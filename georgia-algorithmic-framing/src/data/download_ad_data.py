"""
download_ad_data.py

Queries the Meta Ad Library API for Georgia political advertisements
and performs county-level impression attribution.

This is the methodological heart of the data construction pipeline —
the county attribution logic was missing from the original repo.

Requirements:
  - Meta Ad Library API access token in config/config.yaml
  - Apply for researcher access at https://www.facebook.com/ads/library/api/

Outputs:
  data/raw/ads/ads_{year}_raw.json        raw API response
  data/processed/ads_attributed.csv       county-attributed impressions
"""

import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

GA_STATE_FIPS = "13"

# Election dates (for computing 90-day collection windows)
ELECTION_DATES = {
    2018: "2018-11-06",
    2020: "2020-11-03",
    2022: "2022-11-08",
    2024: "2024-11-05",
}

# Meta Ad Library impression range -> geometric mean estimate
# Using geometric mean (sqrt(lower * upper)) rather than arithmetic midpoint
# because impression distributions are log-normal
IMPRESSION_RANGE_MAP = {
    "<1K":       (1, 1000),
    "1K-5K":     (1000, 5000),
    "5K-10K":    (5000, 10000),
    "10K-50K":   (10000, 50000),
    "50K-100K":  (50000, 100000),
    "100K-200K": (100000, 200000),
    "200K-500K": (200000, 500000),
    "500K-1M":   (500000, 1000000),
    ">1M":       (1000000, 5000000),
}

# Georgia DMA (Designated Market Area) to county FIPS mapping
# Source: Nielsen DMA definitions
GA_DMA_TO_COUNTIES = {
    "Atlanta":         ["13013", "13015", "13035", "13045", "13057", "13059",
                        "13063", "13067", "13077", "13085", "13089", "13097",
                        "13113", "13117", "13121", "13135", "13151", "13159",
                        "13171", "13199", "13211", "13217", "13223", "13247",
                        "13255", "13285", "13295", "13297"],
    "Savannah":        ["13029", "13051", "13103", "13179", "13251", "13267",
                        "13305"],
    "Augusta":         ["13033", "13073", "13105", "13107", "13163", "13189",
                        "13245", "13261", "13277", "13293", "13299", "13301",
                        "13303", "13317"],
    "Columbus":        ["13053", "13079", "13145", "13193", "13215", "13259",
                        "13283", "13285"],
    "Macon":           ["13021", "13023", "13093", "13153", "13169", "13193",
                        "13207", "13225", "13235", "13269"],
    "Albany":          ["13007", "13037", "13071", "13075", "13099", "13177",
                        "13197", "13201", "13253", "13261", "13287"],
    "Valdosta":        ["13065", "13101", "13155", "13173", "13185", "13187",
                        "13253", "13275", "13309"],
    "Chattanooga":     ["13047", "13083", "13111", "13213", "13295"],
    "Greenville":      ["13011", "13119", "13137", "13147", "13195", "13227",
                        "13241", "13257", "13291", "13311"],
}

# Known Georgia cities/metros -> county FIPS (for city-level targeting)
CITY_TO_COUNTY = {
    "Atlanta": "13121",
    "Augusta": "13245",
    "Columbus": "13215",
    "Macon": "13021",
    "Savannah": "13051",
    "Athens": "13059",
    "Sandy Springs": "13121",
    "Roswell": "13117",
    "Albany": "13095",
    "Warner Robins": "13153",
    "Alpharetta": "13117",
    "Marietta": "13067",
    "Smyrna": "13067",
    "Valdosta": "13185",
    "Brookhaven": "13089",
}


def load_config() -> dict:
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


def convert_impression_range(range_str: Optional[str]) -> int:
    """
    Convert Meta impression range string to a point estimate.

    Uses geometric mean of the range bounds rather than arithmetic midpoint,
    consistent with log-normal impression distributions.

    Parameters
    ----------
    range_str : str or None
        Meta API impression range (e.g. "10K-50K")

    Returns
    -------
    int
        Point estimate of impressions
    """
    if not range_str or pd.isna(range_str):
        return 0

    bounds = IMPRESSION_RANGE_MAP.get(range_str.strip())
    if not bounds:
        logger.warning(f"Unknown impression range: {range_str}")
        return 0

    lower, upper = bounds
    return int(np.sqrt(lower * upper))  # Geometric mean


def query_ad_library_page(access_token: str, params: dict) -> dict:
    """Make a single paginated request to the Meta Ad Library API."""
    base_url = "https://graph.facebook.com/v18.0/ads_archive"
    params["access_token"] = access_token

    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def download_ads_for_year(year: int, access_token: str, output_dir: Path) -> list:
    """
    Download all Georgia political ads for the 90-day window before an election.

    Parameters
    ----------
    year : int
        Election year
    access_token : str
        Meta Ad Library API access token
    output_dir : Path
        Directory to save raw JSON

    Returns
    -------
    list of ad records
    """
    election_date = datetime.strptime(ELECTION_DATES[year], "%Y-%m-%d")
    start_date = (election_date - timedelta(days=90)).strftime("%Y-%m-%d")
    end_date = ELECTION_DATES[year]

    logger.info(f"Collecting {year} ads: {start_date} to {end_date}")

    params = {
        "ad_type": "POLITICAL_AND_ISSUE_ADS",
        "ad_reached_countries": "['US']",
        "search_terms": "Georgia",
        "ad_delivery_date_min": start_date,
        "ad_delivery_date_max": end_date,
        "fields": (
            "id,ad_creative_bodies,page_name,impressions,spend,"
            "ad_delivery_start_time,ad_delivery_stop_time,"
            "demographic_distribution,region_distribution,"
            "publisher_platforms,target_ages,target_gender"
        ),
        "limit": 500,
    }

    all_ads = []
    page_count = 0

    while True:
        try:
            data = query_ad_library_page(access_token, params)
            ads = data.get("data", [])
            all_ads.extend(ads)
            page_count += 1

            logger.info(f"Page {page_count}: {len(ads)} ads (total: {len(all_ads)})")

            # Pagination
            next_cursor = data.get("paging", {}).get("cursors", {}).get("after")
            if not next_cursor or len(ads) == 0:
                break

            params["after"] = next_cursor
            time.sleep(0.5)  # Rate limit compliance

        except requests.HTTPError as e:
            logger.error(f"API error on page {page_count}: {e}")
            break

    # Save raw JSON
    raw_path = output_dir / f"ads_{year}_raw.json"
    with open(raw_path, "w") as f:
        json.dump(all_ads, f, indent=2)
    logger.info(f"Saved {len(all_ads)} raw ads to {raw_path}")

    return all_ads


def extract_ad_text(ad: dict) -> str:
    """Extract primary text content from an ad record."""
    bodies = ad.get("ad_creative_bodies", [])
    if bodies:
        return " ".join(bodies)
    return ""


def attribute_ad_to_counties(
    ad: dict,
    county_populations: pd.DataFrame,
) -> dict:
    """
    Attribute advertisement impressions to Georgia counties.

    Attribution hierarchy:
      1. City-level targeting  -> assign to matching county
      2. DMA-level targeting   -> distribute proportionally by county population
      3. Statewide / no target -> distribute proportionally by county population

    This function implements the county attribution algorithm documented in
    the paper's Methods section (§3.1.2) and Appendix B.3.

    Parameters
    ----------
    ad : dict
        Advertisement record from Meta Ad Library API
    county_populations : pd.DataFrame
        DataFrame with columns [county_fips, population]

    Returns
    -------
    dict
        Mapping {county_fips: attributed_impressions}
    """
    total_impressions = convert_impression_range(
        ad.get("impressions", {}).get("upper_bound") or
        ad.get("impressions_range", "")
    )

    if total_impressions == 0:
        return {}

    attributions = {}

    # --- Attempt city-level attribution ---
    # Meta provides region_distribution as list of {region, percentage}
    region_dist = ad.get("region_distribution", [])
    city_matches = {}
    for region_entry in region_dist:
        region_name = region_entry.get("region", "").split(",")[0].strip()
        pct = float(region_entry.get("percentage", 0))
        if region_name in CITY_TO_COUNTY:
            county = CITY_TO_COUNTY[region_name]
            city_matches[county] = city_matches.get(county, 0) + pct

    if city_matches:
        total_pct = sum(city_matches.values())
        if total_pct > 0:
            for county, pct in city_matches.items():
                attributions[county] = int(total_impressions * pct / 100)
            return attributions

    # --- Attempt DMA-level attribution ---
    # Check if any region matches a known Georgia DMA
    for region_entry in region_dist:
        region_name = region_entry.get("region", "").strip()
        for dma_name, dma_counties in GA_DMA_TO_COUNTIES.items():
            if dma_name.lower() in region_name.lower():
                dma_pop = county_populations[
                    county_populations["county_fips"].isin(dma_counties)
                ]["population"].sum()

                if dma_pop == 0:
                    continue

                pct = float(region_entry.get("percentage", 0)) / 100

                for fips in dma_counties:
                    row = county_populations[
                        county_populations["county_fips"] == fips
                    ]
                    if row.empty:
                        continue
                    county_pop = row["population"].values[0]
                    attributions[fips] = attributions.get(fips, 0) + int(
                        total_impressions * pct * (county_pop / dma_pop)
                    )

    if attributions:
        return attributions

    # --- Fallback: statewide population-proportional allocation ---
    # NOTE: This is the weakest attribution method and introduces measurement
    # error that biases regression estimates toward null (see paper §6).
    # Flagged with 'statewide_allocation' = True for sensitivity analysis.
    total_pop = county_populations["population"].sum()
    for _, row in county_populations.iterrows():
        fips = row["county_fips"]
        pop = row["population"]
        attributions[fips] = int(total_impressions * (pop / total_pop))

    return attributions


def build_attributed_ads(
    raw_ads: list,
    year: int,
    county_populations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert raw ad records to county-attributed long-format DataFrame.

    Each row = one ad × county, with proportionally attributed impressions.
    """
    records = []

    for ad in raw_ads:
        text = extract_ad_text(ad)
        if not text.strip():
            continue

        ad_impressions = convert_impression_range(
            str(ad.get("impressions", {}).get("upper_bound", ""))
        )

        attributions = attribute_ad_to_counties(ad, county_populations)
        is_statewide = len(attributions) > 50  # Heuristic flag for statewide allocation

        for county_fips, attributed_impressions in attributions.items():
            records.append({
                "ad_id": ad.get("id"),
                "county_fips": county_fips,
                "year": year,
                "text": text,
                "page_name": ad.get("page_name", ""),
                "impressions": attributed_impressions,
                "total_ad_impressions": ad_impressions,
                "statewide_allocation": is_statewide,
                "delivery_start": ad.get("ad_delivery_start_time"),
                "delivery_stop": ad.get("ad_delivery_stop_time"),
            })

    df = pd.DataFrame(records)
    logger.info(
        f"{year}: {len(raw_ads)} ads → {len(df)} county-ad records "
        f"({df['statewide_allocation'].mean()*100:.1f}% statewide-allocated)"
    )
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config()
    access_token = config.get("meta_access_token", "")
    raw_dir = Path("data/raw/ads")
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not access_token or access_token == "YOUR_META_ACCESS_TOKEN_HERE":
        logger.error(
            "No Meta API token found.\n"
            "Apply for researcher access at https://www.facebook.com/ads/library/api/\n"
            "Then add your token to config/config.yaml under 'meta_access_token'"
        )
        logger.info(
            "\nIf you already have the raw ad data, save it as:\n"
            "  data/raw/ads/ads_{year}_raw.json\n"
            "and re-run this script — it will skip the API calls."
        )
        return

    # Load county populations for attribution
    acs_path = Path("data/processed/acs_panel.csv")
    if not acs_path.exists():
        logger.error(
            "ACS panel not found. Run src/data/prepare_acs_data.py first."
        )
        return

    acs = pd.read_csv(acs_path)

    all_attributed = []

    for year in [2018, 2020, 2022, 2024]:
        raw_path = raw_dir / f"ads_{year}_raw.json"

        # Use cached raw data if available
        if raw_path.exists():
            logger.info(f"Loading cached raw ads for {year} from {raw_path}")
            with open(raw_path) as f:
                raw_ads = json.load(f)
        else:
            raw_ads = download_ads_for_year(year, access_token, raw_dir)

        # Get county populations for this election year
        year_pop = acs[acs["year"] == year][["county_fips", "population"]].dropna()

        attributed = build_attributed_ads(raw_ads, year, year_pop)
        all_attributed.append(attributed)

    if all_attributed:
        combined = pd.concat(all_attributed, ignore_index=True)
        out_path = Path("data/processed/ads_attributed.csv")
        combined.to_csv(out_path, index=False)
        logger.info(f"Saved {len(combined)} county-ad records to {out_path}")


if __name__ == "__main__":
    main()
