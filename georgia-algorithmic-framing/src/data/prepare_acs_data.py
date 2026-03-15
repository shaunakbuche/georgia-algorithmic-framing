"""
prepare_acs_data.py

Downloads county-level socioeconomic data from the American Community Survey
(ACS) 5-year estimates via the Census Bureau API.

Variables collected per county per election year:
  - Median household income (B19013_001E)
  - % adults 25+ with bachelor's degree or higher (S1501_C02_015E)
  - Total population (B01003_001E)

API key required. Get one free at: https://api.census.gov/data/key_signup.html
Set in config/config.yaml.

Outputs:
  data/raw/acs/acs_{year}.csv   for each election year
  data/processed/acs_panel.csv  merged panel
"""

import requests
import pandas as pd
from pathlib import Path
import logging
import yaml
import time

logger = logging.getLogger(__name__)

GA_STATE_FIPS = "13"

# ACS variable codes
ACS_VARIABLES = {
    "B19013_001E": "median_income",
    "B01003_001E": "population",
}

# Subject table variable (requires separate /subject endpoint)
ACS_SUBJECT_VARIABLES = {
    "S1501_C02_015E": "pct_bachelors",
}

# Map election year -> ACS 5-year estimate end year
# Use the estimate that would be available at election time
ACS_YEAR_MAP = {
    2018: 2018,
    2020: 2019,  # 2019 5-year estimate most current at 2020 election
    2022: 2021,
    2024: 2023,
}


def load_config() -> dict:
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        logger.warning(
            "config/config.yaml not found. "
            "Copy config/config_template.yaml and add your Census API key."
        )
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_acs_detail(year: int, variables: list, api_key: str) -> pd.DataFrame:
    """Fetch ACS detail table variables for Georgia counties."""
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    var_str = ",".join(["NAME"] + variables)

    params = {
        "get": var_str,
        "for": "county:*",
        "in": f"state:{GA_STATE_FIPS}",
        "key": api_key,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    return df


def fetch_acs_subject(year: int, variables: list, api_key: str) -> pd.DataFrame:
    """Fetch ACS subject table variables for Georgia counties."""
    url = f"https://api.census.gov/data/{year}/acs/acs5/subject"
    var_str = ",".join(["NAME"] + variables)

    params = {
        "get": var_str,
        "for": "county:*",
        "in": f"state:{GA_STATE_FIPS}",
        "key": api_key,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    return df


def download_acs_year(election_year: int, api_key: str, output_dir: Path) -> pd.DataFrame:
    """
    Download and merge ACS variables for a given election year.

    Returns a DataFrame with county_fips, median_income,
    pct_bachelors, population.
    """
    acs_year = ACS_YEAR_MAP[election_year]
    logger.info(f"Fetching ACS {acs_year} 5-year estimates for election year {election_year}...")

    # Detail table (income, population)
    detail_vars = list(ACS_VARIABLES.keys())
    detail_df = fetch_acs_detail(acs_year, detail_vars, api_key)

    # Subject table (education)
    subj_vars = list(ACS_SUBJECT_VARIABLES.keys())
    subj_df = fetch_acs_subject(acs_year, subj_vars, api_key)

    # Build FIPS code and merge
    detail_df["county_fips"] = detail_df["state"] + detail_df["county"]
    subj_df["county_fips"] = subj_df["state"] + subj_df["county"]

    # Rename raw variable codes to human-readable names
    detail_df = detail_df.rename(columns=ACS_VARIABLES)
    subj_df = subj_df.rename(columns=ACS_SUBJECT_VARIABLES)

    merged = detail_df[["county_fips", "NAME", "median_income", "population"]].merge(
        subj_df[["county_fips", "pct_bachelors"]],
        on="county_fips",
        how="left",
    )

    # Convert to numeric
    for col in ["median_income", "population", "pct_bachelors"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["election_year"] = election_year
    merged["acs_year"] = acs_year

    output_path = output_dir / f"acs_{election_year}.csv"
    merged.to_csv(output_path, index=False)
    logger.info(f"Saved {len(merged)} counties to {output_path}")

    return merged


def build_acs_panel(raw_dir: Path) -> pd.DataFrame:
    """Concatenate per-year ACS files into a single panel."""
    frames = []
    for year in [2018, 2020, 2022, 2024]:
        path = raw_dir / f"acs_{year}.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
        else:
            logger.warning(f"ACS file missing: {path}")

    if not frames:
        return None

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns={"election_year": "year"})
    logger.info(f"Built ACS panel: {len(panel)} county-year observations")
    return panel


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config()
    api_key = config.get("census_api_key", "")

    raw_dir = Path("data/raw/acs")
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not api_key or api_key == "YOUR_CENSUS_API_KEY_HERE":
        logger.error(
            "No Census API key found.\n"
            "1. Get a free key at https://api.census.gov/data/key_signup.html\n"
            "2. Add it to config/config.yaml under 'census_api_key'"
        )
        return

    for election_year in [2018, 2020, 2022, 2024]:
        try:
            download_acs_year(election_year, api_key, raw_dir)
            time.sleep(0.5)  # Be polite to the Census API
        except Exception as e:
            logger.error(f"Failed to download ACS for {election_year}: {e}")

    panel = build_acs_panel(raw_dir)
    if panel is not None:
        out_path = Path("data/processed/acs_panel.csv")
        panel.to_csv(out_path, index=False)
        logger.info(f"Saved ACS panel to {out_path}")


if __name__ == "__main__":
    main()
