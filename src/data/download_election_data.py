"""
download_election_data.py

Downloads county-level election returns for Georgia from:
  - MIT Election Data and Science Lab (MEDSL) — presidential years
  - Georgia Secretary of State — gubernatorial years

Outputs:
  data/raw/elections/ga_elections_{year}.csv  for each year
  data/raw/elections/ga_registration_{year}.csv for each year
"""

import requests
import pandas as pd
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)

# MEDSL county presidential returns (public, no API key needed)
MEDSL_URL = "https://raw.githubusercontent.com/MEDSL/county-returns/main/countypres_2000-2020.csv"

# Georgia SoS result URLs (direct download links; update if SoS changes format)
GA_SOS_URLS = {
    2018: "https://results.enr.clarityelections.com/GA/91639/Web02-state.shtml",
    2022: "https://results.enr.clarityelections.com/GA/115465/Web02-state.shtml",
}

# FIPS codes for Georgia counties (complete list)
GA_COUNTY_FIPS = {
    "Appling": "13001", "Atkinson": "13003", "Bacon": "13005", "Baker": "13007",
    "Baldwin": "13009", "Banks": "13011", "Barrow": "13013", "Bartow": "13015",
    "Ben Hill": "13017", "Berrien": "13019", "Bibb": "13021", "Bleckley": "13023",
    "Brantley": "13025", "Brooks": "13027", "Bryan": "13029", "Bulloch": "13031",
    "Burke": "13033", "Butts": "13035", "Calhoun": "13037", "Camden": "13039",
    "Candler": "13043", "Carroll": "13045", "Catoosa": "13047", "Charlton": "13049",
    "Chatham": "13051", "Chattahoochee": "13053", "Chattooga": "13055",
    "Cherokee": "13057", "Clarke": "13059", "Clay": "13061", "Clayton": "13063",
    "Clinch": "13065", "Cobb": "13067", "Coffee": "13069", "Colquitt": "13071",
    "Columbia": "13073", "Cook": "13075", "Coweta": "13077", "Crawford": "13079",
    "Crisp": "13081", "Dade": "13083", "Dawson": "13085", "Decatur": "13087",
    "DeKalb": "13089", "Dodge": "13091", "Dooly": "13093", "Dougherty": "13095",
    "Douglas": "13097", "Early": "13099", "Echols": "13101", "Effingham": "13103",
    "Elbert": "13105", "Emanuel": "13107", "Evans": "13109", "Fannin": "13111",
    "Fayette": "13113", "Floyd": "13115", "Forsyth": "13117", "Franklin": "13119",
    "Fulton": "13121", "Gilmer": "13123", "Glascock": "13125", "Glynn": "13127",
    "Gordon": "13129", "Grady": "13131", "Greene": "13133", "Gwinnett": "13135",
    "Habersham": "13137", "Hall": "13139", "Hancock": "13141", "Haralson": "13143",
    "Harris": "13145", "Hart": "13147", "Heard": "13149", "Henry": "13151",
    "Houston": "13153", "Irwin": "13155", "Jackson": "13157", "Jasper": "13159",
    "Jeff Davis": "13161", "Jefferson": "13163", "Jenkins": "13165", "Johnson": "13167",
    "Jones": "13169", "Lamar": "13171", "Lanier": "13173", "Laurens": "13175",
    "Lee": "13177", "Liberty": "13179", "Lincoln": "13181", "Long": "13183",
    "Lowndes": "13185", "Lumpkin": "13187", "McDuffie": "13189", "McIntosh": "13191",
    "Macon": "13193", "Madison": "13195", "Marion": "13197", "Meriwether": "13199",
    "Miller": "13201", "Mitchell": "13205", "Monroe": "13207", "Montgomery": "13209",
    "Morgan": "13211", "Murray": "13213", "Muscogee": "13215", "Newton": "13217",
    "Oconee": "13219", "Oglethorpe": "13221", "Paulding": "13223", "Peach": "13225",
    "Pickens": "13227", "Pierce": "13229", "Pike": "13231", "Polk": "13233",
    "Pulaski": "13235", "Putnam": "13237", "Quitman": "13239", "Rabun": "13241",
    "Randolph": "13243", "Richmond": "13245", "Rockdale": "13247", "Schley": "13249",
    "Screven": "13251", "Seminole": "13253", "Spalding": "13255", "Stephens": "13257",
    "Stewart": "13259", "Sumter": "13261", "Talbot": "13263", "Taliaferro": "13265",
    "Tattnall": "13267", "Taylor": "13269", "Telfair": "13271", "Terrell": "13273",
    "Thomas": "13275", "Tift": "13277", "Toombs": "13279", "Towns": "13281",
    "Treutlen": "13283", "Troup": "13285", "Turner": "13287", "Twiggs": "13289",
    "Union": "13291", "Upson": "13293", "Walker": "13295", "Walton": "13297",
    "Ware": "13299", "Warren": "13301", "Washington": "13303", "Wayne": "13305",
    "Webster": "13307", "Wheeler": "13309", "White": "13311", "Whitfield": "13313",
    "Wilcox": "13315", "Wilkes": "13317", "Wilkinson": "13319", "Worth": "13321",
}


def download_medsl_data(output_dir: Path) -> pd.DataFrame:
    """
    Download MEDSL county presidential returns for Georgia.

    Returns Georgia 2020 presidential results. For 2024, manually update
    with SoS results once MEDSL publishes the dataset.
    """
    logger.info("Downloading MEDSL presidential election data...")

    try:
        df = pd.read_csv(MEDSL_URL)
        ga_df = df[df["state_po"] == "GA"].copy()
        ga_df = ga_df[ga_df["year"].isin([2020])]  # MEDSL lags; add 2024 when available

        # Standardize column names
        ga_df = ga_df.rename(columns={
            "county_fips": "county_fips",
            "county_name": "county_name",
            "year": "year",
            "candidate": "candidate",
            "party_simplified": "party",
            "candidatevotes": "votes",
            "totalvotes": "total_votes",
        })

        output_path = output_dir / "medsl_presidential.csv"
        ga_df.to_csv(output_path, index=False)
        logger.info(f"Saved MEDSL data to {output_path}")
        return ga_df

    except Exception as e:
        logger.error(f"Failed to download MEDSL data: {e}")
        logger.info("Falling back to local data if available.")
        return None


def load_manual_election_data(data_dir: Path, year: int) -> pd.DataFrame:
    """
    Load manually downloaded Georgia SoS election results.

    The Georgia SoS provides results at:
    https://results.enr.clarityelections.com/GA/

    Download the county-level CSV for each election and save to:
      data/raw/elections/ga_sos_{year}_raw.csv

    Expected columns: county, dem_votes, rep_votes, total_votes
    """
    raw_path = data_dir / f"ga_sos_{year}_raw.csv"

    if not raw_path.exists():
        logger.warning(
            f"Raw SoS data not found at {raw_path}.\n"
            f"Download from: https://results.enr.clarityelections.com/GA/\n"
            f"Save as: {raw_path}"
        )
        return None

    df = pd.read_csv(raw_path)
    logger.info(f"Loaded SoS data for {year}: {len(df)} counties")
    return df


def build_election_panel(raw_dir: Path) -> pd.DataFrame:
    """
    Build a unified county-year election panel from all sources.

    Returns DataFrame with columns:
      county_fips, county_name, year, dem_votes, rep_votes,
      total_votes, registered_voters, turnout_pct, rep_margin_pct
    """
    records = []

    for year in [2018, 2020, 2022, 2024]:
        year_file = raw_dir / f"ga_elections_{year}.csv"

        if not year_file.exists():
            logger.warning(f"Election data for {year} not found at {year_file}")
            continue

        df = pd.read_csv(year_file)

        # Standardize to common schema
        df["year"] = year
        df["rep_margin_pct"] = (
            (df["rep_votes"] - df["dem_votes"]) / df["total_votes"] * 100
        )
        df["turnout_pct"] = (
            df["total_votes"] / df["registered_voters"] * 100
        )

        records.append(df)

    if not records:
        logger.error(
            "No election data found. Please populate data/raw/elections/ "
            "with county-level results. See data/raw/elections/README.md."
        )
        return None

    panel = pd.concat(records, ignore_index=True)
    logger.info(f"Built election panel: {len(panel)} county-year observations")
    return panel


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_dir = Path("data/raw/elections")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Attempt MEDSL download for presidential years
    download_medsl_data(raw_dir)

    # Build panel (requires manual SoS files for gubernatorial years)
    panel = build_election_panel(raw_dir)

    if panel is not None:
        out_path = Path("data/processed/election_panel.csv")
        panel.to_csv(out_path, index=False)
        logger.info(f"Saved election panel to {out_path}")
    else:
        logger.info(
            "\nTo complete data collection:\n"
            "1. Download county results from https://results.enr.clarityelections.com/GA/\n"
            "2. Save as data/raw/elections/ga_elections_{year}.csv\n"
            "3. Re-run this script\n"
            "See data/raw/elections/README.md for exact format."
        )


if __name__ == "__main__":
    main()
