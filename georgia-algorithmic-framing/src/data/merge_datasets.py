"""
merge_datasets.py

Merges election returns, ACS socioeconomic data, and attributed ad content
features (sentiment, topic prevalence) into the final county-year panel dataset.

Inputs (all from data/processed/):
  election_panel.csv      county × year election outcomes
  acs_panel.csv           county × year ACS covariates
  county_sentiment.csv    county × year sentiment index
  county_topics.csv       county × year topic prevalence

Output:
  data/processed/panel_dataset.csv   final analysis-ready panel (N ≤ 636)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
ELECTION_YEARS = [2018, 2020, 2022, 2024]


def load_and_validate(path: Path, required_cols: list) -> pd.DataFrame:
    """Load a CSV and verify required columns exist."""
    if not path.exists():
        logger.error(f"Required file not found: {path}")
        return None

    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"{path.name} is missing columns: {missing}")

    logger.info(f"Loaded {path.name}: {df.shape}")
    return df


def build_panel() -> pd.DataFrame:
    """
    Merge all data sources into the final panel dataset.

    Returns
    -------
    pd.DataFrame
        County-year panel with all outcome and predictor variables.
        N = 159 counties × 4 years = 636 maximum observations.
    """
    # --- Election outcomes ---
    elections = load_and_validate(
        PROCESSED_DIR / "election_panel.csv",
        ["county_fips", "year", "turnout_pct", "rep_margin_pct"],
    )
    if elections is None:
        return None

    # --- ACS covariates ---
    acs = load_and_validate(
        PROCESSED_DIR / "acs_panel.csv",
        ["county_fips", "year", "median_income", "pct_bachelors", "population"],
    )

    # --- Sentiment index ---
    sentiment = load_and_validate(
        PROCESSED_DIR / "county_sentiment.csv",
        ["county_fips", "year", "sentiment_index"],
    )

    # --- Topic prevalence ---
    topics = load_and_validate(
        PROCESSED_DIR / "county_topics.csv",
        ["county_fips", "year", "topic_social_share", "topic_health_share",
         "topic_election_share"],
    )

    # Start with elections as the base
    panel = elections.copy()

    # Merge in ACS
    if acs is not None:
        panel = panel.merge(
            acs[["county_fips", "year", "median_income", "pct_bachelors", "population"]],
            on=["county_fips", "year"],
            how="left",
        )

    # Merge in sentiment
    if sentiment is not None:
        panel = panel.merge(
            sentiment[["county_fips", "year", "sentiment_index",
                        "n_ads", "total_impressions", "pct_negative"]],
            on=["county_fips", "year"],
            how="left",
        )

    # Merge in topics
    if topics is not None:
        topic_cols = ["county_fips", "year", "topic_social_share",
                      "topic_health_share", "topic_election_share"]
        available_cols = [c for c in topic_cols if c in topics.columns]
        panel = panel.merge(topics[available_cols], on=["county_fips", "year"], how="left")

    # Add lagged outcomes (prior election values)
    panel = panel.sort_values(["county_fips", "year"])
    panel["prior_turnout"] = panel.groupby("county_fips")["turnout_pct"].shift(1)
    panel["prior_margin"] = panel.groupby("county_fips")["rep_margin_pct"].shift(1)

    # Presidential year indicator
    panel["is_presidential"] = panel["year"].isin([2020, 2024]).astype(int)

    # Drop counties with missing outcomes
    n_before = len(panel)
    panel = panel.dropna(subset=["turnout_pct", "rep_margin_pct"])
    n_after = len(panel)
    if n_before > n_after:
        logger.warning(f"Dropped {n_before - n_after} rows with missing outcomes")

    logger.info(
        f"Final panel: {len(panel)} observations, "
        f"{panel['county_fips'].nunique()} counties, "
        f"{panel['year'].nunique()} years"
    )

    # Report missingness
    for col in panel.columns:
        n_missing = panel[col].isna().sum()
        if n_missing > 0:
            logger.info(f"  Missing {col}: {n_missing} ({n_missing/len(panel)*100:.1f}%)")

    return panel


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    panel = build_panel()

    if panel is not None:
        out_path = PROCESSED_DIR / "panel_dataset.csv"
        panel.to_csv(out_path, index=False)
        logger.info(f"\nSaved panel dataset to {out_path}")
        logger.info(f"Shape: {panel.shape}")
        logger.info(f"\nColumns:\n{list(panel.columns)}")
    else:
        logger.error(
            "Panel construction failed. Ensure all upstream scripts have run:\n"
            "  1. src/data/download_election_data.py\n"
            "  2. src/data/prepare_acs_data.py\n"
            "  3. src/data/download_ad_data.py\n"
            "  4. src/preprocessing/preprocess_ads.py\n"
            "  5. src/analysis/sentiment_analysis.py\n"
            "  6. src/analysis/topic_modeling.py\n"
            "  7. Then re-run this script"
        )


if __name__ == "__main__":
    main()
