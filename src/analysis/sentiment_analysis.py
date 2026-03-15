"""
sentiment_analysis.py

Sentiment analysis for political advertisement corpus.

Two methods are implemented:
  - VADER (recommended): Designed for social media text; handles
    capitalization, punctuation intensity, and common online language.
    Does not require preprocessing — run on raw/lightly cleaned text.

  - TextBlob (original): General-purpose lexicon-based method.
    Included for methodological transparency and comparison.
    Known limitation: not trained on political or social media text.
    See paper §3.2.2 and Appendix D for discussion.

The paper reports TextBlob results (for continuity with submitted version),
but this module defaults to VADER. Set method='textblob' in config.yaml
to reproduce the paper's exact numbers.

Input:  data/processed/ads_preprocessed.csv
Output: data/processed/ad_sentiment.csv
        data/processed/county_sentiment.csv
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)

# VADER analyzer is stateless and can be reused
_vader = SentimentIntensityAnalyzer()


def load_config() -> dict:
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Per-document scoring
# ---------------------------------------------------------------------------

def score_vader(text: str) -> float:
    """
    Score text with VADER compound polarity.

    Returns compound score in [-1.0, 1.0].
    Compound is a normalized, weighted composite of positive,
    neutral, and negative scores — the standard single-value
    sentiment summary from VADER.
    """
    if pd.isna(text) or str(text).strip() == "":
        return np.nan
    scores = _vader.polarity_scores(str(text))
    return scores["compound"]


def score_textblob(text: str) -> float:
    """
    Score text with TextBlob polarity.

    Returns polarity in [-1.0, 1.0].
    Note: TextBlob is trained on movie reviews and does not account
    for domain-specific political language or social media conventions.
    """
    if pd.isna(text) or str(text).strip() == "":
        return np.nan
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception as e:
        logger.warning(f"TextBlob error: {e}")
        return np.nan


def score_subjectivity(text: str) -> float:
    """TextBlob subjectivity score [0=objective, 1=subjective]."""
    if pd.isna(text) or str(text).strip() == "":
        return np.nan
    try:
        return TextBlob(str(text)).sentiment.subjectivity
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Corpus-level analysis
# ---------------------------------------------------------------------------

def analyze_ads(ads_df: pd.DataFrame,
                text_column: str = "text_clean",
                method: str = "vader") -> pd.DataFrame:
    """
    Add sentiment scores to advertisement DataFrame.

    Parameters
    ----------
    ads_df : pd.DataFrame
        DataFrame with ad records (use text_clean column from preprocessing)
    text_column : str
        Column containing ad text
    method : str
        'vader' (recommended) or 'textblob'

    Returns
    -------
    pd.DataFrame
        DataFrame with polarity and subjectivity columns added
    """
    df = ads_df.copy()

    # Use text_clean if available, fall back to raw text
    if text_column not in df.columns:
        text_column = "text"
        logger.warning("text_clean not found; falling back to raw text column")

    logger.info(f"Scoring {len(df)} ads with {method.upper()}...")

    if method == "vader":
        df["polarity"] = df[text_column].apply(score_vader)
        # VADER also provides sub-scores; store for diagnostics
        def get_vader_pos(text):
            if pd.isna(text) or str(text).strip() == "":
                return np.nan
            return _vader.polarity_scores(str(text))["pos"]

        def get_vader_neg(text):
            if pd.isna(text) or str(text).strip() == "":
                return np.nan
            return _vader.polarity_scores(str(text))["neg"]

        df["vader_pos"] = df[text_column].apply(get_vader_pos)
        df["vader_neg"] = df[text_column].apply(get_vader_neg)
        df["subjectivity"] = np.nan  # VADER has no subjectivity score

    elif method == "textblob":
        df["polarity"] = df[text_column].apply(score_textblob)
        df["subjectivity"] = df[text_column].apply(score_subjectivity)
        df["vader_pos"] = np.nan
        df["vader_neg"] = np.nan
    else:
        raise ValueError(f"Unknown sentiment method: {method}. Use 'vader' or 'textblob'.")

    df["sentiment_method"] = method

    # Summary statistics
    logger.info(f"  Mean polarity: {df['polarity'].mean():.3f}")
    logger.info(f"  Negative (< 0): {(df['polarity'] < 0).mean()*100:.1f}%")
    logger.info(f"  Positive (> 0): {(df['polarity'] > 0).mean()*100:.1f}%")
    logger.info(f"  Neutral (= 0):  {(df['polarity'] == 0).mean()*100:.1f}%")

    return df


# ---------------------------------------------------------------------------
# County-year aggregation
# ---------------------------------------------------------------------------

def calculate_weighted_sentiment(group: pd.DataFrame,
                                  weight_column: str = "impressions") -> float:
    """
    Impression-weighted mean sentiment for a county-year group.

    Implements: S_ct = Σ(s_i × w_i) / Σ(w_i)  [paper eq. in §3.2.2]
    """
    values = group["polarity"]
    weights = group[weight_column]

    mask = ~(values.isna() | weights.isna()) & (weights > 0)
    if mask.sum() == 0:
        return np.nan

    return float(np.average(values[mask], weights=weights[mask]))


def aggregate_to_county_year(ads_df: pd.DataFrame,
                              county_column: str = "county_fips",
                              year_column: str = "year",
                              weight_column: str = "impressions") -> pd.DataFrame:
    """
    Aggregate advertisement sentiment to county-year level.

    Parameters
    ----------
    ads_df : pd.DataFrame
        Advertisement data with polarity scores
    county_column, year_column, weight_column : str
        Column name overrides

    Returns
    -------
    pd.DataFrame
        County-year aggregated sentiment with columns:
          county_fips, year, sentiment_index, sentiment_std,
          n_ads, total_impressions, pct_negative
    """
    logger.info("Aggregating sentiment to county-year level...")
    results = []

    for (county, year), group in ads_df.groupby([county_column, year_column]):
        sentiment_index = calculate_weighted_sentiment(group, weight_column)
        total_impressions = group[weight_column].sum()
        negative_impressions = group.loc[
            group["polarity"] < 0, weight_column
        ].sum()

        results.append({
            county_column: county,
            year_column: year,
            "sentiment_index": sentiment_index,
            "sentiment_std": group["polarity"].std(),
            "n_ads": len(group["ad_id"].unique()) if "ad_id" in group.columns else len(group),
            "total_impressions": total_impressions,
            "pct_negative": (negative_impressions / total_impressions
                             if total_impressions > 0 else np.nan),
        })

    result_df = pd.DataFrame(results)
    logger.info(f"Created {len(result_df)} county-year observations")
    return result_df


# ---------------------------------------------------------------------------
# Validation: VADER vs TextBlob comparison
# ---------------------------------------------------------------------------

def compare_methods(ads_df: pd.DataFrame,
                    text_column: str = "text_clean",
                    sample_n: int = 200) -> pd.DataFrame:
    """
    Compare VADER and TextBlob polarity on a sample of ads.

    Useful for robustness checks and methodological validation.
    Returns a DataFrame with both scores side by side.
    """
    sample = ads_df.sample(min(sample_n, len(ads_df)), random_state=42)[[text_column]].copy()

    sample["polarity_vader"] = sample[text_column].apply(score_vader)
    sample["polarity_textblob"] = sample[text_column].apply(score_textblob)

    corr = sample[["polarity_vader", "polarity_textblob"]].corr()
    logger.info(f"VADER vs TextBlob Pearson r: {corr.iloc[0,1]:.3f}")

    # Sign agreement (both positive or both negative)
    sign_agreement = (
        (np.sign(sample["polarity_vader"]) == np.sign(sample["polarity_textblob"]))
        & (sample["polarity_vader"] != 0)
        & (sample["polarity_textblob"] != 0)
    ).mean()
    logger.info(f"Sign agreement (non-neutral): {sign_agreement*100:.1f}%")

    return sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config()
    method = config.get("sentiment", {}).get("method", "vader")

    input_path = Path("data/processed/ads_preprocessed.csv")
    if not input_path.exists():
        logger.error(
            f"Preprocessed ads not found at {input_path}\n"
            "Run src/preprocessing/preprocess_ads.py first."
        )
        return

    ads_df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(ads_df)} ad records")

    # Score sentiment
    ads_with_sentiment = analyze_ads(ads_df, method=method)

    # Save ad-level scores
    ad_output = Path("data/processed/ad_sentiment.csv")
    ads_with_sentiment.to_csv(ad_output, index=False)
    logger.info(f"Saved ad-level sentiment to {ad_output}")

    # Aggregate to county-year
    county_sentiment = aggregate_to_county_year(ads_with_sentiment)

    county_output = Path("data/processed/county_sentiment.csv")
    county_sentiment.to_csv(county_output, index=False)
    logger.info(f"Saved county-year sentiment to {county_output}")

    # Run comparison for methodological appendix
    logger.info("\nRunning VADER vs TextBlob comparison (Appendix D)...")
    compare_methods(ads_df)


if __name__ == "__main__":
    main()
