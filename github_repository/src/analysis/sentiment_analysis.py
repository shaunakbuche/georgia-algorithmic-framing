"""
Sentiment Analysis Module

This module implements sentiment analysis for political advertisements
using TextBlob and provides aggregation functions for county-year analysis.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def calculate_sentiment(text: str) -> float:
    """
    Calculate sentiment polarity using TextBlob.
    
    Parameters
    ----------
    text : str
        Raw advertisement text
    
    Returns
    -------
    float
        Polarity score in [-1.0, 1.0]
        -1.0 = extremely negative
        +1.0 = extremely positive
        0.0 = neutral
    """
    if pd.isna(text) or str(text).strip() == '':
        return np.nan
    
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity
    except Exception as e:
        logger.warning(f"Error calculating sentiment: {e}")
        return np.nan


def calculate_subjectivity(text: str) -> float:
    """
    Calculate subjectivity score using TextBlob.
    
    Parameters
    ----------
    text : str
        Raw advertisement text
    
    Returns
    -------
    float
        Subjectivity score in [0.0, 1.0]
        0.0 = objective
        1.0 = subjective
    """
    if pd.isna(text) or str(text).strip() == '':
        return np.nan
    
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.subjectivity
    except Exception as e:
        logger.warning(f"Error calculating subjectivity: {e}")
        return np.nan


def analyze_ads(ads_df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Add sentiment scores to advertisement DataFrame.
    
    Parameters
    ----------
    ads_df : pd.DataFrame
        DataFrame with advertisement data
    text_column : str
        Name of column containing ad text
    
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - polarity: sentiment polarity score
        - subjectivity: sentiment subjectivity score
    """
    logger.info(f"Calculating sentiment for {len(ads_df)} advertisements...")
    
    df = ads_df.copy()
    df['polarity'] = df[text_column].apply(calculate_sentiment)
    df['subjectivity'] = df[text_column].apply(calculate_subjectivity)
    
    # Log summary statistics
    logger.info(f"Mean polarity: {df['polarity'].mean():.3f}")
    logger.info(f"Negative ads: {(df['polarity'] < 0).sum()} ({(df['polarity'] < 0).mean()*100:.1f}%)")
    logger.info(f"Positive ads: {(df['polarity'] > 0).sum()} ({(df['polarity'] > 0).mean()*100:.1f}%)")
    
    return df


def calculate_weighted_sentiment(group: pd.DataFrame, 
                                  weight_column: str = 'impressions') -> float:
    """
    Calculate impression-weighted mean sentiment for a group.
    
    Parameters
    ----------
    group : pd.DataFrame
        Subset of ads for a county-year
    weight_column : str
        Name of column containing impression weights
    
    Returns
    -------
    float
        Weighted mean polarity
    """
    weights = group[weight_column]
    values = group['polarity']
    
    # Remove NaN values
    mask = ~(values.isna() | weights.isna())
    weights = weights[mask]
    values = values[mask]
    
    if weights.sum() == 0 or len(values) == 0:
        return np.nan
    
    return np.average(values, weights=weights)


def aggregate_to_county_year(ads_df: pd.DataFrame,
                             county_column: str = 'county_fips',
                             year_column: str = 'year',
                             weight_column: str = 'impressions') -> pd.DataFrame:
    """
    Aggregate advertisement sentiment to county-year level.
    
    Parameters
    ----------
    ads_df : pd.DataFrame
        Advertisement data with sentiment scores
    county_column : str
        Name of county identifier column
    year_column : str
        Name of year column
    weight_column : str
        Name of impression weight column
    
    Returns
    -------
    pd.DataFrame
        County-year aggregated sentiment with columns:
        - county_fips, year
        - sentiment_index: weighted mean polarity
        - sentiment_std: standard deviation
        - n_ads: number of ads
        - total_impressions: sum of impressions
        - pct_negative: % of impressions from negative ads
    """
    logger.info("Aggregating sentiment to county-year level...")
    
    results = []
    
    for (county, year), group in ads_df.groupby([county_column, year_column]):
        sentiment_index = calculate_weighted_sentiment(group, weight_column)
        
        # Calculate additional metrics
        total_impressions = group[weight_column].sum()
        negative_impressions = group[group['polarity'] < 0][weight_column].sum()
        
        results.append({
            county_column: county,
            year_column: year,
            'sentiment_index': sentiment_index,
            'sentiment_std': group['polarity'].std(),
            'n_ads': len(group),
            'total_impressions': total_impressions,
            'pct_negative': negative_impressions / total_impressions if total_impressions > 0 else np.nan
        })
    
    result_df = pd.DataFrame(results)
    logger.info(f"Created {len(result_df)} county-year observations")
    
    return result_df


def main():
    """Main execution function."""
    # Load advertisement data
    ads_path = Path('data/processed/ads_attributed.csv')
    
    if not ads_path.exists():
        logger.error(f"Advertisement data not found at {ads_path}")
        return
    
    logger.info(f"Loading data from {ads_path}")
    ads_df = pd.read_csv(ads_path)
    
    # Calculate sentiment
    ads_with_sentiment = analyze_ads(ads_df)
    
    # Save ad-level sentiment
    output_path = Path('data/processed/ad_sentiment.csv')
    ads_with_sentiment.to_csv(output_path, index=False)
    logger.info(f"Saved ad-level sentiment to {output_path}")
    
    # Aggregate to county-year
    county_sentiment = aggregate_to_county_year(ads_with_sentiment)
    
    # Save county-year sentiment
    county_output = Path('data/processed/county_sentiment.csv')
    county_sentiment.to_csv(county_output, index=False)
    logger.info(f"Saved county-year sentiment to {county_output}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
