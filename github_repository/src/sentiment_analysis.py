#!/usr/bin/env python3
"""
Sentiment Analysis for Political Advertisements

This script performs sentiment analysis on political advertisement text
using TextBlob and aggregates results to county-year level.

Usage:
    python sentiment_analysis.py
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import os
from tqdm import tqdm

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def analyze_sentiment(text):
    """
    Calculate sentiment polarity and subjectivity for text.
    
    Parameters
    ----------
    text : str
        Advertisement text (original, not preprocessed)
        
    Returns
    -------
    dict
        Dictionary with polarity and subjectivity scores
    """
    try:
        if pd.isna(text) or text.strip() == '':
            return {'polarity': 0.0, 'subjectivity': 0.5}
        
        blob = TextBlob(str(text))
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    except Exception as e:
        print(f"Error processing text: {e}")
        return {'polarity': 0.0, 'subjectivity': 0.5}


def process_ad_sentiments(df):
    """
    Process sentiment for all advertisements in dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Advertisement data with 'ad_text' column
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added sentiment columns
    """
    print("Analyzing sentiment for all advertisements...")
    
    sentiments = []
    for text in tqdm(df['ad_text'], desc="Processing ads"):
        sent = analyze_sentiment(text)
        sentiments.append(sent)
    
    sent_df = pd.DataFrame(sentiments)
    
    df = df.copy()
    df['polarity'] = sent_df['polarity']
    df['subjectivity'] = sent_df['subjectivity']
    
    # Categorize sentiment
    df['sentiment_category'] = pd.cut(
        df['polarity'],
        bins=[-1.01, -0.1, 0.1, 1.01],
        labels=['negative', 'neutral', 'positive']
    )
    
    return df


def aggregate_county_year_sentiment(df):
    """
    Aggregate sentiment to county-year level with impression weighting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Ad-level data with sentiment scores and impressions
        
    Returns
    -------
    pd.DataFrame
        County-year level sentiment aggregates
    """
    print("\nAggregating sentiment to county-year level...")
    
    def weighted_stats(group):
        weights = group['impressions']
        polarity = group['polarity']
        
        # Weighted mean
        weighted_mean = np.average(polarity, weights=weights)
        
        # Weighted standard deviation
        weighted_var = np.average((polarity - weighted_mean)**2, weights=weights)
        weighted_std = np.sqrt(weighted_var)
        
        # Percentages by category
        total_impr = weights.sum()
        pct_negative = weights[group['sentiment_category'] == 'negative'].sum() / total_impr
        pct_positive = weights[group['sentiment_category'] == 'positive'].sum() / total_impr
        pct_neutral = weights[group['sentiment_category'] == 'neutral'].sum() / total_impr
        
        return pd.Series({
            'Sentiment_index': weighted_mean,
            'Sentiment_std': weighted_std,
            'pct_negative': pct_negative,
            'pct_positive': pct_positive,
            'pct_neutral': pct_neutral,
            'total_impressions': total_impr,
            'n_ads': len(group)
        })
    
    aggregated = df.groupby(['county', 'year']).apply(weighted_stats).reset_index()
    
    return aggregated


def generate_sentiment_summary(df):
    """
    Generate summary statistics for sentiment analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Ad-level data with sentiment scores
    """
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Total ads: {len(df):,}")
    print(f"  Mean polarity: {df['polarity'].mean():.4f}")
    print(f"  Std polarity: {df['polarity'].std():.4f}")
    print(f"  Min polarity: {df['polarity'].min():.4f}")
    print(f"  Max polarity: {df['polarity'].max():.4f}")
    
    # By year
    print("\nBy Year:")
    year_stats = df.groupby('year').agg({
        'polarity': ['mean', 'std', 'count'],
        'impressions': 'sum'
    })
    print(year_stats)
    
    # Sentiment category distribution
    print("\nSentiment Category Distribution:")
    cat_dist = df['sentiment_category'].value_counts(normalize=True)
    for cat, pct in cat_dist.items():
        print(f"  {cat}: {pct:.1%}")
    
    # By year category
    print("\nCategory Distribution by Year:")
    year_cat = pd.crosstab(df['year'], df['sentiment_category'], normalize='index')
    print(year_cat)


def main():
    """Main execution function."""
    print("="*60)
    print("SENTIMENT ANALYSIS")
    print("="*60)
    
    # Load advertisement data
    print("\nLoading advertisement data...")
    ad_filepath = os.path.join(DATA_DIR, 'processed', 'ads_with_counties.csv')
    df = pd.read_csv(ad_filepath)
    print(f"Loaded {len(df):,} advertisements")
    
    # Process sentiments
    df = process_ad_sentiments(df)
    
    # Generate summary
    generate_sentiment_summary(df)
    
    # Aggregate to county-year
    county_year_sentiment = aggregate_county_year_sentiment(df)
    
    # Save outputs
    print("\nSaving outputs...")
    
    # Ad-level sentiments
    ad_output_path = os.path.join(DATA_DIR, 'processed', 'ad_sentiment.csv')
    df.to_csv(ad_output_path, index=False)
    print(f"  Saved ad-level sentiments to {ad_output_path}")
    
    # County-year sentiments
    cy_output_path = os.path.join(DATA_DIR, 'processed', 'county_year_sentiment.csv')
    county_year_sentiment.to_csv(cy_output_path, index=False)
    print(f"  Saved county-year sentiments to {cy_output_path}")
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
