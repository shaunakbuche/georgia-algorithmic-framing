#!/usr/bin/env python3
"""
Panel Regression Analysis for Georgia Algorithmic Framing Study

This script performs two-way fixed effects panel regression to analyze
the relationship between political advertising content and electoral outcomes.

Usage:
    python panel_regression.py [--use-processed]
    
Options:
    --use-processed    Use pre-processed data from data/processed/
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import pickle
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
RANDOM_STATE = 42


def load_panel_data(use_processed=True):
    """
    Load panel dataset for analysis.
    
    Parameters
    ----------
    use_processed : bool
        If True, load from processed data directory
        
    Returns
    -------
    pd.DataFrame
        Panel dataset with all variables
    """
    if use_processed:
        filepath = os.path.join(DATA_DIR, 'processed', 'panel_data.csv')
    else:
        filepath = os.path.join(DATA_DIR, 'raw', 'panel_data.csv')
    
    df = pd.read_csv(filepath)
    
    # Ensure proper types
    df['county'] = df['county'].astype(str)
    df['year'] = df['year'].astype(int)
    
    print(f"Loaded {len(df)} observations from {df['county'].nunique()} counties")
    print(f"Years: {sorted(df['year'].unique())}")
    
    return df


def standardize_variables(df, variables):
    """
    Standardize variables to z-scores for coefficient comparison.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    variables : list
        Variable names to standardize
        
    Returns
    -------
    pd.DataFrame
        Dataframe with new standardized columns (*_std)
    """
    df = df.copy()
    
    for var in variables:
        if var in df.columns:
            mean_val = df[var].mean()
            std_val = df[var].std()
            df[f'{var}_std'] = (df[var] - mean_val) / std_val
            print(f"Standardized {var}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    return df


def run_fixed_effects_model(df, dependent_var, independent_vars, 
                            entity_col='county', time_col='year'):
    """
    Estimate two-way fixed effects panel regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    dependent_var : str
        Dependent variable column name
    independent_vars : list
        Independent variable column names
    entity_col : str
        Entity identifier column
    time_col : str
        Time identifier column
        
    Returns
    -------
    PanelOLSResults
        Regression results object
    """
    # Set multi-index for panel structure
    df_panel = df.set_index([entity_col, time_col])
    
    # Extract variables
    y = df_panel[dependent_var]
    X = df_panel[independent_vars]
    
    # Add constant
    X = sm.add_constant(X)
    
    # Estimate model with entity and time fixed effects
    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    
    # Fit with clustered standard errors
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results


def run_turnout_models(df):
    """
    Run panel regression models for voter turnout.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel data with standardized variables
        
    Returns
    -------
    dict
        Dictionary of model results
    """
    print("\n" + "="*60)
    print("TURNOUT MODELS")
    print("="*60)
    
    results = {}
    
    # Model 1: Sentiment only
    print("\nModel 1: Sentiment Index Only")
    results['model1'] = run_fixed_effects_model(
        df, 
        dependent_var='Turnout_pct',
        independent_vars=['Sentiment_index_std']
    )
    print(results['model1'].summary)
    
    # Model 2: Add topic variables
    print("\nModel 2: With Topic Variables")
    results['model2'] = run_fixed_effects_model(
        df,
        dependent_var='Turnout_pct',
        independent_vars=[
            'Sentiment_index_std',
            'TopicEcon_positive_std',
            'TopicAttack_std',
            'TopicElection_std'
        ]
    )
    print(results['model2'].summary)
    
    # Model 3: Full model with controls
    print("\nModel 3: Full Model with Controls")
    results['model3'] = run_fixed_effects_model(
        df,
        dependent_var='Turnout_pct',
        independent_vars=[
            'Sentiment_index_std',
            'TopicEcon_positive_std',
            'TopicAttack_std',
            'TopicElection_std',
            'MedianIncome_std',
            'Edu_BA_pct_std'
        ]
    )
    print(results['model3'].summary)
    
    return results


def run_margin_models(df):
    """
    Run panel regression models for partisan vote margin.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel data with standardized variables
        
    Returns
    -------
    dict
        Dictionary of model results
    """
    print("\n" + "="*60)
    print("VOTE MARGIN MODELS")
    print("="*60)
    
    results = {}
    
    # Model 1: Sentiment only
    print("\nModel 1: Sentiment Index Only")
    results['model1'] = run_fixed_effects_model(
        df,
        dependent_var='RepMargin_pct',
        independent_vars=['Sentiment_index_std']
    )
    print(results['model1'].summary)
    
    # Model 2: Add topic variables
    print("\nModel 2: With Topic Variables")
    results['model2'] = run_fixed_effects_model(
        df,
        dependent_var='RepMargin_pct',
        independent_vars=[
            'Sentiment_index_std',
            'TopicSocial_std',
            'TopicHealth_std',
            'TopicElection_std'
        ]
    )
    print(results['model2'].summary)
    
    # Model 3: Full model with controls
    print("\nModel 3: Full Model with Controls")
    results['model3'] = run_fixed_effects_model(
        df,
        dependent_var='RepMargin_pct',
        independent_vars=[
            'Sentiment_index_std',
            'TopicSocial_std',
            'TopicHealth_std',
            'TopicElection_std',
            'MedianIncome_std',
            'Edu_BA_pct_std'
        ]
    )
    print(results['model3'].summary)
    
    return results


def run_robustness_checks(df):
    """
    Conduct robustness checks on main findings.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel data
        
    Returns
    -------
    dict
        Dictionary of robustness check results
    """
    print("\n" + "="*60)
    print("ROBUSTNESS CHECKS")
    print("="*60)
    
    results = {}
    
    # 1. Year-by-year cross-sectional regressions
    print("\n1. Year-by-Year Cross-Sectional Regressions (Margin Model)")
    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year].copy()
        
        X = sm.add_constant(df_year['Sentiment_index_std'])
        y = df_year['RepMargin_pct']
        
        model = sm.OLS(y, X).fit(cov_type='HC3')
        
        coef = model.params['Sentiment_index_std']
        se = model.bse['Sentiment_index_std']
        pval = model.pvalues['Sentiment_index_std']
        
        print(f"  {year}: β = {coef:.3f} (SE = {se:.3f}), p = {pval:.4f}")
        
        results[f'year_{year}'] = {
            'coefficient': coef,
            'std_error': se,
            'p_value': pval
        }
    
    # 2. Presidential vs Midterm interaction
    print("\n2. Presidential Year Interaction")
    df_interact = df.copy()
    df_interact['presidential'] = df_interact['year'].isin([2020, 2024]).astype(int)
    df_interact['sentiment_x_pres'] = df_interact['Sentiment_index_std'] * df_interact['presidential']
    
    results['interaction'] = run_fixed_effects_model(
        df_interact,
        dependent_var='RepMargin_pct',
        independent_vars=['Sentiment_index_std', 'sentiment_x_pres']
    )
    print(results['interaction'].summary)
    
    # 3. Alternative dependent variable: Absolute margin
    print("\n3. Absolute Margin as Dependent Variable")
    df_abs = df.copy()
    df_abs['AbsMargin_pct'] = df_abs['RepMargin_pct'].abs()
    
    results['abs_margin'] = run_fixed_effects_model(
        df_abs,
        dependent_var='AbsMargin_pct',
        independent_vars=['Sentiment_index_std']
    )
    print(results['abs_margin'].summary)
    
    return results


def extract_results_table(model_results, model_names):
    """
    Extract formatted results table from model results.
    
    Parameters
    ----------
    model_results : dict
        Dictionary of model results
    model_names : list
        Names for each model
        
    Returns
    -------
    pd.DataFrame
        Formatted results table
    """
    rows = []
    
    for name, result in zip(model_names, model_results.values()):
        for param in result.params.index:
            if param != 'const':
                rows.append({
                    'Model': name,
                    'Variable': param,
                    'Coefficient': result.params[param],
                    'Std_Error': result.std_errors[param],
                    'P_Value': result.pvalues[param],
                    'CI_Lower': result.params[param] - 1.96 * result.std_errors[param],
                    'CI_Upper': result.params[param] + 1.96 * result.std_errors[param]
                })
    
    return pd.DataFrame(rows)


def save_results(turnout_results, margin_results, robustness_results, output_dir):
    """
    Save all results to files.
    
    Parameters
    ----------
    turnout_results : dict
        Turnout model results
    margin_results : dict
        Margin model results
    robustness_results : dict
        Robustness check results
    output_dir : str
        Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'model_outputs'), exist_ok=True)
    
    # Save turnout table
    turnout_table = extract_results_table(
        turnout_results, 
        ['Model 1', 'Model 2', 'Model 3']
    )
    turnout_table.to_csv(
        os.path.join(output_dir, 'tables', 'turnout_results.csv'),
        index=False
    )
    
    # Save margin table
    margin_table = extract_results_table(
        margin_results,
        ['Model 1', 'Model 2', 'Model 3']
    )
    margin_table.to_csv(
        os.path.join(output_dir, 'tables', 'margin_results.csv'),
        index=False
    )
    
    # Save model objects
    with open(os.path.join(output_dir, 'model_outputs', 'all_results.pkl'), 'wb') as f:
        pickle.dump({
            'turnout': turnout_results,
            'margin': margin_results,
            'robustness': robustness_results
        }, f)
    
    # Save summary text file
    with open(os.path.join(output_dir, 'regression_summary.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("PANEL REGRESSION RESULTS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("TURNOUT MODEL (Model 3):\n")
        f.write(str(turnout_results['model3'].summary) + "\n\n")
        
        f.write("MARGIN MODEL (Model 3):\n")
        f.write(str(margin_results['model3'].summary) + "\n\n")
    
    print(f"\nResults saved to {output_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Panel regression analysis')
    parser.add_argument('--use-processed', action='store_true', default=True,
                       help='Use pre-processed data')
    args = parser.parse_args()
    
    print("="*60)
    print("GEORGIA ALGORITHMIC FRAMING STUDY")
    print("Panel Regression Analysis")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_panel_data(use_processed=args.use_processed)
    
    # Standardize variables
    print("\nStandardizing variables...")
    variables_to_standardize = [
        'Sentiment_index',
        'TopicSocial_share',
        'TopicHealth_share',
        'TopicElection_share',
        'TopicEcon_positive_share',
        'TopicAttack_share',
        'MedianIncome',
        'Edu_BA_pct'
    ]
    
    # Rename for consistency
    rename_map = {
        'TopicSocial_share_std': 'TopicSocial_std',
        'TopicHealth_share_std': 'TopicHealth_std',
        'TopicElection_share_std': 'TopicElection_std',
        'TopicEcon_positive_share_std': 'TopicEcon_positive_std',
        'TopicAttack_share_std': 'TopicAttack_std'
    }
    
    df = standardize_variables(df, variables_to_standardize)
    df = df.rename(columns=rename_map)
    
    # Run turnout models
    turnout_results = run_turnout_models(df)
    
    # Run margin models
    margin_results = run_margin_models(df)
    
    # Run robustness checks
    robustness_results = run_robustness_checks(df)
    
    # Save all results
    save_results(turnout_results, margin_results, robustness_results, RESULTS_DIR)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
