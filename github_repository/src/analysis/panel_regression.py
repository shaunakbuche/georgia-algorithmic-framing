"""
Panel Regression Analysis Module

This module implements two-way fixed effects panel regression models
for analyzing the relationship between advertising content and electoral outcomes.
"""

import pandas as pd
import numpy as np
from linearmodels import PanelOLS, RandomEffects
from scipy import stats
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def prepare_panel_data(df: pd.DataFrame,
                       entity_col: str = 'county_fips',
                       time_col: str = 'year') -> pd.DataFrame:
    """
    Prepare data for panel regression analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset with county-year observations
    entity_col : str
        Column identifying panel entities (counties)
    time_col : str
        Column identifying time periods (years)
    
    Returns
    -------
    pd.DataFrame
        Panel-indexed DataFrame with standardized variables
    """
    logger.info("Preparing panel data...")
    
    panel_df = df.copy()
    
    # Set multi-index for panel structure
    panel_df = panel_df.set_index([entity_col, time_col])
    
    # Variables to standardize
    vars_to_standardize = [
        'sentiment_index',
        'topic_social_share',
        'topic_health_share',
        'topic_election_share',
        'median_income',
        'pct_bachelors'
    ]
    
    # Standardize (z-score) continuous variables
    for var in vars_to_standardize:
        if var in panel_df.columns:
            mean = panel_df[var].mean()
            std = panel_df[var].std()
            panel_df[f'{var}_std'] = (panel_df[var] - mean) / std
            logger.info(f"Standardized {var}: mean={mean:.3f}, std={std:.3f}")
    
    logger.info(f"Panel data prepared: {len(panel_df)} observations")
    return panel_df


def estimate_fe_model(panel_df: pd.DataFrame,
                      dep_var: str,
                      indep_vars: list,
                      entity_effects: bool = True,
                      time_effects: bool = True) -> object:
    """
    Estimate two-way fixed effects panel regression.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel-indexed DataFrame
    dep_var : str
        Dependent variable name
    indep_vars : list
        List of independent variable names
    entity_effects : bool
        Include entity (county) fixed effects
    time_effects : bool
        Include time (year) fixed effects
    
    Returns
    -------
    PanelOLS results object
    """
    # Build formula
    formula = f"{dep_var} ~ " + " + ".join(indep_vars)
    
    if entity_effects:
        formula += " + EntityEffects"
    if time_effects:
        formula += " + TimeEffects"
    
    logger.info(f"Estimating model: {formula}")
    
    # Estimate model
    model = PanelOLS.from_formula(formula, data=panel_df)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results


def estimate_re_model(panel_df: pd.DataFrame,
                      dep_var: str,
                      indep_vars: list) -> object:
    """
    Estimate random effects panel regression.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel-indexed DataFrame
    dep_var : str
        Dependent variable name
    indep_vars : list
        List of independent variable names
    
    Returns
    -------
    RandomEffects results object
    """
    formula = f"{dep_var} ~ 1 + " + " + ".join(indep_vars)
    
    logger.info(f"Estimating RE model: {formula}")
    
    model = RandomEffects.from_formula(formula, data=panel_df)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results


def hausman_test(fe_results, re_results) -> dict:
    """
    Conduct Hausman specification test.
    
    Parameters
    ----------
    fe_results : PanelOLS results
        Fixed effects model results
    re_results : RandomEffects results
        Random effects model results
    
    Returns
    -------
    dict
        Test results with statistic, df, and p-value
    """
    # Get coefficients (excluding constant and effects)
    fe_params = fe_results.params
    re_params = re_results.params
    
    # Find common parameters
    common = fe_params.index.intersection(re_params.index)
    common = [c for c in common if c != 'Intercept']
    
    if len(common) == 0:
        return {'statistic': np.nan, 'df': 0, 'p_value': np.nan}
    
    # Calculate difference
    b_diff = fe_params[common] - re_params[common]
    
    # Variance of difference (simplified)
    v_fe = np.diag(fe_results.cov.loc[common, common])
    v_re = np.diag(re_results.cov.loc[common, common])
    v_diff = v_fe - v_re
    
    # Hausman statistic
    h_stat = np.sum(b_diff**2 / np.abs(v_diff))
    df = len(common)
    p_value = 1 - stats.chi2.cdf(h_stat, df)
    
    return {
        'statistic': h_stat,
        'df': df,
        'p_value': p_value
    }


def format_results_table(results, model_name: str = 'Model') -> pd.DataFrame:
    """
    Format regression results as a table.
    
    Parameters
    ----------
    results : regression results object
    model_name : str
        Name for the model column
    
    Returns
    -------
    pd.DataFrame
        Formatted results table
    """
    table = pd.DataFrame({
        'Variable': results.params.index,
        'Coefficient': results.params.values,
        'Std. Error': results.std_errors.values,
        't-stat': results.tstats.values,
        'p-value': results.pvalues.values
    })
    
    # Add significance stars
    def add_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''
    
    table['Sig.'] = table['p-value'].apply(add_stars)
    
    return table


def main():
    """Main execution function."""
    # Load panel data
    data_path = Path('data/processed/panel_dataset.csv')
    
    if not data_path.exists():
        logger.error(f"Panel dataset not found at {data_path}")
        return
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare panel data
    panel_df = prepare_panel_data(df)
    
    # Define independent variables
    indep_vars = [
        'sentiment_index_std',
        'topic_social_share_std',
        'topic_health_share_std',
        'topic_election_share_std',
        'median_income_std',
        'pct_bachelors_std'
    ]
    
    # Filter to available variables
    indep_vars = [v for v in indep_vars if v in panel_df.columns]
    
    # Model 1: Turnout
    logger.info("\n" + "="*50)
    logger.info("MODEL 1: VOTER TURNOUT")
    logger.info("="*50)
    
    if 'turnout_pct' in panel_df.columns:
        turnout_fe = estimate_fe_model(panel_df, 'turnout_pct', indep_vars)
        print(turnout_fe.summary)
        
        # Save results
        turnout_table = format_results_table(turnout_fe, 'Turnout FE')
        turnout_table.to_csv('results/tables/turnout_regression.csv', index=False)
    
    # Model 2: Vote Margin
    logger.info("\n" + "="*50)
    logger.info("MODEL 2: PARTISAN VOTE MARGIN")
    logger.info("="*50)
    
    if 'rep_margin_pct' in panel_df.columns:
        margin_fe = estimate_fe_model(panel_df, 'rep_margin_pct', indep_vars)
        print(margin_fe.summary)
        
        # Save results
        margin_table = format_results_table(margin_fe, 'Margin FE')
        margin_table.to_csv('results/tables/margin_regression.csv', index=False)
        
        # Hausman test
        margin_re = estimate_re_model(panel_df, 'rep_margin_pct', indep_vars)
        hausman = hausman_test(margin_fe, margin_re)
        logger.info(f"\nHausman test: chi2={hausman['statistic']:.2f}, "
                   f"df={hausman['df']}, p={hausman['p_value']:.4f}")
    
    logger.info("\nRegression analysis complete. Results saved to results/tables/")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
