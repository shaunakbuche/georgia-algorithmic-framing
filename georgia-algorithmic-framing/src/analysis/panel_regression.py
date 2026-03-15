"""
panel_regression.py

Two-way fixed effects panel regression analysis.

Fixes from original repo:
  1. Hausman test uses correct matrix-inverse formula (not np.abs() patch)
  2. First-differences robustness check implemented
  3. Interaction term (sentiment × presidential year) fully specified
  4. Sensitivity analysis: geographic-targeting-only sample

Models estimated:
  1. Voter turnout ~ sentiment + topics + controls + FE
  2. Partisan vote margin ~ sentiment + topics + controls + FE
  3. Robustness: random effects, first differences, lagged DV, interaction

Input:  data/processed/panel_dataset.csv
Output: results/tables/
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2
from linearmodels import PanelOLS, RandomEffects, FirstDifferenceOLS
from pathlib import Path
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_panel_data(df: pd.DataFrame,
                       entity_col: str = "county_fips",
                       time_col: str = "year") -> pd.DataFrame:
    """
    Set multi-index and standardize continuous variables.

    All continuous predictors are z-scored so coefficients are
    interpretable as "effect of one standard deviation change."
    """
    panel_df = df.copy()
    panel_df = panel_df.set_index([entity_col, time_col])

    vars_to_standardize = [
        "sentiment_index",
        "topic_social_share",
        "topic_health_share",
        "topic_election_share",
        "median_income",
        "pct_bachelors",
    ]

    for var in vars_to_standardize:
        if var in panel_df.columns:
            mean = panel_df[var].mean()
            std = panel_df[var].std()
            if std > 0:
                panel_df[f"{var}_std"] = (panel_df[var] - mean) / std
            else:
                panel_df[f"{var}_std"] = 0
            logger.info(f"  Standardized {var}: μ={mean:.3f}, σ={std:.3f}")

    logger.info(f"Panel prepared: {len(panel_df)} observations")
    return panel_df


def get_available_vars(panel_df: pd.DataFrame, vars: list) -> list:
    """Return only variables that exist in the DataFrame."""
    available = [v for v in vars if v in panel_df.columns]
    missing = [v for v in vars if v not in panel_df.columns]
    if missing:
        logger.warning(f"Variables not found, skipping: {missing}")
    return available


# ---------------------------------------------------------------------------
# Model estimation
# ---------------------------------------------------------------------------

def estimate_fe_model(panel_df: pd.DataFrame,
                      dep_var: str,
                      indep_vars: list,
                      entity_effects: bool = True,
                      time_effects: bool = True):
    """
    Estimate two-way fixed effects OLS panel regression.

    Errors clustered at entity (county) level to account for
    within-county serial correlation.
    """
    indep_vars = get_available_vars(panel_df, indep_vars)
    formula = f"{dep_var} ~ " + " + ".join(indep_vars)

    if entity_effects:
        formula += " + EntityEffects"
    if time_effects:
        formula += " + TimeEffects"

    logger.info(f"Estimating: {formula}")

    model = PanelOLS.from_formula(formula, data=panel_df)
    results = model.fit(cov_type="clustered", cluster_entity=True)
    return results


def estimate_re_model(panel_df: pd.DataFrame,
                      dep_var: str,
                      indep_vars: list):
    """Estimate random effects model (for Hausman test comparison)."""
    indep_vars = get_available_vars(panel_df, indep_vars)
    formula = f"{dep_var} ~ 1 + " + " + ".join(indep_vars)
    logger.info(f"Estimating RE: {formula}")

    model = RandomEffects.from_formula(formula, data=panel_df)
    results = model.fit(cov_type="clustered", cluster_entity=True)
    return results


def estimate_fd_model(panel_df: pd.DataFrame,
                      dep_var: str,
                      indep_vars: list):
    """
    Estimate first-differences model as robustness check.

    Differences out county fixed effects; estimates within-county
    year-to-year changes rather than levels.
    """
    indep_vars = get_available_vars(panel_df, indep_vars)
    formula = f"{dep_var} ~ " + " + ".join(indep_vars)
    logger.info(f"Estimating FD: {formula}")

    model = FirstDifferenceOLS.from_formula(formula, data=panel_df)
    results = model.fit(cov_type="clustered", cluster_entity=True)
    return results


# ---------------------------------------------------------------------------
# Hausman specification test — CORRECTED implementation
# ---------------------------------------------------------------------------

def hausman_test(fe_results, re_results) -> dict:
    """
    Hausman specification test: FE vs RE.

    H0: RE estimator is consistent (random effects are uncorrelated with X)
    H1: FE estimator is consistent; RE is not (use FE)

    The correct statistic is:
        H = (b_FE - b_RE)' [V_FE - V_RE]^{-1} (b_FE - b_RE) ~ χ²(k)

    NOTE: V_FE - V_RE must be positive semi-definite for this to work.
    In finite samples it sometimes isn't (especially with clustered SEs),
    in which case the test is unreliable. We check and warn.

    This fixes the original implementation which used np.abs(v_diff),
    an invalid patch that produced arbitrary test statistics.
    """
    fe_params = fe_results.params
    re_params = re_results.params

    common = fe_params.index.intersection(re_params.index)
    common = [c for c in common if c not in ("Intercept", "const")]

    if len(common) == 0:
        logger.warning("No common parameters for Hausman test")
        return {"statistic": np.nan, "df": 0, "p_value": np.nan, "valid": False}

    b_fe = fe_params[common].values
    b_re = re_params[common].values
    b_diff = b_fe - b_re

    V_fe = fe_results.cov.loc[common, common].values
    V_re = re_results.cov.loc[common, common].values
    V_diff = V_fe - V_re

    # Check positive semi-definiteness of V_diff
    eigenvalues = np.linalg.eigvalsh(V_diff)
    is_psd = bool(np.all(eigenvalues >= -1e-10))

    if not is_psd:
        logger.warning(
            "V_FE - V_RE is not positive semi-definite. "
            "Hausman statistic may be unreliable. "
            "This is common with clustered standard errors. "
            "Interpret with caution."
        )

    try:
        V_diff_inv = np.linalg.inv(V_diff)
        h_stat = float(b_diff @ V_diff_inv @ b_diff)
    except np.linalg.LinAlgError:
        logger.warning("V_diff is singular; cannot invert. Using pseudoinverse.")
        V_diff_inv = np.linalg.pinv(V_diff)
        h_stat = float(b_diff @ V_diff_inv @ b_diff)

    df = len(common)
    p_value = float(1 - chi2.cdf(h_stat, df))

    return {
        "statistic": h_stat,
        "df": df,
        "p_value": p_value,
        "valid": is_psd,
    }


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def format_results_table(results, model_name: str = "Model") -> pd.DataFrame:
    """Format regression results as a clean DataFrame with significance stars."""
    table = pd.DataFrame({
        "Variable": results.params.index,
        "Coefficient": results.params.values,
        "Std. Error": results.std_errors.values,
        "t-stat": results.tstats.values,
        "p-value": results.pvalues.values,
    })

    def stars(p: float) -> str:
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.10:  return "†"
        return ""

    table["Sig."] = table["p-value"].apply(stars)
    table["Model"] = model_name
    return table


def format_latex_table(results_df: pd.DataFrame, title: str = "") -> str:
    """
    Generate a LaTeX table from a formatted results DataFrame.
    Suitable for direct inclusion in the paper.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{title}}}",
        r"\begin{tabular}{lcccc}",
        r"\hline\hline",
        r"Variable & Coefficient & Std. Error & t-stat & p-value \\",
        r"\hline",
    ]

    for _, row in results_df.iterrows():
        sig = row.get("Sig.", "")
        lines.append(
            f"{row['Variable']} & {row['Coefficient']:.3f}{sig} & "
            f"{row['Std. Error']:.3f} & {row['t-stat']:.2f} & "
            f"{row['p-value']:.3f} \\\\"
        )

    lines += [
        r"\hline\hline",
        r"\multicolumn{5}{l}{\footnotesize{$^\dagger$p<0.10, *p<0.05, **p<0.01, ***p<0.001}} \\",
        r"\multicolumn{5}{l}{\footnotesize{Standard errors clustered by county.}} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_path = Path("data/processed/panel_dataset.csv")
    if not data_path.exists():
        logger.error(
            f"Panel dataset not found at {data_path}\n"
            "Run src/data/merge_datasets.py first."
        )
        return

    df = pd.read_csv(data_path)
    logger.info(f"Loaded panel: {df.shape}")

    panel_df = prepare_panel_data(df)

    # Core independent variables
    indep_vars = [
        "sentiment_index_std",
        "topic_social_share_std",
        "topic_health_share_std",
        "topic_election_share_std",
        "median_income_std",
        "pct_bachelors_std",
    ]

    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # MODEL 1: Voter Turnout
    # ----------------------------------------------------------------
    logger.info("\n" + "="*55)
    logger.info("MODEL 1: VOTER TURNOUT")
    logger.info("="*55)

    if "turnout_pct" in panel_df.columns:
        turnout_fe = estimate_fe_model(panel_df, "turnout_pct", indep_vars)
        print(turnout_fe.summary)

        table = format_results_table(turnout_fe, "Turnout — Two-way FE")
        table.to_csv(tables_dir / "turnout_regression.csv", index=False)

        latex = format_latex_table(table, "Fixed Effects Regression: Voter Turnout")
        with open(tables_dir / "turnout_regression.tex", "w") as f:
            f.write(latex)

    # ----------------------------------------------------------------
    # MODEL 2: Partisan Vote Margin
    # ----------------------------------------------------------------
    logger.info("\n" + "="*55)
    logger.info("MODEL 2: PARTISAN VOTE MARGIN")
    logger.info("="*55)

    if "rep_margin_pct" in panel_df.columns:
        margin_fe = estimate_fe_model(panel_df, "rep_margin_pct", indep_vars)
        print(margin_fe.summary)

        table = format_results_table(margin_fe, "Margin — Two-way FE")
        table.to_csv(tables_dir / "margin_regression.csv", index=False)

        latex = format_latex_table(table, "Fixed Effects Regression: Partisan Vote Margin")
        with open(tables_dir / "margin_regression.tex", "w") as f:
            f.write(latex)

        # ---- Hausman test (FE vs RE) ----
        logger.info("\nRunning Hausman test...")
        margin_re = estimate_re_model(panel_df, "rep_margin_pct", indep_vars)
        hausman = hausman_test(margin_fe, margin_re)

        logger.info(
            f"Hausman test: χ²={hausman['statistic']:.2f}, "
            f"df={hausman['df']}, p={hausman['p_value']:.4f} "
            f"({'valid' if hausman['valid'] else 'WARNING: V_diff not PSD'})"
        )

        hausman_df = pd.DataFrame([hausman])
        hausman_df.to_csv(tables_dir / "hausman_test.csv", index=False)

    # ----------------------------------------------------------------
    # ROBUSTNESS CHECK: Presidential vs Midterm interaction
    # ----------------------------------------------------------------
    logger.info("\n" + "="*55)
    logger.info("ROBUSTNESS: PRESIDENTIAL YEAR INTERACTION")
    logger.info("="*55)

    if "is_presidential" in panel_df.columns and "rep_margin_pct" in panel_df.columns:
        # Create interaction term
        panel_df["sentiment_x_presidential"] = (
            panel_df["sentiment_index_std"] * panel_df["is_presidential"]
        )

        interaction_vars = indep_vars + [
            "is_presidential",
            "sentiment_x_presidential",
        ]

        try:
            interaction_fe = estimate_fe_model(
                panel_df, "rep_margin_pct", interaction_vars,
                time_effects=False  # is_presidential collinear with year FE if both included
            )
            print(interaction_fe.summary)

            table = format_results_table(interaction_fe, "Margin — Interaction")
            table.to_csv(tables_dir / "interaction_regression.csv", index=False)
        except Exception as e:
            logger.warning(f"Interaction model failed: {e}")

    # ----------------------------------------------------------------
    # ROBUSTNESS CHECK: First differences
    # ----------------------------------------------------------------
    logger.info("\n" + "="*55)
    logger.info("ROBUSTNESS: FIRST DIFFERENCES")
    logger.info("="*55)

    if "rep_margin_pct" in panel_df.columns:
        try:
            fd_results = estimate_fd_model(panel_df, "rep_margin_pct", indep_vars)
            print(fd_results.summary)

            table = format_results_table(fd_results, "Margin — First Differences")
            table.to_csv(tables_dir / "first_diff_regression.csv", index=False)
        except Exception as e:
            logger.warning(f"First differences model failed: {e}")

    # ----------------------------------------------------------------
    # ROBUSTNESS CHECK: Sensitivity — geographic-targeting-only sample
    # ----------------------------------------------------------------
    logger.info("\n" + "="*55)
    logger.info("ROBUSTNESS: GEOGRAPHIC-TARGETING ONLY SAMPLE")
    logger.info("="*55)

    # This check addresses the measurement error concern in the paper (§6):
    # statewide-allocated ads introduce noise that biases toward null.
    # Re-running on counties with predominantly targeted-ad data tests sensitivity.
    if "pct_statewide" in df.columns and "rep_margin_pct" in panel_df.columns:
        geo_panel = panel_df[panel_df.get("pct_statewide", pd.Series(1)) < 0.5]

        if len(geo_panel) > 0:
            logger.info(f"Geographic-only subsample: {len(geo_panel)} observations")
            try:
                geo_results = estimate_fe_model(
                    geo_panel, "rep_margin_pct", indep_vars
                )
                table = format_results_table(geo_results, "Margin — Geo-targeted only")
                table.to_csv(tables_dir / "geo_targeted_regression.csv", index=False)
                logger.info("Sentiment coefficient in geo-targeted sample: "
                            f"{geo_results.params.get('sentiment_index_std', 'N/A'):.3f}")
            except Exception as e:
                logger.warning(f"Geographic sensitivity check failed: {e}")

    logger.info(
        f"\nAll results saved to {tables_dir}/\n"
        "Tables: turnout_regression.csv/tex, margin_regression.csv/tex,\n"
        "        interaction_regression.csv, first_diff_regression.csv,\n"
        "        hausman_test.csv"
    )


if __name__ == "__main__":
    main()
