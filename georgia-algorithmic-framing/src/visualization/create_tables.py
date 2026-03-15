"""
create_tables.py

Generates summary statistics and descriptive tables for the paper.

Output: results/tables/
  table0_ad_corpus.csv       Advertisement corpus summary by year
  table1_descriptive_stats.csv  County-year panel descriptive statistics
  table1_descriptive_stats.tex  LaTeX version
  table_sentiment_by_year.csv   Sentiment distribution by year (Appendix D.3)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("results/tables")


def load_or_warn(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    return pd.read_csv(path)


def table0_ad_corpus_summary():
    """Summary of advertisement corpus by year (mentioned in §3.1.2)."""
    ads = load_or_warn(PROCESSED_DIR / "ads_preprocessed.csv")
    if ads is None:
        return

    unique_ads = ads.drop_duplicates(subset=["ad_id"]) if "ad_id" in ads.columns else ads

    summary = unique_ads.groupby("year").agg(
        n_ads=("text", "count"),
        total_impressions=("impressions", "sum"),
        mean_impressions=("impressions", "mean"),
        n_pages=("page_name", lambda x: x.nunique()),
    ).reset_index()

    out = TABLES_DIR / "table0_ad_corpus.csv"
    summary.to_csv(out, index=False)
    logger.info(f"Saved {out}\n{summary.to_string()}")


def table1_descriptive_statistics():
    """
    Table 1: County-year panel descriptive statistics.

    Reports N, mean, SD, min, max for all analysis variables.
    """
    panel = load_or_warn(PROCESSED_DIR / "panel_dataset.csv")
    if panel is None:
        return

    analysis_vars = {
        "turnout_pct":           "Voter Turnout (%)",
        "rep_margin_pct":        "Republican Vote Margin (pp)",
        "sentiment_index":       "Sentiment Index",
        "topic_social_share":    "Social Issues Topic Share",
        "topic_health_share":    "Healthcare Topic Share",
        "topic_election_share":  "Election Integrity Topic Share",
        "median_income":         "Median Household Income ($)",
        "pct_bachelors":         "BA Degree Attainment (%)",
        "population":            "County Population",
    }

    rows = []
    for var, label in analysis_vars.items():
        if var not in panel.columns:
            continue
        col = panel[var].dropna()
        rows.append({
            "Variable": label,
            "N": len(col),
            "Mean": col.mean(),
            "SD": col.std(),
            "Min": col.min(),
            "P25": col.quantile(0.25),
            "Median": col.median(),
            "P75": col.quantile(0.75),
            "Max": col.max(),
        })

    stats_df = pd.DataFrame(rows)

    # CSV
    csv_out = TABLES_DIR / "table1_descriptive_stats.csv"
    stats_df.to_csv(csv_out, index=False)
    logger.info(f"Saved {csv_out}")

    # LaTeX
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Descriptive Statistics: County-Year Panel (N\,=\,636 max)}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\hline\hline",
        r"Variable & N & Mean & SD & Min & P25 & Median & P75 & Max \\",
        r"\hline",
    ]

    for _, row in stats_df.iterrows():
        # Format large numbers compactly
        def fmt(x):
            if abs(x) >= 100000:
                return f"{x/1000:.0f}K"
            elif abs(x) >= 1000:
                return f"{x:.0f}"
            elif abs(x) >= 1:
                return f"{x:.2f}"
            else:
                return f"{x:.3f}"

        latex_lines.append(
            f"{row['Variable']} & {int(row['N'])} & {fmt(row['Mean'])} & "
            f"{fmt(row['SD'])} & {fmt(row['Min'])} & {fmt(row['P25'])} & "
            f"{fmt(row['Median'])} & {fmt(row['P75'])} & {fmt(row['Max'])} \\\\"
        )

    latex_lines += [
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_out = TABLES_DIR / "table1_descriptive_stats.tex"
    with open(tex_out, "w") as f:
        f.write("\n".join(latex_lines))
    logger.info(f"Saved {tex_out}")


def table_sentiment_by_year():
    """
    Appendix D.3: Sentiment distribution by year.
    Replicates the table in the paper appendix.
    """
    ads = load_or_warn(PROCESSED_DIR / "ad_sentiment.csv")
    if ads is None:
        return

    unique = ads.drop_duplicates(subset=["ad_id"]) if "ad_id" in ads.columns else ads

    rows = []
    for year, group in unique.groupby("year"):
        pol = group["polarity"].dropna()
        rows.append({
            "Year": year,
            "N Ads": len(pol),
            "Mean Polarity": pol.mean(),
            "SD": pol.std(),
            "% Negative (<0)": (pol < 0).mean() * 100,
            "% Neutral (=0)": (pol == 0).mean() * 100,
            "% Positive (>0)": (pol > 0).mean() * 100,
            "% Highly Negative (<-0.5)": (pol < -0.5).mean() * 100,
        })

    df = pd.DataFrame(rows)
    out = TABLES_DIR / "table_sentiment_by_year.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved {out}\n{df.to_string(index=False)}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    table0_ad_corpus_summary()
    table1_descriptive_statistics()
    table_sentiment_by_year()

    logger.info(f"\nAll tables saved to {TABLES_DIR}/")


if __name__ == "__main__":
    main()
