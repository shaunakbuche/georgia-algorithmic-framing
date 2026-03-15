"""
create_figures.py

Generate all paper figures:
  Figure 1: Sentiment trends over time (line plot)
  Figure 2: Topic prevalence by year (stacked bar)
  Figure 3: Sentiment × margin scatter by county and year
  Figure 4: Coherence scores for topic number selection (optional)

Input:  data/processed/
Output: results/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Publication-quality settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGURES_DIR = Path("results/figures")
PROCESSED_DIR = Path("data/processed")

YEAR_LABELS = {2018: "2018\n(Midterm)", 2020: "2020\n(Presidential)",
               2022: "2022\n(Midterm)", 2024: "2024\n(Presidential)"}


def load_or_warn(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    return pd.read_csv(path)


def figure1_sentiment_trends():
    """
    Figure 1: Mean ad sentiment polarity by election year.
    Shows the increasing negativity trend documented in §4.1.1.
    """
    sentiment = load_or_warn(PROCESSED_DIR / "county_sentiment.csv")
    if sentiment is None:
        return

    yearly = sentiment.groupby("year").agg(
        mean_sentiment=("sentiment_index", "mean"),
        se_sentiment=("sentiment_index", lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(
        yearly["year"],
        yearly["mean_sentiment"],
        yerr=yearly["se_sentiment"] * 1.96,
        marker="o",
        markersize=8,
        linewidth=2,
        capsize=4,
        color="#2c5f8a",
        label="Mean sentiment (±95% CI)",
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xticks([2018, 2020, 2022, 2024])
    ax.set_xticklabels([YEAR_LABELS[y] for y in [2018, 2020, 2022, 2024]])
    ax.set_ylabel("Mean Impression-Weighted Sentiment Polarity")
    ax.set_title("Figure 1. Political Advertisement Sentiment by Election Year")
    ax.legend(framealpha=0.3)
    ax.set_ylim(-0.30, 0.10)

    # Annotate presidential years
    for year in [2020, 2024]:
        ax.axvspan(year - 0.3, year + 0.3, alpha=0.07, color="orange",
                   label="Presidential year" if year == 2020 else "")

    plt.tight_layout()
    out_path = FIGURES_DIR / "figure1_sentiment_trends.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved {out_path}")


def figure2_topic_prevalence():
    """
    Figure 2: Impression-weighted topic prevalence by election year (stacked bar).
    Shows shifts in campaign messaging documented in §4.1.2.
    """
    topics = load_or_warn(PROCESSED_DIR / "county_topics.csv")
    if topics is None:
        return

    # Identify topic share columns
    topic_cols = [c for c in topics.columns if c.startswith("topic_") and c.endswith("_share")
                  and not c.startswith("topic_social") and not c.startswith("topic_health")
                  and not c.startswith("topic_election")]

    if not topic_cols:
        logger.warning("No topic_N_share columns found in county_topics.csv")
        return

    yearly_means = topics.groupby("year")[topic_cols].mean().reset_index()

    # Rename columns to labels from topic_labels.json if available
    labels_path = Path("models/topic_labels.json")
    if labels_path.exists():
        import json
        with open(labels_path) as f:
            labels = {f"topic_{k}_share": v for k, v in json.load(f).items()}
        yearly_means = yearly_means.rename(columns=labels)
        topic_cols = [labels.get(c, c) for c in topic_cols]

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(topic_cols)))
    bottom = np.zeros(len(yearly_means))

    for col, color in zip(topic_cols, colors):
        if col in yearly_means.columns:
            ax.bar(yearly_means["year"], yearly_means[col],
                   bottom=bottom, color=color, label=col, width=0.6)
            bottom += yearly_means[col].values

    ax.set_xticks([2018, 2020, 2022, 2024])
    ax.set_xticklabels([YEAR_LABELS[y] for y in [2018, 2020, 2022, 2024]])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Share of Attributed Impressions")
    ax.set_title("Figure 2. Topic Prevalence by Election Year")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "figure2_topic_prevalence.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved {out_path}")


def figure3_sentiment_margin_scatter():
    """
    Figure 3: Scatter of county sentiment index vs. partisan vote margin.
    Faceted by year; visualizes the β = -1.41 relationship.
    """
    panel = load_or_warn(PROCESSED_DIR / "panel_dataset.csv")
    if panel is None:
        return

    if "sentiment_index" not in panel.columns or "rep_margin_pct" not in panel.columns:
        logger.warning("Required columns not in panel_dataset.csv")
        return

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

    for ax, year in zip(axes, [2018, 2020, 2022, 2024]):
        year_data = panel[panel["year"] == year].dropna(
            subset=["sentiment_index", "rep_margin_pct"]
        )

        ax.scatter(
            year_data["sentiment_index"],
            year_data["rep_margin_pct"],
            alpha=0.45, s=18, color="#2c5f8a",
        )

        # OLS trend line
        if len(year_data) > 5:
            coeffs = np.polyfit(year_data["sentiment_index"],
                                year_data["rep_margin_pct"], 1)
            x_line = np.linspace(year_data["sentiment_index"].min(),
                                  year_data["sentiment_index"].max(), 50)
            ax.plot(x_line, np.polyval(coeffs, x_line),
                    color="firebrick", linewidth=1.5)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_title(YEAR_LABELS[year], fontsize=10)
        ax.set_xlabel("Sentiment Index", fontsize=9)

    axes[0].set_ylabel("Republican Margin (pp)")
    fig.suptitle(
        "Figure 3. County Sentiment vs. Partisan Vote Margin by Year",
        y=1.02, fontsize=12
    )

    plt.tight_layout()
    out_path = FIGURES_DIR / "figure3_sentiment_margin.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved {out_path}")


def figure4_coherence_scores():
    """
    Figure 4: Topic coherence by K (if coherence CSV exists).
    Validates the K=10 selection documented in §3.2.3.
    """
    coherence_path = Path("results/tables/topic_coherence.csv")
    if not coherence_path.exists():
        logger.info("Coherence CSV not found; skipping Figure 4")
        return

    coherence_df = pd.read_csv(coherence_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coherence_df["k"], coherence_df["coherence"],
            marker="o", color="#2c5f8a", linewidth=2)
    ax.axvline(10, color="firebrick", linestyle="--",
               linewidth=1.2, label="K=10 (selected)")
    ax.set_xlabel("Number of Topics (K)")
    ax.set_ylabel("Cv Coherence Score")
    ax.set_title("Figure 4. LDA Topic Coherence by K")
    ax.legend()

    plt.tight_layout()
    out_path = FIGURES_DIR / "figure4_coherence.png"
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved {out_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figure1_sentiment_trends()
    figure2_topic_prevalence()
    figure3_sentiment_margin_scatter()
    figure4_coherence_scores()

    logger.info(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
