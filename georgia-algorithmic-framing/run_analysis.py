#!/usr/bin/env python3
"""
run_analysis.py — Main execution script for Georgia Algorithmic Framing Study.

Usage:
    python run_analysis.py --all           # Full pipeline (requires API keys)
    python run_analysis.py --data          # Download all raw data
    python run_analysis.py --preprocess    # Text preprocessing
    python run_analysis.py --analyze       # Sentiment + topic modeling + regression
    python run_analysis.py --visualize     # Tables and figures only
    python run_analysis.py --regression    # Regression only (fastest re-run)

Pipeline order (if running step-by-step):
    1. src/data/download_election_data.py
    2. src/data/prepare_acs_data.py
    3. src/data/download_ad_data.py
    4. src/preprocessing/preprocess_ads.py
    5. src/analysis/sentiment_analysis.py
    6. src/analysis/topic_modeling.py
    7. src/data/merge_datasets.py
    8. src/analysis/panel_regression.py
    9. src/visualization/create_tables.py
   10. src/visualization/create_figures.py
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def create_directories():
    """Ensure all output directories exist."""
    for d in [
        "data/raw/elections", "data/raw/ads", "data/raw/acs",
        "data/processed", "models", "results/tables", "results/figures",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)


def run_data_download():
    logger.info("─── Downloading election data ───")
    from src.data.download_election_data import main as f; f()

    logger.info("─── Downloading ACS data ───")
    from src.data.prepare_acs_data import main as f; f()

    logger.info("─── Downloading and attributing ad data ───")
    from src.data.download_ad_data import main as f; f()


def run_preprocessing():
    logger.info("─── Preprocessing ad text ───")
    from src.preprocessing.preprocess_ads import main as f; f()


def run_sentiment():
    logger.info("─── Running sentiment analysis ───")
    from src.analysis.sentiment_analysis import main as f; f()


def run_topics():
    logger.info("─── Running topic modeling ───")
    from src.analysis.topic_modeling import main as f; f()


def run_merge():
    logger.info("─── Merging datasets ───")
    from src.data.merge_datasets import main as f; f()


def run_regression():
    logger.info("─── Running panel regression ───")
    from src.analysis.panel_regression import main as f; f()


def run_visualization():
    logger.info("─── Creating tables ───")
    from src.visualization.create_tables import main as f; f()

    logger.info("─── Creating figures ───")
    from src.visualization.create_figures import main as f; f()


def run_all():
    logger.info("=" * 60)
    logger.info("GEORGIA ALGORITHMIC FRAMING — FULL PIPELINE")
    logger.info("=" * 60)

    create_directories()
    run_data_download()
    run_preprocessing()
    run_sentiment()
    run_topics()
    run_merge()
    run_regression()
    run_visualization()

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Results: results/tables/ and results/figures/")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Georgia Algorithmic Framing Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all",        action="store_true", help="Run complete pipeline")
    parser.add_argument("--data",       action="store_true", help="Download raw data only")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess ads only")
    parser.add_argument("--analyze",    action="store_true",
                        help="Sentiment + topics + merge + regression")
    parser.add_argument("--regression", action="store_true", help="Regression only")
    parser.add_argument("--visualize",  action="store_true", help="Tables and figures only")

    args = parser.parse_args()

    create_directories()

    if args.all:
        run_all()
    elif args.data:
        run_data_download()
    elif args.preprocess:
        run_preprocessing()
    elif args.analyze:
        run_sentiment()
        run_topics()
        run_merge()
        run_regression()
    elif args.regression:
        run_merge()
        run_regression()
    elif args.visualize:
        run_visualization()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
