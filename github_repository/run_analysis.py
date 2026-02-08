#!/usr/bin/env python3
"""
Main execution script for Georgia Algorithmic Framing Study.

Usage:
    python run_analysis.py --all           # Run complete pipeline
    python run_analysis.py --preprocess    # Run preprocessing only
    python run_analysis.py --analyze       # Run analysis only
    python run_analysis.py --visualize     # Generate outputs only
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_preprocessing():
    """Run text preprocessing pipeline."""
    logger.info("Starting preprocessing...")
    from src.preprocessing.preprocess_ads import main as preprocess_main
    preprocess_main()
    logger.info("Preprocessing complete.")

def run_sentiment_analysis():
    """Run sentiment analysis."""
    logger.info("Starting sentiment analysis...")
    from src.analysis.sentiment_analysis import main as sentiment_main
    sentiment_main()
    logger.info("Sentiment analysis complete.")

def run_topic_modeling():
    """Run topic modeling."""
    logger.info("Starting topic modeling...")
    from src.analysis.topic_modeling import main as topic_main
    topic_main()
    logger.info("Topic modeling complete.")

def run_regression():
    """Run panel regression analysis."""
    logger.info("Starting regression analysis...")
    from src.analysis.panel_regression import main as regression_main
    regression_main()
    logger.info("Regression analysis complete.")

def run_visualization():
    """Generate tables and figures."""
    logger.info("Generating outputs...")
    from src.visualization.create_tables import main as tables_main
    from src.visualization.create_figures import main as figures_main
    tables_main()
    figures_main()
    logger.info("Output generation complete.")

def run_all():
    """Run complete analysis pipeline."""
    logger.info("="*60)
    logger.info("STARTING COMPLETE ANALYSIS PIPELINE")
    logger.info("="*60)
    
    run_preprocessing()
    run_sentiment_analysis()
    run_topic_modeling()
    run_regression()
    run_visualization()
    
    logger.info("="*60)
    logger.info("ANALYSIS PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info("Results saved to: results/")
    logger.info("Models saved to: models/")

def main():
    parser = argparse.ArgumentParser(
        description='Georgia Algorithmic Framing Study Analysis'
    )
    parser.add_argument('--all', action='store_true',
                       help='Run complete analysis pipeline')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run preprocessing only')
    parser.add_argument('--sentiment', action='store_true',
                       help='Run sentiment analysis only')
    parser.add_argument('--topics', action='store_true',
                       help='Run topic modeling only')
    parser.add_argument('--regression', action='store_true',
                       help='Run regression analysis only')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate outputs only')
    
    args = parser.parse_args()
    
    # Create output directories
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(parents=True, exist_ok=True)
    
    if args.all:
        run_all()
    elif args.preprocess:
        run_preprocessing()
    elif args.sentiment:
        run_sentiment_analysis()
    elif args.topics:
        run_topic_modeling()
    elif args.regression:
        run_regression()
    elif args.visualize:
        run_visualization()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
