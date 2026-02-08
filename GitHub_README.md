# Algorithmic Framing and Voter Polarization: A Georgia Panel Study (2018-2024)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

## Overview

This repository contains the data, code, and documentation for replicating the study "Algorithmic Framing and Voter Polarization: A Georgia Panel Study (2018-2024)." The research investigates how algorithmically delivered political advertising content correlates with voter turnout and partisan vote margins across four election cycles in Georgia.

### Research Questions

1. Is the emotional valence (sentiment) of political advertising content associated with changes in voter turnout?
2. Is advertising sentiment associated with changes in partisan vote margins (polarization)?
3. Do specific thematic frames (topics) predict variation in electoral outcomes?
4. Do these relationships differ between presidential and midterm elections?

### Key Findings

- **Sentiment and Polarization**: Counties exposed to more negatively-valenced advertising showed larger partisan vote margins (β = -1.41, p < 0.01)
- **Turnout Effects**: No statistically significant relationship between overall sentiment and turnout
- **Topic Effects**: Social/cultural issue ads associated with increased Republican margins; healthcare/COVID ads associated with Democratic gains
- **Temporal Variation**: Effects strongest during presidential election years (2020, 2024)

## Repository Structure

```
georgia-algorithmic-framing/
│
├── README.md                 # This file
├── APPENDIX.md              # Detailed reproducibility documentation
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
│
├── config/
│   └── config_template.yaml  # Configuration template
│
├── data/
│   ├── raw/                  # Original unprocessed data
│   │   ├── elections/        # Election returns
│   │   ├── ads/              # Advertisement data
│   │   └── acs/              # Census data
│   ├── processed/            # Cleaned and merged data
│   │   ├── panel_dataset.csv # Main analysis dataset
│   │   ├── ad_sentiment.csv  # Ad-level sentiment
│   │   └── ad_topics.csv     # Ad-level topics
│   └── README.md             # Data documentation
│
├── src/
│   ├── data/
│   │   ├── download_election_data.py
│   │   ├── download_ad_data.py
│   │   ├── prepare_acs_data.py
│   │   └── merge_datasets.py
│   ├── preprocessing/
│   │   ├── text_preprocessing.py
│   │   └── preprocess_ads.py
│   ├── analysis/
│   │   ├── sentiment_analysis.py
│   │   ├── topic_modeling.py
│   │   └── panel_regression.py
│   └── visualization/
│       ├── create_tables.py
│       └── create_figures.py
│
├── models/
│   ├── lda_model.pkl         # Trained topic model
│   └── dictionary.pkl        # Gensim dictionary
│
├── results/
│   ├── tables/               # LaTeX tables
│   └── figures/              # PNG/PDF figures
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_topic_modeling.ipynb
│   └── 04_regression_analysis.ipynb
│
└── run_analysis.py           # Main execution script
```

## Quick Start

### Prerequisites

- Python 3.9+
- pip package manager
- 8GB RAM minimum

### Installation

```bash
# Clone the repository
git clone https://github.com/[username]/georgia-algorithmic-framing.git
cd georgia-algorithmic-framing

# Create and activate virtual environment
python3.9 -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### Running the Analysis

**Option 1: Full Pipeline**
```bash
python run_analysis.py --all
```

**Option 2: Step by Step**
```bash
# 1. Preprocess advertisements
python src/preprocessing/preprocess_ads.py

# 2. Run sentiment analysis
python src/analysis/sentiment_analysis.py

# 3. Run topic modeling
python src/analysis/topic_modeling.py

# 4. Run regression analysis
python src/analysis/panel_regression.py

# 5. Generate outputs
python src/visualization/create_tables.py
python src/visualization/create_figures.py
```

## Data

### Data Sources

| Data | Source | Access |
|------|--------|--------|
| Election returns | MIT Election Data Lab, GA Secretary of State | Public |
| Political ads | Meta Ad Library | Public API |
| Demographics | American Community Survey | Census API |

### Key Variables

- `turnout_pct`: Voter turnout (% of registered voters)
- `rep_margin_pct`: Republican - Democratic vote share (pp)
- `sentiment_index`: Impression-weighted mean ad polarity [-1, 1]
- `topic_X_share`: Share of impressions for topic X [0, 1]

## Methods Summary

### Sentiment Analysis
- **Tool**: TextBlob (lexicon-based)
- **Aggregation**: Impression-weighted mean by county-year

### Topic Modeling
- **Model**: Latent Dirichlet Allocation (10 topics)
- **Selection**: Cv coherence optimization

### Regression Analysis
- **Design**: County-year panel (N=636)
- **Model**: Two-way fixed effects (county + year)
- **Inference**: Clustered standard errors by county

## Results Summary

| Outcome | Predictor | β | SE | p |
|---------|-----------|---|----|----|
| Turnout | Sentiment | 0.31 | 0.22 | 0.16 |
| Margin | Sentiment | -1.41 | 0.52 | <0.01 |
| Margin | Social Topics | 0.89 | 0.38 | <0.05 |
| Margin | Health Topics | -0.76 | 0.34 | <0.05 |

## Citation

```bibtex
@article{author2025algorithmic,
  title={Algorithmic Framing and Voter Polarization: A Georgia Panel Study (2018-2024)},
  author={[Author Name]},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- MIT Election Data and Science Lab
- Meta Ad Library
- U.S. Census Bureau

For detailed reproducibility documentation, see [APPENDIX.md](APPENDIX.md)
