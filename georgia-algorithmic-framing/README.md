# Algorithmic Framing and Voter Polarization: A Georgia Panel Study (2018–2024)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)

Reproducible analysis pipeline for *"Algorithmic Framing and Voter Polarization: A Georgia Panel Study (2018–2024)."* The study investigates associations between algorithmically delivered political advertising content and electoral outcomes across four Georgia election cycles.

---

## Key Findings

| Outcome | Predictor | β | SE | p |
|---------|-----------|---|----|---|
| Voter Turnout | Sentiment Index | 0.31 | 0.22 | 0.16 (n.s.) |
| Partisan Margin | Sentiment Index | −1.41 | 0.52 | <0.01 |
| Partisan Margin | Social/Cultural Topics | 0.89 | 0.38 | <0.05 |
| Partisan Margin | Healthcare/COVID Topics | −0.76 | 0.34 | <0.05 |

Two-way fixed effects (county + year), standard errors clustered by county, N ≤ 636 county-years.

---

## Repository Structure

```
georgia-algorithmic-framing/
│
├── README.md                    ← This file
├── APPENDIX.md                  ← Full reproducibility documentation
├── LICENSE                      ← MIT
├── requirements.txt             ← Python dependencies
├── pytest.ini                   ← Test configuration
├── run_analysis.py              ← Main pipeline orchestrator
│
├── config/
│   └── config_template.yaml    ← Copy to config.yaml, add API keys
│
├── src/
│   ├── data/
│   │   ├── download_election_data.py  ← MEDSL + GA SoS election returns
│   │   ├── prepare_acs_data.py        ← Census ACS socioeconomic data
│   │   ├── download_ad_data.py        ← Meta Ad Library + county attribution
│   │   └── merge_datasets.py          ← Build final panel
│   ├── preprocessing/
│   │   └── preprocess_ads.py          ← Text cleaning, tokenization, lemmatization
│   ├── analysis/
│   │   ├── sentiment_analysis.py      ← VADER + TextBlob scoring, aggregation
│   │   ├── topic_modeling.py          ← LDA with post-training label assignment
│   │   └── panel_regression.py        ← Two-way FE, Hausman, robustness checks
│   └── visualization/
│       ├── create_figures.py          ← Figures 1–4
│       └── create_tables.py           ← Tables 0–1 + appendix tables
│
├── data/
│   ├── raw/
│   │   ├── elections/  ← Manual download required (see data/raw/elections/README.md)
│   │   ├── ads/        ← Populated by download_ad_data.py
│   │   └── acs/        ← Populated by prepare_acs_data.py
│   └── processed/      ← Analysis-ready files (tracked in git)
│       └── README.md   ← Variable codebook
│
├── models/                      ← Trained LDA model (gitignored, regenerate locally)
├── results/
│   ├── tables/                  ← CSV + LaTeX regression tables
│   └── figures/                 ← PNG figures
│
├── notebooks/                   ← Jupyter exploration notebooks
└── tests/                       ← pytest unit tests
    ├── test_sentiment.py
    ├── test_regression.py
    ├── test_topic_modeling.py
    └── test_preprocessing.py
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/shaunakbuche/georgia-algorithmic-framing.git
cd georgia-algorithmic-framing

python3.9 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
```

### 2. Configure API keys

```bash
cp config/config_template.yaml config/config.yaml
# Open config/config.yaml and add:
#   census_api_key: "your_key"        (free: api.census.gov/data/key_signup.html)
#   meta_access_token: "your_token"   (researcher access: facebook.com/ads/library/api/)
```

### 3. Collect data

```bash
python run_analysis.py --data
```

This downloads ACS data automatically. For election returns, follow the instructions printed to console (manual download from GA Secretary of State required for gubernatorial years).

### 4. Run analysis

```bash
# Full pipeline
python run_analysis.py --all

# Or step by step
python run_analysis.py --preprocess   # Text preprocessing
python run_analysis.py --analyze      # Sentiment + topics + regression
python run_analysis.py --visualize    # Tables and figures
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Data Sources

| Data | Source | Access | Notes |
|------|--------|--------|-------|
| Election returns (2020, 2024) | [MIT Election Data Lab](https://electionlab.mit.edu/data) | Public | Auto-downloaded |
| Election returns (2018, 2022) | [GA Secretary of State](https://results.enr.clarityelections.com/GA/) | Public | Manual download |
| Political advertisements | [Meta Ad Library](https://www.facebook.com/ads/library/) | Researcher API | ~2,147 ads |
| Socioeconomic controls | [ACS 5-Year Estimates](https://api.census.gov/) | Census API | Auto-downloaded |

---

## Methods Summary

### County Attribution Algorithm
Advertisements are attributed to counties in three tiers:
1. **City-level targeting** → direct FIPS assignment via city→county lookup
2. **DMA-level targeting** → population-proportional within DMA boundaries
3. **Statewide/no targeting** → population-proportional across all 159 counties

Statewide-allocated impressions are flagged (`statewide_allocation=True`) for sensitivity analysis.

### Sentiment Analysis
- **Primary:** VADER (`vaderSentiment`) — designed for social media text; handles capitalization, punctuation, slang
- **Comparison:** TextBlob — general-purpose lexicon method; reported in paper for transparency
- **Aggregation:** Impression-weighted mean polarity per county-year: *S_ct = Σ(s_i × w_i) / Σ(w_i)*

### Topic Modeling
- LDA via Gensim (K=10, selected by Cv coherence across K ∈ {5,8,10,12,15})
- Labels assigned **post-training** via manual inspection (not pre-coded)
- Three composite categories: Social/Cultural Issues, Healthcare/COVID-19, Election Integrity

### Panel Regression
- Two-way fixed effects OLS (county + year)
- Standard errors clustered by county
- Hausman test using correct matrix-inverse formula: *H = b_diff' [V_FE − V_RE]⁻¹ b_diff*
- Robustness checks: random effects, first differences, lagged DV, interaction terms, geographic-targeting subsample

---

## Reproducing the Paper's Results

Key statistics to verify:

| Statistic | Expected | Tolerance |
|-----------|----------|-----------|
| Panel observations (N) | 636 | Exact |
| Mean ad sentiment, 2020 | −0.15 | ±0.02 |
| Sentiment→Margin β | −1.41 | ±0.15 |
| Hausman χ² | 24.7 | ±2.0 |
| Turnout model R² (within) | ~0.89 | ±0.03 |

> **Note on topic model replication:** LDA results may vary slightly across runs due to stochastic initialization. The paper's results used `random_state=42`. If your K=10 model produces topics in a different order, update `PAPER_TOPIC_LABELS` in `src/analysis/topic_modeling.py` after inspecting `print_topics()` output.

---

## Methodological Notes

**Why VADER instead of TextBlob?**
TextBlob was trained on movie reviews and performs inconsistently on political social media text. VADER was specifically designed for short, informal text and handles the rhetorical features common in political advertising (capitalization, punctuation intensity, common political phrases). See `compare_methods()` in `sentiment_analysis.py` for an empirical comparison on this corpus.

**The county imputation limitation:**
Approximately 40–60% of impressions are statewide-allocated (not explicitly targeted to specific counties). This introduces measurement error that biases regression estimates toward null — it makes the reported effects *conservative*, not inflated. See paper §6 and the geographic-targeting sensitivity check in `panel_regression.py`.

**Hausman test with clustered standard errors:**
The corrected Hausman implementation includes a positive semi-definiteness check on V_FE − V_RE. With clustered standard errors, this matrix sometimes fails the PSD check, making the test unreliable. A warning is issued when this occurs. The paper's reported χ² = 24.7 used unclustered standard errors for the Hausman test specifically.

---

## Citation

```bibtex
@article{buche2025algorithmic,
  title   = {Algorithmic Framing and Voter Polarization: A Georgia Panel Study (2018--2024)},
  author  = {Buche, Shaunak},
  year    = {2025},
  note    = {Data and code: \url{https://github.com/shaunakbuche/georgia-algorithmic-framing}}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

MIT Election Data and Science Lab · Meta Ad Library · U.S. Census Bureau
