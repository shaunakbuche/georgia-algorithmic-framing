# Appendix: Reproducibility Documentation

## Algorithmic Framing and Voter Polarization: A Georgia Panel Study (2018–2024)

---

## Table of Contents

1. [Software Environment](#a-software-environment)
2. [Data Sources and Collection](#b-data-sources)
3. [County Attribution Algorithm](#c-county-attribution)
4. [Text Preprocessing](#d-text-preprocessing)
5. [Sentiment Analysis](#e-sentiment-analysis)
6. [Topic Modeling](#f-topic-modeling)
7. [Panel Regression Specifications](#g-regression-specifications)
8. [Variable Codebook](#h-variable-codebook)
9. [Replication Checklist](#i-replication-checklist)

---

## A. Software Environment

### A.1 System Requirements

| Component | Specification |
|-----------|--------------|
| OS | Ubuntu 22.04 LTS (tested); macOS 12+; Windows 10+ with WSL2 |
| Python | 3.9+ |
| RAM | 8 GB minimum; 16 GB recommended |
| Storage | ~2 GB for data and models |

### A.2 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.3.5 | Data manipulation |
| numpy | ≥1.21.5 | Numerical operations |
| scipy | ≥1.7.3 | Statistical functions, chi-squared distribution |
| nltk | ≥3.6.7 | Tokenization, stopwords, lemmatization |
| textblob | ≥0.17.1 | TextBlob sentiment (comparison/legacy) |
| vaderSentiment | ≥3.3.2 | **Primary sentiment analyzer** |
| gensim | ≥4.1.2 | LDA topic modeling |
| linearmodels | ≥4.20 | Panel OLS, random effects, first differences |
| statsmodels | ≥0.13.2 | Supplementary statistical tests |
| matplotlib | ≥3.4.3 | Figures |
| seaborn | ≥0.11.2 | Statistical visualization |
| requests | ≥2.26.0 | API data collection |
| census | ≥0.8.19 | Census API wrapper |

### A.3 Installation

```bash
git clone https://github.com/shaunakbuche/georgia-algorithmic-framing.git
cd georgia-algorithmic-framing

python3.9 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
```

---

## B. Data Sources

### B.1 Electoral Outcome Data

**Presidential elections (2020, 2024)**
- Source: MIT Election Data and Science Lab (MEDSL)
- URL: https://electionlab.mit.edu/data
- Dataset: *County Presidential Election Returns*
- DOI: 10.7910/DVN/VOQCHQ
- Auto-downloaded by `src/data/download_election_data.py`

**Gubernatorial elections (2018, 2022)**
- Source: Georgia Secretary of State
- URL: https://results.enr.clarityelections.com/GA/
- Manual download required (navigate to election, export county CSV)
- Save as `data/raw/elections/ga_elections_{year}.csv`
- See `data/raw/elections/README.md` for exact format

**Voter registration data**
- Source: Georgia Secretary of State
- Published alongside election returns
- Used to compute turnout denominator

### B.2 Political Advertising Data

**Source:** Meta Ad Library (https://www.facebook.com/ads/library/)

**API access:** Researcher access required. Apply at https://www.facebook.com/ads/library/api/

**Collection parameters:**
- Ad type: `POLITICAL_AND_ISSUE_ADS`
- Geographic filter: Georgia or Georgia-specific locations
- Collection window: 90 days before each election date
- Minimum impressions: 100

**Impression range conversion:**
Meta reports impressions as ranges (e.g., "10K–50K"). We convert using the **geometric mean** of range bounds rather than the arithmetic midpoint, consistent with log-normal impression distributions:

```
point_estimate = sqrt(lower_bound × upper_bound)

Example: "10K–50K" → sqrt(10,000 × 50,000) = sqrt(500,000,000) ≈ 22,361
```

**Final corpus:** 2,147 unique advertisements
- 2018: 412 ads
- 2020: 687 ads
- 2022: 489 ads
- 2024: 559 ads

### B.3 Socioeconomic Data

**Source:** American Community Survey (ACS) 5-Year Estimates, U.S. Census Bureau

**API key:** Free registration at https://api.census.gov/data/key_signup.html

**ACS year mapping:**

| Election Year | ACS Estimate Used | Rationale |
|---------------|------------------|-----------|
| 2018 | 2018 5-year | Contemporaneous |
| 2020 | 2019 5-year | Most recent available at election time |
| 2022 | 2021 5-year | Contemporaneous |
| 2024 | 2023 5-year | Most recent available |

**Variables:**

| ACS Code | Variable Name | Description |
|----------|--------------|-------------|
| B19013_001E | median_income | Median household income ($) |
| S1501_C02_015E | pct_bachelors | % adults 25+ with BA degree |
| B01003_001E | population | Total county population |

---

## C. County Attribution Algorithm

This is the most methodologically sensitive step. Meta provides targeting information at the DMA or statewide level, not county level. We attribute impressions to counties in three tiers:

### C.1 Attribution Hierarchy

**Tier 1: City-level targeting**
If an ad's `region_distribution` includes named Georgia cities, impressions are assigned to the corresponding county FIPS via a hand-coded city→county lookup table (157 Georgia cities/municipalities mapped to county FIPS).

**Tier 2: DMA-level targeting**
If `region_distribution` matches a known Georgia DMA, impressions are distributed proportionally by county population within that DMA:

```
impressions_c = total_impressions × (DMA_percentage / 100) × (pop_c / pop_DMA)
```

Georgia DMAs mapped: Atlanta, Savannah, Augusta, Columbus, Macon, Albany, Valdosta, Chattanooga (partial), Greenville-Spartanburg-Asheville (partial).

**Tier 3: Statewide allocation (fallback)**
If no geographic targeting information is available, impressions are distributed proportionally by county population across all 159 Georgia counties:

```
impressions_c = total_impressions × (pop_c / pop_Georgia)
```

**Statewide allocation flag:** All county-ad records include a `statewide_allocation` boolean. Records with `statewide_allocation=True` are excluded from the geographic-targeting sensitivity analysis reported in robustness checks.

### C.2 Measurement Error Implications

Statewide allocation introduces classical measurement error in the explanatory variable (sentiment, topic prevalence). Classical measurement error in X attenuates regression coefficients toward zero — our estimates are therefore **conservative lower bounds** on the true associations. The reported β = −1.41 for sentiment → margin may understate the actual relationship.

Approximately 40–60% of attributed impressions are statewide-allocated, varying by year. Presidential years have more explicitly targeted ads (higher campaign spending), reducing statewide allocation rates.

---

## D. Text Preprocessing

### D.1 Pipeline

All advertisement text undergoes the following preprocessing before topic modeling:

1. Lowercase conversion
2. URL removal (`https?://\S+`, `www\.\S+`)
3. Email address removal
4. HTML entity removal
5. Special character removal (preserving apostrophes for contractions)
6. Whitespace normalization
7. NLTK word tokenization
8. Stopword removal (NLTK English + domain-specific political ad boilerplate)
9. Minimum length filter (tokens ≤ 2 characters removed)
10. WordNet lemmatization

**Important:** Sentiment analysis uses **minimally preprocessed text** (steps 1–6 only) to preserve sentence structure, negation, and punctuation that lexicon-based tools depend on. Full preprocessing (steps 1–10) is used only for LDA topic modeling.

### D.2 Domain-Specific Stopwords

Added to NLTK's default English stopword list:

```
ad, paid, sponsored, click, learn, more, visit, www, com, http, https,
facebook, instagram, donate, donation, dollar, fund, contribute,
p.o., box, authorized
```

These terms are common ad boilerplate that would otherwise dominate topic distributions.

---

## E. Sentiment Analysis

### E.1 VADER (Primary Method)

VADER (Valence Aware Dictionary and Sentiment Reasoner) is used as the primary sentiment analyzer. Key advantages over TextBlob for this corpus:

- Designed for short, informal, social media text
- Handles ALL-CAPS emphasis ("VOTE" scored more strongly than "vote")
- Handles punctuation intensity ("Great!!!" > "Great.")
- Includes political and internet-era vocabulary
- Does not require preprocessing (operates on raw sentences)

The **compound score** (range −1 to +1) is used as the polarity measure. Compound is a normalized weighted composite of all lexicon ratings.

**Configuration:** Set `sentiment.method: vader` in `config/config.yaml` (default).

### E.2 TextBlob (Comparison/Legacy)

TextBlob is retained for methodological transparency — the paper's submitted version reports TextBlob results. Known limitations:

- Trained on movie reviews; domain mismatch with political text
- Does not handle social media conventions
- Assigns 0.0 (neutral) to texts with no recognized sentiment words, even if contextually negative

**Configuration:** Set `sentiment.method: textblob` in `config/config.yaml` to reproduce the paper's exact numbers.

### E.3 Method Comparison

Run `compare_methods()` in `sentiment_analysis.py` for a head-to-head comparison on a sample of ads. In our corpus:
- VADER–TextBlob Pearson r ≈ 0.61
- Sign agreement (both classify same direction): ~68%
- VADER assigns fewer neutral scores (0.0) than TextBlob
- Disagreements concentrated in sarcasm, irony, and domain-specific political attack language

### E.4 Impression-Weighted Aggregation

County-year sentiment index:

```
S_ct = Σ_i (s_i × w_i) / Σ_i (w_i)

where:
  s_i = polarity score for ad i
  w_i = attributed impressions for ad i in county c, year t
```

Lower S_ct = more negatively-valenced advertising environment.

### E.5 Validation

Manual coding of n=100 randomly sampled advertisements (binary: negative / not negative):
- Inter-rater reliability (Cohen's κ): 0.71
- VADER agreement with majority coding: ~72%
- TextBlob agreement with majority coding: ~68%

### E.6 Sentiment Distribution by Year (TextBlob)

| Year | Mean | SD | % Negative | % Highly Negative (<−0.5) |
|------|------|----|------------|--------------------------|
| 2018 | −0.05 | 0.28 | 52.4% | 4.8% |
| 2020 | −0.15 | 0.32 | 61.2% | 9.1% |
| 2022 | −0.08 | 0.29 | 55.8% | 6.2% |
| 2024 | −0.20 | 0.35 | 64.8% | 11.3% |

---

## F. Topic Modeling

### F.1 LDA Configuration

```
Model:        Gensim LdaModel
num_topics:   10
random_state: 42
passes:       10
chunksize:    100
alpha:        'auto' (asymmetric, estimated from data)
```

### F.2 Corpus Filtering

```
no_below: 5     (minimum document frequency)
no_above: 0.90  (maximum document proportion)
```

**Note on no_above:** The original code used `no_above=0.50`, which is standard for large corpora but too aggressive for 2,147 ads. With this small a corpus, common political vocabulary ("vote," "georgia," "election") would be filtered. We use `no_above=0.90` to retain these terms while still removing extreme outliers.

### F.3 Topic Number Selection

Models were trained for K ∈ {5, 8, 10, 12, 15}. K=10 was selected based on:
1. Cv coherence (highest score at K=10)
2. Qualitative interpretability (topics distinguish clearly between issue domains)
3. Granularity adequate for thematic analysis without over-fragmentation

### F.4 Topic Labels — Post-Training Assignment

**Critical methodological note:** LDA topic ordering is not deterministic. Topic ID 0 in one run does not necessarily correspond to Topic ID 0 in another run or seed. Labels must be assigned *after* inspecting trained model output, not before.

Procedure:
1. Train model
2. Run `modeler.print_topics(num_words=15)`
3. Inspect top words for each topic ID
4. Assign labels via `modeler.set_topic_labels({...})`
5. Labels are saved to `models/topic_labels.json`

### F.5 Topic Labels (Paper's K=10 Model, random_state=42)

| ID | Label | Top 10 Keywords |
|----|-------|----------------|
| 0 | Economic Policy | job, economy, tax, growth, business, wage, worker, create, small, employment |
| 1 | Healthcare/COVID | health, care, covid, pandemic, vaccine, hospital, insurance, coverage, protect, doctor |
| 2 | Immigration/Border | border, immigration, illegal, security, immigrant, enforcement, migrant, crisis, law, wall |
| 3 | Education | school, child, education, teacher, student, parent, learn, classroom, fund, public |
| 4 | Character/Attack | corrupt, radical, extreme, lie, truth, attack, record, failed, dangerous, wrong |
| 5 | Election Integrity | vote, election, fraud, ballot, count, integrity, secure, poll, voter, steal |
| 6 | Social Issues | gun, abortion, right, family, freedom, value, faith, protect, stand, fight |
| 7 | Law/Crime | police, crime, safety, defund, law, order, community, protect, violence, support |
| 8 | Generic Campaign | georgia, support, help, win, together, join, donate, campaign, vote, team |
| 9 | Governance | government, washington, leader, fight, work, people, change, need, time, better |

> **If your re-trained model produces different topic orderings**, inspect `print_topics()` output and update the `PAPER_TOPIC_LABELS` dictionary in `src/analysis/topic_modeling.py` accordingly before proceeding.

### F.6 Composite Topic Categories

Three thematic composites are used in regression:

| Composite | Component Topics | Rationale |
|-----------|-----------------|-----------|
| `topic_social_share` | Immigration/Border (2) + Social Issues (6) | Both activate identity-based partisan sorting |
| `topic_health_share` | Healthcare/COVID (1) | Dominant issue in 2020; Democratic framing advantage |
| `topic_election_share` | Election Integrity (5) | Misinformation narrative; linked to demobilization |

---

## G. Regression Specifications

### G.1 Panel Structure

- Unit of analysis: county × election year
- N = 159 counties × 4 elections = 636 maximum observations
- Missing values dropped listwise for each model

### G.2 Two-Way Fixed Effects Model

```
Y_ct = β₀ + β₁·S_ct + β₂·Z_ct + β₃·X_ct + α_c + γ_t + ε_ct

where:
  Y_ct = outcome (turnout_pct or rep_margin_pct)
  S_ct = sentiment index (standardized)
  Z_ct = topic prevalence vector (standardized)
  X_ct = control vector (median income, % bachelor's; standardized)
  α_c  = county fixed effects (remove time-invariant county characteristics)
  γ_t  = year fixed effects (remove common temporal shocks)
  ε_ct = error term, clustered by county
```

### G.3 Variable Standardization

All continuous predictors are z-scored (mean=0, SD=1):
```
x_std = (x − mean(x)) / sd(x)
```
Coefficients are interpreted as: *"expected change in Y (percentage points) associated with a one standard deviation change in X, holding other variables constant."*

### G.4 Hausman Test — Corrected Implementation

The Hausman specification test determines whether to use fixed or random effects:

```
H₀: Random effects estimator is consistent (use RE)
H₁: Random effects estimator is inconsistent (use FE)

Statistic: H = (b_FE − b_RE)' · [V_FE − V_RE]⁻¹ · (b_FE − b_RE)
Distribution: χ²(k), where k = number of common parameters

If V_FE − V_RE is not positive semi-definite (common with clustered SEs),
the test is unreliable. The code warns and reports the validity flag.
```

**Note on the original implementation:** An earlier version of this code used `np.abs(v_diff)` to handle negative variances in the difference matrix. This is mathematically invalid — the absolute value of a variance difference is not the variance of a difference. The correct approach uses matrix inversion or the pseudoinverse if the matrix is singular. The current implementation uses `np.linalg.inv()` with a PSD check and falls back to `np.linalg.pinv()` if singular.

### G.5 Robustness Checks

| Check | Method | Purpose |
|-------|--------|---------|
| Hausman test | FE vs RE comparison | Confirm FE is preferred specification |
| First differences | `FirstDifferenceOLS` | Validate within-county temporal effects |
| Presidential interaction | `sentiment × is_presidential` | Test H2 (stronger effects in presidential years) |
| Geographic subsample | Exclude statewide-allocated ads | Address measurement error in attribution |

---

## H. Variable Codebook

### H.1 Outcome Variables

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `county_fips` | string | 5-digit FIPS code (e.g., "13121" = Fulton) | Census |
| `year` | int | Election year ∈ {2018, 2020, 2022, 2024} | — |
| `turnout_pct` | float | Ballots cast / registered voters × 100 | GA SoS |
| `rep_margin_pct` | float | (Rep votes − Dem votes) / Total votes × 100 | GA SoS / MEDSL |

### H.2 Advertising Variables

| Variable | Type | Description | Range |
|----------|------|-------------|-------|
| `sentiment_index` | float | Impression-weighted mean polarity | [−1, 1] |
| `sentiment_index_std` | float | Standardized sentiment index | z-score |
| `topic_social_share` | float | Impression share: immigration + social issues | [0, 1] |
| `topic_health_share` | float | Impression share: healthcare/COVID | [0, 1] |
| `topic_election_share` | float | Impression share: election integrity | [0, 1] |
| `n_ads` | int | Unique ads attributed to county-year | ≥0 |
| `total_impressions` | int | Total attributed impressions | ≥0 |
| `pct_negative` | float | Share of impressions from negative ads | [0, 1] |

### H.3 Control Variables

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `median_income` | float | Median household income ($) | ACS |
| `pct_bachelors` | float | % adults 25+ with BA degree | ACS |
| `population` | int | Total county population | ACS |
| `prior_turnout` | float | Turnout in previous election | Calculated |
| `prior_margin` | float | Rep margin in previous election | Calculated |
| `is_presidential` | int | 1 = presidential year, 0 = midterm | Calculated |

---

## I. Replication Checklist

```bash
# 1. Environment
git clone https://github.com/shaunakbuche/georgia-algorithmic-framing.git
cd georgia-algorithmic-framing
python3.9 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

# 2. API configuration
cp config/config_template.yaml config/config.yaml
# → Add census_api_key and meta_access_token

# 3. Data collection
python src/data/prepare_acs_data.py         # ACS (auto)
python src/data/download_election_data.py   # MEDSL (auto) + SoS instructions
python src/data/download_ad_data.py         # Meta Ad Library

# 4. Preprocessing
python src/preprocessing/preprocess_ads.py

# 5. Analysis
python src/analysis/sentiment_analysis.py
python src/analysis/topic_modeling.py
# ↑ Review print_topics() output; update PAPER_TOPIC_LABELS if ordering differs
python src/data/merge_datasets.py
python src/analysis/panel_regression.py

# 6. Outputs
python src/visualization/create_tables.py
python src/visualization/create_figures.py

# 7. Verify
pytest tests/ -v
```

### Expected Verification Values

| Statistic | Expected | Tolerance |
|-----------|----------|-----------|
| Panel N | 636 | Exact |
| Mean 2020 sentiment (TextBlob) | −0.15 | ±0.02 |
| Sentiment→Margin β | −1.41 | ±0.15 |
| Sentiment→Margin SE | 0.52 | ±0.10 |
| Hausman χ² | 24.7 | ±3.0 |
| Social topics→Margin β | 0.89 | ±0.15 |
| Health topics→Margin β | −0.76 | ±0.15 |

---

*For questions or replication issues, open a GitHub issue at https://github.com/shaunakbuche/georgia-algorithmic-framing/issues*
