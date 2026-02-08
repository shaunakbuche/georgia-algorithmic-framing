# Appendix: Reproducibility Documentation

## Algorithmic Framing and Voter Polarization: A Georgia Panel Study (2018-2024)

---

## Table of Contents

1. [Software Environment](#appendix-a-software-environment)
2. [Data Sources and Collection](#appendix-b-data-sources)
3. [Text Preprocessing Pipeline](#appendix-c-text-preprocessing)
4. [Sentiment Analysis Methodology](#appendix-d-sentiment-analysis)
5. [Topic Modeling Methodology](#appendix-e-topic-modeling)
6. [Panel Regression Specifications](#appendix-f-regression-specifications)
7. [Variable Codebook](#appendix-g-variable-codebook)
8. [Replication Instructions](#appendix-h-replication-instructions)

---

## Appendix A: Software Environment

### A.1 System Requirements

- **Operating System:** Ubuntu 22.04 LTS (tested), macOS 12+ (compatible), Windows 10+ with WSL2 (compatible)
- **Python:** 3.9.7
- **Memory:** Minimum 8GB RAM recommended
- **Storage:** ~2GB for data and models

### A.2 Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 1.3.5 | Data manipulation and cleaning |
| numpy | 1.21.5 | Numerical computations |
| scipy | 1.7.3 | Statistical functions |
| nltk | 3.6.7 | Natural language processing, tokenization |
| textblob | 0.17.1 | Sentiment analysis |
| gensim | 4.1.2 | Topic modeling (LDA) |
| scikit-learn | 0.24.2 | Additional preprocessing |
| statsmodels | 0.13.2 | Regression analysis |
| linearmodels | 4.20 | Panel data regression |
| matplotlib | 3.4.3 | Visualization |
| seaborn | 0.11.2 | Statistical visualization |
| requests | 2.26.0 | API data collection |

### A.3 Installation

```bash
# Clone repository
git clone https://github.com/[username]/georgia-algorithmic-framing.git
cd georgia-algorithmic-framing

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

---

## Appendix B: Data Sources

### B.1 Electoral Outcome Data

**Sources:**
1. **MIT Election Data and Science Lab (MEDSL)**
   - URL: https://electionlab.mit.edu/data
   - Dataset: County Presidential Election Returns
   - DOI: 10.7910/DVN/VOQCHQ
   - Used for: 2020, 2024 presidential returns

2. **Georgia Secretary of State**
   - URL: https://results.enr.clarityelections.com/GA/
   - Used for: 2018, 2022 gubernatorial returns; voter registration data

**Collection Procedure:**
```python
# Example: Loading MEDSL data
import pandas as pd

def load_election_data(year):
    """
    Load county-level election results.
    
    Parameters:
    -----------
    year : int
        Election year (2018, 2020, 2022, 2024)
    
    Returns:
    --------
    pd.DataFrame : County-level results with columns:
        - county_fips: 5-digit FIPS code
        - county_name: County name
        - total_votes: Total ballots cast
        - votes_dem: Democratic votes
        - votes_rep: Republican votes
    """
    if year in [2020, 2024]:
        # Load from MEDSL
        df = pd.read_csv(f'data/raw/medsl_{year}.csv')
        df = df[df['state'] == 'GEORGIA']
    else:
        # Load from Georgia SoS
        df = pd.read_csv(f'data/raw/ga_sos_{year}.csv')
    
    return df
```

### B.2 Political Advertising Data

**Source:** Meta Ad Library (https://www.facebook.com/ads/library/)

**Collection Parameters:**
- Time period: 90 days before each election
- Geographic filter: Georgia or Georgia-specific locations
- Ad type: Political and Issue Ads
- Language: English
- Minimum impressions: 100

**API Query Structure:**
```python
import requests

def query_ad_library(start_date, end_date, access_token):
    """
    Query Meta Ad Library for Georgia political ads.
    
    Note: Requires approved Meta Ad Library API access.
    """
    base_url = "https://graph.facebook.com/v18.0/ads_archive"
    
    params = {
        'access_token': access_token,
        'ad_type': 'POLITICAL_AND_ISSUE_ADS',
        'ad_reached_countries': "['US']",
        'search_terms': 'Georgia',
        'ad_delivery_date_min': start_date,
        'ad_delivery_date_max': end_date,
        'fields': 'id,ad_creative_bodies,page_name,impressions,spend,ad_delivery_start_time,ad_delivery_stop_time,demographic_distribution,region_distribution'
    }
    
    response = requests.get(base_url, params=params)
    return response.json()
```

**Impression Range Conversion:**
```python
import numpy as np

def convert_impression_range(range_str):
    """
    Convert Meta impression range to point estimate.
    
    Examples:
    ---------
    "1K-5K" -> 2236 (geometric mean)
    "10K-50K" -> 22360
    """
    range_map = {
        '<1K': (0, 1000),
        '1K-5K': (1000, 5000),
        '5K-10K': (5000, 10000),
        '10K-50K': (10000, 50000),
        '50K-100K': (50000, 100000),
        '100K-200K': (100000, 200000),
        '200K-500K': (200000, 500000),
        '500K-1M': (500000, 1000000),
        '>1M': (1000000, 5000000)
    }
    
    lower, upper = range_map.get(range_str, (0, 0))
    if lower == 0:
        return 0
    return int(np.sqrt(lower * upper))
```

### B.3 County Attribution Algorithm

```python
def attribute_ad_to_counties(ad, county_populations):
    """
    Attribute advertisement impressions to Georgia counties.
    
    Parameters:
    -----------
    ad : dict
        Advertisement record with targeting info
    county_populations : pd.DataFrame
        County population data
    
    Returns:
    --------
    dict : {county_fips: attributed_impressions}
    """
    total_impressions = ad['impressions']
    attributions = {}
    
    if ad.get('city_targeting'):
        # City-level targeting
        cities = ad['city_targeting']
        for city in cities:
            counties = city_to_counties[city]  # Lookup table
            for county, share in counties.items():
                attributions[county] = attributions.get(county, 0) + \
                    total_impressions * share / len(cities)
    
    elif ad.get('dma_targeting'):
        # DMA-level targeting
        dma = ad['dma_targeting']
        dma_counties = dma_to_counties[dma]
        dma_pop = county_populations[
            county_populations['county_fips'].isin(dma_counties)
        ]['population'].sum()
        
        for county in dma_counties:
            county_pop = county_populations[
                county_populations['county_fips'] == county
            ]['population'].values[0]
            attributions[county] = total_impressions * (county_pop / dma_pop)
    
    else:
        # Statewide targeting - distribute by population
        total_pop = county_populations['population'].sum()
        for _, row in county_populations.iterrows():
            attributions[row['county_fips']] = \
                total_impressions * (row['population'] / total_pop)
    
    return attributions
```

### B.4 Socioeconomic Data

**Source:** American Community Survey 5-Year Estimates via Census Bureau API

```python
import requests

def get_acs_data(year, variables, state='13'):
    """
    Retrieve ACS 5-year estimates for Georgia counties.
    
    Parameters:
    -----------
    year : int
        End year of 5-year estimates
    variables : list
        ACS variable codes
    state : str
        State FIPS code ('13' for Georgia)
    """
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    
    params = {
        'get': ','.join(['NAME'] + variables),
        'for': 'county:*',
        'in': f'state:{state}',
        'key': CENSUS_API_KEY
    }
    
    response = requests.get(base_url, params=params)
    return pd.DataFrame(response.json()[1:], columns=response.json()[0])
```

**Variables Used:**
| ACS Variable | Description |
|--------------|-------------|
| B19013_001E | Median household income |
| S1501_C02_015E | Percent bachelor's degree or higher |
| B01003_001E | Total population |

---

## Appendix C: Text Preprocessing

### C.1 Complete Preprocessing Pipeline

```python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    """
    Text preprocessing pipeline for political advertisement analysis.
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add domain-specific stopwords
        self.stop_words.update([
            'ad', 'paid', 'sponsored', 'click', 'learn', 'more',
            'www', 'com', 'http', 'https'
        ])
    
    def clean_text(self, text):
        """Remove URLs, special characters, normalize whitespace."""
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r"[^a-zA-Z0-9\s']", ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.lower()
    
    def tokenize(self, text):
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list."""
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]
    
    def lemmatize(self, tokens):
        """Lemmatize tokens to base form."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def preprocess(self, text, for_lda=True):
        """
        Full preprocessing pipeline.
        
        Parameters:
        -----------
        text : str
            Raw advertisement text
        for_lda : bool
            If True, return tokens for topic modeling
            If False, return joined string
        
        Returns:
        --------
        list or str : Preprocessed tokens or text
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        
        if for_lda:
            return tokens
        return ' '.join(tokens)

# Usage
preprocessor = TextPreprocessor()
processed_texts = [preprocessor.preprocess(ad['text']) for ad in ads]
```

---

## Appendix D: Sentiment Analysis

### D.1 TextBlob Implementation

```python
from textblob import TextBlob
import pandas as pd
import numpy as np

def calculate_sentiment(text):
    """
    Calculate sentiment polarity using TextBlob.
    
    Parameters:
    -----------
    text : str
        Raw advertisement text (NOT preprocessed)
    
    Returns:
    --------
    float : Polarity score in [-1.0, 1.0]
    """
    if pd.isna(text) or text.strip() == '':
        return np.nan
    
    blob = TextBlob(str(text))
    return blob.sentiment.polarity


def calculate_county_year_sentiment(ads_df):
    """
    Calculate impression-weighted sentiment for each county-year.
    
    Parameters:
    -----------
    ads_df : pd.DataFrame
        Advertisement data with columns:
        - county_fips, year, text, impressions
    
    Returns:
    --------
    pd.DataFrame : County-year sentiment indices
    """
    # Calculate sentiment for each ad
    ads_df['polarity'] = ads_df['text'].apply(calculate_sentiment)
    
    # Calculate weighted mean by county-year
    def weighted_mean(group):
        weights = group['impressions']
        values = group['polarity']
        
        if weights.sum() == 0 or values.isna().all():
            return np.nan
        
        return np.average(values.dropna(), 
                         weights=weights[values.notna()])
    
    sentiment_index = ads_df.groupby(['county_fips', 'year']).apply(
        weighted_mean
    ).reset_index(name='sentiment_index')
    
    return sentiment_index
```

### D.2 Validation Results

Manual validation on n=100 randomly sampled advertisements:

| Metric | Value |
|--------|-------|
| Inter-rater reliability (Cohen's κ) | 0.71 |
| TextBlob agreement with majority coding | 68% |
| Disagreements concentrated in | Mixed/subtle sentiment |

### D.3 Sentiment Distribution by Year

| Year | Mean Polarity | SD | % Negative (<0) | % Positive (>0) |
|------|---------------|-----|-----------------|-----------------|
| 2018 | -0.05 | 0.28 | 52.4% | 38.1% |
| 2020 | -0.15 | 0.32 | 61.2% | 29.7% |
| 2022 | -0.08 | 0.29 | 55.8% | 35.4% |
| 2024 | -0.20 | 0.35 | 64.8% | 26.1% |

---

## Appendix E: Topic Modeling

### E.1 LDA Implementation

```python
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                   level=logging.INFO)

class TopicModeler:
    """
    LDA topic modeling for political advertisements.
    """
    
    def __init__(self, num_topics=10, random_state=42):
        self.num_topics = num_topics
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.model = None
    
    def build_corpus(self, texts, no_below=5, no_above=0.5):
        """
        Build dictionary and corpus from preprocessed texts.
        
        Parameters:
        -----------
        texts : list of list
            Preprocessed and tokenized texts
        no_below : int
            Filter tokens appearing in fewer than n documents
        no_above : float
            Filter tokens appearing in more than fraction of documents
        """
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        print(f"Dictionary size: {len(self.dictionary)}")
        print(f"Corpus size: {len(self.corpus)}")
    
    def train(self, passes=10, chunksize=100):
        """Train LDA model."""
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            update_every=1,
            chunksize=chunksize,
            passes=passes,
            alpha='auto',
            per_word_topics=True
        )
    
    def get_coherence(self, texts):
        """Calculate coherence score."""
        coherence_model = CoherenceModel(
            model=self.model,
            texts=texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()
    
    def get_document_topics(self, bow):
        """Get topic distribution for a document."""
        return self.model.get_document_topics(bow)
    
    def get_topic_words(self, topic_id, num_words=10):
        """Get top words for a topic."""
        return self.model.show_topic(topic_id, num_words)

# Model selection
def select_num_topics(texts, k_range=range(5, 16)):
    """
    Select optimal number of topics based on coherence.
    """
    results = []
    
    for k in k_range:
        modeler = TopicModeler(num_topics=k)
        modeler.build_corpus(texts)
        modeler.train()
        coherence = modeler.get_coherence(texts)
        results.append({'k': k, 'coherence': coherence})
        print(f"K={k}: Coherence={coherence:.4f}")
    
    return pd.DataFrame(results)
```

### E.2 Topic Labels and Keywords

| Topic | Label | Top 10 Keywords |
|-------|-------|-----------------|
| 0 | Economic Policy | job, economy, tax, growth, business, wage, worker, employment, create, small |
| 1 | Healthcare/COVID | health, care, covid, pandemic, vaccine, hospital, insurance, coverage, protect, doctor |
| 2 | Immigration/Border | border, immigration, illegal, security, immigrant, wall, enforcement, migrant, crisis, law |
| 3 | Education | school, child, education, teacher, student, parent, learn, classroom, fund, public |
| 4 | Character/Attack | corrupt, radical, extreme, lie, truth, attack, record, failed, dangerous, wrong |
| 5 | Election Integrity | vote, election, fraud, ballot, count, integrity, secure, poll, voter, steal |
| 6 | Social Issues | gun, abortion, right, family, freedom, value, faith, protect, stand, fight |
| 7 | Law/Crime | police, crime, safety, defund, law, order, community, protect, violence, support |
| 8 | Generic Campaign | georgia, support, help, win, together, join, donate, campaign, vote, team |
| 9 | Governance | government, washington, leader, fight, work, people, change, need, time, better |

### E.3 Topic Attribution to Counties

```python
def calculate_county_topic_prevalence(ads_df, topic_modeler):
    """
    Calculate impression-weighted topic prevalence for each county-year.
    
    Returns:
    --------
    pd.DataFrame : County-year topic prevalence shares
    """
    # Get dominant topic for each ad
    def get_dominant_topic(text_tokens):
        bow = topic_modeler.dictionary.doc2bow(text_tokens)
        topic_dist = topic_modeler.get_document_topics(bow)
        if not topic_dist:
            return None
        return max(topic_dist, key=lambda x: x[1])[0]
    
    ads_df['dominant_topic'] = ads_df['tokens'].apply(get_dominant_topic)
    
    # Calculate weighted topic shares
    topic_shares = []
    
    for (county, year), group in ads_df.groupby(['county_fips', 'year']):
        total_impressions = group['impressions'].sum()
        
        shares = {'county_fips': county, 'year': year}
        
        for topic_id in range(topic_modeler.num_topics):
            topic_impressions = group[
                group['dominant_topic'] == topic_id
            ]['impressions'].sum()
            shares[f'topic_{topic_id}_share'] = topic_impressions / total_impressions
        
        topic_shares.append(shares)
    
    return pd.DataFrame(topic_shares)
```

---

## Appendix F: Regression Specifications

### F.1 Panel Data Structure

```python
import pandas as pd
from linearmodels import PanelOLS

def prepare_panel_data(df):
    """
    Prepare data for panel regression.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset with county-year observations
    
    Returns:
    --------
    pd.DataFrame : Panel-indexed DataFrame
    """
    # Set multi-index
    df = df.set_index(['county_fips', 'year'])
    
    # Standardize continuous variables
    vars_to_standardize = [
        'sentiment_index', 
        'topic_social_share',
        'topic_health_share',
        'topic_election_share',
        'median_income',
        'pct_bachelors'
    ]
    
    for var in vars_to_standardize:
        if var in df.columns:
            df[f'{var}_std'] = (df[var] - df[var].mean()) / df[var].std()
    
    return df
```

### F.2 Model Specifications

**Model 1: Turnout**
```python
def estimate_turnout_model(panel_df):
    """
    Estimate two-way fixed effects model for voter turnout.
    """
    formula = """
    turnout_pct ~ 
        sentiment_index_std + 
        topic_social_share_std +
        topic_health_share_std +
        topic_election_share_std +
        median_income_std +
        pct_bachelors_std +
        EntityEffects + TimeEffects
    """
    
    model = PanelOLS.from_formula(formula, data=panel_df)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results
```

**Model 2: Vote Margin**
```python
def estimate_margin_model(panel_df):
    """
    Estimate two-way fixed effects model for partisan vote margin.
    """
    formula = """
    rep_margin_pct ~ 
        sentiment_index_std + 
        topic_social_share_std +
        topic_health_share_std +
        topic_election_share_std +
        median_income_std +
        pct_bachelors_std +
        EntityEffects + TimeEffects
    """
    
    model = PanelOLS.from_formula(formula, data=panel_df)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results
```

### F.3 Robustness Checks

```python
from linearmodels import RandomEffects, FirstDifferenceOLS
from scipy.stats import chi2

def hausman_test(fe_results, re_results):
    """
    Conduct Hausman test comparing FE and RE estimators.
    """
    b_fe = fe_results.params
    b_re = re_results.params
    
    # Common variables
    common = b_fe.index.intersection(b_re.index)
    
    b_diff = b_fe[common] - b_re[common]
    
    # Variance of difference
    V_fe = fe_results.cov[common].loc[common]
    V_re = re_results.cov[common].loc[common]
    V_diff = V_fe - V_re
    
    # Hausman statistic
    H = b_diff @ np.linalg.inv(V_diff) @ b_diff
    df = len(common)
    p_value = 1 - chi2.cdf(H, df)
    
    return {'statistic': H, 'df': df, 'p_value': p_value}

def first_difference_model(panel_df, dep_var):
    """
    Estimate first-differences model.
    """
    model = FirstDifferenceOLS.from_formula(
        f"{dep_var} ~ sentiment_index_std + topic_social_share_std",
        data=panel_df
    )
    return model.fit()
```

---

## Appendix G: Variable Codebook

### G.1 Outcome Variables

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `county_fips` | string | 5-digit FIPS county code | Census |
| `county_name` | string | County name | Census |
| `year` | integer | Election year {2018, 2020, 2022, 2024} | - |
| `turnout_pct` | float | Ballots cast / registered voters × 100 | GA SoS, MEDSL |
| `rep_margin_pct` | float | (Rep votes - Dem votes) / Total votes × 100 | GA SoS, MEDSL |

### G.2 Advertising Content Variables

| Variable | Type | Description | Range |
|----------|------|-------------|-------|
| `sentiment_index` | float | Impression-weighted mean polarity | [-1, 1] |
| `sentiment_index_std` | float | Standardized sentiment index | z-score |
| `topic_0_share` | float | Share of impressions for Topic 0 | [0, 1] |
| ... | ... | ... | ... |
| `topic_9_share` | float | Share of impressions for Topic 9 | [0, 1] |
| `n_ads` | integer | Number of ads attributed to county-year | ≥0 |
| `total_impressions` | integer | Total attributed impressions | ≥0 |

### G.3 Control Variables

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `median_income` | float | Median household income ($) | ACS |
| `pct_bachelors` | float | % adults 25+ with BA degree | ACS |
| `population` | integer | Total population | ACS |
| `prior_turnout` | float | Turnout in previous election | Calculated |
| `prior_margin` | float | Margin in previous election | Calculated |

---

## Appendix H: Replication Instructions

### H.1 Step-by-Step Reproduction

```bash
# 1. Clone repository
git clone https://github.com/[username]/georgia-algorithmic-framing.git
cd georgia-algorithmic-framing

# 2. Set up environment
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

# 3. Configure API keys (if collecting fresh data)
cp config/config_template.yaml config/config.yaml
# Edit config.yaml with your Census API key

# 4. Download/prepare data
python src/data/download_election_data.py
python src/data/prepare_acs_data.py
# Note: Ad data is pre-collected and available in data/processed/

# 5. Run preprocessing
python src/preprocessing/preprocess_ads.py

# 6. Run analysis
python src/analysis/sentiment_analysis.py
python src/analysis/topic_modeling.py
python src/analysis/panel_regression.py

# 7. Generate tables and figures
python src/visualization/create_tables.py
python src/visualization/create_figures.py

# 8. Run all (alternative)
python run_analysis.py --all
```

### H.2 Expected Output

Running the full pipeline produces:

1. **Data files** in `data/processed/`:
   - `panel_dataset.csv` - Final merged panel dataset
   - `ad_sentiment.csv` - Advertisement-level sentiment scores
   - `ad_topics.csv` - Advertisement-level topic assignments

2. **Results** in `results/`:
   - `table1_descriptive_stats.tex` - Summary statistics
   - `table2_turnout_regression.tex` - Turnout model results
   - `table3_margin_regression.tex` - Margin model results
   - `figure1_sentiment_trends.png` - Temporal sentiment plot
   - `figure2_topic_prevalence.png` - Topic prevalence by year

3. **Models** in `models/`:
   - `lda_model.pkl` - Trained LDA model
   - `dictionary.pkl` - Gensim dictionary

### H.3 Runtime Expectations

| Step | Approximate Time |
|------|------------------|
| Data download | 5-10 minutes |
| Preprocessing | 2-3 minutes |
| Sentiment analysis | 1-2 minutes |
| Topic modeling | 10-15 minutes |
| Regression analysis | 1-2 minutes |
| **Total** | **~25 minutes** |

Tested on: Intel i7-10700, 32GB RAM, Ubuntu 22.04

### H.4 Verifying Reproduction

Key statistics to verify successful reproduction:

| Statistic | Expected Value | Tolerance |
|-----------|----------------|-----------|
| N observations (panel) | 636 | Exact |
| Mean sentiment 2020 | -0.15 | ±0.02 |
| Turnout model R² | 0.89 | ±0.02 |
| Margin model R² | 0.94 | ±0.02 |
| Sentiment→Margin β | -1.41 | ±0.15 |

---

## Contact and Support

For questions about replication:
- Open an issue on GitHub
- Email: [author email]

## Citation

If using this code or data, please cite:

```bibtex
@article{author2025algorithmic,
  title={Algorithmic Framing and Voter Polarization: A Georgia Panel Study (2018-2024)},
  author={[Author]},
  journal={[Journal]},
  year={2025},
  note={Data and code available at https://github.com/[username]/georgia-algorithmic-framing}
}
```
