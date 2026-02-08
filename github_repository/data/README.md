# Data Documentation

## Directory Structure

```
data/
├── raw/                    # Original unprocessed data
│   ├── elections/          # Election returns from MEDSL and GA SoS
│   ├── ads/                # Advertisement data from Meta Ad Library
│   └── acs/                # American Community Survey data
└── processed/              # Cleaned and analysis-ready data
    ├── panel_dataset.csv   # Main analysis panel
    ├── ad_sentiment.csv    # Advertisement-level sentiment
    ├── ad_topics.csv       # Advertisement-level topics
    ├── county_sentiment.csv # County-year sentiment
    └── county_topics.csv   # County-year topic prevalence
```

## Data Sources

### Election Data
- **Source**: MIT Election Data and Science Lab (MEDSL); Georgia Secretary of State
- **Years**: 2018, 2020, 2022, 2024
- **Geography**: Georgia counties (N=159)

### Political Advertising Data
- **Source**: Meta Ad Library
- **Collection Period**: 90 days before each election
- **Sample Size**: 2,147 advertisements

### Census Data
- **Source**: American Community Survey 5-Year Estimates
- **Variables**: Median household income, educational attainment

## Variable Codebook

### Panel Dataset

| Variable | Type | Description |
|----------|------|-------------|
| county_fips | string | 5-digit FIPS code |
| year | integer | Election year |
| turnout_pct | float | Voter turnout percentage |
| rep_margin_pct | float | Republican - Democratic margin |
| sentiment_index | float | Weighted mean ad sentiment |
| topic_social_share | float | Social issues topic share |
| topic_health_share | float | Healthcare topic share |
| median_income | float | Median household income |
| pct_bachelors | float | Percent with bachelor degree |

## Topic Labels

| ID | Label |
|----|-------|
| 0 | Economic Policy |
| 1 | Healthcare/COVID |
| 2 | Immigration/Border |
| 3 | Education |
| 4 | Character/Attack |
| 5 | Election Integrity |
| 6 | Social Issues |
| 7 | Law/Crime |
| 8 | Generic Campaign |
| 9 | Governance |
