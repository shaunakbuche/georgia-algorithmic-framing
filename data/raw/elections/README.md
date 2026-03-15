# data/raw/elections/README.md

## Election Data — Manual Collection Required

Georgia election data comes from two sources. This directory is gitignored
but must be populated before running the pipeline.

---

## Required Files

### Presidential Years (2020, 2024)
**Source:** MIT Election Data and Science Lab (MEDSL)
- URL: https://electionlab.mit.edu/data
- Dataset: *County Presidential Election Returns 2000–2020*
- DOI: [10.7910/DVN/VOQCHQ](https://doi.org/10.7910/DVN/VOQCHQ)
- For 2024: Download from GA Secretary of State (MEDSL lags by ~1 year)

### Gubernatorial Years (2018, 2022)
**Source:** Georgia Secretary of State
- URL: https://results.enr.clarityelections.com/GA/
- Navigate to the relevant election and download county-level results

---

## Expected File Format

Save each year as `ga_elections_{year}.csv` with these columns:

```
county_fips   | 5-digit FIPS code (e.g., "13121" for Fulton)
county_name   | County name (e.g., "Fulton")
year          | Election year
dem_votes     | Democratic candidate votes
rep_votes     | Republican candidate votes
total_votes   | Total ballots cast
registered_voters | Total registered voters
```

Also save registration files as `ga_registration_{year}.csv` if separate
from the returns file.

---

## Helper Script

Running `python src/data/download_election_data.py` will:
1. Attempt to auto-download MEDSL data for 2020
2. Print instructions for years requiring manual download
3. Build `data/processed/election_panel.csv` once files are present
