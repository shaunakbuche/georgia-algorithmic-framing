"""
tests/test_sentiment.py

Unit tests for sentiment analysis module.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.sentiment_analysis import (
    score_vader,
    score_textblob,
    calculate_weighted_sentiment,
    analyze_ads,
    aggregate_to_county_year,
)


# ---------------------------------------------------------------------------
# score_vader
# ---------------------------------------------------------------------------

class TestScoreVader:
    def test_positive_text(self):
        score = score_vader("Georgia is thriving! Great jobs and strong economy!")
        assert score > 0, "Clearly positive text should score > 0"

    def test_negative_text(self):
        score = score_vader("Corrupt radical politicians are destroying our state.")
        assert score < 0, "Clearly negative text should score < 0"

    def test_neutral_text(self):
        score = score_vader("Vote on November 5th.")
        assert -0.5 < score < 0.5, "Neutral informational text should score near 0"

    def test_empty_string(self):
        assert np.isnan(score_vader("")), "Empty string should return NaN"

    def test_nan_input(self):
        assert np.isnan(score_vader(float("nan"))), "NaN input should return NaN"

    def test_range(self):
        texts = [
            "Amazing! Best candidate ever!",
            "Terrible corrupt failure.",
            "Election day is Tuesday.",
        ]
        for t in texts:
            s = score_vader(t)
            assert -1.0 <= s <= 1.0, f"Score {s} out of [-1, 1] range"

    def test_negation_handling(self):
        pos = score_vader("This is good.")
        neg = score_vader("This is not good.")
        assert pos > neg, "VADER should handle negation: 'not good' < 'good'"


# ---------------------------------------------------------------------------
# score_textblob
# ---------------------------------------------------------------------------

class TestScoreTextblob:
    def test_positive(self):
        assert score_textblob("Excellent leadership and strong results.") > 0

    def test_negative(self):
        assert score_textblob("Dangerous radical extremist failed policies.") < 0

    def test_empty(self):
        assert np.isnan(score_textblob(""))

    def test_nan(self):
        assert np.isnan(score_textblob(None))

    def test_range(self):
        for text in ["Great!", "Terrible!", "Vote."]:
            s = score_textblob(text)
            if not np.isnan(s):
                assert -1.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# calculate_weighted_sentiment
# ---------------------------------------------------------------------------

class TestWeightedSentiment:
    def _make_group(self, polarities, impressions):
        return pd.DataFrame({"polarity": polarities, "impressions": impressions})

    def test_uniform_weights(self):
        group = self._make_group([0.5, -0.5], [100, 100])
        result = calculate_weighted_sentiment(group)
        assert abs(result) < 1e-10, "Equal weights, opposite polarities → 0"

    def test_skewed_weights(self):
        group = self._make_group([1.0, -1.0], [900, 100])
        result = calculate_weighted_sentiment(group)
        assert result == pytest.approx(0.8, abs=1e-9), "900:100 weight → 0.8"

    def test_all_zero_weights(self):
        group = self._make_group([0.5, 0.5], [0, 0])
        assert np.isnan(calculate_weighted_sentiment(group))

    def test_nan_polarity_excluded(self):
        group = self._make_group([np.nan, 1.0], [100, 100])
        result = calculate_weighted_sentiment(group)
        assert result == pytest.approx(1.0), "NaN polarity rows should be excluded"

    def test_single_row(self):
        group = self._make_group([0.3], [500])
        assert calculate_weighted_sentiment(group) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# analyze_ads
# ---------------------------------------------------------------------------

class TestAnalyzeAds:
    def _sample_df(self):
        return pd.DataFrame({
            "text_clean": [
                "Invest in Georgia's future — jobs and growth!",
                "Corrupt radical politicians are destroying Georgia.",
                "Vote on November 5th.",
                "",
            ],
            "impressions": [10000, 20000, 5000, 1000],
            "county_fips": ["13121"] * 4,
            "year": [2020] * 4,
        })

    def test_vader_scores_added(self):
        df = analyze_ads(self._sample_df(), method="vader")
        assert "polarity" in df.columns
        assert len(df) == 4

    def test_textblob_scores_added(self):
        df = analyze_ads(self._sample_df(), method="textblob")
        assert "polarity" in df.columns
        assert "subjectivity" in df.columns

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            analyze_ads(self._sample_df(), method="bert")

    def test_sentiment_method_column(self):
        df = analyze_ads(self._sample_df(), method="vader")
        assert (df["sentiment_method"] == "vader").all()


# ---------------------------------------------------------------------------
# aggregate_to_county_year
# ---------------------------------------------------------------------------

class TestAggregateToCountyYear:
    def _sample_ads(self):
        return pd.DataFrame({
            "county_fips": ["13121", "13121", "13089", "13089"],
            "year":        [2020, 2020, 2020, 2020],
            "polarity":    [0.4, -0.2, 0.1, 0.3],
            "impressions": [10000, 10000, 5000, 15000],
            "ad_id":       ["a1", "a2", "a3", "a4"],
        })

    def test_output_shape(self):
        ads = self._sample_ads()
        result = aggregate_to_county_year(ads)
        assert len(result) == 2, "Two counties → two rows"

    def test_weighted_mean_correct(self):
        ads = self._sample_ads()
        result = aggregate_to_county_year(ads)
        fulton = result[result["county_fips"] == "13121"]["sentiment_index"].values[0]
        # (0.4 * 10000 + -0.2 * 10000) / 20000 = 0.1
        assert fulton == pytest.approx(0.1, abs=1e-9)

    def test_pct_negative_range(self):
        ads = self._sample_ads()
        result = aggregate_to_county_year(ads)
        assert (result["pct_negative"].between(0, 1) | result["pct_negative"].isna()).all()

    def test_required_columns_present(self):
        ads = self._sample_ads()
        result = aggregate_to_county_year(ads)
        for col in ["county_fips", "year", "sentiment_index", "n_ads",
                    "total_impressions", "pct_negative"]:
            assert col in result.columns, f"Missing column: {col}"
