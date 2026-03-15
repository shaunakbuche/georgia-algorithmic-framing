"""
tests/test_preprocessing.py

Unit tests for text preprocessing and county attribution modules.

Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.preprocess_ads import TextPreprocessor, preprocess_ads_dataframe
from src.data.download_ad_data import (
    convert_impression_range,
    attribute_ad_to_counties,
    extract_ad_text,
)


# ---------------------------------------------------------------------------
# TextPreprocessor
# ---------------------------------------------------------------------------

class TestTextPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()

    def test_clean_removes_urls(self, preprocessor):
        text = "Visit https://example.com for more info"
        cleaned = preprocessor.clean_text(text)
        assert "http" not in cleaned
        assert "example" not in cleaned

    def test_clean_removes_emails(self, preprocessor):
        text = "Contact us at info@campaign.com"
        cleaned = preprocessor.clean_text(text)
        assert "@" not in cleaned

    def test_clean_lowercases(self, preprocessor):
        text = "VOTE for Georgia's Future"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == cleaned.lower()

    def test_clean_normalizes_whitespace(self, preprocessor):
        text = "Vote   for    Georgia"
        cleaned = preprocessor.clean_text(text)
        assert "  " not in cleaned

    def test_stopwords_removed_in_lda_mode(self, preprocessor):
        text = "vote for the future of our great state"
        tokens = preprocessor.preprocess_for_lda(text)
        stop_words = {"for", "the", "of", "our"}
        for sw in stop_words:
            assert sw not in tokens, f"Stopword '{sw}' not removed"

    def test_lemmatization(self, preprocessor):
        text = "voting voters voted"
        tokens = preprocessor.preprocess_for_lda(text)
        # All forms should reduce to 'vote'
        assert "vote" in tokens or "voter" in tokens

    def test_lda_returns_list(self, preprocessor):
        tokens = preprocessor.preprocess_for_lda("Georgia economy jobs growth")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_sentiment_mode_preserves_negation(self, preprocessor):
        text = "This is not a good policy at all!"
        cleaned = preprocessor.preprocess_for_sentiment(text)
        assert "not" in cleaned, "Negation must be preserved for sentiment scoring"

    def test_empty_string(self, preprocessor):
        assert preprocessor.preprocess_for_lda("") == []
        assert preprocessor.preprocess_for_sentiment("") == ""

    def test_nan_input(self, preprocessor):
        assert preprocessor.preprocess_for_lda(None) == []
        assert preprocessor.preprocess_for_lda(float("nan")) == []

    def test_min_token_length(self, preprocessor):
        text = "a an to is it"
        tokens = preprocessor.preprocess_for_lda(text)
        assert all(len(t) > 2 for t in tokens), "Tokens ≤ 2 chars should be removed"


class TestPreprocessAdsDataframe:
    def _sample_df(self):
        return pd.DataFrame({
            "ad_id":       ["a1", "a2", "a3"],
            "text":        [
                "Vote for Georgia! Jobs and growth.",
                "Corrupt radical politician destroying our state https://attack.com",
                "",
            ],
            "impressions": [10000, 20000, 5000],
            "county_fips": ["13121", "13121", "13089"],
            "year":        [2020, 2020, 2020],
        })

    def test_adds_text_clean(self):
        result = preprocess_ads_dataframe(self._sample_df())
        assert "text_clean" in result.columns

    def test_adds_tokens(self):
        result = preprocess_ads_dataframe(self._sample_df())
        assert "tokens" in result.columns

    def test_tokens_parseable(self):
        import ast
        result = preprocess_ads_dataframe(self._sample_df())
        for tokens_str in result["tokens"]:
            try:
                tokens = ast.literal_eval(tokens_str)
                assert isinstance(tokens, list)
            except (ValueError, SyntaxError):
                pytest.fail(f"tokens column not parseable as list: {tokens_str!r}")

    def test_row_count_preserved(self):
        df = self._sample_df()
        result = preprocess_ads_dataframe(df)
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# convert_impression_range
# ---------------------------------------------------------------------------

class TestConvertImpressionRange:
    def test_known_ranges(self):
        # Geometric mean of (1000, 5000) = sqrt(5,000,000) ≈ 2236
        assert convert_impression_range("1K-5K") == pytest.approx(2236, rel=0.01)

    def test_large_range(self):
        # sqrt(10000 * 50000) = sqrt(500,000,000) ≈ 22360
        assert convert_impression_range("10K-50K") == pytest.approx(22360, rel=0.01)

    def test_empty_string(self):
        assert convert_impression_range("") == 0

    def test_none(self):
        assert convert_impression_range(None) == 0

    def test_unknown_range(self):
        assert convert_impression_range("99K-100K") == 0

    def test_all_ranges_positive(self):
        ranges = ["<1K", "1K-5K", "5K-10K", "10K-50K",
                  "50K-100K", "100K-200K", "200K-500K", "500K-1M", ">1M"]
        for r in ranges:
            assert convert_impression_range(r) > 0, f"Range {r} should give positive value"

    def test_monotonically_increasing(self):
        """Higher ranges should produce larger estimates."""
        ranges = ["1K-5K", "5K-10K", "10K-50K", "50K-100K"]
        values = [convert_impression_range(r) for r in ranges]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1], \
                f"Range {ranges[i]} ({values[i]}) >= {ranges[i+1]} ({values[i+1]})"


# ---------------------------------------------------------------------------
# attribute_ad_to_counties
# ---------------------------------------------------------------------------

class TestAttributeAdToCounties:
    @pytest.fixture
    def ga_populations(self):
        """Minimal Georgia county population DataFrame."""
        return pd.DataFrame({
            "county_fips": ["13121", "13089", "13067", "13135", "13117"],
            "population":  [1065000, 760000, 760000, 950000, 240000],
        })

    def _make_ad(self, impressions="10K-50K", region_dist=None):
        return {
            "id": "test_ad_1",
            "impressions": {"upper_bound": impressions},
            "region_distribution": region_dist or [],
        }

    def test_statewide_returns_all_counties(self, ga_populations):
        ad = self._make_ad()
        attributions = attribute_ad_to_counties(ad, ga_populations)
        assert set(attributions.keys()) == set(ga_populations["county_fips"])

    def test_statewide_impressions_sum_to_total(self, ga_populations):
        ad = self._make_ad("10K-50K")
        attributions = attribute_ad_to_counties(ad, ga_populations)
        total = sum(attributions.values())
        expected = convert_impression_range("10K-50K")
        # Allow small rounding error from integer conversion
        assert abs(total - expected) < len(ga_populations)

    def test_population_proportional(self, ga_populations):
        """Larger counties should receive more attributed impressions."""
        ad = self._make_ad()
        attributions = attribute_ad_to_counties(ad, ga_populations)

        # Fulton (13121, pop 1.065M) should get more than Forsyth (13117, pop 240K)
        assert attributions.get("13121", 0) > attributions.get("13117", 0)

    def test_zero_impressions_returns_empty(self, ga_populations):
        ad = {"id": "x", "impressions": {}, "region_distribution": []}
        attributions = attribute_ad_to_counties(ad, ga_populations)
        assert attributions == {}

    def test_city_targeting(self, ga_populations):
        """Atlanta targeting should map to Fulton County (13121)."""
        ad = self._make_ad(region_dist=[{"region": "Atlanta, GA", "percentage": "100"}])
        attributions = attribute_ad_to_counties(ad, ga_populations)
        assert "13121" in attributions
        # City-targeted: should not distribute to all counties
        assert len(attributions) < len(ga_populations)


# ---------------------------------------------------------------------------
# extract_ad_text
# ---------------------------------------------------------------------------

class TestExtractAdText:
    def test_single_body(self):
        ad = {"ad_creative_bodies": ["Vote for Georgia!"]}
        assert extract_ad_text(ad) == "Vote for Georgia!"

    def test_multiple_bodies_joined(self):
        ad = {"ad_creative_bodies": ["First line.", "Second line."]}
        text = extract_ad_text(ad)
        assert "First line." in text
        assert "Second line." in text

    def test_empty_bodies(self):
        assert extract_ad_text({"ad_creative_bodies": []}) == ""

    def test_missing_key(self):
        assert extract_ad_text({}) == ""
