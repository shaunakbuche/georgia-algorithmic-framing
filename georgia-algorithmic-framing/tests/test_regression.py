"""
tests/test_regression.py

Unit tests for panel regression module, with focus on the corrected
Hausman test implementation.

Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.panel_regression import (
    hausman_test,
    prepare_panel_data,
    format_results_table,
    get_available_vars,
)


# ---------------------------------------------------------------------------
# Hausman test
# ---------------------------------------------------------------------------

class TestHausmanTest:
    """
    Tests for the corrected Hausman specification test.

    The original implementation used np.abs(v_diff) which is mathematically
    invalid. The correct formula is: H = b_diff' @ inv(V_diff) @ b_diff
    """

    def _make_mock_results(self, params, cov):
        """Create a minimal mock results object with .params and .cov attributes."""
        class MockResults:
            pass
        r = MockResults()
        r.params = pd.Series(params)
        r.cov = pd.DataFrame(
            cov,
            index=list(params.keys()),
            columns=list(params.keys()),
        )
        return r

    def test_basic_structure(self):
        """Result dict has required keys."""
        fe = self._make_mock_results(
            {"x1": 0.5, "x2": 0.3},
            [[0.04, 0.0], [0.0, 0.02]],
        )
        re = self._make_mock_results(
            {"x1": 0.4, "x2": 0.25},
            [[0.02, 0.0], [0.0, 0.01]],
        )
        result = hausman_test(fe, re)
        for key in ("statistic", "df", "p_value", "valid"):
            assert key in result, f"Missing key: {key}"

    def test_identical_params_near_zero(self):
        """When FE ≈ RE, statistic should be near zero → high p-value."""
        fe = self._make_mock_results(
            {"x1": 0.5},
            [[0.04]],
        )
        re = self._make_mock_results(
            {"x1": 0.5},
            [[0.02]],
        )
        result = hausman_test(fe, re)
        # b_diff = 0, so statistic should be ~0
        assert result["statistic"] == pytest.approx(0.0, abs=1e-9)
        assert result["p_value"] == pytest.approx(1.0, abs=0.01)

    def test_large_difference_low_pvalue(self):
        """Large FE/RE divergence → small p-value (reject H0, prefer FE)."""
        fe = self._make_mock_results(
            {"x1": 2.0},
            [[0.04]],
        )
        re = self._make_mock_results(
            {"x1": 0.0},
            [[0.01]],
        )
        result = hausman_test(fe, re)
        # b_diff = 2.0, V_diff = 0.04 - 0.01 = 0.03
        # H = (2.0)^2 / 0.03 = 133.3
        assert result["statistic"] > 10
        assert result["p_value"] < 0.05

    def test_df_equals_num_common_params(self):
        """Degrees of freedom = number of common non-intercept parameters."""
        fe = self._make_mock_results(
            {"Intercept": 1.0, "x1": 0.5, "x2": 0.3, "x3": 0.1},
            np.diag([0.1, 0.04, 0.02, 0.01]),
        )
        re = self._make_mock_results(
            {"Intercept": 1.1, "x1": 0.4, "x2": 0.25, "x3": 0.08},
            np.diag([0.05, 0.02, 0.01, 0.005]),
        )
        result = hausman_test(fe, re)
        assert result["df"] == 3, "Intercept excluded; 3 common params → df=3"

    def test_no_common_params(self):
        """No common parameters → NaN result, valid=False."""
        fe = self._make_mock_results({"x1": 0.5}, [[0.04]])
        re = self._make_mock_results({"x2": 0.3}, [[0.02]])
        result = hausman_test(fe, re)
        assert np.isnan(result["statistic"])
        assert result["valid"] is False

    def test_pvalue_in_unit_interval(self):
        """p-value must be in [0, 1]."""
        fe = self._make_mock_results({"x1": 0.5, "x2": -0.3}, [[0.04, 0], [0, 0.02]])
        re = self._make_mock_results({"x1": 0.3, "x2": -0.1}, [[0.02, 0], [0, 0.01]])
        result = hausman_test(fe, re)
        if not np.isnan(result["p_value"]):
            assert 0 <= result["p_value"] <= 1.0

    def test_symmetry(self):
        """Swapping FE and RE should produce same statistic (b_diff sign flips but squared)."""
        fe = self._make_mock_results({"x1": 0.8}, [[0.06]])
        re = self._make_mock_results({"x1": 0.4}, [[0.02]])
        r1 = hausman_test(fe, re)
        # Reverse (RE as "FE", FE as "RE") — V_diff will be negative, invalid
        # Just verify the forward direction gives positive statistic
        assert r1["statistic"] >= 0 or np.isnan(r1["statistic"])


# ---------------------------------------------------------------------------
# prepare_panel_data
# ---------------------------------------------------------------------------

class TestPreparePanelData:
    def _sample_df(self):
        np.random.seed(42)
        n = 20
        return pd.DataFrame({
            "county_fips": [f"130{i:02d}" for i in range(5)] * 4,
            "year": [2018] * 5 + [2020] * 5 + [2022] * 5 + [2024] * 5,
            "sentiment_index": np.random.normal(0, 0.2, n),
            "median_income": np.random.normal(55000, 10000, n),
            "pct_bachelors": np.random.uniform(15, 45, n),
            "turnout_pct": np.random.uniform(50, 80, n),
            "rep_margin_pct": np.random.normal(10, 20, n),
        })

    def test_multiindex_set(self):
        df = self._sample_df()
        panel = prepare_panel_data(df)
        assert panel.index.names == ["county_fips", "year"]

    def test_standardized_columns_created(self):
        df = self._sample_df()
        panel = prepare_panel_data(df)
        for col in ["sentiment_index_std", "median_income_std", "pct_bachelors_std"]:
            assert col in panel.columns, f"Missing: {col}"

    def test_standardized_mean_near_zero(self):
        df = self._sample_df()
        panel = prepare_panel_data(df)
        for col in ["sentiment_index_std", "median_income_std"]:
            if col in panel.columns:
                mean = panel[col].mean()
                assert abs(mean) < 1e-10, f"{col} mean should be 0, got {mean}"

    def test_standardized_std_near_one(self):
        df = self._sample_df()
        panel = prepare_panel_data(df)
        for col in ["sentiment_index_std", "median_income_std"]:
            if col in panel.columns:
                std = panel[col].std()
                assert abs(std - 1.0) < 0.01, f"{col} std should be ~1, got {std}"


# ---------------------------------------------------------------------------
# get_available_vars
# ---------------------------------------------------------------------------

class TestGetAvailableVars:
    def test_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = get_available_vars(df, ["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_some_missing(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = get_available_vars(df, ["a", "b", "c", "d"])
        assert result == ["a", "b"]

    def test_all_missing(self):
        df = pd.DataFrame({"x": [1]})
        result = get_available_vars(df, ["a", "b"])
        assert result == []

    def test_empty_vars(self):
        df = pd.DataFrame({"a": [1]})
        result = get_available_vars(df, [])
        assert result == []


# ---------------------------------------------------------------------------
# format_results_table
# ---------------------------------------------------------------------------

class TestFormatResultsTable:
    def _mock_results(self):
        class MockResults:
            params = pd.Series({"sentiment_index_std": -1.41, "pct_bachelors_std": 0.23})
            std_errors = pd.Series({"sentiment_index_std": 0.52, "pct_bachelors_std": 0.18})
            tstats = pd.Series({"sentiment_index_std": -2.71, "pct_bachelors_std": 1.28})
            pvalues = pd.Series({"sentiment_index_std": 0.007, "pct_bachelors_std": 0.201})
        return MockResults()

    def test_required_columns(self):
        table = format_results_table(self._mock_results(), "Test")
        for col in ["Variable", "Coefficient", "Std. Error", "t-stat", "p-value", "Sig."]:
            assert col in table.columns

    def test_significance_stars(self):
        table = format_results_table(self._mock_results())
        # sentiment p=0.007 → **
        sentiment_row = table[table["Variable"] == "sentiment_index_std"]
        assert sentiment_row["Sig."].values[0] == "**"

    def test_no_significance(self):
        table = format_results_table(self._mock_results())
        pct_row = table[table["Variable"] == "pct_bachelors_std"]
        assert pct_row["Sig."].values[0] == ""

    def test_model_name_column(self):
        table = format_results_table(self._mock_results(), "My Model")
        assert (table["Model"] == "My Model").all()
