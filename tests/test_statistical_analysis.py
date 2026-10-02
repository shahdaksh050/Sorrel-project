"""Unit tests for Statistical Analysis Tools — Stage 3 hypothesis testing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.statistical_analysis import (
    SelectStatisticalTestTool,
    _cohens_d,
    _cramers_v,
)


@pytest.fixture
def grouped_csv(tmp_path: pytest.TempPathFactory) -> str:
    """Numeric feature with a binary group column — t-test / Mann-Whitney territory."""
    rng = np.random.default_rng(42)
    n = 120
    group = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "row_id": range(1, n + 1),
            "value": rng.normal(0, 1, n) + group * 0.8,
            "category": rng.choice(["a", "b", "c"], n),
            "group": group,
        }
    )
    p = tmp_path / "grouped.csv"
    df.to_csv(p, index=False)
    return str(p)


class TestSelectStatisticalTestTool:
    def test_numeric_feature_two_groups_succeeds(self, grouped_csv: str) -> None:
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="value", group_column="group"
        )
        assert result.status == "success"
        assert result.output["n_groups"] == 2
        assert result.output["test_name"] in (
            "Independent T-Test", "Welch's T-Test", "Mann-Whitney U"
        )

    def test_p_value_in_valid_range(self, grouped_csv: str) -> None:
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="value", group_column="group"
        )
        assert 0.0 <= result.output["p_value"] <= 1.0

    def test_categorical_feature_uses_chi_square(self, grouped_csv: str) -> None:
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="category", group_column="group"
        )
        assert result.status == "success"
        assert result.output["test_name"] == "Chi-Square Test of Independence"

    def test_missing_feature_column_errors(self, grouped_csv: str) -> None:
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="nope", group_column="group"
        )
        assert result.status == "error"

    def test_id_like_column_rejected(self, grouped_csv: str) -> None:
        """A genuine name-hinted, ~100% unique column is still a row ID."""
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="row_id", group_column="group"
        )
        assert result.status == "error"
        assert "row ID" in (result.error_message or "")

    @pytest.mark.parametrize("n", [6, 500, 5000])
    def test_sorted_continuous_column_not_rejected_as_id(self, tmp_path: Path, n: int) -> None:
        """U0.3 regression guard: a sorted continuous measurement is ~100%
        unique but is NOT a row ID — only name-hinted or exact-uniqueness
        non-float columns are. Row order alone must never decide whether
        the only hypothesis-testing tool in the pipeline runs."""
        rng = np.random.default_rng(42)
        group = rng.integers(0, 2, n)
        measurement = np.sort(rng.normal(0, 1, n))  # sorted by construction
        df = pd.DataFrame({"measurement": measurement, "group": group})
        p = tmp_path / "sorted.csv"
        df.to_csv(p, index=False)
        result = SelectStatisticalTestTool().run(
            file_path=str(p), feature_column="measurement", group_column="group"
        )
        assert result.status == "success"

    def test_customer_id_integer_column_still_rejected(self, tmp_path: Path) -> None:
        n = 200
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "customer_id": range(1, n + 1),
            "group": rng.integers(0, 2, n),
        })
        p = tmp_path / "customers.csv"
        df.to_csv(p, index=False)
        result = SelectStatisticalTestTool().run(
            file_path=str(p), feature_column="customer_id", group_column="group"
        )
        assert result.status == "error"
        assert "row ID" in (result.error_message or "")

    def test_interpretation_present(self, grouped_csv: str) -> None:
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="value", group_column="group"
        )
        assert result.output["interpretation"]
        assert "summary" in result.output

    def test_summary_key_present(self, grouped_csv: str) -> None:
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="value", group_column="group"
        )
        assert "summary" in result.output


class TestEffectSizeFormulas:
    """Unit-test the formulas directly against hand-computed values —
    scipy.stats has no Cohen's d, so it's easy to get subtly wrong."""

    def test_cohens_d_hand_computed(self) -> None:
        a = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        b = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        # mean_a=6, mean_b=5, var_a=var_b=10 (ddof=1) -> pooled_sd=sqrt(10)
        expected = (6.0 - 5.0) / (10.0**0.5)
        assert _cohens_d(a, b) == pytest.approx(expected, abs=1e-9)

    def test_cramers_v_hand_computed(self) -> None:
        # 2x2 table, n=100, chi2 computed by hand for a known association
        # table: [[40, 10], [10, 40]]
        table = np.array([[40, 10], [10, 40]])
        n = int(table.sum())
        row_totals = table.sum(axis=1)
        col_totals = table.sum(axis=0)
        expected_counts = np.outer(row_totals, col_totals) / n
        chi2 = float(((table - expected_counts) ** 2 / expected_counts).sum())
        v = _cramers_v(chi2, n, 2, 2)
        assert v == pytest.approx((chi2 / (n * 1)) ** 0.5, abs=1e-9)
        assert 0.0 <= v <= 1.0


class TestPracticalSignificance:
    def test_large_n_negligible_effect_significant_but_not_practical(self, tmp_path: Path) -> None:
        """The large-n trap: n=50,000 per group, d≈0.038 (negligible) — the
        test must be statistically significant AND flagged as not
        practically significant."""
        rng = np.random.default_rng(42)
        n = 50_000
        group = np.array([0] * n + [1] * n)
        value = np.concatenate([
            rng.normal(0, 1, n),
            rng.normal(0.038, 1, n),  # tiny mean shift -> d ~ 0.038
        ])
        df = pd.DataFrame({"value": value, "group": group})
        p = tmp_path / "large_n.csv"
        df.to_csv(p, index=False)
        result = SelectStatisticalTestTool().run(
            file_path=str(p), feature_column="value", group_column="group"
        )
        assert result.status == "success"
        assert result.output["significant"] is True
        assert result.output["practical_significance"] is False
        assert result.output["sample_size_note"]

    def test_real_effect_significant_and_practical(self, tmp_path: Path) -> None:
        """A real effect (d≈0.8) must be both statistically AND practically
        significant."""
        rng = np.random.default_rng(42)
        n = 100
        group = np.array([0] * n + [1] * n)
        value = np.concatenate([rng.normal(0, 1, n), rng.normal(0.8, 1, n)])
        df = pd.DataFrame({"value": value, "group": group})
        p = tmp_path / "real_effect.csv"
        df.to_csv(p, index=False)
        result = SelectStatisticalTestTool().run(
            file_path=str(p), feature_column="value", group_column="group"
        )
        assert result.status == "success"
        assert result.output["significant"] is True
        assert result.output["practical_significance"] is True


class TestChiSquareValidity:
    def test_low_expected_count_2x2_emits_warning_and_uses_fishers_exact(self, tmp_path: Path) -> None:
        """A 2x2 table with an expected cell count of 2 is unreliable for
        chi-square — must switch to Fisher's exact and say so."""
        df = pd.DataFrame({
            "feature": ["yes"] * 3 + ["no"] * 47,
            "group": (["a"] * 2 + ["b"] * 1) + (["a"] * 23 + ["b"] * 24),
        })
        p = tmp_path / "sparse.csv"
        df.to_csv(p, index=False)
        result = SelectStatisticalTestTool().run(
            file_path=str(p), feature_column="feature", group_column="group"
        )
        assert result.status == "success"
        assert result.output["test_name"] == "Fisher's Exact Test"
        assert "expected_frequency_warning" in result.output


class TestBooleanFeaturesAreCategorical:
    """pandas calls booleans numeric; a yes/no flag still needs a categorical test."""

    def test_balanced_yes_no_feature_gets_chi_square_not_mann_whitney(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        n = 400
        group = rng.choice(["a", "b"], n)
        feature = np.where(rng.random(n) < np.where(group == "a", 0.7, 0.3), "yes", "no")
        p = tmp_path / "balanced.csv"
        pd.DataFrame({"feature": feature, "group": group}).to_csv(p, index=False)
        result = SelectStatisticalTestTool().run(file_path=str(p), feature_column="feature", group_column="group")
        assert result.status == "success"
        assert result.output["test_name"] == "Chi-Square Test of Independence"
        assert result.output["significant"] is True  # the planted association is found

    def test_a_flag_is_not_picked_as_the_default_measure(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(1)
        n = 200
        df = pd.DataFrame(
            {"is_vip": rng.choice(["yes", "no"], n), "spend": rng.normal(50, 10, n), "tier": rng.choice(["a", "b"], n)}
        )
        p = tmp_path / "flag.csv"
        df.to_csv(p, index=False)
        result = SelectStatisticalTestTool().run(file_path=str(p), group_column="tier")
        assert result.status == "success"
        assert result.output.get("feature_column", "spend") != "is_vip"


class TestPValuesAreNeverPrintedAsZero:
    def test_a_tiny_p_value_reads_as_less_than_0_001(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(3)
        n = 600
        group = rng.choice(["a", "b"], n)
        df = pd.DataFrame({"value": rng.normal(0, 1, n) + np.where(group == "a", 2.0, 0.0), "group": group})
        p = tmp_path / "big_effect.csv"
        df.to_csv(p, index=False)
        result = SelectStatisticalTestTool().run(file_path=str(p), feature_column="value", group_column="group")
        assert result.status == "success"
        text = result.output["summary"] + " " + result.output["interpretation"]
        assert "p=0.0000" not in text and "p=0.000 " not in text
        assert "p<0.001" in text
