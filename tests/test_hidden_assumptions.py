"""
Tests for hidden assumption dataset fixtures (FutureScope Phase 0).

Validates that each fixture in tests/fixtures/hidden_assumptions.py genuinely
breaks one or more of the 15 hidden assumptions from FutureScope.md Section 8.
These fixtures act as the baseline harness for Phase 1 (layout/subtotals),
Phase 3 (variable-aware methods), and Phase 4 (dependence/design-aware inference).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tests.fixtures.hidden_assumptions import (
    autocorrelated_series_csv,
    benford_anomalous_csv,
    circular_variables_csv,
    clustered_entities_csv,
    compositional_shares_csv,
    header_offset_and_notes_csv,
    heaped_and_censored_csv,
    subtotals_mixed_csv,
    wide_time_headers_csv,
)


def test_header_offset_and_notes_fixture(tmp_path: Path) -> None:
    """Assumption 1: Header is not on row 1, footnotes present."""
    p = header_offset_and_notes_csv(tmp_path)
    text = p.read_text(encoding="utf-8").splitlines()
    assert "QUARTERLY REGIONAL SALES" in text[0]
    assert "Region,Product,Quarter,Units_Sold,Revenue" in text[3]
    assert any("* Notes:" in line for line in text)


def test_subtotals_mixed_fixture(tmp_path: Path) -> None:
    """Assumption 2: Subtotals and totals mixed with unit records."""
    p = subtotals_mixed_csv(tmp_path)
    df = pd.read_csv(p)
    assert "North Total" in df["Region"].values
    assert "Grand Total" in df["Region"].values
    # Check that Grand Total Headcount equals sum of individual regions
    north_eng = df.loc[(df["Region"] == "North") & (df["Department"] == "Engineering"), "Headcount"].iloc[0]
    north_sales = df.loc[(df["Region"] == "North") & (df["Department"] == "Sales"), "Headcount"].iloc[0]
    north_supp = df.loc[(df["Region"] == "North") & (df["Department"] == "Support"), "Headcount"].iloc[0]
    north_tot = df.loc[df["Region"] == "North Total", "Headcount"].iloc[0]
    assert north_eng + north_sales + north_supp == north_tot


def test_wide_time_headers_fixture(tmp_path: Path) -> None:
    """Assumption 3: Years/time points in header instead of long format."""
    p = wide_time_headers_csv(tmp_path)
    df = pd.read_csv(p)
    year_cols = [c for c in df.columns if c.strip().isdigit()]
    assert len(year_cols) >= 5
    assert "2019" in year_cols
    assert "2023" in year_cols


def test_autocorrelated_series_fixture(tmp_path: Path) -> None:
    """Assumption 4: Serial dependence (autocorrelation) violates row independence."""
    p = autocorrelated_series_csv(tmp_path, n=400)
    df = pd.read_csv(p)
    assert "sensor_reading" in df.columns
    # Calculate lag-1 autocorrelation
    s = df["sensor_reading"]
    r1 = s.autocorr(lag=1)
    assert r1 > 0.85, f"Expected high autocorrelation, got {r1}"


def test_clustered_entities_fixture(tmp_path: Path) -> None:
    """Assumption 4: Clustered entities violate i.i.d. assumption."""
    p = clustered_entities_csv(tmp_path, n_clusters=10, cluster_size=20)
    df = pd.read_csv(p)
    assert "school_id" in df.columns
    assert "student_id" in df.columns
    assert df["school_id"].nunique() == 10
    assert len(df) == 200


def test_compositional_shares_fixture(tmp_path: Path) -> None:
    """Assumption 6: Share columns sum to 100% (spurious correlation trap)."""
    p = compositional_shares_csv(tmp_path, n=50)
    df = pd.read_csv(p)
    share_cols = [c for c in df.columns if c.startswith("share_")]
    assert len(share_cols) == 4
    row_sums = df[share_cols].sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 100.0, rtol=1e-5)


def test_circular_variables_fixture(tmp_path: Path) -> None:
    """Assumption 6: Periodic/circular scale (hour 23 next to hour 0)."""
    p = circular_variables_csv(tmp_path, n=150)
    df = pd.read_csv(p)
    assert "hour_of_day" in df.columns
    assert "wind_direction_deg" in df.columns
    # Confirm values wrap around midnight
    hours = df["hour_of_day"].values
    assert 23 in hours or 22 in hours
    assert 0 in hours or 1 in hours


def test_heaped_and_censored_fixture(tmp_path: Path) -> None:
    """Assumption 11: Digit heaping and detection limits/censoring."""
    p = heaped_and_censored_csv(tmp_path, n=200)
    df = pd.read_csv(p)
    assert "reported_age" in df.columns
    assert "assay_concentration" in df.columns
    # Check heaping: excess of ages ending in 0 or 5
    last_digits = df["reported_age"] % 10
    heaped_count = (last_digits == 0) | (last_digits == 5)
    # With 40% rounding, heaped should be significantly higher than expected 20%
    assert heaped_count.mean() > 0.30
    # Check censored strings
    conc = df["assay_concentration"].astype(str)
    assert any(c.startswith("<") or c.startswith(">") for c in conc)


def test_benford_anomalous_fixture(tmp_path: Path) -> None:
    """Assumption 11: Violations of Benford's Law in invoice amounts."""
    p = benford_anomalous_csv(tmp_path, n=500)
    df = pd.read_csv(p)
    assert "amount" in df.columns
    first_digits = [int(str(v).lstrip("0.")[0]) for v in df["amount"] if v > 0]
    counts = pd.Series(first_digits).value_counts(normalize=True)
    # Under Benford, 1 appears ~30.1% of the time.
    # In this fixture, it is uniformly distributed (~11.1%)
    assert counts.get(1, 0) < 0.20, f"Expected uniform first digit, got {counts.get(1, 0)}"
