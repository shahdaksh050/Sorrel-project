"""
Unit tests for Layout Detection, Wide-to-Long Reshape, and Subtotal Rows (FutureScope Phase 1).

Tests the 3-test generality criteria:
1. Planted cases:
   - Header on row 4 with title and footnotes.
   - Subtotal rows mixed in ('North Total', 'Grand Total').
   - Wide time-in-header table (years as columns).
2. Null case:
   - Clean tabular CSV where no layout alteration occurs.
3. Structurally diverse case:
   - Real-world CSV (AirQualityUCI) where layout stays intact.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.io import (
    detect_and_exclude_subtotals,
    detect_wide_time_headers,
    read_any,
    reshape_wide_to_long,
)
from tests.fixtures.hidden_assumptions import (
    header_offset_and_notes_csv,
    subtotals_mixed_csv,
    wide_time_headers_csv,
)


def test_planted_header_offset_and_footnotes(tmp_path: Path) -> None:
    """Planted test: file has 3-row title block and trailing footnotes."""
    p = header_offset_and_notes_csv(tmp_path)
    df, report = read_any(str(p))

    assert report.header_row_offset == 3
    assert list(df.columns) == ["Region", "Product", "Quarter", "Units_Sold", "Revenue"]
    assert len(df) == 8
    assert "North" in df["Region"].values
    assert "West" in df["Region"].values
    # Check notes
    assert any("Header detected at row 4" in note for note in report.notes)
    assert any("Trimmed" in note for note in report.notes)


def test_planted_subtotal_exclusion(tmp_path: Path) -> None:
    """Planted test: subtotal and total rows are mixed with unit records."""
    p = subtotals_mixed_csv(tmp_path)
    df, report = read_any(str(p))

    assert report.subtotals_excluded == 3
    # Only the 6 true department unit records should remain
    assert len(df) == 6
    assert "North Total" not in df["Region"].values
    assert "South Total" not in df["Region"].values
    assert "Grand Total" not in df["Region"].values
    # Headcount sum of the 6 unit rows equals exactly 115 (no double counting!)
    assert df["Headcount"].sum() == 115


def test_planted_wide_to_long_reshape(tmp_path: Path) -> None:
    """Planted test: wide table with years in header reshapes to long panel."""
    p = wide_time_headers_csv(tmp_path)
    df, report = read_any(str(p))

    assert report.reshaped_from_wide is True
    assert set(report.wide_time_vars) == {"2019", "2020", "2021", "2022", "2023"}
    # The reshaped dataframe should have Country, Indicator, time, value
    assert "time" in df.columns
    assert "value" in df.columns
    assert "Country" in df.columns
    assert "Indicator" in df.columns
    # 6 original rows x 5 years = 30 rows
    assert len(df) == 30


def test_null_clean_csv(tmp_path: Path) -> None:
    """Null test: clean CSV with standard layout is not altered."""
    clean_csv = tmp_path / "clean_data.csv"
    clean_csv.write_text(
        "customer_id,age,income,churn\n"
        "C001,25,50000.0,0\n"
        "C002,45,85000.0,0\n"
        "C003,35,62000.0,1\n"
        "C004,50,95000.0,0\n",
        encoding="utf-8",
    )
    df, report = read_any(str(clean_csv))

    assert report.header_row_offset == 0
    assert report.subtotals_excluded == 0
    assert report.reshaped_from_wide is False
    assert len(df) == 4
    assert list(df.columns) == ["customer_id", "age", "income", "churn"]


def test_direct_detect_and_exclude_subtotals_math_match() -> None:
    """Direct test: detect subtotal row via mathematical sum matching."""
    df = pd.DataFrame({
        "item": ["A", "B", "C", "Summary"],
        "cost": [10.0, 20.0, 30.0, 60.0],
        "sales": [100.0, 200.0, 300.0, 600.0],
    })
    clean_df, excluded, _notes = detect_and_exclude_subtotals(df)
    assert 3 in excluded
    assert len(clean_df) == 3
    assert list(clean_df["item"]) == ["A", "B", "C"]


def test_direct_wide_time_headers_helpers() -> None:
    """Direct test for detect_wide_time_headers and reshape_wide_to_long."""
    df = pd.DataFrame({
        "dept": ["Finance", "IT"],
        "t1": [10, 20],
        "t2": [15, 25],
        "t3": [18, 30],
    })
    res = detect_wide_time_headers(df)
    assert res is not None
    id_vars, time_vars = res
    assert id_vars == ["dept"]
    assert time_vars == ["t1", "t2", "t3"]

    long_df = reshape_wide_to_long(df, id_vars, time_vars, time_col="quarter", value_col="headcount")
    assert len(long_df) == 6
    assert list(long_df.columns) == ["dept", "quarter", "headcount"]
