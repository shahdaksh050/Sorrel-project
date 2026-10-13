from __future__ import annotations

import pandas as pd

from src.core.reshaping import (
    detect_and_strip_summary_rows,
    detect_wide_temporal_columns,
    unpivot_temporal_frame,
)


def test_detect_wide_temporal_columns() -> None:
    # Year columns in header
    cols_years = ["Country", "Sector", "2018", "2019", "2020", "2021"]
    detected = detect_wide_temporal_columns(cols_years)
    assert detected == ["2018", "2019", "2020", "2021"]

    # Month columns in header
    cols_months = ["Region", "Product", "Jan", "Feb", "Mar", "Apr", "May"]
    detected_m = detect_wide_temporal_columns(cols_months)
    assert detected_m == ["Jan", "Feb", "Mar", "Apr", "May"]

    # Standard non-temporal columns
    cols_std = ["age", "gender", "income", "churn"]
    assert detect_wide_temporal_columns(cols_std) == []


def test_unpivot_temporal_frame() -> None:
    df_wide = pd.DataFrame({
        "country": ["Germany", "France"],
        "2019": [100, 200],
        "2020": [110, 220],
        "2021": [120, 240],
        "2022": [130, 260],
    })
    melted = unpivot_temporal_frame(df_wide, var_name="year", value_name="gdp")
    assert len(melted) == 8
    assert "year" in melted.columns
    assert "gdp" in melted.columns
    assert melted["year"].tolist() == [2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022]


def test_detect_and_strip_summary_rows() -> None:
    df = pd.DataFrame({
        "item": ["Widget A", "Widget B", "Widget C", "Widget D", "Widget E", "Widget F", "Grand Total"],
        "sales": [10, 20, 30, 40, 50, 60, 210],
    })
    cleaned_df, notes = detect_and_strip_summary_rows(df)
    assert len(cleaned_df) == 6
    assert "Grand Total" not in cleaned_df["item"].values
    assert len(notes) == 1
    assert "Grand Total" in notes[0] or "summary/total" in notes[0]
