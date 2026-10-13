"""
Reshaping & Ingestion Robustness — wide-table and report-layout normalisation.

Detects non-tidy data representations commonly found in government data,
utility reports, and financial exports:
1. Wide "time in header" tables (e.g. columns named '2015', '2016', ... or 'Jan', 'Feb', ...)
   that represent longitudinal panel or time-series data spread across columns.
2. Summary/subtotal rows (e.g. 'Total', 'Grand Total', 'Average') that distort
   distributions and sample size if left inside data rows.
"""
from __future__ import annotations

import re

import pandas as pd

_YEAR_RE = re.compile(r"^(?:y|year)?(19\d{2}|20\d{2})$", re.IGNORECASE)
_QUARTER_RE = re.compile(r"^(?:q[1-4][-_\s]?\d{2,4}|\d{2,4}[-_\s]?q[1-4])$", re.IGNORECASE)
_MONTH_NAMES = frozenset({
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december",
})

_SUMMARY_TOKENS = frozenset({
    "total", "grand total", "subtotal", "average", "mean", "all", "sum", "overall",
})


def detect_wide_temporal_columns(columns: list[str]) -> list[str]:
    """Detect if a set of columns represents sequential time periods in headers."""
    temporal_cols: list[str] = []
    for col in columns:
        c_str = str(col).strip()
        c_clean = c_str.lower()
        if _YEAR_RE.match(c_clean) or _QUARTER_RE.match(c_clean) or c_clean in _MONTH_NAMES:
            temporal_cols.append(col)
        elif re.match(r"^\d{4}[-_/]\d{1,2}(?:[-_/]\d{1,2})?$", c_str):
            temporal_cols.append(col)

    # Need at least 4 sequential temporal columns to qualify as a wide-time table
    if len(temporal_cols) >= 4:
        return temporal_cols
    return []


def unpivot_temporal_frame(
    df: pd.DataFrame,
    temporal_columns: list[str] | None = None,
    var_name: str = "period",
    value_name: str = "value",
) -> pd.DataFrame:
    """Unpivot / melt wide temporal columns into tidy longitudinal format."""
    cols = temporal_columns or detect_wide_temporal_columns(list(df.columns))
    if not cols:
        return df.copy()

    id_vars = [c for c in df.columns if c not in cols]
    if not id_vars:
        # If all columns are temporal, preserve original index as an entity
        melted = df.reset_index().melt(
            id_vars=["index"], value_vars=cols, var_name=var_name, value_name=value_name
        )
    else:
        melted = df.melt(id_vars=id_vars, value_vars=cols, var_name=var_name, value_name=value_name)

    # Clean period values
    cleaned_periods: list[int | str] = []
    for p in melted[var_name]:
        p_str = str(p).strip()
        y_m = _YEAR_RE.match(p_str)
        if y_m:
            cleaned_periods.append(int(y_m.group(1)))
        else:
            cleaned_periods.append(p_str)

    melted[var_name] = cleaned_periods
    return melted


def detect_and_strip_summary_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Detect and remove trailing or inline subtotal/total rows."""
    if df.empty or len(df) <= 5:
        return df, []

    # Check text/categorical columns for summary tokens
    notes: list[str] = []
    mask_to_drop = pd.Series(False, index=df.index)

    for col in df.columns:
        if df[col].dtype == "object":
            series_clean = df[col].astype(str).str.strip().str.lower()
            is_summary = series_clean.isin(_SUMMARY_TOKENS)
            if is_summary.any():
                n_matches = int(is_summary.sum())
                if n_matches <= max(3, int(len(df) * 0.05)):  # Only drop if it's a small minority of rows
                    mask_to_drop = mask_to_drop | is_summary
                    notes.append(f"Quarantined {n_matches} summary/total row(s) detected via '{col}'.")

    if mask_to_drop.any():
        cleaned = df[~mask_to_drop].copy().reset_index(drop=True)
        return cleaned, notes

    return df, []
