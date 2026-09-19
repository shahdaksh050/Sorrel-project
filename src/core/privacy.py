"""
Small-cell suppression: never report a statistic for a group so small that it
could identify the individuals in it.

Every group-level output (a rate, a mean, a median, a chart bar) is only shown
when the group has at least `min_cell_size()` members. Smaller groups are
folded into a single "Other" row that carries only additive counts.

Pure pandas/numpy, O(groups); no heavy imports.
"""
from __future__ import annotations

import os
import re

import pandas as pd

DEFAULT_MIN_CELL_SIZE = 5
_MIN_ALLOWED = 1
_MAX_ALLOWED = 50
_ENV_VAR = "DSA_MIN_CELL_SIZE"

#: Column-name tokens that mark a column as an additive count (safe to sum
#: when groups are merged). Anything else numeric is a statistic (mean, rate,
#: median, CI, p-value ...) that cannot be recombined and is dropped.
_ADDITIVE_TOKENS = frozenset({
    "n", "count", "counts", "headcount", "total", "sum", "size", "number",
    "departed", "leavers", "events", "successes", "selected",
})
_NON_ADDITIVE_TOKENS = frozenset({
    "mean", "median", "avg", "average", "rate", "pct", "percent", "percentage", "share",
    "ratio", "ci", "lower", "upper", "p", "std", "sd", "min", "max", "lift", "effect",
    "value", "adjusted", "score", "gap",
})
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def min_cell_size() -> int:
    """Smallest group size that may be reported (env DSA_MIN_CELL_SIZE,
    default 5, clamped to 1..50). Read at call time."""
    raw = os.environ.get(_ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_MIN_CELL_SIZE
    try:
        value = int(float(raw.strip()))
    except ValueError:
        return DEFAULT_MIN_CELL_SIZE
    return max(_MIN_ALLOWED, min(_MAX_ALLOWED, value))


def is_small(n: int | float) -> bool:
    """True when a group of size `n` is below the reporting minimum."""
    try:
        return float(n) < min_cell_size()
    except (TypeError, ValueError):
        return False


def _is_additive(column: str, n_col: str) -> bool:
    if column == n_col:
        return True
    tokens = {t for t in _TOKEN_SPLIT.split(str(column).lower()) if t}
    return bool(tokens & _ADDITIVE_TOKENS) and not tokens & _NON_ADDITIVE_TOKENS


def fold_small_groups(
    df: pd.DataFrame, n_col: str, label_col: str, other_label: str = "Other (small groups)"
) -> tuple[pd.DataFrame, int]:
    """Merge rows whose `n_col` is below the minimum into one summed row.

    Additive columns (`n_col` and count-like names) are summed; every other
    column is blanked for the merged row, because a mean or rate cannot be
    recombined from summaries. Returns the new frame (large groups in their
    original order, the merged row last) and how many groups were folded. A
    row already labelled `other_label` is merged into the new one but is not
    counted again. The input is never modified.
    """
    if df.empty or n_col not in df.columns or label_col not in df.columns:
        return df, 0
    n_values = pd.to_numeric(df[n_col], errors="coerce")
    is_other = df[label_col].astype(str) == other_label
    small = (n_values < min_cell_size()) | is_other
    folded = int((small & ~is_other).sum())
    if folded == 0:
        return df, 0
    keep = df[~small]
    merged: dict[str, object] = {}
    for column in df.columns:
        if column == label_col:
            merged[column] = other_label
        elif _is_additive(str(column), n_col) and pd.api.types.is_numeric_dtype(df[column]):
            merged[column] = df.loc[small, column].sum()
        else:
            merged[column] = float("nan") if pd.api.types.is_numeric_dtype(df[column]) else None
    other = pd.DataFrame([merged], columns=df.columns)
    out = other if keep.empty else pd.concat([keep, other], ignore_index=True)
    return out, folded


def suppression_note(folded: int) -> str:
    """Plain-language caveat for `folded` suppressed groups."""
    k = min_cell_size()
    noun = "group" if folded == 1 else "groups"
    return (
        f"{folded} {noun} with fewer than {k} people are combined into 'Other' "
        "to avoid identifying individuals."
    )
