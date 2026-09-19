"""
Shared statistical helpers for the analysis tools.

Lives in src/core (not src/tools) so tools don't import each other's
modules for common logic. Pure: pandas/numpy/scipy only, no memory,
controller or tool imports.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.core.profiler import ColumnProfile, DatasetProfile

#: A final resample bucket covering less than this share of a full period
#: (in time AND in rows) is treated as incomplete.
PARTIAL_PERIOD_COVERAGE = 0.9

#: Rows per entity above which rows are repeated measurements of the same
#: entity, not independent observations — inference must aggregate first.
ENTITY_REPEAT_THRESHOLD = 1.5

_MK_MAX_POINTS = 2_000


def measure_aggregation(col: ColumnProfile | None) -> str:
    """How a measure combines across rows within a period or entity: "sum"
    when it is additive (revenue, counts), "mean" when it is not
    (temperature, a rate, a score). Reads the profile's `aggregation` when
    present and falls back to the unit hint (percent -> mean) otherwise."""
    agg = getattr(col, "aggregation", None)
    if agg in ("sum", "mean"):
        return str(agg)
    return "mean" if col is not None and col.unit_hint == "percent" else "sum"


def is_partial_final_period(counts: pd.Series, last_timestamp: pd.Timestamp, freq: str) -> bool:
    """True when the final bucket of a left-labelled resample (`counts` =
    rows per bucket) covers materially less time and fewer rows than a full
    period — the usual shape of an extract that stops mid-month, which
    otherwise reads as a sudden drop in the last period."""
    if len(counts) < 2:
        return False
    start = counts.index[-1]
    end = pd.date_range(start=start, periods=2, freq=freq)[-1]
    # Date-only timestamps (midnight) stand for the whole day they name.
    observed_end = (
        last_timestamp + pd.Timedelta(days=1)
        if last_timestamp == last_timestamp.normalize()
        else last_timestamp
    )
    coverage = (observed_end - start) / (end - start) if end > start else 1.0
    typical_rows = float(counts.iloc[:-1].median())
    return bool(
        coverage < PARTIAL_PERIOD_COVERAGE
        and counts.iloc[-1] < PARTIAL_PERIOD_COVERAGE * typical_rows
    )


def repeated_entity(profile: DatasetProfile | None, df: pd.DataFrame) -> str | None:
    """The entity column whose rows are repeated measurements (e.g. many
    orders per customer, many readings per sensor), or None when rows are
    already independent units."""
    if profile is None:
        return None
    entity = getattr(profile, "entity_col", None)
    rows_per = getattr(profile, "rows_per_entity", None) or 0.0
    if entity and entity in df.columns and rows_per > ENTITY_REPEAT_THRESHOLD:
        return str(entity)
    return None


def aggregate_to_entity(
    df: pd.DataFrame,
    entity_col: str,
    measure: str,
    agg: str,
    by: str | list[str] | None = None,
) -> pd.DataFrame:
    """One row per entity (per group level when `by` is given), so a
    statistical test counts entities, not repeated rows. `agg` is "sum" or
    "mean" (see measure_aggregation). An entity spanning several levels of
    `by` contributes one row to each level it appears in."""
    keys = [entity_col] + ([by] if isinstance(by, str) else list(by or []))
    frame = df[[*keys, measure]].dropna(subset=[measure])
    grouped = frame.groupby(keys, dropna=True, observed=True)[measure]
    out = grouped.sum() if agg == "sum" else grouped.mean()
    return out.reset_index()


def mann_kendall(values: pd.Series | np.ndarray | list[float]) -> dict[str, Any]:
    """Mann-Kendall monotonic trend test with Sen's slope. Returns
    {"tau", "p_value", "sen_slope", "n", "trend": "increasing"|"decreasing"|"no trend"}.
    Robust to non-normal data and outliers, unlike an OLS-slope R²."""
    from scipy import stats

    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    # Pairwise differences are O(n²) memory; callers pass resampled series,
    # but a long daily series is thinned evenly rather than risked.
    if len(x) > _MK_MAX_POINTS:
        x = x[np.linspace(0, len(x) - 1, _MK_MAX_POINTS).astype(int)]
    n = len(x)
    if n < 4:
        return {"tau": None, "p_value": None, "sen_slope": None, "n": n, "trend": "no trend"}
    diffs = np.sign(x[None, :] - x[:, None])[np.triu_indices(n, k=1)]
    s = float(diffs.sum())
    _, tie_counts = np.unique(x, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in tie_counts)) / 18.0
    if var_s <= 0:
        return {"tau": 0.0, "p_value": 1.0, "sen_slope": 0.0, "n": n, "trend": "no trend"}
    z = (s - 1) / math.sqrt(var_s) if s > 0 else (s + 1) / math.sqrt(var_s) if s < 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    i, j = np.triu_indices(n, k=1)
    sen = float(np.median((x[j] - x[i]) / (j - i)))
    tau = s / (n * (n - 1) / 2)
    trend = "no trend" if p >= 0.05 else ("increasing" if s > 0 else "decreasing")
    return {"tau": round(tau, 4), "p_value": p, "sen_slope": sen, "n": n, "trend": trend}
