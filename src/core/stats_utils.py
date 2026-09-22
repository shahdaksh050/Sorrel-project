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


# ---------------------------------------------------------------------------
# Phase 3: Variable-Aware Statistical Helpers
# ---------------------------------------------------------------------------


def circular_statistics(
    values: pd.Series | np.ndarray | list[float],
    high: float = 360.0,
) -> dict[str, Any]:
    """
    Circular statistics for periodic or angular variables (e.g. wind direction 0-360,
    time of day 0-24, day of week 0-7).

    Returns circular mean, circular dispersion, resultant vector length R,
    and Rayleigh test for directional uniformity.
    """
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(x)
    if n == 0:
        return {
            "circular_mean": None,
            "resultant_length": None,
            "circular_dispersion": None,
            "rayleigh_p": None,
            "is_uniform": True,
            "n": 0,
        }

    theta = (x % high) * (2.0 * np.pi / high)
    c = float(np.mean(np.cos(theta)))
    s = float(np.mean(np.sin(theta)))
    r = float(np.sqrt(c**2 + s**2))

    mean_angle_rad = float(np.arctan2(s, c)) % (2.0 * np.pi)
    mean_angle = float(mean_angle_rad * (high / (2.0 * np.pi)))

    circ_dispersion = float(np.sqrt(-2.0 * np.log(max(r, 1e-12)))) if r > 0 else float("inf")

    # Rayleigh test for uniformity (Wilkie 1983 approximation)
    z = n * (r**2)
    p_val = float(
        np.exp(-z)
        * (
            1.0
            + (2.0 * z - z**2) / (4.0 * n)
            - (24.0 * z - 132.0 * (z**2) + 76.0 * (z**3) - 9.0 * (z**4)) / (288.0 * (n**2))
        )
    ) if n >= 10 else float(np.exp(-z))
    p_val = max(0.0, min(1.0, p_val))

    return {
        "circular_mean": round(mean_angle, 2),
        "resultant_length": round(r, 4),
        "circular_dispersion": round(circ_dispersion, 4) if math.isfinite(circ_dispersion) else None,
        "rayleigh_p": round(p_val, 6),
        "is_uniform": bool(p_val >= 0.05),
        "n": n,
    }


def compositional_clr(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Centered log-ratio (clr) transform for compositional data (parts of a whole,
    shares summing to ~1.0 or ~100.0). Avoids spurious negative correlations
    caused by the simplex constraint.
    """
    target_cols = cols or list(df.select_dtypes(include=[np.number]).columns)
    if len(target_cols) < 2:
        return df[target_cols].copy()

    sub_df = df[target_cols].copy().dropna()
    # Replace zeros with small epsilon (half of minimum positive value)
    for c in target_cols:
        pos = sub_df[c][sub_df[c] > 0]
        eps = float(pos.min() * 0.5) if not pos.empty else 1e-5
        sub_df[c] = sub_df[c].clip(lower=eps)

    log_vals = np.log(sub_df.values)
    geom_means = np.mean(log_vals, axis=1, keepdims=True)
    clr_vals = log_vals - geom_means

    return pd.DataFrame(clr_vals, index=sub_df.index, columns=[f"{c}_clr" for c in target_cols])


def benford_analysis(
    values: pd.Series | np.ndarray | list[float],
) -> dict[str, Any]:
    """
    Benford's Law first-digit analysis for financial, accounting, and integrity audits.
    Evaluates Mean Absolute Deviation (MAD) and chi-square goodness of fit.
    """
    from scipy import stats

    x = pd.Series(values).dropna()
    numeric_x = pd.to_numeric(x, errors="coerce").dropna()
    pos_x = numeric_x[numeric_x > 0]
    n = len(pos_x)
    if n < 30:
        return {
            "n": n,
            "mad": None,
            "conformity": "insufficient_data",
            "chi2_stat": None,
            "p_value": None,
            "first_digit_counts": {},
        }

    # Extract first significant digit
    first_digits: list[int] = []
    for val in pos_x:
        s = f"{val:.10e}"
        for ch in s:
            if ch in "123456789":
                first_digits.append(int(ch))
                break

    digit_counts = dict.fromkeys(range(1, 10), 0)
    for d in first_digits:
        digit_counts[d] += 1

    obs_props = np.array([digit_counts[d] / n for d in range(1, 10)])
    exp_props = np.array([np.log10(1.0 + 1.0 / d) for d in range(1, 10)])

    mad = float(np.mean(np.abs(obs_props - exp_props)))

    # Nigrini MAD conformity thresholds
    if mad < 0.006:
        conformity = "close"
    elif mad < 0.012:
        conformity = "acceptable"
    elif mad < 0.015:
        conformity = "marginally_acceptable"
    else:
        conformity = "non_conforming"

    # Chi-square test
    obs_counts = np.array([digit_counts[d] for d in range(1, 10)])
    exp_counts = exp_props * n
    chi2_stat, p_val = stats.chisquare(obs_counts, f_exp=exp_counts)

    return {
        "n": n,
        "mad": round(mad, 5),
        "conformity": conformity,
        "chi2_stat": round(float(chi2_stat), 2),
        "p_value": round(float(p_val), 6),
        "first_digit_counts": digit_counts,
        "first_digit_props": {d: round(obs_props[d - 1], 4) for d in range(1, 10)},
    }


def digit_heaping_test(
    values: pd.Series | np.ndarray | list[float],
) -> dict[str, Any]:
    """
    Test for digit heaping / rounding preferences (e.g. excess of 0s and 5s
    in ages, prices, or self-reported quantities) using Whipple's Index
    and chi-square terminal digit test.
    """
    from scipy import stats

    x = pd.Series(values).dropna()
    num = pd.to_numeric(x, errors="coerce").dropna()
    n = len(num)
    if n < 20:
        return {"n": n, "whipples_index": None, "heaping_detected": False}

    # Extract terminal digits of rounded integers
    terminals = (np.abs(np.round(num.values)).astype(int)) % 10
    counts = {d: int(np.sum(terminals == d)) for d in range(10)}

    # Whipple's index: 5 * (n0 + n5) / N * 100
    whipple = float(5.0 * (counts[0] + counts[5]) / n * 100.0)

    # Chi-square vs uniform (expected n/10 per digit)
    obs = np.array([counts[d] for d in range(10)])
    exp = np.full(10, n / 10.0)
    chi2_stat, p_val = stats.chisquare(obs, f_exp=exp)

    heaping_detected = bool(whipple > 125.0 or (p_val < 0.01 and (counts[0] + counts[5]) > (0.30 * n)))

    return {
        "n": n,
        "whipples_index": round(whipple, 1),
        "heaping_detected": heaping_detected,
        "terminal_counts": counts,
        "chi2_stat": round(float(chi2_stat), 2),
        "p_value": round(float(p_val), 6),
        "recommendation": "Suggest binning or interval analysis due to rounding" if heaping_detected else "No significant heaping detected",
    }


def cliffs_delta(
    x: pd.Series | np.ndarray | list[float],
    y: pd.Series | np.ndarray | list[float],
) -> dict[str, Any]:
    """
    Cliff's delta non-parametric effect size for ordinal or non-normal data:
    delta = (#(x > y) - #(x < y)) / (n_x * n_y).
    """
    from scipy import stats

    vx = np.asarray(pd.Series(x).dropna(), dtype=float)
    vy = np.asarray(pd.Series(y).dropna(), dtype=float)
    nx, ny = len(vx), len(vy)
    if nx == 0 or ny == 0:
        return {"delta": None, "magnitude": "none", "p_value": None}

    # Pairwise comparison matrix
    diff = vx[:, None] - vy[None, :]
    greater = int(np.sum(diff > 0))
    less = int(np.sum(diff < 0))
    delta = float((greater - less) / (nx * ny))

    # Magnitude thresholds (Romano et al. 2006)
    abs_d = abs(delta)
    if abs_d < 0.147:
        magnitude = "negligible"
    elif abs_d < 0.33:
        magnitude = "small"
    elif abs_d < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    # Mann-Whitney U p-value
    try:
        _u_stat, p_val = stats.mannwhitneyu(vx, vy, alternative="two-sided")
    except Exception:
        p_val = 1.0

    return {
        "delta": round(delta, 4),
        "magnitude": magnitude,
        "p_value": round(float(p_val), 6),
        "nx": nx,
        "ny": ny,
    }


def count_target_diagnostics(
    y: pd.Series | np.ndarray | list[float],
) -> dict[str, Any]:
    """
    Diagnose whether a target variable is a count, whether it is overdispersed,
    and whether zero inflation is present.
    Recommends: 'ols', 'poisson', 'negative_binomial', or 'zero_inflated'.
    """
    vals = pd.Series(y).dropna()
    num = pd.to_numeric(vals, errors="coerce").dropna()
    n = len(num)
    if n < 10:
        return {"is_count": False, "recommended_model": "ols"}

    is_all_nonneg = bool((num >= 0).all())
    is_all_int = bool(np.all(num == np.round(num)))

    if not (is_all_nonneg and is_all_int):
        return {"is_count": False, "recommended_model": "ols"}

    mean_val = float(num.mean())
    var_val = float(num.var()) if n > 1 else 0.0
    dispersion_ratio = float(var_val / mean_val) if mean_val > 0 else 1.0
    zero_count = int((num == 0).sum())
    zero_prop = float(zero_count / n)
    expected_zero_prop = float(np.exp(-mean_val)) if mean_val > 0 else 1.0

    is_overdispersed = bool(dispersion_ratio > 1.25)
    is_zero_inflated = bool(zero_prop > 0.25 and zero_prop > 1.5 * expected_zero_prop)

    if is_zero_inflated:
        rec = "zero_inflated"
    elif is_overdispersed:
        rec = "negative_binomial"
    else:
        rec = "poisson"

    return {
        "is_count": True,
        "mean": round(mean_val, 4),
        "variance": round(var_val, 4),
        "dispersion_ratio": round(dispersion_ratio, 4),
        "zero_proportion": round(zero_prop, 4),
        "expected_zero_proportion": round(expected_zero_prop, 4),
        "is_overdispersed": is_overdispersed,
        "is_zero_inflated": is_zero_inflated,
        "recommended_model": rec,
    }


def robust_location_dispersion(
    x: pd.Series | np.ndarray | list[float],
) -> dict[str, Any]:
    """
    Compute robust location and dispersion statistics (median, MAD, trimmed mean, IQR)
    for heavy-tailed or skewed variables.
    """
    from scipy import stats

    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(arr)
    if n == 0:
        return {}

    median_val = float(np.median(arr))
    mad_val = float(stats.median_abs_deviation(arr, scale="normal")) if n > 1 else 0.0
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr_val = float(q75 - q25)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if std_val < 1e-12:
        skew_val = 0.0
        kurt_val = 0.0
    else:
        skew_val = float(stats.skew(arr)) if n > 2 else 0.0
        kurt_val = float(stats.kurtosis(arr)) if n > 3 else 0.0

    trimmed_5 = float(stats.trim_mean(arr, 0.05)) if n >= 20 else mean_val
    trimmed_10 = float(stats.trim_mean(arr, 0.10)) if n >= 20 else mean_val

    is_heavy_tailed = bool(abs(skew_val) > 2.0 or kurt_val > 5.0)

    return {
        "n": n,
        "median": round(median_val, 4),
        "mad": round(mad_val, 4),
        "iqr": round(iqr_val, 4),
        "q25": round(q25, 4),
        "q75": round(q75, 4),
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "trimmed_mean_5pct": round(trimmed_5, 4),
        "trimmed_mean_10pct": round(trimmed_10, 4),
        "skewness": round(skew_val, 4),
        "kurtosis": round(kurt_val, 4),
        "is_heavy_tailed": is_heavy_tailed,
        "preferred_location": "median" if is_heavy_tailed else "mean",
    }

