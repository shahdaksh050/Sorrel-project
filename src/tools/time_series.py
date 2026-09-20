"""
Time-Series Analysis Tool — Execution Layer.

Stage 3: Trend, stationarity, and autocorrelation diagnostics for datasets
with a genuine time axis (gated on DatasetProfile.is_time_series).

7.7 (grain-aware time series): trend/ADF/seasonality diagnostics are only
statistically meaningful over *one value per calendar period* — not over
raw transaction rows in file order, where multiple rows can share a
timestamp or sit minutes apart. Every row here is therefore resampled to a
natural grain (day/week/month, chosen from the observed date span) and
aggregated with the measure-appropriate reducer (sum for additive
money/count measures, mean for rates) before any diagnostic runs. The
chosen grain and aggregation are reported in the output so the choice is
auditable rather than buried, and a calendar-aware seasonality read
(month-of-year, and day-of-week when the grain is fine enough) supplements
the lag-autocorrelation check, which only ever sees the row-order structure.

The resampled series is calendar-aligned: a period with no rows stays in
the series as missing (NaN), never silently dropped or zero-filled, so
lags mean "N calendar periods". Trend significance is Mann-Kendall with
Sen's slope (the linear fit is kept for reference), and a seasonal
decomposition (STL) runs only once two full cycles are available.
Entity x time panels also get a per-entity trend read.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from src.core.findings import Finding
from src.core.multiple_testing import apply_benjamini_hochberg
from src.core.profiler import pick_measures, profile_dataframe
from src.core.stats_utils import (
    is_partial_final_period,
    mann_kendall,
    measure_aggregation,
    repeated_entity,
)
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

#: Candidate seasonal lags checked via autocorrelation (weekly/monthly/yearly-ish).
#: Lag sets are grain-relative (a lag is "N periods", not "N days") — a lag
#: of 12 means a year on monthly data and a quarter on weekly data, so the
#: candidates checked must change with the chosen grain or the numbers stop
#: meaning anything once the series is resampled instead of raw daily rows.
_SEASONAL_LAGS_BY_GRAIN = {
    "daily": (7, 30, 365),
    "weekly": (4, 13, 52),
    "monthly": (3, 6, 12),
}

#: |autocorrelation| at or above this is reported as a seasonal signal.
_SEASONALITY_THRESHOLD = 0.3

#: Two-sided ADF p-value at/below this rejects the unit-root (non-stationary) null.
_ADF_ALPHA = 0.05

#: Mann-Kendall p-value below which a monotonic trend is reported. R² alone
#: is not a significance test — a short noisy series can clear any R² bar.
_TREND_ALPHA = 0.05

#: Above this share of missing calendar periods, gaps are too large to
#: interpolate: autocorrelation and decomposition are skipped, and trend
#: tests run on the observed periods only.
_MAX_MISSING_SHARE = 0.2

#: Seasonal cycle length (in periods) decomposed at each grain.
_SEASONAL_PERIOD = {"daily": 7, "weekly": 52, "monthly": 12}

#: Month-of-year tests are BH-corrected at this level before any month
#: becomes a finding.
_MONTH_ALPHA = 0.05

#: Entity x time panels: per-entity trends for the entities with the most
#: rows; a direction held by more than half of them is a majority trend,
#: and at least this share moving each way is divergence.
_PANEL_MAX_ENTITIES = 20
_PANEL_DIVERGENCE_SHARE = 0.25
_PANEL_TOP_MOVERS = 3

# ---------------------------------------------------------------------------
# 7.7 — grain selection. Chosen from the observed date span: a multi-year
# range is summarised monthly, a multi-month range weekly, anything shorter
# daily. If the coarse grain picked from the span doesn't leave enough
# periods to diagnose (e.g. a "monthly" series with 3 points), step down to
# the next finer candidate — that is the row-density half of the heuristic.
# ---------------------------------------------------------------------------
_GRAIN_MONTHLY_SPAN_DAYS = 545   # ~1.5 years+ of history -> monthly buckets
_GRAIN_WEEKLY_SPAN_DAYS = 90     # a few months of history -> weekly buckets
_MIN_PERIODS_FOR_GRAIN = 6

_GRAIN_FREQ = {"daily": "D", "weekly": "W", "monthly": "MS"}
_GRAIN_LABEL = {"daily": "Daily", "weekly": "Weekly", "monthly": "Monthly"}
_PERIOD_UNIT = {"daily": "day", "weekly": "week", "monthly": "month"}

#: A calendar month running at/above this fraction away from the yearly
#: average is reported as a seasonal Finding (T5 triviality suppression
#: keeps unremarkable months out of the finding list).
_MONTH_LIFT_THRESHOLD = 0.15

#: A month's factor is only *published* as a Finding once the series has
#: seen that month in at least this many distinct calendar years — one
#: weekly series spanning a single year produces a "month factor" for every
#: month from a single observation each, which is a single number dressed
#: up as a repeating season. Below this, the month stays in the tool's
#: output/summary (marked "single-year, unreplicated") but not the finding
#: list.
_MIN_YEARS_FOR_MONTH_FINDING = 2

#: At most this many calendar-month findings are published per run, ranked
#: by |lift| — a single weekly/daily series has 12 months' worth of factors
#: and publishing all of them (as opposed to, say, one real trend finding)
#: is what let 7 month findings crowd the top of a 12-finding report.
_MAX_MONTH_FINDINGS = 3

#: Periods needed to cover roughly two full annual cycles at each grain —
#: below this, a month-of-year factor and an underlying linear trend are
#: confounded (a later calendar month is also later in time), so a month
#: finding computed from less history than this carries a caveat rather
#: than being reported as a clean repeating seasonal effect.
_PERIODS_FOR_TWO_YEARS = {"daily": 730, "weekly": 104, "monthly": 24}

_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


def _autodetect_datetime_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            return str(col)
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        sample = series.dropna().head(20)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.9:
            return str(col)
    return None


def _fallback_numeric_column(df: pd.DataFrame, date_column: str) -> str | None:
    """Last-resort pick when no profiled measure is available at all: first
    remaining numeric column. Kept as a floor so the tool never fails purely
    because profiling didn't recognise anything as a measure."""
    numeric = [c for c in df.select_dtypes(include="number").columns if c != date_column]
    return str(numeric[0]) if numeric else None


def _choose_value_column_and_aggregation(
    df: pd.DataFrame, date_column: str, requested: str | None
) -> tuple[str | None, str, DatasetProfile | None]:
    """
    Resolve (value_column, aggregation, profile).

    An explicitly requested column always wins on *which* column — only the
    automatic pick is profile-driven. Either way, `aggregation` follows the
    measure's additivity (`measure_aggregation`): "sum" for money/counts,
    "mean" for non-additive measures such as rates, levels or readings. When multiple measures are candidates, one with
    `unit_hint == "currency"` is preferred over an arbitrary numeric pick —
    a revenue question is the most common one — otherwise the profile's
    highest-variance measure is used.
    """
    try:
        profile = profile_dataframe(df)
    except Exception:
        profile = None

    if requested and requested in df.columns:
        col_profile = next(
            (c for c in (profile.columns if profile else []) if c.name == requested), None
        )
        return requested, measure_aggregation(col_profile), profile

    if profile is not None:
        measures = [c for c in profile.measures() if c.name != date_column and c.name in df.columns]
        if measures:
            currency = [c for c in measures if c.unit_hint == "currency"]
            pool = currency if currency else measures
            # Raw variance depends on units (a 1,000-scale sensor always beats a
            # 10-scale concentration); the shared ranking is objective-aware,
            # completeness-first and unit-free.
            allowed = {c.name for c in pool}
            ranked = pick_measures(profile, os.environ.get("USER_OBJECTIVE", "").strip())
            best = next((c for c in ranked if c.name in allowed), None) or max(
                pool, key=lambda c: (c.stats.get("std") or 0.0) ** 2
            )
            return best.name, measure_aggregation(best), profile

    return _fallback_numeric_column(df, date_column), "sum", profile


def _choose_grain_candidates(dates: pd.Series) -> list[str]:
    """Coarsest-appropriate-first candidate list, from the observed span."""
    ordered = dates.dropna().sort_values()
    span_days = float((ordered.iloc[-1] - ordered.iloc[0]).days) if len(ordered) > 1 else 0.0
    if span_days >= _GRAIN_MONTHLY_SPAN_DAYS:
        return ["monthly", "weekly", "daily"]
    if span_days >= _GRAIN_WEEKLY_SPAN_DAYS:
        return ["weekly", "daily"]
    return ["daily"]


def _resample(working: pd.DataFrame, grain: str, aggregation: str) -> pd.DataFrame:
    """One row per calendar period from the first observed period to the
    last: sum (additive measures) or mean (rates). A period with no rows is
    kept as NaN — missing, not zero. Only an event log (one row per event)
    makes an empty period a genuine zero, and nothing here can confirm the
    data is one, so absence is not fabricated into a value."""
    indexed = working.set_index("_date")["_value"]
    # Left-labelled so every period is named by its start (weeks included),
    # which is what is_partial_final_period measures coverage from.
    resampled = indexed.resample(_GRAIN_FREQ[grain], label="left", closed="left")
    agg = resampled.sum(min_count=1) if aggregation == "sum" else resampled.mean()
    return pd.DataFrame({"_period": agg.index, "_value": agg.to_numpy(dtype=float)})


def _resample_series(
    working: pd.DataFrame, aggregation: str, candidates: list[str]
) -> tuple[pd.DataFrame | None, str]:
    """Try grains coarsest-first; step down to the next finer grain if the
    resampled series has too few observed periods to diagnose. Always falls
    back to the finest candidate's result if none clear the bar."""
    for candidate in candidates:
        resampled = _resample(working, candidate, aggregation)
        observed = int(resampled["_value"].notna().sum())
        if observed >= _MIN_PERIODS_FOR_GRAIN or candidate == candidates[-1]:
            return resampled, candidate
    return None, candidates[-1]


def _fill_small_gaps(aligned: pd.Series) -> tuple[pd.Series, bool]:
    """(series, gap_free): a calendar-aligned series with its missing
    periods linearly interpolated when they are at most _MAX_MISSING_SHARE
    of it — lags and Sen's slope then count calendar periods. Larger gaps
    are left as NaN (gap_free=False) rather than invented."""
    if not aligned.isna().any():
        return aligned, True
    if float(aligned.isna().mean()) > _MAX_MISSING_SHARE:
        return aligned, False
    return aligned.interpolate(limit_direction="both"), True


def _seasonal_decomposition(filled: pd.Series, grain: str) -> dict[str, Any] | None:
    """Seasonal strength (1 - var(resid) / var(seasonal + resid), 0..1) and
    the peak/trough season from an STL decomposition, falling back to a
    classical seasonal_decompose. None when fewer than two full cycles are
    available or statsmodels cannot decompose the series."""
    period = _SEASONAL_PERIOD[grain]
    if len(filled) < 2 * period:
        return None
    try:
        from statsmodels.tsa.seasonal import STL

        result: Any = STL(filled, period=period, robust=True).fit()
        method = "STL"
    except Exception:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            result = seasonal_decompose(
                filled, model="additive", period=period, extrapolate_trend="freq"
            )
            method = "seasonal_decompose"
        except Exception:
            return None
    seasonal = pd.Series(np.asarray(result.seasonal, dtype=float), index=filled.index)
    resid = np.asarray(result.resid, dtype=float)
    mask = ~(np.isnan(seasonal.to_numpy()) | np.isnan(resid))
    denom = float(np.var(seasonal.to_numpy()[mask] + resid[mask]))
    strength = max(0.0, 1.0 - float(np.var(resid[mask])) / denom) if denom > 0 else 0.0
    index = pd.DatetimeIndex(filled.index)
    if grain == "monthly":
        labels = pd.Index([_MONTH_NAMES[m] for m in index.month])
    elif grain == "weekly":
        labels = pd.Index([f"week {w}" for w in index.isocalendar().week])
    else:
        labels = pd.Index(index.day_name())
    by_season = seasonal.groupby(labels.to_numpy()).mean()
    return {
        "method": method,
        "period": period,
        "strength": round(strength, 4),
        "peak_season": str(by_season.idxmax()),
        "trough_season": str(by_season.idxmin()),
    }


def _panel_trends(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    entity: str,
    grain: str,
    aggregation: str,
    periods: pd.DatetimeIndex,
) -> dict[str, Any] | None:
    """Mann-Kendall per entity for the _PANEL_MAX_ENTITIES entities with the
    most rows, each resampled on the aggregate's grain and calendar range so
    their trends are comparable. Shares are of entities with a computable
    test (>= 4 observed periods)."""
    frame = pd.DataFrame({
        "_entity": df[entity],
        "_date": pd.to_datetime(df[date_column], errors="coerce", format="mixed"),
        "_value": pd.to_numeric(df[value_column], errors="coerce"),
    }).dropna()
    top = frame["_entity"].value_counts().head(_PANEL_MAX_ENTITIES).index
    rows: list[dict[str, Any]] = []
    for level, sub in frame[frame["_entity"].isin(top)].groupby("_entity", observed=True):
        aligned = (
            _resample(sub, grain, aggregation).set_index("_period")["_value"].reindex(periods)
        )
        filled, gap_free = _fill_small_gaps(aligned)
        mk = mann_kendall(filled if gap_free else aligned.dropna())
        if mk["p_value"] is None:
            continue
        rows.append({
            "entity": str(level),
            "periods_observed": int(aligned.notna().sum()),
            "mk_tau": mk["tau"],
            "mk_p_value": round(mk["p_value"], 6),
            "sen_slope": round(mk["sen_slope"], 6),
            "trend": mk["trend"],
        })
    if not rows:
        return None
    n = len(rows)
    up = [r for r in rows if r["trend"] == "increasing"]
    down = [r for r in rows if r["trend"] == "decreasing"]
    return {
        "entity_column": entity,
        "entities_analysed": n,
        "entities_increasing": len(up),
        "entities_decreasing": len(down),
        "share_increasing": round(len(up) / n, 4),
        "share_decreasing": round(len(down) / n, 4),
        "strongest_risers": [
            r["entity"] for r in sorted(up, key=lambda r: -r["mk_tau"])[:_PANEL_TOP_MOVERS]
        ],
        "strongest_fallers": [
            r["entity"] for r in sorted(down, key=lambda r: r["mk_tau"])[:_PANEL_TOP_MOVERS]
        ],
        "entity_trends": rows,
    }


def _month_of_year_factors(
    period_dates: pd.Series, values: np.ndarray
) -> tuple[dict[str, float], dict[str, int], dict[str, int], dict[str, float | None]]:
    """Each calendar month's average vs. the overall mean, as a signed
    fraction (0.4 = 40% above baseline). Computed on the resampled series so
    a monthly grain compares each month's own point, and a weekly/daily
    grain compares the months' averaged points.

    Alongside the lift, also returns:
      - `years`: how many *distinct calendar years* contributed periods to
        that month — a month factor computed from a single year is one
        observation, not a replicated seasonal effect, however many
        sub-periods (weeks/days) that one year contains.
      - `p_values`: a Welch t-test (that month's periods vs. every other
        period) p-value, where there's enough data to run one — a simple
        significance check rather than treating any |lift| >= threshold as
        equally credible regardless of sample size or spread.
    """
    if len(values) < 4:
        return {}, {}, {}, {}
    s = pd.Series(values, index=pd.DatetimeIndex(period_dates))
    overall_mean = float(s.mean())
    if overall_mean == 0:
        return {}, {}, {}, {}
    grouped = s.groupby(s.index.month)
    factors: dict[str, float] = {}
    counts: dict[str, int] = {}
    years: dict[str, int] = {}
    p_values: dict[str, float | None] = {}
    for month_num, group in grouped:
        name = _MONTH_NAMES[int(month_num)]
        factors[name] = round((float(group.mean()) - overall_mean) / abs(overall_mean), 4)
        counts[name] = int(group.size)
        years[name] = int(pd.DatetimeIndex(group.index).year.nunique())

        rest = s[~s.index.isin(group.index)]
        p_val: float | None = None
        if len(group) >= 2 and len(rest) >= 2 and group.nunique() >= 2:
            try:
                _stat, p = stats.ttest_ind(
                    group.to_numpy(dtype=float), rest.to_numpy(dtype=float), equal_var=False
                )
                p_val = None if np.isnan(p) else round(float(p), 4)
            except (ValueError, ZeroDivisionError):
                p_val = None
        p_values[name] = p_val
    return factors, counts, years, p_values


def _day_of_week_factors(working: pd.DataFrame, aggregation: str) -> dict[str, float]:
    """Weekday effect from a *daily* aggregation of the raw rows, independent
    of the macro grain — weekday patterns only exist at day resolution, so a
    weekly-grain analysis still checks them against the underlying days."""
    daily = _resample(working, "daily", aggregation).dropna()
    if len(daily) < 7:
        return {}
    s = pd.Series(daily["_value"].to_numpy(dtype=float), index=pd.DatetimeIndex(daily["_period"]))
    overall_mean = float(s.mean())
    if overall_mean == 0:
        return {}
    by_dow = s.groupby(s.index.day_name()).mean()
    return {str(k): round((float(v) - overall_mean) / abs(overall_mean), 4) for k, v in by_dow.items()}


#: Time-of-day / day-of-week profile: only for sub-daily data (median gap at
#: most this long) with at least this many days of coverage.
_TOD_MAX_GAP = pd.Timedelta(hours=6)
_TOD_MIN_DAYS = 14
_TOD_MIN_OBS_PER_HOUR = 20
_TOD_MIN_HOURS = 4
_TOD_MIN_OBS_PER_WEEKDAY = 10
#: Measures screened for an hour effect besides the tool's own (first N by
#: profile order), and how many of them may be reported.
_TOD_MAX_CANDIDATES = 8
_TOD_MAX_EXTRA = 3
#: Extra measures need this share of usable rows: a 90%-missing column is not evidence
#: of a cycle worth headlining, however clean its eta^2 looks.
_TOD_MIN_COMPLETENESS = 0.5
#: A cycle is real only at eta^2 >= this and an ANOVA p below _TOD_ALPHA (the
#: p guards short samples, where eta^2 alone is inflated by group count).
_TOD_MIN_ETA_SQ = 0.05
_TOD_ALPHA = 0.01
#: Weekend-vs-weekday gap is quoted in the headline from this size up.
_TOD_WEEKEND_GAP_MIN = 0.10
_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _num(v: float) -> str:
    a = abs(v)
    return f"{v:,.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}"


def _day_part(hour: int) -> str:
    return (
        "morning" if 5 <= hour < 11 else "midday" if 11 <= hour < 14
        else "afternoon" if 14 <= hour < 17 else "evening" if 17 <= hour < 22 else "night"
    )


def _hour_eta_squared(frame: pd.DataFrame, hours: np.ndarray) -> pd.Series:
    """eta^2 of hour-of-day groups for every column at once (one groupby)."""
    grouped = frame.groupby(hours)
    grand = frame.mean()
    between = (grouped.count() * (grouped.mean() - grand) ** 2).sum()
    total = ((frame - grand) ** 2).sum()
    return (between / total.where(total > 0)).dropna()


def _tod_charts(name: str, hours: dict[int, float], weekdays: dict[str, float] | None, mean: float) -> tuple[Any, Any]:
    from src.core.chart_spec import validate_chart_spec

    hour_chart, _ = validate_chart_spec({
        "type": "line", "x": "hour", "y": "average", "x_title": "Hour of day", "y_title": name,
        "data": [{"hour": h, "average": v} for h, v in hours.items()],
        "title": f"When {name} is highest: average by hour of day",
        "annotations": [{"y": round(mean, 4), "label": "overall average"}],
    })
    day_chart = None
    if weekdays:
        day_chart, _ = validate_chart_spec({
            "type": "bar", "x": "weekday", "y": "average", "x_title": "Day of week", "y_title": name,
            "data": [{"weekday": d, "average": v} for d, v in weekdays.items()],
            "title": f"{name} by day of week: average",
            "annotations": [{"y": round(mean, 4), "label": "overall average"}],
        })
    return hour_chart, day_chart


def _measure_tod_profile(
    name: str, values: pd.Series, dates: pd.Series
) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
    """(diurnal, weekly_profile) for one measure, or None when there is no
    real hour-of-day cycle (eta^2 / ANOVA gate) or too few observations."""
    ok = values.notna().to_numpy() & dates.notna().to_numpy()
    v = values.to_numpy(dtype=float)[ok]
    ts = pd.DatetimeIndex(dates[ok])
    hour = np.asarray(ts.hour)
    by_hour = pd.DataFrame({"h": hour, "v": v}).groupby("h")["v"].agg(["mean", "count"])
    by_hour = by_hour[by_hour["count"] >= _TOD_MIN_OBS_PER_HOUR]
    if len(by_hour) < _TOD_MIN_HOURS:
        return None
    keep = np.isin(hour, by_hour.index.to_numpy())
    n, k = int(keep.sum()), len(by_hour)
    grand = float(v[keep].mean())
    ss_tot = float(np.sum((v[keep] - grand) ** 2))
    if grand == 0 or ss_tot <= 0 or n <= k:
        return None
    ss_between = float((by_hour["count"] * (by_hour["mean"] - grand) ** 2).sum())
    eta_sq = ss_between / ss_tot
    if eta_sq >= 1.0:
        return None
    f_stat = (eta_sq / (k - 1)) / ((1 - eta_sq) / (n - k))
    p_value = float(stats.f.sf(f_stat, k - 1, n - k))
    if eta_sq < _TOD_MIN_ETA_SQ or p_value >= _TOD_ALPHA:
        return None
    means = by_hour["mean"]
    peak_hour, trough_hour = int(means.idxmax()), int(means.idxmin())
    peak, trough = float(means.max()), float(means.min())
    hours = {int(h): round(float(m), 4) for h, m in means.items()}

    weekly: dict[str, Any] | None = None
    dow = np.asarray(ts.dayofweek)
    by_dow = pd.DataFrame({"d": dow, "v": v}).groupby("d")["v"].agg(["mean", "count"])
    if len(by_dow) == 7 and int(by_dow["count"].min()) >= _TOD_MIN_OBS_PER_WEEKDAY:
        weekday_mean, weekend_mean = float(v[dow < 5].mean()), float(v[dow >= 5].mean())
        days = {_WEEKDAY_NAMES[int(d)]: round(float(m), 4) for d, m in by_dow["mean"].items()}
        weekly = {
            "means": days,
            "weekday_mean": round(weekday_mean, 4),
            "weekend_mean": round(weekend_mean, 4),
            "weekend_gap": round((weekend_mean - weekday_mean) / abs(weekday_mean), 4) if weekday_mean else None,
            "eta_squared": round(float(
                (by_dow["count"] * (by_dow["mean"] - float(v.mean())) ** 2).sum()
                / float(np.sum((v - v.mean()) ** 2))
            ), 4),
            "n": len(v),
        }
    hour_chart, day_chart = _tod_charts(name, hours, weekly["means"] if weekly else None, grand)
    if weekly is not None:
        weekly["chart"] = day_chart
    diurnal = {
        "hourly_means": hours,
        "hourly_counts": {int(h): int(c) for h, c in by_hour["count"].items()},
        "peak_hour": peak_hour, "peak_value": round(peak, 4),
        "trough_hour": trough_hour, "trough_value": round(trough, 4),
        "peak_trough_ratio": round(peak / trough, 3) if trough > 0 and peak > 0 else None,
        "effect": round((peak - trough) / abs(grand), 4),
        "overall_mean": round(grand, 4),
        "eta_squared": round(eta_sq, 4), "p_value": p_value, "n": n,
        "chart": hour_chart,
    }
    return diurnal, weekly


def _time_of_day_profile(
    df: pd.DataFrame,
    dates: pd.Series,
    date_column: str,
    value_column: str,
    profile: DatasetProfile | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Hour-of-day and day-of-week profile for sub-daily data: the tool's own
    measure plus up to _TOD_MAX_EXTRA more with the strongest hour effect
    (screened over at most _TOD_MAX_CANDIDATES others). None when the axis is
    not sub-daily / too short, or no measure has a real cycle."""
    stamps = pd.DatetimeIndex(dates.dropna().drop_duplicates().sort_values())
    if len(stamps) < 2 * _TOD_MIN_HOURS or (stamps[-1] - stamps[0]).days < _TOD_MIN_DAYS:
        return None
    if stamps.to_series().diff().median() > _TOD_MAX_GAP:
        return None
    names = [c.name for c in profile.measures()] if profile is not None else list(df.select_dtypes("number").columns)
    others = [c for c in dict.fromkeys(names) if c not in (date_column, value_column) and c in df.columns]
    frame = df[others[:_TOD_MAX_CANDIDATES]].apply(pd.to_numeric, errors="coerce")
    valid = dates.notna().to_numpy()
    extras: list[str] = []
    if frame.shape[1]:
        eta = _hour_eta_squared(frame[valid], np.asarray(pd.DatetimeIndex(dates[valid]).hour))
        # ~1.0 is a clock field in disguise, not a measured quantity.
        complete = frame[valid].notna().mean()
        eta = eta[complete.reindex(eta.index).fillna(0.0) >= _TOD_MIN_COMPLETENESS]
        extras = [str(c) for c in eta[eta < 0.999].sort_values(ascending=False).index[:_TOD_MAX_EXTRA]]
    diurnal: dict[str, Any] = {}
    weekly: dict[str, Any] = {}
    for i, col in enumerate([value_column, *extras]):
        values = pd.to_numeric(df[col], errors="coerce")
        got = _measure_tod_profile(col, values, dates)
        if got is None:
            continue
        diurnal[col] = {**got[0], "is_primary": i == 0}
        if got[1] is not None:
            weekly[col] = got[1]
    return (diurnal, weekly) if diurnal else None


class TimeSeriesAnalysisTool(BaseTool):
    """Trend direction, stationarity, and seasonality diagnostics for a time-indexed value."""

    name = "time_series_analysis"
    description = (
        "Resample a time-indexed numeric column to its natural grain (day/week/month) "
        "and analyse: trend direction/slope, stationarity (Augmented Dickey-Fuller test), "
        "lag autocorrelation, and calendar-aware seasonality (month-of-year, and "
        "day-of-week when the grain is fine enough). Use when the data profile shows a "
        "datetime column (is_time_series). Auto-detects date_column and value_column "
        "(preferring a revenue/currency measure) when omitted."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return 1.0 if profile is not None and profile.is_time_series else 0.0

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        date_column: str | None = None,
        value_column: str | None = None,
        target_column: str | None = None,
        aggregation: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)

        if date_column is None:
            date_column = _autodetect_datetime_column(df)
        if date_column is None or date_column not in df.columns:
            raise ToolExecutionError(
                "No usable datetime column found. Pass date_column explicitly."
            )

        requested_value_column = value_column or target_column
        value_column, inferred_aggregation, profile = _choose_value_column_and_aggregation(
            df, date_column, requested_value_column
        )
        if value_column is None or value_column not in df.columns:
            raise ToolExecutionError(
                "No numeric value_column found to analyse. Pass value_column explicitly."
            )
        if aggregation not in ("sum", "mean"):
            aggregation = inferred_aggregation

        dates = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
        raw_values = pd.to_numeric(df[value_column], errors="coerce")
        working = pd.DataFrame({"_date": dates, "_value": raw_values}).dropna()
        working = working.sort_values("_date")

        if len(working) < 10:
            raise ToolExecutionError(
                f"Only {len(working)} usable (date, value) rows after dropping missing "
                f"values — need at least 10 for time-series diagnostics."
            )

        # ---- Resample to a natural grain before diagnosing anything (7.7) ----
        grain_candidates = _choose_grain_candidates(working["_date"])
        resampled, grain = _resample_series(working, aggregation, grain_candidates)
        if resampled is None or int(resampled["_value"].notna().sum()) < 3:
            n = 0 if resampled is None else int(resampled["_value"].notna().sum())
            raise ToolExecutionError(
                f"Only {n} usable periods after resampling to {grain} — need at least 3 "
                f"for time-series diagnostics."
            )

        # The final bucket of an extract is usually incomplete; left in, a
        # half-month of sales reads as a collapse at the end of the series.
        dropped_partial_period: str | None = None
        counts = (
            working.set_index("_date")["_value"]
            .resample(_GRAIN_FREQ[grain], label="left", closed="left")
            .count()
        )
        counts = counts[counts > 0]
        if len(resampled) > 3 and is_partial_final_period(
            counts, working["_date"].iloc[-1], _GRAIN_FREQ[grain]
        ):
            dropped_partial_period = str(pd.Timestamp(resampled["_period"].iloc[-1]).date())
            resampled = resampled.iloc[:-1]

        # ---- Calendar gaps: missing periods stay in the series as NaN ----
        aligned = pd.Series(
            resampled["_value"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(resampled["_period"]),
        )
        observed_mask = aligned.notna().to_numpy()
        missing_periods = [str(ts.date()) for ts in aligned.index[~observed_mask]]
        filled, gap_free = _fill_small_gaps(aligned)
        # What trend/stationarity tests see: the gap-filled calendar series
        # when gaps are small, otherwise the observed periods only.
        test_values = (filled if gap_free else aligned.dropna()).to_numpy(dtype=float)
        values = aligned.dropna().to_numpy(dtype=float)
        period_dates = pd.Series(aligned.index[observed_mask])
        t = np.arange(len(aligned), dtype=float)[observed_mask]

        # ---- Trend: linear fit (reference) + Mann-Kendall significance ----
        slope, intercept = np.polyfit(t, values, 1)
        fitted = slope * t + intercept
        ss_res = float(np.sum((values - fitted) ** 2))
        ss_tot = float(np.sum((values - values.mean()) ** 2))
        r_squared = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0
        # The sign of a fitted slope is never zero on real data, and R² is
        # not a significance test, so the direction is only named when the
        # Mann-Kendall test rejects "no monotonic trend".
        mk = mann_kendall(test_values)
        mk_p_value = mk["p_value"]
        if mk_p_value is not None and mk_p_value < _TREND_ALPHA and mk["trend"] != "no trend":
            direction = str(mk["trend"])
        else:
            direction = "no clear trend"

        # ---- Stationarity (Augmented Dickey-Fuller) ----
        try:
            from statsmodels.tsa.stattools import adfuller

            _adf_stat, adf_p, *_rest = adfuller(test_values, autolag="AIC")
            is_stationary = bool(adf_p <= _ADF_ALPHA)
            adf_p_value: float | None = round(float(adf_p), 4)
        except Exception:
            # statsmodels unavailable or the series is degenerate for ADF
            # (e.g. constant) — fall back rather than failing the whole tool.
            adf_p_value = None
            is_stationary = bool(abs(slope) < 1e-9)

        # ---- Autocorrelation (calendar-aligned, so lag N = N periods) ----
        seasonality: dict[str, float] = {}
        lag1_autocorr: float | None = None
        acf_skipped_reason: str | None = None
        seasonal_decomposition: dict[str, Any] | None = None
        if gap_free:
            series = filled.reset_index(drop=True)
            lag1 = series.autocorr(lag=1) if len(series) > 1 else np.nan
            lag1_autocorr = 0.0 if np.isnan(lag1) else round(float(lag1), 4)
            for lag in _SEASONAL_LAGS_BY_GRAIN[grain]:
                if len(series) > lag * 2:
                    corr = series.autocorr(lag=lag)
                    if corr is not None and not np.isnan(corr):
                        seasonality[str(lag)] = round(float(corr), 4)
            seasonal_decomposition = _seasonal_decomposition(filled, grain)
        else:
            acf_skipped_reason = (
                f"{len(missing_periods)} of {len(aligned)} {grain} periods have no data "
                f"(over {_MAX_MISSING_SHARE:.0%}), too many to interpolate for lag analysis."
            )
        seasonal_lags_detected = [
            lag for lag, corr in seasonality.items() if abs(corr) >= _SEASONALITY_THRESHOLD
        ]

        # ---- Calendar-aware seasonality (7.7), BH-corrected across months ----
        month_factors, month_counts, month_years, month_p_values = _month_of_year_factors(
            period_dates, values
        )
        month_p_adjusted: dict[str, float] = {
            r["month"]: r["p_adjusted"]
            for r in apply_benjamini_hochberg(
                [{"month": m, "p_value": p} for m, p in month_p_values.items() if p is not None],
                alpha=_MONTH_ALPHA,
            )
        }

        # ---- Entity x time panel: per-entity trends alongside the aggregate ----
        panel: dict[str, Any] | None = None
        entity = repeated_entity(profile, df)
        if entity and profile is not None and entity in profile.panel_group_cols:
            panel = _panel_trends(
                df, date_column, value_column, entity, grain, aggregation,
                pd.DatetimeIndex(aligned.index),
            )
        day_of_week_factors: dict[str, float] = {}
        if grain in ("daily", "weekly"):
            day_of_week_factors = _day_of_week_factors(working, aggregation)
        notable_months = {
            name: lift for name, lift in month_factors.items() if abs(lift) >= _MONTH_LIFT_THRESHOLD
        }
        tod: tuple[dict[str, Any], dict[str, Any]] | None
        try:
            tod = _time_of_day_profile(df, dates, date_column, value_column, profile)
        except Exception:
            tod = None

        agg_word = "total" if aggregation == "sum" else "average"
        series_label = f"{_GRAIN_LABEL[grain]} {agg_word} {value_column}"

        # A month is "replicated" once the series has seen it in >= 2
        # distinct calendar years — only those are safe to call a repeating
        # season. The rest are still surfaced here (so the number isn't
        # hidden), just clearly labelled as unreplicated single-year reads
        # rather than left indistinguishable from a genuine seasonal effect.
        seasonal_note = ""
        if notable_months:
            parts = []
            for name, lift in sorted(notable_months.items(), key=lambda kv: -abs(kv[1])):
                replicated = month_years.get(name, 0) >= _MIN_YEARS_FOR_MONTH_FINDING
                tag = "" if replicated else " [single-year, unreplicated]"
                parts.append(f"{name} {lift * 100:+.0f}%{tag}")
            seasonal_note = f" Calendar seasonality: {', '.join(parts)} vs. the yearly average."
        if seasonal_decomposition is not None:
            seasonal_note += (
                f" {seasonal_decomposition['method']} seasonal strength "
                f"{seasonal_decomposition['strength']:.2f} (period {seasonal_decomposition['period']}; "
                f"peak {seasonal_decomposition['peak_season']}, "
                f"trough {seasonal_decomposition['trough_season']})."
            )

        sen_slope = mk["sen_slope"]
        slope_unit = _PERIOD_UNIT[grain] if gap_free else "observed period"
        gap_note = ""
        if missing_periods:
            gap_note = (
                f" {len(missing_periods)} of {len(aligned)} {grain} periods have no data and "
                "are treated as missing, not zero"
                + (
                    " (interpolated for lag and trend tests)."
                    if gap_free
                    else "; lag analysis skipped and trend tested on observed periods only."
                )
            )

        tod_note = ""
        extra_out: dict[str, Any] = {"diurnal": None, "weekly_profile": None}
        if tod is not None:
            first_name, first = next(iter(tod[0].items()))
            tod_note = (
                f" Time of day: {first_name} peaks at {first['peak_hour']:02d}:00 "
                f"({_num(first['peak_value'])}) and is lowest at {first['trough_hour']:02d}:00 "
                f"({_num(first['trough_value'])}); hour of day explains {first['eta_squared']:.0%} of its variation."
            )
            charts = [d["chart"] for d in tod[0].values() if d.get("chart")]
            charts += [w["chart"] for w in tod[1].values() if w.get("chart")][:1]
            extra_out = {"diurnal": tod[0], "weekly_profile": tod[1] or None}
            if charts:
                extra_out["charts"] = charts

        return {
            **extra_out,
            "summary": (
                f"{series_label} ({grain}, {len(values)} periods): "
                + (
                    "no significant monotonic trend "
                    + (f"(Mann-Kendall p={mk_p_value:.3g}). " if mk_p_value is not None else "(too few periods to test). ")
                    if direction == "no clear trend"
                    else f"trend is {direction} (Sen's slope {sen_slope:+.4g} per {slope_unit}, "
                    f"Mann-Kendall p={mk_p_value:.3g}; linear R²={r_squared}). "
                )
                + (
                    f"Series is {'stationary' if is_stationary else 'non-stationary'} "
                    f"(ADF p={adf_p_value})."
                    if adf_p_value is not None
                    else f"Series is {'likely stationary' if is_stationary else 'likely non-stationary'} (ADF unavailable)."
                )
                + (
                    f" Seasonal autocorrelation at lag(s) {', '.join(seasonal_lags_detected)} "
                    f"({grain} periods)."
                    if seasonal_lags_detected
                    else ""
                )
                + seasonal_note
                + tod_note
                + gap_note
                + (
                    f" Final period starting {dropped_partial_period} excluded as incomplete."
                    if dropped_partial_period
                    else ""
                )
                + (
                    f" Per-{panel['entity_column']} trends ({panel['entities_analysed']} entities): "
                    f"{panel['share_increasing']:.0%} increasing, "
                    f"{panel['share_decreasing']:.0%} decreasing."
                    if panel
                    else ""
                )
            ),
            "date_column": date_column,
            "value_column": value_column,
            "aggregation": aggregation,
            "dropped_partial_period": dropped_partial_period,
            "grain": grain,
            "series_label": series_label,
            "chart_title": series_label,
            "rows_used": len(working),
            "periods_used": len(values),
            "calendar_periods": len(aligned),
            "missing_periods": missing_periods,
            "missing_period_count": len(missing_periods),
            "missing_period_handling": (
                "Periods with no rows are treated as missing (NaN), not zero"
                + (", interpolated for lag/trend tests." if gap_free else "; lag analysis skipped.")
            ),
            "trend_direction": direction,
            "trend_slope": round(float(slope), 6),
            "trend_r_squared": r_squared,
            "mk_tau": mk["tau"],
            "mk_p_value": round(mk_p_value, 6) if mk_p_value is not None else None,
            "sen_slope": round(sen_slope, 6) if sen_slope is not None else None,
            "sen_slope_unit": slope_unit,
            "mk_trend": mk["trend"],
            "is_stationary": is_stationary,
            "adf_p_value": adf_p_value,
            "autocorrelation_lag1": lag1_autocorr,
            "seasonality_by_lag": seasonality,
            "seasonal_lags_detected": seasonal_lags_detected,
            "acf_skipped_reason": acf_skipped_reason,
            "seasonal_decomposition": seasonal_decomposition,
            "month_of_year_factors": month_factors,
            "month_of_year_counts": month_counts,
            "month_of_year_years": month_years,
            "month_of_year_p_values": month_p_values,
            "month_of_year_p_adjusted": month_p_adjusted,
            "notable_months": notable_months,
            "day_of_week_factors": day_of_week_factors,
            "panel_trends": panel,
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        results: list[Finding] = []
        value_column = output.get("value_column")
        grain = output.get("grain")
        r_squared = output.get("trend_r_squared")
        direction = output.get("trend_direction")
        slope = output.get("trend_slope")
        mk_p_value = output.get("mk_p_value")
        mk_tau = output.get("mk_tau")
        sen_slope = output.get("sen_slope")
        agg_word = "Total" if output.get("aggregation") == "sum" else "Average"
        measure_label = f"{agg_word} {value_column}"

        # A trend is a finding only when Mann-Kendall rejects "no monotonic
        # trend" — R² alone is not a significance test.
        # Under two full seasonal cycles a visibly seasonal series cannot separate a
        # rise or fall over the window from the season itself: keep the finding but
        # say so, and demote it (a year of humidity 'increasing' is just winter->summer).
        calendar_periods = output.get("calendar_periods") or output.get("periods_used") or 0
        two_year_floor = _PERIODS_FOR_TWO_YEARS.get(str(grain), 24)
        seasonal_series = bool(output.get("notable_months") or output.get("seasonal_lags_detected"))
        short_seasonal = seasonal_series and calendar_periods < two_year_floor
        has_real_trend = False
        if (
            mk_p_value is not None
            and mk_p_value < _TREND_ALPHA
            and mk_tau is not None
            and sen_slope is not None
            and direction in ("increasing", "decreasing")
        ):
            has_real_trend = True
            slope_unit = output.get("sen_slope_unit") or "period"
            results.append(Finding(
                finding_id="",
                kind="trend",
                headline=(
                    f"{measure_label} moved {'up' if direction == 'increasing' else 'down'} across "
                    f"{output.get('periods_used')} {grain} periods, but with under two years of history "
                    f"this may be the season, not a trend (Sen's slope {sen_slope:+.4g} per {slope_unit})."
                    if short_seasonal else
                    f"{measure_label} is {direction} across {output.get('periods_used')} "
                    f"{grain} periods (Sen's slope {sen_slope:+.4g} per {slope_unit}, "
                    f"Mann-Kendall p={mk_p_value:.3g})."
                ),
                detail=(
                    f"Mann-Kendall test on the {grain}-resampled series: tau={mk_tau}, "
                    f"p={mk_p_value:.3g}; Sen's slope {sen_slope:+.4g} per {slope_unit}. "
                    f"Linear fit for reference: slope={slope} per period, R²={r_squared}. "
                    + (
                        f"ADF p-value={output.get('adf_p_value')} "
                        f"({'stationary' if output.get('is_stationary') else 'non-stationary'})."
                        if output.get("adf_p_value") is not None else ""
                    )
                ),
                evidence={
                    "value_column": value_column,
                    "grain": grain,
                    "aggregation": output.get("aggregation"),
                    "trend_slope": slope,
                    "trend_r_squared": r_squared,
                    "periods_used": output.get("periods_used"),
                    "mk_tau": mk_tau,
                    "mk_p_value": mk_p_value,
                    "sen_slope": sen_slope,
                    "sen_slope_unit": slope_unit,
                    "missing_period_count": output.get("missing_period_count"),
                },
                source_tool=self.name,
                measure=value_column,
                effect=mk_tau,
                effect_kind="r",
                p_value=mk_p_value,
                confidence=round(min(0.95, 0.5 + abs(mk_tau) / 2), 3) * (0.5 if short_seasonal else 1),
                chart_hint={"kind": "line", "data": {"series_label": output.get("series_label")}},
                caveats=(
                    ["Less than two full seasonal cycles: a rise or fall over this window cannot be "
                     "separated from the season."] if short_seasonal else []
                ),
                layer="appendix" if short_seasonal else "analyst",
            ))

        panel = output.get("panel_trends")
        if panel:
            results.extend(self._panel_findings(panel, output, measure_label))

        # A real trend confounds a month-of-year read when there isn't
        # enough history to separate "later in the calendar" from "later in
        # time" — flag it on every month finding rather than silently
        # reporting a trend artifact as a repeating season.
        trend_confounded = has_real_trend and calendar_periods < two_year_floor

        notable_months = output.get("notable_months") or {}
        month_counts = output.get("month_of_year_counts") or {}
        month_years = output.get("month_of_year_years") or {}
        month_p_values = output.get("month_of_year_p_values") or {}
        month_p_adjusted = output.get("month_of_year_p_adjusted") or {}

        # Only publish a month as a Finding once it's replicated across >= 2
        # distinct calendar years — a single-year series produces a "month
        # factor" for every month from one observation each, which the
        # summary text already flags but which should never have competed
        # for a top-12 findings slot as if it were a proven repeating season.
        # Twelve month tests are also twelve chances of a false positive, so
        # a month must stay significant after Benjamini-Hochberg correction.
        replicated_months = {
            name: lift for name, lift in notable_months.items()
            if month_years.get(name, 0) >= _MIN_YEARS_FOR_MONTH_FINDING
            and month_p_adjusted.get(name) is not None
            and month_p_adjusted[name] < _MONTH_ALPHA
        }
        # Cap to the strongest few by |lift| — publishing all 12 months
        # (even replicated ones) crowds out every other tool's findings.
        top_months = sorted(
            replicated_months.items(), key=lambda kv: -abs(kv[1])
        )[:_MAX_MONTH_FINDINGS]

        for month_name, lift in top_months:
            direction_word = "above" if lift > 0 else "below"
            p_value = month_p_values.get(month_name)
            # Confidence now tracks the actual significance test instead of
            # a constant (0.6/0.75 regardless of evidence strength): a small
            # p-value earns high confidence, a large one is capped low. When
            # no test could be run (degenerate variance), fall back to a
            # modest default rather than a fixed high constant.
            confidence = round(max(0.3, min(0.9, 1.0 - p_value)), 3) if p_value is not None else 0.4
            caveats: list[str] = []
            if trend_confounded:
                caveats.append(
                    "A statistically real trend is present and less than two years of "
                    "history are available, so this month's lift may partly reflect the "
                    "trend rather than a repeating seasonal effect."
                )
            results.append(Finding(
                finding_id="",
                kind="change",
                headline=(
                    f"{month_name} {measure_label.lower()} runs {abs(lift) * 100:.0f}% "
                    f"{direction_word} the yearly average"
                ),
                detail=(
                    f"Calendar-month factor computed on the {grain}-resampled series "
                    f"({month_counts.get(month_name, '?')} period(s) across "
                    f"{month_years.get(month_name, '?')} years in {month_name}): "
                    f"{lift * 100:+.1f}% vs. the overall mean."
                    + (
                        f" Welch t-test vs. the rest of the year: p={p_value:.4g} "
                        f"(BH-adjusted across months: {month_p_adjusted[month_name]:.4g})."
                        if p_value is not None else ""
                    )
                ),
                evidence={
                    "value_column": value_column,
                    "grain": grain,
                    "month": month_name,
                    "lift": lift,
                    "periods_in_month": month_counts.get(month_name),
                    "years_observed": month_years.get(month_name),
                    "p_adjusted_across_months": month_p_adjusted[month_name],
                },
                source_tool=self.name,
                measure=value_column,
                dimension="month",
                level=month_name,
                effect=lift,
                effect_kind="lift",
                p_value=p_value,
                confidence=confidence,
                caveats=caveats,
                chart_hint={"kind": "bar", "data": {"category": "month_of_year"}},
                layer="analyst",
            ))
        results.extend(self._diurnal_findings(output, profile))
        return results

    def _diurnal_findings(self, output: dict[str, Any], profile: DatasetProfile | None) -> list[Finding]:
        """One finding per measure with a real hour-of-day cycle; the one
        with the highest eta^2 goes on the executive tier."""
        diurnal = output.get("diurnal") or {}
        weekly = output.get("weekly_profile") or {}
        if not diurnal:
            return []
        # The tool's primary measure (objective-aware, completeness-first) comes first
        # in `diurnal`; it carries the executive headline. When it has no cycle the
        # first entry is the strongest of the complete extras.
        strongest = next(iter(diurnal))
        units = {c.name: c.unit_hint for c in (profile.columns if profile else [])}
        out: list[Finding] = []
        for name, d in diurnal.items():
            suffix = "%" if units.get(name) == "percent" else ""
            ph, th = d["peak_hour"], d["trough_hour"]
            ratio = d["peak_trough_ratio"]
            headline = (
                f"{name} peaks at {ph:02d}:00 ({_num(d['peak_value'])}{suffix}) and is lowest at "
                f"{th:02d}:00 ({_num(d['trough_value'])}{suffix})"
            )
            if ratio is not None and ratio >= 1.25:
                headline += f", about {ratio:.1f}x higher in the {_day_part(ph)}".replace(".0x", "x")
            else:
                headline += f", a swing of {d['effect']:.0%} of the average"
            w = weekly.get(name)
            gap = w.get("weekend_gap") if w else None
            if gap is not None and abs(gap) >= _TOD_WEEKEND_GAP_MIN:
                headline += f"; weekends run {abs(gap):.0%} {'lower' if gap < 0 else 'higher'}"
            evidence: dict[str, Any] = {
                "value_column": name, "date_column": output.get("date_column"),
                "peak_hour": ph, "peak_value": d["peak_value"],
                "trough_hour": th, "trough_value": d["trough_value"],
                "peak_trough_ratio": ratio, "overall_mean": d["overall_mean"],
                "eta_squared": d["eta_squared"], "p_value": d["p_value"], "n": d["n"],
                "hourly_means": d["hourly_means"],
            }
            if w:
                evidence.update({
                    "weekday_mean": w["weekday_mean"], "weekend_mean": w["weekend_mean"],
                    "weekend_gap": gap, "weekday_means": w["means"],
                    "weekday_eta_squared": w["eta_squared"],
                })
            out.append(Finding(
                finding_id="",
                kind="trend",
                headline=headline,
                detail=(
                    f"Average {name} by hour of day over {d['n']:,} readings: highest at {ph:02d}:00 "
                    f"({d['peak_value']}), lowest at {th:02d}:00 ({d['trough_value']}), overall average "
                    f"{d['overall_mean']}. Hour of day explains {d['eta_squared']:.1%} of the variation "
                    f"(eta squared {d['eta_squared']}, ANOVA p={d['p_value']:.3g})."
                    + (
                        f" Weekday average {w['weekday_mean']} vs weekend {w['weekend_mean']}"
                        + (f" ({gap:+.0%})." if gap is not None else ".")
                        if w else ""
                    )
                ),
                evidence=evidence,
                source_tool=self.name,
                measure=name,
                dimension="hour_of_day",
                level=f"{ph:02d}:00",
                effect=d["effect"],
                effect_kind="pct",
                p_value=d["p_value"],
                confidence=round(min(0.95, 0.4 + 0.6 * d["eta_squared"] + 0.15 * min(d["n"] / 2000, 1.0)), 3),
                chart_hint=d.get("chart"),
                layer="exec" if name == strongest else "analyst",
            ))
        return out

    def _panel_findings(
        self, panel: dict[str, Any], output: dict[str, Any], measure_label: str
    ) -> list[Finding]:
        """One finding for an entity x time panel: a direction shared by
        most entities, or entities moving materially in both directions."""
        n = panel.get("entities_analysed") or 0
        share_up = panel.get("share_increasing") or 0.0
        share_down = panel.get("share_decreasing") or 0.0
        entity = panel.get("entity_column")
        grain = output.get("grain")
        if n < 4:
            return []
        risers = ", ".join(panel.get("strongest_risers") or []) or "none"
        fallers = ", ".join(panel.get("strongest_fallers") or []) or "none"
        if max(share_up, share_down) > 0.5:
            rising = share_up > share_down
            share = share_up if rising else share_down
            headline = (
                f"{share:.0%} of {entity} values show a significant "
                f"{'increasing' if rising else 'decreasing'} {grain} trend in "
                f"{measure_label.lower()} ({n} {entity} values tested)"
            )
            effect = share if rising else -share
        elif min(share_up, share_down) >= _PANEL_DIVERGENCE_SHARE:
            headline = (
                f"{entity} values diverge on {measure_label.lower()}: {share_up:.0%} trend up "
                f"and {share_down:.0%} trend down ({n} tested)"
            )
            effect = share_up - share_down
        else:
            return []
        return [Finding(
            finding_id="",
            kind="trend",
            headline=headline,
            detail=(
                f"Mann-Kendall test per {entity} on the {grain} series (top {n} by rows, "
                f"same calendar range as the aggregate). Strongest risers: {risers}; "
                f"strongest fallers: {fallers}. Aggregate trend: {output.get('trend_direction')}."
            ),
            evidence={
                "value_column": output.get("value_column"),
                "grain": grain,
                "entity_column": entity,
                "entities_analysed": n,
                "entities_increasing": panel.get("entities_increasing"),
                "entities_decreasing": panel.get("entities_decreasing"),
                "share_increasing": share_up,
                "share_decreasing": share_down,
                "strongest_risers": panel.get("strongest_risers"),
                "strongest_fallers": panel.get("strongest_fallers"),
            },
            source_tool=self.name,
            measure=output.get("value_column"),
            dimension=entity,
            effect=round(effect, 4),
            effect_kind="pct",
            confidence=round(min(0.9, 0.4 + n / 50), 3),
            chart_hint={"kind": "line", "data": {"series_label": output.get("series_label")}},
            layer="analyst",
        )]

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to the (cleaned) dataset.", "required": True},
            "date_column": {
                "type": "string",
                "description": "Datetime column to resample by. Auto-detected if omitted.",
                "required": False,
            },
            "value_column": {
                "type": "string",
                "description": (
                    "Numeric column to analyse over time. Auto-detected if omitted, "
                    "preferring a currency/revenue measure over an arbitrary numeric column."
                ),
                "required": False,
            },
            "target_column": {
                "type": "string",
                "description": "Alias for value_column, accepted for compatibility with generic callers.",
                "required": False,
            },
            "aggregation": {
                "type": "string",
                "description": (
                    "How to aggregate the value per period: 'sum' (additive money/count "
                    "measures) or 'mean' (rates, levels, readings). Auto-chosen from the "
                    "column's profiled additivity if omitted."
                ),
                "required": False,
            },
        }
