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

Kept to statistics that generalise across irregular/coarse-grained data
rather than a full seasonal decomposition, which needs a reliable
inferred frequency that real-world timestamps rarely provide cleanly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from src.core.findings import Finding
from src.core.profiler import profile_dataframe
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import ColumnProfile, DatasetProfile

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

#: R² a fitted trend line must reach before its direction is worth naming.
#: Below this the line explains almost none of the variation, so calling the
#: series "increasing" reports the sign of noise as a finding.
_TREND_MIN_R_SQUARED = 0.05

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


#: A trailing period is "partial" when the data covers less than this share
#: of its calendar span AND it holds fewer rows than this share of a typical
#: period — the row-count half keeps one-row-per-month data (dated on the 1st)
#: from being mistaken for an incomplete month.
_PARTIAL_PERIOD_COVERAGE = 0.9


def measure_aggregation(col: ColumnProfile | None) -> str:
    """How a measure combines across rows within a period: "sum" when it is
    additive (revenue, counts), "mean" when it is not (temperature, a rate,
    a score). Reads the profile's `aggregation` when present and falls back
    to the unit hint (percent -> mean) otherwise."""
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
        coverage < _PARTIAL_PERIOD_COVERAGE
        and counts.iloc[-1] < _PARTIAL_PERIOD_COVERAGE * typical_rows
    )


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
            best = max(pool, key=lambda c: (c.stats.get("std") or 0.0) ** 2)
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
    """One row per observed calendar period: sum (additive measures) or mean
    (rates). Periods with no underlying rows are dropped rather than
    fabricated as zero — the tool aggregates observed data, it doesn't
    assume unobserved periods were genuinely zero."""
    indexed = working.set_index("_date")["_value"]
    # Left-labelled so every period is named by its start (weeks included),
    # which is what is_partial_final_period measures coverage from.
    resampled = indexed.resample(_GRAIN_FREQ[grain], label="left", closed="left")
    agg = resampled.sum(min_count=1) if aggregation == "sum" else resampled.mean()
    agg = agg.dropna()
    return pd.DataFrame({"_period": agg.index, "_value": agg.to_numpy(dtype=float)})


def _resample_series(
    working: pd.DataFrame, aggregation: str, candidates: list[str]
) -> tuple[pd.DataFrame | None, str]:
    """Try grains coarsest-first; step down to the next finer grain if the
    resampled series is too thin to diagnose. Always falls back to the
    finest candidate's result if none clear the minimum-periods bar."""
    for candidate in candidates:
        resampled = _resample(working, candidate, aggregation)
        if len(resampled) >= _MIN_PERIODS_FOR_GRAIN or candidate == candidates[-1]:
            return resampled, candidate
    return None, candidates[-1]


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
    daily = _resample(working, "daily", aggregation)
    if len(daily) < 7:
        return {}
    s = pd.Series(daily["_value"].to_numpy(dtype=float), index=pd.DatetimeIndex(daily["_period"]))
    overall_mean = float(s.mean())
    if overall_mean == 0:
        return {}
    by_dow = s.groupby(s.index.day_name()).mean()
    return {str(k): round((float(v) - overall_mean) / abs(overall_mean), 4) for k, v in by_dow.items()}


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
        value_column, inferred_aggregation, _profile = _choose_value_column_and_aggregation(
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
        if resampled is None or len(resampled) < 3:
            n = 0 if resampled is None else len(resampled)
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

        values = resampled["_value"].to_numpy(dtype=float)
        period_dates = pd.to_datetime(resampled["_period"])
        t = np.arange(len(values), dtype=float)

        # ---- Trend: linear fit over the resampled series (one point per period) ----
        slope, intercept = np.polyfit(t, values, 1)
        fitted = slope * t + intercept
        ss_res = float(np.sum((values - fitted) ** 2))
        ss_tot = float(np.sum((values - values.mean()) ** 2))
        r_squared = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0
        # The sign of a fitted slope is never zero on real data, so reporting
        # "increasing" off the sign alone calls pure noise a trend. R² is what
        # says whether the line describes the series at all: below the
        # threshold the honest answer is that there is no trend, and the
        # direction is not worth naming.
        if r_squared < _TREND_MIN_R_SQUARED:
            direction = "no clear trend"
        else:
            direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"

        # ---- Stationarity (Augmented Dickey-Fuller) ----
        try:
            from statsmodels.tsa.stattools import adfuller

            _adf_stat, adf_p, *_rest = adfuller(values, autolag="AIC")
            is_stationary = bool(adf_p <= _ADF_ALPHA)
            adf_p_value: float | None = round(float(adf_p), 4)
        except Exception:
            # statsmodels unavailable or the series is degenerate for ADF
            # (e.g. constant) — fall back rather than failing the whole tool.
            adf_p_value = None
            is_stationary = bool(abs(slope) < 1e-9)

        # ---- Autocorrelation (over the resampled series, not raw rows) ----
        series = pd.Series(values)
        lag1_autocorr = round(float(series.autocorr(lag=1)), 4) if len(series) > 1 else 0.0

        seasonality: dict[str, float] = {}
        for lag in _SEASONAL_LAGS_BY_GRAIN[grain]:
            if len(series) > lag * 2:
                corr = series.autocorr(lag=lag)
                if corr is not None and not np.isnan(corr):
                    seasonality[str(lag)] = round(float(corr), 4)
        seasonal_lags_detected = [
            lag for lag, corr in seasonality.items() if abs(corr) >= _SEASONALITY_THRESHOLD
        ]

        # ---- Calendar-aware seasonality (7.7) ----
        month_factors, month_counts, month_years, month_p_values = _month_of_year_factors(
            period_dates, values
        )
        day_of_week_factors: dict[str, float] = {}
        if grain in ("daily", "weekly"):
            day_of_week_factors = _day_of_week_factors(working, aggregation)
        notable_months = {
            name: lift for name, lift in month_factors.items() if abs(lift) >= _MONTH_LIFT_THRESHOLD
        }

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

        return {
            "summary": (
                f"{series_label} ({grain}, {len(resampled)} periods): "
                + (
                    f"no clear trend (R²={r_squared} — a fitted line explains "
                    f"almost none of the variation). "
                    if direction == "no clear trend"
                    else f"trend is {direction} (slope={slope:.4g}, R²={r_squared}). "
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
                + (
                    f" Final period starting {dropped_partial_period} excluded as incomplete."
                    if dropped_partial_period
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
            "periods_used": len(resampled),
            "trend_direction": direction,
            "trend_slope": round(float(slope), 6),
            "trend_r_squared": r_squared,
            "is_stationary": is_stationary,
            "adf_p_value": adf_p_value,
            "autocorrelation_lag1": lag1_autocorr,
            "seasonality_by_lag": seasonality,
            "seasonal_lags_detected": seasonal_lags_detected,
            "month_of_year_factors": month_factors,
            "month_of_year_counts": month_counts,
            "month_of_year_years": month_years,
            "month_of_year_p_values": month_p_values,
            "notable_months": notable_months,
            "day_of_week_factors": day_of_week_factors,
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
        agg_word = "Total" if output.get("aggregation") == "sum" else "Average"
        measure_label = f"{agg_word} {value_column}"

        if (
            r_squared is not None
            and r_squared >= _TREND_MIN_R_SQUARED
            and direction not in (None, "no clear trend", "flat")
        ):
            signed_effect = round(min(1.0, r_squared), 4) * (1 if (slope or 0) >= 0 else -1)
            results.append(Finding(
                finding_id="",
                kind="trend",
                headline=(
                    f"{measure_label} is {direction} across {output.get('periods_used')} "
                    f"{grain} periods (slope={slope}, R²={r_squared})."
                ),
                detail=(
                    f"Linear trend fit on the {grain}-resampled series: slope={slope} "
                    f"per period, R²={r_squared}. "
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
                },
                source_tool=self.name,
                measure=value_column,
                effect=signed_effect,
                effect_kind="eta_sq",
                confidence=round(min(1.0, 0.4 + r_squared), 3),
                chart_hint={"kind": "line", "data": {"series_label": output.get("series_label")}},
                layer="analyst",
            ))

        # A real linear trend confounds a month-of-year read when there isn't
        # enough history to separate "later in the calendar" from "later in
        # time" — flag it on every month finding rather than silently
        # reporting a trend artifact as a repeating season.
        has_real_trend = (
            r_squared is not None
            and r_squared >= _TREND_MIN_R_SQUARED
            and direction not in (None, "no clear trend", "flat")
        )
        periods_used = output.get("periods_used") or 0
        two_year_floor = _PERIODS_FOR_TWO_YEARS.get(str(grain), 24)
        trend_confounded = has_real_trend and periods_used < two_year_floor

        notable_months = output.get("notable_months") or {}
        month_counts = output.get("month_of_year_counts") or {}
        month_years = output.get("month_of_year_years") or {}
        month_p_values = output.get("month_of_year_p_values") or {}

        # Only publish a month as a Finding once it's replicated across >= 2
        # distinct calendar years — a single-year series produces a "month
        # factor" for every month from one observation each, which the
        # summary text already flags but which should never have competed
        # for a top-12 findings slot as if it were a proven repeating season.
        replicated_months = {
            name: lift for name, lift in notable_months.items()
            if month_years.get(name, 0) >= _MIN_YEARS_FOR_MONTH_FINDING
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
                        f" Welch t-test vs. the rest of the year: p={p_value:.4g}."
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
        return results

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
