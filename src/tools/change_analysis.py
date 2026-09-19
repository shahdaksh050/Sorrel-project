"""
Change Analysis Tool — Execution Layer (IMPROVEMENTS.md item 7.2 #3).

Period-over-period movement on the correct grain: which period changed, by
how much, and (optionally) which segment drove it. This is the "what
happened" question IMPROVEMENTS.md flags as one no current tool answers
directly — `time_series_analysis` diagnoses trend/seasonality, not which
period moved (Run B: a planted Q4 lift was missed because 4,000 raw rows
were fed straight into a trend model instead of being resampled to a period
sum first).

Grain selection is a simple local heuristic (monthly once the date range
spans a year or more, weekly for a multi-week range, daily otherwise) —
deliberately separate from `time_series_analysis`'s grain choice, which
solves a different problem (trend/seasonality diagnostics, not
period-over-period movement).

Periods are calendar-aligned: a period with no rows is missing, not zero,
so "latest vs prior" always compares adjacent calendar periods and a gap
is reported rather than read as a 100% drop or an infinite rise.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.findings import Finding
from src.core.profiler import profile_dataframe
from src.core.stats_utils import is_partial_final_period, measure_aggregation
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import ColumnProfile, DatasetProfile

#: Date-range width (days) above which the natural reporting grain is
#: monthly; below that but at/above the weekly floor, weekly; otherwise daily.
_MONTHLY_SPAN_DAYS = 365
_WEEKLY_SPAN_DAYS = 60

#: A period-over-period move smaller than this is noise, not a story (T5).
_MIN_CHANGE_FOR_FINDING = 0.10
#: ...and neither is one within this multiple of the series' own typical
#: period-over-period movement — a 15% swing in a series that routinely
#: swings 20% is an ordinary period.
_VOLATILITY_MULTIPLE = 2.0

#: Segment-breakdown dimension cardinality window, mirroring segment_comparison.
_MIN_DIM_CARD = 2
_MAX_DIM_CARD = 20

#: Rows shown in the per-segment contribution breakdown.
_TOP_N_SEGMENTS = 10


def _pick_measure_columns(profile: DatasetProfile) -> list[ColumnProfile]:
    measures = list(profile.measures())
    return sorted(measures, key=lambda c: (c.unit_hint != "currency", c.name))


def _pick_dimension_column(profile: DatasetProfile) -> str | None:
    dims = [
        c for c in profile.columns_of_role("dimension", "flag")
        if _MIN_DIM_CARD <= c.nunique <= _MAX_DIM_CARD
    ]
    if not dims:
        return None
    return sorted(dims, key=lambda c: c.nunique)[0].name


def _grain_for_span(span_days: float) -> tuple[str, str]:
    """(pandas resample rule, human label) for a date range this wide."""
    if span_days >= _MONTHLY_SPAN_DAYS:
        return "MS", "monthly"
    if span_days >= _WEEKLY_SPAN_DAYS:
        return "W", "weekly"
    return "D", "daily"


def _pick_mover(
    breakdown: list[dict[str, Any]] | None, total_change: float
) -> dict[str, Any] | None:
    """The segment to name as "biggest contributor" to the headline change:
    the largest mover in the SAME direction as the total change — a segment
    that moved opposite the headline didn't drive it, even if it happens to
    have the largest absolute delta. Falls back to the largest absolute
    mover (breakdown is already sorted that way) if none moved that way."""
    if not breakdown:
        return None
    if total_change > 0:
        same_dir = [r for r in breakdown if r["delta"] > 0]
    elif total_change < 0:
        same_dir = [r for r in breakdown if r["delta"] < 0]
    else:
        same_dir = []
    if same_dir:
        return max(same_dir, key=lambda r: abs(r["delta"]))
    return breakdown[0]


def _period_label(ts: pd.Timestamp, grain_label: str) -> str:
    if grain_label == "monthly":
        return str(ts.strftime("%Y-%m"))
    if grain_label == "weekly":
        return f"week of {ts.date()}"
    return str(ts.date())


class ChangeAnalysisTool(BaseTool):
    """Period-over-period movement of a measure, on an auto-selected grain,
    optionally broken down by which segment drove the change."""

    name = "change_analysis"
    description = (
        "Resample a measure to a natural period grain (monthly/weekly/daily) "
        "and report the most recent period's change vs the prior period and "
        "vs the trailing average. Optionally breaks the change down by a "
        "segment dimension to say which segment drove it. Auto-selects the "
        "date column and measure when omitted."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None:
            return 0.0
        if not profile.datetime_cols:
            return 0.0
        return 0.85 if _pick_measure_columns(profile) else 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None or not profile.datetime_cols:
            return {}
        measures = _pick_measure_columns(profile)
        if not measures:
            return {}
        params: dict[str, Any] = {
            "date_column": profile.datetime_cols[0],
            "measure_column": measures[0].name,
        }
        dim = _pick_dimension_column(profile)
        if dim:
            params["dimension_column"] = dim
        return params

    def _segment_breakdown(
        self,
        df: pd.DataFrame,
        date_column: str,
        measure_column: str,
        dimension_column: str,
        agg_func: str,
        freq: str,
        latest_period: pd.Timestamp,
        prior_period: pd.Timestamp,
        total_change: float,
    ) -> list[dict[str, Any]] | None:
        """Best-effort: which segment moved the most between the two most
        recent periods. Returns None if the dimension can't be resolved
        against these rows — the core period-over-period result already
        computed does not depend on this succeeding.

        `total_change` is the TOTAL series' actual latest-vs-prior change
        (`latest_value - prior_value` from `execute()`), passed in so
        `share_of_total_change_pct` reconciles to the headline number
        instead of the sum of segment deltas, which can diverge from it.
        """
        try:
            seg = df[[date_column, measure_column, dimension_column]].copy()
            seg[date_column] = pd.to_datetime(seg[date_column], errors="coerce")
            seg[measure_column] = pd.to_numeric(seg[measure_column], errors="coerce")
            # Rows with a missing dimension value still count toward the
            # TOTAL series — fold them into an explicit "(missing)" level
            # instead of dropping them, so those rows are represented in
            # the breakdown rather than silently disappearing from it.
            # `.where(notna(), ...)` (not `.astype(object).fillna(...)`) is
            # deliberate: it's dtype-agnostic across object/category/the
            # pandas-3.0 `str` dtype, whose null sentinel isn't always
            # `np.nan` and can survive an `astype(object)` round-trip.
            dim = seg[dimension_column]
            # astype(str) first: a categorical dimension would reject the
            # new "(missing)" value in .where() (not a known category).
            seg[dimension_column] = dim.astype(str).where(dim.notna(), "(missing)")
            seg = seg.dropna(subset=[date_column, measure_column])
            if seg.empty:
                return None
            grouped = (
                seg.set_index(date_column)
                .groupby(dimension_column)[measure_column]
                .resample(freq, label="left", closed="left")
                .agg(agg_func)
            )
            dates = grouped.index.get_level_values(1)
            latest_by_seg = grouped[dates == latest_period].droplevel(1)
            prior_by_seg = grouped[dates == prior_period].droplevel(1)
            combined = pd.DataFrame({"latest": latest_by_seg, "prior": prior_by_seg})
            # Both periods have data overall, so a segment absent from one of
            # them contributed nothing to a sum there; a mean over no rows has
            # no value at all, so such a segment is left out of the breakdown.
            combined = combined.fillna(0.0) if agg_func == "sum" else combined.dropna()
            if combined.empty:
                return None
            combined["delta"] = combined["latest"] - combined["prior"]
            combined = combined.reindex(combined["delta"].abs().sort_values(ascending=False).index)
            rows = []
            for level, row in combined.head(_TOP_N_SEGMENTS).iterrows():
                delta = float(row["delta"])
                rows.append({
                    "level": str(level),
                    "latest_value": round(float(row["latest"]), 4),
                    "prior_value": round(float(row["prior"]), 4),
                    "delta": round(delta, 4),
                    # Signed: a segment that moved OPPOSITE the total change
                    # (offsetting it) gets a negative share here, not just a
                    # small one — this is "share of the headline move this
                    # segment is responsible for," not "share of magnitude."
                    "share_of_total_change_pct": (
                        round(delta / total_change * 100, 1) if total_change else None
                    ),
                })
            return rows
        except (ValueError, KeyError, TypeError):
            return None

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        date_column: str | None = None,
        measure_column: str | None = None,
        dimension_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows.")

        profile = profile_dataframe(df)

        if date_column and date_column not in df.columns:
            raise ToolExecutionError(f"Column '{date_column}' not found in dataset.")
        if measure_column and measure_column not in df.columns:
            raise ToolExecutionError(f"Column '{measure_column}' not found in dataset.")
        if dimension_column and dimension_column not in df.columns:
            raise ToolExecutionError(f"Column '{dimension_column}' not found in dataset.")

        if not date_column:
            date_column = profile.datetime_cols[0] if profile.datetime_cols else None
        if not date_column:
            raise ToolExecutionError("No datetime column found. Pass date_column explicitly.")

        if not measure_column:
            candidates = _pick_measure_columns(profile)
            if not candidates:
                raise ToolExecutionError(
                    "No numeric measure column found. Pass measure_column explicitly."
                )
            measure_column = candidates[0].name

        work = df[[date_column, measure_column]].copy()
        work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
        work[measure_column] = pd.to_numeric(work[measure_column], errors="coerce")
        work = work.dropna(subset=[date_column, measure_column])
        if work.empty:
            raise ToolExecutionError(
                "No rows have both a parseable date and a numeric measure value."
            )

        span_days = (work[date_column].max() - work[date_column].min()).days
        freq, grain_label = _grain_for_span(float(span_days))

        cp = next((c for c in profile.columns if c.name == measure_column), None)
        agg_func = measure_aggregation(cp)

        # Calendar-aligned: resample emits every period between the first and
        # last observed one, and a period with no rows stays NaN (min_count=1)
        # instead of becoming a zero that a later period is compared against.
        resampler = work.set_index(date_column)[measure_column].resample(
            freq, label="left", closed="left"
        )
        series = resampler.sum(min_count=1) if agg_func == "sum" else resampler.mean()
        if int(series.notna().sum()) < 2:
            raise ToolExecutionError(
                f"Only {int(series.notna().sum())} distinct {grain_label} period(s) after "
                "resampling — need at least 2 to measure change."
            )

        # A dataset extract very often stops mid-period (e.g. the file ends
        # May 15th), which makes the trailing bucket look like a real drop
        # against a full prior month/week when it is really just fewer days
        # of data. Drop it when there is another full period to fall back to.
        latest_period_partial = False
        dropped_partial_period: str | None = None
        counts = (
            work.set_index(date_column)[measure_column]
            .resample(freq, label="left", closed="left")
            .count()
        )
        counts = counts[counts > 0]
        if is_partial_final_period(counts, work[date_column].max(), freq):
            if len(series) >= 3:
                dropped_partial_period = _period_label(series.index[-1], grain_label)
                series = series.iloc[:-1]
            else:
                latest_period_partial = True
        missing_periods = [_period_label(ts, grain_label) for ts in series.index[series.isna()]]
        latest_period = series.index[-1]
        prior_period = series.index[-2]
        latest_value = float(series.iloc[-1])
        prior_value = float(series.iloc[-2])
        # Either side of the comparison having no rows makes the change
        # unknown, not a move to or from zero.
        comparison_missing = bool(np.isnan(latest_value) or np.isnan(prior_value))
        pct_change = (
            (latest_value - prior_value) / prior_value
            if not comparison_missing and prior_value != 0
            else None
        )

        trailing = series.iloc[:-1]
        # The series' own noise floor: the median absolute period-over-period
        # move before the latest period, between adjacent observed periods
        # only (a move across a gap is NaN and dropped).
        prior_moves = (
            trailing.pct_change(fill_method=None)
            .abs()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        typical_volatility = float(prior_moves.median()) if len(prior_moves) else None
        change_threshold = max(
            _MIN_CHANGE_FOR_FINDING, _VOLATILITY_MULTIPLE * (typical_volatility or 0.0)
        )
        trailing_avg = float(trailing.mean()) if trailing.notna().any() else None
        pct_change_vs_trailing = (
            (latest_value - trailing_avg) / trailing_avg
            if trailing_avg is not None and trailing_avg != 0 and not np.isnan(latest_value)
            else None
        )

        if not dimension_column:
            dimension_column = _pick_dimension_column(profile)

        breakdown = None
        if dimension_column and dimension_column in df.columns and not comparison_missing:
            breakdown = self._segment_breakdown(
                df, date_column, measure_column, dimension_column,
                agg_func, freq, latest_period, prior_period,
                latest_value - prior_value,
            )

        latest_label = _period_label(latest_period, grain_label)
        measure_label = f"{'total' if agg_func == 'sum' else 'average'} {measure_column}"
        if comparison_missing:
            empty_side = "latest" if np.isnan(latest_value) else "prior"
            summary = (
                f"{measure_label}: the {empty_side} {grain_label} period in the comparison "
                f"({latest_label} vs the one before) has no data, so the change is unknown "
                "(an empty period is treated as missing, not zero)."
            )
        elif pct_change is None:
            summary = (
                f"{measure_label} moved from 0 to {latest_value:,.2f} in {latest_label} "
                "vs the prior period (prior period was zero — percent change undefined)."
            )
        else:
            direction = "rose" if pct_change > 0 else "fell"
            summary = (
                f"{measure_label} {direction} {abs(pct_change) * 100:.1f}% in "
                f"{latest_label} vs the prior period"
                + (
                    f" (typical move {typical_volatility * 100:.1f}% per period)."
                    if typical_volatility is not None
                    else "."
                )
            )
            mover = _pick_mover(breakdown, latest_value - prior_value)
            if mover:
                contributor_word = "rise" if pct_change > 0 else "drop"
                summary += (
                    f" Biggest contributor to the {contributor_word}: "
                    f"{mover['level']} ({mover['delta']:+,.2f})."
                )
            if latest_period_partial:
                summary += (
                    f" Note: the latest {grain_label} period is not yet complete in "
                    "this data, so the comparison may understate it."
                )
        if dropped_partial_period:
            summary += f" The incomplete final period ({dropped_partial_period}) was excluded."
        if missing_periods and not comparison_missing:
            summary += (
                f" {len(missing_periods)} {grain_label} period(s) have no data and are "
                "treated as missing, not zero."
            )

        return {
            "summary": summary,
            "date_column": date_column,
            "measure_column": measure_column,
            "dimension_column": dimension_column,
            "period_grain": grain_label,
            "aggregation": agg_func,
            "latest_period": latest_label,
            "latest_value": None if np.isnan(latest_value) else round(latest_value, 4),
            "prior_period_value": None if np.isnan(prior_value) else round(prior_value, 4),
            "comparison_period_missing": comparison_missing,
            "missing_periods": missing_periods,
            "missing_period_count": len(missing_periods),
            "pct_change": round(pct_change, 6) if pct_change is not None else None,
            "trailing_avg": round(trailing_avg, 4) if trailing_avg is not None else None,
            "latest_period_partial": latest_period_partial,
            "dropped_partial_period": dropped_partial_period,
            "typical_volatility": (
                round(typical_volatility, 6) if typical_volatility is not None else None
            ),
            "change_threshold": round(change_threshold, 6),
            "pct_change_vs_trailing_avg": (
                round(pct_change_vs_trailing, 6) if pct_change_vs_trailing is not None else None
            ),
            "segment_breakdown": breakdown,
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        pct_change = output.get("pct_change")
        measure = output.get("measure_column")
        if pct_change is None or measure is None:
            return []
        threshold = output.get("change_threshold") or _MIN_CHANGE_FOR_FINDING
        if abs(pct_change) < threshold:
            return []  # T5: within the series' normal period-to-period noise
        if output.get("latest_period_partial"):
            return []  # an incomplete trailing period is a method-fit issue, not a finding

        direction = "rose" if pct_change > 0 else "fell"
        latest_label = output.get("latest_period", "the latest period")
        agg_word = "Total" if output.get("aggregation") == "sum" else "Average"
        headline = (
            f"{agg_word} {measure} {direction} {abs(pct_change) * 100:.1f}% in {latest_label} "
            "vs the prior period."
        )
        detail = (
            f"{output.get('period_grain', 'period')} grain; "
            f"{output.get('prior_period_value')} -> {output.get('latest_value')}; "
            f"flag threshold {threshold * 100:.1f}% (max of 10% and 2x the typical "
            f"period-over-period move)."
        )
        breakdown = output.get("segment_breakdown")
        if breakdown:
            total_change = (output.get("latest_value") or 0.0) - (
                output.get("prior_period_value") or 0.0
            )
            mover = _pick_mover(breakdown, total_change)
            if mover:
                contributor_word = "rise" if pct_change > 0 else "drop"
                detail += (
                    f" Biggest contributor to the {contributor_word}: "
                    f"{mover['level']} ({mover['delta']:+.2f})."
                )

        return [Finding(
            finding_id=f"change_{measure}_{latest_label}",
            kind="change",
            headline=headline,
            detail=detail,
            # The per-period gap list can be long on a daily grain; the count
            # stays in evidence.
            evidence={k: v for k, v in output.items() if k != "missing_periods"},
            source_tool=self.name,
            measure=measure,
            dimension=output.get("dimension_column"),
            effect=round(pct_change, 4),
            effect_kind="pct",
            confidence=0.6,
            surprise=min(1.0, abs(pct_change)),
        )]

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset.",
                "required": True,
            },
            "date_column": {
                "type": "string",
                "description": "Datetime column to resample on. Auto-detected when omitted.",
                "required": False,
            },
            "measure_column": {
                "type": "string",
                "description": (
                    "Numeric measure to aggregate per period. Auto-selected "
                    "(preferring a currency measure) when omitted."
                ),
                "required": False,
            },
            "dimension_column": {
                "type": "string",
                "description": (
                    "Optional categorical/flag column (2-20 levels) to break "
                    "the period-over-period change down by, to say which "
                    "segment drove it."
                ),
                "required": False,
            },
        }
