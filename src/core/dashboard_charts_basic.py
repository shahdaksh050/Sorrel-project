"""
Dashboard Agent — distribution, relationship, time-series and correlation charts.

Split out of dashboard.py; `src.core.dashboard` re-exports every name here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.chart_theme import axis_format, humanize_axis_title, humanize_label
from src.core.dashboard_common import (
    _NON_CONTINUOUS_ROLES,
    CORRELATION_HEATMAP_TITLE,
    MAX_CATEGORIES_SHOWN,
    MAX_POINTS,
    ChartSpec,
    _chartable,
    _fmt_value,
    _histogram_bins,
    _merge_axis_format,
    _num,
    _to_primitive,
)
from src.core.profiler import ColumnProfile, DatasetProfile

#: Scatter is the one panel that genuinely needs raw points (P2.7) — capped
#: much lower than MAX_POINTS since it's visually indistinguishable above a
#: few hundred points at typical opacity, and much lighter in the artifact.
SCATTER_MAX_POINTS = 350


#: How many numeric histograms / categorical bars to show at most — and only
#: for columns a finding actually references (strict curation).
MAX_HISTOGRAMS = 2


MAX_CATEGORY_CHARTS = 3


#: Correlation heatmap: fewest chartable numeric columns worth a matrix, the
#: most it shows (n² cells), and the |r| at which a cell prints its value.
HEATMAP_MIN_COLUMNS = 5


HEATMAP_MAX_COLUMNS = 15


HEATMAP_TEXT_MAX_COLUMNS = 15


HEATMAP_TEXT_MIN_ABS_R = 0.3


#: Average-linkage cut (on 1 - |r|) that defines "a group that moves together".
_HEATMAP_GROUP_DISTANCE = 0.5


#: Scree plots show at most this many leading components.
SCREE_MAX_COMPONENTS = 15


#: |r| from which the scatter panel draws a fitted trend line.
_TREND_MIN_ABS_R = 0.3


#: A category distribution whose largest group is within this ratio of its
#: smallest is "near uniform" — it conveys no story (T5 triviality, applied
#: to panel selection rather than just findings).
_UNIFORM_RATIO_THRESHOLD = 1.15


_SAMPLE_SEED = 42


def _records(df: pd.DataFrame, cols: list[str], max_rows: int = MAX_POINTS) -> list[dict[str, Any]]:
    """Sampled, primitive-typed records for inlining into a Vega-Lite spec."""
    subset = df[cols].dropna()
    if len(subset) > max_rows:
        subset = subset.sample(n=max_rows, random_state=_SAMPLE_SEED)
    return [
        {col: _to_primitive(val) for col, val in row.items()}
        for row in subset.to_dict(orient="records")
    ]


def _five_number_summary(series: pd.Series) -> dict[str, float] | None:
    """Whisker-clipped five-number summary (P2.7) for a manual box plot —
    min/max here are the 1.5xIQR whiskers clipped to the observed range,
    matching the `extent: 1.5` the raw-data boxplot mark used to compute."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    q1, median, q3 = (float(clean.quantile(q)) for q in (0.25, 0.5, 0.75))
    iqr = q3 - q1
    obs_min, obs_max = float(clean.min()), float(clean.max())
    whisker_low = max(obs_min, q1 - 1.5 * iqr)
    whisker_high = min(obs_max, q3 + 1.5 * iqr)
    return {
        "low": round(whisker_low, 4),
        "q1": round(q1, 4),
        "median": round(median, 4),
        "q3": round(q3, 4),
        "high": round(whisker_high, 4),
    }


def _is_near_uniform(counts: pd.Series, ratio_threshold: float = _UNIFORM_RATIO_THRESHOLD) -> bool:
    """True when a category-count distribution has no story — every group is
    roughly the same size, so a bar chart of it conveys nothing (T5 applied
    to panel selection, not just findings)."""
    if len(counts) < 2:
        return False
    lo, hi = float(counts.min()), float(counts.max())
    if lo <= 0:
        return False
    return (hi / lo) <= ratio_threshold


def _pick_measure_column(
    ranked_numeric: list[str], col_by_name: dict[str, ColumnProfile], target: str | None,
) -> str | None:
    """
    Pick the numeric column to use where the panel's purpose is "show the
    measure" (a total/aggregate) rather than generic exploration — P2.7's
    named bug: "monthly mean quantity" should become "monthly total
    revenue" when a currency measure exists. Prefers a semantic `measure`
    column, and among those a `currency` one, over whatever ranked highest
    by pure correlation/variance.
    """
    candidates = [
        c for c in ranked_numeric
        if c != target and col_by_name.get(c) is not None and col_by_name[c].is_measure()
    ]
    if not candidates:
        return None
    currency = [c for c in candidates if col_by_name[c].unit_hint == "currency"]
    return currency[0] if currency else candidates[0]


def _class_balance_chart(
    df: pd.DataFrame, target: str | None, task_type: str | None
) -> ChartSpec | None:
    if not target or target not in df.columns or task_type != "classification":
        return None
    counts = df[target].dropna().astype(str).value_counts().head(MAX_CATEGORIES_SHOWN)
    if counts.empty:
        return None
    values = [{"class": str(k), "count": int(v)} for k, v in counts.items()]
    count_fmt = axis_format("count")
    y_encoding: dict[str, Any] = {"field": "count", "type": "quantitative", "title": "Rows"}
    _merge_axis_format(y_encoding, count_fmt)
    return ChartSpec(
        chart_id="class_balance",
        title=f"Class Balance — {humanize_label(target)}",
        description="Distribution of the target classes. Heavy imbalance means accuracy is misleading.",
        spec={
            "data": {"values": values},
            "mark": {"type": "bar"},
            "height": 220,
            "encoding": {
                # Class order carries no meaning on its own — sort by count
                # descending so the largest class renders first.
                "x": {"field": "class", "type": "nominal", "sort": "-y",
                      "axis": {"labelAngle": 0, "title": None}},
                "y": y_encoding,
                "color": {"field": "class", "type": "nominal", "legend": None},
                "tooltip": [{"field": "class"}, {"field": "count", **count_fmt}],
            },
        },
    )


def _histogram_charts(
    df: pd.DataFrame,
    ranked_numeric: list[str],
    col_by_name: dict[str, ColumnProfile],
    referenced: set[str] | None = None,
    limit: int = MAX_HISTOGRAMS,
) -> list[ChartSpec]:
    """P2.7: bin counts, not raw values. Skips flag/ordinal/identifier
    columns outright (a binary flag's "histogram" is just its two counts,
    reported better as a category chart; an ordinal or identifier should
    never be binned as if it were continuous). With `referenced`, only
    columns a finding uses are drawn (up to `limit`); None means any."""
    charts: list[ChartSpec] = []
    eligible = [
        c for c in ranked_numeric
        if col_by_name.get(c) is not None
        and col_by_name[c].semantic_role not in _NON_CONTINUOUS_ROLES
        and (referenced is None or c in referenced)
    ]
    for col in eligible:
        if len(charts) >= limit:
            break
        # A severely skewed, strictly positive measure is binned and drawn on
        # a log axis — linear bins would pile almost every row into one bar.
        log_x = "severe_skew" in col_by_name[col].flags and bool((pd.to_numeric(df[col], errors="coerce").dropna() > 0).all())
        bins = _histogram_bins(df[col], log=log_x)
        if not bins:
            continue
        unit_hint = col_by_name[col].unit_hint if col_by_name.get(col) else None
        fmt = axis_format(unit_hint)
        axis_title = humanize_axis_title(col, unit_hint)

        x_encoding: dict[str, Any] = {"field": "bin_start", "type": "quantitative", "title": axis_title}
        if log_x:
            x_encoding["scale"] = {"type": "log"}
        else:
            x_encoding["bin"] = "binned"
        _merge_axis_format(x_encoding, fmt)

        # 8.4 — an EDA filler panel (no bound finding, since hist_* chart_ids
        # never appear in _SOURCE_TOOL_CHART_IDS) gets a one-line plain-
        # language caption computed from the same quantile logic the box
        # plot already uses, not a second distribution pass.
        caption: str | None = None
        summary = _five_number_summary(df[col])
        if summary is not None:
            col_label = humanize_label(col).lower()
            lo, hi = _fmt_value(summary["q1"], unit_hint), _fmt_value(summary["q3"], unit_hint)
            caption = f"Most {col_label} values fall between {lo} and {hi}."

        charts.append(ChartSpec(
            chart_id=f"hist_{col}",
            title=f"Distribution — {humanize_label(col)}",
            description=f"Pre-binned histogram of '{col}'. Watch for skew, gaps, and outlier tails."
                        + (" Heavily right-skewed, so bins are log-spaced on a log axis." if log_x else ""),
            spec={
                "data": {"values": bins},
                "mark": {"type": "bar"},
                "height": 200,
                "encoding": {
                    "x": x_encoding,
                    "x2": {"field": "bin_end"},
                    "y": {"field": "count", "type": "quantitative", "title": "Rows"},
                    "tooltip": [
                        {"field": "bin_start", "title": f"{axis_title} from", **fmt},
                        {"field": "bin_end", "title": "to", **fmt},
                        {"field": "count", "title": "Rows"},
                    ],
                },
            },
            caption=caption,
        ))
    return charts


def _category_charts(
    df: pd.DataFrame, profile: DatasetProfile, target: str | None,
    referenced: set[str] | None = None,
) -> tuple[list[ChartSpec], list[ChartSpec]]:
    """Returns (charts, trivial_charts). A near-uniform category distribution
    has no story (T5) and is set aside into `trivial_charts` — the caller
    only falls back to one of those if nothing else in the whole dashboard
    has anything to show. With `referenced`, `charts` holds only columns a
    finding uses (None means any)."""
    charts: list[ChartSpec] = []
    trivial: list[ChartSpec] = []
    cat_cols = [
        c for c in profile.columns_of_kind("categorical", "boolean")
        if c.name != target and "high_cardinality" not in c.flags
    ]
    # Most informative first: how far the largest group sits above an even
    # split. Ascending cardinality used to hand every slot to 2-level columns.
    full_by_col = {c.name: df[c.name].dropna().astype(str).value_counts() for c in cat_cols}
    cat_cols = [c for c in cat_cols if not full_by_col[c.name].empty]
    cat_cols.sort(
        key=lambda c: float(full_by_col[c.name].iloc[0] / full_by_col[c.name].sum()) - 1.0 / len(full_by_col[c.name]),
        reverse=True,
    )
    for col in cat_cols:
        full_counts = full_by_col[col.name]
        counts = full_counts.head(MAX_CATEGORIES_SHOWN)
        values = [{"category": str(k), "count": int(v)} for k, v in counts.items()]
        if len(full_counts) > MAX_CATEGORIES_SHOWN:
            counts = full_counts.head(MAX_CATEGORIES_SHOWN - 1)
            rest = full_counts.iloc[MAX_CATEGORIES_SHOWN - 1:]
            values = [{"category": str(k), "count": int(v)} for k, v in counts.items()]
            values.append({"category": f"Other ({len(rest)} more)", "count": int(rest.sum())})

        count_fmt = axis_format("count")
        x_encoding: dict[str, Any] = {"field": "count", "type": "quantitative", "title": "Rows"}
        _merge_axis_format(x_encoding, count_fmt)

        # 8.4 — cat_* charts never appear in _SOURCE_TOOL_CHART_IDS, so they
        # are always an unbound EDA filler panel; give them a plain-language
        # caption from the counts already computed above (total against the
        # full, untruncated distribution, not just the top-N shown).
        top_category, top_count = str(counts.index[0]), int(counts.iloc[0])
        total = int(full_counts.sum())
        pct = (top_count / total * 100) if total else 0.0
        caption = f"{top_category} is the largest group at {pct:.0f}% of rows."

        spec = ChartSpec(
            chart_id=f"cat_{col.name}",
            title=f"Category Counts — {humanize_label(col.name)}",
            description=f"Frequency of each '{col.name}' value"
                        + (f" (top {MAX_CATEGORIES_SHOWN - 1}, the rest grouped as Other)."
                           if len(full_counts) > MAX_CATEGORIES_SHOWN else "."),
            spec={
                "data": {"values": values},
                "mark": {"type": "bar"},
                "height": max(120, 24 * len(values)),
                "encoding": {
                    # Rows arrive largest-first with "Other" last — keep that order.
                    "y": {"field": "category", "type": "nominal", "sort": None, "title": None},
                    "x": x_encoding,
                    "tooltip": [{"field": "category"}, {"field": "count", **count_fmt}],
                },
            },
            caption=caption,
        )
        if _is_near_uniform(counts):
            trivial.append(spec)
            continue
        if len(charts) < MAX_CATEGORY_CHARTS and (referenced is None or col.name in referenced):
            charts.append(spec)
    return charts, trivial


def _scatter_chart(
    df: pd.DataFrame,
    ranked_numeric: list[str],
    target: str | None,
    task_type: str | None,
    corr_output: dict[str, Any] | None,
    col_by_name: dict[str, ColumnProfile] | None = None,
) -> ChartSpec | None:
    """`ranked_numeric` must already be restricted to continuous columns
    (`_eligible_numeric`); a correlation pair outside it is skipped."""
    pair: tuple[str, str] | None = None
    if corr_output:
        for entry in corr_output.get("top_correlations", []):
            a, b = str(entry.get("col_a", "")), str(entry.get("col_b", ""))
            if a in ranked_numeric and b in ranked_numeric and a != target and b != target:
                pair = (a, b)
                break
    if pair is None and len(ranked_numeric) >= 2:
        pair = (ranked_numeric[0], ranked_numeric[1])
    if pair is None:
        return None

    cols = list(pair)
    color_field: str | None = None
    if (
        task_type == "classification"
        and target
        and target in df.columns
        and df[target].nunique(dropna=True) <= 10
    ):
        color_field = target
        cols.append(target)

    # Scatter is the one panel P2.7 exempts from aggregate-first — it
    # genuinely needs points — but its own cap is much lower than the
    # general MAX_POINTS (visually indistinguishable above a few hundred).
    values = _records(df, cols, max_rows=SCATTER_MAX_POINTS)
    if not values:
        return None
    complete = df[cols].dropna()
    r = complete[pair[0]].astype(float).corr(complete[pair[1]].astype(float))
    if color_field:
        for row in values:
            row[color_field] = str(row[color_field])

    col_by_name = col_by_name or {}
    unit_x = col_by_name[pair[0]].unit_hint if col_by_name.get(pair[0]) else None
    unit_y = col_by_name[pair[1]].unit_hint if col_by_name.get(pair[1]) else None
    fmt_x, fmt_y = axis_format(unit_x), axis_format(unit_y)
    title_x, title_y = humanize_axis_title(pair[0], unit_x), humanize_axis_title(pair[1], unit_y)

    x_encoding: dict[str, Any] = {"field": pair[0], "type": "quantitative",
                                   "scale": {"zero": False}, "title": title_x}
    y_encoding: dict[str, Any] = {"field": pair[1], "type": "quantitative",
                                   "scale": {"zero": False}, "title": title_y}
    _merge_axis_format(x_encoding, fmt_x)
    _merge_axis_format(y_encoding, fmt_y)

    unit_by_field = {pair[0]: fmt_x, pair[1]: fmt_y}
    title_by_field = {pair[0]: title_x, pair[1]: title_y}
    encoding: dict[str, Any] = {
        "x": x_encoding,
        "y": y_encoding,
        "tooltip": [
            {"field": c, "title": title_by_field.get(c, humanize_label(c)), **unit_by_field.get(c, {})}
            for c in cols
        ],
    }
    if color_field:
        encoding["color"] = {"field": color_field, "type": "nominal", "title": humanize_label(color_field)}

    points = {"mark": {"type": "circle", "opacity": 0.55, "size": 36}, "encoding": encoding}
    spec: dict[str, Any] = {"data": {"values": values}, "height": 280, "usermeta": {"columns": list(pair)}}
    has_trend = pd.notna(r) and abs(float(r)) >= _TREND_MIN_ABS_R
    if has_trend:
        # Fitted on the plotted points; drawn in the theme's accent, over the green points.
        spec["layer"] = [points, {
            "mark": {"type": "line", "strokeDash": [4, 3], "style": "accent"},
            "transform": [{"regression": pair[1], "on": pair[0]}],
            "encoding": {"x": {"field": pair[0], "type": "quantitative"},
                         "y": {"field": pair[1], "type": "quantitative"}},
        }]
    else:
        spec.update(points)

    description = "The strongest numeric relationship in the data"
    if pd.notna(r):
        description += f" (r = {float(r):.2f}" + (", dashed line = linear fit)" if has_trend else ")")
    description += " — colored by target class." if color_field else "."
    if len(complete) > len(values):
        description += f" Showing a random sample of {len(values):,} of {len(complete):,} rows."
    return ChartSpec(
        chart_id="scatter_top_pair",
        title=f"Relationship — {humanize_label(pair[0])} vs {humanize_label(pair[1])}",
        description=description,
        spec=spec,
    )


def _box_plot_chart(
    df: pd.DataFrame,
    ranked_numeric: list[str],
    target: str | None,
    task_type: str | None,
    col_by_name: dict[str, ColumnProfile] | None = None,
) -> ChartSpec | None:
    """P2.7: a five-number summary per group (rule+bar+tick layers), not raw
    points — Vega-Lite's `boxplot` mark needs raw data to compute its own
    quartiles, which is exactly the raw-row inlining P2.7 wants gone."""
    if (
        task_type != "classification"
        or not target
        or target not in df.columns
        or not ranked_numeric
        or df[target].nunique(dropna=True) > 10
    ):
        return None
    feature = ranked_numeric[0]
    frame = df[[feature, target]].dropna()
    if frame.empty:
        return None
    summaries: list[dict[str, Any]] = []
    for level, group in frame.groupby(target):
        summary = _five_number_summary(group[feature])
        if summary is None:
            continue
        summaries.append({"level": str(level), **summary})
    if not summaries:
        return None
    col_by_name = col_by_name or {}
    unit_hint = col_by_name[feature].unit_hint if col_by_name.get(feature) else None
    feature_fmt = axis_format(unit_hint)
    feature_title = humanize_axis_title(feature, unit_hint)
    y_low_encoding: dict[str, Any] = {
        "field": "low", "type": "quantitative", "scale": {"zero": False}, "title": feature_title,
    }
    _merge_axis_format(y_low_encoding, feature_fmt)
    return ChartSpec(
        chart_id=f"box_{feature}",
        title=f"Separation — {humanize_label(feature)} by {humanize_label(target)}",
        description=(
            f"Five-number summary of '{feature}' (the most target-linked feature) "
            "across classes — box is the interquartile range, whiskers at 1.5×IQR."
        ),
        spec={
            "data": {"values": summaries},
            "height": 240,
            "layer": [
                {
                    "mark": {"type": "rule"},
                    "encoding": {
                        "x": {"field": "level", "type": "nominal", "axis": {"labelAngle": 0}, "title": None},
                        "y": y_low_encoding,
                        "y2": {"field": "high"},
                    },
                },
                {
                    "mark": {"type": "bar", "size": 30},
                    "encoding": {
                        "x": {"field": "level", "type": "nominal"},
                        "y": {"field": "q1", "type": "quantitative"},
                        "y2": {"field": "q3"},
                        "color": {"field": "level", "type": "nominal", "legend": None},
                    },
                },
                {
                    "mark": {"type": "tick", "size": 30},
                    "encoding": {
                        "x": {"field": "level", "type": "nominal"},
                        "y": {"field": "median", "type": "quantitative"},
                    },
                },
            ],
        },
    )


def _time_series_chart(
    df: pd.DataFrame,
    profile: DatasetProfile,
    ranked_numeric: list[str],
    col_by_name: dict[str, ColumnProfile],
    target_column: str | None,
    ts_output: dict[str, Any] | None = None,
) -> ChartSpec | None:
    datetime_cols = _chartable(profile, "datetime")
    if not datetime_cols or not ranked_numeric:
        return None
    # Prefer the columns time_series_analysis actually examined (it may have
    # auto-detected a different pair than "first datetime, top-ranked
    # numeric") so the chart matches the trend/stationarity findings below
    # instead of silently re-deriving its own, possibly different, series.
    time_col = datetime_cols[0]
    # P2.7 named bug: default to a semantic measure (a currency one first)
    # rather than whatever ranked highest by variance/correlation, so a
    # "monthly mean quantity" chart doesn't outrank "monthly total revenue".
    value_col = _pick_measure_column(ranked_numeric, col_by_name, target_column) or ranked_numeric[0]
    if ts_output:
        ts_time_col = ts_output.get("date_column")
        ts_value_col = ts_output.get("value_column")
        if isinstance(ts_time_col, str) and ts_time_col in datetime_cols:
            time_col = ts_time_col
        if (
            isinstance(ts_value_col, str)
            and ts_value_col in df.columns
            and pd.api.types.is_numeric_dtype(df[ts_value_col])
        ):
            value_col = ts_value_col

    frame = df[[time_col, value_col]].dropna()
    if frame.empty:
        return None
    parsed = pd.to_datetime(frame[time_col], errors="coerce", format="mixed")
    frame = frame.assign(**{time_col: parsed}).dropna()
    if frame.empty:
        return None

    # Prefer time_series_analysis's OWN grain/aggregation choice (7.7 —
    # monthly/weekly/daily picked from the date span, sum for additive
    # measures, mean for rates) over re-deriving it here — otherwise the
    # chart can silently disagree with the tool's own trend/seasonality
    # findings (the exact "monthly mean quantity" vs "monthly total revenue"
    # mismatch this round's audit named). Only fall back to a local guess
    # when no tool output is available (e.g. the tool wasn't scheduled).
    grain_freq = {"daily": "D", "weekly": "W", "monthly": "MS"}
    unit_hint = col_by_name[value_col].unit_hint if col_by_name.get(value_col) else None
    tool_grain = str(ts_output.get("grain") or "") if ts_output else ""
    tool_agg = ts_output.get("aggregation") if ts_output else None
    freq = grain_freq.get(tool_grain, "MS")
    profiled_agg = col_by_name[value_col].aggregation if col_by_name.get(value_col) else None
    agg = tool_agg or profiled_agg or ("sum" if unit_hint in ("currency", "count") else "mean")
    # Empty calendar periods are missing, not zero (a plain resample sum
    # would draw them as 0) — matching time_series_analysis. `position`
    # keeps each period's calendar index for the Sen's-slope line below.
    bucketed = frame.set_index(time_col)[value_col].resample(freq)
    resampled = (bucketed.sum(min_count=1) if agg == "sum" else bucketed.agg(agg)).reset_index()
    resampled["position"] = np.arange(len(resampled))
    resampled = resampled.dropna(subset=[value_col]).reset_index(drop=True)
    grain_label = tool_grain if tool_grain in grain_freq else "monthly"
    # A final bucket the data doesn't fully cover (e.g. a month with 9 days
    # of data) shows a fake drop on a sum; drop it. Only when the data is
    # much finer than the grain — monthly rows dated the 1st are complete.
    # Weekly labels are the week's closing Sunday, monthly the first day.
    trimmed_note = ""
    dates = frame[time_col].dt.normalize().drop_duplicates().sort_values()
    gap = dates.diff().median() if len(dates) > 1 else pd.NaT
    if freq != "D" and len(resampled) > 2 and pd.notna(gap):
        last_label = resampled[time_col].iloc[-1]
        if freq == "W":
            start, end = last_label - pd.Timedelta(days=6), last_label + pd.Timedelta(days=1)
        else:
            start, end = last_label, last_label + pd.offsets.MonthBegin(1)
        if gap <= (end - start) / 2 and dates.iloc[-1] + gap < end:
            resampled = resampled.iloc[:-1]
            trimmed_note = (
                f" The final {grain_label.removesuffix('ly')} is omitted — "
                f"data only runs to {dates.iloc[-1]:%Y-%m-%d}, so it is incomplete."
            )
    values = [
        {"period": ts.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
        for ts, v in zip(resampled[time_col], resampled[value_col], strict=True)
    ]
    if len(values) < 2:
        return None
    verb = "total" if agg == "sum" else "average"
    title_override = ts_output.get("chart_title") if ts_output else None
    description = title_override or f"'{value_col}' aggregated ({grain_label} {verb}) over '{time_col}'."
    description += trimmed_note
    if ts_output:
        direction = ts_output.get("trend_direction")
        is_stationary = ts_output.get("is_stationary")
        seasonal_lags = ts_output.get("seasonal_lags_detected") or []
        if direction:
            description += f" Trend: {direction}."
        if is_stationary is not None:
            description += f" {'Stationary' if is_stationary else 'Non-stationary'} (ADF test)."
        if seasonal_lags:
            description += f" Seasonal signal at lag(s) {', '.join(str(x) for x in seasonal_lags)}."

    value_fmt = axis_format(unit_hint)
    value_title = humanize_axis_title(value_col, unit_hint)
    y_encoding: dict[str, Any] = {
        "field": "value", "type": "quantitative", "title": value_title, "scale": {"zero": False},
    }
    _merge_axis_format(y_encoding, value_fmt)
    x_encoding = {"field": "period", "type": "temporal", "title": None}
    series_layer: dict[str, Any] = {
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": x_encoding,
            "y": y_encoding,
            "tooltip": [
                {"field": "period", "type": "temporal"},
                {"field": "value", "title": value_title, **value_fmt},
            ],
        },
    }
    layers: list[dict[str, Any]] = [series_layer]
    if len(values) >= 20:
        # A noisy series: lighten the raw line and overlay a thicker rolling mean.
        window = max(3, len(values) // 12)
        smooth = pd.Series([v["value"] for v in values]).rolling(window, min_periods=1, center=True).mean()
        for row, s in zip(values, smooth, strict=True):
            row["smooth"] = round(float(s), 4)
        series_layer["mark"] = {"type": "line", "opacity": 0.35}
        layers.append({
            "mark": {"type": "line", "strokeWidth": 3},
            "encoding": {"x": x_encoding, "y": {**y_encoding, "field": "smooth"}},
        })
    spec: dict[str, Any] = {"usermeta": {"value_column": value_col}, "data": {"values": values}, "height": 240}
    # Robust trend annotation: Sen's slope (median pairwise slope per
    # period) through the median-residual intercept, drawn dashed, when
    # time_series_analysis ran the Mann-Kendall test on this series.
    # Only when the chart plots the tool's own series (its columns and
    # grain) — otherwise the slope belongs to a different series.
    same_series = ts_output is not None and tool_grain in grain_freq and (time_col, value_col) == (
        ts_output.get("date_column"), ts_output.get("value_column"))
    sen_slope = _num(ts_output.get("sen_slope")) if ts_output and same_series else None
    if ts_output and sen_slope is not None:
        # The slope is per calendar period, or per observed period when the
        # tool tested a gappy series on its observed points only.
        if ts_output.get("sen_slope_unit") == "observed period":
            x_pos = np.arange(len(values), dtype=float)
        else:
            x_pos = resampled["position"].to_numpy(dtype=float)
        series_y = np.array([v["value"] for v in values], dtype=float)
        intercept = float(np.median(series_y - sen_slope * x_pos))
        for row, pos in zip(values, x_pos, strict=True):
            row["sen_fit"] = round(intercept + sen_slope * float(pos), 4)
        layers.append({
            "mark": {"type": "line", "strokeDash": [4, 3], "style": "accent"},
            "encoding": {"x": x_encoding, "y": {"field": "sen_fit", "type": "quantitative"}},
        })
        unit = ts_output.get("sen_slope_unit") or "period"
        mk_p = _num(ts_output.get("mk_p_value"))
        description += (
            f" Dashed line: Sen's slope {'+' if sen_slope >= 0 else '−'}{_fmt_value(abs(sen_slope), unit_hint)}"
            f" per {unit}"
            + (f" (Mann-Kendall p={mk_p:.3g}{', ' + str(ts_output['mk_trend']) if ts_output.get('mk_trend') else ''})"
               if mk_p is not None else "")
            + "."
        )
    if len(layers) > 1:
        spec["layer"] = layers
    else:
        spec.update(series_layer)
    return ChartSpec(
        chart_id="time_series",
        title=f"Trend — {humanize_label(value_col)} ({grain_label} {verb})",
        description=description,
        spec=spec,
    )


def _cluster_chart(cluster_output: dict[str, Any] | None) -> ChartSpec | None:
    if not cluster_output:
        return None
    points = cluster_output.get("pca_points", [])
    if not isinstance(points, list) or len(points) < 10:
        return None
    n = cluster_output.get("n_clusters", "?")
    sil = cluster_output.get("silhouette_score", "?")
    return ChartSpec(
        chart_id="cluster_scatter",
        title=f"Segments — {n} clusters (silhouette {sil})",
        description="Rows projected to 2-D (PCA), colored by discovered cluster. "
                    "Tight, well-separated colors mean meaningful segments.",
        spec={
            "data": {"values": points},
            "mark": {"type": "circle", "opacity": 0.6, "size": 40},
            "height": 300,
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "title": "PC 1",
                      "scale": {"zero": False}},
                "y": {"field": "y", "type": "quantitative", "title": "PC 2",
                      "scale": {"zero": False}},
                "color": {"field": "cluster", "type": "nominal",
                          "legend": {"orient": "top", "title": None}},
                "tooltip": [{"field": "cluster"}],
            },
        },
    )


def _geospatial_chart(geo_output: dict[str, Any] | None) -> ChartSpec | None:
    if not geo_output:
        return None
    cells = geo_output.get("densest_cells", [])
    if not isinstance(cells, list) or not cells:
        return None
    values = [
        {
            "lat": round((c["lat_range"][0] + c["lat_range"][1]) / 2, 5),
            "lon": round((c["lon_range"][0] + c["lon_range"][1]) / 2, 5),
            "count": c["count"],
        }
        for c in cells
        if isinstance(c, dict) and c.get("lat_range") and c.get("lon_range")
    ]
    if not values:
        return None
    return ChartSpec(
        chart_id="geospatial_hotspots",
        title="Geographic Hotspots",
        description="Densest grid cells by point count — bubble size and color show concentration.",
        spec={
            "data": {"values": values},
            "mark": {"type": "circle", "opacity": 0.75},
            "height": 300,
            "encoding": {
                "x": {"field": "lon", "type": "quantitative", "title": "Longitude", "scale": {"zero": False}},
                "y": {"field": "lat", "type": "quantitative", "title": "Latitude", "scale": {"zero": False}},
                "size": {"field": "count", "type": "quantitative", "title": "Points",
                         "scale": {"range": [50, 800]}},
                "color": {"field": "count", "type": "quantitative", "title": "Points",
                          "scale": {"scheme": "oranges"}},
                "tooltip": [{"field": "lat"}, {"field": "lon"}, {"field": "count"}],
            },
        },
    )


def _scree_chart(dim_output: dict[str, Any] | None) -> ChartSpec | None:
    if not dim_output:
        return None
    explained = dim_output.get("explained_variance_ratio", [])
    cumulative = dim_output.get("cumulative_variance", [])
    if not isinstance(explained, list) or not explained:
        return None
    values = [
        {
            "component": f"PC{i + 1}",
            "order": i,
            "explained": round(float(e) * 100, 2),
            "cumulative": round(float(cumulative[i]) * 100, 2) if i < len(cumulative) else None,
        }
        for i, e in enumerate(explained[:SCREE_MAX_COMPONENTS])
    ]
    n_needed = dim_output.get("n_components_for_threshold")
    threshold = dim_output.get("variance_threshold")
    description = "Share of the data's variance each principal component captures (bars) and the running total (line)."
    if n_needed and threshold:
        description += f" {n_needed} component(s) reach {threshold:.0%} of total variance."
    if len(explained) > SCREE_MAX_COMPONENTS:
        description += f" First {SCREE_MAX_COMPONENTS} of {len(explained)} components shown."
    y_scale = {"domain": [0, 100]}
    x_enc = {"field": "component", "type": "ordinal", "sort": {"field": "order"}, "title": None}
    # One percent axis; 80% / 90% reference lines show where "enough" is.
    return ChartSpec(
        chart_id="pca_scree",
        title="How many underlying factors matter",
        description=description,
        spec={
            "data": {"values": values},
            "height": 280,
            "layer": [
                {
                    # No literal color — the bars take the theme's pen and the
                    # cumulative line its accent (the `accent` style), both from
                    # the injected vega_config().
                    "mark": {"type": "bar"},
                    "encoding": {
                        "x": x_enc,
                        "y": {"field": "explained", "type": "quantitative", "scale": y_scale,
                              "title": "Share of variance (%)"},
                        "tooltip": [{"field": "component"}, {"field": "explained", "title": "Component %"},
                                    {"field": "cumulative", "title": "Cumulative %"}],
                    },
                },
                {
                    "mark": {"type": "line", "point": {"style": "accent"}, "style": "accent"},
                    "encoding": {
                        "x": x_enc,
                        "y": {"field": "cumulative", "type": "quantitative", "scale": y_scale},
                    },
                },
                {
                    "data": {"values": [{"level": 80}, {"level": 90}]},
                    "mark": {"type": "rule", "strokeDash": [4, 3], "opacity": 0.5},
                    "encoding": {"y": {"field": "level", "type": "quantitative", "scale": y_scale}},
                },
            ],
        },
    )


def correlation_heatmap(
    corr: pd.DataFrame, max_columns: int = HEATMAP_MAX_COLUMNS,
) -> tuple[list[dict[str, Any]], dict[str, Any], str] | None:
    """Clustered correlation matrix -> (rows, vega_lite_spec_without_data,
    caption); None when fewer than 3 columns have usable correlations.

    Keeps the `max_columns` columns with the highest mean |r| to the rest,
    orders them by average-linkage clustering on 1 - |r| so variables that
    move together sit in adjacent blocks, and captions the largest such
    group. Rows carry `col_a`/`col_b` (raw names) so a correlation finding
    can be matched to the pair. Shared with the generate_visualizations tool."""
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform

    corr = corr.dropna(how="all").dropna(axis=1, how="all")
    corr = corr.loc[corr.index.intersection(corr.columns), corr.index.intersection(corr.columns)]
    if len(corr) < 3:
        return None
    strength = corr.abs().fillna(0.0).to_numpy(copy=True)  # pandas 3 hands back a read-only view
    np.fill_diagonal(strength, 0.0)
    keep =np.argsort(-strength.sum(axis=1) / (len(corr) - 1), kind="stable")[:max_columns]
    corr = corr.iloc[sorted(keep), sorted(keep)]
    n = len(corr)
    strength = corr.abs().fillna(0.0).to_numpy()
    dist = np.clip(1.0 - strength, 0.0, 1.0)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    tree = linkage(squareform(dist, checks=False), method="average", optimal_ordering=True)
    order = [int(i) for i in leaves_list(tree)]
    clusters = fcluster(tree, t=_HEATMAP_GROUP_DISTANCE, criterion="distance")

    names = [str(c) for c in corr.columns]
    labels = [humanize_label(c) for c in names]
    if len(set(labels)) != n:
        labels = names
    ordered = [labels[i] for i in order]
    rows = [
        {
            "feature_x": labels[j], "feature_y": labels[i], "col_a": names[j], "col_b": names[i],
            "r": round(float(corr.iloc[i, j]), 3), "abs_r": round(abs(float(corr.iloc[i, j])), 3),
        }
        for i in range(n) for j in range(n) if pd.notna(corr.iloc[i, j])
    ]

    groups: list[tuple[int, float, bool, list[int]]] = []
    for cid in {int(c) for c in clusters}:
        members = [i for i in order if int(clusters[i]) == cid]
        if len(members) < 2:
            continue
        block = corr.iloc[members, members].to_numpy()
        off = ~np.eye(len(members), dtype=bool)
        groups.append((len(members), float(np.nanmean(np.abs(block[off]))), bool((block[off] > 0).all()), members))
    if groups:
        _, strength_mean, same_way, members = max(groups, key=lambda g: (g[0], g[1]))
        shown = [labels[i] for i in members]
        if len(shown) <= 3:
            listed = " and ".join([", ".join(shown[:-1]), shown[-1]])
        else:
            listed = f"{', '.join(shown[:3])} and {len(shown) - 3} more"
        caption = (
            f"{listed} {'move together' if same_way else 'are closely linked'} "
            f"(average |r| {strength_mean:.2f})."
            + (f" {len(groups) - 1} other group{'s' if len(groups) > 2 else ''} of related columns also stand out."
               if len(groups) > 1 else "")
        )
    else:
        caption = f"No group of columns moves together strongly — every pair has |r| below {1 - _HEATMAP_GROUP_DISTANCE:.1f}."

    r_field = {"field": "r", "type": "quantitative", "title": "r"}
    x_enc: dict[str, Any] = {"field": "feature_x", "type": "nominal", "sort": ordered, "title": None,
                             "axis": {"labelAngle": -45, "labelLimit": 260, "labelOverlap": False,
                                      "orient": "bottom"}}
    y_enc: dict[str, Any] = {"field": "feature_y", "type": "nominal", "sort": ordered, "title": None,
                             "axis": {"labelLimit": 260}}
    layers: list[dict[str, Any]] = [{
        "mark": {"type": "rect"},
        "encoding": {
            "x": x_enc, "y": y_enc,
            "color": {**r_field, "scale": {"scheme": "blueorange", "domainMid": 0, "domain": [-1, 1]},
                      "legend": {"format": ".1f"}},
            "tooltip": [{"field": "feature_x", "title": "Column"}, {"field": "feature_y", "title": "Column"},
                        {**r_field, "format": ".2f"}],
        },
    }]
    if n <= HEATMAP_TEXT_MAX_COLUMNS:
        # Contrast against the cell fill (the scheme is theme-independent),
        # not against the page: white on strong cells, dark on pale ones.
        # Two static-colour layers rather than a `test` condition (an
        # expression, which the raw-spec sanitiser rejects).
        for colour, text_filter in (
            ("white", {"field": "abs_r", "gte": 0.6}),
            ("black", {"and": [{"field": "abs_r", "gte": HEATMAP_TEXT_MIN_ABS_R}, {"field": "abs_r", "lt": 0.6}]}),
        ):
            layers.append({
                "mark": {"type": "text", "fontSize": 10, "color": colour},
                "transform": [{"filter": text_filter}],
                "encoding": {"x": x_enc, "y": y_enc, "text": {**r_field, "format": ".2f"}},
            })
    return rows, {"width": {"step": 26}, "height": {"step": 26}, "layer": layers}, caption


def _correlation_chart(
    df: pd.DataFrame, numeric_cols: list[str], corr_output: dict[str, Any] | None,
) -> ChartSpec | None:
    """With >= HEATMAP_MIN_COLUMNS chartable numeric columns, a clustered
    correlation heatmap of them; otherwise bars of the tool's top pairs
    (colour only when both signs occur — else it carries no information)."""
    if not corr_output:
        return None
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) >= HEATMAP_MIN_COLUMNS:
        built = correlation_heatmap(df[cols].apply(pd.to_numeric, errors="coerce").corr())
        if built is not None:
            rows, spec, caption = built
            return ChartSpec(
                chart_id="top_correlations",
                title=CORRELATION_HEATMAP_TITLE,
                description="Pearson correlation between numeric columns, grouped so related columns sit "
                            "side by side; orange is positive, blue negative.",
                spec={"data": {"values": rows}, **spec},
                caption=caption,
            )
    top = corr_output.get("top_correlations", [])[:10]
    if not top:
        return None
    # col_a/col_b ride along so a correlation finding is only attached to
    # this chart when its own pair is actually plotted here.
    values = [
        {
            "pair": f"{humanize_label(str(e.get('col_a')))} ↔ {humanize_label(str(e.get('col_b')))}",
            "col_a": str(e.get("col_a")),
            "col_b": str(e.get("col_b")),
            "correlation": round(float(e["correlation"]), 4),
            "abs_correlation": round(abs(float(e["correlation"])), 4),
            "direction": "positive" if float(e["correlation"]) >= 0 else "negative",
        }
        for e in top if isinstance(e, dict) and isinstance(e.get("correlation"), (int, float))
    ]
    if not values:
        return None
    directions = {v["direction"] for v in values}
    encoding: dict[str, Any] = {
        "y": {"field": "pair", "type": "nominal", "title": None,
              "sort": {"field": "abs_correlation", "order": "descending"},
              "axis": {"labelLimit": 360}},
        "x": {"field": "correlation", "type": "quantitative",
              "scale": {"domain": [0, 1] if directions == {"positive"} else [-1, 0] if directions == {"negative"} else [-1, 1]},
              "title": "Correlation (r)"},
        "tooltip": [{"field": "pair", "title": "Pair"},
                    {"field": "correlation", "title": "r", "format": ".2f"}],
    }
    if len(directions) > 1:
        encoding["color"] = {
            "field": "direction", "type": "nominal",
            "scale": {"domain": ["positive", "negative"]},
            "legend": {"orient": "top", "title": None},
        }
    return ChartSpec(
        chart_id="top_correlations",
        title="Top Feature Correlations",
        description="Strongest pairwise relationships, ranked by strength |r|"
                    + ("; colour gives the sign." if len(directions) > 1 else f"; all are {next(iter(directions))}."),
        spec={
            "data": {"values": values},
            "mark": {"type": "bar"},
            "height": max(160, len(values) * 30),
            "encoding": encoding,
        },
    )
