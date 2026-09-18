"""
Dashboard Agent — builds a dashboard that fits the data, not a template.

Given the dataset, its profile, the accumulated tool results, and the ranked
Finding bus (src.core.findings), this agent decides *which* charts are worth
showing (the way a data scientist would) and emits self-contained Vega-Lite
specs:

  - finding-driven panels  built from the top-ranked discoveries first
  - class balance          when a classification target exists
  - histograms             for the most informative numeric features
  - category counts        for low-cardinality categoricals
  - scatter                for the strongest numeric relationship
  - box plots              for the feature that best separates the classes
  - time series            when a datetime column is present
  - model comparison       when training results exist
  - model drivers          when evaluation reports permutation importances
  - LLM-declared charts    validated src.core.chart_spec specs (tool output
                           "chart" or a finding's full-spec chart_hint)
  - correlation bars       when correlation results exist
  - cohort / financial / workforce panels, when those tools ran

Rules:
  - Pure computation: no Streamlit, no file I/O, no LLM. Deterministic.
  - Identifiers and constant columns are never charted.
  - Panels are pre-aggregated (bins, five-number summaries, group totals) —
    not sampled raw rows — except the scatter panel, which genuinely needs
    points (P2.7). Scatter's own cap is lower than the general one.
  - No hardcoded colors anywhere in this file (7.17 / Q1). Every spec either
    omits color entirely (letting the injected src.core.chart_theme.
    vega_config() supply mark defaults) or references a scheme/domain name,
    so the exact same stored spec can be rendered light or dark at
    render/embed time — theme is never baked into a saved artifact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.core.chart_spec import spec_to_vegalite, validate_chart_spec
from src.core.chart_theme import axis_format, humanize_axis_title, humanize_label
from src.core.profiler import ColumnProfile, DatasetProfile

#: Hard cap on inline rows per chart (keeps specs lightweight).
MAX_POINTS = 1_000

#: Scatter is the one panel that genuinely needs raw points (P2.7) — capped
#: much lower than MAX_POINTS since it's visually indistinguishable above a
#: few hundred points at typical opacity, and much lighter in the artifact.
SCATTER_MAX_POINTS = 350

#: How many numeric histograms / categorical bars to show at most.
MAX_HISTOGRAMS = 4
MAX_CATEGORY_CHARTS = 3

#: Categorical columns with more classes than this get truncated to top-N.
MAX_CATEGORIES_SHOWN = 12

#: Default bin count for pre-aggregated histograms.
DEFAULT_HIST_BINS = 20

#: Most charts a dashboard shows; lowest-tier/priority charts are dropped.
MAX_CHARTS = 14

#: Scree plots show at most this many leading components.
SCREE_MAX_COMPONENTS = 15

#: |r| from which the scatter panel draws a fitted trend line.
_TREND_MIN_ABS_R = 0.3

#: Numeric roles that must never be binned/plotted as if continuous
#: (integer-coded dimensions such as year/region codes included).
_NON_CONTINUOUS_ROLES = ("flag", "ordinal", "identifier", "dimension")

#: A category distribution whose largest group is within this ratio of its
#: smallest is "near uniform" — it conveys no story (T5 triviality, applied
#: to panel selection rather than just findings).
_UNIFORM_RATIO_THRESHOLD = 1.15

#: How many top-ranked findings we *attempt* to build/tag a panel for.
#: Most won't produce one (no obvious chart, or the underlying tool didn't
#: run) — this just bounds the work, it isn't a promise of that many panels.
MAX_FINDING_PANELS = 12

#: Findings of these kinds are caveats, not discoveries (src.core.findings.
#: is_trivial never suppresses them for exactly that reason) — they must
#: never be treated as chart-worthy insights.
_CAVEAT_FINDING_KINDS = ("method_fit", "coverage_gap")

#: Which existing chart_id(s) a finding from a given source_tool should be
#: tagged onto (finding_id/priority/layer/caption), so a high-importance
#: discovery's own panel visually leads the dashboard without building a
#: second, redundant chart for it. The first untagged, existing match wins;
#: findings are visited in the caller's importance-descending order.
_SOURCE_TOOL_CHART_IDS: dict[str, tuple[str, ...]] = {
    "train_model": ("model_comparison",),
    "evaluate_model": ("model_drivers", "model_comparison"),
    "cluster_data": ("cluster_scatter",),
    "correlation_analysis": ("top_correlations", "scatter_top_pair"),
    "time_series_analysis": ("time_series",),
    "geospatial_analysis": ("geospatial_hotspots",),
    "dimensionality_analysis": ("pca_scree",),
    "financial_analysis": ("financial_overview",),
    "cohort_analysis": ("cohort_pareto", "cohort_rfm_segments", "cohort_revenue_by_month"),
    "workforce_analysis": ("workforce_headcount_by_dept", "workforce_tenure_hist"),
}

_SAMPLE_SEED = 42


@dataclass
class ChartSpec:
    """One renderable chart: metadata plus a complete Vega-Lite spec.

    `finding_id`/`priority`/`layer`/`caption` are new (7.8 story layer) but
    additive and backward compatible — a chart nobody tagged just keeps the
    defaults, and every existing consumer reading chart_id/title/description/
    spec still gets exactly those keys from to_dict().
    """

    chart_id: str
    title: str
    description: str
    spec: dict[str, Any]
    finding_id: str | None = None
    priority: float = 0.0
    layer: str = "analyst"
    caption: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_id": self.chart_id,
            "title": self.title,
            "description": self.description,
            "spec": self.spec,
            "finding_id": self.finding_id,
            "priority": self.priority,
            "layer": self.layer,
            "caption": self.caption,
        }


def dashboard_to_json(charts: list[ChartSpec]) -> str:
    """Serialise a dashboard for saving to output/dashboard.json."""
    return json.dumps([c.to_dict() for c in charts], indent=2, default=str)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_tool_output(
    tool_results: list[dict[str, Any]], name: str
) -> dict[str, Any] | None:
    """Most recent successful output dict for a tool, or None."""
    for r in reversed(tool_results):
        if r.get("tool_name") == name and r.get("status") == "success":
            out = r.get("output")
            return out if isinstance(out, dict) else None
    return None


def _to_primitive(value: Any) -> Any:
    """Coerce numpy/pandas scalars into JSON-safe Python primitives.

    `dashboard_to_json` serialises with `default=str` — anything that isn't
    already a plain int/float/str/bool/None falls through to `str()`, which
    would silently turn e.g. a numpy.int64 count into the *string* "5" fed
    to a quantitative encoding. Every computed aggregate in this module must
    be routed through this (or an explicit int()/float() cast) before it
    reaches a spec.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _records(df: pd.DataFrame, cols: list[str], max_rows: int = MAX_POINTS) -> list[dict[str, Any]]:
    """Sampled, primitive-typed records for inlining into a Vega-Lite spec."""
    subset = df[cols].dropna()
    if len(subset) > max_rows:
        subset = subset.sample(n=max_rows, random_state=_SAMPLE_SEED)
    return [
        {col: _to_primitive(val) for col, val in row.items()}
        for row in subset.to_dict(orient="records")
    ]


def _chartable(profile: DatasetProfile, *kinds: str) -> list[str]:
    """Column names of the given kinds, excluding identifiers/constants."""
    return [c.name for c in profile.columns_of_kind(*kinds)]


def _safe(fn: Any, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    """Run a panel builder, swallowing any exception into `default`.

    One malformed finding, missing column, or degenerate group must not
    zero out the entire dashboard — the controller already wraps the whole
    of `_generate_dashboard` in a try/except, but that means a single bad
    panel currently costs *every* chart, not just its own.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _merge_axis_format(encoding_entry: dict[str, Any], fmt: dict[str, str]) -> None:
    """Merge `chart_theme.axis_format()`'s fragment into an encoding
    channel's `axis` sub-object in place, without clobbering any axis
    properties the builder already set (e.g. `labelAngle`)."""
    if not fmt:
        return
    axis = dict(encoding_entry.get("axis") or {})
    axis.update(fmt)
    encoding_entry["axis"] = axis


def _histogram_bins(
    series: pd.Series, max_bins: int = DEFAULT_HIST_BINS, log: bool = False
) -> list[dict[str, Any]]:
    """Pre-aggregated bin counts for a numeric series (P2.7) — a histogram
    ships as {bin_start, bin_end, count} triples, never raw values. `log`
    uses geometrically spaced edges (strictly positive series only) so the
    bins are equal-width on a log axis."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return []
    nunique = int(clean.nunique())
    if nunique <= 1:
        return []
    bins: Any = max(1, min(max_bins, nunique))
    if log:
        bins = np.geomspace(float(clean.min()), float(clean.max()), bins + 1)
    counts, edges = np.histogram(clean.to_numpy(dtype=float), bins=bins)
    return [
        {
            "bin_start": round(float(edges[i]), 6),
            "bin_end": round(float(edges[i + 1]), 6),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
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


def _fmt_value(value: float, unit_hint: str | None = None) -> str:
    """Magnitude-aware number for captions: 0.034 stays "0.034", not "0"."""
    if unit_hint == "percent":
        return f"{value:.0%}" if abs(value) >= 0.1 else f"{value:.1%}"
    magnitude = abs(value)
    if magnitude >= 100 or magnitude == 0:
        text = f"{value:,.0f}"
    elif magnitude >= 1:
        text = f"{value:,.2f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:.2g}"
    return f"${text}" if unit_hint == "currency" else text


def _eligible_numeric(ranked: list[str], col_by_name: dict[str, ColumnProfile]) -> list[str]:
    """Numeric columns fit to plot as a continuous quantity: measure-role
    columns when the profile found any, else anything that isn't a flag,
    ordinal or identifier (the histogram rule). Keeps `ranked` order."""
    profiled = [c for c in ranked if col_by_name.get(c) is not None]
    measures = [c for c in profiled if col_by_name[c].is_measure()]
    return measures or [c for c in profiled if col_by_name[c].semantic_role not in _NON_CONTINUOUS_ROLES]


def _eta_squared(feature: pd.Series, groups: pd.Series) -> float:
    """Share of a numeric feature's variance explained by class membership
    (one-way ANOVA effect size) — the association measure for a nominal
    target, where correlating against arbitrary class codes is meaningless."""
    frame = pd.DataFrame({"v": pd.to_numeric(feature, errors="coerce"), "g": groups}).dropna()
    if len(frame) < 3:
        return 0.0
    ss_total = float(((frame["v"] - frame["v"].mean()) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    group_means = frame.groupby("g", observed=True)["v"].transform("mean")
    return 1.0 - float(((frame["v"] - group_means) ** 2).sum()) / ss_total


def _rank_numeric_features(
    df: pd.DataFrame, numeric_cols: list[str], target: str | None, task_type: str | None = None,
) -> list[str]:
    """
    Order numeric features by usefulness: association with the target when
    a usable target exists (eta-squared for a >2-class nominal target,
    |correlation| otherwise), else by variance (normalised).

    Generic exploration ranking — used for histogram/EDA candidate
    selection, where "most informative to look at" is the right criterion.
    Panels whose purpose is specifically "show the measure" (a total/
    aggregate) should use `_pick_measure_column` instead (P2.7).
    """
    usable = [c for c in numeric_cols if c != target and c in df.columns]
    if not usable:
        return []

    if target and target in df.columns:
        target_series = df[target]
        nominal = task_type == "classification" or not pd.api.types.is_numeric_dtype(target_series)
        if nominal and target_series.nunique(dropna=True) > 2:
            eta = {col: _eta_squared(df[col], target_series) for col in usable}
            return sorted(usable, key=lambda c: eta[c], reverse=True)
        if not pd.api.types.is_numeric_dtype(target_series):
            codes, _ = pd.factorize(target_series)
            target_series = pd.Series(codes, index=df.index)
        scores: dict[str, float] = {}
        for col in usable:
            corr = df[col].corr(target_series)
            scores[col] = abs(float(corr)) if pd.notna(corr) else 0.0
        return sorted(usable, key=lambda c: scores[c], reverse=True)

    variances: dict[str, float] = {}
    for col in usable:
        clean = df[col].dropna()
        if clean.empty or float(clean.abs().max()) == 0.0:
            variances[col] = 0.0
        else:
            scale = float(clean.abs().max())
            variances[col] = float((clean / scale).var()) if len(clean) > 1 else 0.0
    return sorted(usable, key=lambda c: variances[c], reverse=True)


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


# ---------------------------------------------------------------------------
# Chart builders — each returns a ChartSpec or None when not applicable
# ---------------------------------------------------------------------------

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
    df: pd.DataFrame, ranked_numeric: list[str], col_by_name: dict[str, ColumnProfile]
) -> list[ChartSpec]:
    """P2.7: bin counts, not raw values. Skips flag/ordinal/identifier
    columns outright (a binary flag's "histogram" is just its two counts,
    reported better as a category chart; an ordinal or identifier should
    never be binned as if it were continuous)."""
    charts: list[ChartSpec] = []
    eligible = [
        c for c in ranked_numeric
        if col_by_name.get(c) is not None
        and col_by_name[c].semantic_role not in _NON_CONTINUOUS_ROLES
    ]
    for col in eligible[:MAX_HISTOGRAMS]:
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
    df: pd.DataFrame, profile: DatasetProfile, target: str | None
) -> tuple[list[ChartSpec], list[ChartSpec]]:
    """Returns (charts, trivial_charts). A near-uniform category distribution
    has no story (T5) and is set aside into `trivial_charts` — the caller
    only falls back to one of those if nothing else in the whole dashboard
    has anything to show."""
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
        if len(charts) < MAX_CATEGORY_CHARTS:
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
    spec: dict[str, Any] = {"data": {"values": values}, "height": 280}
    has_trend = pd.notna(r) and abs(float(r)) >= _TREND_MIN_ABS_R
    if has_trend:
        # Fitted on the plotted points; drawn in the theme's line colour.
        spec["layer"] = [points, {
            "mark": {"type": "line", "strokeDash": [4, 3]},
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
    resampled = (
        frame.set_index(time_col)[value_col]
        .resample(freq)
        .agg(agg)
        .dropna()
        .reset_index()
    )
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
    return ChartSpec(
        chart_id="time_series",
        title=f"Trend — {humanize_label(value_col)} ({grain_label} {verb})",
        description=description,
        spec={
            "usermeta": {"value_column": value_col},
            "data": {"values": values},
            "mark": {"type": "line", "point": True},
            "height": 240,
            "encoding": {
                "x": {"field": "period", "type": "temporal", "title": None},
                "y": y_encoding,
                "tooltip": [
                    {"field": "period", "type": "temporal"},
                    {"field": "value", "title": value_title, **value_fmt},
                ],
            },
        },
    )


def _model_comparison_chart(train_output: dict[str, Any] | None) -> ChartSpec | None:
    if not train_output:
        return None
    models = train_output.get("models_trained", {})
    if not isinstance(models, dict) or not models:
        return None
    task = str(train_output.get("task_type", "classification"))
    metric = "accuracy" if task == "classification" else "r2"
    # Accuracy reads as a percentage; R² stays on its own scale (it can be
    # negative, and the y domain fits the data rather than clipping it).
    scale, metric_title = (100, "Accuracy %") if metric == "accuracy" else (1, "R²")

    rows: list[dict[str, Any]] = []
    for name, m in models.items():
        if not isinstance(m, dict):
            continue
        # A metric the tool didn't report is left out — never drawn as 0.
        for label, raw in (
            ("Train", (m.get("train_metrics") or {}).get(metric)),
            ("Test", (m.get("test_metrics") or {}).get(metric)),
            ("CV mean", m.get("cv_mean")),
        ):
            if isinstance(raw, (int, float)) and not isinstance(raw, bool) and np.isfinite(raw):
                rows.append({"model": str(name), "metric": label, "score": round(float(raw) * scale, 4)})
    if not rows:
        return None
    return ChartSpec(
        chart_id="model_comparison",
        title="Model Comparison",
        description=f"Train vs held-out test vs cross-validated {metric}. "
                    "A large train-test gap signals overfitting.",
        spec={
            "data": {"values": rows},
            "mark": {"type": "bar"},
            "height": 280,
            "encoding": {
                "x": {"field": "model", "type": "nominal", "axis": {"labelAngle": 0, "title": None}},
                "xOffset": {"field": "metric"},
                "y": {"field": "score", "type": "quantitative", "title": metric_title},
                "color": {
                    "field": "metric",
                    "scale": {"domain": ["Train", "Test", "CV mean"]},
                    "legend": {"orient": "top", "title": None},
                },
                "tooltip": [{"field": "model"}, {"field": "metric"},
                            {"field": "score", "title": metric_title, "format": ".3~f"}],
            },
        },
    )


def _drivers_chart(model_output: dict[str, Any] | None) -> ChartSpec | None:
    """Permutation-importance drivers (evaluate_model's `top_drivers`) — the
    one chart that says *what moves the outcome* after a modelling run."""
    if not model_output:
        return None
    drivers = [
        d for d in model_output.get("top_drivers") or []
        if isinstance(d, dict) and d.get("feature") is not None and isinstance(d.get("importance"), (int, float))
    ]
    if not drivers:
        return None
    values = [
        {
            "feature": humanize_label(str(d["feature"])),
            "importance": round(float(d["importance"]), 4),
            "effect": str(d.get("headline") or d.get("direction") or ""),
        }
        for d in drivers
    ]
    return ChartSpec(
        chart_id="model_drivers",
        title="What Drives the Outcome — Top Model Drivers",
        description="Permutation importance on held-out data: how much the model's score drops "
                    "when each feature is shuffled. Longer bar = the model leans on it more.",
        spec={
            "data": {"values": values},
            "mark": {"type": "bar"},
            "height": max(140, 30 * len(values)),
            "encoding": {
                "y": {"field": "feature", "type": "nominal", "sort": "-x", "title": None},
                "x": {"field": "importance", "type": "quantitative", "title": "Permutation importance"},
                "tooltip": [{"field": "feature", "title": "Feature"},
                            {"field": "importance", "title": "Importance", "format": ".3~f"},
                            {"field": "effect", "title": "Effect"}],
            },
        },
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
    description = "Variance explained per principal component (bars, left axis) and running total (line, right axis)."
    if n_needed and threshold:
        description += f" {n_needed} component(s) reach {threshold:.0%} of total variance."
    if len(explained) > SCREE_MAX_COMPONENTS:
        description += f" First {SCREE_MAX_COMPONENTS} of {len(explained)} components shown."
    return ChartSpec(
        chart_id="pca_scree",
        title="Dimensionality — PCA Scree Plot",
        description=description,
        spec={
            "data": {"values": values},
            "height": 280,
            "layer": [
                {
                    # No literal color — bar/line get distinct theme-default
                    # colors from the injected vega_config()'s per-mark-type
                    # config (bar: pen, line: ink).
                    "mark": {"type": "bar"},
                    "encoding": {
                        "x": {"field": "component", "type": "ordinal", "sort": {"field": "order"}, "title": None},
                        "y": {"field": "explained", "type": "quantitative", "title": "Variance explained %"},
                        "tooltip": [{"field": "component"}, {"field": "explained"}],
                    },
                },
                {
                    "mark": {"type": "line", "point": True},
                    "encoding": {
                        "x": {"field": "component", "type": "ordinal", "sort": {"field": "order"}},
                        "y": {"field": "cumulative", "type": "quantitative", "title": "Cumulative %",
                              "scale": {"domain": [0, 100]}, "axis": {"orient": "right"}},
                        "tooltip": [{"field": "component"}, {"field": "cumulative"}],
                    },
                },
            ],
            "resolve": {"scale": {"y": "independent"}},
        },
    )


def _correlation_chart(corr_output: dict[str, Any] | None) -> ChartSpec | None:
    if not corr_output:
        return None
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
    return ChartSpec(
        chart_id="top_correlations",
        title="Top Feature Correlations",
        description="Strongest pairwise relationships, ranked by strength |r|; colour gives the sign.",
        spec={
            "data": {"values": values},
            "mark": {"type": "bar"},
            "height": max(160, len(values) * 30),
            "encoding": {
                "y": {"field": "pair", "type": "nominal", "title": None,
                      "sort": {"field": "abs_correlation", "order": "descending"}},
                "x": {"field": "correlation", "type": "quantitative",
                      "scale": {"domain": [-1, 1]}, "title": "Correlation (r)"},
                "color": {
                    "field": "direction", "type": "nominal",
                    "scale": {"domain": ["positive", "negative"]},
                    "legend": {"orient": "top", "title": None},
                },
                "tooltip": [{"field": "pair", "title": "Pair"},
                            {"field": "correlation", "title": "r", "format": ".2f"}],
            },
        },
    )


# ---------------------------------------------------------------------------
# 6.1 — cohort / financial / workforce panels (previously entirely absent).
# All pre-aggregated from the tool's own output where it already computed
# the right grain (revenue by month, RFM segments, department rollups); the
# tenure histogram and revenue-concentration Pareto recompute from `df`
# because the tool output only carries summary quantiles, not bin counts —
# still aggregate-first (bins / deciles), never raw per-row inlining.
# ---------------------------------------------------------------------------

def _cohort_charts(df: pd.DataFrame, cohort_output: dict[str, Any] | None) -> list[ChartSpec]:
    if not cohort_output:
        return []
    charts: list[ChartSpec] = []

    segments = cohort_output.get("rfm_segments")
    if isinstance(segments, list) and segments:
        values = [
            {
                "segment": str(s.get("segment")),
                "revenue_share_pct": float(s.get("revenue_share_pct") or 0.0),
                "customer_share_pct": float(s.get("customer_share_pct") or 0.0),
            }
            for s in segments if isinstance(s, dict)
        ]
        if values:
            charts.append(ChartSpec(
                chart_id="cohort_rfm_segments",
                title="Customer Segments — Revenue vs Customer Share (RFM)",
                description=(
                    "Share of revenue vs. share of the customer base per RFM segment. "
                    "A segment whose revenue bar towers over its customer bar is doing outsized work."
                ),
                spec={
                    "data": {"values": values},
                    "transform": [{"fold": ["revenue_share_pct", "customer_share_pct"], "as": ["metric", "share"]}],
                    "mark": {"type": "bar"},
                    "height": max(160, 26 * len(values)),
                    "encoding": {
                        "y": {"field": "segment", "type": "nominal", "sort": "-x", "title": None},
                        "x": {"field": "share", "type": "quantitative", "title": "Share %"},
                        "xOffset": {"field": "metric"},
                        "color": {
                            "field": "metric", "type": "nominal",
                            "scale": {"domain": ["revenue_share_pct", "customer_share_pct"]},
                            "legend": {"orient": "top", "title": None},
                        },
                        "tooltip": [{"field": "segment"}, {"field": "metric"}, {"field": "share"}],
                    },
                },
            ))

    revenue_by_month = cohort_output.get("revenue_by_month")
    if isinstance(revenue_by_month, dict) and len(revenue_by_month) >= 2:
        values = [{"period": k, "revenue": float(v)} for k, v in sorted(revenue_by_month.items())]
        charts.append(ChartSpec(
            chart_id="cohort_revenue_by_month",
            title="Revenue by Month",
            description="Total revenue per calendar month.",
            spec={
                "data": {"values": values},
                "mark": {"type": "line", "point": True},
                "height": 220,
                "encoding": {
                    "x": {"field": "period", "type": "temporal", "title": None},
                    "y": {"field": "revenue", "type": "quantitative", "title": "Revenue", "scale": {"zero": False}},
                    "tooltip": [{"field": "period", "type": "temporal"}, {"field": "revenue"}],
                },
            },
        ))

    customer_col = cohort_output.get("customer_column")
    amount_col = cohort_output.get("amount_column")
    if customer_col and amount_col and customer_col in df.columns and amount_col in df.columns:
        amounts = pd.to_numeric(df[amount_col], errors="coerce")
        per_customer = amounts.groupby(df[customer_col]).sum().dropna()
        total = float(per_customer.sum())
        if total > 0 and len(per_customer) >= 10:
            ranked_rev = per_customer.sort_values(ascending=False).reset_index(drop=True)
            n = len(ranked_rev)
            deciles: list[dict[str, Any]] = []
            for decile in range(1, 11):
                cutoff = max(1, round(n * decile / 10))
                cumulative = float(ranked_rev.iloc[:cutoff].sum())
                deciles.append({
                    "customer_decile_pct": decile * 10,
                    "cumulative_revenue_share_pct": round(cumulative / total * 100, 2),
                })
            charts.append(ChartSpec(
                chart_id="cohort_pareto",
                title="Revenue Concentration — Customers Ranked by Spend",
                description=(
                    "Cumulative share of total revenue as more customers (ranked highest-spend "
                    "first) are included — how much of the business rests on a few customers."
                ),
                spec={
                    "data": {"values": deciles},
                    "mark": {"type": "line", "point": True},
                    "height": 220,
                    "encoding": {
                        "x": {"field": "customer_decile_pct", "type": "quantitative", "title": "Top % of customers"},
                        "y": {"field": "cumulative_revenue_share_pct", "type": "quantitative",
                              "title": "Cumulative revenue share %", "scale": {"domain": [0, 100]}},
                        "tooltip": [{"field": "customer_decile_pct"}, {"field": "cumulative_revenue_share_pct"}],
                    },
                },
            ))
    return charts


def _financial_charts(df: pd.DataFrame, financial_output: dict[str, Any] | None) -> list[ChartSpec]:
    if not financial_output:
        return []
    date_col = financial_output.get("date_column")
    price_col = financial_output.get("price_column")
    if not date_col or not price_col or date_col not in df.columns or price_col not in df.columns:
        return []

    symbol_col = financial_output.get("symbol_column")
    cols = [date_col, price_col] + ([symbol_col] if symbol_col and symbol_col in df.columns else [])
    frame = df[cols].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce", format="mixed")
    frame[price_col] = pd.to_numeric(frame[price_col], errors="coerce")
    frame = frame.dropna(subset=[date_col, price_col])
    if frame.empty:
        return []

    label: str | None = None
    if symbol_col and symbol_col in frame.columns:
        best_symbol = (financial_output.get("best_performer") or {}).get("symbol")
        if best_symbol is not None:
            matched = frame[frame[symbol_col].astype(str) == str(best_symbol)]
            if not matched.empty:
                frame = matched
                label = str(best_symbol)

    frame = frame[[date_col, price_col]].sort_values(date_col)
    # Aggregate-first for long series (P2.7) — resample to weekly last-value
    # rather than inlining every row; a further stride decimates the rare
    # case where even weekly resampling exceeds MAX_POINTS.
    if len(frame) > MAX_POINTS:
        frame = frame.set_index(date_col)[price_col].resample("W").last().dropna().reset_index()
    if len(frame) > MAX_POINTS:
        step = max(1, len(frame) // MAX_POINTS)
        frame = frame.iloc[::step]
    if len(frame) < 3:
        return []

    prices = frame[price_col].astype(float).reset_index(drop=True)
    dates = frame[date_col].reset_index(drop=True)
    base = float(prices.iloc[0])
    running_peak = prices.cummax()
    rows: list[dict[str, Any]] = []
    for ts, price, peak in zip(dates, prices, running_peak, strict=True):
        cumulative_return = (float(price) / base - 1.0) * 100 if base else 0.0
        drawdown = (float(price) / float(peak) - 1.0) * 100 if peak else 0.0
        rows.append({
            "date": ts.strftime("%Y-%m-%d"),
            "cumulative_return_pct": round(cumulative_return, 4),
            "drawdown_pct": round(drawdown, 4),
        })

    title_symbol = f" — {label}" if label else ""
    return [ChartSpec(
        chart_id="financial_overview",
        title=f"Return & Drawdown{title_symbol}",
        description=(
            f"Cumulative return and drawdown of '{price_col}' over time"
            + (f" (best performer '{label}')" if label else "")
            + "."
        ),
        spec={
            # Named dataset (P2.7) — both layers read the same rows once
            # instead of each mark inlining its own copy.
            "datasets": {"financial_series": rows},
            "height": 260,
            "layer": [
                {
                    "data": {"name": "financial_series"},
                    "mark": {"type": "area", "opacity": 0.35},
                    "encoding": {
                        "x": {"field": "date", "type": "temporal", "title": None},
                        "y": {"field": "drawdown_pct", "type": "quantitative", "title": "Drawdown %"},
                    },
                },
                {
                    "data": {"name": "financial_series"},
                    "mark": {"type": "line"},
                    "encoding": {
                        "x": {"field": "date", "type": "temporal", "title": None},
                        "y": {"field": "cumulative_return_pct", "type": "quantitative",
                              "title": "Cumulative return %"},
                        "tooltip": [
                            {"field": "date", "type": "temporal"},
                            {"field": "cumulative_return_pct"},
                            {"field": "drawdown_pct"},
                        ],
                    },
                },
            ],
            "resolve": {"scale": {"y": "independent"}},
        },
    )]


def _workforce_charts(df: pd.DataFrame, workforce_output: dict[str, Any] | None) -> list[ChartSpec]:
    if not workforce_output:
        return []
    charts: list[ChartSpec] = []

    by_dept = workforce_output.get("by_department")
    if isinstance(by_dept, list) and by_dept:
        values = [
            {"department": str(r.get("department")), "headcount": int(r.get("headcount") or 0)}
            for r in by_dept if isinstance(r, dict)
        ]
        if values:
            charts.append(ChartSpec(
                chart_id="workforce_headcount_by_dept",
                title="Headcount by Department",
                description="Number of employee records per department.",
                spec={
                    "data": {"values": values},
                    "mark": {"type": "bar"},
                    "height": max(140, 26 * len(values)),
                    "encoding": {
                        "y": {"field": "department", "type": "nominal", "sort": "-x", "title": None},
                        "x": {"field": "headcount", "type": "quantitative", "title": "Headcount"},
                        "tooltip": [{"field": "department"}, {"field": "headcount"}],
                    },
                },
            ))

    hire_col = workforce_output.get("hire_date_column")
    if hire_col and hire_col in df.columns:
        hired = pd.to_datetime(df[hire_col], errors="coerce", format="mixed")
        exit_col = workforce_output.get("exit_date_column")
        if exit_col and exit_col in df.columns:
            ended = pd.to_datetime(df[exit_col], errors="coerce", format="mixed")
        else:
            ended = pd.Series(pd.NaT, index=df.index)
        ref_str = workforce_output.get("tenure_reference_date")
        reference = pd.Timestamp(ref_str) if ref_str else pd.Timestamp.now()
        ended = ended.fillna(reference)
        tenure_years = (ended - hired).dt.total_seconds() / (365.25 * 86400)
        tenure_years = tenure_years[tenure_years.notna() & (tenure_years >= 0)]
        if len(tenure_years) >= 5:
            bins = _histogram_bins(tenure_years, max_bins=15)
            if bins:
                charts.append(ChartSpec(
                    chart_id="workforce_tenure_hist",
                    title="Tenure Distribution",
                    description="Years of tenure across employees (pre-binned).",
                    spec={
                        "data": {"values": bins},
                        "mark": {"type": "bar"},
                        "height": 200,
                        "encoding": {
                            "x": {"field": "bin_start", "bin": "binned", "type": "quantitative", "title": "Years"},
                            "x2": {"field": "bin_end"},
                            "y": {"field": "count", "type": "quantitative", "title": "Employees"},
                            "tooltip": [{"field": "bin_start"}, {"field": "bin_end"}, {"field": "count"}],
                        },
                    },
                ))
    return charts


# ---------------------------------------------------------------------------
# 7.8 — story layer: tag existing panels with the top-ranked finding that
# explains them, so the dashboard's most important chart is visually first
# rather than always being whatever built first in code order.
# ---------------------------------------------------------------------------

def _finding_fits_chart(finding: dict[str, Any], chart: ChartSpec) -> bool:
    """A correlation finding only captions a chart that plots its own pair;
    only driver findings caption the drivers chart."""
    if chart.chart_id == "model_drivers":
        return finding.get("kind") == "driver"
    if chart.chart_id not in ("top_correlations", "scatter_top_pair"):
        return True
    evidence = finding.get("evidence") or {}
    pair = frozenset((str(evidence.get("col_a")), str(evidence.get("col_b"))))
    if chart.chart_id == "top_correlations":
        rows = (chart.spec.get("data") or {}).get("values") or []
        return any(frozenset((r.get("col_a"), r.get("col_b"))) == pair for r in rows)
    encoding = (chart.spec.get("layer") or [chart.spec])[0].get("encoding") or {}
    fields = ((encoding.get("x") or {}).get("field"), (encoding.get("y") or {}).get("field"))
    return None not in fields and frozenset(fields) == pair


def _attach_finding_metadata(charts: list[ChartSpec], findings: list[dict[str, Any]]) -> None:
    """
    Mutates `charts` in place: for each of the top-ranked, non-caveat
    findings (in the caller's own importance-descending order — never
    re-sorted here), tag the first still-untagged chart its source_tool is
    known to produce (and whose data the finding actually describes) with
    finding_id/priority/layer/caption. A chart nobody tags keeps its
    ChartSpec defaults (priority 0.0, no caption). Findings that already
    own a chart (their own chart_hint spec) are skipped.
    """
    by_id = {c.chart_id: c for c in charts}
    used_finding_ids: set[str] = {c.finding_id for c in charts if c.finding_id}
    considered = [f for f in findings if f.get("kind") not in _CAVEAT_FINDING_KINDS][:MAX_FINDING_PANELS]
    for finding in considered:
        source = finding.get("source_tool")
        if not source:
            continue
        fid = finding.get("finding_id") or f"{finding.get('kind')}::{finding.get('measure')}"
        if fid in used_finding_ids:
            continue
        for chart_id in _SOURCE_TOOL_CHART_IDS.get(source, ()):
            chart = by_id.get(chart_id)
            if chart is None or chart.finding_id is not None or not _finding_fits_chart(finding, chart):
                continue
            chart.finding_id = fid
            chart.priority = float(finding.get("importance") or 0.0)
            chart.layer = finding.get("layer") or "analyst"
            chart.caption = finding.get("headline")
            used_finding_ids.add(fid)
            break


# ---------------------------------------------------------------------------
# LLM-authored declarative charts (src.core.chart_spec): a successful tool
# output's "chart" key, or a finding whose chart_hint is a full spec.
# ---------------------------------------------------------------------------

def _llm_chart(clean: dict[str, Any], chart_id: str, finding: dict[str, Any] | None) -> ChartSpec:
    headline = finding.get("headline") if finding else None
    y_label = humanize_label(clean["y"]) if clean.get("y") else "Rows"
    description = clean.get("caption") or headline or "Chart produced during the analysis."
    if clean.get("truncated"):
        description += f" Only the first {len(clean['data'])} rows are plotted."
    return ChartSpec(
        chart_id=chart_id,
        title=clean.get("title") or f"{y_label} by {humanize_label(clean['x'])}",
        description=description,
        spec=spec_to_vegalite(clean),
        finding_id=finding.get("finding_id") if finding else None,
        priority=float(finding.get("importance") or 0.0) if finding else 0.0,
        layer=(finding.get("layer") if finding else None) or "analyst",
        caption=headline or clean.get("caption"),
    )


def _llm_charts(results: list[dict[str, Any]], findings: list[dict[str, Any]]) -> list[ChartSpec]:
    """Validated, themed charts the LLM declared. Finding-borne specs come
    first so a spec reaching both paths keeps its finding link. A tool-output
    chart is linked to a finding only when that tool ran exactly once (the
    only case where "which finding explains this chart" is unambiguous)."""
    charts: list[ChartSpec] = []
    seen: set[str] = set()
    linked: set[str] = set()

    def signature(clean: dict[str, Any]) -> str:
        return json.dumps({k: clean.get(k) for k in ("type", "x", "y", "color", "data")},
                          sort_keys=True, default=str)

    def add(raw: Any, chart_id: str, finding: dict[str, Any] | None) -> None:
        clean, _ = validate_chart_spec(raw)
        if clean is None or signature(clean) in seen:
            return
        chart = _safe(_llm_chart, clean, chart_id, finding)
        if chart is not None:
            seen.add(signature(clean))
            charts.append(chart)
            if chart.finding_id:
                linked.add(chart.finding_id)

    for i, f in enumerate(findings):
        hint = f.get("chart_hint")
        if isinstance(hint, dict) and "type" in hint and "data" in hint:
            fid = f.get("finding_id") or f"finding_{i}"
            add(hint, f"llm_{fid}", {**f, "finding_id": fid})

    ok = [r for r in results if r.get("status") == "success" and isinstance(r.get("output"), dict)]
    runs: dict[str, int] = {}
    for r in ok:
        runs[str(r.get("tool_name"))] = runs.get(str(r.get("tool_name")), 0) + 1
    for i, r in enumerate(ok):
        raw = r["output"].get("chart")
        if raw is None:
            continue
        tool = str(r.get("tool_name"))
        finding: dict[str, Any] | None = None
        if runs[tool] == 1:
            finding = next(
                (f for f in findings
                 if f.get("source_tool") == tool and f.get("finding_id") not in linked
                 and f.get("kind") not in _CAVEAT_FINDING_KINDS),
                None,
            )
        add(raw, f"llm_{finding['finding_id'] if finding else f'{tool}_{i}'}", finding)
    return charts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_dashboard(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_column: str | None = None,
    task_type: str | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    findings: list[dict[str, Any]] | None = None,
) -> list[ChartSpec]:
    """
    Select and build the charts that best fit this dataset and its results.

    Args:
        df:            The dataset (cleaned version preferred when available).
        profile:       DatasetProfile from profile_dataframe().
        target_column: ML target, if any.
        task_type:     "classification" | "regression" | "clustering" | "eda".
        tool_results:  Serialised ToolResult dicts from the memory system.
        findings:      Finding.to_dict() dicts (src.core.findings), already
                       ranked by importance descending — used to tag the
                       panel(s) each finding explains so they lead the
                       dashboard (7.8 story layer). Optional and backward
                       compatible: omitted or empty, the dashboard is built
                       exactly as it was before findings existed.

    Returns:
        Ordered list of at most MAX_CHARTS ChartSpecs — finding-tagged
        panels first (by priority), then results charts, then EDA charts.
    """
    results = tool_results or []
    findings = findings or []
    col_by_name = {c.name: c for c in profile.columns}
    numeric_cols = _chartable(profile, "numeric")
    ranked = _rank_numeric_features(df, numeric_cols, target_column, task_type)
    continuous = _eligible_numeric(ranked, col_by_name)

    train_out = _find_tool_output(results, "train_model")
    corr_out = _find_tool_output(results, "correlation_analysis")
    cluster_out = _find_tool_output(results, "cluster_data")
    ts_out = _find_tool_output(results, "time_series_analysis")
    geo_out = _find_tool_output(results, "geospatial_analysis")
    dim_out = _find_tool_output(results, "dimensionality_analysis")
    cohort_out = _find_tool_output(results, "cohort_analysis")
    financial_out = _find_tool_output(results, "financial_analysis")
    workforce_out = _find_tool_output(results, "workforce_analysis")
    model_out = _find_tool_output(results, "evaluate_model") or train_out

    candidates: list[ChartSpec | None] = [
        _safe(_drivers_chart, model_out),
        _safe(_model_comparison_chart, train_out),
        _safe(_cluster_chart, cluster_out),
        _safe(_correlation_chart, corr_out),
        _safe(_time_series_chart, df, profile, ranked, col_by_name, target_column, ts_out),
        _safe(_geospatial_chart, geo_out),
        _safe(_scree_chart, dim_out),
    ]
    charts: list[ChartSpec] = _safe(_llm_charts, results, findings, default=[])
    charts.extend(c for c in candidates if c is not None)
    charts.extend(_safe(_financial_charts, df, financial_out, default=[]))
    charts.extend(_safe(_cohort_charts, df, cohort_out, default=[]))
    charts.extend(_safe(_workforce_charts, df, workforce_out, default=[]))

    eda_candidates: list[ChartSpec | None] = [
        _safe(_class_balance_chart, df, target_column, task_type),
        _safe(_box_plot_chart, df, continuous, target_column, task_type, col_by_name),
        _safe(_scatter_chart, df, continuous, target_column, task_type, corr_out, col_by_name),
    ]
    eda: list[ChartSpec] = [c for c in eda_candidates if c is not None]
    eda.extend(_safe(_histogram_charts, df, ranked, col_by_name, default=[]))
    cat_charts, trivial_cat_charts = _safe(
        _category_charts, df, profile, target_column, default=([], [])
    )
    eda.extend(cat_charts)
    if not charts and not eda and trivial_cat_charts:
        # Nothing else to show beats a uniform-count bar chart with no story.
        eda.append(trivial_cat_charts[0])

    # Same data twice: the cohort revenue line duplicates the time-series
    # panel of the same measure; the department count bar duplicates the
    # workforce headcount panel.
    ids = {c.chart_id for c in charts}
    drop: set[str] = set()
    ts_chart = next((c for c in charts if c.chart_id == "time_series"), None)
    if (
        ts_chart is not None and cohort_out
        and ts_chart.spec.get("usermeta", {}).get("value_column") == cohort_out.get("amount_column")
    ):
        drop.add("cohort_revenue_by_month")
    if "workforce_headcount_by_dept" in ids and workforce_out and workforce_out.get("department_column"):
        drop.add(f"cat_{workforce_out['department_column']}")
    charts = [c for c in charts if c.chart_id not in drop]
    eda = [c for c in eda if c.chart_id not in drop]

    charts.extend(eda)
    _safe(_attach_finding_metadata, charts, findings)

    # Finding-tagged panels lead (7.8) by priority, then results charts,
    # then EDA; the stable sort keeps build order within each tier.
    eda_ids = {id(c) for c in eda}
    charts.sort(key=lambda c: (0 if c.finding_id else 1 if id(c) not in eda_ids else 2, -c.priority))
    return charts[:MAX_CHARTS]
