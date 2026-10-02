"""
Dashboard Agent — builds a dashboard that fits the data, not a template.

Given the dataset, its profile, the accumulated tool results, and the ranked
Finding bus (src.core.findings), this agent decides *which* charts are worth
showing (the way a data scientist would) and emits self-contained Vega-Lite
specs:

  - finding-driven panels  built from the top-ranked discoveries first:
                           segment lift (bars + CI), concentration (Lorenz),
                           change (waterfall by segment), group test
                           (mean ± CI), plus the drivers dot plot and trend
                           line (Sen's slope) tagged with their findings
  - class balance          when a classification target exists
  - histograms             for the most informative numeric features
  - category counts        for low-cardinality categoricals
  - scatter                for the strongest numeric relationship
  - box plots              for the feature that best separates the classes
  - time series            when a datetime column is present
  - model comparison       when training results exist
  - model drivers          when evaluation reports permutation importances
  - confusion matrix / ROC when evaluate_model reports them (held-out rows)
  - LLM-declared charts    validated src.core.chart_spec specs (tool output
                           "chart" / "charts" or a finding's full-spec
                           chart_hint)
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
import re
from typing import Any

import pandas as pd

from src.core.chart_spec import chart_has_data, spec_to_vegalite, validate_chart_spec
from src.core.chart_theme import humanize_label
from src.core.dashboard_charts_basic import (
    _HEATMAP_GROUP_DISTANCE,
    _SAMPLE_SEED,
    _TREND_MIN_ABS_R,
    _UNIFORM_RATIO_THRESHOLD,
    HEATMAP_MAX_COLUMNS,
    HEATMAP_MIN_COLUMNS,
    HEATMAP_TEXT_MAX_COLUMNS,
    HEATMAP_TEXT_MIN_ABS_R,
    MAX_CATEGORY_CHARTS,
    MAX_HISTOGRAMS,
    SCATTER_MAX_POINTS,
    SCREE_MAX_COMPONENTS,
    _box_plot_chart,
    _category_charts,
    _class_balance_chart,
    _cluster_chart,
    _correlation_chart,
    _five_number_summary,
    _geospatial_chart,
    _histogram_charts,
    _is_near_uniform,
    _pick_measure_column,
    _records,
    _scatter_chart,
    _scree_chart,
    _time_series_chart,
    correlation_heatmap,
)
from src.core.dashboard_charts_domain import (
    _GRAIN_UNIT,
    LORENZ_MAX_POINTS,
    WATERFALL_MAX_SEGMENTS,
    _change_comparison_chart,
    _change_waterfall_chart,
    _cohort_charts,
    _cohort_month_chart,
    _cohort_pareto_chart,
    _cohort_rfm_chart,
    _financial_charts,
    _group_ci_chart,
    _lorenz_chart,
    _merge_small_bins,
    _segment_lift_chart,
    _signed,
    _tool_outputs,
    _workforce_charts,
    _workforce_dept_chart,
    _workforce_tenure_chart,
)
from src.core.dashboard_charts_model import (
    _confusion_matrix_chart,
    _drivers_chart,
    _importance_bounds,
    _model_comparison_chart,
    _roc_chart,
)
from src.core.dashboard_common import (
    _NON_CONTINUOUS_ROLES,
    CORRELATION_HEATMAP_TITLE,
    DEFAULT_HIST_BINS,
    MAX_CATEGORIES_SHOWN,
    MAX_POINTS,
    ChartSpec,
    _chartable,
    _fmt_value,
    _histogram_bins,
    _merge_axis_format,
    _num,
    _safe,
    _slug,
    _to_primitive,
    dashboard_to_json,
)
from src.core.plain_language import fallback_caption
from src.core.profiler import ColumnProfile, DatasetProfile

#: Technical safety ceiling on rendered charts (render-performance protection,
#: not a target): the design stage decides how many charts are meaningful.
MAX_CHARTS = 60

#: Public surface of this module. The chart builders now live in dashboard_common /
#: dashboard_charts_*; they are re-exported here so `src.core.dashboard.<name>` keeps working.
__all__ = [
    "CORRELATION_HEATMAP_TITLE",
    "DEFAULT_HIST_BINS",
    "HEATMAP_MAX_COLUMNS",
    "HEATMAP_MIN_COLUMNS",
    "HEATMAP_TEXT_MAX_COLUMNS",
    "HEATMAP_TEXT_MIN_ABS_R",
    "LORENZ_MAX_POINTS",
    "MAX_AUTO_RESULT_CHARTS",
    "MAX_CATEGORIES_SHOWN",
    "MAX_CATEGORY_CHARTS",
    "MAX_CHARTS",
    "MAX_FINDING_PANELS",
    "MAX_HISTOGRAMS",
    "MAX_PANELS_PER_KIND",
    "MAX_POINTS",
    "SCATTER_MAX_POINTS",
    "SCREE_MAX_COMPONENTS",
    "WATERFALL_MAX_SEGMENTS",
    "_CAVEAT_FINDING_KINDS",
    "_DATE_LABEL_RE",
    "_GRAIN_UNIT",
    "_HEATMAP_GROUP_DISTANCE",
    "_NON_CONTINUOUS_ROLES",
    "_SAMPLE_SEED",
    "_SOURCE_TOOL_CHART_IDS",
    "_TREND_MIN_ABS_R",
    "_UNIFORM_RATIO_THRESHOLD",
    "ChartSpec",
    "_attach_finding_metadata",
    "_box_plot_chart",
    "_category_charts",
    "_change_comparison_chart",
    "_change_waterfall_chart",
    "_chartable",
    "_class_balance_chart",
    "_cluster_chart",
    "_cohort_charts",
    "_cohort_month_chart",
    "_cohort_pareto_chart",
    "_cohort_rfm_chart",
    "_confusion_matrix_chart",
    "_correlation_chart",
    "_covers_pair",
    "_drivers_chart",
    "_eligible_numeric",
    "_eta_squared",
    "_financial_charts",
    "_find_tool_output",
    "_finding_charts",
    "_finding_columns",
    "_finding_fits_chart",
    "_five_number_summary",
    "_fmt_value",
    "_geospatial_chart",
    "_group_ci_chart",
    "_has_data",
    "_histogram_bins",
    "_histogram_charts",
    "_importance_bounds",
    "_is_near_uniform",
    "_llm_chart",
    "_llm_charts",
    "_lorenz_chart",
    "_merge_axis_format",
    "_merge_small_bins",
    "_model_comparison_chart",
    "_num",
    "_pick_measure_column",
    "_rank_numeric_features",
    "_records",
    "_result_table_chart",
    "_roc_chart",
    "_safe",
    "_scatter_chart",
    "_scree_chart",
    "_segment_lift_chart",
    "_signed",
    "_slug",
    "_tag",
    "_time_series_chart",
    "_to_primitive",
    "_tool_outputs",
    "_workforce_charts",
    "_workforce_dept_chart",
    "_workforce_tenure_chart",
    "build_dashboard",
    "correlation_heatmap",
    "dashboard_to_json",
    "designed_chart",
    "merge_designed",
]



#: How many top-ranked findings we *attempt* to build/tag a panel for.
#: Most won't produce one (no obvious chart, or the underlying tool didn't
#: run) — this just bounds the work, it isn't a promise of that many panels.
MAX_FINDING_PANELS = 30


#: At most this many segment-lift / group-test panels (each is one
#: measure x dimension pairing; more would crowd out every other kind).
MAX_PANELS_PER_KIND = 3


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


def _has_data(chart: ChartSpec) -> bool:
    """False only for a chart that would render nothing; never raises."""
    return bool(_safe(chart_has_data, chart.spec, default=True))


def _covers_pair(charts: list[ChartSpec], pair: list[Any]) -> bool:
    """True when a chart's id or title already names both columns of `pair`."""
    if len(pair) != 2:
        return False
    for chart in charts:
        haystack = f"{chart.chart_id} {chart.title}".lower()
        if all(str(c).lower() in haystack or humanize_label(str(c)).lower() in haystack for c in pair):
            return True
    return False


def _find_tool_output(
    tool_results: list[dict[str, Any]], name: str
) -> dict[str, Any] | None:
    """Most recent successful output dict for a tool, or None."""
    for r in reversed(tool_results):
        if r.get("tool_name") == name and r.get("status") == "success":
            out = r.get("output")
            return out if isinstance(out, dict) else None
    return None


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


def _finding_columns(findings: list[dict[str, Any]], known: set[str]) -> set[str]:
    """Dataset columns any non-caveat finding refers to: its measure /
    dimension / `columns`, or any string inside its evidence that is a column
    name (evidence key names vary by tool — measure, feature, col_a, ...)."""
    found: set[str] = set()

    def walk(value: Any, depth: int = 0) -> None:
        if isinstance(value, str):
            if value in known:
                found.add(value)
        elif depth < 3 and isinstance(value, dict):
            for item in value.values():
                walk(item, depth + 1)
        elif depth < 3 and isinstance(value, (list, tuple)):
            for item in value[:50]:
                walk(item, depth + 1)

    for finding in findings:
        if finding.get("kind") in _CAVEAT_FINDING_KINDS:
            continue
        for value in (finding.get("measure"), finding.get("dimension"), finding.get("columns"), finding.get("evidence")):
            walk(value)
    return found


def _tag(chart: ChartSpec, finding: dict[str, Any]) -> ChartSpec:
    chart.finding_id = finding.get("finding_id") or f"{finding.get('kind')}::{finding.get('measure')}"
    chart.priority = float(finding.get("importance") or 0.0)
    chart.layer = finding.get("layer") or "analyst"
    chart.caption = fallback_caption(finding) or finding.get("headline")
    return chart


def _finding_charts(
    df: pd.DataFrame,
    results: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    col_by_name: dict[str, ColumnProfile],
    claimed: set[str],
    skip_pairs: set[tuple[str, str]],
) -> list[ChartSpec]:
    """One panel per top-ranked finding whose kind has a chart form and no
    panel yet: `claimed` holds finding ids already charted (an LLM spec);
    `skip_pairs` holds (measure, dimension) pairs another panel already
    shows. Findings are visited in the caller's importance order."""
    charts: list[ChartSpec] = []
    per_kind: dict[str, int] = {}
    considered = [f for f in findings if f.get("kind") not in _CAVEAT_FINDING_KINDS][:MAX_FINDING_PANELS]
    for finding in considered:
        kind = str(finding.get("kind"))
        if finding.get("finding_id") in claimed or per_kind.get(kind, 0) >= MAX_PANELS_PER_KIND:
            continue
        chart: ChartSpec | None = None
        source = finding.get("source_tool")
        if source == "segment_comparison" and kind == "segment_lift":
            chart = _safe(_segment_lift_chart, finding, results, col_by_name)
        elif source == "concentration_analysis" and kind == "concentration":
            chart = _safe(_lorenz_chart, df, finding)
        elif source == "change_analysis" and kind == "change":
            chart = _safe(_change_waterfall_chart, finding, col_by_name)
        elif (
            source == "select_statistical_test" and kind == "test"
            and (finding.get("measure"), finding.get("dimension")) not in skip_pairs
        ):
            chart = _safe(_group_ci_chart, df, finding, col_by_name)
        if chart is None or any(c.chart_id == chart.chart_id for c in charts):
            continue
        per_kind[kind] = per_kind.get(kind, 0) + 1
        charts.append(_tag(chart, finding))
    return charts


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
    columns = (chart.spec.get("usermeta") or {}).get("columns") or []
    return frozenset(str(c) for c in columns) == pair


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
            chart.caption = fallback_caption(finding) or finding.get("headline")
            used_finding_ids.add(fid)
            break


def _llm_chart(clean: dict[str, Any], chart_id: str, finding: dict[str, Any] | None) -> ChartSpec:
    headline = finding.get("headline") if finding else None
    y_label = humanize_label(clean["y"]) if clean.get("y") else "Rows"
    description = clean.get("caption") or headline or "Chart produced during the analysis."
    if clean.get("note"):
        description += f" {clean['note']}"
    if clean.get("truncated"):
        description += f" Only the first {len(clean['data'])} rows are plotted."
    default_title = f"{y_label} by {humanize_label(clean['x'])}" if clean.get("x") else "Custom chart"
    return ChartSpec(
        chart_id=chart_id,
        title=clean.get("title") or default_title,
        description=description,
        spec=spec_to_vegalite(clean),
        finding_id=finding.get("finding_id") if finding else None,
        priority=(float(finding.get("importance") or 0.0) if finding else 0.0) + clean.get("priority", 0) / 100,
        layer=(finding.get("layer") if finding else None) or "analyst",
        caption=clean.get("caption") or (fallback_caption(finding) if finding else None) or headline,
        size=clean.get("size"),
    )


def designed_chart(raw: dict[str, Any], index: int, findings: list[dict[str, Any]]) -> ChartSpec | None:
    """A validated chart from an LLM-designed recipe (already turned into a raw
    spec), linked to its finding when `finding_id` matches one; None if rejected."""
    clean, _ = validate_chart_spec(raw)
    if clean is None:
        return None
    fid = raw.get("finding_id")
    finding = next((f for f in findings if fid and f.get("finding_id") == fid), None)
    chart: ChartSpec | None = _safe(_llm_chart, clean, f"design_{index}", finding)
    if chart is not None and finding is None:
        chart.priority = 0.5
    return chart


def merge_designed(
    charts: list[ChartSpec], designed: list[ChartSpec], drop_ids: set[str]
) -> list[ChartSpec]:
    """Drop generic charts the designer made redundant, add its charts, re-rank."""
    kept = [
        c for c in charts
        if not (c.chart_id in drop_ids and c.finding_id is None and not c.chart_id.startswith("llm_"))
    ]
    merged = [c for c in kept + designed if _has_data(c)]
    merged.sort(key=lambda c: (
        0 if c.finding_id
        else 2 if (c.priority == 0 and not c.chart_id.startswith("llm_"))
        else 1,
        -c.priority,
    ))
    return merged[:MAX_CHARTS]


#: Sandbox runs whose RESULT table gets a chart although the code declared none.
MAX_AUTO_RESULT_CHARTS = 2


_DATE_LABEL_RE = re.compile(r"\d{4}-\d{2}(-\d{2})?")


def _result_table_chart(result: Any) -> dict[str, Any] | None:
    """A bar (line for date labels) spec for an aggregated RESULT table the
    code never charted: 2-24 uniform rows, one label column, one or more
    numeric columns (the first is plotted)."""
    if not (isinstance(result, list) and 2 <= len(result) <= 24 and all(isinstance(r, dict) for r in result)):
        return None
    cols = list(result[0])
    if len(cols) < 2 or any(list(r) != cols for r in result):
        return None
    nums = [c for c in cols if all(isinstance(r[c], (int, float)) and not isinstance(r[c], bool) for r in result)]
    labels = [c for c in cols if c not in nums]
    if len(labels) != 1 or not nums:
        return None
    x, y = labels[0], nums[0]
    kind = "line" if all(_DATE_LABEL_RE.match(str(r[x])) for r in result) else "bar"
    return {
        "type": kind, "x": x, "y": y, "data": result,
        "caption": f"{humanize_label(y)} by {humanize_label(x)}, from a custom calculation.",
    }


def _llm_charts(results: list[dict[str, Any]], findings: list[dict[str, Any]]) -> list[ChartSpec]:
    """Validated, themed charts the LLM declared. Finding-borne specs come
    first so a spec reaching both paths keeps its finding link. A tool-output
    chart is linked to a finding only when that tool ran exactly once (the
    only case where "which finding explains this chart" is unambiguous)."""
    charts: list[ChartSpec] = []
    seen: set[str] = set()
    linked: set[str] = set()

    def signature(clean: dict[str, Any]) -> str:
        return json.dumps({k: clean.get(k) for k in ("type", "x", "y", "color", "data", "vega_lite")},
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
    auto_left = MAX_AUTO_RESULT_CHARTS
    for i, r in enumerate(ok):
        # A single declared "chart", or a "charts" list (generate_visualizations).
        raw_charts = r["output"].get("charts")
        declared = [r["output"].get("chart"), *(raw_charts if isinstance(raw_charts, list) else [])]
        tool = str(r.get("tool_name"))
        if (
            auto_left and all(c is None for c in declared)
            and "stdout" in r["output"] and not r["output"].get("chart_error")
        ):
            auto = _result_table_chart(r["output"].get("result"))
            if auto is not None:
                declared, auto_left = [auto], auto_left - 1
        for j, raw in enumerate(c for c in declared if c is not None):
            finding: dict[str, Any] | None = None
            if runs[tool] == 1:
                finding = next(
                    (f for f in findings
                     if f.get("source_tool") == tool and f.get("finding_id") not in linked
                     and f.get("kind") not in _CAVEAT_FINDING_KINDS),
                    None,
                )
            suffix = "" if j == 0 else f"_{j}"
            add(raw, f"llm_{finding['finding_id'] if finding else f'{tool}_{i}'}{suffix}", finding)
    return charts


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
    eval_out = _find_tool_output(results, "evaluate_model")
    model_out = eval_out or train_out

    candidates: list[ChartSpec | None] = [
        _safe(_drivers_chart, model_out),
        _safe(_model_comparison_chart, train_out),
        _safe(_confusion_matrix_chart, eval_out),
        _safe(_roc_chart, eval_out),
        _safe(_cluster_chart, cluster_out),
        _safe(_correlation_chart, df, continuous, corr_out),
        _safe(_time_series_chart, df, profile, ranked, col_by_name, target_column, ts_out),
        _safe(_geospatial_chart, geo_out),
    ]
    charts: list[ChartSpec] = _safe(_llm_charts, results, findings, default=[])
    charts.extend(c for c in candidates if c is not None)
    charts.extend(_safe(_financial_charts, df, financial_out, default=[]))
    charts.extend(_safe(_cohort_charts, df, cohort_out, default=[]))
    charts.extend(_safe(_workforce_charts, df, workforce_out, default=[]))

    # Strict curation: distributions and category counts only for columns a
    # finding refers to; the generic box/scatter candidates rank last.
    referenced = _finding_columns(findings, set(df.columns.astype(str)))
    eda: list[ChartSpec] = []
    class_balance = _safe(_class_balance_chart, df, target_column, task_type)
    if class_balance is not None:
        eda.append(class_balance)
    eda.extend(_safe(_histogram_charts, df, ranked, col_by_name, referenced, default=[]))
    cat_charts, trivial_cat_charts = _safe(
        _category_charts, df, profile, target_column, referenced, default=([], [])
    )
    eda.extend(cat_charts)
    eda_candidates: list[ChartSpec | None] = [
        _safe(_box_plot_chart, df, continuous, target_column, task_type, col_by_name),
        _safe(_scatter_chart, df, continuous, target_column, task_type, corr_out, col_by_name),
        _safe(_scree_chart, dim_out),  # technical: ranks with the EDA charts, below findings
    ]
    eda.extend(c for c in eda_candidates if c is not None)

    # Finding-kind panels (segment lift, concentration, change, group test)
    # for findings no LLM chart already covers; a group test on the pair the
    # class box plot already shows is left to that box plot.
    box_pairs = {(c.chart_id.removeprefix("box_"), str(target_column)) for c in eda if c.chart_id.startswith("box_")}
    claimed = {c.finding_id for c in charts if c.finding_id}
    charts.extend(_safe(_finding_charts, df, results, findings, col_by_name, claimed, box_pairs, default=[]))

    # Same data twice: the cohort revenue line duplicates the time-series
    # panel of the same measure; the department count bar duplicates the
    # workforce headcount panel; the cohort spend Pareto duplicates a Lorenz
    # curve of the same measure over the same customers.
    ids = {c.chart_id for c in charts}
    drop: set[str] = set()
    if cohort_out and (
        f"lorenz_{_slug(cohort_out.get('amount_column'))}_{_slug(cohort_out.get('customer_column'))}" in ids
    ):
        drop.add("cohort_pareto")
    ts_chart = next((c for c in charts if c.chart_id == "time_series"), None)
    if (
        ts_chart is not None and cohort_out
        and ts_chart.spec.get("usermeta", {}).get("value_column") == cohort_out.get("amount_column")
    ):
        drop.add("cohort_revenue_by_month")
    if "workforce_headcount_by_dept" in ids and workforce_out and workforce_out.get("department_column"):
        drop.add(f"cat_{workforce_out['department_column']}")
    corr_chart = next((c for c in charts if c.chart_id == "top_correlations"), None)
    if corr_chart is not None and corr_chart.title == CORRELATION_HEATMAP_TITLE and any(
        c.title == CORRELATION_HEATMAP_TITLE and c is not corr_chart for c in charts
    ):
        drop.add("top_correlations")
    # A finding-driven (or LLM) chart of the same column pair makes the generic
    # top-pair scatter a duplicate.
    scatter = next((c for c in eda if c.chart_id == "scatter_top_pair"), None)
    if scatter is not None and _covers_pair(charts, (scatter.spec.get("usermeta") or {}).get("columns") or []):
        drop.add("scatter_top_pair")
    charts = [c for c in charts if c.chart_id not in drop]
    eda = [c for c in eda if c.chart_id not in drop]

    if not charts and not eda:
        # Never an empty dashboard: the single top-ranked histogram, else the
        # most lopsided category count, else (last resort) a uniform one.
        eda = _safe(_histogram_charts, df, ranked, col_by_name, None, 1, default=[])
        if not eda:
            loose, _ = _safe(_category_charts, df, profile, target_column, None, default=([], []))
            eda = (loose or trivial_cat_charts)[:1]

    charts.extend(eda)
    _safe(_attach_finding_metadata, charts, findings)

    # Finding-tagged panels and LLM-declared charts lead (7.8) by priority,
    # then results charts, then EDA; the stable sort keeps build order within
    # each tier. An LLM chart is never outranked by a generic one under
    # MAX_CHARTS.
    eda_ids = {id(c) for c in eda}
    charts = [c for c in charts if _has_data(c)]
    charts.sort(key=lambda c: (
        0 if c.finding_id or c.chart_id.startswith("llm_") else 1 if id(c) not in eda_ids else 2, -c.priority,
    ))
    return charts[:MAX_CHARTS]
