"""
Declarative chart contract shared by every chart producer that is not the
deterministic dashboard builder itself: LLM-authored sandbox code (`CHART =
{...}` / `dsa.chart.*`), generated tools, and tool `Finding.chart_hint`s.

Producers emit this small, whitelisted spec (or, as an escape hatch, a
sanitised raw Vega-Lite spec — see `vega_lite` below); `validate_chart_spec`
cleans it (or rejects it with an actionable message the LLM can fix), and the
dashboard translates the cleaned spec into themed Vega-Lite. A bad spec
degrades to "no chart", never to broken or injected rendering code. Scaling
and layout quality (sorting, top-N + Other, zero baselines, number formats,
label orientation, overplotting) is applied here, deterministically — never
left to the producer.

Spec shape (keys not listed here are dropped):
    type      "bar" | "stacked_bar" | "grouped_bar" | "line" | "area" | "band" |
              "scatter" | "histogram" | "heatmap" | "waterfall" | "lorenz" |
              "dot_ci" | "dual_axis" | "boxplot" | "pareto" | "slope" |
              "bullet" | "vega_lite"                                          (required)
    data      list of flat records (dicts of str/number/bool/None)          (required)
    x         field name in `data`                                          (required)
    y         field name in `data` (required except for histogram; numeric
              for every kind but bar/line/area/scatter/heatmap)
    y2        dual_axis only: numeric field drawn on an independent right axis
              (y2_title / y2_format style it like y_title / y_format)
    color     optional field name — series (line/area/scatter) or group (bar)
    series    stacked_bar / grouped_bar: the series field (same as `color`)
    target    bullet only: numeric field holding each row's target
    facet     small multiples: one panel per value of this field (max 12
              panels, 3 columns) — bar/line/area/scatter/histogram/band/
              boxplot/stacked_bar/grouped_bar
    title     short human title
    x_title / y_title   axis titles (default: humanised field names)
    y_format  "currency" | "percent" | "count" | "number"
              ("percent" expects 0-1 fractions; a 0-100 series is shown as-is)
    sort      "desc" | "asc"  (bar family only; default "desc" for a categorical x)
    log_y     bool — log scale on y (positive values only)
    caption   one-sentence takeaway shown under the chart (<= 200 chars)
    annotations   up to 3 reference lines [{"y": number, "label": str}] on the
              value axis of bar-family/line/area/band/scatter/dot_ci/boxplot
    size      "wide" | "half" | "tall" — dashboard layout hint
    priority  int 0-10; higher panels lead the dashboard
    y_lower / y_upper   optional numeric fields in `data` bounding each y
              value (a confidence interval, ±std) — drawn as error bars on
              bar/scatter/dot_ci and as a band on line/area/band. Both or
              neither (both required for band).
    scale     heatmap only: "zscore" — `color` is already standardised per series
              (diverging scale centred on 0, legend "Std. deviations from that
              series' mean")
    order     heatmap only: list of category labels; matching x/y levels are
              drawn in this order (e.g. a clustered correlation matrix). Default:
              nominal axes sort by mean value (calendar names and square
              matrices keep data order), numeric/date axes ascend.
    Heatmaps need LONG format (one row per x/y cell, >= 2 distinct x and y);
    a numeric axis with > 25 distinct values is binned (mean of `color`).
    A deterministic critic rejects degenerate charts (single distinct x, a
    one-category bar, a constant line) with a message naming the fix.

    waterfall x = step label, y = signed change (running total is derived);
    lorenz x/y = cumulative population/value shares, drawn against the
    equality diagonal; boxplot x = group, y = RAW values (>= 4 per group;
    quartiles are computed by the renderer); pareto y = additive value per
    category (bars sorted desc + cumulative-share line); slope x = exactly 2
    time points, color = group; bullet x = category, y = actual, target.

    vega_lite  {"vega_lite": {...}} (no `type` needed) — a raw Vega-Lite spec
    with inline `data.values` only. Rejected: data.url/name/format, datasets,
    params/selection/signals, expression strings (calculate, string filter,
    test, *Expr), lookup, href/url, image marks. Every encoded field must exist
    in the rows (or be produced by a transform `as`). Any `config`/`background`
    is dropped: the app theme is injected at render time. Width is forced to
    the container.
"""
from __future__ import annotations

import importlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime
from typing import Any, TypeGuard

import src.core.chart_theme

# In long-running processes (e.g. Streamlit runner), chart_theme may have been
# imported before humanize_label was defined. Reload defensively if stale.
if not hasattr(src.core.chart_theme, "humanize_label"):
    importlib.reload(src.core.chart_theme)

from src.core.chart_theme import axis_format, humanize_axis_title, humanize_label

CHART_TYPES: frozenset[str] = frozenset({
    "bar", "stacked_bar", "grouped_bar", "line", "area", "band", "scatter", "histogram",
    "heatmap", "waterfall", "lorenz", "dot_ci", "dual_axis", "boxplot", "pareto", "slope",
    "bullet", "vega_lite",
})
#: Kinds whose value axis (`y`, and `y2`) must be numeric to mean anything.
_NUMERIC_Y_TYPES: frozenset[str] = frozenset({
    "waterfall", "lorenz", "dot_ci", "dual_axis", "boxplot", "pareto", "slope", "bullet",
    "band", "stacked_bar", "grouped_bar",
})
_SERIES_TYPES: frozenset[str] = frozenset({"stacked_bar", "grouped_bar"})
_BAR_FAMILY: frozenset[str] = frozenset({"bar", "stacked_bar", "grouped_bar"})
#: Kinds with a categorical x that is capped to `_MAX_CATEGORIES`.
_CATEGORY_TYPES: frozenset[str] = frozenset({"bar", "stacked_bar", "grouped_bar", "pareto", "dot_ci", "bullet"})
_FACET_TYPES: frozenset[str] = frozenset({
    "bar", "stacked_bar", "grouped_bar", "line", "area", "band", "scatter", "histogram", "boxplot",
})
_ANNOTATE_TYPES: frozenset[str] = frozenset({
    "bar", "stacked_bar", "grouped_bar", "line", "area", "band", "scatter", "dot_ci", "boxplot",
})
_NO_BOUNDS_TYPES: frozenset[str] = frozenset({
    "histogram", "heatmap", "waterfall", "lorenz", "dual_axis", "boxplot", "stacked_bar",
    "grouped_bar", "pareto", "slope", "bullet",
})
Y_FORMATS: frozenset[str] = frozenset({"currency", "percent", "count", "number"})
SIZES: frozenset[str] = frozenset({"wide", "half", "tall"})

#: Rows inlined per chart. Keeps dashboard.json and the self-contained HTML
#: report small; aggregate before charting instead of plotting raw rows.
MAX_CHART_ROWS = 500
#: Categories on one axis (tail folded into "Other" / dropped), facet panels.
_MAX_CATEGORIES = 12
_MAX_ANNOTATIONS = 3
_MAX_PRIORITY = 10

_STR_KEYS = ("title", "x_title", "y_title", "y2_title", "caption")
_MAX_TEXT = 200
_OTHER_LABELS = ("Other", "Other (rest)")
_NON_ADDITIVE = frozenset({
    "avg", "average", "mean", "median", "rate", "ratio", "pct", "percent", "percentage", "share",
    "score", "index", "price", "margin", "std", "stdev", "var", "min", "max", "per", "density",
    "age", "rank", "rating",
})
_DATE_FORMATS = (
    "%Y/%m/%d", "%d-%b-%Y", "%b %Y", "%B %Y", "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%Y%m%d",
)


def _clean_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value if not isinstance(value, str) else value[:_MAX_TEXT]
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and not math.isfinite(value) else value
    if hasattr(value, "item"):  # numpy scalar that slipped through
        return _clean_value(value.item())
    return str(value)[:_MAX_TEXT]


def _is_additive(y: str, y_format: str | None) -> bool:
    """Whether summing `y` across categories is meaningful (a total, not a
    rate/average/price)."""
    if y_format == "percent":
        return False
    return not (set(re.split(r"[^a-z0-9]+", y.lower())) & _NON_ADDITIVE)


def _normalise_dates(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    """Rewrite a column of unambiguous non-ISO date strings ("Jan 2024",
    "2024/03/01") to ISO so it is detected — and sorted — as temporal."""
    present = [r[field] for r in rows if r.get(field) is not None]
    if not present or not all(isinstance(v, str) for v in present) or all(_is_iso_date(v) for v in present):
        return rows
    for fmt in _DATE_FORMATS:
        try:
            parsed = {v: datetime.strptime(v.strip(), fmt).date().isoformat() for v in set(present)}
        except ValueError:
            continue
        return [{**r, field: parsed[r[field]] if r.get(field) is not None else None} for r in rows]
    return rows


def _limit_categories(
    rows: list[dict[str, Any]],
    x: str,
    rank_field: str | None,
    group: str | None,
    sum_fields: list[str],
    aggregate: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    """Keep the largest categories of `x` (by |rank_field|, or row count when
    `rank_field` is None). The tail is summed into one "Other" row per group
    when `aggregate` (additive values), otherwise dropped."""
    levels = {r.get(x) for r in rows}
    if len(levels) <= _MAX_CATEGORIES:
        return rows, None
    score: dict[Any, float] = {}
    for r in rows:
        v = r.get(rank_field) if rank_field else 1.0
        score[r.get(x)] = score.get(r.get(x), 0.0) + (abs(v) if _is_number(v) else 0.0)
    ranked = sorted(score, key=lambda k: (-score[k], str(k)))
    n_keep = _MAX_CATEGORIES - 1 if aggregate else _MAX_CATEGORIES
    kept = set(ranked[:n_keep])
    head = [r for r in rows if r.get(x) in kept]
    if not aggregate:
        return head, f"Showing the {n_keep} largest of {len(levels)} categories."
    label = next((lab for lab in _OTHER_LABELS if lab not in kept), "Other (rest)")
    others: dict[Any, dict[str, Any]] = {}
    for r in rows:
        if r.get(x) in kept:
            continue
        key = r.get(group) if group else None
        row = others.setdefault(key, dict.fromkeys(r) | {x: label} | ({group: key} if group else {}))
        for f in sum_fields:
            if _is_number(r.get(f)):
                row[f] = (row[f] or 0) + r[f]
    note = (
        f"Top {n_keep} of {len(levels)} categories shown; the remaining {len(levels) - n_keep} "
        f"are grouped as {label}."
    )
    return [*head, *others.values()], note


def _prepare_slope(
    rows: list[dict[str, Any]], x: str, y: str, color: str
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    points = list(dict.fromkeys(r.get(x) for r in rows if r.get(x) is not None))
    if len(points) != 2:
        return rows, None, (
            f"CHART['x'] must have exactly 2 distinct values for a slope chart (found {len(points)}) "
            "— e.g. before/after or two years."
        )
    by_group: dict[Any, dict[Any, float]] = {}
    for r in rows:
        if _is_number(r.get(y)) and r.get(x) in points:
            by_group.setdefault(r.get(color), {})[r.get(x)] = float(r[y])
    change = {g: abs(v[points[1]] - v[points[0]]) for g, v in by_group.items() if len(v) == 2}
    if not change:
        return rows, None, "Every slope group (CHART['color']) needs a numeric y at both x points."
    keep = set(sorted(change, key=lambda g: (-change[g], str(g)))[:_MAX_CATEGORIES])
    note = (
        f"The {len(keep)} groups with the largest change are shown (of {len(change)})."
        if len(change) > len(keep) else None
    )
    return [r for r in rows if r.get(color) in keep], note, None


def _clean_annotations(raw: Any) -> tuple[list[dict[str, Any]], str | None]:
    if raw is None:
        return [], None
    out: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw[:_MAX_ANNOTATIONS]:
            if not isinstance(item, dict) or not _is_number(item.get("y")) or not math.isfinite(item["y"]):
                break
            entry: dict[str, Any] = {"y": item["y"]}
            if isinstance(item.get("label"), str) and item["label"].strip():
                entry["label"] = item["label"].strip()[:60]
            out.append(entry)
        else:
            return out, None
    return [], 'CHART[\'annotations\'] must be a list of up to 3 {"y": number, "label": str} dicts.'


def _extras(spec: dict[str, Any], clean: dict[str, Any], strings: tuple[str, ...]) -> None:
    for key in strings:
        if isinstance(spec.get(key), str) and spec[key].strip():
            clean[key] = spec[key].strip()[:_MAX_TEXT]
    if spec.get("size") in SIZES:
        clean["size"] = spec["size"]
    priority = spec.get("priority")
    if _is_number(priority) and math.isfinite(priority):
        clean["priority"] = max(0, min(_MAX_PRIORITY, round(priority)))


_MAX_ORDER = 100


def _clean_order(raw: Any) -> tuple[list[Any], str | None]:
    if raw is None:
        return [], None
    if isinstance(raw, list) and all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in raw):
        return [_clean_value(v) for v in raw[:_MAX_ORDER]], None
    return [], "CHART['order'] must be a list of category labels (strings or numbers)."


def _distinct(rows: list[dict[str, Any]], field: str) -> int:
    return len({r.get(field) for r in rows if r.get(field) is not None})


def _degenerate(kind: str, rows: list[dict[str, Any]], x: str, y: Any) -> str | None:
    """Deterministic chart critic: a specific, fixable reason when the data
    would render as an empty, single-mark or collapsed chart."""
    if kind in ("histogram", "slope"):
        return None
    nx = _distinct(rows, x)
    if kind == "heatmap":
        ny = _distinct(rows, y)
        wide = "heatmap needs long format: pass y=[col1,col2,...] to dsa.chart.heatmap for a wide table"
        if nx < 2 or ny < 2:
            return f"{wide} (x '{x}' has {nx} and y '{y}' has {ny} distinct values; each axis needs at least 2)."
        for cat, val in ((x, y), (y, x)):
            if (_distinct(rows, cat) == len(rows) and _field_type(rows, cat) != "quantitative"
                    and _field_type(rows, val) == "quantitative"):
                return f"{wide} (every '{cat}' appears once and '{val}' is a measurement, not a category, so no cells are shared)."
        return None
    if nx == 1:
        only = next(r[x] for r in rows if r.get(x) is not None)
        if kind in _BAR_FAMILY or kind == "pareto":
            return (
                f"A {kind} chart of one category ('{only}') compares nothing — chart a field with at least 2 "
                "categories (e.g. group by another column), or state the number in the text."
            )
        return (
            f"CHART['x'] ('{x}') has a single distinct value ({only!r}) so the {kind} chart would collapse to "
            "one position — pass an x field that varies (a date or category), or state the value in the text."
        )
    if kind in ("line", "area", "band") and isinstance(y, str):
        values = {r[y] for r in rows if _is_number(r.get(y))}
        if len(values) == 1:
            return (
                f"CHART['y'] ('{y}') is constant ({next(iter(values))}) in every row, so the {kind} chart is a "
                "flat line — chart a field that varies, or state the constant in the text."
            )
    return None


def validate_chart_spec(spec: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Return (clean_spec, None) or (None, reason). Never raises."""
    if not isinstance(spec, dict):
        return None, "CHART must be a dict."
    chart_type = spec.get("type")
    if chart_type == "vega_lite" or (chart_type is None and "vega_lite" in spec):
        return _validate_vega_lite(spec)
    if chart_type not in CHART_TYPES:
        return None, f"CHART['type'] must be one of {sorted(CHART_TYPES)}."
    data = spec.get("data")
    if not isinstance(data, list) or not data:
        return None, "CHART['data'] must be a non-empty list of records (dicts)."
    if not all(isinstance(r, dict) for r in data):
        return None, "Every CHART['data'] row must be a dict — use df.to_dict('records')."

    fields = set().union(*(r.keys() for r in data))
    avail = f"(available: {sorted(fields)[:12]})"
    x = spec.get("x")
    y: Any = spec.get("y")
    y2 = spec.get("y2") if chart_type == "dual_axis" else None
    target = spec.get("target") if chart_type == "bullet" else None
    facet = spec.get("facet")
    color = spec.get("color")
    if color is None and chart_type in _SERIES_TYPES:
        color = spec.get("series")
    if not isinstance(x, str) or x not in fields:
        return None, f"CHART['x'] must name a field in data {avail}."
    if chart_type != "histogram" and (not isinstance(y, str) or y not in fields):
        return None, f"CHART['y'] must name a field in data {avail}."
    if chart_type == "dual_axis" and (not isinstance(y2, str) or y2 not in fields):
        return None, f"CHART['y2'] must name a field in data for dual_axis {avail}."
    if chart_type == "bullet" and (not isinstance(target, str) or target not in fields):
        return None, f"CHART['target'] must name a field in data for bullet {avail}."
    if chart_type in _SERIES_TYPES | {"slope"} and color is None:
        return None, f"{chart_type} needs a series/group field: pass `series` (or `color`) naming a field in data {avail}."
    if color is not None and (not isinstance(color, str) or color not in fields):
        return None, "CHART['color'] must be null or a field in data."
    if facet is not None:
        if chart_type not in _FACET_TYPES or not isinstance(facet, str) or facet not in fields or facet in (x, y):
            return None, (
                f"CHART['facet'] must be a field in data other than x/y {avail}; "
                f"supported for {sorted(_FACET_TYPES)}."
            )
    bounds = (spec.get("y_lower"), spec.get("y_upper"))
    has_bounds = any(b is not None for b in bounds)
    if (has_bounds or chart_type == "band") and (
        chart_type in _NO_BOUNDS_TYPES or not all(isinstance(b, str) and b in fields for b in bounds)
    ):
        return None, (
            "CHART['y_lower'] and CHART['y_upper'] must both name fields in data "
            "(bar/line/area/band/scatter/dot_ci only; required for band), or both be omitted."
        )
    annotations, annotation_error = _clean_annotations(spec.get("annotations"))
    if annotation_error:
        return None, annotation_error
    order, order_error = _clean_order(spec.get("order")) if chart_type == "heatmap" else ([], None)
    if order_error:
        return None, order_error
    if chart_type == "heatmap" and spec.get("scale") not in (None, "zscore"):
        return None, "CHART['scale'] must be 'zscore' (colour is already standardised) or omitted."
    zscore = chart_type == "heatmap" and spec.get("scale") == "zscore"

    keep = {x} | {f for f in (y, color, y2, target, facet) if isinstance(f, str)}
    keep |= {b for b in bounds if isinstance(b, str)}
    rows = [{k: _clean_value(r.get(k)) for k in keep} for r in data[:MAX_CHART_ROWS]]
    rows = _normalise_dates(rows, x)

    if zscore and (not isinstance(color, str) or _field_type(rows, color) != "quantitative"):
        return None, "scale='zscore' needs CHART['color'] to name the numeric standardised value field."
    if error := _degenerate(chart_type, rows, x, y):
        return None, error

    if chart_type in _NUMERIC_Y_TYPES:
        numeric = (x if chart_type == "lorenz" else None, y, y2, target)
        for field in numeric:
            if isinstance(field, str) and _field_type(rows, field) != "quantitative":
                return None, f"CHART['{field}'] must be numeric for a {chart_type} chart."
    if chart_type == "pareto" and any(_is_number(r.get(y)) and r[y] < 0 for r in rows):
        return None, "A pareto chart needs non-negative additive values in y."
    if chart_type == "boxplot":
        per_group = Counter(r.get(x) for r in rows if _is_number(r.get(y)))
        if max(per_group.values(), default=0) < 4:
            return None, (
                "A boxplot needs the RAW values (>= 4 rows per group in x), not pre-aggregated numbers — "
                "pass the row-level data."
            )

    notes: list[str] = []
    if chart_type == "slope":
        rows, note, error = _prepare_slope(rows, x, str(y), str(color))
        if error:
            return None, error
        if note:
            notes.append(note)
    elif chart_type in _CATEGORY_TYPES | {"boxplot"} and _field_type(rows, x) == "nominal":
        sums = [f for f in (y, target) if isinstance(f, str)]
        additive = chart_type in ("pareto", "bullet") or _is_additive(str(y), spec.get("y_format"))
        rows, note = _limit_categories(
            rows, x, None if chart_type == "boxplot" else y, color, sums,
            aggregate=additive and not has_bounds and chart_type != "boxplot",
        )
        if note:
            notes.append(note)
    if isinstance(facet, str):
        counts = Counter(r.get(facet) for r in rows)
        if len(counts) > _MAX_CATEGORIES:
            top = {k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:_MAX_CATEGORIES]}
            rows = [r for r in rows if r.get(facet) in top]
            notes.append(f"Showing the {_MAX_CATEGORIES} largest of {len(counts)} panels.")

    clean: dict[str, Any] = {"type": chart_type, "data": rows, "x": x}
    if chart_type != "histogram":
        clean["y"] = y
    for key, val in (("y2", y2), ("target", target), ("facet", facet), ("color", color)):
        if val:
            clean[key] = val
    if has_bounds:
        clean["y_lower"], clean["y_upper"] = bounds
    _extras(spec, clean, _STR_KEYS)
    if notes:
        clean["note"] = " ".join(notes)
    elif isinstance(spec.get("note"), str) and spec["note"].strip():
        clean["note"] = spec["note"].strip()[:_MAX_TEXT]
    if spec.get("y_format") in Y_FORMATS:
        clean["y_format"] = spec["y_format"]
    if spec.get("y2_format") in Y_FORMATS:
        clean["y2_format"] = spec["y2_format"]
    if chart_type in _BAR_FAMILY and spec.get("sort") in ("asc", "desc"):
        clean["sort"] = spec["sort"]
    if spec.get("log_y") is True:
        clean["log_y"] = True
    if annotations and chart_type in _ANNOTATE_TYPES:
        clean["annotations"] = annotations
    if zscore:
        clean["scale"] = "zscore"
    if order:
        clean["order"] = order

    clean["truncated"] = len(data) > MAX_CHART_ROWS or spec.get("truncated") is True
    return clean, None


# ---------------------------------------------------------------------------
# Raw Vega-Lite escape hatch. The sanitiser emits the same clean-spec shape as
# a catalog chart ({"type": "vega_lite", "data": rows, ...}) so the dashboard
# consumes both through one path; `spec_to_vegalite` re-attaches the rows.
# ---------------------------------------------------------------------------

#: Keys that can fetch, bind or execute anything — reject, never strip.
_VL_REJECTED_KEYS = frozenset({
    "params", "param", "selection", "signal", "signals", "expr", "test", "url", "href",
    "datasets", "lookup", "calculate", "domainRaw", "data",
})
#: Presentation keys the theme owns — dropped silently.
_VL_DROPPED_KEYS = frozenset({"config", "background", "$schema", "usermeta"})
_VL_TRANSFORMS = frozenset({
    "filter", "aggregate", "joinaggregate", "window", "bin", "timeUnit", "fold", "regression",
    "loess", "stack",
})
_VL_NEEDS_AS = frozenset({"bin", "timeUnit", "regression", "loess", "stack"})
_VL_LIST_AS = frozenset({"aggregate", "joinaggregate", "window"})
_VL_REF_KEYS = frozenset({"groupby", "fold", "regression", "loess", "on"})
_VL_COMPOUND = ("facet", "hconcat", "vconcat", "concat", "repeat")
_VL_MAX_DEPTH = 12
_VL_MAX_NODES = 3000
_VL_MIN_PX, _VL_MAX_PX = 60, 640


class _VlContext:
    def __init__(self) -> None:
        self.refs: set[str] = set()
        self.produced: set[str] = set()
        self.nodes = 0


def _vl_transform(item: Any, ctx: _VlContext) -> None:
    if not isinstance(item, dict) or not (ops := [k for k in item if k in _VL_TRANSFORMS]):
        raise ValueError(f"transform must be one of {sorted(_VL_TRANSFORMS)}")
    op = ops[0]
    if op == "filter" and not isinstance(item["filter"], (dict, list)):
        raise ValueError("string (expression) filters are not allowed — use a predicate object")
    if op in _VL_NEEDS_AS and "as" not in item:
        raise ValueError(f"transform '{op}' needs an explicit 'as'")
    if op in _VL_LIST_AS:
        entries = item[op]
        if not isinstance(entries, list) or not all(isinstance(e, dict) and "as" in e for e in entries):
            raise ValueError(f"every '{op}' entry needs an explicit 'as'")
    if op == "fold" and "as" not in item:
        ctx.produced.update(("key", "value"))
    for key in _VL_REF_KEYS:
        ref = item.get(key)
        if isinstance(ref, str):
            ctx.refs.add(ref)
        elif isinstance(ref, list):
            ctx.refs.update(r for r in ref if isinstance(r, str))


def _vl_copy(node: Any, ctx: _VlContext, depth: int = 0) -> Any:
    ctx.nodes += 1
    if depth > _VL_MAX_DEPTH or ctx.nodes > _VL_MAX_NODES:
        raise ValueError("spec is too large or deeply nested")
    if isinstance(node, list):
        return [_vl_copy(item, ctx, depth + 1) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _VL_DROPPED_KEYS:
            continue
        if key in _VL_REJECTED_KEYS or key.endswith("Expr"):
            raise ValueError(f"key '{key}' is not allowed (it can fetch or execute code)")
        if key == "mark" and (value == "image" or (isinstance(value, dict) and value.get("type") == "image")):
            raise ValueError("image marks are not allowed")
        if key == "filter" and isinstance(value, str):
            raise ValueError("string (expression) filters are not allowed — use a predicate object")
        if key == "transform":
            if not isinstance(value, list):
                raise ValueError("'transform' must be a list")
            for item in value:
                _vl_transform(item, ctx)
        elif key == "field" and isinstance(value, str):
            ctx.refs.add(value)
        elif key == "as":
            ctx.produced.update(a for a in (value if isinstance(value, list) else [value]) if isinstance(a, str))
        if key in ("width", "height") and _is_number(value):
            value = max(_VL_MIN_PX, min(_VL_MAX_PX, value))
        out[key] = _vl_copy(value, ctx, depth + 1)
    return out


def _vl_rect_degenerate(vl: dict[str, Any], rows: list[dict[str, Any]], produced: set[str]) -> str | None:
    """Reject a rect-mark (heatmap) spec whose x or y field has < 2 distinct
    values in the rows — the cells would collapse into a single column/row."""
    layers = vl["layer"] if isinstance(vl.get("layer"), list) else [vl]
    for layer in layers:
        mark = layer.get("mark") if isinstance(layer, dict) else None
        if (mark.get("type") if isinstance(mark, dict) else mark) != "rect":
            continue
        enc = {**(vl.get("encoding") or {}), **(layer.get("encoding") or {})}
        for channel in ("x", "y"):
            field = (enc.get(channel) or {}).get("field") if isinstance(enc.get(channel), dict) else None
            field = field.replace("\\", "") if isinstance(field, str) else None
            if field and field not in produced and any(field in r for r in rows) and _distinct(rows, field) < 2:
                return (
                    f"vega_lite rect mark: '{field}' on {channel} has fewer than 2 distinct values, so every cell "
                    "lands in one " + ("column" if channel == "x" else "row")
                    + " — use long format with >= 2 distinct values on both axes (dsa.chart.heatmap with y=[...] "
                    "melts a wide table)."
                )
    return None


def _validate_vega_lite(spec: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    vl = spec.get("vega_lite")
    if not isinstance(vl, dict):
        return None, "CHART['vega_lite'] must be a Vega-Lite spec dict."
    vl = dict(vl)
    raw_data = vl.pop("data", None)
    if raw_data is None:
        raw_rows = spec.get("data")
    elif isinstance(raw_data, dict) and set(raw_data) == {"values"}:
        raw_rows = raw_data["values"]
    else:
        return None, (
            "vega_lite.data must be exactly {'values': [record dicts]} — url/name/format/datasets "
            "are rejected; inline the rows."
        )
    if not isinstance(raw_rows, list) or not raw_rows or not all(isinstance(r, dict) for r in raw_rows):
        return None, "vega_lite data.values must be a non-empty list of record dicts."
    if not any(k in vl for k in ("mark", "layer", *_VL_COMPOUND)):
        return None, "vega_lite spec needs a mark, layer, or facet/concat."
    compound = any(k in vl for k in _VL_COMPOUND)
    title = vl.pop("title", None)
    vl.pop("width", None)
    vl.pop("autosize", None)

    ctx = _VlContext()
    try:
        clean_vl = _vl_copy(vl, ctx)
    except ValueError as exc:
        return None, f"vega_lite rejected: {exc}."
    fields = {k for r in raw_rows for k in r if isinstance(k, str)}
    missing = sorted(f for f in {r.replace("\\", "") for r in ctx.refs} if f not in fields | ctx.produced)
    if missing:
        return None, f"vega_lite references field(s) {missing} not in the data rows (available: {sorted(fields)[:12]})."

    if not compound and (error := _vl_rect_degenerate(clean_vl, raw_rows, ctx.produced)):
        return None, error

    used = (fields & {r.replace("\\", "") for r in ctx.refs}) or set(sorted(fields)[:8])
    rows = [{k: _clean_value(r.get(k)) for k in used} for r in raw_rows[:MAX_CHART_ROWS]]
    if not compound:
        clean_vl["width"] = "container"
        clean_vl["autosize"] = {"type": "fit", "contains": "padding"}
        clean_vl.setdefault("height", 260)
    clean: dict[str, Any] = {"type": "vega_lite", "data": rows, "vega_lite": clean_vl}
    if isinstance(title, str) and title.strip() and not isinstance(spec.get("title"), str):
        spec = {**spec, "title": title}
    _extras(spec, clean, ("title", "caption"))
    clean["truncated"] = len(raw_rows) > MAX_CHART_ROWS or spec.get("truncated") is True
    return clean, None


# ---------------------------------------------------------------------------
# Cleaned spec -> themed Vega-Lite. Colour is never set here: marks inherit
# the injected chart_theme.vega_config(), and quantitative colour references
# a scheme name only (same convention as dashboard.py).
# ---------------------------------------------------------------------------

#: Scatter points needed before a fitted trend line means anything.
_TREND_MIN_POINTS = 10
#: Bar charts with at least this many categories render horizontally so the
#: labels stay readable.
_HORIZONTAL_BAR_MIN = 9
_HIST_MAXBINS = 30
#: Tooltip format for quantitative fields with no unit: 4 significant digits,
#: never scientific notation.
_PLAIN_NUMBER = {"format": ",.4~r"}
#: Magnitude from which axis labels switch to SI-compact ("1.2M").
_COMPACT_FROM = 10_000
_CUM = "cumulative_share"
_FACET_HEIGHT = 140
_FACET_WIDTH = {1: 480, 2: 260, 3: 170}
_PARETO_LINE = 0.8


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str) or len(value) < 7:
        return False
    try:
        datetime.fromisoformat(value + "-01" if len(value) == 7 else value)
    except ValueError:
        return False
    return True


def _field_type(rows: list[dict[str, Any]], field: str) -> str:
    """'quantitative' (all numbers), 'temporal' (all ISO dates) or 'nominal'."""
    values = [r.get(field) for r in rows if r.get(field) is not None]
    if values and all(_is_number(v) for v in values):
        return "quantitative"
    if values and all(_is_iso_date(v) for v in values):
        return "temporal"
    return "nominal"


def _year_like(rows: list[dict[str, Any]], field: str) -> bool:
    values = [r.get(field) for r in rows if r.get(field) is not None]
    return bool(re.search(r"year|yr", field.lower())) and bool(values) and all(
        _is_number(v) and float(v).is_integer() and 1800 <= v <= 2200 for v in values
    )


def _value_format(rows: list[dict[str, Any]], field: str, y_format: str | None) -> dict[str, str]:
    """Axis/tooltip format for the value field. A "percent" series already on
    a 0-100 scale is not re-multiplied by Vega's `%` format."""
    nums = [r[field] for r in rows if _is_number(r.get(field))]
    if y_format == "percent" and nums and max(abs(v) for v in nums) > 1.5:
        return {"format": ",.1~f"}
    return axis_format(y_format)


def _value_axis(rows: list[dict[str, Any]], field: str, fmt: dict[str, str], y_format: str | None) -> dict[str, Any]:
    """Axis settings for a value field: its unit format, or SI-compact labels
    ("$1.2M", "3.4B") once magnitudes make full digits unreadable."""
    axis: dict[str, Any] = dict(fmt)
    nums = [abs(r[field]) for r in rows if _is_number(r.get(field))]
    if nums and max(nums) >= _COMPACT_FROM and y_format in (None, "currency", "count", "number"):
        spec = "$~s" if y_format == "currency" else "~s"
        axis = {"format": spec, "labelExpr": f"replace(format(datum.value, '{spec}'), 'G', 'B')"}
    return axis


def _narrow_range(rows: list[dict[str, Any]], field: str) -> bool:
    """True when a positive series varies within a small band far above zero,
    where a zero baseline would flatten the line."""
    nums = [r[field] for r in rows if _is_number(r.get(field))]
    return bool(nums) and min(nums) > 0 and (max(nums) - min(nums)) <= 0.5 * max(nums)


def _tip(field: str, field_type: str, title: str, fmt: dict[str, str] | None = None) -> dict[str, Any]:
    tip: dict[str, Any] = {"field": field, "type": field_type, "title": title}
    if field_type == "quantitative":
        tip.update(fmt or _PLAIN_NUMBER)
    return tip


def _labels(rows: list[dict[str, Any]], x: str) -> list[str]:
    return [str(v) for v in dict.fromkeys(r.get(x) for r in rows)]


def _prefers_horizontal(labels: list[str]) -> bool:
    longest = max((len(s) for s in labels), default=0)
    return len(labels) >= _HORIZONTAL_BAR_MIN or (len(labels) >= 4 and longest > 12) or longest > 24


def _cat_axis(labels: list[str], horizontal: bool) -> dict[str, Any]:
    if horizontal:
        return {"labelLimit": 220}
    longest = max((len(s) for s in labels), default=0)
    return {"labelLimit": 140, **({"labelAngle": -40} if len(labels) >= 7 and longest >= 6 else {})}


def _ordered_levels(
    rows: list[dict[str, Any]], x: str, y: str, order: str | None, median: bool = False
) -> list[Any]:
    """Category order by total (or median) of `y`; "Other" always last."""
    groups: dict[Any, list[float]] = {}
    for r in rows:
        bucket = groups.setdefault(r.get(x), [])
        if _is_number(r.get(y)):
            bucket.append(float(r[y]))
    score = {k: statistics.median(v) if median and v else sum(v) for k, v in groups.items()}
    levels = sorted(score, key=lambda k: (score[k], str(k)), reverse=order != "asc")
    return [k for k in levels if k not in _OTHER_LABELS] + [k for k in levels if k in _OTHER_LABELS]


def _datum_color(label: str) -> dict[str, Any]:
    """A string datum gives each series its themed category colour and a legend
    entry naming it."""
    return {"datum": label, "type": "nominal", "legend": {"orient": "top", "title": None}}


def spec_to_vegalite(spec: dict[str, Any]) -> dict[str, Any]:
    """Translate a spec already cleaned by `validate_chart_spec` into a
    Vega-Lite spec (no config — the theme is injected at render time)."""
    if spec["type"] == "vega_lite":
        vl = {**spec["vega_lite"], "data": {"values": spec["data"]}}
        if spec.get("size") == "tall" and _is_number(vl.get("height")):
            vl["height"] = round(vl["height"] * 1.5)
        return vl
    vl = _build(spec)
    if spec.get("annotations"):
        vl = _annotate(vl, spec)
    if spec.get("size") == "tall" and _is_number(vl.get("height")):
        vl["height"] = round(vl["height"] * 1.5)
    return _facet(vl, spec) if spec.get("facet") else vl


def _annotate(vl: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Reference rules (+ labels) on the value axis, each drawn from its own
    one-row dataset so they render once, not once per data row."""
    y = spec["y"]
    layers = vl.get("layer") or [{k: vl[k] for k in ("mark", "encoding", "transform") if k in vl}]
    main = next(
        (lay["encoding"] for lay in layers
         if any((lay.get("encoding") or {}).get(c, {}).get("field") == y for c in ("x", "y"))),
        None,
    )
    if main is None:
        return vl
    channel = "y" if main.get("y", {}).get("field") == y else "x"
    other = "x" if channel == "y" else "y"
    scale = main[channel].get("scale") or {}
    title = main[channel].get("title")
    extra: list[dict[str, Any]] = []
    for note in spec["annotations"]:
        if scale.get("type") == "log" and note["y"] <= 0:
            continue
        own = {"values": [{y: note["y"], "_label": note.get("label", "")}]}
        value = {"field": y, "type": "quantitative", "title": title, **({"scale": scale} if scale else {})}
        extra.append({"data": own, "mark": {"type": "rule", "strokeDash": [4, 3]}, "encoding": {channel: value}})
        if note.get("label"):
            extra.append({
                "data": own,
                "mark": {"type": "text", "align": "left", "baseline": "bottom" if channel == "y" else "top",
                         "dx": 4, "dy": -3 if channel == "y" else 3},
                "encoding": {channel: value, other: {"value": 0}, "text": {"field": "_label", "type": "nominal"}},
            })
    if not extra:
        return vl
    rest = {k: v for k, v in vl.items() if k not in ("mark", "encoding", "transform", "layer")}
    return {**rest, "layer": [*layers, *extra]}


def _facet(vl: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    field = str(spec["facet"])
    counts = Counter(r.get(field) for r in vl["data"]["values"] if r.get(field) is not None)
    levels = [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))]
    columns = 1 if len(levels) == 1 else 2 if len(levels) in (2, 4) else 3
    inner = {k: v for k, v in vl.items() if k not in ("data", "height", "width", "padding")}
    inner["height"] = _FACET_HEIGHT
    inner["width"] = _FACET_WIDTH[columns]
    return {
        "data": vl["data"],
        "facet": {"field": field, "type": "nominal", "title": humanize_label(field), "sort": levels},
        "columns": columns,
        "spec": inner,
    }


#: Numeric heatmap axes with more distinct values than this are binned.
_HEAT_MAX_LEVELS = 25
_HEAT_MAXBINS = 20
_HEAT_CELL_PX = 22
_HEAT_MAX_HEIGHT = 520
_ZSCORE_TITLE = "Std. deviations from that series' mean"
_CALENDAR_NAMES = (
    frozenset("jan feb mar apr may jun jul aug sep oct nov dec".split()),
    frozenset("mon tue wed thu fri sat sun".split()),
)


def _is_calendar(levels: list[Any]) -> bool:
    keys = {str(v).strip().lower()[:3] for v in levels}
    return len(levels) >= 2 and any(keys <= names for names in _CALENDAR_NAMES)


def _heat_axis(
    rows: list[dict[str, Any]], field: str, field_type: str, title: str, channel: str,
    value: str | None, order: list[Any], keep_order: bool,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """(encoding, tooltip, cells drawn) for one heatmap axis: numeric axes with
    many distinct values are binned; nominal ones sort by mean value (count when
    there is no value field) unless order is meaningful; `order` wins."""
    levels = list(dict.fromkeys(r[field] for r in rows if r.get(field) is not None))
    if field_type == "quantitative" and len(levels) > _HEAT_MAX_LEVELS:
        enc = {"field": field, "type": "quantitative", "bin": {"maxbins": _HEAT_MAXBINS}, "title": title}
        return enc, {**enc, **_PLAIN_NUMBER}, _HEAT_MAXBINS
    kind = "nominal" if field_type == "nominal" else "ordinal"
    if kind == "ordinal":
        levels.sort()
    elif not keep_order and not _is_calendar(levels):
        scores: dict[Any, list[float]] = {lv: [] for lv in levels}
        for r in rows:
            if r.get(field) is None:
                continue
            if not value:
                scores[r[field]].append(1.0)
            elif _is_number(r.get(value)):
                scores[r[field]].append(float(r[value]))
        rank = {lv: (sum(v) / len(v) if value and v else sum(v)) for lv, v in scores.items()}
        pos = {lv: i for i, lv in enumerate(levels)}
        levels.sort(key=lambda lv: (-rank[lv], pos[lv]))
    sort: list[Any] | None = levels if kind == "nominal" else None
    by_label = {str(lv): lv for lv in levels}
    head = list(dict.fromkeys(by_label[str(o)] for o in order if str(o) in by_label))
    if head:
        sort = [*head, *(lv for lv in levels if lv not in head)]
    enc = {"field": field, "type": kind, "title": title, **({"sort": sort} if sort else {})}
    if kind == "nominal":
        enc["axis"] = _cat_axis([str(lv) for lv in levels], channel == "y")
    return enc, _tip(field, kind, title), len(levels)


def _heatmap(
    spec: dict[str, Any], rows: list[dict[str, Any]], x_type: str, y_type: str, x_title: str, y_title: str,
) -> dict[str, Any]:
    x, y, color, y_format = spec["x"], spec["y"], spec.get("color"), spec.get("y_format")
    value = color if color and _field_type(rows, color) == "quantitative" else None
    zscore = bool(value) and spec.get("scale") == "zscore"
    nums = [r[value] for r in rows if _is_number(r.get(value))] if value else []
    order = spec.get("order") or []
    # Same labels on both axes = a matrix (e.g. correlations): keep its order.
    square = {str(r.get(x)) for r in rows} == {str(r.get(y)) for r in rows}
    x_enc, x_tip, _ = _heat_axis(rows, x, x_type, x_title, "x", value, order, square)
    y_enc, y_tip, n_rows = _heat_axis(rows, y, y_type, y_title, "y", value, order, square)
    binned = "bin" in x_enc or "bin" in y_enc
    if value:
        title = _ZSCORE_TITLE if zscore else humanize_axis_title(value, y_format)
        value_enc: dict[str, Any] = {
            "aggregate": "mean", "field": value, "type": "quantitative",
            "title": f"Average {title}" if binned and not zscore else title,
        }
        v_fmt = {"format": "+.1f"} if zscore else _value_format(rows, value, y_format)
    else:
        value_enc = {"aggregate": "count", "type": "quantitative", "title": "Rows"}
        v_fmt = axis_format("count")
    if zscore:
        m = min(3.0, max(1.0, float(math.ceil(max(abs(v) for v in nums))))) if nums else 1.0
        scale: dict[str, Any] = {"scheme": "blueorange", "domain": [-m, m], "domainMid": 0, "clamp": True}
    elif nums and min(nums) < 0 < max(nums):
        scale = {"scheme": "blueorange", "domainMid": 0}
    else:
        scale = {"scheme": "oranges"}
    return {
        "data": {"values": rows},
        "mark": {"type": "rect"},
        "height": min(_HEAT_MAX_HEIGHT, max(160, _HEAT_CELL_PX * n_rows)),
        "encoding": {
            "x": x_enc,
            "y": y_enc,
            "color": {
                **value_enc,
                "scale": scale,
                "legend": None if value and len(set(nums)) < 2 else dict(v_fmt),
            },
            "tooltip": [x_tip, y_tip, {**value_enc, **v_fmt}],
        },
    }


def _build(spec: dict[str, Any]) -> dict[str, Any]:
    kind = spec["type"]
    rows = spec["data"]
    x, y, color = spec["x"], spec.get("y"), spec.get("color")
    y_format = spec.get("y_format")
    log = {"type": "log"} if spec.get("log_y") else {}

    x_type = _field_type(rows, x)
    x_title = spec.get("x_title") or humanize_axis_title(x, y_format if kind == "histogram" else None)
    y_title = spec.get("y_title") or (humanize_axis_title(y, y_format) if y else "Rows")
    x_axis: dict[str, Any] = {"format": "d"} if _year_like(rows, x) else {}
    color_enc: dict[str, Any] | None = None
    color_tip: list[dict[str, Any]] = []
    if color:
        color_type = "quantitative" if _field_type(rows, color) == "quantitative" else "nominal"
        color_enc = {"field": color, "type": color_type, "title": humanize_label(color),
                     "legend": {"orient": "top"} if color_type == "quantitative" or _distinct(rows, color) > 1 else None}
        color_tip = [_tip(color, color_type, color_enc["title"])]

    if kind == "histogram":
        x_fmt = _value_format(rows, x, y_format)
        if x_type == "quantitative":
            x_enc: dict[str, Any] = {"field": x, "type": "quantitative", "bin": {"maxbins": _HIST_MAXBINS},
                                     "title": x_title, "axis": _value_axis(rows, x, x_fmt, y_format) or x_axis}
            x_tip = {**_tip(x, "quantitative", x_title, x_fmt), "bin": {"maxbins": _HIST_MAXBINS}}
        else:  # a non-numeric "histogram" is a frequency bar chart
            x_enc = {"field": x, "type": "nominal", "sort": "-y", "title": x_title}
            x_tip = _tip(x, "nominal", x_title)
        encoding: dict[str, Any] = {
            "x": x_enc,
            "y": {"aggregate": "count", "type": "quantitative", "title": "Rows",
                  "axis": axis_format("count"), **({"scale": log} if log else {})},
            "tooltip": [x_tip, {"aggregate": "count", "type": "quantitative", "title": "Rows"}, *color_tip],
        }
        if color_enc:
            encoding["color"] = color_enc
        return {"data": {"values": rows}, "mark": {"type": "bar"}, "height": 240, "encoding": encoding}

    assert y is not None  # validate_chart_spec guarantees y for every non-histogram type
    y_type = _field_type(rows, y)

    if kind == "heatmap":
        return _heatmap(spec, rows, x_type, y_type, x_title, y_title)

    y_fmt = _value_format(rows, y, y_format)
    y_enc: dict[str, Any] = {"field": y, "type": "quantitative" if y_type == "quantitative" else "nominal",
                             "title": y_title}
    if y_enc["type"] == "quantitative":
        y_enc["axis"] = _value_axis(rows, y, y_fmt, y_format)
        if log:
            y_enc["scale"] = dict(log)
    y_tip = _tip(y, y_enc["type"], y_title, y_fmt)
    # Uncertainty bounds only mean something on a quantitative value axis.
    lower, upper = str(spec.get("y_lower") or ""), str(spec.get("y_upper") or "")
    bounded = bool(lower and upper) and y_enc["type"] == "quantitative"
    bound_tips = (
        [_tip(lower, "quantitative", "Lower bound", y_fmt), _tip(upper, "quantitative", "Upper bound", y_fmt)]
        if bounded else []
    )

    def bound_layer(mark: dict[str, Any], base: dict[str, Any], val_ch: str = "y") -> dict[str, Any]:
        """Error-bar/band layer sharing `base`'s non-value channels."""
        enc = {k: v for k, v in base.items() if k not in (val_ch, "tooltip")}
        enc[val_ch] = {"field": lower, "type": "quantitative",
                       **({"scale": base[val_ch]["scale"]} if "scale" in base[val_ch] else {})}
        enc[f"{val_ch}2"] = {"field": upper}
        return {"mark": mark, "encoding": enc}

    if kind in _BAR_FAMILY:
        # Dates and years keep chronological order; categories sort by value.
        cat_type = "ordinal" if x_type == "temporal" or _year_like(rows, x) else x_type
        labels = _labels(rows, x)
        horizontal = cat_type == "nominal" and _prefers_horizontal(labels)
        cat_ch, val_ch = ("y", "x") if horizontal else ("x", "y")
        cat_enc: dict[str, Any] = {"field": x, "type": cat_type, "title": x_title}
        if cat_type == "nominal":
            cat_enc["axis"] = _cat_axis(labels, horizontal)
        elif x_axis:
            cat_enc["axis"] = x_axis
        sort = spec.get("sort") or ("desc" if cat_type == "nominal" else None)
        if sort and cat_type == "nominal":
            cat_enc["sort"] = _ordered_levels(rows, x, y, sort)
        elif sort:
            # A layered chart needs one sort both layers agree on, so sort by
            # the value field itself rather than by a channel.
            cat_enc["sort"] = (
                {"field": y, "op": "sum", "order": "descending" if sort == "desc" else "ascending"}
                if bounded else f"-{val_ch}" if sort == "desc" else val_ch
            )
        if y_enc["type"] == "quantitative" and not log:
            y_enc["scale"] = {"zero": True}
        if kind == "stacked_bar":
            y_enc["stack"] = "zero"
        encoding = {cat_ch: cat_enc, val_ch: y_enc,
                    "tooltip": [_tip(x, cat_type, x_title), y_tip, *bound_tips, *color_tip]}
        if color_enc:
            encoding["color"] = color_enc
            if kind != "stacked_bar":
                encoding[f"{cat_ch}Offset"] = {"field": color}
        series = min(len({r.get(color) for r in rows}), 4) if color and kind != "stacked_bar" else 1
        bar_spec: dict[str, Any] = {
            "data": {"values": rows},
            "height": max(160, 24 * len(labels) * series) if horizontal else 260,
        }
        bars = {"mark": {"type": "bar"}, "encoding": encoding}
        if bounded:
            bar_spec["layer"] = [bars, bound_layer({"type": "rule"}, encoding, val_ch)]
        else:
            bar_spec.update(bars)
        return bar_spec

    if kind == "boxplot":
        labels = _labels(rows, x)
        horizontal = _prefers_horizontal(labels)
        cat_ch, val_ch = ("y", "x") if horizontal else ("x", "y")
        y_enc["scale"] = {**y_enc.get("scale", {}), "zero": False}
        cat_enc = {"field": x, "type": "nominal", "title": x_title, "axis": _cat_axis(labels, horizontal),
                   "sort": _ordered_levels(rows, x, y, "desc", median=True)}
        return {
            "data": {"values": rows},
            "height": max(160, 30 * len(labels)) if horizontal else 280,
            "mark": {"type": "boxplot", "extent": 1.5},
            "encoding": {cat_ch: cat_enc, val_ch: y_enc},
        }

    if kind == "pareto":
        levels = _ordered_levels(rows, x, y, "desc")
        totals = dict.fromkeys(levels, 0.0)
        for r in rows:
            if _is_number(r.get(y)):
                totals[r.get(x)] += float(r[y])
        grand = sum(totals.values()) or 1.0
        running = 0.0
        pareto_rows: list[dict[str, Any]] = []
        for k in levels:
            running += totals[k]
            pareto_rows.append({x: k, y: totals[k], _CUM: running / grand})
        labels = [str(k) for k in levels]
        share_title = "Cumulative share"
        cat_enc = {"field": x, "type": "nominal", "title": x_title, "sort": levels,
                   "axis": _cat_axis(labels, False)}
        share_scale = {"domain": [0, 1]}
        return {
            "data": {"values": pareto_rows},
            "height": 280,
            "layer": [
                {"mark": {"type": "bar"},
                 "encoding": {"x": cat_enc, "y": {**y_enc, "scale": {"zero": True}}, "color": _datum_color(y_title),
                              "tooltip": [_tip(x, "nominal", x_title), y_tip,
                                          _tip(_CUM, "quantitative", share_title, {"format": ".0%"})]}},
                {"mark": {"type": "line", "point": True},
                 "encoding": {"x": cat_enc,
                              "y": {"field": _CUM, "type": "quantitative", "title": share_title,
                                    "scale": share_scale, "axis": {"orient": "right", "format": ".0%"}},
                              "color": _datum_color(share_title)}},
                {"data": {"values": [{_CUM: _PARETO_LINE}]}, "mark": {"type": "rule", "strokeDash": [4, 3]},
                 "encoding": {"y": {"field": _CUM, "type": "quantitative", "scale": share_scale, "axis": None}}},
            ],
            "resolve": {"scale": {"y": "independent"}},
        }

    if kind == "slope":
        stops = list(dict.fromkeys(r.get(x) for r in rows if r.get(x) is not None))
        if x_type in ("quantitative", "temporal"):
            stops.sort()
        y_enc["scale"] = {**y_enc.get("scale", {}), "zero": not _narrow_range(rows, y)}
        base = {
            "x": {"field": x, "type": "ordinal", "title": x_title, "sort": stops, "scale": {"padding": 0.5}},
            "y": y_enc,
            "color": {"field": color, "type": "nominal", "legend": None, "title": humanize_label(str(color))},
        }
        tips = [_tip(str(color), "nominal", humanize_label(str(color))), _tip(x, "ordinal", x_title), y_tip]
        return {
            "data": {"values": rows},
            "height": 320,
            "padding": {"left": 5, "top": 5, "bottom": 5, "right": 120},
            "layer": [
                {"mark": {"type": "line"}, "encoding": {**base, "tooltip": tips}},
                {"mark": {"type": "circle", "size": 60}, "encoding": {**base, "tooltip": tips}},
                {"mark": {"type": "text", "align": "left", "dx": 8},
                 "transform": [{"filter": {"field": x, "equal": stops[-1]}}],
                 "encoding": {**base, "text": {"field": color, "type": "nominal"}}},
            ],
        }

    if kind == "bullet":
        target = str(spec["target"])
        levels = _ordered_levels(rows, x, y, "desc")
        axis = _value_axis(rows, y, y_fmt, y_format)

        def measure(field: str) -> dict[str, Any]:
            return {"field": field, "type": "quantitative", "title": y_title, "axis": axis,
                    "scale": {"zero": True}}

        cat_enc = {"field": x, "type": "nominal", "title": x_title, "sort": levels,
                   "axis": {"labelLimit": 220}}
        tips = [_tip(x, "nominal", x_title), _tip(y, "quantitative", "Actual", y_fmt),
                _tip(target, "quantitative", "Target", y_fmt)]
        return {
            "data": {"values": rows},
            "height": max(120, 34 * len(levels)),
            "layer": [
                {"mark": {"type": "bar", "size": 14},
                 "encoding": {"y": cat_enc, "x": measure(y), "color": _datum_color("Actual"), "tooltip": tips}},
                {"mark": {"type": "tick", "thickness": 3, "size": 26},
                 "encoding": {"y": cat_enc, "x": measure(target), "color": _datum_color("Target"), "tooltip": tips}},
            ],
        }

    if kind == "waterfall":
        field = json.dumps(y)
        # Colour comes from the themed category range; the padded domain puts
        # "Decrease" on its risk slot and "Increase" on its positive slot.
        wf_spec: dict[str, Any] = {
            "data": {"values": rows},
            "height": 260,
            "transform": [
                {"window": [{"op": "sum", "field": y, "as": "running_total"}]},
                {"calculate": f"datum.running_total - datum[{field}]", "as": "bar_start"},
                {"calculate": "min(datum.bar_start, datum.running_total)", "as": "bar_bottom"},
                {"calculate": "max(datum.bar_start, datum.running_total)", "as": "bar_top"},
                {"calculate": f"datum[{field}] < 0 ? 'Decrease' : 'Increase'", "as": "direction"},
            ],
            "mark": {"type": "bar"},
            "encoding": {
                "x": {"field": x, "type": "ordinal", "title": x_title, "sort": None},
                "y": {"field": "bar_bottom", "type": "quantitative", "title": y_title,
                      "axis": _value_axis(rows, y, y_fmt, y_format)},
                "y2": {"field": "bar_top"},
                "color": {
                    "field": "direction", "type": "nominal", "legend": None,
                    "scale": {"domain": ["_", "Decrease", "__", "Increase"]},
                },
                "tooltip": [
                    _tip(x, "ordinal", x_title),
                    _tip(y, "quantitative", "Change", y_fmt),
                    {"field": "running_total", "type": "quantitative", "title": "Running total", **y_fmt},
                ],
            },
        }
        return wf_spec

    if kind == "lorenz":
        end_x = max((r[x] for r in rows if _is_number(r.get(x))), default=1)
        end_y = max((r[y] for r in rows if _is_number(r.get(y))), default=1)
        equality = {
            "mark": {"type": "rule", "strokeDash": [4, 4]},
            "encoding": {
                "x": {"datum": 0}, "y": {"datum": 0},
                "x2": {"datum": end_x}, "y2": {"datum": end_y},
            },
        }
        curve = {
            "mark": {"type": "line"},
            "encoding": {
                "x": {"field": x, "type": "quantitative", "title": x_title, "axis": dict(y_fmt)},
                "y": {"field": y, "type": "quantitative", "title": y_title, "axis": dict(y_fmt)},
                "tooltip": [_tip(x, "quantitative", x_title, y_fmt), _tip(y, "quantitative", y_title, y_fmt)],
            },
        }
        return {"data": {"values": rows}, "height": 260, "layer": [equality, curve]}

    if kind == "dot_ci":
        cat_type = "ordinal" if x_type == "temporal" else x_type
        base_enc: dict[str, Any] = {"x": {"field": x, "type": cat_type, "title": x_title}, "y": y_enc,
                                    "tooltip": [_tip(x, cat_type, x_title), y_tip, *bound_tips, *color_tip]}
        if cat_type == "nominal":
            base_enc["x"]["sort"] = None  # producer's row order (usually already ranked)
            base_enc["x"]["axis"] = _cat_axis(_labels(rows, x), False)
        if color_enc:
            base_enc["color"] = color_enc
            base_enc["xOffset"] = {"field": color}
        ci_layers: list[dict[str, Any]] = []
        if bounded:
            ci_layers.append(bound_layer({"type": "rule"}, base_enc))
        ci_layers.append({"mark": {"type": "circle", "size": 60}, "encoding": base_enc})
        return {"data": {"values": rows}, "height": 260, "layer": ci_layers}

    if kind == "dual_axis":
        y2 = str(spec["y2"])
        y2_title = spec.get("y2_title") or humanize_axis_title(y2, spec.get("y2_format"))
        y2_fmt = _value_format(rows, y2, spec.get("y2_format"))
        ds_x_type = "ordinal" if x_type == "nominal" else x_type
        x_enc = {"field": x, "type": ds_x_type, "title": x_title}
        if ds_x_type == "ordinal":
            x_enc["sort"] = None

        def series_layer(field: str, title: str, y_channel: dict[str, Any], mark: dict[str, Any]) -> dict[str, Any]:
            return {
                "mark": mark,
                "encoding": {
                    "x": x_enc,
                    "y": y_channel,
                    "color": _datum_color(title),
                    "tooltip": [_tip(x, ds_x_type, x_title), _tip(field, "quantitative", title, y_fmt if field == y else y2_fmt)],
                },
            }

        return {
            "data": {"values": rows},
            "height": 260,
            "layer": [
                series_layer(y, y_title, y_enc, {"type": "line"}),
                series_layer(
                    y2, y2_title,
                    {"field": y2, "type": "quantitative", "title": y2_title,
                     "axis": {"orient": "right", **_value_axis(rows, y2, y2_fmt, spec.get("y2_format"))}},
                    {"type": "line", "strokeDash": [5, 3]},
                ),
            ],
            "resolve": {"scale": {"y": "independent"}},
        }

    if kind in ("line", "area", "band"):
        # A non-date, non-numeric x keeps the producer's row order.
        line_x_type = "ordinal" if x_type == "nominal" else x_type
        x_enc = {"field": x, "type": line_x_type, "title": x_title}
        if line_x_type == "ordinal":
            x_enc["sort"] = None
        elif x_axis:
            x_enc["axis"] = x_axis
        if kind != "area" and y_enc["type"] == "quantitative" and _narrow_range(rows, y):
            y_enc["scale"] = {**y_enc.get("scale", {}), "zero": False}
        encoding = {"x": x_enc, "y": y_enc,
                    "tooltip": [_tip(x, line_x_type, x_title), y_tip, *bound_tips, *color_tip]}
        if color_enc:
            encoding["color"] = color_enc
        mark: dict[str, Any] = (
            {"type": "area", "line": True} if kind == "area" else {"type": "line", "point": len(rows) <= 60}
        )
        if bounded:
            band = bound_layer({"type": "area", "opacity": 0.2}, encoding)
            return {"data": {"values": rows}, "height": 260,
                    "layer": [band, {"mark": mark, "encoding": encoding}]}
        return {"data": {"values": rows}, "mark": mark, "height": 260, "encoding": encoding}

    # scatter
    x_enc = {"field": x, "type": "quantitative" if x_type == "quantitative" else "nominal", "title": x_title}
    if x_enc["type"] == "quantitative":
        x_enc["scale"] = {"zero": False}
        if x_axis:
            x_enc["axis"] = x_axis
        if y_enc["type"] == "quantitative":
            y_enc["scale"] = {**y_enc.get("scale", {}), "zero": False}
    encoding = {"x": x_enc, "y": y_enc,
                "tooltip": [_tip(x, x_enc["type"], x_title), y_tip, *bound_tips, *color_tip]}
    if color_enc:
        encoding["color"] = color_enc
    opacity, size = (0.6, 40) if len(rows) <= 150 else (0.35, 28) if len(rows) <= 300 else (0.25, 20)
    points = {"mark": {"type": "circle", "opacity": opacity, "size": size}, "encoding": encoding}
    layers: list[dict[str, Any]] = [points]
    if bounded:
        layers.append(bound_layer({"type": "rule"}, encoding))
    if x_enc["type"] == y_enc["type"] == "quantitative" and len(rows) >= _TREND_MIN_POINTS:
        layers.append({
            "mark": {"type": "line", "strokeDash": [4, 3]},
            "transform": [{"regression": y, "on": x}],
            "encoding": {"x": {"field": x, "type": "quantitative"}, "y": {"field": y, "type": "quantitative"}},
        })
    if len(layers) > 1:
        return {"data": {"values": rows}, "height": 280, "layer": layers}
    return {"data": {"values": rows}, "height": 280, **points}
