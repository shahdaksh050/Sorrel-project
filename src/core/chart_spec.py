"""
Declarative chart contract shared by every chart producer that is not the
deterministic dashboard builder itself: LLM-authored sandbox code (`CHART =
{...}` / `dsa.chart.*`), generated tools, and tool `Finding.chart_hint`s.

Producers never emit Vega-Lite. They emit this small, whitelisted spec;
`validate_chart_spec` cleans it (or rejects it with an actionable message the
LLM can fix), and the dashboard translates the cleaned spec into themed
Vega-Lite. Keeping LLM output declarative means a bad spec degrades to "no
chart", never to broken or injected rendering code.

Spec shape (keys not listed here are dropped):
    type      "bar" | "line" | "area" | "scatter" | "histogram" | "heatmap"   (required)
    data      list of flat records (dicts of str/number/bool/None)          (required)
    x         field name in `data`                                          (required)
    y         field name in `data` (required except for histogram)
    color     optional field name — series (line/area/scatter) or group (bar)
    title     short human title
    x_title / y_title   axis titles (default: humanised field names)
    y_format  "currency" | "percent" | "count" | "number"
              ("percent" expects 0-1 fractions; a 0-100 series is shown as-is)
    sort      "desc" | "asc"  (bar only; default "desc" for a categorical x)
    log_y     bool — log scale on y (positive values only)
    caption   one-sentence takeaway shown under the chart
"""
from __future__ import annotations

import importlib
import math
from datetime import datetime
from typing import Any

import src.core.chart_theme

# In long-running processes (e.g. Streamlit runner), chart_theme may have been
# imported before humanize_label was defined. Reload defensively if stale.
if not hasattr(src.core.chart_theme, "humanize_label"):
    importlib.reload(src.core.chart_theme)

from src.core.chart_theme import axis_format, humanize_axis_title, humanize_label

CHART_TYPES: frozenset[str] = frozenset(
    {"bar", "line", "area", "scatter", "histogram", "heatmap"}
)
Y_FORMATS: frozenset[str] = frozenset({"currency", "percent", "count", "number"})

#: Rows inlined per chart. Keeps dashboard.json and the self-contained HTML
#: report small; aggregate before charting instead of plotting raw rows.
MAX_CHART_ROWS = 500

_STR_KEYS = ("title", "x_title", "y_title", "caption")
_MAX_TEXT = 200


def _clean_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value if not isinstance(value, str) else value[:_MAX_TEXT]
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and not math.isfinite(value) else value
    if hasattr(value, "item"):  # numpy scalar that slipped through
        return _clean_value(value.item())
    return str(value)[:_MAX_TEXT]


def validate_chart_spec(spec: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Return (clean_spec, None) or (None, reason). Never raises."""
    if not isinstance(spec, dict):
        return None, "CHART must be a dict."
    chart_type = spec.get("type")
    if chart_type not in CHART_TYPES:
        return None, f"CHART['type'] must be one of {sorted(CHART_TYPES)}."
    data = spec.get("data")
    if not isinstance(data, list) or not data:
        return None, "CHART['data'] must be a non-empty list of records (dicts)."
    if not all(isinstance(r, dict) for r in data):
        return None, "Every CHART['data'] row must be a dict — use df.to_dict('records')."

    fields = set().union(*(r.keys() for r in data))
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    if not isinstance(x, str) or x not in fields:
        return None, f"CHART['x'] must name a field in data (available: {sorted(fields)[:12]})."
    if chart_type != "histogram" and (not isinstance(y, str) or y not in fields):
        return None, f"CHART['y'] must name a field in data (available: {sorted(fields)[:12]})."
    if color is not None and (not isinstance(color, str) or color not in fields):
        return None, "CHART['color'] must be null or a field in data."

    keep = {x} | ({y} if isinstance(y, str) else set()) | ({color} if color else set())
    rows = [{k: _clean_value(r.get(k)) for k in keep} for r in data[:MAX_CHART_ROWS]]

    clean: dict[str, Any] = {"type": chart_type, "data": rows, "x": x}
    if chart_type != "histogram":
        clean["y"] = y
    if color:
        clean["color"] = color
    for key in _STR_KEYS:
        if isinstance(spec.get(key), str) and spec[key].strip():
            clean[key] = spec[key].strip()[:_MAX_TEXT]
    if spec.get("y_format") in Y_FORMATS:
        clean["y_format"] = spec["y_format"]
    if chart_type == "bar" and spec.get("sort") in ("asc", "desc"):
        clean["sort"] = spec["sort"]
    if spec.get("log_y") is True:
        clean["log_y"] = True
    clean["truncated"] = len(data) > MAX_CHART_ROWS
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


def _is_number(value: Any) -> bool:
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


def _value_format(rows: list[dict[str, Any]], field: str, y_format: str | None) -> dict[str, str]:
    """Axis/tooltip format for the value field. A "percent" series already on
    a 0-100 scale is not re-multiplied by Vega's `%` format."""
    nums = [r[field] for r in rows if _is_number(r.get(field))]
    if y_format == "percent" and nums and max(abs(v) for v in nums) > 1.5:
        return {"format": ",.1~f"}
    return axis_format(y_format)


def _tip(field: str, field_type: str, title: str, fmt: dict[str, str] | None = None) -> dict[str, Any]:
    tip: dict[str, Any] = {"field": field, "type": field_type, "title": title}
    if field_type == "quantitative":
        tip.update(fmt or _PLAIN_NUMBER)
    return tip


def spec_to_vegalite(spec: dict[str, Any]) -> dict[str, Any]:
    """Translate a spec already cleaned by `validate_chart_spec` into a
    Vega-Lite spec (no config — the theme is injected at render time)."""
    kind = spec["type"]
    rows = spec["data"]
    x, y, color = spec["x"], spec.get("y"), spec.get("color")
    y_format = spec.get("y_format")
    log = {"type": "log"} if spec.get("log_y") else {}

    x_type = _field_type(rows, x)
    x_title = spec.get("x_title") or humanize_axis_title(x, y_format if kind == "histogram" else None)
    y_title = spec.get("y_title") or (humanize_axis_title(y, y_format) if y else "Rows")
    color_enc: dict[str, Any] | None = None
    color_tip: list[dict[str, Any]] = []
    if color:
        color_type = "quantitative" if _field_type(rows, color) == "quantitative" else "nominal"
        color_enc = {"field": color, "type": color_type, "title": humanize_label(color),
                     "legend": {"orient": "top"}}
        color_tip = [_tip(color, color_type, color_enc["title"])]

    if kind == "histogram":
        x_fmt = _value_format(rows, x, y_format)
        if x_type == "quantitative":
            x_enc: dict[str, Any] = {"field": x, "type": "quantitative", "bin": {"maxbins": _HIST_MAXBINS},
                                     "title": x_title, "axis": dict(x_fmt)}
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
        value = color if color_enc and color_enc["type"] == "quantitative" else None
        v_fmt = _value_format(rows, value, y_format) if value else axis_format("count")
        nums = [r[value] for r in rows if _is_number(r.get(value))] if value else []
        diverging = bool(nums) and min(nums) < 0 < max(nums)
        value_enc: dict[str, Any] = (
            {"field": value, "type": "quantitative", "title": humanize_axis_title(value, y_format)}
            if value else {"aggregate": "count", "type": "quantitative", "title": "Rows"}
        )
        hx = "nominal" if x_type == "nominal" else "ordinal"
        hy = "nominal" if y_type == "nominal" else "ordinal"
        return {
            "data": {"values": rows},
            "mark": {"type": "rect"},
            "height": max(160, 22 * len({r.get(y) for r in rows})),
            "encoding": {
                "x": {"field": x, "type": hx, "title": x_title},
                "y": {"field": y, "type": hy, "title": y_title},
                "color": {
                    **value_enc,
                    "scale": {"scheme": "blueorange", "domainMid": 0} if diverging else {"scheme": "oranges"},
                    "legend": dict(v_fmt),
                },
                "tooltip": [_tip(x, hx, x_title), _tip(y, hy, y_title), {**value_enc, **v_fmt}],
            },
        }

    y_fmt = _value_format(rows, y, y_format)
    y_enc: dict[str, Any] = {"field": y, "type": "quantitative" if y_type == "quantitative" else "nominal",
                             "title": y_title}
    if y_enc["type"] == "quantitative":
        y_enc["axis"] = dict(y_fmt)
        if log:
            y_enc["scale"] = dict(log)
    y_tip = _tip(y, y_enc["type"], y_title, y_fmt)

    if kind == "bar":
        # Dates keep chronological order; categories sort by value.
        cat_type = "ordinal" if x_type == "temporal" else x_type
        n_levels = len({r.get(x) for r in rows})
        horizontal = cat_type == "nominal" and n_levels >= _HORIZONTAL_BAR_MIN
        cat_ch, val_ch = ("y", "x") if horizontal else ("x", "y")
        cat_enc: dict[str, Any] = {"field": x, "type": cat_type, "title": x_title}
        sort = spec.get("sort") or ("desc" if cat_type == "nominal" else None)
        if sort:
            cat_enc["sort"] = f"-{val_ch}" if sort == "desc" else val_ch
        encoding = {cat_ch: cat_enc, val_ch: y_enc,
                    "tooltip": [_tip(x, cat_type, x_title), y_tip, *color_tip]}
        if color_enc:
            encoding["color"] = color_enc
            encoding[f"{cat_ch}Offset"] = {"field": color}
        return {
            "data": {"values": rows},
            "mark": {"type": "bar"},
            "height": max(160, 24 * n_levels) if horizontal else 260,
            "encoding": encoding,
        }

    if kind in ("line", "area"):
        # A non-date, non-numeric x keeps the producer's row order.
        line_x_type = "ordinal" if x_type == "nominal" else x_type
        x_enc = {"field": x, "type": line_x_type, "title": x_title}
        if line_x_type == "ordinal":
            x_enc["sort"] = None
        if kind == "line" and y_enc["type"] == "quantitative":
            y_enc["scale"] = {**y_enc.get("scale", {}), "zero": False}
        encoding = {"x": x_enc, "y": y_enc,
                    "tooltip": [_tip(x, line_x_type, x_title), y_tip, *color_tip]}
        if color_enc:
            encoding["color"] = color_enc
        mark: dict[str, Any] = (
            {"type": "line", "point": len(rows) <= 60} if kind == "line" else {"type": "area", "line": True}
        )
        return {"data": {"values": rows}, "mark": mark, "height": 260, "encoding": encoding}

    # scatter
    x_enc = {"field": x, "type": "quantitative" if x_type == "quantitative" else "nominal", "title": x_title}
    if x_enc["type"] == "quantitative":
        x_enc["scale"] = {"zero": False}
        if y_enc["type"] == "quantitative":
            y_enc["scale"] = {**y_enc.get("scale", {}), "zero": False}
    encoding = {"x": x_enc, "y": y_enc,
                "tooltip": [_tip(x, x_enc["type"], x_title), y_tip, *color_tip]}
    if color_enc:
        encoding["color"] = color_enc
    points = {"mark": {"type": "circle", "opacity": 0.6, "size": 40}, "encoding": encoding}
    if x_enc["type"] == y_enc["type"] == "quantitative" and len(rows) >= _TREND_MIN_POINTS:
        trend = {
            "mark": {"type": "line", "strokeDash": [4, 3]},
            "transform": [{"regression": y, "on": x}],
            "encoding": {"x": {"field": x, "type": "quantitative"}, "y": {"field": y, "type": "quantitative"}},
        }
        return {"data": {"values": rows}, "height": 280, "layer": [points, trend]}
    return {"data": {"values": rows}, "height": 280, **points}
