"""
LLM-designed charts without LLM-written code.

The model proposes declarative chart *recipes* (type, columns, aggregation,
filters); this module executes them deterministically on the FULL dataframe so
every plotted number is exact, and returns raw chart specs in the shape
`chart_spec.validate_chart_spec` accepts (which then applies its own layout
rules). A recipe that names a missing column, is degenerate, or cannot run
yields None — never an exception.

Recipe keys: type, x, y, agg, series, time_grain, filter, top_n, sort, title,
caption, y_format, finding_id; heatmap also takes `value` (the numeric column
filling each x/y cell, aggregated with `agg`).
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from src.core.chart_spec import MAX_CHART_ROWS, Y_FORMATS

#: Safety ceilings against a runaway reply, not targets: the LLM decides how
#: many charts the findings warrant.
_MAX_RECIPES = 16
_MAX_DROPS = 16
_MAX_TOP_N = 15
_DEFAULT_TOP_N = 12
_MAX_CATEGORIES = 12  # chart_spec folds/drops bar-family categories beyond this
_MAX_SERIES = 8
_MAX_LEVELS = 200
_SCATTER_ROWS = 350
_BOX_GROUPS = 12
_BOX_MIN_ROWS = 4
_MAX_FILTERS = 3
_MAX_TEXT = 200

_KINDS = frozenset({
    "bar", "line", "area", "scatter", "heatmap", "histogram", "boxplot", "stacked_bar", "grouped_bar", "pareto",
})
_SERIES_KINDS = frozenset({"bar", "line", "area", "scatter", "stacked_bar", "grouped_bar"})
_BAR_KINDS = frozenset({"bar", "stacked_bar", "grouped_bar", "pareto"})
_AGGS = frozenset({"mean", "sum", "median", "count", "nunique", "min", "max"})
_ADDITIVE = frozenset({"sum", "count"})
_GRAINS = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
_ORDER_OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}
_EQ_OPS = {"==": operator.eq, "!=": operator.ne}


@dataclass
class ChartDesign:
    charts: list[dict[str, Any]] = field(default_factory=list)
    drop_ids: list[str] = field(default_factory=list)


def design_from_reply(df: pd.DataFrame, reply: dict[str, Any]) -> ChartDesign:
    """Run the recipes (up to _MAX_RECIPES) of an LLM reply {"charts": [...], "drop": [...]}."""
    try:
        recipes = reply.get("charts")
        drops = reply.get("drop")
        charts = [
            spec for recipe in (recipes[:_MAX_RECIPES] if isinstance(recipes, list) else [])
            if (spec := run_recipe(df, recipe)) is not None
        ]
        ids = [d.strip() for d in (drops[:_MAX_DROPS] if isinstance(drops, list) else []) if isinstance(d, str)]
        return ChartDesign(charts, list(dict.fromkeys(i for i in ids if i)))
    except Exception:
        return ChartDesign()


def run_recipe(df: pd.DataFrame, recipe: dict[str, Any]) -> dict[str, Any] | None:
    """Execute one recipe into a raw chart spec, or None when invalid/unrunnable."""
    try:
        return _run(df, recipe)
    except Exception:
        return None


# ---------------------------------------------------------------------------


def _py(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        return round(v, 4)
    return v if isinstance(v, (bool, int, str)) else str(v)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: _py(v) for k, v in row.items()} for row in frame.to_dict("records")]


def _text(value: Any) -> str | None:
    return value.strip()[:_MAX_TEXT] or None if isinstance(value, str) else None


def _is_num(s: pd.Series) -> bool:
    return bool(pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s))


def _is_time(s: pd.Series) -> bool:
    return bool(pdt.is_datetime64_any_dtype(s))


def _int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce(s: pd.Series, v: Any) -> Any:
    if not isinstance(v, (str, int, float, bool)):
        raise ValueError("filter value")
    if pdt.is_bool_dtype(s):
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            return v.strip().lower() == "true"
        raise ValueError("filter value")
    if _is_num(s):
        if isinstance(v, bool):
            raise ValueError("filter value")
        return float(v)
    if _is_time(s):
        return pd.Timestamp(v)
    return str(v)


def _mask(s: pd.Series, op: Any, value: Any) -> pd.Series:
    ordered = _is_num(s) or _is_time(s)
    cmp = s if ordered or pdt.is_bool_dtype(s) else s.astype(str)
    if op == "in":
        vals = value if isinstance(value, list) else [value]
        if not 0 < len(vals) <= 50:
            raise ValueError("filter values")
        return cmp.isin([_coerce(s, v) for v in vals])
    if op in _EQ_OPS:
        return _EQ_OPS[op](cmp, _coerce(s, value))
    if op in _ORDER_OPS and ordered:
        return _ORDER_OPS[op](cmp, _coerce(s, value))
    raise ValueError("filter op")


def _filtered(df: pd.DataFrame, filters: Any) -> pd.DataFrame:
    if filters is None:
        return df
    if not isinstance(filters, list) or len(filters) > _MAX_FILTERS:
        raise ValueError("filter")
    for f in filters:
        if not isinstance(f, dict) or not isinstance(f.get("col"), str) or f["col"] not in df.columns:
            raise ValueError("filter column")
        df = df[_mask(df[f["col"]], f.get("op"), f.get("value"))]
    return df


def _bucket(s: pd.Series, grain: str) -> pd.Series:
    return s.dt.to_period(_GRAINS[grain]).dt.start_time


def _axis(w: pd.DataFrame, col: str, grain: Any, cap: int) -> None:
    """Datetime axes are bucketed to `grain` (auto-picked so <= `cap` periods
    when not given); other non-numeric axes become strings. In place."""
    s = w[col]
    if _is_time(s):
        if s.dt.tz is not None:
            s = s.dt.tz_localize(None)
        if grain not in _GRAINS:
            grain = next((g for g in _GRAINS if _bucket(s, g).nunique() <= cap), "year")
        w[col] = _bucket(s, grain)
    elif not _is_num(s):
        w[col] = s.astype(str)


def _fold(g: pd.DataFrame, col: str, val: str, cap: int, additive: bool, keys: list[str]) -> pd.DataFrame:
    """Keep the largest `cap` levels of `col` (by |value|); the tail is summed
    into "Other" when additive, otherwise dropped."""
    if g[col].nunique() <= cap:
        return g
    score = g.groupby(col, observed=True)[val].agg("sum" if additive else "max").abs()
    kept = set(score.sort_values(ascending=False, kind="stable").index[: cap - 1 if additive else cap])
    if not additive:
        return g[g[col].isin(kept)]
    label = next((lab for lab in ("Other", "Other (rest)") if lab not in kept), "Other (rest)")
    g = g.assign(**{col: g[col].where(g[col].isin(kept), label)})
    return g.groupby(keys, sort=False, observed=True)[val].sum().reset_index()


def _run(df: pd.DataFrame, r: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(df, pd.DataFrame) or not isinstance(r, dict) or not df.columns.is_unique:
        return None
    kind = r.get("type")
    if kind not in _KINDS:
        return None
    x, y, series, value = (r.get(k) or None for k in ("x", "y", "series", "value"))
    named = [c for c in (x, y, series, value) if c is not None]
    if x is None or any(not isinstance(c, str) or c not in df.columns for c in named) or len(set(named)) != len(named):
        return None
    if kind not in _SERIES_KINDS:
        series = None
    if kind in ("stacked_bar", "grouped_bar") and series is None:
        return None
    if kind in ("scatter", "boxplot", "heatmap") and y is None:
        return None
    if kind == "histogram":
        y = None
    d = _filtered(df, r.get("filter"))
    if kind == "histogram":
        out = _histogram(d, x)
    elif kind == "scatter":
        out = _scatter(d, x, str(y), series)
    elif kind == "boxplot":
        out = _boxplot(d, x, str(y), r.get("time_grain"))
    else:
        out = _aggregate(d, kind, r, x, y, series, value)
    if out is None:
        return None
    rows, y_name, color, agg = out
    spec: dict[str, Any] = {"type": kind, "x": x, "data": rows}
    if y_name:
        spec["y"] = y_name
    if color:
        spec["color"] = color
    if series:
        spec["color"] = series
        if kind in ("stacked_bar", "grouped_bar"):
            spec["series"] = series
    if kind in _BAR_KINDS and r.get("sort") == "value":
        spec["sort"] = "desc"
    y_format = r.get("y_format")
    if y_format not in Y_FORMATS:
        y_format = "count" if agg in ("count", "nunique") else None
    for key, val in (("title", _text(r.get("title"))), ("caption", _text(r.get("caption"))), ("y_format", y_format)):
        if val:
            spec[key] = val
    fid = r.get("finding_id")
    spec["finding_id"] = fid.strip()[:80] or None if isinstance(fid, str) else None
    return spec


_Out = tuple[list[dict[str, Any]], str | None, str | None, str | None] | None  # rows, y, color, agg


def _histogram(d: pd.DataFrame, x: str) -> _Out:
    if not _is_num(d[x]):
        return None
    s = d[x].dropna()
    if len(s) < 2 or s.nunique() < 2:
        return None
    if len(s) > MAX_CHART_ROWS:
        s = s.sample(MAX_CHART_ROWS, random_state=0)
    return [{x: _py(v)} for v in s], None, None, None


def _scatter(d: pd.DataFrame, x: str, y: str, series: str | None) -> _Out:
    if not _is_num(d[y]) or not (_is_num(d[x]) or _is_time(d[x])):
        return None
    w = d[[x, y, *([series] if series else [])]].dropna()
    if series:
        w = w.assign(**{series: w[series].astype(str)})
        w = w[w[series].isin(w[series].value_counts().index[:_MAX_SERIES])]
    if len(w) > _SCATTER_ROWS:
        w = w.sample(_SCATTER_ROWS, random_state=0)
    if len(w) < 2 or w[x].nunique() < 2:
        return None
    return _records(w), y, None, None


def _boxplot(d: pd.DataFrame, x: str, y: str, grain: Any) -> _Out:
    if not _is_num(d[y]):
        return None
    w = d[[x, y]].dropna()
    _axis(w, x, grain, _BOX_GROUPS)
    w[x] = w[x].dt.strftime("%Y-%m-%d") if _is_time(w[x]) else w[x].astype(str)
    counts = w[x].value_counts()
    w = w[w[x].isin(counts[counts >= _BOX_MIN_ROWS].index[:_BOX_GROUPS])]
    if len(w) > MAX_CHART_ROWS:
        w = w.sample(MAX_CHART_ROWS, random_state=0)
    if len(w) < 2 or w[x].nunique() < 2:
        return None
    return _records(w.sort_values(x, kind="stable")), y, None, None


def _aggregate(
    d: pd.DataFrame, kind: str, r: dict[str, Any], x: str, y: str | None, series: str | None, value: str | None
) -> _Out:
    """bar/line/area/stacked/grouped/pareto rows (x[, series], value) and heatmap
    long rows (x, y, value), grouped on the full frame."""
    heat = kind == "heatmap"
    measure, axis2 = (value, y) if heat else (y, None)
    by = [c for c in (x, axis2, series) if c]
    raw_agg = r.get("agg")
    if raw_agg is not None and raw_agg not in _AGGS:
        return None
    numeric = measure is not None and _is_num(d[measure])
    if measure is None:
        agg = "count"
    else:
        agg = raw_agg or ("sum" if kind == "pareto" and numeric else "mean" if numeric else "count")
    if (agg not in ("count", "nunique") and not numeric) or (kind == "pareto" and agg not in _ADDITIVE):
        return None
    name = measure or "count"
    if name in by:
        return None
    w = d[list(dict.fromkeys([*by, *([measure] if measure else [])]))].dropna()
    if w.empty:
        return None
    axes = [c for c in (x, axis2) if c]
    nominal = {c: not (_is_num(w[c]) or _is_time(w[c])) for c in axes}
    nser = min(w[series].nunique(), _MAX_SERIES) if series else 1
    cap = 24 if heat else max(2, min(_MAX_LEVELS, MAX_CHART_ROWS // nser))
    for c in axes:
        _axis(w, c, r.get("time_grain"), cap)
    if series:
        w[series] = w[series].astype(str)
    gb = w.groupby(by, sort=False, observed=True, dropna=True)
    g = (gb.size() if measure is None else gb[measure].agg(agg)).rename(name).reset_index()
    g = g.dropna(subset=[name])
    if kind == "pareto" and (g[name] < 0).any():
        return None

    additive = agg in _ADDITIVE
    fold_axes = heat or kind in _BAR_KINDS
    top_n = max(2, min(_MAX_TOP_N, _int(r.get("top_n"), _DEFAULT_TOP_N)))
    for c in axes:
        if not nominal[c]:
            continue
        if g[c].nunique() > _MAX_LEVELS and not (additive and fold_axes):
            return None
        if fold_axes:
            g = _fold(g, c, name, top_n if heat else min(top_n, _MAX_CATEGORIES), additive, by)
    if series:
        g = _fold(g, series, name, _MAX_SERIES, additive, by)
    if len(g) < 2 or len(g) > MAX_CHART_ROWS or g[x].nunique() < 2 or (axis2 and g[axis2].nunique() < 2):
        return None

    sort = r.get("sort")
    if kind in _BAR_KINDS and sort != "x" and (sort == "value" or nominal[x]):
        score = g[x].map(g.groupby(x)[name].sum())
        g = g.iloc[np.argsort(-score.to_numpy(dtype=float), kind="stable")]
    else:
        g = g.sort_values(by, kind="stable")
    return _records(g), (axis2 if heat else name), (name if heat else None), agg
