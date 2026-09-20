"""Automated chart guard: every chart producer is driven with adversarial column names and each
result is linted for the failure class "the spec looks valid but renders nothing".

`lint_chart` is reusable: pass a `ChartSpec.to_dict()`, a clean chart spec (what `dsa.chart.*` and
the tools return) or a rendered Vega-Lite spec; it returns a list of problems (empty = fine).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core import sandbox_toolkit as chart
from src.core.chart_spec import alias_fields, chart_has_data, spec_to_vegalite, validate_chart_spec

# ---------------------------------------------------------------------------
# The lint
# ---------------------------------------------------------------------------

_UNSAFE = re.compile(r"""[.\[\]\\'"]""")
_MARKS = frozenset({"bar", "line", "area", "point", "circle", "square", "rect", "rule", "text", "tick", "trail", "boxplot"})
_FORBIDDEN_KEYS = frozenset({"params", "param", "selection", "url", "lookup"})
_SCOPES = ("layer", "hconcat", "vconcat", "concat", "spec")
_DATUM_INDEX = re.compile(r"""datum\[\s*(['"])(.*?)\1\s*\]""")
_DATUM_DOT = re.compile(r"datum\.([A-Za-z_$][\w$]*)")
#: Transforms whose output columns are data-dependent (cannot be enumerated).
_OPEN_OUTPUT = frozenset({"pivot", "density", "regression", "loess", "flatten", "window"})


def _rows_of(node: dict[str, Any], datasets: dict[str, Any]) -> list[Any] | None:
    data = node.get("data")
    if not isinstance(data, dict):
        return None
    rows = data.get("values") if "values" in data else datasets.get(str(data.get("name")))
    return rows if isinstance(rows, list) else None


def _strings(value: Any) -> list[str]:
    return [v for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)]


def _as_names(node: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(node, dict):
        for key, val in node.items():
            out.update(_strings(val) if key == "as" else _as_names(val))
    elif isinstance(node, list):
        for item in node:
            out |= _as_names(item)
    return out


def _fields_in(node: Any) -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for key, val in node.items():
            if key in ("usermeta", "config"):
                continue
            if key == "field" and isinstance(val, str):
                out.append(val)
            else:
                out.extend(_fields_in(val))
    elif isinstance(node, list):
        for item in node:
            out.extend(_fields_in(item))
    return out


def _expressions_in(node: Any) -> list[tuple[str, str]]:
    """(kind, expression) for every calculate / string filter / test."""
    out: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, val in node.items():
            if key in ("usermeta", "config"):
                continue
            if key in ("calculate", "filter", "test") and isinstance(val, str):
                out.append((key, val))
            else:
                out.extend(_expressions_in(val))
    elif isinstance(node, list):
        for item in node:
            out.extend(_expressions_in(item))
    return out


def _check_field(name: str, avail: set[str], open_: bool, where: str, out: list[str]) -> None:
    if _UNSAFE.search(name):
        out.append(f"{where}: field {name!r} contains a character Vega reads as path syntax (. [ ] \\ ' \")")
    elif not open_ and name not in avail:
        out.append(f"{where}: field {name!r} is not a key of the inline rows (keys: {sorted(avail)[:8]})")


def _check_expression(kind: str, expr: str, avail: set[str], open_: bool, where: str, out: list[str]) -> None:
    for _, key in _DATUM_INDEX.findall(expr):
        if _UNSAFE.search(key) or (not open_ and key not in avail):
            out.append(f"{where}: {kind} expression {expr!r} indexes datum[{key!r}] (raw column name)")
    for name in _DATUM_DOT.findall(expr):
        if not open_ and name not in avail:
            out.append(f"{where}: {kind} expression {expr!r} reads datum.{name}, not a row key")


def _walk(node: Any, keys: set[str] | None, produced: set[str], open_: bool, where: str, out: list[str],
          datasets: dict[str, Any]) -> None:
    if not isinstance(node, dict):
        return
    if (rows := _rows_of(node, datasets)) is not None:
        keys = {k for r in rows if isinstance(r, dict) for k in r}
        produced, open_ = set(), False
    own = {k: v for k, v in node.items() if k not in _SCOPES and k not in ("transform", "usermeta", "config", "datasets")}
    scoped = set(produced)
    for i, t in enumerate(node.get("transform") or []):
        avail = (keys or set()) | scoped
        if keys is not None:
            for f in _fields_in(t) + [n for k in ("groupby", "on", "fold", "regression", "loess", "pivot", "stack",
                                                  "density", "impute", "key") for n in _strings(t.get(k))]:
                _check_field(f, avail, open_, f"{where}.transform[{i}]", out)
            for kind, expr in _expressions_in(t):
                _check_expression(kind, expr, avail, open_, f"{where}.transform[{i}]", out)
        scoped |= _as_names(t)
        if "fold" in t and "as" not in t:
            scoped |= {"key", "value"}
        if any(k in t for k in _OPEN_OUTPUT):
            open_ = True
    if keys is not None:
        avail = keys | scoped
        for f in _fields_in(own):
            _check_field(f, avail, open_, where, out)
        for kind, expr in _expressions_in(own):
            _check_expression(kind, expr, avail, open_, where, out)
    for scope in _SCOPES:
        child = node.get(scope)
        for j, sub in enumerate(child if isinstance(child, list) else [child]):
            _walk(sub, keys, scoped, open_, f"{where}.{scope}[{j}]", out, datasets)


def _forbidden(node: Any, where: str, out: list[str]) -> None:
    if isinstance(node, dict):
        for key, val in node.items():
            if key in ("usermeta", "config"):
                continue
            if key in _FORBIDDEN_KEYS:
                out.append(f"{where}: forbidden key {key!r}")
            if key == "mark":
                kind = val.get("type") if isinstance(val, dict) else val
                if kind not in _MARKS:
                    out.append(f"{where}: mark {kind!r} not in the allowed set")
            _forbidden(val, f"{where}.{key}", out)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _forbidden(item, f"{where}[{i}]", out)


@lru_cache(maxsize=1)
def _schema() -> Any:
    """The Vega-Lite JSON schema bundled with altair, or None (jsonschema/altair absent: check skipped)."""
    try:
        import importlib.util

        import jsonschema  # noqa: F401

        spec = importlib.util.find_spec("altair")
        if spec is None or not spec.submodule_search_locations:
            return None
        found = sorted(Path(spec.submodule_search_locations[0]).glob("vegalite/v*/schema/vega-lite-schema.json"))
        return json.loads(found[-1].read_text(encoding="utf-8")) if found else None
    except Exception:
        return None


def _schema_problems(rendered: dict[str, Any]) -> list[str]:
    if (schema := _schema()) is None:
        return []
    import jsonschema

    def leaves(err: Any) -> list[Any]:
        return [leaf for sub in err.context for leaf in leaves(sub)] if err.context else [err]

    top = next((name for key, name in (("layer", "TopLevelLayerSpec"), ("facet", "TopLevelFacetSpec"),
                                       ("hconcat", "TopLevelHConcatSpec"), ("vconcat", "TopLevelVConcatSpec"),
                                       ("concat", "TopLevelConcatSpec"), ("mark", "TopLevelUnitSpec"))
                if key in rendered), None)
    if top is None:
        return []
    validator = jsonschema.Draft7Validator({"definitions": schema["definitions"], "$ref": f"#/definitions/{top}"})
    out: list[str] = []
    for err in list(validator.iter_errors(rendered))[:3]:
        deepest = max(leaves(err), key=lambda e: len(e.absolute_path))  # the branch that got furthest
        if isinstance(deepest.instance, dict) and {"datum", "legend"} <= set(deepest.instance):
            continue  # known: the schema omits `legend` on datum colours, which Vega-Lite honours (series legends)
        out.append(f"Vega-Lite schema at /{'/'.join(map(str, deepest.absolute_path))}: {deepest.message[:140]}")
    return out


def lint_chart(chart_dict: dict[str, Any]) -> list[str]:
    """Problems that would make `chart_dict` render nothing or break; [] when it is sound."""
    problems: list[str] = []
    if "spec" in chart_dict and "chart_id" in chart_dict:  # ChartSpec.to_dict()
        rendered = chart_dict["spec"]
    elif "type" in chart_dict and isinstance(chart_dict.get("data"), list):  # clean chart spec
        clean, error = validate_chart_spec(chart_dict)
        if clean is None:
            return [f"clean spec does not re-pass validate_chart_spec: {error}"]
        rendered = spec_to_vegalite(chart_dict)
    else:
        rendered = chart_dict
    try:
        json.dumps(rendered, allow_nan=False)
    except (TypeError, ValueError) as exc:
        problems.append(f"not JSON-serialisable: {exc}")
    _walk(rendered, None, set(), False, "$", problems, rendered.get("datasets") or {})
    _forbidden(rendered, "$", problems)
    if not chart_has_data(rendered):
        problems.append("chart_has_data is False: nothing would be drawn")
    if alias_fields(rendered) is not rendered:
        problems.append("spec still contains an unaliased unsafe field name")
    return problems + _schema_problems(rendered)


def assert_clean(chart_dict: dict[str, Any], label: str = "") -> None:
    problems = lint_chart(chart_dict)
    assert not problems, f"{label}: " + "; ".join(problems)


# ---------------------------------------------------------------------------
# The lint must itself catch each failure class it exists for
# ---------------------------------------------------------------------------

def _vl(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    return {"data": {"values": rows}, "mark": "point", **kw}


class TestLintCatches:
    def test_clean_spec_passes(self) -> None:
        rows = [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 3, "b": 5}]
        enc = {"x": {"field": "a", "type": "quantitative"}, "y": {"field": "b", "type": "quantitative"}}
        assert lint_chart(_vl(rows, encoding=enc)) == []

    def test_dotted_field(self) -> None:
        rows = [{"PT08.S2(NMHC)": 1, "b": 2}, {"PT08.S2(NMHC)": 2, "b": 3}]
        enc = {"x": {"field": "PT08.S2(NMHC)", "type": "quantitative"}, "y": {"field": "b", "type": "quantitative"}}
        assert any("path syntax" in p for p in lint_chart(_vl(rows, encoding=enc)))

    def test_missing_field(self) -> None:
        rows = [{"a": 1}, {"a": 2}]
        enc = {"x": {"field": "a", "type": "quantitative"}, "y": {"field": "nope", "type": "quantitative"}}
        assert any("not a key" in p for p in lint_chart(_vl(rows, encoding=enc)))

    def test_raw_datum_index_and_unknown_datum(self) -> None:
        rows = [{"a": 1, "raw.col": 2}, {"a": 2, "raw.col": 3}]
        enc = {"x": {"field": "a", "type": "quantitative"}, "y": {"field": "a", "type": "quantitative"}}
        for expr in ("datum['raw.col'] * 2", "datum['gone'] * 2", "datum.ghost + 1"):
            spec = _vl(rows, encoding=enc, transform=[{"calculate": expr, "as": "z"}])
            assert any("datum" in p for p in lint_chart(spec)), expr

    def test_produced_field_is_allowed(self) -> None:
        rows = [{"a": 1, "b": 2}, {"a": 2, "b": 3}]
        spec = _vl(rows, transform=[{"calculate": "datum.a + datum.b", "as": "z"}],
                   encoding={"x": {"field": "a", "type": "quantitative"}, "y": {"field": "z", "type": "quantitative"}})
        assert lint_chart(spec) == []

    def test_no_data_and_forbidden(self) -> None:
        rows = [{"a": None, "b": None}, {"a": None, "b": None}]
        enc = {"x": {"field": "a", "type": "quantitative"}, "y": {"field": "b", "type": "quantitative"}}
        assert any("chart_has_data" in p for p in lint_chart(_vl(rows, encoding=enc)))
        spec = _vl([{"a": 1, "b": 2}] * 2, encoding=enc, params=[{"name": "p"}])
        spec["mark"] = "image"
        problems = lint_chart(spec)
        assert any("params" in p for p in problems) and any("mark 'image'" in p for p in problems)

    def test_test_condition_fails_the_raw_sanitiser(self) -> None:
        rows = [{"a": 1, "b": 2}, {"a": 2, "b": 3}]
        layout = {"mark": "point", "encoding": {
            "x": {"field": "a", "type": "quantitative"}, "y": {"field": "b", "type": "quantitative"},
            "opacity": {"condition": {"test": "datum.a > 1", "value": 1}, "value": 0.3}}}
        problems = lint_chart({"type": "vega_lite", "data": rows, "vega_lite": layout})
        assert problems and "validate_chart_spec" in problems[0]


# ---------------------------------------------------------------------------
# Producers x adversarial names
# ---------------------------------------------------------------------------

NAMES = [
    "PT08.S2(NMHC)", "a[b]", "x.y.z", "Étoile", "co₂ (ppm)", 'say "hi"', "it's", "2fast 2 furious",
    "back\\slash", "plain col",
]
_RNG = np.random.default_rng(3)


def _cols(name: str) -> dict[str, str]:
    return {"x": f"grp {name}", "y": name, "s": f"ser {name}", "t": f"tgt {name}", "lo": f"lo {name}", "hi": f"hi {name}",
            "y2": f"{name} 2", "d": f"date {name}"}


def _cat_frame(c: dict[str, str], n_cat: int = 5, per: int = 1, series: int = 0) -> pd.DataFrame:
    cats = [f"cat{i}" for i in range(n_cat)]
    frame = pd.DataFrame({c["x"]: cats * per, c["y"]: _RNG.uniform(5, 50, n_cat * per)})
    if series:
        frame = pd.concat([frame.assign(**{c["s"]: f"s{k}", c["y"]: _RNG.uniform(5, 50, len(frame))})
                           for k in range(series)], ignore_index=True)
    return frame


def _time_frame(c: dict[str, str]) -> pd.DataFrame:
    y = _RNG.normal(100, 10, 24)
    return pd.DataFrame({c["d"]: pd.date_range("2024-01-01", periods=24, freq="MS").strftime("%Y-%m-%d"), c["y"]: y,
                         c["y2"]: y * 3 + _RNG.normal(0, 5, 24), c["lo"]: y - 5, c["hi"]: y + 5})


def _p_bar(c: dict[str, str]) -> dict[str, Any]:
    return chart.bar(_cat_frame(c), c["x"], c["y"], sort="desc")


def _p_bar_color_facet(c: dict[str, str]) -> dict[str, Any]:
    f = _cat_frame(c, 4, series=2)
    f[f"fac {c['y']}"] = ["p" if i % 2 else "q" for i in range(len(f))]
    return chart.bar(f, c["x"], c["y"], color=c["s"], facet=f"fac {c['y']}")


def _p_line(c: dict[str, str]) -> dict[str, Any]:
    return chart.line(_time_frame(c), c["d"], c["y"])


def _p_line_series(c: dict[str, str]) -> dict[str, Any]:
    f = pd.concat([_time_frame(c).assign(**{c["s"]: s}) for s in ("a", "b")], ignore_index=True)
    return chart.line(f, c["d"], c["y"], color=c["s"], annotations=[{"y": 100, "label": "target"}])


def _p_area(c: dict[str, str]) -> dict[str, Any]:
    return chart.area(_time_frame(c), c["d"], c["y"])


def _p_scatter(c: dict[str, str]) -> dict[str, Any]:
    x = _RNG.normal(0, 1, 60)
    f = pd.DataFrame({c["x"]: x, c["y"]: x * 2 + _RNG.normal(0, 1, 60), c["s"]: ["a", "b"] * 30})
    return chart.scatter(f, c["x"], c["y"], color=c["s"])


def _p_histogram(c: dict[str, str]) -> dict[str, Any]:
    return chart.histogram(pd.DataFrame({c["y"]: _RNG.normal(0, 1, 120)}), c["y"])


def _p_heatmap_long(c: dict[str, str]) -> dict[str, Any]:
    f = pd.DataFrame([(a, b, _RNG.normal()) for a in "wxyz" for b in "pqrs"], columns=[c["x"], c["s"], c["y"]])
    return chart.heatmap(f, c["x"], c["s"], color=c["y"])


def _p_heatmap_wide(c: dict[str, str]) -> dict[str, Any]:
    f = pd.DataFrame({c["x"]: list("abcdef"), c["y"]: _RNG.normal(size=6), c["y2"]: _RNG.normal(size=6),
                      c["t"]: _RNG.normal(size=6)})
    return chart.heatmap(f, c["x"], [c["y"], c["y2"], c["t"]])


def _p_heatmap_zscore(c: dict[str, str]) -> dict[str, Any]:
    f = pd.DataFrame({c["x"]: list("abcdef"), c["y"]: _RNG.normal(size=6), c["y2"]: _RNG.normal(size=6)})
    return chart.heatmap(f, c["x"], [c["y"], c["y2"]], scale="zscore")


def _p_corr(c: dict[str, str]) -> dict[str, Any]:
    base = _RNG.normal(size=80)
    f = pd.DataFrame({c["y"]: base, c["y2"]: base + _RNG.normal(0, .5, 80), c["t"]: _RNG.normal(size=80),
                      c["lo"]: -base + _RNG.normal(0, .5, 80)})
    return chart.corr_heatmap(f)


def _p_stacked(c: dict[str, str]) -> dict[str, Any]:
    return chart.stacked_bar(_cat_frame(c, 4, series=3), c["x"], c["y"], c["s"])


def _p_grouped(c: dict[str, str]) -> dict[str, Any]:
    return chart.grouped_bar(_cat_frame(c, 4, series=2), c["x"], c["y"], c["s"])


def _p_boxplot(c: dict[str, str]) -> dict[str, Any]:
    return chart.boxplot(_cat_frame(c, 3, per=10), c["x"], c["y"])


def _p_pareto(c: dict[str, str]) -> dict[str, Any]:
    return chart.pareto(_cat_frame(c, 8), c["x"], c["y"])


def _p_slope(c: dict[str, str]) -> dict[str, Any]:
    f = pd.DataFrame({c["x"]: ["before"] * 5 + ["after"] * 5, c["s"]: list("abcde") * 2,
                      c["y"]: _RNG.uniform(1, 9, 10)})
    return chart.slope(f, c["x"], c["y"], c["s"])


def _p_bullet(c: dict[str, str]) -> dict[str, Any]:
    f = _cat_frame(c, 4)
    f[c["t"]] = f[c["y"]] * 1.1
    return chart.bullet(f, c["x"], c["y"], c["t"])


def _p_band(c: dict[str, str]) -> dict[str, Any]:
    return chart.band(_time_frame(c), c["d"], c["y"], c["lo"], c["hi"])


def _p_waterfall(c: dict[str, str]) -> dict[str, Any]:
    f = pd.DataFrame({c["x"]: ["start", "up", "down", "more", "end"], c["y"]: [100, 20, -15, 8, -3]})
    return chart.waterfall(f, c["x"], c["y"])


def _p_lorenz(c: dict[str, str]) -> dict[str, Any]:
    pop = np.linspace(0, 1, 11)
    return chart.lorenz(pd.DataFrame({c["x"]: pop, c["y"]: pop ** 2}), c["x"], c["y"])


def _p_dot_ci(c: dict[str, str]) -> dict[str, Any]:
    f = _cat_frame(c, 5)
    f[c["lo"]], f[c["hi"]] = f[c["y"]] - 3, f[c["y"]] + 3
    return chart.dot_ci(f, c["x"], c["y"], y_lower=c["lo"], y_upper=c["hi"])


def _p_dual_axis(c: dict[str, str]) -> dict[str, Any]:
    return chart.dual_axis(_time_frame(c), c["d"], c["y"], c["y2"])


def _p_vega_lite_point(c: dict[str, str]) -> dict[str, Any]:
    f = pd.DataFrame({c["x"]: _RNG.normal(size=40), c["y"]: _RNG.normal(size=40), c["s"]: ["a", "b"] * 20})
    return chart.vega_lite({"mark": "point", "encoding": {
        "x": {"field": c["x"], "type": "quantitative"}, "y": {"field": c["y"], "type": "quantitative"},
        "color": {"field": c["s"], "type": "nominal"},
        "tooltip": [{"field": c["x"]}, {"field": c["y"]}]}}, data=f)


def _p_vega_lite_layers(c: dict[str, str]) -> dict[str, Any]:
    f = _cat_frame(c, 6)
    f["flag"] = [True, False] * 3
    enc = {"x": {"field": c["x"], "type": "nominal", "sort": {"field": c["y"], "order": "descending"}},
           "y": {"field": c["y"], "type": "quantitative"}}
    return chart.vega_lite({"layer": [
        {"mark": "bar", "encoding": enc},
        {"mark": {"type": "text", "dy": -6}, "transform": [{"filter": {"field": c["y"], "gte": 10}}],
         "encoding": {**enc, "text": {"field": c["y"], "type": "quantitative", "format": ".1f"}}}]}, data=f)


def _p_vega_lite_fold(c: dict[str, str]) -> dict[str, Any]:
    f = pd.DataFrame({c["x"]: list("abcd"), c["y"]: [1.0, 2, 3, 4], c["y2"]: [2.0, 3, 1, 5]})
    return chart.vega_lite({"transform": [{"fold": [c["y"], c["y2"]], "as": ["metric", "amount"]}], "mark": "bar",
                            "encoding": {"x": {"field": c["x"], "type": "nominal"},
                                         "y": {"field": "amount", "type": "quantitative"},
                                         "color": {"field": "metric", "type": "nominal"}}}, data=f)


def _p_vega_lite_aggregate(c: dict[str, str]) -> dict[str, Any]:
    f = _cat_frame(c, 4, per=6)
    return chart.vega_lite({"mark": "bar", "encoding": {
        "x": {"field": c["x"], "type": "nominal"},
        "y": {"aggregate": "mean", "field": c["y"], "type": "quantitative"}}}, data=f)


def _p_vega_lite_facet(c: dict[str, str]) -> dict[str, Any]:
    f = pd.DataFrame({c["x"]: _RNG.normal(size=40), c["y"]: _RNG.normal(size=40), c["s"]: ["a", "b"] * 20})
    return chart.vega_lite({"facet": {"column": {"field": c["s"], "type": "nominal"}}, "spec": {
        "mark": "point", "encoding": {"x": {"field": c["x"], "type": "quantitative"},
                                      "y": {"field": c["y"], "type": "quantitative"}}}}, data=f)


CHART_PRODUCERS = {
    f.__name__[3:]: f for f in (
        _p_bar, _p_bar_color_facet, _p_line, _p_line_series, _p_area, _p_scatter, _p_histogram, _p_heatmap_long,
        _p_heatmap_wide, _p_heatmap_zscore, _p_corr, _p_stacked, _p_grouped, _p_boxplot, _p_pareto, _p_slope,
        _p_bullet, _p_band, _p_waterfall, _p_lorenz, _p_dot_ci, _p_dual_axis, _p_vega_lite_point,
        _p_vega_lite_layers, _p_vega_lite_fold, _p_vega_lite_aggregate, _p_vega_lite_facet,
    )
}


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("producer", sorted(CHART_PRODUCERS))
def test_dsa_chart_helpers(producer: str, name: str) -> None:
    assert_clean(CHART_PRODUCERS[producer](_cols(name)), f"{producer}[{name}]")


# ---------------------------------------------------------------------------
# dashboard.build_dashboard: every builder reachable from a synthetic frame
# ---------------------------------------------------------------------------

def _tool(name: str, **output: Any) -> dict[str, Any]:
    return {"tool_name": name, "status": "success", "output": output}


def _dash_inputs(name: str, wide: bool = True) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(11)
    n = 360
    c = {"y": name, "m2": f"m2 {name}", "m3": f"m3 {name}", "m4": f"m4 {name}", "m5": f"m5 {name}", "seg": f"seg {name}",
         "tgt": f"tgt {name}", "when": f"when {name}", "cust": f"cust {name}", "dept": f"dept {name}",
         "hired": f"hired {name}", "sym": f"sym {name}", "price": f"price {name}", "amount": f"amount {name}"}
    seg = rng.choice(["alpha", "beta", "gamma"], n, p=[0.6, 0.3, 0.1])
    base = rng.normal(50, 10, n) + (seg == "beta") * 12
    df = pd.DataFrame({
        c["y"]: base, c["m2"]: base * 1.5 + rng.normal(0, 4, n), c["m3"]: rng.normal(0, 1, n),
        c["m4"]: -base + rng.normal(0, 6, n), c["m5"]: rng.gamma(2.0, 3.0, n),
        c["seg"]: seg, c["tgt"]: (base + rng.normal(0, 8, n) > 55).astype(int),
        c["when"]: pd.date_range("2023-01-01", periods=n, freq="D"),
        c["cust"]: rng.integers(1, 40, n), c["dept"]: rng.choice(["ops", "eng", "hr", "fin"], n),
        c["hired"]: pd.date_range("2015-01-01", periods=n, freq="9D"),
        c["sym"]: rng.choice(["AAA", "BBB"], n), c["price"]: 100 + np.cumsum(rng.normal(0, 1, n)),
        c["amount"]: rng.gamma(2.0, 50.0, n),
    })
    if not wide:
        df = df[[c["y"], c["m2"], c["seg"], c["tgt"], c["when"]]]
    y, m2, seg_c = c["y"], c["m2"], c["seg"]
    results = [
        _tool("correlation_analysis", top_correlations=[
            {"col_a": y, "col_b": m2, "correlation": 0.93}, {"col_a": y, "col_b": c["m4"], "correlation": -0.7}]),
        _tool("cluster_data", n_clusters=3, silhouette_score=0.5,
              pca_points=[{"x": float(i), "y": float(-i % 7), "cluster": f"cluster_{i % 3}"} for i in range(30)]),
        _tool("time_series_analysis", date_column=c["when"], value_column=y, trend_direction="upward"),
        _tool("dimensionality_analysis", explained_variance_ratio=[0.5, 0.3, 0.2], cumulative_variance=[0.5, 0.8, 1.0],
              n_components_for_threshold=2, variance_threshold=0.8),
        _tool("geospatial_analysis", densest_cells=[
            {"lat_range": [10.0, 11.0], "lon_range": [20.0, 21.0], "count": 42},
            {"lat_range": [11.0, 12.0], "lon_range": [21.0, 22.0], "count": 17}]),
        _tool("train_model", task_type="classification", best_model="rf", models_trained={
            "lr": {"train_metrics": {"accuracy": 0.8}, "test_metrics": {"accuracy": 0.78}, "cv_mean": 0.77},
            "rf": {"train_metrics": {"accuracy": 0.95}, "test_metrics": {"accuracy": 0.88}, "cv_mean": 0.86}}),
        _tool("evaluate_model", top_drivers=[
            {"feature": y, "importance": 0.4, "importance_std": 0.05, "headline": "up"},
            {"feature": m2, "importance": 0.2, "importance_std": 0.02, "headline": "down"}],
              confusion_matrix=[[50, 10], [8, 60]], class_labels=["no", "yes"],
              roc_curve={"fpr": [0, .1, .3, 1], "tpr": [0, .5, .8, 1]}, roc_auc=0.85),
        _tool("cohort_analysis", customer_column=c["cust"], amount_column=c["amount"],
              rfm_segments=[{"segment": "champions", "revenue_share_pct": 60.0, "customer_share_pct": 20.0},
                            {"segment": "at risk", "revenue_share_pct": 10.0, "customer_share_pct": 30.0}],
              revenue_by_month={"2023-01": 100.0, "2023-02": 140.0, "2023-03": 90.0}),
        _tool("financial_analysis", date_column=c["when"], price_column=c["price"], symbol_column=c["sym"],
              best_performer={"symbol": "AAA"}),
        _tool("workforce_analysis", department_column=c["dept"], hire_date_column=c["hired"],
              by_department=[{"department": "ops", "headcount": 90}, {"department": "eng", "headcount": 120},
                             {"department": "hr", "headcount": 60}]),
        _tool("segment_comparison", comparisons=[
            {"measure": y, "dimension": seg_c, "level": lv, "level_value": v, "ci_lower": v - 2, "ci_upper": v + 2, "n": 100}
            for lv, v in (("alpha", 50.0), ("beta", 62.0), ("gamma", 49.0))]),
    ]
    findings: list[dict[str, Any]] = [
        {"finding_id": "f1", "kind": "segment_lift", "source_tool": "segment_comparison", "importance": 0.9, "measure": y,
         "evidence": {"measure": y, "dimension": seg_c, "level": "beta", "level_value": 62.0, "baseline_value": 50.0,
                      "overall_mean": 53.0, "n": 100}},
        {"finding_id": "f2", "kind": "concentration", "source_tool": "concentration_analysis", "importance": 0.8,
         "evidence": {"measure_column": c["m5"], "entity_column": c["cust"], "gini_coefficient": 0.4}},
        {"finding_id": "f3", "kind": "change", "source_tool": "change_analysis", "importance": 0.7,
         "evidence": {"measure_column": c["amount"], "prior_period_value": 100.0, "latest_value": 130.0,
                      "aggregation": "sum", "period_grain": "monthly", "latest_period": "2023-03",
                      "segment_breakdown": [{"level": "a", "delta": 20.0}, {"level": "b", "delta": 15.0}]}},
        {"finding_id": "f4", "kind": "test", "source_tool": "select_statistical_test", "importance": 0.6,
         "measure": y, "dimension": seg_c,
         "evidence": {"post_hoc": [{"group_a": "alpha", "group_b": "beta", "p_adjusted": 0.01}]}},
        {"finding_id": "f5", "kind": "distribution", "measure": m2},
        {"finding_id": "f6", "kind": "distribution", "measure": c["m5"]},
        {"finding_id": "f7", "kind": "distribution", "measure": seg_c},
        {"finding_id": "f8", "kind": "distribution", "measure": c["dept"]},
    ]
    return df, c, results, findings


def _dashboard_charts(name: str, wide: bool, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    from src.core import dashboard
    from src.core.profiler import profile_dataframe

    monkeypatch.setattr(dashboard, "MAX_CHARTS", 999)
    df, c, results, findings = _dash_inputs(name, wide)
    profile = profile_dataframe(df, target_column=c["tgt"])
    return dashboard.build_dashboard(df, profile, target_column=c["tgt"], task_type="classification",
                                     tool_results=results, findings=findings)


#: Builders that must be reachable here (a prefix of each chart_id), so a silent
#: "produced nothing" cannot make this guard vacuous.
_DASHBOARD_BUILDERS = (
    "top_correlations", "cluster_scatter", "time_series", "pca_scree", "geospatial_hotspots", "model_comparison",
    "model_drivers", "model_confusion_matrix", "model_roc", "cohort_rfm_segments", "cohort_revenue_by_month",
    "cohort_pareto", "financial_overview", "workforce_headcount_by_dept", "workforce_tenure_hist", "segment_",
    "lorenz_", "change_waterfall_", "groups_", "hist_", "cat_", "box_", "class_balance", "scatter_top_pair",
)


@pytest.mark.parametrize("name", NAMES)
def test_dashboard_wide(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    charts = _dashboard_charts(name, True, monkeypatch)
    ids = [ch.chart_id for ch in charts]
    for builder in _DASHBOARD_BUILDERS:
        assert any(i.startswith(builder) for i in ids), f"{builder} not produced for {name!r}: {ids}"
    for ch in charts:
        assert_clean(ch.to_dict(), f"{ch.chart_id}[{name}]")


@pytest.mark.parametrize("name", NAMES)
def test_dashboard_narrow_uses_correlation_bars(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    charts = _dashboard_charts(name, False, monkeypatch)
    assert charts
    for ch in charts:
        assert_clean(ch.to_dict(), f"{ch.chart_id}[{name}]")


# ---------------------------------------------------------------------------
# chart_designer.run_recipe: every recipe type
# ---------------------------------------------------------------------------

def _recipes(name: str) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rng = np.random.default_rng(5)
    n = 400
    c = _cols(name)
    df = pd.DataFrame({
        c["x"]: rng.choice([f"c{i}" for i in range(6)], n), c["s"]: rng.choice(["p", "q", "r"], n),
        c["y"]: rng.normal(50, 12, n), c["y2"]: rng.normal(0, 1, n),
        c["d"]: pd.date_range("2023-01-01", periods=n, freq="D"),
    })
    df[c["y2"]] = df[c["y"]] * 0.5 + df[c["y2"]]
    x, y, s, d, y2 = c["x"], c["y"], c["s"], c["d"], c["y2"]
    return df, {
        "bar": {"type": "bar", "x": x, "y": y, "agg": "mean", "sort": "value"},
        "bar_count": {"type": "bar", "x": x},
        "bar_filtered": {"type": "bar", "x": x, "y": y, "agg": "sum", "filter": [{"col": y, "op": ">", "value": 40}]},
        "line": {"type": "line", "x": d, "y": y, "agg": "mean", "time_grain": "month"},
        "line_series": {"type": "line", "x": d, "y": y, "series": s, "time_grain": "month"},
        "area": {"type": "area", "x": d, "y": y, "agg": "sum", "time_grain": "quarter"},
        "scatter": {"type": "scatter", "x": y2, "y": y, "series": s},
        "heatmap": {"type": "heatmap", "x": x, "y": s, "value": y, "agg": "mean"},
        "histogram": {"type": "histogram", "x": y},
        "boxplot": {"type": "boxplot", "x": x, "y": y},
        "stacked_bar": {"type": "stacked_bar", "x": x, "y": y, "series": s, "agg": "sum"},
        "grouped_bar": {"type": "grouped_bar", "x": x, "y": y, "series": s, "agg": "mean"},
        "pareto": {"type": "pareto", "x": x, "y": y, "agg": "sum"},
    }


@pytest.mark.parametrize("name", NAMES)
def test_chart_designer_recipes(name: str) -> None:
    from src.core.chart_designer import run_recipe
    from src.core.dashboard import designed_chart

    df, recipes = _recipes(name)
    for label, recipe in recipes.items():
        raw = run_recipe(df, recipe)
        assert raw is not None, f"recipe {label!r} produced nothing for {name!r}"
        assert_clean(raw, f"recipe {label}[{name}]")
        chart_spec = designed_chart(raw, 0, [])
        assert chart_spec is not None, f"designed_chart rejected recipe {label!r} for {name!r}"
        assert_clean(chart_spec.to_dict(), f"designed {label}[{name}]")


def test_chart_designer_covers_every_recipe_kind() -> None:
    from src.core.chart_designer import _KINDS

    assert {r["type"] for r in _recipes("plain")[1].values()} >= _KINDS


# ---------------------------------------------------------------------------
# generate_visualizations tool charts
# ---------------------------------------------------------------------------

def _viz_csv(tmp_path: Path, name: str) -> tuple[str, dict[str, str]]:
    rng = np.random.default_rng(9)
    n = 200
    c = _cols(name)
    base = rng.normal(0, 1, n)
    df = pd.DataFrame({
        c["y"]: base, c["y2"]: base + rng.normal(0, 0.5, n), c["t"]: rng.normal(0, 1, n),
        c["lo"]: -base + rng.normal(0, 0.5, n), c["hi"]: rng.uniform(0, 10, n),
        c["s"]: (rng.normal(0, 1, n) > 0).astype(int),            # balanced-ish 2-value flag
        c["x"]: np.arange(1, n + 1),                              # 1..N id
    })
    path = tmp_path / "viz.csv"
    df.to_csv(path, index=False)
    return str(path), c


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("chart_type", ["correlation_heatmap", "distributions"])
def test_visualization_tool_charts(chart_type: str, name: str, tmp_path: Path) -> None:
    from src.tools.visualization import GenerateVisualizationsTool

    file_path, _ = _viz_csv(tmp_path, name)
    result = GenerateVisualizationsTool().run(file_path=file_path, chart_type=chart_type)
    assert result.status == "success", result.output
    assert result.output["charts"]
    for i, ch in enumerate(result.output["charts"]):
        assert_clean(ch, f"{chart_type}#{i}[{name}]")


def test_visualization_feature_importance(tmp_path: Path) -> None:
    from src.tools.ml_pipeline import TrainModelTool
    from src.tools.visualization import GenerateVisualizationsTool

    name = "PT08.S2(NMHC)"
    file_path, c = _viz_csv(tmp_path, name)
    train = TrainModelTool().run(file_path=file_path, target_column=c["s"], task_type="classification",
                                 models=["random_forest"], n_cv_folds=3, output_dir=str(tmp_path / "models"))
    assert train.status == "success", train.output
    model_path = train.output["models_trained"]["random_forest"]["model_path"]
    result = GenerateVisualizationsTool().run(file_path=file_path, chart_type="feature_importance",
                                              target_column=c["s"], model_path=model_path)
    assert result.status == "success", result.output
    for ch in result.output["charts"]:
        assert_clean(ch, "feature_importance")


class TestDistributionsUniformColumns:
    """Balanced flags and 1..N ids are valid distributions (drawn as bars of counts); constants are not."""

    @staticmethod
    def _charts(tmp_path: Path, frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
        from src.tools.visualization import GenerateVisualizationsTool

        path = tmp_path / "d.csv"
        frame.to_csv(path, index=False)
        result = GenerateVisualizationsTool().run(file_path=str(path), chart_type="distributions")
        assert result.status == "success", result.output
        return {ch["title"].split("— ")[-1]: ch for ch in result.output["charts"]}

    def test_balanced_flag_and_id_are_drawn_as_bars_and_constant_is_skipped(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({
            "flag": [0, 1] * 50, "lopsided": [0] * 30 + [1] * 70, "row_id": np.arange(1, 101),
            "constant": 7, "amount": rng.normal(size=100),
        })
        charts = self._charts(tmp_path, frame)
        assert set(charts) == {"Flag", "Lopsided", "Row ID", "Amount"}
        assert charts["Flag"]["type"] == "bar"
        assert charts["Flag"]["data"] == [{"value": 0, "rows": 50}, {"value": 1, "rows": 50}]
        assert charts["Lopsided"]["data"] == [{"value": 0, "rows": 30}, {"value": 1, "rows": 70}]
        assert charts["Row ID"]["type"] == "bar" and {r["rows"] for r in charts["Row ID"]["data"]} == {5}
        assert charts["Amount"]["type"] == "area"
        for ch in charts.values():
            assert_clean(ch, ch["title"])

    def test_only_constant_columns_is_still_an_error(self, tmp_path: Path) -> None:
        from src.tools.visualization import GenerateVisualizationsTool

        path = tmp_path / "c.csv"
        pd.DataFrame({"a": [1] * 20, "b": [2] * 20}).to_csv(path, index=False)
        assert GenerateVisualizationsTool().run(file_path=str(path), chart_type="distributions").status == "error"


@pytest.mark.parametrize("name", NAMES)
def test_dashboard_renders_every_declared_dsa_chart(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Charts declared by custom code (`output["chart"]`) reach the dashboard through the same lint."""
    from src.core import dashboard
    from src.core.profiler import profile_dataframe

    monkeypatch.setattr(dashboard, "MAX_CHARTS", 999)
    df, c, _, _ = _dash_inputs(name, wide=False)
    results = [_tool(f"custom_{key}", chart=make(_cols(name))) for key, make in CHART_PRODUCERS.items()]
    charts = dashboard.build_dashboard(df, profile_dataframe(df, target_column=c["tgt"]), tool_results=results)
    declared = [ch for ch in charts if ch.chart_id.startswith("llm_")]
    assert len(declared) >= len(CHART_PRODUCERS) - 3, [ch.chart_id for ch in charts]  # identical specs are de-duplicated
    for ch in declared:
        assert_clean(ch.to_dict(), f"{ch.chart_id}[{name}]")
