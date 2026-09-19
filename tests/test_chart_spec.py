"""Validator/renderer behaviour for the chart catalog and the raw Vega-Lite escape hatch."""
from __future__ import annotations

from typing import Any

import pytest

from src.core.chart_spec import MAX_CHART_ROWS, spec_to_vegalite, validate_chart_spec
from src.core.dashboard import _llm_chart


def _ok(spec: dict[str, Any]) -> dict[str, Any]:
    clean, error = validate_chart_spec(spec)
    assert clean is not None, error
    return clean


def _bad(spec: dict[str, Any]) -> str:
    clean, error = validate_chart_spec(spec)
    assert clean is None and error
    return error


def _rows(n: int, value: float = 10.0) -> list[dict[str, Any]]:
    return [{"cat": f"cat_{i:02d}", "sales": value + i} for i in range(n)]


def _layers(vl: dict[str, Any]) -> list[dict[str, Any]]:
    return vl.get("layer") or [vl]


class TestCategoryQuality:
    def test_high_cardinality_additive_tail_becomes_other(self) -> None:
        clean = _ok({"type": "bar", "data": _rows(30), "x": "cat", "y": "sales", "y_format": "currency"})
        cats = [r["cat"] for r in clean["data"]]
        assert len(cats) == 12 and cats[-1] == "Other"
        assert sum(r["sales"] for r in clean["data"]) == sum(r["sales"] for r in _rows(30))
        assert "Other" in clean["note"]
        order = spec_to_vegalite(clean)["encoding"]
        sort = (order.get("x") if "sort" in order.get("x", {}) else order["y"])["sort"]
        assert sort[-1] == "Other" and sort[0] == "cat_29"

    def test_non_additive_tail_is_dropped_not_summed(self) -> None:
        rows = [{"cat": f"c{i}", "avg_price": 5.0 + i} for i in range(20)]
        clean = _ok({"type": "bar", "data": rows, "x": "cat", "y": "avg_price"})
        assert len(clean["data"]) == 12 and all(r["cat"] != "Other" for r in clean["data"])

    def test_revalidation_is_idempotent(self) -> None:
        once = _ok({"type": "pareto", "data": _rows(30), "x": "cat", "y": "sales", "caption": "c", "size": "wide"})
        assert _ok(once) == once

    def test_long_labels_go_horizontal(self) -> None:
        rows = [{"cat": "a very long category label", "v": 1}, {"cat": "b very long category label", "v": 2}]
        enc = spec_to_vegalite(_ok({"type": "bar", "data": rows, "x": "cat", "y": "v"}))["encoding"]
        assert enc["y"]["field"] == "cat" and enc["x"]["field"] == "v"

    def test_bars_are_zero_based_and_narrow_lines_are_not(self) -> None:
        rows = [{"d": f"2024-0{m}-01", "v": 100.0 + m} for m in range(1, 7)]
        bar = spec_to_vegalite(_ok({"type": "bar", "data": rows, "x": "d", "y": "v"}))
        line = spec_to_vegalite(_ok({"type": "line", "data": rows, "x": "d", "y": "v"}))
        assert bar["encoding"]["y"]["scale"] == {"zero": True}
        assert line["encoding"]["y"]["scale"]["zero"] is False

    def test_non_iso_dates_are_normalised_to_temporal(self) -> None:
        rows = [{"d": f"{m} 2024", "v": i} for i, m in enumerate(["Jan", "Feb", "Mar"])]
        clean = _ok({"type": "line", "data": rows, "x": "d", "y": "v"})
        assert clean["data"][0]["d"] == "2024-01-01"
        assert spec_to_vegalite(clean)["encoding"]["x"]["type"] == "temporal"

    def test_large_values_get_compact_axis(self) -> None:
        rows = [{"cat": "a", "v": 2_500_000}, {"cat": "b", "v": 1_000_000}]
        axis = spec_to_vegalite(_ok({"type": "bar", "data": rows, "x": "cat", "y": "v"}))["encoding"]["y"]["axis"]
        assert axis["format"] == "~s"

    def test_dense_scatter_fades_points(self) -> None:
        rows = [{"a": i, "b": i % 7} for i in range(400)]
        vl = spec_to_vegalite(_ok({"type": "scatter", "data": rows, "x": "a", "y": "b"}))
        assert _layers(vl)[0]["mark"]["opacity"] < 0.4


class TestNewKinds:
    def test_stacked_and_grouped_need_series(self) -> None:
        rows = [{"q": q, "g": g, "v": 1 + i} for i, (q, g) in enumerate((q, g) for q in ("Q1", "Q2") for g in "ab")]
        assert "series" in _bad({"type": "stacked_bar", "data": rows, "x": "q", "y": "v"})
        stacked = spec_to_vegalite(_ok({"type": "stacked_bar", "data": rows, "x": "q", "y": "v", "series": "g"}))
        grouped = spec_to_vegalite(_ok({"type": "grouped_bar", "data": rows, "x": "q", "y": "v", "series": "g"}))
        assert stacked["encoding"]["y"]["stack"] == "zero" and "xOffset" not in stacked["encoding"]
        assert "xOffset" in grouped["encoding"]

    def test_boxplot_needs_raw_values(self) -> None:
        assert "RAW" in _bad({"type": "boxplot", "data": [{"g": "a", "v": 1}, {"g": "b", "v": 2}], "x": "g", "y": "v"})
        raw = [{"g": "a", "v": float(i)} for i in range(8)] + [{"g": "b", "v": float(i)} for i in range(8)]
        assert spec_to_vegalite(_ok({"type": "boxplot", "data": raw, "x": "g", "y": "v"}))["mark"]["type"] == "boxplot"

    def test_pareto_builds_cumulative_share(self) -> None:
        rows = [{"c": "a", "v": 60}, {"c": "b", "v": 30}, {"c": "c", "v": 10}]
        vl = spec_to_vegalite(_ok({"type": "pareto", "data": rows, "x": "c", "y": "v"}))
        shares = [r["cumulative_share"] for r in vl["data"]["values"]]
        assert shares == pytest.approx([0.6, 0.9, 1.0])
        assert "independent" in str(vl["resolve"])
        assert "non-negative" in _bad({"type": "pareto", "data": [{"c": "a", "v": -1}, {"c": "b", "v": 2}], "x": "c", "y": "v"})

    def test_slope_needs_exactly_two_points(self) -> None:
        rows = [{"yr": y, "g": g, "v": float(i)} for i, (y, g) in enumerate((y, g) for y in (2020, 2021, 2022) for g in "ab")]
        assert "exactly 2" in _bad({"type": "slope", "data": rows, "x": "yr", "y": "v", "color": "g"})
        two = [r for r in rows if r["yr"] != 2022]
        vl = spec_to_vegalite(_ok({"type": "slope", "data": two, "x": "yr", "y": "v", "color": "g"}))
        assert vl["layer"][0]["encoding"]["x"]["sort"] == [2020, 2021]

    def test_bullet_and_band_require_their_columns(self) -> None:
        rows = [{"k": "a", "act": 5.0, "tgt": 8.0}, {"k": "b", "act": 9.0, "tgt": 8.0}]
        assert "target" in _bad({"type": "bullet", "data": rows, "x": "k", "y": "act"})
        vl = spec_to_vegalite(_ok({"type": "bullet", "data": rows, "x": "k", "y": "act", "target": "tgt"}))
        assert [layer["mark"]["type"] for layer in vl["layer"]] == ["bar", "tick"]
        series = [{"t": f"2024-0{m}-01", "v": float(m), "lo": m - 1.0, "hi": m + 1.0} for m in range(1, 5)]
        assert "y_lower" in _bad({"type": "band", "data": series, "x": "t", "y": "v"})
        band = spec_to_vegalite(_ok({"type": "band", "data": series, "x": "t", "y": "v",
                                     "y_lower": "lo", "y_upper": "hi"}))
        assert len(band["layer"]) == 2

    def test_facet_wraps_and_caps_panels(self) -> None:
        rows = [{"t": i, "v": float(i), "region": f"r{i % 15}"} for i in range(60)]
        clean = _ok({"type": "line", "data": rows, "x": "t", "y": "v", "facet": "region"})
        assert len({r["region"] for r in clean["data"]}) == 12
        vl = spec_to_vegalite(clean)
        assert vl["columns"] == 3 and vl["facet"]["field"] == "region" and "data" not in vl["spec"]
        assert "facet" in _bad({"type": "waterfall", "data": rows, "x": "t", "y": "v", "facet": "region"})

    def test_annotations_add_one_row_reference_layers(self) -> None:
        rows = [{"c": "a", "v": 3.0}, {"c": "b", "v": 5.0}]
        spec = {"type": "bar", "data": rows, "x": "c", "y": "v",
                "annotations": [{"y": 4, "label": "Target"}, {"y": 1}, {"y": 2}, {"y": 9}]}
        clean = _ok(spec)
        assert len(clean["annotations"]) == 3
        rules = [ly for ly in spec_to_vegalite(clean)["layer"] if ly.get("mark", {}).get("type") == "rule"]
        assert len(rules) == 3 and all(len(r["data"]["values"]) == 1 for r in rules)
        assert "annotations" in _bad({**spec, "annotations": [{"y": "high"}]})

    def test_hints_are_normalised(self) -> None:
        rows = [{"c": "a", "v": 1.0}, {"c": "b", "v": 2.0}]
        clean = _ok({"type": "bar", "data": rows, "x": "c", "y": "v", "size": "wide", "priority": 99,
                     "caption": "x" * 500})
        assert clean["size"] == "wide" and clean["priority"] == 10 and len(clean["caption"]) == 200
        assert "size" not in _ok({"type": "bar", "data": rows, "x": "c", "y": "v", "size": "huge"})
        tall = spec_to_vegalite(_ok({"type": "bar", "data": rows, "x": "c", "y": "v", "size": "tall"}))
        assert tall["height"] == 390


class TestVegaLite:
    def _spec(self, **extra: Any) -> dict[str, Any]:
        return {"mark": "bar", "data": {"values": [{"a": "x", "b": 1}, {"a": "y", "b": 2}]},
                "encoding": {"x": {"field": "a", "type": "nominal"}, "y": {"field": "b", "type": "quantitative"}},
                **extra}

    def test_valid_spec_lands_as_catalog_shaped_clean_spec(self) -> None:
        clean = _ok({"vega_lite": self._spec(config={"background": "red"}, title="Mine")})
        assert clean["type"] == "vega_lite" and clean["title"] == "Mine"
        vl = spec_to_vegalite(clean)
        assert vl["width"] == "container" and vl["autosize"]["type"] == "fit"
        assert "config" not in vl and vl["data"]["values"] == clean["data"]
        assert _ok(clean) == clean
        chart = _llm_chart(clean, "llm_x", None)
        assert chart.title == "Mine" and chart.spec["mark"] == "bar"

    @pytest.mark.parametrize("bad", [
        {"data": {"url": "http://evil/x.csv"}},
        {"data": {"name": "table"}},
        {"datasets": {"t": []}},
        {"params": [{"name": "p", "value": 1}]},
        {"transform": [{"calculate": "datum.b * 2", "as": "c"}]},
        {"transform": [{"filter": "datum.b > 1"}]},
        {"transform": [{"lookup": "a", "from": {"data": {"url": "x"}}}]},
        {"mark": "image"},
        {"encoding": {"x": {"field": "a", "type": "nominal", "axis": {"labelExpr": "datum.label"}}}},
        {"encoding": {"href": {"field": "a"}}},
    ])
    def test_code_or_fetch_vectors_are_rejected(self, bad: dict[str, Any]) -> None:
        spec = self._spec()
        if "data" in bad:
            spec.pop("data")
        _bad({"vega_lite": {**spec, **bad}})

    def test_unknown_field_is_rejected_but_transform_output_is_fine(self) -> None:
        assert "nope" in _bad({"vega_lite": self._spec(encoding={"x": {"field": "nope", "type": "nominal"}})})
        derived = self._spec(
            transform=[{"aggregate": [{"op": "sum", "field": "b", "as": "total"}], "groupby": ["a"]}],
            encoding={"x": {"field": "a", "type": "nominal"}, "y": {"field": "total", "type": "quantitative"}},
        )
        _ok({"vega_lite": derived})
        _bad({"vega_lite": {**derived, "transform": [{"aggregate": [{"op": "sum", "field": "b"}]}]}})

    def test_rows_are_capped(self) -> None:
        spec = self._spec()
        spec["data"] = {"values": [{"a": "x", "b": i} for i in range(MAX_CHART_ROWS + 50)]}
        clean = _ok({"vega_lite": spec})
        assert len(clean["data"]) == MAX_CHART_ROWS and clean["truncated"] is True
