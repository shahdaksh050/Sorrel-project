"""
Tests for declarative chart spec translation to interactive Plotly JSON specs.
"""
from __future__ import annotations

from src.core.chart_spec import chart_spec_to_plotly_dict, validate_chart_spec


def test_chart_spec_to_plotly_bar() -> None:
    raw_spec = {
        "type": "bar",
        "data": [
            {"category": "A", "revenue": 100},
            {"category": "B", "revenue": 200},
            {"category": "C", "revenue": 150},
        ],
        "x": "category",
        "y": "revenue",
        "title": "Revenue by Category",
    }
    clean, err = validate_chart_spec(raw_spec)
    assert clean is not None
    assert err is None

    fig_dict = chart_spec_to_plotly_dict(clean)
    assert "data" in fig_dict
    assert "layout" in fig_dict
    assert len(fig_dict["data"]) == 1
    trace = fig_dict["data"][0]
    assert trace["type"] == "bar"
    assert trace["x"] == ["B", "C", "A"] or set(trace["x"]) == {"A", "B", "C"}
    assert fig_dict["layout"]["title"]["text"] == "Revenue by Category"


def test_chart_spec_to_plotly_line_with_series() -> None:
    raw_spec = {
        "type": "line",
        "data": [
            {"month": "Jan", "sales": 50, "region": "North"},
            {"month": "Feb", "sales": 65, "region": "North"},
            {"month": "Jan", "sales": 40, "region": "South"},
            {"month": "Feb", "sales": 45, "region": "South"},
        ],
        "x": "month",
        "y": "sales",
        "color": "region",
        "title": "Monthly Regional Sales",
    }
    clean, err = validate_chart_spec(raw_spec)
    assert clean is not None
    assert err is None

    fig_dict = chart_spec_to_plotly_dict(clean)
    assert len(fig_dict["data"]) == 2
    trace_names = {t["name"] for t in fig_dict["data"]}
    assert trace_names == {"North", "South"}
    assert all(t["type"] == "scatter" and t["mode"] == "lines+markers" for t in fig_dict["data"])


def test_chart_spec_to_plotly_heatmap() -> None:
    raw_spec = {
        "type": "heatmap",
        "data": [
            {"x_var": "A", "y_var": "A", "corr": 1.0},
            {"x_var": "A", "y_var": "B", "corr": 0.5},
            {"x_var": "B", "y_var": "A", "corr": 0.5},
            {"x_var": "B", "y_var": "B", "corr": 1.0},
        ],
        "x": "x_var",
        "y": "y_var",
        "color": "corr",
        "title": "Correlation Matrix",
    }
    clean, err = validate_chart_spec(raw_spec)
    assert clean is not None
    assert err is None

    fig_dict = chart_spec_to_plotly_dict(clean)
    assert len(fig_dict["data"]) == 1
    heatmap_trace = fig_dict["data"][0]
    assert heatmap_trace["type"] == "heatmap"
    assert "z" in heatmap_trace
    assert len(heatmap_trace["z"]) == 2
