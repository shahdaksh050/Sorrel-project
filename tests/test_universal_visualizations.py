from __future__ import annotations

import pandas as pd

from src.core.dashboard import _driver_tornado_chart, _parity_chart
from src.core.profiler import profile_dataframe


def test_parity_chart_spec() -> None:
    df = pd.DataFrame({
        "actual_sales": [10.0, 20.0, 30.0, 40.0, 50.0] * 5,
        "expected_sales": [12.0, 18.0, 35.0, 38.0, 45.0] * 5,
    })
    col_by_name = {c.name: c for c in profile_dataframe(df).columns}
    chart = _parity_chart(df, "actual_sales", "expected_sales", col_by_name, stem="sales")
    assert chart is not None
    assert chart.chart_id == "parity_sales"
    assert "Performance Benchmark" in chart.title
    spec = chart.spec
    assert "layer" in spec
    assert len(spec["layer"]) == 2  # 45-degree reference line + scatter points
    # First layer is reference line
    assert spec["layer"][0]["mark"]["type"] == "line"
    # Second layer is scatter circle
    assert spec["layer"][1]["mark"]["type"] == "circle"


def test_driver_tornado_chart_spec() -> None:
    finding = {
        "kind": "driver",
        "measure": "revenue",
        "dimension": "price",
        "effect": -0.42,
    }
    results = [
        {
            "tool_name": "regression_analysis",
            "status": "success",
            "output": {
                "coefficients": {
                    "price": -0.42,
                    "marketing": 0.35,
                    "discount": -0.18,
                }
            },
        }
    ]
    chart = _driver_tornado_chart(finding, results)
    assert chart is not None
    assert chart.chart_id == "driver_sensitivity_tornado"
    assert "Driver Sensitivity Analysis" in chart.title
    spec = chart.spec
    assert spec["mark"]["type"] == "bar"
    assert spec["encoding"]["y"]["field"] == "feature"
    assert spec["encoding"]["x"]["field"] == "effect"
    # Should sort by effect magnitude
    values = spec["data"]["values"]
    assert len(values) == 3
    assert values[0]["feature"] == "price"
    assert values[0]["direction"] == "Negative Impact"
    assert values[1]["feature"] == "marketing"
    assert values[1]["direction"] == "Positive Impact"
