"""
Unit tests for VariableScaleAnalysisTool (FutureScope Phase 3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.variable_methods import VariableScaleAnalysisTool


@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "wind_deg": [355.0, 0.0, 5.0, 350.0, 10.0, 358.0] * 5,
        "amount": [2**i for i in range(1, 31)],
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65] * 3,
        "visits": rng.poisson(lam=3.0, size=30),
        "share_a": [50.0] * 30,
        "share_b": [30.0] * 30,
        "share_c": [20.0] * 30,
    })
    path = str(tmp_path / "variable_sample.csv")
    df.to_csv(path, index=False)
    return path


def test_variable_scale_tool_auto(sample_csv: str) -> None:
    tool = VariableScaleAnalysisTool()
    res = tool.execute(file_path=sample_csv, analysis_type="auto")

    assert "summary" in res
    assert "results" in res
    assert "findings" in res
    assert len(res["executed_types"]) >= 3
    # Check that circular, benford, and heaping were triggered
    assert "circular" in res["results"]
    assert "benford" in res["results"]
    assert "heaping" in res["results"]


def test_variable_scale_tool_count_diagnostics(sample_csv: str) -> None:
    tool = VariableScaleAnalysisTool()
    res = tool.execute(file_path=sample_csv, analysis_type="count_diagnostics", target_column="visits")

    assert "count_diagnostics" in res["results"]
    diag = res["results"]["count_diagnostics"]
    assert diag["is_count"] is True
    assert diag["recommended_model"] == "poisson"


def test_variable_scale_tool_ordinal(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B"],
        "rating": [1, 2, 2, 5, 5, 4],
    })
    path = str(tmp_path / "ordinal_sample.csv")
    df.to_csv(path, index=False)

    tool = VariableScaleAnalysisTool()
    res = tool.execute(file_path=path, analysis_type="ordinal", group_column="group")

    assert "ordinal" in res["results"]
    rating_res = res["results"]["ordinal"]["rating"]
    assert rating_res["magnitude"] == "large"
    assert any("ordinal difference" in f["headline"].lower() for f in res["findings"])


def test_variable_scale_tool_schema() -> None:
    tool = VariableScaleAnalysisTool()
    schema = tool.get_schema()
    assert "analysis_type" in schema
    assert "file_path" in schema
    assert schema["file_path"].get("required") is True
