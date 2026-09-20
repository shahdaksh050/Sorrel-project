"""Correlation groups (one finding per set of columns moving together) and
the clean_data coverage_gap caveat for mostly-empty numeric columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.tools.data_processing import CleanDataTool, CorrelationAnalysisTool


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    driver = rng.normal(size=n)
    return pd.DataFrame({
        "a": driver + rng.normal(scale=0.2, size=n),
        "b": driver + rng.normal(scale=0.2, size=n),
        "c": driver + rng.normal(scale=0.2, size=n),
        "x": (x := rng.normal(size=n)),
        "y": 0.6 * x + rng.normal(scale=0.5, size=n),
    })


def test_group_reported_once_and_pairs_go_to_appendix(tmp_path):
    path = tmp_path / "d.csv"
    _frame().to_csv(path, index=False)
    tool = CorrelationAnalysisTool()
    out = tool.execute(str(path))
    assert [g["columns"] for g in out["correlation_groups"]] == [["a", "b", "c"]]
    found = [f for f in tool.findings(out, None, None) if f.kind == "correlation"]
    groups = [f for f in found if f.evidence.get("columns")]
    assert len(groups) == 1
    assert groups[0].headline.startswith("a, b and c move together (r between ")
    assert groups[0].layer == "analyst" and groups[0].caveats
    assert groups[0].effect == out["correlation_groups"][0]["mean_abs_r"]
    pair_layers = {frozenset((f.evidence["col_a"], f.evidence["col_b"])): f.layer for f in found if "col_a" in f.evidence}
    assert pair_layers[frozenset("xy")] == "analyst"
    assert all(pair_layers[frozenset(p)] == "appendix" for p in ("ab", "ac", "bc"))
    assert any(p.get("in_group") for p in out["top_correlations"])


def test_two_strong_columns_are_not_a_group(tmp_path):
    path = tmp_path / "d.csv"
    _frame()[["a", "b", "x", "y"]].to_csv(path, index=False)
    out = CorrelationAnalysisTool().execute(str(path))
    assert out["correlation_groups"] == []
    assert not any(p.get("in_group") for p in out["top_correlations"])


def test_coverage_gap_for_mostly_missing_column(tmp_path):
    df = _frame()
    df.loc[df.index[: int(0.6 * len(df))], "y"] = np.nan
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    tool = CleanDataTool()
    out = tool.execute(str(path))
    found = tool.findings(out, None, None)
    assert len(found) == 1 and found[0].kind == "coverage_gap"
    assert found[0].headline.startswith("y is 60% missing (160 of 400 rows usable)")
    assert found[0].evidence["columns"][0]["column"] == "y"
    assert "pairwise" in found[0].caveats[0]


def test_no_coverage_gap_when_missing_is_modest(tmp_path):
    df = _frame()
    df.loc[df.index[:100], "y"] = np.nan
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    tool = CleanDataTool()
    assert tool.findings(tool.execute(str(path)), None, None) == []
