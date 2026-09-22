"""
Unit tests for SharedAnalysisContext (FutureScope Phase 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.shared_context import SharedAnalysisContext


def test_shared_context_correlations() -> None:
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [5.0, 4.0, 3.0, 2.0, 1.0],
        "C": [1.0, 2.0, 1.0, 2.0, 1.0],
        "cat": ["x", "y", "x", "y", "x"],
    })
    ctx = SharedAnalysisContext(df=df)

    corr = ctx.get_correlation_matrix("pearson")
    assert corr.shape == (3, 3)
    assert np.isclose(corr.loc["A", "B"], -1.0)

    # Calling again should use cached version
    corr2 = ctx.get_correlation_matrix("pearson")
    assert corr is corr2


def test_shared_context_column_summaries() -> None:
    df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0, 40.0, np.nan],
        "cat": ["apple", "banana", "apple", "apple", "cherry"],
    })
    ctx = SharedAnalysisContext(df=df)

    num_summary = ctx.get_column_summary("num")
    assert num_summary["count"] == 4
    assert num_summary["missing"] == 1
    assert num_summary["mean"] == 25.0
    assert num_summary["median"] == 25.0

    cat_summary = ctx.get_column_summary("cat")
    assert cat_summary["count"] == 5
    assert cat_summary["unique"] == 3
    assert cat_summary["top_values"]["apple"] == 3


def test_shared_context_group_indices() -> None:
    df = pd.DataFrame({
        "group": ["A", "B", "A", "B", "A"],
        "val": [1, 2, 3, 4, 5],
    })
    ctx = SharedAnalysisContext(df=df)

    groups = ctx.get_group_indices("group")
    assert set(groups.keys()) == {"A", "B"}
    assert np.array_equal(groups["A"], np.array([0, 2, 4]))
    assert np.array_equal(groups["B"], np.array([1, 3]))

    ctx.clear()
    assert len(ctx._group_indices) == 0
