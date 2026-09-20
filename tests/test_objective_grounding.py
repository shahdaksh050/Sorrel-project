"""Objective grounding: pick_measures ordering and target detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.memory import DatasetMetadata
from src.core.profiler import ground_objective_column, pick_measures, profile_dataframe
from src.tools.anomaly import AnomalyAnalysisTool

OBJECTIVE = "Which pollutants move together, and what drives CO levels?"


def _frame(order: list[str] | None = None, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=n, freq="h"),
        "CO(GT)": rng.gamma(4.0, 0.5, n),
        "PT08.S1(CO)": 1100 + rng.normal(0, 150, n),
        "AH": 0.9 + rng.normal(0, 0.1, n),
        "T": 12 + rng.normal(0, 4, n),
    })
    return df[order] if order else df


def _metadata(df: pd.DataFrame) -> DatasetMetadata:
    return DatasetMetadata(
        file_path="x.csv", row_count=len(df), column_count=df.shape[1],
        columns={c: str(t) for c, t in df.dtypes.items()},
        missing_values=dict.fromkeys(df.columns, 0),
        numerical_cols=[c for c in df.columns if c != "Date"], categorical_cols=[],
    )


def test_objective_grounds_plain_name_before_embedded_token() -> None:
    profile = profile_dataframe(_frame())
    assert ground_objective_column(profile, "what drives CO levels") == "CO(GT)"
    assert pick_measures(profile, "what drives CO levels")[0].name == "CO(GT)"
    assert ground_objective_column(profile, "pollutants move together") is None


def test_no_objective_order_is_stable_across_column_shuffles() -> None:
    cols = ["CO(GT)", "PT08.S1(CO)", "AH", "T"]
    ref = [c.name for c in pick_measures(profile_dataframe(_frame()))]
    for order in (cols[::-1], ["AH", "T", "CO(GT)", "PT08.S1(CO)"]):
        got = [c.name for c in pick_measures(profile_dataframe(_frame(["Date", *order])))]
        assert got == ref


def test_incomplete_column_ranks_below_complete_ones() -> None:
    df = _frame()
    df.loc[df.index[:150], "PT08.S1(CO)"] = np.nan   # most variable, but 50% missing
    names = [c.name for c in pick_measures(profile_dataframe(df))]
    assert names[-1] == "PT08.S1(CO)"


def test_driver_objective_raises_target_confidence() -> None:
    meta = _metadata(_frame())
    col, conf = meta.detect_target_with_confidence("what drives CO levels")
    assert col == "CO(GT)" and conf > 0.40
    assert meta.detect_target_with_confidence()[1] <= 0.40
    # Naming a column without driver phrasing does not create a target.
    assert meta.detect_target_with_confidence("show CO levels")[1] <= 0.40


def test_anomaly_default_measure_follows_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = profile_dataframe(_frame())
    monkeypatch.setenv("USER_OBJECTIVE", "what drives CO levels")
    assert AnomalyAnalysisTool().default_params(profile, None)["value_column"] == "CO(GT)"
