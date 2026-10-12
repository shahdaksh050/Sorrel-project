"""Unit tests for ScenarioSimulationTool."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.core.profiler import profile_dataframe
from src.tools.base import ToolExecutionError
from src.tools.scenario_simulation import ScenarioSimulationTool


def test_scenario_simulation_basic(tmp_path: Path) -> None:
    csv_path = tmp_path / "marketing_spend.csv"
    np.random.seed(42)
    n = 100
    spend = np.linspace(10, 100, n)
    revenue = 2.5 * spend + 50.0 + np.random.normal(0, 5, n)

    df = pd.DataFrame({"ad_spend": spend, "revenue": revenue})
    df.to_csv(csv_path, index=False)

    tool = ScenarioSimulationTool()
    res = tool.execute(file_path=str(csv_path), target_column="revenue", driver_column="ad_spend")

    assert "summary" in res
    assert res["target_column"] == "revenue"
    assert res["population_size"] == 100
    assert "driver_model" in res
    assert res["driver_model"]["driver"] == "ad_spend"
    assert res["driver_model"]["slope"] > 2.0
    assert len(res["scenarios"]) == 4

    # Check shift scenario details
    top_scenario = next(s for s in res["scenarios"] if s["shift_pct"] == 10.0)
    assert top_scenario["shift_pct"] == 10.0
    assert top_scenario["delta_target"] > 0
    assert top_scenario["ci_lower"] < top_scenario["delta_target"] < top_scenario["ci_upper"]
    assert top_scenario["projected_total_impact"] > 0
    assert "decision_table" in res
    assert "| Shift ad_spend +10%" in res["decision_table"]


def test_scenario_simulation_segment_opportunity(tmp_path: Path) -> None:
    csv_path = tmp_path / "store_sales.csv"
    np.random.seed(42)
    region = ["East"] * 60 + ["West"] * 60
    # West has higher sales than East
    sales = [50.0 + np.random.normal(0, 2) for _ in range(60)] + [80.0 + np.random.normal(0, 2) for _ in range(60)]

    df = pd.DataFrame({"region": region, "sales": sales})
    df.to_csv(csv_path, index=False)

    tool = ScenarioSimulationTool()
    res = tool.execute(file_path=str(csv_path), target_column="sales", segment_column="region")

    assert res["segment_opportunity"] is not None
    opp = res["segment_opportunity"]
    assert opp["dimension"] == "region"
    assert opp["segment"] == "East"
    assert opp["gap_per_unit"] > 25.0
    assert opp["recoverable_total"] > 1500.0


def test_scenario_simulation_findings(tmp_path: Path) -> None:
    csv_path = tmp_path / "sim_data.csv"
    np.random.seed(42)
    n = 80
    x = np.linspace(1, 50, n)
    y = 3.0 * x + 10.0
    reg = ["A"] * 40 + ["B"] * 40
    y[0:40] -= 15.0

    df = pd.DataFrame({"feature_x": x, "outcome_y": y, "tier": reg})
    df.to_csv(csv_path, index=False)

    tool = ScenarioSimulationTool()
    output = tool.execute(
        file_path=str(csv_path),
        target_column="outcome_y",
        driver_column="feature_x",
        segment_column="tier",
    )

    findings = tool.findings(output, None, None)
    assert len(findings) >= 1
    decision_scenarios = [f for f in findings if f.kind == "decision_scenario"]
    assert len(decision_scenarios) >= 1
    for f in decision_scenarios:
        assert f.source_tool == "scenario_simulation"
        assert f.measure == "outcome_y"
        assert f.confidence >= 0.70


def test_scenario_simulation_applies_to_and_default_params() -> None:
    tool = ScenarioSimulationTool()
    df = pd.DataFrame({
        "sales": [10.0 + i for i in range(25)],
        "profit": [2.0 + i for i in range(25)],
        "category": ["A", "B"] * 12 + ["A"],
    })
    prof = profile_dataframe(df)
    meta = SimpleNamespace(target_column="sales")

    score = tool.applies_to(prof, meta)  # type: ignore[arg-type]
    assert score == 0.85

    params = tool.default_params(prof, meta)  # type: ignore[arg-type]
    assert params == {"target_column": "sales"}


def test_scenario_simulation_applies_to_single_measure_with_dimension() -> None:
    tool = ScenarioSimulationTool()
    df = pd.DataFrame({
        "sales": [10.0 + i for i in range(30)],
        "region": ["East", "West"] * 15,
    })
    prof = profile_dataframe(df)

    assert tool.applies_to(prof, None) == 0.55
    assert tool.default_params(prof, None) == {"target_column": "sales"}


def test_scenario_simulation_error_handling(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(empty_csv, index=False)

    tool = ScenarioSimulationTool()
    with pytest.raises(ToolExecutionError):
        tool.execute(file_path=str(empty_csv))

    csv_non_numeric = tmp_path / "text_only.csv"
    pd.DataFrame({"name": ["Alice", "Bob"] * 20}).to_csv(csv_non_numeric, index=False)
    with pytest.raises(ToolExecutionError, match="No numeric target column"):
        tool.execute(file_path=str(csv_non_numeric))
