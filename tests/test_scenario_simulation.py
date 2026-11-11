"""Unit tests for ScenarioSimulationTool."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.core.profiler import profile_dataframe
from src.tools.base import ToolExecutionError
from src.tools.scenario_simulation import ScenarioSimulationTool


@pytest.mark.parametrize(
    ("target_name", "expected_goal"),
    [
        ("customer_churn", "minimize"),
        ("loan_default_rate", "minimize"),
        ("fraud_rate", "minimize"),
        ("machine_failure_rate", "minimize"),
        ("patient_mortality", "minimize"),
        ("air_pollution", "minimize"),
        ("monthly_revenue", "maximize"),
        ("quality_index", "sensitivity_only"),
    ],
)
def test_target_goal_is_inferred_conservatively(
    tmp_path: Path, target_name: str, expected_goal: str
) -> None:
    n = 40
    driver = np.arange(n, dtype=float)
    target = 2.0 * driver + 10.0
    path = tmp_path / f"{target_name}.csv"
    pd.DataFrame({"driver": driver, target_name: target}).to_csv(path, index=False)

    output = ScenarioSimulationTool().execute(file_path=str(path), target_column=target_name)

    assert output["target_goal"] == expected_goal
    assert output["goal_source"] == (
        "inferred_ambiguous" if expected_goal == "sensitivity_only" else "domain_policy"
    )
    assert output["goal_rationale"]


@pytest.mark.parametrize("goal", ["sensitivity_only", "hold"])
def test_non_prescriptive_goals_do_not_create_joint_optimization(tmp_path: Path, goal: str) -> None:
    n = 60
    x_one = np.linspace(1.0, 10.0, n)
    x_two = np.linspace(3.0, 15.0, n)
    path = tmp_path / f"{goal}.csv"
    pd.DataFrame({"quality_index": x_one + x_two, "x_one": x_one, "x_two": x_two}).to_csv(
        path, index=False
    )

    output = ScenarioSimulationTool().execute(
        file_path=str(path), target_column="quality_index", target_goal=goal
    )

    assert output["target_goal"] == goal
    assert output["goal_source"] == "user"
    assert output["joint_package"] is None
    assert "non-prescriptive" in output["summary"]


def test_binary_target_never_generates_prescriptive_joint_optimum(tmp_path: Path) -> None:
    n = 60
    x_one = np.linspace(1.0, 10.0, n)
    x_two = np.linspace(10.0, 20.0, n)
    path = tmp_path / "binary.csv"
    pd.DataFrame({"fraud_flag": [0, 1] * (n // 2), "x_one": x_one, "x_two": x_two}).to_csv(
        path, index=False
    )

    output = ScenarioSimulationTool().execute(
        file_path=str(path), target_column="fraud_flag", target_goal="minimize"
    )

    assert output["is_binary_target"] is True
    assert output["joint_package"] is None
    assert "binary" in output["summary"].lower()
    assert output["bounded_target"] is True
    assert output["model_family"] == "bounded_probability_sensitivity"
    assert all(0.0 <= scenario["projected_mean"] <= 1.0 for scenario in output["scenarios"])


def test_explicit_goal_overrides_inference_and_schema_exposes_contract(tmp_path: Path) -> None:
    path = tmp_path / "revenue.csv"
    value = np.arange(40, dtype=float)
    pd.DataFrame({"revenue": value, "driver": value + 2.0}).to_csv(path, index=False)
    tool = ScenarioSimulationTool()

    output = tool.execute(file_path=str(path), target_column="revenue", target_goal="hold")

    assert output["target_goal"] == "hold"
    assert output["goal_source"] == "user"
    assert set(tool.get_schema()["target_goal"]["enum"]) == {
        "minimize", "maximize", "hold", "sensitivity_only"
    }


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


def test_adverse_target_cannot_be_maximized(tmp_path: Path) -> None:
    csv_path = tmp_path / "churn.csv"
    churn = np.linspace(0.1, 0.9, 40)
    pd.DataFrame({"churn_rate": churn, "price": churn * 10.0}).to_csv(csv_path, index=False)
    tool = ScenarioSimulationTool()
    with pytest.raises(ToolExecutionError, match="Cannot maximize adverse target"):
        tool.execute(file_path=str(csv_path), target_column="churn_rate", target_goal="maximize")


def test_age_is_protected_attribute_excluded_from_levers(tmp_path: Path) -> None:
    csv_path = tmp_path / "demographics.csv"
    np.random.seed(42)
    n = 60
    sales = np.linspace(100, 200, n)
    age = np.linspace(20, 70, n)
    discount = np.linspace(5, 25, n)
    pd.DataFrame({"sales": sales, "age": age, "discount": discount}).to_csv(csv_path, index=False)
    tool = ScenarioSimulationTool()
    out = tool.execute(file_path=str(csv_path), target_column="sales")
    assert "age" not in out["actionable_levers"]
    assert any(x["column"] == "age" and x["reason"] == "protected attribute" for x in out["excluded_levers"])


def test_compound_protected_attribute_names_excluded_from_levers(tmp_path: Path) -> None:
    csv_path = tmp_path / "compound_demo.csv"
    np.random.seed(42)
    n = 60
    sales = np.linspace(100, 200, n)
    customer_age = np.linspace(20, 70, n)
    user_gender = ["M", "F"] * 30
    discount = np.linspace(5, 25, n)
    pd.DataFrame({
        "sales": sales,
        "customer_age": customer_age,
        "user_gender": user_gender,
        "discount": discount,
    }).to_csv(csv_path, index=False)
    tool = ScenarioSimulationTool()
    out = tool.execute(file_path=str(csv_path), target_column="sales")
    assert "customer_age" not in out["actionable_levers"]
    assert any(x["column"] == "customer_age" and x["reason"] == "protected attribute" for x in out["excluded_levers"])
    assert out.get("segment_opportunity") is None


def test_inflected_adverse_target_cannot_be_maximized(tmp_path: Path) -> None:
    csv_path = tmp_path / "inflected.csv"
    churned = np.linspace(0.1, 0.9, 40)
    pd.DataFrame({"customer_churned": churned, "price": churned * 10.0}).to_csv(csv_path, index=False)
    tool = ScenarioSimulationTool()
    with pytest.raises(ToolExecutionError, match="Cannot maximize adverse target"):
        tool.execute(file_path=str(csv_path), target_column="customer_churned", target_goal="maximize")



def test_backward_compatible_aliases_present(tmp_path: Path) -> None:
    csv_path = tmp_path / "compat.csv"
    n = 60
    sales = np.linspace(100, 200, n)
    spend = np.linspace(10, 50, n)
    region = ["North"] * 30 + ["South"] * 30
    pd.DataFrame({"sales": sales, "spend": spend, "region": region}).to_csv(csv_path, index=False)
    tool = ScenarioSimulationTool()
    out = tool.execute(file_path=str(csv_path), target_column="sales", driver_column="spend", segment_column="region")
    top_scenario = out["scenarios"][0]
    assert "projected_total_impact" in top_scenario
    assert "sensitivity_total" in top_scenario
    assert top_scenario["projected_total_impact"] == top_scenario["sensitivity_total"]
    if out.get("segment_opportunity"):
        assert "recoverable_total" in out["segment_opportunity"]
        assert "cohort_gap_total" in out["segment_opportunity"]
        assert out["segment_opportunity"]["recoverable_total"] == out["segment_opportunity"]["cohort_gap_total"]

