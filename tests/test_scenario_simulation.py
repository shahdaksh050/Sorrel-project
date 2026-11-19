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
    assert opp["cohort_gap_total"] > 1500.0
    assert opp["recoverable_total"] is None


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
        assert out["segment_opportunity"]["recoverable_total"] is None
        assert out["segment_opportunity"]["cohort_gap_total"] > 0.0


@pytest.mark.parametrize("levels", [(10.0, 20.0), (10.0, 10.0), (0.0, 1.0)])
def test_zero_driver_shift_preserves_observed_units(tmp_path: Path, levels: tuple[float, float]) -> None:
    path = tmp_path / "costs.csv"
    pd.DataFrame({"service_cost": list(levels) * 30, "volume": np.arange(60)}).to_csv(path, index=False)
    result = ScenarioSimulationTool().run(file_path=str(path), target_column="service_cost",
                                          driver_column="volume", shifts=[0.0])
    assert result.status == "success"
    assert result.output["bounded_target"] == (levels == (0.0, 1.0))
    scenario = result.output["scenarios"][0]
    assert scenario["projected_mean"] == sum(levels) / 2
    assert scenario["delta_target"] == scenario["projected_total_impact"] == 0.0
    assert scenario["ci_lower"] == scenario["ci_upper"] == 0.0


@pytest.mark.parametrize("goal", ["minimize", "maximize", "sensitivity_only", "hold"])
def test_cohort_benchmark_respects_goal(tmp_path: Path, goal: str) -> None:
    path = tmp_path / "costs.csv"
    pd.DataFrame({"plan": ["A"] * 30 + ["B"] * 30,
                  "service_cost": [10, 11] * 15 + [20, 21] * 15}).to_csv(path, index=False)
    tool = ScenarioSimulationTool()
    result = tool.run(file_path=str(path), target_column="service_cost", segment_column="plan", target_goal=goal)
    assert result.status == "success"
    benchmark = result.output["segment_opportunity"]
    if goal in {"sensitivity_only", "hold"}:
        assert benchmark is None
        return
    minimize = goal == "minimize"
    assert benchmark["segment"] == ("B" if minimize else "A")
    assert benchmark["gap_per_unit"] == (-10.0 if minimize else 10.0)
    assert benchmark["cohort_gap_total"] == (-300.0 if minimize else 300.0)
    assert benchmark["recoverable_total"] is None
    assert f"{benchmark['cohort_gap_total']:+,.2f} across the cohort" in result.output["decision_table"]
    finding, = tool.findings(result.output, None, None)
    assert ("10.00 higher" if minimize else "10.00 lower") in finding.headline


def test_scenario_simulation_population_separation(tmp_path: Path) -> None:
    csv_path = tmp_path / "pop_sep.csv"
    n = 100
    df = pd.DataFrame({
        "revenue": np.linspace(100, 200, n),
        "marketing_cost": np.linspace(10, 50, n),
    })
    df.to_csv(csv_path, index=False)
    tool = ScenarioSimulationTool()
    res = tool.execute(
        file_path=str(csv_path),
        target_column="revenue",
        driver_column="marketing_cost",
        projection_population=50000,
    )
    assert res["estimation_population"] == 100
    assert res["projection_population"] == 50000
    top_scenario = res["scenarios"][0]
    expected_tot = round(top_scenario["delta_target"] * 50000, 2)
    assert top_scenario["projected_total_impact"] == expected_tot


def test_scenario_simulation_intensive_target_omits_total(tmp_path: Path) -> None:
    csv_path = tmp_path / "satisfaction.csv"
    n = 80
    np.random.seed(42)
    wait_time = np.linspace(5, 60, n)
    # Satisfaction score on 1-5 scale (intensive measure)
    sat = 5.0 - 0.05 * wait_time + np.random.normal(0, 0.2, n)
    df = pd.DataFrame({"support_wait_time": wait_time, "satisfaction_score": sat})
    df.to_csv(csv_path, index=False)

    tool = ScenarioSimulationTool()
    res = tool.execute(
        file_path=str(csv_path),
        target_column="satisfaction_score",
        driver_column="support_wait_time",
    )
    assert res["is_additive_target"] is False
    top_scenario = res["scenarios"][0]
    assert top_scenario["projected_total_impact"] is None
    assert top_scenario["sensitivity_total"] is None
    assert "N/A (Intensive Measure)" in res["decision_table"]
    assert "Total impact omitted" in res["summary"]

    findings = tool.findings(res, None, None)
    assert len(findings) > 0
    f = findings[0]
    assert "Total impact omitted: 'satisfaction_score' is an intensive/non-additive measure" in f.detail
    assert "Observational linear sensitivity benchmark under stated counterfactual assumption" in f.detail
    assert f.evidence["is_additive"] is False


def test_scenario_simulation_extrapolation_detection(tmp_path: Path) -> None:
    csv_path = tmp_path / "extrap.csv"
    # Tight range of driver: 10 to 12
    n = 50
    spend = np.linspace(10.0, 12.0, n)
    revenue = 2.0 * spend + 10.0
    pd.DataFrame({"spend": spend, "revenue": revenue}).to_csv(csv_path, index=False)

    tool = ScenarioSimulationTool()
    # A +50% shift on mean 11.0 is 16.5, which far exceeds driver max of 12.0
    res = tool.execute(
        file_path=str(csv_path),
        target_column="revenue",
        driver_column="spend",
        shifts=[50.0],
    )
    scenario = res["scenarios"][0]
    assert scenario["is_extrapolation"] is True
    assert scenario["simulated_driver_value"] > scenario["driver_max"]

    findings = tool.findings(res, None, None)
    assert len(findings) > 0
    assert "[Extrapolation]" in findings[0].headline
