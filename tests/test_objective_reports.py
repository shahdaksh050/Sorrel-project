"""Reader-visible regression checks for the sample investigations."""
from __future__ import annotations

from html import unescape
from pathlib import Path

import pandas as pd
import pytest
from ui.components.cards import render_evidence_html
from ui.components.swarm_dashboard import build_swarm_payload

from src.core.controller import AgentController
from src.core.findings import (
    Finding,
    classify_action_matrix,
    matrix_recommendations,
    report_action_matrix,
)
from src.core.profiler import profile_dataframe
from src.core.question_router import comparison_objective
from src.core.run_view import build_run_view


@pytest.mark.parametrize(("dataset", "objective", "expected"), [
    ("sample_customer_churn", "Which customer groups have the highest churn rate?",
     ("Month-to-month", "42.7%", "302/708", "3.9%", "31/792")),
    ("AirQualityUCI", "How do CO concentrations vary by time of day?",
     ("CO(GT)", "19:00", "3.73", "05:00", "0.71")),
])
def test_direct_answer_across_surfaces(
    tmp_path: Path, dataset: str, objective: str, expected: tuple[str, ...],
) -> None:
    controller = AgentController(use_llm=False, enable_rlm=False, objective=objective, output_dir=str(tmp_path))
    controller.load_dataset(str(Path(__file__).resolve().parents[1] / "data" / f"{dataset}.csv"), interactive=False)
    result = controller.analyze()
    summary = result["executive_summary"]
    assert all(value in summary.split(". ")[0] for value in expected), summary
    view = build_run_view(result, {}, objective=objective, is_sample=True)
    assert view.reasoning == summary
    assert view.headline_findings[0]["headline"] == result["insights"][0]
    assert all(v in render_evidence_html(view.headline_findings[0], business_first=True) for v in expected)
    payload = build_swarm_payload(report=result)
    assert payload["metadata"]["lead_narrative"] == summary
    assert payload["action_matrix"]["data"] == result["action_matrix"]
    for file in (f"{dataset}_report.md", "report.html"):
        report = unescape((tmp_path / "reports" / file).read_text(encoding="utf-8"))
        assert summary in report
        assert all(action in report for action in result["recommendations"])
        assert "Target partner" not in report
        assert "'hour_of_day' sensor telemetry" not in report
    assert "action_matrix" in result


@pytest.mark.parametrize("state", ["demonstrated_equivalence", "inconclusive"])
def test_null_results_never_propose_an_intervention(state: str) -> None:
    finding = Finding("comp", "test", "Tenure by partner", measure="tenure", dimension="partner",
                      effect=-0.01, evidence={"statistical_state": state})
    for representation in ([finding], [finding.to_dict()]):
        assert not any(classify_action_matrix(representation).values())


def test_bound_and_unavailable_comparisons(tmp_path: Path) -> None:
    df = pd.DataFrame({"outcome": [0, 0, 1, 1] * 30, "region": ["west", "east"] * 60})
    profile = profile_dataframe(df)
    request = comparison_objective("Compare outcome by region", profile)
    assert request is not None and request.tool_parameters() == {"measure_column": "outcome", "dimension_column": "region"}
    match = Finding("f", "segment_lift", "Observed rates", dimension="region", measure="outcome")
    assert request.matches(match)
    assert not request.matches(Finding("g", "test", "Reverse comparison", dimension="outcome", measure="region"))
    assert comparison_objective("", profile) is None
    csv = tmp_path / "regions.csv"
    df.to_csv(csv, index=False)
    controller = AgentController(use_llm=False, enable_rlm=False, use_ml=False,
                                 objective="Compare outcome by department", output_dir=str(tmp_path))
    controller.load_dataset(str(csv), interactive=False)
    result = controller.analyze()
    assert result["executive_summary"].startswith(result["objective_gap"])
    assert "could not be identified" in result["objective_gap"]
    assert "lead_finding" not in result


def test_saved_report_actions_and_explicit_empty_decision() -> None:
    finding = Finding("f", "driver", "Offer reduces abandonment in a randomized trial",
                      dimension="offer", measure="abandonment", evidence_tier="causal_action",
                      effect=-0.2, evidence={"feasibility_supported": True})
    data = [finding.to_dict()]
    matrix = report_action_matrix({}, data)
    assert matrix == classify_action_matrix([finding])
    records = matrix_recommendations(matrix)
    assert len(records) == 1 and records[0]["disposition"] == "act"
    assert records[0]["finding_ids"] == ["f"]
    assert records[0]["action"] == matrix["quick_wins"][0]["lever"]
    assert report_action_matrix({"action_matrix": {}}, data) == {}
    saved = build_run_view({"findings": data, "recommendations": ["Old advice"]}, {}, objective="", is_sample=False)
    assert saved.recommendations == (records[0]["action"],)
    data[0]["evidence"]["checks"] = {"robust": False}
    data[0]["disposition"] = "act"
    assert not report_action_matrix({}, data)["quick_wins"]
