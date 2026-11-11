"""Deterministic EVR-09 guards for objective-led tool selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.agenda import build_agenda
from src.core.controller import AgentController
from src.core.profiler import profile_dataframe
from src.core.question_router import route_question


def test_churn_objective_has_high_decision_value_and_driver_agenda() -> None:
    frame = pd.DataFrame({
        "tenure": np.arange(80),
        "monthly_charge": np.linspace(10.0, 90.0, 80),
        "churn": [0, 1] * 40,
    })
    objective = "What drives churn and where should retention be investigated?"
    routing = route_question(objective, profile_dataframe(frame))
    agenda = build_agenda(profile_dataframe(frame), {"mode": "model", "target": "churn"}, objective)

    assert routing.primary_family == "explain"
    assert routing.expected_decision_value >= 0.8
    assert any(question.kind == "driver" and question.objective_fit >= 0.75 for question in agenda)


def test_air_quality_objective_preserves_time_series_relevance() -> None:
    frame = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=90, freq="D"),
        "pm25": np.linspace(12.0, 55.0, 90),
        "no2": np.linspace(30.0, 20.0, 90),
    })
    profile = profile_dataframe(frame)
    agenda = build_agenda(profile, {"mode": "describe"}, "Forecast PM2.5 trends for next month")

    assert profile.is_time_series
    assert any(question.suggested_tool == "time_series_analysis" and question.objective_fit >= 0.75 for question in agenda)


def test_text_only_dataset_cannot_schedule_numeric_or_generic_scenario_tools(tmp_path: Path) -> None:
    path = tmp_path / "notes.csv"
    pd.DataFrame({
        "message": ["late shipment", "happy customer", "needs follow-up"] * 15,
        "channel": ["email", "chat", "phone"] * 15,
    }).to_csv(path, index=False)
    controller = AgentController(
        output_dir=str(tmp_path / "out"), objective="Summarize customer feedback", use_llm=False, use_ml=True
    )
    controller.load_dataset(str(path))
    planned = {step["tool_name"] for step in controller._build_fallback_plan()["steps"]}
    llm_steps = controller._parse_steps({"steps": [{"tool_name": "cluster_data"}, {"tool_name": "correlation_analysis"}]})

    assert not planned & {"correlation_analysis", "cluster_data", "scenario_simulation", "select_statistical_test"}
    assert not llm_steps
