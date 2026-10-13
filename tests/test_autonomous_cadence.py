from __future__ import annotations

import pandas as pd
import pytest

from src.core.controller import AgentController
from src.core.memory import AnalysisStep, DatasetMetadata, ToolResult
from src.core.profiler import profile_dataframe


def test_autonomous_cadence_cycle1_and_cycle2(tmp_path: pytest.TempPathFactory) -> None:
    csv_path = tmp_path / "cadence_test.csv"  # type: ignore[operator]
    df = pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0] * 15,
        "spend": [10.0, 20.0, 30.0, 40.0] * 15,
        "region": ["North", "South", "East", "West"] * 15,
    })
    df.to_csv(csv_path, index=False)

    controller = AgentController(
        output_dir=str(tmp_path),
        use_ml=True,
        use_llm=False,
    )
    meta = DatasetMetadata(
        file_path=str(csv_path),
        row_count=len(df),
        column_count=len(df.columns),
        columns={c: str(df[c].dtype) for c in df.columns},
        missing_values={c: int(df[c].isna().sum()) for c in df.columns},
        numerical_cols=["revenue", "spend"],
        categorical_cols=["region"],
    )
    controller.memory.dataset_metadata = meta
    controller.last_profile = profile_dataframe(df)

    # Agenda with segment question
    controller.memory.set_context("question_agenda", [
        {"suggested_tool": "segment_comparison", "columns": ["revenue", "region"], "text": "Test segment"}
    ])

    # In Cycle 1: ensure segment_comparison is anchored
    controller.memory.iteration_count = 1
    steps_c1: list[AnalysisStep] = []
    anchored_c1 = controller._ensure_agenda_anchor_steps(steps_c1)
    tool_names_c1 = [s.tool_name for s in anchored_c1]
    assert "segment_comparison" in tool_names_c1

    # In Cycle 2: simulate Cycle 1 tools succeeded, ensure cluster_data & scenario_simulation are anchored
    controller.memory.iteration_count = 2
    controller.memory.append_tool_result(ToolResult(
        tool_name="clean_data", status="success", output={}, iteration=1
    ))
    controller.memory.append_tool_result(ToolResult(
        tool_name="segment_comparison", status="success", output={}, iteration=1
    ))

    steps_c2: list[AnalysisStep] = [
        AnalysisStep(step_number=1, tool_name="regression_analysis", parameters={"target_column": "revenue"}, rationale="Analyze drivers")
    ]
    anchored_c2 = controller._ensure_agenda_anchor_steps(steps_c2)
    tool_names_c2 = [s.tool_name for s in anchored_c2]

    # Archetype Clustering and What-If Simulation must both be present
    assert "cluster_data" in tool_names_c2
    assert "scenario_simulation" in tool_names_c2
