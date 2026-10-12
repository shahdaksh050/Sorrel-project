"""Unit tests for Autonomous Mode (when no user question/objective is supplied)."""
from pathlib import Path

import pandas as pd

from src.core.controller import AgentController, _synthesize_autonomous_objective
from src.core.memory import MemorySystem
from src.core.profiler import profile_dataframe
from src.core.prompt_manager import PromptManager
from src.core.run_view import build_run_view


def test_synthesize_autonomous_objective_picks_currency_north_star() -> None:
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "tier": ["Pro", "Free", "Pro", "Free"],
        "arr_revenue": [1200.0, 0.0, 2400.0, 0.0],
        "logins": [15, 2, 30, 4],
    })
    prof = profile_dataframe(df)
    agenda = [{"text": "Does arr_revenue differ significantly across tier?", "columns": ["arr_revenue", "tier"]}]

    obj = _synthesize_autonomous_objective(prof, agenda)
    assert "Autonomous Investigation" in obj
    assert "arr_revenue" in obj
    assert "tier" in obj
    assert "Does arr_revenue differ significantly across tier?" in obj


def test_prompt_manager_injects_autonomous_objective() -> None:
    memory = MemorySystem()
    auto_mission = "Autonomous Investigation: Uncover key drivers for 'arr_revenue' across 'tier'."
    memory.set_context("autonomous_objective", auto_mission)

    pm = PromptManager(memory, tool_descriptions={})
    block = pm._objective_block()

    assert "## Analysis Mission (Autonomous Discovery)" in block
    assert auto_mission in block
    assert "No specific question was entered" in block


def test_run_view_falls_back_to_autonomous_objective() -> None:
    context = {"autonomous_objective": "Autonomous Investigation: Drivers of sales."}
    view = build_run_view(
        final_result={"findings": [], "reasoning": "Done."},
        context=context,
        objective="",
        is_sample=False,
    )
    assert view.objective == "Autonomous Investigation: Drivers of sales."


def test_controller_load_dataset_autonomous_mode(tmp_path: Path) -> None:
    csv_file = tmp_path / "saas_metrics.csv"
    df = pd.DataFrame({
        "company": ["Acme", "Beta", "Gamma", "Delta"] * 6,
        "mrr": [100.0, 250.0, 400.0, 800.0] * 6,
        "plan": ["Basic", "Standard", "Pro", "Enterprise"] * 6,
    })
    df.to_csv(csv_file, index=False)

    controller = AgentController(output_dir=str(tmp_path / "out"), use_llm=False, use_ml=False)
    controller.load_dataset(str(csv_file))

    assert controller.autonomous_objective is not None
    assert "Autonomous Investigation" in controller.autonomous_objective
    assert "mrr" in controller.autonomous_objective
    assert controller.memory.get_context("autonomous_objective") == controller.autonomous_objective

    # Verify fallback plan includes segment_comparison on mrr
    plan = controller._build_fallback_plan()
    tool_names = [s["tool_name"] for s in plan.get("steps", [])]
    assert "segment_comparison" in tool_names
