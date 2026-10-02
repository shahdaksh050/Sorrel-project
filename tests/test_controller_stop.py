"""Cooperative stop and the new-finding callback on AgentController (offline, LLM off)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.controller import AgentController
from src.core.findings import Finding


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHART_DESIGN", "false")


@pytest.fixture
def csv_path(tmp_path: Path) -> str:
    rng = np.random.default_rng(42)
    n = 120
    group = rng.choice(["a", "b", "c"], n)
    value = rng.normal(10, 2, n) + np.where(group == "a", 4.0, 0.0)  # planted group effect
    df = pd.DataFrame({"group": group, "value": value, "other": rng.normal(0, 1, n)})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def _agent(tmp_path: Path) -> AgentController:
    return AgentController(
        max_iterations=3,
        enable_rlm=False,
        use_llm=False,
        use_ml=False,
        output_dir=str(tmp_path / "out"),
    )


def _run(agent: AgentController, csv_path: str) -> dict[str, Any]:
    agent.load_dataset(csv_path, interactive=False)
    return agent.analyze()


def test_stop_before_any_work_returns_a_marked_partial_result(tmp_path: Path, csv_path: str) -> None:
    agent = _agent(tmp_path)
    agent.should_stop = lambda: True

    result = _run(agent, csv_path)

    assert result["stopped"] is True
    assert result["status"] == "complete"
    # Ingestion (stage 1) and the report step still run; no analysis tool does.
    ran = {r.tool_name for r in agent.memory.tool_results}
    assert ran <= {"ingest_dataset", "planner", "generate_report"}
    assert result["status"] == "complete"
    notes = agent.memory.get_context("degradations") or []
    assert any("stopped" in n.lower() for n in notes)


def test_stop_between_steps_keeps_finished_steps_and_skips_the_rest(tmp_path: Path, csv_path: str) -> None:
    agent = _agent(tmp_path)
    finished_steps: list[str] = []
    agent.on_step_callback = lambda tool, status, detail: (
        finished_steps.append(tool) if status in ("success", "error") else None
    )
    # Request the stop as soon as the first step has finished.
    agent.should_stop = lambda: len(finished_steps) >= 1

    result = _run(agent, csv_path)

    assert result["stopped"] is True
    assert len(finished_steps) == 1
    assert len(agent.memory.tool_results) >= 1


def test_run_without_a_stop_hook_has_no_stopped_marker(tmp_path: Path, csv_path: str) -> None:
    result = _run(_agent(tmp_path), csv_path)
    assert "stopped" not in result


def test_a_hook_that_never_asks_leaves_the_run_unchanged(tmp_path: Path, csv_path: str) -> None:
    agent = _agent(tmp_path)
    agent.should_stop = lambda: False
    result = _run(agent, csv_path)
    assert "stopped" not in result
    assert not any("stopped" in n.lower() for n in agent.memory.get_context("degradations") or [])


def test_a_raising_hook_is_ignored(tmp_path: Path, csv_path: str) -> None:
    def boom() -> bool:
        raise RuntimeError("hook failed")

    agent = _agent(tmp_path)
    agent.should_stop = boom
    result = _run(agent, csv_path)
    assert "stopped" not in result


def test_finding_callback_receives_findings_as_steps_add_them(tmp_path: Path, csv_path: str) -> None:
    agent = _agent(tmp_path)
    received: list[Finding] = []
    agent.on_finding_callback = received.append

    _run(agent, csv_path)

    assert received, "expected the planted group effect to produce at least one finding"
    assert all(isinstance(f, Finding) and f.headline for f in received)
    memory_ids = {f.finding_id for f in agent.memory.findings}
    assert {f.finding_id for f in received} <= memory_ids


def test_a_raising_finding_callback_does_not_break_the_run(tmp_path: Path, csv_path: str) -> None:
    def boom(_: Finding) -> None:
        raise RuntimeError("display failed")

    agent = _agent(tmp_path)
    agent.on_finding_callback = boom
    result = _run(agent, csv_path)
    assert result["status"] == "complete"
    assert agent.memory.findings
