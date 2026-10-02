"""Direct tests for the reasoning-loop helpers split out of AgentController._analyze."""
from __future__ import annotations

from typing import Any

import pytest

from src.core.controller import AgentController


class _FakeEngine:
    """Stands in for RLMEngine.invoke: replays scripted replies or raises them."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[str] = []

    def invoke(self, prompt: str, depth: int = 0, stage: str = "") -> Any:
        self.calls.append(stage)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def agent(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> AgentController:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    return AgentController(min_iterations=2, max_iterations=4, enable_rlm=False, use_llm=False)


def _with_engine(agent: AgentController, replies: list[Any]) -> _FakeEngine:
    engine = _FakeEngine(replies)
    agent._rlm_engine = engine  # type: ignore[assignment]
    return engine


def test_non_complete_reply_passes_through(agent: AgentController) -> None:
    reply = {"status": "in_progress", "steps": []}
    assert agent._handle_completion(1, "p", "s", reply) == (reply, None)


def test_complete_at_or_after_min_iterations_is_accepted(agent: AgentController) -> None:
    reply = {"status": "complete"}
    out, final = agent._handle_completion(2, "p", "s", reply)
    assert out is reply and final is reply


def test_early_complete_reprompts_and_uses_new_plan(agent: AgentController) -> None:
    engine = _with_engine(agent, [{"status": "in_progress", "steps": [1]}])
    agent._llm_budget_exhausted = lambda: False  # type: ignore[method-assign]
    out, final = agent._handle_completion(1, "p", "s", {"status": "complete"})
    assert final is None and out["steps"] == [1]
    assert engine.calls == ["s:deepen_exploration"]


def test_early_complete_confirmed_on_reprompt_ends_loop(agent: AgentController) -> None:
    _with_engine(agent, [{"status": "complete", "summary": "done"}])
    agent._llm_budget_exhausted = lambda: False  # type: ignore[method-assign]
    out, final = agent._handle_completion(1, "p", "s", {"status": "complete"})
    assert final is out and final["summary"] == "done"


def test_reprompt_failure_accepts_original_completion(agent: AgentController) -> None:
    _with_engine(agent, [RuntimeError("boom")])
    agent._llm_budget_exhausted = lambda: False  # type: ignore[method-assign]
    first = {"status": "complete"}
    out, final = agent._handle_completion(1, "p", "s", first)
    assert out is first and final is first


def test_request_plan_returns_reply(agent: AgentController) -> None:
    _with_engine(agent, [{"status": "in_progress", "steps": []}])
    reply, final = agent._request_plan(1, "p", "s")
    assert final is None and reply["status"] == "in_progress"


def test_request_plan_retries_once_on_unusable_reply(agent: AgentController) -> None:
    engine = _with_engine(agent, [ValueError("truncated"), {"status": "in_progress", "steps": []}])
    reply, final = agent._request_plan(1, "p", "stage")
    assert final is None and reply["status"] == "in_progress"
    assert engine.calls == ["stage", "stage:retry"]


def test_request_plan_failure_after_first_cycle_synthesises_final(agent: AgentController) -> None:
    _with_engine(agent, [RuntimeError("down")])
    agent._deterministic_final = lambda: {"status": "complete", "marker": 1}  # type: ignore[method-assign]
    reply, final = agent._request_plan(2, "p", "s")
    assert reply == {} and final == {"status": "complete", "marker": 1}
    assert "RuntimeError" in str(agent.memory.get_context("llm_error"))
