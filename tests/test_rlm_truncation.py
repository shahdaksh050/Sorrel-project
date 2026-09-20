"""RLM sub-task resilience: a reply that is prose / truncated is retried once
with a bigger budget, siblings survive a permanent failure, and tokens spent by
failed calls still count."""
from __future__ import annotations

import json
import threading
from typing import Any

from src.core.llm_client import LLMClient
from src.rlm.engine import RLMEngine, RLMSubTask

_PROSE = "Here's a thinking process: 1. **Analyze User Input:** - Sub-Task: numerical_group_2 ..."
_GOOD = json.dumps({"status": "complete", "task_id": "x", "insights": ["ok"], "recommendations": []})


class ScriptedClient(LLMClient):
    """A real LLMClient (real truncation flag, real _parse_json, real usage
    plumbing) whose network layer is a script: task id in prompt -> replies."""

    def __init__(self, script: dict[str, list[tuple[str, bool]]]) -> None:
        super().__init__()
        self.provider = "openai"
        self.max_tokens = 4096
        self.script = script
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _dispatch(self, system_prompt: str, user_prompt: str) -> str:
        task = next(t for t in self.script if f"[{t}]" in user_prompt)
        with self._lock:
            replies = self.script[task]
            text, truncated = replies.pop(0) if len(replies) > 1 else replies[0]
            self.calls.append({
                "task": task,
                "max_tokens": self._call_max_tokens(),
                "effort": getattr(self._usage_local, "effort", None),
                "json_only": "ONE JSON object only" in user_prompt,
            })
        self._usage_local.truncated = truncated
        self._usage_local.value = {"prompt_tokens": 100, "completion_tokens": 40, "provider": "openai"}
        return text


def _engine(client: ScriptedClient) -> RLMEngine:
    return RLMEngine(client.call, "system", base_max_tokens=client.max_tokens)


def _tasks(*ids: str) -> list[RLMSubTask]:
    return [RLMSubTask(i, f"task {i}", {}) for i in ids]


def _build(task: RLMSubTask) -> str:
    return f"prompt for [{task.task_id}]"


def test_prose_truncation_retried_once_with_bigger_budget() -> None:
    client = ScriptedClient({"numerical_group_2": [(_PROSE, True), (_GOOD, False)]})
    engine = _engine(client)

    results = engine.decompose_and_invoke(_tasks("numerical_group_2"), _build)

    assert results["numerical_group_2"]["status"] == "complete"
    assert engine.last_failures == {}
    assert len(client.calls) == 2  # first attempt + exactly one retry
    first, retry = client.calls
    assert retry["max_tokens"] == 8192 > first["max_tokens"]
    assert retry["effort"] == "low"
    assert retry["json_only"] and not first["json_only"]


def test_permanent_failure_keeps_siblings_and_is_reported() -> None:
    client = ScriptedClient({
        "numerical_group_1": [(_GOOD, False)],
        "numerical_group_2": [(_PROSE, True)],  # every attempt fails
        "categorical_group": [(_GOOD, False)],
    })
    engine = _engine(client)

    results = engine.decompose_and_invoke(
        _tasks("numerical_group_1", "numerical_group_2", "categorical_group"), _build
    )

    assert list(results) == ["numerical_group_1", "categorical_group"]
    failure = engine.last_failures["numerical_group_2"]
    assert failure["status"] == "failed" and failure["attempts"] == 2
    assert "truncated" in failure["error"]
    # Bounded: the bad sub-task got one retry, not a loop.
    assert sum(c["task"] == "numerical_group_2" for c in client.calls) == 2


def test_failed_call_tokens_are_counted() -> None:
    client = ScriptedClient({"numerical_group_2": [(_PROSE, True)]})
    engine = _engine(client)

    engine.decompose_and_invoke(_tasks("numerical_group_2"), _build)

    # Two failed calls, 140 tokens each: nothing returned, all counted.
    assert engine.usage_summary()["total_tokens"] == 280


def test_record_usage_is_public() -> None:
    engine = RLMEngine(lambda s, u: {}, "system")
    engine.record_usage({"prompt_tokens": 7, "completion_tokens": 3, "provider": "openai"})
    engine.record_usage(None)
    assert engine.usage_summary()["total_tokens"] == 10


def test_plain_two_argument_callable_still_works_and_retries() -> None:
    attempts: list[str] = []

    def llm(system_prompt: str, user_prompt: str) -> dict[str, Any]:
        attempts.append(user_prompt)
        if len(attempts) == 1:
            raise ValueError("LLM returned non-JSON: prose")
        return {"status": "complete"}

    engine = RLMEngine(llm, "system")
    results = engine.decompose_and_invoke(_tasks("t"), _build)

    assert "t" in results and len(attempts) == 2
    assert "ONE JSON object only" in attempts[1]
