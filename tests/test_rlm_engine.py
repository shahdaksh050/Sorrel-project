"""Unit tests for the RLM Engine — recursive invocation, decomposition, and REPL state."""
from __future__ import annotations

from typing import Any

import pytest

from src.rlm.engine import REPLEnvironment, RLMEngine, RLMSubTask


def _echo_llm(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    return {"status": "complete", "echo": user_prompt[:40]}


@pytest.fixture
def engine() -> RLMEngine:
    return RLMEngine(llm_callable=_echo_llm, system_prompt="system", max_depth=2)


class TestREPLEnvironment:
    def test_set_and_get(self) -> None:
        env = REPLEnvironment()
        env.set("k", [1, 2, 3])
        assert env.get("k") == [1, 2, 3]

    def test_get_default(self) -> None:
        assert REPLEnvironment().get("missing", "fallback") == "fallback"

    def test_summary_empty(self) -> None:
        assert REPLEnvironment().summary() == "(empty)"


class TestRLMEngine:
    def test_invoke_returns_llm_dict(self, engine: RLMEngine) -> None:
        resp = engine.invoke("hello", depth=0, stage="test")
        assert resp["status"] == "complete"

    def test_invoke_records_trace(self, engine: RLMEngine) -> None:
        engine.set_iteration(3)
        engine.invoke("prompt-1", depth=0, stage="s1")
        engine.invoke("prompt-2", depth=1, stage="s2")
        trace = engine.trace
        assert len(trace) == 2
        assert trace[0].iteration == 3
        assert trace[1].depth == 1
        assert trace[1].stage == "s2"

    def test_max_depth_exceeded_raises(self, engine: RLMEngine) -> None:
        with pytest.raises(RecursionError):
            engine.invoke("too deep", depth=3, stage="overflow")

    def test_trace_property_returns_copy(self, engine: RLMEngine) -> None:
        engine.invoke("x", depth=0, stage="s")
        trace = engine.trace
        trace.clear()
        assert len(engine.trace) == 1

    def test_decompose_and_invoke_returns_per_task_results(self, engine: RLMEngine) -> None:
        tasks = [
            RLMSubTask("t1", "first", {"cols": ["a"]}),
            RLMSubTask("t2", "second", {"cols": ["b"]}),
        ]
        results = engine.decompose_and_invoke(tasks, lambda t: f"analyse {t.task_id}", depth=1)
        assert set(results.keys()) == {"t1", "t2"}
        assert all(r["status"] == "complete" for r in results.values())

    def test_decompose_stores_repl_context_and_results(self, engine: RLMEngine) -> None:
        tasks = [RLMSubTask("grp", "group", {"cols": ["a", "b"]})]
        engine.decompose_and_invoke(tasks, lambda t: t.description, depth=1)
        assert engine.repl_env.get("subtask_ctx_grp") == {"cols": ["a", "b"]}
        assert engine.repl_env.get("subtask_result_grp") is not None

    def test_decompose_runs_sub_tasks_concurrently(self) -> None:
        """P1.5: sub-tasks are independent LLM round trips and must run in
        parallel, not one at a time — this is what makes it worth having a
        thread pool at all rather than the old sequential loop."""
        import time as _time

        def _slow_llm(system_prompt: str, user_prompt: str) -> dict[str, Any]:
            _time.sleep(0.05)
            return {"status": "complete"}

        engine = RLMEngine(llm_callable=_slow_llm, system_prompt="s", max_depth=2)
        tasks = [RLMSubTask(f"t{i}", f"task {i}", {}) for i in range(5)]

        start = _time.perf_counter()
        results = engine.decompose_and_invoke(tasks, lambda t: t.description, depth=1)
        elapsed = _time.perf_counter() - start

        assert set(results.keys()) == {t.task_id for t in tasks}
        # Serial would take >= 0.25s (5 * 0.05s); a shared bounded pool
        # keeps it well under that even with scheduling overhead.
        assert elapsed < 0.20, f"expected concurrent execution, took {elapsed:.3f}s"

    def test_decompose_result_order_is_deterministic(self, engine: RLMEngine) -> None:
        """Results and REPL-stored state must not depend on which thread
        happens to finish first."""
        tasks = [RLMSubTask(f"t{i}", f"task {i}", {"n": i}) for i in range(8)]
        results = engine.decompose_and_invoke(tasks, lambda t: t.description, depth=1)
        assert list(results.keys()) == [f"t{i}" for i in range(8)]
        for i in range(8):
            assert engine.repl_env.get(f"subtask_ctx_t{i}") == {"n": i}

    def test_usage_payload_is_accounted(self) -> None:
        def metered(system_prompt: str, user_prompt: str) -> dict[str, Any]:
            return {
                "status": "complete",
                "_rlm_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "provider": "openai",
                },
            }

        engine = RLMEngine(metered, "system")
        engine.invoke("hello", stage="test")
        usage = engine.usage_summary()
        assert usage["total_tokens"] == 140
        assert usage["call_count"] == 1
        assert usage["estimated_cost_usd"] > 0

    def test_record_usage_counts_auxiliary_calls(self, engine: RLMEngine) -> None:
        engine.record_usage({"prompt_tokens": 7, "completion_tokens": 3, "provider": "openai"})
        assert engine.usage_summary()["total_tokens"] == 10
