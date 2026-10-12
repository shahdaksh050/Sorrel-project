"""Unit tests for inference tracing and token/cost accounting."""
from __future__ import annotations

from typing import Any

import pytest

from src.rlm.engine import RLMEngine


def _echo_llm(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    return {"status": "complete", "echo": user_prompt[:40]}


@pytest.fixture
def engine() -> RLMEngine:
    return RLMEngine(llm_callable=_echo_llm, system_prompt="system", max_depth=2)


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
