"""Regression tests for inference failure accounting after decomposition retirement."""
from __future__ import annotations

from typing import Any, ClassVar

from src.rlm.engine import RLMEngine


def test_failed_call_usage_is_counted() -> None:
    class MeteredError(ValueError):
        rlm_usage: ClassVar[dict[str, Any]] = {
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "provider": "openai",
        }

    def fail(system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise MeteredError("invalid JSON")

    engine = RLMEngine(fail, "system")
    try:
        engine.invoke("prompt", stage="planning")
    except MeteredError:
        pass

    assert engine.usage_summary()["total_tokens"] == 140


def test_plain_two_argument_callable_still_works() -> None:
    def llm(system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {"status": "complete"}

    engine = RLMEngine(llm, "system")

    assert engine.invoke("prompt")["status"] == "complete"
