"""Tests for operational telemetry added at the BaseTool boundary."""
from __future__ import annotations

import time
from typing import Any

from src.tools.base import BaseTool, ToolExecutionError


class _TelemetryTool(BaseTool):
    name = "telemetry_probe"
    description = "Test-only resource telemetry probe."

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        # Hold the allocation across a sampler interval so peak RSS has a
        # meaningful observation on all supported platforms.
        allocation = bytearray(2_000_000)
        time.sleep(0.06)
        return {
            "summary": "telemetry captured",
            "compute_backend": "cuda",
            "sampling": {"rows_used": 100, "sampled": True},
            "allocation_size": len(allocation),
        }

    def get_schema(self) -> dict[str, Any]:
        return {}


class _FailingTelemetryTool(BaseTool):
    name = "failing_telemetry_probe"
    description = "Test-only failure telemetry probe."

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise ToolExecutionError("expected test failure")

    def get_schema(self) -> dict[str, Any]:
        return {}


def test_base_tool_records_resource_backend_and_sampling_telemetry() -> None:
    result = _TelemetryTool().run()

    assert result.status == "success"
    assert "telemetry" not in result.output
    assert result.telemetry["wall_time_ms"] >= 0.0
    assert result.telemetry["cpu_time_ms"] is None or result.telemetry["cpu_time_ms"] >= 0.0
    assert isinstance(result.telemetry["peak_rss_bytes"], int)
    assert result.telemetry["peak_rss_bytes"] > 0
    assert result.telemetry["backend"] == "cuda"
    assert result.telemetry["sampling"] == {"rows_used": 100, "sampled": True}
    assert result.to_dict()["telemetry"] == result.telemetry


def test_base_tool_records_telemetry_when_execution_fails() -> None:
    result = _FailingTelemetryTool().run()

    assert result.status == "error"
    assert result.telemetry["backend"] is None
    assert result.telemetry["sampling"] == {}
    assert result.telemetry["wall_time_ms"] >= 0.0


class _SlowTool(BaseTool):
    name = "slow_tool"
    description = "Test-only slow tool for timeout testing."

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        time.sleep(0.5)
        return {"summary": "completed"}

    def get_schema(self) -> dict[str, Any]:
        return {}


def test_base_tool_times_out_when_exceeding_timeout_s() -> None:
    result = _SlowTool().run(timeout_s=0.05)
    assert result.status == "error"
    assert "timed out after 0.1 seconds" in result.error_message or "timed out" in result.error_message

