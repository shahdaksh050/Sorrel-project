"""
Unit tests for Dynamic Precondition Interrupt Signaling (Phase 10).
"""
from __future__ import annotations

from typing import Any

from src.core.controller import AgentController
from src.core.memory import AnalysisStep
from src.tools.base import BaseTool, ToolInterruptSignal


class DummyInterruptTool(BaseTool):
    name = "dummy_interrupt"
    description = "Test tool that raises ToolInterruptSignal."

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise ToolInterruptSignal(
            reason="Zero variance in critical feature column",
            recommended_pivot="Switch to non-parametric anomaly detection or drop constant column.",
            details={"col": "critical_feat", "variance": 0.0},
        )

    def get_schema(self) -> dict[str, Any]:
        return {}


class DummyNormalTool(BaseTool):
    name = "dummy_normal"
    description = "Test tool that succeeds normally."

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"summary": "Executed successfully", "value": 42}

    def get_schema(self) -> dict[str, Any]:
        return {}


def test_tool_interrupt_signal_packaging() -> None:
    tool = DummyInterruptTool()
    result = tool.run()
    assert result.status == "interrupt"
    assert "Zero variance" in result.error_message if result.error_message else False
    assert result.output["interrupt_reason"] == "Zero variance in critical feature column"
    assert "Switch to non-parametric" in result.output["recommended_pivot"]


def test_controller_halts_on_interrupt() -> None:
    controller = AgentController(objective="Test dynamic interrupt handling")
    interrupt_tool = DummyInterruptTool()
    normal_tool = DummyNormalTool()

    controller.tool_registry.register(interrupt_tool)
    controller.tool_registry.register(normal_tool)

    steps = [
        AnalysisStep(step_number=1, tool_name="dummy_interrupt", parameters={}, rationale="First step triggers interrupt"),
        AnalysisStep(step_number=2, tool_name="dummy_normal", parameters={}, rationale="Should be halted"),
    ]

    controller._execute_steps(steps)

    # Verify that only step 1 was executed and step 2 was halted
    executed_tools = [r.tool_name for r in controller.memory.tool_results]
    assert "dummy_interrupt" in executed_tools
    assert "dummy_normal" not in executed_tools

    # Verify interrupt signal stored in memory context
    interrupt_ctx = controller.memory.get_context("interrupt_signal")
    assert interrupt_ctx is not None
    assert "Zero variance" in interrupt_ctx["interrupt_reason"]

    # Verify hypothesis tree updated
    htree_ctx = controller.memory.get_context("hypothesis_tree")
    assert htree_ctx is not None
    refuted = [n for n in htree_ctx["nodes"].values() if n["status"] == "refuted"]
    assert len(refuted) >= 1
