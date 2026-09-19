"""Tests for src/tools/dynamic_code.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.base import ToolExecutionError
from src.tools.dynamic_code import DynamicCodeExecutionTool
from tests.fixtures import single_column


class TestDynamicCodeExecutionTool:
    def test_execute_success(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DynamicCodeExecutionTool()
        output = tool.execute(
            file_path=str(dataset), code="RESULT = float(df['value'].mean())\n"
        )
        assert "summary" in output
        assert output["status"] == "ok"
        assert isinstance(output["result"], float)

    def test_execute_raises_on_sandbox_failure(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DynamicCodeExecutionTool()
        with pytest.raises(ToolExecutionError, match="static_check"):
            tool.execute(file_path=str(dataset), code="x = 1\n")  # no RESULT

    def test_run_records_sandbox_failure_as_tool_error(self, tmp_path: Path) -> None:
        # The controller's retry budget and failure accounting key on
        # ToolResult.status, so failing generated code must not read as success.
        dataset = single_column(tmp_path)
        tool = DynamicCodeExecutionTool()
        tool_result = tool.run(file_path=str(dataset), code="x = 1\n")
        assert tool_result.status == "error"
        assert "static_check" in (tool_result.error_message or "")

    def test_empty_code_raises(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DynamicCodeExecutionTool()
        with pytest.raises(ToolExecutionError):
            tool.execute(file_path=str(dataset), code="")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        tool = DynamicCodeExecutionTool()
        with pytest.raises(ToolExecutionError):
            tool.execute(file_path=str(tmp_path / "nope.csv"), code="RESULT = 1\n")

    def test_get_schema(self) -> None:
        tool = DynamicCodeExecutionTool()
        schema = tool.get_schema()
        assert schema["file_path"]["required"] is True
        assert schema["code"]["required"] is True

    def test_prepare_params_extracts_prior_results(self) -> None:
        from src.core.memory import MemorySystem, ToolResult

        tool = DynamicCodeExecutionTool()
        memory = MemorySystem()
        memory.append_tool_result(
            ToolResult(
                tool_name="select_statistical_test",
                status="success",
                output={"p_value": 0.005, "significant": True},
            )
        )
        params = tool.prepare_params(
            params={"file_path": "dummy.csv", "code": "RESULT = 1"},
            memory=memory,
            output_root="output",
        )
        assert "prior_results" in params
        assert "select_statistical_test" in params["prior_results"]
        assert params["prior_results"]["select_statistical_test"]["p_value"] == 0.005


class TestRegistration:
    def test_tool_is_registered(self) -> None:
        from src.core.controller import ToolRegistry

        registry = ToolRegistry()
        assert registry.has("execute_dynamic_code")
