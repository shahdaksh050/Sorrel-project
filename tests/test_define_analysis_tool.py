"""Tests for src/tools/define_analysis_tool.py (Round 8, item 8.2).

DefineAnalysisToolTool.execute() must stay PURE (AGENTS.md layer rule,
design decision 3): it validates/smoke-tests a proposed tool and reports
status "ready"/"error", but never registers anything on a ToolRegistry and
never touches MemorySystem. That side-effect boundary is exercised together
with the controller in tests/test_controller.py; this file covers the tool
in isolation.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.tools.define_analysis_tool import DefineAnalysisToolTool
from tests.fixtures import single_column


class TestExecuteValidation:
    def test_valid_code_reports_ready(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DefineAnalysisToolTool()
        output = tool.execute(
            tool_name="row_counter",
            description="counts rows",
            params_schema={},
            code="RESULT = len(df)\n",
            file_path=str(dataset),
        )
        assert output["status"] == "ready"
        assert output["spec"]["name"] == "row_counter"
        assert "summary" in output

    def test_empty_code_reports_error(self) -> None:
        tool = DefineAnalysisToolTool()
        output = tool.execute(
            tool_name="row_counter", description="x", params_schema={}, code="",
        )
        assert output["status"] == "error"
        assert output["hint"]

    def test_static_check_failure_reports_error(self) -> None:
        tool = DefineAnalysisToolTool()
        output = tool.execute(
            tool_name="row_counter", description="x", params_schema={}, code="x = 1\n",
        )
        assert output["status"] == "error"
        assert "RESULT" in output["hint"]

    def test_disallowed_import_reports_error(self) -> None:
        tool = DefineAnalysisToolTool()
        output = tool.execute(
            tool_name="row_counter",
            description="x",
            params_schema={},
            code="import os\nRESULT = 1\n",
        )
        assert output["status"] == "error"

    def test_non_dict_schema_reports_error(self) -> None:
        tool = DefineAnalysisToolTool()
        output = tool.execute(
            tool_name="row_counter",
            description="x",
            params_schema=["not", "a", "dict"],
            code="RESULT = 1\n",
        )
        assert output["status"] == "error"
        assert "params_schema" in output["hint"]

    def test_params_schema_as_json_string_is_parsed(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DefineAnalysisToolTool()
        output = tool.execute(
            tool_name="row_counter",
            description="x",
            params_schema=json.dumps({"bump": {"type": "number", "description": "d", "required": False}}),
            code="RESULT = len(df)\n",
            file_path=str(dataset),
        )
        assert output["status"] == "ready"
        assert output["spec"]["params_schema"] == {
            "bump": {"type": "number", "description": "d", "required": False}
        }

    def test_invalid_json_string_schema_reports_error(self) -> None:
        tool = DefineAnalysisToolTool()
        output = tool.execute(
            tool_name="row_counter",
            description="x",
            params_schema="{not valid json",
            code="RESULT = 1\n",
        )
        assert output["status"] == "error"
        assert "JSON" in output["hint"]


class TestSmokeTest:
    def test_smoke_test_runs_against_real_dataset(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DefineAnalysisToolTool()
        output = tool.execute(
            tool_name="mean_tool",
            description="computes mean",
            params_schema={},
            code="RESULT = float(df['value'].mean())\n",
            file_path=str(dataset),
        )
        assert output["status"] == "ready"

    def test_smoke_test_failure_surfaces_sandbox_hint(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DefineAnalysisToolTool()
        output = tool.execute(
            tool_name="broken_tool",
            description="x",
            params_schema={},
            code="RESULT = undefined_name\n",  # passes static check, fails at runtime
            file_path=str(dataset),
        )
        assert output["status"] == "error"
        assert output["hint"]

    def test_example_params_reach_the_smoke_test(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DefineAnalysisToolTool()
        output = tool.execute(
            tool_name="bump_tool",
            description="x",
            params_schema={"bump": {"type": "number", "description": "d", "required": False}},
            code="RESULT = bump + 1\n",
            example_params={"bump": 41},
            file_path=str(dataset),
        )
        assert output["status"] == "ready"

    def test_no_file_path_is_not_ready(self) -> None:
        # file_path is declared required in get_schema() — a missing path
        # must not silently register a never-executed tool as "ready".
        tool = DefineAnalysisToolTool()
        output = tool.execute(
            tool_name="row_counter", description="x", params_schema={}, code="RESULT = 1\n",
        )
        assert output["status"] == "error"
        assert "not provided" in output["hint"]

    def test_nonexistent_file_path_is_not_ready(self, tmp_path: Path) -> None:
        tool = DefineAnalysisToolTool()
        output = tool.execute(
            tool_name="row_counter",
            description="x",
            params_schema={},
            code="RESULT = 1\n",
            file_path=str(tmp_path / "does_not_exist.csv"),
        )
        assert output["status"] == "error"
        assert "does not exist" in output["hint"]


class TestPurity:
    """Design decision 3 (AGENTS.md layer rule) — this tool must never
    register or persist anything itself; that's the controller's job after
    a "ready" status. Checked by inspecting the module's own source rather
    than behavior, since a passing behavioral test can't prove an absence."""

    def test_module_never_imports_tool_factory_or_registry(self) -> None:
        import ast
        import inspect

        import src.tools.define_analysis_tool as module

        tree = ast.parse(inspect.getsource(module))
        imported_names: set[str] = set()
        # Type-only imports (under `if TYPE_CHECKING:`) don't exist at runtime.
        type_only = {
            id(n)
            for top in ast.walk(tree)
            if isinstance(top, ast.If) and "TYPE_CHECKING" in ast.unparse(top.test)
            for n in ast.walk(top)
        }
        for node in ast.walk(tree):
            if id(node) in type_only:
                continue
            if isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
        assert not any("tool_factory" in name for name in imported_names)
        assert not any("memory" in name.lower() for name in imported_names)


class TestGetSchemaAndRegistration:
    def test_get_schema_declares_required_fields(self) -> None:
        schema = DefineAnalysisToolTool().get_schema()
        for field in ("tool_name", "description", "params_schema", "code", "file_path"):
            assert schema[field]["required"] is True
        assert schema["example_params"]["required"] is False

    def test_requires_llm_is_true(self) -> None:
        assert DefineAnalysisToolTool().requires_llm is True

    def test_tool_is_registered_in_registry(self) -> None:
        from src.core.controller import ToolRegistry

        registry = ToolRegistry()
        assert registry.has("define_analysis_tool")
