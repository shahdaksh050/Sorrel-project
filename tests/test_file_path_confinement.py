"""Regression tests for planner-controlled dataset path confinement."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.core.controller import AgentController
from src.core.memory import AnalysisStep, DatasetMetadata, MemorySystem
from src.tools.base import BaseTool, ToolExecutionError


class _PathTool(BaseTool):
    name = "test_path_tool"
    description = "Test-only path consumer."

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"summary": "ok"}

    def get_schema(self) -> dict[str, Any]:
        return {"file_path": {"type": "string", "required": True}}


class _RawPathTool(_PathTool):
    uses_cleaned_file = False


def _memory(raw: Path, cleaned: Path | None = None, derived: Path | None = None) -> MemorySystem:
    memory = MemorySystem()
    memory.dataset_metadata = DatasetMetadata(
        file_path=str(raw),
        row_count=1,
        column_count=1,
        columns={"value": "int64"},
        missing_values={"value": 0},
        numerical_cols=["value"],
        categorical_cols=[],
    )
    if cleaned is not None:
        memory.set_context("cleaned_file_path", str(cleaned))
    if derived is not None:
        memory.set_context(
            "derived_datasets",
            {"subset": {"path": str(derived), "columns": ["value"]}},
        )
    return memory


def test_base_tool_rejects_arbitrary_path_before_cleaning(tmp_path: Path) -> None:
    raw = tmp_path / "loaded.csv"
    arbitrary = tmp_path / "other-tenant.csv"
    memory = _memory(raw)

    with pytest.raises(ToolExecutionError, match="not the loaded dataset"):
        _PathTool().prepare_params({"file_path": str(arbitrary)}, memory, str(tmp_path))


def test_base_tool_rejects_arbitrary_path_when_clean_redirect_is_disabled(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "loaded.csv"
    memory = _memory(raw)

    with pytest.raises(ToolExecutionError, match="not the loaded dataset"):
        _RawPathTool().prepare_params(
            {"file_path": str(tmp_path / "secret.csv")}, memory, str(tmp_path)
        )


def test_base_tool_redirects_raw_to_cleaned_but_preserves_registered_derived(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "loaded.csv"
    cleaned = tmp_path / "loaded_cleaned.csv"
    derived = tmp_path / "subset.csv"
    memory = _memory(raw, cleaned, derived)
    tool = _PathTool()

    raw_params = tool.prepare_params({"file_path": str(raw)}, memory, str(tmp_path))
    derived_params = tool.prepare_params({"file_path": str(derived)}, memory, str(tmp_path))

    assert raw_params["file_path"] == str(cleaned)
    assert derived_params["file_path"] == str(derived)


def test_base_tool_rejects_path_without_any_loaded_dataset(tmp_path: Path) -> None:
    with pytest.raises(ToolExecutionError, match="not the loaded dataset"):
        _PathTool().prepare_params(
            {"file_path": str(tmp_path / "untrusted.csv")}, MemorySystem(), str(tmp_path)
        )


@pytest.fixture
def path_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentController:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    agent = AgentController(max_iterations=1, enable_rlm=False)
    agent.memory = _memory(
        tmp_path / "loaded.csv",
        tmp_path / "loaded_cleaned.csv",
        tmp_path / "subset.csv",
    )
    return agent


def test_controller_defaults_omitted_path_and_resolves_registered_alias(
    path_agent: AgentController,
) -> None:
    tool = _PathTool()

    defaulted = path_agent._resolve_file_path(tool, {})
    aliased = path_agent._resolve_file_path(tool, {"file_path": "subset"})

    assert defaulted["file_path"] == path_agent.memory.dataset_metadata.file_path
    assert aliased["file_path"].endswith("subset.csv")


@pytest.mark.parametrize("tool", [_PathTool(), _RawPathTool()])
def test_controller_rejects_arbitrary_path_for_every_tool_mode(
    path_agent: AgentController, tmp_path: Path, tool: BaseTool
) -> None:
    with pytest.raises(ValueError, match="not the loaded dataset"):
        path_agent._resolve_file_path(tool, {"file_path": str(tmp_path / "secret.csv")})


def test_controller_records_rejected_path_as_plan_error(
    path_agent: AgentController, tmp_path: Path
) -> None:
    step = AnalysisStep(
        step_number=1,
        tool_name="clean_data",
        parameters={"file_path": str(tmp_path / "secret.csv")},
        rationale="Read an untrusted path.",
    )

    path_agent._execute_steps([step])

    result = path_agent.memory.tool_results[-1]
    assert result.tool_name == "clean_data"
    assert result.status == "error"
    assert "Plan validation failed" in (result.error_message or "")
    assert "not the loaded dataset" in (result.error_message or "")
