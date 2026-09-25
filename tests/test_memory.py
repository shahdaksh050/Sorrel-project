"""Unit tests for the Memory System — Stage 1 state management."""
from __future__ import annotations

import pytest

from src.core.memory import AnalysisStep, DatasetMetadata, MemorySystem, ToolResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        file_path="tests/fixtures/test.csv",
        row_count=1_000,
        column_count=5,
        columns={"age": "int64", "income": "float64", "churn": "int64"},
        missing_values={"income": 12},
        numerical_cols=["age", "income"],
        categorical_cols=[],
        target_column="churn",
        task_type="classification",
    )


@pytest.fixture
def memory() -> MemorySystem:
    return MemorySystem()


# ---------------------------------------------------------------------------
# Metadata storage
# ---------------------------------------------------------------------------

class TestDatasetMetadata:
    def test_task_type_inferred_as_classification(self) -> None:
        meta = DatasetMetadata(
            file_path="x.csv",
            row_count=100,
            column_count=3,
            columns={"x": "float64", "y": "int64"},
            missing_values={},
            numerical_cols=["x"],
            categorical_cols=[],
            target_column="y",
        )
        assert meta.infer_task_type() == "classification"

    def test_task_type_inferred_as_clustering_when_no_target(self) -> None:
        meta = DatasetMetadata(
            file_path="x.csv",
            row_count=100,
            column_count=2,
            columns={"x": "float64"},
            missing_values={},
            numerical_cols=["x"],
            categorical_cols=[],
        )
        assert meta.infer_task_type() == "clustering"

    def test_all_null_column_never_target_fallback(self) -> None:
        meta = DatasetMetadata(
            file_path="x.csv", row_count=100, column_count=3,
            columns={"a": "float64", "b": "float64", "Unnamed: 2": "float64"},
            missing_values={"Unnamed: 2": 100},
            numerical_cols=["a", "b", "Unnamed: 2"], categorical_cols=[],
        )
        col, _ = meta.detect_target_with_confidence()
        assert col == "b"

    def test_to_prompt_string_contains_row_count(self, sample_metadata: DatasetMetadata) -> None:
        s = sample_metadata.to_prompt_string()
        assert "1,000" in s or "1000" in s

    def test_to_prompt_string_contains_target(self, sample_metadata: DatasetMetadata) -> None:
        s = sample_metadata.to_prompt_string()
        assert "churn" in s


# ---------------------------------------------------------------------------
# MemorySystem core operations
# ---------------------------------------------------------------------------

class TestMemorySystem:
    def test_store_and_retrieve_metadata(
        self, memory: MemorySystem, sample_metadata: DatasetMetadata
    ) -> None:
        memory.store_dataset_metadata(sample_metadata)
        assert memory.dataset_metadata is not None
        assert memory.dataset_metadata.row_count == 1_000

    def test_get_metadata_prompt_is_string(
        self, memory: MemorySystem, sample_metadata: DatasetMetadata
    ) -> None:
        memory.store_dataset_metadata(sample_metadata)
        prompt = memory.get_metadata_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 10

    def test_get_metadata_prompt_raises_without_dataset(self, memory: MemorySystem) -> None:
        with pytest.raises(ValueError, match="No dataset metadata"):
            memory.get_metadata_prompt()

    def test_append_tool_result(self, memory: MemorySystem) -> None:
        r = ToolResult(tool_name="clean_data", status="success", output={"summary": "ok"})
        memory.append_tool_result(r)
        assert len(memory.tool_results) == 1

    def test_get_results_summary_includes_tool_names(self, memory: MemorySystem) -> None:
        for name in ["clean_data", "detect_outliers"]:
            memory.append_tool_result(
                ToolResult(tool_name=name, status="success", output={"summary": "done"})
            )
        summary = memory.get_results_summary()
        assert "clean_data" in summary
        assert "detect_outliers" in summary

    def test_get_results_summary_empty(self, memory: MemorySystem) -> None:
        assert memory.get_results_summary() == "No tool results yet."

    def test_pending_steps_filters_incomplete(self, memory: MemorySystem) -> None:
        steps = [
            AnalysisStep(1, "clean_data", {}, "rationale"),
            AnalysisStep(2, "detect_outliers", {}, "rationale"),
        ]
        memory.store_analysis_plan(steps)
        memory.mark_step_complete(
            1, ToolResult(tool_name="clean_data", status="success", output={})
        )
        pending = memory.get_pending_steps()
        assert len(pending) == 1
        assert pending[0].step_number == 2

    def test_failed_steps_filtered(self, memory: MemorySystem) -> None:
        steps = [AnalysisStep(1, "train_model", {}, "rationale")]
        memory.store_analysis_plan(steps)
        memory.mark_step_complete(
            1, ToolResult(tool_name="train_model", status="error", output={}, error_message="fail")
        )
        assert len(memory.get_failed_steps()) == 1

    def test_generic_context_store_and_retrieve(self, memory: MemorySystem) -> None:
        memory.set_context("cleaned_path", "/tmp/data_cleaned.csv")
        assert memory.get_context("cleaned_path") == "/tmp/data_cleaned.csv"

    def test_generic_context_default_value(self, memory: MemorySystem) -> None:
        assert memory.get_context("missing", default="fallback") == "fallback"

    def test_increment_retry_counter(self, memory: MemorySystem) -> None:
        steps = [AnalysisStep(1, "train_model", {}, "rationale")]
        memory.store_analysis_plan(steps)
        memory.increment_retry(1)
        assert memory.analysis_plan[0].retry_count == 1

    def test_get_last_result_for(self, memory: MemorySystem) -> None:
        memory.append_tool_result(
            ToolResult(tool_name="clean_data", status="success", output={"v": 1})
        )
        memory.append_tool_result(
            ToolResult(tool_name="clean_data", status="success", output={"v": 2})
        )
        last = memory.get_last_result_for("clean_data")
        assert last is not None
        assert last.output["v"] == 2

    def test_get_last_result_none_for_unknown(self, memory: MemorySystem) -> None:
        assert memory.get_last_result_for("nonexistent") is None

    def test_list_context_keys(self, memory: MemorySystem) -> None:
        memory.set_context("a", 1)
        memory.set_context("b", 2)
        assert set(memory.list_context_keys()) == {"a", "b"}

    def test_clear_context_removes_key(self, memory: MemorySystem) -> None:
        memory.set_context("interrupt_signal", {"interrupt_reason": "x"})
        memory.clear_context("interrupt_signal")
        assert memory.get_context("interrupt_signal") is None
        assert "interrupt_signal" not in memory.list_context_keys()

    def test_clear_context_missing_key_is_a_no_op(self, memory: MemorySystem) -> None:
        memory.clear_context("never_set")
        assert memory.get_context("never_set") is None

    def test_save_skips_when_no_persist_path(self, memory: MemorySystem) -> None:
        # Should not raise even without a persist path
        memory.save()
