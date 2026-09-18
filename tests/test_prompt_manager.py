"""Unit tests for the Prompt Manager — Stages 2/4/5/7 prompt assembly."""
from __future__ import annotations

import pytest

from src.core.memory import DatasetMetadata, MemorySystem, ToolResult
from src.core.prompt_manager import PromptManager

TOOL_DESCRIPTIONS = "tool_name: clean_data\ndescription: cleans data\nparameters:\n  - file_path"


@pytest.fixture
def memory() -> MemorySystem:
    mem = MemorySystem()
    mem.store_dataset_metadata(
        DatasetMetadata(
            file_path="data/churn.csv",
            row_count=500,
            column_count=4,
            columns={"age": "int64", "income": "float64", "plan": "object", "churn": "int64"},
            missing_values={"income": 7},
            numerical_cols=["age", "income"],
            categorical_cols=["plan"],
            target_column="churn",
            task_type="classification",
        )
    )
    return mem


@pytest.fixture
def pm(memory: MemorySystem) -> PromptManager:
    return PromptManager(memory, TOOL_DESCRIPTIONS)


class TestPromptManager:
    def test_system_prompt_embeds_tool_descriptions(self, pm: PromptManager) -> None:
        prompt = pm.get_system_prompt()
        assert "clean_data" in prompt
        assert "Form 1" in prompt and "Form 2" in prompt

    def test_initial_prompt_contains_concrete_values(self, pm: PromptManager) -> None:
        prompt = pm.get_initial_user_prompt()
        assert "data/churn.csv" in prompt
        assert '"churn"' in prompt
        assert '"classification"' in prompt

    def test_initial_prompt_contains_metadata_summary(self, pm: PromptManager) -> None:
        prompt = pm.get_initial_user_prompt()
        assert "500" in prompt

    def test_iteration_prompt_injects_cleaned_path(
        self, pm: PromptManager, memory: MemorySystem
    ) -> None:
        memory.set_context("cleaned_file_path", "/tmp/churn_cleaned.csv")
        prompt = pm.get_iteration_user_prompt()
        assert "/tmp/churn_cleaned.csv" in prompt

    def test_iteration_prompt_lists_failed_steps(
        self, pm: PromptManager, memory: MemorySystem
    ) -> None:
        memory.append_tool_result(
            ToolResult(
                tool_name="train_model",
                status="error",
                output={},
                error_message="column not found",
            )
        )
        prompt = pm.get_iteration_user_prompt()
        assert "train_model" in prompt
        assert "column not found" in prompt

    def test_final_prompt_includes_results_summary(
        self, pm: PromptManager, memory: MemorySystem
    ) -> None:
        memory.append_tool_result(
            ToolResult(tool_name="clean_data", status="success", output={"summary": "cleaned"})
        )
        prompt = pm.get_final_interpretation_prompt()
        assert "clean_data" in prompt
        assert "Form 2" in prompt

    def test_rlm_subtask_prompt_scopes_to_task(self, pm: PromptManager) -> None:
        prompt = pm.get_rlm_subtask_prompt(
            task_id="numerical_group_1",
            description="Analyse 2-feature group",
            context_summary='{"columns": ["age", "income"]}',
        )
        assert "numerical_group_1" in prompt
        assert '"age"' in prompt


class TestObjectiveInjection:
    """The user's natural-language objective must reach every reasoning prompt."""

    OBJECTIVE = "Which customers should we target with retention offers?"

    def test_objective_in_initial_prompt(
        self, pm: PromptManager, memory: MemorySystem
    ) -> None:
        memory.set_context("user_objective", self.OBJECTIVE)
        prompt = pm.get_initial_user_prompt()
        assert "User Objective" in prompt
        assert self.OBJECTIVE in prompt

    def test_objective_in_iteration_prompt(
        self, pm: PromptManager, memory: MemorySystem
    ) -> None:
        memory.set_context("user_objective", self.OBJECTIVE)
        assert self.OBJECTIVE in pm.get_iteration_user_prompt()

    def test_objective_in_final_prompt(
        self, pm: PromptManager, memory: MemorySystem
    ) -> None:
        memory.set_context("user_objective", self.OBJECTIVE)
        prompt = pm.get_final_interpretation_prompt()
        assert self.OBJECTIVE in prompt
        assert "MUST directly address" in prompt

    def test_no_objective_means_no_section(self, pm: PromptManager) -> None:
        assert "User Objective" not in pm.get_initial_user_prompt()
        assert "User Objective" not in pm.get_iteration_user_prompt()


class TestProfileInjection:
    """The automated data profile summary feeds the initial planning prompt."""

    def test_profile_summary_in_initial_prompt(
        self, pm: PromptManager, memory: MemorySystem
    ) -> None:
        memory.set_context(
            "data_profile_summary",
            "Data profile: quality score 88/100; 0 duplicate rows.",
        )
        prompt = pm.get_initial_user_prompt()
        assert "Data Profile" in prompt
        assert "quality score 88/100" in prompt

    def test_no_profile_means_no_section(self, pm: PromptManager) -> None:
        assert "Data Profile" not in pm.get_initial_user_prompt()

    def test_profile_summary_in_iteration_prompt(
        self, pm: PromptManager, memory: MemorySystem
    ) -> None:
        memory.set_context(
            "data_profile_summary",
            "Data profile: quality score 88/100; 0 duplicate rows.",
        )
        prompt = pm.get_iteration_user_prompt()
        assert "Data Profile" in prompt
        assert "quality score 88/100" in prompt

    def test_system_prompt_mentions_sandbox_and_dynamic_code(
        self, pm: PromptManager
    ) -> None:
        prompt = pm.get_system_prompt()
        assert "execute_dynamic_code" in prompt
        assert "Safe Sandboxed Code Execution" in prompt

    def test_initial_prompt_guides_phased_foundation(
        self, pm: PromptManager
    ) -> None:
        prompt = pm.get_initial_user_prompt()
        assert "Phase 1: Foundation & Reconnaissance" in prompt
        assert "1–3 targeted foundational steps" in prompt
