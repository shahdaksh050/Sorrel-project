"""Integration tests for the Agent Controller — full 7-stage workflow with a mock LLM."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.controller import AgentController, LLMClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path: pytest.TempPathFactory) -> str:
    rng = np.random.default_rng(42)
    n = 80
    fa = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "feature_a": fa,
            "feature_b": rng.normal(5, 2, n),
            "feature_c": rng.uniform(0, 10, n),
            "label": (fa + rng.normal(0, 0.5, n) > 0).astype(int),
        }
    )
    df.loc[::15, "feature_a"] = np.nan
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def agent(
    sample_csv: str, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> AgentController:
    monkeypatch.delenv("MAX_ITERATIONS", raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    return AgentController(max_iterations=5, enable_rlm=False)


class _ScriptedLLM:
    """Stands in for LLMClient — returns scripted responses per call."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self._responses = responses
        self.calls = 0

    def call(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        item = self._responses[idx]
        if isinstance(item, Exception):
            raise item
        return item


FINAL_RESPONSE: dict[str, Any] = {
    "status": "complete",
    "reasoning": "Done.",
    "insights": ["feature_a drives the label."],
    "recommendations": ["Ship it."],
    "best_model": "logistic_regression",
    "key_metrics": {"cv_mean": "0.8"},
}


# ---------------------------------------------------------------------------
# Step parsing
# ---------------------------------------------------------------------------

class TestParseSteps:
    def test_valid_steps_parsed(self, agent: AgentController) -> None:
        steps = agent._parse_steps(
            {"steps": [{"step_number": 1, "tool_name": "clean_data", "parameters": {"a": 1}}]}
        )
        assert len(steps) == 1
        assert steps[0].tool_name == "clean_data"

    def test_malformed_entries_skipped(self, agent: AgentController) -> None:
        steps = agent._parse_steps(
            {
                "steps": [
                    "not a dict",
                    {"parameters": {}},  # missing tool_name
                    {"tool_name": "", "parameters": {}},  # empty tool_name
                    {"tool_name": "detect_outliers", "parameters": "oops"},  # bad params
                ]
            }
        )
        assert len(steps) == 1
        assert steps[0].tool_name == "detect_outliers"
        assert steps[0].parameters == {}

    def test_missing_step_number_defaults_to_index(self, agent: AgentController) -> None:
        steps = agent._parse_steps(
            {"steps": [{"tool_name": "clean_data"}, {"tool_name": "detect_outliers"}]}
        )
        assert [s.step_number for s in steps] == [1, 2]

    def test_non_list_steps_returns_empty(self, agent: AgentController) -> None:
        assert agent._parse_steps({"steps": "garbage"}) == []


# ---------------------------------------------------------------------------
# Full workflow (mock LLM)
# ---------------------------------------------------------------------------

class TestAnalyzeWorkflow:
    def test_happy_path_runs_all_stages(self, agent: AgentController, sample_csv: str) -> None:
        plan = {
            "status": "in_progress",
            "reasoning": "Clean then explore.",
            "steps": [
                {
                    "step_number": 1,
                    "tool_name": "clean_data",
                    "parameters": {"file_path": sample_csv, "strategy": "median"},
                    "rationale": "Handle missing values.",
                },
                {
                    "step_number": 2,
                    "tool_name": "correlation_analysis",
                    "parameters": {"file_path": sample_csv, "target_column": "label"},
                    "rationale": "Explore relationships.",
                },
            ],
        }
        agent.llm_client = _ScriptedLLM([plan, FINAL_RESPONSE])  # type: ignore[assignment]

        meta = agent.load_dataset(sample_csv, target_hint="label", interactive=False)
        assert meta.task_type == "classification"

        final = agent.analyze()
        assert final["status"] == "complete"
        assert agent.memory.get_context("cleaned_file_path")

        report_dir = Path(agent._output_dir) / "reports"
        assert list(report_dir.glob("*.md")), "Stage 7 must write a Markdown report"

    def test_llm_failure_triggers_fallback_and_deterministic_final(
        self, agent: AgentController, sample_csv: str
    ) -> None:
        agent.llm_client = _ScriptedLLM([RuntimeError("api down")])  # type: ignore[assignment]
        agent.load_dataset(sample_csv, target_hint="label", interactive=False)

        final = agent.analyze()

        assert final["status"] == "complete"
        assert agent.memory.get_context("llm_error")
        # Fallback plan must have actually executed tools
        executed = {r.tool_name for r in agent.memory.tool_results}
        assert {"clean_data", "detect_outliers", "correlation_analysis"} <= executed
        # _deterministic_final only ever echoes numbers straight out of real
        # tool output — it must never trip its own verbatim-metric validator.
        assert not agent.memory.get_context("unverified_claims")
        # Deterministic synthesis must reference real results
        assert final["insights"]

    def test_repeatedly_failing_tool_is_skipped(
        self, agent: AgentController, sample_csv: str
    ) -> None:
        bad_step = {
            "status": "in_progress",
            "reasoning": "Try a bad tool call.",
            "steps": [
                {
                    "step_number": 1,
                    "tool_name": "select_statistical_test",
                    "parameters": {
                        "file_path": sample_csv,
                        "feature_column": "does_not_exist",
                        "group_column": "label",
                    },
                    "rationale": "This will fail.",
                }
            ],
        }
        agent.llm_client = _ScriptedLLM(  # type: ignore[assignment]
            [bad_step, bad_step, bad_step, FINAL_RESPONSE]
        )
        agent.load_dataset(sample_csv, target_hint="label", interactive=False)
        agent.analyze()

        statuses = [
            r.status for r in agent.memory.tool_results
            if r.tool_name == "select_statistical_test"
        ]
        assert statuses.count("error") == 2, "two real attempts allowed"
        assert "skipped" in statuses, "third attempt must be skipped, not executed"

    def test_analyze_without_dataset_raises(self, agent: AgentController) -> None:
        with pytest.raises(RuntimeError, match="No dataset loaded"):
            agent.analyze()

    def test_identical_replanned_step_is_served_from_cache(
        self, agent: AgentController, sample_csv: str
    ) -> None:
        """A step the LLM re-plans verbatim must be reused, not re-executed —
        otherwise a repeated train_model step can silently overwrite the
        model file a prior step already pointed memory context at."""
        clean_step = {
            "status": "in_progress",
            "reasoning": "Clean, then clean again (redundant replan).",
            "steps": [
                {
                    "step_number": 1,
                    "tool_name": "clean_data",
                    "parameters": {"file_path": sample_csv, "strategy": "median"},
                    "rationale": "Handle missing values.",
                }
            ],
        }
        agent.llm_client = _ScriptedLLM(  # type: ignore[assignment]
            [clean_step, clean_step, FINAL_RESPONSE]
        )
        agent.load_dataset(sample_csv, target_hint="label", interactive=False)
        agent.analyze()

        clean_results = [
            r for r in agent.memory.tool_results if r.tool_name == "clean_data"
        ]
        assert len(clean_results) == 2, "both planning cycles must record a result"
        assert clean_results[0] is clean_results[1], (
            "second occurrence must be the cached object, not a fresh execution"
        )


# ---------------------------------------------------------------------------
# Verbatim-metric validation (P0.7)
# ---------------------------------------------------------------------------

class TestVerbatimMetricValidation:
    """SYSTEM_PROMPT_CORE tells the LLM to cite only numbers that appear in
    tool results; these assert the rule is actually enforced, not just
    stated."""

    def test_number_present_in_tool_results_is_not_flagged(
        self, agent: AgentController
    ) -> None:
        from src.core.memory import ToolResult

        agent.memory.append_tool_result(
            ToolResult(
                tool_name="train_model",
                status="success",
                output={"models_trained": {"logistic_regression": {"cv_mean": 0.8123}}},
            )
        )
        final = {"insights": ["Accuracy of 0.8123 is strong."], "recommendations": []}
        flagged = agent._flag_unverified_claims(final)
        assert flagged == []
        assert final["insights"][0] == "Accuracy of 0.8123 is strong."

    def test_fabricated_number_is_flagged_and_annotated(
        self, agent: AgentController
    ) -> None:
        from src.core.memory import ToolResult

        agent.memory.append_tool_result(
            ToolResult(
                tool_name="train_model",
                status="success",
                output={"models_trained": {"logistic_regression": {"cv_mean": 0.8123}}},
            )
        )
        final = {"insights": ["The model reaches 0.999 accuracy."], "recommendations": []}
        flagged = agent._flag_unverified_claims(final)
        assert len(flagged) == 1
        assert "[unverified: 0.999" in final["insights"][0]

    def test_small_integer_counts_are_not_flagged(
        self, agent: AgentController
    ) -> None:
        """'3 models' etc. are structural counts, not cited metrics —
        flagging every small integer would drown the real signal."""
        final = {"insights": ["Trained 3 models on 2 splits."], "recommendations": []}
        flagged = agent._flag_unverified_claims(final)
        assert flagged == []

    def test_unverified_key_metric_is_flagged(self, agent: AgentController) -> None:
        final = {"insights": [], "recommendations": [], "key_metrics": {"cv_mean": "0.9999"}}
        flagged = agent._flag_unverified_claims(final)
        assert len(flagged) == 1
        assert "cv_mean" in flagged[0]

    def test_comma_formatted_large_number_is_verified(self, agent: AgentController) -> None:
        from src.core.memory import ToolResult

        agent.memory.append_tool_result(
            ToolResult(
                tool_name="detect_outliers",
                status="success",
                output={"total_outliers": 14349666, "summary": "Found 14,349,666 outliers."},
            )
        )
        final = {
            "insights": ["Global_reactive_power showed 14,349,666 outliers."],
            "recommendations": [],
        }
        flagged = agent._flag_unverified_claims(final)
        assert flagged == []
        assert "[unverified:" not in final["insights"][0]

    def test_dataset_row_count_with_comma_is_verified_in_attributed_sentence(
        self, agent: AgentController
    ) -> None:
        from src.core.memory import DatasetMetadata, ToolResult

        agent.memory.dataset_metadata = DatasetMetadata(
            file_path="dummy.csv",
            row_count=260640,
            column_count=10,
            columns=["col1"],
            missing_values={},
            numerical_cols=["col1"],
            categorical_cols=[],
            task_type="regression",
        )
        agent.memory.append_tool_result(
            ToolResult(
                tool_name="detect_outliers",
                status="success",
                output={"total_outliers": 14349, "summary": "Found 14,349 outliers."},
            )
        )
        final = {
            "insights": ["Across 260,640 records, 14,349 outliers were detected."],
            "recommendations": [],
        }
        flagged = agent._flag_unverified_claims(final)
        assert flagged == []
        assert "[unverified:" not in final["insights"][0]



# ---------------------------------------------------------------------------
# Objective, profiling, and dashboard integration
# ---------------------------------------------------------------------------

class TestObjectiveAndProfile:
    def test_objective_env_stored_in_memory(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("USER_OBJECTIVE", "what drives the label?")
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
        agent = AgentController(max_iterations=3, enable_rlm=False)
        assert agent.objective == "what drives the label?"
        assert agent.memory.get_context("user_objective") == "what drives the label?"

    def test_no_objective_leaves_context_empty(self, agent: AgentController) -> None:
        assert agent.memory.get_context("user_objective") is None

    def test_load_dataset_produces_profile(
        self, agent: AgentController, sample_csv: str
    ) -> None:
        agent.load_dataset(sample_csv, target_hint="label", interactive=False)
        assert agent.last_profile is not None
        profile_dict = agent.memory.get_context("data_profile")
        assert profile_dict["row_count"] == 80
        assert agent.memory.get_context("data_profile_summary")

    def test_objective_echoed_in_deterministic_final(
        self, sample_csv: str, tmp_path: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("USER_OBJECTIVE", "find churn drivers")
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
        agent = AgentController(max_iterations=3, enable_rlm=False)
        agent.llm_client = _ScriptedLLM([RuntimeError("api down")])  # type: ignore[assignment]
        agent.load_dataset(sample_csv, target_hint="label", interactive=False)
        final = agent.analyze()
        assert "find churn drivers" in final["reasoning"]

    def test_dashboard_json_written_after_analyze(
        self, agent: AgentController, sample_csv: str
    ) -> None:
        agent.llm_client = _ScriptedLLM([RuntimeError("api down")])  # type: ignore[assignment]
        agent.load_dataset(sample_csv, target_hint="label", interactive=False)
        agent.analyze()
        dash_path = agent.memory.get_context("dashboard_path")
        assert dash_path and Path(dash_path).exists()
        import json
        charts = json.loads(Path(dash_path).read_text(encoding="utf-8"))
        assert charts, "dashboard must contain at least one chart"
        assert all("spec" in c for c in charts)


# ---------------------------------------------------------------------------
# LLM client JSON parsing
# ---------------------------------------------------------------------------

class TestLLMClientParseJson:
    def test_plain_json(self) -> None:
        assert LLMClient._parse_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self) -> None:
        assert LLMClient._parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_truncated_json_repaired(self) -> None:
        parsed = LLMClient._parse_json('{"status": "in_progress", "steps": [{"tool_name": "clean')
        assert parsed["status"] == "in_progress"

    def test_garbage_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            LLMClient._parse_json("complete nonsense !!!")

    def test_literal_newline_in_string_value_still_parses(self) -> None:
        """Observed live against a small OpenRouter model (Round 8 testing
        pass): a structurally complete response whose "reasoning" string
        contains a literal, unescaped newline instead of `\\n`. Strict
        json.loads rejects this as "Invalid control character" even though
        every brace is balanced and no data is missing — must not be treated
        as unparseable."""
        raw = (
            '{"status": "in_progress", "reasoning": "line one\n'
            'line two", "steps": [{"step_number": 1, "tool_name": "clean_data", '
            '"parameters": {}, "rationale": "start here"}]}'
        )
        parsed = LLMClient._parse_json(raw)
        assert parsed["status"] == "in_progress"
        assert parsed["steps"][0]["tool_name"] == "clean_data"
        assert "line one" in parsed["reasoning"]


class TestMinIterationsOrchestration:
    def test_min_iterations_defaults_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MIN_ITERATIONS", raising=False)
        agent = AgentController()
        assert agent.min_iterations == 1

    def test_min_iterations_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIN_ITERATIONS", "4")
        agent = AgentController()
        assert agent.min_iterations == 4

    def test_min_iterations_reprompts_early_completion(
        self, sample_csv: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        agent = AgentController(min_iterations=2, max_iterations=5, enable_rlm=False)
        agent._output_dir = str(tmp_path / "output")

        # LLM tries to complete on iteration 1, then provides steps on reprompt, then completes on iteration 2
        step1 = {
            "status": "in_progress",
            "reasoning": "Exploring data.",
            "steps": [
                {
                    "step_number": 1,
                    "tool_name": "clean_data",
                    "parameters": {"file_path": sample_csv},
                    "rationale": "Clean initial data.",
                }
            ],
        }
        # First call: complete (early!). Second call (reprompt): step1. Third call (iter 2): FINAL_RESPONSE.
        agent.llm_client = _ScriptedLLM([FINAL_RESPONSE, step1, FINAL_RESPONSE])  # type: ignore[assignment]
        agent.load_dataset(sample_csv, target_hint="label", interactive=False)
        final = agent.analyze()
        assert final["status"] == "complete"
        assert agent.memory.iteration_count >= 2
