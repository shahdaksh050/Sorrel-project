"""Direct tests for the reasoning-loop helpers split out of AgentController._analyze."""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.core.controller import AgentController
from src.core.memory import AnalysisStep, DatasetMetadata, ToolResult
from src.core.profiler import profile_dataframe


class _FakeEngine:
    """Stands in for RLMEngine.invoke: replays scripted replies or raises them."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[str] = []

    def invoke(self, prompt: str, depth: int = 0, stage: str = "") -> Any:
        self.calls.append(stage)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def agent(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> AgentController:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    return AgentController(min_iterations=2, max_iterations=4, enable_rlm=False, use_llm=False)


def _with_engine(agent: AgentController, replies: list[Any]) -> _FakeEngine:
    engine = _FakeEngine(replies)
    agent._rlm_engine = engine  # type: ignore[assignment]
    return engine


def test_non_complete_reply_passes_through(agent: AgentController) -> None:
    reply = {"status": "in_progress", "steps": []}
    assert agent._handle_completion(1, "p", "s", reply) == (reply, None)


def test_complete_at_or_after_min_iterations_is_accepted(agent: AgentController) -> None:
    reply = {"status": "complete"}
    out, final = agent._handle_completion(2, "p", "s", reply)
    assert out is reply and final is reply


def test_early_complete_reprompts_and_uses_new_plan(agent: AgentController) -> None:
    engine = _with_engine(agent, [{"status": "in_progress", "steps": [1]}])
    agent._llm_budget_exhausted = lambda: False  # type: ignore[method-assign]
    out, final = agent._handle_completion(1, "p", "s", {"status": "complete"})
    assert final is None and out["steps"] == [1]
    assert engine.calls == ["s:deepen_exploration"]


def test_early_complete_confirmed_on_reprompt_ends_loop(agent: AgentController) -> None:
    _with_engine(agent, [{"status": "complete", "summary": "done"}])
    agent._llm_budget_exhausted = lambda: False  # type: ignore[method-assign]
    out, final = agent._handle_completion(1, "p", "s", {"status": "complete"})
    assert final is out and final["summary"] == "done"


def test_reprompt_failure_accepts_original_completion(agent: AgentController) -> None:
    _with_engine(agent, [RuntimeError("boom")])
    agent._llm_budget_exhausted = lambda: False  # type: ignore[method-assign]
    first = {"status": "complete"}
    out, final = agent._handle_completion(1, "p", "s", first)
    assert out is first and final is first


def test_request_plan_returns_reply(agent: AgentController) -> None:
    _with_engine(agent, [{"status": "in_progress", "steps": []}])
    reply, final = agent._request_plan(1, "p", "s")
    assert final is None and reply["status"] == "in_progress"


def test_request_plan_retries_once_on_unusable_reply(agent: AgentController) -> None:
    engine = _with_engine(agent, [ValueError("truncated"), {"status": "in_progress", "steps": []}])
    reply, final = agent._request_plan(1, "p", "stage")
    assert final is None and reply["status"] == "in_progress"
    assert engine.calls == ["stage", "stage:retry"]


def test_request_plan_failure_after_first_cycle_synthesises_final(agent: AgentController) -> None:
    _with_engine(agent, [RuntimeError("down")])
    agent._deterministic_final = lambda: {"status": "complete", "marker": 1}  # type: ignore[method-assign]
    reply, final = agent._request_plan(2, "p", "s")
    assert reply == {} and final == {"status": "complete", "marker": 1}
    assert "RuntimeError" in str(agent.memory.get_context("llm_error"))


def test_required_preprocessing_is_restored_when_planner_omits_it(
    agent: AgentController,
) -> None:
    frame = pd.DataFrame(
        {"specification": [f"{index + 1} GB memory with feature package" for index in range(40)]}
    )
    agent.last_profile = profile_dataframe(frame)
    agent.memory.store_dataset_metadata(
        DatasetMetadata(
            file_path="catalog.csv",
            row_count=len(frame),
            column_count=1,
            columns={"specification": "str"},
            missing_values={"specification": 0},
            numerical_cols=[],
            categorical_cols=["specification"],
        )
    )
    steps = [AnalysisStep(1, "text_analysis", {}, "planner choice")]

    restored = agent._ensure_preprocessing_steps(steps)

    assert [step.tool_name for step in restored[:2]] == [
        "clean_data",
        "extract_structured_features",
    ]
    assert [step.step_number for step in restored] == [1, 2, 3]


def test_deterministic_cycle_discovers_tools_unlocked_by_extraction(
    agent: AgentController,
) -> None:
    plans = iter(
        [
            {
                "status": "in_progress",
                "deterministic": True,
                "steps": [
                    {"step_number": 1, "tool_name": "clean_data"},
                    {"step_number": 2, "tool_name": "extract_structured_features"},
                ],
            },
            {
                "status": "in_progress",
                "deterministic": True,
                "steps": [
                    {"step_number": 1, "tool_name": "clean_data"},
                    {"step_number": 2, "tool_name": "extract_structured_features"},
                    {"step_number": 3, "tool_name": "segment_comparison"},
                ],
            },
        ]
    )
    executed: list[list[str]] = []
    agent._build_fallback_plan = lambda: next(plans)  # type: ignore[method-assign]

    def execute(steps: list[AnalysisStep]) -> None:
        names = [step.tool_name for step in steps]
        executed.append(names)
        if len(executed) == 1:
            agent.memory.set_context("structured_feature_provenance", [{"derived_column": "memory_gb"}])
            for name in names:
                agent.memory.append_tool_result(
                    ToolResult(tool_name=name, status="success", output={"summary": "ok"})
                )

    agent._execute_steps = execute  # type: ignore[method-assign]
    agent._deterministic_final = lambda: {"status": "complete"}  # type: ignore[method-assign]

    class _Progress:
        def update(self, *_args: object, **_kwargs: object) -> None:
            pass

    result = agent._deterministic_cycle(_Progress(), 1)  # type: ignore[arg-type]

    assert result == {"status": "complete"}
    assert executed == [
        ["clean_data", "extract_structured_features"],
        ["segment_comparison"],
    ]


def test_reasoning_contracts_ground_domain_and_causal_roles(
    agent: AgentController,
) -> None:
    agent.objective = "Estimate the causal effect of treatment on outcome"
    agent.memory.store_dataset_metadata(
        DatasetMetadata(
            file_path="study.csv",
            row_count=500,
            column_count=4,
            columns={"treatment": "int64", "outcome": "float64", "age": "float64", "region": "str"},
            missing_values={},
            numerical_cols=["treatment", "outcome", "age"],
            categorical_cols=["region"],
        )
    )

    agent._store_reasoning_contracts({
        "domain_analysis": {
            "domain": "public health",
            "row_entity": "one patient",
            "key_measures": ["outcome", "invented"],
            "key_dimensions": ["region"],
            "domain_questions": ["How does outcome vary after treatment?"],
            "suggested_methods": ["causal adjustment"],
            "assumptions": ["Measurements are comparable."],
            "prohibited_analyses": ["Do not call correlation causation."],
        },
        "causal_design": {
            "requested": True,
            "design": "observational",
            "treatment": "treatment",
            "outcome": "outcome",
            "confounders": ["age"],
            "assumptions": ["conditional_exchangeability", "positivity", "consistency"],
        },
    })

    domain = agent.memory.get_context("domain_analysis")
    causal = agent.memory.get_context("causal_design")
    assert list(domain["key_measures"]) == ["outcome"]
    assert list(domain["key_dimensions"]) == ["region"]
    assert domain["custom_execution_policy"] == "sandbox_only"
    assert causal["status"] == "partially_identified"
    assert causal["execution_route"] == "built_in:causal_analysis"
    assert causal["randomization_verified"] is False


def test_llm_cannot_initiate_causal_analysis_without_user_intent(
    agent: AgentController,
) -> None:
    agent.objective = "Describe the dataset"
    agent.memory.store_dataset_metadata(
        DatasetMetadata(
            file_path="study.csv",
            row_count=100,
            column_count=2,
            columns={"treatment": "int64", "outcome": "float64"},
            missing_values={},
            numerical_cols=["treatment", "outcome"],
            categorical_cols=[],
        )
    )

    agent._store_reasoning_contracts({
        "causal_design": {
            "requested": True,
            "design": "randomized_experiment",
            "treatment": "treatment",
            "outcome": "outcome",
            "assumptions": ["random_assignment"],
        }
    })

    assert agent.memory.get_context("causal_design")["status"] == "not_requested"


def test_validated_observational_design_adds_guarded_estimator(
    agent: AgentController,
) -> None:
    agent.memory.set_context("causal_design", {
        "status": "partially_identified",
        "study_design": "observational_cross_section",
        "treatment": "treatment",
        "outcome": "outcome",
        "pre_treatment_confounders": ["age"],
    })

    steps = agent._ensure_causal_analysis_step([
        AnalysisStep(step_number=1, tool_name="clean_data", parameters={}, rationale="")
    ])

    assert [step.tool_name for step in steps] == ["clean_data", "causal_analysis"]
    assert steps[-1].parameters == {
        "treatment": "treatment",
        "outcome": "outcome",
        "covariates": ["age"],
        "method": "ipw",
    }


def test_validated_panel_did_design_adds_difference_in_differences_estimator(
    agent: AgentController,
) -> None:
    agent.memory.set_context("causal_design", {
        "status": "partially_identified",
        "study_design": "panel_difference_in_differences",
        "treatment": "treatment",
        "outcome": "outcome",
        "time_column": "year",
        "entity_column": "firm_id",
        "pre_treatment_confounders": ["assets"],
    })

    steps = agent._ensure_causal_analysis_step([
        AnalysisStep(step_number=1, tool_name="clean_data", parameters={}, rationale="")
    ])

    assert [step.tool_name for step in steps] == ["clean_data", "difference_in_differences"]
    assert steps[-1].parameters == {
        "treatment": "treatment",
        "outcome": "outcome",
        "time_column": "year",
        "entity_column": "firm_id",
        "covariates": ["assets"],
    }


def test_adversarial_causal_review_attaches_caveats_and_warnings(
    agent: AgentController,
) -> None:
    from src.core.findings import Finding

    agent.memory.set_context("causal_design", {
        "status": "partially_identified",
        "study_design": "observational_cross_section",
        "treatment": "treatment",
        "outcome": "outcome",
    })
    agent.memory.add_findings([
        Finding(
            finding_id="causal_outcome_treatment",
            kind="test",
            headline="Observed contrast 4.2",
            source_tool="causal_analysis",
            caveats=[],
        )
    ])

    final_result: dict[str, Any] = {}
    agent._run_adversarial_causal_review(final_result)

    causal = agent.memory.get_context("causal_design")
    assert causal["status"] == "partially_identified"
    assert "adversarial_critique" in causal
    assert causal["warnings"]
    finding = agent.memory.findings[0]
    assert any("unmeasured" in c.lower() or "observational" in c.lower() for c in finding.caveats)


def test_extract_groupby_columns() -> None:
    code1 = "g = df.groupby('platform')['price'].agg(['count', 'mean']).reset_index()"
    dim1, measure1 = AgentController._extract_groupby_columns(code1)
    assert dim1 == "platform"
    assert measure1 == ["price"]

    code2 = "res = df.groupby(['category'])['sales'].mean()"
    dim2, measure2 = AgentController._extract_groupby_columns(code2)
    assert dim2 == "category"
    assert measure2 == ["sales"]

    code3 = "df['ratio'] = df['a'] / df['b']"
    dim3, measure3 = AgentController._extract_groupby_columns(code3)
    assert dim3 is None
    assert measure3 == []

    code4 = "agg = df.groupby('brand').agg({'rating': ['count', 'mean'], 'price': 'mean', 'memory_gb': 'mean'}).reset()"
    dim4, measure4 = AgentController._extract_groupby_columns(code4)
    assert dim4 == "brand"
    assert measure4 == ["rating", "price", "memory_gb"]


def test_dynamic_code_fallback_direct_groupby(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.controller_common import _PreparedStep

    csv_file = tmp_path / "products.csv"
    brands = [f"Brand_{i}" for i in range(25)]
    data = {
        "brand": [brands[i % 25] for i in range(100)],
        "price": [10.0 + i * 2.0 for i in range(100)],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    agent = AgentController(min_iterations=1, max_iterations=2, enable_rlm=False, use_llm=False)
    agent.load_dataset(str(csv_file))

    code = "agg = df.groupby('brand').agg({'price': 'mean'}).reset()"
    step = AnalysisStep(
        step_number=1,
        tool_name="execute_dynamic_code",
        parameters={"code": code},
        rationale="group aggregation",
    )
    tool = agent.tool_registry.get("execute_dynamic_code")
    params = {"code": code, "file_path": str(csv_file)}
    prepared = _PreparedStep(
        idx=1,
        step=step,
        tool=tool,
        params=params,
        dropped_note="",
        cache_key="key",
        cached=None,
    )
    failed_result = ToolResult(
        tool_name="execute_dynamic_code",
        status="error",
        output={},
        error_message="AttributeError: 'DataFrame' object has no attribute 'reset'",
    )

    fallback = agent._try_dynamic_code_fallback(prepared, failed_result)
    assert fallback is not None
    _new_prepared, healed_result = fallback
    assert healed_result.status == "success"
    assert "[Self-healed from dynamic code fallback]" in str(healed_result.output.get("summary", ""))
    assert "groups" in healed_result.output
    assert len(agent.memory.findings) == 1
    assert agent.memory.findings[0].kind == "segment_lift"


def test_deterministic_recommendations_from_ranked_findings(agent: AgentController) -> None:
    from src.core.findings import Finding

    f1 = Finding(
        finding_id="f1",
        kind="segment_lift",
        headline="iOS devices 3.37x higher price than Android",
        measure="price",
        dimension="platform",
        level="iOS",
        effect=2.37,
        confidence=0.8,
    )
    f2 = Finding(
        finding_id="f2",
        kind="driver",
        headline="RAM is primary driver of performance",
        measure="performance",
        dimension="ram_gb",
        confidence=0.75,
    )
    agent.memory.add_findings([f1, f2])

    result = agent._deterministic_final()
    recs = result.get("recommendations", [])
    assert any("platform='iOS'" in r and "price" in r for r in recs)
    assert any("ram_gb" in r and "driver" in r for r in recs)

