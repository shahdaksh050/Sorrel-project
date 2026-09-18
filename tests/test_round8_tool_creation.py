"""
Round 8 integration — LLM Sandbox mode, dynamic tool creation/modification.

Exercises the full loop across all four owners' code together (sandbox,
tool_factory/generated_tool/define_analysis_tool, and the controller/memory/
prompt_manager wiring): define a tool -> controller registers it -> it's
callable by name -> its findings reach the shared finding bus -> it stays
invisible to the deterministic planner -> a built-in name collision is
rejected. This formalizes the manual verification script run during
implementation (see IMPROVEMENTS.md's Round 8 "Status" section) as a real,
repeatable test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.controller import AgentController
from src.core.memory import ToolResult


@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "amount": rng.normal(50, 10, n),
        "region": rng.choice(["East", "West"], n),
    })
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def agent(sample_csv: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentController:
    monkeypatch.delenv("MAX_ITERATIONS", raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    controller = AgentController(use_llm=False, use_ml=False, max_iterations=1)
    controller.load_dataset(sample_csv, interactive=False)
    return controller


DEFINE_CODE = (
    "n = int(df.shape[0])\n"
    "RESULT = {'row_count': n}\n"
    "FINDING = {'headline': f'{n} rows in this dataset', "
    "'detail': 'generated tool test', 'evidence': {'row_count': n}}\n"
)


class TestFullLifecycle:
    def test_define_then_register_then_call(self, agent: AgentController, sample_csv: str) -> None:
        define_tool = agent.tool_registry.get("define_analysis_tool")
        define_result = define_tool.run(
            tool_name="row_count_tool",
            description="counts rows",
            params_schema={},
            code=DEFINE_CODE,
            file_path=sample_csv,
        )
        assert define_result.status == "success"
        assert define_result.output["status"] == "ready"

        # The controller's post-step hook is what actually registers a
        # proposed tool (AGENTS.md layer rule — define_analysis_tool's own
        # execute() is pure and never does this itself).
        assert not agent.tool_registry.has("row_count_tool")
        agent._maybe_register_generated_tool(define_result)
        assert define_result.status == "success"  # unchanged — registration succeeded
        assert agent.tool_registry.has("row_count_tool")

        gen_tool = agent.tool_registry.get("row_count_tool")
        call_result = gen_tool.run(file_path=sample_csv)
        assert call_result.status == "success"
        assert call_result.output["status"] == "ok"
        assert call_result.output["result"]["row_count"] == 60

    def test_findings_reach_the_shared_finding_bus(self, agent: AgentController, sample_csv: str) -> None:
        define_tool = agent.tool_registry.get("define_analysis_tool")
        define_result = define_tool.run(
            tool_name="row_count_tool", description="counts rows",
            params_schema={}, code=DEFINE_CODE, file_path=sample_csv,
        )
        agent._maybe_register_generated_tool(define_result)
        gen_tool = agent.tool_registry.get("row_count_tool")
        call_result = gen_tool.run(file_path=sample_csv)

        new_findings = gen_tool.findings(call_result.output, agent.last_profile, agent.memory.dataset_metadata)
        assert len(new_findings) == 1
        agent.memory.add_findings(new_findings)

        ranked = agent.memory.ranked_findings(suppress_trivial=False)
        assert any(f.source_tool == "row_count_tool" for f in ranked)

    def test_registered_tool_is_recorded_in_memory(self, agent: AgentController, sample_csv: str) -> None:
        define_tool = agent.tool_registry.get("define_analysis_tool")
        define_result = define_tool.run(
            tool_name="row_count_tool", description="counts rows",
            params_schema={}, code=DEFINE_CODE, file_path=sample_csv,
        )
        agent._maybe_register_generated_tool(define_result)

        recorded = agent.memory.list_generated_tools()
        assert len(recorded) == 1
        assert recorded[0]["name"] == "row_count_tool"

    def test_spec_persisted_to_disk(self, agent: AgentController, sample_csv: str) -> None:
        define_tool = agent.tool_registry.get("define_analysis_tool")
        define_result = define_tool.run(
            tool_name="row_count_tool", description="counts rows",
            params_schema={}, code=DEFINE_CODE, file_path=sample_csv,
        )
        agent._maybe_register_generated_tool(define_result)

        spec_path = Path(agent._output_dir) / "generated_tools" / "row_count_tool.json"
        assert spec_path.exists()


class TestSafetyInvariants:
    def test_deterministic_planner_never_sees_generated_or_meta_tools(
        self, agent: AgentController, sample_csv: str
    ) -> None:
        define_tool = agent.tool_registry.get("define_analysis_tool")
        define_result = define_tool.run(
            tool_name="row_count_tool", description="counts rows",
            params_schema={}, code=DEFINE_CODE, file_path=sample_csv,
        )
        agent._maybe_register_generated_tool(define_result)

        candidates = agent.tool_registry.candidate_tools(
            agent.last_profile, agent.memory.dataset_metadata, use_ml=False, use_llm=False
        )
        names = [t.name for t in candidates]
        assert "row_count_tool" not in names
        assert "define_analysis_tool" not in names

    def test_builtin_name_collision_rejected_and_not_registered(
        self, agent: AgentController, sample_csv: str
    ) -> None:
        define_tool = agent.tool_registry.get("define_analysis_tool")
        collide_result = define_tool.run(
            tool_name="clean_data",
            description="malicious overwrite attempt",
            params_schema={},
            code="RESULT = 1\n",
            file_path=sample_csv,
        )
        original_clean_data = agent.tool_registry.get("clean_data")
        agent._maybe_register_generated_tool(collide_result)

        assert collide_result.status == "error"
        assert agent.tool_registry.get("clean_data") is original_clean_data

    def test_invalid_spec_missing_name_reported_as_error(self, agent: AgentController) -> None:
        fake_result = ToolResult(
            tool_name="define_analysis_tool",
            status="success",
            output={"status": "ready", "spec": {"name": "", "description": "x", "params_schema": {}, "code": "RESULT=1\n"}},
        )
        agent._maybe_register_generated_tool(fake_result)
        assert fake_result.status == "error"

    def test_non_ready_status_is_a_no_op(self, agent: AgentController) -> None:
        fake_result = ToolResult(
            tool_name="define_analysis_tool", status="success",
            output={"status": "error", "spec": {}},
        )
        agent._maybe_register_generated_tool(fake_result)
        assert fake_result.status == "success"  # untouched — nothing to register

    def test_modification_bumps_version(self, agent: AgentController, sample_csv: str) -> None:
        define_tool = agent.tool_registry.get("define_analysis_tool")

        first = define_tool.run(
            tool_name="row_count_tool", description="v1",
            params_schema={}, code=DEFINE_CODE, file_path=sample_csv,
        )
        agent._maybe_register_generated_tool(first)

        second = define_tool.run(
            tool_name="row_count_tool", description="v2 (fixed)",
            params_schema={}, code=DEFINE_CODE, file_path=sample_csv,
        )
        agent._maybe_register_generated_tool(second)
        assert second.status == "success"

        recorded = agent.memory.list_generated_tools()
        assert len(recorded) == 1  # replaced, not duplicated
        assert recorded[0]["version"] == 2
        assert recorded[0]["description"] == "v2 (fixed)"


class TestPromptWiring:
    def test_iteration_prompt_lists_generated_tools(self, agent: AgentController, sample_csv: str) -> None:
        from src.core.prompt_manager import PromptManager

        define_tool = agent.tool_registry.get("define_analysis_tool")
        define_result = define_tool.run(
            tool_name="row_count_tool", description="counts all rows in the dataset",
            params_schema={}, code=DEFINE_CODE, file_path=sample_csv,
        )
        agent._maybe_register_generated_tool(define_result)

        pm = PromptManager(agent.memory, tool_descriptions="(irrelevant for this test)")
        prompt = pm.get_iteration_user_prompt()
        assert "row_count_tool" in prompt
        assert "counts all rows in the dataset" in prompt

    def test_iteration_prompt_omits_block_when_no_generated_tools(self, agent: AgentController) -> None:
        from src.core.prompt_manager import PromptManager

        pm = PromptManager(agent.memory, tool_descriptions="(irrelevant for this test)")
        prompt = pm.get_iteration_user_prompt()
        assert "Tools You Created During This Run" not in prompt

    def test_system_prompt_explains_sandbox_contract_once(self, agent: AgentController) -> None:
        from src.core.prompt_manager import PromptManager

        pm = PromptManager(agent.memory, tool_descriptions="(irrelevant for this test)")
        system_prompt = pm.get_system_prompt()
        assert "define_analysis_tool" in system_prompt
        assert "RESULT" in system_prompt
        assert "FINDING" in system_prompt


class TestMemoryApi:
    def test_add_generated_tool_replaces_same_name(self) -> None:
        from src.core.memory import MemorySystem

        memory = MemorySystem()
        memory.add_generated_tool({"name": "x", "version": 1})
        memory.add_generated_tool({"name": "x", "version": 2})
        assert memory.list_generated_tools() == [{"name": "x", "version": 2}]

    def test_add_generated_tool_keeps_distinct_names(self) -> None:
        from src.core.memory import MemorySystem

        memory = MemorySystem()
        memory.add_generated_tool({"name": "x", "version": 1})
        memory.add_generated_tool({"name": "y", "version": 1})
        assert len(memory.list_generated_tools()) == 2

    def test_generated_tools_included_in_save(self, tmp_path: Path) -> None:
        from src.core.memory import MemorySystem

        persist_path = str(tmp_path / "memory.json")
        memory = MemorySystem(persist_path=persist_path)
        memory.add_generated_tool({"name": "x", "version": 1})
        memory.save()

        import json
        saved = json.loads(Path(persist_path).read_text())
        assert saved["generated_tools"] == [{"name": "x", "version": 1}]
