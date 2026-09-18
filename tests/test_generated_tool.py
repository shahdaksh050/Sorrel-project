"""Tests for src/tools/generated_tool.py (Round 8, item 8.2)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.tool_factory import GeneratedToolSpec
from src.tools.base import ToolExecutionError
from src.tools.generated_tool import GeneratedTool
from tests.fixtures import single_column


def _spec(code: str, params_schema: dict | None = None) -> GeneratedToolSpec:
    return GeneratedToolSpec(
        name="row_counter",
        description="counts rows",
        params_schema=params_schema or {},
        code=code,
        version=1,
        created_at="2026-01-01T00:00:00+00:00",
        dataset_fingerprint="fp1",
    )


class TestExecute:
    def test_basic_execution(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = GeneratedTool(_spec("RESULT = int(df.shape[0])\n"))
        output = tool.execute(file_path=str(dataset))
        assert output["status"] == "ok"
        assert output["result"] == 50

    def test_declared_parameter_reaches_code(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        spec = _spec(
            "RESULT = int(df.shape[0]) + bump\n",
            {"bump": {"type": "number", "description": "x", "required": False}},
        )
        tool = GeneratedTool(spec)
        output = tool.execute(file_path=str(dataset), bump=5)
        assert output["status"] == "ok"
        assert output["result"] == 55

    def test_finding_payload_and_evidence_populated(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        code = (
            "n = int(df.shape[0])\n"
            "RESULT = n\n"
            "FINDING = {'headline': f'{n} rows', 'detail': 'd', 'evidence': {'n': n}}\n"
        )
        tool = GeneratedTool(_spec(code))
        output = tool.execute(file_path=str(dataset))
        assert output["finding_payload"] == {"headline": "50 rows", "detail": "d", "evidence": {"n": 50}}
        # Decision 7 — evidence numbers copied to the top-level output too,
        # so controller._flag_unverified_claims can verify them.
        assert output["evidence"] == {"n": 50}

    def test_no_finding_key_when_absent(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = GeneratedTool(_spec("RESULT = 1\n"))
        output = tool.execute(file_path=str(dataset))
        assert "finding_payload" not in output
        assert "evidence" not in output

    def test_sandbox_error_surfaces_without_raising(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = GeneratedTool(_spec("RESULT = 1 / 0\n"))
        output = tool.execute(file_path=str(dataset))
        assert output["status"] == "error"
        assert output["error_type"] == "runtime"

    def test_empty_code_raises(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = GeneratedTool(_spec("   "))
        with pytest.raises(ToolExecutionError):
            tool.execute(file_path=str(dataset))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        tool = GeneratedTool(_spec("RESULT = 1\n"))
        with pytest.raises(ToolExecutionError):
            tool.execute(file_path=str(tmp_path / "nope.csv"))

    def test_run_wraps_sandbox_failure_as_tool_success(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = GeneratedTool(_spec("x = 1\n"))  # no RESULT
        tool_result = tool.run(file_path=str(dataset))
        assert tool_result.status == "success"
        assert tool_result.output["status"] == "error"


class TestGetSchemaAndIdentity:
    def test_schema_matches_spec(self) -> None:
        schema = {"x": {"type": "number", "description": "d", "required": True}}
        tool = GeneratedTool(_spec("RESULT = 1\n", schema))
        assert tool.get_schema() == schema

    def test_name_and_description_from_spec(self) -> None:
        tool = GeneratedTool(_spec("RESULT = 1\n"))
        assert tool.name == "row_counter"
        assert tool.description == "counts rows"

    def test_requires_llm_is_true(self) -> None:
        assert GeneratedTool(_spec("RESULT = 1\n")).requires_llm is True

    def test_to_prompt_description_works(self) -> None:
        tool = GeneratedTool(_spec("RESULT = 1\n"))
        text = tool.to_prompt_description()
        assert "row_counter" in text
        assert "counts rows" in text


class TestFindings:
    def test_builds_finding_from_payload(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        code = "RESULT = 1\nFINDING = {'headline': 'h', 'detail': 'd', 'evidence': {'n': 1}}\n"
        tool = GeneratedTool(_spec(code))
        output = tool.execute(file_path=str(dataset))
        findings = tool.findings(output, None, None)
        assert len(findings) == 1
        f = findings[0]
        assert f.headline == "h"
        assert f.source_tool == "row_counter"
        assert f.kind == "generated_tool"
        assert "AI-generated analysis tool" in f.caveats

    def test_deterministic_finding_id_across_identical_runs(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        code = "RESULT = 1\nFINDING = {'headline': 'stable headline'}\n"
        tool = GeneratedTool(_spec(code))
        out1 = tool.execute(file_path=str(dataset))
        out2 = tool.execute(file_path=str(dataset))
        f1 = tool.findings(out1, None, None)[0]
        f2 = tool.findings(out2, None, None)[0]
        assert f1.finding_id == f2.finding_id

    def test_no_findings_when_payload_absent(self) -> None:
        tool = GeneratedTool(_spec("RESULT = 1\n"))
        assert tool.findings({"status": "ok", "result": 1}, None, None) == []

    def test_no_findings_when_payload_malformed(self) -> None:
        tool = GeneratedTool(_spec("RESULT = 1\n"))
        # finding_payload present but with no usable headline/summary
        output = {"status": "ok", "result": 1, "finding_payload": {"detail": "no headline here"}}
        assert tool.findings(output, None, None) == []

    def test_findings_never_raises_on_garbage_input(self) -> None:
        tool = GeneratedTool(_spec("RESULT = 1\n"))
        assert tool.findings({"finding_payload": "not a dict"}, None, None) == []
        assert tool.findings({"finding_payload": None}, None, None) == []
