"""
Generated Tool — Execution Layer for LLM-defined, re-callable analysis
tools (Round 8, item 8.2).

Architecturally a sibling of src.tools.dynamic_code.DynamicCodeExecutionTool
(same run_sandboxed delegation, same df/SCHEMA/RESULT contract) but named,
reusable across iterations, and parameterized via a `GeneratedToolSpec`
instead of one-shot inline code. One GeneratedTool instance wraps exactly
one spec; a "modification" (self-correction) is a new GeneratedTool built
from a version-bumped spec and re-registered under the same name, not a
mutation of this instance.

Pure per the AGENTS.md layer rule: execute() never touches MemorySystem or
ToolRegistry. Registration/persistence happens in
src.core.tool_factory.register_and_persist, called by the controller.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.sandbox import run_sandboxed
from src.tools.base import BaseTool, ToolExecutionError

#: Matches the deterministic finding_id convention every other
#: finding-emitting tool uses (e.g. concentration_analysis.py's
#: f"concentration_{measure}_{entity}", segment_comparison.py's slugged
#: composite ids) — never Python's built-in hash(), which is
#: process-randomized (PYTHONHASHSEED) and would make finding_id differ
#: between two identical runs, breaking BaseTool's documented "same inputs
#: -> same outputs" contract and any report/dashboard binding keyed on it.
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", text.lower()).strip("_") or "na"


if TYPE_CHECKING:
    from src.core.findings import Finding
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile
    from src.core.tool_factory import GeneratedToolSpec


class GeneratedTool(BaseTool):
    """A named, re-callable analysis tool whose body is LLM-authored code,
    executed through the same isolated-subprocess sandbox as
    execute_dynamic_code."""

    requires_llm = True  # invisible to the --no-llm deterministic planner (decision 5)

    def __init__(self, spec: GeneratedToolSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.description = spec.description

    def get_schema(self) -> dict[str, Any]:
        return self.spec.params_schema

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self.spec.code or not self.spec.code.strip():
            raise ToolExecutionError(f"Generated tool '{self.name}' has no code to execute.")
        if not Path(file_path).exists():
            raise ToolExecutionError(f"Dataset file not found: {file_path}")

        # kwargs minus file_path reach the sandboxed code as plain data via
        # extra_globals — never templated into the code body (decision 2).
        extra_globals = {k: v for k, v in kwargs.items() if k != "file_path"}

        sandbox_result = run_sandboxed(
            code=self.spec.code,
            dataset_ref=file_path,
            extra_globals=extra_globals,
        )

        finding_payload = getattr(sandbox_result, "finding", None)

        if sandbox_result.status != "ok":
            return {
                "summary": (
                    f"Execution failed ({sandbox_result.error_type}): "
                    f"{sandbox_result.hint or 'see traceback for details.'}"
                ),
                "status": "error",
                "result": None,
                "stdout": sandbox_result.stdout,
                "error_type": sandbox_result.error_type,
                "traceback": sandbox_result.traceback,
                "hint": sandbox_result.hint,
                "duration_ms": sandbox_result.duration_ms,
            }

        output: dict[str, Any] = {
            "summary": f"Executed '{self.name}' successfully in {sandbox_result.duration_ms:.0f} ms.",
            "status": "ok",
            "result": sandbox_result.result,
            "stdout": sandbox_result.stdout,
            "error_type": None,
            "traceback": None,
            "hint": None,
            "duration_ms": sandbox_result.duration_ms,
        }

        # Decision 7: FINDING evidence numbers must also land in `output`
        # itself, verbatim, so controller._flag_unverified_claims can verify
        # any narrative built from this finding against the tool's own
        # output — not a workaround, a requirement.
        if isinstance(finding_payload, dict):
            output["finding_payload"] = finding_payload
            evidence = finding_payload.get("evidence")
            if isinstance(evidence, dict):
                output["evidence"] = dict(evidence)

        return output

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        try:
            payload = output.get("finding_payload")
            if not isinstance(payload, dict):
                return []

            headline = payload.get("headline") or payload.get("summary")
            if not headline or not isinstance(headline, str):
                return []

            from src.core.findings import Finding

            evidence = payload.get("evidence")
            if not isinstance(evidence, dict):
                evidence = {}

            finding = Finding(
                finding_id=f"{self.name}_{_slug(headline)[:60]}",
                kind="generated_tool",
                headline=headline,
                detail=str(payload.get("detail", "")),
                evidence=evidence,
                source_tool=self.name,
                caveats=["AI-generated analysis tool"],
            )
            return [finding]
        except Exception:
            # findings() must never raise — a malformed FINDING payload
            # from LLM-authored code means "no finding", not a crash.
            return []
