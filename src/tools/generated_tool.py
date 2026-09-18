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

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.sandbox import run_sandboxed
from src.tools.base import BaseTool, ToolExecutionError, collect_prior_results
from src.tools.dynamic_code import sandbox_failure_message, sandbox_findings, sandbox_output

if TYPE_CHECKING:
    from src.core.findings import Finding
    from src.core.memory import DatasetMetadata, MemorySystem
    from src.core.profiler import DatasetProfile
    from src.core.tool_factory import GeneratedToolSpec


class GeneratedTool(BaseTool):
    """A named, re-callable analysis tool whose body is LLM-authored code,
    executed through the same isolated-subprocess sandbox as
    execute_dynamic_code."""

    requires_llm = True  # invisible to the --no-llm deterministic planner (decision 5)
    executes_code = True
    #: A DF_OUT set by the tool body is saved as derived/<tool name>.parquet.
    output_subdir = "derived"

    def __init__(self, spec: GeneratedToolSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.description = spec.description

    def get_schema(self) -> dict[str, Any]:
        return self.spec.params_schema

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        params = super().prepare_params(params, memory, output_root)
        params["_prior_results"] = collect_prior_results(memory)
        return params

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self.spec.code or not self.spec.code.strip():
            raise ToolExecutionError(f"Generated tool '{self.name}' has no code to execute.")
        if not Path(file_path).exists():
            raise ToolExecutionError(f"Dataset file not found: {file_path}")

        prior_results = kwargs.pop("_prior_results", None)
        output_dir = kwargs.pop("output_dir", None)
        out_dir = Path(output_dir) if output_dir else Path(file_path).parent / "derived"
        # kwargs minus file_path reach the sandboxed code as plain data via
        # extra_globals — never templated into the code body (decision 2).
        extra_globals = {k: v for k, v in kwargs.items() if k != "file_path"}

        sandbox_result = run_sandboxed(
            code=self.spec.code,
            dataset_ref=file_path,
            extra_globals=extra_globals,
            prior_results=prior_results,
            derived_dest=str(out_dir / f"{self.name}.parquet"),
        )
        if sandbox_result.status != "ok":
            raise ToolExecutionError(sandbox_failure_message(sandbox_result))
        return sandbox_output(sandbox_result, f"'{self.name}'", self.name)

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        return sandbox_findings(output, self.name, "generated_tool", "AI-generated analysis tool")
