"""
Abstract Base Tool for the Agentic Data Analysis System.

Every tool in src/tools/ must subclass BaseTool.

Contract:
  - Deterministic: same inputs → same outputs (no randomness unless seeded).
  - Structured output: every execute() must return a dict with a "summary" key.
  - Typed: full PEP 484 type hints required.
  - Safe failure: raise ToolExecutionError on expected failures; never swallow
    exceptions silently.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from src.core.memory import MemorySystem, ToolResult

if TYPE_CHECKING:
    from src.core.findings import Finding
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile


class ToolExecutionError(Exception):
    """Raised when a tool encounters an unrecoverable, expected error."""
    pass


#: A prior tool output whose JSON exceeds this is cut down to its small
#: values before being shipped into the sandbox as PRIOR_RESULTS.
PRIOR_RESULT_CAP_CHARS = 20_000
_PRIOR_VALUE_CAP_CHARS = 2_000
#: Never useful as prior data (a chart spec's rows, console output) and large
#: enough to push the computed RESULT over the cap.
_PRIOR_SKIP_KEYS = frozenset({"chart", "stdout"})


def _needs_cleaned_redirect(file_path: Any, memory: MemorySystem) -> bool:
    """Redirect to the cleaned dataset unless the planner named a derived
    dataset this run registered (execute_dynamic_code's `save_as`). Only
    registered paths are honoured — an arbitrary existing path from the
    planner would otherwise let a tool read any file on disk into a prompt."""
    if not file_path or not isinstance(file_path, str):
        return True
    derived = memory.get_context("derived_datasets") or {}
    try:
        target = Path(file_path).resolve()
        registered = {
            Path(info["path"]).resolve()
            for info in derived.values()
            if isinstance(info, dict) and info.get("path")
        }
    except OSError:
        return True
    return target not in registered


def collect_prior_results(memory: MemorySystem) -> dict[str, Any]:
    """{tool_name: JSON-safe output} of every successful step so far, for the
    sandbox's PRIOR_RESULTS. Oversized outputs keep only their small values
    (summary, scalars, short lists) so the subprocess input stays lean."""
    ok = [tr for tr in memory.tool_results if tr.status == "success" and isinstance(tr.output, dict)]
    totals = Counter(tr.tool_name for tr in ok)
    seen: Counter[str] = Counter()
    prior: dict[str, Any] = {}
    for tr in ok:
        sizes: dict[str, int] = {}
        for k, v in tr.output.items():
            if k in _PRIOR_SKIP_KEYS:
                continue
            try:
                sizes[k] = len(json.dumps(v))
            except (TypeError, ValueError, OverflowError):
                continue
        if sum(sizes.values()) > PRIOR_RESULT_CAP_CHARS:
            sizes = {k: n for k, n in sizes.items() if n <= _PRIOR_VALUE_CAP_CHARS}
        entry = {k: tr.output[k] for k in sizes}
        seen[tr.tool_name] += 1
        prior[tr.tool_name] = entry  # the latest run
        if totals[tr.tool_name] > 1:
            prior[f"{tr.tool_name}#{seen[tr.tool_name]}"] = entry  # every run, in order
    return prior


class BaseTool(ABC):
    """Abstract base class for all deterministic analysis tools."""

    #: Unique snake_case name referenced by the LLM in its JSON plans.
    name: str
    #: Human-readable description injected into LLM system prompts.
    description: str

    #: Subfolder under the run's output_dir this tool writes into (e.g.
    #: "models", "visualizations"). None means the tool writes no files —
    #: ToolRegistry/AgentController use this instead of a hardcoded map to
    #: inject `output_dir`.
    output_subdir: str | None = None

    #: When True (the default), `file_path` in this tool's parameters is
    #: redirected to the pipeline's cleaned dataset once clean_data has
    #: produced one. Ingestion/cleaning tools themselves opt out.
    uses_cleaned_file: bool = True

    #: True when this tool fits a model (supervised training, clustering,
    #: dimensionality reduction). Declared per tool rather than held as a
    #: name list in the controller, so a tool added later opts itself in and
    #: "run without ML" keeps meaning what it says. These are also the
    #: slowest tools by a wide margin — turning them off is the single
    #: biggest speed lever the system has.
    requires_ml: ClassVar[bool] = False

    #: True when this tool is meaningless without a reachable LLM (e.g. it
    #: executes LLM-generated code). Excluded outright in no-LLM mode.
    requires_llm: ClassVar[bool] = False

    #: True when this tool executes LLM-authored code in the sandbox. Such
    #: tools are governed (src/core/governance.py): hidden when code
    #: execution is disabled, budgeted per run, and audit-logged.
    executes_code: ClassVar[bool] = False

    #: {memory_context_key: param_name} — filled in from MemorySystem
    #: context whenever the plan step left `param_name` empty. Covers the
    #: common case (e.g. target_column); tools with bespoke injection logic
    #: (forced overrides, existence checks) override prepare_params instead.
    requires_context: ClassVar[dict[str, str]] = {}

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        """
        Relevance score in [0.0, 1.0] for this dataset, used to build the
        candidate tool set the planner sees (ToolRegistry.candidate_tools).

        0.0 excludes the tool entirely. The default (1.0) suits
        general-purpose EDA/reporting tools that apply to any dataset;
        tools tied to a specific data nature (time-series, text, geo, a
        target column...) override this to gate themselves in or out.
        """
        return 1.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        """
        Parameters this tool can choose for itself from the data profile.

        Used by the deterministic (no-LLM) planner, which otherwise knows
        only the file path. A tool whose schema requires more than that —
        `select_statistical_test` needs a feature and a group column,
        `generate_visualizations` needs a chart type — is unrunnable without
        this hook, and was being scheduled and failing on every no-LLM run.

        Returning {} means "I need nothing beyond file_path"; the planner
        skips any tool whose required parameters are still unfilled rather
        than scheduling a step it knows will fail.
        """
        return {}

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        """
        Resolve this step's parameters against pipeline state before execute().

        Generic policy driven by the declarations above:
          - redirect file_path to cleaned_file_path (if uses_cleaned_file)
          - inject output_dir under output_root/output_subdir (if unset)
          - fill any param named in requires_context from memory context
            (only when the planner left it empty)

        Override to add tool-specific injection (e.g. a forced override that
        must win even when the planner supplied a value, or a value that
        must be computed from accumulated results rather than read back).
        Always call super().prepare_params() first so the generic rules
        still apply.
        """
        params = dict(params)
        if self.uses_cleaned_file:
            cleaned = memory.get_context("cleaned_file_path")
            if cleaned and "file_path" in params and _needs_cleaned_redirect(params["file_path"], memory):
                params["file_path"] = cleaned
        if self.output_subdir and not params.get("output_dir"):
            params["output_dir"] = str(Path(output_root) / self.output_subdir)
        elif self.output_subdir and params.get("output_dir"):
            # P2.3 — output_dir can arrive from the LLM planner's own JSON
            # plan, not just this injection. Every write location, planner-
            # chosen or not, must still resolve under the run's output root;
            # resolve_output_path refuses any escape (absolute path outside
            # the root, `..` traversal). A planner value that fails the
            # check is replaced with the safe default rather than rejected
            # outright, so a bad plan degrades instead of failing the step.
            from src.core.security import UploadValidationError, resolve_output_path
            try:
                params["output_dir"] = str(resolve_output_path(output_root, params["output_dir"]))
            except UploadValidationError:
                params["output_dir"] = str(Path(output_root) / self.output_subdir)
        for ctx_key, param_name in self.requires_context.items():
            if not params.get(param_name):
                value = memory.get_context(ctx_key)
                if value is not None:
                    params[param_name] = value
        return params

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        """
        Derive structured Finding objects (src.core.findings) from this
        tool's own successful `output` dict — no tool rewrite required, this
        is purely a projection of data the tool already computed.

        Called once per successful step by the controller, which accumulates
        the results on MemorySystem.findings (7.1 finding bus). The default
        (no findings) suits tools that are pure data preparation (clean_data,
        ingest_dataset) rather than analysis. Every analysis tool should
        override this so its results reach the reports and dashboard through
        the one shared projection instead of a per-surface hardcoded parser.
        """
        return []

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the tool with the given parameters.

        Returns:
            JSON-serialisable dict.  Must always include:
              - "summary": one-sentence human-readable outcome description.
        Raises:
            ToolExecutionError: On expected, recoverable failures.
        """
        ...

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """
        Return the parameter schema for LLM prompt injection.

        Returns:
            {param_name: {"type": str, "description": str, "required": bool}}
        """
        ...

    def run(self, **kwargs: Any) -> ToolResult:
        """
        Public wrapper — timing, error handling, ToolResult packaging.

        Always call this method rather than execute() directly.
        Wraps ToolExecutionError and any unexpected exception into a
        structured ToolResult so the agent loop never crashes.
        """
        start = time.monotonic()
        try:
            output = self.execute(**kwargs)
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=output,
                execution_time_ms=elapsed,
            )
        except ToolExecutionError as exc:
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                status="error",
                output={},
                error_message=str(exc),
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                status="error",
                output={},
                error_message=f"Unexpected error in {self.name}: {exc!r}",
                execution_time_ms=elapsed,
            )

    def to_prompt_description(self) -> str:
        """Format tool info for LLM system prompt injection."""
        schema = self.get_schema()
        if "properties" in schema and isinstance(schema["properties"], dict):
            req_set = set(schema.get("required", [])) if isinstance(schema.get("required"), list) else set()
            params_dict: dict[str, Any] = {
                k: {**v, "required": k in req_set} if isinstance(v, dict) else {"type": "any", "description": str(v)}
                for k, v in schema["properties"].items()
            }
        else:
            params_dict = schema

        params = "\n".join(
            f"  - {k} ({v.get('type', 'any') if isinstance(v, dict) else 'any'}) "
            f"{'[required]' if isinstance(v, dict) and v.get('required') else '[optional]'}: "
            f"{v.get('description', '') if isinstance(v, dict) else ''}"
            for k, v in params_dict.items()
        )
        return (
            f"tool_name: {self.name}\n"
            f"description: {self.description}\n"
            f"parameters:\n{params}"
        )
