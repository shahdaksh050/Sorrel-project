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

import concurrent.futures
import json
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import psutil

from src.core.memory import MemorySystem, ToolResult

if TYPE_CHECKING:
    from src.core.findings import Finding
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile


class ToolExecutionError(Exception):
    """Raised when a tool encounters an unrecoverable, expected error."""
    pass


class ToolInterruptSignal(Exception):  # noqa: N818
    """
    Raised by a tool when a statistical precondition or critical assumption is broken,
    signaling to the controller that the current linear execution plan should halt and pivot.
    """

    def __init__(
        self,
        reason: str,
        recommended_pivot: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.recommended_pivot = recommended_pivot
        self.details = details or {}


#: A prior tool output whose JSON exceeds this is cut down to its small
#: values before being shipped into the sandbox as PRIOR_RESULTS.
PRIOR_RESULT_CAP_CHARS = 20_000
_PRIOR_VALUE_CAP_CHARS = 2_000
#: Never useful as prior data (a chart spec's rows, console output) and large
#: enough to push the computed RESULT over the cap.
_PRIOR_SKIP_KEYS = frozenset({"chart", "stdout"})


class _ProcessTelemetrySampler:
    """Bounded process resource sampler for one tool invocation.

    ``psutil`` has no portable per-function peak-RSS counter.  Sampling the
    current process at a short fixed interval gives a useful observed peak
    without coupling tools to the UI runner or a platform-specific profiler.
    """

    _SAMPLE_INTERVAL_S: ClassVar[float] = 0.05

    def __init__(self) -> None:
        self._process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes: int | None = None
        self._start_cpu_seconds: float | None = None

    def _sample(self) -> None:
        """Record one RSS observation, tolerating process-inspection races."""
        try:
            rss = int(self._process.memory_info().rss)
        except psutil.Error:
            return
        self._peak_rss_bytes = max(self._peak_rss_bytes or 0, rss)

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self._SAMPLE_INTERVAL_S):
            self._sample()

    def start(self) -> None:
        """Take a baseline sample and start the bounded background sampler."""
        self._sample()
        try:
            times = self._process.cpu_times()
            self._start_cpu_seconds = float(times.user + times.system)
        except psutil.Error:
            self._start_cpu_seconds = None
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="dsa-tool-resource-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> tuple[float | None, int | None]:
        """Stop sampling and return ``(cpu_ms, observed_peak_rss_bytes)``."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._SAMPLE_INTERVAL_S * 2)
        self._sample()
        cpu_ms: float | None = None
        if self._start_cpu_seconds is not None:
            try:
                times = self._process.cpu_times()
                cpu_ms = max(0.0, (float(times.user + times.system) - self._start_cpu_seconds) * 1000)
            except psutil.Error:
                pass
        return cpu_ms, self._peak_rss_bytes


def _output_execution_metadata(output: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Extract optional backend and sampling decisions from a tool output."""
    backend: str | None = None
    raw_backend = output.get("compute_backend", output.get("backend"))
    if isinstance(raw_backend, str):
        backend = raw_backend
    elif isinstance(raw_backend, dict):
        candidate = raw_backend.get("backend", raw_backend.get("device"))
        if isinstance(candidate, str):
            backend = candidate

    raw_sampling = output.get("sampling")
    sampling = dict(raw_sampling) if isinstance(raw_sampling, dict) else {}
    return backend, sampling


def _tool_telemetry(
    output: dict[str, Any],
    elapsed_ms: float,
    sampler: _ProcessTelemetrySampler,
) -> dict[str, Any]:
    """Create the JSON-safe operational record attached to a ``ToolResult``."""
    cpu_time_ms, peak_rss_bytes = sampler.stop()
    backend, sampling = _output_execution_metadata(output)
    return {
        "wall_time_ms": elapsed_ms,
        "cpu_time_ms": cpu_time_ms,
        "peak_rss_bytes": peak_rss_bytes,
        "backend": backend,
        "sampling": sampling,
    }


def _resolved_path(value: Any) -> Path | None:
    """Return a canonical path for a planner value, or ``None`` if invalid."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return Path(value).resolve()
    except (OSError, RuntimeError):
        return None


def _registered_derived_paths(memory: MemorySystem) -> set[Path]:
    """Return canonical paths created and registered during this run."""
    derived = memory.get_context("derived_datasets") or {}
    if not isinstance(derived, dict):
        return set()
    paths: set[Path] = set()
    for info in derived.values():
        if not isinstance(info, dict):
            continue
        path = _resolved_path(info.get("path"))
        if path is not None:
            paths.add(path)
    return paths


def _confine_file_path(file_path: Any, memory: MemorySystem) -> tuple[Path, bool]:
    """Validate a tool input path against this analysis run's allowlist.

    The execution layer checks this independently of the controller so a
    direct tool caller cannot use planner-controlled parameters to read an
    arbitrary host file.
    """
    derived_dict = memory.get_context("derived_datasets") or {}
    if isinstance(file_path, str) and isinstance(derived_dict, dict) and file_path in derived_dict:
        info = derived_dict[file_path]
        if isinstance(info, dict) and info.get("path"):
            file_path = info["path"]
    target = _resolved_path(file_path)
    raw = _resolved_path(
        memory.dataset_metadata.file_path if memory.dataset_metadata is not None else None
    )
    cleaned = _resolved_path(memory.get_context("cleaned_file_path"))
    derived = _registered_derived_paths(memory)
    allowed = {path for path in (raw, cleaned) if path is not None} | derived
    if target is None or target not in allowed:
        raise ToolExecutionError(
            "file_path is not the loaded dataset, cleaned dataset, or a registered "
            "derived dataset."
        )
    return target, target in derived


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

    #: True when this tool is read-only and thread-safe, allowing consecutive
    #: instances to execute concurrently in parallel worker threads.
    is_concurrent_safe: ClassVar[bool] = False

    #: True when a concurrent-safe tool writes a fixed-name file, requiring
    #: output destination deduplication or sequencing.
    writes_fixed_file: ClassVar[bool] = False

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
          - confine file_path to this run's loaded/cleaned/derived datasets,
            then redirect it to cleaned_file_path (if uses_cleaned_file)
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
        if "file_path" in params:
            target, is_derived = _confine_file_path(params["file_path"], memory)
            params["file_path"] = str(target)
            cleaned = memory.get_context("cleaned_file_path")
            if self.uses_cleaned_file and cleaned and not is_derived:
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
        sampler = _ProcessTelemetrySampler()
        sampler.start()
        try:
            timeout_s = kwargs.pop("timeout_s", None)
            if timeout_s is None:
                try:
                    from src.core.compute_policy import ComputePolicy
                    timeout_s = ComputePolicy.from_env().tool_timeout_s
                except Exception:
                    timeout_s = 120.0

            if timeout_s is not None and float(timeout_s) > 0:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    from src.core.run_context import in_run_context
                    fn = in_run_context(self.execute)
                except Exception:
                    fn = self.execute
                try:
                    future = executor.submit(fn, **kwargs)
                    output = future.result(timeout=float(timeout_s))
                except concurrent.futures.TimeoutError as err:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise ToolExecutionError(
                        f"Tool '{self.name}' timed out after {float(timeout_s):.1f} seconds."
                    ) from err
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
            else:
                output = self.execute(**kwargs)

            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=output,
                execution_time_ms=elapsed,
                telemetry=_tool_telemetry(output, elapsed, sampler),
            )
        except ToolInterruptSignal as sig:
            elapsed = (time.monotonic() - start) * 1000
            interrupt_output = {
                "summary": f"Tool interrupt signaled by {self.name}: {sig.reason}",
                "interrupt_reason": sig.reason,
                "recommended_pivot": sig.recommended_pivot,
                "details": sig.details,
            }
            return ToolResult(
                tool_name=self.name,
                status="interrupt",
                output=interrupt_output,
                error_message=sig.reason,
                execution_time_ms=elapsed,
                telemetry=_tool_telemetry(interrupt_output, elapsed, sampler),
            )
        except ToolExecutionError as exc:
            elapsed = (time.monotonic() - start) * 1000
            error_output: dict[str, Any] = {}
            return ToolResult(
                tool_name=self.name,
                status="error",
                output=error_output,
                error_message=str(exc),
                execution_time_ms=elapsed,
                telemetry=_tool_telemetry(error_output, elapsed, sampler),
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            unexpected_output: dict[str, Any] = {}
            return ToolResult(
                tool_name=self.name,
                status="error",
                output=unexpected_output,
                error_message=f"Unexpected error in {self.name}: {exc!r}",
                execution_time_ms=elapsed,
                telemetry=_tool_telemetry(unexpected_output, elapsed, sampler),
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
