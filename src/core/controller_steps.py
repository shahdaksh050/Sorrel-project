"""
AgentController mixin: Stage 3 — validating, preparing, running and recording plan steps.

Split out of controller.py; `AgentController` inherits it, so the methods keep
their `self.*` state. Shared constants live in controller_common.
"""
from __future__ import annotations

import ast
import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.controller_common import (
    MAX_CODE_STEP_RETRIES,
    MAX_STEP_RETRIES,
    ControllerState,
    _PreparedStep,
    console,
)
from src.core.findings import Finding, score_objective_fit
from src.core.hypothesis import generate_counterfactual_probes
from src.core.memory import AnalysisStep, ToolResult
from src.core.multiple_testing import adjust_findings_run_level
from src.core.run_context import in_run_context
from src.core.step_validation import columns_for, is_column_param, validate_step

logger = logging.getLogger(__name__)


class StepMixin(ControllerState):
    """Stage 3 — validating, preparing, running and recording plan steps."""

    #: Read-only analytical tools that only read the dataset file and touch
    #: no shared run state, so a cycle's consecutive steps of these can run
    #: concurrently. Excluded on purpose: tools that read `prior_results`
    #: (generate_visualizations, generate_report), model training/clustering
    #: (CPU- and memory-heavy), and anything that executes LLM-authored code.
    _CONCURRENT_SAFE_TOOLS: frozenset[str] = frozenset({
        "detect_outliers", "correlation_analysis", "select_statistical_test",
        "segment_comparison", "time_series_analysis", "text_analysis",
        "geospatial_analysis", "experiment_analysis", "anomaly_analysis",
        "survival_analysis", "basket_analysis", "price_elasticity_analysis", "equity_analysis",
        "curve_fit_analysis",
    })

    #: Concurrent-safe tools that write a fixed-name output file.
    _FILE_WRITING_TOOLS: frozenset[str] = frozenset({"detect_outliers"})

    def _all_steps_cached(self, steps: list[AnalysisStep]) -> bool:
        """True when every planned step would be answered from the step cache."""
        for step in steps:
            if not self.tool_registry.has(step.tool_name):
                return False
            tool = self.tool_registry.get(step.tool_name)
            params = tool.prepare_params(
                self._resolve_file_path(tool, step.parameters), self.memory, self._output_dir
            )
            params, _dropped, errors, _normalised = self._validate_step(tool, step.parameters, params)
            if errors or self._step_cache_key(step.tool_name, params) not in self._step_cache:
                return False
        return bool(steps)

    def _tool_blocks(self) -> tuple[str, str, str]:
        """(full, short, compact) candidate-tool descriptions for the current profile."""
        args = (self.last_profile, self.memory.dataset_metadata)
        kwargs = {"use_ml": self.use_ml, "use_llm": self.use_llm}
        return (
            self.tool_registry.get_candidate_descriptions(*args, **kwargs),
            self.tool_registry.get_candidate_short_descriptions(*args, **kwargs),
            self.tool_registry.get_candidate_descriptions(*args, **kwargs, compact=True),
        )

    @staticmethod
    def _code_of(tool: Any, params: dict[str, Any]) -> str:
        """The LLM-authored code a code-executing step runs: inline for
        execute_dynamic_code/define_analysis_tool, the spec body for a
        generated tool."""
        code = params.get("code")
        if isinstance(code, str):
            return code
        spec = getattr(tool, "spec", None)
        return str(getattr(spec, "code", "") or "")

    #: Import roots / sklearn submodules and bare estimator names that fit
    #: predictive or clustering models. Preprocessing, metrics and other
    #: utility submodules are not listed — scaling a column is not "machine
    #: learning".
    _ML_IMPORT_ROOTS = frozenset({"xgboost", "lightgbm", "catboost"})

    _ML_SKLEARN_MODULES = frozenset({
        "ensemble", "linear_model", "tree", "svm", "neighbors", "naive_bayes",
        "neural_network", "cluster", "mixture", "gaussian_process",
        "discriminant_analysis", "cross_decomposition",
    })

    _ML_ESTIMATORS = frozenset({
        "KMeans", "SVC", "SVR", "LogisticRegression", "LinearRegression", "Ridge", "Lasso",
    })

    @classmethod
    def _detects_ml_code(cls, code: str) -> bool:
        """Whether LLM-authored code imports a model-fitting library."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules = [f"{node.module}.{alias.name}" for alias in node.names]
            elif isinstance(node, ast.Name):
                if node.id in cls._ML_ESTIMATORS or node.id.endswith(("Classifier", "Regressor")):
                    return True
                continue
            else:
                continue
            for module in modules:
                parts = module.split(".")
                if parts[0] in cls._ML_IMPORT_ROOTS or (
                    parts[0] == "sklearn" and (len(parts) == 1 or parts[1] in cls._ML_SKLEARN_MODULES)
                ):
                    return True
        return False

    def _resolve_file_path(self, tool: Any, params: dict[str, Any]) -> dict[str, Any]:
        """
        The planner may omit file_path (the system prompt tells it to) or
        name a derived dataset instead of giving its path. Fill in the raw
        dataset path — BaseTool.prepare_params then redirects it to the
        cleaned file — or the derived dataset's path.
        """
        try:
            schema = tool.get_schema()
        except Exception:
            schema = {}
        if "file_path" not in schema:
            return params
        params = dict(params)
        derived = self.memory.get_context("derived_datasets") or {}
        current = params.get("file_path")
        if isinstance(current, str) and current in derived:
            params["file_path"] = derived[current]["path"]
        elif not current and self.memory.dataset_metadata is not None:
            params["file_path"] = self.memory.dataset_metadata.file_path
        return params

    @staticmethod
    def _is_column_param(key: str) -> bool:
        return is_column_param(key)

    def _columns_for(self, file_path: Any) -> list[str] | None:
        """Columns of the dataset a step will read — see `step_validation.columns_for`."""
        return columns_for(self.memory, file_path)

    def _validate_step(
        self, tool: Any, raw: dict[str, Any], params: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str], list[str], str]:
        """Check a step against its tool's schema before it runs — see
        `step_validation.validate_step`."""
        return validate_step(self.memory, tool, raw, params)

    #: Params injected from run state (not chosen by the planner) that must
    #: not key the step cache — they change every step and are large.
    _CACHE_EXCLUDED_PARAMS = frozenset({"prior_results", "_prior_results"})

    @staticmethod
    def _step_cache_key(tool_name: str, params: dict[str, Any]) -> str:
        """
        Content-address a step: same tool, same resolved params, same input
        file content → same key. File-valued params are stamped with
        mtime+size (not just path) so an in-place edit still invalidates.
        """
        parts = [tool_name]
        for key in sorted(params):
            if key in StepMixin._CACHE_EXCLUDED_PARAMS:
                continue
            value = params[key]
            parts.append(f"{key}={value!r}")
            if isinstance(value, str):
                candidate = Path(value)
                if candidate.is_file():
                    stat = candidate.stat()
                    parts.append(f"{key}.stat={stat.st_mtime_ns}:{stat.st_size}")
        return "|".join(parts)

    def _record_step(self, step: AnalysisStep, result: ToolResult) -> None:
        """Append a step's outcome, tagged with the current cycle, to memory."""
        result.iteration = self.memory.iteration_count
        self.memory.append_tool_result(result)
        self.memory.mark_step_complete(step.step_number, result)

    def _process_step_result(self, prepared: _PreparedStep, result: ToolResult, total_steps: int) -> None:
        """Record a freshly executed step and fold its output into run state."""
        idx, step, tool, params = prepared.idx, prepared.step, prepared.tool, prepared.params
        executes_code = getattr(tool, "executes_code", False)
        if executes_code:
            self._governor.record(
                tool_name=step.tool_name, code=self._code_of(tool, params), params=params,
                status=result.status, iteration=self.memory.iteration_count,
                step_number=step.step_number, output=result.output,
                error=result.error_message,
            )
        if prepared.dropped_note:
            if result.status == "success":
                result.output = {**result.output, "plan_note": prepared.dropped_note}
            else:
                result.error_message = f"{result.error_message or ''} ({prepared.dropped_note})".strip()
        self._record_step(step, result)

        if result.status == "success":
            try:
                new_findings: list[Finding] = tool.findings(result.output, self.last_profile, self.memory.dataset_metadata)
            except Exception as exc:
                new_findings = []
                console.print(f"  [yellow]⚠ findings() failed for {step.tool_name} (non-fatal): {exc}[/]")
            if new_findings:
                seen_ids = {f.finding_id for f in self.memory.findings}
                for i, finding in enumerate(new_findings):
                    if not finding.finding_id:
                        finding.finding_id = f"{step.tool_name}_{step.step_number}_{i}"
                    if finding.finding_id in seen_ids:
                        finding.finding_id = f"{finding.finding_id}_i{self.memory.iteration_count}s{step.step_number}_{i}"
                    seen_ids.add(finding.finding_id)
                    if not finding.source_tool:
                        finding.source_tool = step.tool_name
                    if self.objective:
                        finding.objective_fit = score_objective_fit(finding, self.objective)
                    if not isinstance(finding.evidence, dict):
                        finding.evidence = {}
                    unit = finding.evidence.get("unit_of_analysis") or result.output.get("unit_of_analysis")
                    if isinstance(unit, str) and unit and unit != "row":
                        finding.evidence["unit_of_analysis"] = unit
                        note = f"Counts are per '{unit}' (repeated rows aggregated), not per row."
                        if note not in finding.caveats:
                            finding.caveats.append(note)
                self.memory.add_findings(new_findings)
                self._notify_findings(new_findings)
                col_names = [c.name for c in self.last_profile.columns] if self.last_profile else []
                for f in new_findings:
                    m = f.measure or ""
                    d = f.dimension or ""
                    confounders = [c for c in col_names if c not in (m, d)][:3]
                    cf_probes = generate_counterfactual_probes(m, d, confounders) if (m and d) else []
                    key = (m, d) if (m and d) else None
                    # Nest under the first hypothesis already recorded about
                    # this exact (measure, dimension) pair — a later finding
                    # on the same relationship is a refinement/counterfactual
                    # of it, not an unrelated top-level hypothesis. Exact-key
                    # match only, so unrelated findings stay flat siblings.
                    parent_id = self._hypothesis_parent_by_key.get(key) if key else None
                    node = self.hypothesis_tree.add_hypothesis(
                        statement=f"{f.headline or f.detail}",
                        parent_id=parent_id,
                        status="supported",
                        rationale=f"Observed in step {step.step_number} ({step.tool_name})",
                        counterfactuals=cf_probes,
                    )
                    if key and key not in self._hypothesis_parent_by_key:
                        self._hypothesis_parent_by_key[key] = node.id
                    self._hypothesis_parent_by_tool[step.tool_name] = node.id
                self.memory.set_context("hypothesis_tree", self.hypothesis_tree.to_dict())
                if any(f.p_value is not None for f in new_findings):
                    try:
                        adjust_findings_run_level(self.memory.findings)
                    except Exception as exc:
                        console.print(f"  [yellow]⚠ Run-level FDR correction skipped (non-fatal): {exc}[/]")

            derived = result.output.get("derived_dataset")
            if isinstance(derived, dict) and derived.get("name") and derived.get("path"):
                registry = dict(self.memory.get_context("derived_datasets") or {})
                registry[str(derived["name"])] = derived
                self.memory.set_context("derived_datasets", registry)
                console.print(
                    f"  [dim]Derived dataset '{derived['name']}' → {derived['path']}[/]"
                )

        if step.tool_name == "define_analysis_tool" and result.status == "success":
            self._maybe_register_generated_tool(result)

        # Cached only now: registration can flip a define_analysis_tool result
        # to "error", and an error must never be served from the cache.
        if result.status == "success":
            self._step_cache[prepared.cache_key] = result
            # A prior interrupt's CRITICAL PRECONDITION VIOLATION block would
            # otherwise keep re-injecting into every subsequent prompt for
            # the rest of the run — a later step succeeding means the agent
            # has already pivoted past it.
            if self.memory.get_context("interrupt_signal") is not None:
                self.memory.clear_context("interrupt_signal")

        rationales = self.memory.get_context("plan_rationales") or []
        rationales.append({
            "step_number": step.step_number,
            "tool_name": step.tool_name,
            "rationale": step.rationale,
        })
        self.memory.set_context("plan_rationales", rationales)

        if step.tool_name == "select_statistical_test" and result.status == "success":
            pvalue_tests = self.memory.get_context("statistical_test_pvalues") or []
            family = result.output.get("family_results")
            if family:
                for entry in family:
                    feature = entry.get("feature_column") or result.output.get("feature_column")
                    group = entry.get("group_column")
                    pvalue_tests.append({
                        "step_number": step.step_number,
                        "feature_column": f"{feature} by {group}" if group else feature,
                        "test_name": entry.get("test_name"),
                        "p_value": entry.get("p_value"),
                    })
            elif "p_value" in result.output:
                pvalue_tests.append({
                    "step_number": step.step_number,
                    "feature_column": result.output.get("feature_column"),
                    "test_name": result.output.get("test_name"),
                    "p_value": result.output["p_value"],
                })
            self.memory.set_context("statistical_test_pvalues", pvalue_tests)

        if step.tool_name == "segment_comparison" and result.status == "success":
            pvalue_tests = self.memory.get_context("statistical_test_pvalues") or []
            for c in result.output.get("comparisons", []):
                if c.get("p_value") is None:
                    continue
                test_name = "proportions_ztest" if c.get("is_rate") else "welch_ttest"
                pvalue_tests.append({
                    "step_number": step.step_number,
                    "feature_column": f"{c.get('measure')} by {c.get('dimension')}={c.get('level')}",
                    "test_name": test_name,
                    "p_value": c["p_value"],
                })
            self.memory.set_context("statistical_test_pvalues", pvalue_tests)

        if result.status == "error":
            self._tool_failure_counts[step.tool_name] = (
                self._tool_failure_counts.get(step.tool_name, 0) + 1
            )
            self.memory.increment_retry(step.step_number)

        if step.tool_name == "train_model" and result.status == "success":
            best = result.output.get("best_model", "")
            mt = result.output.get("models_trained", {})
            if best and best in mt:
                model_path = mt[best].get("model_path", "")
                if model_path:
                    self.memory.set_context("best_model_path", model_path)
                    self.memory.set_context("best_model_name", best)
            trained_test_size = result.output.get("test_size")
            if trained_test_size is not None:
                self.memory.set_context("train_test_size", trained_test_size)
            trained_split_strategy = result.output.get("split_strategy")
            if trained_split_strategy in ("random", "time_series", "panel"):
                self.memory.set_context("split_strategy", trained_split_strategy)
                self.memory.set_context("split_time_column", result.output.get("time_column"))
                self.memory.set_context("split_group_column", result.output.get("group_column"))

        if result.status == "interrupt":
            self.memory.set_context("interrupt_signal", result.output)
            reason = result.error_message or "Execution interrupted by tool precondition violation."
            pivot = result.output.get("recommended_pivot")
            # A precondition violation for a tool that already produced
            # findings is a refutation of *that* hypothesis, not a new
            # unrelated top-level one — nest it under the most recent
            # hypothesis this same tool contributed.
            parent_id = self._hypothesis_parent_by_tool.get(step.tool_name)
            self.hypothesis_tree.add_hypothesis(
                statement=f"Precondition check for {step.tool_name}",
                parent_id=parent_id,
                status="refuted",
                rationale=reason,
                counterfactuals=[f"Pivot recommendation: {pivot}"] if pivot else [],
            )
            self.memory.set_context("hypothesis_tree", self.hypothesis_tree.to_dict())
            console.print(f"  [red bold]⚡ Step {step.step_number}: {step.tool_name} triggered INTERRUPT:[/] {reason}")
            if self.on_step_callback:
                self.on_step_callback(step.tool_name, "interrupt", f"Halted by interrupt: {reason[:80]}")

        if step.tool_name == "clean_data" and result.status == "success":
            cleaned_path = result.output.get("cleaned_file_path")
            if cleaned_path:
                self.memory.set_context("cleaned_file_path", cleaned_path)
                console.print(
                    f"  [dim]Cleaned file stored → {cleaned_path}[/]"
                )

        if self.on_step_callback:
            summary = result.output.get("summary", "")[:80] if result.status == "success" else result.error_message
            self.on_step_callback(step.tool_name, result.status, f"{idx}/{total_steps} done — {summary}")

    def _step_budget(self, step: AnalysisStep) -> int:
        """Failures a tool may accumulate before its steps are skipped."""
        is_code_tool = self.tool_registry.has(step.tool_name) and getattr(
            self.tool_registry.get(step.tool_name), "executes_code", False
        )
        return MAX_CODE_STEP_RETRIES if is_code_tool else MAX_STEP_RETRIES

    def _prepare_step(self, idx: int, step: AnalysisStep, total_steps: int) -> _PreparedStep | None:
        """
        Resolve a step against current run state and run every pre-execution
        gate (retry budget, registry, plan validation, ML switch, governance).
        A step that fails a gate is recorded here and yields None.
        """
        budget = self._step_budget(step)
        if self._tool_failure_counts.get(step.tool_name, 0) >= budget:
            console.print(
                f"  [yellow]⏭ Step {step.step_number}: {step.tool_name} skipped "
                f"(exceeded {budget} retries).[/]"
            )
            self._record_step(step, ToolResult(
                tool_name=step.tool_name, status="skipped",
                output={"summary": f"Skipped: '{step.tool_name}' already failed {budget} times. Do not plan it again."},
            ))
            return None
        # Defense in depth: _parse_steps filters unknown tools, but never
        # let a registry miss crash the whole pipeline.
        try:
            tool = self.tool_registry.get(step.tool_name)
        except KeyError as exc:
            self._record_step(step, ToolResult(
                tool_name=step.tool_name, status="error", output={}, error_message=str(exc),
            ))
            return None
        if getattr(tool, "requires_ml", False) and not self.use_ml:
            console.print(
                f"  [yellow]⏭ Step {step.step_number}: {step.tool_name} skipped "
                f"(use_ml=False).[/]"
            )
            self._record_step(step, ToolResult(
                tool_name=step.tool_name, status="skipped",
                output={"summary": f"Skipped: '{step.tool_name}' needs machine learning, which is turned off (use_ml=False). Do not plan it."},
            ))
            return None

        # Parameter resolution is driven by each tool's own declarations
        # (BaseTool.prepare_params) so this controller never grows a per-tool
        # if-ladder. It must run after earlier steps of the cycle have been
        # recorded: it reads cleaned_file_path, best_model_path and the like.
        params = tool.prepare_params(
            self._resolve_file_path(tool, step.parameters), self.memory, self._output_dir
        )
        # Plan validation: an invalid step never runs (and never spends
        # code-execution budget); its error — with close column matches —
        # goes back to the planner and counts against the retry budget.
        params, dropped, plan_errors, normalised = self._validate_step(tool, step.parameters, params)
        dropped_note = f"Ignored unknown parameter(s): {', '.join(dropped)}." if dropped else ""
        if dropped:
            console.print(f"  [yellow]⚠ Step {step.step_number}: {step.tool_name} — {dropped_note}[/]")
        if normalised:
            console.print(f"  [dim]Step {step.step_number}: {step.tool_name} — {normalised}[/]")
            dropped_note = f"{dropped_note} {normalised}.".strip()
        if plan_errors:
            message = " ".join(["Plan validation failed, step not run:", *plan_errors, dropped_note]).strip()
            console.print(f"  [yellow]✗ Step {step.step_number}: {step.tool_name} — {message}[/]")
            self._record_step(step, ToolResult(
                tool_name=step.tool_name, status="error", output={}, error_message=message,
            ))
            self._tool_failure_counts[step.tool_name] = self._tool_failure_counts.get(step.tool_name, 0) + 1
            self.memory.increment_retry(step.step_number)
            if self.on_step_callback:
                self.on_step_callback(step.tool_name, "error", f"{idx}/{total_steps} — {message[:80]}")
            return None

        if getattr(tool, "executes_code", False):
            code = self._code_of(tool, params)
            refusal = (
                "Machine learning is turned off for this analysis (use_ml=False); ML code was not run."
                if not self.use_ml and self._detects_ml_code(code)
                else self._governor.refusal_reason()
            )
            if refusal:
                console.print(f"  [yellow]⛔ Step {step.step_number}: {step.tool_name} refused — {refusal}[/]")
                self._governor.record(
                    tool_name=step.tool_name, code=code, params=params,
                    status="refused", iteration=self.memory.iteration_count,
                    step_number=step.step_number, error=refusal, refused=True,
                )
                self._record_step(step, ToolResult(
                    tool_name=step.tool_name, status="skipped",
                    output={"summary": f"Refused by policy: {refusal}"},
                ))
                return None

        cache_key = self._step_cache_key(step.tool_name, params)
        cached = self._step_cache.get(cache_key)
        if cached is None:
            console.print(f"  [cyan]→ Step {step.step_number}: {step.tool_name}[/] [dim]{step.rationale[:60]}[/]")
            if self.on_step_callback:
                self.on_step_callback(step.tool_name, "running", f"{idx}/{total_steps} — {step.tool_name}…")
        return _PreparedStep(idx, step, tool, params, dropped_note, cache_key, cached)

    def _record_cached_step(self, prepared: _PreparedStep, total_steps: int) -> None:
        """Answer a step from the cache. Only the result is re-recorded (so
        this cycle's digest shows it): its findings, p-values and context
        side effects already exist from the original run, and repeating them
        would duplicate findings and inflate the multiple-testing family."""
        assert prepared.cached is not None
        step = prepared.step
        console.print(
            f"  [dim]↺ Step {step.step_number}: {step.tool_name} — "
            f"identical to a prior successful step, reusing its result.[/]"
        )
        result = copy.copy(prepared.cached)
        result.output = {
            **result.output,
            "plan_note": (
                f"Identical to the iteration-{prepared.cached.iteration} step; its result was "
                "reused, not recomputed. Plan something new instead of repeating it."
            ),
        }
        self._record_step(step, result)
        if self.on_step_callback:
            summary = str(result.output.get("summary", ""))[:80]
            self.on_step_callback(step.tool_name, "success", f"{prepared.idx}/{total_steps} done — {summary}")

    def _run_batch(self, batch: list[tuple[int, AnalysisStep]], total_steps: int) -> bool:
        """Prepare and execute consecutive steps; a batch of two or more
        uncached steps runs concurrently, and results are recorded in plan
        order either way so findings and ids stay deterministic. Returns True
        if an interrupt signal was encountered."""
        prepared = [p for idx, step in batch if (p := self._prepare_step(idx, step, total_steps)) is not None]
        to_run = [p for p in prepared if p.cached is None]
        results: dict[int, ToolResult] = {}
        # detect_outliers writes <stem>_outliers_flagged.csv, so two of them
        # in one batch would clobber each other's file: run those serially.
        if sum(p.step.tool_name in self._FILE_WRITING_TOOLS for p in to_run) > 1:
            to_run = [p for p in to_run if p.step.tool_name not in self._FILE_WRITING_TOOLS]
        if len(to_run) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(to_run))) as pool:
                run_tool = in_run_context(lambda p: p.tool.run(**p.params))
                outcomes = pool.map(run_tool, to_run)
                results = {p.idx: result for p, result in zip(to_run, outcomes, strict=True)}
        for p in prepared:
            if p.cached is not None:
                self._record_cached_step(p, total_steps)
            else:
                result = results[p.idx] if p.idx in results else p.tool.run(**p.params)
                self._process_step_result(p, result, total_steps)
                if result.status == "interrupt":
                    return True
        return False

    def _execute_steps(self, steps: list[AnalysisStep]) -> None:
        """
        Stage 3 — execute the plan with retry budgets. clean_data goes first
        (later steps read the cleaned file); runs of independent read-only
        analytical tools execute concurrently, everything else sequentially.
        If any tool raises a ToolInterruptSignal, execution of the remaining
        steps is halted so the agent can replan.
        """
        ordered = sorted(steps, key=lambda s: s.tool_name != "clean_data")
        total_steps = len(ordered)
        batch: list[tuple[int, AnalysisStep]] = []
        for idx, step in enumerate(ordered, 1):
            if self._stop_requested():
                return  # steps not yet started, including a pending batch, are dropped
            if step.tool_name in self._CONCURRENT_SAFE_TOOLS:
                batch.append((idx, step))
                continue
            if batch and self._run_batch(batch, total_steps):
                return
            batch = []
            if self._run_batch([(idx, step)], total_steps):
                return
        if batch:
            self._run_batch(batch, total_steps)

    def _maybe_register_generated_tool(self, result: ToolResult) -> None:
        """
        Round 8 — validate and, if it passes, register+persist a tool the
        LLM proposed via define_analysis_tool.

        define_analysis_tool.execute() only validates/smoke-tests (it must
        never mutate the registry or memory itself, per AGENTS.md's layer
        rule); this is where that proposal actually becomes a callable tool.
        A validation failure rewrites `result` in place to status="error" so
        the LLM sees exactly what to fix on its next iteration, the same way
        any other tool failure is surfaced.
        """
        if result.output.get("status") != "ready":
            return
        spec_dict = result.output.get("spec") or {}
        name = spec_dict.get("name")
        description = spec_dict.get("description", "")
        params_schema = spec_dict.get("params_schema", {}) or {}
        code = spec_dict.get("code", "")
        if not name:
            result.status = "error"
            result.error_message = "define_analysis_tool returned status='ready' with no tool name."
            return

        from src.core.tool_factory import (
            GeneratedToolSpec,
            compute_dataset_fingerprint,
            register_and_persist,
            save_to_library,
            tool_library_dir,
            tool_library_enabled,
            validate_spec,
        )

        existing_generated = {g["name"]: g for g in self.memory.list_generated_tools() if g.get("name")}
        errors = validate_spec(
            name=name,
            description=description,
            params_schema=params_schema,
            code=code,
            existing_tool_names=[n for n in self.tool_registry.names() if n not in existing_generated],
            existing_generated=existing_generated,
        )
        if errors:
            result.status = "error"
            result.error_message = (
                f"'{name}' was not registered: " + "; ".join(errors)
            )
            console.print(f"  [yellow]⚠ Generated tool '{name}' rejected: {result.error_message}[/]")
            return

        prior = existing_generated.get(name)
        version = (prior.get("version", 0) + 1) if prior else 1
        # GeneratedTool inherits uses_cleaned_file=True (the default), so it
        # always runs against the cleaned dataset once one exists — fingerprint
        # that same path, or a later opt-in reload (load_persisted_tools)
        # would never match.
        runtime_path = (
            self.memory.get_context("cleaned_file_path")
            or (self.memory.dataset_metadata.file_path if self.memory.dataset_metadata else "")
        )
        fingerprint = compute_dataset_fingerprint(runtime_path) if runtime_path else ""
        spec = GeneratedToolSpec(
            name=name,
            description=description,
            params_schema=params_schema,
            code=code,
            version=version,
            created_at=datetime.now(UTC).isoformat(),
            dataset_fingerprint=fingerprint,
        )
        register_and_persist(spec, self.tool_registry, self.memory, self._output_dir)
        if tool_library_enabled() and self.last_profile is not None:
            save_to_library(
                spec, tool_library_dir(), {c.name: c.kind for c in self.last_profile.columns}
            )
        console.print(
            f"  [bold green]✓ Registered generated tool[/] '{name}' "
            f"(v{version}, callable starting next iteration)."
        )
