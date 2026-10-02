"""
Agent Controller — Orchestration and Reasoning Layer.

Implements the full 7-stage agent workflow:

  Stage 1 — Dataset Ingestion          load_dataset()
  Stage 2 — Initial Reasoning Phase    analyze() → first RLM invoke
  Stage 3 — Tool Selection & Execution analyze() → execution loop
  Stage 4 — Result Interpretation      analyze() → iteration prompt
  Stage 5 — Iterative Refinement       analyze() → loop until complete
  Stage 6 — RLM Workflow Management    _run_rlm_decomposition()
  Stage 7 — Report Generation          _generate_final_report()

ARCHITECTURAL BOUNDARY:
  - This file handles PLANNING and ORCHESTRATION only.
  - Raw data never enters LLM prompts — only summaries via PromptManager.
  - Tools are always called through ToolRegistry, never directly.
  - The LLM is always called through RLMEngine, never directly.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn

from src.core.agenda import Question, build_agenda, coverage_report
from src.core.causal_guard import (
    CAVEAT_OBSERVATIONAL,
    audit_findings_causal_language,
    classify_study_design,
)
from src.core.claim_verification import _CANON_PRECISIONS as _CANON_PRECISIONS
from src.core.claim_verification import _KEYWORD_TOOL_MAP as _KEYWORD_TOOL_MAP
from src.core.claim_verification import _KEYWORD_TOOL_PATTERNS as _KEYWORD_TOOL_PATTERNS
from src.core.claim_verification import _NUMBER_RE as _NUMBER_RE
from src.core.claim_verification import _UNVERIFIABLE_SKIP_ABS_INT as _UNVERIFIABLE_SKIP_ABS_INT
from src.core.claim_verification import _canon_number as _canon_number
from src.core.claim_verification import _collect_numbers as _collect_numbers
from src.core.coercion import coerce_types
from src.core.controller_common import (  # noqa: F401 - re-exported: other modules import these from here
    _AUTODETECT_HIGH,
    _AUTODETECT_LOW,
    _COMPACT_RETRY_NOTE,
    _MAX_PLANNER_CONTEXT_TOKENS,
    _MAX_STEPS_PER_CYCLE,
    _MIN_CHART_DESIGN_CONTEXT,
    MAX_CODE_STEP_RETRIES,
    MAX_STEP_RETRIES,
    _PreparedStep,
    _read_dataframe,
    console,
)
from src.core.controller_planning import PlanMixin
from src.core.controller_report import ReportMixin
from src.core.controller_steps import StepMixin
from src.core.degradations import collect_degradations
from src.core.deliverable_contract import audit_deliverables, parse_deliverable_contract
from src.core.domain_packs import detect_domain_pack, evaluate_domain_pack
from src.core.domains import infer_domains
from src.core.findings import Finding, attach_finding_checks
from src.core.governance import (
    AUDIT_SUBDIR,
    LOCAL_PROVIDERS,
    CodeGovernor,
    code_execution_enabled,
    local_only,
    max_llm_tokens_per_run,
)
from src.core.hypothesis import HypothesisTree
from src.core.integrity import evaluate_data_integrity
from src.core.io import get_max_rows, read_any
from src.core.joins import join_related
from src.core.llm_client import _DEFAULT_MODELS as _DEFAULT_MODELS
from src.core.llm_client import LLMClient as LLMClient
from src.core.llm_client import LocalOnlyError as LocalOnlyError
from src.core.memory import DatasetMetadata, MemorySystem, ToolResult
from src.core.model_telemetry import get_limiter
from src.core.profiler import DatasetProfile, profile_dataframe
from src.core.prompt_manager import (
    ARCHETYPES,
    RLM_SUBTASK_SYSTEM,
    PromptManager,
)
from src.core.question_router import route_question
from src.core.run_config import RunConfig
from src.core.run_context import objective_scope
from src.core.security import sanitize_for_prompt
from src.core.shared_context import SharedAnalysisContext
from src.core.step_validation import _COLUMN_PARAM_NAMES as _COLUMN_PARAM_NAMES
from src.core.tool_registry import _INJECTED_PARAMS as _INJECTED_PARAMS
from src.core.tool_registry import ToolRegistry as ToolRegistry
from src.core.tool_registry import _short_tool_description as _short_tool_description
from src.rlm.engine import RLMEngine, RLMSubTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent Controller — the top-level orchestrator
# ---------------------------------------------------------------------------

class AgentController(PlanMixin, StepMixin, ReportMixin):
    """
    Top-level orchestrator implementing the full 7-stage workflow.

    Reasoning and execution are strictly separated:
      - Reasoning: LLMClient → RLMEngine → PromptManager
      - Execution: ToolRegistry → BaseTool.run()
    """

    def __init__(
        self,
        max_iterations: int | None = None,
        enable_rlm: bool | None = None,
        memory_persist_path: str | None = None,
        use_llm: bool | None = None,
        use_ml: bool | None = None,
        min_iterations: int | None = None,
        objective: str | None = None,
        output_dir: str | None = None,
        run_config: RunConfig | None = None,
    ) -> None:
        # Capability switches. Both default on, and both are honest about
        # what they cost: with use_llm off the run is fully deterministic
        # (no network, no narrative synthesis, plans come from the profile);
        # with use_ml off no model is fitted, which is the single largest
        # time saving available since training dominates every run.
        self.use_llm = (
            use_llm
            if use_llm is not None
            else os.getenv("ENABLE_LLM", "true").strip().lower() == "true"
        )
        self.use_ml = (
            use_ml
            if use_ml is not None
            else os.getenv("ENABLE_ML", "true").strip().lower() == "true"
        )
        self.min_iterations = (
            min_iterations
            if min_iterations is not None
            else int(os.getenv("MIN_ITERATIONS", "1"))
        )
        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else int(os.getenv("MAX_ITERATIONS", "15"))
        )
        self.enable_rlm = (
            enable_rlm
            if enable_rlm is not None
            else os.getenv("ENABLE_RLM_INFERENCE", "true").lower() == "true"
        )
        self.memory = MemorySystem(persist_path=memory_persist_path)
        self.llm_client = LLMClient(run_config)
        self.tool_registry = ToolRegistry()
        self._rlm_engine: RLMEngine | None = None
        self._prompt_manager: PromptManager | None = None
        # P2.4 — concurrent runs must not corrupt each other's models/reports.
        # An explicit OUTPUT_DIR (the Streamlit app always sets one, into a
        # fresh tempdir per run) is honoured as-is; with no override, default
        # to a session-scoped subdirectory rather than a fixed "output" path,
        # so two unattended CLI runs in flight never share a models/ or
        # reports/ directory. A pointer file (not a symlink — no elevated
        # rights needed on Windows) records the most recent run for humans
        # poking at the output tree by hand.
        if output_dir:
            self._output_dir = output_dir
            Path(self._output_dir).mkdir(parents=True, exist_ok=True)
        elif os.getenv("OUTPUT_DIR"):
            self._output_dir = os.environ["OUTPUT_DIR"]
        else:
            self._output_dir = str(Path("output") / "runs" / self.memory.session_id)
            try:
                Path(self._output_dir).mkdir(parents=True, exist_ok=True)
                Path("output").mkdir(parents=True, exist_ok=True)
                Path("output", "latest.txt").write_text(self._output_dir, encoding="utf-8")
            except OSError:
                pass
        # Governance for LLM-authored code: kill switch, per-run budget, audit log.
        self._governor = CodeGovernor(self._output_dir, self.memory.session_id)
        self.llm_client.audit_dir = str(Path(self._output_dir) / AUDIT_SUBDIR)
        # Natural-language analysis objective supplied by the user (optional).
        self.objective = (objective or os.getenv("USER_OBJECTIVE") or "").strip()
        self.deliverable_contract = parse_deliverable_contract(self.objective)
        self.question_routing = route_question(self.objective)
        self.hypothesis_tree = HypothesisTree(primary_objective=self.objective)
        self.memory.set_context("hypothesis_tree", self.hypothesis_tree.to_dict())
        # Deterministic parent-lookup for hypothesis-tree nesting: the first
        # hypothesis recorded about a given (measure, dimension) pair, and the
        # most recent one sourced from a given tool — so a later counterfactual
        # or a precondition refutation about the same relationship attaches
        # under it instead of flattening the tree. Exact-key match only; no
        # fuzzy relationship-guessing.
        self._hypothesis_parent_by_key: dict[tuple[str, str], str] = {}
        self._hypothesis_parent_by_tool: dict[str, str] = {}
        if self.objective:
            self.memory.set_context("user_objective", self.objective)
            self.memory.set_context("question_routing", {
                "primary_family": self.question_routing.primary_family,
                "secondary_families": self.question_routing.secondary_families,
                "recommended_tools": self.question_routing.recommended_tools,
                "rationale": self.question_routing.rationale,
            })
        # Most recent dataset profile (set during load_dataset).
        self.last_profile: DatasetProfile | None = None
        # The coerced dataframe from load_dataset's first profiling pass,
        # held only long enough to re-profile once the target is known
        # (7.4 needs profile-before-target; class-imbalance warnings need
        # target-before-profile). Cleared at the end of load_dataset.
        self._pending_df: pd.DataFrame | None = None
        self.shared_context: SharedAnalysisContext | None = None
        self.study_design: str = "observational"
        # Charts from the most recent dashboard build (for the HTML report).
        self._last_charts: list[dict[str, Any]] = []
        self._rlm_decomposed: bool = False   # run decomposition at most once per session
        # Failures per tool across iterations — LLM-replanned steps are new
        # objects each cycle, so retry budgets must be tracked here.
        self._tool_failure_counts: dict[str, int] = {}
        # Cache of successful step results, keyed on (tool_name, resolved
        # params, input-file mtime+size) — the LLM replans from scratch each
        # iteration, so an identical step (same clean_data call, same
        # correlation_analysis params) would otherwise re-execute in full,
        # burning time and — for train_model — silently overwriting an
        # already-referenced model file with a fresh random draw.
        self._step_cache: dict[str, ToolResult] = {}
        # Optional callback fired after each tool: (tool_name, status, detail) -> None
        self.on_step_callback: Any = None
        # Optional callback fired after each LLM iteration: (iteration, stage) -> None
        self.on_iteration_callback: Any = None
        # Optional zero-arg callable, polled at safe checkpoints (before each
        # LLM call, between steps, after a step batch). True asks the run to
        # wind down: a cooperative stop, never a forced cancel, so a tool that
        # is already running finishes first.
        self.should_stop: Callable[[], bool] | None = None
        # Optional callback fired with each new Finding as a step adds it.
        self.on_finding_callback: Callable[[Finding], None] | None = None
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # 7.4 — Analysis-mode decision
    # ------------------------------------------------------------------

    #: Objective keywords that count as "the user actually asked for a
    #: prediction" — without one of these, a name-matched numeric target at
    #: transaction grain is treated as a descriptive subject (see below).
    _PREDICTION_INTENT_KEYWORDS = (
        "predict", "forecast", "model", "classif", "regress",
        "estimate", "will churn", "likely to", "propensity",
        "drive", "what affect", "what influenc", "what cause", "depends on", "explain", "why ",
    )

    def _decide_analysis_mode(self, col: str | None, confidence: float) -> dict[str, Any]:
        """
        Decide — and return a recorded rationale for — whether prediction is
        even the right mode, before `load_dataset`'s confidence-banded logic
        commits to training on `col`. `_AUTODETECT_HIGH` alone used to be
        sufficient to start training a regressor on any name-matched numeric
        column (`_NUMERIC_TARGET_NAMES` in memory.py); this adds one veto: a
        measure at transaction grain in a domain-matched dataset is a
        descriptive subject, not a modelling target, unless the objective
        actually asks for prediction. Autonomy includes the autonomy to
        decline to model (T2).
        """
        profile = self.last_profile
        objective_l = self.objective.lower()
        wants_prediction = any(kw in objective_l for kw in self._PREDICTION_INTENT_KEYWORDS)

        # A candidate below the autonomy floor (_AUTODETECT_LOW) is not a
        # real target — it's exactly the "very low confidence" band that
        # load_dataset's own confidence-banded logic (below) leaves
        # metadata.target_column unset for and defaults to EDA. Without this
        # check, this function unconditionally recorded `mode: "model"` for
        # ANY name-matched or positional-fallback column, however weak the
        # signal — including a column the caller never actually models.
        if col and confidence < _AUTODETECT_LOW:
            return {
                "mode": "describe",
                "target": None,
                # "candidate" (Round 8) — a column WAS found, just too weak
                # a signal to model on; kept distinct from `target` (whose
                # None means "not modelling", and load_dataset relies on
                # that) so downstream text (agenda.py) can still name the
                # column it declined rather than saying "None".
                "candidate": col,
                "rationale": (
                    f"Candidate target '{col}' reached only {confidence:.0%} confidence "
                    f"(below the {_AUTODETECT_LOW:.0%} autonomy floor) — too weak a signal "
                    "to commit to modelling; proceeding with descriptive EDA instead."
                ),
                "alternatives_rejected": [
                    f"model on '{col}' (confidence {confidence:.0%}, below autonomy floor)"
                ],
            }

        if col and profile and not wants_prediction:
            col_profile = next((c for c in profile.columns if c.name == col), None)
            is_transactional = bool(profile.domains) or (
                profile.entity_col is not None and (profile.rows_per_entity or 0) > 1.5
            )
            if col_profile is not None and col_profile.semantic_role == "measure" and is_transactional:
                return {
                    "mode": "describe",
                    "target": None,
                    "candidate": col,
                    "rationale": (
                        f"'{col}' is a descriptive measure at transaction grain "
                        f"({'domain: ' + profile.domains[0].domain if profile.domains else 'repeat-row entity structure'}), "
                        "and no prediction was requested — describing it (totals, "
                        "concentration, trend) is a better answer than training a "
                        "regressor on a name-matched column."
                    ),
                    "alternatives_rejected": [
                        f"regression on '{col}' (name-matched only, confidence {confidence:.2f})"
                    ],
                }

        if col:
            return {
                "mode": "model",
                "target": col,
                "candidate": col,
                "rationale": f"Auto-detected target '{col}' (confidence {confidence:.0%}).",
                "alternatives_rejected": [],
            }
        return {
            "mode": "describe",
            "target": None,
            "candidate": None,
            "rationale": "No plausible target column found — proceeding with descriptive EDA.",
            "alternatives_rejected": [],
        }

    # ------------------------------------------------------------------
    # Stage 1 — Dataset Ingestion
    # ------------------------------------------------------------------

    def _join_related_files(
        self, file_path: str, related: list[str], overrides: dict[str, dict[str, Any]] | None = None
    ) -> tuple[str, list[str]]:
        """Join related tables onto the main file; returns (dataset path, notes)."""
        original = file_path
        base, base_report = read_any(file_path)
        notes = list(base_report.notes)
        tables: list[tuple[str, pd.DataFrame]] = []
        for path in related:
            try:
                df, report = read_any(path)
            except Exception as exc:
                notes.append(f"Could not read related file {Path(path).name}: {exc}")
                continue
            tables.append((Path(path).stem, df))
            notes.extend(report.notes)
        merged, join_notes = join_related(base, tables, get_max_rows(), Path(file_path).stem, overrides)
        notes.extend(join_notes)
        if merged is not base:
            dest = Path(self._output_dir) / "derived" / f"{Path(file_path).stem}_joined.parquet"
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                merged.to_parquet(dest, index=False)
                file_path = str(dest)
            except Exception as exc:
                notes.append(f"Joined table could not be saved ({exc}); the main table alone was analysed.")
        if file_path == original:
            notes = notes[len(base_report.notes):]   # the main file's own notes are reported by its read
        self.memory.set_context("joins", notes)
        return file_path, notes

    def load_dataset(
        self,
        file_path: str,
        target_hint: str | None = None,
        interactive: bool = True,
        related_files: list[str] | None = None,
        join_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> DatasetMetadata:
        """Stage 1: see `_load_dataset`. Runs with this run's objective in scope,
        so the profile-driven tools never read another run's question."""
        with objective_scope(self.objective):
            return self._load_dataset(
                file_path, target_hint, interactive, related_files, join_overrides
            )

    def _load_dataset(
        self,
        file_path: str,
        target_hint: str | None = None,
        interactive: bool = True,
        related_files: list[str] | None = None,
        join_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> DatasetMetadata:
        """
        Stage 1: Ingest the dataset and store metadata in Memory.

        The LLM never sees raw data — only the compact metadata string.

        Args:
            file_path:   Path to the CSV or Excel file.
            target_hint: Optional column name to use as the ML target.
            interactive: When True and auto-detection confidence is low,
                         prompt the user via stdin. Set to False in
                         non-interactive environments (Streamlit, API).
            related_files: Optional extra tables (customers, products...) joined
                         onto the main file on inferred keys; the merged frame
                         becomes the dataset for every later stage.
            join_overrides: Optional per-table review choices keyed by related
                         file stem: {"skip": True} or {"left_key", "right_key"}.

        Returns:
            DatasetMetadata stored in the Memory System.
        """
        console.print(
            Panel(
                f"[bold]Stage 1 — Dataset Ingestion[/]\nFile: {file_path}",
                title="[bold green]Agent Controller",
                border_style="green",
            )
        )

        # Override target from CLI hint or environment
        target = target_hint or os.getenv("TARGET_COLUMN_HINT")

        join_notes: list[str] = []
        if related_files:
            file_path, join_notes = self._join_related_files(file_path, related_files, join_overrides)

        ingest_tool = self.tool_registry.get("ingest_dataset")
        result = ingest_tool.run(file_path=file_path, target_column=target)

        if result.status == "error":
            raise RuntimeError(f"Dataset ingestion failed: {result.error_message}")

        raw_meta = result.output["metadata"]
        metadata = DatasetMetadata(**raw_meta)

        # ---- Data profiling runs before target auto-detection (7.4) — the
        # analysis-mode decision below needs the semantic/domain picture
        # (is this a transaction log? is the candidate column a measure?),
        # not just a column name match. Failure here is non-fatal: the
        # decision function below degrades to name-only reasoning when
        # self.last_profile is still None. ----
        try:
            df, read_report = read_any(file_path)
            df, coercions = coerce_types(df, delimiter=read_report.delimiter)
            self._pending_df = df
            self.shared_context = SharedAnalysisContext(df=df)
            self.memory.set_context("shared_analysis_context", self.shared_context)
            self.study_design = classify_study_design(df, self.objective)
            self.memory.set_context("study_design", self.study_design)
            profile = profile_dataframe(df, target_column=None)
            try:
                profile.domains = infer_domains(df, profile)
            except Exception as exc:
                profile.domains = []
                console.print(f"  [yellow]⚠ Domain inference skipped: {exc}[/]")
            self.last_profile = profile
            self.memory.set_context("data_profile", profile.to_dict())
            self.memory.set_context("data_profile_summary", profile.to_prompt_string())

            # ---- Domain Pack Evaluation (Phase 5) ----
            pack = detect_domain_pack(df, self.objective)
            if pack:
                self.memory.set_context("active_domain_pack", pack.name)
                pack_findings = evaluate_domain_pack(df, pack=pack, objective=self.objective)
                if pack_findings:
                    self.memory.add_findings(pack_findings)
                    console.print(f"  [cyan]🏛 Domain Pack '{pack.display_name}': {len(pack_findings)} contextual finding(s) generated.[/]")

            # ---- Data Integrity & Constraint Audit (Phase 6) ----
            integrity_findings = evaluate_data_integrity(df)
            if integrity_findings:
                self.memory.add_findings(integrity_findings)
                console.print(f"  [yellow]🛡 Integrity Audit: {len(integrity_findings)} data integrity finding(s) detected.[/]")
            self.memory.set_context(
                "read_report",
                {
                    "path": read_report.path,
                    "format": read_report.format,
                    "encoding": read_report.encoding,
                    "encoding_confident": read_report.encoding_confident,
                    "delimiter": read_report.delimiter,
                    "delimiter_sniffed": read_report.delimiter_sniffed,
                    "duplicate_headers": read_report.duplicate_headers,
                    "notes": [*read_report.notes, *join_notes],
                },
            )
            self.memory.set_context("coercions", [c.to_dict() for c in coercions])
            self.memory.set_context("profile_status", "ok")
            self.memory.set_context(
                "degradations",
                collect_degradations(
                    self.memory.get_context("read_report"),
                    self.memory.get_context("coercions"),
                    profile.to_dict(),
                    "ok",
                ),
            )
            console.print(
                f"  [cyan]🔬 Profile: quality {profile.quality_score}/100, "
                f"{len(profile.warnings)} warning(s).[/]"
            )
            if coercions:
                console.print(
                    f"  [cyan]🔧 Repaired {len(coercions)} column(s): "
                    + ", ".join(f"{c.column} ({c.rule})" for c in coercions) + "[/]"
                )
        except Exception as exc:
            profile_status = f"failed: {exc}"
            self.memory.set_context("profile_status", profile_status)
            self.memory.set_context(
                "degradations", collect_degradations(None, None, None, profile_status)
            )
            console.print(
                f"  [yellow]⚠ Data profiling failed (non-fatal): {exc}. "
                "Running in degraded mode — dataset-nature tools (time-series, "
                "text, geo...) are unavailable without a profile.[/]"
            )

        # Auto-detect target column when user hasn't provided one
        if not metadata.target_column:
            col, confidence = metadata.detect_target_with_confidence(self.objective)
            decision = self._decide_analysis_mode(col, confidence)
            self.memory.set_context("analysis_decision", decision)

            if decision["mode"] == "describe" and col is not None:
                # 7.4 veto: a name-matched numeric column that would
                # otherwise auto-model itself is, in this dataset's context,
                # a descriptive subject — record why and skip modelling
                # rather than training a near-tautological regressor.
                console.print(
                    f"  [cyan]📋 Analysis-mode decision: describe, not model — "
                    f"{decision['rationale']}[/]"
                )
                metadata.task_type = "eda"

            elif col and confidence >= _AUTODETECT_HIGH:
                # High confidence — proceed autonomously
                metadata.target_column = col
                console.print(
                    f"  [green]Auto-detected '{col}' as the target "
                    f"(confidence {confidence:.0%}). Proceeding with analysis...[/]"
                )
                metadata.task_type = metadata.infer_task_type()

            elif col and confidence >= _AUTODETECT_LOW:
                # Medium confidence — prompt if interactive, else use best guess
                if interactive:
                    console.print(
                        f"\n[yellow]Low-confidence target detection: '{col}' "
                        f"(confidence {confidence:.0%}).[/]"
                    )
                    user_input = input(
                        f"  Press ENTER to accept '{col}', or type another column name: "
                    ).strip()
                    chosen = user_input if user_input else col
                    if chosen in metadata.columns:
                        metadata.target_column = chosen
                        metadata.task_type = metadata.infer_task_type()
                    else:
                        console.print(f"  [yellow]Column '{chosen}' not found. Using EDA mode.[/]")
                        metadata.task_type = "eda"
                else:
                    console.print(
                        f"  [yellow]Using best-guess target '{col}' "
                        f"(confidence {confidence:.0%}). Pass target_hint to override.[/]"
                    )
                    metadata.target_column = col
                    metadata.task_type = metadata.infer_task_type()

            else:
                # Very low confidence / no viable candidate
                if interactive:
                    console.print("\n[yellow]Could not confidently determine the target column.[/]")
                    avail = ", ".join(list(metadata.columns.keys())[:15])
                    console.print(f"  Available columns: {avail}")
                    user_input = input(
                        "  Please input the target column name"
                        " (or press ENTER for EDA mode): "
                    ).strip()
                    if user_input and user_input in metadata.columns:
                        metadata.target_column = user_input
                        metadata.task_type = metadata.infer_task_type()
                    else:
                        metadata.task_type = "eda"
                else:
                    console.print("  [dim]No target column detected. Defaulting to EDA mode.[/]")
                    metadata.task_type = "eda"

        else:
            # Target was already known (explicit hint). IngestDatasetTool now
            # delegates to infer_task_type() itself, but recompute here too —
            # this is the one place a divergence would silently break
            # training (see IMPROVEMENTS.md #1), so don't gate it on
            # `not metadata.task_type` ever again.
            metadata.task_type = metadata.infer_task_type()
            # Record the decision too, so the agenda's "driver" question and
            # the methodology "why modelled" note appear for an explicit target.
            self.memory.set_context("analysis_decision", {
                "mode": "model",
                "target": metadata.target_column,
                "candidate": metadata.target_column,
                "rationale": f"Target '{metadata.target_column}' was specified explicitly.",
                "alternatives_rejected": [],
            })

        self.memory.store_dataset_metadata(metadata)
        # Generic context store, not just the DatasetMetadata field — lets
        # BaseTool.requires_context declarations (e.g. select_statistical_test's
        # group_column) fill themselves in without controller-side special-casing.
        self.memory.set_context("target_column", metadata.target_column)
        self.memory.append_tool_result(result)

        # Re-profile with the now-known target column so class-imbalance
        # warnings (which need a target to check) are included — the first
        # pass above ran before the target was decided, deliberately, since
        # the analysis-mode decision needed the profile first. Cheap next to
        # the read/coerce already done; non-fatal, and only runs when the
        # first pass actually succeeded.
        if self._pending_df is not None and self.last_profile is not None:
            try:
                reprofiled = profile_dataframe(self._pending_df, target_column=metadata.target_column)
                reprofiled.domains = self.last_profile.domains
                reprofiled.grain = self.last_profile.grain
                reprofiled.entity_col = self.last_profile.entity_col
                reprofiled.rows_per_entity = self.last_profile.rows_per_entity
                self.last_profile = reprofiled
                self.memory.set_context("data_profile", reprofiled.to_dict())
                self.memory.set_context("data_profile_summary", reprofiled.to_prompt_string())
                self.memory.set_context(
                    "degradations",
                    collect_degradations(
                        self.memory.get_context("read_report"),
                        self.memory.get_context("coercions"),
                        reprofiled.to_dict(),
                        "ok",
                    ),
                )
            except Exception:
                pass  # keep the pre-target profile rather than lose it
        self._pending_df = None
        self._stash_profile_context()

        # 7.5 — question agenda: what a human analyst would ask of this
        # data, generated once profiling and the analysis-mode decision are
        # both settled. Stored in context so the report's methodology
        # section can show real coverage ("N questions, M answered") rather
        # than only ever listing which tools ran.
        try:
            agenda = build_agenda(
                self.last_profile, self.memory.get_context("analysis_decision"), self.objective
            )
            self.memory.set_context("question_agenda", [q.to_dict() for q in agenda])
        except Exception as exc:
            console.print(f"  [yellow]⚠ Question agenda build skipped (non-fatal): {exc}[/]")

        return metadata

    def _stash_profile_context(self) -> None:
        """Archetype and PII columns from the live profile into context,
        where PromptManager reads them. Read with getattr: profiles built
        before those attributes existed simply lack them."""
        profile = self.last_profile
        if profile is None:
            return
        archetype = getattr(profile, "archetype", None)
        if archetype:
            self.memory.set_context(
                "data_archetype",
                {"archetype": str(archetype), "evidence": getattr(profile, "archetype_evidence", None)},
            )
        self.memory.set_context(
            "pii_columns", [c.name for c in profile.columns if getattr(c, "pii", None)]
        )

    def _add_degradation(self, note: str) -> None:
        degradations = list(self.memory.get_context("degradations") or [])
        if note not in degradations:
            degradations.append(note)
            self.memory.set_context("degradations", degradations)

    def _stop_requested(self) -> bool:
        """True once the caller's `should_stop` hook asks the run to end.

        Sticky: after the first True the run stays stopped. The first time it
        fires it records a degradation, so both reports say the result is
        partial instead of presenting it as a finished analysis.
        """
        if self._stopped:
            return True
        if self.should_stop is None:
            return False
        try:
            requested = bool(self.should_stop())
        except Exception:
            # A broken hook must never abort an analysis; treat it as "keep going".
            return False
        if requested:
            self._stopped = True
            self._add_degradation(
                "The run was stopped before it finished, so this report covers only the "
                "steps completed up to that point."
            )
        return requested

    def _notify_findings(self, findings: list[Finding]) -> None:
        """Hand each new finding to `on_finding_callback`, if one is set.

        A progress display must never break the analysis, so a failing callback
        is skipped rather than raised.
        """
        callback = self.on_finding_callback
        if callback is None:
            return
        for finding in findings:
            try:
                callback(finding)
            except Exception:  # display hook: never fatal to the run
                continue

    def _llm_budget_exhausted(self) -> bool:
        """MAX_LLM_TOKENS_PER_RUN reached — no further LLM calls this run.
        Records the degradation (once) the first time it is hit."""
        cap = max_llm_tokens_per_run()
        if not cap or self._rlm_engine is None:
            return False
        used = int(self._rlm_engine.usage_summary().get("total_tokens", 0) or 0)
        if used < cap:
            return False
        self.memory.set_context("llm_error", f"LLM token cap reached ({used:,} of {cap:,} tokens)")
        self._add_degradation(
            f"LLM token cap reached ({used:,} tokens, MAX_LLM_TOKENS_PER_RUN={cap:,}) — "
            "the rest of the run was synthesised deterministically from tool output."
        )
        return True

    def _register_library_tools(self) -> None:
        """ENABLE_TOOL_LIBRARY: register generated tools from earlier runs
        whose required columns exist here with the same kinds."""
        from src.core.tool_factory import (
            MAX_GENERATED_TOOLS,
            load_compatible_tools,
            record_library_use,
            register_and_persist,
            tool_library_dir,
        )

        if self.last_profile is None:
            return
        column_kinds = {c.name: c.kind for c in self.last_profile.columns}
        library = tool_library_dir()
        registered = 0
        for spec in load_compatible_tools(library, column_kinds):
            if registered >= MAX_GENERATED_TOOLS or self.tool_registry.has(spec.name):
                continue
            register_and_persist(spec, self.tool_registry, self.memory, self._output_dir)
            # use_count = runs that registered the tool, counted here once per run.
            record_library_use(library, spec)
            registered += 1
        if registered:
            console.print(f"  [green]📚 Registered {registered} tool(s) from the generated-tool library.[/]")

    # ------------------------------------------------------------------
    # Stages 2-7 — Full autonomous analysis pipeline
    # ------------------------------------------------------------------

    def analyze(
        self,
        file_path: str | None = None,
        target_hint: str | None = None,
        interactive: bool = False,
    ) -> dict[str, Any]:
        """Run Stages 2-7: see `_analyze`. Runs with this run's objective in
        scope for the whole call, including the tool thread pool."""
        with objective_scope(self.objective):
            return self._analyze(file_path, target_hint, interactive)

    def _analyze(
        self,
        file_path: str | None = None,
        target_hint: str | None = None,
        interactive: bool = False,
    ) -> dict[str, Any]:
        """
        Run the complete autonomous analysis pipeline (Stages 2-7).
        Optionally loads the dataset first if file_path is provided.

        Returns:
            Final analysis report as a structured dict.
        """
        if file_path is not None:
            self.load_dataset(file_path, target_hint=target_hint, interactive=interactive)

        if not self.memory.dataset_metadata:
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")

        self._prepare_reasoning()

        console.print("\n[bold magenta]🚀 Starting Autonomous Analysis — Stages 2-7[/]\n")
        final_result = self._reasoning_loop()
        final_result = self._finalize_analysis(final_result)
        return final_result

    def _prepare_reasoning(self) -> None:
        """Stage 2 setup: enforce LOCAL_ONLY, build the PromptManager and RLMEngine,
        seed the planner's draft plan and register library tools."""
        # LOCAL_ONLY: nothing may leave the machine. A cloud provider is
        # refused up front and the run continues deterministically.
        if self.use_llm and local_only() and self.llm_client.provider not in LOCAL_PROVIDERS:
            console.print(
                f"[yellow]⚠ LOCAL_ONLY=true and provider '{self.llm_client.provider}' is not "
                "local — running the deterministic plan instead.[/]"
            )
            self._add_degradation(
                f"LOCAL_ONLY=true refused the non-local LLM provider '{self.llm_client.provider}' — "
                "the run was deterministic and no data was sent to an LLM."
            )
            self.use_llm = False

        # Initialise PromptManager and RLMEngine. Tool descriptions are
        # filtered/ranked against the dataset's profile — the planner only
        # ever sees tools that actually apply to this data's nature.
        tool_desc, short_desc, compact_desc = self._tool_blocks()
        ctx_tokens = (
            min(self.llm_client.get_context_window(), _MAX_PLANNER_CONTEXT_TOKENS)
            if self.use_llm and not os.getenv("LLM_CONTEXT_TOKENS")
            else None
        )
        self._prompt_manager = PromptManager(
            self.memory,
            tool_desc,
            self.max_iterations,
            short_tool_descriptions=short_desc,
            compact_tool_descriptions=compact_desc,
            use_ml=self.use_ml,
            context_tokens=ctx_tokens,
        )
        if self.use_llm:
            # The deterministic plan doubles as the planner's cycle-1 draft:
            # a small model that edits a profile-grounded plan does far
            # better than one planning from a blank page.
            try:
                self.memory.set_context("draft_plan", self._build_fallback_plan()["steps"])
            except Exception:
                logger.debug("could not seed the planner's draft plan", exc_info=True)
            # Registered after the tool list and draft plan are built, so
            # library tools are listed once — in the per-cycle "Tools you
            # created" block, like any generated tool.
            if code_execution_enabled():
                from src.core.tool_factory import tool_library_enabled

                if tool_library_enabled():
                    try:
                        self._register_library_tools()
                    except Exception as exc:
                        console.print(f"  [yellow]⚠ Tool library skipped (non-fatal): {exc}[/]")
        self._rlm_engine = RLMEngine(
            llm_callable=self.llm_client.call,
            system_prompt=self._prompt_manager.get_system_prompt(),
            max_depth=int(os.getenv("RLM_MAX_DEPTH", "5")),
            base_max_tokens=getattr(self.llm_client, "max_tokens", 4096),
        )

    def _reasoning_loop(self) -> dict[str, Any]:
        """Stages 2-6: run reasoning cycles until the analysis completes, stops or
        hits max_iterations. Returns the raw final result (before post-run audits)."""
        assert self._prompt_manager is not None and self._rlm_engine is not None
        final_result: dict[str, Any] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task_id = progress.add_task("Reasoning…", total=None)

            for iteration in range(1, self.max_iterations + 1):
                self.memory.iteration_count = iteration
                self._rlm_engine.set_iteration(iteration)
                progress.update(
                    task_id,
                    description=f"Stage 2/4/5 — Reasoning cycle {iteration}/{self.max_iterations}",
                )

                # ---- Stage 2 / 4+5: Reasoning Phase ----
                if iteration == 1:
                    user_prompt = self._prompt_manager.get_initial_user_prompt()
                    stage_label = "stage2:initial_reasoning"
                else:
                    user_prompt = self._prompt_manager.get_iteration_user_prompt()
                    stage_label = f"stage4-5:iteration_{iteration}"

                if self.on_iteration_callback:
                    self.on_iteration_callback(iteration, stage_label)

                # ---- Cooperative stop: before any LLM call or step this cycle ----
                if self._stop_requested():
                    console.print("[yellow]⏹ Stop requested — synthesising a report from results so far.[/]")
                    final_result = self._deterministic_final()
                    break

                # ---- Deterministic mode: no LLM, by choice ----
                if not self.use_llm:
                    final_result = self._deterministic_cycle(progress, task_id)
                    break

                # ---- Per-run token cap (MAX_LLM_TOKENS_PER_RUN) ----
                if self._llm_budget_exhausted():
                    console.print("[yellow]⚠ LLM token cap reached — synthesising final report from results.[/]")
                    final_result = self._deterministic_final()
                    break
                self.llm_client.stage = stage_label

                # ---- Reasoning with graceful degradation ----
                llm_response, final = self._request_plan(iteration, user_prompt, stage_label)
                if final is not None:
                    final_result = final
                    break

                if iteration == 1:
                    self._store_data_understanding(llm_response)

                # A "complete" reply before any tool ran can only be
                # invented insights: run the profile-driven plan first.
                if llm_response.get("status") == "complete" and not any(
                    r.tool_name != "planner" for r in self.memory.tool_results
                ):
                    console.print(
                        "[yellow]ℹ LLM signalled complete before any analysis ran — "
                        "running the profile-driven plan first.[/]"
                    )
                    llm_response = self._build_fallback_plan()

                llm_response, final = self._handle_completion(
                    iteration, user_prompt, stage_label, llm_response
                )
                if final is not None:
                    final_result = final
                    break

                # ---- Parse plan steps (tolerant of malformed entries) ----
                steps = self._parse_steps(llm_response, cap=not llm_response.get("deterministic"))
                if not steps:
                    console.print(
                        f"[yellow]⚠ No valid steps on iteration {iteration} — synthesising final report.[/]"
                    )
                    final_result = self._final_synthesis("stage7:no_steps_synthesis")
                    break
                # ---- No-progress exit: the planner only repeats work already done ----
                if iteration >= max(2, self.min_iterations) and self._all_steps_cached(steps):
                    console.print(
                        f"\n[bold green]✅ Iteration {iteration} re-planned only completed steps — "
                        "analysis has converged.[/]"
                    )
                    final_result = self._final_synthesis("stage7:converged_synthesis")
                    break
                self.memory.store_analysis_plan(steps)

                # ---- Stage 3: Tool Selection & Execution ----
                progress.update(task_id, description=f"Stage 3 — Executing {len(steps)} tool(s)…")
                self._execute_steps(steps)

                # ---- Cooperative stop: after a step batch ----
                if self._stop_requested():
                    console.print("[yellow]⏹ Stop requested — synthesising a report from results so far.[/]")
                    self.memory.save()
                    final_result = self._deterministic_final()
                    break

                # ---- Stage 6: RLM Decomposition (if enabled & many features, once only) ----
                if self.enable_rlm and not self._rlm_decomposed and self._should_decompose():
                    progress.update(task_id, description="Stage 6 — RLM task decomposition…")
                    self._run_rlm_decomposition()
                    self._rlm_decomposed = True

                self.memory.save()

            else:
                # Max iterations reached
                console.print("[yellow]⚠ Max iterations reached — generating final report.[/]")
                final_result = self._final_synthesis("stage7:max_iter_synthesis")

        return final_result

    def _request_plan(
        self, iteration: int, user_prompt: str, stage_label: str
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Ask the LLM for this cycle's reply, degrading gracefully on failure.

        An LLM/API failure must never abort a running analysis: iteration 1 falls
        back to a deterministic plan, later iterations synthesise a final answer
        from existing results. Returns `(reply, final)`; a non-None `final` means
        the loop should stop with that result."""
        assert self._rlm_engine is not None
        try:
            try:
                llm_response = self._rlm_engine.invoke(
                    user_prompt, depth=0, stage=stage_label
                )
            except ValueError as first_exc:
                # A malformed/truncated/empty reply (ValueError from
                # LLMClient.call) is usually a one-off — one retry
                # asking for a compact plan is far cheaper than
                # discarding the LLM planner for the whole first
                # cycle. Transport errors (timeouts, rate limits) are
                # already retried by the SDK and fall straight
                # through to the fallback below.
                console.print(
                    f"[yellow]⚠ Unusable LLM reply on iteration {iteration} "
                    f"({first_exc}) — retrying once with a compact-JSON reminder.[/]"
                )
                llm_response = self._rlm_engine.invoke(
                    user_prompt + _COMPACT_RETRY_NOTE,
                    depth=0,
                    stage=f"{stage_label}:retry",
                )
            llm_response = self._normalise_reply(llm_response)
            if llm_response.get("status") == "error":
                raise RuntimeError(
                    str(llm_response.get("error", "Unknown LLM error"))
                )
        except Exception as exc:
            self.memory.set_context("llm_error", f"{type(exc).__name__}: {exc}")
            console.print(
                f"[yellow]⚠ LLM failure on iteration {iteration}: {exc}[/]"
            )
            if iteration == 1:
                console.print("[yellow]  → Using deterministic fallback plan.[/]")
                # Record it in the degradations log both reports
                # render — otherwise, when a later cycle's LLM call
                # succeeds and supplies the final answer, nothing in
                # the report says the plan itself wasn't the LLM's.
                degradations = list(self.memory.get_context("degradations") or [])
                degradations.append(
                    "LLM planning call failed on the first reasoning cycle "
                    f"({type(exc).__name__}: {str(exc)[:200]}) — the analysis plan "
                    "was chosen by the deterministic profile-driven fallback."
                )
                self.memory.set_context("degradations", degradations)
                llm_response = self._build_fallback_plan()
            else:
                console.print("[yellow]  → Synthesising final report from results.[/]")
                return {}, self._deterministic_final()
        return llm_response, None

    def _handle_completion(
        self, iteration: int, user_prompt: str, stage_label: str, llm_response: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Decide what a "complete" reply means (Stage 7 trigger).

        Before `min_iterations` the planner is pushed for one more hypothesis;
        otherwise the reply is accepted. Returns `(reply, final)`: a non-None
        `final` ends the loop with that result, else `reply` is the plan to run.
        """
        if llm_response.get("status") != "complete":
            return llm_response, None
        if iteration >= self.min_iterations:
            console.print("\n[bold green]✅ LLM signalled analysis complete.[/]")
            return llm_response, llm_response
        assert self._rlm_engine is not None
        console.print(
            f"[yellow]ℹ LLM signalled complete on iteration {iteration} "
            f"(< min_iterations {self.min_iterations}) — prompting for deeper hypothesis exploration.[/]"
        )
        continue_prompt = (
            user_prompt
            + f"\n\n[SYSTEM DIRECTIVE: Minimum exploration cycles not yet reached "
            f"(currently iteration {iteration} of minimum {self.min_iterations}). "
            "Do NOT return Form 2 yet. Formulate a specific follow-up hypothesis, anomaly check, "
            "or deeper investigation, and return Form 1 (Action Plan) with 1–3 steps. "
            "Use execute_dynamic_code if you need a custom calculation.]"
        )
        try:
            if self._llm_budget_exhausted():
                raise RuntimeError("LLM token cap reached")
            self.llm_client.stage = f"{stage_label}:deepen_exploration"
            reprompt_res = self._rlm_engine.invoke(
                continue_prompt, depth=0, stage=f"{stage_label}:deepen_exploration"
            )
            reprompt_res = self._normalise_reply(reprompt_res)
            if reprompt_res.get("status") != "complete":
                return reprompt_res, None
            console.print("\n[bold green]✅ LLM confirmed analysis complete.[/]")
            return reprompt_res, reprompt_res
        except Exception as exc:
            console.print(f"[yellow]⚠ Exploration reprompt failed ({exc}) — accepting completion.[/]")
            return llm_response, llm_response

    def _deterministic_cycle(self, progress: Progress, task_id: TaskID) -> dict[str, Any]:
        """Deterministic mode (LLM off by choice): run the profile-driven plan once.

        Distinct from the failure path in `_request_plan`. Nothing is "degraded"
        here — the user asked for a deterministic run, so the plan executes once
        and the report is synthesised from tool output without any network call.
        """
        console.print(
            "[cyan]🔌 LLM disabled — running the deterministic "
            "profile-driven plan.[/]"
        )
        llm_response = self._build_fallback_plan()
        llm_response["reasoning"] = (
            "LLM disabled for this run — plan selected from the "
            "dataset profile and domain inference."
        )
        steps = self._parse_steps(llm_response, cap=False)
        if steps:
            self.memory.store_analysis_plan(steps)
            progress.update(task_id, description=f"Stage 3 — Executing {len(steps)} tool(s)…")
            self._execute_steps(steps)
            self.memory.save()
        return self._deterministic_final()

    def _attach_run_metadata(self, final_result: dict[str, Any]) -> None:
        """Strip plumbing from the raw result and attach the run's findings,
        governance, API telemetry and question coverage."""
        # P3.1 — `final_result` is, on the "complete"/max-iteration paths,
        # the raw LLM response dict returned by RLMEngine.invoke(), which
        # LLMClient.call() may have stamped with `_rlm_usage` (this run's
        # own token/cost accounting — see LLMClient.call). That's plumbing
        # for RLMEngine's running totals, not an analytical result, and
        # must not leak into the persisted report/raw JSON. The engine's
        # own running totals are the real place for this to live.
        final_result.pop("_rlm_usage", None)
        if self._stopped:
            # Only present on a stopped run, so a finished run's result is unchanged.
            final_result["stopped"] = True
        if self._rlm_engine is not None:
            self.memory.set_context("llm_usage", self._rlm_engine.usage_summary())

        # Every path through this loop must leave `findings` on the result —
        # the LLM-complete and max-iteration branches return the raw LLM
        # response dict, which doesn't carry the bus. Backfilling here means
        # the reports/dashboard can always read final_result["findings"]
        # regardless of which branch produced the final answer.
        if "findings" not in final_result:
            final_result["findings"] = [f.to_dict() for f in self.memory.ranked_findings()]
        if self.memory.get_context("data_understanding"):
            final_result.setdefault("data_understanding", self.memory.get_context("data_understanding"))
        column_roles = (self.memory.get_context("column_roles") or {}).get("accepted")
        if column_roles:
            final_result.setdefault("column_roles", column_roles)
        governance = self._governor.summary(self.memory.get_context("llm_usage"))
        self.memory.set_context("governance", governance)
        final_result["governance"] = governance
        if self.use_llm:
            profile = get_limiter().get_profile(self.llm_client.provider, self.llm_client.model)
            api_telem = {
                "provider": profile.provider,
                "model": profile.model,
                "context_window": profile.context_window,
                "rpm_limit": profile.rpm_limit,
                "rpm_remaining": profile.rpm_remaining,
                "tpm_limit": profile.tpm_limit,
                "tpm_remaining": profile.tpm_remaining,
                "speed_tag": profile.speed_tag,
            }
            final_result["api_telemetry"] = api_telem
            self.memory.set_context("api_telemetry", api_telem)
        try:
            agenda = self.memory.get_context("question_agenda") or []
            final_result["coverage"] = coverage_report(
                [Question(**q) for q in agenda], final_result["findings"]
            )
        except Exception:
            logger.debug("question coverage report skipped", exc_info=True)

    def _run_post_hoc_audits(self, final_result: dict[str, Any]) -> None:
        """Claim verification plus the post-hoc audits (deliverable contract,
        dependence, target leakage, fragility, causal language) and check marks."""
        # ---- Verbatim-metric validation: enforce "cite only verbatim
        # metrics" as a mechanism, not just a prompt instruction ----
        unverified = self._flag_unverified_claims(final_result)
        if unverified:
            self.memory.set_context("unverified_claims", unverified)
            console.print(
                f"[yellow]⚠ {len(unverified)} unverified metric claim(s) in the "
                f"final synthesis — see memory context 'unverified_claims'.[/]"
            )

        # ---- Shared cleaned-dataset read for the post-hoc audits below
        # (Deliverable Contract, Dependence/Confounding, Target Leakage,
        # Finding Fragility) — one read, reused, instead of reading the same
        # path twice. ----
        audit_df: pd.DataFrame | None = None
        try:
            from src.core.analysis_io import read_analysis_df
            audit_path = self.memory.get_context("cleaned_file_path") or getattr(self.memory.dataset_metadata, "file_path", None)
            if audit_path:
                audit_df = read_analysis_df(str(audit_path))
        except Exception:
            audit_df = None

        self._audit_contract(final_result, audit_df)
        if audit_df is not None:
            self._audit_dependence(final_result, audit_df)
        leakage_alerts = self._audit_target_leakage(final_result, audit_df)
        fragility_by_id = self._audit_fragility(final_result, audit_df)
        self._attach_audit_checks(final_result, leakage_alerts, fragility_by_id)

    def _audit_contract(self, final_result: dict[str, Any], audit_df: pd.DataFrame | None) -> None:
        """Deliverable Contract Audit (Phase 2)."""
        if not self.deliverable_contract.is_empty():
            audit_rep = audit_deliverables(self.deliverable_contract, final_result, df=audit_df)
            self.memory.set_context("deliverable_audit", {
                "delivered": audit_rep.delivered,
                "missing": audit_rep.missing,
                "repaired": audit_rep.repaired,
            })
            if audit_rep.missing:
                console.print(
                    f"[yellow]⚠ Deliverable Contract: {len(audit_rep.missing)} deliverable(s) missing: "
                    f"{', '.join(audit_rep.missing)}[/]"
                )
            if audit_rep.repaired:
                console.print(
                    f"[green]✓ Deliverable Contract: {len(audit_rep.repaired)} deliverable(s) repaired: "
                    f"{', '.join(audit_rep.repaired)}[/]"
                )


    def _audit_dependence(self, final_result: dict[str, Any], df: pd.DataFrame) -> None:
        """Dependence & Confounding Audit (Phase 4)."""
        try:
            self._audit_dependence_structure(df)
        except Exception as exc:
            console.print(f"  [yellow]⚠ Dependence/Simpson's-paradox audit skipped (non-fatal): {exc}[/]")
        # Re-derive from ranked_findings(), not the raw list — the raw
        # list bypasses _drop_unskilled_drivers/is_trivial suppression,
        # which would resurrect noise-level "driver" findings on every
        # run that reaches this point (this block used to overwrite with
        # raw findings unconditionally and broke that suppression).
        final_result["findings"] = [f.to_dict() for f in self.memory.ranked_findings()]

    def _audit_target_leakage(
        self, final_result: dict[str, Any], df: pd.DataFrame | None
    ) -> list[dict[str, Any]]:
        """Target Leakage Audit (sensitivity.py). Returns the alerts, also stored on the result."""
        leakage_alerts: list[dict[str, Any]] = []
        target_col = getattr(self.memory.dataset_metadata, "target_column", None)
        if df is not None and target_col:
            try:
                from src.core.sensitivity import detect_target_leakage
                leakage_alerts = detect_target_leakage(df, target_col)
            except Exception as exc:
                console.print(f"  [yellow]⚠ Target leakage audit skipped (non-fatal): {exc}[/]")
            if leakage_alerts:
                self.memory.set_context("target_leakage_alerts", leakage_alerts)
                critical = [a for a in leakage_alerts if a.get("severity") == "critical"]
                console.print(
                    f"  [red]⚡ Target Leakage Audit: {len(leakage_alerts)} alert(s), "
                    f"{len(critical)} critical — {', '.join(a['column'] for a in leakage_alerts[:5])}[/]"
                )
        final_result["target_leakage_alerts"] = leakage_alerts
        return leakage_alerts

    def _audit_fragility(
        self, final_result: dict[str, Any], df: pd.DataFrame | None
    ) -> dict[str, bool]:
        """Finding Fragility Audit (sensitivity.py): finding_id -> is_fragile."""
        fragility_by_id: dict[str, bool] = {}
        if df is None:
            return fragility_by_id
        try:
            from src.core.sensitivity import audit_finding_sensitivity
            # Bounded to the top-ranked findings — a jackknife pass
            # recomputes the effect once per finding, which is too
            # costly to run over every finding on a long analysis.
            for f in self.memory.ranked_findings()[:8]:
                if f.measure and f.effect is not None:
                    report = audit_finding_sensitivity(df, f)
                    is_fragile = bool(report.get("is_fragile"))
                    fragility_by_id[f.finding_id] = is_fragile
                    if is_fragile:
                        f.caveats.append(report["diagnosis"])
            final_result["findings"] = [f.to_dict() for f in self.memory.ranked_findings()]
        except Exception as exc:
            console.print(f"  [yellow]⚠ Finding fragility audit skipped (non-fatal): {exc}[/]")
        return fragility_by_id

    def _attach_audit_checks(
        self,
        final_result: dict[str, Any],
        leakage_alerts: list[dict[str, Any]],
        fragility_by_id: dict[str, bool],
    ) -> None:
        """Causal Claim Guard (Phase 4), then the audited-entry check marks."""
        study_design = getattr(self, "study_design", "observational")
        causal_warns = audit_findings_causal_language(self.memory.findings, study_design=study_design)
        if causal_warns:
            console.print(
                f"[yellow]⚠ Causal Claim Guard: {len(causal_warns)} claim(s) downgraded to association "
                f"due to {study_design} study design.[/]"
            )

        # ---- Audited-entry check marks (FrontendPlan.md section 5) ----
        # Runs after every audit above has had a chance to write onto
        # self.memory.findings, so evidence["checks"] reflects the same run
        # the Markdown/HTML reports and the dashboard are about to read.
        causal_flagged_ids = {
            f.finding_id for f in self.memory.findings
            if CAVEAT_OBSERVATIONAL in f.caveats
        }
        attach_finding_checks(
            self.memory.findings,
            leakage_alerts=leakage_alerts,
            fragility_by_id=fragility_by_id,
            causal_flagged_ids=causal_flagged_ids,
        )
        final_result["findings"] = [f.to_dict() for f in self.memory.ranked_findings()]

    def _finalize_analysis(self, final_result: dict[str, Any]) -> dict[str, Any]:
        """Stage 7: attach metadata, run the audits, then write the reports."""
        self._attach_run_metadata(final_result)
        self._run_post_hoc_audits(final_result)

        # ---- Stage 7: Report Generation ----
        self._generate_final_report(final_result)
        self._generate_dashboard()
        self._generate_html_report(final_result)
        self._write_run_summary()

        # Print reasoning trace
        if self._rlm_engine:
            console.print()
            self._rlm_engine.print_reasoning_trace()


        return final_result



    # ------------------------------------------------------------------
    # Resilience helpers — plan parsing and LLM-failure fallbacks
    # ------------------------------------------------------------------


    def _apply_column_roles(self, proposals: Any) -> None:
        """Validate the planner's proposed column roles against the data, keep
        only the confirmed ones on the profile (tools then match those columns
        like a name match) and refresh the tool blocks so the next cycle lists
        newly applicable tools. Rejections are logged dim only."""
        if not isinstance(proposals, dict) or not proposals or self.last_profile is None:
            return
        meta = self.memory.dataset_metadata
        if meta is None:
            return
        try:
            from src.core.analysis_io import read_analysis_df
            from src.core.roles import validate_roles

            df = read_analysis_df(str(self.memory.get_context("cleaned_file_path") or meta.file_path))
            accepted, rejected = validate_roles(df, proposals)
        except Exception as exc:
            console.print(f"  [dim]Column roles skipped: {exc}[/]")
            return
        self.memory.set_context("column_roles", {"accepted": accepted, "rejected": rejected})
        for column, reason in rejected.items():
            console.print(f"  [dim]Column role rejected: {column} ({reason})[/]")
        if not accepted:
            return
        self.last_profile.role_overrides = accepted
        self.memory.set_context("data_profile", self.last_profile.to_dict())
        if self._prompt_manager is not None:
            self._prompt_manager.refresh_tools(*self._tool_blocks())
            if self._rlm_engine is not None:
                self._rlm_engine.set_system_prompt(self._prompt_manager.get_system_prompt())

    #: Caps on the planner's cycle-1 data_understanding block — it is carried
    #: into every later prompt, so an over-long answer taxes every cycle.
    _UNDERSTANDING_TEXT_KEYS = ("subject", "domain", "time_column")
    _UNDERSTANDING_LIST_KEYS = ("key_measures", "key_dimensions", "caveats", "questions")

    def _store_data_understanding(self, llm_response: dict[str, Any]) -> None:
        """Keep the planner's cycle-1 `data_understanding`, bounded, in
        memory context: the prompts carry it forward and the reports can
        show how the agent read the data. Column lists are filtered to real
        columns so a hallucinated name never propagates."""
        raw = llm_response.get("data_understanding")
        if not isinstance(raw, dict):
            return
        self._apply_column_roles(raw.get("roles"))
        columns = set(self.memory.dataset_metadata.columns) if self.memory.dataset_metadata else set()
        clean: dict[str, Any] = {}
        for key in self._UNDERSTANDING_TEXT_KEYS:
            value = raw.get(key)
            if isinstance(value, str) and value.strip() and value.strip().lower() != "null":
                clean[key] = value.strip()[:160]
        if clean.get("time_column") and clean["time_column"] not in columns:
            clean.pop("time_column")
        archetype = str(raw.get("archetype") or "").strip().lower().replace("-", "_").replace(" ", "_")
        if archetype in ARCHETYPES:
            clean["archetype"] = archetype
        for key in self._UNDERSTANDING_LIST_KEYS:
            values = raw.get(key)
            if not isinstance(values, list):
                continue
            items = [str(v).strip()[:160] for v in values if str(v).strip()]
            if key in ("key_measures", "key_dimensions"):
                items = [v for v in items if v in columns]
            if items:
                clean[key] = items[:6]
        if clean:
            self.memory.set_context("data_understanding", clean)
















    # ------------------------------------------------------------------
    # Stage 3 execution helper
    # ------------------------------------------------------------------





    # ------------------------------------------------------------------
    # Plan validation — deterministic, before a step spends a tool run
    # ------------------------------------------------------------------














    # ------------------------------------------------------------------
    # Stage 6 — RLM decomposition
    # ------------------------------------------------------------------

    def _should_decompose(self) -> bool:
        """Trigger decomposition for wide datasets — the same structural
        fact (is_high_dimensional) that gates dimensionality_analysis, so
        there's one source of truth for "this data has a lot of features"."""
        mode = os.getenv("RLM_DECOMPOSE", "auto").strip().lower()
        if mode == "off":
            return False
        if mode == "on":
            return True
        if self.last_profile is not None:
            wide = self.last_profile.is_high_dimensional
        else:
            meta = self.memory.dataset_metadata
            wide = meta is not None and meta.column_count > 15
        if not wide:
            return False
        # Auto: each sub-task is a paid LLM call, so skip it when the
        # deterministic tools already covered the numeric columns.
        reason = self._rlm_skip_reason()
        if reason:
            if not getattr(self, "_rlm_skip_logged", False):
                self._rlm_skip_logged = True
                console.print(f"  [dim]RLM decomposition skipped: {reason}.[/]")
            return False
        return True

    def _rlm_skip_reason(self) -> str:
        meta = self.memory.dataset_metadata
        num_cols = meta.numerical_cols if meta is not None else []
        if len(num_cols) < 8:
            return f"only {len(num_cols)} numeric columns (< 8)"
        if self._prompt_manager is None:
            return ""
        groups = [num_cols[i: i + 8] for i in range(0, len(num_cols), 8)]
        if all(self._prompt_manager.rlm_group_findings(g) for g in groups):
            return f"findings already cover all {len(groups)} numeric column groups"
        return ""

    def _run_rlm_decomposition(self) -> None:
        """
        Stage 6: Decompose the feature space into sub-groups and run
        targeted LLM sub-calls on each group.

        This is the core RLM innovation: instead of one monolithic context,
        each feature group gets its own focused reasoning call.
        """
        if self._rlm_engine is None or self._prompt_manager is None:
            return

        meta = self.memory.dataset_metadata
        if meta is None:
            return

        num_cols = meta.numerical_cols
        cat_cols = meta.categorical_cols

        # Partition numerical columns into groups of ≤8
        groups: dict[str, list[str]] = {}
        chunk_size = 8
        for i in range(0, len(num_cols), chunk_size):
            groups[f"numerical_group_{i // chunk_size + 1}"] = num_cols[i: i + chunk_size]
        if cat_cols:
            groups["categorical_group"] = cat_cols[:10]

        sub_tasks = [
            RLMSubTask(
                task_id=gid,
                description=(
                    f"Analyse {len(cols)}-feature group: "
                    f"{', '.join(sanitize_for_prompt(c) for c in cols[:5])}…"
                ),
                context={"columns": [sanitize_for_prompt(c) for c in cols], "group_id": gid},
            )
            for gid, cols in groups.items()
        ]

        if not sub_tasks:
            return

        def build_prompt(task: RLMSubTask) -> str:
            assert self._prompt_manager is not None
            ctx_summary = json.dumps(task.context)
            return self._prompt_manager.get_rlm_subtask_prompt(
                task_id=task.task_id,
                description=task.description,
                context_summary=ctx_summary,
                columns=groups[task.task_id],
            )

        # Graceful degradation, same rationale as the stage-2 planning call
        # (line ~1184): decomposition results are optional context for final
        # synthesis, not a required step, so a transient LLM/API failure here
        # (rate limit, timeout, provider outage) must not abort the run —
        # it should just mean synthesis proceeds without the extra detail.
        cap = max_llm_tokens_per_run()
        if self._llm_budget_exhausted():
            return
        self.llm_client.stage = "stage6:rlm_decomposition"
        try:
            sub_results = self._rlm_engine.decompose_and_invoke(
                sub_tasks=sub_tasks,
                prompt_builder=build_prompt,
                depth=1,
                max_total_tokens=cap or None,
                system_prompt=RLM_SUBTASK_SYSTEM,
            )
        except Exception as exc:
            self.memory.set_context("rlm_decomposition_error", f"{type(exc).__name__}: {exc}")
            console.print(f"[yellow]⚠ RLM decomposition failed: {exc}[/]")
            console.print("[yellow]  → Continuing without sub-task decomposition.[/]")
            return

        # Store sub-results in memory context for final synthesis
        self.memory.set_context("rlm_sub_results", sub_results)
        console.print(
            f"  [green]✓ RLM decomposition complete: "
            f"{len(sub_results)} sub-task(s) resolved.[/]"
        )
        if self._rlm_engine.last_failures:
            self.memory.set_context("rlm_sub_failures", self._rlm_engine.last_failures)
            console.print(
                f"  [yellow]  {len(self._rlm_engine.last_failures)} sub-task(s) unresolved: "
                f"{', '.join(self._rlm_engine.last_failures)}[/]"
            )

    # ------------------------------------------------------------------
    # Stage 7 — Report generation
    # ------------------------------------------------------------------





