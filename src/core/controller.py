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

import ast
import copy
import difflib
import json
import logging
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.core.agenda import Question, build_agenda, coverage_report
from src.core.causal_guard import (
    CAVEAT_OBSERVATIONAL,
    audit_findings_causal_language,
    classify_study_design,
)
from src.core.chart_designer import design_from_reply
from src.core.claim_verification import _CANON_PRECISIONS as _CANON_PRECISIONS
from src.core.claim_verification import _KEYWORD_TOOL_MAP as _KEYWORD_TOOL_MAP
from src.core.claim_verification import _KEYWORD_TOOL_PATTERNS as _KEYWORD_TOOL_PATTERNS
from src.core.claim_verification import _NUMBER_RE as _NUMBER_RE
from src.core.claim_verification import _UNVERIFIABLE_SKIP_ABS_INT as _UNVERIFIABLE_SKIP_ABS_INT
from src.core.claim_verification import _canon_number as _canon_number
from src.core.claim_verification import _collect_numbers as _collect_numbers
from src.core.claim_verification import flag_unverified_claims, verified_number_pools
from src.core.coercion import coerce_types
from src.core.dashboard import (
    ChartSpec,
    build_dashboard,
    dashboard_to_json,
    designed_chart,
    merge_designed,
)
from src.core.degradations import collect_degradations
from src.core.deliverable_contract import audit_deliverables, parse_deliverable_contract
from src.core.dependence import check_simpsons_paradox, intraclass_correlation
from src.core.domain_packs import detect_domain_pack, evaluate_domain_pack
from src.core.domains import infer_domains
from src.core.findings import Finding, attach_finding_checks, score_objective_fit
from src.core.governance import (
    AUDIT_SUBDIR,
    LOCAL_PROVIDERS,
    CodeGovernor,
    code_execution_enabled,
    local_only,
    max_llm_tokens_per_run,
)
from src.core.hypothesis import HypothesisTree, generate_counterfactual_probes
from src.core.integrity import evaluate_data_integrity
from src.core.io import get_max_rows, read_any
from src.core.joins import join_related
from src.core.llm_client import _DEFAULT_MODELS as _DEFAULT_MODELS
from src.core.llm_client import LLMClient as LLMClient
from src.core.llm_client import LocalOnlyError as LocalOnlyError
from src.core.memory import AnalysisStep, DatasetMetadata, MemorySystem, ToolResult
from src.core.model_telemetry import get_limiter
from src.core.multiple_testing import adjust_findings_run_level
from src.core.profiler import DatasetProfile, profile_dataframe
from src.core.prompt_manager import (
    ARCHETYPES,
    CHART_DESIGN_PROMPT,
    RLM_SUBTASK_SYSTEM,
    PromptManager,
)
from src.core.question_router import route_question
from src.core.run_config import RunConfig
from src.core.run_context import in_run_context, objective_scope
from src.core.security import sanitize_for_prompt
from src.core.shared_context import SharedAnalysisContext
from src.core.stats_utils import repeated_entity
from src.core.step_validation import _COLUMN_PARAM_NAMES as _COLUMN_PARAM_NAMES
from src.core.step_validation import columns_for, is_column_param, validate_step
from src.core.tool_registry import _INJECTED_PARAMS as _INJECTED_PARAMS
from src.core.tool_registry import ToolRegistry as ToolRegistry
from src.core.tool_registry import _short_tool_description as _short_tool_description
from src.rlm.engine import RLMEngine, RLMSubTask

logger = logging.getLogger(__name__)

console = Console()


class _PreparedStep(NamedTuple):
    """A plan step resolved against run state and cleared to execute."""

    idx: int
    step: AnalysisStep
    tool: Any
    params: dict[str, Any]
    dropped_note: str
    cache_key: str
    cached: ToolResult | None


def _read_dataframe(file_path: str) -> pd.DataFrame:
    """Load a CSV/TSV/Excel dataset for dashboard generation."""
    df, _report = read_any(file_path)
    return df

# Max retries before abandoning a failed step
MAX_STEP_RETRIES = 2
#: Code-running tools fail as part of normal self-correction (a wrong column
#: name, a pandas idiom) — each failure returns a hint the next attempt uses,
#: so they get a larger budget than a deterministic tool that is simply broken.
MAX_CODE_STEP_RETRIES = 5

#: Appended to a reasoning prompt when the first reply was unusable
#: (non-JSON, truncated at max_tokens, or empty) — see analyze().
_COMPACT_RETRY_NOTE = (
    "\n\n## Retry Note\nYour previous reply could not be used (it was not "
    "complete, valid JSON — possibly cut off by the output length limit). "
    "Reply again with ONLY the JSON object: at most 8 steps, one short "
    "sentence per rationale, no markdown fences, no text outside the JSON.\n"
)

#: Ceiling on the prompt budget derived from the model's context window. A
#: 1M-token window is not a reason to send 750k-token planner prompts: cost
#: and latency grow with prompt size while the ranked findings already keep
#: what matters compact. LLM_CONTEXT_TOKENS, when set, overrides all of this.
_MAX_PLANNER_CONTEXT_TOKENS = 32_000

#: Steps run per reasoning cycle — a model that lists twenty is padding, and
#: each extra step costs a tool run plus digest tokens on every later prompt.
_MAX_STEPS_PER_CYCLE = 8

#: Below this context window the chart-design call is skipped: a small model
#: would spend its scarce budget on it and the deterministic dashboard stands.
_MIN_CHART_DESIGN_CONTEXT = 16_000

# Target auto-detection confidence thresholds
_AUTODETECT_HIGH = 0.75   # proceed autonomously above this
_AUTODETECT_LOW  = 0.40   # prompt user (CLI) or best-guess (UI) above this


# ---------------------------------------------------------------------------
# Agent Controller — the top-level orchestrator
# ---------------------------------------------------------------------------

class AgentController:
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

        console.print("\n[bold magenta]🚀 Starting Autonomous Analysis — Stages 2-7[/]\n")
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
                # Distinct from the failure path below. Nothing is "degraded"
                # here — the user asked for a deterministic run, so the
                # profile-driven plan executes once and the report is
                # synthesised from tool output without any network call.
                if not self.use_llm:
                    if iteration == 1:
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
                        progress.update(
                            task_id, description=f"Stage 3 — Executing {len(steps)} tool(s)…"
                        )
                        self._execute_steps(steps)
                        self.memory.save()
                    final_result = self._deterministic_final()
                    break

                # ---- Per-run token cap (MAX_LLM_TOKENS_PER_RUN) ----
                if self._llm_budget_exhausted():
                    console.print("[yellow]⚠ LLM token cap reached — synthesising final report from results.[/]")
                    final_result = self._deterministic_final()
                    break
                self.llm_client.stage = stage_label

                # ---- Reasoning with graceful degradation ----
                # An LLM/API failure must never abort a running analysis:
                # iteration 1 falls back to a deterministic plan, later
                # iterations synthesise a final answer from existing results.
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
                        final_result = self._deterministic_final()
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

                # ---- Check for completion (Stage 7 trigger) ----
                if llm_response.get("status") == "complete":
                    if iteration < self.min_iterations:
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
                                llm_response = reprompt_res
                            else:
                                console.print("\n[bold green]✅ LLM confirmed analysis complete.[/]")
                                final_result = reprompt_res
                                break
                        except Exception as exc:
                            console.print(f"[yellow]⚠ Exploration reprompt failed ({exc}) — accepting completion.[/]")
                            final_result = llm_response
                            break
                    else:
                        console.print("\n[bold green]✅ LLM signalled analysis complete.[/]")
                        final_result = llm_response
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

        # ---- Deliverable Contract Audit (Phase 2) ----
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

        # ---- Dependence & Confounding Audit (Phase 4) ----
        dep_df = audit_df
        if dep_df is not None:
            try:
                self._audit_dependence_structure(dep_df)
            except Exception as exc:
                console.print(f"  [yellow]⚠ Dependence/Simpson's-paradox audit skipped (non-fatal): {exc}[/]")
            # Re-derive from ranked_findings(), not the raw list — the raw
            # list bypasses _drop_unskilled_drivers/is_trivial suppression,
            # which would resurrect noise-level "driver" findings on every
            # run that reaches this point (this block used to overwrite with
            # raw findings unconditionally and broke that suppression).
            final_result["findings"] = [f.to_dict() for f in self.memory.ranked_findings()]

        # ---- Target Leakage & Finding Fragility Audit (sensitivity.py) ----
        leakage_alerts: list[dict[str, Any]] = []
        target_col = getattr(self.memory.dataset_metadata, "target_column", None)
        if dep_df is not None and target_col:
            try:
                from src.core.sensitivity import detect_target_leakage
                leakage_alerts = detect_target_leakage(dep_df, target_col)
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

        fragility_by_id: dict[str, bool] = {}
        if dep_df is not None:
            try:
                from src.core.sensitivity import audit_finding_sensitivity
                # Bounded to the top-ranked findings — a jackknife pass
                # recomputes the effect once per finding, which is too
                # costly to run over every finding on a long analysis.
                for f in self.memory.ranked_findings()[:8]:
                    if f.measure and f.effect is not None:
                        report = audit_finding_sensitivity(dep_df, f)
                        is_fragile = bool(report.get("is_fragile"))
                        fragility_by_id[f.finding_id] = is_fragile
                        if is_fragile:
                            f.caveats.append(report["diagnosis"])
                final_result["findings"] = [f.to_dict() for f in self.memory.ranked_findings()]
            except Exception as exc:
                console.print(f"  [yellow]⚠ Finding fragility audit skipped (non-fatal): {exc}[/]")

        # ---- Causal Claim Guard (Phase 4) ----
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

    def _final_synthesis(self, stage: str) -> dict[str, Any]:
        """One LLM call for the final interpretation of everything found so
        far; the deterministic synthesis when the LLM is off, capped, failing,
        or does not return a usable Form 2 reply."""
        if not self.use_llm or self._prompt_manager is None or self._rlm_engine is None:
            return self._deterministic_final()
        try:
            if self._llm_budget_exhausted():
                raise RuntimeError("LLM token cap reached")
            self.llm_client.stage = stage
            result: dict[str, Any] = self._normalise_reply(self._rlm_engine.invoke(
                self._prompt_manager.get_final_interpretation_prompt(), depth=0, stage=stage
            ))
            if result.get("status") == "error":
                raise RuntimeError(str(result.get("error", "Unknown LLM error")))
            if result.get("status") != "complete" or not result.get("insights"):
                raise RuntimeError("final interpretation reply had no insights")
            return result
        except Exception as exc:
            self.memory.set_context("llm_error", f"{type(exc).__name__}: {exc}")
            return self._deterministic_final()

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

    # ------------------------------------------------------------------
    # Resilience helpers — plan parsing and LLM-failure fallbacks
    # ------------------------------------------------------------------

    def _tool_blocks(self) -> tuple[str, str, str]:
        """(full, short, compact) candidate-tool descriptions for the current profile."""
        args = (self.last_profile, self.memory.dataset_metadata)
        kwargs = {"use_ml": self.use_ml, "use_llm": self.use_llm}
        return (
            self.tool_registry.get_candidate_descriptions(*args, **kwargs),
            self.tool_registry.get_candidate_short_descriptions(*args, **kwargs),
            self.tool_registry.get_candidate_descriptions(*args, **kwargs, compact=True),
        )

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

    _PLAN_KEYS = ("steps", "plan", "next_steps", "actions", "tasks")
    _TOOL_KEYS = ("tool_name", "tool", "name", "function", "action", "tool_call")
    _PARAM_KEYS = ("parameters", "params", "arguments", "args", "input", "inputs")
    _STEP_NO_KEYS = ("step_number", "step", "id", "index")
    _TOOL_PREFIX_RE = re.compile(r"^(?:(?:functions?|tools?)\s*[.:/]\s*)+", re.IGNORECASE)
    _DONE_STATUSES = frozenset({"complete", "completed", "done", "finished", "final", "finish"})
    _GOING_STATUSES = frozenset({"in_progress", "inprogress", "continue", "continuing", "running", "ongoing", "working"})

    @classmethod
    def _plan_items(cls, reply: Any) -> list[Any]:
        """The plan's step list wherever a drifting model put it: a top-level
        list, under steps/plan/next_steps/actions/tasks, or a dict keyed by
        step number."""
        if isinstance(reply, list):
            return reply
        if not isinstance(reply, dict):
            return []
        for key in cls._PLAN_KEYS:
            value = reply.get(key)
            if isinstance(value, list) and value:
                return value
            if isinstance(value, dict) and value:
                if any(k in value for k in cls._TOOL_KEYS):
                    return [value]
                if nested := cls._plan_items(value):
                    return nested
                keyed = [(k, v) for k, v in value.items() if isinstance(v, dict)]
                if keyed:
                    return [
                        {**v, "step_number": int(m.group())} if (m := re.search(r"\d+", str(k))) else v
                        for k, v in keyed
                    ]
        return []

    @classmethod
    def _normalise_reply(cls, reply: Any) -> Any:
        """Repair reply-form drift: a bare step list, a missing or oddly
        spelled `status`. Error replies pass through untouched."""
        if isinstance(reply, list):
            return {"status": "in_progress", "steps": reply}
        if not isinstance(reply, dict) or reply.get("status") == "error":
            return reply
        reply = dict(reply)
        status = re.sub(r"[\s-]+", "_", str(reply.get("status") or "").strip().lower())
        if status in cls._DONE_STATUSES:
            reply["status"] = "complete"
        elif status in cls._GOING_STATUSES or cls._plan_items(reply):
            reply["status"] = "in_progress"
        elif reply.get("insights") or reply.get("recommendations"):
            reply["status"] = "complete"
        return reply

    def _resolve_tool_name(self, raw: str) -> tuple[str | None, str]:
        """(registered tool name or None, the name cleaned of decoration).
        Small models write "functions.clean_data", "Clean_Data", "clean_dat":
        match case-/underscore-insensitively, then by a unique close match."""
        name = self._TOOL_PREFIX_RE.sub("", raw.strip()).strip("`'\" ").removesuffix("()").split(".")[-1].strip()
        if not name or self.tool_registry.has(name):
            return (name or None), name
        names = self.tool_registry.names()
        key = re.sub(r"[\W_]+", "", name.lower())
        exact = [n for n in names if re.sub(r"[\W_]+", "", n.lower()) == key]
        if len(exact) == 1:
            return exact[0], name
        close = difflib.get_close_matches(name.lower(), names, n=2, cutoff=0.8)
        return (close[0] if len(close) == 1 else None), name

    def _parse_steps(self, llm_response: Any, *, cap: bool = True) -> list[AnalysisStep]:
        """
        Parse LLM plan steps, skipping malformed entries instead of crashing.
        Field names, tool names and the step container are matched tolerantly;
        exact duplicates are dropped and the plan is capped at
        _MAX_STEPS_PER_CYCLE (`cap=False` for a plan the profile built).

        Anti-hallucination guard: steps naming tools that do not exist in the
        ToolRegistry are rejected here (never executed), and a planner note is
        recorded so the next reasoning cycle sees the correction.
        """
        steps: list[AnalysisStep] = []
        rejected: list[str] = []
        notes: list[str] = []
        seen: set[str] = set()
        duplicates = 0
        for idx, s in enumerate(self._plan_items(llm_response), 1):
            if isinstance(s, str):
                s = {"tool_name": s}
            if not isinstance(s, dict):
                continue
            for key in ("function", "tool_call"):  # OpenAI style: {"function": {"name", "arguments"}}
                if isinstance(s.get(key), dict):
                    s = {**s, **s[key]}
            candidates = [v for k in self._TOOL_KEYS if isinstance(v := s.get(k), str) and v.strip()]
            if not candidates:
                continue
            resolved = [self._resolve_tool_name(c) for c in candidates]
            hit = next(((n, c) for n, c in resolved if n), None)
            if hit is None:
                rejected.append(resolved[0][1] or candidates[0])
                continue
            tool_name, cleaned = hit
            if (
                getattr(self.tool_registry.get(tool_name), "executes_code", False)
                and not code_execution_enabled()
            ):
                rejected.append(tool_name)
                continue
            if tool_name != cleaned:
                notes.append(f"corrected {cleaned!r} -> {tool_name!r}")
            parameters: Any = next((s[k] for k in self._PARAM_KEYS if s.get(k)), {})
            if isinstance(parameters, str):
                # ...and sometimes send the parameter object as a JSON string.
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError:
                    parameters = {}
            if not isinstance(parameters, dict):
                parameters = {}
            signature = f"{tool_name}|{json.dumps(parameters, sort_keys=True, default=str)}"
            if signature in seen:
                duplicates += 1
                continue
            seen.add(signature)
            number = next((s[k] for k in self._STEP_NO_KEYS if s.get(k) is not None), None)
            if isinstance(number, str) and number.strip().isdigit():
                number = int(number)
            steps.append(
                AnalysisStep(
                    step_number=number if isinstance(number, int) and not isinstance(number, bool) and number > 0 else idx,
                    tool_name=tool_name,
                    parameters=parameters,
                    rationale=str(s.get("rationale", "")),
                )
            )
        if duplicates:
            notes.append(f"dropped {duplicates} duplicate step(s)")
        if cap and len(steps) > _MAX_STEPS_PER_CYCLE:
            notes.append(
                f"only the first {_MAX_STEPS_PER_CYCLE} of {len(steps)} steps were run, the rest "
                "were dropped; plan them in a later cycle"
            )
            steps = steps[:_MAX_STEPS_PER_CYCLE]
        if rejected:
            console.print(
                f"  [yellow]⚠ Rejected {len(rejected)} hallucinated tool name(s): "
                f"{', '.join(rejected)}[/]"
            )
        if rejected or notes:
            summary = [
                *(
                    [
                        f"Rejected unknown tool name(s): {', '.join(sorted(set(rejected)))}. "
                        f"Only these tools exist: {', '.join(self.tool_registry.names())}."
                    ]
                    if rejected else []
                ),
                *(f"Plan note: {n}." for n in notes),
            ]
            self.memory.append_tool_result(
                ToolResult(
                    tool_name="planner",
                    status="skipped",
                    output={"summary": " ".join(summary)},
                    iteration=self.memory.iteration_count,
                )
            )
        return steps

    #: Nature-driven tools included in the fallback plan when applies_to()
    #: scores them at full confidence (1.0) — same tools, same gating logic
    #: the LLM planner sees, so there's one source of truth for "what suits
    #: this data" (controller._should_decompose folds in the same way).
    #: Tools the deterministic plan sequences explicitly (or never runs from
    #: the profile sweep): pipeline control, the supervised branch decided
    #: below on the target, and code execution, which needs an LLM to write
    #: the code and is meaningless without one.
    _FALLBACK_EXCLUDED_TOOLS = frozenset({
        "ingest_dataset", "clean_data", "generate_report",
        "train_model", "evaluate_model", "cluster_data",
        "execute_dynamic_code",
        # Now returns chart specs the dashboard renders, but everything it
        # draws unprompted (distributions, correlation heatmap, importances)
        # the deterministic dashboard already builds from the profile and
        # tool outputs — scheduling it here would only duplicate panels. The
        # LLM planner can still call it for a specific chart_type.
        "generate_visualizations",
    })

    #: applies_to score a tool must reach to earn a slot in the deterministic
    #: plan. Below 1.0 so a domain matched on partial evidence (0.65 for a
    #: ticker+price file with no OHLC) still contributes its analysis.
    _FALLBACK_MIN_SCORE = 0.6

    _ARM_NAME_RE = re.compile(r"arm|variant|treat|group|condition|cohort|bucket|version", re.IGNORECASE)

    @classmethod
    def _experiment_arm_column(cls, profile: DatasetProfile | None, target: str | None) -> str | None:
        """The arm column of an experiment-archetype profile, else None:
        a low-cardinality dimension named in the archetype evidence (whatever
        its shape), else one with an arm-like name, else the first one."""
        if profile is None or getattr(profile, "archetype", None) != "experiment":
            return None
        dims = [c.name for c in profile.dimensions() if 2 <= c.nunique <= 6 and c.name != target]
        raw_evidence = getattr(profile, "archetype_evidence", None)
        if isinstance(raw_evidence, dict):
            evidence = [f"{k} {v}" for k, v in raw_evidence.items()]
        elif isinstance(raw_evidence, (list, tuple)):
            evidence = [str(e) for e in raw_evidence]
        else:
            evidence = [str(raw_evidence or "")]
        # A name counts only as a whole word/quoted token of an evidence item.
        named = [d for d in dims if any(re.search(rf"(?<!\w){re.escape(d)}(?!\w)", e) for e in evidence)]
        for group in (
            named,
            [d for d in dims if cls._ARM_NAME_RE.search(d)],
            dims,
        ):
            if group:
                return group[0]
        return None

    def _build_fallback_plan(self) -> dict[str, Any]:
        """
        Deterministic analysis plan used when the LLM is unreachable on the
        first reasoning cycle. Always: clean → outliers → correlation, plus
        whichever nature-specific tools the profile-driven gating says
        apply at full confidence, plus (train + evaluate) when a target
        exists or (cluster + visualise) otherwise.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            raise RuntimeError("No dataset loaded — cannot build a fallback plan.")
        fp = meta.file_path
        profile = self.last_profile
        steps: list[dict[str, Any]] = [
            {
                "step_number": 1,
                "tool_name": "clean_data",
                "parameters": {
                    "file_path": fp,
                    "target_column": meta.target_column,
                },
                # No imputation: inferential tools use complete cases and
                # models impute inside cross-validation.
                "rationale": "Fallback plan: clean the dataset (types, duplicates) before analysis.",
            },
        ]

        # detect_outliers/correlation_analysis are near-universal but not
        # unconditional — e.g. correlation_analysis needs 2+ numeric columns
        # — so they go through the same applies_to gate as everything else
        # rather than being hardcoded past it (a single-numeric-column
        # dataset would otherwise error out here every time).
        for name, params, rationale in (
            ("detect_outliers", {"file_path": fp, "method": "iqr"}, "Fallback plan: flag anomalous rows."),
            ("correlation_analysis", {"file_path": fp, "target_column": meta.target_column}, "Fallback plan: quantify feature relationships."),
        ):
            if self.tool_registry.has(name) and self.tool_registry.get(name).applies_to(profile, meta) > 0.0:
                steps.append({
                    "step_number": len(steps) + 1,
                    "tool_name": name,
                    "parameters": params,
                    "rationale": rationale,
                })

        # Every registered tool the profile says fits, ranked by its own
        # applies_to score — not a hardcoded name list. This is what makes the
        # no-LLM path a real analyst rather than a stub: a tool registered
        # after this function was written (the domain tools, anything added
        # later) is planned automatically, and a dataset recognised as
        # transactional gets its cohort analysis without an LLM ever being
        # reachable. The previous hardcoded tuple silently excluded every
        # tool it predated.
        already = {s["tool_name"] for s in steps} | self._FALLBACK_EXCLUDED_TOOLS
        for tool in self.tool_registry.candidate_tools(
            profile, meta, use_ml=self.use_ml, use_llm=self.use_llm
        ):
            name = getattr(tool, "name", "")
            if name in already:
                continue
            score = tool.applies_to(profile, meta)
            if score < self._FALLBACK_MIN_SCORE:
                continue

            tool_params: dict[str, Any] = {"file_path": fp}
            try:
                tool_params.update(tool.default_params(profile, meta) or {})
            except Exception:
                logger.debug("default_params failed for a scheduled tool", exc_info=True)

            # Never schedule a step that cannot run. file_path and output_dir
            # are injected by BaseTool.prepare_params, and requires_context
            # entries are filled from memory, so only genuinely unfilled
            # required parameters disqualify a tool.
            try:
                schema = tool.get_schema()
            except Exception:
                schema = {}
            injected = {"file_path", "output_dir"} | set(
                getattr(tool, "requires_context", {}).values()
            )
            unfilled = [
                key
                for key, spec in schema.items()
                if spec.get("required")
                and key not in tool_params
                and key not in injected
            ]
            if unfilled:
                continue

            steps.append({
                "step_number": len(steps) + 1,
                "tool_name": name,
                "parameters": tool_params,
                "rationale": (
                    f"Fallback plan: profile-driven selection scored '{name}' "
                    f"at {score:.2f} for this dataset."
                ),
            })
            already.add(name)

        # An experiment's primary question is the arm comparison — make sure
        # the draft tests it rather than leaving the grouping to chance.
        arm = self._experiment_arm_column(profile, meta.target_column)
        if arm and self.tool_registry.has("select_statistical_test"):
            test_tool = self.tool_registry.get("select_statistical_test")
            if test_tool.applies_to(profile, meta) > 0.0:
                test_step = next((s for s in steps if s["tool_name"] == "select_statistical_test"), None)
                if test_step is None:
                    test_params: dict[str, Any] = {"file_path": fp}
                    try:
                        test_params.update(test_tool.default_params(profile, meta) or {})
                    except Exception:
                        logger.debug("default_params failed for a test tool", exc_info=True)
                    test_step = {
                        "step_number": len(steps) + 1,
                        "tool_name": "select_statistical_test",
                        "parameters": test_params,
                        "rationale": f"Fallback plan: experiment data — compare outcomes across arms of '{arm}'.",
                    }
                    steps.append(test_step)
                test_step["parameters"]["group_column"] = arm

        if not self.use_ml:
            # No model-fitting branch at all: no supervised training and no
            # clustering fallback. The profile-driven analyses above already
            # ran, so the plan is complete and genuinely ML-free.
            pass
        elif meta.target_column and meta.task_type in ("classification", "regression"):
            steps += [
                {
                    "step_number": len(steps) + 1,
                    "tool_name": "train_model",
                    "parameters": {
                        "file_path": fp,
                        "target_column": meta.target_column,
                        "task_type": meta.task_type,
                    },
                    "rationale": "Fallback plan: train baseline models with CV.",
                },
                {
                    "step_number": len(steps) + 2,
                    "tool_name": "evaluate_model",
                    "parameters": {
                        "file_path": fp,
                        "target_column": meta.target_column,
                        "task_type": meta.task_type,
                    },
                    "rationale": "Fallback plan: evaluate the best model on held-out data.",
                },
            ]
        else:
            steps += [
                {
                    "step_number": len(steps) + 1,
                    "tool_name": "cluster_data",
                    "parameters": {"file_path": fp},
                    "rationale": "Fallback plan: no target — discover natural segments.",
                },
            ]
        return {
            "status": "in_progress",
            "reasoning": "LLM unavailable — executing deterministic fallback plan.",
            "steps": steps,
            "deterministic": True,   # _parse_steps does not cap a plan the profile built
        }

    def _generate_dashboard(self) -> None:
        """
        Build the dynamic dashboard from the final state of the analysis and
        save it as JSON next to the reports. Non-fatal on any failure.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            return
        source = str(self.memory.get_context("cleaned_file_path") or meta.file_path)
        try:
            df = _read_dataframe(source)
            profile = profile_dataframe(df, target_column=meta.target_column)
            charts = build_dashboard(
                df,
                profile,
                target_column=meta.target_column,
                task_type=meta.task_type,
                tool_results=[r.to_dict() for r in self.memory.tool_results],
                findings=[f.to_dict() for f in self.memory.ranked_findings()],
            )
            charts = self._design_charts(df, profile, charts)
            out_path = Path(self._output_dir) / "reports" / "dashboard.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(dashboard_to_json(charts), encoding="utf-8")
            self._last_charts = [c.to_dict() for c in charts]
            self.memory.set_context("dashboard_path", str(out_path))
            console.print(
                f"  [green]📊 Dashboard generated: {len(charts)} chart(s) → {out_path}[/]"
            )
        except Exception as exc:
            console.print(f"  [yellow]⚠ Dashboard generation failed (non-fatal): {exc}[/]")

    def _design_charts(
        self, df: pd.DataFrame, profile: DatasetProfile, charts: list[ChartSpec]
    ) -> list[ChartSpec]:
        """One low-effort LLM call that designs a few finding-driven charts and
        drops generic ones they make redundant. Any failure keeps `charts`."""
        if (
            not self.use_llm
            or self._prompt_manager is None
            or os.getenv("CHART_DESIGN", "").strip().lower() == "false"
        ):
            return charts
        try:
            findings = [f.to_dict() for f in self.memory.ranked_findings()]
            if not findings or self._llm_budget_exhausted():
                return charts
            if self.llm_client.get_context_window() < _MIN_CHART_DESIGN_CONTEXT:
                console.print("  [dim]Chart design skipped: the model's context window is too small for it.[/]")
                return charts
            self.llm_client.stage = "chart_design"
            reply = self.llm_client.call(
                CHART_DESIGN_PROMPT,
                self._prompt_manager.get_chart_design_prompt(
                    findings, [c.to_dict() for c in charts], profile
                ),
                reasoning_effort="low",
                max_tokens=3000,  # room for the recipes of as many charts as it judges worthwhile
            )
            if self._rlm_engine is not None:
                self._rlm_engine.record_usage(reply.get("_rlm_usage"))
                self.memory.set_context("llm_usage", self._rlm_engine.usage_summary())
            design = design_from_reply(df, reply)
            designed = [
                c for i, raw in enumerate(design.charts)
                if (c := designed_chart(raw, i, findings)) is not None
            ]
            return merge_designed(charts, designed, set(design.drop_ids))
        except Exception as exc:
            if self._rlm_engine is not None:  # a failed call still spent tokens
                self._rlm_engine.record_usage(getattr(exc, "rlm_usage", None))
            console.print(f"  [dim]Chart design skipped (non-fatal): {exc}[/]")
            return charts

    def _write_run_summary(self) -> None:
        """reports/summary.json: a compact record of this run for run-to-run
        comparison. Non-fatal on any failure."""
        meta = self.memory.dataset_metadata
        if meta is None:
            return
        try:
            times: dict[str, float] = {}
            for r in self.memory.tool_results:
                times[r.tool_name] = times.get(r.tool_name, 0.0) + float(r.execution_time_ms or 0.0) / 1000.0
            summary = {
                "dataset": {"name": Path(meta.file_path).stem, "rows": meta.row_count, "cols": meta.column_count},
                "objective": self.objective,
                "llm": {"enabled": bool(self.use_llm), "model": self.llm_client.model if self.use_llm else None},
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "findings": [
                    {"finding_id": f.finding_id, "kind": f.kind, "layer": f.layer,
                     "importance": f.importance, "headline": f.headline}
                    for f in self.memory.ranked_findings()
                ],
                "charts": [{"chart_id": c.get("chart_id"), "title": c.get("title")} for c in self._last_charts],
                "tool_seconds": {k: round(v, 3) for k, v in times.items()},
            }
            out = Path(self._output_dir) / "reports" / "summary.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            console.print(f"  [yellow]Run summary not written (non-fatal): {exc}[/]")

    def _generate_html_report(self, llm_final: dict[str, Any]) -> None:
        """
        Write the self-contained HTML report (summary + interactive dashboard).
        Non-fatal on any failure.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            return
        try:
            from src.core.html_report import build_html_report

            html_doc = build_html_report(
                dataset_name=Path(meta.file_path).stem,
                llm_insights=llm_final,
                tool_results=[r.to_dict() for r in self.memory.tool_results],
                charts=self._last_charts,
                objective=self.objective,
                profile=self.memory.get_context("data_profile"),
                read_report=self.memory.get_context("read_report"),
                coercions=self.memory.get_context("coercions"),
                plan_rationales=self.memory.get_context("plan_rationales"),
                statistical_test_pvalues=self.memory.get_context("statistical_test_pvalues"),
                unverified_claims=self.memory.get_context("unverified_claims"),
                profile_status=self.memory.get_context("profile_status"),
                degradations=self.memory.get_context("degradations"),
                findings=[f.to_dict() for f in self.memory.ranked_findings()],
                analysis_decision=self.memory.get_context("analysis_decision"),
            )
            out_path = Path(self._output_dir) / "reports" / "report.html"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(html_doc, encoding="utf-8")
            self.memory.set_context("html_report_path", str(out_path))
            console.print(f"  [green]🌐 HTML report generated → {out_path}[/]")
        except Exception as exc:
            console.print(f"  [yellow]⚠ HTML report generation failed (non-fatal): {exc}[/]")

    def _deterministic_final(self) -> dict[str, Any]:
        """
        Synthesise a final report dict, projected from the finding bus (7.1)
        rather than re-deriving a narrative from raw tool output per tool.
        `best_model`/`key_metrics` stay derived directly from train_model's
        output since they're model-internal facts, not audience-facing
        findings; every analytical claim ("X differs by Y", "Z is trending")
        comes from `MemorySystem.ranked_findings()`, which is the same list
        every other surface (both reports, the dashboard) reads — a tool
        that gains a findings() implementation reaches all of them at once
        instead of needing a hardcoded case in each narrator.
        """
        recommendations: list[str] = []
        key_metrics: dict[str, Any] = {}
        best_model = self.memory.get_context("best_model_name")

        train = self.memory.get_last_result_for("train_model")
        if train and train.status == "success":
            models_trained = train.output.get("models_trained", {})
            best = train.output.get("best_model")
            if best and best in models_trained:
                best_model = best
                metrics = models_trained[best]
                key_metrics["cv_mean"] = metrics.get("cv_mean")
                key_metrics["cv_std"] = metrics.get("cv_std")
                key_metrics["train_test_gap"] = metrics.get("train_test_gap")
            warnings = train.output.get("overfit_warnings", [])
            if warnings:
                recommendations.append(
                    "Reduce model complexity (lower max_depth) or add regularisation "
                    "to close the train-test gap."
                )

        ranked = self.memory.ranked_findings()
        insights = [f.headline for f in ranked[:12]]
        if not insights:
            insights.append("Analysis produced no tool results to synthesise.")
        # Chosen deterministic mode and a mid-run LLM failure produce the same
        # findings but are not the same event, and saying so matters: telling
        # someone to "re-run with a reachable provider" when they deliberately
        # switched the LLM off reads as a malfunction rather than the mode
        # working as asked.
        if not self.use_llm:
            reasoning = (
                "Deterministic run: the LLM was switched off, so the plan came "
                "from the dataset profile and domain inference and these "
                "findings were compiled directly from tool output."
            )
            if not recommendations:
                recommendations.append(
                    "Turn the AI narrative on to get these same findings "
                    "interpreted and prioritised in plain language."
                )
        else:
            if not recommendations:
                recommendations.append(
                    "Re-run with a reachable LLM provider for narrative interpretation "
                    "of these deterministic findings."
                )
            reasoning = (
                "Deterministic synthesis: the LLM became unreachable mid-run, so "
                "findings were compiled directly from tool outputs."
            )
            llm_error = self.memory.get_context("llm_error")
            if llm_error:
                reasoning += f" (LLM error: {llm_error})"

        if not self.use_ml:
            recommendations.append(
                "Machine learning was switched off for this run — no model was "
                "fitted. Turn it on for predictive modelling and clustering."
            )
        if self.objective:
            reasoning = f"User objective: {self.objective}\n{reasoning}"

        return {
            "status": "complete",
            "llm_fallback": not self.use_llm or bool(self.memory.get_context("llm_error")),
            "deterministic_mode": not self.use_llm,
            "ml_enabled": self.use_ml,
            "reasoning": reasoning,
            "insights": insights,
            "recommendations": recommendations,
            "best_model": best_model,
            "key_metrics": key_metrics,
            "findings": [f.to_dict() for f in ranked],
        }

    # ------------------------------------------------------------------
    # Stage 3 execution helper
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Plan validation — deterministic, before a step spends a tool run
    # ------------------------------------------------------------------

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
            if key in AgentController._CACHE_EXCLUDED_PARAMS:
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

    def _verified_number_pool(self) -> set[str]:
        """Every numeric literal that actually appears in accumulated tool
        results, canonicalised for verbatim-citation checking."""
        pool, _per_tool = self._verified_number_pools()
        return pool

    def _verified_number_pools(self) -> tuple[set[str], dict[str, set[str]]]:
        """Global and per-tool verified-number pools — see
        `claim_verification.verified_number_pools`."""
        return verified_number_pools(self.memory)

    def _flag_unverified_claims(self, final_result: dict[str, Any]) -> list[str]:
        """Enforce the "cite only verbatim metrics" rule — see
        `claim_verification.flag_unverified_claims`."""
        return flag_unverified_claims(self.memory, final_result)

    def _audit_dependence_structure(self, df: pd.DataFrame | None) -> None:
        """Phase 4 (FutureScope §5.3) — dependence- and design-aware
        inference, wired as a post-hoc audit over already-collected findings
        and the profiled dataframe, the same pattern the Causal Claim Guard
        just below uses. Cost is ~0 when the triggering structure is absent:
        the clustering check only runs when `repeated_entity` finds a repeat-
        measurement column, and the Simpson's-paradox check only runs
        against `segment_lift` findings that already exist — a dataset with
        no lift claim to double-check pays nothing."""
        profile = self.last_profile
        if profile is None or df is None or df.empty:
            return

        def slug(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "x"

        # ---- Cluster/hierarchy check: rows nested in an entity (students in
        # schools, orders per customer) are not independent; naive p-values
        # overstate significance unless cluster-robust. ----
        entity_col = repeated_entity(profile, df)
        if entity_col and entity_col in df.columns:
            measure_col = next(
                (c.name for c in profile.measures() if c.name in df.columns and c.name != entity_col),
                None,
            )
            if measure_col:
                try:
                    icc_res = intraclass_correlation(df, entity_col, measure_col)
                except Exception:
                    icc_res = {}
                if icc_res.get("needs_cluster_robust"):
                    icc_finding = Finding(
                        finding_id=f"dependence_icc_{slug(entity_col)}_{slug(measure_col)}",
                        kind="method_fit",
                        headline=(
                            f"'{measure_col}' is clustered by '{entity_col}' "
                            f"(ICC={icc_res['icc']:.2f}, design effect={icc_res['deff']:.2f}) — "
                            f"treat this as {icc_res['n_eff']:.0f} effective observations, not {icc_res['n']}."
                        ),
                        detail=icc_res.get("recommendation", ""),
                        evidence=icc_res,
                        source_tool="dependence_audit",
                        measure=measure_col,
                        dimension=entity_col,
                        confidence=0.7,
                        layer="analyst",
                        caveats=["Rows are repeated measurements within an entity; standard-error estimates assuming independence are optimistic."],
                    )
                    self.memory.add_findings([icc_finding])
                    console.print(
                        f"[yellow]⚠ Clustering detected: '{measure_col}' rows are not independent "
                        f"within '{entity_col}' (ICC={icc_res['icc']:.2f}); significance tests overstate "
                        "confidence unless cluster-robust.[/]"
                    )

        # ---- Simpson's paradox: does a real segment_lift finding reverse
        # sign once stratified by another dimension? ----
        segment_findings = [f for f in self.memory.findings if f.kind == "segment_lift"]
        if not segment_findings:
            return
        candidate_dims = [
            c.name for c in profile.dimensions()
            if c.kind == "categorical" and 2 <= c.nunique <= 12 and c.name in df.columns
        ]
        if not candidate_dims:
            return

        checked_pairs: set[tuple[str, str, str]] = set()
        top_findings = sorted(segment_findings, key=lambda f: -abs(f.effect or 0.0))[:5]
        for sfinding in top_findings:
            if not sfinding.measure or not sfinding.dimension or sfinding.level is None:
                continue
            if sfinding.measure not in df.columns or sfinding.dimension not in df.columns:
                continue
            confounders = [c for c in candidate_dims if c != sfinding.dimension][:2]
            for confounder in confounders:
                key = (sfinding.measure, sfinding.dimension, confounder)
                if key in checked_pairs:
                    continue
                checked_pairs.add(key)
                level_str = str(sfinding.level)
                try:
                    sub = df[[sfinding.measure, sfinding.dimension, confounder]].copy()
                    dim_as_str = sub[sfinding.dimension].astype(str)
                    sub["_dep_group"] = dim_as_str.where(dim_as_str == level_str, other="rest")
                    res = check_simpsons_paradox(sub, sfinding.measure, "_dep_group", confounder)
                except Exception:
                    continue
                if not res.get("paradox_detected"):
                    continue
                n_level = int((df[sfinding.dimension].astype(str) == level_str).sum())
                opposing = res.get("opposing_strata_count", 0)
                valid_strata = res.get("valid_strata_count", 0)
                para_finding = Finding(
                    finding_id=f"dependence_simpsons_{slug(sfinding.measure)}_{slug(sfinding.dimension)}_{slug(confounder)}",
                    kind="method_fit",
                    headline=(
                        f"The '{sfinding.level}' vs rest difference in '{sfinding.measure}' reverses direction "
                        f"in {opposing} of {valid_strata} strata of '{confounder}' — possible confounding; "
                        "treat the unstratified effect with caution."
                    ),
                    detail=res.get("explanation", ""),
                    evidence={**res, "n_level": n_level},
                    source_tool="dependence_audit",
                    measure=sfinding.measure,
                    dimension=sfinding.dimension,
                    level=sfinding.level,
                    confidence=0.6,
                    layer="analyst",
                    caveats=["Simpson's paradox: the aggregate and stratified effects disagree in sign."],
                )
                sfinding.caveats.append(
                    f"Possible Simpson's paradox when stratified by '{confounder}' — see the dependence audit finding."
                )
                self.memory.add_findings([para_finding])
                console.print(
                    f"[yellow]⚠ Simpson's paradox: '{sfinding.level}' vs rest on '{sfinding.measure}' "
                    f"reverses when stratified by '{confounder}'.[/]"
                )
                break

    def _generate_final_report(self, llm_final: dict[str, Any]) -> None:
        """
        Stage 7: Invoke GenerateReportTool to produce the Markdown/JSON report.

        The report tool is called with the serialised tool results and the
        LLM's final interpretation so it can produce a complete document.
        """
        meta = self.memory.dataset_metadata
        dataset_name = Path(meta.file_path).stem if meta else "dataset"

        tool_results_json = json.dumps(
            [r.to_dict() for r in self.memory.tool_results], default=str
        )

        report_tool = self.tool_registry.get("generate_report")
        result = report_tool.run(
            dataset_name=dataset_name,
            tool_results_json=tool_results_json,
            llm_insights=llm_final,
            output_dir=str(Path(self._output_dir) / "reports"),
            data_profile=self.memory.get_context("data_profile"),
            read_report=self.memory.get_context("read_report"),
            coercions=self.memory.get_context("coercions"),
            plan_rationales=self.memory.get_context("plan_rationales"),
            statistical_test_pvalues=self.memory.get_context("statistical_test_pvalues"),
            unverified_claims=self.memory.get_context("unverified_claims"),
            profile_status=self.memory.get_context("profile_status"),
            degradations=self.memory.get_context("degradations"),
            findings=[f.to_dict() for f in self.memory.ranked_findings()],
            analysis_decision=self.memory.get_context("analysis_decision"),
        )

        self.memory.append_tool_result(result)

        if result.status == "success":
            console.print(
                Panel(
                    f"[bold green]Stage 7 — Report Generated[/]\n"
                    f"Markdown: {result.output.get('markdown_path')}\n"
                    f"JSON:     {result.output.get('json_path')}",
                    border_style="green",
                )
            )
        else:
            console.print(
                f"[yellow]⚠ Report generation failed: {result.error_message}[/]"
            )
