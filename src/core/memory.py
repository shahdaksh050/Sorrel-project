"""
Memory System for the Agentic Data Analysis System.

Maintains persistent state across all reasoning-execution cycles.
Implements the 'external environment' (REPL state) from the RLM paradigm:
  - Full dataset metadata lives here, NOT in the LLM context window.
  - Only compact, token-efficient summaries are passed to the LLM.
  - All tool results and analysis steps are accumulated and queryable.

Workflow position: used by every other layer — ingested first, read last.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.findings import Finding, rank_findings

console = Console()


# Target-detection keyword sets (used by detect_target_with_confidence)
_PRIMARY_TARGET_NAMES = frozenset({"target", "label", "class", "outcome", "y"})
_DOMAIN_TARGET_NAMES = frozenset({
    "churn", "survived", "fraud", "default", "purchased", "converted",
    "cancelled", "canceled", "clicked", "subscribed", "is_fraud",
    "is_churn", "is_default",
})
_NUMERIC_TARGET_NAMES = frozenset({
    "profit", "sales", "revenue", "price", "amount", "total", "cost",
    "score", "rating", "demand", "margin", "result", "status",
})
_PARTIAL_TARGET_HINTS = ("target", "label", "class", "outcome", "predict", "response")


def _shrink_to_fit(obj: Any, max_chars: int) -> Any:
    """Shrink `obj` so `json.dumps(obj)` fits within `max_chars`, by
    replacing oversized nested values with a short placeholder — never by
    slicing the serialised string. A blind character slice can cut JSON off
    mid-structure (invalid JSON, and possibly cut before the metric value
    that matters); this always returns something that serialises cleanly.

    Only touches dict values (and, recursively, list items) — scalars are
    returned untouched since there's no way to shrink a number or a short
    string without changing what it says.
    """
    if len(json.dumps(obj, default=str)) <= max_chars:
        return obj

    if isinstance(obj, dict):
        shrunk = dict(obj)
        # Largest serialised value first, so the biggest offender is
        # summarised before smaller — likely more important — fields
        # (e.g. `best_model`, `task_type`) are touched at all.
        by_size = sorted(
            shrunk.keys(),
            key=lambda k: len(json.dumps(shrunk[k], default=str)),
            reverse=True,
        )
        for key in by_size:
            if len(json.dumps(shrunk, default=str)) <= max_chars:
                break
            value = shrunk[key]
            if isinstance(value, (dict, list)) and value:
                size = len(json.dumps(value, default=str))
                count = len(value)
                kind = "entries" if isinstance(value, dict) else "items"
                shrunk[key] = f"<{count} {kind}, {size} chars — omitted for length>"
            elif isinstance(value, str) and len(value) > 80:
                shrunk[key] = value[:80] + "…"
        return shrunk

    if isinstance(obj, list):
        items = list(obj)
        while items and len(json.dumps(items, default=str)) > max_chars:
            items.pop()
        if len(items) < len(obj):
            return [*items, f"…and {len(obj) - len(items)} more"]
        return items

    return obj


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DatasetMetadata:
    """Structured metadata extracted from an uploaded dataset."""

    file_path: str
    row_count: int
    column_count: int
    columns: dict[str, str]           # {col_name: dtype_string}
    missing_values: dict[str, int]    # {col_name: n_missing}
    numerical_cols: list[str]
    categorical_cols: list[str]
    target_column: str | None = None
    task_type: str | None = None      # "classification" | "regression" | "clustering"
    summary_stats: dict[str, Any] = field(default_factory=dict)
    class_balance: dict[str, int] = field(default_factory=dict)  # for classification targets
    high_cardinality_cols: list[str] = field(default_factory=list)
    column_nunique: dict[str, int] = field(default_factory=dict)   # {col: nunique count}

    # ------------------------------------------------------------------
    def detect_target_with_confidence(self) -> tuple[str | None, float]:
        """
        Heuristic target detection returning (column_name, confidence).

        Scoring pipeline:
          1. Naming convention match — exact keywords, domain names, partial hints
          2. Statistical adjustments — cardinality, dtype, high-cardinality flag
          3. Positional fallback — last column when no keyword match found

        Returns:
            (column, confidence) where confidence ∈ [0.0, 1.0].
            (None, 0.0) when the dataset has no columns.
        """
        col_names = list(self.columns.keys())
        if not col_names:
            return None, 0.0

        # A datetime/date-typed column is never a supervised-learning target
        # — there is no regressor/classifier task for "predict the
        # timestamp". Without this exclusion, a date column with no other
        # target-shaped column in the dataset falls through to the
        # positional fallback below (or, worse, a coincidental name match
        # like "target_date") and gets auto-selected to train on, which is
        # never the right call regardless of confidence.
        non_datetime_cols = [
            c for c in col_names if "datetime" not in self.columns.get(c, "").lower()
        ]
        if not non_datetime_cols:
            return None, 0.0

        lower_to_orig = {c.lower(): c for c in non_datetime_cols}
        candidates: list[tuple[str, float]] = []

        for lower, original in lower_to_orig.items():
            base = 0.0
            if lower in _PRIMARY_TARGET_NAMES:
                base = 0.95
            elif lower in _DOMAIN_TARGET_NAMES:
                base = 0.90
            elif lower in _NUMERIC_TARGET_NAMES:
                base = 0.85
            elif any(h in lower for h in _PARTIAL_TARGET_HINTS):
                base = 0.75

            if base == 0.0:
                continue

            # Statistical adjustments
            nunique = self.column_nunique.get(original, -1)
            if nunique == 2:                          # binary → strong target signal
                base = min(1.0, base + 0.05)
            elif nunique > 100:                       # very high cardinality → likely ID
                base = max(0.0, base - 0.15)
            if original in self.high_cardinality_cols:  # >50 unique categoricals
                base = max(0.0, base - 0.20)
            if original in self.categorical_cols:       # categorical dtype → slight bonus
                base = min(1.0, base + 0.03)

            candidates.append((original, round(base, 4)))

        if not candidates:
            # Positional fallback — last column is the common ML convention,
            # but with no name hint it is weak evidence. Score non-binary
            # guesses below the autonomy threshold so the pipeline prefers
            # EDA + clustering over modelling an arbitrary column.
            last_col = non_datetime_cols[-1]
            nunique = self.column_nunique.get(last_col, -1)
            if nunique == 2:
                pos_score = 0.65   # binary last column is the usual label shape
            elif nunique > 100:
                pos_score = 0.25
            else:
                pos_score = 0.35
            if last_col in self.high_cardinality_cols:
                pos_score = max(0.20, pos_score - 0.20)
            candidates.append((last_col, round(pos_score, 4)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    def auto_detect_target(self) -> str | None:
        """Compatibility shim — delegates to detect_target_with_confidence()."""
        col, _ = self.detect_target_with_confidence()
        return col

    def infer_task_type(self) -> str | None:
        """
        Infer the ML task type from the target column.

        Rules
        -----
        - No target column → clustering (unsupervised fallback)
        - Boolean / low-cardinality integer (≤20 unique) → classification
        - High-cardinality numeric → regression
        - Object/string targets → classification
        """
        if not self.target_column:
            return "clustering"
        dtype = self.columns.get(self.target_column, "")
        nunique = self.column_nunique.get(self.target_column, -1)
        if "bool" in dtype:
            return "classification"
        if "int" in dtype:
            return "classification" if nunique == -1 or nunique <= 20 else "regression"
        if "float" in dtype:
            return "regression"
        return "classification"  # default for object/string targets

    def to_prompt_string(self, compact: bool = False) -> str:
        """
        Render a compact, LLM-friendly metadata summary (≈200 tokens).

        `compact` drops the column listings and stats — used when the data
        profile (which describes every column in more detail) is also in the
        prompt, so a small model's context isn't spent on the same columns
        twice.

        All dataset-derived strings (column names, class labels) pass through
        sanitize_for_prompt so a hostile dataset cannot inject instructions
        into the planner.
        """
        from src.core.security import sanitize_for_prompt as _sp

        missing_total = sum(self.missing_values.values())
        target_note = (
            _sp(self.target_column)
            if self.target_column
            else "None (perform full EDA — describe distributions, correlations, outliers, and key patterns)"
        )
        num_cols = ", ".join(_sp(c) for c in self.numerical_cols[:12])
        cat_cols = ", ".join(_sp(c) for c in self.categorical_cols[:8])
        lines = [
            f"Dataset: {_sp(Path(self.file_path).name)}",
            f"Shape: {self.row_count:,} rows × {self.column_count} columns",
        ]
        if not compact:
            lines += [
                f"Numerical ({len(self.numerical_cols)}): {num_cols}",
                f"Categorical ({len(self.categorical_cols)}): {cat_cols}",
            ]
        lines += [
            f"Missing values: {missing_total:,} total cells across "
            f"{sum(1 for v in self.missing_values.values() if v > 0)} columns",
            f"Target: {target_note}",
            f"Task type: {self.task_type or 'Not identified'}",
        ]
        if self.class_balance:
            bal = ", ".join(f"{_sp(k)}={v}" for k, v in list(self.class_balance.items())[:6])
            lines.append(f"Class balance: {bal}")
        if compact:
            return "\n".join(lines)
        if self.high_cardinality_cols:
            hc = ", ".join(_sp(c) for c in self.high_cardinality_cols)
            lines.append(f"High-cardinality columns: {hc}")
        if self.summary_stats:
            stats_snippets = []
            for col, stats in list(self.summary_stats.items())[:6]:
                if isinstance(stats, dict):
                    st_str = ", ".join(f"{k}={v}" for k, v in list(stats.items())[:4])
                    stats_snippets.append(f"{_sp(col)} ({st_str})")
            if stats_snippets:
                lines.append(f"Key stats: {'; '.join(stats_snippets)}")
        return "\n".join(lines)


@dataclass
class ToolResult:
    """The structured output of a single tool execution."""

    tool_name: str
    status: str          # "success" | "error" | "skipped"
    output: dict[str, Any]
    error_message: str | None = None
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    iteration: int = 0   # P1.6 — which reasoning cycle produced this result

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "output": self.output,
            "error": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class AnalysisStep:
    """A single step in the LLM-generated analysis plan."""

    step_number: int
    tool_name: str
    parameters: dict[str, Any]
    rationale: str
    result: ToolResult | None = None
    retry_count: int = 0

    def is_complete(self) -> bool:
        return self.result is not None and self.result.status == "success"

    def is_failed(self) -> bool:
        return self.result is not None and self.result.status == "error"


# ---------------------------------------------------------------------------
# Memory System
# ---------------------------------------------------------------------------

class MemorySystem:
    """
    Central state store — the RLM 'external environment'.

    Design principles:
      1. Context offloading: full data stays here; LLM gets summaries only.
      2. Persistent REPL state: survives across all reasoning iterations.
      3. Generic context store: supports arbitrary RLM sub-call variables.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self.dataset_metadata: DatasetMetadata | None = None
        self.analysis_plan: list[AnalysisStep] = []
        self.tool_results: list[ToolResult] = []
        self.iteration_count: int = 0
        self.session_id: str = str(int(time.time()))
        self.persist_path: str | None = persist_path
        self._ctx: dict[str, Any] = {}     # generic RLM sub-call context store
        self.findings: list[Finding] = []  # the finding bus (7.1) — one list, every surface projects it
        #: Round 8 — specs (as dicts, see src.core.tool_factory.GeneratedToolSpec)
        #: for tools the LLM has defined during this run via define_analysis_tool.
        #: Populated by the controller (never by a tool's own execute()) once a
        #: proposed tool passes validation — see AGENTS.md's layer rule and
        #: IMPROVEMENTS.md Round 8 decision 3. Read back each iteration so the
        #: prompt can tell the LLM which tools it created earlier in this run.
        self.generated_tools: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Stage 1 — Dataset metadata
    # ------------------------------------------------------------------

    def store_dataset_metadata(self, metadata: DatasetMetadata) -> None:
        """Store extracted dataset metadata as the foundational context."""
        self.dataset_metadata = metadata
        console.print(
            Panel(
                f"[bold green]✓ Dataset metadata stored[/]\n{metadata.to_prompt_string()}",
                title="[bold cyan]Memory System",
                border_style="cyan",
            )
        )

    def get_metadata_prompt(self, compact: bool = False) -> str:
        """Return compact metadata string for LLM context injection."""
        if not self.dataset_metadata:
            raise ValueError("No dataset metadata. Call store_dataset_metadata() first.")
        return self.dataset_metadata.to_prompt_string(compact=compact)

    # ------------------------------------------------------------------
    # Stage 2 — Analysis plan (LLM-generated)
    # ------------------------------------------------------------------

    def store_analysis_plan(self, steps: list[AnalysisStep]) -> None:
        self.analysis_plan = steps
        console.print(f"[cyan]📋 Plan stored: {len(steps)} step(s)[/]")

    def get_pending_steps(self) -> list[AnalysisStep]:
        return [s for s in self.analysis_plan if not s.is_complete()]

    def get_failed_steps(self) -> list[AnalysisStep]:
        return [s for s in self.analysis_plan if s.is_failed()]

    def mark_step_complete(self, step_number: int, result: ToolResult) -> None:
        for step in self.analysis_plan:
            if step.step_number == step_number:
                step.result = result
                break

    def increment_retry(self, step_number: int) -> None:
        for step in self.analysis_plan:
            if step.step_number == step_number:
                step.retry_count += 1
                break

    # ------------------------------------------------------------------
    # Stage 3 — Tool results (execution layer outputs)
    # ------------------------------------------------------------------

    def append_tool_result(self, result: ToolResult) -> None:
        self.tool_results.append(result)
        color = {"success": "green", "skipped": "yellow"}.get(result.status, "red")
        icon = {"success": "✓", "skipped": "⏭"}.get(result.status, "✗")
        console.print(
            f"  [{color}]{icon} {result.tool_name}[/] → {result.status} "
            f"[dim]({result.execution_time_ms:.0f} ms)[/]"
        )

    def get_results_summary(self, max_chars_per_result: int = 1200) -> str:
        """
        Compact summary of all tool outputs for LLM context injection.

        Only non-raw-data fields are included to keep tokens low. Oversized
        nested fields are summarised field-by-field (see `_shrink_to_fit`),
        never blindly sliced — a mid-structure character cut can produce
        broken JSON and can cut off before a model's actual metric values,
        which directly undermines the system prompt's hard rule to "cite
        ONLY metric values that appear verbatim in the results"
        (IMPROVEMENTS.md #3).
        """
        if not self.tool_results:
            return "No tool results yet."
        lines: list[str] = []
        for r in self.tool_results:
            if r.status == "success":
                slim = {k: v for k, v in r.output.items() if k not in {"raw_data", "dataframe"}}
                shrunk = _shrink_to_fit(slim, max_chars_per_result)
                serialised = json.dumps(shrunk, default=str)
                lines.append(f"[{r.tool_name}] SUCCESS → {serialised}")
            elif r.status == "skipped":
                lines.append(f"[{r.tool_name}] SKIPPED → {r.output.get('summary', '')}")
            else:
                lines.append(f"[{r.tool_name}] ERROR → {r.error_message}")
        return "\n".join(lines)

    def get_results_summary_digest(
        self, current_iteration: int, max_chars_per_result: int = 1200
    ) -> str:
        """
        P1.6 — the iteration prompt re-sent every accumulated result in full
        on every cycle (up to 48 entries by iteration 8), which grows
        linearly and re-bills tokens for results the LLM already reacted to.
        This gives the current iteration's results in full (the LLM's job
        this cycle is to react to what just happened) and a digest for
        everything earlier.

        Round 8 hardening — the digest used to be a bare "tool -> status"
        line with no numbers at all, so once a result aged out of "current"
        the LLM could never again cite anything from it verbatim (it only
        ever saw the metrics for a SINGLE iteration's worth of results at a
        time). Earlier successes now carry a small shrunk payload — key
        numbers survive, at the cost of some tokens — while errors/skips
        stay a one-liner since there's nothing numeric worth preserving.
        """
        if not self.tool_results:
            return "No tool results yet."
        current = [r for r in self.tool_results if r.iteration == current_iteration]
        earlier = [r for r in self.tool_results if r.iteration != current_iteration]

        lines: list[str] = []
        if earlier:
            lines.append("Earlier iterations (digest — key numbers only, already reacted to):")
            for r in earlier:
                if r.status == "success":
                    slim = {k: v for k, v in r.output.items() if k not in {"raw_data", "dataframe"}}
                    shrunk = _shrink_to_fit(slim, 250)
                    serialised = json.dumps(shrunk, default=str)
                    lines.append(f"  [iter {r.iteration}] {r.tool_name} → success → {serialised}")
                elif r.status == "skipped":
                    lines.append(f"  [iter {r.iteration}] {r.tool_name} → skipped")
                else:
                    lines.append(f"  [iter {r.iteration}] {r.tool_name} → error")
        if current:
            lines.append("This iteration's results (in full):")
            for r in current:
                if r.status == "success":
                    slim = {k: v for k, v in r.output.items() if k not in {"raw_data", "dataframe"}}
                    shrunk = _shrink_to_fit(slim, max_chars_per_result)
                    lines.append(f"  [{r.tool_name}] SUCCESS → {json.dumps(shrunk, default=str)}")
                elif r.status == "skipped":
                    lines.append(f"  [{r.tool_name}] SKIPPED → {r.output.get('summary', '')}")
                else:
                    lines.append(f"  [{r.tool_name}] ERROR → {r.error_message}")
        return "\n".join(lines)

    def get_last_result_for(self, tool_name: str) -> ToolResult | None:
        """Return the most recent result for a given tool name."""
        for r in reversed(self.tool_results):
            if r.tool_name == tool_name:
                return r
        return None

    # ------------------------------------------------------------------
    # Finding bus (7.1) — every projection (deterministic synthesis, the
    # Markdown/HTML reports, the dashboard panel selector) reads
    # ranked_findings() instead of re-deriving its own narrative from raw
    # tool output, so a tool that gains a findings() implementation reaches
    # every surface at once.
    # ------------------------------------------------------------------

    def add_findings(self, findings: list[Finding]) -> None:
        self.findings.extend(findings)

    def ranked_findings(self, suppress_trivial: bool = True) -> list[Finding]:
        return rank_findings(self.findings, suppress_trivial=suppress_trivial)

    def findings_by_kind(self, *kinds: str) -> list[Finding]:
        return [f for f in self.findings if f.kind in kinds]

    # ------------------------------------------------------------------
    # Generated tools (Round 8) — tools the LLM defined during this run via
    # define_analysis_tool, registered by the controller after validation.
    # ------------------------------------------------------------------

    def add_generated_tool(self, spec: dict[str, Any]) -> None:
        """Record a newly-registered (or modified) generated tool spec.

        A modification (same `name` as an existing entry) replaces that
        entry rather than appending a duplicate, since only one live version
        of a generated tool exists in the registry at a time — the spec's
        own `history` list (see tool_factory.GeneratedToolSpec) is where
        prior code versions are kept for audit.
        """
        name = spec.get("name")
        self.generated_tools = [g for g in self.generated_tools if g.get("name") != name]
        self.generated_tools.append(spec)

    def list_generated_tools(self) -> list[dict[str, Any]]:
        return list(self.generated_tools)

    # ------------------------------------------------------------------
    # Generic context store — RLM sub-call environments (Stage 6)
    # ------------------------------------------------------------------

    def set_context(self, key: str, value: Any) -> None:
        self._ctx[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self._ctx.get(key, default)

    def list_context_keys(self) -> list[str]:
        return list(self._ctx.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        if not self.persist_path:
            return
        state = {
            "session_id": self.session_id,
            "iteration_count": self.iteration_count,
            "dataset_metadata": self.dataset_metadata.__dict__ if self.dataset_metadata else None,
            "tool_results": [r.to_dict() for r in self.tool_results],
            "findings": [f.to_dict() for f in self.findings],
            "generated_tools": self.generated_tools,
            "additional_context": {k: v for k, v in self._ctx.items() if isinstance(v, (str, int, float, bool, list, dict))},
        }
        Path(self.persist_path).write_text(json.dumps(state, indent=2, default=str))

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_status(self) -> None:
        table = Table(title="Memory State", show_header=True, header_style="bold magenta")
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        table.add_row("Session ID", self.session_id)
        table.add_row("Iterations", str(self.iteration_count))
        table.add_row("Dataset loaded", "✓" if self.dataset_metadata else "✗")
        table.add_row("Plan steps", str(len(self.analysis_plan)))
        table.add_row("Tool results", str(len(self.tool_results)))
        table.add_row("Pending steps", str(len(self.get_pending_steps())))
        table.add_row("Failed steps", str(len(self.get_failed_steps())))
        table.add_row("Context keys", str(len(self._ctx)))
        console.print(table)
