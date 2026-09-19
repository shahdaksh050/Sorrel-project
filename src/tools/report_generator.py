"""
Report Generator Tool — Execution Layer.

Stage 7: Report Generation.

Compiles all tool results, LLM insights, and metric summaries into:
  - A structured Markdown report  (output/reports/analysis_report.md)
  - A raw JSON dump               (output/reports/final_report.json)

The Markdown report is human-readable and suitable for conversion to PDF
via pandoc or any Markdown renderer.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.html_report import data_understanding_rows, governance_rows
from src.core.multiple_testing import DEFAULT_ALPHA as _BH_ALPHA
from src.core.multiple_testing import apply_benjamini_hochberg as _apply_benjamini_hochberg
from src.core.plain_language import describe_uncertainty, plainify
from src.tools.base import BaseTool

if TYPE_CHECKING:
    from src.core.memory import MemorySystem


# ---------------------------------------------------------------------------
# 3a — executive summary: prefer the LLM's own `insights` over its raw
# `reasoning` scratch text, which has leaked internal plumbing ("completed
# successfully across iter 1", "RLM Sub-Analysis Findings", "Form 2",
# literal tool names) straight into shipped reports. Mirrored in
# src.core.html_report so both reports agree on what the reader sees.
# ---------------------------------------------------------------------------

#: Word-bounded patterns (case-insensitive) that mark a sentence as internal
#: plumbing rather than analyst-facing prose — see prompt_manager.py's
#: "Form 1"/"Form 2"/iteration-numbered system prompt, which the model
#: occasionally echoes back into its own reasoning/insights text.
_JARGON_RE = re.compile(
    r"\biter(ation)? \d|\brlm\b|\bsub-analys|\bform [12]\b|\btool results\b", re.IGNORECASE
)


def _strip_jargon_sentences(text: str) -> str:
    """Drop any sentence containing internal-plumbing jargon. Splits on
    sentence-ending punctuation — approximate, but the text here is prose
    the LLM itself wrote, not something requiring a real parser."""
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences if not _JARGON_RE.search(s)]
    return " ".join(kept).strip()


def _build_executive_summary(llm_insights: dict[str, Any]) -> str:
    """Build the executive summary from `reasoning` (with internal-plumbing jargon
    stripped). If `reasoning` is empty or only contained jargon, fall back to
    joining the top `insights`."""
    clean_reasoning = _strip_jargon_sentences(str(llm_insights.get("reasoning", "") or ""))
    if clean_reasoning:
        return plainify(clean_reasoning)
    insights = llm_insights.get("insights") or []
    if insights:
        text = " ".join(str(i).strip().rstrip(".") + "." for i in insights[:4] if str(i).strip())
        return plainify(_strip_jargon_sentences(text))
    return ""


def _model_was_trained(llm_insights: dict[str, Any], tool_results: list[dict[str, Any]]) -> bool:
    """True only when a model was actually fitted (item 3b) — `best_model`
    is the surest signal (TrainModelTool only ever runs for classification/
    regression, so its presence already implies that task_type), backed up
    by a successful `train_model` tool result in case `best_model` wasn't
    threaded through `llm_insights`. Distinguishes a real "Model
    Performance" section from a describe-only (EDA) run whose key_metrics
    happens to be populated with something else (e.g. a gini_coefficient
    from concentration_analysis)."""
    if llm_insights.get("best_model"):
        return True
    return any(
        r.get("tool_name") == "train_model"
        and r.get("status") == "success"
        and isinstance(r.get("output"), dict)
        and r["output"].get("best_model")
        for r in tool_results
    )


def _format_data_overview(
    data_profile: dict[str, Any] | None,
    read_report: dict[str, Any] | None,
    coercions: list[dict[str, Any]] | None,
) -> list[str]:
    """Shape, quality, and — critically — what was *detected or assumed* at
    read time (item 2) and *repaired* before profiling (item 3), so the
    report explains what it's actually looking at, not just what it found."""
    if not (data_profile or read_report or coercions):
        return []
    lines = ["## Data Overview", ""]
    if data_profile:
        kind_counts: dict[str, int] = {}
        for col in data_profile.get("columns", []):
            kind = col.get("kind", "?")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        kinds_str = ", ".join(f"{v} {k}" for k, v in sorted(kind_counts.items()))
        lines.append(
            f"- **Shape**: {data_profile.get('row_count', '—')} rows × "
            f"{data_profile.get('column_count', '—')} columns"
        )
        lines.append(f"- **Column kinds**: {kinds_str or '—'}")
        lines.append(f"- **Quality score**: {data_profile.get('quality_score', '—')}/100")
        lines.append(f"- **Duplicate rows**: {data_profile.get('duplicate_rows', '—')}")
        if not data_profile.get("is_sufficient", True):
            lines.append(
                f"- **⚠ Data sufficiency**: {data_profile.get('sufficiency_reason') or 'Insufficient data.'}"
            )
        lines.append("")
        # What the data was recognised *as* — this is what selected the
        # domain-specific analyses, so the reader can see (and challenge)
        # the classification rather than wondering why an RFM table appeared.
        for match in data_profile.get("domains") or []:
            roles = ", ".join(
                f"`{col}` as {role}" for role, col in sorted((match.get("roles") or {}).items())
            )
            lines.append(
                f"**Recognised as {match.get('domain')} data** "
                f"(confidence {float(match.get('confidence', 0)):.2f}). "
                f"Columns read as: {roles or '—'}."
            )
            for item in match.get("evidence") or []:
                lines.append(f"  - {item}")
            lines.append("")
    if read_report:
        bits = [
            f"format `{read_report.get('format')}`",
            f"encoding `{read_report.get('encoding')}`"
            + ("" if read_report.get("encoding_confident", True) else " (guessed)"),
        ]
        if read_report.get("delimiter"):
            bits.append(
                f"delimiter `{read_report.get('delimiter')!r}`"
                + ("" if read_report.get("delimiter_sniffed") else " (assumed from extension)")
            )
        lines.append(f"**Detected at read time**: {', '.join(bits)}.")
        for note in read_report.get("notes") or []:
            lines.append(f"  - ⚠ {note}")
        lines.append("")
    if coercions:
        lines.append(f"**Repaired before analysis** ({len(coercions)} column(s)):")
        lines.append("")
        lines.append("| Column | Rule | Converted | Failed |")
        lines.append("|--------|------|-----------|--------|")
        for c in coercions:
            lines.append(f"| {c.get('column')} | {c.get('rule')} | {c.get('n_converted')} | {c.get('n_failed')} |")
        lines.append("")
    return lines


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _format_data_understanding(
    llm_insights: dict[str, Any], data_profile: dict[str, Any] | None,
) -> list[str]:
    """How the planner read the data on its first pass (subject, domain,
    table archetype, key measures/dimensions, caveats) — rows shared with
    src.core.html_report."""
    rows, caveats = data_understanding_rows(llm_insights, data_profile)
    if not (rows or caveats):
        return []
    lines = ["## How the Agent Read the Data", ""]
    for label, value in rows:
        lines.append(f"- **{label}**: {value}")
    if caveats:
        if rows:
            lines.append("")
        lines.append("What it flagged to watch out for:")
        lines.extend(f"- {c}" for c in caveats)
    lines.append("")
    return lines


def _format_governance(llm_insights: dict[str, Any]) -> list[str]:
    """Code-execution and LLM-usage accounting for the run — rows shared
    with src.core.html_report."""
    rows = governance_rows(llm_insights)
    if not rows:
        return []
    lines = ["## Governance", "", "| Item | Value |", "|------|-------|"]
    lines.extend(f"| {_md_cell(label)} | {_md_cell(value)} |" for label, value in rows)
    lines.append("")
    return lines


def _format_methodology(
    plan_rationales: list[dict[str, Any]] | None,
    analysis_decision: dict[str, Any] | None = None,
) -> list[str]:
    """The planner is required to justify every step (prompt_manager.py's
    SYSTEM_PROMPT_CORE), but that rationale used to be truncated to 60 chars
    in a console panel and then discarded — this is the narrative the brief
    asks for, generated all along and just never surfaced.

    `analysis_decision` (7.9) adds the T2 "why we did or didn't model X"
    transparency when the controller recorded one — additive, so a caller
    without one gets exactly the table this always produced."""
    lines: list[str] = []
    if plan_rationales:
        lines += [
            "## Methodology",
            "",
            "Why each analysis was chosen, in the planner's own words:",
            "",
            "| Step | Tool | Rationale |",
            "|------|------|-----------|",
        ]
        for r in plan_rationales:
            rationale = str(r.get("rationale", "")).replace("|", "\\|")
            lines.append(f"| {r.get('step_number', '—')} | {r.get('tool_name', '—')} | {rationale} |")
        lines.append("")
    if analysis_decision:
        mode = analysis_decision.get("mode")
        decision_rationale = analysis_decision.get("rationale")
        rejected = analysis_decision.get("alternatives_rejected") or []
        if mode or decision_rationale:
            if not lines:
                lines += ["## Methodology", ""]
            lines.append(f"**Approach taken**: {mode or '—'}.")
            if decision_rationale:
                lines.append("")
                lines.append(str(decision_rationale))
            if rejected:
                lines.append("")
                lines.append("Alternatives considered and not taken:")
                for r in rejected:
                    lines.append(f"- {r}")
            lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 7.9 — the layered report's finding-driven sections, mirroring
# src.core.html_report's selectors so the Markdown and HTML reports agree on
# what "headline" / "evidence" / "caveat" mean. `findings` is the shared bus
# (src.core.findings): the caller (AgentController) already ranks it by
# importance descending, but every selector here re-sorts defensively rather
# than trusting call-site order.
# ---------------------------------------------------------------------------

_CAVEAT_FINDING_KINDS = ("method_fit", "coverage_gap")
_MAX_HEADLINE_FINDINGS = 5
_MAX_EVIDENCE_FINDINGS = 15


def _headline_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [f for f in findings if f.get("kind") not in _CAVEAT_FINDING_KINDS]
    preferred = [f for f in candidates if f.get("layer") in ("exec", "analyst")]
    pool = preferred if preferred else candidates
    pool = list(pool)  # keep rank_findings() order (source-diversity pass), no re-sort
    return pool[:_MAX_HEADLINE_FINDINGS]


def _evidence_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool = [
        f for f in findings
        if f.get("layer") == "analyst" and f.get("kind") not in _CAVEAT_FINDING_KINDS
    ]
    pool = list(pool)  # keep rank_findings() order (source-diversity pass), no re-sort
    return pool[:_MAX_EVIDENCE_FINDINGS]


def _caveat_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in findings if f.get("kind") in _CAVEAT_FINDING_KINDS]


def _format_top_findings(findings: list[dict[str, Any]] | None) -> list[str]:
    """The headline layer: the top 3-5 findings' own words, in plain
    language. Purely additive — an old caller passing no findings simply
    gets no section here, and every other section renders as before."""
    top = _headline_findings(findings or [])
    if not top:
        return []
    lines = ["## Top Findings", ""]
    for i, f in enumerate(top, 1):
        lines.append(f"{i}. {plainify(str(f.get('headline', '')))}")
    lines.append("")
    return lines


def _format_evidence(findings: list[dict[str, Any]] | None) -> list[str]:
    """The analyst layer: every finding's detail/evidence dict, effect sizes
    and p-values — the traceable numbers behind the headlines above, listed
    exactly as computed under a plain-language sentence. This is
    exactly where a tool's own `findings()` (e.g. cohort_analysis's revenue-
    concentration or RFM-lift findings) now reaches the report regardless of
    whether it also has a bespoke subsection in `_format_additional_analyses`."""
    evidence = _evidence_findings(findings or [])
    if not evidence:
        return []
    lines = ["## Evidence", ""]
    for f in evidence:
        lines.append(f"### {plainify(str(f.get('headline', '')))}")
        lines.append("")
        detail = f.get("detail")
        if detail:
            lines.append(plainify(str(detail)))
            lines.append("")
        confidence = describe_uncertainty(f)
        if confidence:
            lines.append(confidence)
            lines.append("")
        bits: list[str] = []
        effect = f.get("effect")
        if effect is not None:
            kind = f.get("effect_kind") or ""
            bits.append(f"effect {round(float(effect), 4)}" + (f" ({kind})" if kind else ""))
        p_value = f.get("p_value")
        if p_value is not None:
            bits.append(f"p={round(float(p_value), 4)}")
        p_adj = f.get("p_adjusted")
        if p_adj is not None:
            bits.append(f"p(adj)={round(float(p_adj), 4)}")
        raw_evidence = f.get("evidence") or {}
        if isinstance(raw_evidence, dict):
            for k, v in raw_evidence.items():
                if isinstance(v, (int, float, str)) and not isinstance(v, bool):
                    bits.append(f"{k}={v}")
        if bits:
            lines.append("- " + " · ".join(str(b) for b in bits))
        source = f.get("source_tool")
        if source:
            lines.append(f"- source: `{source}`")
        lines.append("")
    return lines


#: 3d — the controller now feeds the BH correction every family-mode test
#: *and* every segment_comparison test in one run, which can run into the
#: dozens. A table that long stops being readable, so only the most
#: decision-relevant rows are shown: significant results first, then by
#: ascending BH-adjusted p, capped at this many rows with a "N more not
#: shown" trailer. Mirrored in src.core.html_report.
_MAX_BH_ROWS = 15


def _sort_bh_tests(bh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        bh,
        key=lambda t: (not t.get("significant_after_correction"), t.get("p_adjusted") or 1.0),
    )


def _format_limitations(
    data_profile: dict[str, Any] | None,
    statistical_test_pvalues: list[dict[str, Any]] | None,
    unverified_claims: list[str] | None,
    profile_status: str | None,
    degradations: list[str] | None = None,
    findings: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Everything a careful reader needs to know before trusting a number in
    this report: every recorded fallback/repair (item 10's degradations
    log, when the caller has it — the controller always does), the
    multiple-comparisons correction (item 4), anything the verbatim-metric
    guard (P0.7) couldn't verify, and (7.9) every `method_fit`/`coverage_gap`
    finding — caveats, not headline insights, per
    src.core.findings.is_trivial's own triviality rules."""
    if degradations is None:
        # Direct callers that don't accumulate a degradations log (tests,
        # scripts) still get the same information, derived on the spot.
        from src.core.degradations import collect_degradations

        degradations = collect_degradations(None, None, data_profile, profile_status)
    caveat_findings = [
        plainify(str(f.get("headline", ""))) for f in _caveat_findings(findings or []) if f.get("headline")
    ]
    bh = _apply_benjamini_hochberg(statistical_test_pvalues or [])
    if not (degradations or bh or unverified_claims or caveat_findings):
        return []

    lines = ["## Limitations & Caveats", ""]
    for item in degradations:
        lines.append(f"- {item}")
    for item in caveat_findings:
        lines.append(f"- {item}")
    if degradations or caveat_findings:
        lines.append("")

    if bh:
        lines.append(
            f"**Multiple-comparison correction**: {len(bh)} statistical test(s) ran this session. "
            f"Benjamini-Hochberg-corrected significance (FDR, α={_BH_ALPHA}):"
        )
        lines.append("")
        ordered = _sort_bh_tests(bh)
        shown = ordered[:_MAX_BH_ROWS]
        has_group = any(t.get("group_column") for t in shown)
        if has_group:
            lines.append("| Feature | Group | Test | p-value | BH-adjusted p | Significant after correction |")
            lines.append("|---------|-------|------|---------|----------------|-------------------------------|")
        else:
            lines.append("| Feature | Test | p-value | BH-adjusted p | Significant after correction |")
            lines.append("|---------|------|---------|----------------|-------------------------------|")
        for t in shown:
            row = f"| {t.get('feature_column', '—')} | "
            if has_group:
                row += f"{t.get('group_column') or '—'} | "
            row += (
                f"{t.get('test_name', '—')} | "
                f"{t.get('p_value', 0):.4f} | {t.get('p_adjusted', 0):.4f} | "
                f"{'Yes' if t.get('significant_after_correction') else 'No'} |"
            )
            lines.append(row)
        remaining = len(ordered) - len(shown)
        if remaining > 0:
            lines.append("")
            lines.append(f"… {remaining} more test(s) not shown.")
        lines.append("")

    if unverified_claims:
        lines.append("**Unverified claims** (numeric literals in the synthesis not traceable to a tool result):")
        for c in unverified_claims:
            lines.append(f"- {c}")
        lines.append("")

    return lines

#: Tools already covered by a dedicated report section (Model Performance)
#: or not an analytical finding at all — everything else gets a rich
#: subsection below instead of surviving only as one truncated line in the
#: flat Tool Execution Log table.
_BESPOKE_REPORTED_TOOLS = frozenset({
    "ingest_dataset", "clean_data", "detect_outliers", "correlation_analysis",
    "select_statistical_test", "train_model", "evaluate_model",
    "generate_report", "generate_visualizations", "planner",
})


def _format_additional_analyses(tool_results: list[dict[str, Any]]) -> list[str]:
    """
    Markdown subsection per successful tool result without a bespoke
    section above — cluster_data, time_series_analysis, text_analysis,
    geospatial_analysis, dimensionality_analysis, and any future tool.
    Mirrors app.py's `_render_other_findings` so the file-based report and
    the Streamlit UI agree on what "the full picture" contains.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for r in tool_results:
        name = r.get("tool_name", "")
        if name in _BESPOKE_REPORTED_TOOLS or name in seen or r.get("status") != "success":
            continue
        out = r.get("output")
        if not isinstance(out, dict):
            continue
        seen.add(name)
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append("")
        summary = out.get("summary")
        if summary:
            lines.append(str(summary))
            lines.append("")

        if name == "cluster_data":
            lines.append(f"- Clusters found: {out.get('n_clusters', '—')}")
            lines.append(f"- Silhouette score: {out.get('silhouette_score', '—')}")
            lines.append(f"- Separation: {out.get('separation_quality', '—')}")
        elif name == "time_series_analysis":
            lines.append(f"- Trend: {out.get('trend_direction', '—')}")
            lines.append(f"- Stationary: {'Yes' if out.get('is_stationary') else 'No'}")
            lags = out.get("seasonal_lags_detected") or []
            lines.append(f"- Seasonal lag(s): {', '.join(str(x) for x in lags) or 'None found'}")
        elif name == "text_analysis":
            lines.append(f"- Vocabulary size: {out.get('vocab_size', '—')}")
            lines.append(f"- Avg. words/row: {out.get('avg_word_count', '—')}")
            top = out.get("top_tokens") or []
            words = [t.get("token", t) if isinstance(t, dict) else t for t in top[:6]]
            if words:
                lines.append(f"- Most frequent words: {', '.join(str(w) for w in words)}")
        elif name == "geospatial_analysis":
            lines.append(f"- Points mapped: {out.get('n_points', '—')}")
            centroid = out.get("centroid") or {}
            if centroid:
                lines.append(f"- Centroid: {centroid.get('lat', '—')}, {centroid.get('lon', '—')}")
        elif name == "dimensionality_analysis":
            lines.append(f"- Numeric features: {out.get('n_features', '—')}")
            lines.append(f"- Components for target variance: {out.get('n_components_for_threshold', '—')}")
            pairs = out.get("high_correlation_pairs") or []
            lines.append(f"- Highly correlated pairs: {len(pairs)}")

        lines.append("")
    return lines


class GenerateReportTool(BaseTool):
    """
    Compile all analysis results into a structured Markdown report.

    Inputs come from the MemorySystem via the `tool_results_json` parameter
    (a JSON string of the serialised tool results list) and the `llm_insights`
    parameter (the LLM's final interpretation dict).
    """

    name = "generate_report"
    description = (
        "Assemble the final analysis report from all tool results and LLM insights. "
        "Produces a Markdown report and a JSON data file. "
        "Returns output file paths."
    )
    output_subdir = "reports"
    uses_cleaned_file = False  # takes no file_path param

    def applies_to(self, profile: Any, metadata: Any) -> float:
        # AgentController._generate_final_report calls this tool directly
        # and unconditionally after the reasoning loop ends (Stage 7) — it
        # is never something the planner itself needs to schedule.
        return 0.0

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        # The accumulated tool results live in memory, not in anything the
        # LLM plan can supply — always inject them fresh. Note: in practice
        # AgentController._generate_final_report calls this tool directly
        # via .run(), bypassing prepare_params entirely (applies_to() is
        # 0.0, so the planner never schedules it) — that call site injects
        # the same context explicitly. This still fills in the same
        # defaults for any other caller (tests, a future planned use).
        params = super().prepare_params(params, memory, output_root)
        meta = memory.dataset_metadata
        params.setdefault("dataset_name", Path(meta.file_path).stem if meta else "dataset")
        params["tool_results_json"] = json.dumps(
            [r.to_dict() for r in memory.tool_results], default=str
        )
        params.setdefault("llm_insights", {})
        params.setdefault("data_profile", memory.get_context("data_profile"))
        params.setdefault("read_report", memory.get_context("read_report"))
        params.setdefault("coercions", memory.get_context("coercions"))
        params.setdefault("plan_rationales", memory.get_context("plan_rationales"))
        params.setdefault("statistical_test_pvalues", memory.get_context("statistical_test_pvalues"))
        params.setdefault("unverified_claims", memory.get_context("unverified_claims"))
        params.setdefault("profile_status", memory.get_context("profile_status"))
        params.setdefault("degradations", memory.get_context("degradations"))
        params.setdefault("findings", [f.to_dict() for f in memory.ranked_findings()])
        params.setdefault("analysis_decision", memory.get_context("analysis_decision"))
        return params

    def execute(
        self,
        dataset_name: str = "dataset",
        tool_results_json: str = "[]",
        llm_insights: dict[str, Any] | None = None,
        output_dir: str = "output/reports",
        data_profile: dict[str, Any] | None = None,
        read_report: dict[str, Any] | None = None,
        coercions: list[dict[str, Any]] | None = None,
        plan_rationales: list[dict[str, Any]] | None = None,
        statistical_test_pvalues: list[dict[str, Any]] | None = None,
        unverified_claims: list[str] | None = None,
        profile_status: str | None = None,
        degradations: list[str] | None = None,
        findings: list[dict[str, Any]] | None = None,
        analysis_decision: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if llm_insights is None:
            llm_insights = {}
        # LLM sometimes double-serializes llm_insights as a JSON string
        if isinstance(llm_insights, str):
            try:
                llm_insights = json.loads(llm_insights)
            except json.JSONDecodeError:
                llm_insights = {}
        if not isinstance(llm_insights, dict):
            llm_insights = {}
        # 3c — internal-only keys (e.g. "_rlm_usage") never belong in a
        # report a human reads or in the raw JSON dump; drop them once,
        # up front, so every render/write below is already clean.
        llm_insights = {k: v for k, v in llm_insights.items() if not str(k).startswith("_")}

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if not tool_results_json or not tool_results_json.strip():
            tool_results_json = "[]"
        try:
            _parsed = json.loads(tool_results_json)
        except json.JSONDecodeError:
            try:
                _parsed, _offset = json.JSONDecoder().raw_decode(tool_results_json.strip())
            except json.JSONDecodeError:
                _parsed = []
        tool_results: list[dict[str, Any]] = _parsed if isinstance(_parsed, list) else []

        # Build Markdown
        md_lines: list[str] = [
            "# Agentic Data Analysis Report",
            "",
            f"**Dataset**: {dataset_name}  ",
            f"**Generated**: {timestamp}  ",
            "",
            "---",
            "",
        ]

        # Data Overview — shape/quality plus what was detected-or-assumed at
        # read time (item 2) and repaired before analysis (item 3).
        md_lines += _format_data_overview(data_profile, read_report, coercions)
        md_lines += _format_data_understanding(llm_insights, data_profile)

        # 7.9 layered skeleton, headline first: the top-ranked findings from
        # the shared bus, in plain language with the numbers embedded. Falls
        # back to the old insights list below when no findings were passed
        # (an old caller, or a run that produced none), so nothing regresses.
        md_lines += _format_top_findings(findings)

        # Executive summary — 3a: prefer `insights` over the LLM's raw
        # `reasoning` scratch text (which has leaked plumbing like "iter 1"
        # and "RLM Sub-Analysis Findings" into shipped reports), and strip
        # any jargon sentences from whichever source is used.
        executive_summary = _build_executive_summary(llm_insights)
        if executive_summary:
            md_lines += ["## Executive Summary", "", executive_summary, ""]

        insights = llm_insights.get("insights") or []
        recs = llm_insights.get("recommendations") or []

        if insights and not findings:
            md_lines += ["## Key Insights", ""]
            for i, insight in enumerate(insights, 1):
                md_lines.append(f"{i}. {plainify(str(insight))}")
            md_lines.append("")

        # Recommendations
        if recs:
            md_lines += ["## Recommendations", ""]
            for rec in recs:
                md_lines.append(f"- {plainify(str(rec))}")
            md_lines.append("")

        # Model performance — 3b: only call this "Model Performance" when a
        # model was actually trained; a describe-only (EDA) run's
        # key_metrics can be populated with unrelated numbers (e.g. a
        # gini_coefficient), which "Model Performance" would misrepresent.
        best_model = llm_insights.get("best_model")
        key_metrics = llm_insights.get("key_metrics", {})
        if best_model or key_metrics:
            section_title = "Model Performance" if _model_was_trained(llm_insights, tool_results) else "Key Metrics"
            md_lines += [f"## {section_title}", ""]
            if best_model:
                md_lines.append(f"**Best model**: {best_model}")
            if key_metrics:
                md_lines.append("")
                md_lines.append("| Metric | Value |")
                md_lines.append("|--------|-------|")
                for k, v in key_metrics.items():
                    md_lines.append(f"| {k} | {v} |")
            md_lines.append("")

        # Additional analyses (cluster/time-series/text/geo/dimensionality —
        # anything without a bespoke section above)
        additional = _format_additional_analyses(tool_results)
        if additional:
            md_lines += ["## Additional Analyses", "", *additional]

        # 7.9 — the analyst layer: every finding's effect size, p-value and
        # traceable evidence dict. This is exactly where a tool's own
        # findings() (cohort's revenue concentration, workforce's pay gap...)
        # now reaches the report regardless of whether it also got a bespoke
        # subsection above — the orphaning bug this round set out to fix.
        md_lines += _format_evidence(findings)

        # Methodology — why each analysis was chosen, from the planner's own
        # rationale (previously generated every run and then discarded), plus
        # (7.9/7.4) the analysis-mode decision — the honest "we considered
        # modelling X and declined, here's why" record.
        md_lines += _format_methodology(plan_rationales, analysis_decision)

        # Tool execution log
        if tool_results:
            md_lines += ["## Tool Execution Log", "", "| Tool | Status | Summary |", "|------|--------|---------|"]
            for r in tool_results:
                name = r.get("tool_name", "?")
                status = r.get("status", "?")
                summary = str((r.get("output") or {}).get("summary") or r.get("error") or "")[:120]
                md_lines.append(f"| {name} | {status} | {summary} |")
            md_lines.append("")

        # Limitations & Caveats — data-quality warnings, multiple-comparison
        # correction (item 4), degraded-profiling notice (item 7), and any
        # unverified metric claims (P0.7).
        md_lines += _format_limitations(
            data_profile, statistical_test_pvalues, unverified_claims, profile_status,
            degradations, findings=findings,
        )
        md_lines += _format_governance(llm_insights)

        md_lines += [
            "---",
            "",
            "*Report generated by the Agentic Data Analysis System.*",
            "*Architecture: Reasoning ↔ Execution separation with RLM context offloading.*",
        ]

        md_content = "\n".join(md_lines)
        md_path = Path(output_dir) / f"{dataset_name}_report.md"
        md_path.write_text(md_content, encoding="utf-8")

        json_path = Path(output_dir) / f"{dataset_name}_raw.json"
        json_path.write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "dataset": dataset_name,
                    "llm_insights": llm_insights,
                    "tool_results": tool_results,
                    "data_profile": data_profile,
                    "read_report": read_report,
                    "coercions": coercions,
                    "plan_rationales": plan_rationales,
                    "statistical_test_pvalues_bh_corrected": _apply_benjamini_hochberg(
                        statistical_test_pvalues or []
                    ),
                    "unverified_claims": unverified_claims,
                    "profile_status": profile_status,
                    "degradations": degradations,
                    "findings": findings,
                    "analysis_decision": analysis_decision,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        return {
            "summary": f"Report generated: {md_path} and {json_path}.",
            "markdown_path": str(md_path),
            "json_path": str(json_path),
            "n_insights": len(insights),
            "n_recommendations": len(recs),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "dataset_name": {
                "type": "string",
                "description": "Name used in the report filename and header.",
                "required": False,
            },
            "tool_results_json": {
                "type": "string",
                "description": "JSON-serialised list of tool result dicts.",
                "required": False,
            },
            "llm_insights": {
                "type": "dict",
                "description": "Final LLM interpretation dict (insights, recommendations, metrics).",
                "required": False,
            },
            "output_dir": {
                "type": "string",
                "description": "Output directory. Default: output/reports.",
                "required": False,
            },
            "findings": {
                "type": "list",
                "description": "Ranked Finding dicts from the shared finding bus (7.1). Injected automatically from memory.",
                "required": False,
            },
            "analysis_decision": {
                "type": "dict",
                "description": "The 7.4 analysis-mode decision record (mode/rationale/alternatives_rejected). Injected automatically from memory.",
                "required": False,
            },
        }
