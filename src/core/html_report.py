"""
HTML Report Builder — a single shareable artifact for the whole analysis.

Produces a self-contained `report.html`: executive summary driven by the
user's objective, key insights, model drivers, metrics, the full interactive
dashboard (Vega-Lite, embedded inline from static/ so it opens offline, with a
data table behind every chart), and the tool execution log.

Rules:
  - Pure string building, no I/O — the controller writes the file.
  - Every dataset- or LLM-derived string is HTML-escaped.
  - Charts render client-side from embedded JSON. Fonts and the Vega bundles are
    inlined from static/ (src.core.report_assets), so the file opens offline;
    only if those files are missing does it fall back to CDN tags.
"""
from __future__ import annotations

import html
import json
import re
import time
from typing import Any

from src.core.audited_entry import GLYPH, audited_checks
from src.core.chart_theme import vega_config
from src.core.design_tokens import css_root_block
from src.core.multiple_testing import apply_benjamini_hochberg
from src.core.plain_language import describe_uncertainty, format_p, plainify
from src.core.report_assets import report_assets

# ---------------------------------------------------------------------------
# 3a — executive summary: prefer the LLM's own `insights` over its raw
# `reasoning` scratch text, which has leaked internal plumbing ("completed
# successfully across iter 1", "RLM Sub-Analysis Findings", "Form 2",
# literal tool names) straight into shipped reports. Mirrored in
# src.tools.report_generator so both reports agree on what the reader sees.
# ---------------------------------------------------------------------------

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
    """True only when a model was actually fitted (item 3b) — see the
    identical helper (and its full rationale) in
    src.tools.report_generator, which this mirrors so the two reports agree
    on when to title the section "Model Performance" vs "Key Metrics"."""
    if llm_insights.get("best_model"):
        return True
    return any(
        r.get("tool_name") == "train_model"
        and r.get("status") == "success"
        and isinstance(r.get("output"), dict)
        and r["output"].get("best_model")
        for r in tool_results
    )

# "Sorrel" (DESIGN.md): warm unbleached paper, near-black ink, one forest-green
# pen, hairline rules and square-ish corners. The shared report is the same
# sheet as the console, printed: no shadows, no gradients, nothing that moves.
# Token values come from src.core.design_tokens (FrontendPlan.md 2.1) — this
# file used to hand-type its own fourth copy of the same hex codes. Only
# color-scheme is not a palette token, so it stays here.
# Text that must be read uses the AA-safe --accent-text / --danger-text: raw
# --pen and --risk are fills and borders (Night --pen is ~2.5:1 as text).
# Built as an f-string root block plus a plain string for the rest of the
# rules, rather than one big f-string, so none of the CSS below needs its
# braces doubled.
_CSS_ROOT = f"""
:root {{
  color-scheme: light dark;
{css_root_block("day")}
}}

@media (prefers-color-scheme: dark) {{
  :root {{
{css_root_block("night")}
  }}
}}
"""

_CSS = _CSS_ROOT + """
* { box-sizing: border-box; }
body {
  background: var(--stock);
  color: var(--ink);
  font-family: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  margin: 0; padding: 3.5rem 1.5rem; line-height: 1.65;
}
.wrap { max-width: 900px; margin: 0 auto; }
.brand { display: flex; align-items: baseline; gap: .7rem; padding-bottom: .8rem;
         border-bottom: 1px solid var(--rule); margin-bottom: 1.6rem; }
.brand .mark { font-family: 'Newsreader', Georgia, serif; font-weight: 500; font-size: 1.5rem;
               letter-spacing: -.01em; color: var(--accent-text); }
.brand .kind { font: 500 .72rem/1.2 'Geist Mono', ui-monospace, monospace; letter-spacing: .08em;
               text-transform: uppercase; color: var(--graphite); }
h1 { font-family: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
     font-weight: 600; letter-spacing: -.02em;
     font-size: clamp(30px, 4.6vw, 46px); line-height: 1.12;
     margin: .3rem 0 .2rem; max-width: 20ch; color: var(--ink); }
h2 { font-family: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
     font-weight: 600; letter-spacing: -.01em; font-size: 23px;
     line-height: 1.2; color: var(--ink);
     border-bottom: 1px solid var(--rule); padding-bottom: .4rem; margin: 3rem 0 1rem; }
.sub { color: var(--graphite); font-size: .85rem; margin-bottom: 2rem; }
.objective { border-left: 3px solid var(--pen); border-radius: 0 4px 4px 0;
             background: var(--sheet); padding: .6rem 0 .6rem 1rem;
             margin: 1.4rem 0; max-width: 72ch; }
.card { padding: .5rem 0 .5rem 1rem; margin: .1rem 0 .8rem;
        border-left: 3px solid var(--rule); border-radius: 0 4px 4px 0;
        background: var(--sheet); max-width: 74ch; }
.card.exec { background: var(--sheet); border: 1px solid var(--rule); border-left: 3px solid var(--pen);
             border-radius: 4px;
             padding: 1.5rem 1.7rem; max-width: 72ch; }
.insight { border-left-color: var(--graphite); }
.rec { border-left-color: var(--pen); }
.driver { border-left-color: var(--graphite); font-size: .95rem; }
.treat { border-left-color: var(--rule); font-size: .9rem; color: var(--graphite); }
.warn { border-left-color: var(--risk); color: var(--danger-text); }
table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: .88rem;
        background: var(--sheet); border: 1px solid var(--rule); border-radius: 4px; overflow: hidden; }
th, td { border-bottom: 1px solid var(--rule-faint); padding: .55rem .8rem; text-align: left; }
th { background: var(--sheet-alt); color: var(--ink); font-weight: 600; font-size: .75rem;
     font-family: 'Geist Mono', ui-monospace, monospace; letter-spacing: .06em; text-transform: uppercase; }
.chart { background: var(--sheet); border: 1px solid var(--rule); border-radius: 4px;
         padding: 1.2rem 1.3rem 1.1rem; margin: 1.4rem 0; }
.chart h3 { margin: .1rem 0 .2rem; font-weight: 600; font-size: 1.05rem;
            font-family: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; }
.chart p { margin: .15rem 0 .9rem; color: var(--graphite); font-size: .85rem; max-width: 68ch; }
.vega-holder { width: 100%; }
/* vega-embed makes its target inline-block; with width:"container" that shrinks to 0 unless it is full width. */
.vega-holder .vega-embed { width: 100%; }
.chart-data { font-size: .78rem; margin: .4rem 0; }
.chart-data caption { text-align: left; color: var(--graphite); padding-bottom: .3rem; }
.chart-data-details summary { cursor: pointer; font-size: .8rem; color: var(--graphite); margin-top: .5rem; }
.badge { display: inline-block; border: 1px solid var(--pen); color: var(--accent-text);
         background: var(--sheet); border-radius: 999px; font-weight: 600;
         padding: .25rem .85rem; font-size: .78rem; margin: 0 .4rem .4rem 0; }
.footer { margin-top: 4rem; border-top: 1px solid var(--rule); padding-top: .8rem;
          color: var(--graphite); font-size: .8rem; }
.check-row { display: flex; flex-wrap: wrap; gap: .7rem; margin-top: .5rem; }
.check { font-size: .78rem; font-weight: 600; display: inline-flex; align-items: center; gap: 4px; }
.check.ok { color: var(--positive); }
.check.risk { color: var(--danger-text); }
.check.note { color: var(--graphite); font-weight: 500; }

@media print {
  .card, .chart, table { page-break-inside: avoid; }
  .chart-data-details { display: none; }
  .wrap { max-width: none; }
  a[href]:after { content: " (" attr(href) ")"; font-size: .75em; color: var(--graphite); }
}
"""

def _esc(value: Any) -> str:
    return html.escape(str(value))


def _cards(items: list[Any], css_class: str, prefix: str = "") -> str:
    return "\n".join(
        f'<div class="card {css_class}">{prefix}{_esc(item)}</div>' for item in items
    )


def _find_tool_output(tool_results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for r in reversed(tool_results):
        if r.get("tool_name") == name and r.get("status") == "success":
            out = r.get("output")
            if isinstance(out, dict):
                return out
    return {}


# ---------------------------------------------------------------------------
# 7.9 — the layered report's finding-driven sections. `findings` is the
# shared bus (src.core.findings): the caller (AgentController) already ranks
# it by importance descending, but every selector here re-sorts defensively
# rather than trusting call-site order.
#
# `method_fit`/`coverage_gap` findings are caveats, not discoveries — they
# are the one exception `is_trivial()` never suppresses (so they always
# reach *some* surface) but they must never occupy a headline/evidence slot.
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


#: FrontendPlan.md section 5, "Design rules" — same state->CSS-class mapping
#: as the Streamlit console's version of this component (ui/components/cards.py),
#: so the report and the console show identical trust marks (item 3.5).
_CHECK_STATE_CLASS: dict[str, str] = {"pass": "ok", "fail": "risk", "neutral": "note"}


def _check_row_html(evidence: dict[str, Any] | None) -> str:
    """`<div class="check-row">` for one finding's audited checks, or ""
    when it got no marks at all — an unrun check must never render as if it
    passed (FrontendPlan.md risk #2), so "no marks" means no row, not an
    empty one."""
    marks = audited_checks(evidence)
    if not marks:
        return ""
    spans = "".join(
        f'<span class="check {_CHECK_STATE_CLASS[m.state]}">{GLYPH[m.state]} {_esc(m.label)}</span>'
        for m in marks
    )
    return f'<div class="check-row">{spans}</div>'


def _finding_evidence_html(finding: dict[str, Any]) -> str:
    """One evidence card: headline, analyst-facing detail, and the raw
    evidence numbers (effect/p-value/confidence) traceable to a tool result."""
    bits: list[str] = [f"<strong>{_esc(plainify(str(finding.get('headline', ''))))}</strong>"]
    detail = finding.get("detail")
    if detail:
        bits.append(f"<br>{_esc(plainify(str(detail)))}")
    confidence = describe_uncertainty(finding)
    if confidence:
        bits.append(f"<br>{_esc(confidence)}")
    numeric_bits: list[str] = []
    effect = finding.get("effect")
    if effect is not None:
        kind = finding.get("effect_kind") or ""
        numeric_bits.append(f"effect {_esc(round(float(effect), 4))}" + (f" ({_esc(kind)})" if kind else ""))
    p_value = finding.get("p_value")
    if p_value is not None:
        numeric_bits.append(f"p={_esc(format_p(p_value))}")
    p_adj = finding.get("p_adjusted")
    if p_adj is not None:
        numeric_bits.append(f"p(adj)={_esc(format_p(p_adj))}")
    evidence = finding.get("evidence") or {}
    if isinstance(evidence, dict):
        for k, v in evidence.items():
            if isinstance(v, (int, float, str)) and not isinstance(v, bool):
                numeric_bits.append(f"{_esc(k)}={_esc(v)}")
    if numeric_bits:
        bits.append(f"<br><span style='color:var(--graphite);font-size:.85rem;'>{' · '.join(numeric_bits)}</span>")
    source = finding.get("source_tool")
    if source:
        bits.append(f"<br><span style='color:var(--graphite);font-size:.78rem;'>source: {_esc(source)}</span>")
    bits.append(_check_row_html(finding.get("evidence")))
    return "".join(bits)


#: 3d — mirrors src.tools.report_generator's _MAX_BH_ROWS/_sort_bh_tests:
#: the controller now feeds the BH correction every family-mode test *and*
#: every segment_comparison test in one run, which can run into the dozens.
_MAX_BH_ROWS = 15


def _sort_bh_tests(bh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        bh,
        key=lambda t: (not t.get("significant_after_correction"), t.get("p_adjusted") or 1.0),
    )


# ---------------------------------------------------------------------------
# "How the agent read the data" and "Governance" — shared with
# src.tools.report_generator so both reports show the same rows. Plain
# (label, value) text; each report escapes/formats it its own way.
# ---------------------------------------------------------------------------

def data_understanding_rows(
    llm_insights: dict[str, Any], profile: dict[str, Any] | None,
) -> tuple[list[tuple[str, str]], list[str]]:
    """(rows, caveats) from the planner's `data_understanding` block, with
    the profiler's archetype when the block doesn't carry one."""
    du = llm_insights.get("data_understanding")
    du = du if isinstance(du, dict) else {}
    profile = profile or {}
    rows: list[tuple[str, str]] = []
    for key, label in (("subject", "What the data is about"), ("domain", "Domain")):
        if du.get(key):
            rows.append((label, str(du[key])))
    archetype = du.get("archetype") or profile.get("archetype")
    if archetype:
        evidence = profile.get("archetype_evidence") if not du.get("archetype") else None
        rows.append((
            "Kind of table",
            str(archetype)
            + (f" ({'; '.join(str(e) for e in evidence[:3])})" if isinstance(evidence, list) and evidence else ""),
        ))
    if du.get("time_column"):
        rows.append(("Time column", str(du["time_column"])))
    for key, label in (("key_measures", "Key measures"), ("key_dimensions", "Key dimensions")):
        if isinstance(du.get(key), list) and du[key]:
            rows.append((label, ", ".join(str(v) for v in du[key])))
    roles = llm_insights.get("column_roles")
    if isinstance(roles, dict) and roles:
        rows.append((
            "Column roles inferred (validated against the data)",
            ", ".join(f"`{col}` → {role}" for col, role in roles.items()),
        ))
    caveats = [str(c) for c in du.get("caveats") or [] if str(c).strip()] if isinstance(du.get("caveats"), list) else []
    return rows, caveats


def governance_rows(llm_insights: dict[str, Any]) -> list[tuple[str, str]]:
    """Code-execution and LLM-usage accounting for the run, from
    `llm_insights["governance"]` (src.core.governance) plus `llm_usage`
    when the caller carried it."""
    gov = llm_insights.get("governance")
    gov = gov if isinstance(gov, dict) else {}
    usage = llm_insights.get("llm_usage")
    usage = usage if isinstance(usage, dict) else {}
    rows: list[tuple[str, str]] = []
    if "code_execution_enabled" in gov:
        rows.append(("Code execution", "enabled" if gov["code_execution_enabled"] else "disabled"))
    if gov.get("code_executions") is not None:
        budget = gov.get("execution_budget")
        rows.append(("Code runs", f"{gov['code_executions']}" + (f" of {budget} allowed" if budget is not None else "")))
    for keys, label in ((("code_failures", "failures"), "Failed runs"), (("code_refusals", "refusals"), "Refused runs")):
        value = next((gov[k] for k in keys if gov.get(k) is not None), None)
        if value is not None:
            rows.append((label, str(value)))
    if gov.get("sandbox_backends"):
        rows.append(("Sandbox", ", ".join(str(b) for b in gov["sandbox_backends"])))
    if gov.get("audit_log"):
        rows.append(("Audit log", str(gov["audit_log"])))
    if "local_only" in gov:
        rows.append(("Local-only mode", "on — no data sent to an external LLM" if gov["local_only"] else "off"))
    calls = gov.get("llm_calls", usage.get("call_count"))
    tokens = gov.get("llm_tokens", usage.get("total_tokens"))
    if calls is not None:
        rows.append(("LLM calls", str(calls)))
    if tokens is not None:
        cap = gov.get("llm_token_cap")
        rows.append((
            "LLM tokens",
            (f"{tokens:,}" if isinstance(tokens, int) else str(tokens)) + (f" of {cap:,} allowed" if isinstance(cap, int) and cap else ""),
        ))
    if gov.get("llm_audit_log"):
        rows.append(("LLM audit log", str(gov["llm_audit_log"])))
    cost = gov.get("estimated_cost_usd", usage.get("estimated_cost_usd"))
    if isinstance(cost, (int, float)) and cost > 0:
        rows.append(("Estimated LLM cost", f"${cost:,.4f} (approximate)"))
    return rows


#: A chart's data table is a readable fallback and an accessible alternative, not an export.
_MAX_TABLE_ROWS = 20
_MAX_TABLE_COLS = 8


def spec_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The data rows a Vega-Lite spec carries inline (top level, else its first layer)."""
    candidates: list[Any] = [spec.get("data")]
    candidates.extend(layer.get("data") for layer in spec.get("layer") or [] if isinstance(layer, dict))
    for data in candidates:
        values = data.get("values") if isinstance(data, dict) else None
        if isinstance(values, list) and values:
            return [row for row in values if isinstance(row, dict)]
    return []


def _table_cell(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return _esc("" if value is None else value)
    if isinstance(value, float):
        return _esc(f"{value:.4g}")
    return _esc(str(value)[:60])


def _chart_data_table(spec: dict[str, Any]) -> str:
    """Server-rendered table of the data behind a chart; "" when the spec carries none."""
    rows = spec_rows(spec)
    if not rows:
        return ""
    columns: list[str] = []
    for row in rows[:_MAX_TABLE_ROWS]:
        for key in row:
            if key not in columns and len(columns) < _MAX_TABLE_COLS:
                columns.append(str(key))
    shown = rows[:_MAX_TABLE_ROWS]
    note = (
        f"Data behind this chart (first {len(shown)} of {len(rows)} rows)"
        if len(rows) > len(shown)
        else f"Data behind this chart ({len(rows)} rows)"
    )
    head = "".join(f"<th>{_esc(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_table_cell(row.get(c))}</td>" for c in columns) + "</tr>"
        for row in shown
    )
    return f'<table class="chart-data"><caption>{_esc(note)}</caption><tr>{head}</tr>{body}</table>'


#: Draws each chart into a fresh node; on success the data table moves into a collapsed
#: <details>, on failure (or no Vega) the table stays visible as the fallback.
_CHART_SCRIPT = (
    "if (typeof vegaEmbed !== 'undefined') { SPECS.forEach((s, i) => {"
    "const holder = document.getElementById('chart_' + i);"
    "const table = holder.querySelector('.chart-data');"
    "const target = document.createElement('div');"
    "holder.insertBefore(target, holder.firstChild);"
    "vegaEmbed(target, Object.assign({}, s, {config: _cfg}), {actions: false}).then(() => {"
    "if (table) { const d = document.createElement('details'); d.className = 'chart-data-details';"
    "const m = document.createElement('summary'); m.textContent = 'Show the data behind this chart';"
    "d.appendChild(m); d.appendChild(table); holder.appendChild(d); }"
    "}).catch(() => { target.remove(); });"
    "}); }"
)


#: Where this module's own head ends. The embedded Vega bundles contain the text "</head>" inside their
#: scripts, so the closing tag alone is not a safe place to insert anything; this exact sequence is.
_HEAD_END = "</style></head><body>"

#: The line in a saved report that picks chart colours from the viewer's browser setting.
_AUTO_DARK_JS = "const _dark = !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);"


def retheme_report_html(document: str, theme: str) -> str:
    """A finished report page forced to the app's Day or Night theme.

    A saved report follows the viewer's browser setting (`prefers-color-scheme`), which is right for
    a file that is opened on its own. Embedded in the app it should follow the app's own Day/Night
    toggle instead, which the browser setting does not know about. This leaves the saved file
    untouched and returns a copy: the colour tokens are re-declared after the report's own rules, and
    the charts' light/dark choice is fixed to match.
    """
    night = theme in ("night", "dark")
    forced = (
        '<style id="forced-theme">:root {\n'
        f"  color-scheme: {'dark' if night else 'light'};\n"
        f"{css_root_block('night' if night else 'day')}\n"
        "}</style>"
    )
    out = document.replace(_AUTO_DARK_JS, f"const _dark = {'true' if night else 'false'};")
    end = out.rfind(_HEAD_END)
    if end < 0:
        return out
    at = end + len("</style>")  # after the report's own stylesheet, so the forced tokens win
    return out[:at] + forced + out[at:]


def build_html_report(
    dataset_name: str,
    llm_insights: dict[str, Any],
    tool_results: list[dict[str, Any]],
    charts: list[dict[str, Any]],
    objective: str = "",
    profile: dict[str, Any] | None = None,
    read_report: dict[str, Any] | None = None,
    coercions: list[dict[str, Any]] | None = None,
    plan_rationales: list[dict[str, Any]] | None = None,
    statistical_test_pvalues: list[dict[str, Any]] | None = None,
    unverified_claims: list[str] | None = None,
    profile_status: str | None = None,
    degradations: list[str] | None = None,
    findings: list[dict[str, Any]] | None = None,
    analysis_decision: dict[str, Any] | None = None,
) -> str:
    """
    Assemble the full self-contained HTML report.

    Args:
        dataset_name: Display name for the header.
        llm_insights: Final report dict (reasoning/insights/recommendations/...).
        tool_results: Serialised ToolResult dicts.
        charts:       Dashboard ChartSpec dicts ({chart_id,title,description,spec}).
        objective:    The user's natural-language goal, if any.
        profile:      DatasetProfile.to_dict(), if available.
        read_report:  src.core.io.ReadReport as a dict, if available (item 2).
        coercions:    src.core.coercion.Coercion dicts, if any (item 3).
        plan_rationales: [{step_number, tool_name, rationale}, ...] (item 6).
        statistical_test_pvalues: p-values accumulated this run, for the
            Benjamini-Hochberg correction (item 4).
        unverified_claims: Numeric literals P0.7 couldn't trace to a tool result.
        profile_status: "ok" or "failed: <reason>" (item 7).
        findings: Finding.to_dict() dicts (7.9 layered report) — the shared
            "what did we discover" bus every projection reads from. When
            empty/None (an old caller, or a run that produced none), the
            report still renders fully from tool_results/llm_insights as
            before; findings only ever *add* sections on top of that.
        analysis_decision: {"mode", "rationale", "alternatives_rejected", ...}
            — T2 "why we did or didn't model X" transparency, surfaced under
            "Why these analyses" when present.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    sections: list[str] = []

    # ---- header + badges ----
    badges = ""
    if profile:
        badges += f'<span class="badge">quality {_esc(profile.get("quality_score", "?"))}/100</span>'
        badges += f'<span class="badge">{_esc(profile.get("row_count", "?"))} rows</span>'
        badges += f'<span class="badge">{_esc(profile.get("column_count", "?"))} columns</span>'
        # Mirrors the Markdown report's "Recognised as ..." line so the two
        # reports do not diverge on how the data was classified.
        for match in profile.get("domains") or []:
            badges += (
                f'<span class="badge">{_esc(match.get("domain"))} data '
                f'({float(match.get("confidence", 0)):.2f})</span>'
            )
    best_model = llm_insights.get("best_model")
    if best_model:
        badges += f'<span class="badge">best model: {_esc(best_model)}</span>'

    sections.append(
        '<div class="brand"><span class="mark">Sorrel</span><span class="kind">Analysis report</span></div>'
        f"<h1>What we found in {_esc(dataset_name)}</h1>"
        f'<div class="sub">Prepared {timestamp} by Sorrel, your data assistant</div>'
        f"<div>{badges}</div>"
    )

    # ---- objective + executive summary ----
    if objective:
        sections.append(
            f'<div class="objective">You asked: {_esc(objective)}</div>'
        )
    # 3a — prefer `insights` over the LLM's raw `reasoning` scratch text,
    # which has leaked internal plumbing into shipped reports; strip any
    # jargon sentences from whichever source ends up used.
    executive_summary = _build_executive_summary(llm_insights)
    if executive_summary:
        sections.append(
            f"<h2>The short version</h2><div class='card exec'>{_esc(executive_summary)}</div>"
        )

    # ---- 7.9 headline layer's pool, computed early because the run-level
    # verdict (just below) counts over the same findings — matching
    # ui/tabs/answers_tab.py's `card_findings` (layer in exec/analyst, no
    # caveat kinds, capped at 5), so the report and the console verdict
    # agree on M, not just on the wording. ----
    findings = findings or []
    top_findings = _headline_findings(findings)

    # ---- 3.5 run-level verdict — mirrors ui/tabs/answers_tab.py's "N of M
    # held up" banner so the report and the console tell the same story.
    # Findings the audits never touched (no marks at all) don't count
    # toward M — a run with zero checked findings renders no verdict
    # rather than a fake one. ----
    audited_pool = [(f, audited_checks(f.get("evidence"))) for f in top_findings]
    audited_pool = [(f, marks) for f, marks in audited_pool if marks]
    total_checked = len(audited_pool)
    if total_checked > 0:
        held_up = sum(1 for _, marks in audited_pool if not any(m.state == "fail" for m in marks))
        verdict = f"{held_up} of {total_checked} finding{'s' if total_checked != 1 else ''} held up."
        needing_scrutiny = total_checked - held_up
        if needing_scrutiny > 0:
            verdict += f" {needing_scrutiny} need{'s' if needing_scrutiny == 1 else ''} more data."
        sections.append(f'<div class="card exec">{_esc(verdict)}</div>')

    # ---- 7.9 headline layer — top findings from the shared finding bus.
    # Purely additive: when `findings` is empty/None this renders nothing
    # and the report falls back to the sections above/below exactly as
    # before (the required graceful-degradation path). ----
    if top_findings:
        headline_cards = "\n".join(
            f'<div class="card {"exec" if i == 0 else "insight"}">'
            f'{_esc(plainify(str(f.get("headline", ""))))}{_check_row_html(f.get("evidence"))}</div>'
            for i, f in enumerate(top_findings)
        )
        sections.append("<h2>Top findings</h2>" + headline_cards)

    # ---- insights & recommendations ----
    insights = llm_insights.get("insights") or []
    if insights:
        sections.append("<h2>What the data shows</h2>" + _cards([plainify(str(i)) for i in insights], "insight"))
    recs = llm_insights.get("recommendations") or []
    if recs:
        sections.append("<h2>What to do next</h2>" + _cards([plainify(str(r)) for r in recs], "rec"))

    # ---- drivers (explainability) ----
    eval_out = _find_tool_output(tool_results, "evaluate_model")
    narrative = eval_out.get("driver_narrative") or []
    if narrative:
        sections.append("<h2>What drives the predictions</h2>" + _cards(narrative, "driver"))

    # ---- automatic treatments ----
    train_out = _find_tool_output(tool_results, "train_model")
    treatments = train_out.get("treatments_applied") or []
    if treatments:
        sections.append(
            "<h2>What the agent changed before modelling</h2>"
            + _cards(treatments, "treat")
        )

    # ---- key metrics — 3b: only call this "Model Performance" when a model
    # was actually trained; a describe-only (EDA) run's key_metrics can be
    # populated with unrelated numbers (e.g. a gini_coefficient), which
    # "Model Performance" would misrepresent. ----
    key_metrics = llm_insights.get("key_metrics") or {}
    if key_metrics:
        title = "Model Performance" if _model_was_trained(llm_insights, tool_results) else "Key Metrics"
        rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in key_metrics.items()
        )
        sections.append(
            f"<h2>{_esc(title)}</h2><table><tr><th>Metric</th><th>Value</th></tr>"
            + rows + "</table>"
        )

    # ---- 7.9 evidence layer — the analyst-facing detail behind the
    # headlines: every finding's detail/evidence dict, effect sizes and
    # p-values. Additive; no-op when `findings` is empty. ----
    evidence = _evidence_findings(findings)
    if evidence:
        sections.append(
            "<h2>Evidence</h2>"
            + "\n".join(f'<div class="card insight">{_finding_evidence_html(f)}</div>' for f in evidence)
        )

    # ---- interactive dashboard ----
    if charts:
        # Finding-tagged panels (priority > 0, set by src.core.dashboard) lead;
        # ties keep the dashboard-builder's own ordering (stable sort).
        charts = sorted(charts, key=lambda c: c.get("priority") or 0.0, reverse=True)
        chart_divs = "".join(
            f'<div class="chart"><h3>{_esc(c.get("title", ""))}</h3>'
            + (f'<p><strong>{_esc(c["caption"])}</strong></p>' if c.get("caption") else "")
            + (f'<p>{_esc(c["description"])}</p>' if c.get("description") and c["description"] != c.get("caption") else "")
            + f'<div class="vega-holder" id="chart_{i}">{_chart_data_table(c.get("spec") or {})}</div></div>'
            for i, c in enumerate(charts)
        )
        # Colors are injected here, at render time, from the shared theme
        # module — never stored in the spec itself — so this same HTML file
        # renders correctly whichever way the viewer's OS/browser theme is
        # set (7.17 / Q1: one chart theme module, no baked-in colors).
        specs = [
            dict(c.get("spec", {}), width="container") if {"mark", "layer"} & set(c.get("spec", {})) else c.get("spec", {})
            for c in charts
        ]
        specs_json = json.dumps(specs, default=str).replace("</", "<\\/")
        day_cfg_json = json.dumps(vega_config(dark=False), default=str).replace("</", "<\\/")
        night_cfg_json = json.dumps(vega_config(dark=True), default=str).replace("</", "<\\/")
        sections.append(
            "<h2>The charts</h2>"
            + chart_divs
            + f"<script>const SPECS = {specs_json};"
            + f"const VEGA_CFG_DAY = {day_cfg_json};"
            + f"const VEGA_CFG_NIGHT = {night_cfg_json};"
            + "const _dark = !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);"
            + "const _cfg = _dark ? VEGA_CFG_NIGHT : VEGA_CFG_DAY;"
            + _CHART_SCRIPT
            + "</script>"
        )

    # ---- tool log ----
    if tool_results:
        rows = "".join(
            f"<tr><td>{_esc(r.get('tool_name', '?'))}</td>"
            f"<td>{_esc(r.get('status', '?'))}</td>"
            f"<td>{_esc(str(r.get('output', {}).get('summary', r.get('error', '')) or '')[:140])}</td></tr>"
            for r in tool_results
        )
        sections.append(
            "<h2>Every step it ran</h2><table><tr><th>Tool</th><th>Status</th>"
            "<th>Summary</th></tr>" + rows + "</table>"
        )

    # ---- methodology (item 6) — why each step ran, from the planner, plus
    # (7.9) the T2 "why we did or didn't model X" transparency when the
    # controller recorded an analysis_decision. ----
    understanding, understanding_caveats = data_understanding_rows(llm_insights, profile)
    if understanding or understanding_caveats:
        section = "<h2>How the agent read the data</h2>"
        if understanding:
            section += "<table>" + "".join(
                f"<tr><td>{_esc(label)}</td><td>{_esc(value)}</td></tr>" for label, value in understanding
            ) + "</table>"
        if understanding_caveats:
            section += "<p>What it flagged to watch out for:</p>" + _cards(understanding_caveats, "treat")
        sections.append(section)

    methodology_html = ""
    if plan_rationales:
        rows = "".join(
            f"<tr><td>{_esc(r.get('step_number', '—'))}</td>"
            f"<td>{_esc(r.get('tool_name', '—'))}</td>"
            f"<td>{_esc(r.get('rationale', ''))}</td></tr>"
            for r in plan_rationales
        )
        methodology_html += (
            "<table><tr><th>Step</th><th>Tool</th><th>Rationale</th></tr>" + rows + "</table>"
        )
    if analysis_decision:
        mode = analysis_decision.get("mode")
        rationale = analysis_decision.get("rationale")
        rejected = analysis_decision.get("alternatives_rejected") or []
        decision_bits = []
        if mode:
            decision_bits.append(f"<strong>Approach taken: {_esc(mode)}.</strong>")
        if rationale:
            decision_bits.append(_esc(rationale))
        if decision_bits:
            card = "<br>".join(decision_bits)
            if rejected:
                card += "<br>Alternatives considered and not taken: " + "; ".join(_esc(r) for r in rejected) + "."
            methodology_html += f'<div class="card treat">{card}</div>'
    if methodology_html:
        sections.append("<h2>Why these analyses</h2>" + methodology_html)

    # ---- limitations & caveats (item 4 / 5 / 7 / 10 / P0.7) ----
    if degradations is None:
        # Direct callers without an accumulated degradations log (tests,
        # scripts) still get the same information, derived on the spot.
        from src.core.degradations import collect_degradations

        degradations = collect_degradations(read_report, coercions, profile, profile_status)
    limitation_cards: list[str] = list(degradations)
    if unverified_claims:
        limitation_cards.extend(str(c) for c in unverified_claims)
    # 7.9 — method-fit / coverage-gap findings are caveats, not headline
    # insights (src.core.findings.is_trivial never suppresses them for
    # exactly this reason); this is where they belong.
    limitation_cards.extend(plainify(str(f.get("headline", ""))) for f in _caveat_findings(findings))

    bh = apply_benjamini_hochberg(statistical_test_pvalues or [])
    if limitation_cards or bh:
        sections.append("<h2>Limitations &amp; caveats</h2>" + _cards(limitation_cards, "warn"))
        if bh:
            ordered = _sort_bh_tests(bh)
            shown = ordered[:_MAX_BH_ROWS]
            has_group = any(t.get("group_column") for t in shown)
            bh_rows = []
            for t in shown:
                p_val = format_p(t.get('p_value'))
                p_adj = format_p(t.get('p_adjusted'))
                sig = "Yes" if t.get("significant_after_correction") else "No"
                group_cell = f"<td>{_esc(t.get('group_column') or '—')}</td>" if has_group else ""
                bh_rows.append(
                    f"<tr><td>{_esc(t.get('feature_column', '—'))}</td>"
                    f"{group_cell}"
                    f"<td>{_esc(t.get('test_name', '—'))}</td>"
                    f"<td>{_esc(p_val)}</td><td>{_esc(p_adj)}</td><td>{sig}</td></tr>"
                )
            rows = "".join(bh_rows)
            group_header = "<th>Group</th>" if has_group else ""
            remaining = len(ordered) - len(shown)
            trailer = f"<p>… {remaining} more test(s) not shown.</p>" if remaining > 0 else ""
            sections.append(
                f"<p>{len(bh)} statistical test(s) ran this session — "
                "Benjamini-Hochberg-corrected significance (FDR, α=0.05):</p>"
                f"<table><tr><th>Feature</th>{group_header}<th>Test</th><th>p-value</th>"
                "<th>BH-adjusted p</th><th>Significant after correction</th></tr>"
                + rows + "</table>" + trailer
            )

    governance = governance_rows(llm_insights)
    if governance:
        sections.append(
            "<h2>Governance</h2><p>What the agent was allowed to run, and what it used.</p><table>"
            + "".join(f"<tr><td>{_esc(label)}</td><td>{_esc(value)}</td></tr>" for label, value in governance)
            + "</table>"
        )

    sections.append(
        '<div class="footer">Made for you by Sorrel, the working name of DSA Agent.</div>'
    )

    body = "\n".join(sections)
    assets = report_assets(need_vega=bool(charts))
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Sorrel analysis report — {_esc(dataset_name)}</title>"
        f"{assets.fonts_html}{assets.vega_html}<style>{_CSS}</style></head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    )
