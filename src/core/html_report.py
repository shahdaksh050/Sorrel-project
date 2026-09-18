"""
HTML Report Builder — a single shareable artifact for the whole analysis.

Produces a self-contained `report.html`: executive summary driven by the
user's objective, key insights, model drivers, metrics, the full interactive
dashboard (Vega-Lite via the vega-embed CDN), and the tool execution log.

Rules:
  - Pure string building, no I/O — the controller writes the file.
  - Every dataset- or LLM-derived string is HTML-escaped.
  - Charts render client-side from embedded JSON; viewing needs internet
    access for the CDN scripts (acceptable for a shareable artifact).
"""
from __future__ import annotations

import html
import json
import time
from typing import Any

from src.core.chart_theme import vega_config
from src.core.multiple_testing import apply_benjamini_hochberg

# "Ledger" (DESIGN.md): warm paper, friendly ink, one terracotta pen.
# The shared report is the same warm sheet as the console, printed.
_CSS = """
:root {
  color-scheme: light dark;
  --stock:       #f7eedd;
  --sheet:       #fffbf2;
  --sheet-alt:   #f1e4cb;
  --ink:         #3a2b1e;
  --graphite:    #8a7660;
  --pen:         #a34f20;
  --risk:        #a33526;
  --rule:        #e4d4bc;
  --rule-faint:  #eee3cb;
  --lift:        0 4px 14px rgba(58,43,30,.14);
}

@media (prefers-color-scheme: dark) {
  :root {
    --stock:       #241c14;
    --sheet:       #2f251a;
    --sheet-alt:   #3a2e1f;
    --ink:         #f3e9d8;
    --graphite:    #d0c2a8;
    --pen:         #f0a24a;
    --risk:        #e2685a;
    --rule:        #4a3c28;
    --rule-faint:  #3a2e1f;
    --lift:        0 4px 18px rgba(0,0,0,.35);
  }
}

* { box-sizing: border-box; }
body {
  background: var(--stock);
  color: var(--ink);
  font-family: 'Mukta', 'Segoe UI', system-ui, sans-serif;
  margin: 0; padding: 3.5rem 1.5rem; line-height: 1.65;
}
.wrap { max-width: 900px; margin: 0 auto; }
h1 { font-family: 'Baloo 2', 'Mukta', sans-serif; font-weight: 800;
     font-size: clamp(32px, 5vw, 52px); line-height: 1.06;
     margin: .3rem 0 .2rem; max-width: 18ch; color: var(--ink); }
h2 { font-family: 'Baloo 2', 'Mukta', sans-serif; font-weight: 700; font-size: 25px;
     line-height: 1.15; color: var(--ink);
     border-bottom: 2px solid var(--rule); padding-bottom: .4rem; margin: 3rem 0 1rem; }
.sub { color: var(--graphite); font-size: .85rem; margin-bottom: 2rem; }
.objective { border-left: 4px solid var(--pen); border-radius: 0 10px 10px 0;
             background: var(--sheet); padding: .6rem 0 .6rem 1rem;
             margin: 1.4rem 0; max-width: 72ch; }
.card { padding: .5rem 0 .5rem 1rem; margin: .1rem 0 .8rem;
        border-left: 3px solid var(--rule); border-radius: 0 10px 10px 0;
        background: var(--sheet); max-width: 74ch; }
.card.exec { background: var(--sheet); border: 1px solid var(--rule); border-left: 4px solid var(--pen);
             border-radius: 14px; box-shadow: var(--lift);
             padding: 1.5rem 1.7rem; max-width: 72ch; }
.insight { border-left-color: var(--graphite); }
.rec { border-left-color: var(--pen); }
.driver { border-left-color: var(--graphite); font-size: .95rem; }
.treat { border-left-color: var(--rule); font-size: .9rem; color: var(--graphite); }
.warn { border-left-color: var(--risk); color: var(--risk); }
table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: .88rem;
        background: var(--sheet); border: 1px solid var(--rule); border-radius: 12px; overflow: hidden;
        box-shadow: var(--lift); }
th, td { border-bottom: 1px solid var(--rule-faint); padding: .55rem .8rem; text-align: left; }
th { background: var(--sheet-alt); color: var(--ink); font-weight: 700; font-size: .8rem; }
.chart { background: var(--sheet); border: 1px solid var(--rule); border-radius: 14px;
         box-shadow: var(--lift);
         padding: 1.2rem 1.3rem 1.1rem; margin: 1.4rem 0; }
.chart h3 { margin: .1rem 0 .2rem; font-weight: 700; font-size: 1.05rem;
            font-family: 'Baloo 2', 'Mukta', sans-serif; }
.chart p { margin: .15rem 0 .9rem; color: var(--graphite); font-size: .85rem; max-width: 68ch; }
.vega-holder { width: 100%; }
.badge { display: inline-block; border: 1px solid var(--pen); color: var(--pen);
         background: var(--sheet); border-radius: 999px; font-weight: 600;
         padding: .25rem .85rem; font-size: .78rem; margin: 0 .4rem .4rem 0; }
.footer { margin-top: 4rem; border-top: 1px solid var(--rule); padding-top: .8rem;
          color: var(--graphite); font-size: .8rem; }
"""

_FONTS_CDN = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800'
    '&family=Mukta:wght@400;500;600;700&display=swap" rel="stylesheet">'
)

_VEGA_CDN = (
    '<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>'
)

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
    pool = sorted(pool, key=lambda f: f.get("importance") or 0.0, reverse=True)
    return pool[:_MAX_HEADLINE_FINDINGS]


def _evidence_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool = [
        f for f in findings
        if f.get("layer") == "analyst" and f.get("kind") not in _CAVEAT_FINDING_KINDS
    ]
    pool = sorted(pool, key=lambda f: f.get("importance") or 0.0, reverse=True)
    return pool[:_MAX_EVIDENCE_FINDINGS]


def _caveat_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in findings if f.get("kind") in _CAVEAT_FINDING_KINDS]


def _finding_evidence_html(finding: dict[str, Any]) -> str:
    """One evidence card: headline, analyst-facing detail, and the raw
    evidence numbers (effect/p-value/confidence) traceable to a tool result."""
    bits: list[str] = [f"<strong>{_esc(finding.get('headline', ''))}</strong>"]
    detail = finding.get("detail")
    if detail:
        bits.append(f"<br>{_esc(detail)}")
    numeric_bits: list[str] = []
    effect = finding.get("effect")
    if effect is not None:
        kind = finding.get("effect_kind") or ""
        numeric_bits.append(f"effect {_esc(round(float(effect), 4))}" + (f" ({_esc(kind)})" if kind else ""))
    p_value = finding.get("p_value")
    if p_value is not None:
        numeric_bits.append(f"p={_esc(round(float(p_value), 4))}")
    p_adj = finding.get("p_adjusted")
    if p_adj is not None:
        numeric_bits.append(f"p(adj)={_esc(round(float(p_adj), 4))}")
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
    return "".join(bits)


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
        f"<h1>What we found in {_esc(dataset_name)}</h1>"
        f'<div class="sub">Prepared {timestamp} by your data assistant</div>'
        f"<div>{badges}</div>"
    )

    # ---- objective + executive summary ----
    if objective:
        sections.append(
            f'<div class="objective">You asked: {_esc(objective)}</div>'
        )
    reasoning = llm_insights.get("reasoning", "")
    if reasoning:
        sections.append(
            f"<h2>The short version</h2><div class='card exec'>{_esc(reasoning)}</div>"
        )

    # ---- 7.9 headline layer — top findings from the shared finding bus.
    # Purely additive: when `findings` is empty/None this renders nothing
    # and the report falls back to the sections above/below exactly as
    # before (the required graceful-degradation path). ----
    findings = findings or []
    top_findings = _headline_findings(findings)
    if top_findings:
        headline_cards = "\n".join(
            f'<div class="card {"exec" if i == 0 else "insight"}">{_esc(f.get("headline", ""))}</div>'
            for i, f in enumerate(top_findings)
        )
        sections.append("<h2>Top findings</h2>" + headline_cards)

    # ---- insights & recommendations ----
    insights = llm_insights.get("insights") or []
    if insights:
        sections.append("<h2>What the data shows</h2>" + _cards(insights, "insight"))
    recs = llm_insights.get("recommendations") or []
    if recs:
        sections.append("<h2>What to do next</h2>" + _cards(recs, "rec"))

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

    # ---- key metrics ----
    key_metrics = llm_insights.get("key_metrics") or {}
    if key_metrics:
        rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in key_metrics.items()
        )
        sections.append(
            "<h2>Key numbers</h2><table><tr><th>Metric</th><th>Value</th></tr>"
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
            + f'<p>{_esc(c.get("description", ""))}</p>'
            f'<div class="vega-holder" id="chart_{i}"></div></div>'
            for i, c in enumerate(charts)
        )
        # Colors are injected here, at render time, from the shared theme
        # module — never stored in the spec itself — so this same HTML file
        # renders correctly whichever way the viewer's OS/browser theme is
        # set (7.17 / Q1: one chart theme module, no baked-in colors).
        specs = [dict(c.get("spec", {}), width="container") for c in charts]
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
            + "SPECS.forEach((s, i) => vegaEmbed('#chart_' + i, Object.assign({}, s, {config: _cfg}), {actions: false}));"
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
    limitation_cards.extend(f.get("headline", "") for f in _caveat_findings(findings))

    bh = apply_benjamini_hochberg(statistical_test_pvalues or [])
    if limitation_cards or bh:
        sections.append("<h2>Limitations &amp; caveats</h2>" + _cards(limitation_cards, "warn"))
        if bh:
            bh_rows = []
            for t in bh:
                p_val = f"{t.get('p_value', 0):.4f}"
                p_adj = f"{t.get('p_adjusted', 0):.4f}"
                sig = "Yes" if t.get("significant_after_correction") else "No"
                bh_rows.append(
                    f"<tr><td>{_esc(t.get('feature_column', '—'))}</td>"
                    f"<td>{_esc(t.get('test_name', '—'))}</td>"
                    f"<td>{_esc(p_val)}</td><td>{_esc(p_adj)}</td><td>{sig}</td></tr>"
                )
            rows = "".join(bh_rows)
            sections.append(
                f"<p>{len(bh)} statistical test(s) ran this session — "
                "Benjamini-Hochberg-corrected significance (FDR, α=0.05):</p>"
                "<table><tr><th>Feature</th><th>Test</th><th>p-value</th>"
                "<th>BH-adjusted p</th><th>Significant after correction</th></tr>"
                + rows + "</table>"
            )

    sections.append(
        '<div class="footer">Made for you by your data assistant.</div>'
    )

    body = "\n".join(sections)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Analysis Report — {_esc(dataset_name)}</title>"
        f"{_FONTS_CDN}{_VEGA_CDN}<style>{_CSS}</style></head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    )
