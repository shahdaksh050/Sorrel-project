"""
Presentation Cards, Gauges, Stamps, and Layout Elements for Streamlit UI.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import streamlit as st

from src.core.audited_entry import GLYPH, audited_checks
from src.core.plain_language import describe_uncertainty, plainify
from ui.components.icons import (
    ICON_CHART,
    ICON_COMPASS,
    ICON_GLOBE,
    ICON_REPEAT,
    ICON_RULER,
    ICON_SEARCH,
    ICON_SHIELD,
    ICON_ZAP,
)

STAGE_DEFS: list[tuple[str, str]] = [
    ("1", "Reading Your File"),
    ("2", "Understanding Your Question"),
    ("3", "Running the Numbers"),
    ("4", "Making Sense of It"),
    ("5", "Double-Checking"),
    ("6", "Solving the Tricky Parts"),
    ("7", "Writing Your Report"),
]


def gauge(label: str, value: str, sub: str = "", *, flag: bool = False) -> str:
    """One cell of the instrument readout: the number leads, the label follows."""
    value = str(value)
    fit = "" if len(value) <= 11 else " long" if len(value) <= 18 else " longer"
    value, label, sub = html.escape(value), html.escape(str(label)), html.escape(str(sub))
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    return (
        f'<div class="gauge{fit}{" flag" if flag else ""}">'
        f'<div class="v">{value}</div>'
        f'<div class="k">{label}</div>{sub_html}</div>'
    )


def gap_is_risky(gap: float) -> bool:
    """A train-test gap above 10 points means the model memorised the split."""
    from src.tools.ml_pipeline import OVERFIT_THRESHOLD

    return gap > OVERFIT_THRESHOLD


def section(title: str, note: str = "", level: str = "h3") -> None:
    """Section head: the title sits on its own rule, with an optional mono note."""
    note_html = f'<div class="note">{html.escape(str(note))}</div>' if note else ""
    st.markdown(
        f'<div class="sect"><{level}>{html.escape(str(title))}</{level}>{note_html}</div>',
        unsafe_allow_html=True,
    )


def md_text(text: object) -> str:
    """Escape `$` so finding/chart text isn't rendered as LaTeX by st.markdown/st.caption."""
    return str(text).replace("$", "\\$")


def safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any datetime/Timestamp columns to strings so PyArrow can serialise them."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)
    return out


def stage_card(num: str, name: str, status: str, detail: str = "", is_new: bool = False) -> str:
    """One row of the stage ledger. Numbered: the pipeline is a real sequence."""
    cls = {
        "done": "done",
        "active": "active",
        "skipped": "skip",
        "error": "err",
    }.get(status, "")
    if is_new and status != "pending":
        cls += " is-new"
    det = f'<span class="detail">{html.escape(str(detail))}</span>' if detail else ""
    aria = ' aria-live="polite"' if status == "active" else ""
    return (
        f'<div class="sc {cls.strip()}"{aria}>'
        f'<span class="sc-num">{num.zfill(2)}</span>'
        f'<span class="nm">{html.escape(str(name))}</span>{det}</div>'
    )


def render_steps_list(stage_log: list[tuple[str, str, str]], seen_stages: set[str]) -> str:
    """Render the primary, always-visible numbered steps list."""
    log_map = {n: (s, d) for n, s, d in stage_log}
    html_out = ['<div aria-live="polite" aria-atomic="false">']
    for num, name in STAGE_DEFS:
        status, detail = log_map.get(num, ("pending", ""))
        is_new = False
        if status != "pending" and num not in seen_stages:
            is_new = True
            seen_stages.add(num)
        html_out.append(stage_card(num, name, status, detail, is_new))
    html_out.append('</div>')
    return "".join(html_out)


def render_datum(cells: list[tuple[str, str]]) -> str:
    """A ruled measurement bar. Each reading gets its own cell and hairline."""
    body = "".join(
        f'<div class="cell"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div></div>'
        for k, v in cells
    )
    return f'<div class="datum">{body}</div>'


#: CheckMark.state -> the CSS class the audited-entry check row uses for it.
_CHECK_CSS: dict[str, str] = {"pass": "ok", "fail": "risk", "neutral": "note"}


_LONG_DECIMAL = re.compile(r"(?<![\w.])-?\d+\.\d{3,}(?![\w.])")


def round_for_reading(text: str) -> str:
    """Round long decimals in prose for display (1,201.1968 reads as 1,201). The finding keeps its original values.

    Magnitude sets the precision: 100 and above shows no decimals, 10 and above one, 1 and above
    two, below 1 three. Values under 0.001 (p-values) are left alone so they are not rounded to zero.
    """
    def _round(match: re.Match[str]) -> str:
        value = float(match.group(0))
        magnitude = abs(value)
        if 0 < magnitude < 0.001:
            return match.group(0)
        if magnitude >= 100:
            return f"{value:,.0f}"
        if magnitude >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}" if magnitude >= 1 else f"{value:.3f}"

    return _LONG_DECIMAL.sub(_round, text)


FindingState = Literal["held", "needs_more", "unchecked"]

#: The three verdicts. Each carries a word and a different shape, so colour is never the only cue.
_STATE_WORD: dict[FindingState, str] = {
    "held": "Held up", "needs_more": "Needs more data", "unchecked": "Not checked",
}
_STATE_MARK: dict[FindingState, str] = {"held": "●", "needs_more": "⊗", "unchecked": "○"}


def finding_state(finding: dict[str, Any]) -> FindingState:
    """Held up, needs more data, or not checked, from the finding's own check marks.

    A check that did not run leaves no mark, and a finding whose only mark is the neutral
    "pattern, not proof" note was not checked either: absence of a check is never a failure.
    """
    marks = [c for c in audited_checks(finding.get("evidence")) if c.state != "neutral"]
    if not marks:
        return "unchecked"
    return "needs_more" if any(c.state == "fail" for c in marks) else "held"


def state_label(state: FindingState) -> str:
    """Shape and word for a verdict, as plain text (for radio options and captions)."""
    return f"{_STATE_MARK[state]} {_STATE_WORD[state]}"


def render_evidence_html(finding: dict[str, Any]) -> str:
    """The selected finding's evidence: headline, verdict, plain detail, checks that ran, caveats, source."""
    state = finding_state(finding)
    headline = html.escape(plainify(str(finding.get("headline", ""))))
    detail = round_for_reading(plainify(str(finding.get("detail") or "")))
    checks = audited_checks(finding.get("evidence"))
    marks = "".join(
        f'<span class="check {_CHECK_CSS[c.state]}">{GLYPH[c.state]} {html.escape(c.label)}</span>' for c in checks
    )
    caveats = "".join(
        f"<li>{html.escape(plainify(str(c)))}</li>" for c in (finding.get("caveats") or [])[:4]
    )
    source = str(finding.get("source_tool") or "").replace("_", " ")
    return (
        f'<div class="evidence-head"><span class="verdict-mark {state}">{html.escape(state_label(state))}</span>'
        f'<h3 class="evidence-title">{headline}</h3></div>'
        + (f'<p class="evidence-detail">{html.escape(detail)}</p>' if detail else "")
        + (f'<div class="check-row">{marks}</div>' if marks else '<p class="evidence-none">No check ran on this finding.</p>')
        + (f'<div class="evidence-caveats"><b>Be careful</b><ul>{caveats}</ul></div>' if caveats else "")
        + (f'<p class="evidence-source">Found by: {html.escape(source)}.</p>' if source else "")
    )


def render_finding_card(finding: dict[str, Any], chart_finding_ids: set[str], is_primary: bool = False) -> str:
    """One sentence card for a top-ranked Finding, with its audited-entry check row."""
    headline = html.escape(plainify(str(finding.get("headline", ""))))
    detail = finding.get("detail")
    caveats = finding.get("caveats") or []
    sub_text = round_for_reading(
        plainify(str(detail))
        if detail
        else plainify(str(caveats[0]))
        if caveats
        else describe_uncertainty(finding) or ""
    )
    sub_html = f'<div class="finding-detail">{html.escape(sub_text)}</div>' if sub_text else ""
    xref = ""
    if finding.get("finding_id") and finding["finding_id"] in chart_finding_ids:
        xref = '<div class="finding-chart-note">Chart available in the Charts tab.</div>'
    checks = audited_checks(finding.get("evidence"))
    check_html = ""
    is_flagged = False
    if checks:
        if any(c.state == "fail" for c in checks):
            is_flagged = True
        marks = "".join(
            f'<span class="check {_CHECK_CSS[c.state]}">{GLYPH[c.state]} {html.escape(c.label)}</span>'
            for c in checks
        )
        check_html = f'<div class="check-row animate">{marks}</div>'

    classes = ["finding-card"]
    if is_primary:
        classes.append("full-width")
    if is_flagged:
        classes.append("flagged")

    return (
        f'<div class="{" ".join(classes)}">'
        f'<div class="finding-headline">{headline}</div>'
        f'{sub_html}{xref}{check_html}'
        '</div>'
    )


def search_findings(query: str, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cheap, deterministic keyword search over the finding bus."""
    terms = [t for t in query.lower().split() if t]
    if not terms:
        return []
    scored: list[tuple[int, float, dict[str, Any]]] = []
    for f in findings:
        haystack = " ".join(
            str(f.get(k) or "") for k in ("headline", "measure", "dimension", "detail")
        ).lower()
        matches = sum(1 for t in terms if t in haystack)
        if matches:
            scored.append((matches, float(f.get("importance", 0.0)), f))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [f for _, _, f in scored]


def render_defect_stamp(gap_val: float | None) -> str:
    """Render authentic engineering defect stamp or certification seal for generalization."""
    if gap_val is None:
        return ""
    if gap_is_risky(gap_val):
        return f"""
        <div class="defect-stamp">
            <span class="stamp-tag">⚠ Heads up: this might not hold up</span>
            <div class="stamp-title">The model memorised the examples ({gap_val*100:.1f}% gap)</div>
            <div class="stamp-desc">
                It did noticeably better on the data it trained on than on data it hadn't seen.
                A sign it memorised quirks rather than learning the real pattern.
                We've ranked it lower because of this.
            </div>
        </div>
        """
    return f"""
    <div class="cert-stamp">
        <span class="stamp-tag">✓ Good news: this should hold up</span>
        <div class="stamp-title">The model performed consistently ({gap_val*100:.1f}% gap)</div>
        <div class="stamp-desc">
            It did about as well on new data as on the data it trained on.
            A good sign the pattern it found is real, not a fluke.
        </div>
    </div>
    """


def render_agent_grid(
    stage_log: list[tuple[str, str, str]],
    tool_results: list[dict[str, Any]] | None = None,
    report: dict[str, Any] | None = None,
) -> str:
    """Render visual architecture cards for the autonomous multi-agent teamwork roster."""
    log_map = {n: s for n, s, _ in stage_log}
    tool_results = tool_results or []
    report = report or {}

    agents = [
        {
            "icon": ICON_COMPASS,
            "name": "Planner",
            "role": "Plans the approach",
            "desc": "Reads your question and breaks it into a step-by-step plan, then decides when enough checking has been done.",
            "stage": "2",
            "tool": "Reasoning",
        },
        {
            "icon": ICON_SHIELD,
            "name": "File Checker",
            "role": "Checks your file is safe and healthy",
            "desc": "Makes sure your file is safe to open, figures out what each column means, spots anything unusual, and gives your data a health score out of 100.",
            "stage": "1",
            "tool": "Checks & cleans",
        },
        {
            "icon": ICON_RULER,
            "name": "Fact-Checker",
            "role": "Tests what's actually true",
            "desc": "Runs the right statistical tests to check whether a pattern is real or could just be chance, and finds which columns move together.",
            "stage": "3",
            "tool": "Statistical tests",
        },
        {
            "icon": ICON_ZAP,
            "name": "Model Builder",
            "role": "Builds and tests prediction models",
            "desc": "Trains several different prediction models and tests each one on different slices of your data, so a lucky guess doesn't get mistaken for a good model.",
            "stage": "3",
            "tool": "Model training",
        },
        {
            "icon": ICON_SEARCH,
            "name": "Reality-Checker",
            "role": "Catches models that just memorised",
            "desc": "Compares how each model does on data it trained on versus data it's never seen. If a model only looks good because it memorised the examples, this agent flags it and marks it down.",
            "stage": "4",
            "tool": "Model checking",
        },
        {
            "icon": ICON_REPEAT,
            "name": "Double-Checker",
            "role": "Goes back for another pass",
            "desc": "Looks at what's been found so far, and if there are loose ends or your question isn't fully answered yet, sends the work back for another round.",
            "stage": "5",
            "tool": "Another pass",
        },
        {
            "icon": ICON_GLOBE,
            "name": "Detail Handler",
            "role": "Handles the tricky, many-part questions",
            "desc": "When a question has too many moving parts to answer in one go, this splits it into smaller pieces, solves each one separately, and brings the answers back together.",
            "stage": "6",
            "tool": "Splitting up work",
        },
        {
            "icon": ICON_CHART,
            "name": "Report Writer",
            "role": "Builds your charts and report",
            "desc": "Builds charts that fit your data, then puts everything together into the report you can download and share.",
            "stage": "7",
            "tool": "Charts & report",
        },
    ]

    cards_html = []
    for ag in agents:
        st_val = log_map.get(ag["stage"], "pending")
        if st_val == "done":
            badge_cls = "done"
            badge_txt = "Done"
            card_cls = "agent-card"
        elif st_val == "active":
            badge_cls = "running"
            badge_txt = "Working"
            card_cls = "agent-card agent-active"
        elif st_val == "error":
            badge_cls = "error"
            badge_txt = "Needs attention"
            card_cls = "agent-card agent-flagged"
        else:
            badge_cls = ""
            badge_txt = "Waiting"
            card_cls = "agent-card"

        agent_output = ""
        if ag["name"] == "Planner":
            agent_output = report.get("reasoning", "Waiting for a plan.")
        elif ag["name"] == "File Checker":
            _prof = st.session_state.get("profile") or {}
            agent_output = f"Health score {_prof.get('quality_score', 'N/A')}/100. Cleaned up any issues found."
        elif ag["name"] == "Fact-Checker":
            agent_output = "Tests complete: checked which columns move together and whether the differences are real."
        elif ag["name"] == "Model Builder":
            agent_output = f"Best model so far: {report.get('best_model', 'N/A')}. Tested multiple times on different slices of your data."
        elif ag["name"] == "Reality-Checker":
            agent_output = "Checked every model for memorisation. Applied a penalty to any that didn't hold up."
        elif ag["name"] == "Double-Checker":
            agent_output = "Finished reviewing. Went back for more passes where needed."
        elif ag["name"] == "Detail Handler":
            agent_output = f"{len(report.get('rlm_sub_results', []))} smaller questions solved separately and combined."
        elif ag["name"] == "Report Writer":
            agent_output = "Report finished, with charts, key findings, and what to do next."

        cards_html.append(
            f"""
        <div class="{card_cls} agent-card">
            <details class="agent-details">
                <summary class="agent-summary">
                    <div class="agent-header">
                        <span class="agent-role">{ag['icon']} {ag['name']}</span>
                        <span class="agent-badge {badge_cls}">{badge_txt}</span>
                    </div>
                    <div class="agent-desc">{ag['desc']}</div>
                    <div class="agent-metric">Role: {ag['role']}. Tool: {ag['tool']}</div>
                </summary>
                <div class="agent-more">
                    <div class="agent-more-rule"><strong>Rule:</strong> {ag['desc']}</div>
                    <div class="agent-more-found"><strong>Found:</strong> {html.escape(str(agent_output))}</div>
                </div>
            </details>
        </div>
        """.strip()
        )

    return f'<div class="agent-grid org-chart-layout">{"".join(cards_html)}</div>'


def render_handoff_stream(progress_lines: list[str], tool_results: list[dict[str, Any]]) -> str:
    """Render timeline feed of inter-agent messages and handoffs."""
    items_html = []
    if progress_lines:
        for line in progress_lines[:20]:
            if not line.strip():
                continue
            meta = "Note"
            if "[done]" in line:
                meta = "Done"
            elif "[run ]" in line:
                meta = "Started"
            elif "ok" in line:
                meta = "Finished"
            clean_text = (
                line.replace("[done]", "")
                .replace("[run ]", "")
                .replace("[    ]", "")
                .replace("[fail]", "⚠ ")
                .strip()
            )
            items_html.append(
                f"""
            <div class="handoff-item">
                <div class="handoff-meta">{meta}</div>
                <div class="handoff-text">{html.escape(clean_text)}</div>
            </div>
            """.strip()
            )
    elif tool_results:
        for r in tool_results:
            name = r.get("tool_name", "Step")
            status = r.get("status", "success")
            summary = r.get("output", {}).get("summary", "") or r.get("error", "")
            time_ms = r.get("execution_time_ms", 0)
            items_html.append(
                f"""
            <div class="handoff-item">
                <div class="handoff-meta">{html.escape(str(name))} · {html.escape(str(status))} · {time_ms:.0f}ms</div>
                <div class="handoff-text">{html.escape(str(summary))}</div>
            </div>
            """.strip()
            )
    else:
        items_html.append(
            """
        <div class="handoff-item">
            <div class="handoff-meta">Waiting</div>
            <div class="handoff-text">Your helpers are ready. Upload a file to get started.</div>
        </div>
        """.strip()
        )
    return f'<div class="handoff-stream">{"".join(items_html)}</div>'


def render_agent_deep_dive(
    agent_name: str, tool_results: list[dict[str, Any]], report: dict[str, Any]
) -> None:
    """Render structured details for an inspected agent persona."""
    details = {
        "🧭 Planner": {
            "mission": "Reads your question and turns it into a step-by-step plan: what to check first, what to try next, and when the plan needs adjusting.",
            "directive": "Only works from summaries and statistics, never your raw data rows, the way a manager works from a report rather than the raw ledger.",
            "tools": "Reasoning and planning",
            "output": report.get("reasoning", "Waiting for a plan."),
        },
        "🛡️ File Checker": {
            "mission": "Checks your file is safe to open, figures out what each column means, and gives your data a health score.",
            "directive": "Scores your data 0–100 based on missing values, duplicate rows, and anything that looks off.",
            "tools": "File safety checks, data profiling",
            "output": f"Health score {st.session_state.get('profile', {}).get('quality_score', 'N/A')}/100. Cleaned up any issues found.",
        },
        "📐 Fact-Checker": {
            "mission": "Runs statistical tests to check whether a pattern in your data is real, or could just be chance.",
            "directive": "Checks how your data is shaped before picking which test is fair to use: the right test depends on the shape.",
            "tools": "Statistical tests, correlation checks",
            "output": "Tests complete: checked which columns move together and whether the differences are real.",
        },
        "⚡ Model Builder": {
            "mission": "Trains a few different prediction models on your data and scores each one.",
            "directive": "Tests every model on several different slices of the data, not just one, so a lucky split doesn't make a bad model look good.",
            "tools": "Model training (several approaches, tested against each other)",
            "output": f"Best model so far: {report.get('best_model', 'N/A')}. Tested multiple times on different slices of your data.",
        },
        "🔍 Reality-Checker": {
            "mission": "Compares how each model performs on data it trained on versus data it's never seen.",
            "directive": "If a model does noticeably better on familiar data than new data, it's flagged as having memorised rather than learned, and marked down.",
            "tools": "Model checking",
            "output": "Checked every model for memorisation. Applied a penalty to any that didn't hold up.",
        },
        "🔁 Double-Checker": {
            "mission": "Looks at what's been found so far and decides whether your question has really been answered.",
            "directive": "Sends the work back for another pass if things haven't settled down yet, up to a set limit of tries.",
            "tools": "Review and another pass",
            "output": "Finished reviewing. Went back for more passes where needed.",
        },
        "🌐 Detail Handler": {
            "mission": "Splits a big, many-part question into smaller pieces, solves each on its own, then brings the answers back together.",
            "directive": "Keeps each piece small and separate, so a complicated question doesn't overwhelm any single step.",
            "tools": "Splitting up and recombining work",
            "output": f"{len(report.get('rlm_sub_results', []))} smaller questions solved separately and combined.",
        },
        "📊 Report Writer": {
            "mission": "Builds charts that fit your data and puts everything into a report you can download and share.",
            "directive": "Uses the same easy-to-read style for the charts and the report as the rest of the app, and gives you both a written version and a webpage version.",
            "tools": "Charts and report writing",
            "output": "Report finished, with charts, key findings, and what to do next.",
        },
    }
    info = details.get(agent_name, details["🧭 Planner"])
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.markdown(f"**What it does:** {info['mission']}")
        st.markdown(f"**Its rule:** {info['directive']}")
    with c2:
        st.markdown(f"**What it uses:** {info['tools']}")
        st.markdown(f"**What it found:** {info['output']}")


def find_tool(tool_results: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    """Find the output dict of a successful tool run by tool name."""
    for r in tool_results:
        if r.get("tool_name") == name and r.get("status") == "success":
            out = r.get("output")
            return out if isinstance(out, dict) else {}
    return None


def get_vega_config() -> dict[str, Any]:
    """Ledger palette for Vega-Lite charts."""
    from src.core.chart_theme import vega_config as _vega_config_impl

    dark = st.session_state.get("theme", "day") in ("night", "dark")
    return _vega_config_impl(dark=dark)


def find_chart_by_id(dash: list[dict[str, Any]] | None, chart_id: str) -> dict[str, Any] | None:
    """Look up a dashboard panel by its chart_id."""
    for ch in (dash or []):
        if ch.get("chart_id") == chart_id:
            return ch
    return None


def _render_chart_data(spec: dict[str, Any]) -> None:
    """The numbers behind a chart, as a table: a non-visual alternative to the canvas."""
    from src.core.html_report import spec_rows

    rows = spec_rows(spec)
    if not rows:
        return
    shown = rows[:20]
    note = f"first {len(shown)} of {len(rows)} rows" if len(rows) > len(shown) else f"{len(rows)} rows"
    with st.expander(f"Data behind this chart ({note})", expanded=False):
        st.dataframe(safe_df(pd.DataFrame(shown)), width="stretch")


def render_dashboard_chart(ch: dict[str, Any], vega_cfg: dict[str, Any], finding_note: str = "") -> None:
    """Render one dashboard panel (title, chart, caption/description).

    `finding_note` says which finding the chart supports and its verdict, when there is one.
    """
    with st.container(border=True):
        st.markdown(
            f'<h4 class="chart-title">{html.escape(str(ch.get("title", "")))}</h4>',
            unsafe_allow_html=True,
        )
        if finding_note:
            st.caption(finding_note)
        spec = dict(ch.get("spec", {}))
        spec.setdefault("background", "transparent")
        spec.setdefault("config", vega_cfg)
        st.vega_lite_chart(spec, width="stretch")
        caption = ch.get("caption")
        if caption:
            st.caption(md_text(caption))
        elif ch.get("description"):
            st.markdown(
                f'<div class="chart-desc">{html.escape(str(ch["description"]))}</div>',
                unsafe_allow_html=True,
            )
        _render_chart_data(spec)


def render_data_understanding(du: dict[str, Any]) -> str:
    """The planner's first read of the dataset, as one compact card."""
    def _row(label: str, value: Any) -> str:
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        if not value:
            return ""
        return f'<div><span class="k">{html.escape(label)}</span>{html.escape(str(value))}</div>'

    kind = " · ".join(str(du[k]) for k in ("domain", "archetype") if du.get(k))
    rows = "".join([
        _row("Subject", du.get("subject")),
        _row("Kind", kind),
        _row("Measures", du.get("key_measures")),
        _row("Dimensions", du.get("key_dimensions")),
        _row("Time", du.get("time_column")),
    ])
    caveats = "".join(
        f'<div class="note">⚠ {html.escape(str(c))}</div>' for c in du.get("caveats") or []
    )
    if not rows and not caveats:
        return ""
    return f'<div class="du">{rows}{caveats}</div>'


BESPOKE_RENDERED_TOOLS: frozenset[str] = frozenset({
    "ingest_dataset", "clean_data", "detect_outliers", "correlation_analysis",
    "select_statistical_test", "train_model", "evaluate_model",
    "generate_report", "generate_visualizations", "planner",
})


def render_other_findings(tool_results: list[dict[str, Any]]) -> None:
    """Fallback card for any successful tool result without a bespoke section."""
    seen: set[str] = set()
    shown_any = False
    for r in tool_results:
        name = r.get("tool_name", "")
        if name in BESPOKE_RENDERED_TOOLS or name in seen or r.get("status") != "success":
            continue
        out = r.get("output")
        if not isinstance(out, dict):
            continue
        seen.add(name)
        shown_any = True
        st.markdown(f"#### {name.replace('_', ' ').title()}")
        summary = out.get("summary")
        if summary:
            st.info(md_text(summary))

        if name == "cluster_data":
            c1, c2, c3 = st.columns(3)
            c1.metric("Clusters found", out.get("n_clusters", "N/A"))
            c2.metric("Silhouette score", out.get("silhouette_score", "N/A"))
            c3.metric("Separation", out.get("separation_quality", "N/A"))
        elif name == "time_series_analysis":
            c1, c2, c3 = st.columns(3)
            c1.metric("Trend", str(out.get("trend_direction", "N/A")).title())
            c2.metric("Stationary?", "Yes" if out.get("is_stationary") else "No")
            lags = out.get("seasonal_lags_detected") or []
            c3.metric("Seasonal lag(s)", ", ".join(str(x) for x in lags) or "None found")
        elif name == "text_analysis":
            c1, c2, c3 = st.columns(3)
            c1.metric("Vocabulary size", out.get("vocab_size", "N/A"))
            c2.metric("Avg. words / row", out.get("avg_word_count", "N/A"))
            top = out.get("top_tokens") or []
            # TextAnalysisTool's real shape is a dict ({token: count}, see
            # src/tools/text_analysis.py), not a list — `top[:6]` on a plain
            # dict raised TypeError pre-3.12, but slice objects became
            # hashable in Python 3.12+, so it now does a dict lookup and
            # raises KeyError instead. Normalize to a token list first, kept
            # defensive for a list-of-dicts/list-of-strings shape too.
            if isinstance(top, dict):
                top = list(top.keys())
            elif not isinstance(top, list):
                top = []
            words = [t.get("token", t) if isinstance(t, dict) else t for t in top[:6]]
            if words:
                st.caption(md_text("Most frequent words: " + ", ".join(str(w) for w in words)))
        elif name == "geospatial_analysis":
            c1, c2 = st.columns(2)
            c1.metric("Points mapped", out.get("n_points", "N/A"))
            centroid = out.get("centroid") or {}
            if centroid:
                c2.metric("Centroid", f"{centroid.get('lat', 'N/A')}, {centroid.get('lon', 'N/A')}")
        elif name == "dimensionality_analysis":
            c1, c2, c3 = st.columns(3)
            c1.metric("Numeric features", out.get("n_features", "N/A"))
            threshold = out.get("variance_threshold")
            c2.metric(
                f"Components for {threshold:.0%} variance" if threshold else "Components needed",
                out.get("n_components_for_threshold", "N/A"),
            )
            pairs = out.get("high_correlation_pairs") or []
            c3.metric("Highly correlated pairs", len(pairs))

    if not shown_any:
        st.caption("No additional analyses ran for this dataset.")


def read_audit_log(path: str | None) -> list[dict[str, Any]]:
    """Audit records from the append-only JSONL; unreadable lines are skipped."""
    if not path:
        return []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def render_governance(gov: dict[str, Any]) -> None:
    """Governance summary and the code-execution audit trail for this run."""
    budget = gov.get("execution_budget")
    cells = [
        ("Code runs", f"{gov.get('code_executions', 0)} / {budget}" if budget is not None
         else str(gov.get("code_executions", 0)), False),
        ("Failed", str(gov.get("code_failures", 0)), bool(gov.get("code_failures"))),
        ("Refused", str(gov.get("code_refusals", 0)), bool(gov.get("code_refusals"))),
    ]
    if gov.get("llm_calls") is not None:
        cells.append(("LLM calls", f"{gov['llm_calls']:,}", False))
    if gov.get("llm_tokens") is not None:
        cap = gov.get("llm_token_cap") or 0
        cells.append(("LLM tokens", f"{gov['llm_tokens']:,}" + (f" / {cap:,}" if cap else ""),
                      bool(cap) and gov["llm_tokens"] >= cap))
    for col, (label, value, flag) in zip(st.columns(len(cells)), cells, strict=True):
        col.markdown(gauge(label, value, flag=flag), unsafe_allow_html=True)

    if not gov.get("code_execution_enabled", True):
        st.caption("AI-written code was switched off for this run.")
    elif gov.get("code_execution_blocker"):
        st.caption(f"AI-written code could not run in this analysis. {gov['code_execution_blocker']}")
    backends = [b for b in gov.get("sandbox_backends") or [] if b != "refused"]
    if "subprocess" in backends:
        note = ", some runs used Docker" if "docker" in backends else ""
        st.markdown(
            '<span class="iso-badge warn">Process-level isolation only (subprocess)'
            f'{note}</span>',
            unsafe_allow_html=True,
        )
        st.caption("Set SANDBOX_BACKEND=docker, or turn on \"Require container isolation\", "
                   "for a kernel boundary around AI-written code.")
    elif "docker" in backends:
        st.markdown('<span class="iso-badge ok">Container isolation (Docker)</span>',
                    unsafe_allow_html=True)

    entries = read_audit_log(gov.get("audit_log"))
    if not entries:
        st.caption("No AI-written code ran or was refused in this run."
                   if not gov.get("audit_log") else "The audit log could not be read.")
        return

    def _ms(value: Any) -> str:
        return f"{value:,.0f}" if isinstance(value, (int, float)) else "N/A"

    with st.expander(f"Audit log: {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}"):
        st.dataframe(safe_df(pd.DataFrame([{
            "Time": str(e.get("timestamp") or "")[:19].replace("T", " "),
            "Tool": e.get("tool_name"),
            "Status": e.get("status"),
            "Backend": e.get("backend") or "N/A",
            "Duration (ms)": _ms(e.get("duration_ms")),
            "SHA-256": (e.get("code_sha256") or "")[:12] or "N/A",
        } for e in entries])), width='stretch')
        st.caption(f"Full record: {gov.get('audit_log')}")

    for i, e in enumerate(entries, 1):
        sha = (e.get("code_sha256") or "")[:12]
        with st.expander(f"#{i:02d} `{e.get('tool_name', '?')}` · {e.get('status', '?')}"
                         + (f" · `{sha}`" if sha else "")):
            if e.get("code"):
                st.code(e["code"], language="python")
            if e.get("error"):
                st.code(e["error"], language=None)


def render_run_compare() -> None:
    """Diff the current run against a chosen earlier one (output/runs)."""
    from src.core.run_compare import compare_runs, list_runs

    cur_path = st.session_state.get("current_summary_path")
    if not cur_path:
        return
    others = [r for r in list_runs() if r["path"] != cur_path]
    with st.expander("Compare with a previous run"):
        if not others:
            st.caption("No earlier runs saved yet.")
            return
        pick = st.selectbox("Previous run", others, format_func=lambda r: r["label"], key="cmp_pick")
        cur = json.loads(Path(cur_path).read_text(encoding="utf-8"))
        d = compare_runs(pick, cur)
        for title, key in (("New findings", "findings_added"), ("No longer found", "findings_removed"),
                           ("New charts", "charts_added"), ("Charts dropped", "charts_removed")):
            if d[key]:
                st.markdown(f"**{title}**")
                for h in d[key]:
                    st.markdown(f"- {md_text(h)}")
        if d["findings_changed"]:
            st.markdown("**Changed importance**")
            for c in d["findings_changed"]:
                st.markdown(f"- {md_text(c['headline'])} ({c['old']:.2f} to {c['new']:.2f})")
        slow = [t for t in d["tool_time_deltas"] if abs(t["delta"]) >= 0.5]
        if slow:
            st.markdown("**Tool time change (seconds)**")
            for t in slow[:8]:
                st.markdown(f"- `{t['tool']}`: {t['delta']:+.1f}s ({t['old']:.1f} to {t['new']:.1f})")
        if not any(d[k] for k in ("findings_added", "findings_removed", "findings_changed",
                                  "charts_added", "charts_removed")):
            st.caption("No differences in findings or charts.")

