"""
Answers Tab (Executive Briefing & Key Findings).

Reading order: the question and its answer, the verdict, the evidence (finding
cards), what to be careful about, what to do, then supporting context and the
collapsed technical detail. Findings, verdict, coverage and recommendations
come from the `RunView`; `tool_results` is read only for model detail.
"""
from __future__ import annotations

import html
from typing import Any

import streamlit as st

from src.core.plain_language import plainify
from src.core.run_view import RunView, build_run_view
from ui.components.cards import (
    find_chart_by_id,
    find_tool,
    gap_is_risky,
    md_text,
    render_dashboard_chart,
    render_data_understanding,
    render_defect_stamp,
    render_finding_card,
    render_run_compare,
    search_findings,
)


def render_answers_tab(
    report: dict[str, Any],
    tool_results: list[dict[str, Any]],
    meta: Any,
    dash: list[dict[str, Any]] | None,
    prof: dict[str, Any] | None,
    preview_df: Any | None,
    vega_cfg: dict[str, Any],
    run_view: RunView | None = None,
) -> None:
    """Render Tier 1: Executive Briefing & Key Discoveries."""
    # A finished run always carries a view; the fallback only protects a session
    # that predates it, and it shows nothing the report did not already hold.
    view = run_view or build_run_view(report, {}, objective="", is_sample=False)

    train_out = find_tool(tool_results, "train_model")
    corr_out = find_tool(tool_results, "correlation_analysis")

    best_model = report.get("best_model") or "N/A"
    best_cv = "0"
    best_gap_str = "0"
    gap_val: float | None = None

    if train_out:
        mt_map = train_out.get("models_trained", {})
        best = train_out.get("best_model", "")
        if best and best in mt_map:
            bm = mt_map[best]
            best_cv = f"{bm.get('cv_mean', 0)*100:.1f}"
            gap_val = bm.get("train_test_gap")
            best_gap_str = f"{gap_val*100:.1f}" if gap_val is not None else "0"

    task_type = (
        (train_out.get("task_type") if train_out else None)
        or (meta.task_type if meta else "N/A")
        or "N/A"
    )

    prof = prof or {}
    q = int(prof.get("quality_score", 0))

    # 1. The question as the heading, the answer under it.
    answer = html.escape(plainify(view.reasoning)) if view.reasoning else ""
    if view.objective:
        st.markdown(
            '<div class="exec-directive">'
            f"<h2>You asked: {html.escape(view.objective)}</h2>"
            + (f'<p class="dir-content">{answer}</p>' if answer else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    elif answer:
        st.markdown(
            f'<div class="exec-directive"><p class="dir-content">{answer}</p></div>',
            unsafe_allow_html=True,
        )

    # 2. Verdict. Only when audits actually ran: a "0 of 0" would claim a check
    # that did not happen.
    verdict = view.verdict
    if verdict is not None:
        sub_lines = []
        if verdict.needs_more > 0:
            sub_lines.append(
                f"{verdict.needs_more} need{'s' if verdict.needs_more == 1 else ''} more data."
            )
        if gap_val is not None:
            sub_lines.append(
                "The model held up on new data."
                if not gap_is_risky(gap_val)
                else "The model may have memorised examples, see below."
            )
        st.markdown(
            f'<div class="run-banner">{verdict.held_up} of {verdict.audited} '
            f'finding{"s" if verdict.audited != 1 else ""} held up.'
            + "".join(f'<span class="sub">{html.escape(s)}</span>' for s in sub_lines)
            + "</div>",
            unsafe_allow_html=True,
        )

    # 3. Evidence: the findings and their check rows.
    st.markdown("#### What we found")
    if view.headline_findings:
        dash_finding_ids: set[str] = {
            str(c["finding_id"]) for c in (dash or []) if c.get("finding_id")
        }
        for f in view.headline_findings:
            st.markdown(render_finding_card(f, dash_finding_ids), unsafe_allow_html=True)
    else:
        # No cards to lead with: promote whatever analysis actually ran. With
        # cards present this line would only repeat the first one.
        dominant = None
        if not train_out:
            dominant = (
                find_tool(tool_results, "cluster_data")
                or find_tool(tool_results, "time_series_analysis")
                or find_tool(tool_results, "geospatial_analysis")
                or find_tool(tool_results, "text_analysis")
                or find_tool(tool_results, "dimensionality_analysis")
            )
        has_dominant = bool(dominant and dominant.get("summary"))
        if dominant and has_dominant:
            st.info(f"**What we found:** {dominant['summary']}")
        if view.findings:
            st.caption(
                "No headline-worthy findings cleared the bar for this run. "
                "See the Details tab for the full finding list."
            )
        elif view.insights:
            for ins in view.insights:
                st.markdown(
                    f'<div class="ic"><span class="mk">Insight</span>{html.escape(plainify(ins))}</div>',
                    unsafe_allow_html=True,
                )
        elif not has_dominant:
            st.caption("No explicit statistical discoveries recorded.")

    # 4. Caveats, in one always-visible block. The model stamp is never hidden.
    unanswered = view.coverage.get("unanswered") or []
    small_notes = list(
        dict.fromkeys(
            str(c)
            for f in view.findings
            for c in f.get("caveats") or []
            if "to avoid identifying individuals" in str(c)
        )
    )
    du_html = render_data_understanding(report.get("data_understanding") or {})
    if gap_val is not None or unanswered or small_notes or du_html:
        st.markdown("#### Be careful about")
        if gap_val is not None:
            st.markdown(render_defect_stamp(gap_val), unsafe_allow_html=True)
        if unanswered:
            st.caption(f"⚠ {len(unanswered)} question(s) considered, not answered.")
            with st.expander("What wasn't answered, and why"):
                for u in unanswered:
                    st.markdown(f"- {md_text(u.get('text', ''))}")
        if small_notes:
            st.caption("Small groups combined. " + " ".join(small_notes[:3]))
        if du_html:
            st.markdown("##### How we read your data")
            st.markdown(du_html, unsafe_allow_html=True)

    # 5. Next action.
    st.markdown("#### What to do")
    if view.recommendations:
        for rec in view.recommendations:
            st.markdown(
                f'<div class="rc"><span class="mk">Do</span>{html.escape(plainify(rec))}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No operational recommendations generated.")

    # 6. Supporting context.
    st.markdown("#### At a glance")
    row_count = (
        (meta.row_count if meta else None)
        or (prof.get("row_count") if prof else None)
        or (len(preview_df) if preview_df is not None else None)
    )
    caveat_count = sum(
        1 for f in view.findings if f.get("kind") in ("method_fit", "coverage_gap")
    )
    trust_cells = [
        ("Data quality", f"{q}/100"),
        ("Rows analyzed", f"{row_count:,}" if isinstance(row_count, int) else "N/A"),
        ("Caveats flagged", str(caveat_count)),
    ]
    st.markdown(
        '<div class="trust-strip">'
        + "".join(
            f'<div class="trust-cell"><div class="k">{html.escape(k)}</div>'
            f'<div class="v">{html.escape(v)}</div></div>'
            for k, v in trust_cells
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    render_run_compare()

    # Search these findings: a keyword search over headlines, not free-form Q&A.
    st.markdown("#### Search these findings")
    ask_q = st.text_input(
        "Search these findings",
        placeholder="e.g. tenure churn",
        label_visibility="collapsed",
        key="ask_data_q",
    )
    if ask_q.strip():
        matches = search_findings(ask_q, list(view.findings))
        if matches:
            for m in matches:
                st.success(f"Based on what was found: {plainify(str(m.get('headline', '')))}")
        else:
            st.info(
                "Nothing in this analysis directly answers that. Try rephrasing, "
                "or check the Details tab for full coverage."
            )

    # Technical detail expander
    with st.expander("Show technical detail"):
        radial_color = (
            "var(--positive)"
            if q >= 80
            else ("var(--accent)" if q >= 60 else "var(--risk)")
        )
        gap_flag = gap_val is not None and gap_is_risky(gap_val)
        st.markdown(
            f"""<div class="kpi-row">
<div class="kpi-gauge-card">
<div class="kpi-gauge-label">Data Quality</div>
<div class="kpi-ring-wrap">
<svg width="140" height="140" viewBox="0 0 140 140" style="position: absolute; top: 0; left: 0; transform: rotate(-90deg); overflow: visible; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));">
<circle cx="70" cy="70" r="56" fill="none" stroke="var(--rule-faint)" stroke-width="12" />
<circle cx="70" cy="70" r="56" fill="none" stroke="{radial_color}" stroke-width="12" stroke-linecap="round" stroke-dasharray="351.86" stroke-dashoffset="{351.86 - (351.86 * q / 100)}" class="anime-gauge" data-q="{q}" />
</svg>
<div class="kpi-ring-value">
<div class="kpi-ring-num count-up" data-value="{q}" data-suffix="">{q}</div>
<div class="kpi-ring-sub">Score</div>
</div>
</div>
</div>
<div class="kpi-tiles">
<div class="kpi-tile">
<div class="k">Best Model</div>
<div class="v">{html.escape(best_model)}</div>
<div class="s">Task: {html.escape(str(task_type))}</div>
</div>
<div class="kpi-tile">
<div class="k">CV Score</div>
<div class="v big count-up" data-value="{best_cv}" data-suffix="%">{best_cv}%</div>
</div>
<div class="kpi-tile{' flagged' if gap_flag else ''}">
<div class="k">Train-Test Gap</div>
<div class="v big{' risk' if gap_flag else ''} count-up" data-value="{best_gap_str}" data-suffix="%">{best_gap_str}%</div>
</div>
</div>
</div>""",
            unsafe_allow_html=True,
        )

        for w in train_out.get("overfit_warnings", []) if train_out else []:
            st.markdown(
                f'<div class="wc"><span class="mk">Risk</span>{html.escape(str(w))}</div>',
                unsafe_allow_html=True,
            )

        mc_chart = find_chart_by_id(dash, "model_comparison")
        if mc_chart:
            st.markdown("##### How each model scored (Train vs Test vs CV)")
            render_dashboard_chart(mc_chart, vega_cfg)
        elif train_out:
            st.caption("Model comparison chart not available for this run.")

        tc_chart = find_chart_by_id(dash, "top_correlations")
        if tc_chart:
            st.markdown("##### Strongest feature correlations")
            render_dashboard_chart(tc_chart, vega_cfg)
        elif corr_out:
            st.caption("Correlation chart not available for this run.")
