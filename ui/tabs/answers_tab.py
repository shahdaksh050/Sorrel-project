"""
Answers Tab (Executive Briefing & Key Findings).

Reading order: the question and its answer, the verdict, the evidence (finding
cards), what to be careful about, what to do, then supporting context and the
collapsed technical detail. Findings, verdict, coverage and recommendations
come from the `RunView`; `tool_results` is read only for model detail.
"""
from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st

from src.core.audited_entry import GLYPH
from src.core.plain_language import business_headline, plainify
from src.core.run_view import RunView, build_run_view
from ui.components.cards import (
    find_chart_by_id,
    find_tool,
    finding_state,
    gap_is_risky,
    md_text,
    render_dashboard_chart,
    render_data_understanding,
    render_defect_stamp,
    render_evidence_html,
    render_methodology_drawer,
    render_run_compare,
    round_for_reading,
    search_findings,
)


def _debloated_finding_label(finding: dict[str, Any]) -> str:
    """Format a crisp, 1-line label for the left-hand navigation list."""
    st_val = finding_state(finding)
    st_mark = {"held": "✓", "needs_more": "⚠", "unchecked": "○"}.get(st_val, "○")

    head = round_for_reading(business_headline(finding))
    # Remove long trailing statistical parentheticals (e.g. " (difference +2.15, d=1.42, n=68 vs 121)")
    head = re.sub(r"\s*\([^)]*(?:difference|p=|d=|n=|r=|z=|vs|CI)[^)]*\)", "", head).strip(". ")

    if len(head) > 52:
        head = head[:49].rsplit(" ", 1)[0] + "…"
    return f"{st_mark}  {head}"


@st.fragment
def _evidence_inspector(
    findings: list[dict[str, Any]],
    chart_by_finding: dict[str, dict[str, Any]],
    vega_cfg: dict[str, Any],
) -> None:
    """Ranked findings beside the selected one's evidence. A fragment, so choosing a finding does not rerun the page."""
    if not findings:
        return

    n_held = sum(1 for f in findings if finding_state(f) == "held")
    n_needs = sum(1 for f in findings if finding_state(f) == "needs_more")
    n_obs = sum(1 for f in findings if finding_state(f) == "unchecked")

    filter_choices = [f"All ({len(findings)})"]
    if n_held:
        filter_choices.append(f"✓ Held up ({n_held})")
    if n_needs:
        filter_choices.append(f"⚠ Needs data ({n_needs})")
    if n_obs:
        filter_choices.append(f"○ Observations ({n_obs})")

    left, right = st.columns([0.40, 0.60], gap="medium")
    with left:
        active_filter = filter_choices[0]
        if len(filter_choices) > 2:
            sel_filter = st.segmented_control(
                "Filter findings",
                filter_choices,
                default=filter_choices[0],
                key="findings_status_filter",
                label_visibility="collapsed",
            )
            if sel_filter:
                active_filter = sel_filter

        if "Held up" in active_filter:
            subset = [f for f in findings if finding_state(f) == "held"]
        elif "Needs data" in active_filter:
            subset = [f for f in findings if finding_state(f) == "needs_more"]
        elif "Observations" in active_filter:
            subset = [f for f in findings if finding_state(f) == "unchecked"]
        else:
            subset = findings

        if not subset:
            subset = findings

        ids = [str(f.get("finding_id") or f"finding-{i}") for i, f in enumerate(subset)]
        if st.session_state.get("selected_finding") not in ids:
            st.session_state["selected_finding"] = ids[0]

        labels = {
            fid: _debloated_finding_label(f)
            for fid, f in zip(ids, subset, strict=True)
        }

        # Bounded scrollable container when finding count is large (> 4), else compact container
        container_kwargs = {"height": 520, "border": True} if len(findings) > 4 else {"border": True}
        with st.container(**container_kwargs):
            chosen = st.radio(
                "Ranked findings",
                ids,
                format_func=lambda fid: labels[fid],
                key="selected_finding",
                label_visibility="collapsed",
            )

    position = ids.index(chosen)
    finding = subset[position]
    # Each choice is its own card (the key carries the position), so choosing mounts a fresh one and it
    # can settle in from the side the choice came from. The first draw, and a redraw of the same finding,
    # carry no marker and do not animate.
    previous = st.session_state.get("_evidence_position")
    st.session_state["_evidence_position"] = position
    came_from = "" if previous is None or previous == position else "ev-down" if position > previous else "ev-up"
    marker = f'<span class="{came_from}"></span>' if came_from else ""
    with right:
        with st.container(border=True, key=f"evidence_panel_{position}"):
            st.markdown(marker + render_evidence_html(finding, with_figures=False, business_first=True), unsafe_allow_html=True)
            chart = chart_by_finding.get(str(finding.get("finding_id") or ""))
            if chart:
                # Unframed (the pane is the frame) and at the common chart height.
                render_dashboard_chart(chart, vega_cfg, framed=False, height=240, suppress_caption=True)
            st.markdown(render_methodology_drawer(finding), unsafe_allow_html=True)



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
    best_cv = "Unavailable"
    best_gap_str = "Unavailable"
    gap_val: float | None = None

    if train_out:
        mt_map = train_out.get("models_trained", {})
        best = train_out.get("best_model", "")
        if best and best in mt_map:
            bm = mt_map[best]
            if isinstance(bm.get("cv_mean"), (float, int)):
                best_cv = f"{bm['cv_mean']:.3f}"
            gap_val = bm.get("train_test_gap")
            if gap_val is not None:
                best_gap_str = f"{gap_val:.3f}"

    task_type = (
        (train_out.get("task_type") if train_out else None)
        or (meta.task_type if meta else "N/A")
        or "N/A"
    )

    prof = prof or {}
    q = prof.get("quality_score")
    quality_label = f"{q}/100" if isinstance(q, (int, float)) else "Unavailable"
    row_count = meta.row_count if meta is not None else prof.get("row_count")

    # 1. Executive Briefing Header & Verdict
    answer = html.escape(plainify(view.reasoning)) if view.reasoning else ""
    headline_raw = list(view.headline_findings)

    def _finding_sort_key(f: dict[str, Any]) -> tuple[int, float]:
        st_val = finding_state(f)
        state_order = {"held": 0, "needs_more": 1, "unchecked": 2}
        score = float(f.get("importance") or f.get("score") or f.get("confidence") or 0.0)
        return (state_order.get(st_val, 3), -score)

    headline = sorted(headline_raw, key=_finding_sort_key)
    total = len(headline)
    states = [finding_state(f) for f in headline]
    checked = sum(1 for x in states if x != "unchecked")
    held = sum(1 for x in states if x == "held")
    needs_more = sum(1 for x in states if x == "needs_more")
    not_checked = len(states) - checked

    k_tests = len(report.get("statistical_test_pvalues") or report.get("statistical_test_pvalues_bh_corrected") or [])

    sub_chips: list[str] = []
    if k_tests > 0:
        sub_chips.append(f"Benjamini-Hochberg FDR (k={k_tests}, α=0.05)")
    if gap_val is not None and gap_is_risky(gap_val):
        sub_chips.append("⚠ Train-test gap exceeded threshold")
    lift = train_out.get("lift_over_baseline") if train_out else None
    if train_out and isinstance(lift, (float, int)) and not train_out.get("model_training_gated"):
        sub_chips.append(f"CV Lift: {lift:+.3f}")
    if not_checked > 0:
        plural_obs = "s" if not_checked != 1 else ""
        sub_chips.append(f"{not_checked} exploratory observation{plural_obs}")

    chips_html = ""
    if sub_chips:
        chips_html = '<div class="briefing-chips">' + "".join(
            f'<span class="briefing-chip">· {html.escape(c)}</span>' for c in sub_chips
        ) + '</div>'

    verdict_badge = ""
    if checked:
        v_cls = "ok" if needs_more == 0 else "caution"
        v_icon = GLYPH["pass"] if needs_more == 0 else GLYPH["fail"]
        verdict_badge = f'<span class="briefing-verdict {v_cls}">{v_icon} {held}/{checked} Audited Findings Verified</span>'
    elif total:
        verdict_badge = f'<span class="briefing-verdict unchecked">○ {total} Exploratory Findings</span>'

    if view.objective:
        header_title = f'<span class="briefing-tag">QUESTION</span> <span class="briefing-q">"{html.escape(view.objective)}"</span>'
    else:
        header_title = '<span class="briefing-tag">EXECUTIVE SUMMARY</span> <span class="briefing-q">Autonomous Dataset Exploration</span>'

    briefing_body = f'<p class="briefing-body">{answer}</p>' if answer else ""

    st.markdown(
        f"""<div class="briefing-box">
<div class="briefing-header">
    <div class="briefing-title">{header_title}</div>
    {verdict_badge}
</div>
{briefing_body}
{chips_html}
</div>""",
        unsafe_allow_html=True,
    )

    # 2. Evidence: a ranked list on the left, the selected finding with its chart and checks beside it.
    st.markdown("## What we found")
    if headline:
        dash_by_finding = {str(c["finding_id"]): c for c in (dash or []) if c.get("finding_id")}
        _evidence_inspector(headline, dash_by_finding, vega_cfg)
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
    answered_q = view.coverage.get("answered", 0)
    total_q = view.coverage.get("total", 0)
    small_notes = list(
        dict.fromkeys(
            str(c)
            for f in view.findings
            for c in f.get("caveats") or []
            if "to avoid identifying individuals" in str(c)
        )
    )
    du_html = render_data_understanding(report.get("data_understanding") or {})
    if gap_val is not None or unanswered or small_notes or du_html or total_q:
        st.markdown("## Be careful about")
        if gap_val is not None:
            st.markdown(render_defect_stamp(gap_val), unsafe_allow_html=True)
        if total_q > 0:
            st.markdown(
                f"**Business question coverage:** {answered_q} of {total_q} questions answered "
                f"({len(unanswered)} remain open)."
            )
        if unanswered:
            with st.expander(f"What wasn't answered, and why ({len(unanswered)} open)", expanded=False):
                for u in unanswered:
                    reason = u.get("reason") or u.get("blocked_reason") or "Not answered by executed steps"
                    st.markdown(f"- **{md_text(u.get('text', ''))}** — *reason: {reason}*")
        if small_notes:
            st.caption("Small groups combined. " + " ".join(small_notes[:3]))
        if du_html:
            with st.expander("Data Schema & Ingestion Profiling", expanded=False):
                st.markdown(du_html, unsafe_allow_html=True)

    # 4. Next action.
    st.markdown("## What to do")
    if view.recommendations:
        for idx, rec in enumerate(view.recommendations, start=1):
            rec_plain = plainify(rec)
            badge = f"Step {idx}"
            st.markdown(
                f'<div class="rc" style="margin-bottom:0.35rem;padding:9px 13px;">'
                f'<span class="mk">{badge}</span>{html.escape(rec_plain)}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No operational recommendations generated.")

    # 6. Supporting context.
    st.markdown("## At a glance")
    caveat_count = sum(
        1 for f in view.findings if f.get("kind") in ("method_fit", "coverage_gap")
    )
    clean_tool = find_tool(tool_results, "clean_data")
    raw_rows = clean_tool.get("rows_before") if clean_tool else row_count
    analyzed_rows = clean_tool.get("rows_after") if clean_tool else row_count
    trust_cells = [
        ("Data quality", quality_label),
        ("Source rows", f"{raw_rows:,}" if isinstance(raw_rows, int) else "Unavailable"),
        ("Analyzed rows", f"{analyzed_rows:,}" if isinstance(analyzed_rows, int) else "Unavailable"),
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
    st.markdown("## Search these findings")
    st.caption(
        "A keyword search over all the findings from this run. It is not a chat, "
        "and it does not run a new analysis."
    )
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

    # Technical detail expander: the plain answer above leads, the exact figures fold away here.
    with st.expander("Details for analysts"):
        # A low data-quality score is a warning (amber), not a "this may not hold" verdict: the
        # score is printed in the ring, so the colour only adds to the number.
        st.write(f"Data quality: {quality_label}")
        if train_out and task_type in ("classification", "regression"):
            metric = "weighted F1" if task_type == "classification" else "R²"
            st.write(f"Selected model: {best_model} · CV {metric}: {best_cv} · Train–test gap: {best_gap_str}")

        for w in train_out.get("overfit_warnings", []) if train_out else []:
            st.markdown(
                f'<div class="wc"><span class="mk">Risk</span>{html.escape(str(w))}</div>',
                unsafe_allow_html=True,
            )

        mc_chart = find_chart_by_id(dash, "model_comparison")
        if mc_chart:
            st.markdown("### How each model scored (Train vs Test vs CV)")
            render_dashboard_chart(mc_chart, vega_cfg)
        elif train_out:
            st.caption("Model comparison chart not available for this run.")

        tc_chart = find_chart_by_id(dash, "top_correlations")
        if tc_chart:
            st.markdown("### Strongest feature correlations")
            render_dashboard_chart(tc_chart, vega_cfg)
        elif corr_out:
            st.caption("Correlation chart not available for this run.")
