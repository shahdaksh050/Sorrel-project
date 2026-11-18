"""
Actions Tab (Prescriptive Action Engine & 2x2 Prioritization Matrix).

Categorizes findings into operational quadrants (Quick Wins, Strategic Bets,
Pilot / A/B Tests, Incremental Tuning, and Guardrails) and displays
interactive what-if scenario levers.
"""
from __future__ import annotations

import html
from typing import Any

import streamlit as st

from src.core.findings import classify_action_matrix
from src.core.plain_language import plainify
from ui.components.cards import find_tool, tab_head


def render_actions_tab(
    report: dict[str, Any],
    tool_results: list[dict[str, Any]],
    run_view: Any | None = None,
) -> None:
    """Render Tier 3: Prescriptive Action Engine."""
    tab_head("Actions", "What to do next, by impact and", "feasibility")

    # 1. 2x2 Action Prioritization Matrix
    findings = list(run_view.findings) if run_view and hasattr(run_view, "findings") else (report.get("findings") or [])
    matrix = classify_action_matrix(findings)

    qw_items = matrix.get("quick_wins", [])
    sb_items = matrix.get("strategic_bets", [])
    pilot_items = matrix.get("pilot", [])
    inc_items = matrix.get("incremental", [])
    watch_items = matrix.get("watch_caveats", [])

    total_actionable = len(qw_items) + len(sb_items) + len(pilot_items) + len(inc_items)
    top_lever = qw_items[0].get("lever") if qw_items else (pilot_items[0].get("lever") if pilot_items else (sb_items[0].get("lever") if sb_items else "Review audit"))

    # Top Executive Action Ribbon
    st.markdown(
        f"""<div class="trust-strip" style="margin-bottom:1.25rem;">
<div class="trust-cell"><div class="k">Actionable Levers</div><div class="v">{total_actionable} Levers</div></div>
<div class="trust-cell"><div class="k">Primary Execution Candidate</div><div class="v">{html.escape(str(top_lever)[:36])}</div></div>
<div class="trust-cell"><div class="k">Validation Pilots</div><div class="v">{len(pilot_items)} Recommended</div></div>
<div class="trust-cell"><div class="k">Monitored Risk Guardrails</div><div class="v">{len(watch_items)} Boundaries</div></div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### Action Prioritization Matrix (2x2)")
    st.caption("Strategic levers categorized by business impact and operational execution feasibility.")

    # 2x2 Grid Layout: Row 1 = High Impact, Row 2 = Moderate / Empirical Impact
    quad_defs = [
        # Row 1: High Impact
        [
            ("quick_wins", "⚡ Quick Wins", "High Impact · High Feasibility", "var(--pen)", qw_items, "Immediate Execution Candidate"),
            ("strategic_bets", "🎯 Strategic Bets", "High Impact · Structural / Complex", "var(--accent)", sb_items, "Requires Cross-Functional Alignment"),
        ],
        # Row 2: Moderate Impact / Empirical Validation
        [
            ("pilot", "🧪 Pilot / A/B Tests", "High Opportunity · Test Before Scaling", "var(--ink-muted)", pilot_items, "Validate Via A/B Trial First"),
            ("incremental", "🔧 Incremental Gains", "Moderate Impact · High Feasibility", "var(--ink-faint)", inc_items, "Routine Operational Tuning"),
        ],
    ]

    for row_quads in quad_defs:
        cols = st.columns(2, gap="medium")
        for c_idx, (_, title, subtitle, accent_color, items, badge_text) in enumerate(row_quads):
            with cols[c_idx]:
                with st.container(border=True):
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.15rem;">'
                        f'<div style="font-weight:700;font-size:1.02rem;color:{accent_color};">{html.escape(title)}</div>'
                        f'<span class="mono-badge" style="font-size:0.7rem;padding:0.15rem 0.45rem;">{html.escape(badge_text)}</span>'
                        f'</div>'
                        f'<div style="font-size:0.75rem;color:var(--ink-muted);margin-bottom:0.65rem;text-transform:uppercase;letter-spacing:0.04em;">{html.escape(subtitle)}</div>',
                        unsafe_allow_html=True,
                    )
                    if items:
                        for it in items[:3]:
                            lever = html.escape(str(it.get("lever", it.get("headline", ""))))
                            rationale = html.escape(str(it.get("rationale", "")))
                            eff = it.get("effect")
                            eff_badge = f' <span style="font-size:0.8rem;color:var(--pen);font-weight:700;">({eff:+0.1%})</span>' if isinstance(eff, (int, float)) and abs(eff) < 10 else ""
                            samp = it.get("sample_size")
                            samp_str = f' <span style="font-size:0.75rem;color:var(--ink-faint);">(n={samp:,})</span>' if isinstance(samp, (int, float)) else ""

                            st.markdown(
                                f'<div style="margin-bottom:0.65rem;border-left:3px solid {accent_color};padding-left:0.6rem;">'
                                f'<div style="font-weight:600;font-size:0.88rem;line-height:1.35;">{lever}{eff_badge}{samp_str}</div>'
                                f'<div style="font-size:0.8rem;color:var(--ink-muted);margin-top:0.2rem;line-height:1.4;">{rationale}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            '<div style="font-size:0.82rem;color:var(--ink-faint);font-style:italic;padding:0.4rem 0;">'
                            'No findings currently meet the threshold for this quadrant.'
                            '</div>',
                            unsafe_allow_html=True,
                        )

    # 2. Risk Management & Guardrails Section
    if watch_items:
        st.markdown("")
        with st.container(border=True):
            st.markdown(
                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;">'
                '<div style="font-weight:700;font-size:1.02rem;color:var(--risk);">🛡️ Operational Risk Guardrails & Monitored Covariates</div>'
                '<span class="mono-badge" style="font-size:0.7rem;padding:0.15rem 0.45rem;">RISK MANAGEMENT</span>'
                '</div>'
                '<div style="font-size:0.78rem;color:var(--ink-muted);margin-bottom:0.65rem;">'
                'External covariates, non-controllable factors, and sensitivity boundaries to monitor in operational governance.'
                '</div>',
                unsafe_allow_html=True,
            )
            w_cols = st.columns(min(len(watch_items), 2), gap="medium")
            for w_idx, it in enumerate(watch_items[:4]):
                target_col = w_cols[w_idx % len(w_cols)]
                with target_col:
                    lever = html.escape(str(it.get("lever", it.get("headline", ""))))
                    rationale = html.escape(str(it.get("rationale", "")))
                    st.markdown(
                        f'<div style="margin-bottom:0.5rem;border-left:2px solid var(--risk);padding-left:0.5rem;">'
                        f'<div style="font-weight:600;font-size:0.85rem;">{lever}</div>'
                        f'<div style="font-size:0.78rem;color:var(--ink-muted);margin-top:0.15rem;">{rationale}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # 3. What-If Counterfactual Scenario Simulations
    scenario_out = find_tool(tool_results, "scenario_simulation")
    if scenario_out and isinstance(scenario_out, dict):
        st.divider()
        st.markdown("### What-If Counterfactual Interventions")
        target_name = scenario_out.get("target_column") or "Target"
        scenarios = scenario_out.get("scenarios") or []
        if scenarios:
            st.caption(f"Predicted response of {html.escape(str(target_name))} under targeted driver modifications.")
            for sc in scenarios[:3]:
                with st.container(border=True):
                    sc_name = html.escape(str(sc.get("scenario_name") or sc.get("driver") or "Scenario"))
                    pred_delta = sc.get("predicted_delta")
                    base_val = sc.get("base_value")
                    new_val = sc.get("counterfactual_value")
                    delta_str = f"{pred_delta:+0.2f}" if isinstance(pred_delta, (int, float)) else str(pred_delta)
                    base_str = f"{base_val:0.2f}" if isinstance(base_val, (int, float)) else str(base_val)
                    new_str = f"{new_val:0.2f}" if isinstance(new_val, (int, float)) else str(new_val)

                    c1, c2, c3 = st.columns([0.45, 0.25, 0.3])
                    c1.markdown(f"**{sc_name}**")
                    c2.markdown(f"Baseline: `{base_str}` → Simulated: `{new_str}`")
                    c3.markdown(f"Net Shift: **{delta_str}**")

    # 4. Operational Directives Roadmap
    st.divider()
    st.markdown("### Operational Directives")
    recs = (run_view.recommendations if run_view and hasattr(run_view, "recommendations") else None) or report.get("recommendations") or []
    if recs:
        for idx, rec in enumerate(recs, start=1):
            rec_plain = plainify(str(rec))
            rec_lower = rec_plain.lower()
            if any(rec_lower.startswith(p) for p in ("pilot", "test", "validate", "evaluate")):
                badge = "Test"
            elif any(rec_lower.startswith(p) for p in ("monitor", "watch", "track")):
                badge = "Watch"
            elif any(rec_lower.startswith(p) for p in ("investigate", "explore", "research")):
                badge = "Explore"
            else:
                badge = "Do"
            st.markdown(
                f'<div class="rc"><span class="mk">{badge}</span><b>{idx}.</b> {html.escape(rec_plain)}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No operational directives recorded for this run.")

