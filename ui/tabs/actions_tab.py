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

    quadrants = [
        ("quick_wins", "⚡ Quick Wins", "High Impact · High Feasibility", "var(--pen)"),
        ("strategic_bets", "🎯 Strategic Bets", "High Impact · Structural / Complex", "var(--accent)"),
        ("pilot", "🧪 Pilot / A/B Tests", "High Opportunity · Test Before Scaling", "var(--ink-muted)"),
        ("incremental", "🔧 Incremental Gains", "Moderate Impact · High Feasibility", "var(--ink-faint)"),
        ("watch_caveats", "⚠️ Guardrails & Watch", "Confounders / Sensitive / Fragile", "var(--risk)"),
    ]

    has_matrix_items = any(matrix.get(k) for k, _, _, _ in quadrants)

    if has_matrix_items:
        st.markdown("### Action Prioritization Matrix (2x2)")
        st.caption("Strategic levers prioritized by statistical impact and operational implementation feasibility.")

        cols = st.columns(2, gap="medium")
        for idx, (key, title, subtitle, accent_color) in enumerate(quadrants):
            items = matrix.get(key, [])
            if not items:
                continue
            target_col = cols[idx % 2]
            with target_col:
                with st.container(border=True):
                    st.markdown(
                        f'<div style="font-weight:700;font-size:1.05rem;color:{accent_color};margin-bottom:0.15rem;">{html.escape(title)}</div>'
                        f'<div style="font-size:0.75rem;color:var(--ink-muted);margin-bottom:0.6rem;text-transform:uppercase;letter-spacing:0.04em;">{html.escape(subtitle)}</div>',
                        unsafe_allow_html=True,
                    )
                    for it in items[:4]:
                        lever = html.escape(str(it.get("lever", it.get("headline", ""))))
                        rationale = html.escape(str(it.get("rationale", "")))
                        eff = it.get("effect")
                        eff_badge = f' <span style="font-size:0.8rem;color:var(--pen);font-weight:600;">({eff:+0.1%})</span>' if isinstance(eff, (int, float)) and abs(eff) < 10 else ""
                        st.markdown(
                            f'<div style="margin-bottom:0.6rem;border-left:2px solid {accent_color};padding-left:0.5rem;">'
                            f'<div style="font-weight:600;font-size:0.9rem;">{lever}{eff_badge}</div>'
                            f'<div style="font-size:0.82rem;color:var(--ink-muted);margin-top:0.15rem;">{rationale}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
    else:
        st.caption("No multivariate levers met the threshold for the 2x2 prioritization matrix.")

    # 2. What-If Scenario Simulations
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

    # 3. Operational Directives
    st.divider()
    st.markdown("### Operational Directives")
    recs = (run_view.recommendations if run_view and hasattr(run_view, "recommendations") else None) or report.get("recommendations") or []
    if recs:
        for rec in recs:
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
                f'<div class="rc"><span class="mk">{badge}</span>{html.escape(rec_plain)}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No operational directives recorded for this run.")
