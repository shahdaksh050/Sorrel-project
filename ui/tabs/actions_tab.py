"""
Actions Tab (Prescriptive Action Engine & 2x2 Prioritization Matrix).

Categorizes findings into operational quadrants (Quick Wins, Strategic Bets,
Pilot / A/B Tests, Incremental Tuning, and Guardrails) and displays
interactive what-if scenario levers adapted to the dataset's archetype.
"""
from __future__ import annotations

import html
from typing import Any

import streamlit as st

from src.core.findings import matrix_recommendations, report_action_matrix
from src.core.plain_language import plainify
from ui.components.cards import find_tool, round_for_reading, tab_head


def get_archetype_action_config(archetype: str | None) -> dict[str, Any]:
    """Return archetype-specific terminology, titles, and quadrant configurations."""
    arch = (archetype or "").lower().replace("-", "_").replace(" ", "_")

    # 1. Sensor / Continuous / IoT / Physical / Energy
    if any(k in arch for k in ("continuous", "sensor", "energy", "iot", "machinery", "physical")):
        return {
            "archetype_cluster": "sensor_iot",
            "tab_title": ("Actions", "Control levers and priority interventions, by impact and", "feasibility"),
            "caption": "Empirical control points, operational calibrations, and system guardrails prioritized by magnitude and feasibility.",
            "ribbon": {
                "levers": "Control Levers",
                "candidate": "Primary Control Point",
                "pilots": "Experimental Calibrations",
                "guardrails": "System Baselines & Guardrails",
            },
            "quadrants": [
                [
                    ("quick_wins", "⚡ Immediate Control Levers", "High Impact · High Controllability", "var(--pen)", "Active Control Candidate"),
                    ("strategic_bets", "🎯 Structural System Drivers", "High Impact · Systemic / Multi-Factor", "var(--accent)", "System-Level Intervention"),
                ],
                [
                    ("pilot", "🧪 Experimental Field Trials", "High Opportunity · Calibrate & Validate", "var(--graphite)", "Field Calibration First"),
                    ("incremental", "🔧 Routine Parameter Tuning", "Moderate Impact · Readily Tunable", "var(--rule-strong)", "Parameter Calibration"),
                ],
            ],
            "guardrails": {
                "title": "🛡️ Physical Telemetry, Sensor Covariates & Baselines",
                "badge": "SYSTEM GUARDRAILS",
                "subtitle": "Physical instrumentation limits, ambient environmental covariates, and baseline noise boundaries to monitor.",
            },
            "directives_title": "Engineering & Operational Directives",
        }

    # 2. Longitudinal Panel / Geospatial / Multi-Entity Tracking (e.g. city_hour.csv, regions, plants)
    if any(k in arch for k in ("panel", "geospatial", "spatial", "fleet", "hierarchical")):
        return {
            "archetype_cluster": "longitudinal_panel",
            "tab_title": ("Actions", "Targeted interventions and cohort strategies, by impact and", "feasibility"),
            "caption": "Cross-entity levers, priority regimes, and monitoring guardrails categorized by impact and intervention feasibility.",
            "ribbon": {
                "levers": "Actionable Levers",
                "candidate": "Primary Priority Target",
                "pilots": "Targeted Pilot Regimes",
                "guardrails": "Monitored Covariates & Boundaries",
            },
            "quadrants": [
                [
                    ("quick_wins", "⚡ High-Priority Regimes", "High Impact · Actionable Cohort", "var(--pen)", "Immediate Priority Target"),
                    ("strategic_bets", "🎯 Structural Macro Drivers", "High Impact · Cross-Entity / Systemic", "var(--accent)", "Structural Policy Shift"),
                ],
                [
                    ("pilot", "🧪 Targeted Field Pilots", "High Opportunity · Pilot in Target Cohort", "var(--graphite)", "Pilot in Target Cohort First"),
                    ("incremental", "🔧 Incremental Cohort Tuning", "Moderate Impact · Readily Tunable", "var(--rule-strong)", "Localized Tuning"),
                ],
            ],
            "guardrails": {
                "title": "🛡️ Contextual Covariates & Observational Boundaries",
                "badge": "COHORT GUARDRAILS",
                "subtitle": "Regional covariates, longitudinal confounders, and sensitivity boundaries to monitor across entities.",
            },
            "directives_title": "Strategic & Targeted Directives",
        }

    # 3. Scientific / Clinical / Healthcare / Survey / Psychology
    if any(k in arch for k in ("survey", "cross_sectional", "cross_section", "survival", "clinical", "scientific", "medical", "trial")):
        return {
            "archetype_cluster": "scientific_clinical",
            "tab_title": ("Actions", "Empirical drivers and recommended protocols, by impact and", "feasibility"),
            "caption": "Empirical drivers, priority interventions, and research guardrails categorized by evidence strength and impact.",
            "ribbon": {
                "levers": "Key Action Levers",
                "candidate": "Primary Focus Area",
                "pilots": "Confirmatory Studies",
                "guardrails": "Confounders & Observational Limits",
            },
            "quadrants": [
                [
                    ("quick_wins", "⚡ Core Action Levers", "High Impact · High Confidence", "var(--pen)", "Direct Implementation"),
                    ("strategic_bets", "🎯 Strategic Determinants", "High Impact · Structural Determinants", "var(--accent)", "Strategic Initiative"),
                ],
                [
                    ("pilot", "🧪 Confirmatory Studies", "High Opportunity · Prospective Validation", "var(--graphite)", "Confirmatory Study First"),
                    ("incremental", "🔧 Incremental Enhancements", "Moderate Impact · High Feasibility", "var(--rule-strong)", "Targeted Optimization"),
                ],
            ],
            "guardrails": {
                "title": "🛡️ Observational Confounders & Boundary Conditions",
                "badge": "STUDY GUARDRAILS",
                "subtitle": "Unmeasured confounders, sampling caveats, and observational boundaries to account for in decision-making.",
            },
            "directives_title": "Research & Implementation Directives",
        }

    # 4. Commercial / Business / E-commerce / Retail
    if any(k in arch for k in ("transaction", "basket", "rfm", "retail", "ecommerce", "saas", "subscription", "sales", "churn")):
        return {
            "archetype_cluster": "commercial",
            "tab_title": ("Actions", "What to do next, by impact and", "feasibility"),
            "caption": "Strategic levers categorized by business impact and operational execution feasibility.",
            "ribbon": {
                "levers": "Actionable Levers",
                "candidate": "Primary Execution Candidate",
                "pilots": "Validation Pilots",
                "guardrails": "Monitored Risk Guardrails",
            },
            "quadrants": [
                [
                    ("quick_wins", "⚡ Quick Wins", "High Impact · High Feasibility", "var(--pen)", "Immediate Execution Candidate"),
                    ("strategic_bets", "🎯 Strategic Bets", "High Impact · Structural / Complex", "var(--accent)", "Requires Cross-Functional Alignment"),
                ],
                [
                    ("pilot", "🧪 Pilot / A/B Tests", "High Opportunity · Test Before Scaling", "var(--graphite)", "Validate Via A/B Trial First"),
                    ("incremental", "🔧 Incremental Gains", "Moderate Impact · High Feasibility", "var(--rule-strong)", "Routine Operational Tuning"),
                ],
            ],
            "guardrails": {
                "title": "🛡️ Operational Risk Guardrails & Monitored Covariates",
                "badge": "RISK MANAGEMENT",
                "subtitle": "External market factors, non-controllable covariates, and risk boundaries to monitor in operational governance.",
            },
            "directives_title": "Operational Directives",
        }

    # 5. General Analytical Default
    return {
        "archetype_cluster": "general",
        "tab_title": ("Actions", "Actionable insights and priorities, by impact and", "feasibility"),
        "caption": "Prioritized findings and actionable levers categorized by analytical impact and operational feasibility.",
        "ribbon": {
            "levers": "Actionable Insights",
            "candidate": "Primary Action Target",
            "pilots": "Targeted Trials / Pilots",
            "guardrails": "Contextual Guardrails",
        },
        "quadrants": [
            [
                ("quick_wins", "⚡ High-Impact Priorities", "High Impact · Readily Actionable", "var(--pen)", "Immediate Priority"),
                ("strategic_bets", "🎯 Strategic Drivers", "High Impact · Systemic / Structural", "var(--accent)", "Structural Focus"),
            ],
            [
                ("pilot", "🧪 Pilot & Validation Targets", "High Opportunity · Validate Before Scaling", "var(--graphite)", "Validate First"),
                ("incremental", "🔧 Incremental Optimizations", "Moderate Impact · High Feasibility", "var(--rule-strong)", "Routine Tuning"),
            ],
        ],
        "guardrails": {
            "title": "🛡️ Contextual Covariates & Risk Guardrails",
            "badge": "GUARDRAILS",
            "subtitle": "External covariates, non-controllable factors, and sensitivity boundaries to monitor.",
        },
        "directives_title": "Actionable Directives",
    }


def format_archetype_lever_and_rationale(
    lever: str,
    rationale: str,
    cluster: str,
) -> tuple[str, str]:
    """Adapt lever and rationale phrasing to match the domain archetype."""
    out_lever = lever
    out_rat = rationale

    if cluster in ("longitudinal_panel", "panel"):
        out_lever = out_lever.replace("Pilot A/B test: Target ", "Targeted Pilot: Priority Cohort ")
        out_lever = out_lever.replace("Pilot validation: Target ", "Targeted Pilot: Priority Cohort ")
        out_lever = out_lever.replace("Pilot A/B test: ", "Targeted Pilot: ")
        out_lever = out_lever.replace("Pilot validation: ", "Targeted Pilot: ")
        out_lever = out_lever.replace("Target ", "Priority Cohort: ")
        out_rat = out_rat.replace("prospective A/B test or pilot before scaling", "targeted field pilot or localized cohort trial")
        out_rat = out_rat.replace("before changing downstream policy", "before enacting broad cohort interventions")
        out_rat = out_rat.replace("before changing policy", "before enacting broad cohort interventions")
        if out_rat.strip().startswith("Counts are per"):
            out_rat = f"Cohort-level aggregated signal — {out_rat}"
    elif cluster in ("sensor_iot", "continuous_series"):
        out_lever = out_lever.replace("Pilot A/B test: Target ", "Experimental Calibration: Regime ")
        out_lever = out_lever.replace("Pilot validation: Target ", "Experimental Calibration: Regime ")
        out_lever = out_lever.replace("Pilot A/B test: ", "Experimental Calibration: ")
        out_lever = out_lever.replace("Pilot validation: ", "Experimental Calibration: ")
        out_lever = out_lever.replace("Target ", "Control Regime: ")
        out_rat = out_rat.replace("prospective A/B test or pilot before scaling", "controlled field calibration or threshold testing")
        out_rat = out_rat.replace("before changing downstream policy", "before adjusting operational thresholds")
        out_rat = out_rat.replace("before changing policy", "before adjusting operational thresholds")
    elif cluster in ("scientific_clinical", "survey"):
        out_lever = out_lever.replace("Pilot A/B test: Target ", "Confirmatory Study: Subgroup ")
        out_lever = out_lever.replace("Pilot validation: Target ", "Confirmatory Study: Subgroup ")
        out_lever = out_lever.replace("Pilot A/B test: ", "Confirmatory Study: ")
        out_lever = out_lever.replace("Pilot validation: ", "Confirmatory Study: ")
        out_lever = out_lever.replace("Target ", "Key Factor: ")
        out_rat = out_rat.replace("prospective A/B test or pilot before scaling", "prospective confirmatory study or trial")
        out_rat = out_rat.replace("before changing downstream policy", "before drawing scientific conclusions")
        out_rat = out_rat.replace("before changing policy", "before drawing clinical or scientific conclusions")

    return out_lever, out_rat


def render_actions_tab(
    report: dict[str, Any],
    tool_results: list[dict[str, Any]],
    run_view: Any | None = None,
    archetype: str | None = None,
) -> None:
    """Render Tier 3: Prescriptive Action Engine (Universal Archetype-Native)."""
    # Resolve data archetype from run_view, report, profile, or preview_df
    prof_obj = getattr(st, "session_state", {}).get("profile") if hasattr(st, "session_state") else None
    resolved_arch = (
        archetype
        or (report.get("understanding") or {}).get("archetype")
        or report.get("archetype")
        or report.get("data_archetype")
        or getattr(run_view, "archetype", None)
        or (getattr(prof_obj, "archetype", None) if prof_obj else None)
    )
    if not resolved_arch and hasattr(st, "session_state"):
        preview = st.session_state.get("preview_df")
        if preview is not None and hasattr(preview, "columns"):
            cols_lower = [str(c).lower() for c in preview.columns]
            if any(k in cols_lower for k in ("pm2.5", "pm10", "aqi", "aqi_bucket", "no2", "so2", "co")):
                resolved_arch = "panel"
            elif any(k in cols_lower for k in ("sub_metering_1", "active_power", "global_active_power", "voltage")):
                resolved_arch = "continuous_series"

    cfg = get_archetype_action_config(resolved_arch)

    tab_head(cfg["tab_title"][0], cfg["tab_title"][1], cfg["tab_title"][2])

    # 1. 2x2 Action Prioritization Matrix
    findings = list(run_view.findings) if run_view and hasattr(run_view, "findings") else (report.get("findings") or [])
    matrix = report_action_matrix(report, findings)

    qw_items = matrix.get("quick_wins", [])
    sb_items = matrix.get("strategic_bets", [])
    pilot_items = matrix.get("pilot", [])
    inc_items = matrix.get("incremental", [])
    watch_items = matrix.get("watch_caveats", [])

    total_actionable = len(qw_items) + len(sb_items) + len(pilot_items) + len(inc_items)
    raw_top_lever = qw_items[0].get("lever") if qw_items else (pilot_items[0].get("lever") if pilot_items else (sb_items[0].get("lever") if sb_items else "Review audit"))
    top_lever = round_for_reading(plainify(str(raw_top_lever)))

    # Top Executive Action Ribbon
    rib = cfg["ribbon"]
    st.markdown(
        f"""<div class="trust-strip" style="margin-bottom:1.25rem;">
<div class="trust-cell"><div class="k">{html.escape(rib["levers"])}</div><div class="v">{total_actionable} Levers</div></div>
<div class="trust-cell"><div class="k">{html.escape(rib["candidate"])}</div><div class="v">{html.escape(top_lever[:42])}</div></div>
<div class="trust-cell"><div class="k">{html.escape(rib["pilots"])}</div><div class="v">{len(pilot_items)} Recommended</div></div>
<div class="trust-cell"><div class="k">{html.escape(rib["guardrails"])}</div><div class="v">{len(watch_items)} Boundaries</div></div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### Action Prioritization Matrix (2x2)")
    st.caption(cfg["caption"])

    # 2x2 Grid Layout
    quad_defs = cfg["quadrants"]

    for row_quads in quad_defs:
        cols = st.columns(2, gap="medium")
        for c_idx, (q_key, title, subtitle, accent_color, badge_text) in enumerate(row_quads):
            items = matrix.get(q_key, [])
            with cols[c_idx]:
                with st.container(border=True):
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.15rem;">'
                        f'<div style="font-weight:700;font-size:1.02rem;color:{accent_color};">{html.escape(title)}</div>'
                        f'<span class="mono-badge" style="font-size:0.7rem;padding:0.15rem 0.45rem;">{html.escape(badge_text)}</span>'
                        f'</div>'
                        f'<div style="font-size:0.75rem;color:var(--graphite);margin-bottom:0.65rem;text-transform:uppercase;letter-spacing:0.04em;">{html.escape(subtitle)}</div>',
                        unsafe_allow_html=True,
                    )
                    if items:
                        for it in items[:3]:
                            lever = round_for_reading(plainify(str(it.get("lever", it.get("headline", "")))))
                            rationale = round_for_reading(plainify(str(it.get("rationale", ""))))
                            samp = it.get("sample_size")
                            samp_str = f' <span style="font-size:0.75rem;color:var(--rule-strong);">(n={samp:,})</span>' if isinstance(samp, (int, float)) else ""

                            st.markdown(
                                f'<div style="margin-bottom:0.65rem;border-left:3px solid {accent_color};padding-left:0.6rem;">'
                                f'<div style="font-weight:600;font-size:0.88rem;line-height:1.35;">{html.escape(lever)}{samp_str}</div>'
                                f'<div style="font-size:0.8rem;color:var(--graphite);margin-top:0.2rem;line-height:1.4;">{html.escape(rationale)}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            '<div style="font-size:0.82rem;color:var(--graphite);font-style:italic;padding:0.4rem 0;">'
                            'No findings currently meet the threshold for this quadrant.'
                            '</div>',
                            unsafe_allow_html=True,
                        )

    # 2. Risk Management & Guardrails Section
    g_cfg = cfg["guardrails"]
    if watch_items:
        st.markdown("")
        with st.container(border=True):
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;">'
                f'<div style="font-weight:700;font-size:1.02rem;color:var(--risk);">{html.escape(g_cfg["title"])}</div>'
                f'<span class="mono-badge" style="font-size:0.7rem;padding:0.15rem 0.45rem;">{html.escape(g_cfg["badge"])}</span>'
                f'</div>'
                f'<div style="font-size:0.78rem;color:var(--ink-muted);margin-bottom:0.65rem;">'
                f'{html.escape(g_cfg["subtitle"])}'
                f'</div>',
                unsafe_allow_html=True,
            )
            w_cols = st.columns(min(len(watch_items), 2), gap="medium")
            for w_idx, it in enumerate(watch_items[:4]):
                target_col = w_cols[w_idx % len(w_cols)]
                with target_col:
                    lever = round_for_reading(plainify(str(it.get("lever", it.get("headline", "")))))
                    rationale = round_for_reading(plainify(str(it.get("rationale", ""))))
                    st.markdown(
                        f'<div style="margin-bottom:0.5rem;border-left:2px solid var(--risk);padding-left:0.5rem;">'
                        f'<div style="font-weight:600;font-size:0.85rem;">{html.escape(lever)}</div>'
                        f'<div style="font-size:0.78rem;color:var(--graphite);margin-top:0.15rem;">{html.escape(rationale)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # 3. What-If Counterfactual Scenario Simulations
    scenario_out = find_tool(tool_results, "scenario_simulation")
    if scenario_out and isinstance(scenario_out, dict):
        target_name = scenario_out.get("target_column") or "Target"
        scenarios = scenario_out.get("scenarios") or []
        base_fallback = scenario_out.get("baseline_mean")

        # Resolve valid numeric scenarios only (suppress broken None entries)
        valid_scenarios: list[dict[str, Any]] = []
        for sc in scenarios:
            base_val = sc.get("base_value") if sc.get("base_value") is not None else (
                sc.get("base_mean") if sc.get("base_mean") is not None else base_fallback
            )
            new_val = sc.get("counterfactual_value") if sc.get("counterfactual_value") is not None else sc.get("projected_mean")
            pred_delta = sc.get("predicted_delta") if sc.get("predicted_delta") is not None else (
                sc.get("delta_target") if sc.get("delta_target") is not None else sc.get("predicted_delta_target")
            )
            if pred_delta is None and base_val is not None and new_val is not None:
                pred_delta = new_val - base_val

            if isinstance(base_val, (int, float)) and isinstance(new_val, (int, float)) and isinstance(pred_delta, (int, float)):
                shift = sc.get("shift_pct")
                driver = sc.get("driver") or "Driver"
                name = sc.get("scenario_name") or (f"Simulate {shift:+0.0f}% Shift in '{driver}'" if isinstance(shift, (int, float)) else f"Simulate Shift in '{driver}'")
                valid_scenarios.append({
                    "name": name,
                    "base_val": float(base_val),
                    "new_val": float(new_val),
                    "pred_delta": float(pred_delta),
                })

        if valid_scenarios:
            st.divider()
            st.markdown("### What-If Counterfactual Interventions")
            st.caption(f"Predicted response of {html.escape(str(target_name))} under targeted driver modifications.")
            for sc in valid_scenarios[:3]:
                with st.container(border=True):
                    sc_name = html.escape(sc["name"])
                    base_str = f"{sc['base_val']:0.2f}"
                    new_str = f"{sc['new_val']:0.2f}"
                    delta_str = f"{sc['pred_delta']:+0.2f}"

                    c1, c2, c3 = st.columns([0.45, 0.25, 0.3])
                    c1.markdown(f"**{sc_name}**")
                    c2.markdown(f"Baseline: `{base_str}` → Simulated: `{new_str}`")
                    c3.markdown(f"Net Shift: **{delta_str}**")

    # 4. Directives Roadmap
    st.divider()
    st.markdown(f"### {html.escape(cfg['directives_title'])}")
    recs = (run_view.recommendations if run_view and hasattr(run_view, "recommendations") else None) or report.get("recommendations") or []
    if findings and "action_matrix" not in report:
        recs = [entry["action"] for entry in matrix_recommendations(matrix)]
    if recs:
        for idx, rec in enumerate(recs, start=1):
            rec_plain = plainify(str(rec))
            rec_lower = rec_plain.lower()
            if any(rec_lower.startswith(p) for p in ("remediate", "repair", "fix", "audit", "address")):
                badge = "Repair"
            elif any(rec_lower.startswith(p) for p in ("pilot", "test", "validate", "evaluate")):
                badge = "Test"
            elif any(rec_lower.startswith(p) for p in ("monitor", "watch", "track")):
                badge = "Watch"
            elif any(rec_lower.startswith(p) for p in ("investigate", "explore", "research", "compare", "recheck")):
                badge = "Explore"
            elif any(rec_lower.startswith(p) for p in ("optimize", "tune", "refine", "calibrate")):
                badge = "Tune"
            else:
                badge = "Do"
            st.markdown(
                f'<div class="rc"><span class="mk">{badge}</span><b>{idx}.</b> {html.escape(rec_plain)}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No operational directives recorded for this run.")


