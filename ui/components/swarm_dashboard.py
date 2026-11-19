"""
Swarm Dashboard Animation Component.

Renders an interactive dashboard constructed from a blank slate by a swarm
of 10 autonomous agent cursors with selection marquees, card stamping, and
live Vega-Lite analytical visualization binding.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.core.chart_theme import vega_config
from src.core.json_normalization import strict_json_dumps
from src.core.plain_language import plainify
from ui.components.cards import finding_state, render_dashboard_chart, state_label

SWARM_HTML_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "prototypes" / "swarm_dashboard.html"


def build_swarm_payload(
    dashboard: list[dict[str, Any]] | None = None,
    report: dict[str, Any] | None = None,
    findings: list[dict[str, Any]] | None = None,
    run_view: Any | None = None,
    meta: Any | None = None,
    prof: Any | None = None,
    source_name: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Extract metadata, telemetry, KPIs, Vega-Lite chart specs, model leaderboard,
    action matrix, and audit ledger text for the Swarm Dashboard animation."""
    rep = report or {}
    finds = findings or []

    # Resolve metadata & profile objects from arguments or Streamlit session state
    session = getattr(st, "session_state", {})
    meta_obj = meta or rep.get("metadata") or (session.get("metadata") if isinstance(session, dict) or hasattr(session, "get") else None)
    prof_obj = prof or rep.get("profile") or (session.get("profile") if isinstance(session, dict) or hasattr(session, "get") else None)

    # 1. Source & Header Metadata
    meta_path = getattr(meta_obj, "file_path", None)
    actual_source = (
        source_name
        or getattr(run_view, "source_name", None)
        or (session.get("orig_name") if isinstance(session, dict) or hasattr(session, "get") else None)
        or (session.get("preview_name") if isinstance(session, dict) or hasattr(session, "get") else None)
        or rep.get("source_name")
        or (Path(str(meta_path)).name if meta_path else None)
        or (isinstance(meta_obj, dict) and meta_obj.get("source_name"))
        or "Unavailable"
    )
    actual_run_id = (
        run_id
        or getattr(run_view, "run_id", None)
        or (session.get("run_id") if isinstance(session, dict) or hasattr(session, "get") else None)
        or rep.get("run_id")
        or "Unknown"
    )
    actual_run_id_str = str(actual_run_id)
    if actual_run_id_str.startswith("#"):
        actual_run_id_str = actual_run_id_str[1:]
    if len(actual_run_id_str) > 8:
        actual_run_id = actual_run_id_str[:8]
    else:
        actual_run_id = actual_run_id_str
    title = getattr(run_view, "objective", None) or rep.get("objective") or "Autonomous Executive Briefing"

    headline_finding = finds[0] if finds else {}
    rep_lead = rep.get("lead_narrative") or rep.get("summary")
    if rep_lead and not any(k in str(rep_lead).lower() for k in ("robust sd", "rolling_median", "z-score")):
        lead_narrative = str(rep_lead)
    else:
        from src.core.plain_language import business_headline, business_takeaway
        biz_hl = business_headline(headline_finding) if headline_finding else ""
        biz_takeaway = business_takeaway(headline_finding) if headline_finding else ""
        if biz_hl and biz_takeaway and not biz_takeaway.lower().startswith(biz_hl.lower()):
            lead_narrative = f"{biz_hl}. {biz_takeaway}"
        else:
            lead_narrative = (
                biz_takeaway
                or biz_hl
                or headline_finding.get("headline")
                or ""
            )
    if any(k in lead_narrative.lower() for k in ("robust sd", "rolling_median", "z-score")):
        from src.core.plain_language import business_takeaway
        lead_narrative = business_takeaway(headline_finding) if headline_finding else ""

    states = [finding_state(f) for f in finds]
    checked = sum(state != "unchecked" for state in states)
    held = states.count("held")

    stat_counts = {
        "demonstrated_equivalence": 0,
        "detected_difference": 0,
        "inconclusive": 0,
    }
    for f in finds:
        st_state = f.get("statistical_state") or (f.get("evidence", {}).get("statistical_state") if isinstance(f.get("evidence"), dict) else None)
        if st_state in stat_counts:
            stat_counts[st_state] += 1

    coverage = rep.get("coverage") or rep.get("run_status", {}).get("coverage") or (getattr(run_view, "coverage", None) if hasattr(run_view, "coverage") else {})
    if not isinstance(coverage, dict):
        coverage = {}
    answered_q = coverage.get("answered")
    total_q = coverage.get("total")
    if answered_q is None or total_q is None:
        qa = rep.get("question_agenda") or getattr(run_view, "question_agenda", None)
        if isinstance(qa, dict):
            answered_q = len(qa.get("answered_questions", []))
            total_q = len(qa.get("all_questions", [])) or (answered_q + len(qa.get("open_questions", [])))

    q_str = f" · {answered_q}/{total_q} questions answered" if (total_q is not None and total_q > 0) else ""
    stat_parts = []
    if stat_counts["demonstrated_equivalence"]:
        stat_parts.append(f"{stat_counts['demonstrated_equivalence']} equiv")
    if stat_counts["detected_difference"]:
        stat_parts.append(f"{stat_counts['detected_difference']} diff")
    if stat_counts["inconclusive"]:
        stat_parts.append(f"{stat_counts['inconclusive']} inconclusive")
    stat_str = f" · {', '.join(stat_parts)}" if stat_parts else ""

    headline_badge = f"{len(finds)} findings · {checked} checked{q_str}{stat_str}"


    # 2. Telemetry
    rep_meta = rep.get("metadata")
    rep_meta_dict = rep_meta if isinstance(rep_meta, dict) else {}
    row_count = (
        getattr(meta_obj, "row_count", None)
        or (meta_obj.get("row_count") if isinstance(meta_obj, dict) else None)
        or getattr(prof_obj, "row_count", None)
        or (prof_obj.get("row_count") if isinstance(prof_obj, dict) else None)
        or rep.get("row_count")
        or rep_meta_dict.get("row_count")
    )
    col_count = (
        getattr(meta_obj, "column_count", None)
        or (meta_obj.get("column_count") if isinstance(meta_obj, dict) else None)
        or getattr(prof_obj, "column_count", None)
        or (prof_obj.get("column_count") if isinstance(prof_obj, dict) else None)
        or rep.get("column_count")
        or rep_meta_dict.get("column_count")
    )
    if col_count is None and (isinstance(session, dict) or hasattr(session, "get")):
        preview = session.get("preview_df")
        if preview is not None and hasattr(preview, "columns"):
            col_count = len(preview.columns)

    records_str = f"{row_count:,}" if isinstance(row_count, (int, float)) else "N/A"
    dim_str = f"{col_count} Cols" if isinstance(col_count, (int, float)) else "N/A"
    profile_quality = prof_obj.get("quality_score") if isinstance(prof_obj, dict) else getattr(prof_obj, "quality_score", None)
    quality_score = rep.get("quality_score", profile_quality)
    if quality_score is None:
        quality_score = "Unavailable"
    if isinstance(quality_score, (int, float)):
        quality_score = f"{quality_score:.1f} / 100"
    missingness = rep.get("missingness", "Unavailable")
    if isinstance(missingness, (int, float)):
        missingness = f"{missingness:.2%}"

    kpis: list[dict[str, Any]] = []
    evidence = headline_finding.get("evidence") or {}
    deviation = evidence.get("z") if isinstance(evidence, dict) else None
    if isinstance(deviation, (int, float)):
        kpis.append({"label": "Statistical Deviation", "badge": "Observed",
                     "target": deviation, "prefix": "z = ", "suffix": "",
                     "subtext": headline_finding.get("headline") or ""})
    reversals = sum(
        1 for f in finds if (f.get("evidence") or {}).get("checks", {}).get("paradox_free") is False
    )
    if reversals:
        kpis.append({"label": "Subgroup reversals", "badge": "CAUTION", "target": reversals,
                     "prefix": "", "suffix": "", "subtext": "Direction changes across subgroups."})
    effect = headline_finding.get("effect")
    if isinstance(effect, (int, float)):
        kind = headline_finding.get("effect_kind")
        pct = kind == "pct"
        kpis.append({
            "label": "Lead finding effect", "badge": str(kind or "Effect"),
            "target": round(abs(effect) * (100 if pct else 1), 3),
            "prefix": "+" if effect >= 0 else "-", "suffix": "%" if pct else "",
            "subtext": headline_finding.get("headline") or "",
        })
    adjusted = headline_finding.get("p_adjusted")
    pvalue = adjusted if adjusted is not None else headline_finding.get("p_value")
    if isinstance(pvalue, (int, float)):
        kpis.append({
            "label": "Adjusted p-value" if adjusted is not None else "Raw p-value",
            "badge": "p < 0.001" if pvalue < 0.001 else f"p = {pvalue:.3g}",
            "target": f"{pvalue:.3g}", "prefix": "", "suffix": "",
            "subtext": headline_finding.get("headline") or "",
        })


    # 4. Chart Specs from Dashboard
    dash_charts = dashboard or []
    hero_chart = None
    segment_chart = None
    extra_charts = []

    if len(dash_charts) > 0:
        c0 = dash_charts[0]
        hero_chart = {
            "title": c0.get("title") or "Chart",
            "description": c0.get("description") or "",
            "badge": "VEGA SPEC V5",
            "spec": c0.get("spec", {}),
        }
    if len(dash_charts) > 1:
        c1 = dash_charts[1]
        segment_chart = {
            "title": c1.get("title", "Segment Contrast: Distribution"),
            "description": c1.get("description", "Distribution across profiled strata"),
            "badge": "VEGA SPEC V5",
            "spec": c1.get("spec", {}),
        }
    if len(dash_charts) > 2:
        for extra in dash_charts[2:]:
            extra_charts.append({
                "title": extra.get("title", "Analytical Insight"),
                "description": extra.get("description", "Stratified analytical feature view"),
                "badge": "VEGA SPEC V5",
                "spec": extra.get("spec", {}),
            })

    # 5. Model Leaderboard
    models_raw = rep.get("models") or rep.get("model_leaderboard") or []
    has_models = bool(models_raw)
    model_list = []
    if isinstance(models_raw, list):
        for m in models_raw:
            if isinstance(m, dict):
                model_list.append({
                    "name": m.get("name") or m.get("model", "Model"),
                    "cv_mean": m.get("cv_mean", m.get("score", "Unavailable")),
                    "gap": m.get("gap", m.get("train_test_gap", "Unavailable")),
                    "warning": m.get("warning", m.get("overfit_warning", "Unavailable")),
                })

    # 6. Action Matrix
    action_raw = rep.get("action_matrix") or rep.get("action_prioritization") or {}
    has_action_matrix = bool(action_raw)

    # 7. Audit
    ledger = f"{held} of {checked} checked findings held up" if checked else "No finding checks recorded"

    return {
        "metadata": {
            "source_name": actual_source,
            "run_id": actual_run_id,
            "title": title,
            "lead_narrative": lead_narrative,
            "headline_badge": headline_badge,
        },
        "telemetry": {
            "records": records_str,
            "dimensions": dim_str,
            "quality_score": quality_score,
            "missingness": missingness,
        },
        "kpis": kpis,
        "charts": {
            "hero": hero_chart,
            "segment": segment_chart,
            "extra": extra_charts,
        },
        "models": {
            "has_models": has_models,
            "model_list": model_list,
        },
        "action_matrix": {
            "has_action_matrix": has_action_matrix,
            "data": action_raw,
        },
        "audit": {
            "claims_verified": ledger,
            "ledger_text": ledger,
            "subtext": f"{len(finds) - checked} findings without checks",
            "checksum": rep.get("checksum"),
        },
    }


def get_swarm_dashboard_html(
    payload: dict[str, Any] | None = None,
    *,
    inline_vega: bool = True,
) -> str:
    """Return the standalone HTML content for the swarm dashboard animation,
    optionally inlining vendored Vega scripts and injecting runtime payload."""
    if not SWARM_HTML_PATH.is_file():
        return "<p>Swarm dashboard prototype HTML not found.</p>"

    html = SWARM_HTML_PATH.read_text(encoding="utf-8")

    # 1. Inline vendor Vega scripts into <head> if requested
    if inline_vega:
        static_vega_dir = Path(__file__).resolve().parent.parent.parent / "static" / "vendor" / "vega"
        vega_scripts = []
        for name in ("vega.min.js", "vega-lite.min.js", "vega-embed.min.js"):
            script_file = static_vega_dir / name
            if script_file.is_file():
                code = script_file.read_text(encoding="utf-8")
                vega_scripts.append(f"<script>{code}</script>")
        if vega_scripts:
            vega_bundle = "\n".join(vega_scripts)
            # Remove any relative script tags if present
            for name in ("vega.min.js", "vega-lite.min.js", "vega-embed.min.js"):
                rel_tag = f'<script src="../../static/vendor/vega/{name}"></script>'
                html = html.replace(rel_tag, "")
            if "</head>" in html:
                html = html.replace("</head>", f"{vega_bundle}\n</head>", 1)
            else:
                html = f"{vega_bundle}\n{html}"

    # 2. Inject payload and theme configs before </body>
    if payload is not None:
        payload_json = strict_json_dumps(payload).replace("</", "<\\/")
        day_cfg_json = strict_json_dumps(vega_config(dark=False)).replace("</", "<\\/")
        night_cfg_json = strict_json_dumps(vega_config(dark=True)).replace("</", "<\\/")
        injection = (
            f"<script>\n"
            f"window.__DSA_SWARM_PAYLOAD__ = {payload_json};\n"
            f"window.__DSA_VEGA_CFG_DAY__ = {day_cfg_json};\n"
            f"window.__DSA_VEGA_CFG_NIGHT__ = {night_cfg_json};\n"
            f"</script>\n"
        )
        if "</body>" in html:
            html = html.replace("</body>", f"{injection}</body>", 1)
        else:
            html += f"\n{injection}"

    return html


def render_native_swarm_cockpit(
    dashboard: list[dict[str, Any]] | None = None,
    payload: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    findings: list[dict[str, Any]] | None = None,
    run_view: Any | None = None,
    vega_cfg: dict[str, Any] | None = None,
    meta: Any | None = None,
    prof: Any | None = None,
    source_name: str | None = None,
    run_id: str | None = None,
) -> None:
    """Render the native Sorrel Swarm Cockpit using Streamlit containers, cards, and Vega-Lite charts."""
    if payload is None:
        payload = build_swarm_payload(
            dashboard=dashboard,
            report=report,
            findings=findings,
            run_view=run_view,
            meta=meta,
            prof=prof,
            source_name=source_name,
            run_id=run_id,
        )

    cfg = vega_cfg or {}
    meta = payload.get("metadata", {})
    telem = payload.get("telemetry", {})
    kpis = payload.get("kpis", [])
    audit = payload.get("audit", {})
    finds = findings or []

    # 1. Cockpit Header
    title = meta.get("title") or "Autonomous Executive Briefing"
    lead_narrative = meta.get("lead_narrative") or ""
    run_id = meta.get("run_id") or "Unknown"
    source_name = meta.get("source_name") or "Unavailable"
    headline_badge = meta.get("headline_badge") or "No finding checks recorded"

    st.markdown(
        f"""<div class="exec-directive" style="margin-bottom:1.35rem;">
<h2>{html.escape(str(title))}</h2>
<p class="dir-content">{html.escape(str(lead_narrative))}</p>
<div style="display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;margin-top:0.75rem;">
<span class="mono-badge" style="background:var(--paper-card);border:1px solid var(--pen);color:var(--pen);font-weight:700;">{html.escape(str(headline_badge))}</span>
<span class="mono-badge">Run #{html.escape(str(run_id))}</span>
<span class="mono-badge">{html.escape(str(source_name))}</span>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    # 2. Top Telemetry Scorecard
    st.markdown(
        f"""<div class="trust-strip" style="margin-bottom:1.5rem;">
<div class="trust-cell"><div class="k">Population Records</div><div class="v">{html.escape(str(telem.get("records", "")))}</div></div>
<div class="trust-cell"><div class="k">Data Dimensions</div><div class="v">{html.escape(str(telem.get("dimensions", "")))}</div></div>
<div class="trust-cell"><div class="k">Data Quality Score</div><div class="v">{html.escape(str(telem.get("quality_score", "")))}</div></div>
<div class="trust-cell"><div class="k">Missingness Rate</div><div class="v">{html.escape(str(telem.get("missingness", "")))}</div></div>
</div>""",
        unsafe_allow_html=True,
    )

    # 4. Key Analytical KPIs
    if kpis:
        st.markdown("")
        kpi_cols = st.columns(min(len(kpis), 3), gap="medium")
        for k_idx, kpi in enumerate(kpis[:3]):
            with kpi_cols[k_idx]:
                with st.container(border=True):
                    target_val = f"{kpi.get('prefix', '')}{kpi.get('target', '')}{kpi.get('suffix', '')}"
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;">'
                        f'<div style="font-size:0.76rem;font-weight:600;color:var(--ink-muted);text-transform:uppercase;letter-spacing:0.04em;">{html.escape(str(kpi.get("label", "")))}</div>'
                        f'<span class="mono-badge" style="color:var(--pen);font-size:0.7rem;">{html.escape(str(kpi.get("badge", "")))}</span>'
                        f'</div>'
                        f'<div style="font-size:1.75rem;font-weight:700;color:var(--ink-text);font-family:var(--sans);margin:0.2rem 0;">{html.escape(target_val)}</div>'
                        f'<div style="font-size:0.78rem;color:var(--ink-muted);line-height:1.35;">{html.escape(str(kpi.get("subtext", "")))}</div>',
                        unsafe_allow_html=True,
                    )

    # 5. Synchronized Analytical Vega-Lite Visualizations
    st.markdown("")
    st.markdown("### Synchronized Analytical Visualizations")

    charts = dashboard or []
    if charts:
        finding_by_id = {str(f.get("finding_id")): f for f in finds if f.get("finding_id")}

        def _get_note(ch: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
            f = finding_by_id.get(str(ch.get("finding_id") or ""))
            if not f:
                return "", None
            return f"Evidence for: {plainify(str(f.get('headline', '')))} ({state_label(finding_state(f))})", f

        # Lead Chart
        c0 = charts[0]
        note0, f0_match = _get_note(c0)
        render_dashboard_chart(c0, cfg, note0, f0_match, framed=True, height=340, card_key="swarm_chart_lead")

        # Subsequent charts
        if len(charts) > 1:
            for j in range(1, len(charts), 2):
                pair = charts[j : j + 2]
                if len(pair) == 1:
                    note_j, f_j = _get_note(pair[0])
                    render_dashboard_chart(pair[0], cfg, note_j, f_j, framed=True, height=280, card_key=f"swarm_chart_{j}")
                else:
                    c_left, c_right = st.columns(2, gap="medium")
                    with c_left:
                        n_l, f_l = _get_note(pair[0])
                        render_dashboard_chart(pair[0], cfg, n_l, f_l, framed=True, height=280, card_key=f"swarm_chart_{j}")
                    with c_right:
                        n_r, f_r = _get_note(pair[1])
                        render_dashboard_chart(pair[1], cfg, n_r, f_r, framed=True, height=280, card_key=f"swarm_chart_{j+1}")
    else:
        st.info("No dashboard charts generated for this run.")

    # 6. Audited Swarm Execution Ledger
    st.markdown("")
    with st.container(border=True):
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">'
            f'<div style="font-weight:700;font-size:0.95rem;">{html.escape(str(audit.get("ledger_text", "Audited Autonomous Ledger")))}</div>'
            f'</div><div>{html.escape(str(audit.get("subtext") or ""))}</div>',

            unsafe_allow_html=True,
        )


def render_swarm_dashboard(
    dashboard: list[dict[str, Any]] | None = None,
    payload: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    findings: list[dict[str, Any]] | None = None,
    run_view: Any | None = None,
    vega_cfg: dict[str, Any] | None = None,
    meta: Any | None = None,
    prof: Any | None = None,
    source_name: str | None = None,
    run_id: str | None = None,
    *,
    height: int = 860,
    scrolling: bool = True,
    run_once: bool = True,
    session_key: str = "swarm_dash_built",
) -> None:
    """
    Render the swarm-constructed dashboard inside a Streamlit container.

    When `vega_cfg` is provided, renders the Native Sorrel Cockpit directly using Streamlit
    components and Vega-Lite charts. If `vega_cfg` is None, falls back to the iframe
    prototype component for backward-compatible tests.
    """
    if vega_cfg is not None:
        render_native_swarm_cockpit(
            dashboard=dashboard,
            payload=payload,
            report=report,
            findings=findings,
            run_view=run_view,
            vega_cfg=vega_cfg,
            meta=meta,
            prof=prof,
            source_name=source_name,
            run_id=run_id,
        )
        return

    if payload is None and (dashboard is not None or report is not None or findings is not None or run_view is not None):
        payload = build_swarm_payload(
            dashboard=dashboard,
            report=report,
            findings=findings,
            run_view=run_view,
            meta=meta,
            prof=prof,
            source_name=source_name,
            run_id=run_id,
        )

    html_content = get_swarm_dashboard_html(payload=payload, inline_vega=True)

    # Track one-shot trigger if requested
    if run_once:
        has_run = bool(st.session_state.get(session_key, False))
        if not has_run:
            st.session_state[session_key] = True

    components.html(html_content, height=height, scrolling=scrolling)
