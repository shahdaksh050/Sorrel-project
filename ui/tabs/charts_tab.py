"""
Charts Tab (Dynamic Vega-Lite Dashboard).
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui.components.cards import tab_head
from ui.components.swarm_dashboard import render_native_swarm_cockpit

#: One chart height per panel type, so the panels of a row line up: half-width panels share one, and
#: the full-width panels (the lead chart, time series, anything marked "wide") share a taller one.
HALF_HEIGHT = 260
WIDE_HEIGHT = 340


def render_charts_tab(
    dashboard: list[dict[str, Any]] | None,
    report: dict[str, Any],
    vega_cfg: dict[str, Any],
    findings: list[dict[str, Any]] | None = None,
    run_view: Any | None = None,
    meta: Any | None = None,
    prof: Any | None = None,
) -> None:
    """Render Tier 3: Dynamic Visual Dashboard (Unified Cockpit & Synchronized Analytical Visualizations)."""
    tab_head("Charts", "The charts, most important", "first")
    if not dashboard:
        dash_coverage = report.get("coverage") or {}
        if dash_coverage.get("total", 0) > 0 and dash_coverage.get("answered", 0) == 0:
            st.info(
                "No chart-worthy findings were produced for this run — every "
                "question on the agenda was either declined or came back without "
                "a chartable pattern. See the Details tab for the full coverage breakdown."
            )
        else:
            st.info(
                "No dashboard was generated for this run. The run may still be "
                "processing, or the analysis produced no chartable results."
            )
        return

    st.caption(
        "Charts are chosen to fit your data, most important first. "
        "A chart tied to a finding says which one and whether it held up."
    )

    render_native_swarm_cockpit(
        dashboard=dashboard,
        report=report,
        findings=findings,
        run_view=run_view,
        vega_cfg=vega_cfg,
        meta=meta,
        prof=prof,
    )

