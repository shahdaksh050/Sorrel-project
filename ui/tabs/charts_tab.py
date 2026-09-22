"""
Charts Tab (Dynamic Vega-Lite Dashboard).
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui.components.cards import render_dashboard_chart


def render_charts_tab(
    dashboard: list[dict[str, Any]] | None,
    report: dict[str, Any],
    vega_cfg: dict[str, Any],
) -> None:
    """Render Tier 3: Dynamic Visual Dashboard."""
    if dashboard:
        st.caption("Built automatically to fit your data — the most important panels lead.")

        top_priority_ids = {c.get("chart_id") for c in dashboard[:3]}
        grid_charts: list[dict[str, Any]] = []

        for ch in dashboard:
            is_full_width = ch.get("size") == "wide" or (
                ch.get("size") != "half"
                and (ch.get("layer") == "exec" or ch.get("chart_id") in top_priority_ids)
            )
            if is_full_width:
                render_dashboard_chart(ch, vega_cfg)
            else:
                grid_charts.append(ch)

        if grid_charts:
            for j in range(0, len(grid_charts), 2):
                row = grid_charts[j : j + 2]
                if len(row) == 1:
                    render_dashboard_chart(row[0], vega_cfg)
                    continue
                for col, ch in zip(st.columns(2), row, strict=True):
                    with col:
                        render_dashboard_chart(ch, vega_cfg)
    else:
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
