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

        def flush(pending: list[dict[str, Any]]) -> None:
            """Two panels per row; a lone last panel takes the full width."""
            for j in range(0, len(pending), 2):
                row = pending[j : j + 2]
                if len(row) == 1:
                    render_dashboard_chart(row[0], vega_cfg)
                    continue
                for col, ch in zip(st.columns(2), row, strict=True):
                    with col:
                        render_dashboard_chart(ch, vega_cfg)

        # Ranking order is kept. Only the lead chart and charts that need width (time series,
        # survival curves, anything marked "wide") span the page; the rest sit two to a row, so
        # a three-bar chart is not stretched across a full-width canvas.
        pending: list[dict[str, Any]] = []
        for index, ch in enumerate(dashboard):
            spans_page = ch.get("size") == "wide" or (index == 0 and ch.get("size") != "half")
            if spans_page:
                flush(pending)
                pending = []
                render_dashboard_chart(ch, vega_cfg)
            else:
                pending.append(ch)
        flush(pending)
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
