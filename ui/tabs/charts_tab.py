"""
Charts Tab (Dynamic Vega-Lite Dashboard).
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from src.core.plain_language import plainify
from ui.components.cards import finding_state, render_dashboard_chart, state_label

#: One chart height per panel type, so the panels of a row line up: half-width panels share one, and
#: the full-width panels (the lead chart, time series, anything marked "wide") share a taller one.
HALF_HEIGHT = 260
WIDE_HEIGHT = 340


def render_charts_tab(
    dashboard: list[dict[str, Any]] | None,
    report: dict[str, Any],
    vega_cfg: dict[str, Any],
    findings: list[dict[str, Any]] | None = None,
) -> None:
    """Render Tier 3: Dynamic Visual Dashboard."""
    by_id = {str(f["finding_id"]): f for f in (findings or []) if f.get("finding_id")}

    def finding_of(ch: dict[str, Any]) -> dict[str, Any] | None:
        """The finding a chart supports, so its checks and its source can sit beside it."""
        return by_id.get(str(ch.get("finding_id") or ""))

    def note(ch: dict[str, Any]) -> str:
        """Which finding a chart supports, with its verdict, so a chart is never an unexplained picture."""
        f = by_id.get(str(ch.get("finding_id") or ""))
        if f is None:
            return ""
        return f"Evidence for: {plainify(str(f.get('headline', '')))} ({state_label(finding_state(f))})"

    if dashboard:
        st.caption(
            "Charts are chosen to fit your data, most important first. "
            "A chart tied to a finding says which one and whether it held up."
        )

        def flush(pending: list[dict[str, Any]]) -> None:
            """Two panels per row, in bordered columns so the two cards are always the same height; a
            lone last panel takes the full width."""
            for j in range(0, len(pending), 2):
                row = pending[j : j + 2]
                if len(row) == 1:
                    render_dashboard_chart(row[0], vega_cfg, note(row[0]), finding_of(row[0]), height=HALF_HEIGHT)
                    continue
                for col, ch in zip(st.columns(2, border=True), row, strict=True):
                    with col:
                        render_dashboard_chart(
                            ch, vega_cfg, note(ch), finding_of(ch), height=HALF_HEIGHT, framed=False
                        )

        # Ranking order is kept. Only the lead chart and charts that need width (time series,
        # survival curves, anything marked "wide") span the page; the rest sit two to a row, so
        # a three-bar chart is not stretched across a full-width canvas.
        pending: list[dict[str, Any]] = []
        for index, ch in enumerate(dashboard):
            spans_page = ch.get("size") == "wide" or (index == 0 and ch.get("size") != "half")
            if spans_page:
                flush(pending)
                pending = []
                render_dashboard_chart(ch, vega_cfg, note(ch), finding_of(ch), height=WIDE_HEIGHT)
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
