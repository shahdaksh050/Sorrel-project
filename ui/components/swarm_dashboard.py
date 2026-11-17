"""
Swarm Dashboard Animation Component.

Renders an interactive dummy dashboard constructed from a blank slate by a swarm
of 10 autonomous agent cursors with selection marquees and card stamping.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

SWARM_HTML_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "prototypes" / "swarm_dashboard.html"


def get_swarm_dashboard_html() -> str:
    """Return the standalone HTML content for the swarm dashboard animation."""
    if SWARM_HTML_PATH.is_file():
        return SWARM_HTML_PATH.read_text(encoding="utf-8")
    return "<p>Swarm dashboard prototype HTML not found.</p>"


def render_swarm_dashboard(
    *,
    height: int = 860,
    run_once: bool = True,
    session_key: str = "swarm_dash_built",
) -> None:
    """
    Render the swarm-constructed dummy dashboard inside a Streamlit container.

    Args:
        height: Height in pixels for the embedded iframe component.
        run_once: If True, tracks execution in `st.session_state` so the animation
                  is triggered only once per session or analysis run.
        session_key: The session state key used to track one-shot completion.
    """
    html_content = get_swarm_dashboard_html()

    # Track one-shot trigger if requested
    if run_once:
        has_run = bool(st.session_state.get(session_key, False))
        if not has_run:
            st.session_state[session_key] = True

    components.html(html_content, height=height, scrolling=True)
