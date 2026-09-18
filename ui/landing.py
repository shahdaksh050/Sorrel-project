"""
Landing Page (Phase 2 Narrative)
Serves the custom Streamlit component for the scroll-driven landing experience.
"""
from __future__ import annotations

import os
from typing import Any

import streamlit.components.v1 as components

# Canonical "Ledger" design-system tokens (see DESIGN.md, "Tokens — ink").
# This is the single Python source of truth for the two hexes this module needs
# (the iframe/background colour shown while the static landing component loads).
# The static HTML assets (ui/landing_component/index.html, frontend-landing/index.html,
# and their non-Python siblings) cannot import this constant, so their inline
# :root/.theme-day CSS custom properties are hand-kept in sync with these values
# and with DESIGN.md — look for the "Keep in sync" comments above each :root block.
# Ledger design-system tokens (DESIGN.md) — single source of truth. The
# static landing HTML (ui/landing_component/index.html, frontend-landing/)
# duplicates these as literal CSS hexes (no build step wires them together),
# so tests assert against THIS dict rather than re-typing hex literals a
# second time — the exact "give the palette a single source of truth and
# have both the code and the tests import it" fix IMPROVEMENTS.md's Q1 asks
# for. Keep in sync with DESIGN.md's ink token table if it ever changes.
LEDGER_TOKENS_DAY: dict[str, str] = {
    "stock": "#f7eedd",
    "sheet": "#fffbf2",
    "ink": "#3a2b1e",
    "graphite": "#8a7660",
    "pen": "#a34f20",
}
LEDGER_TOKENS_NIGHT: dict[str, str] = {
    "stock": "#241c14",
    "sheet": "#2f251a",
    "ink": "#f3e9d8",
    "graphite": "#b8a688",
    "pen": "#f0a24a",
}

# Determine component directory: prefers the next-generation frontend-landing directory if present
_LEGACY_DIR: str = os.path.join(os.path.dirname(__file__), "landing_component")
_V2_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend-landing"))

_env_override: str | None = os.environ.get("DSA_LANDING_DIR")
if _env_override and os.path.isdir(_env_override):
    _component_dir: str = _env_override
elif os.path.isdir(_V2_DIR) and os.path.isfile(os.path.join(_V2_DIR, "index.html")):
    _component_dir = _V2_DIR
else:
    _component_dir = _LEGACY_DIR

# Declare the component
_landing_component: Any = components.declare_component("landing", path=_component_dir)

def show_landing_page() -> bool:
    """Renders the full-screen narrative landing page.

    Returns True if the user clicked the CTA button to enter the workspace, otherwise False.
    A theme toggled on the landing page is written back to ``st.session_state["theme"]``
    so the workspace opens in the same theme.
    """
    import streamlit as st

    # Day is the console's default theme (app.py _DEFAULTS); the landing runs before those defaults
    theme = "night" if st.session_state.get("theme", "day") in ("night", "dark") else "day"
    # Matches the console's --stock token for each theme (app.py _inject_theme_css)
    bg_color = LEDGER_TOKENS_DAY["stock"] if theme == "day" else LEDGER_TOKENS_NIGHT["stock"]

    # Hide Streamlit UI completely and force iframe to be fixed full-screen
    st.markdown(f"""
        <style>
            header, footer, [data-testid="stSidebar"] {{ display: none !important; }}
            .block-container {{ padding: 0 !important; max-width: 100% !important; margin: 0 !important; }}
            .stApp {{ background: {bg_color} !important; }}

            iframe {{
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                border: none !important;
                z-index: 999999 !important;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Render the component. The JS sends {"enter": bool, "theme": "day" | "night"}.
    value = _landing_component(theme=theme, key="landing_narrative", default=None)

    if isinstance(value, dict):
        chosen = value.get("theme")
        if chosen in ("day", "night") and chosen != st.session_state.get("theme"):
            st.session_state["theme"] = chosen
        return bool(value.get("enter"))
    return bool(value)
