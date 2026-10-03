"""
Landing Page (Phase 2 Narrative)
Serves the custom Streamlit component for the scroll-driven landing experience.
"""
from __future__ import annotations

import os
from typing import Any

import streamlit.components.v1 as components

from src.core.design_tokens import palette

# Canonical "Ledger" design-system tokens (see DESIGN.md, "Tokens — ink").
# This is the single Python source of truth for the two hexes this module needs
# (the iframe/background colour shown while the static landing component loads).
# The static HTML asset (ui/landing_component/index.html) cannot import this
# constant, so its inline :root/.theme-day CSS custom properties are hand-kept
# in sync with these values — look for the "Keep in sync" comment above its
# :root block.
# FrontendPlan.md item 2.1 — these two dicts used to hand-type a third
# independent copy of the same hex codes (the audit that found T2/T3/T6
# missed this file); they now pull the 5 keys this module needs out of
# src.core.design_tokens.palette(), the single Python source of truth.
# Names and shape (dict[str, str], these exact 5 keys) are unchanged because
# tests/test_landing.py and tests/test_landing_v2.py import them by name.
LEDGER_TOKENS_DAY: dict[str, str] = {k: palette("day")[k] for k in ("stock", "sheet", "ink", "graphite", "pen")}
LEDGER_TOKENS_NIGHT: dict[str, str] = {k: palette("night")[k] for k in ("stock", "sheet", "ink", "graphite", "pen")}

# landing_component is the one live implementation. This resolver used to prefer a sibling
# directory whenever its index.html existed, which it always did — so every edit to
# landing_component/index.html was silently rendering nothing. Fixed by dropping that preference.
_component_dir: str = os.path.join(os.path.dirname(__file__), "landing_component")

_env_override: str | None = os.environ.get("DSA_LANDING_DIR")
if _env_override and os.path.isdir(_env_override):
    _component_dir = _env_override

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
    from ui.components.islands import island_base_url

    value = _landing_component(
        theme=theme, islands_url=island_base_url("hero"), key="landing_narrative", default=None
    )

    if isinstance(value, dict):
        chosen = value.get("theme")
        if chosen in ("day", "night") and chosen != st.session_state.get("theme"):
            st.session_state["theme"] = chosen
        return bool(value.get("enter"))
    return bool(value)
