"""
Streamlit UI Tab Modules for Results Dossier.
"""
from __future__ import annotations

from ui.tabs.actions_tab import render_actions_tab
from ui.tabs.answers_tab import render_answers_tab
from ui.tabs.charts_tab import render_charts_tab
from ui.tabs.details_tab import render_details_tab
from ui.tabs.downloads_tab import render_downloads_tab

__all__ = [
    "render_actions_tab",
    "render_answers_tab",
    "render_charts_tab",
    "render_details_tab",
    "render_downloads_tab",
]
