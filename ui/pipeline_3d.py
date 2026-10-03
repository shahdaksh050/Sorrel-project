"""
The plate: the seven-stage RLM workflow drawn as a live technical drawing, in the Sorrel inks.

The scene itself lives in ``assets/pipeline_3d.js`` (Three.js + GSAP); this
module is the seam between it and Streamlit. It owns the state contract —
what a stage is, which inks the scene may use — and inlines both assets
into a single sandboxed iframe, because ``components.html`` has no way to
serve sibling files.

Streamlit re-runs the whole script on every interaction, so the component
mounts fresh each time and renders one snapshot of ``st.session_state``.
The scene replays the run on mount rather than streaming frames live.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import streamlit as st
import streamlit.components.v1 as components

from src.core import design_tokens
from ui.styles import static_url

__all__ = [
    "PALETTE",
    "PALETTES",
    "Stage",
    "StageStatus",
    "build_cinematic_document",
    "build_document",
    "export_cinematic_html",
    "extract_cinematic_state",
    "render",
    "render_cinematic",
]

_ASSETS = Path(__file__).parent / "assets"

#: Matches the status vocabulary ``app.py`` already writes into ``stage_log``.
StageStatus = Literal["pending", "active", "done", "skipped", "error"]

#: Day / Night palettes for the Sorrel theme, sourced from the shared token
#: module, every token included (the scene reads `accent_text`, `accent_ink`,
#: `bg_deep` ... as well as the original inks). The scene's "grid" key predates
#: `design_tokens`' "rule" naming, so it is mapped explicitly rather than left
#: missing by a plain dict spread.
PALETTES: dict[str, dict[str, str]] = {
    mode: {**palette, "grid": palette["rule"]}
    for mode, palette in design_tokens.PALETTES.items()
}

#: Default palette for backwards compatibility
PALETTE: dict[str, str] = PALETTES["day"]


@dataclass(frozen=True, slots=True)
class Stage:
    """One workflow stage, in the shape the scene needs to draw it.

    Args:
        num: Stage number as displayed, e.g. ``"3"``.
        name: Human-readable stage name.
        status: Where the run got to for this stage.
        detail: Short free-text note shown when the stage is hovered.
    """

    num: str
    name: str
    status: StageStatus = "pending"
    detail: str = ""


_THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"


@lru_cache(maxsize=2)
def _asset(name: str) -> str:
    """Read a bundled asset once per process."""
    return (_ASSETS / name).read_text(encoding="utf-8")


def _embed(document: str, height: int) -> None:
    """Mount ``document`` in a sandboxed iframe.

    ``st.iframe`` is the current API and ``components.html`` is deprecated. The
    ``hasattr`` guard stays for environments that install an older Streamlit than
    requirements.txt asks for.
    """
    if hasattr(st, "iframe"):
        st.iframe(document, height=height)
    else:  # pragma: no cover - only reached on older Streamlit
        components.html(document, height=height, scrolling=False)


def build_document(stages: Sequence[Stage], theme: str = "day") -> str:
    """Assemble the standalone HTML document for ``stages``.

    Split out from :func:`render` so the scene can be previewed without a
    Streamlit server — see ``scripts/preview_pipeline_3d.py``.

    Args:
        stages: The workflow stages, in execution order.
        theme: "day" or "night".

    Returns:
        A self-contained HTML document, bar the Three.js and GSAP CDN tags.
    """
    palette = PALETTES.get(theme, PALETTES["day"])
    mode: design_tokens.Mode = "night" if theme == "night" else "day"
    state = {
        "stages": [
            {"num": s.num, "name": s.name, "status": s.status, "detail": s.detail}
            for s in stages
        ],
        "palette": palette,
        "theme": theme,
    }
    # Stage details come from tool output, so escape anything that could close
    # the inline <script> early. The scene renders them with textContent.
    state_json = json.dumps(state, ensure_ascii=False).replace("<", "\\u003c")

    # Local copies first (Streamlit's static route), the CDN only as a fallback: the plate must
    # draw with no network. The page's absolute paths resolve against the Streamlit origin.
    three_urls = json.dumps([static_url("vendor/three/three.module.js"), _THREE_CDN]).replace("<", "\\u003c")
    return (
        _asset("pipeline_3d.html")
        .replace("__ROOT_TOKENS__", design_tokens.css_root_block(mode))
        .replace("__FONT_LINKS__", f'<link rel="stylesheet" href="{static_url("vendor/fonts/ledger-fonts.css")}">')
        .replace("__GSAP_URL__", static_url("vendor/gsap/3.15.0/gsap.min.js"))
        .replace("__SCENE_SCRIPT__", f"window.__THREE_URLS__ = {three_urls};\n" + _asset("pipeline_3d.js"))
        .replace("__STATE_JSON__", state_json)  # last: stage text that contains a placeholder name stays text
    )


def render(stages: Sequence[Stage], *, height: int = 420, theme: str = "day") -> None:
    """Draw the plate for ``stages``.

    Args:
        stages: The workflow stages, in execution order.
        height: Iframe height in pixels. The scene reframes itself to fit.
        theme: Theme name ("day" or "night").
    """
    _embed(build_document(stages, theme=theme), height)


# ── Cinematic 3D Fullpage Master Architecture Bridge ─────────────────────────
def build_cinematic_document(
    state_dict: dict[str, Any] | None = None,
    theme: str = "night",
) -> str:
    """Assemble the 6-section 3D cinematic presentation document."""
    from ui.cinematic_3d import build_cinematic_document as _bcd
    return _bcd(state_dict=state_dict, theme=theme)


def render_cinematic(
    state_or_session: Any = None,
    *,
    height: int = 860,
    theme: str = "night",
) -> None:
    """Render the 6-section 3D cinematic showcase in Streamlit."""
    from ui.cinematic_3d import render_cinematic as _rc
    _rc(state_or_session=state_or_session, height=height, theme=theme)


def extract_cinematic_state(session_state: Any) -> dict[str, Any]:
    """Extract live or preview state for the 6-section 3D cinematic showcase."""
    from ui.cinematic_3d import extract_cinematic_state as _ecs
    return _ecs(session_state)


def export_cinematic_html(
    output_path: Path | str,
    state_dict: dict[str, Any] | None = None,
    theme: str = "night",
) -> Path:
    """Export the 6-section 3D cinematic showcase as a standalone HTML presentation."""
    from ui.cinematic_3d import export_cinematic_html as _ech
    return _ech(output_path=output_path, state_dict=state_dict, theme=theme)
