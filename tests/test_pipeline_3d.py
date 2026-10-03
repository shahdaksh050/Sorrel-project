"""Tests for the 3D pipeline hero component (ui/pipeline_3d.py)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.pipeline_3d import PALETTE, Stage, _asset, build_document, render  # noqa: E402


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the document render() hands to Streamlit, without a browser."""
    calls: list[dict[str, Any]] = []

    def fake_iframe(src: str, **kwargs: Any) -> None:
        calls.append({"document": src, **kwargs})

    monkeypatch.setattr("ui.pipeline_3d.st.iframe", fake_iframe)
    return calls


STAGES = [
    Stage("1", "Dataset Ingestion", "done", "1,000 rows"),
    Stage("2", "Initial Reasoning", "done"),
    Stage("3", "Tool Execution", "active", "running"),
    Stage("4", "Result Interpretation"),
]


def _state_of(document: str) -> dict[str, Any]:
    """Pull the injected state object back out of the rendered document."""
    match = re.search(r"window\.__PIPELINE_STATE__ = (\{.*?\});", document, re.S)
    assert match, "state payload not found in document"
    return json.loads(match.group(1))  # type: ignore[no-any-return]


# ── Assets ───────────────────────────────────────────────────────────────────
def test_assets_exist() -> None:
    assert _asset("pipeline_3d.html").lstrip().startswith("<!doctype html>")
    assert "THREE" in _asset("pipeline_3d.js")


def test_no_placeholders_left_in_document(captured: list[dict[str, Any]]) -> None:
    render(STAGES)
    document = captured[0]["document"]
    assert "__STATE_JSON__" not in document
    assert "__SCENE_SCRIPT__" not in document


def test_scene_and_libraries_are_inlined(captured: list[dict[str, Any]]) -> None:
    render(STAGES)
    document = captured[0]["document"]
    assert "three.module.js" in document, "three.js not loaded"
    assert "gsap.min.js" in document, "gsap not loaded"
    assert "setAnimationLoop" in document, "scene script not inlined"


# ── State contract ───────────────────────────────────────────────────────────
def test_stage_state_round_trips(captured: list[dict[str, Any]]) -> None:
    render(STAGES)
    state = _state_of(captured[0]["document"])

    assert state["palette"] == PALETTE
    assert [s["num"] for s in state["stages"]] == ["1", "2", "3", "4"]
    assert state["stages"][2] == {
        "num": "3",
        "name": "Tool Execution",
        "status": "active",
        "detail": "running",
    }
    assert state["stages"][3]["status"] == "pending", "default status"


def test_empty_stage_list_is_allowed(captured: list[dict[str, Any]]) -> None:
    render([])
    assert _state_of(captured[0]["document"])["stages"] == []


def test_height_is_forwarded(captured: list[dict[str, Any]]) -> None:
    render(STAGES, height=512)
    assert captured[0]["height"] == 512


def test_falls_back_to_components_html_without_st_iframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """requirements.txt still allows streamlit versions that predate st.iframe."""
    import ui.pipeline_3d as mod

    calls: list[dict[str, Any]] = []
    monkeypatch.delattr(mod.st, "iframe", raising=False)
    monkeypatch.setattr(
        "ui.pipeline_3d.components.html",
        lambda document, **kw: calls.append({"document": document, **kw}),
    )

    render(STAGES)
    assert len(calls) == 1
    assert calls[0]["scrolling"] is False
    assert "__PIPELINE_STATE__" in calls[0]["document"]


# ── Injection safety ─────────────────────────────────────────────────────────
def test_stage_detail_cannot_break_out_of_the_script_tag(
    captured: list[dict[str, Any]],
) -> None:
    """Tool output reaches `detail`, so it must not be able to inject markup."""
    hostile = "</script><script>window.pwned=1</script>"
    render([Stage("1", "Dataset Ingestion", "done", hostile)])
    document = captured[0]["document"]

    assert "</script><script>" not in document, "script tag was not neutralised"

    # …and the escaped form still parses back to the original text.
    assert _state_of(document)["stages"][0]["detail"] == hostile


def test_hostile_stage_name_is_escaped(captured: list[dict[str, Any]]) -> None:
    render([Stage("1", "<img src=x onerror=alert(1)>", "done")])
    document = captured[0]["document"]
    assert "<img src=x" not in document


# ── Design system ────────────────────────────────────────────────────────────
def test_palette_matches_design_tokens() -> None:
    """The scene's inks come from src/core/design_tokens.py; it may not invent others.

    Compared to the tokens, not to typed hex values: a hard-coded copy here went stale
    when the palette changed, which is the drift this test exists to catch."""
    from src.core import design_tokens

    day = design_tokens.palette("day")
    for key in ("ink", "pen", "risk", "stock", "sheet", "graphite", "accent"):
        assert PALETTE[key] == day[key], f"plate {key} drifted from design_tokens"
    assert PALETTE["grid"] == day["rule"]  # the scene's older name for the hairline
    assert set(PALETTE) <= set(day) | {"grid"}, "the scene adds no inks beyond the tokens and its grid alias"


def test_stage_is_immutable() -> None:
    stage = Stage("1", "Dataset Ingestion")
    with pytest.raises(AttributeError):
        stage.status = "done"  # type: ignore[misc]


# ── The plate must draw with no network, and never be an empty box ───────────────────────────
def test_plate_loads_three_locally_before_the_cdn_and_never_from_google() -> None:
    document = build_document([Stage("1", "Reading", "done", "")], theme="day")
    urls = json.loads(re.search(r"window\.__THREE_URLS__ = (.*);", document).group(1))  # type: ignore[union-attr]
    assert urls[0].endswith("/app/static/vendor/three/three.module.js")
    assert urls[1].startswith("https://cdn.jsdelivr.net/")
    assert "fonts.googleapis.com" not in document and "fonts.gstatic.com" not in document
    assert "/app/static/vendor/fonts/ledger-fonts.css" in document


def test_a_failed_three_load_cannot_kill_the_script_and_shows_the_stages_as_text() -> None:
    document = build_document([Stage("1", "Reading", "done", "")], theme="day")
    assert "import * as THREE" not in document  # a static import would die before any fallback ran
    assert "await import(url)" in document
    assert "fallback(" in document and 'document.createElement("ol")' in document


def test_a_gsap_stand_in_keeps_the_scene_drawing_when_gsap_is_missing() -> None:
    document = build_document([Stage("1", "Reading", "done", "")], theme="day")
    assert "if (window.gsap) return;" in document
    for method in ("timeline", "globalTimeline", "to:", "from:", "set:"):
        assert method in document


# ── Sorrel ─────────────────────────────────────────────────────────────────────────────────────
def _doc(theme: str = "day") -> str:
    return build_document([Stage("1", "Reading", "done", "")], theme=theme)


@pytest.mark.parametrize("theme", ["day", "night"])
def test_the_document_carries_that_themes_tokens_and_no_placeholder(theme: str) -> None:
    from src.core import design_tokens

    document = _doc(theme)
    assert "__ROOT_TOKENS__" not in document
    for key in ("pen", "stock", "sheet", "accent", "accent_text", "accent_ink", "rule"):
        css_var = f"--{key.replace('_', '-')}: {design_tokens.palette(theme)[key]};"  # type: ignore[arg-type]
        assert css_var in document, css_var


def test_the_plate_is_set_in_the_sorrel_families_not_the_old_ones() -> None:
    document = _doc()
    assert "'Geist'" in document and "'Geist Mono'" in document
    for old in ("Public Sans", "Bricolage", "IBM Plex", "Baloo", "Mukta"):
        assert old not in document


def test_no_old_ledger_brown_is_left_in_the_plate() -> None:
    document = _doc()
    for old in ("#3a2b1e", "#a34f20", "#a33526", "#f7eedd", "#fffbf2", "#8a7660", "rgba(58, 43, 30"):
        assert old not in document, old


def test_a_failed_stage_is_amber_never_the_may_not_hold_brick() -> None:
    """DESIGN.md: --risk only ever means "this finding may not hold"; a stage error is a generic error."""
    js = _asset("pipeline_3d.js")
    assert "C.risk" not in js
    assert 'if (status === "error") return WARN;' in js
    html = _asset("pipeline_3d.html")
    assert "var(--risk)" not in html


def test_stage_tokens_are_square_and_flat() -> None:
    html = _asset("pipeline_3d.html")
    assert "border-radius: 999px" not in html and "box-shadow" not in html
    assert "gradient" not in html and "backdrop-filter" not in html


def test_nothing_spins_or_flows_at_rest_or_under_reduced_motion() -> None:
    js = _asset("pipeline_3d.js")
    assert "!REDUCED_MOTION && activeIndex >= 0" in js
    assert "activeIndex < 0 || REDUCED_MOTION" in js  # the particle stream
    assert "core.rotation.y = t *" not in js and "Math.floor(t * 1.5)" not in js  # no continuous spin
    assert "prefers-reduced-motion: reduce" in _asset("pipeline_3d.html")


def test_plate_text_and_marks_meet_contrast_in_both_themes() -> None:
    from src.core import design_tokens

    def lum(hex_color: str) -> float:
        def ch(c: int) -> float:
            v = c / 255
            return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

        r, g, b = (ch(int(hex_color[i : i + 2], 16)) for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def ratio(a: str, b: str) -> float:
        hi, lo = sorted((lum(a), lum(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    for theme in ("day", "night"):
        p = design_tokens.palette(theme)  # type: ignore[arg-type]
        for fg, bg in (("ink", "sheet"), ("ink_2", "sheet"), ("accent_text", "sheet"), ("accent", "sheet"),
                       ("graphite", "stock"), ("accent_ink", "pen"), ("ink", "sheet_alt")):
            assert ratio(p[fg], p[bg]) >= 4.5, f"{theme}: {fg} on {bg}"
        assert ratio(p["accent_text"], p["stock"]) >= 3.0  # the Night stage ink against the page


def test_stage_text_naming_a_placeholder_stays_text_and_never_splices_the_script_in() -> None:
    trick = "__SCENE_SCRIPT__ __GSAP_URL__"
    document = build_document([Stage("1", "Reading", "done", trick)], theme="day")
    assert _state_of(document)["stages"][0]["detail"] == trick
    assert document.count("setAnimationLoop") == _asset("pipeline_3d.js").count("setAnimationLoop")
