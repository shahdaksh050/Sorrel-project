"""Tests for the 6-Section 3D Cinematic Fullpage Experience (ui/cinematic_3d.py)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]

from ui.cinematic_3d import (  # noqa: E402
    CINEMATIC_PALETTES,
    _read_asset,
    build_cinematic_document,
    export_cinematic_html,
    extract_cinematic_state,
    render_cinematic,
)


@pytest.fixture
def captured_iframe(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_iframe(src: str, **kwargs: Any) -> None:
        calls.append({"document": src, **kwargs})

    import ui.cinematic_3d as cmod
    monkeypatch.setattr(cmod.st, "iframe", fake_iframe)
    return calls


def _state_of(document: str) -> dict[str, Any]:
    match = re.search(r"window\.__CINEMATIC_STATE__ = (\{.*?\});", document, re.S)
    assert match, "state payload not found in document"
    return json.loads(match.group(1))


def test_cinematic_assets_exist() -> None:
    html_content = _read_asset("cinematic_3d.html")
    assert html_content.lstrip().startswith("<!doctype html>")
    assert "fullpage" in html_content
    assert "anime" in html_content
    js_content = _read_asset("cinematic_3d.js")
    assert "THREE" in js_content
    assert "setAnimationLoop" in js_content


def test_no_placeholders_left_in_cinematic_doc() -> None:
    doc = build_cinematic_document()
    assert "__CINEMATIC_STATE_JSON__" not in doc
    assert "__CINEMATIC_SCENE_SCRIPT__" not in doc


def test_cinematic_state_structure() -> None:
    doc = build_cinematic_document(theme="night")
    state = _state_of(doc)
    assert state["theme"] == "night"
    assert "dataset" in state
    assert "stages" in state
    assert len(state["stages"]) == 7
    assert "statistics" in state
    assert "ml" in state
    assert "synthesis" in state
    assert state["palette"] == CINEMATIC_PALETTES["night"]


def test_render_cinematic_height_and_iframe(captured_iframe: list[dict[str, Any]]) -> None:
    render_cinematic(height=900, theme="day")
    assert len(captured_iframe) == 1
    assert captured_iframe[0]["height"] == 900
    doc = captured_iframe[0]["document"]
    assert "window.__CINEMATIC_STATE__" in doc
    assert "day" in doc


def test_export_cinematic_html(tmp_path: Path) -> None:
    target = tmp_path / "cinematic_presentation.html"
    exported = export_cinematic_html(target, theme="night")
    assert exported.exists()
    content = exported.read_text(encoding="utf-8")
    assert "<!doctype html>" in content
    assert "fullpage" in content


def test_injection_safety_in_cinematic_doc() -> None:
    hostile = "</script><script>window.pwned=1</script>"
    state = extract_cinematic_state({
        "preview_name": hostile,
        "user_objective": hostile,
        "theme": "night",
    })
    doc = build_cinematic_document(state)
    assert "</script><script>" not in doc
    extracted = _state_of(doc)
    assert extracted["dataset"]["name"] == hostile


def test_extract_cinematic_state_with_nones() -> None:
    """Session state can contain None values before analysis runs."""
    state = extract_cinematic_state({
        "final_report": None,
        "profile": None,
        "tool_results": None,
        "preview_df": None,
        "metadata": None,
        "stage_log": None,
        "theme": "day",
    })
    assert state["theme"] == "day"
    assert state["dataset"]["name"] == "sample_dataset.csv"
    assert len(state["stages"]) == 7
    assert state["ml"]["best_model"] == "GradientBoosting"
    assert "findings" in state["synthesis"]


def test_extract_cinematic_state_with_real_profile_columns() -> None:
    """profile["columns"] is a LIST of column dicts, not a mapping.

    Both DatasetProfile.to_dict() (`[c.to_dict() for c in self.columns]`) and
    the sample-report demo state build it as a list. Every other test here
    passes profile=None, which lands on the `{}` fallback and hides the
    difference — so the list shape went unexercised.
    """
    state = extract_cinematic_state({
        "profile": {
            "quality_score": 92,
            "column_count": 4,
            "columns": [
                {"name": "tenure", "kind": "numeric", "dtype": "int64"},
                {"name": "charges", "kind": "numeric", "dtype": "float64"},
                {"name": "contract", "kind": "categorical", "dtype": "str"},
                {"name": "signed_at", "kind": "datetime", "dtype": "datetime64[ns]"},
            ],
        },
        "theme": "day",
    })
    assert state["dataset"]["quality_score"] == 92
    assert state["dataset"]["column_types"]["numeric"] == 2
    assert state["dataset"]["column_types"]["categorical"] == 1
    assert state["dataset"]["column_types"]["datetime"] == 1


def test_extract_cinematic_state_with_mapping_profile_columns() -> None:
    """A name->column mapping must keep working alongside the list shape."""
    state = extract_cinematic_state({
        "profile": {
            "quality_score": 88,
            "columns": {
                "tenure": {"kind": "numeric"},
                "contract": {"kind": "categorical"},
            },
        },
        "theme": "day",
    })
    assert state["dataset"]["quality_score"] == 88
    assert state["dataset"]["column_types"]["numeric"] == 1
    assert state["dataset"]["column_types"]["categorical"] == 1


# ── Sorrel ─────────────────────────────────────────────────────────────────────────────────────
OLD_LEDGER = ("#241c14", "#1c1610", "#f6eedf", "#f0a24a", "#e2685a", "#7fb77e", "#f7eedd", "#fffbf2",
              "#a34f20", "#a33526", "#3a2b1e", "#2e2015", "rgba(240, 162, 74", "rgba(79, 195, 247")


def test_both_modes_tokens_are_written_into_the_stylesheet_from_design_tokens() -> None:
    from src.core import design_tokens

    doc = build_cinematic_document(theme="night")
    assert "__TOKENS_NIGHT__" not in doc and "__TOKENS_DAY__" not in doc
    for mode in ("day", "night"):
        pal = design_tokens.palette(mode)  # type: ignore[arg-type]
        for key in ("stock", "pen", "accent_text", "danger_text", "rule", "sheet"):
            assert f"--{key.replace('_', '-')}: {pal[key]};" in doc, (mode, key)
        assert f"--card-border: {pal['rule']};" in doc


def test_state_carries_both_palettes_so_the_theme_toggle_never_needs_a_typed_colour() -> None:
    state = _state_of(build_cinematic_document(theme="day"))
    assert state["palettes"] == CINEMATIC_PALETTES
    assert state["palette"] == CINEMATIC_PALETTES["day"]
    assert {"accent_text", "danger_text", "margin", "paper_elevated"} <= set(state["palettes"]["night"])


def test_no_old_ledger_colour_or_face_is_left_in_the_assets() -> None:
    for name in ("cinematic_3d.html", "cinematic_3d.js"):
        text = _read_asset(name)
        for old in OLD_LEDGER:
            assert old not in text, (name, old)
        for face in ("Public Sans", "Bricolage", "Baloo", "Mukta", "IBM Plex"):
            assert face not in text, (name, face)
    html = _read_asset("cinematic_3d.html")
    assert "'Geist'" in html and "'Newsreader'" in html


def test_the_workspace_rules_hold_no_glow_blur_gradient_or_hover_lift() -> None:
    html = _read_asset("cinematic_3d.html")
    for banned in ("pen-glow", "backdrop-filter", "box-shadow", "gradient(", "text-shadow", "translateY(-2px)",
                   "pulse-dot", "@keyframes bounce", "border-radius: 999px"):
        assert banned not in html, banned
    assert "prefers-reduced-motion: reduce" in html and "prefers-reduced-transparency: reduce" in html
    js = _read_asset("cinematic_3d.js")
    assert "pen-glow" not in js and "boxShadow" not in js


def test_visible_names_say_sorrel_and_carry_no_overclaim() -> None:
    doc = build_cinematic_document(theme="night")
    assert "<title>Sorrel" in doc and '<div class="brand-title">Sorrel</div>' in doc
    visible = _read_asset("cinematic_3d.html") + _read_asset("cinematic_3d.js")
    for banned in ("DSA AGENT", "DSA Agent", "Ledger", "LEDGER", "ertified", "CERTIFIED",
                   "Zero-Hallucination", "holographic"):
        assert banned not in visible, banned
    assert "anifold" not in _read_asset("cinematic_3d.html")  # the identifier manifoldPlates is code, not copy
    for old_copy in ("Continuous Feature Manifolds", "STRATIFIED MANIFOLDS", "FEATURE MANIFOLDS", "04 Manifolds"):
        assert old_copy not in visible, old_copy


def test_dataset_and_llm_text_is_escaped_before_it_reaches_innerhtml() -> None:
    js = _read_asset("cinematic_3d.js")
    assert "export function esc(" in js
    for expr in ("${esc(c.pair || '')}", "${esc(mod.name)}", "${esc(f)}"):
        assert expr in js, expr
    assert "<li>${f}</li>" not in js and "${c.pair || ''}" not in js


def test_export_is_standalone_with_the_sorrel_variable_fonts_inlined(tmp_path: Path) -> None:
    content = export_cinematic_html(tmp_path / "walk.html", theme="day").read_text(encoding="utf-8")
    assert "url('./" not in content and 'url("./' not in content  # no relative font link survives
    assert "fonts.googleapis.com" not in content and "fonts.gstatic.com" not in content
    assert 'rel="stylesheet" href=' not in content
    faces = re.findall(r"@font-face\s*\{[^}]*\}", content)
    families = {re.search(r"font-family:\s*'([^']+)'", f).group(1) for f in faces}  # type: ignore[union-attr]
    assert families == {"Geist", "Geist Mono", "Newsreader"}
    assert len(faces) == 4 and all("data:font/woff2;base64," in f and "woff2-variations" in f for f in faces)
    assert all("unicode-range" in f for f in faces)
    assert "Baloo" not in content and "Mukta" not in content


def test_export_drops_a_face_whose_font_file_is_missing_rather_than_link_to_it(tmp_path: Path) -> None:
    from ui.cinematic_3d import _inline_export_fonts

    fonts = tmp_path / "static" / "vendor" / "fonts"
    fonts.mkdir(parents=True)
    (fonts / "geist-latin-wght-normal.woff2").write_bytes(b"wOF2")
    (fonts / "ledger-fonts.css").write_text(
        "@font-face { font-family: 'Geist'; font-weight: 100 900; "
        "src: url('./geist-latin-wght-normal.woff2') format('woff2-variations'); }\n"
        "@font-face { font-family: 'Newsreader'; font-weight: 200 800; "
        "src: url('./newsreader-latin-wght-normal.woff2') format('woff2-variations'); }\n"
        "@font-face { font-family: 'Baloo 2'; src: url('./baloo-2-500-latin.woff2'); }\n",
        encoding="utf-8",
    )
    style = _inline_export_fonts(tmp_path)
    assert "'Geist'" in style and "data:font/woff2;base64," in style
    assert "Newsreader" not in style and "Baloo" not in style and "url('./" not in style


def test_cinematic_text_pairs_meet_contrast_in_both_modes() -> None:
    def lum(hex_color: str) -> float:
        def ch(c: int) -> float:
            v = c / 255
            return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

        r, g, b = (ch(int(hex_color[i : i + 2], 16)) for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def ratio(a: str, b: str) -> float:
        hi, lo = sorted((lum(a), lum(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    for mode, pal in CINEMATIC_PALETTES.items():
        for fg, bg in (("ink", "sheet"), ("ink_2", "sheet"), ("accent_text", "sheet"), ("danger_text", "sheet_alt"),
                       ("positive", "accent_soft"), ("accent_text", "accent_soft"), ("ink", "stock"),
                       ("graphite", "stock"), ("accent_ink", "pen")):
            assert ratio(pal[fg], pal[bg]) >= 4.5, f"{mode}: {fg} on {bg} = {ratio(pal[fg], pal[bg]):.2f}"


def test_text_naming_a_placeholder_stays_text_and_never_splices_the_script_in() -> None:
    """The state is substituted last, so a dataset name that looks like a template slot is just a string."""
    trick = "__CINEMATIC_SCENE_SCRIPT__ and __FONT_LINKS__"
    state = extract_cinematic_state({"preview_name": trick, "theme": "night"})
    doc = build_cinematic_document(state)
    assert _state_of(doc)["dataset"]["name"] == trick
    assert doc.count("setAnimationLoop") == _read_asset("cinematic_3d.js").count("setAnimationLoop")
