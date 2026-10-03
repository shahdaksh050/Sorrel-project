"""Regression checks for the native Streamlit presentation primitives."""
from __future__ import annotations

from pathlib import Path

import pytest
from ui.components.cards import render_finding_card

ROOT = Path(__file__).resolve().parents[1]


def test_finding_card_uses_semantic_classes_and_escapes_text() -> None:
    """Finding content stays escaped while visual treatment lives in the stylesheet."""
    card = render_finding_card(
        {
            "finding_id": "sales",
            "headline": "Sales < target",
            "detail": "A & B need review",
        },
        {"sales"},
        is_primary=True,
    )

    assert 'class="finding-card full-width"' in card
    assert 'class="finding-detail"' in card
    assert 'class="finding-chart-note"' in card
    assert "Sales &lt; target" in card
    assert "A &amp; B need review" in card
    assert "style=" not in card


def test_workspace_styles_keep_motion_and_embedding_scoped() -> None:
    """Decorative scripts and iframe chrome cannot leak into native Streamlit UI."""
    styles = (ROOT / "ui" / "styles.py").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "--mono:" in styles
    assert "--ease-in-out:" in styles
    assert ".finding-card.full-width" in styles
    assert ".st-key-plate iframe" in styles
    assert "inject_micro_interactions()" not in app


# ── The Sorrel workspace: shape plus words, folded figures, shelf rows ────────────────────────────────


def test_a_flagged_gauge_always_says_so_in_words() -> None:
    from ui.components.cards import gauge

    assert '<div class="s">! Some cells are empty</div>' in gauge("Missing cells", "5", "Some cells are empty", flag=True)
    assert '<div class="s">! Worth a look</div>' in gauge("Failed", "2", flag=True)
    assert 'class="s"' not in gauge("Rows", "9")


def test_verdict_marks_reuse_the_check_row_shapes() -> None:
    from ui.components.cards import state_label

    from src.core.audited_entry import GLYPH

    assert state_label("held") == f"{GLYPH['pass']} Held up"
    assert state_label("needs_more") == f"{GLYPH['fail']} Needs more data"
    assert state_label("unchecked") == f"{GLYPH['neutral']} Not checked"


def test_exact_figures_are_folded_behind_the_plain_sentence_and_escaped() -> None:
    from ui.components.cards import analyst_figures, render_evidence_html

    finding = {
        "finding_id": "f1",
        "headline": "A and B move together",
        "detail": "Sales rise with spend.",
        "effect": 0.9311,
        "effect_kind": "r",
        "p_value": 0.00001,
        "confidence": 0.82,
        "source_tool": "correlation_analysis<x>",
    }
    figures = analyst_figures(finding)
    assert "r = 0.9311" in figures and "p = <0.001" in figures
    assert "confidence" not in figures  # a ranking weight, not statistical confidence
    out = render_evidence_html(finding)
    assert '<details class="tech-note"><summary>Details for analysts</summary>' in out
    assert "correlation analysis&lt;x&gt;" in out and "<x>" not in out
    assert out.index("evidence-detail") < out.index("tech-note")  # the plain sentence leads
    assert analyst_figures({"headline": "bare"}) == ""
    assert "tech-note" not in render_evidence_html({"headline": "bare"})


def test_a_chart_shows_the_checks_that_ran_and_where_it_came_from_or_nothing() -> None:
    from ui.components.cards import chart_provenance_html

    assert chart_provenance_html(None) == ""
    assert chart_provenance_html({"evidence": {}}) == ""
    out = chart_provenance_html({"evidence": {}, "source_tool": "cluster_data"})
    assert "Worked out by the cluster data step of this run." in out
    assert "check-row" not in out  # no check ran, so no tick is invented


def test_shelf_rows_carry_type_purpose_size_and_state_and_escape_the_name() -> None:
    from ui.components.cards import artifact_info_html, format_size

    ready = artifact_info_html("HTML", "report<1>.html", "A clean reading view.", 143 * 1024, "Ready")
    assert 'class="ty">HTML<' in ready and "report&lt;1&gt;.html" in ready
    assert 'class="sz">143 KB<' in ready and 'class="avail">Ready<' in ready
    off = artifact_info_html("HTML", "3D presentation", "Replays the run.", None, "Not prepared yet")
    assert 'class="avail off">Not prepared yet<' in off and 'class="sz"' not in off
    assert "\n" not in ready and "style=" not in ready
    assert [format_size(n) for n in (0, 1023, 1024, 1536, 20 * 1024, 5 * 1024 * 1024)] == [
        "0 B", "1023 B", "1.0 KB", "1.5 KB", "20 KB", "5.0 MB",
    ]


def test_audit_heads_and_agent_badges_escape_and_carry_shapes() -> None:
    from ui.components.cards import audit_head_html, render_agent_grid

    out = audit_head_html("Models <b>", "How they scored & why")
    assert "Models &lt;b&gt;" in out and "scored &amp; why" in out
    grid = render_agent_grid([("1", "done", ""), ("2", "active", ""), ("3", "error", "")])
    for word in ("✓ Done", "● Working", "! Needs attention", "○ Waiting"):
        assert word in grid


def _media_blocks(css: str, header: str) -> list[str]:
    """Every `header { ... }` block in `css`, braces matched, as full text (header included)."""
    blocks: list[str] = []
    start = css.find(header)
    while start != -1:
        depth, i = 0, css.index("{", start)
        while True:
            depth += css[i] == "{"
            depth -= css[i] == "}"
            i += 1
            if depth == 0:
                break
        blocks.append(css[start:i])
        start = css.find(header, i)
    return blocks


def test_the_workspace_stylesheet_has_no_motion_gradient_or_red_for_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import re

    import streamlit as st
    from ui import styles as ui_styles

    emitted: list[str] = []
    monkeypatch.setattr(st, "markdown", lambda body, **kwargs: emitted.append(body))
    monkeypatch.setattr(st, "get_option", lambda key: "")
    monkeypatch.setattr(st, "session_state", {"theme": "night"})
    ui_styles.inject_theme_css()
    css = emitted[0]
    assert "gradient" not in css and "drop-shadow" not in css and "blur(" not in css
    # Motion is allowed as reveals only (expand, collapse, fade): inside the one block that applies when the
    # browser has NOT asked for reduced motion, at 250 ms or less, never looping, never a hover lift.
    gated = _media_blocks(css, "@media (prefers-reduced-motion: no-preference)")
    assert gated, "reveal motion must live in a reduced-motion-gated block"
    ungated = css
    for block in gated:
        ungated = ungated.replace(block, "")
    assert not re.search(r"transition\s*:", ungated) and "@keyframes" not in ungated
    assert not re.search(r"animation\s*:\s*(?!none)\S", ungated)  # `animation: none` is fine, any other is not
    motion = " ".join(gated)
    assert "infinite" not in motion and "gradient" not in motion
    for amount, unit in re.findall(r"(?<![\w-])(\d+(?:\.\d+)?)(ms|s)\b", motion):
        assert float(amount) * (1000 if unit == "s" else 1) <= 250, f"{amount}{unit} is too slow for a reveal"
    for token in ("fast", "base"):  # the two durations the gated block may use
        assert int(re.search(rf"--dur-{token}:\s*(\d+)ms", css).group(1)) <= 250  # type: ignore[union-attr]
    assert "--dur-slow" not in motion
    assert not re.search(r":hover[^{{}}]*\{{[^}}]*(?:transform|translate|scale|box-shadow)", css)
    # and reduced motion still wins everywhere
    assert "@media (prefers-reduced-motion: reduce)" in css and "animation-duration: .01ms" in css
    assert "--risk-text:   var(--danger-text)" in css  # one text colour for "may not hold", both modes
    assert "ff8a8a" not in css
    # Error and warning alerts are amber, never the "may not hold" colour.
    start = css.index('stAlertContentError"]),')
    alert = css[start : css.index("}", start)]
    assert "var(--accent)" in alert and "var(--risk)" not in alert


def test_visible_naming_is_sorrel() -> None:
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'page_title="Sorrel"' in app
    assert 'class="brand-name">Sorrel<' in app and 'class="side-word">Sorrel<' in app
    assert "Agentic Data Analysis" not in app
    assert "Sorrel is the working name of DSA Agent" in app


def test_the_stepper_is_a_list_of_seven_with_a_shape_and_a_word_each() -> None:
    from ui.components.cards import STAGE_DEFS, render_stepper

    out = render_stepper([("1", "done", ""), ("2", "active", ""), ("3", "error", ""), ("4", "skipped", "")])
    assert out.count('role="listitem"') == len(STAGE_DEFS) == 7
    # Plain divs with list roles, not ol/li: Streamlit styles those, and an ol collapsed to a narrow column.
    assert 'role="list"' in out and "<ol" not in out and "<li" not in out
    for word in ("✓ Done", "● Working", "! Stopped", "– Skipped", "○ Waiting"):
        assert word in out
    assert 'aria-label="Analysis steps, 1 of 7 done"' in out
    # one polite sentence names the stage in progress; the row itself is not a live region
    assert out.count('role="status"') == 1 and "Working on step 2 of 7" in out
    assert "aria-live" not in out
    assert 'role="status"' not in render_stepper([("1", "done", "")])


def test_the_stepper_names_every_step_and_escapes_what_it_is_given() -> None:
    from ui.components.cards import STAGE_DEFS, render_stepper

    out = render_stepper([("1", "done", "<script>x</script>")])
    for _, name in STAGE_DEFS:
        assert name in out
    assert "<script" not in out


def test_team_cards_are_one_line_until_opened() -> None:
    from ui.components.cards import render_agent_grid

    grid = render_agent_grid([("1", "done", "")])
    summary = grid.split("<summary")[1].split("</summary>")[0]
    assert "agent-line" in summary and "agent-desc" not in summary and "agent-metric" not in summary
    assert grid.count("agent-more") >= 8  # the description and what it found open with each card


def test_a_chart_gets_one_evidence_line_with_a_verdict_chip_and_never_repeats_the_finding() -> None:
    from ui.components.cards import _same_sentence, evidence_line_html

    held = {"headline": "Total duration is <b>rising</b>.", "evidence": {"p_adj": 0.001, "effect": 0.4}}
    out = evidence_line_html(held)
    assert 'class="verdict-chip' in out and 'aria-hidden="true"' in out  # a shape and a word
    assert "&lt;b&gt;rising&lt;/b&gt;" in out and "<b>rising" not in out
    assert "Not checked" in evidence_line_html({"headline": "x", "evidence": {}})
    assert _same_sentence("Total duration is rising.", "total  duration is rising")
    assert not _same_sentence("Total duration is rising.", "Total duration is falling.")
    assert not _same_sentence("", "")


def test_panels_in_a_row_get_one_height_unless_the_chart_needs_more() -> None:
    from ui.components.cards import _fit_height

    bar = {"mark": "bar", "encoding": {}}
    assert _fit_height(bar, 260)["height"] == 260
    assert _fit_height({**bar, "height": 120}, 260)["height"] == 260  # shorter ones are lifted to match
    assert _fit_height({**bar, "height": 420}, 260)["height"] == 420  # a tall chart keeps its height
    stepped = {**bar, "height": {"step": 18}}
    assert _fit_height(stepped, 260) is stepped  # sized by step: left alone
    assert _fit_height({"facet": {}, "spec": {}}, 260) == {"facet": {}, "spec": {}}
    assert _fit_height({"hconcat": [], "mark": "bar"}, None) == {"hconcat": [], "mark": "bar"}
    # compound charts and row/column facets are sized per cell, so a panel height would make them huge
    for odd in ({"hconcat": [], "mark": "bar"}, {"vconcat": [], "layer": []}, {"mark": "bar", "encoding": {"row": {}}},
                {"mark": "bar", "encoding": {"column": {}}}):
        assert _fit_height(odd, 260) is odd
