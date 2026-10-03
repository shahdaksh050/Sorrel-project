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


def test_every_stage_status_is_a_shape_and_a_word() -> None:
    from ui.components.cards import stage_card

    expected = {
        "done": "✓ Done",
        "active": "● Working",
        "skipped": "– Skipped",
        "error": "! Stopped",
        "pending": "○ Waiting",
    }
    for status, word in expected.items():
        out = stage_card("3", "Running the Numbers", status)
        assert f'<span class="st">{word}</span>' in out
        assert '<span class="sc-num">03</span>' in out


def test_the_timeline_lists_all_seven_stages_and_announces_only_the_active_one() -> None:
    from ui.components.cards import STAGE_DEFS, render_steps_list

    seen: set[str] = set()
    out = render_steps_list([("1", "done", ""), ("2", "active", "reading <b>")], seen)
    assert out.count('<li class="sc') == len(STAGE_DEFS) == 7
    assert out.count('role="status"') == 1
    assert "Working on step 2 of 7: Understanding Your Question" in out
    assert "reading &lt;b&gt;" in out and "reading <b>" not in out
    assert "aria-live" not in out
    assert seen == {"1", "2"}
    assert 'role="status"' not in render_steps_list([("1", "done", "")], set())


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
    assert not re.search(r"transition\s*:", css)
    assert "@keyframes" not in css
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


def test_the_stepper_is_one_even_row_of_seven_with_a_shape_and_a_word_each() -> None:
    from ui.components.cards import STAGE_DEFS, render_stepper

    out = render_stepper([("1", "done", ""), ("2", "active", ""), ("3", "error", ""), ("4", "skipped", "")])
    assert out.count('<li class="step') == len(STAGE_DEFS) == 7
    for word in ("✓ Done", "● Working", "! Stopped", "– Skipped", "○ Waiting"):
        assert word in out
    assert 'aria-label="Analysis steps, 1 of 7 done"' in out
    # one polite sentence names the stage in progress; the row itself is not a live region
    assert out.count('role="status"') == 1 and "Working on step 2 of 7" in out
    assert "aria-live" not in out
    assert 'role="status"' not in render_stepper([("1", "done", "")])


def test_step_notes_show_only_steps_that_reported_and_escape_them() -> None:
    from ui.components.cards import render_step_notes

    out = render_step_notes([("1", "done", "8,789 rows <b>"), ("2", "active", "")])
    assert "8,789 rows &lt;b&gt;" in out and "<b>x" not in out
    assert out.count("<li>") == 1 and "Reading Your File" in out
    assert "appears here as it finishes" in render_step_notes([("1", "pending", "")])


def test_team_cards_are_one_line_until_opened() -> None:
    from ui.components.cards import render_agent_grid

    grid = render_agent_grid([("1", "done", "")])
    summary = grid.split("<summary")[1].split("</summary>")[0]
    assert "agent-line" in summary and "agent-desc" not in summary and "agent-metric" not in summary
    assert grid.count("agent-more") >= 8  # the description and what it found open with each card
