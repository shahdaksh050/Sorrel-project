"""Regression checks for the native Streamlit presentation primitives."""
from __future__ import annotations

from pathlib import Path

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
