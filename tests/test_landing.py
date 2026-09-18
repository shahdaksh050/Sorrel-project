"""Tests for the Landing Page experience (ui/landing.py and ui/landing_component/index.html)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "ui" / "landing_component" / "index.html"


def test_landing_index_html_exists() -> None:
    assert INDEX_HTML.exists(), "ui/landing_component/index.html must exist"
    content = INDEX_HTML.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "DSA Agent" in content


def test_landing_typography_tokens() -> None:
    content = INDEX_HTML.read_text(encoding="utf-8")
    # Must use warm Ledger typography (Baloo 2 and Mukta)
    assert "family=Baloo+2" in content
    assert "family=Mukta" in content
    assert "--heading: 'Baloo 2'" in content
    assert "--sans: 'Mukta'" in content


def test_landing_day_and_night_themes() -> None:
    # Q1 — assert against ui.landing's LEDGER_TOKENS_DAY/NIGHT (which mirror
    # DESIGN.md) rather than re-typed hex literals, so this test and the
    # rendered page can't silently diverge again.
    from ui.landing import LEDGER_TOKENS_DAY, LEDGER_TOKENS_NIGHT

    content = INDEX_HTML.read_text(encoding="utf-8")
    # Night theme tokens
    assert f"--stock: {LEDGER_TOKENS_NIGHT['stock']}" in content
    assert f"--pen: {LEDGER_TOKENS_NIGHT['pen']}" in content
    assert f"--ink: {LEDGER_TOKENS_NIGHT['ink']}" in content

    # Day theme tokens
    assert "html.theme-day" in content
    assert f"--stock: {LEDGER_TOKENS_DAY['stock']}" in content
    assert f"--pen: {LEDGER_TOKENS_DAY['pen']}" in content
    assert f"--ink: {LEDGER_TOKENS_DAY['ink']}" in content

    # Theme toggle button
    assert "btn-theme-toggle" in content


def test_landing_beautiful_3d_invariants() -> None:
    content = INDEX_HTML.read_text(encoding="utf-8")
    # Invariant 1: Single clock via setAnimationLoop
    assert "setAnimationLoop" in content

    # Invariant 6: Tone mapping & Color space
    assert "ACESFilmicToneMapping" in content
    assert "SRGBColorSpace" in content

    # Invariant 7: Preallocated scratch vector and frame-rate damping
    assert "scratchVec" in content or "THREE.Vector3" in content
    assert "Math.exp" in content

    # Invariant 11: prefers-reduced-motion
    assert "prefers-reduced-motion" in content


def test_landing_fullpage_licensing_and_cta() -> None:
    content = INDEX_HTML.read_text(encoding="utf-8")
    # FullPage.js GPLv3 license key fix
    assert "licenseKey: 'gplv3-license'" in content

    # CTA messaging
    assert "setComponentValue" in content
    assert "HIDDEN_ENTER" in content
    assert "enter-btn" in content


def test_landing_cro_best_practices() -> None:
    """Validates CRO and SEO best practices required by /landing-page-generator."""
    content = INDEX_HTML.read_text(encoding="utf-8")

    # 1. SEO & OpenGraph meta tags
    assert 'name="description"' in content
    assert 'property="og:title"' in content
    assert 'name="twitter:card"' in content

    # 2. JSON-LD structured schema
    assert "application/ld+json" in content
    assert "SoftwareApplication" in content

    # 3. Above-the-fold dual CTA in Hero section
    assert "hero-enter-btn" in content
    assert "hero-explore-btn" in content

    # 4. Trust signals and microcopy
    assert "trust-list" in content
    assert "Local REPL Sandbox" in content
    assert "Generalization Guard" in content

    # 5. Keyboard navigation affordances
    assert "keydown" in content
    assert "Space" in content


def test_show_landing_page_signature_and_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    import streamlit as st
    import ui.landing as landing_mod

    # Mock session state
    monkeypatch.setattr(st, "session_state", {"theme": "day"})

    captured_kwargs: dict[str, Any] = {}

    def fake_component(*args: Any, **kwargs: Any) -> bool:
        captured_kwargs.update(kwargs)
        return True

    monkeypatch.setattr(landing_mod, "_landing_component", fake_component)

    result = landing_mod.show_landing_page()
    assert result is True
    assert captured_kwargs.get("theme") == "day"
    assert captured_kwargs.get("key") == "landing_narrative"
