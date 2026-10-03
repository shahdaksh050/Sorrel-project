"""Tests for the Landing Page experience (ui/landing.py and ui/landing_component/index.html)."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "ui" / "landing_component" / "index.html"


def test_landing_index_html_exists() -> None:
    assert INDEX_HTML.exists(), "ui/landing_component/index.html must exist"
    content = INDEX_HTML.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    # The product is named Sorrel on screen; DSA Agent survives only as its working name.
    assert "<title>Sorrel" in content
    assert 'class="brand-name">Sorrel<' in content
    assert "working name of DSA Agent" in content


def test_landing_typography_tokens() -> None:
    content = INDEX_HTML.read_text(encoding="utf-8")
    # Sorrel typography: Geist (interface and headlines), Newsreader (serif accents, wordmark), Geist Mono.
    # Served from ./fonts beside the page, not from Google (see test_landing_works_offline).
    assert "./fonts/landing-fonts.css" in content
    assert "family=Geist" not in content
    assert "--heading: 'Geist'" in content
    assert "--sans: 'Geist'" in content
    assert "--serif: 'Newsreader'" in content
    assert "--mono: 'Geist Mono'" in content


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

    # 4. Trust signals: a plain statement of where the data goes (the page used
    #    to claim "runs entirely on your machine", which a hosted server cannot).
    assert "trust-list" in content
    assert 'id="data-path"' in content

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


class _VisibleText(HTMLParser):
    """Text a visitor can read: no script, style or comments, plus aria-label and title attributes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        for key, value in attrs:
            if key in ("aria-label", "title", "alt", "placeholder") and value:
                self.chunks.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self.chunks.append(" ".join(data.split()))


def _visible_text() -> str:
    parser = _VisibleText()
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
    return "\n".join(parser.chunks)


#: Claims and jargon the page must not make: the runtime cannot prove them, or they
#: mean nothing to a reader. (Class names, ids and comments may still use some of
#: these words; only text a visitor can read is checked.)
_BANNED_PHRASES = (
    "swarm", "synapse", "manifold", "undulation", "certified", "telemetry",
    "never leaves", "runs entirely on your machine", "100% local", "zero cloud",
    "cloud transmissions", "air-gapped", "wcag", "compliance", "cryptographic",
    "formal mathematical verification", "guard status", "operational",
    "rlm v2", "high risk", "no credit card",
)


def test_visible_text_makes_no_unsupported_claims() -> None:
    text = _visible_text().lower()
    found = [phrase for phrase in _BANNED_PHRASES if phrase in text]
    assert not found, f"visible landing text still contains: {found}"


def test_data_path_statement_is_honest_and_complete() -> None:
    text = _visible_text()
    statement = text[text.index("Where your data goes:") :].split("\n", 1)[0]
    assert "processed on the server" in statement
    assert "deleted when you start a new analysis" in statement
    assert "AI provider you choose" in statement and "switch AI off" in statement
    assert "academic project" in statement.lower()
    assert "sensitive data" in statement
    # No retention or compliance promise.
    for promise in ("never stored", "gdpr", "hipaa", "soc 2", "retention"):
        assert promise not in statement.lower()


def test_example_findings_are_labelled_as_an_illustration() -> None:
    """The findings shown are a real run on a public table, labelled as such, and the 3D scene is
    labelled as an illustration; neither can be mistaken for the visitor's own results."""
    text = _visible_text()
    assert "These come from a real run on a public air-quality table" in text
    assert "not from your data" in text
    assert "illustration, not your data" in text
    # The verdict counts exactly the entries shown.
    assert "From this run: 1 of the 2 findings shown held up" in text
    assert INDEX_HTML.read_text(encoding="utf-8").count('class="audited-entry"') == 2


def test_no_pricing_or_faq_remnants() -> None:
    content = INDEX_HTML.read_text(encoding="utf-8")
    for selector in (".pricing-", ".faq-", ".plan-price", ".popular-flag", ".discount-pill"):
        assert selector not in content, f"dead CSS left behind: {selector}"
    visible = _visible_text().lower()
    for word in ("pricing", "faq", "per month", "annual"):
        assert word not in visible


def test_streamlit_bridge_is_intact() -> None:
    content = INDEX_HTML.read_text(encoding="utf-8")
    assert "setComponentValue({ enter: true, theme: currentTheme })" in content
    assert "setComponentValue({ enter: false, theme: currentTheme })" in content
    for element_id in ("hero-enter-btn", "enter-btn", "cta-launch-btn", "nav-launch-btn", "btn-theme-toggle"):
        assert f'id="{element_id}"' in content


# ── The page must work with no network and when the 3D scene cannot start ────────────────────
COMPONENT = INDEX_HTML.parent


def test_landing_works_offline() -> None:
    """Regression: the page imported Three.js from a CDN inside the same module that told Streamlit
    it was ready, so with the CDN unreachable (an offline demo) Streamlit showed its loading
    skeleton forever. Everything the page needs now ships beside it."""
    content = INDEX_HTML.read_text(encoding="utf-8")
    for host in ("cdn.jsdelivr.net", "cdnjs.cloudflare.com", "fonts.googleapis.com", "fonts.gstatic.com", "unpkg.com"):
        assert host not in content, f"the landing page still loads from {host}"
    assert "from './vendor/three.module.js'" in content
    assert 'src="./vendor/anime.min.js"' in content


def test_vendored_libraries_and_fonts_are_present_with_licenses() -> None:
    vendor = COMPONENT / "vendor"
    assert (vendor / "three.module.js").stat().st_size > 500_000
    assert (vendor / "anime.min.js").stat().st_size > 10_000
    for licence in ("LICENSE-three.txt", "LICENSE-animejs.txt"):
        assert "MIT" in (vendor / licence).read_text(encoding="utf-8")
    # three.module.js must be one self-contained file: a relative import would 404 in the iframe.
    assert not re.search(r"^\s*import\s", (vendor / "three.module.js").read_text(encoding="utf-8"), re.M)
    css = (COMPONENT / "fonts" / "landing-fonts.css").read_text(encoding="utf-8")
    files = re.findall(r"url\('\./([^']+)'\)", css)
    assert files and all((COMPONENT / "fonts" / name).is_file() for name in files)
    assert "http" not in css


def test_the_streamlit_bridge_does_not_depend_on_the_3d_module() -> None:
    """The classic script must run before, and independent of, the ES module."""
    content = INDEX_HTML.read_text(encoding="utf-8")
    bridge = content.index("function dsaBridge()")
    module = content.index('<script type="module">')
    assert bridge < module
    classic = content[bridge:module]
    assert "streamlit:componentReady" in classic
    assert "THREE" not in classic
    for element_id in ("hero-enter-btn", "enter-btn", "cta-launch-btn", "nav-launch-btn", "btn-theme-toggle"):
        assert element_id in classic
    # The fallback steps aside once the module has bound its own handlers.
    assert "window.__dsaModuleReady" in classic
    assert content.rindex("window.__dsaModuleReady = true;") > module
