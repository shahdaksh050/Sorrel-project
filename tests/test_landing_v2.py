"""
Structure, SEO, token, 3D-engine and Streamlit-messaging checks for the live landing page
(ui/landing_component/index.html).

These used to run against a duplicate prototype in `frontend-landing/`, which has been
removed. What the prototype alone contained (pricing tiers, an FAQ, its FAQPage schema and
TypeScript definitions) is gone from the live page too, so those checks went with it.
Copy and claims are covered in test_landing.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "ui" / "landing_component" / "index.html"


def test_landing_section_stack() -> None:
    """The page keeps its full section stack and the ids the navigation points at."""
    content = INDEX_HTML.read_text(encoding="utf-8")

    assert 'class="site-header"' in content
    assert "DSA AGENT" in content

    assert 'id="hero"' in content
    assert "hero-enter-btn" in content
    assert "hero-explore-btn" in content

    assert 'id="workflow"' in content
    for stage in range(4):
        assert f'data-stage="{stage}"' in content

    assert 'id="comparison"' in content
    assert 'id="features"' in content
    assert 'id="cta"' in content
    assert "cta-launch-btn" in content
    assert 'class="site-footer"' in content


def test_landing_seo_and_jsonld_schema() -> None:
    content = INDEX_HTML.read_text(encoding="utf-8")

    assert 'name="description"' in content
    assert 'property="og:title"' in content
    assert 'property="og:description"' in content
    assert 'name="twitter:card"' in content
    assert 'rel="canonical"' in content

    match = re.search(r'<script type="application/ld\+json">(.*?)</script>', content, re.DOTALL)
    assert match is not None, "JSON-LD script block must exist"
    data = json.loads(match.group(1))
    assert "@graph" in data or "@context" in data
    types = {node.get("@type") for node in data.get("@graph", [])}
    assert "SoftwareApplication" in types
    assert "FAQPage" not in types, "the FAQ section is gone, so its schema must be too"


def test_landing_ledger_tokens() -> None:
    """The page's CSS tokens mirror ui.landing's LEDGER_TOKENS (which read design_tokens)."""
    from ui.landing import LEDGER_TOKENS_DAY, LEDGER_TOKENS_NIGHT

    content = INDEX_HTML.read_text(encoding="utf-8")

    assert "./fonts/landing-fonts.css" in content
    fonts = (INDEX_HTML.parent / "fonts" / "landing-fonts.css").read_text(encoding="utf-8")
    assert "font-family: 'Baloo 2'" in fonts and "font-family: 'Mukta'" in fonts
    assert "--heading: 'Baloo 2'" in content
    assert "--sans: 'Mukta'" in content

    assert f"--stock: {LEDGER_TOKENS_NIGHT['stock']}" in content
    assert f"--pen: {LEDGER_TOKENS_NIGHT['pen']}" in content
    assert f"--ink: {LEDGER_TOKENS_NIGHT['ink']}" in content

    assert "html.theme-day" in content
    assert f"--stock: {LEDGER_TOKENS_DAY['stock']}" in content
    assert f"--pen: {LEDGER_TOKENS_DAY['pen']}" in content
    assert f"--ink: {LEDGER_TOKENS_DAY['ink']}" in content


def test_landing_3d_engine_invariants() -> None:
    content = INDEX_HTML.read_text(encoding="utf-8")

    assert "setAnimationLoop" in content  # a single clock
    assert "Math.exp" in content  # frame-rate independent damping
    assert "calculateScrollProgress" in content
    assert "currentScalar" in content
    assert "SRGBColorSpace" in content
    assert "ACESFilmicToneMapping" in content
    assert "TOKEN_COUNT = 720" in content
    assert "THREE.InstancedMesh" in content
    assert "prefers-reduced-motion" in content


def test_landing_streamlit_messaging() -> None:
    content = INDEX_HTML.read_text(encoding="utf-8")

    assert "streamlit:componentReady" in content
    assert "setComponentValue" in content
    assert "HIDDEN_ENTER" in content
    assert "keydown" in content
    assert "Space" in content
