"""
Unit tests for the Next-Gen Landing Page architecture (frontend-landing/).
Validates standards from /landing-page-generator, /frontend-dev-guidelines, and 3d-ui-skill.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANDING_DIR = ROOT / "frontend-landing"
INDEX_HTML = LANDING_DIR / "index.html"
TYPES_TS = LANDING_DIR / "src" / "types" / "landing.ts"
THREE_STAGE_TS = LANDING_DIR / "src" / "three" / "ThreeStage.ts"


def test_v2_landing_files_exist() -> None:
    """Ensure all core architectural files for frontend-landing exist."""
    assert INDEX_HTML.exists(), "frontend-landing/index.html must exist"
    assert TYPES_TS.exists(), "frontend-landing/src/types/landing.ts must exist"
    assert THREE_STAGE_TS.exists(), "frontend-landing/src/three/ThreeStage.ts must exist"
    assert (LANDING_DIR / "README.md").exists(), "frontend-landing/README.md must exist"


def test_v2_landing_conversion_sections() -> None:
    """Validates full section stack required by /landing-page-generator."""
    content = INDEX_HTML.read_text(encoding="utf-8")

    # 1. Header & Brand
    assert 'class="site-header"' in content
    assert "DSA AGENT" in content

    # 2. Hero Section (AIDA Attention)
    assert 'id="hero"' in content
    assert "hero-enter-btn" in content
    assert "hero-explore-btn" in content
    assert "Stop debugging" in content
    assert "hallucinated math" in content

    # 3. Interactive 3D Journey (7-Stage Timeline)
    assert 'id="workflow"' in content
    assert 'data-stage="0"' in content
    assert 'data-stage="1"' in content
    assert 'data-stage="2"' in content
    assert 'data-stage="3"' in content

    # 4. Comparison Section (Problem vs Solution)
    assert 'id="comparison"' in content
    assert "Chat LLMs &amp; Spreadsheets" in content or "Chat LLMs" in content
    assert "DSA Agent Autonomous Engine" in content

    # 5. Production Features Matrix
    assert 'id="features"' in content
    assert "Zero-Leakage Dataset Ingestion" in content
    assert "Stratified 5-Fold Cross-Validation" in content
    assert "Anti-Overfit Generalization Envelope" in content

    # 6. Pricing & Deployment Tiers
    assert 'id="pricing"' in content
    assert "pricing-toggle" in content
    assert "Community" in content
    assert "Workgroup" in content
    assert "Enterprise" in content

    # 7. FAQ Accordion Section
    assert 'id="faq"' in content
    assert "faq-button" in content
    assert "faq-answer" in content

    # 8. Closing CTA Banner & Footer
    assert 'id="cta"' in content
    assert "cta-launch-btn" in content
    assert 'class="site-footer"' in content


def test_v2_seo_and_jsonld_schemas() -> None:
    """Validates comprehensive SEO and structured data schema."""
    content = INDEX_HTML.read_text(encoding="utf-8")

    # Meta Tags
    assert 'name="description"' in content
    assert 'property="og:title"' in content
    assert 'property="og:description"' in content
    assert 'name="twitter:card"' in content
    assert 'rel="canonical"' in content

    # JSON-LD Schemas (SoftwareApplication + FAQPage)
    assert 'type="application/ld+json"' in content
    assert "SoftwareApplication" in content
    assert "FAQPage" in content

    # Extract and parse JSON-LD
    schema_match = re.search(r'<script type="application/ld\+json">(.*?)</script>', content, re.DOTALL)
    assert schema_match is not None, "JSON-LD script block must exist"
    data = json.loads(schema_match.group(1))
    assert "@graph" in data or "@context" in data


def test_v2_warm_ledger_tokens() -> None:
    """Validates adherence to DESIGN.md Warm Ledger specifications."""
    # Q1 — assert against ui.landing's LEDGER_TOKENS_DAY/NIGHT (mirrors
    # DESIGN.md) instead of re-typed hex literals; see test_landing.py's
    # sibling test for the same fix and rationale.
    from ui.landing import LEDGER_TOKENS_DAY, LEDGER_TOKENS_NIGHT

    content = INDEX_HTML.read_text(encoding="utf-8")

    # Typography
    assert "family=Baloo+2" in content
    assert "family=Mukta" in content
    assert "--heading: 'Baloo 2'" in content
    assert "--sans: 'Mukta'" in content

    # Night Tokens
    assert f"--stock: {LEDGER_TOKENS_NIGHT['stock']}" in content
    assert f"--pen: {LEDGER_TOKENS_NIGHT['pen']}" in content
    assert f"--ink: {LEDGER_TOKENS_NIGHT['ink']}" in content

    # Day Tokens
    assert "html.theme-day" in content
    assert f"--stock: {LEDGER_TOKENS_DAY['stock']}" in content
    assert f"--pen: {LEDGER_TOKENS_DAY['pen']}" in content
    assert f"--ink: {LEDGER_TOKENS_DAY['ink']}" in content


def test_v2_3d_engine_invariants() -> None:
    """Validates strict adherence to the 12 Invariants of 3d-ui-skill."""
    content = INDEX_HTML.read_text(encoding="utf-8")

    # Invariant 1: Single clock via setAnimationLoop
    assert "setAnimationLoop" in content

    # Invariant 2: Frame-rate independent dt damping
    assert "Math.exp" in content

    # Invariant 3: Continuous progress scalar (native scroll)
    assert "calculateScrollProgress" in content
    assert "currentScalar" in content

    # Invariant 6: Output color space and ACES tonemapping
    assert "SRGBColorSpace" in content
    assert "ACESFilmicToneMapping" in content

    # Invariant 7: Preallocated instanced mesh for 720 data tokens
    assert "TOKEN_COUNT = 720" in content
    assert "THREE.InstancedMesh" in content

    # Invariant 11: Accessibility prefers-reduced-motion
    assert "prefers-reduced-motion" in content


def test_v2_streamlit_messaging_integration() -> None:
    """Validates two-way messaging with parent Streamlit workspace."""
    content = INDEX_HTML.read_text(encoding="utf-8")

    assert "streamlit:componentReady" in content
    assert "setComponentValue" in content
    assert "HIDDEN_ENTER" in content
    assert "keydown" in content
    assert "Space" in content


def test_v2_typescript_definitions() -> None:
    """Validates strict TypeScript types for /frontend-dev-guidelines compliance."""
    content = TYPES_TS.read_text(encoding="utf-8")

    assert "export type ThemeMode" in content
    assert "export interface ThemePalette" in content
    assert "export interface FeatureCard" in content
    assert "export interface PricingPlan" in content
    assert "export interface FaqItem" in content
    assert "export interface ThreeMotionTokens" in content
    # Strict typing: no raw any types
    assert not re.search(r"\bany\b", content), "TypeScript definitions should avoid 'any' type"
