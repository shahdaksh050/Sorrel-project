"""The Streamlit console loads its fonts from this repo, not from a font CDN."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import streamlit as st
from ui import styles

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "static" / "vendor" / "fonts"


def test_styles_module_has_no_remote_font_reference() -> None:
    source = (ROOT / "ui" / "styles.py").read_text(encoding="utf-8")
    assert "fonts.googleapis" not in source
    assert "fonts.gstatic" not in source


def test_static_serving_is_enabled_in_the_config() -> None:
    with (ROOT / ".streamlit" / "config.toml").open("rb") as fh:
        config = tomllib.load(fh)
    assert config["server"]["enableStaticServing"] is True


def test_font_css_url_default_and_with_a_base_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(st, "get_option", lambda key: "")
    assert styles.font_css_url() == "/app/static/vendor/fonts/ledger-fonts.css"
    monkeypatch.setattr(st, "get_option", lambda key: "/dsa/")
    assert styles.font_css_url() == "/dsa/app/static/vendor/fonts/ledger-fonts.css"


def test_injected_css_imports_the_local_stylesheet(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[str] = []
    monkeypatch.setattr(st, "markdown", lambda body, **kwargs: emitted.append(body))
    monkeypatch.setattr(st, "get_option", lambda key: "")
    monkeypatch.setattr(st, "session_state", {"theme": "day"})
    styles.inject_theme_css()
    css = "\n".join(emitted)
    assert "@import url('/app/static/vendor/fonts/ledger-fonts.css');" in css
    assert "fonts.googleapis" not in css and "fonts.gstatic" not in css


def test_every_font_file_the_stylesheet_names_exists_and_is_local() -> None:
    css = (FONT_DIR / "ledger-fonts.css").read_text(encoding="utf-8")
    urls = re.findall(r"url\(['\"]?([^'\")]+)['\"]?\)", css)
    assert urls, "the stylesheet should declare font files"
    for url in urls:
        assert not url.startswith(("http:", "https:", "//")), f"remote font URL: {url}"
        assert (FONT_DIR / url).resolve().is_file(), f"missing font file: {url}"


def test_the_families_the_console_uses_are_declared() -> None:
    css = (FONT_DIR / "ledger-fonts.css").read_text(encoding="utf-8")
    assert "font-family: 'Baloo 2'" in css and "font-family: 'Mukta'" in css


def test_static_folder_holds_only_public_assets() -> None:
    """Static serving exposes the whole folder to every visitor."""
    allowed = {".woff2", ".css", ".js", ".json"}
    files = [p for p in (ROOT / "static").rglob("*") if p.is_file()]
    assert files
    other = [p for p in files if p.suffix not in allowed]
    # The only other files are the licence texts that ship with vendored libraries.
    assert all(p.suffix == ".txt" and (p.name.upper().startswith("LICENSE") or p.name.upper() == "OFL.TXT") for p in other), [
        str(p.relative_to(ROOT)) for p in other
    ]


def test_vendored_three_in_static_matches_the_landing_copy() -> None:
    """The plate and the landing page load the same Three.js build, from two places."""
    static = (ROOT / "static" / "vendor" / "three" / "three.module.js").read_bytes()
    landing = (ROOT / "ui" / "landing_component" / "vendor" / "three.module.js").read_bytes()
    assert static == landing
