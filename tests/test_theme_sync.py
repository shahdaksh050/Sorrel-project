"""The Streamlit theme file must carry the same colours as src/core/design_tokens.py."""
from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from src.core.design_tokens import palette

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".streamlit" / "config.toml"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sync_streamlit_theme", ROOT / "scripts" / "sync_streamlit_theme.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


@pytest.fixture(scope="module")
def parsed() -> dict[str, object]:
    with CONFIG.open("rb") as fh:
        return dict(tomllib.load(fh))


def test_every_mapped_key_equals_its_day_token(parsed: dict[str, object]) -> None:
    theme = parsed["theme"]
    assert isinstance(theme, dict)
    tokens = palette("day")
    for key, token in script.THEME_KEY_TOKENS.items():
        assert theme[key].lower() == tokens[token].lower(), f"theme.{key} drifted from {token}"


def test_native_theme_does_not_follow_the_browser_colour_scheme(parsed: dict[str, object]) -> None:
    """With [theme.light]/[theme.dark] present Streamlit picks its theme from the
    browser, which the app's Day/Night toggle cannot control: a dark browser then
    showed dark-themed widgets on the Day page. One flat native theme avoids that."""
    theme = parsed["theme"]
    assert isinstance(theme, dict)
    assert "light" not in theme and "dark" not in theme
    assert "base" not in theme


def test_a_stale_palette_value_is_detected() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    stale = text.replace(palette("day")["stock"], "#000000", 1)
    assert stale != text
    assert script.sync(stale) == text  # the script restores exactly the committed file


def test_committed_file_is_in_sync() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert script.sync(text) == text


def test_sync_leaves_hand_edited_keys_alone() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    edited = text.replace('baseRadius = "4px"', 'baseRadius = "9px"')
    assert 'baseRadius = "9px"' in script.sync(edited)


def test_missing_markers_raise() -> None:
    with pytest.raises(ValueError, match="markers"):
        script.sync("[theme]\nfont = 'x'\n")


def test_non_colour_settings_are_kept(parsed: dict[str, object]) -> None:
    theme = parsed["theme"]
    assert isinstance(theme, dict)
    for key in ("font", "headingFont", "codeFont", "baseRadius", "buttonRadius", "chartCategoricalColors"):
        assert key in theme


def test_no_reference_to_the_deleted_design_doc() -> None:
    assert "DESIGN.md" not in CONFIG.read_text(encoding="utf-8")


def test_app_defaults_to_day_and_never_reads_the_browser_theme() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert '_DEFAULT_THEME = "day"' in source
    assert "st.context.theme" not in source


def test_native_text_rules_use_tokens_only() -> None:
    styles = (ROOT / "ui" / "styles.py").read_text(encoding="utf-8")
    start = styles.index("/* ── Native widget text follows the page tokens")
    block = styles[start : styles.index("</style>", start)]
    assert "var(--graphite)" in block and "var(--ink)" in block
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)
