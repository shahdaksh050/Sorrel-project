"""The Streamlit theme file must carry the same colours as src/core/design_tokens.py."""
from __future__ import annotations

import importlib.util
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
def theme() -> dict[str, object]:
    with CONFIG.open("rb") as fh:
        loaded = tomllib.load(fh)
    return dict(loaded["theme"])


@pytest.mark.parametrize(("section", "mode"), [("light", "day"), ("dark", "night")])
def test_every_mapped_key_equals_its_token(
    theme: dict[str, object], section: str, mode: str
) -> None:
    block = theme[section]
    assert isinstance(block, dict)
    tokens = palette(mode)  # type: ignore[arg-type]
    for key, token in script.THEME_KEY_TOKENS.items():
        assert block[key].lower() == tokens[token].lower(), f"{section}.{key} drifted from {token}"


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
    edited = text.replace('baseRadius = "14px"', 'baseRadius = "9px"')
    assert 'baseRadius = "9px"' in script.sync(edited)


def test_missing_markers_raise() -> None:
    with pytest.raises(ValueError, match="markers"):
        script.sync("[theme]\nfont = 'x'\n")


def test_non_colour_settings_are_kept(theme: dict[str, object]) -> None:
    for key in ("font", "headingFont", "codeFont", "baseRadius", "buttonRadius", "chartCategoricalColors"):
        assert key in theme


def test_no_reference_to_the_deleted_design_doc() -> None:
    assert "DESIGN.md" not in CONFIG.read_text(encoding="utf-8")
