"""The Sorrel token set: both modes carry the same keys, every value is a hex, and the CSS block is complete."""
import re

import pytest

from src.core.design_tokens import PALETTES, TOKEN_ORDER, css_root_block, palette

HEX = re.compile(r"^#[0-9a-f]{6}$")
#: Roles the Sorrel prototype added to the original Ledger set.
NEW_KEYS = ("ink_2", "ink_4", "accent_soft", "accent_ink", "accent_text", "danger_text", "bg_deep", "paper_elevated")


@pytest.mark.parametrize("mode", ["day", "night"])
def test_every_token_is_a_lowercase_hex(mode: str) -> None:
    for key, value in palette(mode).items():  # type: ignore[arg-type]
        assert HEX.match(value), f"{mode} {key} = {value!r}"


def test_both_modes_have_exactly_the_ordered_keys() -> None:
    assert set(PALETTES["day"]) == set(PALETTES["night"]) == set(TOKEN_ORDER)
    assert len(TOKEN_ORDER) == len(set(TOKEN_ORDER))


def test_new_prototype_roles_are_present() -> None:
    for key in NEW_KEYS:
        assert key in TOKEN_ORDER


@pytest.mark.parametrize("mode", ["day", "night"])
def test_css_block_declares_every_token_once(mode: str) -> None:
    block = css_root_block(mode)  # type: ignore[arg-type]
    names = re.findall(r"--([a-z0-9-]+):", block)
    assert names == [key.replace("_", "-") for key in TOKEN_ORDER]
    for key in NEW_KEYS:
        assert f"--{key.replace('_', '-')}: {palette(mode)[key]};" in block  # type: ignore[arg-type]


def test_text_roles_differ_from_fills_only_where_contrast_needs_it() -> None:
    # Day: the accent is dark enough to be text, so the roles agree. Night: it is not, so they must differ.
    assert PALETTES["day"]["accent_text"] == PALETTES["day"]["pen"]
    assert PALETTES["night"]["accent_text"] != PALETTES["night"]["pen"]
    assert PALETTES["night"]["danger_text"] != PALETTES["night"]["risk"]
