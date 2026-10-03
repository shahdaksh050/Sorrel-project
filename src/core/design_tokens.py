"""
Design Tokens — the single Python source of truth for the "Sorrel" palette
(DESIGN.md), Day and Night.

Sorrel is the working name of this app's look: warm unbleached paper, near-black
ink, and one forest-green pen, with hairline rules and a deliberate amber for
"look twice" and a brick for "this may not hold". The values below are the ones
the landing and workspace prototype (`docs/prototypes/verdacert_dsa_preview.html`)
settled on, and every text pairing was checked against WCAG AA (4.5:1).

History worth knowing: before FrontendPlan.md item 2.1 five call sites each
hand-rolled their own copy of the same hex codes (`ui/styles.py`, `ui/landing.py`,
`ui/pipeline_3d.py`, `ui/cinematic_3d.py` and `src/core/html_report.py`'s CSS
block). They now import from here, and `src/core/chart_theme.py` builds the
*chart* inks on top of this module.

Placement note: this lives in `src/core`, not `ui/`, even though most consumers
are `ui/*.py`. A design token is not UI behaviour — `chart_theme.py` and
`html_report.py` (both `src/core`) need the same hexes, and `src/core` must never
depend on `ui/`.

Names kept, colours changed (the Sorrel port, see `docs/prototypes/SORREL_PORT_SPEC.md`):
the key and CSS-variable names below date from the earlier "Ledger" look and are
kept so every import and stylesheet keeps working. A few no longer match their
colour — `pen` is now the forest-green primary accent, `accent` is the amber
secondary, `stock` is the page paper — and the comment on each says what it is now.

Two roles needed new keys because raw `pen` and `risk` are fills, not text, in
Night mode: `accent_text` and `danger_text` are the AA-safe versions to use as
TEXT colour (Night `pen` on a Night card is only about 2.5:1; `accent_text` is
about 8:1). Use `pen` / `risk` for fills, borders and marks; use the `*_text`
keys when the colour is a glyph or a word.

The rule about red (DESIGN.md): `risk` and `danger_text` mean "this number or
finding may not hold". They are never a generic error colour.
"""
from __future__ import annotations

from typing import Literal

Mode = Literal["day", "night"]

__all__ = [
    "ACCENT_DAY",
    "ACCENT_INK_DAY",
    "ACCENT_INK_NIGHT",
    "ACCENT_NIGHT",
    "ACCENT_SOFT_DAY",
    "ACCENT_SOFT_NIGHT",
    "ACCENT_TEXT_DAY",
    "ACCENT_TEXT_NIGHT",
    "BG_DEEP_DAY",
    "BG_DEEP_NIGHT",
    "CODE_BG_DAY",
    "CODE_BG_NIGHT",
    "DANGER_TEXT_DAY",
    "DANGER_TEXT_NIGHT",
    "GRAPHITE_DAY",
    "GRAPHITE_NIGHT",
    "INK_2_DAY",
    "INK_2_NIGHT",
    "INK_4_DAY",
    "INK_4_NIGHT",
    "INK_DAY",
    "INK_NIGHT",
    "MARGIN_DAY",
    "MARGIN_NIGHT",
    "PALETTES",
    "PAPER_ELEVATED_DAY",
    "PAPER_ELEVATED_NIGHT",
    "PEN_DAY",
    "PEN_HOVER_DAY",
    "PEN_HOVER_NIGHT",
    "PEN_NIGHT",
    "POSITIVE_DAY",
    "POSITIVE_NIGHT",
    "RISK_DAY",
    "RISK_NIGHT",
    "RULE_DAY",
    "RULE_FAINT_DAY",
    "RULE_FAINT_NIGHT",
    "RULE_NIGHT",
    "SHEET_ALT_DAY",
    "SHEET_ALT_NIGHT",
    "SHEET_DAY",
    "SHEET_NIGHT",
    "STOCK_DAY",
    "STOCK_NIGHT",
    "TOKEN_ORDER",
    "css_root_block",
    "palette",
]

# ---------------------------------------------------------------------------
# Day — warm unbleached paper
# ---------------------------------------------------------------------------
STOCK_DAY = "#f7f4ed"          # page paper
SHEET_DAY = "#fffdf7"          # card surface
SHEET_ALT_DAY = "#efeae0"      # alternate surface; also the warm tone of the pipeline card
INK_DAY = "#1a1a17"            # text
GRAPHITE_DAY = "#66665e"       # secondary text (4.5:1 on every day surface)
PEN_DAY = "#1f4634"            # primary accent: forest green (name kept from "Ledger", where it was rust)
PEN_HOVER_DAY = "#15311f"
RISK_DAY = "#8a3a2a"           # "this may not hold": brick
ACCENT_DAY = "#a35a18"         # secondary accent: amber, for "look twice"
POSITIVE_DAY = "#24643c"
RULE_DAY = "#d9d3c4"           # hairline
RULE_FAINT_DAY = "#e6e0d1"     # softer hairline
MARGIN_DAY = "#9a8650"         # decorative only; a brass tone, far in hue from `risk`
CODE_BG_DAY = SHEET_ALT_DAY
INK_2_DAY = "#3a3a35"          # body text on cards
INK_4_DAY = "#9a9a90"          # quiet / disabled; decorative, never text that must be read
ACCENT_SOFT_DAY = "#e4ebe5"    # tinted surface behind accent text
ACCENT_INK_DAY = "#f7f4ed"     # text on a `pen` fill
ACCENT_TEXT_DAY = "#1f4634"    # the accent as TEXT (AA on paper, card and tint)
DANGER_TEXT_DAY = "#8a3a2a"    # "may not hold" as TEXT
BG_DEEP_DAY = "#12241a"        # deep forest band
PAPER_ELEVATED_DAY = "#ffffff"

# ---------------------------------------------------------------------------
# Night — carbon ledger stock
# ---------------------------------------------------------------------------
STOCK_NIGHT = "#141712"
SHEET_NIGHT = "#1c2219"
SHEET_ALT_NIGHT = "#1a1e17"
INK_NIGHT = "#edf0e4"
GRAPHITE_NIGHT = "#8a927d"
PEN_NIGHT = "#326d48"          # a fill: use ACCENT_TEXT_NIGHT for text
PEN_HOVER_NIGHT = "#2b6040"    # deeper than `pen`, so button text stays at 6.4:1 on hover (the prototype's lighter hover was 4.0:1)
RISK_NIGHT = "#cf5544"         # a fill/mark: use DANGER_TEXT_NIGHT for text
ACCENT_NIGHT = "#d48239"
POSITIVE_NIGHT = "#4ca167"
RULE_NIGHT = "#2a3324"
RULE_FAINT_NIGHT = "#20271b"
MARGIN_NIGHT = "#b3a06a"
CODE_BG_NIGHT = SHEET_ALT_NIGHT
INK_2_NIGHT = "#c4c9b6"
INK_4_NIGHT = "#5a6150"
ACCENT_SOFT_NIGHT = "#1c2b20"
ACCENT_INK_NIGHT = "#edf0e4"
ACCENT_TEXT_NIGHT = "#7fc79a"
DANGER_TEXT_NIGHT = "#e8826a"
BG_DEEP_NIGHT = "#0d120e"
PAPER_ELEVATED_NIGHT = "#22291e"

#: Iteration order for `css_root_block` and any table/report of the full
#: token set — CSS custom-property name (hyphenated) is this key with `_`
#: replaced by `-`.
TOKEN_ORDER: tuple[str, ...] = (
    "stock", "sheet", "sheet_alt", "ink", "graphite",
    "pen", "pen_hover", "risk", "accent", "positive",
    "rule", "rule_faint", "margin", "code_bg",
    "ink_2", "ink_4", "accent_soft", "accent_ink",
    "accent_text", "danger_text", "bg_deep", "paper_elevated",
)

PALETTES: dict[Mode, dict[str, str]] = {
    "day": {
        "stock": STOCK_DAY, "sheet": SHEET_DAY, "sheet_alt": SHEET_ALT_DAY,
        "ink": INK_DAY, "graphite": GRAPHITE_DAY,
        "pen": PEN_DAY, "pen_hover": PEN_HOVER_DAY,
        "risk": RISK_DAY, "accent": ACCENT_DAY, "positive": POSITIVE_DAY,
        "rule": RULE_DAY, "rule_faint": RULE_FAINT_DAY,
        "margin": MARGIN_DAY, "code_bg": CODE_BG_DAY,
        "ink_2": INK_2_DAY, "ink_4": INK_4_DAY,
        "accent_soft": ACCENT_SOFT_DAY, "accent_ink": ACCENT_INK_DAY,
        "accent_text": ACCENT_TEXT_DAY, "danger_text": DANGER_TEXT_DAY,
        "bg_deep": BG_DEEP_DAY, "paper_elevated": PAPER_ELEVATED_DAY,
    },
    "night": {
        "stock": STOCK_NIGHT, "sheet": SHEET_NIGHT, "sheet_alt": SHEET_ALT_NIGHT,
        "ink": INK_NIGHT, "graphite": GRAPHITE_NIGHT,
        "pen": PEN_NIGHT, "pen_hover": PEN_HOVER_NIGHT,
        "risk": RISK_NIGHT, "accent": ACCENT_NIGHT, "positive": POSITIVE_NIGHT,
        "rule": RULE_NIGHT, "rule_faint": RULE_FAINT_NIGHT,
        "margin": MARGIN_NIGHT, "code_bg": CODE_BG_NIGHT,
        "ink_2": INK_2_NIGHT, "ink_4": INK_4_NIGHT,
        "accent_soft": ACCENT_SOFT_NIGHT, "accent_ink": ACCENT_INK_NIGHT,
        "accent_text": ACCENT_TEXT_NIGHT, "danger_text": DANGER_TEXT_NIGHT,
        "bg_deep": BG_DEEP_NIGHT, "paper_elevated": PAPER_ELEVATED_NIGHT,
    },
}


def palette(mode: Mode) -> dict[str, str]:
    """The token dict for one mode.

    `mode` is whatever the caller already normalised to `"day"`/`"night"` —
    that mapping from Streamlit's `"dark"`/session-state spelling happens
    once, at the session-state boundary (`ui/styles.py`), not here, so this
    module has exactly one fallback rule to keep in sync with, not several.
    """
    return PALETTES[mode]


def css_root_block(mode: Mode, *, indent: str = "    ") -> str:
    """The `--token: #hex;` lines for one mode, in `TOKEN_ORDER`.

    The shared body every CSS-emitting call site (`ui/styles.py`,
    `src/core/html_report.py`) interpolates into its own `:root { ... }` —
    so the CSS custom-property *declarations* are written once, not just
    the hex values they carry.
    """
    values = palette(mode)
    lines = (f"{indent}--{key.replace('_', '-')}: {values[key]};" for key in TOKEN_ORDER)
    return "\n".join(lines)
