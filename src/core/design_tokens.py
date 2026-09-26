"""
Design Tokens — the single Python source of truth for the "Ledger" palette
(DESIGN.md, "Tokens — ink"), Day and Night.

FrontendPlan.md item 2.1. Before this module, five call sites each hand-rolled
their own copy of the same hex codes: `ui/styles.py`, `ui/landing.py`,
`ui/pipeline_3d.py`, `ui/cinematic_3d.py`, and `src/core/html_report.py`'s
CSS block (`src/core/chart_theme.py` already centralised the *chart* colors
correctly — see its own docstring — but did not cover the page-chrome tokens
the other five files each retyped). Four independent copies is how the
FrontendPlan.md T2/T3/T6 findings happened: `cinematic_3d.py` alone carried
three values (night `--graphite`, `--sheet`, and a bare `#4fc3f7` "third pen")
that matched none of the other four copies. Every one of those files now
imports from here instead of retyping a hex.

Placement note: this lives in `src/core`, not `ui/`, even though most current
consumers are `ui/*.py`. A design token is not UI behaviour — `chart_theme.py`
and `html_report.py` (both `src/core`) need the same hexes as `ui/*.py` does,
and `src/core` must never depend on `ui/` (AGENTS.md's architectural
boundaries table names `controller.py`/`engine.py`/`tools/*`/`memory.py`
explicitly, but "core has no upward dependency on presentation" is the same
rule applied one layer further out — putting the tokens in `ui/` would make
`src/core/chart_theme.py` import from `ui/`, which is backwards).

"Ledger paper" (decision A2, 2026-09-25): stock is a pale buff-green rather
than warm cream, hairline rules are a faint blue (the feint ruling on real
accounting paper — intentionally low-contrast; that is what "feint" means),
and the decorative double margin rule is a desaturated wine, not `--risk`'s
saturated brick. The margin color was chosen for hue separation from `--risk`
(342 degrees vs 7 degrees on the day palette) specifically so the page's one
red status color keeps its single meaning (DESIGN.md, "The rule about red") —
see `MARGIN_DAY`/`MARGIN_NIGHT` below. Every text/background pairing in this
module was checked against WCAG AA (4.5:1) before being chosen.
"""
from __future__ import annotations

from typing import Literal

Mode = Literal["day", "night"]

__all__ = [
    "ACCENT_DAY",
    "ACCENT_NIGHT",
    "CODE_BG_DAY",
    "CODE_BG_NIGHT",
    "GRAPHITE_DAY",
    "GRAPHITE_NIGHT",
    "INK_DAY",
    "INK_NIGHT",
    "MARGIN_DAY",
    "MARGIN_NIGHT",
    "PALETTES",
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
# Day
# ---------------------------------------------------------------------------
STOCK_DAY = "#eef1e0"
SHEET_DAY = "#f7f8ef"
SHEET_ALT_DAY = "#e6e9d6"
INK_DAY = "#33362a"
GRAPHITE_DAY = "#5f6850"
PEN_DAY = "#a34f20"
PEN_HOVER_DAY = "#7e3d18"
RISK_DAY = "#a33526"
ACCENT_DAY = "#e08a3e"
POSITIVE_DAY = "#3f7a44"
RULE_DAY = "#a9bfd1"
RULE_FAINT_DAY = "#c7d8e2"
MARGIN_DAY = "#7d3f52"
CODE_BG_DAY = SHEET_ALT_DAY

# ---------------------------------------------------------------------------
# Night
# ---------------------------------------------------------------------------
STOCK_NIGHT = "#1c211a"
SHEET_NIGHT = "#242a20"
SHEET_ALT_NIGHT = "#2a3024"
INK_NIGHT = "#eef0e4"
GRAPHITE_NIGHT = "#a8ad98"
PEN_NIGHT = "#f0a24a"
PEN_HOVER_NIGHT = "#ffb86b"
RISK_NIGHT = "#e2685a"
ACCENT_NIGHT = "#d99a4e"
POSITIVE_NIGHT = "#7fb77e"
RULE_NIGHT = "#3d4a52"
RULE_FAINT_NIGHT = "#2c3630"
MARGIN_NIGHT = "#a85f74"
CODE_BG_NIGHT = "#232922"

#: Iteration order for `css_root_block` and any table/report of the full
#: token set — CSS custom-property name (hyphenated) is this key with `_`
#: replaced by `-`.
TOKEN_ORDER: tuple[str, ...] = (
    "stock", "sheet", "sheet_alt", "ink", "graphite",
    "pen", "pen_hover", "risk", "accent", "positive",
    "rule", "rule_faint", "margin", "code_bg",
)

PALETTES: dict[Mode, dict[str, str]] = {
    "day": {
        "stock": STOCK_DAY, "sheet": SHEET_DAY, "sheet_alt": SHEET_ALT_DAY,
        "ink": INK_DAY, "graphite": GRAPHITE_DAY,
        "pen": PEN_DAY, "pen_hover": PEN_HOVER_DAY,
        "risk": RISK_DAY, "accent": ACCENT_DAY, "positive": POSITIVE_DAY,
        "rule": RULE_DAY, "rule_faint": RULE_FAINT_DAY,
        "margin": MARGIN_DAY, "code_bg": CODE_BG_DAY,
    },
    "night": {
        "stock": STOCK_NIGHT, "sheet": SHEET_NIGHT, "sheet_alt": SHEET_ALT_NIGHT,
        "ink": INK_NIGHT, "graphite": GRAPHITE_NIGHT,
        "pen": PEN_NIGHT, "pen_hover": PEN_HOVER_NIGHT,
        "risk": RISK_NIGHT, "accent": ACCENT_NIGHT, "positive": POSITIVE_NIGHT,
        "rule": RULE_NIGHT, "rule_faint": RULE_FAINT_NIGHT,
        "margin": MARGIN_NIGHT, "code_bg": CODE_BG_NIGHT,
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
