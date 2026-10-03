"""
Chart Theme — single source of truth for the Ledger palette as it applies to
*charts* (Round 7 items 7.17 and the Vega half of Q1).

The page-chrome tokens (stock/sheet/ink/etc.) now live in
`src/core/design_tokens.py`, which this module imports rather than
retranscribing (FrontendPlan.md item 2.1). What stays local to this module is
chart-only: the categorical/sequential/diverging ink sets DESIGN.md calls out
as separate from the two UI pens ("Chart categories extend the two pens with
four warm plot inks"), plus the Vega-Lite config assembly.

Why this module exists: three places used to each hand-roll their own copy
of the same handful of hex codes (`src/core/html_report.py`'s
`_VEGA_PLOT_CONFIG`, `app.py`'s `_get_vega_config`, and ad hoc hexes sprinkled
through `src/core/dashboard.py`'s chart specs). A stored chart spec must
never bake a color in — `vega_config()` is injected fresh at render/embed
time, so the exact same `dashboard.json` / `report.html` artifact can be
re-rendered light or dark from whatever the viewer's own theme is, without
regenerating anything.

NOTE for a future pass: `app.py`'s `_get_vega_config()` should be changed to
`return vega_config()` instead of hand-maintaining its own copy of this
palette. Not done here because app.py is off-limits for this change set.

FrontendPlan.md item 1.4: the previous six-slot categorical range failed the
dataviz skill's colorblind-separation and normal-vision checks in both modes,
and used `--risk` as slot 2 — meaning a chart series colored red meant
"second category" while the rest of the UI reserves red for "may not hold"
(DESIGN.md, "The rule about red"). The replacement below is validated (all
checks PASS, no warnings) against the new A2 buff-green chart surfaces via
the dataviz skill's `validate_palette.js`:
    node validate_palette.js "<hexes>" --mode light --surface "#f7f8ef"
    node validate_palette.js "<hexes>" --mode dark  --surface "#242a20"
`--risk` no longer appears in either range — it stays a status color only.
"""
from __future__ import annotations

from typing import Any

from src.core.design_tokens import (
    ACCENT_DAY,
    ACCENT_NIGHT,
    GRAPHITE_DAY,
    GRAPHITE_NIGHT,
    INK_DAY,
    INK_NIGHT,
    PEN_DAY,
    PEN_NIGHT,
    POSITIVE_DAY,
    POSITIVE_NIGHT,
    RISK_DAY,
    RISK_NIGHT,
    RULE_DAY,
    RULE_FAINT_DAY,
    RULE_FAINT_NIGHT,
    RULE_NIGHT,
    SHEET_DAY,
    SHEET_NIGHT,
    STOCK_DAY,
    STOCK_NIGHT,
)

__all__ = [
    "ACCENT_DAY",
    "ACCENT_NIGHT",
    "CATEGORY_RANGE_DAY",
    "CATEGORY_RANGE_NIGHT",
    "FONT_BODY",
    "FONT_HEADING",
    "GRAPHITE_DAY",
    "GRAPHITE_NIGHT",
    "INK_DAY",
    "INK_NIGHT",
    "PEN_DAY",
    "PEN_NIGHT",
    "POSITIVE_DAY",
    "POSITIVE_NIGHT",
    "RISK_DAY",
    "RISK_NIGHT",
    "RULE_DAY",
    "RULE_FAINT_DAY",
    "RULE_FAINT_NIGHT",
    "RULE_NIGHT",
    "SHEET_DAY",
    "SHEET_NIGHT",
    "STOCK_DAY",
    "STOCK_NIGHT",
    "axis_format",
    "humanize_axis_title",
    "humanize_label",
    "vega_config",
]

# ---------------------------------------------------------------------------
# Page-chrome tokens (stock/sheet/ink/etc.) are imported from
# `src/core/design_tokens.py` above — this module transcribes nothing.
# ---------------------------------------------------------------------------

# Colorblind-safe categorical palette (FrontendPlan.md section 4.2), fixed
# hue order, never cycled. Slot 1 is the Ledger pen in both modes except one
# deliberate exception: Night slot 1 is NOT `PEN_NIGHT` (`#f0a24a`), which is
# too light to read as a chart mark against the dark chart surface — chart
# inks and UI pens are separate token sets by design. `--risk` (`RISK_DAY`/
# `RISK_NIGHT`) is deliberately absent from both ranges; see the module
# docstring. A 7th series folds into "Other" rather than cycling a 7th hue.
CATEGORY_RANGE_DAY: list[str] = [
    PEN_DAY,     # 1. Ledger rust (the pen)
    "#1f6aa0",   # 2. Ink blue
    "#3f7f4a",   # 3. Leaf green
    "#6d4a8c",   # 4. Plum
    "#9a7418",   # 5. Ochre
    "#c4648a",   # 6. Rose
]
CATEGORY_RANGE_NIGHT: list[str] = [
    "#cc7f34",   # 1. Ledger rust, chart-surface variant (not PEN_NIGHT)
    "#4d97cf",   # 2. Ink blue
    "#4fa46a",   # 3. Leaf green
    "#9d7fd0",   # 4. Plum
    "#a8892a",   # 5. Ochre
    "#c86e92",   # 6. Rose
]

FONT_HEADING = "Bricolage Grotesque, 'Public Sans', sans-serif"
FONT_BODY = "Public Sans, 'Segoe UI', sans-serif"


#: Round 8 (8.4) — Vega-Lite format-string fragments keyed by `unit_hint`
#: (`src.core.profiler.ColumnProfile.unit_hint`). One place this lives, same
#: reasoning as `vega_config()` being the one place colour lives: a stored
#: chart spec must never hand-roll its own format string per column.
#:
#: `"percent"` uses `.0%` (not a literal `%` suffix on a 0-100 number)
#: because `src.core.coercion._parse_percent` divides by 100 at parse time —
#: a string like "17%" becomes the float 0.17, not 17 — so every
#: `unit_hint == "percent"` column in this codebase is already a 0-1
#: fraction, which is exactly what Vega-Lite's `%` format expects (it does
#: its own ×100 before appending the sign). Using `,.0f` + a literal "%"
#: here would silently 100x every percent axis/tooltip.
_UNIT_FORMATS: dict[str, dict[str, str]] = {
    "currency": {"format": "$,.0f"},
    "percent": {"format": ".0%"},
    "count": {"format": ",d"},
}

#: Human-readable axis-title suffix per unit_hint, appended after
#: title-casing the raw column name (see `humanize_axis_title`).
_UNIT_TITLE_SUFFIX: dict[str, str] = {
    "currency": " ($)",
    "percent": " (%)",
}


def axis_format(unit_hint: str | None) -> dict[str, str]:
    """Vega-Lite format-string fragment for a column's `unit_hint`.

    Returns `{"format": "$,.0f"}` for `"currency"`, `{"format": ".0%"}` for
    `"percent"`, `{"format": ",d"}` for `"count"`, and `{}` (no override,
    Vega-Lite's own default formatting applies) for `None`/anything else.
    Callers merge this into an encoding channel's `axis` sub-object, and/or
    spread it directly into a tooltip channel definition (both accept a
    bare `"format"` key), rather than ever writing a format string inline
    at the chart-builder call site.
    """
    return dict(_UNIT_FORMATS.get(unit_hint or "", {}))


#: Short tokens that read as acronyms, not words, once a column is humanised.
_ACRONYMS: frozenset[str] = frozenset(
    {"id", "usd", "eur", "gbp", "inr", "kpi", "roi", "gdp", "pca", "rfm", "url", "sku", "ltv", "cac", "aov"}
)


def humanize_label(name: str) -> str:
    """Raw column/field name -> readable label: `revenue_usd` -> "Revenue USD".

    Underscores become spaces; all-lowercase words are capitalised; words the
    author already cased (`MRR`, `iPhone`) are left alone. The one helper for
    every chart title, axis title and tooltip title.
    """
    words = str(name).replace("_", " ").split()
    return " ".join(
        w.upper() if w.lower() in _ACRONYMS else (w.capitalize() if w.islower() else w)
        for w in words
    )


def humanize_axis_title(column: str, unit_hint: str | None = None) -> str:
    """Human-readable axis/tooltip title for a raw column name.

    `humanize_label` plus a unit suffix for a known currency/percent hint —
    e.g. a column literally named `amount` with `unit_hint == "currency"`
    becomes `"Amount ($)"` instead of the bare string `"amount"`.
    `count`/`None` get no suffix (a plain, already-legible title is left alone).
    """
    return humanize_label(column) + _UNIT_TITLE_SUFFIX.get(unit_hint or "", "")


def vega_config(dark: bool = False) -> dict[str, Any]:
    """
    A complete Vega-Lite `config` object for the Ledger theme.

    Callers inject this at render/embed time (`spec["config"] = vega_config(...)`
    right before handing the spec to vega-embed) — it must never be baked into
    a saved chart spec (dashboard.json), so the same artifact renders in
    either mode from the viewer's live preference.

    Includes per-mark-type color defaults (bar/line/area/circle/point/tick/
    rule) so a spec that emits *no* color encoding at all still renders
    in-theme, and two layered mark types in one chart (e.g. a bar+line combo
    chart) read as visually distinct without either layer hardcoding a hex.
    """
    ink = INK_NIGHT if dark else INK_DAY
    graphite = GRAPHITE_NIGHT if dark else GRAPHITE_DAY
    sheet = SHEET_NIGHT if dark else SHEET_DAY
    rule = RULE_NIGHT if dark else RULE_DAY
    pen = PEN_NIGHT if dark else PEN_DAY
    category = CATEGORY_RANGE_NIGHT if dark else CATEGORY_RANGE_DAY

    return {
        "background": sheet,
        "font": FONT_BODY,
        "axis": {
            "labelColor": graphite,
            "titleColor": graphite,
            "gridColor": rule,
            "gridDash": [2, 3],
            "domainColor": rule,
            "tickColor": rule,
            "labelFont": FONT_BODY,
            "labelFontSize": 11,
            "titleFont": FONT_HEADING,
            "titleFontWeight": 600,
        },
        "legend": {
            "labelColor": ink,
            "titleColor": graphite,
            "labelFont": FONT_BODY,
            "titleFont": FONT_HEADING,
            "symbolType": "square",
        },
        "view": {"stroke": "transparent"},
        "range": {"category": category},
        "mark": {"color": pen},
        "bar": {"color": pen},
        "line": {"color": ink},
        "area": {"color": pen, "opacity": 0.35, "line": {"color": pen}},
        "circle": {"color": pen},
        "point": {"color": pen},
        "tick": {"color": ink},
        "rule": {"color": graphite},
    }
