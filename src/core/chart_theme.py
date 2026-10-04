"""
Chart Theme — single source of truth for the Sorrel palette as it applies to
*charts* (Round 7 items 7.17 and the Vega half of Q1).

The page-chrome tokens (stock/sheet/ink/etc.) now live in
`src/core/design_tokens.py`, which this module imports rather than
retranscribing (FrontendPlan.md item 2.1). What stays local to this module is
chart-only: the categorical/sequential/diverging ink sets DESIGN.md calls out
as separate from the UI pens, plus the Vega-Lite config assembly.

Why this module exists: three places used to each hand-roll their own copy
of the same handful of hex codes (`src/core/html_report.py`'s
`_VEGA_PLOT_CONFIG`, `app.py`'s `_get_vega_config`, and ad hoc hexes sprinkled
through `src/core/dashboard.py`'s chart specs). A stored chart spec must
never bake a color in — `vega_config()` is injected fresh at render/embed
time, so the exact same `dashboard.json` / `report.html` artifact can be
re-rendered light or dark from whatever the viewer's own theme is, without
regenerating anything.

NOTE: `ui/components/cards.py`'s `get_vega_config()` already returns
`vega_config(dark=...)`, so the console and the HTML report draw the same charts.

FrontendPlan.md item 1.4: an earlier six-slot range used `--risk` as slot 2 —
meaning a chart series colored red meant "second category" while the rest of
the UI reserves red for "may not hold" (DESIGN.md, "The rule about red").
`--risk` is in neither range — it stays a status color only. The current ranges
were re-validated for the Sorrel surfaces (see the comment above them for the
thresholds); the chart surface is `sheet` in both modes, since that is the Vega
`background`.
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
    "CHART_GRID_NIGHT",
    "CHART_PEN_DAY",
    "CHART_PEN_NIGHT",
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
# hue order, never cycled. Slot 1 is the forest-green pen in both modes except one
# deliberate exception: Night slot 1 is NOT `PEN_NIGHT` (`#326d48`), which is a
# fill and only ~2.6:1 against the Night chart surface (marks need 3:1) — chart
# inks and UI pens are separate token sets by design, so Night uses a lighter
# green of the same hue. `--risk` (`RISK_DAY`/`RISK_NIGHT`) is deliberately
# absent from both ranges; see the module docstring. A 7th series folds into
# "Other" rather than cycling a 7th hue.
#
# Sorrel redesign: the old range had the pen as slot 1 and a leaf green as slot 3;
# with the pen now green those two were one hue, so slot 3 became amber-brown and
# no other slot is green. Validated with a script (not by eye): CIEDE2000 for all
# 15 pairs under normal vision (every pair >= 20 Day / >= 22 Night) and under
# protanopia, deuteranopia and tritanopia (Machado 2009, severity 1.0; every pair
# >= 8 Day / >= 11 Night), each mark >= 3:1 against the chart surface (`sheet`),
# and every slot >= 15 CIEDE2000 away from `risk`.
CHART_PEN_NIGHT = "#4a9a69"
#: Day's mark green. `PEN_DAY` (#1f4634) is a UI fill so dark it reads as black on a chart; marks use
#: this brighter green of the same hue (5:1 on the card, still the first slot of the range).
CHART_PEN_DAY = "#2e7d57"
#: Night gridlines. `RULE_NIGHT` is a hairline for the page; on the lighter Night chart surface it is
#: only ~1.1:1 and the grid vanishes. This is ~1.4:1, the ratio Day's grid has on its surface.
CHART_GRID_NIGHT = "#363f2f"
CATEGORY_RANGE_DAY: list[str] = [
    CHART_PEN_DAY,  # 1. Forest green, chart-mark variant (not PEN_DAY)
    "#2f86d6",   # 2. Ink blue
    "#a8620f",   # 3. Amber-brown
    "#6b2a73",   # 4. Plum
    "#a08c10",   # 5. Ochre / gold
    "#c26a8f",   # 6. Rose
]
CATEGORY_RANGE_NIGHT: list[str] = [
    CHART_PEN_NIGHT,   # 1. Forest green, chart-surface variant (not PEN_NIGHT)
    "#7ca9ec",   # 2. Ink blue
    "#d7a85b",   # 3. Amber-brown
    "#7863d4",   # 4. Plum / violet
    "#85700b",   # 5. Ochre / gold
    "#b75280",   # 6. Rose
]

#: Sorrel's type. Chart text is Geist throughout (labels, axis titles, legend); the serif
#: Newsreader is the app's accent face and never carries data. Same families the report embeds.
FONT_HEADING = "Geist, -apple-system, 'Segoe UI', system-ui, sans-serif"
FONT_BODY = "Geist, -apple-system, 'Segoe UI', system-ui, sans-serif"


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
    A complete Vega-Lite `config` object for the Sorrel theme.

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
    rule = CHART_GRID_NIGHT if dark else RULE_DAY
    accent = ACCENT_NIGHT if dark else ACCENT_DAY
    category = CATEGORY_RANGE_NIGHT if dark else CATEGORY_RANGE_DAY
    # Default mark colour = slot 1, so a mark with no colour encoding matches the first series and
    # clears 3:1 on the chart surface in both modes (raw PEN_NIGHT is only ~2.6:1 there).
    pen = category[0]

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
        # No `cornerRadiusEnd` here: on a pre-binned histogram (x + x2) Vega-Lite turns the bars into
        # paths of zero width, so every histogram drew empty.
        "bar": {"color": pen},
        # The main line is the pen green, not ink: near-white lines on the Night surface glare. Night
        # strokes are 0.5px heavier and fills lighter, so thin marks and tints keep their weight there.
        "line": {"color": pen, "strokeWidth": 2.5 if dark else 2},
        "area": {"color": pen, "opacity": 0.28 if dark else 0.35, "line": {"color": pen}},
        "circle": {"color": pen},
        "point": {"color": pen},
        "tick": {"color": ink},
        "rule": {"color": graphite},
        # Text marks (a reference line's label, bar values) take the ink, not Vega's default black.
        "text": {"color": ink, "font": FONT_BODY, "fontSize": 11},
        # The pop: a second thing drawn over the data (a fitted or cumulative line, an uncertainty
        # band) is the theme's amber accent. A mark opts in with {"style": "accent"}.
        "style": {"accent": {"color": accent, "strokeWidth": 3 if dark else 2.5}},
    }
