"""
Chart Theme — single source of truth for the Ledger palette as it applies to
*charts* (Round 7 items 7.17 and the Vega half of Q1).

The other half of Q1 — the landing-page HTML/CSS palette — is maintained
separately (see DESIGN.md and ui/landing.py) and is out of scope here.

DESIGN.md's "Tokens — ink" table is canonical; this module is a direct
transcription of it, not an independent source. If the two ever disagree,
DESIGN.md wins and this file is out of date.

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
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Tokens — transcribed from DESIGN.md's "Tokens — ink" table verbatim.
# Day = light palette, Night = dark palette (`prefers-color-scheme: dark`).
# ---------------------------------------------------------------------------

STOCK_DAY = "#f7eedd"
STOCK_NIGHT = "#241c14"

SHEET_DAY = "#fffbf2"
SHEET_NIGHT = "#2f251a"

INK_DAY = "#3a2b1e"
INK_NIGHT = "#f3e9d8"

GRAPHITE_DAY = "#8a7660"
GRAPHITE_NIGHT = "#b8a688"

PEN_DAY = "#a34f20"
PEN_NIGHT = "#f0a24a"

RISK_DAY = "#a33526"
RISK_NIGHT = "#e2685a"

POSITIVE_DAY = "#5b8c5a"
POSITIVE_NIGHT = "#7fb77e"

RULE_DAY = "#e4d4bc"
RULE_NIGHT = "#4a3c28"

RULE_FAINT_DAY = "#eee3cb"
RULE_FAINT_NIGHT = "#3a2e1f"

# Decorative only, per DESIGN.md.
ACCENT_DAY = "#e08a3e"
ACCENT_NIGHT = "#d99a4e"

# "Chart categories extend the two pens with four warm plot inks" (DESIGN.md,
# "Tokens — ink"). DESIGN.md gives the Day row verbatim:
#   Day `#a34f20 #a33526 #c08a2e #5b8c5a #8a7660 #b5714a`
# It does not specify a Night chart-category row. The Night list below is
# built only from documented Night tokens (pen, risk, positive, graphite,
# accent) plus exactly one interpolated gold, so it keeps the same six-color
# shape as Day without inventing more palette than necessary.
CATEGORY_RANGE_DAY: list[str] = [
    PEN_DAY, RISK_DAY, "#c08a2e", POSITIVE_DAY, GRAPHITE_DAY, "#b5714a",
]
CATEGORY_RANGE_NIGHT: list[str] = [
    PEN_NIGHT, RISK_NIGHT, "#d9a53e", POSITIVE_NIGHT, GRAPHITE_NIGHT, ACCENT_NIGHT,
]

FONT_HEADING = "Baloo 2, 'Mukta', sans-serif"
FONT_BODY = "Mukta, 'Segoe UI', sans-serif"


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


def humanize_axis_title(column: str, unit_hint: str | None = None) -> str:
    """Human-readable axis/tooltip title for a raw column name.

    Title-cases the column name and replaces underscores with spaces, then
    appends a unit suffix for a known currency/percent hint — e.g. a column
    literally named `amount` with `unit_hint == "currency"` becomes
    `"Amount ($)"` instead of the bare string `"amount"`. `count`/`None`
    get no suffix (a plain, already-legible title is left alone).
    """
    title = column.replace("_", " ").strip().title()
    return title + _UNIT_TITLE_SUFFIX.get(unit_hint or "", "")


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
            "domainColor": ink,
            "tickColor": ink,
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
