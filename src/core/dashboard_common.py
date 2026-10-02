"""
Dashboard Agent — shared chart helpers and the ChartSpec record.

Split out of dashboard.py; `src.core.dashboard` re-exports every name here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.core.chart_spec import alias_fields
from src.core.profiler import DatasetProfile

#: Hard cap on inline rows per chart (keeps specs lightweight).
MAX_POINTS = 1_000


#: Title of the correlation heatmap; the generate_visualizations tool uses the
#: same one, so the dashboard can tell it already has this view.
CORRELATION_HEATMAP_TITLE = "Which Columns Move Together"


#: Categorical columns with more classes than this get truncated to top-N.
MAX_CATEGORIES_SHOWN = 12


#: Default bin count for pre-aggregated histograms.
DEFAULT_HIST_BINS = 20


#: Numeric roles that must never be binned/plotted as if continuous
#: (integer-coded dimensions such as year/region codes included).
_NON_CONTINUOUS_ROLES = ("flag", "ordinal", "identifier", "dimension")


@dataclass
class ChartSpec:
    """One renderable chart: metadata plus a complete Vega-Lite spec.

    `finding_id`/`priority`/`layer`/`caption` are new (7.8 story layer) but
    additive and backward compatible — a chart nobody tagged just keeps the
    defaults, and every existing consumer reading chart_id/title/description/
    spec still gets exactly those keys from to_dict().
    """

    chart_id: str
    title: str
    description: str
    spec: dict[str, Any]
    finding_id: str | None = None
    priority: float = 0.0
    layer: str = "analyst"
    caption: str | None = None
    size: str | None = None

    def __post_init__(self) -> None:
        # Vega treats . [ ] \ ' " in a field name as path syntax; alias them
        # (idempotent) so every consumer gets a spec that actually renders.
        self.spec = alias_fields(self.spec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_id": self.chart_id,
            "title": self.title,
            "description": self.description,
            "spec": self.spec,
            "finding_id": self.finding_id,
            "priority": self.priority,
            "layer": self.layer,
            "caption": self.caption,
            "size": self.size,
        }


def dashboard_to_json(charts: list[ChartSpec]) -> str:
    """Serialise a dashboard for saving to output/dashboard.json."""
    return json.dumps([c.to_dict() for c in charts], indent=2, default=str)


def _to_primitive(value: Any) -> Any:
    """Coerce numpy/pandas scalars into JSON-safe Python primitives.

    `dashboard_to_json` serialises with `default=str` — anything that isn't
    already a plain int/float/str/bool/None falls through to `str()`, which
    would silently turn e.g. a numpy.int64 count into the *string* "5" fed
    to a quantitative encoding. Every computed aggregate in this module must
    be routed through this (or an explicit int()/float() cast) before it
    reaches a spec.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _chartable(profile: DatasetProfile, *kinds: str) -> list[str]:
    """Column names of the given kinds, excluding identifiers/constants."""
    return [c.name for c in profile.columns_of_kind(*kinds)]


def _safe(fn: Any, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    """Run a panel builder, swallowing any exception into `default`.

    One malformed finding, missing column, or degenerate group must not
    zero out the entire dashboard — the controller already wraps the whole
    of `_generate_dashboard` in a try/except, but that means a single bad
    panel currently costs *every* chart, not just its own.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _merge_axis_format(encoding_entry: dict[str, Any], fmt: dict[str, str]) -> None:
    """Merge `chart_theme.axis_format()`'s fragment into an encoding
    channel's `axis` sub-object in place, without clobbering any axis
    properties the builder already set (e.g. `labelAngle`)."""
    if not fmt:
        return
    axis = dict(encoding_entry.get("axis") or {})
    axis.update(fmt)
    encoding_entry["axis"] = axis


def _histogram_bins(
    series: pd.Series, max_bins: int = DEFAULT_HIST_BINS, log: bool = False
) -> list[dict[str, Any]]:
    """Pre-aggregated bin counts for a numeric series (P2.7) — a histogram
    ships as {bin_start, bin_end, count} triples, never raw values. `log`
    uses geometrically spaced edges (strictly positive series only) so the
    bins are equal-width on a log axis."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return []
    nunique = int(clean.nunique())
    if nunique <= 1:
        return []
    bins: Any = max(1, min(max_bins, nunique))
    if log:
        bins = np.geomspace(float(clean.min()), float(clean.max()), bins + 1)
    counts, edges = np.histogram(clean.to_numpy(dtype=float), bins=bins)
    return [
        {
            "bin_start": round(float(edges[i]), 6),
            "bin_end": round(float(edges[i + 1]), 6),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]


def _fmt_value(value: float, unit_hint: str | None = None) -> str:
    """Magnitude-aware number for captions: 0.034 stays "0.034", not "0"."""
    if unit_hint == "percent":
        return f"{value:.0%}" if abs(value) >= 0.1 else f"{value:.1%}"
    magnitude = abs(value)
    if magnitude >= 100 or magnitude == 0:
        text = f"{value:,.0f}"
    elif magnitude >= 1:
        text = f"{value:,.2f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:.2g}"
    return f"${text}" if unit_hint == "currency" else text


def _num(value: Any) -> float | None:
    """A finite real number as float, else None (bools and strings excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _slug(text: Any) -> str:
    return re.sub(r"\W+", "_", str(text)).strip("_").lower() or "x"
