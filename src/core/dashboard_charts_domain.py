"""
Dashboard Agent — domain-pack charts (cohort, financial, workforce, segment, concentration, change, group CI).

Split out of dashboard.py; `src.core.dashboard` re-exports every name here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.chart_theme import axis_format, humanize_axis_title, humanize_label
from src.core.dashboard_common import (
    MAX_CATEGORIES_SHOWN,
    MAX_POINTS,
    ChartSpec,
    _fmt_value,
    _histogram_bins,
    _merge_axis_format,
    _num,
    _safe,
    _slug,
    _to_primitive,
)
from src.core.privacy import fold_small_groups, is_small, min_cell_size, suppression_note
from src.core.profiler import ColumnProfile
from src.core.stats_utils import aggregate_to_entity, measure_aggregation

#: Segments drawn individually in a change waterfall; the rest are one bar.
WATERFALL_MAX_SEGMENTS = 8


#: Points kept on a Lorenz curve (evenly spaced over the ranked entities).
LORENZ_MAX_POINTS = 100


def _signed(value: float, unit_hint: str | None = None) -> str:
    """`_fmt_value` with an explicit sign: "+$1,200" / "−$300"."""
    return ("+" if value >= 0 else "−") + _fmt_value(abs(value), unit_hint)


def _tool_outputs(tool_results: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Every successful output dict of a tool, most recent first."""
    return [
        r["output"] for r in reversed(tool_results)
        if r.get("tool_name") == name and r.get("status") == "success" and isinstance(r.get("output"), dict)
    ]


def _cohort_charts(df: pd.DataFrame, cohort_output: dict[str, Any] | None) -> list[ChartSpec]:
    if not cohort_output:
        return []
    # Each panel is isolated so one bad value drops only that panel.
    return [
        *_safe(_cohort_rfm_chart, cohort_output, default=[]),
        *_safe(_cohort_month_chart, cohort_output, default=[]),
        *_safe(_cohort_pareto_chart, df, cohort_output, default=[]),
    ]


def _cohort_rfm_chart(cohort_output: dict[str, Any]) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    segments = cohort_output.get("rfm_segments")
    if isinstance(segments, list) and segments:
        values = [
            {
                "segment": str(s.get("segment")),
                "revenue_share_pct": float(s.get("revenue_share_pct") or 0.0),
                "customer_share_pct": float(s.get("customer_share_pct") or 0.0),
            }
            for s in segments if isinstance(s, dict)
        ]
        if values:
            charts.append(ChartSpec(
                chart_id="cohort_rfm_segments",
                title="Customer Segments — Revenue vs Customer Share (RFM)",
                description=(
                    "Share of revenue vs. share of the customer base per RFM segment. "
                    "A segment whose revenue bar towers over its customer bar is doing outsized work."
                ),
                spec={
                    "data": {"values": values},
                    "transform": [{"fold": ["revenue_share_pct", "customer_share_pct"], "as": ["metric", "share"]}],
                    "mark": {"type": "bar"},
                    "height": max(160, 26 * len(values)),
                    "encoding": {
                        "y": {"field": "segment", "type": "nominal", "sort": "-x", "title": None},
                        "x": {"field": "share", "type": "quantitative", "title": "Share %"},
                        "yOffset": {"field": "metric"},
                        "color": {
                            "field": "metric", "type": "nominal",
                            "scale": {"domain": ["revenue_share_pct", "customer_share_pct"]},
                            "legend": {"orient": "top", "title": None},
                        },
                        "tooltip": [{"field": "segment"}, {"field": "metric"}, {"field": "share"}],
                    },
                },
            ))

    return charts


def _cohort_month_chart(cohort_output: dict[str, Any]) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    revenue_by_month = cohort_output.get("revenue_by_month")
    if isinstance(revenue_by_month, dict) and len(revenue_by_month) >= 2:
        values = [{"period": k, "revenue": float(v)} for k, v in sorted(revenue_by_month.items())]
        charts.append(ChartSpec(
            chart_id="cohort_revenue_by_month",
            title="Revenue by Month",
            description="Total revenue per calendar month.",
            spec={
                "data": {"values": values},
                "mark": {"type": "line", "point": True},
                "height": 220,
                "encoding": {
                    "x": {"field": "period", "type": "temporal", "title": None},
                    "y": {"field": "revenue", "type": "quantitative", "title": "Revenue", "scale": {"zero": False}},
                    "tooltip": [{"field": "period", "type": "temporal"}, {"field": "revenue"}],
                },
            },
        ))

    return charts


def _cohort_pareto_chart(df: pd.DataFrame, cohort_output: dict[str, Any]) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    customer_col = cohort_output.get("customer_column")
    amount_col = cohort_output.get("amount_column")
    if customer_col and amount_col and customer_col in df.columns and amount_col in df.columns:
        amounts = pd.to_numeric(df[amount_col], errors="coerce")
        per_customer = amounts.groupby(df[customer_col]).sum().dropna()
        total = float(per_customer.sum())
        if total > 0 and len(per_customer) >= 10:
            ranked_rev = per_customer.sort_values(ascending=False).reset_index(drop=True)
            n = len(ranked_rev)
            deciles: list[dict[str, Any]] = []
            for decile in range(1, 11):
                cutoff = max(1, round(n * decile / 10))
                cumulative = float(ranked_rev.iloc[:cutoff].sum())
                deciles.append({
                    "customer_decile_pct": decile * 10,
                    "cumulative_revenue_share_pct": round(cumulative / total * 100, 2),
                })
            charts.append(ChartSpec(
                chart_id="cohort_pareto",
                title="Revenue Concentration — Customers Ranked by Spend",
                description=(
                    "Cumulative share of total revenue as more customers (ranked highest-spend "
                    "first) are included — how much of the business rests on a few customers."
                ),
                spec={
                    "data": {"values": deciles},
                    "mark": {"type": "line", "point": True},
                    "height": 220,
                    "encoding": {
                        "x": {"field": "customer_decile_pct", "type": "quantitative", "title": "Top % of customers"},
                        "y": {"field": "cumulative_revenue_share_pct", "type": "quantitative",
                              "title": "Cumulative revenue share %", "scale": {"domain": [0, 100]}},
                        "tooltip": [{"field": "customer_decile_pct"}, {"field": "cumulative_revenue_share_pct"}],
                    },
                },
            ))
    return charts


def _financial_charts(df: pd.DataFrame, financial_output: dict[str, Any] | None) -> list[ChartSpec]:
    if not financial_output:
        return []
    date_col = financial_output.get("date_column")
    price_col = financial_output.get("price_column")
    if not date_col or not price_col or date_col not in df.columns or price_col not in df.columns:
        return []

    symbol_col = financial_output.get("symbol_column")
    cols = [date_col, price_col] + ([symbol_col] if symbol_col and symbol_col in df.columns else [])
    frame = df[cols].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce", format="mixed")
    frame[price_col] = pd.to_numeric(frame[price_col], errors="coerce")
    frame = frame.dropna(subset=[date_col, price_col])
    if frame.empty:
        return []

    label: str | None = None
    best = False
    if symbol_col and symbol_col in frame.columns:
        symbols = frame[symbol_col].astype(str)
        best_symbol = (financial_output.get("best_performer") or {}).get("symbol")
        if best_symbol is not None and (symbols == str(best_symbol)).any():
            label = str(best_symbol)
            best = True
        else:
            # No usable best performer: pick the most frequent symbol so several
            # tickers are never interleaved into one series.
            label = str(symbols.mode().iloc[0])
            best = False
        frame = frame[symbols == label]

    frame = frame[[date_col, price_col]].sort_values(date_col)
    # Aggregate-first for long series (P2.7) — resample to weekly last-value
    # rather than inlining every row; a further stride decimates the rare
    # case where even weekly resampling exceeds MAX_POINTS.
    if len(frame) > MAX_POINTS:
        frame = frame.set_index(date_col)[price_col].resample("W").last().dropna().reset_index()
    if len(frame) > MAX_POINTS:
        step = max(1, len(frame) // MAX_POINTS)
        frame = frame.iloc[::step]
    if len(frame) < 3:
        return []

    prices = frame[price_col].astype(float).reset_index(drop=True)
    dates = frame[date_col].reset_index(drop=True)
    base = float(prices.iloc[0])
    running_peak = prices.cummax()
    rows: list[dict[str, Any]] = []
    for ts, price, peak in zip(dates, prices, running_peak, strict=True):
        cumulative_return = (float(price) / base - 1.0) * 100 if base else 0.0
        drawdown = (float(price) / float(peak) - 1.0) * 100 if peak else 0.0
        rows.append({
            "date": ts.strftime("%Y-%m-%d"),
            "cumulative_return_pct": round(cumulative_return, 4),
            "drawdown_pct": round(drawdown, 4),
        })

    title_symbol = f" — {label}" if label else ""
    return [ChartSpec(
        chart_id="financial_overview",
        title=f"Return & Drawdown{title_symbol}",
        description=(
            f"Cumulative return and drawdown of '{price_col}' over time"
            + (f" ({'best performer' if best else 'most frequent symbol'} '{label}')" if label else "")
            + "."
        ),
        spec={
            # Named dataset (P2.7) — both layers read the same rows once
            # instead of each mark inlining its own copy.
            "datasets": {"financial_series": rows},
            "height": 260,
            "layer": [
                {
                    "data": {"name": "financial_series"},
                    "mark": {"type": "area", "opacity": 0.35},
                    "encoding": {
                        "x": {"field": "date", "type": "temporal", "title": None},
                        "y": {"field": "drawdown_pct", "type": "quantitative", "title": "Drawdown %"},
                    },
                },
                {
                    "data": {"name": "financial_series"},
                    "mark": {"type": "line"},
                    "encoding": {
                        "x": {"field": "date", "type": "temporal", "title": None},
                        "y": {"field": "cumulative_return_pct", "type": "quantitative",
                              "title": "Cumulative return %"},
                        "tooltip": [
                            {"field": "date", "type": "temporal"},
                            {"field": "cumulative_return_pct"},
                            {"field": "drawdown_pct"},
                        ],
                    },
                },
            ],
            "resolve": {"scale": {"y": "independent"}},
        },
    )]


def _workforce_charts(df: pd.DataFrame, workforce_output: dict[str, Any] | None) -> list[ChartSpec]:
    if not workforce_output:
        return []
    # Each panel is isolated so one bad value drops only that panel.
    return [
        *_safe(_workforce_dept_chart, workforce_output, default=[]),
        *_safe(_workforce_tenure_chart, df, workforce_output, default=[]),
    ]


def _workforce_dept_chart(workforce_output: dict[str, Any]) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    by_dept =workforce_output.get("by_department")
    if isinstance(by_dept, list) and by_dept:
        values = [
            {"department": str(r.get("department")), "headcount": int(r.get("headcount") or 0)}
            for r in by_dept if isinstance(r, dict)
        ]
        # Small-cell suppression: departments under the minimum headcount are
        # merged into one "Other" bar (a no-op for output the tool already folded).
        folded_frame, folded = fold_small_groups(pd.DataFrame(values), "headcount", "department")
        if folded:
            values = [
                {"department": str(r["department"]), "headcount": int(r["headcount"])}
                for r in folded_frame.to_dict("records")
            ]
        if values:
            charts.append(ChartSpec(
                chart_id="workforce_headcount_by_dept",
                title="Headcount by Department",
                description="Number of employee records per department."
                + (f" {suppression_note(folded)}" if folded else ""),
                spec={
                    "data": {"values": values},
                    "mark": {"type": "bar"},
                    "height": max(140, 26 * len(values)),
                    "encoding": {
                        "y": {"field": "department", "type": "nominal", "sort": "-x", "title": None},
                        "x": {"field": "headcount", "type": "quantitative", "title": "Headcount"},
                        "tooltip": [{"field": "department"}, {"field": "headcount"}],
                    },
                },
            ))
    return charts


def _merge_small_bins(bins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Small-cell suppression for a pre-binned histogram: a non-empty bin
    with fewer than the minimum count is merged into its nearest non-empty
    neighbour (widening that bar) so no bar shows a handful of people."""
    if len(bins) < 2 or sum(b["count"] for b in bins) < min_cell_size():
        return bins
    merged: list[dict[str, Any]] = []
    carry: dict[str, Any] | None = None
    for b in bins:
        b = dict(b)
        if carry is not None:
            b["bin_start"] = carry["bin_start"]
            b["count"] += carry["count"]
            carry = None
        if 0 < b["count"] < min_cell_size():
            carry = b
            continue
        merged.append(b)
    if carry is not None:
        if merged:
            merged[-1]["bin_end"] = carry["bin_end"]
            merged[-1]["count"] += carry["count"]
        else:
            merged.append(carry)
    return merged


def _workforce_tenure_chart(df: pd.DataFrame, workforce_output: dict[str, Any]) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    hire_col = workforce_output.get("hire_date_column")
    if hire_col and hire_col in df.columns:
        hired = pd.to_datetime(df[hire_col], errors="coerce", format="mixed")
        exit_col = workforce_output.get("exit_date_column")
        if exit_col and exit_col in df.columns:
            ended = pd.to_datetime(df[exit_col], errors="coerce", format="mixed")
        else:
            ended = pd.Series(pd.NaT, index=df.index)
        ref_str = workforce_output.get("tenure_reference_date")
        reference = pd.Timestamp(ref_str) if ref_str else pd.Timestamp.now()
        ended = ended.fillna(reference)
        tenure_years = (ended - hired).dt.total_seconds() / (365.25 * 86400)
        tenure_years = tenure_years[tenure_years.notna() & (tenure_years >= 0)]
        if len(tenure_years) >= 5:
            bins = _merge_small_bins(_histogram_bins(tenure_years, max_bins=15))
            if bins:
                charts.append(ChartSpec(
                    chart_id="workforce_tenure_hist",
                    title="Tenure Distribution",
                    description="Years of tenure across employees (pre-binned).",
                    spec={
                        "data": {"values": bins},
                        "mark": {"type": "bar"},
                        "height": 200,
                        "encoding": {
                            "x": {"field": "bin_start", "bin": "binned", "type": "quantitative", "title": "Years"},
                            "x2": {"field": "bin_end"},
                            "y": {"field": "count", "type": "quantitative", "title": "Employees"},
                            "tooltip": [{"field": "bin_start"}, {"field": "bin_end"}, {"field": "count"}],
                        },
                    },
                ))
    return charts


_GRAIN_UNIT = {"monthly": "month", "weekly": "week", "daily": "day"}


def _segment_lift_chart(
    finding: dict[str, Any], results: list[dict[str, Any]], col_by_name: dict[str, ColumnProfile],
) -> ChartSpec | None:
    """Every level of the finding's dimension (segment_comparison's own
    comparisons) with its 95% CI as an error bar and the overall average as
    a dashed reference; the finding's own level is drawn solid. Falls back
    to that level vs everyone else when no sibling levels were reported."""
    ev = finding.get("evidence") or {}
    measure, dimension, level = ev.get("measure"), ev.get("dimension"), str(ev.get("level"))
    baseline = _num(ev.get("baseline_value"))
    if not measure or not dimension or baseline is None or _num(ev.get("level_value")) is None:
        return None
    if ev.get("n") is not None and is_small(ev.get("n") or 0):
        return None  # the finding's own segment is under the minimum cell size
    siblings: list[dict[str, Any]] = []
    for out in _tool_outputs(results, "segment_comparison"):
        siblings = [
            c for c in out.get("comparisons") or []
            if isinstance(c, dict) and c.get("measure") == measure and c.get("dimension") == dimension
            and _num(c.get("level_value")) is not None
            and not (c.get("n") is not None and is_small(c.get("n") or 0))
        ]
        if len(siblings) >= 2:
            break
    else:
        siblings = []

    values: list[dict[str, Any]] = [
        {
            "segment": str(c.get("level")),
            "value": _num(c.get("level_value")),
            "ci_lower": _num(c.get("ci_lower")),
            "ci_upper": _num(c.get("ci_upper")),
            "n": _to_primitive(c.get("n")),
            "focus": str(c.get("level")) == level,
        }
        for c in (siblings or [ev])
    ]
    values.sort(key=lambda r: r["value"], reverse=True)
    if len(values) > MAX_CATEGORIES_SHOWN:
        head = values[:MAX_CATEGORIES_SHOWN - 1]
        values = head + [r for r in values[MAX_CATEGORIES_SHOWN - 1:] if r["focus"]]
    if not siblings:
        values.append({"segment": "Everyone else", "value": baseline, "ci_lower": None, "ci_upper": None,
                       "n": _to_primitive(ev.get("n_rest")), "focus": False})

    is_rate = bool(ev.get("is_rate"))
    col = col_by_name.get(str(measure))
    unit_hint = "percent" if is_rate else (ev.get("unit_hint") or (col.unit_hint if col else None))
    fmt = axis_format(unit_hint)
    value_title = f"{humanize_label(measure)} rate" if is_rate else humanize_axis_title(measure, unit_hint)
    dim_title = humanize_label(dimension)

    horizontal = len(values) >= 9
    cat_ch, val_ch = ("y", "x") if horizontal else ("x", "y")
    sort = {"field": "value", "order": "descending"}
    cat_enc: dict[str, Any] = {"field": "segment", "type": "nominal", "sort": sort, "title": dim_title}
    if not horizontal:
        cat_enc["axis"] = {"labelAngle": 0}
    val_enc: dict[str, Any] = {"field": "value", "type": "quantitative", "title": value_title}
    _merge_axis_format(val_enc, fmt)
    layers: list[dict[str, Any]] = [{"mark": {"type": "bar"}, "encoding": {
        cat_ch: cat_enc,
        val_ch: val_enc,
        "opacity": {"condition": {"test": "datum.focus", "value": 1}, "value": 0.45},
        "tooltip": [
            {"field": "segment", "title": dim_title},
            {"field": "value", "title": value_title, **fmt},
            {"field": "ci_lower", "title": "95% CI from", **fmt},
            {"field": "ci_upper", "title": "95% CI to", **fmt},
            {"field": "n", "title": "Sample size"},
        ],
    }}]
    has_ci = any(r["ci_lower"] is not None and r["ci_upper"] is not None for r in values)
    if has_ci:
        layers.append({"mark": {"type": "rule"}, "encoding": {
            cat_ch: {"field": "segment", "type": "nominal", "sort": sort},
            val_ch: {"field": "ci_lower", "type": "quantitative", "title": value_title},
            f"{val_ch}2": {"field": "ci_upper"},
        }})
    overall = _num(ev.get("overall_mean")) if siblings else None
    if overall is not None:
        layers.append({
            "data": {"values": [{"overall": overall}]},
            "mark": {"type": "rule", "strokeDash": [4, 3]},
            "encoding": {val_ch: {"field": "overall", "type": "quantitative", "title": value_title}},
        })

    noun = "rate" if is_rate else "average"
    description = (
        f"{humanize_label(measure)} {noun} for each {dim_title} level" if siblings
        else f"{humanize_label(measure)} {noun} for {level} vs everyone else"
    ) + "; the solid bar is the segment this finding is about."
    if has_ci:
        description += " Error bars: 95% confidence interval of each segment's own value."
    if overall is not None:
        description += f" Dashed line: overall {noun} ({_fmt_value(overall, unit_hint)})."
    return ChartSpec(
        chart_id=f"segment_{_slug(measure)}_{_slug(dimension)}",
        title=f"{humanize_label(measure)} by {dim_title}",
        description=description,
        spec={
            "data": {"values": values},
            "height": max(160, 24 * len(values)) if horizontal else 260,
            "layer": layers,
        },
    )


def _lorenz_chart(df: pd.DataFrame, finding: dict[str, Any]) -> ChartSpec | None:
    """Lorenz curve of a concentration finding: entities ranked smallest to
    largest against their cumulative share of the measure, with the
    equality diagonal. Recomputed from the data when its columns are
    present, else drawn through the tool's own top-10/20/50% checkpoints."""
    ev = finding.get("evidence") or {}
    measure, entity = ev.get("measure_column"), ev.get("entity_column")
    if not measure or not entity:
        return None
    points: list[tuple[float, float]] = []
    if measure in df.columns and entity in df.columns:
        # Same rows the tool summed: both an entity and a numeric measure.
        pair = pd.DataFrame({"entity": df[entity], "value": pd.to_numeric(df[measure], errors="coerce")}).dropna()
        per_entity = pair.groupby("entity")["value"].sum()
        total = float(per_entity.sum())
        if len(per_entity) >= 2 and total > 0 and bool((per_entity >= 0).all()):
            cumulative = per_entity.sort_values().cumsum().to_numpy(dtype=float) / total
            n = len(cumulative)
            idx = np.unique(np.linspace(0, n - 1, min(n, LORENZ_MAX_POINTS)).astype(int))
            points = [(0.0, 0.0)] + [((i + 1) / n, float(cumulative[i])) for i in idx]
    if not points:
        # The bottom (100 - p)% of entities hold 1 - (top p% share).
        shares = [(p, _num(ev.get(f"top_{p}_pct_share"))) for p in (50, 20, 10)]
        if any(s is None for _, s in shares):
            return None
        points = [(0.0, 0.0), *((1 - p / 100, 1 - (s or 0.0)) for p, s in shares), (1.0, 1.0)]

    values = [{"entity_share": round(x * 100, 2), "measure_share": round(y * 100, 2)} for x, y in points]
    entity_label, measure_label = humanize_label(entity).lower(), humanize_label(measure).lower()
    domain = {"scale": {"domain": [0, 100]}}
    gini = _num(ev.get("gini_coefficient"))
    return ChartSpec(
        chart_id=f"lorenz_{_slug(measure)}_{_slug(entity)}",
        title=f"Concentration — {humanize_label(measure)} across {humanize_label(entity)}",
        description=(
            f"Lorenz curve: {entity_label} ranked from smallest to largest {measure_label}, against "
            f"their cumulative share of the total. The dashed diagonal is a perfectly even split; the "
            f"further the curve sags below it, the more the total rests on a few {entity_label}."
            + (f" Gini = {gini:.2f}." if gini is not None else "")
        ),
        spec={
            "height": 260,
            "layer": [
                {"data": {"values": values}, "mark": {"type": "area", "line": True}, "encoding": {
                    "x": {"field": "entity_share", "type": "quantitative",
                          "title": f"Cumulative % of {entity_label} (smallest first)", **domain},
                    "y": {"field": "measure_share", "type": "quantitative",
                          "title": f"Cumulative % of {measure_label}", **domain},
                    "tooltip": [{"field": "entity_share", "title": f"% of {entity_label}"},
                                {"field": "measure_share", "title": f"% of {measure_label}"}],
                }},
                {"data": {"values": [{"entity_share": 0, "measure_share": 0},
                                     {"entity_share": 100, "measure_share": 100}]},
                 "mark": {"type": "line", "strokeDash": [4, 3]},
                 "encoding": {"x": {"field": "entity_share", "type": "quantitative"},
                              "y": {"field": "measure_share", "type": "quantitative"}}},
            ],
        },
    )


def _change_comparison_chart(
    measure: str, prior: float, latest: float, unit: str, latest_label: str,
    unit_hint: str | None, agg_word: str,
) -> ChartSpec:
    """Two bars from a zero baseline — previous period vs latest — with the
    change written above the latest bar as "+x (+y%)". A waterfall of two
    levels and one thin step says nothing; this is the honest version when
    there is no additive segment breakdown to decompose."""
    change = latest - prior
    pct = f" ({change / abs(prior):+.1%})" if prior else ""
    rows = [
        {"period": f"Previous {unit}", "order": 0, "value": round(prior, 6),
         "label": _fmt_value(prior, unit_hint), "note": "", "latest": False},
        {"period": latest_label, "order": 1, "value": round(latest, 6),
         "label": _fmt_value(latest, unit_hint), "note": f"{_signed(change, unit_hint)}{pct}", "latest": True},
    ]
    fmt = axis_format(unit_hint)
    x_enc: dict[str, Any] = {"field": "period", "type": "nominal", "sort": {"field": "order"}, "title": None,
                             "axis": {"labelAngle": 0}}
    y_enc: dict[str, Any] = {"field": "value", "type": "quantitative", "scale": {"zero": True},
                             "title": f"{agg_word.capitalize()} {humanize_axis_title(measure, unit_hint)}"}
    _merge_axis_format(y_enc, fmt)
    return ChartSpec(
        chart_id=f"change_waterfall_{_slug(measure)}",
        title=f"{humanize_label(measure)} — {latest_label} vs Previous {unit.capitalize()}",
        description=f"{agg_word.capitalize()} {humanize_label(measure).lower()} in the previous {unit} and in "
                    f"{latest_label}, both from zero, with the change between them.",
        spec={
            "data": {"values": rows},
            "height": 240,
            "layer": [
                {"mark": {"type": "bar"}, "encoding": {
                    "x": x_enc, "y": y_enc,
                    "tooltip": [{"field": "period", "title": "Period"}, {"field": "label", "title": "Value"}],
                }},
                {"mark": {"type": "text", "dy": -7}, "encoding": {
                    "x": x_enc, "y": {"field": "value", "type": "quantitative"}, "text": {"field": "label"},
                }},
                {"mark": {"type": "text", "dy": -22, "fontWeight": "bold"},
                 "transform": [{"filter": {"field": "latest", "equal": True}}],
                 "encoding": {"x": x_enc, "y": {"field": "value", "type": "quantitative"}, "text": {"field": "note"}}},
            ],
        },
    )


def _change_waterfall_chart(
    finding: dict[str, Any], col_by_name: dict[str, ColumnProfile],
) -> ChartSpec | None:
    """Waterfall from the previous period's total to the latest one, one
    step per segment's contribution (change_analysis `segment_breakdown`)
    plus an explicit "Other segments" remainder so the bars reconcile to the
    headline change. Only an additive (summed) measure with a segment
    breakdown gets one; an averaged measure's segment deltas don't add up,
    and with no breakdown a waterfall is just two levels — both get a
    two-bar previous-vs-latest comparison instead."""
    ev = finding.get("evidence") or {}
    prior, latest = _num(ev.get("prior_period_value")), _num(ev.get("latest_value"))
    if prior is None or latest is None:
        return None
    measure = str(ev.get("measure_column") or finding.get("measure") or "value")
    col = col_by_name.get(measure)
    unit_hint = col.unit_hint if col else None
    unit = _GRAIN_UNIT.get(str(ev.get("period_grain")), "period")
    latest_label = str(ev.get("latest_period") or f"Latest {unit}")
    additive = ev.get("aggregation") == "sum"
    raw_breakdown = ev.get("segment_breakdown")
    breakdown: list[Any] = raw_breakdown if additive and isinstance(raw_breakdown, list) else []

    steps: list[tuple[str, float, bool]] = [(f"Previous {unit}", prior, True)]
    for seg in breakdown[:WATERFALL_MAX_SEGMENTS]:
        delta = _num(seg.get("delta")) if isinstance(seg, dict) else None
        if delta is not None:
            steps.append((str(seg.get("level")), delta, False))
    if len(steps) == 1:
        return _change_comparison_chart(
            measure, prior, latest, unit, latest_label, unit_hint, "total" if additive else "average",
        )
    rest = (latest - prior) - sum(value for _, value, is_total in steps if not is_total)
    if abs(rest) > 1e-9 * max(abs(prior), abs(latest), 1.0):
        steps.append(("Other segments", rest, False))
    steps.append((latest_label, latest, True))

    rows: list[dict[str, Any]] = []
    running = 0.0
    for order, (label, value, is_total) in enumerate(steps):
        start, end = (0.0, value) if is_total else (running, running + value)
        running = end
        rows.append({
            "step": label, "order": order,
            "start": round(start, 6), "end": round(end, 6), "top": round(max(start, end), 6),
            "total": is_total,
            "label": _fmt_value(value, unit_hint) if is_total else _signed(value, unit_hint),
        })

    agg_word = "total" if additive else "average"
    fmt = axis_format(unit_hint)
    x_enc: dict[str, Any] = {"field": "step", "type": "nominal", "sort": {"field": "order"}, "title": None}
    if len(rows) <= 6:
        x_enc["axis"] = {"labelAngle": 0}
    y_enc: dict[str, Any] = {"field": "start", "type": "quantitative",
                             "title": f"{agg_word.capitalize()} {humanize_axis_title(measure, unit_hint)}"}
    _merge_axis_format(y_enc, fmt)
    description = (
        f"How {agg_word} {humanize_label(measure).lower()} moved from the previous {unit} to {latest_label}"
        + (": each lighter bar is one segment's contribution." if len(steps) > 3 else ".")
        + " Solid bars are the period totals."
    )
    return ChartSpec(
        chart_id=f"change_waterfall_{_slug(measure)}",
        title=f"What Moved {humanize_label(measure)} — {latest_label} vs Previous {unit.capitalize()}",
        description=description,
        spec={
            "data": {"values": rows},
            "height": 260,
            "layer": [
                {"mark": {"type": "bar"}, "encoding": {
                    "x": x_enc, "y": y_enc, "y2": {"field": "end"},
                    "opacity": {"condition": {"test": "datum.total", "value": 1}, "value": 0.55},
                    "tooltip": [{"field": "step", "title": "Step"}, {"field": "label", "title": "Amount"}],
                }},
                {"mark": {"type": "text", "dy": -7}, "encoding": {
                    "x": x_enc, "y": {"field": "top", "type": "quantitative"}, "text": {"field": "label"},
                }},
            ],
        },
    )


def _group_ci_chart(
    df: pd.DataFrame, finding: dict[str, Any], col_by_name: dict[str, ColumnProfile],
) -> ChartSpec | None:
    """Mean ± 95% CI of a test finding's measure in every group of its
    dimension, on the test's own unit of analysis (one value per entity when
    it aggregated repeated rows), the leading post-hoc pair highlighted."""
    from scipy import stats

    ev = finding.get("evidence") or {}
    measure, dimension = str(finding.get("measure")), str(finding.get("dimension"))
    if (
        measure not in df.columns or dimension not in df.columns
        or not pd.api.types.is_numeric_dtype(df[measure])
        # The test quartile-binned a float grouping; regrouping it here
        # differently would show groups the test never compared.
        or pd.api.types.is_float_dtype(df[dimension])
    ):
        return None
    unit = ev.get("unit_of_analysis")
    entity = unit if isinstance(unit, str) and unit != "row" and unit in df.columns else None
    frame = df[[measure, dimension] + ([entity] if entity else [])].dropna()
    col = col_by_name.get(measure)
    if entity:
        agg = measure_aggregation(col) if col is not None and col.semantic_role == "measure" else "mean"
        frame = aggregate_to_entity(frame, entity, measure, agg, by=dimension)
    grouped = frame.groupby(frame[dimension].astype(str))[measure].agg(["mean", "std", "count"])
    # Small-cell suppression: a group under the minimum size is not drawn.
    dropped = int((grouped["count"] < max(2, min_cell_size())).sum())
    grouped = grouped[grouped["count"] >= max(2, min_cell_size())].nlargest(MAX_CATEGORIES_SHOWN, "count")
    if len(grouped) < 2:
        return None
    half = stats.t.ppf(0.975, grouped["count"] - 1) * grouped["std"].fillna(0.0) / np.sqrt(grouped["count"])

    post_hoc = ev.get("post_hoc")
    pair = post_hoc[0] if isinstance(post_hoc, list) and post_hoc and isinstance(post_hoc[0], dict) else None
    focus = {str(pair.get("group_a")), str(pair.get("group_b"))} if pair else set()
    values = [
        {
            "group": str(g),
            "mean": round(float(row["mean"]), 6),
            "ci_lower": round(float(row["mean"] - h), 6),
            "ci_upper": round(float(row["mean"] + h), 6),
            "n": int(row["count"]),
            "focus": not focus or str(g) in focus,
        }
        for (g, row), h in zip(grouped.iterrows(), half, strict=True)
    ]
    unit_hint = col.unit_hint if col else None
    fmt = axis_format(unit_hint)
    value_title = f"Average {humanize_axis_title(measure, unit_hint)}"
    dim_title = humanize_label(dimension)
    horizontal = len(values) >= 9
    cat_ch, val_ch = ("y", "x") if horizontal else ("x", "y")
    cat_enc: dict[str, Any] = {"field": "group", "type": "nominal",
                               "sort": {"field": "mean", "order": "descending"}, "title": dim_title}
    if not horizontal:
        cat_enc["axis"] = {"labelAngle": 0}
    val_enc: dict[str, Any] = {"field": "ci_lower", "type": "quantitative", "title": value_title,
                               "scale": {"zero": False}}
    _merge_axis_format(val_enc, fmt)
    opacity = {"condition": {"test": "datum.focus", "value": 1}, "value": 0.4}

    description = (
        f"Average {humanize_label(measure).lower()} in each {dim_title} group; whiskers are the "
        "95% confidence interval of each group's mean."
    )
    if dropped:
        description += f" {suppression_note(dropped)}"
    if pair:
        p_adj = _num(pair.get("p_adjusted"))
        description += (
            f" Largest follow-up gap: {pair.get('group_a')} vs {pair.get('group_b')}"
            + (f" (adjusted p={p_adj:.3g})" if p_adj is not None else "") + ", highlighted."
        )
    return ChartSpec(
        chart_id=f"groups_{_slug(measure)}_{_slug(dimension)}",
        title=f"{humanize_label(measure)} across {dim_title}",
        description=description,
        spec={
            "data": {"values": values},
            "height": max(160, 24 * len(values)) if horizontal else 240,
            "layer": [
                {"mark": {"type": "rule"}, "encoding": {
                    cat_ch: cat_enc, val_ch: val_enc, f"{val_ch}2": {"field": "ci_upper"}, "opacity": opacity,
                }},
                {"mark": {"type": "circle", "size": 90, "opacity": 1}, "encoding": {
                    cat_ch: cat_enc,
                    val_ch: {"field": "mean", "type": "quantitative", "title": value_title},
                    "opacity": opacity,
                    "tooltip": [
                        {"field": "group", "title": dim_title},
                        {"field": "mean", "title": value_title, **fmt},
                        {"field": "ci_lower", "title": "95% CI from", **fmt},
                        {"field": "ci_upper", "title": "95% CI to", **fmt},
                        {"field": "n", "title": "Sample size"},
                    ],
                }},
            ],
        },
    )
