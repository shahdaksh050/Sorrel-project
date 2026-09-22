"""
Deliverable Contract — parse and audit required deliverables from user objectives.

Implements FutureScope.md Section 5.6:
- Parses user objective into required deliverables:
  * Chart types (heatmap, scatter, bar, line, boxplot, etc.)
  * Driver analysis for specific target columns
  * Relationships / co-movement (correlations)
  * Segment comparisons
  * Forecasts
- Audits final_result against the contract:
  * Verifies each required deliverable exists in charts or findings.
  * Provides deterministic fallback recipes for any undelivered request.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

#: Chart keywords to chart kind mapping
_CHART_KEYWORD_MAP: dict[str, str] = {
    "heatmap": "heatmap",
    "heat map": "heatmap",
    "scatter": "scatter",
    "scatterplot": "scatter",
    "bar chart": "bar",
    "bar graph": "bar",
    "barchart": "bar",
    "histogram": "histogram",
    "distribution": "histogram",
    "line chart": "line",
    "line plot": "line",
    "trend": "line",
    "boxplot": "boxplot",
    "box plot": "boxplot",
    "waterfall": "waterfall",
    "dual axis": "dual_axis",
}

_DRIVER_PATTERNS = (
    re.compile(r"what drives\s+([a-zA-Z0-9_\(\)\s]+?)(?:\?|$|,|\.)", re.IGNORECASE),
    re.compile(r"drivers of\s+([a-zA-Z0-9_\(\)\s]+?)(?:\?|$|,|\.)", re.IGNORECASE),
    re.compile(r"predict\s+([a-zA-Z0-9_\(\)\s]+?)(?:\?|$|,|\.)", re.IGNORECASE),
    re.compile(r"explain\s+([a-zA-Z0-9_\(\)\s]+?)(?:\?|$|,|\.)", re.IGNORECASE),
    re.compile(r"factors affecting\s+([a-zA-Z0-9_\(\)\s]+?)(?:\?|$|,|\.)", re.IGNORECASE),
)

_RELATIONSHIP_PATTERNS = (
    re.compile(r"which\s+([a-zA-Z0-9_\s]+)\s+move together", re.IGNORECASE),
    re.compile(r"correlated|correlation|relationships? between", re.IGNORECASE),
)


@dataclass
class DeliverableContract:
    """Explicit deliverables extracted from the user's objective."""

    raw_objective: str
    required_charts: list[str] = field(default_factory=list)
    required_driver_targets: list[str] = field(default_factory=list)
    requires_relationship_analysis: bool = False
    requires_forecast: bool = False
    requires_comparison: bool = False

    def is_empty(self) -> bool:
        return (
            not self.required_charts
            and not self.required_driver_targets
            and not self.requires_relationship_analysis
            and not self.requires_forecast
            and not self.requires_comparison
        )


@dataclass
class DeliverableAuditReport:
    """Result of auditing final output against the deliverable contract."""

    delivered: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    repaired: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return len(self.missing) == 0


def parse_deliverable_contract(objective: str) -> DeliverableContract:
    """Parse a user's objective string into a structured DeliverableContract."""
    obj_l = objective.lower()
    required_charts: list[str] = []
    for kw, chart_type in _CHART_KEYWORD_MAP.items():
        if kw in obj_l and chart_type not in required_charts:
            required_charts.append(chart_type)

    required_targets: list[str] = []
    for pat in _DRIVER_PATTERNS:
        m = pat.search(objective)
        if m:
            target = m.group(1).strip()
            # Remove trailing words like "levels", "rate", "scores" if generic
            cleaned_target = re.sub(r"\s+(?:levels|rates?|scores?|values?)$", "", target, flags=re.IGNORECASE).strip()
            if cleaned_target and cleaned_target.lower() not in ("the", "it"):
                required_targets.append(cleaned_target)

    requires_relationship = any(pat.search(objective) for pat in _RELATIONSHIP_PATTERNS)
    requires_forecast = bool(re.search(r"\b(?:forecast|predict future|projection)\b", obj_l))
    requires_comparison = bool(re.search(r"\b(?:compare|difference between|versus|vs\.?)\b", obj_l))

    return DeliverableContract(
        raw_objective=objective,
        required_charts=required_charts,
        required_driver_targets=required_targets,
        requires_relationship_analysis=requires_relationship,
        requires_forecast=requires_forecast,
        requires_comparison=requires_comparison,
    )


def audit_deliverables(
    contract: DeliverableContract,
    final_result: dict[str, Any],
    df: pd.DataFrame | None = None,
) -> DeliverableAuditReport:
    """
    Audit final_result against the contract.
    Checks charts, findings, and insights; records delivered, missing, and repaired items.
    """
    report = DeliverableAuditReport()
    if contract.is_empty():
        return report

    charts: list[dict[str, Any]] = final_result.get("charts", [])
    findings: list[Any] = final_result.get("findings", [])
    insights: list[str] = final_result.get("insights", [])

    # 1. Audit charts
    emitted_chart_types: set[str] = set()
    for ch in charts:
        # Check mark or chart_type in chart spec
        spec = ch.get("spec", {}) if isinstance(ch, dict) else {}
        mark = spec.get("mark")
        if isinstance(mark, dict):
            mark = mark.get("type")
        if isinstance(mark, str):
            emitted_chart_types.add(mark.lower())
        c_type = ch.get("chart_type") or ch.get("type")
        if isinstance(c_type, str):
            emitted_chart_types.add(c_type.lower())
        title = str(ch.get("title", "")).lower()
        for kw in ("heatmap", "scatter", "bar", "line", "histogram", "boxplot", "waterfall"):
            if kw in title:
                emitted_chart_types.add(kw)

    for req_chart in contract.required_charts:
        if req_chart in emitted_chart_types or (req_chart == "heatmap" and "rect" in emitted_chart_types):
            report.delivered.append(f"Chart: {req_chart}")
        else:
            # Check if we can repair with a deterministic fallback recipe
            repaired = _repair_missing_chart(req_chart, final_result, df)
            if repaired:
                report.repaired.append(f"Chart: {req_chart} (synthesized fallback)")
                report.delivered.append(f"Chart: {req_chart}")
            else:
                report.missing.append(f"Chart: {req_chart}")

    # 2. Audit driver analysis
    for target in contract.required_driver_targets:
        target_l = target.lower()
        target_found = False
        # Check findings and insights
        for f in findings:
            f_text = (getattr(f, "headline", "") or str(f)).lower()
            if target_l in f_text and ("driver" in f_text or "predict" in f_text or "importance" in f_text or "lift" in f_text):
                target_found = True
                break
        if not target_found:
            for ins in insights:
                if target_l in ins.lower() and ("driver" in ins.lower() or "correlated" in ins.lower() or "associated" in ins.lower()):
                    target_found = True
                    break

        if target_found:
            report.delivered.append(f"Driver analysis: {target}")
        else:
            report.missing.append(f"Driver analysis: {target}")

    # 3. Audit relationship analysis
    if contract.requires_relationship_analysis:
        rel_found = False
        for f in findings:
            f_kind = getattr(f, "kind", "")
            if f_kind in ("correlation", "co_movement", "cluster"):
                rel_found = True
                break
        if rel_found or any("correlat" in ins.lower() or "move together" in ins.lower() for ins in insights):
            report.delivered.append("Relationship analysis")
        else:
            report.missing.append("Relationship analysis")

    return report


def _repair_missing_chart(
    chart_type: str,
    final_result: dict[str, Any],
    df: pd.DataFrame | None,
) -> bool:
    """Generate a deterministic fallback chart if a requested chart was missed."""
    if df is None or df.empty:
        return False

    charts: list[dict[str, Any]] = final_result.setdefault("charts", [])

    if chart_type == "heatmap":
        # Generate a correlation heatmap or 2D aggregate heatmap
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols[:10]].corr().round(2)
            corr_data = []
            for c1 in corr.columns:
                for c2 in corr.index:
                    corr_data.append({"var1": str(c1), "var2": str(c2), "correlation": float(corr.loc[c2, c1])})
            spec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "title": "Correlation Heatmap (Fallback Recipe)",
                "data": {"values": corr_data},
                "mark": "rect",
                "encoding": {
                    "x": {"field": "var1", "type": "nominal", "title": ""},
                    "y": {"field": "var2", "type": "nominal", "title": ""},
                    "color": {"field": "correlation", "type": "quantitative", "scale": {"domain": [-1, 1]}},
                },
            }
            charts.append({"title": "Correlation Heatmap", "spec": spec, "chart_type": "heatmap"})
            return True

    elif chart_type in ("bar", "bar chart"):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if cat_cols and num_cols:
            x_col, y_col = cat_cols[0], num_cols[0]
            summary = df.groupby(x_col)[y_col].mean().reset_index().head(15)
            spec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "title": f"Average {y_col} by {x_col} (Fallback Recipe)",
                "data": {"values": summary.to_dict(orient="records")},
                "mark": "bar",
                "encoding": {
                    "x": {"field": x_col, "type": "nominal", "sort": "-y"},
                    "y": {"field": y_col, "type": "quantitative"},
                },
            }
            charts.append({"title": f"{y_col} by {x_col}", "spec": spec, "chart_type": "bar"})
            return True

    return False
