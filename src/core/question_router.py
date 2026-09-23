"""
Question-Type Router — route user objective into Axis E analysis families.

Implements FutureScope.md Section 5.6 & Axis E taxonomy:
Families:
- describe: summary statistics, distributions, histograms
- compare: group differences, ANOVA, segment comparisons
- associate: correlations, cross-tabs, scatterplots, relations
- predict: regression, classification models, driver importances
- explain: driver analysis, causal-guard checked associations
- forecast: time-series forecasting, trend projections
- detect: anomalies, outliers, regime level-shifts
- segment: clustering, cohort analysis, customer segmentation
- monitor: control charts, threshold checks, trend stability
- audit: Benford's law, duplicate checks, data integrity

Also recommends the primary tool pipeline matching the intent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

QuestionFamily = Literal[
    "describe",
    "compare",
    "associate",
    "predict",
    "explain",
    "forecast",
    "detect",
    "segment",
    "monitor",
    "audit",
    "fairness",
    "benchmark",
    "decide",
]

_FAMILY_PATTERNS: dict[QuestionFamily, list[re.Pattern[str]]] = {
    "forecast": [
        re.compile(r"\b(?:forecast|future|project|predict future|next (?:month|year|quarter|week))\b", re.IGNORECASE),
    ],
    "detect": [
        re.compile(r"\b(?:detect|anomal|outlier|unusual|spike|drop|glitch|defect)\b", re.IGNORECASE),
    ],
    "segment": [
        re.compile(r"\b(?:segment|cluster|cohort|group customers|persona)\b", re.IGNORECASE),
    ],
    "compare": [
        re.compile(r"\b(?:compare|difference between|versus|vs\.?|across (?:groups|regions|departments))\b", re.IGNORECASE),
    ],
    "predict": [
        re.compile(r"\b(?:predict|model|classification|machine learning|train|estimate)\b", re.IGNORECASE),
    ],
    "explain": [
        re.compile(r"\b(?:what drives|drivers? of|explain|factors affecting|cause|root cause|why)\b", re.IGNORECASE),
    ],
    "associate": [
        re.compile(r"\b(?:correlat|move together|relationship between|associated with|co-movement)\b", re.IGNORECASE),
    ],
    "audit": [
        re.compile(r"\b(?:audit|fraud|integrity|benford|falsif|manipulat)\b", re.IGNORECASE),
    ],
    "monitor": [
        re.compile(r"\b(?:monitor|control chart|stability|tracking|alert|drift)\b", re.IGNORECASE),
    ],
    "describe": [
        re.compile(r"\b(?:describe|summary|overview|distribution|breakdown|profile)\b", re.IGNORECASE),
    ],
}

_FAMILY_TOOL_MAP: dict[QuestionFamily, list[str]] = {
    "describe": ["statistical_analysis", "visualization"],
    "compare": ["segment_comparison", "statistical_analysis", "visualization"],
    "associate": ["statistical_analysis", "visualization"],
    "predict": ["regression_analysis", "train_model", "evaluate_model"],
    "explain": ["regression_analysis", "segment_comparison", "statistical_analysis"],
    "forecast": ["time_series_analysis", "forecast_analysis", "visualization"],
    "detect": ["detect_outliers", "anomaly_analysis", "visualization"],
    "segment": ["clustering", "cohort_analysis", "visualization"],
    "monitor": ["time_series_analysis", "change_analysis", "visualization"],
    "audit": ["statistical_analysis", "detect_outliers"],
    "fairness": ["equity_analysis", "segment_comparison"],
    "benchmark": ["statistical_analysis", "segment_comparison"],
    "decide": ["statistical_analysis", "segment_comparison"],
}


@dataclass
class QuestionRoutingResult:
    """Routing decisions for the user's objective."""

    primary_family: QuestionFamily
    secondary_families: list[QuestionFamily] = field(default_factory=list)
    recommended_tools: list[str] = field(default_factory=list)
    rationale: str = ""


def route_question(
    objective: str,
    profile: Any = None,
) -> QuestionRoutingResult:
    """
    Route an objective string into an Axis E QuestionFamily and recommended tools.
    Optionally informed by the DatasetProfile (e.g. time-series data reinforces forecast).
    """
    matches: list[tuple[QuestionFamily, int]] = []

    for family, patterns in _FAMILY_PATTERNS.items():
        score = 0
        for pat in patterns:
            if pat.search(objective):
                score += 1
        if score > 0:
            matches.append((family, score))

    # Profile-informed adjustments. Previously this only boosted "forecast"
    # when a forecast keyword had *already* matched (i.e. it boosted a
    # signal that was already the strongest one, doing nothing useful) —
    # the point of a profile-informed adjustment is to add forecast as a
    # candidate even when the objective never used the word.
    if profile is not None and getattr(profile, "is_time_series", False):
        matches.append(("forecast", 1))

    primary: QuestionFamily
    secondary: list[QuestionFamily]
    if not matches:
        # Default family is describe
        primary = "describe"
        secondary = []
        rationale = "No explicit analytical intent keywords found; defaulted to descriptive overview."
    else:
        matches.sort(key=lambda m: m[1], reverse=True)
        primary = matches[0][0]
        secondary = [f for f, _ in matches[1:] if f != primary]
        rationale = f"Matched primary intent '{primary}' based on objective keywords."

    recommended_tools = list(_FAMILY_TOOL_MAP.get(primary, ["statistical_analysis", "visualization"]))
    for sec in secondary:
        for t in _FAMILY_TOOL_MAP.get(sec, []):
            if t not in recommended_tools:
                recommended_tools.append(t)

    return QuestionRoutingResult(
        primary_family=primary,
        secondary_families=secondary,
        recommended_tools=recommended_tools,
        rationale=rationale,
    )
