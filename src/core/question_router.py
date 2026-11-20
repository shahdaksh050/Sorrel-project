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

from src.core.findings import Finding
from src.core.profiler import DatasetProfile, ground_objective_column

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
    expected_decision_value: float = 0.0


_DECISION_VALUE: dict[QuestionFamily, float] = {
    "decide": 0.95, "explain": 0.85, "predict": 0.80, "forecast": 0.75,
    "compare": 0.70, "monitor": 0.65, "audit": 0.65, "detect": 0.60,
    "associate": 0.45, "segment": 0.45, "describe": 0.35,
    "fairness": 0.75, "benchmark": 0.70,
}


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
        expected_decision_value=_DECISION_VALUE[primary],
    )


@dataclass(frozen=True)
class ComparisonObjective:
    """Column roles for an explicit comparison, shared by planning and reporting."""

    measure: str | None
    dimension: str | None
    hourly: bool
    ascending: bool
    descending: bool
    missing_dimension: bool

    def tool_parameters(self) -> dict[str, str]:
        if not self.measure or self.missing_dimension:
            return {}
        if self.hourly:
            return {"value_column": self.measure}
        params = {"measure_column": self.measure}
        if self.dimension:
            params["dimension_column"] = self.dimension
        return params

    def matches(self, finding: Finding) -> bool:
        if self.missing_dimension or not self.measure or finding.measure != self.measure:
            return False
        if self.hourly:
            return finding.dimension == "hour_of_day" and finding.source_tool == "time_series_analysis"
        return (
            finding.kind in ("segment_lift", "test", "invariance", "uniformity")
            and bool(finding.dimension)
            and (self.dimension is None or finding.dimension == self.dimension)
        )


def comparison_objective(objective: str, profile: DatasetProfile | None) -> ComparisonObjective | None:
    """Resolve named measures independently of prediction targets and grouping columns."""
    if not objective or profile is None:
        return None
    text = objective.lower()
    hourly = bool(re.search(r"\b(time of day|hour of (?:the )?day|hourly pattern|by hour)\b", text))
    grouped = bool(re.search(r"\b(groups?|cohorts?|segments?|categories|compare|differs?|across|by)\b", text))
    if not (hourly or grouped) or re.search(r"\b(predict|forecast|cluster|segment customers)\b", text):
        return None
    parts = re.split(r"\b(?:by|across)\b", text, maxsplit=1)
    candidates = [c.name for c in profile.columns if c.is_measure() or c.is_flag() or c.kind in ("numeric", "boolean")]
    measure = ground_objective_column(candidates, parts[0])
    dim_names = [
        c.name for c in profile.columns
        if c.name != measure and c.kind not in ("identifier", "constant")
        and (c.is_dimension() or c.kind == "categorical")
    ]
    dim_text = parts[-1] if len(parts) == 2 else text
    dimension = None if hourly else ground_objective_column(dim_names, dim_text)
    missing_dim = not hourly and len(parts) == 2 and not dimension and not re.fullmatch(
        r"\s*(?:customer |user )?(?:groups?|cohorts?|segments?)[?.! ]*", dim_text
    )
    return ComparisonObjective(
        measure=measure, dimension=dimension, hourly=hourly,
        ascending=bool(re.search(r"\b(lowest|smallest)\b", text)),
        descending=bool(re.search(r"\b(highest|largest)\b", text)),
        missing_dimension=missing_dim,
    )
