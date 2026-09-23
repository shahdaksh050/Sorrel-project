"""
Unit tests for Question-Type Router (FutureScope Phase 2).
"""
from __future__ import annotations

from src.core.question_router import route_question


def test_route_forecast() -> None:
    res = route_question("Forecast electricity demand for next month")
    assert res.primary_family == "forecast"
    assert "time_series_analysis" in res.recommended_tools
    assert "forecast_analysis" in res.recommended_tools


def test_route_explain_driver() -> None:
    res = route_question("What drives employee attrition in our engineering team?")
    assert res.primary_family == "explain"
    assert "regression_analysis" in res.recommended_tools or "segment_comparison" in res.recommended_tools


def test_route_compare() -> None:
    res = route_question("Compare customer satisfaction between North and South regions")
    assert res.primary_family == "compare"
    assert "segment_comparison" in res.recommended_tools


def test_route_detect_outliers() -> None:
    res = route_question("Detect anomalous transactions and spikes in fraud")
    assert res.primary_family == "detect"
    assert "detect_outliers" in res.recommended_tools


def test_route_segment() -> None:
    res = route_question("Segment our users into cohorts based on activity")
    assert res.primary_family == "segment"
    assert "clustering" in res.recommended_tools or "cohort_analysis" in res.recommended_tools


def test_route_default_describe() -> None:
    res = route_question("Give me an initial overview of this dataset")
    assert res.primary_family == "describe"
    assert "statistical_analysis" in res.recommended_tools


class _FakeProfile:
    def __init__(self, is_time_series: bool) -> None:
        self.is_time_series = is_time_series


def test_route_time_series_profile_surfaces_forecast_when_ambiguous() -> None:
    """Planted case: no explicit intent keywords at all, but the profile says
    the data is a time series -> forecast must be surfaced as the primary
    family. Previously the profile-informed boost was gated on forecast
    already having matched by keyword, i.e. it never fired for exactly this
    (the actual intended) case."""
    res = route_question("Tell me about this data", profile=_FakeProfile(is_time_series=True))
    assert res.primary_family == "forecast"
    assert "forecast_analysis" in res.recommended_tools


def test_route_time_series_profile_does_not_override_explicit_intent() -> None:
    """Null case: an explicit non-forecast ask on time-series data keeps its
    own primary family — the profile signal augments, it doesn't hijack."""
    res = route_question(
        "Compare revenue between North and South regions",
        profile=_FakeProfile(is_time_series=True),
    )
    assert res.primary_family == "compare"
