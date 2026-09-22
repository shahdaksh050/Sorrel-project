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
