"""
Unit tests for Data Science Cockpit UI and Actions Tab.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from ui.components.cards import render_evidence_html, render_finding_card
from ui.tabs.actions_tab import render_actions_tab

from src.core.dashboard_charts_basic import _forecast_chart


def test_forecast_chart_spec_generation() -> None:
    forecast_output = {
        "value_column": "active_power",
        "period_grain": "hour",
        "model": "exponential_smoothing",
        "mase": 0.654,
        "chart": {
            "type": "vega_lite",
            "title": "Hourly active power forecast",
            "vega_lite": {"data": {"values": []}},
        },
    }
    spec = _forecast_chart(forecast_output)
    assert spec is not None
    assert spec.chart_id == "forecast_trajectory"
    assert spec.size == "wide"
    assert "Hourly active power forecast" in spec.title
    assert "MASE 0.65" in spec.description


def test_render_evidence_html_suppresses_duplicate_headline() -> None:
    same_text = "Daily average Global_active_power is forecast to reach 0.65 (-30% vs latest)"
    finding = {
        "finding_id": "f_1",
        "headline": same_text,
        "detail": same_text,  # Identical to headline
        "evidence": {"status": "held", "tests": [{"name": "forecast", "passed": True}]},
    }
    html_out = render_evidence_html(finding)
    assert '<h3 class="evidence-title">' in html_out
    # Must NOT render <p class="evidence-detail"> when detail is identical to headline
    assert '<p class="evidence-detail">' not in html_out


def test_render_evidence_html_renders_distinct_detail() -> None:
    finding = {
        "finding_id": "f_2",
        "headline": "Headline statement",
        "detail": "Distinct supporting elaboration explaining the mechanics.",
        "evidence": {"status": "held", "tests": [{"name": "t_test", "passed": True}]},
    }
    html_out = render_evidence_html(finding)
    assert '<h3 class="evidence-title">' in html_out
    assert '<p class="evidence-detail">Distinct supporting elaboration' in html_out


def test_render_finding_card_suppresses_duplicate_subtext() -> None:
    same_text = "Customer churn rises among monthly contract holders"
    finding = {
        "finding_id": "f_3",
        "headline": same_text,
        "detail": same_text,
        "evidence": {"status": "held"},
    }
    card_html = render_finding_card(finding, chart_finding_ids=set())
    # Subtext should not duplicate the headline
    assert card_html.count(same_text) == 1


@patch("streamlit.markdown")
@patch("streamlit.columns")
@patch("streamlit.container")
def test_render_actions_tab_execution(mock_container: MagicMock, mock_columns: MagicMock, mock_markdown: MagicMock) -> None:
    mock_columns.return_value = [MagicMock(), MagicMock()]
    mock_container.return_value.__enter__.return_value = MagicMock()

    mock_run_view = MagicMock()
    mock_run_view.findings = [
        {"kind": "segment_lift", "headline": "High tenure group lift", "effect": 0.25, "action_category": "quick_wins"}
    ]
    mock_run_view.recommendations = ["Pilot load shifting strategy during peak hours."]

    tool_results = [
        {
            "tool_name": "scenario_simulation",
            "output": {
                "target_column": "active_power",
                "scenarios": [
                    {
                        "scenario_name": "Shift 20% peak load",
                        "predicted_delta": -0.45,
                        "base_value": 2.1,
                        "counterfactual_value": 1.65,
                    }
                ],
            },
        }
    ]

    render_actions_tab(report={}, tool_results=tool_results, run_view=mock_run_view)
    assert mock_markdown.called
