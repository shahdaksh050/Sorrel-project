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


def test_business_headline_and_takeaway() -> None:
    from src.core.plain_language import business_headline, business_takeaway

    # Case 1: Structured segment lift finding with rating metric
    finding_delivery = {
        "finding_id": "f_delivery",
        "kind": "segment_lift",
        "dimension": "category",
        "level": "Delivery",
        "measure": "rating",
        "effect": 1.72,
        "evidence": {"n": 68, "n_rest": 121, "ci_lower": 3.27, "ci_upper": 3.53},
        "headline": "Delivery orders average rating of 4.85 vs 1.78 for everyone else (difference +2.15, d=1.42, n=68 vs 121).",
    }

    hl = business_headline(finding_delivery)
    assert "Delivery Category Leads Satisfaction with +172% Higher Rating" in hl
    assert "difference +2.15" not in hl
    assert "n=68" not in hl

    takeaway = business_takeaway(finding_delivery)
    assert "Operational Takeaway: Prioritize the 'Delivery' segment" in takeaway
    assert "n=68 records" in takeaway

    # Case 2: Stripping trailing academic statistics notation
    raw_finding = {
        "finding_id": "f_academic",
        "headline": "West-region orders run 25% above every other region (1.25x, n=340 vs n=980).",
    }
    hl_stripped = business_headline(raw_finding)
    assert "West-region orders run 25% above every other region." in hl_stripped
    assert "n=340" not in hl_stripped
    assert "1.25x" not in hl_stripped


def test_render_methodology_drawer() -> None:
    from ui.components.cards import render_methodology_drawer

    finding = {
        "finding_id": "f_method",
        "effect": 1.42,
        "effect_kind": "d",
        "p_value": 0.0001,
        "p_adjusted": 0.0004,
        "source_tool": "segment_comparison",
        "detail": "n=68 in segment vs 121 elsewhere; p=0.0001, p_adjusted=0.0004; 95% CI [3.27, 3.53].",
        "evidence": {
            "n": 68,
            "n_rest": 121,
            "ci_lower": 3.27,
            "ci_upper": 3.53,
            "status": "held",
            "tests": [{"name": "robust", "passed": True}],
        },
    }

    drawer_html = render_methodology_drawer(finding)
    assert '<details class="tech-note"' in drawer_html
    assert "<summary>Evidence & Statistical Methodology</summary>" in drawer_html
    assert "n = 68 in segment vs n = 121 baseline" in drawer_html
    assert "p = &lt;0.001" in drawer_html
    assert "[3.27, 3.53]" in drawer_html
    assert "Segment Comparison" in drawer_html


def test_classify_action_matrix_populates_all_four_quadrants() -> None:
    from src.core.findings import Finding, classify_action_matrix

    # 1. Quick Win: High impact, high feasibility controllable lever
    f1 = Finding(
        finding_id="f_qw",
        kind="segment_lift",
        headline="Enterprise tier customers yield +140% higher spend",
        dimension="subscription_tier",
        level="Enterprise",
        measure="spend",
        effect=1.40,
        confidence=0.88,
        evidence={"n": 240, "checks": {"robust": True, "paradox_free": True}},
    )

    # 2. Strategic Bet: High impact, multi-factor synergy / complex feasibility
    f2 = Finding(
        finding_id="f_sb",
        kind="interaction_synergy",
        headline="Joint Channel='Mobile' x Promo='VIP' yields +65% lift",
        dimension="channel_x_promo",
        level="Mobile x VIP",
        measure="conversion",
        effect=0.65,
        confidence=0.70,
        evidence={"n": 180, "checks": {"robust": True, "paradox_free": True}},
    )

    # 3. Pilot / A/B Test: High opportunity observational contrast with disclaimer
    f3 = Finding(
        finding_id="f_pilot",
        kind="segment_lift",
        headline="New delivery category shows +85% repeat order frequency",
        dimension="delivery_category",
        level="Express",
        measure="order_frequency",
        effect=0.85,
        confidence=0.78,
        caveats=["Observational cohort contrast; unadjusted for selection effect."],
        evidence={"n": 150, "checks": {"robust": True, "paradox_free": True}},
    )

    # 4. Incremental Gain: Moderate impact, high feasibility
    f4 = Finding(
        finding_id="f_inc",
        kind="segment_lift",
        headline="Email opt-in users spend +4% more session time",
        dimension="channel",
        level="Email",
        measure="session_time",
        effect=0.04,
        confidence=0.82,
        evidence={"n": 500, "checks": {"robust": True, "paradox_free": True}},
    )

    # 5. Risk Guardrail / Watch: Paradox reversal or fragile
    f5 = Finding(
        finding_id="f_risk",
        kind="segment_lift",
        headline="Flash discount increases order volume",
        dimension="discount",
        level="Flash",
        measure="volume",
        effect=0.30,
        confidence=0.65,
        evidence={"checks": {"paradox_free": False}},
    )

    matrix = classify_action_matrix([f1, f2, f3, f4, f5])

    assert len(matrix["quick_wins"]) >= 1
    assert matrix["quick_wins"][0]["finding_id"] == "f_qw"
    assert "Enterprise" in matrix["quick_wins"][0]["lever"]

    assert len(matrix["strategic_bets"]) >= 1
    assert matrix["strategic_bets"][0]["finding_id"] == "f_sb"

    assert len(matrix["pilot"]) >= 1
    assert matrix["pilot"][0]["finding_id"] == "f_pilot"
    assert "Pilot" in matrix["pilot"][0]["lever"]

    assert len(matrix["incremental"]) >= 1
    assert matrix["incremental"][0]["finding_id"] == "f_inc"

    assert len(matrix["watch_caveats"]) >= 1
    assert matrix["watch_caveats"][0]["finding_id"] == "f_risk"


@patch("streamlit.markdown")
@patch("streamlit.columns")
@patch("streamlit.container")
def test_render_native_swarm_cockpit(mock_container: MagicMock, mock_columns: MagicMock, mock_markdown: MagicMock) -> None:
    from ui.components.swarm_dashboard import render_swarm_dashboard

    mock_columns.return_value = [MagicMock() for _ in range(5)]
    mock_container.return_value.__enter__.return_value = MagicMock()

    mock_dashboard = [
        {"chart_id": "c_hero", "title": "Lead Power Trajectory", "spec": {"mark": "line"}},
        {"chart_id": "c_seg", "title": "Segment Distribution", "spec": {"mark": "bar"}},
    ]
    mock_report = {
        "title": "Air Quality Assessment",
        "lead_narrative": "Autonomous analysis of continuous feature distributions.",
        "row_count": 9357,
        "column_count": 15,
        "quality_score": "98.2 / 100",
    }

    # Verify native cockpit rendering (no components.html)
    with patch("streamlit.components.v1.html") as mock_html:
        render_swarm_dashboard(
            dashboard=mock_dashboard,
            report=mock_report,
            vega_cfg={"view": {"stroke": "transparent"}},
        )
        assert mock_markdown.called
        # Verify that native mode does NOT invoke components.html
        mock_html.assert_not_called()


def test_render_evidence_html_business_first_shows_takeaway_and_folds_stats() -> None:
    finding = {
        "finding_id": "f_bf_test",
        "kind": "segment_lift",
        "dimension": "category",
        "level": "Delivery",
        "measure": "rating",
        "effect": 1.72,
        "detail": "n=68 in segment vs 121 elsewhere; p=0.0001, p_adjusted=0.0004; 95% CI [3.27, 3.53].",
        "evidence": {
            "n": 68,
            "n_rest": 121,
            "ci_lower": 3.27,
            "ci_upper": 3.53,
            "status": "held",
            "tests": [{"name": "robust", "passed": True}],
        },
        "headline": "Delivery orders average rating of 4.85 vs 1.78 for everyone else (difference +2.15, d=1.42, n=68 vs 121).",
    }
    html_out = render_evidence_html(finding, business_first=True)
    # The main visible detail must be the operational takeaway, NOT raw statistics
    assert "Operational Takeaway: Prioritize the" in html_out
    assert "Delivery" in html_out
    assert "n=68 in segment vs 121 elsewhere" not in html_out
    assert "p=0.0001" not in html_out
    assert "95% CI" not in html_out


def test_business_headline_from_raw_unstructured_string() -> None:
    from src.core.plain_language import business_headline

    raw_finding = {
        "headline": "Delivery orders average rating of 4.85 vs 1.78 for everyone else (difference +2.15, d=1.42, n=68 vs 121)."
    }
    hl = business_headline(raw_finding)
    assert hl == "Delivery Category Leads Satisfaction with +172% Higher Rating"


def test_classify_action_matrix_benign_caveats_not_fatal() -> None:
    from src.core.findings import Finding, classify_action_matrix

    f = Finding(
        finding_id="f_benign",
        kind="segment_lift",
        headline="Enterprise segment delivers +45% higher ARR",
        measure="arr",
        dimension="tier",
        level="Enterprise",
        effect=0.45,
        confidence=0.85,
        evidence={"checks": {"robust": True, "paradox_free": True}},
        evidence_tier="causal_action",
        caveats=["Two-tailed t-test with Welch correction.", "FDR Benjamini-Hochberg adjusted."],
    )
    matrix = classify_action_matrix([f])
    assert len(matrix["quick_wins"]) == 1
    assert len(matrix["watch_caveats"]) == 0


def test_classify_action_matrix_pilot_with_empty_caveats() -> None:
    from src.core.findings import Finding, classify_action_matrix

    f = Finding(
        finding_id="f_pilot_empty_caveats",
        kind="segment_lift",
        headline="Pilot group shows potential +30% lift",
        measure="lift",
        dimension="group",
        level="Pilot",
        effect=0.30,
        confidence=0.80,
        evidence_tier="pilot",
        caveats=[],
    )
    matrix = classify_action_matrix([f])
    assert len(matrix["pilot"]) == 1
    assert len(matrix["quick_wins"]) == 0


def test_classify_action_matrix_low_impact_driver_is_incremental() -> None:
    from src.core.findings import Finding, classify_action_matrix

    f_low = Finding(
        finding_id="f_driver_low",
        kind="driver",
        headline="Feature X drives 1.5% shift",
        measure="churn",
        dimension="feature_x",
        effect=0.015,
        confidence=0.85,
        evidence_tier="causal_action",
    )
    matrix = classify_action_matrix([f_low])
    assert len(matrix["incremental"]) == 1
    assert len(matrix["quick_wins"]) == 0


def test_swarm_dashboard_dynamic_agent_telemetry_and_kpis() -> None:
    from ui.components.swarm_dashboard import build_swarm_payload

    # With no findings and empty report: no hardcoded mock numbers
    p_empty = build_swarm_payload()
    assert p_empty["telemetry"]["records"] == "N/A"
    assert p_empty["kpis"][0]["target"] == 0.0
    assert p_empty["kpis"][1]["badge"] == "AUDITED"
    assert p_empty["kpis"][2]["target"] == 0
    assert p_empty["kpis"][2]["badge"] == "INVARIANT"

    # With paradox finding: KPI 3 must flag reversal
    p_paradox = build_swarm_payload(
        findings=[
            {
                "finding_id": "f_px",
                "headline": "Reversal finding",
                "caveats": ["Subgroup reversal detected (Simpson's paradox)."],
            }
        ]
    )
    assert p_paradox["kpis"][2]["target"] == 1
    assert p_paradox["kpis"][2]["badge"] == "CAUTION"


