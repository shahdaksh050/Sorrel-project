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
    assert "4.85 vs 1.78" in hl
    assert "difference +2.15" not in hl
    assert "n=68" not in hl

    takeaway = business_takeaway(finding_delivery)
    assert "68 in the segment and 121 in the comparison group" in takeaway

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
        evidence_tier="causal_action",
        evidence={"feasibility_supported": True, "n": 240, "checks": {"robust": True, "paradox_free": True}},
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
        evidence_tier="causal_action",
        evidence={"feasibility_supported": True, "n": 180, "checks": {"robust": True, "paradox_free": True}},
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
        evidence_tier="causal_action",
        evidence={"feasibility_supported": True, "n": 500, "checks": {"robust": True, "paradox_free": True}},
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
    assert "68 in the segment and 121 in the comparison group" in html_out
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
    assert "4.85 vs 1.78" in hl


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
        evidence={"feasibility_supported": True, "checks": {"robust": True, "paradox_free": True}},
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
        evidence={"feasibility_supported": True},
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
    assert p_empty["kpis"] == []
    assert p_empty["audit"]["ledger_text"] == "No finding checks recorded"

    # With paradox finding: KPI 3 must flag reversal
    p_paradox = build_swarm_payload(
        findings=[
            {
                "finding_id": "f_px",
                "headline": "Reversal finding",
                "evidence": {"checks": {"paradox_free": False}},
                "caveats": ["Subgroup reversal detected (Simpson's paradox)."],
            }
        ]
    )
    reversal = next(k for k in p_paradox["kpis"] if k["label"] == "Subgroup reversals")
    assert reversal["target"] == 1 and reversal["badge"] == "CAUTION"


def test_forecast_chart_preserves_data_points_and_layers() -> None:
    from src.core.dashboard_charts_basic import _forecast_chart
    raw_fc = {
        "value_column": "price",
        "period_grain": "day",
        "model": "linear_trend",
        "mase": 0.45,
        "chart": {
            "type": "vega_lite",
            "title": "Daily Price per Unit: history and forecast (95% range shaded)",
            "data": [
                {"period": "2023-01-01", "value": 10.0, "part": "History", "lower": 10.0, "upper": 10.0},
                {"period": "2023-01-02", "value": 12.0, "part": "Forecast", "lower": 11.0, "upper": 13.0},
            ],
            "vega_lite": {
                "height": 260,
                "layer": [{"mark": "line"}, {"mark": "area"}],
            },
        },
    }
    chart = _forecast_chart(raw_fc)
    assert chart is not None
    assert "data" in chart.spec
    assert len(chart.spec["data"]["values"]) == 2
    assert chart.spec["data"]["values"][0]["period"] == "2023-01-01"


def test_correlation_heatmap_square_cells_and_borders() -> None:
    import numpy as np
    import pandas as pd

    from src.core.dashboard_charts_basic import correlation_heatmap

    corr_df = pd.DataFrame(
        np.eye(4),
        columns=["Age", "Quantity", "Total Amount", "Price Per Unit"],
        index=["Age", "Quantity", "Total Amount", "Price Per Unit"],
    )
    res = correlation_heatmap(corr_df)
    assert res is not None
    _rows, spec, _cap = res
    assert spec["layer"][0]["mark"]["stroke"] == "#dcd7cc"
    assert spec["width"]["step"] <= 30
    assert spec["width"]["step"] == spec["height"]["step"]
    assert spec["autosize"]["type"] == "pad"


def test_swarm_dashboard_telemetry_hydration_and_lead_narrative() -> None:
    from dataclasses import dataclass

    from ui.components.swarm_dashboard import build_swarm_payload

    @dataclass
    class MockMeta:
        row_count: int = 260640
        column_count: int = 10
        file_path: str = "household_power.csv"

    mock_f = [
        {
            "finding_id": "f_anomaly",
            "kind": "outlier_scan",
            "measure": "Total Amount",
            "headline": "Total Amount spiked in 2023-05-23: $8,455 vs about $1,060 expected (+697.6%).",
            "detail": "7.4 robust SDs from the rolling_median baseline (total per day).",
            "evidence": {"z": 7.4},
            "state": "held_up",
        }
    ]
    p = build_swarm_payload(
        meta=MockMeta(),
        findings=mock_f,
    )
    assert p["telemetry"]["records"] == "260,640"
    assert p["telemetry"]["dimensions"] == "10 Cols"
    # The lead narrative must NOT leak raw z-score equations
    assert "robust sds" not in p["metadata"]["lead_narrative"].lower()
    deviation = next(k for k in p["kpis"] if k["label"] == "Statistical Deviation")
    assert deviation["target"] == 7.4
    assert deviation["prefix"] == "z = "


def test_entity_contrasts_preserve_evidence_disposition_for_either_sign() -> None:
    from src.core.findings import Finding, classify_action_matrix

    # Category='Delivery': +196.9% lift, n=68 states vs 121
    f_delivery = Finding(
        finding_id="f_del",
        kind="segment_lift",
        headline="Delivery entities average Rating of 3.40 vs 1.25 for everyone else (difference +2.15, d=1.42, 68 vs 121).",
        dimension="Category",
        level="Delivery",
        measure="Rating",
        effect=1.969,
        confidence=0.85,
        caveats=[
            "Observation evidence — monitor or test; do not treat as a causal lever.",
            "Counts are per 'State' (repeated rows aggregated), not per row.",
        ],
        evidence={"n": 68, "n_rest": 121, "checks": {"robust": True, "paradox_free": True}},
    )

    # Category='Plumbing': -171.7% deficit, n=114 states vs 76
    f_plumbing = Finding(
        finding_id="f_plumb",
        kind="segment_lift",
        headline="Plumbing entities average Rating of 1.13 vs 3.05 for everyone else (difference -1.92, d=-1.20, 114 vs 76).",
        dimension="Category",
        level="Plumbing",
        measure="Rating",
        effect=-1.717,
        confidence=0.85,
        caveats=[
            "Observation evidence — monitor or test; do not treat as a causal lever.",
            "Counts are per 'State' (repeated rows aggregated), not per row.",
        ],
        evidence={"n": 114, "n_rest": 76, "checks": {"robust": True, "paradox_free": True}},
    )

    matrix = classify_action_matrix([f_delivery, f_plumbing])

    assert not matrix["strategic_bets"] and not matrix["quick_wins"]
    assert not matrix["pilot"]
    assert {record["finding_id"] for record in matrix["watch_caveats"]} == {"f_del", "f_plumb"}
    assert f_plumbing.desired_direction is None


def test_cluster_and_correlation_plain_headlines() -> None:
    from src.core.plain_language import business_headline

    # Cluster separation
    f_sep = {
        "kind": "cluster",
        "headline": "Data splits into 2 segments with reasonable separation (silhouette=0.56)",
    }
    hl_sep = business_headline(f_sep)
    assert hl_sep == "Population Partitions into 2 Distinct Behavioral Cohorts"

    # Cluster profile
    f_prof = {
        "kind": "cluster",
        "headline": "Largest segment (cluster_0, 65 'State' entities, 51.2% of data) stands out on 'NumberReview' (mean 0.42 vs 16.7 overall)",
    }
    hl_prof = business_headline(f_prof)
    assert "Dominant Segment (51.2% of Population)" in hl_prof
    assert "Numberreview" in hl_prof

    # Correlation
    f_corr = {
        "kind": "correlation",
        "measure": "Rating",
        "dimension": "NumberReview",
        "effect": 0.55,
        "headline": "Rating and NumberReview move together (a strong positive relationship)",
    }
    hl_corr = business_headline(f_corr)
    assert "Rating and Numberreview Move in Positive Alignment" in hl_corr


def test_correlation_takeaway_resolves_col_b_without_none() -> None:
    from src.core.plain_language import business_headline, business_takeaway

    # Measure-to-measure correlation finding where finding.dimension is None
    finding = {
        "kind": "correlation",
        "measure": "Global_active_power",
        "dimension": None,
        "effect": 0.61,
        "headline": "Global_active_power and Sub_metering_3 move together.",
        "evidence": {
            "col_a": "Global_active_power",
            "col_b": "Sub_metering_3",
            "r": 0.61,
        },
    }
    takeaway = business_takeaway(finding)
    assert "'None'" not in takeaway
    assert "Global_active_power" in takeaway
    assert "Sub_metering_3" in takeaway
    assert "does not establish a causal effect" in takeaway

    hl = business_headline(finding)
    assert "Global Active Power and Sub Metering 3 Move in Positive Alignment" in hl


def test_anomaly_segment_plain_headline_formats_cleanly() -> None:
    from src.core.plain_language import business_headline

    finding = {
        "kind": "outlier_scan",
        "dimension": "day_of_week",
        "level": "Sunday",
        "measure": "Sub_metering_1",
        "effect": 1.38,
        "headline": "day_of_week Sunday averages 2.48 Sub_metering_1, well above its peers' typical 1.04 (+138%).",
        "evidence": {
            "dimension": "day_of_week",
            "level": "Sunday",
            "deviation_pct": 1.38,
        },
    }
    hl = business_headline(finding)
    assert "Sunday" in hl and "+138%" in hl
    assert "2.48" in hl and "1.04" in hl


def test_correlation_heatmap_adaptive_step_sizing() -> None:
    import pandas as pd

    from src.core.dashboard_charts_basic import correlation_heatmap

    # 7-column dataframe
    data = {f"col_{i}": [float(j) for j in range(20)] for i in range(7)}
    df = pd.DataFrame(data)
    corr_df = df.corr()

    res = correlation_heatmap(corr_df)
    assert res is not None
    _rows, spec, _caption = res
    assert "width" in spec
    step = spec["width"]["step"]
    # For n=7, step should be 28px, not 36px, fitting within the card container
    assert step <= 30
    assert step >= 18


def test_swarm_dashboard_run_id_clamping() -> None:
    from ui.components.swarm_dashboard import build_swarm_payload

    payload = build_swarm_payload(
        report={"run_id": "dc77abded9264b4b8d65c2f9fc3f36b5"},
        findings=[{"kind": "segment_lift", "headline": "Test finding", "effect": 0.2}],
        run_id="dc77abded9264b4b8d65c2f9fc3f36b5",
    )
    assert payload["metadata"]["run_id"] == "dc77abde"
    assert len(payload["metadata"]["run_id"]) == 8


def test_action_matrix_excludes_cluster_synthetic_features() -> None:
    from src.core.findings import Finding, _is_actionable_feature, classify_action_matrix

    # _is_actionable_feature must reject cluster/segment
    assert not _is_actionable_feature("cluster")
    assert not _is_actionable_feature("cluster_0")
    assert not _is_actionable_feature("segment")
    assert not _is_actionable_feature("segment_1")

    # Finding with cluster dimension
    f_cluster = Finding(
        finding_id="f_clust",
        kind="cluster",
        headline="Largest segment (cluster_0, 65 entities) stands out",
        dimension="cluster",
        level="cluster_0",
        measure="NumberReview",
        effect=0.975,
        confidence=0.75,
    )

    matrix = classify_action_matrix([f_cluster])
    # Must not be placed in quick_wins, strategic_bets, pilot, or incremental
    assert len(matrix["quick_wins"]) == 0
    assert len(matrix["strategic_bets"]) == 0
    assert len(matrix["pilot"]) == 0
    assert len(matrix["incremental"]) == 0


def test_swarm_dashboard_pvalue_formatting_avoids_zero() -> None:
    from ui.components.swarm_dashboard import build_swarm_payload

    # When p-value is extremely small (< 0.001), target should be '< 0.001' not '0'
    payload = build_swarm_payload(
        findings=[
            {
                "finding_id": "f_sig",
                "kind": "segment_lift",
                "headline": "Significant segment lift observed",
                "p_adjusted": 0.000002,
                "effect": 0.45,
            }
        ]
    )
    p_kpi = next(k for k in payload["kpis"] if "p-value" in k["label"].lower())
    assert p_kpi["target"] == "< 0.001"
    assert p_kpi["badge"] == "p < 0.001"
    assert "FDR" in p_kpi["subtext"]


@patch("streamlit.markdown")
@patch("streamlit.columns")
@patch("streamlit.container")
@patch("streamlit.caption")
def test_render_charts_tab_unified_view(
    mock_caption: MagicMock,
    mock_container: MagicMock,
    mock_columns: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    from ui.tabs.charts_tab import render_charts_tab

    mock_columns.return_value = [MagicMock() for _ in range(5)]
    mock_container.return_value.__enter__.return_value = MagicMock()

    mock_dashboard = [
        {"chart_id": "c_lead", "title": "Lead Wide Chart", "size": "wide", "spec": {"mark": "line"}},
        {"chart_id": "c_half1", "title": "Paired Chart 1", "size": "half", "spec": {"mark": "bar"}},
        {"chart_id": "c_half2", "title": "Paired Chart 2", "size": "half", "spec": {"mark": "bar"}},
    ]
    mock_report = {
        "title": "Autonomous Executive Briefing",
        "lead_narrative": "Air quality measures show significant cluster variance.",
        "row_count": 50000,
        "column_count": 12,
        "quality_score": "95.0 / 100",
    }
    mock_findings = [
        {
            "finding_id": "f1",
            "headline": "Lead finding holds",
            "effect": 0.35,
            "p_adjusted": 0.0005,
            "evidence": {"checks": {"robust": True, "paradox_free": True}},
        }
    ]

    render_charts_tab(
        dashboard=mock_dashboard,
        report=mock_report,
        vega_cfg={"view": {"stroke": "transparent"}},
        findings=mock_findings,
    )

    assert mock_caption.called
    assert mock_markdown.called


@patch("streamlit.info")
def test_render_charts_tab_empty_dashboard(mock_info: MagicMock) -> None:
    from ui.tabs.charts_tab import render_charts_tab

    render_charts_tab(
        dashboard=[],
        report={"coverage": {"total": 5, "answered": 0}},
        vega_cfg={},
    )
    assert mock_info.called


def test_get_archetype_action_config_all_archetypes() -> None:
    from ui.tabs.actions_tab import get_archetype_action_config

    cfg_panel = get_archetype_action_config("panel")
    assert cfg_panel["archetype_cluster"] == "longitudinal_panel"
    assert "Cohort" in cfg_panel["quadrants"][0][0][2]
    assert "COHORT GUARDRAILS" == cfg_panel["guardrails"]["badge"]

    cfg_sensor = get_archetype_action_config("continuous_series")
    assert cfg_sensor["archetype_cluster"] == "sensor_iot"
    assert "Control" in cfg_sensor["quadrants"][0][0][1]
    assert "SYSTEM GUARDRAILS" == cfg_sensor["guardrails"]["badge"]

    cfg_survey = get_archetype_action_config("survey")
    assert cfg_survey["archetype_cluster"] == "scientific_clinical"
    assert "Confirmatory" in cfg_survey["quadrants"][1][0][1]

    cfg_comm = get_archetype_action_config("transactional")
    assert cfg_comm["archetype_cluster"] == "commercial"
    assert "Quick Wins" in cfg_comm["quadrants"][0][0][1]

    cfg_gen = get_archetype_action_config("unknown")
    assert cfg_gen["archetype_cluster"] == "general"


def test_format_archetype_lever_and_rationale() -> None:
    from ui.tabs.actions_tab import format_archetype_lever_and_rationale

    # In panel / environmental data:
    lever_p, rat_p = format_archetype_lever_and_rationale(
        "Target AQI_Bucket='Very Poor'",
        "Investigate the observed difference before changing policy.",
        "longitudinal_panel",
    )
    assert "Priority Cohort: AQI_Bucket='Very Poor'" in lever_p
    assert "before changing policy" not in rat_p
    assert "before enacting broad cohort interventions" in rat_p

    # In sensor data:
    lever_s, rat_s = format_archetype_lever_and_rationale(
        "Target Sub_metering_1='High'",
        "Investigate before changing policy.",
        "sensor_iot",
    )
    assert "Control Regime: Sub_metering_1='High'" in lever_s
    assert "before adjusting operational thresholds" in rat_s


@patch("streamlit.markdown")
@patch("streamlit.columns")
@patch("streamlit.container")
def test_render_actions_tab_panel_archetype_awareness(
    mock_container: MagicMock,
    mock_columns: MagicMock,
    mock_markdown: MagicMock,
) -> None:
    from ui.tabs.actions_tab import render_actions_tab

    mock_columns.return_value = [MagicMock(), MagicMock()]
    mock_container.return_value.__enter__.return_value = MagicMock()

    mock_finding = {
        "finding_id": "f_aqi",
        "kind": "segment_lift",
        "headline": "Very Poor AQI leads PM2.5 with +154% higher performance",
        "dimension": "AQI_Bucket",
        "level": "Very Poor",
        "measure": "PM2.5",
        "effect": 1.54,
        "confidence": 0.85,
        "caveats": ["Counts are per 'City' (repeated rows aggregated), not per row."],
        "evidence": {"n": 22, "n_rest": 26, "checks": {"robust": True, "paradox_free": True}},
    }
    mock_report = {
        "understanding": {"archetype": "panel"},
        "findings": [mock_finding],
    }

    render_actions_tab(report=mock_report, tool_results=[], run_view=None)
    assert mock_markdown.called






