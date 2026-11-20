"""Unit tests for Action Prioritization Matrix (2x2) and its report rendering."""
from src.core.findings import Finding, classify_action_matrix
from src.core.html_report import _action_matrix_html
from src.tools.report_generator import _format_action_matrix


def test_classify_action_matrix_buckets() -> None:
    # 1. Quick win: high impact, high confidence, robust
    f_quick = Finding(
        finding_id="f1",
        kind="segment_lift",
        headline="Enterprise segment delivers +45% higher ARR",
        measure="arr",
        dimension="tier",
        level="Enterprise",
        effect=0.45,
        confidence=0.85,
        evidence={"feasibility_supported": True, "checks": {"not_luck": True, "robust": True, "paradox_free": True}},
        evidence_tier="causal_action",
    )

    # 2. Strategic bet: high impact interaction synergy
    f_bet = Finding(
        finding_id="f2",
        kind="interaction_synergy",
        headline="Compound synergy: Region='West' x Channel='Online' yields +32% lift",
        measure="sales",
        dimension="region_x_channel",
        effect=0.32,
        confidence=0.60,
        evidence={"feasibility_supported": True},
        evidence_tier="causal_action",
    )

    # 3. Incremental: moderate impact, high confidence
    f_inc = Finding(
        finding_id="f3",
        kind="segment_lift",
        headline="Mobile app users spend +4% more time",
        measure="time",
        dimension="device",
        level="Mobile",
        effect=0.04,
        confidence=0.80,
        evidence={"feasibility_supported": True},
        evidence_tier="causal_action",
    )

    # 4. Watch / Caveat item: paradox or sensitivity fragile
    f_watch = Finding(
        finding_id="f4",
        kind="segment_lift",
        headline="Discount tier appears to drive higher volume",
        measure="volume",
        dimension="discount",
        level="High",
        effect=0.25,
        confidence=0.75,
        evidence={"checks": {"paradox_free": False}},
    )

    matrix = classify_action_matrix([f_quick, f_bet, f_inc, f_watch])

    assert len(matrix["quick_wins"]) == 1
    assert matrix["quick_wins"][0]["finding_id"] == "f1"
    assert "Enterprise" in matrix["quick_wins"][0]["lever"]

    assert len(matrix["strategic_bets"]) == 1
    assert matrix["strategic_bets"][0]["finding_id"] == "f2"

    assert len(matrix["incremental"]) == 1
    assert matrix["incremental"][0]["finding_id"] == "f3"

    assert len(matrix["watch_caveats"]) == 1
    assert matrix["watch_caveats"][0]["finding_id"] == "f4"
    assert "Simpson's paradox" in matrix["watch_caveats"][0]["rationale"]


def test_classify_action_matrix_accepts_dicts() -> None:
    f_dict = {
        "finding_id": "f_dict_1",
        "kind": "decision_scenario",
        "headline": "What-If: +10% price shift yields +$120k margin",
        "measure": "margin",
        "dimension": "price",
        "effect": 0.22,
        "confidence": 0.88,
        "evidence": {"feasibility_supported": True, "checks": {"not_luck": True, "robust": True, "paradox_free": True}},
        "evidence_tier": "causal_action",
    }
    matrix = classify_action_matrix([f_dict])
    assert len(matrix["quick_wins"]) == 1
    assert matrix["quick_wins"][0]["finding_id"] == "f_dict_1"


def test_action_matrix_report_renderers() -> None:
    findings = [
        {
            "finding_id": "f1",
            "kind": "segment_lift",
            "headline": "Enterprise segment delivers +45% higher ARR",
            "measure": "arr",
            "dimension": "tier",
            "level": "Enterprise",
            "effect": 0.45,
            "confidence": 0.85,
            "evidence": {"feasibility_supported": True, "checks": {"robust": True, "paradox_free": True}},
            "evidence_tier": "causal_action",
        },
        {
            "finding_id": "f2",
            "kind": "segment_lift",
            "headline": "Fragile finding reverses in subgroup",
            "measure": "arr",
            "dimension": "tier",
            "effect": 0.15,
            "confidence": 0.70,
            "evidence": {"checks": {"paradox_free": False}},
        },
    ]

    # Markdown renderer
    md_lines = _format_action_matrix(findings)
    assert any("Action Prioritization Matrix (2x2)" in line for line in md_lines)
    assert any("⚡ Quick Wins" in line for line in md_lines)
    assert any("⚠️ Risk & Watch Items" in line for line in md_lines)

    # HTML renderer
    html = _action_matrix_html(findings)
    assert "Action Prioritization Matrix (2x2)" in html
    assert "⚡ Quick Wins" in html
    assert "⚠️ Guardrails & Watch" in html


def test_classify_action_matrix_filters_hygiene_alarms() -> None:
    f_flatline = Finding(
        finding_id="f_flat",
        kind="flatline",
        headline="Sensor flatline in 'release_year': identical value 2018 repeated for 19 consecutive periods.",
        source_tool="integrity_audit",
        effect=0.0,
        confidence=0.5,
    )
    f_constraint = Finding(
        finding_id="f_cons",
        kind="outlier_scan",
        headline="Constraint violation: 'Transaction ID >= Quantity' violated by 1 rows",
        source_tool="integrity_audit",
        effect=0.001,
        confidence=0.5,
    )
    f_shape = Finding(
        finding_id="f_shape",
        kind="metadata",
        headline="Dataset contains 1020 rows and 24 columns.",
        source_tool="execute_dynamic_code",
        effect=None,
        confidence=0.5,
    )

    matrix = classify_action_matrix([f_flatline, f_constraint, f_shape])
    assert len(matrix["quick_wins"]) == 0
    assert len(matrix["strategic_bets"]) == 0
    assert len(matrix["incremental"]) == 0
    assert len(matrix["watch_caveats"]) == 0


def test_observational_disclaimer_disqualifies_action_quadrants() -> None:
    # Statistical confidence is not causal/action evidence.
    f_driver = Finding(
        finding_id="f_driver",
        kind="driver",
        headline="Holding the other factors fixed, contract = One year rather than Month-to-month lowers churn by 20.0 percentage points",
        measure="churn",
        dimension="contract",
        level="One year vs Month-to-month",
        effect=-0.86,
        confidence=0.90,
        caveats=[
            "Effects are adjusted associations between the columns, not proof of cause.",
            "Imputed (median/mode) columns: total_charges.",
        ],
    )
    matrix = classify_action_matrix([f_driver])
    assert len(matrix["quick_wins"]) == 0
    item = matrix["watch_caveats"][0]
    assert item["finding_id"] == "f_driver"
    assert item["disposition"] == "investigate"
    assert "contract='One year vs Month-to-month'" in item["lever"]
    assert not item["lever"].startswith("Target")


def test_observational_segment_contrast_enters_pilot_quadrant() -> None:
    f_seg = Finding(
        finding_id="f_seg_contrast",
        kind="segment_lift",
        headline="Premium tier users show +35% higher engagement",
        measure="engagement",
        dimension="tier",
        level="Premium",
        effect=0.35,
        confidence=0.80,
        caveats=[
            "Observational cohort difference; unadjusted for confounding or selection effects."
        ],
        evidence={"checks": {"paradox_free": True, "robust": True}, "n": 250},
    )
    matrix = classify_action_matrix([f_seg])
    assert len(matrix["pilot"]) == 1
    assert len(matrix["a_b_test"]) == 1
    item = matrix["pilot"][0]
    assert item["finding_id"] == "f_seg_contrast"
    assert "Pilot" in item["lever"]
    assert "prospective A/B test or pilot" in item["rationale"]


def test_observational_segment_contrast_small_sample_rejected_from_pilot() -> None:
    f_small = Finding(
        finding_id="f_seg_small",
        kind="segment_lift",
        headline="Pilot group shows +40% engagement",
        measure="engagement",
        dimension="tier",
        level="Pilot",
        effect=0.40,
        confidence=0.80,
        caveats=[
            "Observational cohort difference; unadjusted for confounding or selection effects."
        ],
        evidence={"checks": {"paradox_free": True, "robust": True}, "n": 45},
    )
    matrix = classify_action_matrix([f_small])
    assert len(matrix["pilot"]) == 0
    assert len(matrix["watch_caveats"]) == 1
    assert "Small sample size" in matrix["watch_caveats"][0]["rationale"]


def test_household_power_sensor_telemetry_guardrail() -> None:
    """Household power telemetry: sub-metering columns are passive sensor telemetry
    and additive components of active power, and MUST NEVER receive an 'act' or 'pilot'
    recommendation, or recommend raising consumption or changing meter readings."""
    f_power = Finding(
        finding_id="f_power_submeter",
        kind="driver",
        headline="Global_active_power increases when Sub_metering_3 is higher",
        measure="Global_active_power",
        dimension="Sub_metering_3",
        effect=0.60,
        confidence=0.95,
        evidence={"n": 200000},
    )
    disposition = f_power.recommendation_disposition()
    assert disposition == "monitor"
    assert f_power.controllability == "sensor_telemetry"

    matrix = classify_action_matrix([f_power])
    # Must NOT be in quick_wins, strategic_bets, or pilot
    assert len(matrix["quick_wins"]) == 0
    assert len(matrix["strategic_bets"]) == 0
    assert len(matrix["pilot"]) == 0
    assert len(matrix["watch_caveats"]) == 1
    caveat_item = matrix["watch_caveats"][0]
    assert "Sensor telemetry" in caveat_item["rationale"]
    assert "manipulate meter readings" in caveat_item["rationale"]


def test_positive_control_intervention_action_recommendation() -> None:
    """Positive control: well-supported intervention with causal evidence and
    explicit feasibility receives an action ('act') recommendation in quick_wins."""
    f_positive_control = Finding(
        finding_id="f_pos_control",
        kind="segment_lift",
        headline="Automated checkout reduces cart abandonment by 18% (RCT p<0.001)",
        measure="abandonment_rate",
        dimension="checkout_flow",
        level="OneClick",
        effect=-0.18,
        confidence=0.92,
        evidence_tier="causal_action",
        evidence={
            "feasibility_supported": True,
            "experiment_design": "randomized_controlled_trial",
            "n": 5000,
            "checks": {"robust": True, "paradox_free": True},
        },
    )
    disposition = f_positive_control.recommendation_disposition()
    assert disposition == "act"
    assert f_positive_control.controllability == "manipulable"

    matrix = classify_action_matrix([f_positive_control])
    assert len(matrix["quick_wins"]) == 1
    assert matrix["quick_wins"][0]["finding_id"] == "f_pos_control"
    assert matrix["quick_wins"][0]["feasibility_level"] == "Immediate Execution"


def test_actionable_descriptive_advice_duplicate_billing() -> None:
    """Actionable descriptive advice such as investigating duplicate billing clusters
    is preserved as 'investigate' rather than ignored or turned into direct action."""
    f_billing = Finding(
        finding_id="f_dup_billing",
        kind="cluster",
        headline="Duplicate billing cluster identified: 142 invoice pairs with identical timestamp and amount",
        detail="Possible double-charging anomaly in payment gateway reconciliation.",
        measure="amount",
        dimension="cluster_billing_anomaly",
        effect=0.30,
        confidence=0.85,
        evidence={"cluster_id": 2, "n": 142},
    )
    disposition = f_billing.recommendation_disposition()
    assert disposition == "investigate"
    assert f_billing.controllability == "descriptive"

    matrix = classify_action_matrix([f_billing])
    assert len(matrix["watch_caveats"]) == 1
    item = matrix["watch_caveats"][0]
    assert "Investigate" in item["lever"] or "billing" in item["lever"]
    assert "duplicate records" in item["rationale"]
