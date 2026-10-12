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
        evidence={"checks": {"not_luck": True, "robust": True, "paradox_free": True}},
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
        "evidence": {"checks": {"not_luck": True, "robust": True, "paradox_free": True}},
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
            "evidence": {"checks": {"robust": True, "paradox_free": True}},
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
