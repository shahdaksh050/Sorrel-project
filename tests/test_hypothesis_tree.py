"""
Unit tests for HypothesisTree and Counterfactual Reasoning (Phase 10).
"""
from __future__ import annotations

from src.core.hypothesis import HypothesisTree, generate_counterfactual_probes


def test_hypothesis_tree_lifecycle() -> None:
    tree = HypothesisTree(primary_objective="Evaluate churn risk drivers")
    assert len(tree.nodes) == 1
    root = tree.nodes["H1"]
    assert "churn" in root.statement.lower()
    assert root.status == "untested"

    # Add child hypothesis
    child = tree.add_hypothesis(
        statement="High monthly charges directly correlate with increased churn",
        parent_id="H1",
        status="untested",
        counterfactuals=["Does effect persist when controlling for contract length?"],
    )
    assert child.id == "H2"
    assert child.parent_id == "H1"
    assert len(tree.get_active_hypotheses()) == 2

    # Update child status with finding
    tree.update_status("H2", status="supported", confidence=0.85, finding_id="F1", rationale="p < 0.001, r = 0.45")
    assert child.status == "supported"
    assert child.confidence == 0.85
    assert "F1" in child.evidence_finding_ids
    assert len(tree.get_active_hypotheses()) == 1  # only H1 is untested now

    # Summary prompt formatting
    summary = tree.to_prompt_summary()
    assert "✓ Supported" in summary
    assert "Counterfactuals: 1" in summary


def test_generate_counterfactual_probes() -> None:
    probes = generate_counterfactual_probes(
        col1="pricing",
        col2="conversion",
        candidate_confounders=["device_type", "region", "pricing", "conversion", "marketing_channel"],
    )
    assert len(probes) == 3
    assert "pricing" not in probes[0].split("controlling for")[1]
    assert "controlling for 'device_type'" in probes[0]
    assert "controlling for 'region'" in probes[1]
