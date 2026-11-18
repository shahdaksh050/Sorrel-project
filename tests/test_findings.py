"""Unit tests for Finding model null-hypothesis equivalence and invariance."""
from __future__ import annotations

from src.core.findings import Finding, attach_finding_checks, is_trivial
from src.tools.statistical_analysis import SelectStatisticalTestTool


def test_invariance_and_uniformity_not_trivial() -> None:
    """Findings with kind in ('invariance', 'uniformity') are not suppressed by is_trivial."""
    inv = Finding(
        finding_id="test:inv",
        kind="invariance",
        headline="No difference across groups",
        effect=0.005,
        effect_kind="cohens_d",
        p_value=0.45,
        p_adjusted=0.45,
    )
    uni = Finding(
        finding_id="test:uni",
        kind="uniformity",
        headline="Uniform distribution across groups",
        effect=0.001,
        effect_kind="cohens_d",
        p_value=0.85,
        p_adjusted=0.85,
    )
    assert not is_trivial(inv)
    assert not is_trivial(uni)


def test_invariance_confidence_and_normalized_effect() -> None:
    """Invariance findings receive appropriate non-zero confidence and normalized effect."""
    inv = Finding(
        finding_id="test:inv",
        kind="invariance",
        headline="Invariance finding",
        effect=0.01,
        effect_kind="cohens_d",
        p_value=0.25,
        p_adjusted=0.25,
    )
    assert inv.effective_confidence() >= 0.75
    assert inv.normalized_effect() >= 0.8  # near-zero effect means high invariance

    # Invariance finding with effect=0.0 and effect_kind=None returns 1.0 (not 0.0)
    zero_inv = Finding(finding_id="t", kind="invariance", headline="h", effect=0.0, effect_kind=None)
    assert zero_inv.normalized_effect() == 1.0


def test_invariance_passes_checks() -> None:
    """Invariance findings pass 'not_luck' and 'invariance_verified' checks when p >= 0.05."""
    inv = Finding(
        finding_id="test:inv",
        kind="invariance",
        headline="Invariance finding",
        effect=0.01,
        effect_kind="cohens_d",
        p_value=0.30,
        p_adjusted=0.30,
        evidence={"n": 100},
    )
    attach_finding_checks([inv])
    checks = inv.evidence["checks"]
    assert checks["not_luck"] is True
    assert checks["invariance_verified"] is True
    assert inv.primary_eligible is True


def test_statistical_test_elevates_non_significant_result_to_invariance() -> None:
    """SelectStatisticalTestTool elevates p >= 0.05 tests with adequate n into invariance findings."""
    tool = SelectStatisticalTestTool()
    output = {
        "feature_column": "metric",
        "group_column": "treatment",
        "test_name": "Mann-Whitney U",
        "effect_size": 0.02,
        "effect_size_metric": "rank_biserial",
        "p_value": 0.42,
        "n": 80,
        "interpretation": "No significant difference.",
        "summary": "Mann-Whitney U: stat=400, p=0.42",
    }
    findings = tool.findings(output, None, None)
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "invariance"
    assert "empirical invariance" in f.headline.lower()
    assert f.p_value == 0.42
