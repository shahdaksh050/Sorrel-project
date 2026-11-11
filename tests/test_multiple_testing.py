"""Tests for honest run- and tool-level multiple-testing correction."""
from __future__ import annotations

from math import nan

import pytest

from src.core.findings import Finding
from src.core.multiple_testing import adjust_findings_run_level, apply_benjamini_hochberg


def test_non_finite_p_values_do_not_contaminate_valid_bh_family() -> None:
    corrected = apply_benjamini_hochberg(
        [
            {"name": "strong", "p_value": 0.01},
            {"name": "undefined", "p_value": nan},
            {"name": "weak", "p_value": 0.2},
            {"name": "missing", "p_value": None},
        ]
    )

    by_name = {row["name"]: row for row in corrected}
    assert by_name["strong"]["p_adjusted"] == pytest.approx(0.02)
    assert by_name["strong"]["significant_after_correction"] is True
    assert by_name["weak"]["p_adjusted"] == pytest.approx(0.2)
    assert by_name["undefined"]["p_adjusted"] is None
    assert by_name["missing"]["p_adjusted"] is None
    assert by_name["undefined"]["p_value"] is None
    assert by_name["undefined"]["significant_after_correction"] is False


def test_all_invalid_family_is_retained_as_non_testable() -> None:
    corrected = apply_benjamini_hochberg(
        [{"p_value": -0.1}, {"p_value": 1.1}, {"p_value": "not-a-number"}]
    )

    assert len(corrected) == 3
    assert all(row["p_adjusted"] is None for row in corrected)
    assert not any(row["significant_after_correction"] for row in corrected)


def test_adjust_findings_partitions_exploratory_and_confirmatory() -> None:
    f_exploratory = Finding(
        finding_id="corr_0",
        kind="correlation",
        headline="x and y move together",
        p_value=0.01,
        p_adjusted=0.04,  # already corrected locally
        evidence={"test_family": "exploratory_screening", "p_adjusted_within_tool": 0.04},
    )
    f_confirmatory_1 = Finding(
        finding_id="test_1",
        kind="test",
        headline="group A differs from B",
        p_value=0.01,
        evidence={"test_family": "confirmatory_objective"},
    )
    f_confirmatory_2 = Finding(
        finding_id="test_2",
        kind="test",
        headline="group C differs from D",
        p_value=0.03,
        evidence={"test_family": "confirmatory_objective"},
    )

    findings = [f_exploratory, f_confirmatory_1, f_confirmatory_2]
    family_size = adjust_findings_run_level(findings)

    # Only the 2 confirmatory findings should be in the run-wide family
    assert family_size == 2
    # Exploratory finding retains its within-tool p_adjusted and is not diluted by run-level tests
    assert f_exploratory.p_adjusted == 0.04
    # Confirmatory findings get adjusted together
    assert f_confirmatory_1.p_adjusted is not None
    assert f_confirmatory_2.p_adjusted is not None
