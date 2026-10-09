"""Tests for honest run- and tool-level multiple-testing correction."""
from __future__ import annotations

from math import nan

import pytest

from src.core.multiple_testing import apply_benjamini_hochberg


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
