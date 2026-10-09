"""Evidence-contract tests for LLM-designed chart recipes."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.chart_designer import design_from_reply, run_recipe


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": ["a", "a", "b", "b", "c", "c"],
            "rating": [70.0, 72.0, 80.0, 82.0, 75.0, 77.0],
            "other": [1, 2, 3, 4, 5, 6],
        }
    )


def _finding(estimand: str | None = None) -> dict[str, Any]:
    evidence: dict[str, Any] = {"n": 6}
    if estimand is not None:
        evidence["estimand"] = estimand
    return {
        "finding_id": "rating_by_segment",
        "kind": "segment_lift",
        "measure": "rating",
        "dimension": "segment",
        "evidence": evidence,
    }


def _recipe(agg: str = "mean", **extra: Any) -> dict[str, Any]:
    return {
        "type": "bar",
        "x": "segment",
        "y": "rating",
        "agg": agg,
        "finding_id": "rating_by_segment",
        **extra,
    }


def test_linked_recipe_requires_matching_finding_measure_dimension_and_estimand() -> None:
    frame = _frame()
    finding = _finding()
    accepted = design_from_reply(frame, {"charts": [_recipe()]}, [finding])
    assert len(accepted.charts) == 1
    assert "Mean" in accepted.charts[0]["caption"]
    assert "n=6" in accepted.charts[0]["caption"]

    wrong_measure = _recipe()
    wrong_measure["y"] = "other"
    assert design_from_reply(frame, {"charts": [wrong_measure]}, [finding]).charts == []
    assert design_from_reply(frame, {"charts": [_recipe()]}, []).charts == []


def test_reply_metadata_is_accepted_without_changing_existing_callers() -> None:
    design = design_from_reply(
        _frame(), {"charts": [_recipe()], "finding_metadata": [_finding()]}
    )
    assert len(design.charts) == 1

    # Unlinked goal-oriented recipes remain supported for historical callers.
    assert run_recipe(_frame(), {"type": "bar", "x": "segment", "y": "rating", "agg": "mean"})


def test_extrema_cannot_stand_in_for_a_comparison_unless_explicitly_estimated() -> None:
    frame = _frame()
    assert run_recipe(frame, _recipe("max"), findings=[_finding()]) is None
    accepted = run_recipe(frame, _recipe("max"), findings=[_finding("max")])
    assert accepted is not None
    assert "Maximum" in accepted["caption"]
