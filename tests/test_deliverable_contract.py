"""
Unit tests for Deliverable Contract (FutureScope Phase 2).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.deliverable_contract import (
    audit_deliverables,
    parse_deliverable_contract,
)


def test_parse_deliverable_contract_full() -> None:
    objective = (
        "Which pollutants move together, show a heatmap of monthly average per pollutant, "
        "and what drives CO levels?"
    )
    contract = parse_deliverable_contract(objective)

    assert "heatmap" in contract.required_charts
    assert any("co" in t.lower() for t in contract.required_driver_targets)
    assert contract.requires_relationship_analysis is True
    assert contract.requires_forecast is False


def test_parse_deliverable_contract_forecast_and_compare() -> None:
    objective = "Compare revenue between North and South, and forecast Q4 sales"
    contract = parse_deliverable_contract(objective)

    assert contract.requires_comparison is True
    assert contract.requires_forecast is True


def test_parse_deliverable_contract_ignores_generic_explain_phrasing() -> None:
    """Null case: a rhetorical "explain"/"what drives" with no real column
    reference must not manufacture a driver target that can never be
    satisfied (previously "Explain the findings from this analysis" ->
    required_driver_targets == ["the findings from this analysis"], which
    audit_deliverables would then report MISSING on every single run)."""
    for objective in (
        "Explain the findings from this analysis",
        "Please explain what happened here",
        "Explain the relationship",
    ):
        contract = parse_deliverable_contract(objective)
        assert contract.required_driver_targets == [], (objective, contract.required_driver_targets)
        assert contract.is_empty()


def test_parse_deliverable_contract_still_extracts_real_targets() -> None:
    """Planted case: real, specific targets in the same phrasings must still
    be extracted — the generic-phrase guard must not overcorrect."""
    assert parse_deliverable_contract("What drives customer churn?").required_driver_targets == ["customer churn"]
    assert parse_deliverable_contract("What are the drivers of monthly revenue?").required_driver_targets == ["monthly revenue"]
    contract = parse_deliverable_contract(
        "Show a heatmap of monthly average per pollutant and tell me what drives CO levels"
    )
    assert any("co" in t.lower() for t in contract.required_driver_targets)


def test_audit_deliverables_complete() -> None:
    contract = parse_deliverable_contract("Show a heatmap and what drives CO")
    final_result = {
        "charts": [
            {"title": "Monthly Heatmap", "chart_type": "heatmap", "spec": {"mark": "rect"}},
        ],
        "findings": [
            {"headline": "Traffic volume is the primary driver of CO levels (importance 0.45)", "kind": "driver"},
        ],
        "insights": ["CO levels are driven by traffic."],
    }
    report = audit_deliverables(contract, final_result)
    assert "Chart: heatmap" in report.delivered
    assert any("Driver analysis: CO" in d for d in report.delivered)
    assert report.is_complete is True


def test_audit_deliverables_repair_fallback() -> None:
    contract = parse_deliverable_contract("Show a heatmap of pollutants")
    # Missing heatmap in charts
    final_result: dict[str, Any] = {"charts": [], "findings": [], "insights": []}
    df = pd.DataFrame({
        "CO": [1.0, 2.0, 3.0, 4.0],
        "NOx": [2.0, 3.0, 4.0, 5.0],
        "O3": [0.5, 0.4, 0.3, 0.2],
    })
    report = audit_deliverables(contract, final_result, df=df)

    assert report.is_complete is True
    assert any("synthesized fallback" in r for r in report.repaired)
    # Confirm chart was added to final_result
    assert len(final_result["charts"]) == 1
    assert final_result["charts"][0]["chart_type"] == "heatmap"


def test_audit_deliverables_bar_repair_finds_pandas3_str_dtype_column() -> None:
    # pandas 3.0 gives plain text columns dtype `str`, not `object` — the
    # bar-chart fallback must still find them as the categorical axis.
    contract = parse_deliverable_contract("Show a bar chart of sales by region")
    final_result: dict[str, Any] = {"charts": [], "findings": [], "insights": []}
    df = pd.DataFrame({
        "region": pd.Series(["north", "south", "north", "south"], dtype="str"),
        "sales": [10.0, 20.0, 30.0, 40.0],
    })
    assert df["region"].dtype == "str"

    report = audit_deliverables(contract, final_result, df=df)

    assert any("synthesized fallback" in r for r in report.repaired)
    assert len(final_result["charts"]) == 1
    assert final_result["charts"][0]["chart_type"] == "bar"
