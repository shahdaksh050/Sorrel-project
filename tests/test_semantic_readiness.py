"""Tests for aggregate-only semantic conformance checks."""
from __future__ import annotations

import json

import pandas as pd

from src.core.semantic_readiness import assess_semantic_readiness


def test_detects_name_value_semantic_shift_without_returning_cells() -> None:
    """An OS-labelled column containing capabilities is flagged in aggregate."""
    frame = pd.DataFrame(
        {
            "operating_system": ["Android v14", "No FM Radio", "Android v13", "No NFC"],
            "price": ["₹14,999", "₹18,000", "₹20,000", "₹22,000"],
        }
    )

    result = assess_semantic_readiness(frame)
    by_column = {item["column"]: item for item in result["per_column"]}

    operating_system = by_column["operating_system"]
    assert "platform" in operating_system["expected_semantics"]
    assert operating_system["detected_semantic_coverage_pct"]["platform"] == 50.0
    assert operating_system["detected_semantic_coverage_pct"]["capability"] == 25.0
    assert operating_system["detected_semantic_coverage_pct"]["connectivity"] == 25.0
    assert "mixed_or_shifted_semantics" in operating_system["issues"]
    assert result["columns_with_issues"] >= 1
    assert "Android v14" not in json.dumps(result)
    assert "No FM Radio" not in json.dumps(result)


def test_reports_high_conformance_for_consistent_unit_columns() -> None:
    """Unit-bearing values agreeing with their names score as analysis-ready."""
    frame = pd.DataFrame(
        {
            "battery_capacity": ["5000 mAh", "4500mAh"],
            "charging_power": ["33 W", "20 watt"],
            "processor_speed": ["2.4 GHz", "1800 MHz"],
        }
    )

    result = assess_semantic_readiness(frame)

    assert result["readiness_score"] == 100.0
    assert result["columns_with_issues"] == 0


def test_unit_normalised_numeric_measure_is_not_marked_misaligned() -> None:
    """Cleaning may remove currency glyphs while retaining a valid numeric price."""
    result = assess_semantic_readiness(pd.DataFrame({"price": [14999.0, 22000.0]}))

    price = result["per_column"][0]
    assert price["conformance_pct"] == 100.0
    assert price["issues"] == []
