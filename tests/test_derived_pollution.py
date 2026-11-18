"""
Unit tests for synthetic & self-derived feature pollution prevention and
action matrix lever sanity across all archetypes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.findings import Finding, _is_actionable_feature, classify_action_matrix
from src.core.stats_utils import get_base_column_name, is_derived_pair
from src.tools.data_processing import CorrelationAnalysisTool


def test_is_derived_pair_detects_all_archetype_transforms() -> None:
    # Base vs derived transform (suffix)
    assert is_derived_pair("Price", "Price_log_return")
    assert is_derived_pair("price_per_unit", "price_per_unit_lag1")
    assert is_derived_pair("sales", "sales_delta1")
    assert is_derived_pair("returns", "returns_vol30")
    assert is_derived_pair("latitude", "latitude_centroid")
    assert is_derived_pair("customer_id", "customer_id_rfm")
    assert is_derived_pair("feature", "feature_norm")
    assert is_derived_pair("feature", "feature_scaled")
    assert is_derived_pair("review_text", "review_text_sentiment")
    assert is_derived_pair("text", "text_tfidf_word")

    # Base vs derived transform (prefix)
    assert is_derived_pair("Price", "log_price")
    assert is_derived_pair("sales", "diff_sales")
    assert is_derived_pair("revenue", "lag1_revenue")

    # Derived vs derived of the same base variable
    assert is_derived_pair("Price_log_return", "Price_lag1")
    assert is_derived_pair("sales_delta1", "sales_lag2")
    assert is_derived_pair("log_price", "price_lag1")

    # Distinct base variables (not self-derived)
    assert not is_derived_pair("Price", "Quantity")
    assert not is_derived_pair("Price_log_return", "Quantity_log_return")
    assert not is_derived_pair("latitude", "longitude")

    # Crucial regression test: one column name is a prefix of another followed by underscore,
    # but the suffix is NOT a mathematical transform
    assert not is_derived_pair("user", "user_id")
    assert not is_derived_pair("loan", "loan_amount")
    assert not is_derived_pair("order", "order_date")
    assert not is_derived_pair("store", "store_sales")
    assert not is_derived_pair("customer", "customer_age")
    assert not is_derived_pair("product", "product_category")
    assert not is_derived_pair("ad", "ad_spend")


def test_get_base_column_name() -> None:
    assert get_base_column_name("Price_log_return") == "Price"
    assert get_base_column_name("sales_delta2") == "sales"
    assert get_base_column_name("Total Amount") == "Total Amount"


def test_correlation_tool_filters_derived_pairs(tmp_path: Path) -> None:
    # Construct a dataset with a base column and its own derived log return and lag
    df = pd.DataFrame({
        "price": [10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 12.0, 11.9, 12.5, 13.0] * 5,
        "quantity": [100, 95, 90, 92, 88, 85, 80, 82, 78, 75] * 5,
    })
    df["price_lag1"] = df["price"].shift(1).bfill()
    df["price_log_return"] = (df["price"] / df["price_lag1"]).apply(lambda x: 0.0 if x <= 0 else float(x))

    csv_path = str(tmp_path / "test_data.csv")
    df.to_csv(csv_path, index=False)

    tool = CorrelationAnalysisTool()
    result = tool.run(file_path=csv_path)
    assert result.status == "success"
    findings = tool.findings(result.output, None, None)

    # Ensure no finding links price to price_lag1 or price_log_return
    for f in findings:
        col_a = f.evidence.get("col_a")
        col_b = f.evidence.get("col_b")
        if col_a and col_b:
            assert not is_derived_pair(str(col_a), str(col_b)), f"Self-derived correlation finding leaked: {f.headline}"


def test_is_actionable_feature_rejects_outcomes_and_transforms() -> None:
    # Controllable levers: should be actionable
    assert _is_actionable_feature("Product Category")
    assert _is_actionable_feature("marketing_channel")
    assert _is_actionable_feature("discount_tier")

    # Outcome KPIs: should NOT be actionable levers
    assert not _is_actionable_feature("Total Amount")
    assert not _is_actionable_feature("revenue")
    assert not _is_actionable_feature("sales_revenue")
    assert not _is_actionable_feature("profit_margin")
    assert not _is_actionable_feature("churn_rate")

    # Derived transforms: should NOT be actionable levers
    assert not _is_actionable_feature("Price_log_return")
    assert not _is_actionable_feature("sales_lag1")


def test_classify_action_matrix_excludes_correlations_and_uncontrollable_levers() -> None:
    findings = [
        # Correlation finding on self-derived feature (must be dropped completely)
        Finding(
            finding_id="f1",
            source_tool="correlation_analysis",
            headline="Price and Price_log_return move together",
            dimension="Price_log_return",
            measure="Price",
            kind="correlation",
            effect=0.95,
            p_value=0.001,
            evidence={"col_a": "Price", "col_b": "Price_log_return"},
        ),
        # Segment comparison on an outcome variable (Total Amount should NOT be an action lever)
        Finding(
            finding_id="f2",
            source_tool="segment_comparison",
            headline="Total Amount differs by Region",
            dimension="Total Amount",
            measure="Region",
            kind="segment_lift",
            effect=0.45,
            p_value=0.01,
            evidence={"dimension": "Total Amount", "measure": "Region", "segment_n": 500},
        ),
        # Valid actionable lever on controllable category
        Finding(
            finding_id="f3",
            source_tool="segment_comparison",
            headline="Conversion differs by Category",
            dimension="Category",
            measure="Conversion",
            kind="segment_lift",
            effect=0.35,
            p_value=0.005,
            evidence_tier="causal_action",
            evidence={"dimension": "Category", "measure": "Conversion", "segment_n": 500},
        ),
    ]

    matrix = classify_action_matrix(findings)
    action_items = matrix["quick_wins"] + matrix["strategic_bets"] + matrix["incremental"] + matrix["pilot"]

    # In action quadrants: self-derived and uncontrollable outcome metrics must NEVER appear
    for item in action_items:
        assert "log_return" not in item["headline"].lower()
        assert "co-movement" not in item["headline"].lower()
        assert item["headline"] != "Total Amount differs by Region"
        assert item["lever"] != "Target Total Amount"

    # Only Category should be classified as an action quadrant lever
    assert any("Category" in item["lever"] for item in action_items)

    # Self-derived correlation finding (f1) is completely dropped from the matrix
    all_items = action_items + matrix["watch_caveats"]
    assert not any(item["finding_id"] == "f1" for item in all_items)


def test_driver_tornado_chart_filters_derived_features_and_outcomes() -> None:
    from src.core.dashboard_charts_basic import _driver_tornado_chart

    finding = {"measure": "Price", "finding_id": "f_driver"}
    results = [
        {
            "tool_name": "regression_analysis",
            "status": "success",
            "output": {
                "coefficients": {
                    "Price_log_return": 0.95,  # Self-derived transform of target
                    "log_price": 0.88,         # Prefix transform of target
                    "Total Amount": 0.75,      # Outcome KPI
                    "Product Category": 0.45,  # Controllable lever
                    "marketing_channel": 0.30, # Controllable lever
                }
            },
        }
    ]

    spec = _driver_tornado_chart(finding, results)
    assert spec is not None
    data_values = spec.spec["data"]["values"]
    features_shown = [d["feature"] for d in data_values]

    # Neither self-derived transforms nor outcome KPIs should be listed as top levers
    assert "Price_log_return" not in features_shown
    assert "log_price" not in features_shown
    assert "Total Amount" not in features_shown

    # Controllable levers should be present
    assert "Product Category" in features_shown
    assert "marketing_channel" in features_shown
