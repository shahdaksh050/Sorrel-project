"""Join review: preview without merging, user overrides, bare-key refusal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.joins import join_related, preview_joins, suggest_join


def _orders_customers() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    orders = pd.DataFrame({
        "order_id": range(200),
        "customer_id": rng.integers(0, 40, 200),
        "amount": rng.normal(50, 10, 200),
    })
    customers = pd.DataFrame({
        "id": range(40),
        "region": rng.choice(["N", "S", "E", "W"], 40),
    })
    return orders, customers


def test_preview_does_not_merge_and_proposes_keys() -> None:
    orders, customers = _orders_customers()
    before = orders.copy()
    out = preview_joins(orders, [("customers", customers)], "orders")
    assert orders.equals(before)
    assert len(out) == 1
    row = out[0]
    assert row["name"] == "customers"
    assert (row["plan"]["left_key"], row["plan"]["right_key"]) == ("customer_id", "id")
    assert row["plan"]["cardinality"] == "many_to_one"
    assert row["plan"]["coverage"] == 1.0
    assert "region" in row["columns"] and "customer_id" in row["left_columns"]


def test_preview_reads_paths(tmp_path) -> None:
    orders, customers = _orders_customers()
    orders.to_csv(tmp_path / "orders.csv", index=False)
    customers.to_csv(tmp_path / "customers.csv", index=False)
    out = preview_joins(tmp_path / "orders.csv", [("customers", tmp_path / "customers.csv")])
    assert out[0]["plan"] is not None


def test_override_skip_leaves_table_out() -> None:
    orders, customers = _orders_customers()
    merged, notes = join_related(
        orders, [("customers", customers)], 10_000, "orders", overrides={"customers": {"skip": True}}
    )
    assert merged is orders
    assert any("chose not to join" in n for n in notes)


def test_override_keys_are_honoured() -> None:
    orders, _ = _orders_customers()
    tiers = pd.DataFrame({"cust": range(40), "tier": ["gold", "silver"] * 20})
    auto, _ = join_related(orders, [("tiers", tiers)], 10_000, "orders")
    assert auto is orders   # no shared name, so nothing is proposed automatically
    merged, notes = join_related(
        orders, [("tiers", tiers)], 10_000, "orders",
        overrides={"tiers": {"left_key": "customer_id", "right_key": "cust"}},
    )
    assert "tier" in merged.columns and len(merged) == len(orders)
    assert merged["tier"].notna().all()
    assert any("customer_id = cust" in n for n in notes)


def test_override_with_missing_column_is_skipped() -> None:
    orders, customers = _orders_customers()
    merged, notes = join_related(
        orders, [("customers", customers)], 10_000, "orders",
        overrides={"customers": {"left_key": "nope", "right_key": "id"}},
    )
    assert merged is orders
    assert any("Skipped customers" in n for n in notes)


def test_bare_id_tables_with_low_overlap_are_not_proposed() -> None:
    a = pd.DataFrame({"id": range(1000), "x": range(1000)})
    b = pd.DataFrame({"id": range(200, 1200), "y": range(1000)})   # 80% overlap only
    assert suggest_join(a, b, "a", "b") is None
    c = pd.DataFrame({"index": range(100), "x": range(100)})
    d = pd.DataFrame({"index": range(65, 165), "y": range(100)})   # 35% overlap
    assert suggest_join(c, d, "c", "d") is None
    assert preview_joins(a, [("b", b)])[0]["plan"] is None


def test_generic_key_needs_90_percent_containment() -> None:
    left = pd.DataFrame({"code": [f"c{i}" for i in range(100)], "x": range(100)})
    mid = pd.DataFrame({"code": [f"c{i}" for i in range(75)], "y": range(75)})       # 75%
    high = pd.DataFrame({"code": [f"c{i}" for i in range(95)], "y": range(95)})      # 95%
    assert suggest_join(left, mid, "left", "mid") is None
    plan = suggest_join(left, high, "left", "high")
    assert plan is not None and plan.left_key == "code"
