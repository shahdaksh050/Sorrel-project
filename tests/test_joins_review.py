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


def test_parent_as_base_joins_child_one_to_many() -> None:
    """When base is parent (customers) and related is child (orders), suggest_join must propose one-to-many."""
    orders, customers = _orders_customers()
    plan = suggest_join(customers, orders, "customers", "orders")
    assert plan is not None
    assert (plan.left_key, plan.right_key) == ("id", "customer_id")
    assert plan.cardinality == "one_to_many"
    merged, notes = join_related(customers, [("orders", orders)], 10_000, "customers")
    # Left join retains all customers (including any with 0 orders) plus all 200 orders
    assert merged["order_id"].notna().sum() == len(orders)
    assert len(merged) >= len(orders)
    assert any("one-to-many" in n for n in notes)


def test_nullable_float_foreign_key_aligns_with_integer() -> None:
    """CSVs with missing values parse integer IDs as float64 (e.g. 101.0, NaN)."""
    parent = pd.DataFrame({"id": range(10, 50), "category": ["A"] * 40})
    child = pd.DataFrame({
        "parent_id": [10.0, 11.0, 12.0, np.nan, 13.0, 14.0] * 10,
        "amount": range(60),
    })
    plan = suggest_join(child, parent, "child", "parent")
    assert plan is not None
    assert (plan.left_key, plan.right_key) == ("parent_id", "id")
    merged, _ = join_related(child, [("parent", parent)], 10_000, "child")
    assert len(merged) == len(child)
    assert "category" in merged.columns


def test_case_insensitive_key_containment() -> None:
    """Keys in different casing (e.g. 'US' vs 'us') should align."""
    states_a = pd.DataFrame({"country_code": ["US", "CA", "GB", "FR"], "val": [1, 2, 3, 4]})
    states_b = pd.DataFrame({"country_code": ["us", "ca", "gb", "fr"], "name": ["USA", "Canada", "UK", "France"]})
    plan = suggest_join(states_a, states_b, "states_a", "states_b")
    assert plan is not None
    assert (plan.left_key, plan.right_key) == ("country_code", "country_code")
    merged, _ = join_related(states_a, [("states_b", states_b)], 10_000, "states_a")
    assert len(merged) == len(states_a)
    assert "name" in merged.columns


def test_canonical_stem_synonym_matching() -> None:
    """Abbreviation synonyms (cust -> customer) should be recognized."""
    orders = pd.DataFrame({"cust_id": range(30), "amount": range(30)})
    customers = pd.DataFrame({"id": range(30), "name": [f"User_{i}" for i in range(30)]})
    plan = suggest_join(orders, customers, "orders", "customers")
    assert plan is not None
    assert (plan.left_key, plan.right_key) == ("cust_id", "id")


def test_cartesian_fanout_aggregated_safely() -> None:
    """Joining two 1:N child tables onto a parent must aggregate the second child to prevent Cartesian explosion."""
    games = pd.DataFrame({"game_id": [1, 2], "season": [2021, 2021]})
    appearances = pd.DataFrame({
        "game_id": [1, 1, 1, 2, 2],
        "player_id": [10, 11, 12, 10, 11],
        "goals": [1, 0, 2, 0, 1],
    })
    shots = pd.DataFrame({
        "game_id": [1, 1, 1, 1, 2, 2, 2],
        "minute": [10, 25, 45, 80, 15, 30, 75],
        "xGoal": [0.1, 0.4, 0.7, 0.2, 0.3, 0.5, 0.1],
    })
    # Joining both onto games:
    merged, notes = join_related(games, [("appearances", appearances), ("shots", shots)], 10_000, "games")
    # Base expanded to 5 appearances rows; shots was aggregated by game_id so it did not multiply to 5 * 4 = 20 rows!
    assert len(merged) == 5
    assert "shots_xGoal_sum" in merged.columns
    assert "shots_count" in merged.columns
    assert any("aggregated child table" in n for n in notes)


def test_large_one_to_many_child_table_aggregated_safely() -> None:
    """A massive child table on a 1:N join must roll up by key to preserve entity grain."""
    parent = pd.DataFrame({"game_id": list(range(500)), "home_goals": [1] * 500})
    child_rows = 6000
    child = pd.DataFrame({
        "game_id": [i % 500 for i in range(child_rows)],
        "player_id": list(range(child_rows)),
        "player_rating": [7.5] * child_rows,
    })
    merged, notes = join_related(parent, [("appearances", child)], 100_000, "games")
    assert len(merged) == 500
    assert "appearances_count" in merged.columns
    assert "appearances_player_rating_mean" in merged.columns
    assert any("aggregated child table to avoid Cartesian fan-out" in n for n in notes)


def test_transitive_join_column_ownership() -> None:
    """A 3rd table joining onto a column introduced by a 2nd table attributes ownership to the 2nd table."""
    orders = pd.DataFrame({"order_id": [1, 2, 3], "product_id": [10, 20, 30]})
    products = pd.DataFrame({"id": [10, 20, 30], "category_id": [100, 200, 300], "pname": ["A", "B", "C"]})
    categories = pd.DataFrame({"id": [100, 200, 300], "cat_name": ["Electronics", "Books", "Clothing"]})

    merged, notes = join_related(orders, [("products", products), ("categories", categories)], 10_000, "orders")
    assert len(merged) == 3
    assert "cat_name" in merged.columns
    # Check that categories was joined to products, not orders
    cat_note = next(n for n in notes if n.startswith("Joined categories"))
    assert "to products" in cat_note

