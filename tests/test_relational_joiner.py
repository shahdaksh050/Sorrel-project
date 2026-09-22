"""
Unit tests for Automated Relational Discovery and Star Schema Joiner (Phase 11).
"""
from __future__ import annotations

import pandas as pd

from src.core.relational_joiner import assemble_star_schema, discover_foreign_keys


def test_discover_foreign_keys_and_star_schema() -> None:
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
        "region": ["US", "EU", "US", "APAC"],
    })

    products = pd.DataFrame({
        "product_id": [101, 102, 103],
        "product_name": ["Laptop", "Mouse", "Keyboard"],
        "price": [1200.0, 25.0, 75.0],
    })

    orders = pd.DataFrame({
        "order_id": [1001, 1002, 1003, 1004, 1005],
        "customer_id": [1, 2, 1, 3, 2],
        "product_id": [101, 102, 103, 101, 102],
        "quantity": [1, 2, 1, 1, 3],
    })

    tables = {
        "customers": customers,
        "products": products,
        "orders": orders,
    }

    relations = discover_foreign_keys(tables)
    assert len(relations) >= 2

    # Verify orders is detected as child of customers and products
    rel_parents = {r.parent_table: r.child_table for r in relations}
    assert rel_parents.get("customers") == "orders"
    assert rel_parents.get("products") == "orders"

    # Assemble star schema
    star_df, notes = assemble_star_schema(tables, relations)
    assert len(star_df) == 5
    assert "name" in star_df.columns or "customers_name" in star_df.columns
    assert "product_name" in star_df.columns or "products_product_name" in star_df.columns
    assert "quantity" in star_df.columns
    assert any("Selected 'orders' as central fact table" in n for n in notes)
