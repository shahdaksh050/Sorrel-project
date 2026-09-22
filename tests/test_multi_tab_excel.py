"""
Unit tests for Multi-Tab Excel Ingestion and Schema Discovery (Phase 11).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.io import read_any


def test_multi_tab_excel_matching_schemas(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "sales_quarters.xlsx"
    q1 = pd.DataFrame({"region": ["North", "South"], "revenue": [100, 200]})
    q2 = pd.DataFrame({"region": ["East", "West"], "revenue": [150, 250]})

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        q1.to_excel(writer, sheet_name="Q1", index=False)
        q2.to_excel(writer, sheet_name="Q2", index=False)

    df, report = read_any(str(xlsx_path))

    assert len(df) == 4
    assert "_sheet_name" in df.columns
    assert set(df["_sheet_name"]) == {"Q1", "Q2"}
    assert any("Auto-concatenated 2 sheets" in note for note in report.notes)


def test_multi_tab_excel_distinct_schemas(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "ecommerce.xlsx"
    orders = pd.DataFrame({"order_id": [1, 2, 3], "customer_id": [10, 20, 10], "amount": [50.0, 99.0, 25.0]})
    customers = pd.DataFrame({"customer_id": [10, 20], "name": ["Alice", "Bob"], "tier": ["Gold", "Silver"]})

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        orders.to_excel(writer, sheet_name="Orders", index=False)
        customers.to_excel(writer, sheet_name="Customers", index=False)

    df, report = read_any(str(xlsx_path))

    # Primary should be Orders (3 rows)
    assert len(df) == 3
    assert "order_id" in df.columns
    assert "Customers" in report.extra_tables
    assert len(report.extra_tables["Customers"]) == 2
    assert any("Detected multi-tab Excel with 2 distinct sheets" in note for note in report.notes)
