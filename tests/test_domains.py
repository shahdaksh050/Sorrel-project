"""Domain inference tests for generic dataset semantics."""
from __future__ import annotations

import pandas as pd

from src.core.domains import infer_domains
from src.core.profiler import profile_dataframe


def test_product_catalog_detected_from_unique_products_and_prices() -> None:
    frame = pd.DataFrame(
        {
            "model": [f"item-{index}" for index in range(100)],
            "price": [100.0 + index for index in range(100)],
            "rating": [60.0 + index % 30 for index in range(100)],
            "category": ["a", "b"] * 50,
        }
    )
    matches = infer_domains(frame, profile_dataframe(frame))
    product = next(match for match in matches if match.domain == "product_catalog")
    assert product.roles["product"] == "model"
    assert product.roles["price"] == "price"
    assert product.confidence >= 0.55


def test_repeated_order_lines_do_not_claim_product_catalog() -> None:
    frame = pd.DataFrame(
        {
            "product": [f"item-{index % 5}" for index in range(100)],
            "price": [10.0 + index % 5 for index in range(100)],
            "order_date": pd.date_range("2026-01-01", periods=100, freq="h"),
        }
    )
    matches = infer_domains(frame, profile_dataframe(frame))
    assert all(match.domain != "product_catalog" for match in matches)
