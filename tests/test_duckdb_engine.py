"""
Tests for high-throughput in-memory query engine (DuckDB with Polars fallback)
and sandbox dsa.query_sql integration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.duckdb_engine import (
    DuckDBQueryError,
    get_query_engine,
    query_dataframe,
)
from src.core.sandbox_toolkit import Toolkit


@pytest.fixture
def sample_sales_df() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5, 6],
        "category": ["Tech", "Tech", "Auto", "Auto", "Health", "Tech"],
        "amount": [100.0, 150.0, 50.0, 80.0, 20.0, 200.0],
        "quantity": [1, 2, 1, 3, 1, 4],
    })


def test_duckdb_basic_query(sample_sales_df: pd.DataFrame) -> None:
    sql = "SELECT category, SUM(amount) as total_amt FROM sales GROUP BY category ORDER BY total_amt DESC"
    result = query_dataframe(sql, {"sales": sample_sales_df})

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert result.iloc[0]["category"] == "Tech"
    assert result.iloc[0]["total_amt"] == 450.0


def test_duckdb_fast_groupby(sample_sales_df: pd.DataFrame) -> None:
    engine = get_query_engine()
    grouped = engine.fast_groupby(
        sample_sales_df,
        group_cols=["category"],
        aggregations={"amount": ["sum", "avg", "count"]},
    )

    assert isinstance(grouped, pd.DataFrame)
    assert "amount_sum" in grouped.columns
    assert "amount_avg" in grouped.columns
    assert "amount_count" in grouped.columns
    tech_row = grouped[grouped["category"] == "Tech"].iloc[0]
    assert tech_row["amount_sum"] == 450.0
    assert tech_row["amount_count"] == 3


def test_duckdb_fast_quantiles() -> None:
    engine = get_query_engine()
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"score": rng.normal(100, 15, 1000)})

    qs = engine.fast_quantiles(df, "score", [0.25, 0.50, 0.75])
    assert 0.25 in qs
    assert 0.50 in qs
    assert 0.75 in qs
    assert qs[0.25] < qs[0.50] < qs[0.75]
    assert 90 < qs[0.50] < 110


def test_duckdb_unsafe_patterns_rejected(sample_sales_df: pd.DataFrame) -> None:
    unsafe_queries = [
        "COPY sales TO 'output.csv'",
        "ATTACH 'evil.db'",
        "INSTALL httpfs",
        "SELECT * FROM read_csv('secret.csv')",
        "   ",
    ]

    for q in unsafe_queries:
        with pytest.raises(DuckDBQueryError):
            query_dataframe(q, {"sales": sample_sales_df})


def test_dsa_query_sql_toolkit(sample_sales_df: pd.DataFrame) -> None:
    res = Toolkit.query_sql(
        "SELECT category, AVG(amount) as avg_amt FROM df GROUP BY category",
        df=sample_sales_df,
    )
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 3
    assert "avg_amt" in res.columns
