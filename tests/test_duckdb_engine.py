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
    HighThroughputQueryEngine,
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


def test_duckdb_regex_evasions_rejected(sample_sales_df: pd.DataFrame, tmp_path: object) -> None:
    """`_auto`-suffixed table functions and bare quoted paths used to evade the
    word-boundary regex denylist entirely (`\\bread_csv\\b` does not match
    `read_csv_auto`). These must be rejected."""
    evasions = [
        "SELECT * FROM read_csv_auto('requirements.txt')",
        "SELECT * FROM read_json_auto('requirements.txt')",
        "SELECT * FROM read_parquet_auto('requirements.txt')",
        "SELECT * FROM 'requirements.txt'",
        "SELECT * FROM glob('*')",
        "SET enable_external_access=true",
    ]
    for q in evasions:
        with pytest.raises(DuckDBQueryError):
            query_dataframe(q, {"sales": sample_sales_df})


def test_duckdb_read_star_functions_all_rejected(sample_sales_df: pd.DataFrame) -> None:
    """The denylist previously enumerated individual `read_*` function names
    and missed several real ones (read_ipc, read_avro, read_excel,
    read_database) — on the Polars SQLContext fallback (used whenever the
    duckdb package is unavailable) these are real, unblocked filesystem
    reads. A single catch-all `read_\\w*` pattern must reject all of them."""
    evasions = [
        "SELECT * FROM read_ipc('requirements.txt')",
        "SELECT * FROM read_avro('requirements.txt')",
        "SELECT * FROM read_excel('requirements.txt')",
        "SELECT * FROM read_database('requirements.txt')",
    ]
    for q in evasions:
        with pytest.raises(DuckDBQueryError):
            query_dataframe(q, {"sales": sample_sales_df})


def test_duckdb_engine_level_lock_holds_even_without_regex(sample_sales_df: pd.DataFrame) -> None:
    """The regex denylist is defense-in-depth only. Bypass it entirely by calling
    the private executor directly and confirm DuckDB's own
    enable_external_access=false + lock_configuration=true still blocks file access —
    this is the actual security boundary the sandbox relies on."""
    engine = HighThroughputQueryEngine()
    with pytest.raises(DuckDBQueryError):
        engine._execute_duckdb(
            "SELECT * FROM read_csv_auto('requirements.txt')", {"sales": sample_sales_df}
        )


def test_dsa_query_sql_toolkit(sample_sales_df: pd.DataFrame) -> None:
    res = Toolkit.query_sql(
        "SELECT category, AVG(amount) as avg_amt FROM df GROUP BY category",
        df=sample_sales_df,
    )
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 3
    assert "avg_amt" in res.columns
