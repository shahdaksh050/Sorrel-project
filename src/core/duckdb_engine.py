"""
High-throughput in-memory query engine powered by DuckDB (with Polars SQL fallback).

Provides vectorized, multi-threaded columnar querying for DataFrames, group-bys,
window calculations, and aggregations without single-threaded Pandas overhead.

Safety & Governance:
- Runs in-memory only (":memory:").
- Disallows disk mutations, external file reads, ATTACH, COPY, INSTALL, LOAD.
- Defensively copies results to prevent accidental DataFrame state mutation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


class DuckDBQueryError(RuntimeError):
    """Raised when an in-memory SQL query fails or violates safety rules."""
    pass


#: SQL keywords / functions disallowed to prevent disk I/O, installation, or privilege escalation.
#: This is defense-in-depth only — the authoritative boundary is DuckDB's own
#: `enable_external_access=false` + `lock_configuration=true`, set on every connection
#: in `_execute_duckdb` below, since any regex denylist over a full SQL dialect is
#: gameable (e.g. `read_csv` doesn't match `read_csv_auto`; DuckDB also allows a bare
#: quoted path as a table reference with no function name at all: `FROM 'x.csv'`).
#: The Polars fallback has no equivalent engine-level lock, so these patterns are its
#: only protection — kept broad (prefix match, not `\b...\b`) for that reason.
_DISALLOWED_SQL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bATTACH\b", re.IGNORECASE),
    re.compile(r"\bDETACH\b", re.IGNORECASE),
    re.compile(r"\bINSTALL\b", re.IGNORECASE),
    re.compile(r"\bLOAD\b", re.IGNORECASE),
    re.compile(r"\bEXPORT\b", re.IGNORECASE),
    re.compile(r"\bIMPORT\b", re.IGNORECASE),
    re.compile(r"\bPRAGMA\b", re.IGNORECASE),
    re.compile(r"\bCOPY\s+.*?\s+TO\b", re.IGNORECASE),
    # Catch-all for every `read_*` table function (read_csv, read_parquet, read_json,
    # read_ndjson, read_text, read_blob, read_ipc, read_avro, read_excel, read_database,
    # and any future/undocumented one) instead of enumerating each name individually —
    # the enumeration previously missed read_ipc/read_avro/read_excel/read_database,
    # which the Polars SQLContext fallback still executes as real filesystem reads.
    re.compile(r"\bread_\w*", re.IGNORECASE),
    re.compile(r"\bscan_\w*", re.IGNORECASE),
    re.compile(r"\bparquet_scan\w*", re.IGNORECASE),
    re.compile(r"\bglob\s*\(", re.IGNORECASE),
    re.compile(r"\bsniff_csv\w*", re.IGNORECASE),
    re.compile(r"""FROM\s+['"]""", re.IGNORECASE),  # bare quoted-path table reference
)


def _validate_safe_sql(sql: str) -> None:
    """Ensure query operates only on registered in-memory tables and contains no unsafe commands."""
    stripped = sql.strip()
    if not stripped:
        raise DuckDBQueryError("SQL query cannot be empty.")

    for pattern in _DISALLOWED_SQL_PATTERNS:
        if pattern.search(stripped):
            raise DuckDBQueryError(f"Unsafe SQL pattern detected: {pattern.pattern}")


@dataclass
class QueryEngineStatus:
    engine_name: str
    version: str
    in_memory: bool = True


class HighThroughputQueryEngine:
    """
    In-memory columnar query engine.
    Prefers DuckDB for multi-threaded vectorized SQL; falls back to Polars SQLContext.
    """

    def __init__(self) -> None:
        self._duckdb_available = False
        self._polars_available = False
        self._duckdb: Any = None
        self._polars: Any = None

        try:
            import duckdb
            self._duckdb = duckdb
            self._duckdb_available = True
        except ImportError:
            self._duckdb = None

        try:
            import polars as pl
            self._polars = pl
            self._polars_available = True
        except ImportError:
            self._polars = None

        if not self._duckdb_available and not self._polars_available:
            raise RuntimeError("Neither DuckDB nor Polars is available in the environment.")

    def get_status(self) -> QueryEngineStatus:
        if self._duckdb_available and self._duckdb is not None:
            return QueryEngineStatus(
                engine_name="DuckDB",
                version=getattr(self._duckdb, "__version__", "unknown"),
            )
        elif self._polars_available and self._polars is not None:
            return QueryEngineStatus(
                engine_name="Polars",
                version=getattr(self._polars, "__version__", "unknown"),
            )
        return QueryEngineStatus(engine_name="None", version="0.0.0")

    def execute_query(
        self,
        sql: str,
        tables: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """
        Execute safe in-memory SQL query against provided pandas DataFrames.

        Args:
            sql: SQL statement (SELECT/WITH only).
            tables: Mapping of table names to DataFrames.

        Returns:
            Resulting pd.DataFrame.
        """
        _validate_safe_sql(sql)
        table_map = tables or {}

        if self._duckdb_available and self._duckdb is not None:
            return self._execute_duckdb(sql, table_map)
        elif self._polars_available and self._polars is not None:
            return self._execute_polars(sql, table_map)
        else:
            raise DuckDBQueryError("No query backend available.")

    def _execute_duckdb(
        self,
        sql: str,
        tables: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        con = self._duckdb.connect(":memory:")
        try:
            # Authoritative safety boundary (not the regex denylist above): disables all
            # file-system and network access at the engine level, then locks the setting
            # so the query itself cannot re-enable it via `SET enable_external_access=true`.
            con.execute("SET enable_external_access=false")
            con.execute("SET autoinstall_known_extensions=false")
            con.execute("SET autoload_known_extensions=false")
            con.execute("SET lock_configuration=true")
            for name, df in tables.items():
                con.register(name, df)
            res = con.execute(sql).fetchdf()
            return res.copy()
        except Exception as exc:
            raise DuckDBQueryError(f"DuckDB query failed: {exc}") from exc
        finally:
            con.close()

    def _execute_polars(
        self,
        sql: str,
        tables: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        try:
            ctx = self._polars.SQLContext()
            for name, df in tables.items():
                pldf = self._polars.from_pandas(df)
                ctx.register(name, pldf)
            res_pl = ctx.execute(sql).collect()
            return res_pl.to_pandas()
        except Exception as exc:
            raise DuckDBQueryError(f"Polars SQL query failed: {exc}") from exc

    def fast_groupby(
        self,
        df: pd.DataFrame,
        group_cols: list[str],
        aggregations: dict[str, list[str]],
    ) -> pd.DataFrame:
        """
        Vectorized high-speed group-by aggregation.

        Example:
            aggregations = {"amount": ["sum", "avg", "count"], "price": ["min", "max"]}
        """
        if df.empty or not group_cols or not aggregations:
            return pd.DataFrame()

        grp_str = ", ".join(f'"{c}"' for c in group_cols)
        agg_exprs: list[str] = [f'"{c}"' for c in group_cols]

        for col, funcs in aggregations.items():
            for fn in funcs:
                fn_clean = fn.upper()
                if fn_clean == "MEAN":
                    fn_clean = "AVG"
                out_name = f"{col}_{fn.lower()}"
                agg_exprs.append(f'{fn_clean}("{col}") AS "{out_name}"')

        select_clause = ", ".join(agg_exprs)
        sql = f'SELECT {select_clause} FROM dataset GROUP BY {grp_str}'
        return self.execute_query(sql, {"dataset": df})

    def fast_quantiles(
        self,
        df: pd.DataFrame,
        col: str,
        quantiles: list[float] | None = None,
    ) -> dict[float, float]:
        """Compute exact quantiles using DuckDB/Polars percentile functions."""
        if df.empty or col not in df.columns:
            return {}

        qs = quantiles or [0.25, 0.50, 0.75]
        q_exprs = [
            f'QUANTILE_CONT("{col}", {q}) AS "q_{int(q*100)}"'
            for q in qs
        ]
        sql = f'SELECT {", ".join(q_exprs)} FROM dataset'
        res = self.execute_query(sql, {"dataset": df})
        if res.empty:
            return {}

        out: dict[float, float] = {}
        for q in qs:
            key = f"q_{int(q*100)}"
            if key in res.columns:
                out[q] = float(res.iloc[0][key])
        return out


# Global singleton engine instance
_ENGINE: HighThroughputQueryEngine | None = None


def get_query_engine() -> HighThroughputQueryEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = HighThroughputQueryEngine()
    return _ENGINE


def query_dataframe(sql: str, tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Convenience helper to query DataFrames in-memory via DuckDB/Polars."""
    return get_query_engine().execute_query(sql, tables)
