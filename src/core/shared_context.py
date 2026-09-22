"""
Shared Analysis Context for the Agentic Data Analysis System.

FutureScope Phase 2:
Computes heavy analytical primitives once (correlation matrices, column summaries,
group indices) and caches them across multiple tools to eliminate redundant
expensive calculations on large datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SharedAnalysisContext:
    """
    Cached intermediate computations and statistical primitives shared across tools.
    """

    df: pd.DataFrame
    _correlations: dict[str, pd.DataFrame] = field(default_factory=dict)
    _column_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    _group_indices: dict[str, dict[Any, np.ndarray]] = field(default_factory=dict)

    def get_correlation_matrix(self, method: str = "pearson") -> pd.DataFrame:
        """
        Get or compute a correlation matrix for numeric columns.
        Supported methods: 'pearson', 'spearman', 'kendall'.
        """
        if method not in self._correlations:
            numeric_df = self.df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] > 1:
                self._correlations[method] = numeric_df.corr(method=method)
            else:
                self._correlations[method] = pd.DataFrame()
        return self._correlations[method]

    def get_column_summary(self, col: str) -> dict[str, Any]:
        """
        Get or compute descriptive statistics for a single column.
        """
        if col not in self._column_summaries:
            if col not in self.df.columns:
                return {}
            series = self.df[col].dropna()
            if series.empty:
                self._column_summaries[col] = {"count": 0, "missing": len(self.df)}
            elif pd.api.types.is_numeric_dtype(series):
                q25, q75 = series.quantile(0.25), series.quantile(0.75)
                self._column_summaries[col] = {
                    "count": int(series.count()),
                    "missing": int(self.df[col].isna().sum()),
                    "mean": float(series.mean()),
                    "std": float(series.std()) if len(series) > 1 else 0.0,
                    "median": float(series.median()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "q25": float(q25),
                    "q75": float(q75),
                    "iqr": float(q75 - q25),
                    "skew": float(series.skew()) if len(series) > 2 else 0.0,
                }
            else:
                top_counts = series.value_counts().head(5).to_dict()
                self._column_summaries[col] = {
                    "count": int(series.count()),
                    "missing": int(self.df[col].isna().sum()),
                    "unique": int(series.nunique()),
                    "top_values": {str(k): int(v) for k, v in top_counts.items()},
                }
        return self._column_summaries[col]

    def get_group_indices(self, col: str) -> dict[Any, np.ndarray]:
        """
        Get or compute row index arrays grouped by a categorical column.
        """
        if col not in self._group_indices:
            if col not in self.df.columns:
                return {}
            # Group rows by value
            groups: dict[Any, list[int]] = {}
            for idx, val in enumerate(self.df[col]):
                if pd.isna(val):
                    continue
                if val not in groups:
                    groups[val] = []
                groups[val].append(idx)
            self._group_indices[col] = {k: np.array(v, dtype=int) for k, v in groups.items()}
        return self._group_indices[col]

    def clear(self) -> None:
        """Invalidate all cached computations."""
        self._correlations.clear()
        self._column_summaries.clear()
        self._group_indices.clear()
