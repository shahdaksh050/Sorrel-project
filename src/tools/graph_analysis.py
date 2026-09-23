"""
Graph and network structure analysis tool.

FutureScope Phase 5 (FutureScope.md §5.4):
- Detects edge list representations in datasets.
- Computes degree distributions, connected components, and PageRank centrality.
- Identifies critical network hubs and graph fragmentation using scipy.sparse.csgraph.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.core.findings import Finding
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

#: The loose fallback patterns originally matched "source"/"target" as an
#: unanchored substring — which also matches ordinary business columns like
#: "marketing_source" and "sales_target" (a real false positive: an
#: ordinary customer/order table with those two columns and zero graph
#: structure scored applies_to=0.85, "very likely a network"). Bare
#: "source"/"target" now live only in the exact-full-name strict patterns
#: below; the loose fallback keeps only compound "*_id"-style tokens, which
#: are far less likely to appear in non-graph tabular data, bounded to a
#: whole underscore-delimited token so "source_data" doesn't match either.
_SRC_PATTERNS = (
    re.compile(r"^(?:source|src|from|sender|parent|origin|caller|user1|node1)$", re.I),
    re.compile(r"(?:^|_)(?:from_id|sender_id)(?:$|_)", re.I),
)
_DST_PATTERNS = (
    re.compile(r"^(?:target|dst|to|receiver|recipient|child|dest|destination|callee|user2|node2)$", re.I),
    re.compile(r"(?:^|_)(?:to_id|recipient_id)(?:$|_)", re.I),
)


def _detect_edge_columns(columns: list[str]) -> tuple[str, str] | None:
    src_col = None
    dst_col = None
    for c in columns:
        if not src_col and any(p.search(c) for p in _SRC_PATTERNS):
            src_col = c
        elif not dst_col and any(p.search(c) for p in _DST_PATTERNS):
            dst_col = c

    if src_col and dst_col and src_col != dst_col:
        return src_col, dst_col
    return None


def _detect_edge_columns_from_profile(profile: DatasetProfile) -> tuple[str, str] | None:
    """Name-based detection, then a same-kind sanity gate: a genuine edge
    list's source and target columns hold the same *kind* of value (both
    categorical node IDs, or both identifiers). A numeric measure that
    happens to be named "...target" paired by name alone with an unrelated
    categorical "...source" column is not a graph — it just has column
    names that collide with edge-list vocabulary. This uses only the
    profile (already computed), so it stays ~0 extra cost."""
    col_names = [c.name for c in profile.columns]
    detected = _detect_edge_columns(col_names)
    if detected is None:
        return None
    src_col, dst_col = detected
    kind_by_name = {c.name: c.kind for c in profile.columns}
    if kind_by_name.get(src_col) != kind_by_name.get(dst_col):
        return None
    return detected


class GraphAnalysisTool(BaseTool):
    """Network graph structure and centrality analysis from edge list data."""

    name = "graph_analysis"
    description = (
        "Analyze network graph structure from edge lists: degree distribution, "
        "connected components, PageRank centrality, and critical hub identification."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or profile.row_count < 5:
            return 0.0
        edge_cols = _detect_edge_columns_from_profile(profile)
        return 0.85 if edge_cols is not None else 0.1

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        if profile is None:
            return {}
        edge_cols = _detect_edge_columns_from_profile(profile)
        if edge_cols:
            return {"source_column": edge_cols[0], "target_column": edge_cols[1]}
        return {}

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the edge list dataset.",
                "required": True,
            },
            "source_column": {
                "type": "string",
                "description": "Column representing source nodes.",
                "required": False,
            },
            "target_column": {
                "type": "string",
                "description": "Column representing target nodes.",
                "required": False,
            },
            "weight_column": {
                "type": "string",
                "description": "Optional column for edge weights.",
                "required": False,
            },
        }

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        source_column: str | None = None,
        target_column: str | None = None,
        weight_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset is empty.")

        src_col = source_column
        dst_col = target_column

        if not src_col or not dst_col or src_col not in df.columns or dst_col not in df.columns:
            detected = _detect_edge_columns(list(df.columns))
            if detected:
                src_col, dst_col = detected
            else:
                raise ToolExecutionError(
                    "Could not identify source and target columns for graph analysis. "
                    "Please specify 'source_column' and 'target_column'."
                )

        edges_df = df[[src_col, dst_col]].dropna()
        if edges_df.empty:
            raise ToolExecutionError("No valid edges found after dropping missing values.")

        # Map nodes to contiguous integer IDs
        all_nodes = pd.concat([edges_df[src_col], edges_df[dst_col]]).unique()
        node_to_id = {node: idx for idx, node in enumerate(all_nodes)}
        id_to_node = {idx: node for node, idx in node_to_id.items()}
        num_nodes = len(all_nodes)
        num_edges = len(edges_df)

        src_ids = edges_df[src_col].map(node_to_id).values
        dst_ids = edges_df[dst_col].map(node_to_id).values

        if weight_column and weight_column in df.columns:
            weights = pd.to_numeric(df[weight_column], errors="coerce").fillna(1.0).values
        else:
            weights = np.ones(num_edges, dtype=float)

        # Build sparse adjacency matrix
        adj = csr_matrix((weights, (src_ids, dst_ids)), shape=(num_nodes, num_nodes))

        # Connected components
        n_components, labels = connected_components(adj, directed=False)
        _uniq, comp_counts = np.unique(labels, return_counts=True)
        largest_comp_size = int(np.max(comp_counts))
        largest_comp_pct = round(largest_comp_size / num_nodes * 100.0, 1)

        # Degree calculation
        out_degrees = np.asarray(adj.sum(axis=1)).flatten()
        in_degrees = np.asarray(adj.sum(axis=0)).flatten()
        total_degrees = out_degrees + in_degrees

        top_hub_idx = int(np.argmax(total_degrees))
        top_hub_node = str(id_to_node[top_hub_idx])
        top_hub_degree = float(total_degrees[top_hub_idx])

        # PageRank via power iteration
        d = 0.85
        p = np.full(num_nodes, 1.0 / num_nodes)
        out_deg_safe = np.maximum(out_degrees, 1.0)
        norm_adj = adj.copy().astype(float)
        norm_adj.data /= out_deg_safe[norm_adj.nonzero()[0]]

        for _step in range(25):
            p = d * (norm_adj.T.dot(p)) + (1.0 - d) / num_nodes

        top_pagerank_idx = int(np.argmax(p))
        top_pagerank_node = str(id_to_node[top_pagerank_idx])
        top_pagerank_val = round(float(p[top_pagerank_idx]), 5)

        findings: list[Finding] = []

        # Finding: Hub identification
        findings.append(
            Finding(
                finding_id="graph_top_hub",
                kind="network",
                headline=f"Top network hub is '{top_hub_node}' with {int(top_hub_degree)} connections (PageRank={top_pagerank_val})",
                detail=(
                    f"Graph contains {num_nodes} nodes and {num_edges} edges across {n_components} connected component(s). "
                    f"Largest connected component contains {largest_comp_pct}% of all nodes."
                ),
                importance=0.8,
                effect=top_hub_degree / max(1.0, float(num_nodes)),
                effect_kind="share",
                evidence={
                    "top_hub": top_hub_node,
                    "degree": top_hub_degree,
                    "top_pagerank_node": top_pagerank_node,
                    "pagerank": top_pagerank_val,
                    "nodes": num_nodes,
                    "edges": num_edges,
                    "components": n_components,
                },
                source_tool=self.name,
            )
        )

        # Finding: Fragmentation if disconnected
        if n_components > 1:
            findings.append(
                Finding(
                    finding_id="graph_fragmentation",
                    kind="network",
                    headline=f"Network is fragmented into {n_components} disconnected components",
                    detail=f"The largest component contains {largest_comp_size} nodes ({largest_comp_pct}% of network).",
                    importance=0.7,
                    effect=1.0 - (largest_comp_size / num_nodes),
                    effect_kind="share",
                    evidence={
                        "components": n_components,
                        "largest_component_size": largest_comp_size,
                        "largest_component_pct": largest_comp_pct,
                    },
                    source_tool=self.name,
                )
            )

        summary = (
            f"Graph analysis of {num_nodes} nodes and {num_edges} edges across {n_components} component(s). "
            f"Top hub: '{top_hub_node}' (degree={int(top_hub_degree)})."
        )

        results = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_components": n_components,
            "largest_component_size": largest_comp_size,
            "largest_component_pct": largest_comp_pct,
            "top_hub": {"node": top_hub_node, "degree": top_hub_degree},
            "top_pagerank": {"node": top_pagerank_node, "score": top_pagerank_val},
        }

        return {
            "summary": summary,
            "results": results,
            "findings": [f.to_dict() for f in findings],
            "finding_payloads": [f.to_dict() for f in findings],
        }
