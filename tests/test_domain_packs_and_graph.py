"""
Unit tests for Domain Packs and Graph Analysis (FutureScope Phase 5).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.domain_packs import (
    _AIR_QUALITY_PACK,
    detect_domain_pack,
    evaluate_domain_pack,
)
from src.tools.graph_analysis import GraphAnalysisTool


def test_air_quality_domain_pack_exceedance() -> None:
    df = pd.DataFrame({
        "co": [2.0, 4.0, 8.0, 11.5, 12.0],  # 2 of 5 exceed 10.0
        "no2": [50.0, 80.0, 120.0, 150.0, 180.0],  # none exceed 200.0
    })
    pack = detect_domain_pack(df, objective="Analyze air pollutants")
    assert pack is not None
    assert pack.name == "air_quality"

    findings = evaluate_domain_pack(df, pack=pack)
    assert len(findings) == 1
    f = findings[0]
    assert "co" in f.headline.lower()
    assert "40.0%" in f.headline  # 2/5 = 40%
    assert "12.0" in f.headline  # peak
    assert f.evidence["exceedance_rate"] == 0.4


def test_healthcare_domain_pack_exceedance() -> None:
    df = pd.DataFrame({
        "systolic": [115, 120, 145, 150],  # 2 exceed 140
        "diastolic": [75, 80, 85, 95],  # 1 exceeds 90
    })
    pack = detect_domain_pack(df, objective="Check patient blood pressure")
    assert pack is not None
    assert pack.name == "healthcare"

    findings = evaluate_domain_pack(df, pack=pack)
    assert len(findings) == 2
    assert any("systolic" in f.headline.lower() for f in findings)
    assert any("diastolic" in f.headline.lower() for f in findings)


def test_domain_pack_null() -> None:
    # All values clean and below limits
    df = pd.DataFrame({
        "co": [1.0, 2.0, 3.0],
        "no2": [20.0, 30.0, 40.0],
    })
    findings = evaluate_domain_pack(df, pack=_AIR_QUALITY_PACK)
    assert len(findings) == 0


@pytest.fixture
def graph_csv(tmp_path: Path) -> str:
    # Star graph: hub 'Node_A' connected to 5 other nodes
    edges = [
        {"source": "Node_A", "target": "Node_B"},
        {"source": "Node_A", "target": "Node_C"},
        {"source": "Node_A", "target": "Node_D"},
        {"source": "Node_A", "target": "Node_E"},
        {"source": "Node_A", "target": "Node_F"},
    ]
    df = pd.DataFrame(edges)
    path = str(tmp_path / "edges.csv")
    df.to_csv(path, index=False)
    return path


def test_graph_analysis_star_hub(graph_csv: str) -> None:
    tool = GraphAnalysisTool()
    res = tool.execute(file_path=graph_csv)

    assert res["results"]["num_nodes"] == 6
    assert res["results"]["num_edges"] == 5
    assert res["results"]["num_components"] == 1
    assert res["results"]["top_hub"]["node"] == "Node_A"
    assert res["results"]["top_hub"]["degree"] == 5.0
    assert any("top network hub" in f["headline"].lower() for f in res["findings"])


def test_graph_analysis_disconnected(tmp_path: Path) -> None:
    # Two disconnected components: (A-B) and (C-D)
    edges = [
        {"from_node": "A", "to_node": "B"},
        {"from_node": "C", "to_node": "D"},
    ]
    df = pd.DataFrame(edges)
    path = str(tmp_path / "disconnected_edges.csv")
    df.to_csv(path, index=False)

    tool = GraphAnalysisTool()
    res = tool.execute(file_path=path, source_column="from_node", target_column="to_node")

    assert res["results"]["num_components"] == 2
    assert any("fragmented" in f["headline"].lower() for f in res["findings"])


def test_graph_analysis_schema_and_applies() -> None:
    tool = GraphAnalysisTool()
    schema = tool.get_schema()
    assert "source_column" in schema
    assert "target_column" in schema
