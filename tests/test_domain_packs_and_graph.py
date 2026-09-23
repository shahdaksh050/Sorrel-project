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


def test_air_quality_domain_pack_matches_real_world_column_names() -> None:
    """Regression: the limit-key regexes originally required the pollutant
    token to be immediately followed by "_" or end-of-string, so they never
    matched the project's own reference dataset's real column names, which
    carry a method suffix in parentheses with no underscore
    ("CO(GT)", "NO2(GT)", "C6H6(GT)" — see AirQualityUCI.csv). This is the
    Phase 5 exit gate itself ("air quality analysis automatically cites WHO
    limit exceedances"), so it must pass against those exact names."""
    df = pd.DataFrame({
        "CO(GT)": [2.0, 4.0, 8.0, 11.5, 12.0],  # 2 of 5 exceed 10.0 mg/m3
        "NO2(GT)": [50.0, 80.0, 120.0, 150.0, 180.0],  # none exceed 200.0
        "C6H6(GT)": [1.0, 2.0, 3.0, 6.0, 7.0],  # 2 of 5 exceed 5.0
    })
    findings = evaluate_domain_pack(df, pack=_AIR_QUALITY_PACK)
    flagged_cols = {f.evidence["column"] for f in findings}
    assert flagged_cols == {"CO(GT)", "C6H6(GT)"}
    co_finding = next(f for f in findings if f.evidence["column"] == "CO(GT)")
    assert co_finding.evidence["exceedance_rate"] == 0.4


def test_air_quality_domain_pack_ignores_sensor_response_columns() -> None:
    """Null case: AirQualityUCI also carries raw tin-oxide sensor response
    columns named "PT08.S1(CO)" / "PT08.S4(NO2)" right next to the real
    ground-truth concentration columns. These are unitless sensor readings
    in the hundreds/thousands, not CO/NO2 concentrations on the mg/m3 or
    ug/m3 scale the limits assume — matching them would fabricate a bogus
    exceedance claim against the wrong unit, so they must be excluded even
    though the pollutant token appears as a substring."""
    df = pd.DataFrame({
        "PT08.S1(CO)": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        "PT08.S4(NO2)": [1500.0, 1600.0, 1700.0, 1800.0, 1900.0],
    })
    findings = evaluate_domain_pack(df, pack=_AIR_QUALITY_PACK)
    assert findings == []


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


def test_graph_analysis_ignores_ordinary_business_columns() -> None:
    """Null case (regression): the original unanchored loose patterns
    matched "source"/"target" as a substring anywhere in a column name, so
    an ordinary customer/order table with "marketing_source" and
    "sales_target" columns — no graph structure at all — scored
    applies_to=0.85, "very likely a network". The two columns also differ
    in kind (categorical vs numeric), which a real edge list's source/target
    columns never do."""
    from src.core.profiler import profile_dataframe

    df = pd.DataFrame({
        "customer_id": list(range(50)),
        "marketing_source": (["Facebook", "Google", "Email", "Referral"] * 13)[:50],
        "sales_target": [1000.0 + i * 10 for i in range(50)],
        "revenue": [900.0 + i * 9 for i in range(50)],
    })
    profile = profile_dataframe(df)
    tool = GraphAnalysisTool()
    assert tool.applies_to(profile, None) < 0.2
    assert tool.default_params(profile, None) == {}
