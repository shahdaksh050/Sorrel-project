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


def test_retail_ecommerce_domain_pack() -> None:
    df = pd.DataFrame({
        "order_id": [f"ORD_{i // 2}" for i in range(20)],
        "customer_id": [f"CUST_{i % 5}" for i in range(20)],
        "revenue": [50.0 + i * 5 for i in range(19)] + [-25.0],  # 1 refund
        "discount": [0.10] * 18 + [0.60, 0.70],  # 2 exceedances (> 0.50)
    })
    pack = detect_domain_pack(df, objective="Retail e-commerce revenue and order metrics")
    assert pack is not None
    assert pack.name == "retail_ecommerce"

    findings = evaluate_domain_pack(df, pack=pack)
    assert any("average order value" in f.headline.lower() for f in findings)
    assert any("repeat purchase rate" in f.headline.lower() for f in findings)
    assert any("negative transactions" in f.headline.lower() for f in findings)
    assert any("discount" in f.headline.lower() for f in findings)


def test_retail_disambiguation_unit_price_and_quantity() -> None:
    """When only unit_price and quantity exist, synthesize line total for accurate volume and AOV."""
    df = pd.DataFrame({
        "invoice_no": [f"INV_{i // 2}" for i in range(10)],
        "unit_price": [10.0] * 10,
        "quantity": [2] * 10,  # line total is 20.0 each, total volume 200.0, 5 orders -> AOV $40.00
    })
    pack = detect_domain_pack(df, objective="Retail shopping basket analysis")
    assert pack is not None
    findings = evaluate_domain_pack(df, pack=pack)
    aov_finding = next((f for f in findings if "average order value" in f.headline.lower()), None)
    assert aov_finding is not None
    assert "$40.00" in aov_finding.headline
    assert "synthesized" in aov_finding.headline
    assert aov_finding.evidence["total_revenue"] == 200.0


def test_retail_disambiguation_unit_price_and_total_amount() -> None:
    """When both unit_price and total_amount exist, prioritize explicit total_amount."""
    df = pd.DataFrame({
        "invoice_no": [f"INV_{i // 2}" for i in range(10)],
        "unit_price": [10.0] * 10,
        "quantity": [2] * 10,
        "total_amount": [25.0] * 10,  # explicit line total (e.g. after taxes/fees)
    })
    pack = detect_domain_pack(df, objective="Retail sales analysis")
    assert pack is not None
    findings = evaluate_domain_pack(df, pack=pack)
    aov_finding = next((f for f in findings if "average order value" in f.headline.lower()), None)
    assert aov_finding is not None
    assert "$50.00" in aov_finding.headline
    assert aov_finding.evidence["total_revenue"] == 250.0
    assert "synthesized" not in aov_finding.headline


def test_retail_disambiguation_unit_price_only() -> None:
    """When only unit_price exists without quantity, report Average Unit Price rather than gross sum."""
    df = pd.DataFrame({
        "product_id": [f"P_{i}" for i in range(10)],
        "unit_price": [15.0 + i for i in range(10)],  # mean = 19.50
    })
    pack = detect_domain_pack(df, objective="Retail product catalog and store sales")
    assert pack is not None
    findings = evaluate_domain_pack(df, pack=pack)
    price_finding = next((f for f in findings if "average unit price" in f.headline.lower()), None)
    assert price_finding is not None
    assert "$19.50" in price_finding.headline
    assert not any("average order value" in f.headline.lower() for f in findings)


def test_retail_disambiguation_all_nan_amount_no_spurious_aov() -> None:
    """When transaction value or unit rate is entirely NaN, no artificial $0.00 AOV is generated."""
    nan_val = float("nan")
    df = pd.DataFrame({
        "order_id": ["ORD_1", "ORD_2", "ORD_3"],
        "total_amount": [nan_val, nan_val, nan_val],
    })
    pack = detect_domain_pack(df, objective="Retail sales and transactions")
    assert pack is not None
    findings = evaluate_domain_pack(df, pack=pack)
    assert not any("average order value" in f.headline.lower() for f in findings)

    df2 = pd.DataFrame({
        "order_id": ["ORD_1", "ORD_2", "ORD_3"],
        "unit_price": [nan_val, nan_val, nan_val],
        "quantity": [nan_val, nan_val, nan_val],
    })
    findings2 = evaluate_domain_pack(df2, pack=pack)
    assert not any("average order value" in f.headline.lower() for f in findings2)
    assert not any("average unit price" in f.headline.lower() for f in findings2)


def test_saas_subscription_domain_pack() -> None:
    df = pd.DataFrame({
        "subscriber_id": [f"SUB_{i:03d}" for i in range(50)],
        "churn": [0] * 40 + [1] * 10,  # 20% churn rate
        "mrr": [100.0] * 50,
        "tenure_months": [12.0] * 50,
    })
    pack = detect_domain_pack(df, objective="Analyze SaaS subscription churn and ARR")
    assert pack is not None
    assert pack.name == "saas_subscription"

    findings = evaluate_domain_pack(df, pack=pack)
    assert any("churn rate" in f.headline.lower() for f in findings)
    assert any("mrr" in f.headline.lower() or "monthly recurring revenue" in f.headline.lower() for f in findings)


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


def test_graph_analysis_weighted_leaders_differ_and_mass_conserved(tmp_path: Path) -> None:
    # Construct a weighted directed graph where leaders differ and sinks exist:
    # - Node A has the highest unweighted degree (4 outgoing connections)
    # - Node B has the highest weighted strength (massive weights 100 + 100 = 200)
    # - Node C has the highest PageRank (accumulates heavy incoming transitions, out-degree 0 / sink)
    edges = [
        {"from_node": "A", "to_node": "B", "volume": 1.0},
        {"from_node": "A", "to_node": "C", "volume": 1.0},
        {"from_node": "A", "to_node": "D", "volume": 1.0},
        {"from_node": "A", "to_node": "E", "volume": 1.0},
        {"from_node": "B", "to_node": "C", "volume": 100.0},
        {"from_node": "B", "to_node": "D", "volume": 100.0},
    ]
    df = pd.DataFrame(edges)
    path = str(tmp_path / "weighted_graph.csv")
    df.to_csv(path, index=False)

    tool = GraphAnalysisTool()
    res = tool.execute(
        file_path=path,
        source_column="from_node",
        target_column="to_node",
        weight_column="volume",
    )

    results = res["results"]
    assert results["top_hub"]["node"] == "A"
    assert results["top_hub"]["degree"] == 4

    assert results["top_strength"]["node"] == "B"
    assert results["top_strength"]["strength"] > 200.0

    assert results["top_pagerank"]["node"] == "C"

    # Leaders are distinct
    assert results["top_hub"]["node"] != results["top_strength"]["node"]
    assert results["top_strength"]["node"] != results["top_pagerank"]["node"]

    # PageRank mass conservation: sum of PageRank scores across all nodes must equal 1.0
    assert results["pagerank_mass_sum"] == pytest.approx(1.0, abs=1e-5)

    # Check evidence reporting
    hub_finding = next(f for f in res["findings"] if f["finding_id"] == "graph_top_hub")
    ev = hub_finding["evidence"]
    assert ev["top_hub"] == "A"
    assert ev["top_strength_node"] == "B"
    assert ev["top_pagerank_node"] == "C"
    assert ev["pagerank_mass_sum"] == pytest.approx(1.0, abs=1e-5)
