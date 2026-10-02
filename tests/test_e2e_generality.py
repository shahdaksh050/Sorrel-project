"""
End-to-End Generality & Multi-Dataset Integration Tests (FutureScope Phase 7).
Tests end-to-end deterministic pipeline execution across structurally diverse datasets:
1. Environmental time-series / AirQualityUCI (domain pack, limits, correlations).
2. A/B test experimental dataset (study design classification).
3. Clustered hierarchical dataset (intraclass correlation & design effects).
4. Edge list network dataset (graph analysis).
5. Wide-to-long table with subtotals (layout intelligence).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.controller import AgentController
from src.core.dependence import intraclass_correlation
from src.core.io import read_any
from src.tools.graph_analysis import GraphAnalysisTool


def test_e2e_deterministic_air_quality(tmp_path: Path) -> None:
    # Build a representative sample of AirQuality data if root file exists or synthetic
    root_csv = Path(__file__).resolve().parents[1] / "data" / "AirQualityUCI.csv"
    out_dir = str(tmp_path / "output_aq")
    os.makedirs(out_dir, exist_ok=True)

    if root_csv.exists():
        # Use first 150 rows of real file to keep test snappy
        df_sample = pd.read_csv(root_csv, sep=";", decimal=",", nrows=150)
        # Drop completely unnamed/empty columns from trailing semicolons
        df_sample = df_sample.loc[:, ~df_sample.columns.str.contains("^Unnamed")]
        target_csv = str(tmp_path / "air_quality_sample.csv")
        df_sample.to_csv(target_csv, index=False)
    else:
        df_sample = pd.DataFrame({
            "Date": pd.date_range("2004-03-10", periods=100, freq="h"),
            "CO(GT)": [1.5 + 0.1 * i if i % 10 != 0 else 12.0 for i in range(100)],
            "NO2(GT)": [50.0 + i for i in range(100)],
            "C6H6(GT)": [3.0 + 0.05 * i for i in range(100)],
            "T": [15.0 + 0.1 * i for i in range(100)],
        })
        target_csv = str(tmp_path / "air_quality_sample.csv")
        df_sample.to_csv(target_csv, index=False)

    controller = AgentController(
        objective="Which pollutants move together and does CO exceed air quality guidelines?",
        output_dir=out_dir,
        use_llm=False,
    )

    result = controller.analyze(file_path=target_csv)

    assert result["status"] == "complete"
    assert "insights" in result
    assert "findings" in result
    # Check that domain pack for air quality activated
    active_pack = controller.memory.get_context("active_domain_pack")
    assert active_pack == "air_quality"

    # Check report files generated
    rep_dir = Path(out_dir) / "reports"
    assert rep_dir.exists()
    md_files = list(rep_dir.glob("*.md"))
    json_files = list(rep_dir.glob("*.json"))
    assert len(md_files) >= 1
    assert len(json_files) >= 1


def test_e2e_ab_test_study_design(tmp_path: Path) -> None:
    df_ab = pd.DataFrame({
        "user_id": list(range(100)),
        "treatment": [0, 1] * 50,
        "converted": [0, 1, 0, 1] * 25,
        "spend": [10.0 + i for i in range(100)],
    })
    csv_path = str(tmp_path / "ab_test.csv")
    df_ab.to_csv(csv_path, index=False)

    out_dir = str(tmp_path / "output_ab")
    controller = AgentController(
        objective="Analyze conversion lift in the A/B test experiment",
        output_dir=out_dir,
        use_llm=False,
    )
    result = controller.analyze(file_path=csv_path)

    assert result["status"] == "complete"
    assert controller.study_design == "randomized_experiment"
    assert controller.memory.get_context("study_design") == "randomized_experiment"


def test_e2e_clustered_hierarchical_data() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for hospital in range(5):
        hosp_bias = hospital * 15.0
        for _ in range(30):
            rows.append({
                "hospital_id": f"Hosp_{hospital}",
                "recovery_days": hosp_bias + rng.normal(scale=2.0),
            })
    df_cluster = pd.DataFrame(rows)

    icc_info = intraclass_correlation(df_cluster, "hospital_id", "recovery_days")
    assert icc_info["icc"] > 0.80
    assert icc_info["deff"] > 10.0
    assert icc_info["needs_cluster_robust"] is True


def test_e2e_edge_list_graph_analysis(tmp_path: Path) -> None:
    edges = [
        {"from": "Server_Main", "to": f"Client_{i}", "weight": 1.0}
        for i in range(20)
    ]
    df_graph = pd.DataFrame(edges)
    csv_path = str(tmp_path / "network.csv")
    df_graph.to_csv(csv_path, index=False)

    tool = GraphAnalysisTool()
    res = tool.execute(file_path=csv_path)
    assert res["results"]["num_nodes"] == 21
    assert res["results"]["top_hub"]["node"] == "Server_Main"
    assert res["results"]["top_hub"]["degree"] == 20.0


def test_e2e_wide_to_long_and_subtotals(tmp_path: Path) -> None:
    content = (
        "Country,Sector,2020,2021,2022\n"
        "USA,Tech,100,120,140\n"
        "USA,Auto,50,55,60\n"
        "Total,All,150,175,200\n"
    )
    csv_path = str(tmp_path / "wide_totals.csv")
    Path(csv_path).write_text(content, encoding="utf-8")

    df, report = read_any(csv_path)
    # Subtotal excluded
    assert report.subtotals_excluded == 1
    assert len(df[df["Country"].astype(str).str.lower() == "total"]) == 0
    # Wide reshaped to long
    assert report.reshaped_from_wide is True
    assert "time" in df.columns
    assert "value" in df.columns
    assert set(df["Country"].unique()) == {"USA"}
