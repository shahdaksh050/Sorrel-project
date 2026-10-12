"""Regression tests for question-led segment discovery."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.core.profiler import profile_dataframe
from src.tools.segment_comparison import SegmentComparisonTool
from src.tools.statistical_analysis import SelectStatisticalTestTool


def _profile() -> object:
    frame = pd.DataFrame(
        {
            "monthly_charges": [40.0, 42.0, 80.0, 82.0, 55.0, 57.0],
            "churn": [0, 0, 1, 1, 0, 1],
            "contract": ["annual", "annual", "monthly", "monthly", "quarterly", "quarterly"],
        }
    )
    return profile_dataframe(frame)


def test_blank_objective_leaves_measure_unpinned_for_family_sweep() -> None:
    tool = SegmentComparisonTool()

    assert tool.default_params(_profile(), None) == {}  # type: ignore[arg-type]


def test_explicit_target_remains_the_primary_segment_measure() -> None:
    tool = SegmentComparisonTool()
    metadata = SimpleNamespace(target_column="churn")

    assert tool.default_params(_profile(), metadata) == {"measure_column": "churn"}  # type: ignore[arg-type]


def test_autonomous_discovery_skips_tiny_residual_groups_but_explicit_request_honours_them(
    tmp_path: Path,
) -> None:
    source = tmp_path / "catalog.csv"
    frame = pd.DataFrame(
        {
            "price": [float(index + 1) for index in range(100)],
            "platform": ["Android"] * 50 + ["iOS"] * 45 + ["Other"] * 5,
            "has_wifi": [True] * 95 + [False] * 5,
        }
    )
    frame.to_csv(source, index=False)

    autonomous = SegmentComparisonTool().execute(file_path=str(source))
    assert {row["dimension"] for row in autonomous["comparisons"]} == {"platform"}

    explicit = SegmentComparisonTool().execute(
        file_path=str(source),
        measure_column="price",
        dimension_column="has_wifi",
    )
    assert {row["dimension"] for row in explicit["comparisons"]} == {"has_wifi"}

    family = SelectStatisticalTestTool().execute(file_path=str(source), feature_column="price")
    assert {row["group_column"] for row in family["family_results"]} == {"platform"}
    assert family["excluded_underpowered_levels"] == 1
    assert "were omitted" in family["sample_size_note"]


def test_two_way_interaction_synergy_discovered(tmp_path: Path) -> None:
    # Construct a dataset where Tier=VIP has +15% lift, Channel=Direct has +10% lift,
    # but the joint segment (VIP x Direct) has a super-additive synergy (+60% lift).
    import numpy as np

    np.random.seed(42)
    n = 200
    tiers = np.random.choice(["VIP", "Standard"], n, p=[0.3, 0.7])
    channels = np.random.choice(["Direct", "Partner"], n, p=[0.4, 0.6])

    spend = 100.0 + np.random.normal(0, 10, n)
    for i in range(n):
        if tiers[i] == "VIP" and channels[i] == "Direct":
            spend[i] += 70.0  # Big joint synergy!
        elif tiers[i] == "VIP":
            spend[i] += 15.0
        elif channels[i] == "Direct":
            spend[i] += 10.0

    csv_path = tmp_path / "synergy.csv"
    pd.DataFrame({"spend": spend, "tier": tiers, "channel": channels}).to_csv(csv_path, index=False)

    tool = SegmentComparisonTool()
    res = tool.execute(file_path=str(csv_path), measure_column="spend")
    assert "interactions" in res
    assert len(res["interactions"]) > 0
    vip_direct = next(i for i in res["interactions"] if "VIP" in i["joint_level"] and "Direct" in i["joint_level"])
    assert vip_direct["interaction_type"] == "synergy"
    assert vip_direct["synergy_diff"] > 0.05

    findings = tool.findings(res, None, None)
    synergy_findings = [f for f in findings if f.kind == "interaction_synergy"]
    assert len(synergy_findings) >= 1
    assert any("VIP" in f.headline and "Direct" in f.headline for f in synergy_findings)

