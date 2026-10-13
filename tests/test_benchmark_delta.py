from __future__ import annotations

import pandas as pd
import pytest

from src.core.agenda import build_agenda
from src.core.profiler import profile_dataframe
from src.tools.structured_features import StructuredFeatureExtractionTool, _find_benchmark_pairs


def test_find_benchmark_pairs() -> None:
    # 1. Prefix pairs
    cols1 = ["actual_revenue", "target_revenue", "region"]
    pairs1 = _find_benchmark_pairs(cols1)
    assert len(pairs1) == 1
    assert pairs1[0] == ("actual_revenue", "target_revenue", "revenue")

    # 2. Suffix pairs
    cols2 = ["sales_actual", "sales_budget", "department"]
    pairs2 = _find_benchmark_pairs(cols2)
    assert len(pairs2) == 1
    assert pairs2[0] == ("sales_actual", "sales_budget", "sales")

    # 3. Domain equivalents (e.g. football, logistics, saas)
    cols3 = ["home_goals", "home_xgoals", "away_goals", "away_xgoals"]
    pairs3 = _find_benchmark_pairs(cols3)
    assert len(pairs3) == 2
    assert ("home_goals", "home_xgoals", "home_goals") in pairs3
    assert ("away_goals", "away_xgoals", "away_goals") in pairs3

    cols4 = ["duration", "eta", "driver_id"]
    pairs4 = _find_benchmark_pairs(cols4)
    assert len(pairs4) == 1
    assert pairs4[0] == ("duration", "eta", "duration")


def test_benchmark_delta_extraction(tmp_path: pytest.TempPathFactory) -> None:
    csv_path = tmp_path / "benchmark_test.csv"  # type: ignore[operator]
    df = pd.DataFrame({
        "actual_sales": [100.0, 150.0, 200.0, 80.0, 120.0] * 5,
        "target_sales": [90.0, 140.0, 220.0, 100.0, 110.0] * 5,
        "region": ["North", "South", "East", "West", "Central"] * 5,
    })
    df.to_csv(csv_path, index=False)

    tool = StructuredFeatureExtractionTool()
    assert tool.applies_to(profile_dataframe(df), None) > 0.0

    output = tool.execute(file_path=str(csv_path))
    derived_path = output["derived_dataset"]["path"]
    derived_df = pd.read_csv(derived_path)

    # Derived dataset must have delta and ratio columns
    assert "sales_delta" in derived_df.columns
    assert "sales_ratio" in derived_df.columns
    # Check row 0: actual 100 - target 90 = +10
    assert derived_df["sales_delta"].iloc[0] == 10.0
    assert round(derived_df["sales_ratio"].iloc[0], 2) == round(100.0 / 90.0, 2)


def test_agenda_prioritizes_benchmark_delta() -> None:
    df = pd.DataFrame({
        "sales_delta": [10.0, -10.0, 20.0, -20.0] * 10,
        "raw_cost": [50.0, 60.0, 70.0, 80.0] * 10,
        "region": ["North", "South", "East", "West"] * 10,
    })
    profile = profile_dataframe(df)
    agenda = build_agenda(profile, decision=None)
    segment_questions = [q for q in agenda if q.kind == "segment"]
    assert len(segment_questions) > 0
    # Top measure should be the performance delta
    assert segment_questions[0].columns[0] == "sales_delta"
    assert "performance delta" in segment_questions[0].text
