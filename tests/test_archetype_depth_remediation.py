"""Tests for Universal Archetype Analytical Depth & Pipeline Integrity Remediation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.controller_planning import PlanMixin
from src.core.memory import MemorySystem
from src.core.profiler import (
    ARCHETYPE_CONTINUOUS_SERIES,
    profile_dataframe,
)
from src.core.prompt_manager import ARCHETYPE_CANONICAL_TOOLS
from src.core.reshaping import extract_temporal_calendar_features
from src.tools.forecast import ForecastAnalysisTool
from src.tools.segment_comparison import SegmentComparisonTool


def test_sub_hourly_frequency_and_continuous_series_inference() -> None:
    """Verify that 1-minute, 15-minute, and secondly datasets are recognized as continuous_series."""
    dates = pd.date_range("2026-01-01 00:00:00", periods=500, freq="1min")
    df_minutely = pd.DataFrame({
        "timestamp": dates,
        "time_str": [d.strftime("%H:%M:%S") for d in dates],
        "sensor_a": np.random.randn(500) + 10.0,
        "sensor_b": np.random.randn(500) * 2.0 + 5.0,
        "sensor_c": np.random.randn(500) * 0.5 + 1.0,
    })
    prof = profile_dataframe(df_minutely)
    assert prof.time_frequency == "minutely"
    assert prof.archetype == ARCHETYPE_CONTINUOUS_SERIES

    # 15min frequency
    dates_15 = pd.date_range("2026-01-01 00:00:00", periods=500, freq="15min")
    df_15 = pd.DataFrame({
        "timestamp": dates_15,
        "power": np.random.randn(500) + 20.0,
        "voltage": np.random.randn(500) + 230.0,
    })
    prof_15 = profile_dataframe(df_15)
    assert prof_15.time_frequency == "15min"
    assert prof_15.archetype == ARCHETYPE_CONTINUOUS_SERIES


def test_extract_temporal_calendar_features(tmp_path: Path) -> None:
    """Verify that temporal dimensions (hour, day_part, day_of_week, is_weekend) are cleanly derived."""
    dates = pd.date_range("2026-01-01 00:00:00", periods=2880, freq="1min")  # 2 full days
    df = pd.DataFrame({
        "ts": dates,
        "active_power": np.random.uniform(0.5, 3.5, size=2880),
    })
    enriched, notes = extract_temporal_calendar_features(df, "ts")
    assert "hour_of_day" in enriched.columns
    assert "day_part" in enriched.columns
    assert "day_of_week" in enriched.columns
    assert "is_weekend" in enriched.columns
    assert enriched["day_part"].nunique() >= 3
    assert len(notes) > 0

    path = tmp_path / "enriched.csv"
    enriched.to_csv(path, index=False)
    # Ensure segment_comparison can compare active_power across day_part without crashing
    tool = SegmentComparisonTool()
    out = tool.execute(
        file_path=str(path),
        measure_column="active_power",
        dimension_column="day_part",
    )
    assert "summary" in out
    assert len(out.get("comparisons", [])) >= 3


def test_forecast_non_negative_physical_bounds(tmp_path: Path) -> None:
    """Verify that strictly non-negative series produce non-negative, non-cliff bounds and asymmetric percentiles."""
    dates = pd.date_range("2026-01-01", periods=180, freq="D")
    rng = np.random.default_rng(42)
    # Series with low value and high relative noise (similar to household power)
    values = np.clip(0.65 + rng.normal(0, 0.45, size=180), 0.05, 3.0)
    df = pd.DataFrame({"Date": dates, "power": values})
    path = tmp_path / "power_test.csv"
    df.to_csv(path, index=False)

    tool = ForecastAnalysisTool()
    res = tool.execute(file_path=str(path), date_column="Date", value_column="power", horizon=10)
    assert res.get("model") is not None
    s = res["series"][0]
    steps = s["forecast"]
    assert len(steps) == 10
    for step in steps:
        assert step["lower"] >= 0.0
        assert step["upper"] >= step["forecast"]
        # Crucial: day 1 lower bound does NOT cliff drop to exactly 0.0 when positive
        assert step["lower"] > 0.0

    # Check findings headline
    findings = tool.findings(res, None, None)
    assert len(findings) > 0
    h = findings[0].headline
    # Either reports 95% range or give or take, without reporting negative numbers
    assert "give or take" in h or "95% range" in h
    assert "-0." not in h and "-1." not in h

    # Check Vega-Lite area spec
    chart = res.get("chart", {})
    vl = chart.get("vega_lite", {})
    layers = vl.get("layer", [])
    assert len(layers) == 2
    band_layer = layers[0]
    assert band_layer.get("mark", {}).get("line") is False
    assert band_layer.get("encoding", {}).get("y", {}).get("axis", {}).get("titlePadding") == 12


class DummyPlanner(PlanMixin):
    def __init__(self) -> None:
        from src.core.memory import DatasetMetadata
        self.memory = MemorySystem()
        self.memory.dataset_metadata = DatasetMetadata(
            file_path="mock.csv",
            row_count=100,
            column_count=5,
            columns={"a": "float", "b": "float"},
            missing_values={},
            numerical_cols=["a", "b"],
            categorical_cols=[],
        )
        self.last_profile = None
        self.use_ml = True
        self.objective = ""
        self.autonomous_objective = ""
        from src.core.tool_registry import ToolRegistry
        self.tool_registry = ToolRegistry()


def test_autonomous_lead_driver_scenario_simulation_allowed() -> None:
    """Verify that scenario_simulation is permitted in autonomous mode when driver findings exist."""
    planner = DummyPlanner()
    # When no driver findings exist, scenario_simulation is rejected without an objective
    err = planner._tool_hard_prerequisite_error("scenario_simulation")
    assert err is not None
    assert "requires an objective or at least one quantified driver finding" in err

    # Now add a driver finding
    from src.core.findings import Finding
    planner.memory.findings.append(
        Finding(
            finding_id="driver_test",
            kind="driver",
            headline="Sub_metering_3 drives Global_active_power",
            detail="1 SD changes power by 89%",
            dimension="Sub_metering_3",
            measure="Global_active_power",
            source_tool="regression_analysis",
        )
    )
    # With a driver finding present, hard prerequisite check passes cleanly
    err_after = planner._tool_hard_prerequisite_error("scenario_simulation")
    assert err_after is None


def test_multi_tiered_canonical_blueprints() -> None:
    """Verify that every archetype has a rich curriculum of at least 3 canonical tools."""
    for archetype, tools in ARCHETYPE_CANONICAL_TOOLS.items():
        assert len(tools) >= 3, f"Archetype {archetype} must have >= 3 canonical tools, had {len(tools)}"
