from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tools.multi_seasonal_analysis import MultiSeasonalAnalysisTool


def test_multi_seasonal_detects_fft_harmonics(tmp_path: Path) -> None:
    # 240 hourly steps = 10 days
    t = np.arange(240)
    # 24-hour cycle + 12-hour sub-harmonic + noise
    daily_cycle = 15.0 * np.sin(2 * np.pi * t / 24.0)
    sub_cycle = 7.0 * np.cos(2 * np.pi * t / 12.0)
    noise = np.random.default_rng(42).normal(0, 1.0, 240)
    grid_load = 100.0 + daily_cycle + sub_cycle + noise

    timestamps = pd.date_range("2026-01-01", periods=240, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "grid_load_mw": grid_load})
    csv_path = tmp_path / "grid_telemetry.csv"
    df.to_csv(csv_path, index=False)

    tool = MultiSeasonalAnalysisTool()
    res = tool.run(file_path=str(csv_path), time_column="timestamp", measure_column="grid_load_mw")
    assert res.status == "success"
    out: dict[str, Any] = res.output

    assert "dominant_cycles" in out
    assert len(out["dominant_cycles"]) >= 1
    # Top period should be around 24 hours
    top_period = out["dominant_cycles"][0]["period_steps"]
    assert 22.0 <= top_period <= 26.0

    assert out["periodic_variance_explained_pct"] > 50.0

    findings = tool.findings(out)
    assert len(findings) >= 1
    f = findings[0]
    assert f.kind == "periodicity"
    assert f.chart_hint == "line"
    assert "periodic" in f.headline.lower()
