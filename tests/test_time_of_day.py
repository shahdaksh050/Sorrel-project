"""Tests for the time-of-day / day-of-week profile in time_series_analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tools.time_series import TimeSeriesAnalysisTool


def _sensor(days: int = 60, seed: int = 3, cycle: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-03-01", periods=days * 24, freq="h")
    hour = ts.hour.to_numpy()
    if cycle:
        # trough 2.0 at 07:00, peak 8.0 at 19:00 (ratio 4), weekends 30% lower
        base = 5.0 + 3.0 * np.cos(2 * np.pi * (hour - 19) / 24)
        base = base * np.where(ts.dayofweek >= 5, 0.7, 1.0)
        co = base + rng.normal(0, 0.4, len(ts))
        noox = 100 + 40 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 10, len(ts))
    else:
        co = 5.0 + rng.normal(0, 1.0, len(ts))
        noox = 100 + rng.normal(0, 10, len(ts))
    return pd.DataFrame({"Date": ts, "CO": co, "NOx": noox, "flat": rng.normal(50, 5, len(ts))})


def _run(tmp_path: Path, df: pd.DataFrame) -> dict[str, Any]:
    path = tmp_path / "s.csv"
    df.to_csv(path, index=False)
    result = TimeSeriesAnalysisTool().run(file_path=str(path), date_column="Date", value_column="CO")
    assert result.status == "success", result.error_message
    return result.output


def test_daily_cycle_and_weekend_dip(tmp_path: Path) -> None:
    out = _run(tmp_path, _sensor())
    d = out["diurnal"]["CO"]
    assert d["is_primary"] and d["peak_hour"] == 19 and d["trough_hour"] == 7
    assert abs(d["peak_trough_ratio"] - 4.0) / 4.0 < 0.15
    assert d["eta_squared"] >= 0.5 and len(d["hourly_means"]) == 24
    w = out["weekly_profile"]["CO"]
    assert w["weekend_gap"] < 0 and abs(w["weekend_gap"] + 0.3) < 0.05
    assert len(w["means"]) == 7
    assert out["charts"] and "average by hour of day" in out["charts"][0]["title"]
    # a second measure with its own hour effect is picked up; the flat one is not
    assert "NOx" in out["diurnal"] and "flat" not in out["diurnal"]

    findings = TimeSeriesAnalysisTool().findings(out, None, None)
    diurnal = [f for f in findings if f.dimension == "hour_of_day"]
    co = next(f for f in diurnal if f.measure == "CO")
    assert co.kind == "trend" and co.effect_kind == "pct" and co.effect > 1.0
    assert "19:00" in co.headline and "07:00" in co.headline and "weekends run 31% lower" in co.headline
    assert co.evidence["eta_squared"] == d["eta_squared"]
    assert sum(f.layer == "exec" for f in diurnal) == 1


def test_no_cycle_no_diurnal_finding(tmp_path: Path) -> None:
    out = _run(tmp_path, _sensor(days=14, cycle=False))
    assert out["diurnal"] is None and out["weekly_profile"] is None
    assert not [f for f in TimeSeriesAnalysisTool().findings(out, None, None) if f.dimension == "hour_of_day"]


def test_daily_data_skipped(tmp_path: Path) -> None:
    df = _sensor(days=200)
    out = _run(tmp_path, df.iloc[::24].reset_index(drop=True))
    assert out["diurnal"] is None and "charts" not in out
