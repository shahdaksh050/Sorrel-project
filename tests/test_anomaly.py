"""Unit tests for AnomalyAnalysisTool (src/tools/anomaly.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tools.anomaly import AnomalyAnalysisTool


def _run(tmp_path: Path, df: pd.DataFrame, **params: Any) -> dict[str, Any]:
    path = tmp_path / "anomaly.csv"
    df.to_csv(path, index=False)
    result = AnomalyAnalysisTool().run(file_path=str(path), **params)
    assert result.status == "success", result.error_message
    return result.output


def _daily(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    days = pd.date_range("2025-01-01", periods=n, freq="D")
    weekly = 10.0 * np.sin(2 * np.pi * np.arange(n) / 7)
    return pd.DataFrame({"date": days, "sales": 200.0 + weekly + rng.normal(0, 3.0, n)})


def test_spike_is_reported_with_date_and_magnitude(tmp_path: Path) -> None:
    df = _daily()
    df.loc[100, "sales"] += 120.0
    out = _run(tmp_path, df, date_column="date", value_column="sales")
    top = out["spikes"][0]
    stamp = str(df.loc[100, "date"].date())
    assert top["period"] == stamp
    assert top["direction"] == "spike" and top["deviation"] > 80
    assert len(out["spikes"]) <= 5
    findings = AnomalyAnalysisTool().findings(out, None, None)
    assert any(f.kind == "outlier_scan" and stamp in f.headline for f in findings)


def test_level_shift_is_located(tmp_path: Path) -> None:
    df = _daily()
    df.loc[150:, "sales"] += 60.0
    out = _run(tmp_path, df, date_column="date", value_column="sales")
    assert out["level_shifts"], out["summary"]
    shift = out["level_shifts"][0]
    assert abs((pd.Timestamp(shift["period"]) - df.loc[150, "date"]).days) <= 7
    assert shift["delta"] > 40


def test_smooth_trend_is_not_a_level_shift(tmp_path: Path) -> None:
    df = _daily()
    df["sales"] = df["sales"] + np.linspace(0, 80, len(df))
    out = _run(tmp_path, df, date_column="date", value_column="sales")
    assert out["level_shifts"] == []


def test_unusual_segment_is_found_without_dates(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    regions = [f"R{i}" for i in range(8)]
    df = pd.DataFrame({
        "region": np.repeat(regions, 60),
        "amount": np.concatenate([rng.normal(100.0 + (60.0 if r == "R3" else 0.0), 8.0, 60) for r in regions]),
    })
    out = _run(tmp_path, df, value_column="amount", group_column="region")
    assert out["segments"] and out["segments"][0]["level"] == "R3"
    assert out["segments"][0]["deviation"] > 0
    assert not out["spikes"]


def test_errors_without_anything_to_scan(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": range(30), "b": np.arange(30) * 1.5})
    path = tmp_path / "flat.csv"
    df.to_csv(path, index=False)
    assert AnomalyAnalysisTool().run(file_path=str(path)).status == "error"
