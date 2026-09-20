"""Tests for ForecastAnalysisTool (src/tools/forecast.py)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.profiler import profile_dataframe
from src.tools.forecast import ForecastAnalysisTool


def _monthly(n: int = 60, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n + 12)
    truth = 200.0 + 3.0 * t + 25.0 * np.sin(2 * np.pi * t / 12)
    dates = pd.date_range("2019-01-01", periods=n, freq="MS")
    return pd.DataFrame({"month": dates, "revenue": truth[:n] + rng.normal(0, 4.0, n)}), truth[n:]


def _run(tmp_path: Path, df: pd.DataFrame, **params: Any) -> dict[str, Any]:
    path = tmp_path / "f.csv"
    df.to_csv(path, index=False)
    result = ForecastAnalysisTool().run(file_path=str(path), **params)
    assert result.status == "success", result.error_message
    return result.output


def test_trend_and_seasonality_forecast(tmp_path: Path) -> None:
    df, truth = _monthly()
    out = _run(tmp_path, df, date_column="month", value_column="revenue", horizon=12)
    s = out["series"][0]
    assert s["mase"] < 1 and not s["naive_no_better"]
    steps = s["forecast"]
    assert len(steps) == 12
    fc = np.array([x["forecast"] for x in steps])
    lo = np.array([x["lower"] for x in steps])
    hi = np.array([x["upper"] for x in steps])
    assert float(np.mean(np.abs(fc - truth) / truth)) < 0.15
    assert float(np.mean((lo <= truth) & (truth <= hi))) >= 0.8
    assert "chart" in out
    findings = ForecastAnalysisTool().findings(out, None, None)
    assert findings[0].kind == "forecast"
    assert "forecast to reach" in findings[0].headline and "give or take" in findings[0].headline


def test_random_walk_carries_naive_caveat(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "month": pd.date_range("2019-01-01", periods=60, freq="MS"),
        "revenue": 500.0 + np.cumsum(rng.normal(0, 10, 60)),
    })
    out = _run(tmp_path, df, date_column="month", value_column="revenue")
    assert any("no better than repeating the last value" in c for c in out["caveats"])


def test_short_history_declines_or_caveats(tmp_path: Path) -> None:
    df, _ = _monthly(18)
    path = tmp_path / "short.csv"
    df.to_csv(path, index=False)
    result = ForecastAnalysisTool().run(file_path=str(path), date_column="month", value_column="revenue")
    if result.status == "success":
        text = " ".join(result.output["caveats"])
        assert "Short history" in text and "two full seasons" in text
    else:
        assert result.error_message


def test_applies_to_negative_and_positive() -> None:
    tool = ForecastAnalysisTool()
    rng = np.random.default_rng(3)
    n = 300
    survey = pd.DataFrame({
        "respondent_id": np.arange(n), "region": rng.choice(["N", "S", "E", "W"], n),
        "age": rng.integers(18, 80, n), "satisfaction": rng.normal(6, 1.5, n),
    })
    assert tool.applies_to(profile_dataframe(survey), None) == 0.0
    two_dates = pd.DataFrame({
        "date": pd.to_datetime(rng.choice(["2024-01-05", "2024-01-06", "2024-01-07"], n)),
        "amount": rng.normal(100, 10, n),
    })
    assert tool.applies_to(profile_dataframe(two_dates), None) == 0.0
    assert tool.applies_to(profile_dataframe(_monthly()[0]), None) > 0.0
    assert tool.requires_ml is False


def test_large_series_is_fast_and_threads_agree(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 5_000
    big = pd.DataFrame({
        "day": pd.date_range("2010-01-01", periods=n, freq="D"),
        "sales": 100 + 0.01 * np.arange(n) + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 2, n),
    })
    path = tmp_path / "big.csv"
    big.to_csv(path, index=False)
    tool = ForecastAnalysisTool()
    start = time.perf_counter()
    res = tool.run(file_path=str(path), date_column="day", value_column="sales")
    assert res.status == "success", res.error_message
    assert time.perf_counter() - start < 8.0

    df, _ = _monthly()
    small = tmp_path / "small.csv"
    df.to_csv(small, index=False)
    with ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(lambda _: tool.run(file_path=str(small), date_column="month", value_column="revenue").output, range(4)))
    assert all(o["series"] == outs[0]["series"] and o["summary"] == outs[0]["summary"] for o in outs)
