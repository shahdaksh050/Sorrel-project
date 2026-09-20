"""Tests for CurveFitAnalysisTool (src/tools/curve_fit.py) against planted truth."""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.profiler import profile_dataframe
from src.tools.curve_fit import CurveFitAnalysisTool


def _csv(tmp_path: Path, df: pd.DataFrame) -> Path:
    path = tmp_path / "fit.csv"
    df.to_csv(path, index=False)
    return path


def _run(tmp_path: Path, df: pd.DataFrame, **params: Any) -> dict[str, Any]:
    result = CurveFitAnalysisTool().run(file_path=str(_csv(tmp_path, df)), **params)
    assert result.status == "success", result.error_message
    return result.output


def _decay(n: int = 120, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 15, n)
    return pd.DataFrame({"time_days": x, "signal": 5 * np.exp(-0.3 * x) + 1 + rng.normal(0, 0.1, n)})


def _dose(seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dose = np.repeat(np.geomspace(0.01, 100, 14), 4)
    ec50, hill = 2.0, 1.2
    resp = 5 + 95 * dose**hill / (ec50**hill + dose**hill) + rng.normal(0, 2.0, len(dose))
    return pd.DataFrame({"dose": dose, "response": resp})


def _derived(out: dict[str, Any], name: str) -> dict[str, Any]:
    return next(d for d in out["derived"] if d["name"] == name)


def test_decay_half_life_recovered(tmp_path: Path) -> None:
    out = _run(tmp_path, _decay(), x_column="time_days", y_column="signal")
    assert out["model"] in ("decay", "exponential")
    assert _derived(out, "half_life")["value"] == pytest.approx(math.log(2) / 0.3, rel=0.10)
    assert out["r2"] > 0.95
    found = CurveFitAnalysisTool().findings(out, None, None)
    assert found and found[0].kind == "curve_fit" and "half" in found[0].headline
    assert len(out["chart"]["data"]) > 0


def test_columns_are_auto_detected(tmp_path: Path) -> None:
    out = _run(tmp_path, _decay())
    assert out["x_column"] == "time_days" and out["y_column"] == "signal"


def test_hill_ec50_recovered(tmp_path: Path) -> None:
    out = _run(tmp_path, _dose(), x_column="dose", y_column="response")
    assert out["model"] == "hill"
    assert _derived(out, "ec50")["value"] == pytest.approx(2.0, rel=0.15)
    assert out["r2"] > 0.95


def test_pure_noise_says_no_clear_curve(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"time_days": rng.uniform(0, 10, 100), "signal": rng.normal(0, 1, 100)})
    out = _run(tmp_path, df, x_column="time_days", y_column="signal")
    assert out["r2"] < 0.5 and out["low_fit"] is True
    assert any("No clear curve" in c for c in out["caveats"])
    found = CurveFitAnalysisTool().findings(out, None, None)
    assert found and found[0].headline.startswith("No clear curve")


def test_applies_to_is_zero_for_unrelated_frames() -> None:
    rng = np.random.default_rng(1)
    n = 200
    sales = pd.DataFrame({
        "region": rng.choice(["N", "S", "E"], n), "product": rng.choice(["p1", "p2", "p3"], n),
        "amount": rng.gamma(2, 50, n), "date": pd.date_range("2025-01-01", periods=n),
    })
    survey = pd.DataFrame({f"q{i}_rating": rng.integers(1, 6, n) for i in range(5)})
    tool = CurveFitAnalysisTool()
    assert tool.applies_to(profile_dataframe(sales), None) == 0.0
    assert tool.applies_to(profile_dataframe(survey), None) == 0.0
    assert tool.applies_to(profile_dataframe(_decay()), None) > 0.0


def test_large_frame_is_fast(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 15, 50_000)
    df = pd.DataFrame({"time_days": x, "signal": 5 * np.exp(-0.3 * x) + 1 + rng.normal(0, 0.1, len(x))})
    path = _csv(tmp_path, df)
    start = time.perf_counter()
    result = CurveFitAnalysisTool().run(file_path=str(path), x_column="time_days", y_column="signal")
    assert result.status == "success", result.error_message
    assert time.perf_counter() - start < 6.0


def test_concurrent_runs_are_identical(tmp_path: Path) -> None:
    path = _csv(tmp_path, _dose())

    def go(_: int) -> str:
        res = CurveFitAnalysisTool().run(file_path=str(path), x_column="dose", y_column="response")
        assert res.status == "success"
        return json.dumps(res.output, sort_keys=True, default=str)

    with ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(go, range(4)))
    assert len(set(outs)) == 1
