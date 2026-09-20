"""Tests for MixedModelAnalysisTool (src/tools/mixed_model.py)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.profiler import profile_dataframe
from src.tools.mixed_model import MixedModelAnalysisTool


def _grouped(group_sd: float, n_groups: int = 40, size: int = 12, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    site = np.repeat([f"S{i:03d}" for i in range(n_groups)], size)
    effect = np.repeat(rng.normal(0, group_sd, n_groups), size)
    x = rng.normal(0, 1, n_groups * size)
    y = 2.0 * x + effect + rng.normal(0, 1, n_groups * size)
    return pd.DataFrame({"site": site, "x": x, "y": y})


def _csv(tmp_path: Path, df: pd.DataFrame, name: str = "d.csv") -> str:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def _run(tmp_path: Path, df: pd.DataFrame, **params: Any) -> dict[str, Any]:
    result = MixedModelAnalysisTool().run(file_path=_csv(tmp_path, df), **params)
    assert result.status == "success", result.error_message
    return result.output


def test_recovers_fixed_effect_and_high_icc(tmp_path: Path) -> None:
    out = _run(tmp_path, _grouped(3.0), target_column="y", group_column="site", feature_columns=["x"])
    fx = next(e for e in out["fixed_effects"] if e["feature"] == "x")
    assert abs(fx["per_unit"] - 2.0) < 0.3
    assert fx["per_unit_low"] < 2.0 < fx["per_unit_high"]
    assert abs(out["icc"] - 0.9) < 0.08
    assert out["design_effect_note"] and out["design_effect"] > 5
    findings = MixedModelAnalysisTool().findings(out, None, None)
    assert findings and all(f.kind == "mixed_model" for f in findings)


def test_recovers_low_icc(tmp_path: Path) -> None:
    out = _run(tmp_path, _grouped(np.sqrt(0.3 / 0.7), seed=11), target_column="y")
    assert out["group_column"] == "site" and out["target_column"] == "y"
    assert abs(out["icc"] - 0.3) < 0.08
    assert out["method"].startswith("mixed")


def test_binary_target_refused(tmp_path: Path) -> None:
    df = _grouped(1.0)
    df["y"] = (df["y"] > 0).astype(int)
    result = MixedModelAnalysisTool().run(file_path=_csv(tmp_path, df), target_column="y", group_column="site")
    assert result.status != "success"
    assert "yes/no" in result.error_message and "regression_analysis" in result.error_message


def test_unique_group_per_row_declined(tmp_path: Path) -> None:
    df = _grouped(1.0)
    df["site"] = [f"row{i}" for i in range(len(df))]
    path = _csv(tmp_path, df)
    for params in ({}, {"group_column": "site"}):
        result = MixedModelAnalysisTool().run(file_path=path, target_column="y", **params)
        assert result.status != "success"
        assert result.error_message


def test_applies_to_negative_and_positive() -> None:
    tool = MixedModelAnalysisTool()
    rng = np.random.default_rng(3)
    n = 300
    survey = pd.DataFrame({
        "respondent_id": np.arange(n),
        "region": rng.choice(["North", "South", "East", "West", "Central"], n),
        "age": rng.integers(18, 80, n),
        "income": rng.normal(50_000, 9_000, n),
        "satisfaction": rng.normal(6, 1.5, n),
    })
    assert tool.applies_to(profile_dataframe(survey), None) == 0.0
    sales = pd.DataFrame({
        "order_id": np.arange(n), "product": rng.choice(list("ABCDEFGH"), n),
        "units": rng.integers(1, 20, n), "revenue": rng.normal(400, 90, n),
    })
    assert tool.applies_to(profile_dataframe(sales), None) == 0.0
    assert tool.applies_to(profile_dataframe(_grouped(3.0)), None) > 0.0


def test_large_frame_is_fast_and_threads_agree(tmp_path: Path) -> None:
    big = _grouped(2.0, n_groups=500, size=100, seed=5)
    path = _csv(tmp_path, big, "big.csv")
    tool = MixedModelAnalysisTool()
    start = time.perf_counter()
    first = tool.run(file_path=path, target_column="y", group_column="site")
    assert first.status == "success", first.error_message
    assert time.perf_counter() - start < 8.0
    assert first.output["n_obs"] == 50_000

    small = _csv(tmp_path, _grouped(3.0), "small.csv")
    with ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(lambda _: tool.run(file_path=small, target_column="y", group_column="site").output, range(4)))
    assert all(o["summary"] == outs[0]["summary"] and o["fixed_effects"] == outs[0]["fixed_effects"] for o in outs)
