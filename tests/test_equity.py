"""Tests for EquityAnalysisTool (src/tools/equity.py)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.privacy import min_cell_size
from src.core.profiler import profile_dataframe
from src.tools.equity import EquityAnalysisTool


def _pay_frame(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["F", "M"], n, p=[0.4, 0.6])  # reference = largest group (M)
    level = np.where(
        gender == "F",
        rng.choice([1, 2, 3, 4], n, p=[0.4, 0.3, 0.2, 0.1]),
        rng.choice([1, 2, 3, 4], n, p=[0.1, 0.2, 0.3, 0.4]),
    )
    tenure = rng.uniform(0, 15, n)
    log_pay = (
        np.log(50000) + 0.20 * level + 0.01 * tenure
        + np.where(gender == "F", np.log(0.94), 0.0) + rng.normal(0, 0.03, n)
    )
    return pd.DataFrame({"gender": gender, "level": level, "tenure": tenure, "pay": np.exp(log_pay)})


def _run(tmp_path: Path, df: pd.DataFrame, **kw: Any) -> tuple[EquityAnalysisTool, dict[str, Any]]:
    path = tmp_path / "q.csv"
    df.to_csv(path, index=False)
    tool = EquityAnalysisTool()
    res = tool.run(file_path=str(path), **kw)
    assert res.status == "success", res.error_message
    return tool, res.output


def test_numeric_adjusted_gap(tmp_path: Path) -> None:
    tool, out = _run(tmp_path, _pay_frame(), outcome_column="pay", group_column="gender")
    row = out["groups"][0]
    assert out["reference_group"] == "M" and row["group"] == "F"
    assert abs(row["adjusted_gap_pct"] - (-6.0)) < 2.0
    assert row["ci_lower_pct"] <= -6.0 <= row["ci_upper_pct"]
    assert row["unadjusted_mean_gap_pct"] < row["adjusted_gap_pct"] < 0  # raw gap larger in magnitude
    assert set(out["control_columns"]) == {"level", "tenure"}
    text = " ".join(out["caveats"]).lower()
    assert "not proof of discrimination" in text and "not measured" in text
    f = tool.findings(out, None, None)
    assert f and f[0].kind == "equity" and any("proof of discrimination" in c for c in f[0].caveats)


def test_binary_four_fifths_and_small_cell_suppression(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 2000
    df = pd.DataFrame({
        "gender": ["M"] * n + ["F"] * n,
        "promoted": np.concatenate([rng.binomial(1, 0.30, n), rng.binomial(1, 0.18, n)]),
    })
    tiny = pd.DataFrame({"gender": ["X"] * 3, "promoted": [1, 1, 1]})
    tool, out = _run(tmp_path, pd.concat([df, tiny], ignore_index=True), outcome_column="promoted", group_column="gender")
    by = {r["group"]: r for r in out["groups"]}
    assert "X" not in by and out["suppressed_groups"] == 1 and 3 < min_cell_size()
    assert out["reference_group"] == "M"
    assert abs(by["F"]["air"] - 0.6) < 0.12 and by["F"]["flag_four_fifths"] is True
    f = tool.findings(out, None, None)
    assert f and "four-fifths" in f[0].headline and f[0].evidence["group"] == "F"
    blob = str(out) + str([(x.headline, x.detail, x.evidence, x.level) for x in f])
    assert "'X'" not in blob
    assert "fewer than" in " ".join(out["caveats"])


def test_applies_to() -> None:
    tool = EquityAnalysisTool()
    no_protected = pd.DataFrame({
        "pay": np.random.default_rng(0).normal(5e4, 5e3, 300), "level": [1, 2, 3] * 100, "team": ["a", "b"] * 150,
    })
    assert tool.applies_to(profile_dataframe(no_protected), None) == 0.0
    unrelated = pd.DataFrame({"x": np.arange(300.0), "y": np.arange(300.0) ** 2, "region": ["n", "s"] * 150})
    assert tool.applies_to(profile_dataframe(unrelated), None) == 0.0
    assert tool.applies_to(profile_dataframe(_pay_frame()), None) > 0


def test_perf_and_thread_safety(tmp_path: Path) -> None:
    df = _pay_frame(n=50000)
    path = tmp_path / "big.csv"
    df.to_csv(path, index=False)
    tool = EquityAnalysisTool()
    kw: dict[str, Any] = {"file_path": str(path), "outcome_column": "pay", "group_column": "gender"}
    t0 = time.perf_counter()
    first = tool.run(**kw)
    assert time.perf_counter() - t0 < 6 and first.status == "success"
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: tool.run(**kw).output, range(4)))
    assert all(o == first.output for o in outs)
