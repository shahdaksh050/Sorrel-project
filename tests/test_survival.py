"""Tests for SurvivalAnalysisTool (src/tools/survival.py) against planted truth."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.agenda import _specialist_questions
from src.core.profiler import profile_dataframe
from src.tools.survival import SurvivalAnalysisTool

RATE_A, RATE_B, RATE_CENSOR = 0.10, 0.20, 0.06  # B leaves 2x faster; ~30% censored overall


def _lifetimes(n_a: int = 700, n_b: int = 500, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plan = np.array(["A"] * n_a + ["B"] * n_b)
    rate = np.where(plan == "A", RATE_A, RATE_B)
    life = rng.exponential(1.0 / rate)
    cens = rng.exponential(1.0 / RATE_CENSOR, len(plan))
    return pd.DataFrame({
        "plan": plan,
        "tenure_months": np.minimum(life, cens) + 1e-6,
        "churned": (life <= cens).astype(int),
    })


def _csv(tmp_path: Path, df: pd.DataFrame) -> Path:
    path = tmp_path / "surv.csv"
    df.to_csv(path, index=False)
    return path


def _run(tmp_path: Path, df: pd.DataFrame, **params: Any) -> dict[str, Any]:
    result = SurvivalAnalysisTool().run(file_path=str(_csv(tmp_path, df)), **params)
    assert result.status == "success", result.error_message
    return result.output


def test_planted_hazard_ratio_is_recovered(tmp_path: Path) -> None:
    df = _lifetimes()
    assert 0.2 < 1 - df["churned"].mean() < 0.4
    out = _run(tmp_path, df, group_column="plan")
    med = {g["group"]: g["median_survival"]["median"] for g in out["groups"]}
    assert med["A"] == pytest.approx(np.log(2) / RATE_A, rel=0.15)
    assert med["B"] == pytest.approx(np.log(2) / RATE_B, rel=0.15)
    assert out["log_rank"]["p_value"] < 0.01
    hr = next(t for t in out["cox"]["hazard_ratios"] if t["label"] == "B")
    assert hr["hr"] == pytest.approx(2.0, rel=0.25)
    assert hr["ci_lower"] < 2.0 < hr["ci_upper"]
    assert "faster" in hr["sentence"]

    found = SurvivalAnalysisTool().findings(out, None, None)
    assert found and all(f.kind == "survival" for f in found)
    assert "Half" in found[0].headline and "months" in found[0].headline
    assert any("faster" in f.headline for f in found)
    chart = out["chart"]
    assert len(chart["data"]) > 0 and {"group", "time", "survival"} <= set(chart["data"][0])


def test_yes_no_event_column(tmp_path: Path) -> None:
    df = _lifetimes()
    base = _run(tmp_path, df, group_column="plan")
    words = df.assign(churned=np.where(df["churned"] == 1, "yes", "no"))
    out = _run(tmp_path, words, group_column="plan")
    assert out["n_events"] == base["n_events"]
    assert out["median_survival"]["median"] == base["median_survival"]["median"]


def test_censored_column_is_read_inverted(tmp_path: Path) -> None:
    df = _lifetimes()
    base = _run(tmp_path, df, group_column="plan")
    inv = df.rename(columns={"churned": "censored"}).assign(censored=1 - df["churned"])
    out = _run(tmp_path, inv, group_column="plan", event_column="censored")
    assert out["n_events"] == base["n_events"]
    assert "censoring" in out["event_coding"]
    assert out["median_survival"]["median"] == base["median_survival"]["median"]


def test_too_few_events_is_caveated_or_errors(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"tenure_months": rng.uniform(1, 24, 60), "churned": [1] * 5 + [0] * 55})
    out = _run(tmp_path, df)
    assert any("Only 5 events" in c for c in out["caveats"])
    result = SurvivalAnalysisTool().run(file_path=str(_csv(tmp_path, df.assign(churned=0))), event_column="churned")
    assert result.status != "success" and "no events" in (result.error_message or "")


def test_applies_to_is_zero_for_unrelated_frames() -> None:
    rng = np.random.default_rng(1)
    n = 200
    sales = pd.DataFrame({
        "region": rng.choice(["N", "S", "E"], n), "product": rng.choice(["p1", "p2", "p3"], n),
        "amount": rng.gamma(2, 50, n), "date": pd.date_range("2025-01-01", periods=n),
    })
    survey = pd.DataFrame({f"q{i}_rating": rng.integers(1, 6, n) for i in range(5)})
    tool = SurvivalAnalysisTool()
    assert tool.applies_to(profile_dataframe(sales), None) == 0.0
    assert tool.applies_to(profile_dataframe(survey), None) == 0.0
    assert tool.applies_to(profile_dataframe(_lifetimes()), None) > 0.0


def test_default_params_and_agenda_include_group_column() -> None:
    profile = profile_dataframe(_lifetimes())
    params = SurvivalAnalysisTool().default_params(profile, None)
    assert params == {"duration_column": "tenure_months", "event_column": "churned", "group_column": "plan"}
    qs = [q for q in _specialist_questions(profile) if q.kind == "survival"]
    assert qs and qs[0].columns == ["tenure_months", "churned", "plan"]
    ungrouped = profile_dataframe(_lifetimes().drop(columns="plan"))
    assert "group_column" not in SurvivalAnalysisTool().default_params(ungrouped, None)


def test_large_frame_is_fast(tmp_path: Path) -> None:
    path = _csv(tmp_path, _lifetimes(30_000, 20_000))
    start = time.perf_counter()
    result = SurvivalAnalysisTool().run(file_path=str(path), group_column="plan")
    assert result.status == "success", result.error_message
    assert time.perf_counter() - start < 6.0


def test_concurrent_runs_are_identical(tmp_path: Path) -> None:
    path = _csv(tmp_path, _lifetimes())

    def go(_: int) -> str:
        res = SurvivalAnalysisTool().run(file_path=str(path), group_column="plan")
        assert res.status == "success"
        return json.dumps(res.output, sort_keys=True, default=str)

    with ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(go, range(4)))
    assert len(set(outs)) == 1
