"""
Null-data guard (IMPROVEMENTS.md Round 9, 9.14-8): the deterministic
pipeline must not invent insight. Each generator below builds a dataset with
NO real structure; the pipeline is run end to end (`use_llm=False`) and the
findings are checked. Every null generator has a PLANTED CONTROL (the same
generator plus exactly one real effect) that must be found, proving the
"nothing found" assertions are not vacuous.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.controller import AgentController


def run_pipeline(csv_path: Path) -> dict[str, Any]:
    """Deterministic run; artifacts go to a sibling dir of the CSV (tmp)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OUTPUT_DIR", str(Path(csv_path).parent / f"out_{Path(csv_path).stem}"))
        agent = AgentController(use_llm=False, enable_rlm=False)
        agent.load_dataset(str(csv_path), interactive=False)
        return agent.analyze()


# ---------------------------------------------------------------------------
# Generators (fixed seeds)
# ---------------------------------------------------------------------------

def sensor_table(seed: int = 11, n: int = 8000, cycle: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="h")
    data: dict[str, Any] = {"timestamp": ts}
    for i in range(12):
        data[f"sensor_{i + 1:02d}"] = np.round(rng.normal(50.0, 5.0, n), 3)
    if cycle:  # +/-30% around 50, peak at 15:00
        data["sensor_03"] = np.round(
            data["sensor_03"] + 15.0 * np.cos(2 * np.pi * (ts.hour - 15) / 24), 3
        )
    data["site"] = rng.choice(["alpha", "bravo", "charlie", "delta"], n)
    data["status"] = rng.choice(["ok", "warn", "idle"], n)
    return pd.DataFrame(data)


def customer_table(seed: int = 12, n: int = 4000, lift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "region": rng.choice(["North", "South", "East", "West"], n),
        "plan": rng.choice(["basic", "plus", "pro"], n),
        "segment": rng.choice(["retail", "smb", "enterprise"], n),
        "amount": np.round(rng.gamma(9.0, 6.0, n), 2),
    })
    if lift:
        df.loc[df["region"] == "West", "amount"] = np.round(
            df.loc[df["region"] == "West", "amount"] * (1 + lift), 2
        )
    return df


def experiment_table(seed: int = 13, n: int = 3000, effect: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arm = rng.choice(["control", "treatment"], n)
    outcome = rng.normal(100.0, 15.0, n) * np.where(arm == "treatment", 1 + effect, 1.0)
    return pd.DataFrame({
        "user_id": [f"U{i:05d}" for i in range(n)],
        "arm": arm,
        "outcome": np.round(outcome, 3),
    })


@pytest.fixture(scope="module")
def null_runs(tmp_path_factory) -> dict[str, list[dict[str, Any]]]:
    d = tmp_path_factory.mktemp("null")
    out = {}
    for name, df in {
        "sensor": sensor_table(), "customer": customer_table(), "experiment": experiment_table(),
    }.items():
        p = d / f"{name}.csv"
        df.to_csv(p, index=False)
        out[name] = run_pipeline(p)["findings"]
    return out


# ---------------------------------------------------------------------------
# Null assertions
#
# Calibration (5 seeds x 3 generators, deterministic pipeline): the only
# analyst findings on pure noise are the honest no-skill model_performance
# line, method_fit warnings about that model, and (before the no-skill driver
# suppression in findings.rank_findings) spurious "drivers". So the cap is
# MAX_ANALYST = 5 (observed max 4 after the fix); anything that is an actual
# claim about the data (exec layer, correlation, hour/day/month profile,
# change/outlier with importance > 0.5, test/lift with p_adjusted < 0.05) must
# be absent. 0.5 importance is the level where a change/spike would be
# headlined; 0.05 is the standard significance bar after BH correction.
# ---------------------------------------------------------------------------

MAX_ANALYST = 5
_PROFILE_DIMS = {"hour_of_day", "day_of_week", "weekday", "month", "hour"}
_CHANGE_KINDS = {"change", "outlier_scan", "anomaly"}
_TEST_KINDS = {"segment_lift", "test"}


def _assert_null(findings: list[dict[str, Any]]) -> None:
    where = dump(findings)
    assert not [f for f in findings if f.get("layer") == "exec"], where
    assert not [f for f in findings if f["kind"] == "correlation"], where
    assert not [f for f in findings if f.get("dimension") in _PROFILE_DIMS], where
    assert not [
        f for f in findings
        if f["kind"] in _CHANGE_KINDS and (f.get("importance") or 0) > 0.5
    ], where
    assert not [
        f for f in findings
        if f["kind"] in _TEST_KINDS and (f.get("p_adjusted") is not None) and f["p_adjusted"] < 0.05
    ], where
    assert not [f for f in findings if f["kind"] == "driver" and f.get("source_tool") == "evaluate_model"], where
    analyst = [f for f in findings if f.get("layer") == "analyst"]
    assert len(analyst) <= MAX_ANALYST, where


def dump(findings):
    return [(f["kind"], f.get("layer"), f.get("measure"), f.get("dimension"), f.get("level"),
             f.get("effect"), f.get("p_adjusted"), f.get("importance")) for f in findings]


@pytest.mark.parametrize("name", ["sensor", "customer", "experiment"])
def test_null_dataset_yields_no_insight(null_runs, name):
    _assert_null(null_runs[name])


# ---------------------------------------------------------------------------
# Planted controls: same generators + ONE real effect. Must be found.
# ---------------------------------------------------------------------------

def _run_df(tmp_path: Path, name: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    p = tmp_path / f"{name}.csv"
    df.to_csv(p, index=False)
    return run_pipeline(p)["findings"]


def _within(actual, planted, tol=0.35):
    return actual is not None and abs(actual - planted) <= tol * abs(planted)


def test_planted_hourly_cycle_is_found(tmp_path):
    findings = _run_df(tmp_path, "sensor", sensor_table(cycle=True))
    hits = [f for f in findings if f["kind"] == "trend" and f.get("dimension") == "hour_of_day"
            and f.get("measure") == "sensor_03"]
    assert hits, dump(findings)
    f = hits[0]
    assert f["level"] == "15:00"  # planted peak hour
    assert _within(f["effect"], 0.6)  # peak-to-trough 30 on mean 50
    # and only the planted measure has a diurnal profile
    assert {h["measure"] for h in findings if h.get("dimension") == "hour_of_day"} == {"sensor_03"}


def test_planted_segment_lift_is_found(tmp_path):
    findings = _run_df(tmp_path, "customer", customer_table(lift=0.4))
    hits = [f for f in findings if f["kind"] == "segment_lift" and f.get("dimension") == "region"
            and f.get("level") == "West" and f.get("measure") == "amount"]
    assert hits, dump(findings)
    assert hits[0]["effect"] > 0 and _within(hits[0]["effect"], 0.4)
    assert hits[0]["p_adjusted"] < 0.05


def test_planted_arm_effect_is_found(tmp_path):
    findings = _run_df(tmp_path, "experiment", experiment_table(effect=0.15))
    hits = [f for f in findings if f["kind"] == "test" and f.get("dimension") == "arm"
            and f.get("level") == "treatment" and f.get("measure") == "outcome"]
    assert hits, dump(findings)
    assert hits[0]["effect"] > 0 and _within(hits[0]["effect"], 0.15)
    assert hits[0]["p_adjusted"] < 0.05


def test_drivers_of_a_no_skill_model_are_dropped():
    """Regression for findings.rank_findings: permutation-importance drivers
    from evaluate_model are invented insight when the model does not beat a
    no-skill baseline; a skilled model's drivers stay."""
    from src.core.findings import Finding, rank_findings

    def mk(lift, measure):
        return [
            Finding(finding_id=f"m{measure}", kind="model_performance", headline="m", measure=measure,
                    source_tool="train_model", effect=0.3, effect_kind="share",
                    evidence={"lift_over_baseline": lift}),
            Finding(finding_id=f"d{measure}", kind="driver", headline="d", measure=measure,
                    dimension="site", level="a", effect=1.0, effect_kind="lift", source_tool="evaluate_model"),
        ]

    def kinds(fs):
        return sorted(f.kind for f in rank_findings(fs))

    assert kinds(mk(-0.05, "y")) == ["model_performance"]
    assert kinds(mk(0.4, "y")) == ["driver", "model_performance"]
    assert kinds(mk(-0.05, "y") + mk(0.4, "z")[1:]) == ["driver", "model_performance"]  # other target untouched
