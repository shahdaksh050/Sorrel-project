"""Focused safety and estimation tests for CausalAnalysisTool."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.causal_analysis import CausalAnalysisTool, DifferenceInDifferencesTool


def _write(tmp_path: Path, df: pd.DataFrame) -> str:
    path = tmp_path / "causal.csv"
    df.to_csv(path, index=False)
    return str(path)


def _overlap_frame() -> pd.DataFrame:
    rng = np.random.default_rng(19)
    n = 600
    age = rng.normal(45, 10, n)
    risk = rng.normal(0, 1, n)
    probability = 1 / (1 + np.exp(-(-0.2 + 0.02 * age + 0.25 * risk)))
    treatment = rng.binomial(1, probability)
    outcome = 4.0 * treatment + 0.1 * age + risk + rng.normal(0, 1, n)
    return pd.DataFrame({"outcome": outcome, "treatment": treatment, "age": age, "risk": risk})


def test_ipw_reports_guarded_estimate_and_balance(tmp_path: Path) -> None:
    tool = CausalAnalysisTool()
    output = tool.run(file_path=_write(tmp_path, _overlap_frame()), outcome="outcome", treatment="treatment", covariates=["age", "risk"]).output
    assert output["status"] == "estimated_with_assumptions"
    assert output["method"] == "stabilized_ipw"
    assert 2.5 < output["estimate"] < 5.5
    assert len(output["balance"]) == 2
    assert output["causal_claim_permitted"] is False
    assert tool.findings(output, None, None)[0].kind == "test"


def test_bad_overlap_is_non_identifiable(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(-4, 0.2, 120), rng.normal(4, 0.2, 120)]
    df = pd.DataFrame({"outcome": rng.normal(size=240), "treatment": [0] * 120 + [1] * 120, "x": x})
    output = CausalAnalysisTool().run(file_path=_write(tmp_path, df), outcome="outcome", treatment="treatment", covariates=["x"]).output
    assert output["status"] == "not_identifiable"
    assert "overlap" in output["reason"].lower()


def test_post_treatment_named_covariate_is_refused(tmp_path: Path) -> None:
    df = _overlap_frame().assign(post_purchase_score=lambda value: value["outcome"])
    output = CausalAnalysisTool().run(file_path=_write(tmp_path, df), outcome="outcome", treatment="treatment", covariates=["post_purchase_score"]).output
    assert output["status"] == "not_identifiable"
    assert "post-treatment" in output["reason"]


def test_requires_binary_treatment(tmp_path: Path) -> None:
    df = _overlap_frame().assign(treatment=["a", "b", "c"] * 200)
    result = CausalAnalysisTool().run(file_path=_write(tmp_path, df), outcome="outcome", treatment="treatment", covariates=["age"])
    assert result.status == "error"


def _panel_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    entities = [f"unit_{i}" for i in range(40)]
    times = [1, 2, 3, 4]
    rows = []
    treated_units = set(entities[:20])
    for u in entities:
        is_treated = 1 if u in treated_units else 0
        unit_fe = float(rng.normal(0, 1))
        for t in times:
            is_post = 1 if t >= 3 else 0
            time_fe = 0.5 * t
            effect = 5.0 * (is_treated * is_post)
            noise = float(rng.normal(0, 0.5))
            y = 10.0 + unit_fe + time_fe + effect + noise
            rows.append({
                "outcome": y,
                "treatment": is_treated,
                "time": t,
                "entity": u,
                "covariate": float(rng.normal(0, 1)),
            })
    return pd.DataFrame(rows)


def test_did_estimates_treatment_effect_with_cluster_robust_se(tmp_path: Path) -> None:
    tool = DifferenceInDifferencesTool()
    csv_path = _write(tmp_path, _panel_frame())
    output = tool.run(
        file_path=csv_path,
        outcome="outcome",
        treatment="treatment",
        time_column="time",
        entity_column="entity",
        post_treatment_time=3,
        covariates=["covariate"],
    ).output

    assert output["status"] == "estimated_with_assumptions"
    assert output["method"] == "panel_difference_in_differences"
    assert 4.0 < output["estimate"] < 6.0
    assert output["causal_claim_permitted"] is False
    assert output["parallel_trends"]["status"] == "passed"
    assert output["n_clusters"] == 40
    findings = tool.findings(output, None, None)
    assert len(findings) == 1
    assert findings[0].kind == "test"


def test_did_detects_pre_trend_violation(tmp_path: Path) -> None:
    df = _panel_frame()
    mask = (df["treatment"] == 1) & (df["time"] <= 2)
    df.loc[mask, "outcome"] += df.loc[mask, "time"] * 10.0

    tool = DifferenceInDifferencesTool()
    output = tool.run(
        file_path=_write(tmp_path, df),
        outcome="outcome",
        treatment="treatment",
        time_column="time",
        entity_column="entity",
        post_treatment_time=3,
    ).output

    assert output["status"] == "estimated_with_assumptions"
    assert output["parallel_trends"]["status"] == "violated"
    assert any("failed" in w or "differ" in w for w in output["warnings"])


def test_did_handles_untested_single_pre_period(tmp_path: Path) -> None:
    df = _panel_frame().query("time in [2, 3]").copy()
    tool = DifferenceInDifferencesTool()
    output = tool.run(
        file_path=_write(tmp_path, df),
        outcome="outcome",
        treatment="treatment",
        time_column="time",
        entity_column="entity",
        post_treatment_time=3,
    ).output

    assert output["status"] == "estimated_with_assumptions"
    assert output["parallel_trends"]["status"] == "untested_single_pre_period"


def test_did_refuses_post_treatment_covariate(tmp_path: Path) -> None:
    df = _panel_frame().assign(post_intervention_mediator=lambda d: d["outcome"])
    tool = DifferenceInDifferencesTool()
    output = tool.run(
        file_path=_write(tmp_path, df),
        outcome="outcome",
        treatment="treatment",
        time_column="time",
        entity_column="entity",
        post_treatment_time=3,
        covariates=["post_intervention_mediator"],
    ).output

    assert output["status"] == "not_identifiable"
    assert "post-treatment" in output["reason"]


def test_did_refuses_missing_cell(tmp_path: Path) -> None:
    df = _panel_frame()
    df = df[~((df["treatment"] == 1) & (df["time"] < 3))]
    tool = DifferenceInDifferencesTool()
    output = tool.run(
        file_path=_write(tmp_path, df),
        outcome="outcome",
        treatment="treatment",
        time_column="time",
        entity_column="entity",
        post_treatment_time=3,
    ).output

    assert output["status"] == "not_identifiable"
    assert "Missing observations" in output["reason"]

