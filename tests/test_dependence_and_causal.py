"""
Unit tests for Dependence- and Design-Aware Inference (FutureScope Phase 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.causal_guard import (
    CAVEAT_OBSERVATIONAL,
    audit_findings_causal_language,
    classify_study_design,
    guard_causal_claims,
)
from src.core.dependence import (
    check_simpsons_paradox,
    effective_sample_size_ar1,
    intraclass_correlation,
    partial_correlation,
)
from src.core.findings import Finding


def test_effective_sample_size_ar1() -> None:
    # Highly autocorrelated AR(1) series
    rng = np.random.default_rng(42)
    n = 200
    e1 = rng.normal(size=n)
    e2 = rng.normal(size=n)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + e1[t]
        y[t] = 0.9 * y[t - 1] + e2[t]

    res = effective_sample_size_ar1(x, y)
    assert res["n"] == 200
    # With rho ~ 0.9, Neff should be substantially smaller than N
    assert res["n_eff"] < 100
    assert res["inflation_factor"] > 2.0


def test_partial_correlation() -> None:
    # X and Y are both driven entirely by Z: X = 2*Z + noise, Y = 3*Z + noise
    rng = np.random.default_rng(42)
    n = 200
    z = rng.normal(size=n)
    x = 2.0 * z + rng.normal(scale=0.5, size=n)
    y = 3.0 * z + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"x": x, "y": y, "z": z})
    raw_r = float(df["x"].corr(df["y"]))
    assert raw_r > 0.8  # Strong spurious raw correlation

    part_res = partial_correlation(df, "x", "y", "z")
    assert part_res["partial_r"] is not None
    # Controlling for Z, partial correlation should collapse toward 0
    assert abs(part_res["partial_r"]) < 0.20


def test_check_simpsons_paradox() -> None:
    # Classic Simpson's paradox:
    # Strata 1: Treatment diff = 10 - 20 = -10
    # Strata 2: Treatment diff = 40 - 50 = -10
    # But treatment group is mostly in Strata 2, control group mostly in Strata 1!
    data = []
    # Treatment group: 10 in Strata 1 (mean=10), 90 in Strata 2 (mean=40) -> overall mean = 37
    data.extend([{"outcome": 10.0, "group": "treatment", "strata": "small"}] * 10)
    data.extend([{"outcome": 40.0, "group": "treatment", "strata": "large"}] * 90)
    # Control group: 90 in Strata 1 (mean=20), 10 in Strata 2 (mean=50) -> overall mean = 23
    data.extend([{"outcome": 20.0, "group": "control", "strata": "small"}] * 90)
    data.extend([{"outcome": 50.0, "group": "control", "strata": "large"}] * 10)

    df = pd.DataFrame(data)
    res = check_simpsons_paradox(df, "outcome", "group", "strata")
    assert res["paradox_detected"] is True
    # Aggregate diff is positive (treatment > control), but within every stratum it is negative!
    assert res["aggregate_diff"] > 0
    assert res["opposing_strata_count"] == 2


def test_intraclass_correlation() -> None:
    # Strong cluster effect (students in schools)
    rng = np.random.default_rng(42)
    rows = []
    for school_id in range(10):
        school_mean = float(school_id * 10.0)
        for _ in range(20):
            score = school_mean + rng.normal(scale=1.0)
            rows.append({"school": f"school_{school_id}", "score": score})

    df = pd.DataFrame(rows)
    icc_res = intraclass_correlation(df, "school", "score")
    assert icc_res["icc"] > 0.70
    assert icc_res["deff"] > 5.0
    assert icc_res["needs_cluster_robust"] is True


def test_causal_guard_observational() -> None:
    text = "Feature A drives sales and causes higher customer churn."
    sanitized, warnings = guard_causal_claims(text, study_design="observational")
    assert "causes" not in sanitized
    assert "drives" not in sanitized
    assert "is associated with" in sanitized
    assert "correlates with" in sanitized
    assert len(warnings) == 2


def test_causal_guard_experiment() -> None:
    text = "Feature A causes higher conversion."
    sanitized, warnings = guard_causal_claims(text, study_design="randomized_experiment")
    # In randomized experiments, causal verbs are permitted
    assert sanitized == text
    assert len(warnings) == 0


def test_audit_findings_causal_language() -> None:
    finding = Finding(
        finding_id="f1",
        kind="driver",
        headline="Discount causes higher volume",
        detail="Pricing strategy drives repeat orders",
    )
    warnings = audit_findings_causal_language([finding], study_design="observational")
    assert len(warnings) == 2
    assert "causes" not in finding.headline
    assert "drives" not in finding.detail
    assert CAVEAT_OBSERVATIONAL in finding.caveats


def test_classify_study_design() -> None:
    df_exp = pd.DataFrame({"treatment": [0, 1, 0, 1], "y": [1, 2, 1, 3]})
    assert classify_study_design(df_exp) == "randomized_experiment"

    df_obs = pd.DataFrame({"age": [20, 30], "income": [50000, 70000]})
    assert classify_study_design(df_obs) == "observational"

    assert classify_study_design(objective="Run an A/B test analysis") == "randomized_experiment"
