"""
Unit tests for Dependence- and Design-Aware Inference (FutureScope Phase 4).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.causal_guard import (
    CAVEAT_OBSERVATIONAL,
    audit_findings_causal_language,
    classify_study_design,
    guard_causal_claims,
)
from src.core.controller import AgentController
from src.core.dependence import (
    check_simpsons_paradox,
    effective_sample_size_ar1,
    intraclass_correlation,
    partial_correlation,
)
from src.core.findings import Finding
from src.core.profiler import profile_dataframe


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


def test_classify_study_design_treatment_column_name_alone_is_not_enough() -> None:
    """Null case: a real observational medical dataset can have a column
    literally named 'treatment' naming the actual drug/procedure each
    patient received (many distinct values, not a random 2-4-arm
    assignment). Classifying that as randomized_experiment would silently
    disable the causal-language guard on genuinely observational data."""
    df_obs_treatment = pd.DataFrame({
        "treatment": ["chemo_A", "chemo_B", "radiation", "surgery", "palliative"] * 4,
        "age": list(range(20)),
        "outcome": [0, 1] * 10,
    })
    assert df_obs_treatment["treatment"].nunique() == 5
    assert classify_study_design(df_obs_treatment) == "observational"

    # Planted case: a genuine small-arm assignment column still classifies
    # as an experiment (guard against over-fixing away the real signal).
    df_three_arm = pd.DataFrame({
        "treatment": ["control", "low_dose", "high_dose"] * 10,
        "y": list(range(30)),
    })
    assert classify_study_design(df_three_arm) == "randomized_experiment"


# ---------------------------------------------------------------------------
# Wiring tests — dependence.py's functions are pure and were unit-tested
# above in isolation, but nothing called them from the actual pipeline
# (controller.py) until AgentController._audit_dependence_structure was
# added. These prove the wiring itself: a planted case that fires through
# the real analyze() pipeline, and a null case proving ~0 misfire cost.
# ---------------------------------------------------------------------------


def test_dependence_icc_wired_end_to_end(tmp_path: Path) -> None:
    """Planted case: strongly clustered data (10 schools x 20 students) run
    through the real deterministic pipeline must surface a clustering
    method_fit finding — not just pass the standalone intraclass_correlation
    unit test above."""
    rng = np.random.default_rng(7)
    rows = []
    for school_id in range(10):
        school_mean = float(school_id * 10.0)
        for _ in range(20):
            rows.append({
                "school_id": f"school_{school_id}",
                "score": school_mean + rng.normal(scale=1.0),
            })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "clustered_schools.csv"
    df.to_csv(csv_path, index=False)

    out_dir = str(tmp_path / "output_icc")
    os.makedirs(out_dir, exist_ok=True)
    controller = AgentController(
        objective="Analyze scores by school", output_dir=out_dir, use_llm=False
    )
    result = controller.analyze(file_path=str(csv_path))

    assert result["status"] == "complete"
    dep_findings = [
        f for f in result["findings"] if f.get("source_tool") == "dependence_audit"
    ]
    assert dep_findings, "expected a dependence_audit finding on strongly clustered data"
    icc_findings = [f for f in dep_findings if "clustered" in f["headline"]]
    assert icc_findings, f"expected a clustering finding, got: {dep_findings}"


def test_dependence_icc_null_case_no_misfire() -> None:
    """Null case: independent rows with no repeated-entity structure must
    not produce a dependence_audit finding (Generality Rule — ~0 cost /
    no misfire when the triggering property is absent)."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "amount": rng.normal(100, 10, size=200),
        "category": rng.choice(["a", "b", "c"], size=200),
    })
    profile = profile_dataframe(df)

    controller = AgentController.__new__(AgentController)
    controller.last_profile = profile
    from src.core.memory import MemorySystem
    controller.memory = MemorySystem()

    controller._audit_dependence_structure(df)
    assert controller.memory.findings == []


def test_dependence_simpsons_paradox_wired_via_audit() -> None:
    """The controller-level audit reuses a real `segment_lift` Finding (as
    segment_comparison.py would have produced) and must (a) detect the
    planted paradox, (b) add a companion method_fit Finding with the exact
    numbers, and (c) attach a caveat to the original finding — the
    Disclose-Don't-Hide rule."""
    data = []
    data.extend([{"outcome": 10.0, "group": "treatment", "strata": "small"}] * 10)
    data.extend([{"outcome": 40.0, "group": "treatment", "strata": "large"}] * 90)
    data.extend([{"outcome": 20.0, "group": "control", "strata": "small"}] * 90)
    data.extend([{"outcome": 50.0, "group": "control", "strata": "large"}] * 10)
    df = pd.DataFrame(data)
    profile = profile_dataframe(df)

    original = Finding(
        finding_id="segment_lift_0_outcome_group_treatment",
        kind="segment_lift",
        headline="'treatment' group averages a higher outcome.",
        evidence={"n": 100, "n_rest": 100},
        source_tool="segment_comparison",
        measure="outcome",
        dimension="group",
        level="treatment",
        effect=0.5,
        effect_kind="lift",
    )

    controller = AgentController.__new__(AgentController)
    controller.last_profile = profile
    from src.core.memory import MemorySystem
    controller.memory = MemorySystem()
    controller.memory.add_findings([original])

    controller._audit_dependence_structure(df)

    dep_findings = [
        f for f in controller.memory.findings if f.source_tool == "dependence_audit"
    ]
    assert dep_findings, "expected a Simpson's paradox finding"
    assert "reverses" in dep_findings[0].headline
    # Honest counts, not an overclaim: this fixture has 2 strata and both
    # oppose, so the headline must say "2 of 2", not blanket "every stratum".
    assert "2 of 2" in dep_findings[0].headline
    assert any("Simpson's paradox" in c for c in original.caveats)


def test_dependence_simpsons_paradox_headline_reports_honest_majority_not_unanimous() -> None:
    """The underlying `check_simpsons_paradox` fires on a majority of
    opposing strata (>= half), not unanimity. The controller's headline text
    must report the real `{opposing}/{valid}` counts it received rather than
    claiming "reverses within every stratum" when only a majority did —
    caught by /advisor as an overclaim risk the 2-of-2 planted case above
    can't detect by itself."""
    data = []
    # 3 strata; overall aggregate favors treatment, but only 2 of 3 strata
    # oppose that direction (the 3rd, "medium", agrees with the aggregate).
    data.extend([{"outcome": 10.0, "group": "treatment", "strata": "small"}] * 10)
    data.extend([{"outcome": 30.0, "group": "treatment", "strata": "medium"}] * 50)
    data.extend([{"outcome": 40.0, "group": "treatment", "strata": "large"}] * 90)
    data.extend([{"outcome": 20.0, "group": "control", "strata": "small"}] * 90)
    data.extend([{"outcome": 20.0, "group": "control", "strata": "medium"}] * 50)
    data.extend([{"outcome": 50.0, "group": "control", "strata": "large"}] * 10)
    df = pd.DataFrame(data)
    profile = profile_dataframe(df)

    original = Finding(
        finding_id="segment_lift_0_outcome_group_treatment",
        kind="segment_lift",
        headline="'treatment' group averages a higher outcome.",
        evidence={"n": 150, "n_rest": 150},
        source_tool="segment_comparison",
        measure="outcome",
        dimension="group",
        level="treatment",
        effect=0.5,
        effect_kind="lift",
    )

    controller = AgentController.__new__(AgentController)
    controller.last_profile = profile
    from src.core.memory import MemorySystem
    controller.memory = MemorySystem()
    controller.memory.add_findings([original])

    controller._audit_dependence_structure(df)

    dep_findings = [
        f for f in controller.memory.findings if f.source_tool == "dependence_audit"
    ]
    assert dep_findings, "expected a Simpson's paradox finding (2/3 strata oppose, >= half)"
    headline = dep_findings[0].headline
    assert "3 of 3" not in headline, f"overclaims unanimity when only a majority opposed: {headline}"
    assert "of 3" in headline


def test_dependence_simpsons_paradox_null_case_no_segment_findings() -> None:
    """No segment_lift findings exist yet (e.g. a --no-llm run before
    segment_comparison ever ran) -> the Simpson's-paradox half of the audit
    must be a no-op, not an error, and cost ~0."""
    df = pd.DataFrame({
        "outcome": [1.0, 2.0, 3.0, 4.0],
        "group": ["a", "b", "a", "b"],
        "strata": ["x", "x", "y", "y"],
    })
    profile = profile_dataframe(df)

    controller = AgentController.__new__(AgentController)
    controller.last_profile = profile
    from src.core.memory import MemorySystem
    controller.memory = MemorySystem()

    controller._audit_dependence_structure(df)
    assert controller.memory.findings == []
