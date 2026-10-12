"""
Unit tests for Sensitivity & Jackknife Fragility Audits (Phase 12).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.sensitivity import audit_finding_sensitivity, detect_target_leakage


def test_sensitivity_robust_finding() -> None:
    np.random.seed(42)
    x = np.random.normal(50, 10, 200)
    y = 2.0 * x + np.random.normal(0, 5, 200)
    df = pd.DataFrame({"age": x, "income": y})

    finding = {
        "measure": "income",
        "dimension": "age",
        "effect": 0.85,
    }

    report = audit_finding_sensitivity(df, finding, trim_pct=0.01)
    assert report["status"] == "passed"
    assert report["is_fragile"] is False
    assert report["fragility_score"] < 0.5
    assert "Robust" in report["diagnosis"]


def test_sensitivity_fragile_finding_with_extreme_outlier() -> None:
    np.random.seed(42)
    # Uncorrelated noise with 1 extreme leverage point that dictates correlation
    x = np.random.normal(10, 2, 100)
    y = np.random.normal(10, 2, 100)

    # Inject extreme leverage point
    x[0] = 200.0
    y[0] = 200.0

    df = pd.DataFrame({"ad_spend": x, "signups": y})

    finding = {
        "measure": "signups",
        "dimension": "ad_spend",
        "effect": 0.90,
    }

    report = audit_finding_sensitivity(df, finding, trim_pct=0.01)
    assert report["status"] == "failed"
    assert report["is_fragile"] is True
    assert report["effect_shift_pct"] > 30.0
    assert "Fragile finding" in report["diagnosis"]


def test_sensitivity_non_numeric_measure_is_not_a_robust_pass() -> None:
    df = pd.DataFrame({"label": ["a"] * 30, "segment": ["x"] * 30})

    report = audit_finding_sensitivity(df, {"measure": "label", "dimension": "segment"})

    assert report["status"] == "not_applicable"
    assert report["is_fragile"] is None
    assert report["fragility_score"] is None


def test_sensitivity_unsupported_model_driver_is_not_a_robust_pass() -> None:
    x = np.arange(30, dtype=float)
    df = pd.DataFrame({"feature": x, "outcome": x * 2.0})

    report = audit_finding_sensitivity(
        df,
        {"kind": "driver", "measure": "outcome", "dimension": "feature", "effect_kind": "r"},
    )

    assert report["status"] == "not_applicable"
    assert report["is_fragile"] is None


def test_sensitivity_categorical_segment_recomputes_level_vs_rest_lift() -> None:
    df = pd.DataFrame({
        "segment": ["A"] * 20 + ["B"] * 20,
        "score": [10.0] * 20 + [5.0] * 20,
    })
    finding = {
        "kind": "segment_lift",
        "measure": "score",
        "dimension": "segment",
        "level": "A",
        "effect_kind": "lift",
    }

    report = audit_finding_sensitivity(df, finding)

    assert report["status"] == "passed"
    assert report["baseline_effect"] == 1.0
    assert report["trimmed_effect"] == 1.0


def test_sensitivity_categorical_without_supported_contrast_is_not_a_pass() -> None:
    df = pd.DataFrame({
        "segment": ["A"] * 20 + ["B"] * 20,
        "score": list(range(40)),
    })

    report = audit_finding_sensitivity(
        df,
        {"kind": "test", "measure": "score", "dimension": "segment", "effect_kind": "eta_sq"},
    )

    assert report["status"] == "not_applicable"
    assert report["is_fragile"] is None


def test_leakage_ignores_near_unique_identifier_column() -> None:
    # A transaction id (or timestamp) is unique per row, so it trivially
    # "determines" a categorical target — that isn't leakage, it's an
    # identifier, and must not be flagged (mirrors the same guard already
    # applied in src/tools/ml_pipeline.py's _detect_target_leakage).
    n = 100
    df = pd.DataFrame({
        "transaction_id": [f"tx_{i}" for i in range(n)],
        "region": np.random.RandomState(0).choice(["north", "south"], n),
        "outcome": np.random.RandomState(1).choice(["yes", "no"], n),
    })

    alerts = detect_target_leakage(df, "outcome")
    assert not any(a["column"] == "transaction_id" for a in alerts)


def test_leakage_flags_genuine_deterministic_mapping() -> None:
    # A low-cardinality feature that perfectly determines the target class
    # is real leakage and must still be caught.
    n = 100
    plan = np.random.RandomState(0).choice(["basic", "premium"], n)
    outcome = np.where(plan == "premium", "churn", "retain")
    df = pd.DataFrame({"plan_tier": plan, "outcome": outcome})

    alerts = detect_target_leakage(df, "outcome")
    assert any(
        a["column"] == "plan_tier" and a["leakage_type"] == "deterministic_mapping"
        for a in alerts
    )


def test_simpsons_paradox_detected_on_reversal() -> None:
    # Classic Simpson's paradox:
    # Subgroup 1: X in [10, 20], Y = -2*X + 50 (negative correlation within group)
    # Subgroup 2: X in [30, 40], Y = -2*X + 130 (negative correlation within group)
    # But Subgroup 2 has much higher X and Y, making aggregate correlation positive!
    np.random.seed(42)
    n = 40
    x1 = np.linspace(10, 20, n)
    y1 = -1.5 * x1 + 50 + np.random.normal(0, 1, n)
    g1 = ["GroupA"] * n

    x2 = np.linspace(30, 40, n)
    y2 = -1.5 * x2 + 130 + np.random.normal(0, 1, n)
    g2 = ["GroupB"] * n

    df = pd.DataFrame({
        "study_hours": np.concatenate([x1, x2]),
        "exam_score": np.concatenate([y1, y2]),
        "department": g1 + g2,
    })

    from src.core.sensitivity import audit_simpsons_paradox

    finding = {
        "kind": "correlation",
        "measure": "exam_score",
        "dimension": "study_hours",
        "effect": 0.80,
    }

    report = audit_simpsons_paradox(df, finding)
    assert report["status"] == "failed"
    assert report["has_paradox"] is True
    assert report["stratifier"] == "department"
    assert "Simpson's Paradox detected" in report["diagnosis"]


def test_simpsons_paradox_passes_when_consistent() -> None:
    np.random.seed(42)
    n = 40
    x1 = np.linspace(10, 20, n)
    y1 = 2.0 * x1 + np.random.normal(0, 1, n)
    g1 = ["GroupA"] * n

    x2 = np.linspace(10, 20, n)
    y2 = 2.0 * x2 + 5 + np.random.normal(0, 1, n)
    g2 = ["GroupB"] * n

    df = pd.DataFrame({
        "tenure": np.concatenate([x1, x2]),
        "salary": np.concatenate([y1, y2]),
        "location": g1 + g2,
    })

    from src.core.sensitivity import audit_simpsons_paradox

    finding = {
        "kind": "correlation",
        "measure": "salary",
        "dimension": "tenure",
        "effect": 0.90,
    }

    report = audit_simpsons_paradox(df, finding)
    assert report["status"] == "passed"
    assert report["has_paradox"] is False
    assert "Robust" in report["diagnosis"]

