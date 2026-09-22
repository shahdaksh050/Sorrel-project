"""
Unit tests for Sensitivity & Jackknife Fragility Audits (Phase 12).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.sensitivity import audit_finding_sensitivity


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
    assert report["is_fragile"] is True
    assert report["effect_shift_pct"] > 30.0
    assert "Fragile finding" in report["diagnosis"]
