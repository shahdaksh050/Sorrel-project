"""
Unit tests for Target Leakage Guard (Phase 12).
"""
from __future__ import annotations

import pandas as pd

from src.core.sensitivity import detect_target_leakage


def test_target_leakage_detection() -> None:
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "churn_is_churned": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],          # semantic leakage
        "post_cancellation_survey": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # temporal prefix leakage
        "almost_perfect_leak": [0.01, 0.99, 0.02, 0.98, 0.03, 0.97, 0.01, 0.99, 0.02, 0.98], # r > 0.99
        "normal_feature": [15, 42, 33, 21, 55, 60, 12, 80, 25, 40], # benign
    })

    alerts = detect_target_leakage(df, target_col="churn")

    leaked_cols = {a["column"]: a["leakage_type"] for a in alerts}
    assert "churn_is_churned" in leaked_cols
    assert "post_cancellation_survey" in leaked_cols
    assert "almost_perfect_leak" in leaked_cols
    assert "normal_feature" not in leaked_cols
