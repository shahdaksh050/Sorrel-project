"""
Unit tests for data integrity and logical constraint discovery (FutureScope Phase 6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.integrity import (
    check_missingness_mechanism,
    check_value_continuity_and_flatlines,
    discover_logical_constraints,
    evaluate_data_integrity,
)


def test_discover_logical_inequality_constraint() -> None:
    # high >= low for 98 rows, but 2 rows have low > high
    highs = [100.0 + i for i in range(100)]
    lows = [50.0 + i for i in range(100)]
    # Violations at index 10 and 20
    lows[10] = 200.0
    lows[20] = 250.0

    df = pd.DataFrame({"high": highs, "low": lows})
    constraints = discover_logical_constraints(df)

    assert len(constraints) >= 1
    c = constraints[0]
    assert c["type"] == "inequality"
    assert "high >= low" in c["rule"]
    assert c["violating_count"] == 2
    assert 10 in c["violating_indices"]
    assert 20 in c["violating_indices"]


def test_discover_logical_sum_identity() -> None:
    part_a = [10.0 + i for i in range(50)]
    part_b = [20.0 + i for i in range(50)]
    total = [part_a[i] + part_b[i] for i in range(50)]
    # Violation at index 5
    total[5] = 999.0

    df = pd.DataFrame({"part_a": part_a, "part_b": part_b, "total": total})
    constraints = discover_logical_constraints(df)

    sum_c = next((c for c in constraints if c["type"] == "sum_identity"), None)
    assert sum_c is not None
    assert sum_c["violating_count"] == 1
    assert 5 in sum_c["violating_indices"]


def test_missingness_mechanism_systematic_mar() -> None:
    # Missingness in 'income' strongly correlated with 'age'
    rng = np.random.default_rng(42)
    n = 200
    age = rng.normal(loc=40, scale=10, size=n)
    income = rng.normal(loc=50000, scale=15000, size=n)

    # Missing income when age > 45 (strong correlation between missingness and age)
    income_with_missing = income.copy()
    for i in range(n):
        if age[i] > 45 and rng.random() > 0.3:
            income_with_missing[i] = np.nan

    df = pd.DataFrame({"age": age, "income": income_with_missing})
    res = check_missingness_mechanism(df)

    assert "income" in res
    assert res["income"]["mechanism"] == "MAR_systematic"
    assert len(res["income"]["correlated_covariates"]) >= 1
    assert res["income"]["correlated_covariates"][0]["column"] == "age"


def test_missingness_block_dropout() -> None:
    # 25 consecutive NaNs (sensor blackout)
    vals = [10.0] * 100
    for i in range(30, 55):
        vals[i] = np.nan  # 25 consecutive missing

    df = pd.DataFrame({"sensor": vals, "temp": [20.0] * 100})
    res = check_missingness_mechanism(df)

    assert "sensor" in res
    assert res["sensor"]["has_block_missingness"] is True
    assert res["sensor"]["max_missing_block"] == 25


def test_continuity_flatlines_and_negatives() -> None:
    # 15 consecutive identical values
    temps = [20.0 + i * 0.1 for i in range(50)]
    for i in range(10, 25):
        temps[i] = 25.0  # 15 identical values

    # Negative prices
    prices = [10.0, 20.0, -5.0, 30.0, -10.0] + [15.0] * 45

    df = pd.DataFrame({"temperature": temps, "price": prices})
    issues = check_value_continuity_and_flatlines(df, max_consecutive_identical=10)

    flatline_issue = next((iss for iss in issues if iss["type"] == "flatline"), None)
    assert flatline_issue is not None
    assert flatline_issue["column"] == "temperature"
    assert flatline_issue["run_length"] == 15

    negative_issue = next((iss for iss in issues if iss["type"] == "negative_values"), None)
    assert negative_issue is not None
    assert negative_issue["column"] == "price"
    assert negative_issue["negative_count"] == 2


def test_evaluate_data_integrity_full() -> None:
    # Composite dataframe with constraint violation, flatline, and negative values
    highs = [100.0] * 50
    lows = [50.0] * 50
    lows[5] = 150.0  # constraint violation

    df = pd.DataFrame({
        "high": highs,
        "low": lows,
        "age": [25, 30, -5, 40] + [35] * 46,  # negative age
    })
    findings = evaluate_data_integrity(df)
    assert len(findings) >= 2
    assert any("constraint violation" in f.headline.lower() for f in findings)
    assert any("negative values" in f.headline.lower() for f in findings)
