"""
Unit tests for variable-aware statistical methods (FutureScope Phase 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.stats_utils import (
    benford_analysis,
    circular_statistics,
    cliffs_delta,
    compositional_clr,
    count_target_diagnostics,
    digit_heaping_test,
    robust_location_dispersion,
)


def test_circular_statistics_planted() -> None:
    # Angles clustered tightly around 0 / 360 degrees (e.g. 355, 0, 5)
    angles = [355.0, 0.0, 5.0, 350.0, 10.0]
    res = circular_statistics(angles, high=360.0)
    assert res["circular_mean"] is not None
    # Mean should be near 0 / 360, definitely NOT near 144 (which a regular arithmetic mean would give!)
    mean_val = res["circular_mean"]
    assert mean_val < 15.0 or mean_val > 345.0
    assert res["resultant_length"] > 0.90
    assert res["is_uniform"] is False  # Non-uniform (strongly clustered)


def test_circular_statistics_null() -> None:
    # Uniform angles around the clock
    angles = [0.0, 90.0, 180.0, 270.0]
    res = circular_statistics(angles, high=360.0)
    assert np.isclose(res["resultant_length"], 0.0, atol=1e-5)
    assert res["is_uniform"] is True


def test_compositional_clr() -> None:
    # Budget shares summing to 100
    df = pd.DataFrame({
        "housing": [40.0, 50.0, 30.0],
        "food": [30.0, 20.0, 35.0],
        "transport": [30.0, 30.0, 35.0],
    })
    clr_df = compositional_clr(df, ["housing", "food", "transport"])
    assert clr_df.shape == (3, 3)
    # The sum of CLR coordinates across parts for each row is identically 0
    row_sums = clr_df.sum(axis=1)
    assert np.allclose(row_sums, 0.0, atol=1e-5)


def test_benford_analysis_planted_and_null() -> None:
    # Planted Benford distribution: 2^n powers span several orders of magnitude
    powers_of_two = [2**i for i in range(1, 100)]
    res_benford = benford_analysis(powers_of_two)
    assert res_benford["conformity"] in ("close", "acceptable")
    assert res_benford["first_digit_counts"][1] > res_benford["first_digit_counts"][9]

    # Null / uniform first digits: non-conforming
    uniform_digits = [i * 100 for i in range(1, 10)] * 20
    res_uniform = benford_analysis(uniform_digits)
    assert res_uniform["conformity"] == "non_conforming"


def test_digit_heaping_planted() -> None:
    # Planted heaping: excess of 0s and 5s
    heaped_ages = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65] * 5 + [21, 32, 43, 54]
    res = digit_heaping_test(heaped_ages)
    assert res["heaping_detected"] is True
    assert res["whipples_index"] > 175.0

    # Unheaped uniform ages
    unheaped = list(range(20, 80))
    res_unheaped = digit_heaping_test(unheaped)
    assert res_unheaped["heaping_detected"] is False


def test_cliffs_delta() -> None:
    # Completely separated ordinal groups
    group_a = [1, 2, 3, 4]
    group_b = [5, 6, 7, 8]
    res = cliffs_delta(group_b, group_a)
    assert res["delta"] == 1.0
    assert res["magnitude"] == "large"
    assert res["p_value"] < 0.05

    # Identical groups -> delta = 0
    res_null = cliffs_delta(group_a, group_a)
    assert res_null["delta"] == 0.0
    assert res_null["magnitude"] == "negligible"


def test_count_target_diagnostics() -> None:
    # Continuous data -> not a count
    continuous = [1.2, 3.4, 5.6, 7.8, 9.0] * 5
    res_cont = count_target_diagnostics(continuous)
    assert res_cont["is_count"] is False
    assert res_cont["recommended_model"] == "ols"

    # Poisson-like count (variance ~ mean)
    rng = np.random.default_rng(42)
    poisson_counts = rng.poisson(lam=3.0, size=200)
    res_pois = count_target_diagnostics(poisson_counts)
    assert res_pois["is_count"] is True
    assert res_pois["recommended_model"] == "poisson"

    # Overdispersed counts (Negative Binomial)
    neg_bin_counts = rng.negative_binomial(n=2, p=0.1, size=200)
    res_nb = count_target_diagnostics(neg_bin_counts)
    assert res_nb["is_count"] is True
    assert res_nb["is_overdispersed"] is True
    assert res_nb["recommended_model"] in ("negative_binomial", "zero_inflated")


def test_robust_location_dispersion() -> None:
    # Normal data -> mean preferred
    normal_data = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9] * 5
    res_norm = robust_location_dispersion(normal_data)
    assert res_norm["is_heavy_tailed"] is False
    assert res_norm["preferred_location"] == "mean"

    # Extreme outlier / heavy tail -> median preferred
    heavy_tailed = [1.0, 1.2, 1.1, 1.3, 0.9, 1.0, 1000.0]
    res_heavy = robust_location_dispersion(heavy_tailed)
    assert res_heavy["is_heavy_tailed"] is True
    assert res_heavy["preferred_location"] == "median"
