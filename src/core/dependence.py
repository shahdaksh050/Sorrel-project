"""
Dependence- and design-aware statistical inference.

FutureScope Phase 4 (FutureScope.md §5.3):
- Autocorrelation-aware inference: Effective Sample Size (ESS) adjustment for time series.
- Deseasonalized and partial correlation: Conditioning on common drivers / cycles.
- Simpson's paradox and confounder check: Stratified effect comparison.
- Cluster- and hierarchy-aware inference: Intraclass correlation (ICC) and design effects (Deff).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def effective_sample_size_ar1(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """
    Compute the effective sample size (N_eff) and autocorrelation-adjusted
    p-value for the correlation between two time series with serial dependence.
    Uses the Pyper-Peterman (1998) / Quenouille (1947) adjustment:
    N_eff = N * (1 - r1_x * r1_y) / (1 + r1_x * r1_y).
    """
    s_x = pd.Series(x).dropna()
    s_y = pd.Series(y).dropna()
    valid = s_x.index.intersection(s_y.index)
    if len(valid) < 6:
        return {
            "n": len(valid),
            "n_eff": len(valid),
            "r": None,
            "raw_p": None,
            "adjusted_p": None,
            "inflation_factor": 1.0,
        }

    vx = s_x.loc[valid].values
    vy = s_y.loc[valid].values
    n = len(valid)

    raw_r, raw_p = stats.pearsonr(vx, vy)

    # Lag-1 autocorrelation
    r1_x = float(np.corrcoef(vx[:-1], vx[1:])[0, 1]) if n > 3 else 0.0
    r1_y = float(np.corrcoef(vy[:-1], vy[1:])[0, 1]) if n > 3 else 0.0

    r1_x = float(np.nan_to_num(r1_x, nan=0.0))
    r1_y = float(np.nan_to_num(r1_y, nan=0.0))

    denom = 1.0 + r1_x * r1_y
    if denom > 1e-6:
        n_eff_float = n * (1.0 - r1_x * r1_y) / denom
    else:
        n_eff_float = float(n)

    n_eff = max(4.0, min(float(n), n_eff_float))
    inflation_factor = n / n_eff

    # Recompute p-value using adjusted degrees of freedom
    t_stat = raw_r * math.sqrt((n_eff - 2.0) / max(1e-12, 1.0 - raw_r**2)) if abs(raw_r) < 1.0 else float("inf")
    adj_p = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n_eff - 2.0)))

    return {
        "n": n,
        "n_eff": round(n_eff, 1),
        "r": round(float(raw_r), 4),
        "r1_x": round(r1_x, 4),
        "r1_y": round(r1_y, 4),
        "raw_p": round(float(raw_p), 6),
        "adjusted_p": round(float(adj_p), 6),
        "inflation_factor": round(inflation_factor, 2),
    }


def partial_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    covariates: list[str] | str,
) -> dict[str, Any]:
    """
    Compute partial correlation between x and y controlling for one or more covariates.
    Residualizes x and y on covariates via OLS and correlates the residuals.
    """
    if isinstance(covariates, str):
        covariates = [c.strip() for c in covariates.split(",") if c.strip()]

    all_cols = [x, y, *covariates]
    sub_df = df[all_cols].dropna()
    n = len(sub_df)
    k = len(covariates)

    if n <= k + 2:
        return {"partial_r": None, "p_value": None, "n": n, "df": n - k - 2}

    vx = sub_df[x].values
    vy = sub_df[y].values
    z = np.column_stack([np.ones(n), sub_df[covariates].values])

    # Residualize x on z
    beta_x, _, _, _ = np.linalg.lstsq(z, vx, rcond=None)
    res_x = vx - z @ beta_x

    # Residualize y on z
    beta_y, _, _, _ = np.linalg.lstsq(z, vy, rcond=None)
    res_y = vy - z @ beta_y

    r, _ = stats.pearsonr(res_x, res_y)
    deg_free = n - k - 2
    t_stat = r * math.sqrt(deg_free / max(1e-12, 1.0 - r**2)) if abs(r) < 1.0 else float("inf")
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=deg_free)))

    return {
        "partial_r": round(float(r), 4),
        "p_value": round(float(p_val), 6),
        "n": n,
        "df": deg_free,
        "covariates": covariates,
    }


def check_simpsons_paradox(
    df: pd.DataFrame,
    outcome: str,
    group_col: str,
    confounder_col: str,
) -> dict[str, Any]:
    """
    Detect Simpson's Paradox between two groups on an outcome across strata of a confounder.
    Compares the sign of the overall aggregate difference to the stratum-specific differences.
    """
    sub = df[[outcome, group_col, confounder_col]].dropna()
    groups = sub[group_col].unique()
    if len(groups) != 2:
        return {"paradox_detected": False, "reason": "Requires exactly 2 groups to compare"}

    g1, g2 = groups[0], groups[1]
    agg_mean1 = float(sub[sub[group_col] == g1][outcome].mean())
    agg_mean2 = float(sub[sub[group_col] == g2][outcome].mean())
    agg_diff = agg_mean1 - agg_mean2

    strata = sub[confounder_col].unique()
    strata_diffs: dict[str, dict[str, Any]] = {}
    opposing_strata_count = 0
    valid_strata_count = 0

    for s in strata:
        s_data = sub[sub[confounder_col] == s]
        m1_vals = s_data[s_data[group_col] == g1][outcome]
        m2_vals = s_data[s_data[group_col] == g2][outcome]
        if len(m1_vals) >= 2 and len(m2_vals) >= 2:
            m1 = float(m1_vals.mean())
            m2 = float(m2_vals.mean())
            diff = m1 - m2
            strata_diffs[str(s)] = {
                "mean_g1": round(m1, 3),
                "mean_g2": round(m2, 3),
                "diff": round(diff, 3),
                "n": len(s_data),
            }
            valid_strata_count += 1
            if agg_diff != 0 and np.sign(diff) != np.sign(agg_diff):
                opposing_strata_count += 1

    paradox_detected = bool(valid_strata_count >= 2 and opposing_strata_count >= (valid_strata_count / 2))

    return {
        "paradox_detected": paradox_detected,
        "aggregate_diff": round(agg_diff, 4),
        "strata_diffs": strata_diffs,
        "opposing_strata_count": opposing_strata_count,
        "valid_strata_count": valid_strata_count,
        "explanation": (
            f"Simpson's Paradox detected: overall difference is {round(agg_diff, 3)}, but {opposing_strata_count}/{valid_strata_count} "
            f"strata of '{confounder_col}' show an effect in the opposite direction."
            if paradox_detected
            else "No Simpson's Paradox detected across strata."
        ),
    }


def intraclass_correlation(
    df: pd.DataFrame,
    cluster_col: str,
    measure_col: str,
) -> dict[str, Any]:
    """
    Compute Intraclass Correlation Coefficient (ICC(1)) and Design Effect (Deff)
    for clustered / hierarchical data (e.g. patients in hospitals, students in schools).
    """
    sub = df[[cluster_col, measure_col]].dropna()
    clusters = sub[cluster_col].unique()
    k = len(clusters)
    n = len(sub)

    if k < 2 or n <= k:
        return {"icc": 0.0, "deff": 1.0, "n_eff": float(n), "warning": "Insufficient clusters"}

    grand_mean = float(sub[measure_col].mean())
    cluster_stats = sub.groupby(cluster_col)[measure_col].agg(["count", "mean"])
    cluster_sizes = cluster_stats["count"].values
    cluster_means = cluster_stats["mean"].values

    ss_between = float(np.sum(cluster_sizes * (cluster_means - grand_mean) ** 2))
    ms_between = ss_between / (k - 1)

    # Within-cluster sum of squares
    ss_within = float(np.sum(sub.groupby(cluster_col)[measure_col].apply(lambda x: np.sum((x - x.mean()) ** 2))))
    ms_within = ss_within / (n - k) if (n - k) > 0 else 1e-6

    # Average cluster size (harmonic or adjusted average m_0)
    m_0 = (n - float(np.sum(cluster_sizes**2)) / n) / (k - 1) if (k - 1) > 0 else float(np.mean(cluster_sizes))
    m_0 = max(1.0, m_0)

    # ICC formula
    denom = ms_between + (m_0 - 1.0) * ms_within
    icc = float((ms_between - ms_within) / denom) if denom > 0 else 0.0
    icc = max(0.0, min(1.0, icc))

    # Design effect: Deff = 1 + (mean_m - 1) * ICC
    mean_m = float(np.mean(cluster_sizes))
    deff = float(1.0 + (mean_m - 1.0) * icc)
    n_eff = float(n / deff)

    needs_clustering_warning = bool(deff > 1.25 or icc > 0.05)

    return {
        "icc": round(icc, 4),
        "deff": round(deff, 2),
        "n": n,
        "n_clusters": k,
        "avg_cluster_size": round(mean_m, 1),
        "n_eff": round(n_eff, 1),
        "needs_cluster_robust": needs_clustering_warning,
        "recommendation": (
            f"Clustering detected (ICC={round(icc, 3)}, Deff={round(deff, 2)}): standard errors must use "
            f"cluster-robust standard errors or mixed-effects models (effective sample size is {round(n_eff, 0)} vs {n})."
            if needs_clustering_warning
            else "Minimal clustering effect (Deff ~ 1.0)."
        ),
    }
