"""
Data integrity, logical constraint discovery, and missingness mechanism checks.

FutureScope Phase 6 (FutureScope.md §5.5):
- Logical constraint discovery: learns inequalities (high >= low, end >= start)
  and sum identities (A + B == total), flagging violating rows.
- Missingness mechanism check: classifies MCAR vs MAR/systematic missingness,
  detecting when missing data introduces bias.
- Value continuity and flatline pass: detects stuck values, sensor flatlines,
  and impossible negative values.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.core.findings import Finding


def discover_logical_constraints(
    df: pd.DataFrame,
    min_support: float = 0.95,
) -> list[dict[str, Any]]:
    """
    Discover logical relationships and constraints between columns:
    - Inequalities: B >= A (e.g. high >= low, end >= start, max >= min).
    - Additive identities: A + B == C (e.g. parts summing to total).
    Returns discovered constraints along with any violating row indices.
    """
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    constraints: list[dict[str, Any]] = []

    # 1. Pairwise inequalities (col_b >= col_a)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            if i == j:
                continue
            col_a, col_b = num_cols[i], num_cols[j]
            valid = df[[col_a, col_b]].dropna()
            if len(valid) < 20:
                continue

            diff = valid[col_b] - valid[col_a]
            satisfied = (diff >= -1e-6).sum()
            support = float(satisfied / len(valid))

            # Only consider meaningful candidate pairs: names match (high/low, max/min, start/end)
            # or support is very high (>= min_support)
            name_hint = bool(
                re.search(r"high|max|end|total|gross|after", str(col_b), re.I)
                and re.search(r"low|min|start|net|before|part", str(col_a), re.I)
            )

            if support >= min_support and (name_hint or support > 0.99):
                violating_indices = list(valid[diff < -1e-6].index)
                if violating_indices:
                    constraints.append({
                        "type": "inequality",
                        "rule": f"{col_b} >= {col_a}",
                        "support": round(support, 4),
                        "total_checked": len(valid),
                        "violating_count": len(violating_indices),
                        "violating_indices": violating_indices[:50],  # cap preview
                    })

    # 2. Additive identities (A + B == C) for small column counts
    if 3 <= len(num_cols) <= 15:
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                for k in range(len(num_cols)):
                    if k == i or k == j:
                        continue
                    ca, cb, cc = num_cols[i], num_cols[j], num_cols[k]
                    valid = df[[ca, cb, cc]].dropna()
                    if len(valid) < 20:
                        continue
                    sum_diff = np.abs((valid[ca] + valid[cb]) - valid[cc])
                    match_mask = sum_diff < 1e-4
                    support = float(match_mask.sum() / len(valid))
                    if support >= min_support:
                        violating = list(valid[~match_mask].index)
                        if violating:
                            constraints.append({
                                "type": "sum_identity",
                                "rule": f"{ca} + {cb} == {cc}",
                                "support": round(support, 4),
                                "total_checked": len(valid),
                                "violating_count": len(violating),
                                "violating_indices": violating[:50],
                            })

    return constraints


def check_missingness_mechanism(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Check whether missing values are Missing Completely At Random (MCAR)
    or systematic / Missing At Random (MAR).
    Tests correlation of missingness indicators with observed measures.
    """
    results: dict[str, Any] = {}
    n = len(df)
    if n < 30:
        return results

    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    missing_cols = [c for c in df.columns if 0.02 * n <= df[c].isna().sum() <= 0.90 * n]

    for col in missing_cols:
        miss_ind = df[col].isna().astype(int).values
        # Correlate with other numeric columns
        correlated_vars: list[dict[str, Any]] = []
        for other in num_cols:
            if other == col:
                continue
            valid = df[[other]].dropna()
            if len(valid) < 20:
                continue
            y_obs = df.loc[valid.index, other].values
            m_obs = miss_ind[valid.index]
            if len(np.unique(m_obs)) < 2 or len(np.unique(y_obs)) < 2 or np.std(y_obs) < 1e-9:
                continue
            r, p = stats.pointbiserialr(m_obs, y_obs)
            if p < 0.01 and abs(r) > 0.15:
                correlated_vars.append({
                    "column": other,
                    "correlation": round(float(r), 3),
                    "p_value": round(float(p), 6),
                })

        # Check for consecutive missing blocks (sensor outage / flatline dropout)
        runs: list[int] = []
        curr = 0
        for v in miss_ind:
            if v == 1:
                curr += 1
            else:
                if curr > 0:
                    runs.append(curr)
                curr = 0
        if curr > 0:
            runs.append(curr)

        max_block = max(runs) if runs else 0
        has_block_missingness = max_block >= 10

        if correlated_vars:
            mechanism = "MAR_systematic"
            bias_warning = (
                f"Missingness in '{col}' is systematically correlated with {len(correlated_vars)} "
                f"other column(s) ({', '.join(v['column'] for v in correlated_vars[:3])}). "
                "Naive listwise deletion or simple mean imputation will introduce substantial bias."
            )
        elif has_block_missingness:
            mechanism = "block_dropout"
            bias_warning = (
                f"Missingness in '{col}' occurs in contiguous blocks of up to {max_block} observations. "
                "Suggests sensor outage or systematic reporting pause."
            )
        else:
            mechanism = "MCAR_likely"
            bias_warning = f"Missingness in '{col}' shows no strong correlation with other measures."

        results[col] = {
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(float(df[col].isna().sum() / n * 100.0), 1),
            "mechanism": mechanism,
            "has_block_missingness": has_block_missingness,
            "max_missing_block": max_block,
            "correlated_covariates": correlated_vars,
            "warning": bias_warning,
        }

    return results


def check_value_continuity_and_flatlines(
    df: pd.DataFrame,
    max_consecutive_identical: int = 10,
) -> list[dict[str, Any]]:
    """
    Detect stuck sensors / flatlines and impossible negative values.
    """
    issues: list[dict[str, Any]] = []
    num_cols = list(df.select_dtypes(include=[np.number]).columns)

    for col in num_cols:
        series = df[col].dropna()
        if len(series) < max_consecutive_identical * 2:
            continue

        # Check for flatlines (runs of identical non-zero values)
        vals = series.values
        diffs = vals[1:] == vals[:-1]
        longest_run = 0
        curr_run = 1
        stuck_val = None
        for i, same in enumerate(diffs):
            if same:
                curr_run += 1
                if curr_run > longest_run:
                    longest_run = curr_run
                    stuck_val = vals[i]
            else:
                curr_run = 1

        if longest_run >= max_consecutive_identical:
            issues.append({
                "type": "flatline",
                "column": col,
                "stuck_value": float(stuck_val) if stuck_val is not None else None,
                "run_length": longest_run,
                "description": f"Sensor flatline in '{col}': identical value {stuck_val} repeated for {longest_run} consecutive periods.",
            })

        # Check for negative values in naturally positive variables
        if re.search(r"\b(age|price|cost|revenue|count|visits|quantity|sales|spend)\b", col, re.I):
            negs = series[series < 0]
            if not negs.empty:
                issues.append({
                    "type": "negative_values",
                    "column": col,
                    "negative_count": len(negs),
                    "min_value": float(negs.min()),
                    "description": f"Impossible negative values in '{col}' ({len(negs)} observations < 0, min={negs.min()}).",
                })

    return issues


def evaluate_data_integrity(
    df: pd.DataFrame,
) -> list[Finding]:
    """
    Run full data integrity audit (constraints, missingness bias, flatlines)
    and produce Finding objects.
    """
    findings: list[Finding] = []

    # 1. Logical constraints
    constraints = discover_logical_constraints(df)
    for c in constraints:
        viol_count = c["violating_count"]
        findings.append(
            Finding(
                finding_id=f"integrity_viol_{c['rule']}",
                kind="outlier_scan",
                headline=f"Constraint violation: '{c['rule']}' violated by {viol_count} rows ({round(100 - c['support']*100, 2)}% error rate)",
                detail=f"Discovered logical rule {c['rule']} holds for {round(c['support']*100, 1)}% of rows but fails in {viol_count} cases.",
                importance=0.8,
                effect=1.0 - c["support"],
                effect_kind="pct",
                evidence=c,
                source_tool="integrity_audit",
            )
        )

    # 2. Missingness mechanism
    miss_res = check_missingness_mechanism(df)
    for col, m_info in miss_res.items():
        if m_info["mechanism"] == "MAR_systematic":
            findings.append(
                Finding(
                    finding_id=f"missing_bias_{col}",
                    kind="outlier_scan",
                    headline=f"Systematic missingness in '{col}' ({m_info['missing_pct']}% missing): potential estimation bias",
                    detail=m_info["warning"],
                    importance=0.75,
                    effect=m_info["missing_pct"] / 100.0,
                    effect_kind="pct",
                    evidence=m_info,
                    source_tool="integrity_audit",
                )
            )

    # 3. Flatlines and continuity
    continuity_issues = check_value_continuity_and_flatlines(df)
    for issue in continuity_issues:
        findings.append(
            Finding(
                finding_id=f"continuity_{issue['column']}_{issue['type']}",
                kind="outlier_scan",
                headline=issue["description"],
                detail=issue["description"],
                importance=0.7,
                evidence=issue,
                source_tool="integrity_audit",
            )
        )

    return findings
