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
from src.core.multiple_testing import apply_benjamini_hochberg

#: Caps the "other" side of the missingness pairwise-correlation scan so a
#: very wide table (hundreds of numeric columns) can't turn a per-load check
#: into an O(cols^2) cost. The most-complete columns are kept (see caller).
_MAX_OTHER_COLS_FOR_MISSINGNESS = 60

#: Above this many numeric columns, `discover_logical_constraints` only pays
#: for the expensive per-pair dataframe check (dropna/diff/sum) on pairs
#: whose names actually hint at an inequality; below it, every pair is still
#: checked empirically (>99% support), matching the pre-existing behavior on
#: any realistically-sized table.
_MAX_COLS_FOR_UNHINTED_INEQUALITY_SWEEP = 40


def _block_missingness_info(miss_mask: np.ndarray) -> dict[str, Any]:
    """Length of the longest run of consecutive True values in `miss_mask`,
    vectorized (no per-row Python loop — this runs once per missing column
    on every dataset load, including on 500k-row tables)."""
    if miss_mask.size == 0 or not miss_mask.any():
        return {"max_block": 0, "has_block_missingness": False}
    padded = np.concatenate(([0], miss_mask.astype(np.int8), [0]))
    diffs = np.diff(padded)
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    max_block = int((ends - starts).max()) if starts.size else 0
    return {"max_block": max_block, "has_block_missingness": max_block >= 10}


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
    if not num_cols:
        return constraints

    # Pre-extract once. Per-pair *pandas* overhead (dropna()/__getitem__/
    # indexing machinery), not the arithmetic, dominates at normal column
    # counts — profiled at 0.455s of a 0.541s call on a 15-column, 9.5k-row
    # table. Raw numpy indexing on a pre-extracted matrix runs the identical
    # O(n) per-pair algorithm with far less per-call overhead, and — unlike
    # a full pairwise 3D broadcast — stays O(rows x cols) in memory
    # regardless of column count, so it's safe at wide-table sizes too.
    values = df[num_cols].to_numpy(dtype=float, copy=False)
    valid_mask = ~np.isnan(values)
    index_arr = df.index.to_numpy()

    # 1. Pairwise inequalities (col_b >= col_a). A full i != j sweep is
    # O(cols^2) pairs — on a wide table (hundreds of numeric columns) that's
    # tens of thousands of pairs if nothing gates on relevance. The name-hint
    # regex is essentially free (string match, no array touched), so it's
    # checked FIRST for every pair; only a pair that names an inequality
    # (high>=low, end>=start...) pays for the actual computation. The old
    # "no name hint but empirically >99% support" catch-all is real signal on
    # a normal-width table but becomes noise-prone AND unbounded-cost on a
    # very wide one, so it's capped the same way the additive-identity pass
    # below already caps itself (`3 <= len(num_cols) <= 15`) — here at a
    # wider but still bounded `_MAX_COLS_FOR_UNHINTED_INEQUALITY_SWEEP`.
    allow_unhinted = len(num_cols) <= _MAX_COLS_FOR_UNHINTED_INEQUALITY_SWEEP
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            if i == j:
                continue
            col_a, col_b = num_cols[i], num_cols[j]
            name_hint = bool(
                re.search(r"high|max|end|total|gross|after", str(col_b), re.I)
                and re.search(r"low|min|start|net|before|part", str(col_a), re.I)
            )
            if not name_hint and not allow_unhinted:
                continue

            mask = valid_mask[:, i] & valid_mask[:, j]
            n_valid = int(mask.sum())
            if n_valid < 20:
                continue

            diff = values[mask, j] - values[mask, i]
            satisfied = int((diff >= -1e-6).sum())
            support = satisfied / n_valid

            if support >= min_support and (name_hint or support > 0.99):
                violating_mask = diff < -1e-6
                if violating_mask.any():
                    violating_indices = index_arr[mask][violating_mask].tolist()
                    constraints.append({
                        "type": "inequality",
                        "rule": f"{col_b} >= {col_a}",
                        "support": round(support, 4),
                        "total_checked": n_valid,
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
                    mask = valid_mask[:, i] & valid_mask[:, j] & valid_mask[:, k]
                    n_valid = int(mask.sum())
                    if n_valid < 20:
                        continue
                    sum_diff = np.abs((values[mask, i] + values[mask, j]) - values[mask, k])
                    match_mask = sum_diff < 1e-4
                    support = float(match_mask.sum() / n_valid)
                    if support >= min_support:
                        violating_mask = ~match_mask
                        if violating_mask.any():
                            violating = index_arr[mask][violating_mask].tolist()
                            constraints.append({
                                "type": "sum_identity",
                                "rule": f"{ca} + {cb} == {cc}",
                                "support": round(support, 4),
                                "total_checked": n_valid,
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

    Every (missing_col, other_col) pair is one point-biserial test; a wide
    table with many numeric columns runs dozens of these per missing column,
    so an uncorrected p<0.01 floor alone produces "systematic bias" verdicts
    on genuinely MCAR data far above the nominal 1% rate (empirically ~25%
    of missing columns at 30 numeric columns, n=200). Benjamini-Hochberg
    correction across every test this scan runs -- same utility and
    reasoning `multiple_testing.py` already applies to statistical_analysis's
    test family -- keeps the false-positive rate honest regardless of how
    many columns the table has.

    Performance: this runs on every dataset load (no applies_to gate skips
    it), so both hot loops are vectorized instead of pure Python: block-run
    detection uses a numpy diff trick instead of a per-row Python loop
    (matters at hundreds of thousands of rows), and the pairwise correlation
    test uses one `DataFrame.corrwith()` call per missing column instead of
    one `scipy.stats.pointbiserialr` call per (missing_col, other_col) pair
    -- this loop was previously O(missing_cols * all_numeric_cols), unbounded
    on a wide table. `_MAX_OTHER_COLS_FOR_MISSINGNESS` caps the "other" side
    to the most-complete columns so worst-case cost stays bounded on a very
    wide table; a table under that width is unaffected.
    """
    results: dict[str, Any] = {}
    n = len(df)
    if n < 30:
        return results

    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    missing_cols = [c for c in df.columns if 0.02 * n <= df[c].isna().sum() <= 0.90 * n]
    if not missing_cols or not num_cols:
        return results

    other_cols = num_cols
    if len(num_cols) > _MAX_OTHER_COLS_FOR_MISSINGNESS:
        other_cols = (
            df[num_cols].notna().sum()
            .sort_values(ascending=False)
            .index[:_MAX_OTHER_COLS_FOR_MISSINGNESS]
            .tolist()
        )
    other_df = df[other_cols]
    valid_counts = other_df.notna().sum()

    raw_tests: list[dict[str, Any]] = []
    block_info: dict[str, dict[str, Any]] = {}
    for col in missing_cols:
        miss_mask = df[col].isna()
        block_info[col] = _block_missingness_info(miss_mask.to_numpy())

        candidates = [c for c in other_cols if c != col]
        if not candidates:
            continue
        corrs = other_df[candidates].corrwith(miss_mask.astype(int))
        for other in candidates:
            r = corrs.get(other)
            valid_n = int(valid_counts[other])
            if r is None or pd.isna(r) or valid_n < 20:
                continue
            # r = +/-1 (missingness perfectly predicted by another column,
            # e.g. discount_amount is NaN exactly when has_discount == 0) is
            # the STRONGEST possible signal, not a case to skip — the old
            # `abs(r) >= 1.0: continue` here silently reclassified the most
            # obvious MAR_systematic case as MCAR_likely. The `max(1e-12, ...)`
            # floor already keeps the t-statistic finite at r == +/-1.
            t_stat = float(r) * np.sqrt((valid_n - 2) / max(1e-12, 1.0 - float(r) ** 2))
            p = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=valid_n - 2)))
            raw_tests.append({"missing_col": col, "other": other, "r": float(r), "p_value": p})

    corrected = apply_benjamini_hochberg(raw_tests, alpha=0.05) if raw_tests else []
    tests_by_col: dict[str, list[dict[str, Any]]] = {}
    for t in corrected:
        tests_by_col.setdefault(t["missing_col"], []).append(t)

    for col in missing_cols:
        correlated_vars = [
            {
                "column": t["other"],
                "correlation": round(t["r"], 3),
                "p_value": round(t["p_value"], 6),
                "p_adjusted": t["p_adjusted"],
            }
            for t in tests_by_col.get(col, [])
            if t["significant_after_correction"] and abs(t["r"]) > 0.15
        ]
        max_block = block_info[col]["max_block"]
        has_block_missingness = block_info[col]["has_block_missingness"]

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
            "candidate_columns_scanned": len(other_cols),
            "candidate_columns_total": len(num_cols),
        }
        if len(other_cols) < len(num_cols):
            # Disclose, Don't Hide: a wide table's scan was truncated to the
            # most-complete columns, so an MCAR_likely verdict here is only
            # "no correlation found among the columns checked."
            results[col]["warning"] += (
                f" (Checked the {len(other_cols)} most-complete of {len(num_cols)} numeric "
                "columns; a real correlation with a sparser column could be missed.)"
            )

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
