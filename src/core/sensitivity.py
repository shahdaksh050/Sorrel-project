"""
Statistical Sensitivity, Jackknife Fragility, and Target Leakage Audits.

Provides rigorous guards against brittle findings driven by extreme leverage points
and target leakage in predictive features.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from src.core.findings import Finding

#: Semantic indicators of post-outcome or target-derived columns
_POST_OUTCOME_PREFIXES = ("post_", "after_", "resolved_", "exit_", "outcome_")
_POST_OUTCOME_SUFFIXES = ("_after", "_resolved", "_outcome", "_date", "_reason")


def audit_finding_sensitivity(
    df: pd.DataFrame,
    finding: dict[str, Any] | Finding,
    trim_pct: float = 0.01,
) -> dict[str, Any]:
    """
    Audit whether a quantitative finding relies critically on a tiny fraction
    of extreme leverage points (Jackknife Fragility).

    Drops the top 1% leverage observations and checks if the metric/effect shifts
    dramatically or flips sign.
    """
    f_dict = finding.to_dict() if hasattr(finding, "to_dict") else dict(finding)
    measure = f_dict.get("measure")
    dimension = f_dict.get("dimension")

    if not measure or measure not in df.columns or not pd.api.types.is_numeric_dtype(df[measure]):
        return {
            "is_fragile": False,
            "fragility_score": 0.0,
            "diagnosis": "Non-numeric or unanchored measure; sensitivity audit skipped.",
            "effect_shift_pct": 0.0,
        }

    valid_df = df.dropna(subset=[measure])
    if len(valid_df) < 20:
        return {
            "is_fragile": False,
            "fragility_score": 0.0,
            "diagnosis": "Sample size too small for robust leverage trimming.",
            "effect_shift_pct": 0.0,
        }

    # Identify top leverage points
    if dimension and dimension in df.columns and pd.api.types.is_numeric_dtype(df[dimension]):
        # Bivariate leverage (Mahalanobis distance on (dimension, measure))
        pts = valid_df[[dimension, measure]].dropna()
        if len(pts) < 20:
            return {"is_fragile": False, "fragility_score": 0.0, "diagnosis": "Insufficient bivariate points."}
        try:
            cov = np.cov(pts.values, rowvar=False)
            inv_cov = np.linalg.pinv(cov)
            diff = pts.values - pts.mean().values
            dists = np.sum(diff @ inv_cov * diff, axis=1)
            k_drop = max(1, int(len(pts) * trim_pct))
            top_leverage_idx = pts.index[np.argsort(dists)[-k_drop:]]
            trimmed_pts = pts.drop(index=top_leverage_idx)

            r_base, _ = stats.pearsonr(pts[dimension], pts[measure])
            r_trim, _ = stats.pearsonr(trimmed_pts[dimension], trimmed_pts[measure])

            base_val = float(r_base)
            trim_val = float(r_trim)
        except Exception:
            return {"is_fragile": False, "fragility_score": 0.0, "diagnosis": "Leverage calculation error."}
    else:
        # Univariate leverage (Z-score / extreme values on measure)
        vals = valid_df[measure].values
        k_drop = max(1, int(len(vals) * trim_pct))
        z_scores = np.abs((vals - np.mean(vals)) / max(np.std(vals), 1e-9))
        top_leverage_idx = valid_df.index[np.argsort(z_scores)[-k_drop:]]
        trimmed_df = valid_df.drop(index=top_leverage_idx)

        base_val = float(np.mean(vals))
        trim_val = float(trimmed_df[measure].mean())

    denom = abs(base_val) if abs(base_val) > 1e-5 else 1.0
    shift_pct = abs(base_val - trim_val) / denom
    sign_flipped = (base_val * trim_val) < 0 and abs(base_val) > 0.05
    is_fragile = shift_pct > 0.30 or sign_flipped
    fragility_score = min(1.0, shift_pct / 0.5)

    if is_fragile:
        diagnosis = (
            f"Fragile finding: dropping top {trim_pct * 100:.0f}% leverage points ({k_drop} row(s)) "
            f"shifts effect by {shift_pct * 100:.1f}% ({base_val:.3f} -> {trim_val:.3f})"
            + (" with sign reversal." if sign_flipped else ".")
        )
    else:
        diagnosis = (
            f"Robust finding: effect persists under {trim_pct * 100:.0f}% leverage trimming "
            f"({base_val:.3f} vs {trim_val:.3f}, {shift_pct * 100:.1f}% change)."
        )

    return {
        "is_fragile": is_fragile,
        "fragility_score": round(fragility_score, 3),
        "baseline_effect": round(base_val, 4),
        "trimmed_effect": round(trim_val, 4),
        "effect_shift_pct": round(shift_pct * 100, 2),
        "trimmed_rows": k_drop,
        "diagnosis": diagnosis,
    }


def detect_target_leakage(df: pd.DataFrame, target_col: str) -> list[dict[str, Any]]:
    """
    Inspect dataset features for suspicious target leakage risks.
    Detects extreme statistical correlation, semantic post-outcome naming,
    and identifier separation.
    """
    alerts: list[dict[str, Any]] = []
    if target_col not in df.columns:
        return alerts

    target_s = df[target_col]
    target_clean = re.sub(r"[^a-z0-9]", "", target_col.lower())

    for col in df.columns:
        if col == target_col:
            continue

        c_clean = re.sub(r"[^a-z0-9]", "", str(col).lower())

        # 1. Semantic post-outcome / target copy leakage
        if c_clean == target_clean or (c_clean.endswith(target_clean) and len(c_clean) > len(target_clean)):
            alerts.append({
                "column": col,
                "leakage_type": "semantic_target_derivative",
                "severity": "critical",
                "rationale": f"Column '{col}' name strongly derives from target column '{target_col}'.",
            })
            continue

        if any(str(col).lower().startswith(p) for p in _POST_OUTCOME_PREFIXES) or any(
            str(col).lower().endswith(s) for s in _POST_OUTCOME_SUFFIXES
        ):
            alerts.append({
                "column": col,
                "leakage_type": "temporal_post_outcome",
                "severity": "warning",
                "rationale": f"Column '{col}' appears to record post-outcome or resolution information.",
            })

        # 2. Extreme statistical correlation
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(target_s):
            sub = df[[col, target_col]].dropna()
            if len(sub) >= 10:
                try:
                    r, _ = stats.pearsonr(sub[col], sub[target_col])
                    if abs(r) >= 0.98:
                        alerts.append({
                            "column": col,
                            "leakage_type": "perfect_correlation",
                            "severity": "critical",
                            "metric_value": round(float(r), 4),
                            "rationale": f"Near-perfect Pearson correlation (|r| = {abs(r):.3f}) with target.",
                        })
                except Exception:
                    pass

        # 3. Exact categorical mapping (target uniquely determined by feature)
        elif not pd.api.types.is_numeric_dtype(target_s):
            sub = df[[col, target_col]].dropna()
            if len(sub) >= 10 and sub[col].nunique() > 1:
                grouped = sub.groupby(col)[target_col].nunique()
                if (grouped == 1).all():
                    alerts.append({
                        "column": col,
                        "leakage_type": "deterministic_mapping",
                        "severity": "critical",
                        "rationale": f"Every category of '{col}' uniquely and deterministically identifies the target class.",
                    })

    return alerts
