"""
Statistical sensitivity, jackknife fragility, and target-leakage audits.

Sensitivity results distinguish a completed audit from one that could not be
run.  In particular, an unsupported or skipped audit is never evidence that a
finding is robust.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from src.core.findings import Finding

logger = logging.getLogger(__name__)

#: Semantic indicators of post-outcome or target-derived columns
_POST_OUTCOME_PREFIXES = ("post_", "after_", "resolved_", "exit_", "outcome_")
_POST_OUTCOME_SUFFIXES = ("_after", "_resolved", "_outcome", "_date", "_reason")

#: Below this many distinct feature values relative to row count, the
#: "determines the target" test is vacuous — a near-unique ID or timestamp
#: column trivially has one target value per group. Mirrors
#: `_LEAKAGE_MIN_GROUP_RATIO` in src/tools/ml_pipeline.py.
_LEAKAGE_MIN_GROUP_RATIO = 0.5

AuditStatus = Literal["passed", "failed", "skipped", "not_applicable"]


def _non_passing_result(
    status: Literal["skipped", "not_applicable"],
    diagnosis: str,
) -> dict[str, Any]:
    """Build an explicit result for an audit that did not establish robustness."""
    return {
        "status": status,
        # ``None`` is intentionally distinct from False: legacy consumers that
        # only know ``is_fragile`` must not interpret this as an audit pass.
        "is_fragile": None,
        "fragility_score": None,
        "diagnosis": diagnosis,
        "effect_shift_pct": None,
    }


def _cohens_d(level_values: pd.Series, rest_values: pd.Series) -> float | None:
    """Return the two-sample pooled-standard-deviation effect size."""
    if len(level_values) < 2 or len(rest_values) < 2:
        return None
    level_var = float(level_values.var(ddof=1))
    rest_var = float(rest_values.var(ddof=1))
    denominator_df = len(level_values) + len(rest_values) - 2
    pooled_variance = (
        (len(level_values) - 1) * level_var + (len(rest_values) - 1) * rest_var
    ) / denominator_df
    if pooled_variance <= 0:
        return None
    return float((level_values.mean() - rest_values.mean()) / np.sqrt(pooled_variance))


def _categorical_contrast_effect(
    frame: pd.DataFrame,
    measure: str,
    level: str,
    effect_kind: str | None,
) -> tuple[float | None, np.ndarray, str | None]:
    """Recompute a level-versus-rest effect and return per-row residuals.

    The residuals identify observations most able to distort this exact group
    contrast.  Only effect definitions used by ``segment_comparison`` are
    accepted; guessing a regression coefficient or an omnibus test would make
    a false robustness claim.
    """
    group_values = frame["_dimension"].astype("string")
    in_level = group_values.eq(str(level)).to_numpy()
    level_values = frame.loc[in_level, measure]
    rest_values = frame.loc[~in_level, measure]
    if len(level_values) < 2 or len(rest_values) < 2:
        return None, np.array([], dtype=float), "The selected level or its comparison group has fewer than two rows."

    if effect_kind == "lift":
        rest_mean = float(rest_values.mean())
        if abs(rest_mean) <= 1e-12:
            return None, np.array([], dtype=float), "The comparison-group mean is zero, so lift is undefined."
        effect = float(level_values.mean() / rest_mean - 1.0)
    elif effect_kind == "cohens_d":
        cohens_d = _cohens_d(level_values, rest_values)
        if cohens_d is None:
            return None, np.array([], dtype=float), "The group contrast has no pooled variance for Cohen's d."
        effect = cohens_d
    else:
        return None, np.array([], dtype=float), (
            "Categorical sensitivity requires a supported level-versus-rest effect "
            "(lift or cohens_d)."
        )

    group_means = frame.groupby("_dimension", observed=True)[measure].transform("mean")
    residuals = (frame[measure] - group_means).abs().to_numpy(dtype=float)
    return effect, residuals, None


def _categorical_audit(
    valid_df: pd.DataFrame,
    measure: str,
    dimension: str,
    level: str | None,
    effect_kind: str | None,
    trim_pct: float,
) -> tuple[float, float, int] | dict[str, Any]:
    """Audit a supported categorical level-versus-rest contrast."""
    if level is None:
        return _non_passing_result(
            "not_applicable",
            "Categorical sensitivity needs a named level to reproduce the group contrast.",
        )

    points = valid_df[[dimension, measure]].dropna().rename(columns={dimension: "_dimension"})
    if len(points) < 20:
        return _non_passing_result("skipped", "Insufficient categorical comparison points for leverage trimming.")

    baseline_effect, residuals, error = _categorical_contrast_effect(
        points, measure, level, effect_kind
    )
    if baseline_effect is None or error is not None:
        return _non_passing_result("not_applicable", error or "Unable to reproduce the categorical contrast.")

    k_drop = max(1, int(len(points) * trim_pct))
    top_leverage_idx = points.index[np.argsort(residuals)[-k_drop:]]
    trimmed_points = points.drop(index=top_leverage_idx)
    trimmed_effect, _, trim_error = _categorical_contrast_effect(
        trimmed_points, measure, level, effect_kind
    )
    if trimmed_effect is None or trim_error is not None:
        return _non_passing_result(
            "skipped", trim_error or "Leverage trimming left no valid categorical contrast."
        )
    return baseline_effect, trimmed_effect, k_drop


def audit_finding_sensitivity(
    df: pd.DataFrame,
    finding: dict[str, Any] | Finding,
    trim_pct: float = 0.01,
) -> dict[str, Any]:
    """Audit whether a finding is sensitive to a small set of leverage points.

    A completed result has status ``passed`` or ``failed``. ``skipped`` and
    ``not_applicable`` mean the pipeline has no robustness verdict and set
    ``is_fragile`` to ``None`` rather than silently treating it as robust.
    """
    if not 0.0 < trim_pct < 1.0:
        return _non_passing_result("not_applicable", "trim_pct must be between zero and one.")

    f_dict = finding.to_dict() if hasattr(finding, "to_dict") else dict(finding)
    measure = f_dict.get("measure")
    dimension = f_dict.get("dimension")
    level = f_dict.get("level")
    effect_kind = f_dict.get("effect_kind")
    finding_kind = f_dict.get("kind")

    if not isinstance(measure, str) or measure not in df.columns:
        return _non_passing_result("not_applicable", "The finding has no measure anchored to this dataset.")
    if not pd.api.types.is_numeric_dtype(df[measure]):
        return _non_passing_result("not_applicable", "The finding measure is non-numeric; this audit does not apply.")

    valid_df = df.dropna(subset=[measure])
    if len(valid_df) < 20:
        return _non_passing_result("skipped", "Sample size is too small for robust leverage trimming.")

    base_val: float
    trim_val: float
    k_drop: int

    if isinstance(dimension, str) and dimension in df.columns:
        if pd.api.types.is_numeric_dtype(df[dimension]):
            # A regression driver is not a Pearson correlation even if its
            # effect_kind is ``r``; audit only correlations or legacy findings
            # without a kind, where the historical contract was correlation.
            if finding_kind not in (None, "", "correlation"):
                return _non_passing_result(
                    "not_applicable",
                    "Numeric sensitivity can reproduce correlations, not this finding's model-specific effect.",
                )
            points = valid_df[[dimension, measure]].dropna()
            if len(points) < 20:
                return _non_passing_result("skipped", "Insufficient bivariate points for correlation trimming.")
            try:
                cov = np.cov(points.values, rowvar=False)
                inv_cov = np.linalg.pinv(cov)
                diff = points.values - points.mean().values
                distances = np.sum(diff @ inv_cov * diff, axis=1)
                k_drop = max(1, int(len(points) * trim_pct))
                top_leverage_idx = points.index[np.argsort(distances)[-k_drop:]]
                trimmed_points = points.drop(index=top_leverage_idx)
                base_val = float(stats.pearsonr(points[dimension], points[measure])[0])
                trim_val = float(stats.pearsonr(trimmed_points[dimension], trimmed_points[measure])[0])
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                return _non_passing_result("skipped", "Correlation leverage calculation could not be completed.")
        else:
            if finding_kind not in (None, "", "segment_lift"):
                return _non_passing_result(
                    "not_applicable",
                    "Categorical sensitivity can reproduce a level-versus-rest contrast, not this finding type.",
                )
            categorical_result = _categorical_audit(
                valid_df, measure, dimension, str(level) if level is not None else None,
                str(effect_kind) if effect_kind is not None else None, trim_pct,
            )
            if isinstance(categorical_result, dict):
                return categorical_result
            base_val, trim_val, k_drop = categorical_result
    else:
        if finding_kind not in (None, "", "univariate"):
            return _non_passing_result(
                "not_applicable",
                "Univariate sensitivity cannot reproduce this finding's stated effect.",
            )
        values = valid_df[measure].to_numpy(dtype=float)
        k_drop = max(1, int(len(values) * trim_pct))
        z_scores = np.abs((values - np.mean(values)) / max(np.std(values), 1e-9))
        top_leverage_idx = valid_df.index[np.argsort(z_scores)[-k_drop:]]
        trimmed_df = valid_df.drop(index=top_leverage_idx)
        base_val = float(np.mean(values))
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
        "status": "failed" if is_fragile else "passed",
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
                    logger.debug("near-perfect-correlation check skipped for a column", exc_info=True)

        # 3. Exact categorical mapping (target uniquely determined by feature)
        elif not pd.api.types.is_numeric_dtype(target_s):
            sub = df[[col, target_col]].dropna()
            n_groups = sub[col].nunique()
            if len(sub) >= 10 and 1 < n_groups <= len(sub) * _LEAKAGE_MIN_GROUP_RATIO:
                grouped = sub.groupby(col)[target_col].nunique()
                if (grouped == 1).all():
                    alerts.append({
                        "column": col,
                        "leakage_type": "deterministic_mapping",
                        "severity": "critical",
                        "rationale": f"Every category of '{col}' uniquely and deterministically identifies the target class.",
                    })

    return alerts


def audit_simpsons_paradox(
    df: pd.DataFrame,
    finding: dict[str, Any] | Finding,
    candidate_stratifiers: list[str] | tuple[str, ...] | None = None,
    min_stratum_n: int = 15,
) -> dict[str, Any]:
    """Audit whether a finding suffers from Simpson's Paradox across subgroups.

    Checks whether the sign of a bivariate correlation or categorical contrast reverses
    when conditioning on third-party categorical variables (stratifiers).
    """
    f_dict = finding.to_dict() if hasattr(finding, "to_dict") else dict(finding)
    measure = f_dict.get("measure")
    dimension = f_dict.get("dimension")
    level = f_dict.get("level")
    effect_kind = f_dict.get("effect_kind")
    finding_kind = f_dict.get("kind")

    if not isinstance(measure, str) or measure not in df.columns:
        return _non_passing_result("not_applicable", "The finding has no measure anchored to this dataset.")
    if not pd.api.types.is_numeric_dtype(df[measure]):
        return _non_passing_result("not_applicable", "The finding measure is non-numeric; Simpson's audit does not apply.")
    if not isinstance(dimension, str) or dimension not in df.columns:
        return _non_passing_result("not_applicable", "The finding has no dimension anchored to test for subgroup reversal.")

    # Identify potential stratifiers (confounders)
    if candidate_stratifiers is not None:
        stratifiers = [c for c in candidate_stratifiers if c in df.columns and c not in (measure, dimension)]
    else:
        stratifiers = [
            c for c in df.columns
            if c not in (measure, dimension)
            and 2 <= df[c].nunique(dropna=True) <= 8
            and not pd.api.types.is_numeric_dtype(df[c])
        ]
        # If no non-numeric columns, allow low-cardinality discrete numeric
        if not stratifiers:
            stratifiers = [
                c for c in df.columns
                if c not in (measure, dimension)
                and 2 <= df[c].nunique(dropna=True) <= 6
            ]

    if not stratifiers:
        return {
            "status": "not_applicable",
            "has_paradox": None,
            "is_fragile": None,
            "diagnosis": "No suitable categorical stratifiers available to test subgroup reversal.",
            "stratifier": None,
            "strata_effects": {},
        }

    is_num_dim = pd.api.types.is_numeric_dtype(df[dimension])
    tested_stratifiers: list[str] = []

    for strat in stratifiers[:4]:  # test up to 4 most plausible stratifiers
        valid = df[[dimension, measure, strat]].dropna()
        if len(valid) < max(30, min_stratum_n * 2):
            continue

        strata_vals = valid[strat].unique()
        strata_effects: dict[str, float] = {}

        if is_num_dim:
            if finding_kind not in (None, "", "correlation"):
                continue
            try:
                base_r, _ = stats.pearsonr(valid[dimension], valid[measure])
            except Exception:
                continue
            if abs(base_r) < 0.12:
                continue

            base_sign = 1 if base_r > 0 else -1
            reversal_count = 0
            valid_strata_count = 0

            for s_val in strata_vals:
                sub = valid[valid[strat] == s_val]
                if len(sub) < min_stratum_n:
                    continue
                try:
                    s_r, _ = stats.pearsonr(sub[dimension], sub[measure])
                    if np.isnan(s_r):
                        continue
                    strata_effects[str(s_val)] = round(float(s_r), 3)
                    valid_strata_count += 1
                    if (1 if s_r > 0 else -1) != base_sign and abs(s_r) >= 0.08:
                        reversal_count += 1
                except Exception:
                    continue

            if valid_strata_count >= 2:
                tested_stratifiers.append(strat)
                if reversal_count >= 2 and (reversal_count / valid_strata_count) >= 0.60:
                    strata_str = ", ".join(f"{k}: {v:+0.2f}" for k, v in strata_effects.items())
                    return {
                        "status": "failed",
                        "has_paradox": True,
                        "is_fragile": True,
                        "stratifier": strat,
                        "aggregate_effect": round(float(base_r), 3),
                        "strata_effects": strata_effects,
                        "diagnosis": (
                            f"Simpson's Paradox detected: overall correlation ({base_r:+0.2f}) "
                            f"reverses within subgroups of '{strat}' ({strata_str}). "
                            f"Controlling for '{strat}' inverts the direction."
                        ),
                    }
        else:
            # Categorical dimension: level-vs-rest contrast
            if level is None:
                continue
            base_effect_val, _, err = _categorical_contrast_effect(
                valid.rename(columns={dimension: "_dimension"}),
                measure,
                str(level),
                str(effect_kind or "lift"),
            )
            if base_effect_val is None or err is not None or abs(base_effect_val) < 0.04:
                continue

            base_sign = 1 if base_effect_val > 0 else -1
            reversal_count = 0
            valid_strata_count = 0

            for s_val in strata_vals:
                sub = valid[valid[strat] == s_val].rename(columns={dimension: "_dimension"})
                if len(sub) < min_stratum_n:
                    continue
                s_effect, _, s_err = _categorical_contrast_effect(
                    sub, measure, str(level), str(effect_kind or "lift")
                )
                if s_effect is None or s_err is not None:
                    continue
                strata_effects[str(s_val)] = round(float(s_effect), 3)
                valid_strata_count += 1
                if (1 if s_effect > 0 else -1) != base_sign and abs(s_effect) >= 0.02:
                    reversal_count += 1

            if valid_strata_count >= 2:
                tested_stratifiers.append(strat)
                if reversal_count >= 2 and (reversal_count / valid_strata_count) >= 0.60:
                    strata_str = ", ".join(f"{k}: {v:+0.2f}" for k, v in strata_effects.items())
                    return {
                        "status": "failed",
                        "has_paradox": True,
                        "is_fragile": True,
                        "stratifier": strat,
                        "aggregate_effect": round(float(base_effect_val), 3),
                        "strata_effects": strata_effects,
                        "diagnosis": (
                            f"Simpson's Paradox detected: overall {effect_kind or 'lift'} ({base_effect_val:+0.2f}) "
                            f"reverses within subgroups of '{strat}' ({strata_str}). "
                            f"Controlling for '{strat}' inverts the conclusion."
                        ),
                    }

    if tested_stratifiers:
        return {
            "status": "passed",
            "has_paradox": False,
            "is_fragile": False,
            "stratifier": None,
            "tested_stratifiers": tested_stratifiers,
            "strata_effects": {},
            "diagnosis": f"Robust across subgroup stratification (tested on: {', '.join(tested_stratifiers)}); no Simpson's reversal detected.",
        }

    return {
        "status": "skipped",
        "has_paradox": None,
        "is_fragile": None,
        "stratifier": None,
        "strata_effects": {},
        "diagnosis": "Insufficient observations per subgroup stratum to evaluate Simpson's paradox.",
    }

