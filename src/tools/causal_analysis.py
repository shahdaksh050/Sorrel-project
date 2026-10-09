"""Guarded causal diagnostics and a conservative binary-treatment estimate.

This execution-layer tool deliberately does not infer a causal graph.  It
only estimates an average treatment effect after reporting the observable
requirements that the data can (and cannot) check.  In particular, an
observational estimate remains conditional on the no-unmeasured-confounding
assumption; callers must not present it as proof of causation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.findings import Finding
from src.core.memory import MemorySystem
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

_MIN_ARM_N = 30
_MIN_COMPLETE_N = 100
_BOOTSTRAPS = 300
_RANDOM_STATE = 42
_PROPENSITY_LOW = 0.05
_PROPENSITY_HIGH = 0.95
_MIN_OVERLAP_SHARE = 0.90
_MAX_WEIGHT = 20.0
_POST_TREATMENT_TOKENS = ("post", "after", "mediator", "intermediate", "followup")
_SUPPORTED_DESIGNS = {"observational", "randomized_experiment"}


def _binary_treatment(values: pd.Series) -> tuple[pd.Series, dict[str, str]]:
    """Convert exactly two non-null treatment labels to a deterministic 0/1 coding."""
    non_null = values.dropna()
    levels = sorted(str(value) for value in non_null.unique())
    if len(levels) != 2:
        raise ToolExecutionError("treatment must contain exactly two non-missing levels.")
    mapping = {levels[0]: 0, levels[1]: 1}
    coded = values.astype(str).map(mapping).where(values.notna())
    return coded, {"control": levels[0], "treated": levels[1]}


def _is_post_treatment(name: str) -> bool:
    lowered = name.lower().replace("_", " ").replace("-", " ")
    return any(token in lowered.split() for token in _POST_TREATMENT_TOKENS)


def _smd(values: pd.Series, treatment: pd.Series, weights: np.ndarray | None = None) -> float | None:
    """Absolute standardized mean difference for one numeric/binary covariate."""
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    t = treatment.to_numpy(dtype=int)
    valid = np.isfinite(x)
    x, t = x[valid], t[valid]
    if len(x) == 0 or len(np.unique(t)) != 2:
        return None
    if weights is None:
        first, second = x[t == 1], x[t == 0]
        variance = (float(np.var(first)) + float(np.var(second))) / 2
        return 0.0 if variance == 0 else abs(float(np.mean(first) - np.mean(second))) / np.sqrt(variance)
    w = weights[valid]
    means = [np.average(x[t == group], weights=w[t == group]) for group in (0, 1)]
    variances = [np.average((x[t == group] - means[group]) ** 2, weights=w[t == group]) for group in (0, 1)]
    variance = float(sum(variances) / 2)
    return 0.0 if variance == 0 else abs(float(means[1] - means[0])) / np.sqrt(variance)


class CausalAnalysisTool(BaseTool):
    """Diagnose overlap/balance and estimate an IPW treatment contrast safely."""

    name: str = "causal_analysis"
    description: str = (
        "Guarded binary-treatment causal diagnostic. Checks treatment overlap and covariate "
        "balance, then reports a stabilized inverse-probability-weighted outcome contrast "
        "with a deterministic bootstrap interval. It refuses inadequate overlap or unsupported "
        "designs and never turns observational associations into causal proof."
    )
    requires_ml: ClassVar[bool] = True

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        """Never auto-schedule causal estimation from suggestive column names alone."""
        return 0.0

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        """Only a validated design record may authorize a non-observational label."""
        prepared = super().prepare_params(params, memory, output_root)
        design = memory.get_context("causal_design")
        validated_design = "observational"
        if isinstance(design, dict):
            candidate = design.get("study_design")
            if candidate == "randomized_experiment" and bool(
                design.get("randomization_verified")
            ):
                validated_design = "randomized_experiment"
            if not prepared.get("outcome") and isinstance(design.get("outcome"), str):
                prepared["outcome"] = design["outcome"]
            if not prepared.get("treatment") and isinstance(design.get("treatment"), str):
                prepared["treatment"] = design["treatment"]
            if not prepared.get("covariates"):
                covariates = design.get("pre_treatment_confounders")
                if isinstance(covariates, list) and all(isinstance(value, str) for value in covariates):
                    prepared["covariates"] = covariates
        prepared["study_design"] = validated_design
        return prepared

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        outcome: str,
        treatment: str,
        covariates: list[str] | None = None,
        method: str = "ipw",
        study_design: str = "observational",
        **_: Any,
    ) -> dict[str, Any]:
        if method not in {"auto", "ipw"}:
            return self._not_identifiable(f"Unsupported method '{method}'; only IPW is available.")
        if study_design not in _SUPPORTED_DESIGNS:
            return self._not_identifiable(
                f"Unsupported study_design '{study_design}'; provide observational or randomized_experiment."
            )
        df = _read_df(file_path)
        requested_covariates = list(covariates or [])
        required = [outcome, treatment, *requested_covariates]
        missing = sorted(set(required).difference(df.columns))
        if missing:
            raise ToolExecutionError(f"Column(s) not found: {', '.join(missing)}.")
        if outcome == treatment or outcome in requested_covariates or treatment in requested_covariates:
            raise ToolExecutionError("outcome, treatment, and covariates must be distinct columns.")
        suspicious = [name for name in requested_covariates if _is_post_treatment(name)]
        if suspicious:
            return self._not_identifiable(
                "Potential post-treatment covariates were supplied: " + ", ".join(suspicious) + "."
            )
        if not requested_covariates:
            return self._not_identifiable("At least one pre-treatment covariate is required for IPW adjustment.")

        y = pd.to_numeric(df[outcome], errors="coerce")
        t, labels = _binary_treatment(df[treatment])
        work = df.loc[y.notna() & t.notna(), [*requested_covariates]].copy()
        y = y.loc[work.index].reset_index(drop=True)
        t = t.loc[work.index].astype(int).reset_index(drop=True)
        work = work.reset_index(drop=True)
        if len(work) < _MIN_COMPLETE_N:
            return self._not_identifiable(f"Need at least {_MIN_COMPLETE_N} usable rows; found {len(work)}.")
        arm_counts = t.value_counts()
        if min(int(arm_counts.get(0, 0)), int(arm_counts.get(1, 0))) < _MIN_ARM_N:
            return self._not_identifiable(f"Each treatment arm needs at least {_MIN_ARM_N} rows.")

        numeric = [name for name in requested_covariates if pd.api.types.is_numeric_dtype(work[name])]
        categorical = [name for name in requested_covariates if name not in numeric]
        high_cardinality = [name for name in categorical if work[name].nunique(dropna=True) > 20]
        if high_cardinality:
            return self._not_identifiable(
                "Categorical covariates have unsafe cardinality for this diagnostic: " + ", ".join(high_cardinality) + "."
            )
        transformer = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                    numeric,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical,
                ),
            ],
            remainder="drop",
        )
        propensity_model = Pipeline([
            ("features", transformer),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    max_iter=1000,
                    random_state=_RANDOM_STATE,
                ),
            ),
        ])
        try:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=_RANDOM_STATE)
            propensity = np.empty(len(work), dtype=float)
            for train_index, test_index in splitter.split(work, t):
                fold_model = clone(propensity_model)
                fold_model.fit(work.iloc[train_index], t.iloc[train_index])
                propensity[test_index] = fold_model.predict_proba(work.iloc[test_index])[:, 1]
        except (ValueError, TypeError) as exc:
            raise ToolExecutionError(f"Could not fit propensity model: {exc}") from exc
        overlap_share = float(np.mean((propensity >= _PROPENSITY_LOW) & (propensity <= _PROPENSITY_HIGH)))
        raw_weights = np.where(t.to_numpy() == 1, 1.0 / propensity, 1.0 / (1.0 - propensity))
        if overlap_share < _MIN_OVERLAP_SHARE or float(np.max(raw_weights)) > _MAX_WEIGHT:
            return self._not_identifiable(
                f"Insufficient propensity overlap (share in [{_PROPENSITY_LOW}, {_PROPENSITY_HIGH}] = {overlap_share:.1%}; "
                f"max IPW weight = {float(np.max(raw_weights)):.1f}).",
                overlap_share=overlap_share,
                max_weight=float(np.max(raw_weights)),
            )
        marginal_treated = float(t.mean())
        weights = np.where(t.to_numpy() == 1, marginal_treated / propensity, (1.0 - marginal_treated) / (1.0 - propensity))
        treated_mean = float(np.average(y[t == 1], weights=weights[t.to_numpy() == 1]))
        control_mean = float(np.average(y[t == 0], weights=weights[t.to_numpy() == 0]))
        estimate = treated_mean - control_mean
        ci_low, ci_high = self._bootstrap_interval(y.to_numpy(), t.to_numpy(), weights)
        balance = self._balance(work, t, numeric, categorical, weights)
        max_after_smd = max((row["after_smd"] for row in balance if row["after_smd"] is not None), default=0.0)
        caveats = [
            "Observational estimates require no unmeasured confounding, correct propensity specification, consistency, and positivity.",
            "Column names cannot prove temporal order; confirm every covariate was measured before treatment.",
        ]
        if study_design == "randomized_experiment":
            caveats[0] = "Randomization must be verified externally; this tool does not verify treatment assignment."
        if max_after_smd > 0.1:
            caveats.append("Residual covariate imbalance exceeds 0.10 SMD; interpret estimate cautiously.")
        wording = "adjusted association" if study_design == "observational" else "design-conditional treatment contrast"
        return {
            "summary": f"{wording.capitalize()} for {outcome}: {estimate:.4g} (95% bootstrap interval {ci_low:.4g} to {ci_high:.4g}).",
            "status": "estimated_with_assumptions",
            "identification_status": "conditionally_identified",
            "study_design": study_design,
            "method": "stabilized_ipw",
            "propensity_cross_fitting": {"folds": 5, "shuffle": True, "random_state": _RANDOM_STATE},
            "outcome": outcome,
            "treatment": treatment,
            "treatment_levels": labels,
            "n": len(work),
            "arm_counts": {"control": int(arm_counts[0]), "treated": int(arm_counts[1])},
            "estimate": estimate,
            "confidence_interval_95": [ci_low, ci_high],
            "overlap": {"share_in_common_range": overlap_share, "max_raw_weight": float(np.max(raw_weights))},
            "balance": balance,
            "max_after_smd": max_after_smd,
            "assumptions": caveats,
            "warnings": caveats,
            # A caller needs external evidence of assignment and temporal order
            # before upgrading this output to a causal claim.
            "causal_claim_permitted": False,
        }

    @staticmethod
    def _bootstrap_interval(y: np.ndarray, t: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
        rng = np.random.default_rng(_RANDOM_STATE)
        estimates = np.empty(_BOOTSTRAPS)
        for index in range(_BOOTSTRAPS):
            rows = rng.integers(0, len(y), len(y))
            sampled_t, sampled_y, sampled_w = t[rows], y[rows], weights[rows]
            if not np.any(sampled_t == 0) or not np.any(sampled_t == 1):
                estimates[index] = np.nan
                continue
            estimates[index] = (
                np.average(sampled_y[sampled_t == 1], weights=sampled_w[sampled_t == 1])
                - np.average(sampled_y[sampled_t == 0], weights=sampled_w[sampled_t == 0])
            )
        quantiles = np.nanquantile(estimates, [0.025, 0.975])
        return float(quantiles[0]), float(quantiles[1])

    @staticmethod
    def _balance(
        work: pd.DataFrame,
        treatment: pd.Series,
        numeric: list[str],
        categorical: list[str],
        weights: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Report numeric SMDs and the worst SMD across each categorical's levels."""
        rows = [
            {
                "covariate": name,
                "before_smd": _smd(work[name], treatment),
                "after_smd": _smd(work[name], treatment, weights),
            }
            for name in numeric
        ]
        for name in categorical:
            dummy = pd.get_dummies(work[name].fillna("<missing>"), dtype=float)
            before = [_smd(dummy[column], treatment) for column in dummy.columns]
            after = [_smd(dummy[column], treatment, weights) for column in dummy.columns]
            rows.append({
                "covariate": name,
                "before_smd": max((value for value in before if value is not None), default=None),
                "after_smd": max((value for value in after if value is not None), default=None),
            })
        return rows

    @staticmethod
    def _not_identifiable(reason: str, **diagnostics: Any) -> dict[str, Any]:
        return {
            "summary": f"Causal estimate not identified: {reason}",
            "status": "not_identifiable",
            "identification_status": "not_identified",
            "reason": reason,
            "diagnostics": diagnostics,
            "assumptions": [],
            "warnings": [reason],
            "causal_claim_permitted": False,
        }

    def findings(self, output: dict[str, Any], profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> list[Finding]:
        if output.get("status") != "estimated_with_assumptions":
            return [Finding(finding_id="causal_not_identifiable", kind="method_fit", headline=str(output["summary"]), evidence=output, source_tool=self.name, confidence=0.95)]
        return [Finding(
            finding_id=f"causal_{output['outcome']}_{output['treatment']}", kind="test", headline=str(output["summary"]),
            detail="Conditional estimate; causal interpretation depends on the listed assumptions.", evidence=output,
            source_tool=self.name, measure=str(output["outcome"]), dimension=str(output["treatment"]),
            effect=float(output["estimate"]), effect_kind="estimate", confidence=0.7,
            caveats=list(output.get("assumptions", [])),
        )]

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to dataset.", "required": True},
            "outcome": {"type": "string", "description": "Numeric outcome column.", "required": True},
            "treatment": {"type": "string", "description": "Binary treatment column.", "required": True},
            "covariates": {"type": "array", "description": "Known pre-treatment covariate columns.", "required": True},
            "method": {"type": "string", "description": "auto or ipw.", "required": False},
            "study_design": {"type": "string", "description": "observational or randomized_experiment.", "required": False},
        }


class DifferenceInDifferencesTool(BaseTool):
    """Guarded panel Difference-in-Differences estimator with cluster-robust standard errors."""

    name: str = "difference_in_differences"
    description: str = (
        "Panel Difference-in-Differences (DiD) estimator. Evaluates pre-treatment "
        "parallel trends, fits two-way group/time contrast with cluster-robust standard errors, "
        "and checks for differential pre-trends. It refuses data lacking pre/post periods or entity clusters "
        "and enforces conditional interpretation."
    )
    requires_ml: ClassVar[bool] = False

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        """Only explicitly scheduled when a panel/DiD design is validated."""
        return 0.0

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        prepared = super().prepare_params(params, memory, output_root)
        design = memory.get_context("causal_design")
        if isinstance(design, dict):
            if not prepared.get("outcome") and isinstance(design.get("outcome"), str):
                prepared["outcome"] = design["outcome"]
            if not prepared.get("treatment") and isinstance(design.get("treatment"), str):
                prepared["treatment"] = design["treatment"]
            if not prepared.get("time_column") and isinstance(design.get("time_column"), str):
                prepared["time_column"] = design["time_column"]
            if not prepared.get("entity_column") and isinstance(design.get("entity_column"), str):
                prepared["entity_column"] = design["entity_column"]
            if not prepared.get("covariates"):
                covariates = design.get("pre_treatment_confounders")
                if isinstance(covariates, list) and all(isinstance(v, str) for v in covariates):
                    prepared["covariates"] = covariates
        return prepared

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        outcome: str,
        treatment: str,
        time_column: str,
        entity_column: str,
        post_treatment_time: Any | None = None,
        covariates: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        covs = list(covariates or [])
        required = [outcome, treatment, time_column, entity_column, *covs]
        missing = sorted(set(required).difference(df.columns))
        if missing:
            raise ToolExecutionError(f"Column(s) not found: {', '.join(missing)}.")

        cols_distinct = [outcome, treatment, time_column, entity_column]
        if len(set(cols_distinct)) != len(cols_distinct):
            raise ToolExecutionError("outcome, treatment, time_column, and entity_column must be distinct columns.")

        suspicious = [name for name in covs if _is_post_treatment(name)]
        if suspicious:
            return self._not_identifiable(
                "Potential post-treatment covariates were supplied: " + ", ".join(suspicious) + "."
            )

        work = df[required].dropna()
        if len(work) < _MIN_ARM_N:
            return self._not_identifiable(f"Insufficient observations (n={len(work)} < {_MIN_ARM_N}).")

        # Encode treatment to 0/1
        coded_treatment, levels_map = _binary_treatment(work[treatment])
        work = work.assign(_treated=coded_treatment)

        # Encode pre/post
        time_series = work[time_column]
        if post_treatment_time is not None:
            try:
                num_time = pd.to_numeric(time_series)
                num_cutoff = float(post_treatment_time)
                is_post = (num_time >= num_cutoff).astype(int)
            except (ValueError, TypeError):
                is_post = (time_series.astype(str) >= str(post_treatment_time)).astype(int)
        else:
            unique_times = sorted(time_series.unique())
            if len(unique_times) == 2:
                is_post = (time_series == unique_times[1]).astype(int)
            elif len(unique_times) > 2:
                if set(unique_times) == {0, 1} or set(unique_times) == {"0", "1"}:
                    is_post = (time_series == unique_times[1]).astype(int)
                else:
                    return self._not_identifiable(
                        f"time_column '{time_column}' has {len(unique_times)} periods; specify 'post_treatment_time' cutoff."
                    )
            else:
                return self._not_identifiable(f"time_column '{time_column}' has only 1 unique time period.")

        work = work.assign(_post=is_post, _did=work["_treated"] * is_post)

        cell_counts = {
            "ctrl_pre": int(((work["_treated"] == 0) & (work["_post"] == 0)).sum()),
            "ctrl_post": int(((work["_treated"] == 0) & (work["_post"] == 1)).sum()),
            "treat_pre": int(((work["_treated"] == 1) & (work["_post"] == 0)).sum()),
            "treat_post": int(((work["_treated"] == 1) & (work["_post"] == 1)).sum()),
        }
        if any(c == 0 for c in cell_counts.values()):
            return self._not_identifiable(
                f"Missing observations in 2x2 DiD design: {cell_counts}."
            )

        pre_data = work[work["_post"] == 0]
        unique_pre_times = pre_data[time_column].nunique()
        parallel_trends_test: dict[str, Any] = {"status": "untested_single_pre_period"}
        caveats = [
            "Parallel trends assumption is untestable for the post-treatment counterfactual.",
            "Conditional estimate: relies on absence of unmeasured contemporaneous shocks.",
        ]

        if unique_pre_times >= 2:
            try:
                pre_t = pd.to_numeric(pre_data[time_column], errors="coerce")
                if pre_t.notna().all():
                    x_pre = np.column_stack([
                        np.ones(len(pre_data)),
                        pre_data["_treated"].to_numpy(dtype=float),
                        pre_t.to_numpy(dtype=float),
                        (pre_data["_treated"] * pre_t).to_numpy(dtype=float),
                    ])
                    y_pre = pd.to_numeric(pre_data[outcome], errors="coerce").to_numpy(dtype=float)
                    beta_pre = np.linalg.lstsq(x_pre, y_pre, rcond=None)[0]
                    res_pre = y_pre - x_pre @ beta_pre
                    dof_pre = len(y_pre) - x_pre.shape[1]
                    s2_pre = float((res_pre @ res_pre) / max(1, dof_pre))
                    cov_pre = np.linalg.pinv(x_pre.T @ x_pre) * s2_pre
                    se_diff = float(np.sqrt(max(1e-12, cov_pre[3, 3])))
                    t_pre = float(beta_pre[3] / se_diff)
                    p_trend = float(2 * (1 - stats.t.cdf(abs(t_pre), max(1, dof_pre))))
                    if p_trend < 0.05:
                        parallel_trends_test = {
                            "status": "violated",
                            "p_value": round(p_trend, 4),
                            "slope_difference": round(float(beta_pre[3]), 4),
                        }
                        caveats.append(
                            f"Parallel pre-trends test failed (p={p_trend:.4f}): pre-intervention trends significantly differ."
                        )
                    else:
                        parallel_trends_test = {
                            "status": "passed",
                            "p_value": round(p_trend, 4),
                            "slope_difference": round(float(beta_pre[3]), 4),
                        }
            except Exception:
                parallel_trends_test = {"status": "error_during_pre_trend_check"}

        y = pd.to_numeric(work[outcome], errors="coerce").to_numpy(dtype=float)
        x_cols = [
            np.ones(len(work)),
            work["_treated"].to_numpy(dtype=float),
            work["_post"].to_numpy(dtype=float),
            work["_did"].to_numpy(dtype=float),
        ]
        for c in covs:
            col_num = pd.to_numeric(work[c], errors="coerce").fillna(0.0)
            x_cols.append(col_num.to_numpy(dtype=float))
        x_mat = np.column_stack(x_cols)

        beta, _residuals, _rank, _ = np.linalg.lstsq(x_mat, y, rcond=None)
        res = y - x_mat @ beta

        clusters = work[entity_column].to_numpy()
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)
        n_obs, n_features = x_mat.shape

        if n_clusters < 2:
            return self._not_identifiable(
                f"Cluster variable '{entity_column}' has fewer than 2 clusters."
            )

        inv_xx = np.linalg.pinv(x_mat.T @ x_mat)
        meat = np.zeros((n_features, n_features))
        for cluster in unique_clusters:
            idx = (clusters == cluster)
            x_cluster = x_mat[idx]
            res_g = res[idx]
            score_g = x_cluster.T @ res_g
            meat += np.outer(score_g, score_g)

        dfc = (n_clusters / max(1, n_clusters - 1)) * ((n_obs - 1) / max(1, n_obs - n_features))
        cov_cluster = inv_xx @ meat @ inv_xx * dfc

        did_idx = 3
        estimate = float(beta[did_idx])
        se = float(np.sqrt(max(1e-12, cov_cluster[did_idx, did_idx])))
        dof = max(1, n_clusters - 1)
        t_stat = float(estimate / se) if se > 0 else 0.0
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), dof)))
        t_crit = float(stats.t.ppf(0.975, dof))
        ci_lower = float(estimate - t_crit * se)
        ci_upper = float(estimate + t_crit * se)

        if n_clusters < 20:
            caveats.append(f"Number of clusters is small (G={n_clusters} < 20); cluster-robust standard errors may be downward biased.")

        summary = (
            f"Difference-in-Differences estimate for '{outcome}' (treatment: '{treatment}'): "
            f"{estimate:.4f} (SE: {se:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}], p={p_val:.4f})."
        )

        return {
            "summary": summary,
            "status": "estimated_with_assumptions",
            "identification_status": "partially_identified",
            "method": "panel_difference_in_differences",
            "outcome": outcome,
            "treatment": treatment,
            "time_column": time_column,
            "entity_column": entity_column,
            "levels": levels_map,
            "cell_counts": cell_counts,
            "estimate": round(estimate, 4),
            "std_err": round(se, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_val, 4),
            "ci": (round(ci_lower, 4), round(ci_upper, 4)),
            "n_clusters": n_clusters,
            "parallel_trends": parallel_trends_test,
            "assumptions": caveats,
            "warnings": caveats,
            "causal_claim_permitted": False,
        }

    @staticmethod
    def _not_identifiable(reason: str, **diagnostics: Any) -> dict[str, Any]:
        return {
            "summary": f"DiD causal estimate not identified: {reason}",
            "status": "not_identifiable",
            "identification_status": "not_identified",
            "reason": reason,
            "diagnostics": diagnostics,
            "assumptions": [],
            "warnings": [reason],
            "causal_claim_permitted": False,
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        if output.get("status") != "estimated_with_assumptions":
            return [
                Finding(
                    finding_id="did_not_identifiable",
                    kind="method_fit",
                    headline=str(output["summary"]),
                    evidence=output,
                    source_tool=self.name,
                    confidence=0.95,
                )
            ]
        return [
            Finding(
                finding_id=f"did_{output['outcome']}_{output['treatment']}",
                kind="test",
                headline=str(output["summary"]),
                detail="Panel Difference-in-Differences estimate; causal interpretation requires parallel trends.",
                evidence=output,
                source_tool=self.name,
                measure=str(output["outcome"]),
                dimension=str(output["treatment"]),
                effect=float(output["estimate"]),
                effect_kind="estimate",
                confidence=0.75,
                caveats=list(output.get("assumptions", [])),
            )
        ]

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to dataset.", "required": True},
            "outcome": {"type": "string", "description": "Numeric outcome column.", "required": True},
            "treatment": {"type": "string", "description": "Binary treatment indicator column.", "required": True},
            "time_column": {"type": "string", "description": "Time period column.", "required": True},
            "entity_column": {"type": "string", "description": "Unit / entity column for cluster-robust SEs.", "required": True},
            "post_treatment_time": {"type": "any", "description": "Cutoff dividing pre from post.", "required": False},
            "covariates": {"type": "array", "description": "Optional pre-treatment controls.", "required": False},
        }
