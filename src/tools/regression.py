"""
Regression Analysis Tool — Execution Layer.

"What drives this outcome, holding the other factors fixed, and by how
much?" — an inferential model, not a predictive one. It fits exactly one
model family per call (OLS for a continuous target, logistic regression for
a binary one) on a deliberately lean predictor set and reports, in plain
English, the adjusted effect of each driver with a confidence interval.

Why this is not the ML pipeline: no estimator search, no hyperparameters,
nothing persisted (no `output_subdir`, no model file). Cross-validation is
used only to report an honest out-of-sample R2 / AUC next to the in-sample
one, and to measure each driver's importance as the drop in that
out-of-sample score when it is left out.

Design choices that keep the inference honest:
  - The predictor set is screened by univariate association, pruned for
    near-collinearity, and capped by an events-per-variable budget. Screening
    is repeated inside each CV fold, so the out-of-sample score is not
    inflated by choosing predictors on the data it is then tested on.
  - Effects are per +1 SD of a numeric predictor (per doubling for a
    right-skewed positive predictor that was logged), so drivers measured in
    different units are comparable; a 0/1 or category term is its raw
    contrast. A skewed positive target is modelled on the log scale and its
    effects are reported as percent changes.
  - Standard errors are robust: heteroscedasticity-consistent by default,
    cluster-robust when rows repeat per entity, HAC for an ordered
    time series. Terms are corrected together with Benjamini-Hochberg.
  - Splitter follows the data's structure (AGENTS.md): TimeSeriesSplit for a
    time series, GroupKFold for repeated entities, stratified K-fold for a
    binary target, plain K-fold otherwise.
  - Ordinary least squares / maximum likelihood are unpenalised on purpose:
    penalised coefficients are biased, and this tool reports confidence
    intervals on them. Overfit is controlled by the predictor cap instead
    and surfaced as `train_test_gap` / `overfit_warnings`.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from scipy import stats

from src.core.findings import Finding
from src.core.multiple_testing import apply_benjamini_hochberg
from src.core.profiler import profile_dataframe
from src.core.stats_utils import repeated_entity
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df
from src.tools.experiment_analysis import _coerce_metric

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import ColumnProfile, DatasetProfile

_ALPHA = 0.05
_MAX_ROWS = 100_000
_MIN_ROWS = 40
_MIN_EVENTS = 15
_MAX_FEATURES = 8
_MAX_TERMS = 12
_TOP_LEVELS = 5
_MAX_CARDINALITY = 50
_MAX_MISSING = 0.30
_COLLINEAR_R = 0.9
_NEAR_DUPLICATE_R2 = 0.96
_MIN_SCREEN_R2 = 0.001
_SKEW_FOR_LOG = 1.5
_K_FOLDS = 5
_OVERFIT_GAP = 0.10
_MAX_COEF_ROWS = 12
_MAX_IMPORTANCE_ROWS = 8
_MAX_FINDINGS = 5
_MIN_STD_BETA = 0.05
_MIN_ODDS_EFFECT = 0.15
_VIF_MODERATE = 5.0
_VIF_SEVERE = 10.0


@dataclass
class _Term:
    name: str
    feature: str
    kind: str          # numeric | log | binary | level
    scale: float       # multiplies the column coefficient into the reported unit
    sd_raw: float      # SD of the untransformed predictor (numeric only)
    reference: str = ""
    level: str = ""


def _skew(values: np.ndarray) -> float:
    return float(stats.skew(values)) if len(values) > 2 and float(np.std(values)) > 0 else 0.0


def _g(value: float) -> str:
    return f"{value:+,.0f}" if abs(value) >= 1000 else f"{value:+.3g}"


def _classify(cp: ColumnProfile | None, series: pd.Series) -> str | None:
    """numeric | binary | categorical, or None when the column cannot be a predictor."""
    if cp is not None and (
        cp.pii or cp.semantic_role in ("identifier", "text", "constant", "time")
        or cp.kind in ("datetime", "identifier", "constant")
    ):
        return None
    nunique = int(series.nunique(dropna=True))
    if nunique < 2:
        return None
    role = cp.semantic_role if cp is not None else ""
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        if set(series.dropna().unique()) <= {0, 1}:
            return "binary"
        if nunique == 2 or (role == "dimension" and nunique <= _MAX_CARDINALITY):
            return "categorical"
        return "numeric"
    return "categorical" if nunique <= _MAX_CARDINALITY else None


def _build_design(
    df: pd.DataFrame, features: list[str], profile: DatasetProfile
) -> tuple[pd.DataFrame, list[_Term], dict[str, str], list[str]]:
    """Encoded predictor matrix (imputed, standardised, dummy-coded) plus one
    `_Term` per column. Returns (X, terms, {feature: why dropped}, imputed)."""
    by_name = {c.name: c for c in profile.columns}
    columns: dict[str, np.ndarray] = {}
    terms: list[_Term] = []
    dropped: dict[str, str] = {}
    imputed: list[str] = []
    for feature in features:
        series = df[feature]
        kind = _classify(by_name.get(feature), series)
        if kind is None:
            dropped[feature] = "not usable as a predictor (identifier, free text, personal data or constant)"
            continue
        if series.isna().mean() > _MAX_MISSING:
            dropped[feature] = f"more than {_MAX_MISSING:.0%} missing"
            continue
        if series.isna().any():
            imputed.append(feature)
        if kind == "categorical":
            text = series.astype(str).where(series.notna(), "Missing")
            counts = text.value_counts()
            top = [str(x) for x in counts.index[:_TOP_LEVELS]]
            text = text.where(text.isin(top), "Other")
            reference = top[0]
            for level in [x for x in text.value_counts().index if x != reference]:
                name = f"{feature}[{level}]"
                columns[name] = (text == level).to_numpy(dtype=float)
                terms.append(_Term(name, feature, "level", 1.0, 0.0, reference, str(level)))
            continue
        x = pd.to_numeric(series.astype("float64") if pd.api.types.is_bool_dtype(series) else series, errors="coerce")
        x = x.fillna(x.mode().iloc[0] if kind == "binary" else x.median())
        values = x.to_numpy(dtype=float)
        if kind == "binary":
            columns[feature] = values
            terms.append(_Term(feature, feature, "binary", 1.0, 0.0))
            continue
        sd_raw = float(np.std(values, ddof=1))
        if sd_raw == 0:
            dropped[feature] = "constant"
            continue
        logged = float(values.min()) >= 0 and _skew(values) > _SKEW_FOR_LOG
        if logged:
            log_values = np.log1p(values)
            logged = abs(_skew(log_values)) < 0.5 * abs(_skew(values)) and float(np.std(log_values)) > 0
            values = log_values if logged else values
        sd = float(np.std(values, ddof=1))
        columns[feature] = (values - float(values.mean())) / sd
        terms.append(_Term(feature, feature, "log" if logged else "numeric", math.log(2) / sd if logged else 1.0, sd_raw))
    return pd.DataFrame(columns, index=df.index), terms, dropped, imputed


def _r2(design: np.ndarray, y: np.ndarray) -> float:
    a = np.column_stack([np.ones(len(y)), design])
    beta = np.linalg.lstsq(a, y, rcond=None)[0]
    resid = y - a @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(resid @ resid) / sst if sst > 0 else 0.0


def _select(
    x: np.ndarray, y: np.ndarray, term_idx: dict[str, list[int]], budget: int
) -> tuple[list[str], dict[str, str]]:
    """Greedy screen: strongest univariate association first, skipping a
    feature that is near-collinear with one already chosen or that would
    exceed the term budget."""
    scores = {f: _r2(x[:, idx], y) for f, idx in term_idx.items()}
    with np.errstate(all="ignore"):
        corr = np.atleast_2d(np.nan_to_num(np.corrcoef(x, rowvar=False)))
    chosen: list[str] = []
    skipped: dict[str, str] = {}
    used = 0
    for feature in sorted(scores, key=lambda f: -scores[f]):
        idx = term_idx[feature]
        if scores[feature] < _MIN_SCREEN_R2:
            skipped[feature] = "no association with the target"
        elif scores[feature] >= _NEAR_DUPLICATE_R2:
            skipped[feature] = "near-duplicate of the target (likely leakage)"
        elif len(chosen) >= _MAX_FEATURES or used + len(idx) > budget:
            skipped[feature] = "beyond the lean predictor budget"
        elif any(
            float(np.abs(corr[np.ix_(idx, term_idx[c])]).max()) > _COLLINEAR_R for c in chosen
        ):
            skipped[feature] = f"collinear (|r| > {_COLLINEAR_R}) with a stronger predictor"
        else:
            chosen.append(feature)
            used += len(idx)
    return chosen, skipped


def _fit_predict(kind: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray | None:
    a_train = np.column_stack([np.ones(len(x_train)), x_train])
    a_test = np.column_stack([np.ones(len(x_test)), x_test])
    if kind == "ols":
        beta = np.linalg.lstsq(a_train, y_train, rcond=None)[0]
        return np.asarray(a_test @ beta)
    if float(y_train.min()) == float(y_train.max()):
        return None
    import statsmodels.api as sm

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Logit(y_train, a_train).fit(disp=0, maxiter=100)
            return np.asarray(fit.predict(a_test))
    except Exception:
        return None


def _score(kind: str, y: np.ndarray, pred: np.ndarray) -> float | None:
    if kind == "ols":
        sst = float(((y - y.mean()) ** 2).sum())
        return 1.0 - float(((y - pred) ** 2).sum()) / sst if sst > 0 else None
    if len(np.unique(y)) < 2 or not np.isfinite(pred).all():
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, pred))


def _cv_scores(
    kind: str,
    x: np.ndarray,
    y: np.ndarray,
    term_idx: dict[str, list[int]],
    splits: list[tuple[np.ndarray, np.ndarray]],
    fixed: list[str] | None,
    budget: int,
) -> tuple[list[float], int]:
    """Per-fold out-of-sample scores and the number of folds that failed. With
    `fixed=None` the predictors are re-selected on each training fold."""
    scores: list[float] = []
    failed = 0
    for train, test in splits:
        features = fixed if fixed is not None else _select(x[train], y[train], term_idx, budget)[0]
        cols = [i for f in features for i in term_idx[f]]
        pred = _fit_predict(kind, x[train][:, cols], y[train], x[test][:, cols]) if cols else None
        score = _score(kind, y[test], pred) if pred is not None else None
        if score is None:
            failed += 1
        else:
            scores.append(score)
    return scores, failed


def _make_splits(
    kind: str, y: np.ndarray, groups: np.ndarray | None, ordered: bool
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

    idx = np.arange(len(y))
    if ordered:
        return list(TimeSeriesSplit(n_splits=_K_FOLDS).split(idx)), "time_series_split"
    if groups is not None and len(np.unique(groups)) >= _K_FOLDS:
        return list(GroupKFold(n_splits=_K_FOLDS).split(idx, y, groups)), "group_k_fold"
    if kind == "logit":
        return list(StratifiedKFold(n_splits=_K_FOLDS, shuffle=True, random_state=42).split(idx, y)), "stratified_k_fold"
    return list(KFold(n_splits=_K_FOLDS, shuffle=True, random_state=42).split(idx)), "k_fold"


def _cov_spec(kind: str, groups: np.ndarray | None, ordered: bool, n: int) -> tuple[str, dict[str, Any], str]:
    if groups is not None:
        codes = pd.factorize(pd.Series(groups))[0]
        return "cluster", {"groups": codes}, "cluster-robust by entity"
    if ordered and kind == "ols":
        return "HAC", {"maxlags": max(1, int(n ** 0.25))}, "HAC (autocorrelation-robust)"
    return ("HC3", {}, "HC3 heteroscedasticity-robust") if kind == "ols" else ("HC0", {}, "HC0 heteroscedasticity-robust")


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


class RegressionAnalysisTool(BaseTool):
    """Inferential OLS / logistic regression with plain-English adjusted effects."""

    name = "regression_analysis"
    description = (
        "Explain what drives an outcome: OLS for a numeric target or logistic "
        "regression for a yes/no target, on a lean auto-selected set of "
        "predictors. Reports each driver's adjusted effect in plain English "
        "(per +1 SD, with a 95% CI, corrected for multiple testing), driver "
        "importance as the drop in out-of-sample R2/AUC, collinearity (VIF), "
        "heteroscedasticity and influential-point diagnostics, and a "
        "cross-validated out-of-sample R2/AUC. Inference only - use train_model "
        "for prediction."
    )
    #: Fits models, so it goes wherever "run without ML" turns model-fitting off.
    requires_ml = True
    requires_context: ClassVar[dict[str, str]] = {"target_column": "target_column"}

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None or profile.row_count < _MIN_ROWS:
            return 0.0
        target = getattr(metadata, "target_column", None)
        if target and any(c.name == target for c in profile.columns):
            return 0.8
        return 0.4 if profile.measures() and len(profile.columns) >= 3 else 0.0

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        target = getattr(metadata, "target_column", None)
        if profile is None or not target or not any(c.name == target for c in profile.columns):
            return {}
        return {"target_column": target}

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        target_column: str,
        feature_columns: list[str] | str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        if target_column not in df.columns:
            raise ToolExecutionError(f"Target column '{target_column}' not found in dataset.")
        if isinstance(feature_columns, str):
            feature_columns = [c.strip() for c in feature_columns.split(",") if c.strip()]
        missing = [c for c in feature_columns or [] if c not in df.columns]
        if missing:
            raise ToolExecutionError(f"Feature column(s) not found: {', '.join(missing)}.")

        profile = profile_dataframe(df)
        entity = repeated_entity(profile, df)
        time_col = next(
            (c for c in profile.datetime_cols if profile.is_time_series and c in df.columns), None
        )
        y_series = _coerce_metric(df[target_column])
        if y_series.notna().sum() == 0:
            raise ToolExecutionError(f"Target '{target_column}' is neither numeric nor a yes/no column.")
        keep = y_series.notna()
        if not keep.all():
            df, y_series = df[keep], y_series[keep]
        if len(df) > _MAX_ROWS:
            df = df.sample(_MAX_ROWS, random_state=42)
            y_series = y_series.loc[df.index]
        if time_col:
            stamps = pd.to_datetime(df[time_col], errors="coerce", format="mixed").to_numpy()
            order = np.argsort(stamps, kind="stable")
            df, y_series = df.iloc[order], y_series.iloc[order]

        levels = y_series.dropna().unique()
        if len(levels) < 2:
            raise ToolExecutionError(f"Target '{target_column}' is constant - nothing to explain.")
        if len(levels) == 2:
            kind = "logit"
            y = (y_series == y_series.max()).to_numpy(dtype=float)
        else:
            kind = "ols"
            y = y_series.to_numpy(dtype=float)
        n = len(y)
        events = int(min(y.sum(), n - y.sum())) if kind == "logit" else n
        if n < _MIN_ROWS or events < _MIN_EVENTS:
            raise ToolExecutionError(
                f"Too little data for a regression: {n} rows"
                + (f", {events} in the rarer class (need {_MIN_EVENTS})." if kind == "logit" else f" (need {_MIN_ROWS}).")
            )
        target_log = False
        if kind == "ols" and float(y.min()) > 0 and _skew(y) > _SKEW_FOR_LOG and abs(_skew(np.log(y))) < 0.5 * abs(_skew(y)):
            y, target_log = np.log(y), True

        explicit = list(dict.fromkeys(feature_columns)) if feature_columns else None
        excluded = {target_column, *([entity] if entity else []), *([time_col] if time_col else [])}
        candidates = [c for c in (explicit or df.columns) if c not in excluded]
        if not candidates:
            raise ToolExecutionError("No candidate predictor columns.")
        design, terms, dropped, imputed = _build_design(df, candidates, profile)
        if not terms:
            raise ToolExecutionError("None of the candidate columns can be used as a predictor.")
        x = design.to_numpy(dtype=float)
        term_idx: dict[str, list[int]] = {}
        for i, t in enumerate(terms):
            term_idx.setdefault(t.feature, []).append(i)

        budget = max(2, min(_MAX_TERMS, (events if kind == "logit" else n) // (10 if kind == "logit" else 15)))
        if explicit is None:
            chosen, skipped = _select(x, y, term_idx, budget)
            dropped.update(skipped)
        else:
            chosen = [f for f in explicit if f in term_idx]
        if not chosen:
            return self._empty(target_column, kind, n, dropped)

        groups = df[entity].to_numpy() if entity else None
        ordered = time_col is not None
        splits, cv_scheme = _make_splits(kind, y, groups, ordered)
        cols = [i for f in chosen for i in term_idx[f]]
        fit_result = self._fit(kind, design.iloc[:, cols], y, groups, ordered)
        cv_scores, failed = _cv_scores(kind, x, y, term_idx, splits, None if explicit is None else chosen, budget)
        fixed_scores = cv_scores if explicit is not None else _cv_scores(kind, x, y, term_idx, splits, chosen, budget)[0]
        cv_mean = float(np.mean(cv_scores)) if cv_scores else None
        train_score = self._train_score(kind, fit_result, y)
        importance = self._importance(kind, x, y, term_idx, splits, chosen, budget, fixed_scores)

        coefficients = self._coefficients(kind, fit_result, terms, design, y, target_log, target_column)
        diagnostics = self._diagnostics(kind, fit_result, terms, ordered)

        gap = train_score - cv_mean if cv_mean is not None else None
        warnings_out: list[str] = []
        if gap is not None and gap > _OVERFIT_GAP:
            metric = "R2" if kind == "ols" else "AUC"
            warnings_out.append(
                f"In-sample {metric} ({train_score:.2f}) exceeds cross-validated {metric} ({cv_mean:.2f}) by "
                f"{gap:.2f}; the fitted effects are partly noise - treat them as tentative."
            )
        if explicit and sum(len(term_idx[f]) for f in chosen) > budget:
            warnings_out.append(
                f"{len(chosen)} predictors ({len(cols)} terms) exceed the {budget}-term budget for {n:,} rows"
                + (f" ({events} events)." if kind == "logit" else ".")
            )
        if failed:
            warnings_out.append(f"{failed} of {len(splits)} CV folds could not be scored.")

        caveats = [c for c in (
            "Effects are adjusted associations between the columns, not proof of cause.",
            f"Rows repeat per '{entity}'; standard errors are clustered by it." if entity else "",
            f"Time series: predictors were tested on later periods only; '{time_col}' order was used." if ordered else "",
            f"Imputed (median/mode) columns: {', '.join(imputed[:5])}." if imputed else "",
            "The target was modelled on the log scale; effects are percent changes." if target_log else "",
        ) if c]
        metric_name = "r2" if kind == "ols" else "auc"
        drivers = [c["sentence"] for c in coefficients if c["significant_after_correction"]][:_MAX_FINDINGS]

        output: dict[str, Any] = {
            "summary": self._summary(target_column, kind, n, coefficients, metric_name, train_score, cv_mean),
            "model_type": "ols" if kind == "ols" else "logistic",
            "target_column": target_column,
            "target_transform": "log" if target_log else None,
            "n_obs": n,
            "unit_of_analysis": entity or "row",
            "standard_errors": fit_result["cov_label"],
            "features_used": chosen,
            "features_dropped": dict(list(dropped.items())[:8]),
            "predictor_budget_terms": budget,
            "coefficients": coefficients[:_MAX_COEF_ROWS],
            "n_terms_significant": sum(1 for c in coefficients if c["significant_after_correction"]),
            "drivers": drivers,
            "importance": importance[:_MAX_IMPORTANCE_ROWS],
            "importance_metric": f"drop in cross-validated {metric_name.upper()} when the feature is left out",
            "metric": metric_name,
            "in_sample": _round(train_score),
            "cv_mean": _round(cv_mean),
            "cv_std": _round(float(np.std(cv_scores))) if len(cv_scores) > 1 else None,
            "cv_scheme": cv_scheme,
            "cv_folds": len(splits),
            "train_test_gap": _round(gap),
            "overfit_warnings": warnings_out,
            "diagnostics": diagnostics,
            "caveats": caveats,
        }
        if kind == "ols":
            output["adj_r2_in_sample"] = _round(float(fit_result["res"].rsquared_adj))
        return output

    @staticmethod
    def _empty(target: str, kind: str, n: int, dropped: dict[str, str]) -> dict[str, Any]:
        return {
            "summary": f"No candidate column shows a usable association with {target} (n={n:,}); nothing to report as a driver.",
            "model_type": "ols" if kind == "ols" else "logistic", "target_column": target, "n_obs": n,
            "features_used": [], "features_dropped": dict(list(dropped.items())[:8]),
            "coefficients": [], "drivers": [], "importance": [], "overfit_warnings": [], "diagnostics": {}, "caveats": [],
        }

    @staticmethod
    def _fit(kind: str, xdf: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None, ordered: bool) -> dict[str, Any]:
        import statsmodels.api as sm
        from statsmodels.tools.sm_exceptions import PerfectSeparationError

        cov_type, cov_kwds, cov_label = _cov_spec(kind, groups, ordered, len(y))
        exog = sm.add_constant(xdf, has_constant="add")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if kind == "ols":
                    res = sm.OLS(y, exog).fit(cov_type=cov_type, cov_kwds=cov_kwds)
                else:
                    res = sm.Logit(y, exog).fit(disp=0, maxiter=200, cov_type=cov_type, cov_kwds=cov_kwds)
        except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as exc:
            raise ToolExecutionError(
                f"The regression could not be fitted ({exc}); a predictor may perfectly separate the outcome or be collinear."
            ) from exc
        return {"res": res, "exog": exog, "cov_label": cov_label}

    @staticmethod
    def _train_score(kind: str, fit: dict[str, Any], y: np.ndarray) -> float:
        if kind == "ols":
            return float(fit["res"].rsquared)
        return _score("logit", y, np.asarray(fit["res"].predict(fit["exog"]))) or 0.5

    @staticmethod
    def _importance(
        kind: str, x: np.ndarray, y: np.ndarray, term_idx: dict[str, list[int]],
        splits: list[tuple[np.ndarray, np.ndarray]], chosen: list[str], budget: int, full_scores: list[float],
    ) -> list[dict[str, Any]]:
        if not full_scores:
            return []
        full = float(np.mean(full_scores))
        null = 0.0 if kind == "ols" else 0.5
        rows: list[dict[str, Any]] = []
        for feature in chosen:
            rest = [f for f in chosen if f != feature]
            scores, _ = _cv_scores(kind, x, y, term_idx, splits, rest, budget) if rest else ([], 0)
            without = float(np.mean(scores)) if scores else null
            rows.append({"feature": feature, "drop_in_score": _round(full - without)})
        rows.sort(key=lambda r: -(r["drop_in_score"] or 0.0))
        total = sum(max(r["drop_in_score"] or 0.0, 0.0) for r in rows)
        for r in rows:
            r["share"] = _round(max(r["drop_in_score"] or 0.0, 0.0) / total, 3) if total > 0 else None
        return rows

    def _coefficients(
        self, kind: str, fit: dict[str, Any], terms: list[_Term], design: pd.DataFrame, y: np.ndarray,
        target_log: bool, target: str,
    ) -> list[dict[str, Any]]:
        res, exog = fit["res"], fit["exog"]
        by_name = {t.name: t for t in terms}
        conf = np.asarray(res.conf_int(alpha=_ALPHA))
        names = list(exog.columns)
        sd_y = float(np.std(y, ddof=1))
        base_p = np.asarray(res.predict(exog)) if kind == "logit" else None
        rows: list[dict[str, Any]] = []
        for i, name in enumerate(names):
            if name == "const":
                continue
            term = by_name[name]
            coef, se, p = float(res.params.iloc[i]), float(res.bse.iloc[i]), float(res.pvalues.iloc[i])
            if not (math.isfinite(coef) and math.isfinite(se) and math.isfinite(p)):
                continue
            lo, hi = float(conf[i, 0]), float(conf[i, 1])
            k = term.scale
            row: dict[str, Any] = {
                "feature": term.feature, "term": name, "term_kind": term.kind,
                "unit": self._unit(term), "p_value": p, "t_or_z": coef / se if se > 0 else 0.0,
            }
            if kind == "ols":
                std_beta = coef * float(design[name].std(ddof=1)) / sd_y if sd_y > 0 else 0.0
                if target_log:
                    change, ci = math.expm1(coef * k), (math.expm1(lo * k), math.expm1(hi * k))
                else:
                    change, ci = coef * k, (lo * k, hi * k)
                row.update({"effect": change, "ci_lower": ci[0], "ci_upper": ci[1], "std_beta": std_beta,
                            "effect_scale": "percent" if target_log else "target_units"})
            else:
                bumped = exog.copy()
                bumped.iloc[:, i] = bumped.iloc[:, i] + k
                ame = float(np.mean(np.asarray(res.predict(bumped)) - base_p))
                row.update({
                    "odds_ratio": math.exp(min(coef * k, 50.0)),
                    "ci_lower": math.exp(min(lo * k, 50.0)), "ci_upper": math.exp(min(hi * k, 50.0)),
                    "effect": ame, "effect_scale": "probability", "std_beta": None,
                })
            rows.append(row)
        adjusted = apply_benjamini_hochberg([r for r in rows if math.isfinite(r["p_value"])], alpha=_ALPHA)
        for r in adjusted:
            r["sentence"] = self._sentence(r, by_name[r["term"]], kind, target)
            for key in ("p_value", "p_adjusted", "t_or_z", "effect", "ci_lower", "ci_upper", "odds_ratio", "std_beta"):
                if key in r and r[key] is not None:
                    r[key] = _round(r[key], 6 if key.startswith("p_") else 4)
        adjusted.sort(key=lambda r: (not r["significant_after_correction"], -abs(r["t_or_z"] or 0.0)))
        return adjusted

    @staticmethod
    def _unit(term: _Term) -> str:
        return {
            "numeric": f"+1 SD (about {term.sd_raw:,.3g})",
            "log": "a doubling",
            "binary": "1 vs 0",
            "level": f"{term.level} vs {term.reference}",
        }[term.kind]

    @staticmethod
    def _sentence(r: dict[str, Any], term: _Term, kind: str, target: str) -> str:
        subject = {
            "numeric": f"each +1 SD in {term.feature} (about {term.sd_raw:,.3g})",
            "log": f"each doubling of {term.feature}",
            "binary": f"having {term.feature} = 1 rather than 0",
            "level": f"{term.feature} = {term.level} rather than {term.reference}",
        }[term.kind]
        lead = f"Holding the other factors fixed, {subject}"
        if kind == "logit":
            direction = "raises" if r["effect"] > 0 else "lowers"
            return (
                f"{lead} {direction} the chance of {target} by about {abs(r['effect']) * 100:.1f} percentage points "
                f"(odds ratio {r['odds_ratio']:.2f}, 95% CI {r['ci_lower']:.2f} to {r['ci_upper']:.2f})."
            )
        if r["effect_scale"] == "percent":
            return (
                f"{lead} changes {target} by about {r['effect'] * 100:+.1f}% "
                f"(95% CI {r['ci_lower'] * 100:+.1f}% to {r['ci_upper'] * 100:+.1f}%)."
            )
        return f"{lead} changes {target} by {_g(r['effect'])} (95% CI {_g(r['ci_lower'])} to {_g(r['ci_upper'])})."

    @staticmethod
    def _diagnostics(kind: str, fit: dict[str, Any], terms: list[_Term], ordered: bool) -> dict[str, Any]:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        res, exog = fit["res"], fit["exog"]
        feature_of = {t.name: t.feature for t in terms}
        out: dict[str, Any] = {}
        vif_by_feature: dict[str, float] = {}
        if exog.shape[1] > 2:
            values = exog.to_numpy(dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for j, name in enumerate(exog.columns):
                    if name != "const":
                        feature = feature_of[name]
                        vif_by_feature[feature] = max(vif_by_feature.get(feature, 0.0), float(variance_inflation_factor(values, j)))
        out["max_vif"] = _round(max(vif_by_feature.values(), default=1.0), 2)
        out["collinearity_flags"] = {
            f: "severe" if v >= _VIF_SEVERE else "moderate" for f, v in vif_by_feature.items() if v >= _VIF_MODERATE
        }
        n = int(exog.shape[0])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if kind == "ols":
                    from statsmodels.stats.diagnostic import het_breuschpagan
                    from statsmodels.stats.stattools import durbin_watson

                    bp_p = float(het_breuschpagan(res.resid, exog)[1])
                    out["heteroscedasticity"] = {
                        "breusch_pagan_p": _round(bp_p, 6), "flagged": bp_p < _ALPHA,
                        "note": "robust standard errors already account for this" if bp_p < _ALPHA else "",
                    }
                    if ordered:
                        dw = float(durbin_watson(res.resid))
                        out["durbin_watson"] = _round(dw, 2)
                        out["autocorrelated_residuals"] = dw < 1.5 or dw > 2.5
                    cooks = np.asarray(res.get_influence().cooks_distance[0])
                else:
                    glm = sm.GLM(res.model.endog, exog, family=sm.families.Binomial()).fit()
                    cooks = np.asarray(glm.get_influence().cooks_distance[0])
            threshold = 4.0 / n
            count = int((cooks > threshold).sum())
            out["influential_points"] = {
                "count": count, "share": _round(count / n, 4), "cooks_threshold": _round(threshold, 6),
            }
        except (ValueError, np.linalg.LinAlgError, AttributeError, NotImplementedError):
            out["influential_points"] = None
        return out

    @staticmethod
    def _summary(
        target: str, kind: str, n: int, coefficients: list[dict[str, Any]], metric: str,
        train_score: float, cv_mean: float | None,
    ) -> str:
        top = next((c for c in coefficients if c["significant_after_correction"]), None)
        head = top["sentence"] if top else f"No predictor has a statistically reliable effect on {target}."
        oos = f"{cv_mean:.2f}" if cv_mean is not None else "n/a"
        model = "OLS" if kind == "ols" else "Logistic"
        return (
            f"{model} regression of {target} (n={n:,}): {head} "
            f"Out-of-sample {metric.upper()} {oos} (in-sample {train_score:.2f}, {_K_FOLDS}-fold)."
        )

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        target = str(output.get("target_column"))
        cv = output.get("cv_mean")
        n = int(output.get("n_obs") or 0)
        logit = output.get("model_type") == "logistic"
        model_caveats = [*(output.get("overfit_warnings") or []), *(output.get("caveats") or [])]
        if cv is not None and cv < 0.1 and not logit:
            model_caveats.append(f"The model explains little out of sample (R2 {cv:.2f}); read effects as weak associations.")
        flags = (output.get("diagnostics") or {}).get("collinearity_flags") or {}
        seen: set[str] = set()
        results: list[Finding] = []
        for c in output.get("coefficients", []):
            feature = c["feature"]
            if feature in seen or not c.get("significant_after_correction"):
                continue
            effect = (c["odds_ratio"] - 1.0) if logit else c.get("std_beta")
            floor = _MIN_ODDS_EFFECT if logit else _MIN_STD_BETA
            if effect is None or abs(effect) < floor:
                continue
            seen.add(feature)
            results.append(Finding(
                finding_id=f"driver_{target}_{c['term']}".replace(" ", "_"),
                kind="driver",
                headline=c["sentence"],
                detail=f"p={c['p_value']:.4g}, adjusted p={c['p_adjusted']:.4g}; {output.get('standard_errors')}.",
                evidence={**c, "n_obs": n, "cv_mean": cv, "metric": output.get("metric")},
                source_tool=self.name,
                measure=target,
                dimension=feature,
                level=c.get("unit"),
                effect=effect,
                effect_kind="lift" if logit else "r",
                p_value=c["p_value"],
                p_adjusted=c["p_adjusted"],
                confidence=min(1.0, n / 200.0) * (0.6 + 0.4 * max(0.0, cv or 0.0)),
                surprise=min(1.0, abs(effect)),
                caveats=[*model_caveats, *([f"{feature} is collinear with other predictors ({flags[feature]} VIF)."] if feature in flags else [])],
            ))
            if len(results) >= _MAX_FINDINGS:
                break
        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset.",
                "required": True,
            },
            "target_column": {
                "type": "string",
                "description": (
                    "Outcome to explain: a numeric column (OLS) or a yes/no / 0-1 "
                    "column (logistic). Filled from the dataset's detected target when omitted."
                ),
                "required": True,
            },
            "feature_columns": {
                "type": "array",
                "description": (
                    "Optional predictors to use as given. When omitted, a lean set "
                    "(at most 8, collinearity-pruned) is chosen automatically."
                ),
                "required": False,
            },
        }
