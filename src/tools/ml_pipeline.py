"""
ML Pipeline Tools — Execution Layer.

Stage 3: Model Training   (TrainModelTool)
Stage 3: Model Evaluation (EvaluateModelTool)

Anti-overfitting measures built in:
  - Stratified K-Fold cross-validation (k=5, default)
  - Explicit train/validation/test split reporting
  - Train–test accuracy gap warning (>0.10 threshold)
  - Tree depth capping via max_depth parameter
  - L2 regularisation active by default (LogisticRegression, Ridge)
  - XGBoost early stopping when eval_set is available
"""
from __future__ import annotations

import logging
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

from src.core.findings import Finding
from src.core.model_io import save_model
from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import (
    _read_df as _read_df,  # re-exported: src.tools.clustering imports it from here
)

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata, MemorySystem
    from src.core.profiler import DatasetProfile
from src.tools.ml_common import (
    OVERFIT_THRESHOLD,
    _cap_train_rows,
    _encode_target,
    _prepare_features,
    _resolve_split_strategy,
    _split_train_test,
)
from src.tools.ml_evaluate import (
    _EVAL_TRAIN_MAX_ROWS,
    _PERMUTATION_MAX_ROWS,
    EvaluateModelTool,
    _seeded_rows,
)

#: Public surface. Helpers and EvaluateModelTool live in ml_common / ml_evaluate and are
#: re-exported here so `src.tools.ml_pipeline.<name>` keeps working.
__all__ = [
    "IMBALANCE_THRESHOLD",
    "LINEAR_MODELS",
    "OVERFIT_PENALTY_WEIGHT",
    "OVERFIT_THRESHOLD",
    "SKEW_TREATMENT_THRESHOLD",
    "_EVAL_TRAIN_MAX_ROWS",
    "_LEAKAGE_MIN_GROUP_RATIO",
    "_LEAKAGE_PURITY",
    "_MIN_TIME_AXIS_VALUES",
    "_NEAR_PERFECT_SCORE",
    "_ORDINAL_CARD_MAX",
    "_ORDINAL_CARD_MIN",
    "_ORDINAL_NAME_HINTS",
    "_PERMUTATION_MAX_ROWS",
    "_TUNE_SAMPLE_ROWS",
    "EvaluateModelTool",
    "TrainModelTool",
    "_SkewLog1pTransformer",
    "_build_preprocessor",
    "_cap_train_rows",
    "_detect_cuda_gpu",
    "_detect_formula_leakage",
    "_detect_target_leakage",
    "_encode_target",
    "_is_flag_or_ordinal",
    "_prepare_features",
    "_read_df",
    "_resolve_split_strategy",
    "_seeded_rows",
    "_split_train_test",
    "_tuning_sample",
]

logger = logging.getLogger(__name__)


#: Composite-score weight for _pick_best's overfit penalty: how many points
#: of cv_mean one point of (train_test_gap - OVERFIT_THRESHOLD) costs a
#: model when ranking. At 1.0, a model 0.10 over the threshold loses 0.10
#: off its effective score — enough to lose to a close runner-up that
#: wasn't flagged, without disqualifying a clear overall winner outright.
OVERFIT_PENALTY_WEIGHT = 1.0


#: Absolute skewness at which a numeric feature gets a skew treatment
#: (log1p for a positive/right skew, Yeo-Johnson for a negative/left skew).
SKEW_TREATMENT_THRESHOLD = 2.0


#: Local mirror of profiler.py's ordinal-detection name hints (IMPROVEMENTS.md
#: 7.10) — duplicated rather than importing a private profiler helper, since
#: the fitted Pipeline step here only ever sees a bare numeric frame, never
#: the DatasetProfile that classified it.
_ORDINAL_NAME_HINTS = (
    "rating", "score", "grade", "level", "tier", "stars", "priority",
    "satisfaction", "nps", "rank", "severity", "star",
)


_ORDINAL_CARD_MIN = 3


_ORDINAL_CARD_MAX = 10


@lru_cache(maxsize=1)
def _detect_cuda_gpu() -> bool:
    """Whether an NVIDIA GPU answers `nvidia-smi` (probed once per process;
    importing torch just to ask would cost seconds and gigabytes)."""
    try:
        res = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0


#: Hyperparameter search never sees more than this many training rows.
_TUNE_SAMPLE_ROWS = 5_000


def _tuning_sample(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series[Any] | None,
    split_strategy: str,
    task_type: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series[Any] | None]:
    """Subsample the training rows the tuning search runs on, keeping what the
    splitter needs: the most recent rows (in order) for a time series —
    shuffled rows would let TimeSeriesSplit train on the future — and a
    class-stratified draw for classification."""
    if len(X) <= _TUNE_SAMPLE_ROWS:
        return X, y, groups
    if split_strategy == "time_series":
        keep = np.arange(len(X) - _TUNE_SAMPLE_ROWS, len(X))
    else:
        keep = np.random.default_rng(42).choice(len(X), _TUNE_SAMPLE_ROWS, replace=False)
        if task_type == "classification" and y.nunique() > 1:
            from sklearn.model_selection import train_test_split

            try:
                keep = train_test_split(
                    np.arange(len(X)), train_size=_TUNE_SAMPLE_ROWS, stratify=y, random_state=42
                )[0]
            except ValueError:  # a class with a single member cannot be stratified
                pass
        keep = np.sort(keep)
    return X.iloc[keep], y.iloc[keep], (groups.iloc[keep] if groups is not None else None)


def _is_flag_or_ordinal(name: str, clean: pd.Series[Any]) -> bool:
    """
    Cheap local replica of `profiler.py`'s semantic-role test for a numeric
    column being a flag (0/1) or an ordinal scale (a 1-5 rating) rather than
    a true continuous measure.

    A flag or ordinal must never be log1p'd or power-transformed: log1p
    implies a magnitude/skew reading that doesn't apply to a code, and a 0/1
    flag skewed toward one value isn't "heavy-tailed" — it's just imbalanced,
    which class weighting already handles. This was the P0.1/6.4 bug: a
    negatively-skewed 0/1 flag was still getting log1p'd because the old
    check only skipped `min < 0`, not "this isn't a continuous quantity".
    """
    if len(clean) == 0:
        return False
    try:
        is_integer_valued = bool((clean % 1 == 0).all())
    except TypeError:
        return False
    if not is_integer_valued:
        return False
    nunique = int(clean.nunique())
    if nunique == 2:
        return True  # flag
    name_l = name.lower()
    if _ORDINAL_CARD_MIN <= nunique <= _ORDINAL_CARD_MAX and any(h in name_l for h in _ORDINAL_NAME_HINTS):
        return True  # ordinal
    return False


#: Minority-class fraction below which class weighting is applied.
IMBALANCE_THRESHOLD = 0.10


class _SkewLog1pTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """
    Fixes each severely-skewed numeric column with whichever transform
    actually applies to its skew direction — but the decision is made once,
    at `fit`, from whatever frame `fit` is called on. Used inside a Pipeline
    fit only on the training fold (IMPROVEMENTS.md P0.1), so the decision —
    and the values it's based on — never see the test fold.

    log1p only compresses a long *right* tail (positive skew) on non-negative
    data; it was previously gated on `abs(skew)` and `min < 0` alone, which
    both log1p'd negatively-skewed columns (silently wrong — log1p on a left
    tail does nothing useful) and left a 0/1 flag skewed toward one class
    treated as "heavy-tailed" numeric data (IMPROVEMENTS.md 7.10/6.4). Now:
      - flag/ordinal columns (see `_is_flag_or_ordinal`) are never touched.
      - positive skew on non-negative data -> log1p.
      - negative skew, or positive skew the column's negative values make
        log1p inapplicable to -> `PowerTransformer(method="yeo-johnson")`,
        which handles both tail directions and negative values.

    Must inherit BaseEstimator/TransformerMixin/OneToOneFeatureMixin rather
    than duck-typing fit/transform: sklearn 1.8's Pipeline requires
    `__sklearn_tags__` on every step, which only BaseEstimator provides.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> _SkewLog1pTransformer:
        from sklearn.preprocessing import PowerTransformer

        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        skewed: list[str] = []
        power_cols: list[str] = []
        power_transformers: dict[str, Any] = {}
        skew_values: dict[str, float] = {}
        for col in X.columns:
            clean = X[col].dropna()
            if len(clean) < 3 or _is_flag_or_ordinal(str(col), clean):
                continue
            skew = float(clean.skew())
            if abs(skew) < SKEW_TREATMENT_THRESHOLD:
                continue
            if skew > 0 and float(clean.min()) >= 0:
                skewed.append(str(col))
                skew_values[str(col)] = skew
            else:
                # Negative skew (log1p doesn't apply to a left tail), or
                # positive skew on data that goes negative (log1p undefined) —
                # Yeo-Johnson handles both.
                pt = PowerTransformer(method="yeo-johnson")
                try:
                    pt.fit(clean.to_numpy().reshape(-1, 1))
                except Exception:
                    continue
                power_cols.append(str(col))
                power_transformers[str(col)] = pt
                skew_values[str(col)] = skew
        self.skewed_cols_ = skewed
        self.power_cols_ = power_cols
        self.power_transformers_ = power_transformers
        self.skew_values_ = skew_values
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        X = X.copy()
        for col in self.skewed_cols_:
            # Clip: a test-fold negative in a column the training fold saw
            # as non-negative must not silently produce NaN.
            X[col] = np.log1p(X[col].clip(lower=0))
        for col in self.power_cols_:
            pt = self.power_transformers_[col]
            X[col] = pt.transform(X[[col]].to_numpy()).ravel()
        return X

    def describe(self) -> list[str]:
        """Human-readable treatment strings for the report, one per column
        a skew treatment was applied to (decided at fit time), naming which
        rule fired."""
        lines = [
            f"Applied log1p to '{col}' (skew={self.skew_values_[col]:.2f} — heavy right tail compressed)."
            for col in self.skewed_cols_
        ]
        lines.extend(
            f"Applied Yeo-Johnson power transform to '{col}' (skew={self.skew_values_[col]:.2f} — "
            f"negatively skewed or not log1p-eligible)."
            for col in self.power_cols_
        )
        return lines


#: Linear models get OneHotEncoder (no fake ordinality); tree/ensemble models
#: get OrdinalEncoder (cheaper, and trees can recover from arbitrary codes).
LINEAR_MODELS = {"logistic_regression", "linear_regression", "ridge"}


def _build_preprocessor(X: pd.DataFrame, encoding: str) -> Any:
    """
    Build the ColumnTransformer that becomes a Pipeline's "prep" step,
    fit exclusively on whatever frame is passed to it (the training fold).

    ``encoding``: "onehot" for linear models, "ordinal" for tree/ensemble
    and clustering models (IMPROVEMENTS.md P0.6).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    numeric_cols = [
        c for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
    ]
    bool_cols = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in numeric_cols and c not in bool_cols]

    encoder: Any = (
        OneHotEncoder(handle_unknown="ignore")
        if encoding == "onehot"
        else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("skew", _SkewLog1pTransformer()),
        ]).set_output(transform="pandas")
        transformers.append(("num", numeric_pipe, numeric_cols))
    if bool_cols:
        transformers.append(("bool", SimpleImputer(strategy="most_frequent"), bool_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", encoder),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers, remainder="drop")


#: A feature explaining at least this much of the target is reported as
#: leakage rather than as a finding. Set just below 1.0 because the cases
#: that matter are near-deterministic, not merely strong. For classification
#: the bar also rises with the majority-class rate (see
#: _detect_target_leakage): on a 99.5%-one-class target every feature is
#: already >= 99% "pure".
_LEAKAGE_PURITY = 0.99


#: Cross-validated score at or above which the result is reported as a data
#: check rather than a finding. Genuine business problems do not score here.
_NEAR_PERFECT_SCORE = 0.99


#: Distinct timestamps a datetime column needs before it is treated as the
#: dataset's time axis for a chronological split.
_MIN_TIME_AXIS_VALUES = 20


#: Below this many distinct feature values the "determines the target"
#: test is vacuous — a column that is unique per row trivially "predicts"
#: anything, which is a different defect (an identifier) already handled.
_LEAKAGE_MIN_GROUP_RATIO = 0.5


def _detect_target_leakage(
    X: pd.DataFrame, y: pd.Series[Any], task_type: str
) -> list[str]:
    """
    Find features that trivially determine the target.

    A near-perfect model is the most misleading thing this system can
    report, because it looks like the best possible result. On a shop
    export, `Unit Price` and `Product Name` are the same fact written twice
    — every product has exactly one price — so a model "predicting" the
    product from its price scores 1.00 and means nothing. An analyst asks
    "what leaked?" the moment they see 100% accuracy; this asks it
    automatically.

    Returns human-readable warnings, empty when nothing looks tautological.
    """
    warnings: list[str] = []
    if len(X) == 0 or y.nunique(dropna=True) < 2:
        return warnings
    # Purity a feature gets for free by predicting the majority class; a
    # leak must close at least half the remaining gap to 1.0.
    majority_rate = float(y.value_counts(normalize=True).iloc[0])
    purity_bar = max(_LEAKAGE_PURITY, majority_rate + 0.5 * (1.0 - majority_rate))

    for column in X.columns:
        feature = X[column]
        try:
            if task_type == "classification":
                n_groups = int(feature.nunique(dropna=True))
                # Skip near-unique columns: they separate every row by
                # construction and say nothing about the target.
                if n_groups < 2 or n_groups > len(X) * _LEAKAGE_MIN_GROUP_RATIO:
                    continue
                # Share of rows whose target equals their group's majority
                # class. 1.0 means the feature fixes the target exactly.
                # = sum of each group's majority count / rows with a feature
                # value, from one groupby instead of a Python call per group.
                pair_counts = (
                    pd.DataFrame({"f": feature, "y": y})
                    .groupby(["f", "y"], observed=True)
                    .size()
                )
                majority = pair_counts.groupby(level=0, observed=True).max()
                if len(majority) < n_groups:  # a group with no target value
                    continue
                purity = majority.sum() / int(feature.notna().sum())
                if float(purity) >= purity_bar:
                    warnings.append(
                        f"'{column}' determines the target in "
                        f"{float(purity) * 100:.1f}% of rows — the model is "
                        f"likely restating a definition, not learning a "
                        f"relationship. Drop it and re-train to get a "
                        f"meaningful score."
                    )
            elif pd.api.types.is_numeric_dtype(feature):
                corr = float(pd.Series(feature).corr(pd.Series(y)))
                if abs(corr) >= _LEAKAGE_PURITY:
                    warnings.append(
                        f"'{column}' correlates with the target at r={corr:.4f} "
                        f"— near-perfect, so the model is likely restating a "
                        f"definition. Drop it and re-train."
                    )
        except Exception:
            # A diagnostic must never take the training run down.
            continue
    return warnings


def _detect_formula_leakage(df: pd.DataFrame, target: str, features: list[str]) -> list[str]:
    """Features that are inputs of an exact/near formula involving the target
    (`revenue = price * qty` with `revenue` as the target): a model learns the
    formula, not a relationship. Relation discovery is a diagnostic — any
    failure returns no warnings."""
    from src.core.relations import find_relations

    try:
        relations = find_relations(df)
    except Exception:
        return []
    warnings: list[str] = []
    for rel in relations:
        members = {rel["target"], *rel["terms"]}
        if target not in members:
            continue
        leaked = [f for f in features if f in members and f != target]
        if leaked:
            names = ", ".join(f"'{f}'" for f in leaked)
            warnings.append(
                f"{names} and the target are linked by a formula ({rel['expr']}) — the model "
                "would restate it rather than learn a relationship. Drop them and re-train."
            )
    return warnings


class TrainModelTool(BaseTool):
    """
    Train one or more ML models with anti-overfitting safeguards.

    Supports:
      Classification : RandomForest, XGBoost, LogisticRegression
      Regression     : RandomForest, XGBoost, LinearRegression, Ridge
      Clustering     : KMeans, DBSCAN
    """

    requires_ml = True

    name = "train_model"
    description = (
        "Train one or more ML models with built-in cross-validation (k=5) "
        "and train/test gap monitoring to detect overfitting. "
        "Returns per-model metrics, CV scores, and the best model name."
    )
    output_subdir = "models"
    requires_context: ClassVar[dict[str, str]] = {
        "target_column": "target_column",
        # IMPROVEMENTS.md 7.16 — the Streamlit sidebar's "Max tree depth" /
        # "Test split %" / "CV folds" sliders write these context keys
        # before analyze() runs (see app.py's "Analysis Settings" section).
        # Same fallback-fill semantics as target_column: only applied when
        # the planner (LLM or deterministic) left the parameter empty, so an
        # explicit plan value still wins — in practice the planner never
        # sets these itself, so the UI's choice is what actually trains.
        "ui_max_depth": "max_depth",
        "ui_test_size": "test_size",
        "ui_n_cv_folds": "n_cv_folds",
        "ui_tune_hyperparameters": "tune_hyperparameters",
    }

    CLASSIFICATION_MODELS: ClassVar[list[str]] = ["random_forest", "xgboost", "logistic_regression"]
    REGRESSION_MODELS: ClassVar[list[str]] = ["random_forest", "xgboost", "linear_regression", "ridge"]
    CLUSTERING_MODELS: ClassVar[list[str]] = ["kmeans", "dbscan"]

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return 1.0 if metadata and metadata.target_column and metadata.task_type in ("classification", "regression") else 0.0

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        """
        Wire the profiler's dataset-nature detection into the splitter.

        The profiler already flags time-series and panel/grouped structure
        (`DatasetProfile.is_time_series`, `.panel_group_cols`), but until now
        nothing downstream consumed those facts — `train_test_split` shuffled
        rows regardless, training on the future and testing on the past for
        time-series data, or leaking the same entity into both splits for
        panel data. Only fills in when the planner didn't already choose a
        strategy, and time-series takes priority when a dataset is both.
        """
        params = super().prepare_params(params, memory, output_root)
        if not params.get("split_strategy"):
            profile = memory.get_context("data_profile") or {}
            columns = profile.get("columns") or []
            nunique = {c.get("name"): c.get("nunique", 0) for c in columns}
            # Only a genuine time axis orders the rows: enough distinct
            # timestamps to form a sequence, and not a per-person attribute
            # like a birth date that carries no "past vs future" meaning.
            time_axis = next(
                (
                    c for c in profile.get("datetime_cols") or []
                    if (not columns or nunique.get(c, 0) >= _MIN_TIME_AXIS_VALUES)
                    and not any(h in c.lower() for h in ("birth", "dob"))
                ),
                None,
            )
            # Grouped splitting is about the same entity repeating across
            # rows (customer_id, patient_id) — never a low-cardinality
            # dimension like gender, which would hold out a whole category.
            panel_cols = profile.get("panel_group_cols") or []
            entity_col = profile.get("entity_col") or (panel_cols[0] if panel_cols else None)
            rows_per_entity = profile.get("rows_per_entity") or (2.0 if panel_cols else 0.0)
            if profile.get("is_time_series") and time_axis:
                params["split_strategy"] = "time_series"
                params.setdefault("time_column", time_axis)
            elif entity_col and rows_per_entity > 1.5 and entity_col != params.get("target_column"):
                params["split_strategy"] = "panel"
                params.setdefault("group_column", entity_col)
        return params

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        """
        Project the training run's own headline result — best model + score,
        overfit warnings, leakage warnings — onto the finding bus
        (IMPROVEMENTS.md 7.2 item 4 / item 4). `driver_narrative`/top_drivers
        are EvaluateModelTool's output, not this tool's, so per-driver
        findings are emitted there; this covers what train_model itself
        actually determined.
        """
        found: list[Finding] = []
        models_trained = output.get("models_trained") or {}
        best_model = output.get("best_model")
        best = models_trained.get(best_model) or {}
        task_type = output.get("task_type")
        target_column = metadata.target_column if metadata else None
        cv_mean = best.get("cv_mean")
        cv_std = best.get("cv_std")
        baseline = output.get("baseline_cv_mean")
        lift = output.get("lift_over_baseline")

        if best_model and best_model != "none" and isinstance(cv_mean, (int, float)):
            scoring_label = "F1 (weighted)" if task_type == "classification" else "R2"
            # Lift below this over a no-skill predictor is not a result.
            weak = isinstance(lift, (int, float)) and lift < 0.05
            found.append(
                Finding(
                    finding_id=f"{self.name}_best_model",
                    kind="model_performance",
                    headline=(
                        f"{best_model} best predicts {target_column or 'the target'} "
                        f"— cross-validated {scoring_label} = {cv_mean:.3f}"
                        + (f" (+/- {cv_std:.3f})" if isinstance(cv_std, (int, float)) else "")
                        + (
                            f", {lift:+.3f} over a no-skill baseline ({baseline:.3f})"
                            if isinstance(lift, (int, float)) and isinstance(baseline, (int, float))
                            else ""
                        )
                    ),
                    detail=(
                        f"Selected from {len(models_trained)} candidate model(s) trained on a "
                        f"'{output.get('split_strategy', 'random')}' split "
                        f"({output.get('train_samples')} train / {output.get('test_samples')} test rows)."
                    ),
                    evidence={
                        "models_trained": {
                            name: res.get("cv_mean") for name, res in models_trained.items()
                        },
                        "best_model": best_model,
                        "baseline_cv_mean": baseline,
                        "lift_over_baseline": lift,
                    },
                    source_tool=self.name,
                    measure=target_column,
                    effect=round(float(cv_mean), 4),
                    effect_kind="share",
                    caveats=(
                        ["Model barely beats a no-skill baseline — not a usable predictor."]
                        if weak
                        else []
                    ),
                    confidence=0.3 if weak else 0.7,
                    layer="analyst",
                )
            )

        for warning in output.get("overfit_warnings") or []:
            model_name = warning.split(":", 1)[0].strip()
            found.append(
                Finding(
                    finding_id=f"{self.name}_overfit_{model_name}",
                    kind="method_fit",
                    headline=f"{model_name} shows signs of overfitting (train-test gap warning)",
                    detail=warning,
                    evidence={"model": model_name},
                    source_tool=self.name,
                    measure=target_column,
                    caveats=[warning],
                    confidence=0.6,
                    layer="analyst",
                )
            )

        for i, warning in enumerate(output.get("leakage_warnings") or []):
            found.append(
                Finding(
                    finding_id=f"{self.name}_leakage_{i}",
                    kind="method_fit",
                    headline="Possible target leakage detected",
                    detail=warning,
                    evidence={},
                    source_tool=self.name,
                    measure=target_column,
                    caveats=[warning],
                    confidence=0.6,
                    layer="analyst",
                )
            )

        return found

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        target_column: str,
        task_type: str = "auto",
        models: list[str] | None = None,
        test_size: float = 0.2,
        n_cv_folds: int = 5,
        max_depth: int = 6,
        tune_hyperparameters: bool = False,
        output_dir: str = "output/models",
        split_strategy: str = "random",
        time_column: str | None = None,
        group_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        from sklearn.model_selection import cross_val_score

        df = _read_df(file_path)
        target_words: dict[str, str] | None = (df.attrs.get("boolean_labels") or {}).get(target_column)

        if target_column not in df.columns:
            raise ToolExecutionError(f"Target column '{target_column}' not in dataset.")

        df, split_strategy, split_notes = _resolve_split_strategy(
            df, split_strategy, time_column, group_column
        )
        X, y, treatments = _prepare_features(df, target_column)
        treatments.extend(split_notes)

        # Auto-detect task type. Must agree with
        # DatasetMetadata.infer_task_type, which is the canonical rule: a
        # FLOAT target is continuous no matter how few distinct values it
        # happens to take. The previous "nunique <= 20 -> classification"
        # test ignored dtype, so a revenue column taking 18 distinct prices
        # was treated as an 18-class problem and every model failed with
        # "Supported target types are ('binary', 'multiclass'). Got
        # 'continuous'" — the whole ML stage dying on ordinary money data.
        if task_type == "auto":
            if not pd.api.types.is_numeric_dtype(y):
                task_type = "classification"
            elif pd.api.types.is_bool_dtype(y):
                task_type = "classification"
            elif pd.api.types.is_integer_dtype(y) and y.nunique() <= 20:
                task_type = "classification"
            else:
                task_type = "regression"

        leakage_warnings = _detect_target_leakage(X, y, task_type)
        leakage_warnings += _detect_formula_leakage(df, target_column, list(X.columns))

        # Encode non-numeric classification targets (XGBoost requires
        # numeric labels; roc_auc_score requires {0,1} for binary tasks)
        class_labels: list[str] = []
        balanced = False
        scale_pos_weight = 1.0
        if task_type == "classification":
            y, class_labels = _encode_target(y, target_words)
            # Act on class imbalance instead of just warning about it
            counts = y.value_counts()
            if len(counts) >= 2:
                minority_frac = float(counts.iloc[-1]) / float(counts.sum())
                if minority_frac < IMBALANCE_THRESHOLD:
                    balanced = True
                    if len(counts) == 2:
                        scale_pos_weight = float(counts.get(0, 0)) / max(float(counts.get(1, 0)), 1.0)
                    treatments.append(
                        f"Class weighting applied — minority class is {minority_frac:.1%} "
                        f"of rows (threshold {IMBALANCE_THRESHOLD:.0%})."
                    )

        do_tune = bool(tune_hyperparameters)

        if models is None:
            models = (
                self.CLASSIFICATION_MODELS
                if task_type == "classification"
                else self.REGRESSION_MODELS
                if task_type == "regression"
                else self.CLUSTERING_MODELS
            )

        if not models:
            raise ToolExecutionError(
                f"'models' list is empty. Pass a non-empty list or omit the parameter "
                f"to use the defaults for task_type='{task_type}'."
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {}
        overfit_warnings: list[str] = []
        build_errors: list[str] = []

        if task_type in {"classification", "regression"}:
            X_train, X_test, y_train, y_test, cv, groups_train = _split_train_test(
                X, y, df, split_strategy, group_column, task_type, test_size, n_cv_folds
            )
            X_train, y_train, groups_train, orig_train_len = _cap_train_rows(
                X_train, y_train, groups_train, split_strategy, task_type
            )
            if orig_train_len is not None:
                treatments.append(
                    f"Subsampled training set to {len(X_train):,} rows (from {orig_train_len:,}) "
                    "for fast, memory-bounded model training."
                )
            scoring = "f1_weighted" if task_type == "classification" else "r2"

            # The skew decision only depends on X_train's numeric columns, not
            # on which encoder a given model's preprocessor uses — identical
            # across every model trained in this call, so it's reported once
            # here rather than once per model.
            numeric_cols = [
                c for c in X_train.columns
                if pd.api.types.is_numeric_dtype(X_train[c]) and not pd.api.types.is_bool_dtype(X_train[c])
            ]
            if numeric_cols:
                skew_probe = _SkewLog1pTransformer().fit(X_train[numeric_cols])
                treatments.extend(skew_probe.describe())

            for model_name in models:
                try:
                    estimator = self._build_model(
                        model_name, task_type, max_depth,
                        balanced=balanced, scale_pos_weight=scale_pos_weight,
                    )
                except Exception as exc:
                    build_errors.append(f"{model_name}: build failed — {exc}")
                    continue
                if estimator is None:
                    build_errors.append(
                        f"{model_name}: unknown model name for task_type='{task_type}'. "
                        f"Valid names: {self.CLASSIFICATION_MODELS if task_type == 'classification' else self.REGRESSION_MODELS}"
                    )
                    continue

                try:
                    from sklearn.pipeline import Pipeline

                    preprocessor = _build_preprocessor(
                        X_train, "onehot" if model_name in LINEAR_MODELS else "ordinal"
                    )
                    model: Any = Pipeline([("prep", preprocessor), ("model", estimator)])

                    best_params: dict[str, Any] = {}
                    cv_mean: float | None = None
                    cv_std: float | None = None
                    if do_tune:
                        X_tune, y_tune, groups_tune = _tuning_sample(
                            X_train, y_train, groups_train, split_strategy, task_type
                        )
                        model, best_params, tune_cv_mean, tune_cv_std = self._tune(
                            model, model_name, X_tune, y_tune, cv, scoring, max_depth,
                            groups=groups_tune,
                        )
                        # The search's own CV score is only the model's score when
                        # it saw every training row; otherwise cross_val_score below
                        # measures the tuned model on the full training set.
                        if len(X_tune) == len(X_train):
                            cv_mean, cv_std = tune_cv_mean, tune_cv_std

                    try:
                        model.fit(X_train, y_train)
                    except Exception as fit_exc:
                        if any(k in str(fit_exc).lower() for k in ("cuda", "gpu", "out of memory", "device")):
                            treatments.append(f"{model_name}: GPU fit failed ({fit_exc}); refit on CPU.")
                            estimator = self._build_model(
                                model_name, task_type, max_depth, balanced=balanced,
                                scale_pos_weight=scale_pos_weight, force_cpu=True,
                            )
                            model = Pipeline([("prep", preprocessor), ("model", estimator)])
                            model.set_params(**{f"model__{k}": v for k, v in best_params.items()})
                            model.fit(X_train, y_train)
                        else:
                            raise

                    train_metrics = self._evaluate(model, X_train, y_train, task_type)
                    test_metrics = self._evaluate(model, X_test, y_test, task_type)

                    # Cross-validation (anti-overfitting measure). Fit only on the
                    # training fold — X/y here would leak the held-out test rows
                    # into every CV fold. When tuning ran, RandomizedSearchCV
                    # already measured this with the same splitter/scorer, so
                    # _tune's cv_mean/cv_std above are reused instead of paying
                    # for a second cross_val_score pass.
                    if cv_mean is None:
                        cv_scores = cross_val_score(
                            model, X_train, y_train, groups=groups_train,
                            cv=cv, scoring=scoring, n_jobs=1,
                        )
                        cv_mean = round(float(cv_scores.mean()), 4)
                        cv_std = round(float(cv_scores.std()), 4)

                    # Train–test gap check
                    primary_train = train_metrics.get("accuracy", train_metrics.get("r2", 0.0))
                    primary_test = test_metrics.get("accuracy", test_metrics.get("r2", 0.0))
                    gap = round(primary_train - primary_test, 4)
                    if gap > OVERFIT_THRESHOLD:
                        overfit_warnings.append(
                            f"{model_name}: train-test gap={gap:.3f} > {OVERFIT_THRESHOLD} "
                            f"— possible overfitting. Consider reducing max_depth or adding regularisation."
                        )

                    # Save model
                    model_path = Path(output_dir) / f"{model_name}.pkl"
                    save_model(model, model_path)

                    results[model_name] = {
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                        "train_test_gap": gap,
                        "model_path": str(model_path),
                        "best_params": best_params,
                    }
                except Exception as exc:
                    build_errors.append(f"{model_name}: training failed — {exc}")

            if not results:
                raise ToolExecutionError(
                    f"No models could be trained for task_type='{task_type}'. "
                    f"Errors: {'; '.join(build_errors) or 'all _build_model calls returned None — check model names and task_type.'}"
                )

            best_model = self._pick_best(results)

            # A CV score only means something against what a no-skill
            # predictor scores on the same folds — F1 0.82 on an 80%-majority
            # target is barely better than always guessing the majority. On
            # weighted F1, guessing the majority scores poorly on a balanced
            # multiclass target, so the stronger of the two no-skill
            # classifiers is the bar.
            from sklearn.dummy import DummyClassifier, DummyRegressor

            dummies = (
                [
                    DummyClassifier(strategy="most_frequent"),
                    DummyClassifier(strategy="stratified", random_state=42),
                ]
                if task_type == "classification"
                else [DummyRegressor(strategy="mean")]
            )
            try:
                baseline_cv_mean = round(max(
                    float(cross_val_score(
                        dummy, X_train, y_train, groups=groups_train,
                        cv=cv, scoring=scoring, n_jobs=1,
                    ).mean())
                    for dummy in dummies
                ), 4)
            except Exception:
                baseline_cv_mean = None

        else:
            # Clustering
            X_train, X_test = X, X
            for model_name in models:
                try:
                    from sklearn.pipeline import Pipeline

                    estimator = self._build_model(model_name, task_type, max_depth)
                    if estimator is None:
                        build_errors.append(f"{model_name}: unknown clustering model name.")
                        continue
                    preprocessor = _build_preprocessor(X_train, "ordinal")
                    model = Pipeline([("prep", preprocessor), ("model", estimator)])
                    model.fit(X_train)
                    model_path = Path(output_dir) / f"{model_name}.pkl"
                    save_model(model, model_path)
                    results[model_name] = {"model_path": str(model_path)}
                except Exception as exc:
                    build_errors.append(f"{model_name}: {exc}")

            if not results:
                raise ToolExecutionError(
                    f"No clustering models could be trained. "
                    f"Errors: {'; '.join(build_errors)}"
                )
            best_model = next(iter(results))

        best_summary = results.get(best_model, {})
        if task_type not in {"classification", "regression"}:
            baseline_cv_mean = None
        lift_over_baseline = (
            round(float(best_summary["cv_mean"]) - baseline_cv_mean, 4)
            if baseline_cv_mean is not None and isinstance(best_summary.get("cv_mean"), (int, float))
            else None
        )

        # A near-perfect score is itself evidence, even when no single column
        # explains it. Total = Unit Price x Qty is a definition spread across
        # two features, so the per-feature check above cannot see it, but an
        # R² of 0.995 on ordinary business data still means the model is
        # reconstructing an identity rather than learning anything.
        best_cv = best_summary.get("cv_mean")
        if isinstance(best_cv, (int, float)) and float(best_cv) >= _NEAR_PERFECT_SCORE:
            leakage_warnings.append(
                f"Cross-validated score is {float(best_cv):.4f} — near-perfect. "
                f"On real data this almost always means a feature (or a "
                f"combination of them, such as a total that is the product of "
                f"two other columns) defines the target. Treat this as a data "
                f"check, not a result."
            )

        # A near-perfect score is a red flag, not a headline. Say so in the
        # summary itself, because the summary is what reaches the report and
        # the LLM synthesis — a caveat buried in a sibling key gets read as
        # an endorsement of the score.
        leak_note = (
            f" ⚠ {leakage_warnings[0]}"
            if leakage_warnings
            else ""
        )
        return {
            "summary": (
                f"Trained {len(results)} model(s) [{task_type}, "
                f"split={split_strategy}]. "
                f"Best: {best_model} | "
                f"CV mean={best_summary.get('cv_mean', 'N/A')} "
                f"± {best_summary.get('cv_std', 'N/A')}."
                + (
                    f" Baseline (no-skill) CV={baseline_cv_mean}, lift={lift_over_baseline:+.4f}."
                    if lift_over_baseline is not None
                    else ""
                )
                + f"{leak_note}"
            ),
            "task_type": task_type,
            "models_trained": results,
            "best_model": best_model,
            "baseline_cv_mean": baseline_cv_mean,
            "lift_over_baseline": lift_over_baseline,
            "class_labels": class_labels,
            "overfit_warnings": overfit_warnings,
            "leakage_warnings": leakage_warnings,
            "treatments_applied": treatments,
            "hyperparameter_tuning": do_tune,
            "test_size": test_size,
            "n_cv_folds": n_cv_folds,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "split_strategy": split_strategy if task_type in {"classification", "regression"} else "n/a",
            "time_column": time_column if split_strategy == "time_series" else None,
            "group_column": group_column if split_strategy == "panel" else None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(
        self,
        name: str,
        task_type: str,
        max_depth: int,
        balanced: bool = False,
        scale_pos_weight: float = 1.0,
        force_cpu: bool = False,
    ) -> Any:
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

        # class_weight counteracts imbalanced targets (set when the minority
        # class falls below IMBALANCE_THRESHOLD)
        cw = "balanced" if balanced else None

        model_map: dict[str, Any] = {
            "random_forest_classification": RandomForestClassifier(
                n_estimators=100, max_depth=max_depth, random_state=42, n_jobs=-1,
                class_weight=cw,
            ),
            "random_forest_regression": RandomForestRegressor(
                n_estimators=100, max_depth=max_depth, random_state=42, n_jobs=-1
            ),
            "logistic_regression": LogisticRegression(
                max_iter=500, C=1.0, random_state=42,  # L2 regularisation (default)
                class_weight=cw,
            ),
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=1.0),  # L2 regularisation
            "kmeans": KMeans(n_clusters=3, random_state=42, n_init="auto"),
            "dbscan": DBSCAN(eps=0.5, min_samples=5),
        }

        try:
            from xgboost import XGBClassifier, XGBRegressor
            xgb_kwargs: dict[str, Any] = {
                "n_estimators": 200,
                "max_depth": max_depth,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbosity": 0,
                "n_jobs": -1,
                "tree_method": "hist",
            }
            if not force_cpu and _detect_cuda_gpu():
                xgb_kwargs["device"] = "cuda"

            model_map["xgboost_classification"] = XGBClassifier(
                **xgb_kwargs,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight if balanced else 1.0,
            )
            model_map["xgboost_regression"] = XGBRegressor(**xgb_kwargs)
        except ImportError:
            pass  # XGBoost not installed; xgboost model names will resolve to None

        # NOTE: do not use `model_map.get(key) or model_map.get(name)` here.
        # sklearn ensembles define __len__ via the unfitted `estimators_`
        # attribute, so truthiness checks raise AttributeError before fit.
        model = model_map.get(f"{name}_{task_type}")
        if model is None:
            model = model_map.get(name)
        return model

    #: Search spaces for the light hyperparameter tuning pass.
    #: random_forest's max_depth space is rebuilt at tune time so the search
    #: never exceeds the user's max_depth cap.
    TUNING_GRIDS: ClassVar[dict[str, dict[str, list[Any]]]] = {
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 6],
            "min_samples_leaf": [1, 2, 4],
        },
        "xgboost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
        },
        "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0]},
        "ridge": {"alpha": [0.1, 1.0, 10.0]},
    }

    def _tune(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: Any,
        scoring: str,
        max_depth: int = 6,
        groups: pd.Series[Any] | None = None,
    ) -> tuple[Any, dict[str, Any], float | None, float | None]:
        """
        Light randomized hyperparameter search; returns
        (best_model, best_params, cv_mean, cv_std).

        ``model`` is a `Pipeline([("prep", ...), ("model", estimator)])` —
        the grid keys are prefixed `model__` so `RandomizedSearchCV` tunes
        the estimator step, not the whole pipeline; the prefix is stripped
        from the returned `best_params` so the report shows plain names.

        Bounded by design: n_iter ≤ 8, the caller's CV splitter, seeded. Models
        without a defined grid pass through untuned. Depth-bearing grids are
        clamped to the user's max_depth so tuning can never undo the
        anti-overfitting cap.

        ``cv_mean``/``cv_std`` are derived from the search's own CV results
        (``best_score_`` and the spread of ``mean_test_score``) rather than a
        second ``cross_val_score`` call — same splitter, same scorer, so a
        separate call would just re-measure what the search already measured.
        """
        from sklearn.model_selection import RandomizedSearchCV

        grid = self.TUNING_GRIDS.get(model_name)
        if not grid:
            return model, {}, None, None
        grid = dict(grid)
        if "max_depth" in grid:
            grid["max_depth"] = sorted({max(2, max_depth // 2), max(2, max_depth - 1), max_depth})
        n_combos = 1
        for values in grid.values():
            n_combos *= len(values)
        prefixed_grid = {f"model__{k}": v for k, v in grid.items()}
        search = RandomizedSearchCV(
            model,
            param_distributions=prefixed_grid,
            n_iter=min(8, n_combos),
            cv=cv,
            scoring=scoring,
            random_state=42,
            n_jobs=1,
        )
        search.fit(X_train, y_train, groups=groups)
        cv_mean = round(float(search.best_score_), 4)
        # std_test_score at best_index_, not std(mean_test_score) across all
        # candidates — the latter is spread *between configurations*, not the
        # fold-to-fold variability of the selected one, which is what the
        # untuned cross_val_score() path (and the "± X" summary text) means.
        std_scores = np.asarray(search.cv_results_["std_test_score"], dtype=float)
        cv_std = round(float(std_scores[search.best_index_]), 4)
        best_params = {
            k.removeprefix("model__"): v for k, v in search.best_params_.items()
        }
        return search.best_estimator_, best_params, cv_mean, cv_std

    def _evaluate(
        self, model: Any, X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )
        # predict() on every classifier used here (RF, XGBoost,
        # LogisticRegression) is exactly classes_[argmax(predict_proba(X))] —
        # calling both separately ran the whole Pipeline (preprocessing +
        # model) twice per _evaluate() call. Deriving y_pred from the one
        # predict_proba() pass halves that cost with an identical result;
        # only a classifier without predict_proba falls back to predict().
        if task_type == "classification" and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)
            y_pred = model.classes_[np.argmax(y_prob, axis=1)]
            metrics: dict[str, float] = {
                "accuracy": round(float(accuracy_score(y, y_pred)), 4),
                "f1_score": round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4),
            }
            if y_prob.shape[1] == 2:
                metrics["roc_auc"] = round(float(roc_auc_score(y, y_prob[:, 1])), 4)
        elif task_type == "classification":
            y_pred = model.predict(X)
            metrics = {
                "accuracy": round(float(accuracy_score(y, y_pred)), 4),
                "f1_score": round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4),
            }
        else:
            y_pred = model.predict(X)
            metrics = {
                "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
                "mae": round(float(mean_absolute_error(y, y_pred)), 4),
                "r2": round(float(r2_score(y, y_pred)), 4),
            }
        return metrics

    def _pick_best(self, results: dict[str, Any]) -> str:
        """Rank by cv_mean, penalised for overfitting past OVERFIT_THRESHOLD.

        A model that tops cv_mean but is already flagged in overfit_warnings
        (train_test_gap too high) used to still be crowned "best" and
        propagate as best_model_path/best_model_name through evaluate_model,
        the feature-importance chart, and the final report — inconsistent
        with the system's own anti-overfitting stance (IMPROVEMENTS.md #5).
        """
        if not results:
            return "none"

        def _score(name: str) -> float:
            r = results[name]
            cv_mean = float(r.get("cv_mean", -float("inf")))
            gap = float(r.get("train_test_gap", 0.0))
            penalty = OVERFIT_PENALTY_WEIGHT * max(0.0, gap - OVERFIT_THRESHOLD)
            return cv_mean - penalty

        return max(results, key=_score)

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to cleaned dataset.", "required": True},
            "target_column": {"type": "string", "description": "Column to predict.", "required": True},
            "task_type": {
                "type": "string",
                "description": "classification | regression | clustering | auto.",
                "required": False,
            },
            "models": {
                "type": "list[string]",
                "description": "Model names to train. Defaults to all suitable models.",
                "required": False,
            },
            "test_size": {"type": "float", "description": "Test split fraction. Default: 0.2.", "required": False},
            "n_cv_folds": {"type": "int", "description": "Number of CV folds. Default: 5.", "required": False},
            "max_depth": {"type": "int", "description": "Max tree depth (RF, XGB). Default: 6.", "required": False},
            "tune_hyperparameters": {
                "type": "bool",
                "description": (
                    "Run a light randomized hyperparameter search (auto-skipped above 20k "
                    "rows). Opt-in ('Thorough' mode) — Default: false, since tuning multiplies "
                    "training time for a usually-small score gain ('Quick' mode)."
                ),
                "required": False,
            },
            "split_strategy": {
                "type": "string",
                "description": (
                    "'random' | 'time_series' | 'panel'. Auto-filled from the data "
                    "profile when the dataset is time-series or has grouped/panel "
                    "structure — leave unset to use the detected strategy."
                ),
                "required": False,
            },
            "time_column": {
                "type": "string",
                "description": "Datetime column to sort by for split_strategy='time_series'. Auto-filled from the profile.",
                "required": False,
            },
            "group_column": {
                "type": "string",
                "description": "Entity/group column for split_strategy='panel' (e.g. customer_id). Auto-filled from the profile.",
                "required": False,
            },
        }
