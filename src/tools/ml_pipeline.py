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

import os
import pickle
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

from src.core.findings import Finding
from src.core.io import DatasetReadError, read_any
from src.tools.base import BaseTool, ToolExecutionError

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata, MemorySystem
    from src.core.profiler import DatasetProfile


def _read_df(file_path: str) -> pd.DataFrame:
    """Read a dataset via the unified reader (src.core.io.read_any)."""
    try:
        df, _report = read_any(file_path)
    except DatasetReadError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return df

# Overfitting warning threshold: gap between train and test accuracy
OVERFIT_THRESHOLD = 0.10

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


def _prepare_features(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series[Any], list[str]]:
    """
    Shared train/evaluate/visualise feature preparation.

    Deterministic given the same data, so a saved model always sees the
    same feature matrix at train, evaluate, and visualisation time:
      - drops rows with a missing target
      - expands datetime columns into year/month/day/dayofweek/hour/
        is_weekend/days_since_min trend features, and drops ID-like
        columns (near-unique strings, and near-unique integer identifiers)

    This is purely structural feature engineering — no statistic is fit
    here. Skew-based log1p and categorical encoding (IMPROVEMENTS.md P0.1/
    P0.5/P0.6) are decided and fit exclusively on the training fold, inside
    the `Pipeline` built by `_build_preprocessor` — fitting them here, over
    whatever frame is passed in, would leak test-fold statistics into
    "held-out" metrics. Returned features therefore still contain raw
    categorical (string) columns and NaNs; every consumer feeds them
    through a fitted Pipeline rather than using them directly.

    Returns:
        (features, target, treatments) — treatments is a human-readable
        list of every automatic action taken, for the report.
    """
    treatments: list[str] = []
    df = df.dropna(subset=[target_column])
    y = df[target_column]
    features = df.drop(columns=[target_column]).copy()
    n_rows = max(len(features), 1)

    # is_numeric_dtype, not `dtype == object`: pandas 3 strings are `str` dtype
    for col in list(features.columns):
        series = features[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            dt = pd.to_datetime(series)
            days_since_min = (dt - dt.min()).dt.days
            features[f"{col}_year"] = dt.dt.year.fillna(-1).astype(int)
            features[f"{col}_month"] = dt.dt.month.fillna(-1).astype(int)
            features[f"{col}_day"] = dt.dt.day.fillna(-1).astype(int)
            features[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
            features[f"{col}_hour"] = dt.dt.hour.fillna(-1).astype(int)
            features[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
            # Left NaN: the Pipeline's SimpleImputer fills it from the
            # training fold only (a whole-frame median would leak the test fold).
            features[f"{col}_days_since_min"] = days_since_min.astype(float)
            features = features.drop(columns=[col])
            treatments.append(
                f"Expanded datetime column '{col}' into year/month/day/dayofweek/"
                f"hour/is_weekend/days_since_min trend features."
            )
        elif (
            not pd.api.types.is_numeric_dtype(series)
            and series.nunique() / n_rows > 0.5
        ):
            features = features.drop(columns=[col])
            treatments.append(f"Dropped ID-like text column '{col}' (>50% unique).")
        elif (
            pd.api.types.is_integer_dtype(series)
            and series.nunique() / n_rows >= 0.98
        ):
            features = features.drop(columns=[col])
            treatments.append(f"Dropped identifier column '{col}' (~100% unique integers).")

    return features, y, treatments


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


def _resolve_split_strategy(
    df: pd.DataFrame, split_strategy: str, time_column: str | None, group_column: str | None
) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Validate the requested split strategy against this dataframe and, for
    time-series, sort chronologically *before* feature prep so a later
    positional split is a chronological split (train on the past, test on
    the future). Falls back to "random" with a note — never a hard error —
    since this is often an auto-injected hint, not an explicit user choice.

    Returns (possibly-resorted df, resolved split_strategy, notes).
    """
    notes: list[str] = []
    if split_strategy == "time_series":
        if time_column and time_column in df.columns:
            sort_key = pd.to_datetime(df[time_column], errors="coerce")
            df = df.assign(**{time_column: sort_key}).sort_values(time_column).reset_index(drop=True)
        else:
            notes.append(
                f"split_strategy='time_series' requested but time_column="
                f"'{time_column}' not found — fell back to a random split."
            )
            split_strategy = "random"
    if split_strategy == "panel" and (not group_column or group_column not in df.columns):
        notes.append(
            f"split_strategy='panel' requested but group_column="
            f"'{group_column}' not found — fell back to a random split."
        )
        split_strategy = "random"
    return df, split_strategy, notes


def _split_train_test(
    X: pd.DataFrame,
    y: pd.Series[Any],
    df: pd.DataFrame,
    split_strategy: str,
    group_column: str | None,
    task_type: str,
    test_size: float,
    n_cv_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series[Any], pd.Series[Any], Any, pd.Series[Any] | None]:
    """
    Shared by TrainModelTool and EvaluateModelTool so both ever partition
    the data the same way for a given split_strategy — evaluate's whole
    purpose is to score a model on the exact rows it didn't train on.

    ``df`` must be the same (already time-sorted, if applicable) frame X/y
    were derived from, so the group column can be recovered by index even
    though _prepare_features may have transformed or dropped it.

    Returns (X_train, X_test, y_train, y_test, cv, groups_train). ``cv`` is
    unused by EvaluateModelTool but costs nothing extra to compute here.
    """
    from sklearn.model_selection import (
        GroupKFold,
        GroupShuffleSplit,
        KFold,
        StratifiedKFold,
        TimeSeriesSplit,
        train_test_split,
    )

    groups = df.loc[X.index, group_column] if split_strategy == "panel" and group_column else None
    groups_train: pd.Series[Any] | None = None
    if groups is not None and group_column in X.columns:
        # The group column is the split key, not a feature — a customer/
        # store/device id a linear or tree model would otherwise see as an
        # arbitrary label-encoded number is meaningless as model input and,
        # under a group split, the test fold carries codes train never saw.
        X = X.drop(columns=[group_column])

    if split_strategy == "time_series":
        # X is already sorted by time_column (see _resolve_split_strategy).
        split_idx = max(1, min(len(X) - 1, int(len(X) * (1 - test_size))))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        cv = TimeSeriesSplit(n_splits=min(n_cv_folds, max(2, split_idx - 1)))
    elif split_strategy == "panel" and groups is not None:
        # Same entity never appears in both train and test.
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        n_groups = int(groups_train.nunique())
        cv = GroupKFold(n_splits=max(2, min(n_cv_folds, n_groups)))
    else:
        stratify = y if task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        cv = (
            StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            if task_type == "classification"
            else KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
        )
    return X_train, X_test, y_train, y_test, cv, groups_train


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
                purity = (
                    pd.DataFrame({"f": feature, "y": y})
                    .groupby("f", observed=True)["y"]
                    .transform(lambda g: g.value_counts().iloc[0] / len(g))
                    .mean()
                )
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


def _encode_target(y: pd.Series[Any]) -> tuple[pd.Series[Any], list[str]]:
    """
    Deterministically encode non-numeric classification targets to integers.

    LabelEncoder sorts classes, so train and evaluate produce identical
    encodings for the same data. Returns (encoded_y, class_labels);
    class_labels is empty when no encoding was needed.
    """
    from sklearn.preprocessing import LabelEncoder

    if not pd.api.types.is_numeric_dtype(y) or str(y.dtype) == "bool":
        encoder = LabelEncoder()
        encoded = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index, name=y.name)
        return encoded, [str(c) for c in encoder.classes_]
    return y, []


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
            y, class_labels = _encode_target(y)
            # Act on class imbalance instead of just warning about it
            counts = y.value_counts()
            if len(counts) >= 2:
                minority_frac = float(counts.iloc[-1]) / float(counts.sum())
                if minority_frac < IMBALANCE_THRESHOLD:
                    balanced = True
                    if len(counts) == 2:
                        scale_pos_weight = float(counts.max()) / max(float(counts.min()), 1.0)
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
            max_train_samples = int(os.getenv("MAX_TRAIN_SAMPLES", "50000"))
            if max_train_samples > 0 and len(X_train) > max_train_samples:
                orig_train_len = len(X_train)
                if split_strategy == "time_series":
                    X_train = X_train.iloc[-max_train_samples:]
                    y_train = y_train.iloc[-max_train_samples:]
                elif task_type == "classification" and y_train.nunique() > 1:
                    from sklearn.model_selection import train_test_split as _tts
                    X_train, _, y_train, _ = _tts(
                        X_train, y_train, train_size=max_train_samples,
                        stratify=y_train, random_state=42
                    )
                else:
                    sample_idx = X_train.sample(n=max_train_samples, random_state=42).index
                    X_train = X_train.loc[sample_idx]
                    y_train = y_train.loc[sample_idx]
                if groups_train is not None:
                    groups_train = groups_train.loc[X_train.index]
                treatments.append(
                    f"Subsampled training set to {max_train_samples:,} rows (from {orig_train_len:,}) "
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
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)

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
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)
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
        y_pred = model.predict(X)
        if task_type == "classification":
            metrics: dict[str, float] = {
                "accuracy": round(float(accuracy_score(y, y_pred)), 4),
                "f1_score": round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4),
            }
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)
                if y_prob.shape[1] == 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y, y_prob[:, 1])), 4)
        else:
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


class EvaluateModelTool(BaseTool):
    """
    Load a saved model and run detailed evaluation on a held-out split.

    Recreates the same train/test split used by TrainModelTool
    (random_state=42) so the reported metrics describe generalisation,
    not memorisation. Produces a classification report or regression
    metrics plus the train-test gap as an overfitting diagnostic.
    """

    requires_ml = True

    name = "evaluate_model"
    description = (
        "Load a saved .pkl model and evaluate it on the held-out test split "
        "of a dataset (same random_state=42 split as train_model). "
        "Produces a full classification report (or regression metrics) "
        "plus an overfitting diagnostic (train_test_gap)."
    )
    requires_context: ClassVar[dict[str, str]] = {"target_column": "target_column"}

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return 1.0 if metadata and metadata.target_column and metadata.task_type in ("classification", "regression") else 0.0

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        params = super().prepare_params(params, memory, output_root)
        # best_model_path only overrides when the planner's model_path is
        # missing or doesn't exist — it may legitimately name a different
        # saved model on a re-evaluation step.
        best_path = memory.get_context("best_model_path")
        if best_path:
            raw_mp = params.get("model_path", "")
            if not raw_mp or not Path(raw_mp).exists():
                params["model_path"] = best_path
        # evaluate_model's whole purpose is to recreate train_model's exact
        # split ("held-out data only") — a different test_size, or a
        # different split_strategy/time_column/group_column, produces a
        # different partition, so these are forced overrides, never a
        # fill-if-absent: a plan step naming a stale value must still lose
        # to what train_model actually used.
        trained_test_size = memory.get_context("train_test_size")
        if trained_test_size is not None:
            params["test_size"] = trained_test_size
        split_strategy = memory.get_context("split_strategy")
        if split_strategy is not None:
            params["split_strategy"] = split_strategy
            params["time_column"] = memory.get_context("split_time_column")
            params["group_column"] = memory.get_context("split_group_column")
        return params

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        """
        Project `_explain_drivers`'s upgraded output — per-driver level
        effects/directions, plus the overfit gap this tool itself measured —
        onto the finding bus (IMPROVEMENTS.md 7.2 item 4).
        """
        found: list[Finding] = []
        target_column = metadata.target_column if metadata else None
        task_type = output.get("task_type")

        for drv in output.get("top_drivers") or []:
            feature = drv.get("feature")
            importance = drv.get("importance")
            level_effect = drv.get("level_effect")
            headline = drv.get("headline") or f"'{feature}' is a top driver of {target_column or 'the outcome'}"
            if level_effect:
                level = level_effect.get("level")
                lift = level_effect.get("lift")
                effect = lift if lift is not None else level_effect.get("diff")
                effect_kind = "lift" if lift is not None else "pct"
                found.append(
                    Finding(
                        finding_id=f"{self.name}_driver_{feature}_{level}",
                        kind="driver",
                        headline=headline,
                        detail=f"Permutation importance {importance}.",
                        evidence={"level_effect": level_effect, "importance": importance},
                        source_tool=self.name,
                        measure=target_column,
                        dimension=feature,
                        level=str(level) if level is not None else None,
                        effect=round(float(effect), 4) if effect is not None else None,
                        effect_kind=effect_kind,
                        confidence=0.6,
                        surprise=0.4,
                        layer="analyst",
                    )
                )
            else:
                corr = drv.get("correlation")
                effect = corr if corr is not None else importance
                found.append(
                    Finding(
                        finding_id=f"{self.name}_driver_{feature}",
                        kind="driver",
                        headline=headline,
                        detail=f"Permutation importance {importance}.",
                        evidence={"importance": importance, "correlation": corr},
                        source_tool=self.name,
                        measure=target_column,
                        dimension=feature,
                        effect=round(float(effect), 4) if effect is not None else None,
                        effect_kind="r" if corr is not None else None,
                        confidence=0.5,
                        layer="analyst",
                    )
                )

        gap = output.get("train_test_gap")
        if isinstance(gap, (int, float)) and abs(gap) > OVERFIT_THRESHOLD:
            found.append(
                Finding(
                    finding_id=f"{self.name}_overfit_gap",
                    kind="method_fit",
                    headline=f"Train-test gap of {gap:+.3f} suggests possible overfitting",
                    detail=output.get("summary", ""),
                    evidence={"train_test_gap": gap, "task_type": task_type},
                    source_tool=self.name,
                    measure=target_column,
                    caveats=[f"train_test_gap={gap:+.4f} exceeds the {OVERFIT_THRESHOLD} threshold."],
                    confidence=0.6,
                    layer="analyst",
                )
            )

        return found

    def execute(  # type: ignore[override]
        self,
        model_path: str,
        file_path: str,
        target_column: str,
        task_type: str = "classification",
        test_size: float = 0.2,
        split_strategy: str = "random",
        time_column: str | None = None,
        group_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        from sklearn.metrics import classification_report

        if not Path(model_path).exists():
            raise ToolExecutionError(f"Model file not found: {model_path}")

        df = _read_df(file_path)
        if target_column not in df.columns:
            raise ToolExecutionError(f"Target column '{target_column}' not in dataset.")

        df, split_strategy, _notes = _resolve_split_strategy(
            df, split_strategy, time_column, group_column
        )
        X, y, _treatments = _prepare_features(df, target_column)
        class_labels: list[str] = []
        if task_type == "classification":
            y, class_labels = _encode_target(y)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Recreate train_model's exact split so evaluation runs on rows the
        # model never trained on, whichever strategy produced them.
        X_train, X_test, y_train, y_test, _cv, _groups_train = _split_train_test(
            X, y, df, split_strategy, group_column, task_type, test_size
        )
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        drivers, driver_narrative = self._explain_drivers(
            model, X_test, y_test, task_type, target_column, class_labels
        )

        if task_type == "classification":
            from sklearn.metrics import accuracy_score

            report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
            test_acc = float(accuracy_score(y_test, y_pred_test))
            train_acc = float(accuracy_score(y_train, y_pred_train))
            gap = round(train_acc - test_acc, 4)
            # Map encoded integer class keys back to original label names
            if class_labels:
                report = {
                    (class_labels[int(k)] if k.isdigit() and int(k) < len(class_labels) else k): v
                    for k, v in report.items()
                }
            return {
                "summary": (
                    f"Held-out evaluation complete. Test accuracy: {test_acc:.4f} "
                    f"(train: {train_acc:.4f}, gap: {gap:+.4f})."
                ),
                "classification_report": report,
                "task_type": task_type,
                "accuracy": round(test_acc, 4),
                "train_accuracy": round(train_acc, 4),
                "train_test_gap": gap,
                "class_labels": class_labels,
                "top_drivers": drivers,
                "driver_narrative": driver_narrative,
                **self._held_out_curves(model, X_test, y_test, y_pred_test, class_labels),
            }
        else:
            from sklearn.metrics import mean_squared_error, r2_score

            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            r2_test = float(r2_score(y_test, y_pred_test))
            r2_train = float(r2_score(y_train, y_pred_train))
            gap = round(r2_train - r2_test, 4)
            return {
                "summary": (
                    f"Held-out evaluation complete. RMSE: {rmse:.4f}, "
                    f"R²: {r2_test:.4f} (train R²: {r2_train:.4f}, gap: {gap:+.4f})."
                ),
                "rmse": round(rmse, 4),
                "r2": round(r2_test, 4),
                "train_r2": round(r2_train, 4),
                "train_test_gap": gap,
                "task_type": task_type,
                "top_drivers": drivers,
                "driver_narrative": driver_narrative,
            }

    #: ROC points kept for the dashboard — enough for a smooth curve, small
    #: enough to inline into dashboard.json and the HTML report.
    _MAX_ROC_POINTS = 100

    @classmethod
    def _held_out_curves(
        cls,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred_test: Any,
        class_labels: list[str],
    ) -> dict[str, Any]:
        """Confusion matrix (rows = actual, columns = predicted, ordered like
        `class_labels`) and, for a binary target, ROC points + AUC — all on the
        held-out split. Never fails evaluation."""
        from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

        out: dict[str, Any] = {}
        try:
            labels = list(range(len(class_labels))) if class_labels else sorted(pd.Series(y_test).unique())
            out["confusion_matrix"] = confusion_matrix(y_test, y_pred_test, labels=labels).tolist()
        except Exception:
            pass
        try:
            if hasattr(model, "predict_proba") and pd.Series(y_test).nunique() == 2:
                proba = model.predict_proba(X_test)[:, 1]
                positive = sorted(pd.Series(y_test).unique())[-1]
                fpr, tpr, _ = roc_curve(y_test, proba, pos_label=positive)
                if len(fpr) > cls._MAX_ROC_POINTS:
                    keep = np.linspace(0, len(fpr) - 1, cls._MAX_ROC_POINTS).astype(int)
                    fpr, tpr = fpr[keep], tpr[keep]
                out["roc_curve"] = {"fpr": [round(float(v), 4) for v in fpr],
                                    "tpr": [round(float(v), 4) for v in tpr]}
                out["roc_auc"] = round(float(roc_auc_score(y_test == positive, proba)), 4)
        except Exception:
            pass
        return out

    @staticmethod
    def _explain_drivers(
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
        target_column: str,
        class_labels: list[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Model-agnostic explainability: permutation importance on the held-out
        split ranks the top drivers; how each is then explained depends on
        its shape (IMPROVEMENTS.md 7.2 item 4):

          - numeric/measure driver: direction (increases/decreases) plus the
            raw feature-target correlation, as before — just made explicit
            in the returned dict instead of living only in the ranking.
          - categorical/dimension driver: a "level effect" — the per-level
            mean of the outcome (predicted probability of the positive class
            for classification, the target itself for regression), compared
            to the overall baseline, so a driver like `contract` is reported
            as "month-to-month customers churn at 3.1x the base rate"
            instead of "a categorical feature" with no direction named.

        X_test carries raw (post-P0.1) columns, including string categoricals
        for a Pipeline-wrapped model, which is what makes the per-level
        groupby possible here without re-deriving the encoding.

        Failure here must never fail evaluation — returns empty results instead.
        """
        try:
            from sklearn.inspection import permutation_importance

            perm = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
            )
            order = perm.importances_mean.argsort()[::-1][:5]
            positive_label: str | None = None
            if len(class_labels) == 2:
                positive_label = class_labels[-1]
            elif task_type == "classification" and pd.Series(y_test).nunique() == 2:
                # Numeric binary target — name the positive class by its value
                positive_label = f"{target_column}={sorted(pd.Series(y_test).unique())[-1]}"

            # The "outcome" per-row used for level-effect / correlation
            # comparisons: predicted probability of the positive class when
            # available (classification), else the actual target — a
            # constant, reused for every driver rather than recomputed.
            outcome = pd.Series(y_test).astype(float)
            is_rate = False
            if task_type == "classification" and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test)
                    if proba.shape[1] == 2:
                        outcome = pd.Series(proba[:, 1], index=X_test.index)
                        is_rate = True
                except Exception:
                    pass
            baseline = float(outcome.mean())

            drivers: list[dict[str, Any]] = []
            narrative: list[str] = []
            for rank, idx in enumerate(order, 1):
                importance = float(perm.importances_mean[idx])
                if importance <= 0:
                    continue
                feature = str(X_test.columns[idx])
                col = X_test.iloc[:, idx]

                if pd.api.types.is_numeric_dtype(col):
                    corr = float(col.corr(pd.Series(y_test).astype(float)))
                    direction = "increases" if corr >= 0 else "decreases"
                    entry: dict[str, Any] = {
                        "feature": feature,
                        "importance": round(importance, 4),
                        "importance_std": round(float(perm.importances_std[idx]), 4),
                        "kind": "numeric",
                        "direction": direction,
                        "correlation": round(corr, 4),
                        "level_effect": None,
                    }
                    if task_type == "classification":
                        toward = f"'{positive_label}'" if positive_label else "the higher-encoded class"
                        headline = (
                            f"'{feature}' — higher values "
                            f"{'push predictions toward ' + toward if direction == 'increases' else 'push predictions away from ' + toward}"
                            f" (r={corr:.2f})"
                        )
                    else:
                        headline = (
                            f"'{feature}' — higher values {direction} "
                            f"predicted '{target_column}' (r={corr:.2f})"
                        )
                    entry["headline"] = headline
                    drivers.append(entry)
                    narrative.append(
                        f"#{rank} driver: {headline} (permutation importance {importance:.3f})."
                    )
                    continue

                # Categorical/dimension driver — per-level mean of the
                # outcome vs. the overall baseline, so the direction and
                # magnitude are named instead of collapsing to "categorical".
                level_frame = pd.DataFrame({"level": col.astype(str), "outcome": outcome})
                level_means = level_frame.groupby("level", observed=True)["outcome"].mean()
                if level_means.empty:
                    continue
                deviations = (level_means - baseline).abs()
                best_level = str(deviations.idxmax())
                level_value = float(level_means.loc[best_level])
                diff = level_value - baseline
                lift = (level_value / baseline) if abs(baseline) > 1e-9 else None
                direction_word = "up" if diff >= 0 else "down"
                level_effect = {
                    "level": best_level,
                    "level_value": round(level_value, 4),
                    "baseline": round(baseline, 4),
                    "diff": round(diff, 4),
                    "lift": round(lift, 4) if lift is not None else None,
                    "direction": direction_word,
                }
                entry = {
                    "feature": feature,
                    "importance": round(importance, 4),
                    "importance_std": round(float(perm.importances_std[idx]), 4),
                    "kind": "categorical",
                    "direction": None,
                    "level_effect": level_effect,
                }

                if is_rate:
                    if lift is not None:
                        headline = (
                            f"'{feature}' = '{best_level}': predicted {target_column} rate is "
                            f"{lift:.2f}x the baseline ({level_value:.1%} vs {baseline:.1%})"
                        )
                    else:
                        headline = (
                            f"'{feature}' = '{best_level}': predicted {target_column} rate is "
                            f"{level_value:.1%}, {abs(diff) * 100:.1f} pts {direction_word} "
                            f"vs baseline {baseline:.1%}"
                        )
                elif lift is not None:
                    headline = (
                        f"'{feature}' = '{best_level}': average {target_column} is {lift:.2f}x "
                        f"the overall baseline ({level_value:.3g} vs {baseline:.3g})"
                    )
                else:
                    headline = (
                        f"'{feature}' = '{best_level}': average {target_column} is "
                        f"{level_value:.3g}, {diff:+.3g} vs baseline {baseline:.3g}"
                    )
                entry["headline"] = headline
                drivers.append(entry)
                narrative.append(
                    f"#{rank} driver: {headline} (permutation importance {importance:.3f})."
                )
            return drivers, narrative
        except Exception:
            return [], []

    def get_schema(self) -> dict[str, Any]:
        return {
            "model_path": {"type": "string", "description": "Path to .pkl model file.", "required": True},
            "file_path": {"type": "string", "description": "Path to evaluation dataset.", "required": True},
            "target_column": {"type": "string", "description": "Target column name.", "required": True},
            "task_type": {
                "type": "string",
                "description": "classification | regression.",
                "required": False,
            },
            "test_size": {
                "type": "float",
                "description": "Held-out fraction — must match train_model. Default: 0.2.",
                "required": False,
            },
            "split_strategy": {
                "type": "string",
                "description": (
                    "'random' | 'time_series' | 'panel' — must match the train_model "
                    "call that produced model_path. Auto-filled from that step's result."
                ),
                "required": False,
            },
            "time_column": {"type": "string", "description": "Must match train_model. Auto-filled.", "required": False},
            "group_column": {"type": "string", "description": "Must match train_model. Auto-filled.", "required": False},
        }
