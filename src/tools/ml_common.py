"""
ML Pipeline — shared data preparation, preprocessing, split and leakage helpers.

Split out of ml_pipeline.py; `src.tools.ml_pipeline` re-exports every name here.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.tools.data_processing import (
    _read_df as _read_df,  # re-exported: src.tools.clustering imports it from here
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Overfitting warning threshold: gap between train and test accuracy
OVERFIT_THRESHOLD = 0.10


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


def _cap_train_rows(
    X_train: pd.DataFrame,
    y_train: pd.Series[Any],
    groups_train: pd.Series[Any] | None,
    split_strategy: str,
    task_type: str,
) -> tuple[pd.DataFrame, pd.Series[Any], pd.Series[Any] | None, int | None]:
    """
    Apply MAX_TRAIN_SAMPLES (env, default 50,000) to the training split.

    Shared by TrainModelTool (which fits on the result) and EvaluateModelTool
    (which must score the train side on the *same* rows to report an honest
    train_test_gap) — same env var, same seed, same selection rule per
    split_strategy, so the two tools can never disagree about which rows were
    actually "training" rows for a given saved model.

    Returns (X_train, y_train, groups_train, orig_rows); ``orig_rows`` is the
    pre-cap row count, or None when the cap didn't trigger (len(X_train) is
    already <= the cap, or MAX_TRAIN_SAMPLES <= 0 disables it).
    """
    max_train_samples = int(os.getenv("MAX_TRAIN_SAMPLES", "50000"))
    if max_train_samples <= 0 or len(X_train) <= max_train_samples:
        return X_train, y_train, groups_train, None
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
    return X_train, y_train, groups_train, orig_train_len


def _encode_target(
    y: pd.Series[Any], words: dict[str, str] | None = None
) -> tuple[pd.Series[Any], list[str]]:
    """
    Deterministically encode classification targets to contiguous integers 0..k-1.

    LabelEncoder sorts classes (numerically for numeric targets), so train and
    evaluate produce identical encodings for the same data. Numeric labels such
    as 1..5 must be encoded too: XGBoost rejects classes that do not start at 0.
    Returns (encoded_y, class_labels) with the original labels as strings.

    A yes/no target that ingestion turned into a boolean column keeps the file's own
    words (`words`, recorded by `coerce_types`), so reports say "no"/"yes" and not 0/1.
    """
    from sklearn.preprocessing import LabelEncoder

    if pd.api.types.is_bool_dtype(y) and not y.isna().any():
        present = [flag for flag in (False, True) if bool((y == flag).any())]
        names = {False: (words or {}).get("false", "False"), True: (words or {}).get("true", "True")}
        index = {flag: i for i, flag in enumerate(present)}
        encoded_bool = pd.Series(
            [index[bool(v)] for v in y], index=y.index, name=y.name, dtype="int64"
        )
        return encoded_bool, [names[flag] for flag in present]

    numeric = pd.api.types.is_numeric_dtype(y) and str(y.dtype) != "bool"
    encoder = LabelEncoder()
    encoded = pd.Series(encoder.fit_transform(y if numeric else y.astype(str)), index=y.index, name=y.name)
    return encoded, [str(c) for c in encoder.classes_]
