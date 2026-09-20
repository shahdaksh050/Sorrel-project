"""
Data Processing Tools — Execution Layer.

Stage 1: Dataset Ingestion  (IngestDatasetTool)
Stage 3: Data Cleaning      (CleanDataTool)
Stage 3: Outlier Detection  (DetectOutliersTool)
Stage 3: Correlation EDA    (CorrelationAnalysisTool)

All tools are deterministic and return structured dicts.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.coercion import coerce_types
from src.core.findings import Finding
from src.core.io import (
    _READ_CACHE,
    _READ_CACHE_LOCK,
    DatasetReadError,
    _cache_key,
    invalidate_read_cache,
    read_any,
)
from src.core.memory import DatasetMetadata, MemorySystem
from src.core.security import UploadValidationError, escape_csv_formulas, resolve_output_path
from src.tools.base import BaseTool, ToolExecutionError

if TYPE_CHECKING:
    from src.core.profiler import DatasetProfile

#: Semantic roles that must never be run through outlier detection — a flag
#: (0/1) has no distribution to speak of, an ordinal is a small closed scale
#: where "outlier" is meaningless, and an identifier is a key, not a measure.
_OUTLIER_EXCLUDED_ROLES = frozenset({"flag", "ordinal", "identifier"})

#: Outlier rate above which the method is judged unsuitable for a column's
#: distribution rather than having found genuine anomalies (7.10 / 6.4).
_METHOD_UNSUITABLE_PCT = 20.0

#: Modified z-score (MAD-based) threshold conventionally used as the robust
#: analogue of a 3-sigma rule (Iglewicz & Hoaglin).
_MAD_MODIFIED_Z_THRESHOLD = 3.5

#: |r| floor for CorrelationAnalysisTool.findings() — below this, a
#: correlation is noise-level on most real datasets and not worth promoting.
_CORR_FINDING_THRESHOLD = 0.3

#: Cap on how many top correlation pairs become Findings per run.
_CORR_TOP_N = 3

#: Columns linked by |r| >= this form a "group" (one shared driver reported
#: once) when three or more are connected; a lone strong pair stays a pair.
_CORR_GROUP_R = 0.9
_CORR_GROUP_MIN_COLS = 3
_CORR_GROUP_NAMES = 5  # member names spelled out in a group headline

#: A numeric column missing more than this share of its values gets a
#: coverage_gap finding from clean_data; at most _COVERAGE_GAP_MAX_COLS listed.
_COVERAGE_GAP_SHARE = 0.5
_COVERAGE_GAP_MAX_COLS = 5

#: |r| floor for promoting a feature<->target correlation to a Finding
#: (eta-squared targets use its square, the same share of variance).
_TARGET_CORR_FINDING_THRESHOLD = 0.2

#: Definitional relationships (correlation_analysis): |r| at/above this is
#: the same quantity recorded twice; a column matching a product/sum/ratio
#: of two others within _RELATION_RTOL on _RELATION_MATCH_SHARE of sampled
#: complete rows is a formula (total = qty x price), not a finding. The
#: three-column search is O(k^3), so it only runs up to _MAX_RELATION_COLS.
_IDENTITY_R = 0.995
_RELATION_RTOL = 0.01
_RELATION_MATCH_SHARE = 0.99
_RELATION_SAMPLE = 2_000
_MIN_RELATION_ROWS = 10
_MAX_RELATION_COLS = 15


def _resolve_write_path(out_dir: Path, filename: str) -> Path:
    """Join filename under out_dir via resolve_output_path (P2.3) so a
    crafted filename (e.g. from an uploaded file's stem) can't escape the
    intended output directory."""
    try:
        return resolve_output_path(out_dir, filename)
    except UploadValidationError as exc:
        raise ToolExecutionError(str(exc)) from exc


def _read_raw_df(file_path: str) -> pd.DataFrame:
    """Read a dataset exactly as stored, with no repair applied."""
    try:
        df, _report = read_any(file_path)
    except DatasetReadError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return df


#: Coerced frames by the read cache's (path, mtime, size) key. Coercing a
#: 500k-row export costs seconds per call and every tool re-reads, so the
#: repaired frame is kept for the few most recent files. An entry is only
#: served while the raw frame is still in src.core.io's cache: a writer's
#: invalidate_read_cache() (the guard against same-tick rewrites) drops both.
_COERCED_MAX_ENTRIES = 2
_COERCED: OrderedDict[tuple[str, int, int], pd.DataFrame] = OrderedDict()
_COERCED_LOCK = threading.Lock()
_COERCED_KEY_LOCKS: dict[tuple[str, int, int], threading.Lock] = {}


def analysis_sample_rows() -> int:
    """Row count above which exploratory statistics run on a seeded random
    sample (env DSA_ANALYSIS_SAMPLE_ROWS, default 200,000)."""
    try:
        return max(1_000, int(os.environ.get("DSA_ANALYSIS_SAMPLE_ROWS", "200000")))
    except ValueError:
        return 200_000


def sample_note(n_from: int, n_to: int) -> dict[str, Any]:
    """Output fields marking an exploratory result computed on a random sample."""
    return {
        "sampled_from": n_from,
        "sampled_to": n_to,
        "sample_caveat": f"computed on a random sample of {n_to:,} of {n_from:,} rows",
    }


#: pearson_matrix uses the matrix-product path from this many columns up.
_FAST_CORR_MIN_COLS = 30


def pearson_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """`df.corr(method="pearson")` (pairwise-complete). pandas loops over every
    column pair, O(k^2 n) single-threaded; on a wide frame the complete columns
    go through one matrix product instead, and only pairs involving a column
    with missing values are computed pairwise. Narrow, mostly-incomplete or
    non-finite frames use pandas as before."""
    k = df.shape[1]
    if k < _FAST_CORR_MIN_COLS:
        return df.corr(method="pearson")
    arr = df.to_numpy(dtype=float, na_value=np.nan)
    has_nan = np.isnan(arr).any(axis=0)
    if int(has_nan.sum()) * 4 > k or bool(np.isinf(arr).any()):
        return df.corr(method="pearson")
    ok, bad = np.flatnonzero(~has_nan), np.flatnonzero(has_nan)
    out = np.full((k, k), np.nan)
    with np.errstate(all="ignore"):
        out[np.ix_(ok, ok)] = np.corrcoef(arr[:, ok], rowvar=False)
        for j in bad:
            rows = np.flatnonzero(~np.isnan(arr[:, j]))
            xc = arr[rows, j] - arr[rows, j].mean()
            yc = arr[np.ix_(rows, ok)]
            yc = yc - yc.mean(axis=0)
            out[j, ok] = out[ok, j] = (yc.T @ xc) / np.sqrt((yc**2).sum(axis=0) * (xc**2).sum())
    if len(bad):
        out[np.ix_(bad, bad)] = df.iloc[:, bad].corr(method="pearson").to_numpy()
    return pd.DataFrame(out, index=df.columns, columns=df.columns)


def _read_df(file_path: str) -> pd.DataFrame:
    """
    Read a dataset ready for analysis: unified reader + type coercion.

    Coercion belongs here, not at individual call sites. A retail export
    carries money as "$1,234.56" and rates as "45.3%", which read back as
    strings. `controller.load_dataset` coerces before profiling, so the
    *profile* saw them as numeric — but every tool re-read the file through
    this helper and got the strings back, so revenue was invisible to the
    entire analysis. On a real sales file that left correlation running on
    a customer ID and a quantity, and reporting r=-0.06 between them as the
    headline finding, while never once looking at revenue.

    Coercion is idempotent, and every repair is recorded and reported by
    the ingestion path (memory context "coercions" -> the report's Data
    Overview), so nothing here is silent.
    """
    key = _cache_key(file_path)
    if key is None:
        return coerce_types(_read_raw_df(file_path))[0]
    # One lock per file: tools run in parallel batches, and N threads coercing
    # the same 500k rows at once would each pay N times the GIL-bound cost.
    with _COERCED_LOCK:
        key_lock = _COERCED_KEY_LOCKS.setdefault(key, threading.Lock())
    with key_lock:
        with _READ_CACHE_LOCK:
            raw_cached = key in _READ_CACHE
        with _COERCED_LOCK:
            hit = _COERCED.get(key) if raw_cached else _COERCED.pop(key, None)
            if hit is not None:
                _COERCED.move_to_end(key)
        if hit is not None:
            return hit.copy()
        df = _read_raw_df(file_path)
        repaired, _coercions = coerce_types(df)
        with _COERCED_LOCK:
            _COERCED[key] = repaired.copy()
            while len(_COERCED) > _COERCED_MAX_ENTRIES:
                _COERCED_KEY_LOCKS.pop(_COERCED.popitem(last=False)[0], None)
        return repaired


# ============================================================
# Tool 1: Dataset Ingestion — Stage 1
# ============================================================

class IngestDatasetTool(BaseTool):
    """
    Load a CSV/Excel file and extract rich schema metadata.

    Stage 1 of the agent workflow. The metadata returned here feeds
    directly into the Memory System and is the only dataset information
    passed to the LLM — raw data is never put in LLM context.
    """

    name = "ingest_dataset"
    description = (
        "Load a dataset from a file path. Detects schema, dtypes, missing values, "
        "class balance, high-cardinality columns, and basic statistics. "
        "Returns structured metadata — do NOT pass raw data to the LLM."
    )
    uses_cleaned_file = False

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        # Stage 1 already ingests the dataset before planning starts — never
        # a candidate for the LLM's own plan.
        return 0.0

    def execute(self, file_path: str, target_column: str | None = None, **_: Any) -> dict[str, Any]:  # type: ignore[override]
        path = Path(file_path)
        if not path.exists():
            raise ToolExecutionError(f"File not found: {file_path}")

        try:
            # Coerced, deliberately. This tool's metadata drives target
            # auto-detection and task-type inference for the whole run, so it
            # must describe the same frame every other tool analyses. Reading
            # raw here made a "$18.50" revenue column look like a string, so
            # the task was inferred as classification while train_model saw a
            # float and every model failed on a continuous target. What was
            # repaired is reported separately, in the report's Data Overview.
            df = _read_df(file_path)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Failed to read file: {exc}") from exc

        numerical_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
        missing_values: dict[str, int] = {
            k: int(v) for k, v in df.isnull().sum().items() if v > 0
        }

        # Class balance for classification targets
        class_balance: dict[str, int] = {}
        if target_column and target_column in df.columns:
            class_balance = df[target_column].value_counts().to_dict()
            class_balance = {str(k): int(v) for k, v in class_balance.items()}

        # Unique value counts for all columns (used by target-detection confidence scoring)
        column_nunique = {col: int(df[col].nunique()) for col in df.columns}

        # High-cardinality detection (>50 unique values in a categorical col)
        high_card = [
            c for c in categorical_cols
            if df[c].nunique() > 50
        ]

        metadata_dict: dict[str, Any] = {
            "file_path": str(path.resolve()),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": missing_values,
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "target_column": target_column if target_column in df.columns else None,
            "task_type": None,
            "class_balance": class_balance,
            "high_cardinality_cols": high_card,
            "column_nunique": column_nunique,
            "summary_stats": {},
        }
        # Task type has exactly one implementation: DatasetMetadata.infer_task_type().
        # A previous version duplicated this logic here with a different (wrong)
        # rule for high-cardinality object/string targets — see IMPROVEMENTS.md #1.
        task_type = DatasetMetadata(**metadata_dict).infer_task_type()
        metadata_dict["task_type"] = task_type

        # Cosmetic-but-visible fix (Round 7 audit): `infer_task_type()` with
        # no target correctly returns "clustering" as the *unsupervised
        # fallback*, but that reads as a settled, wrong answer in the log
        # when the run goes on to auto-detect a target and train a
        # classifier — target auto-detection runs later, in the controller,
        # not here. Say so honestly instead of stating a task nobody chose.
        task_type_display = task_type if target_column else "TBD (target not yet selected)"

        return {
            "summary": (
                f"Ingested: {len(df):,} rows × {len(df.columns)} cols | "
                f"{len(numerical_cols)} numerical, {len(categorical_cols)} categorical | "
                f"{sum(missing_values.values()):,} missing cells | "
                f"task={task_type_display}"
            ),
            "metadata": metadata_dict,
            "column_list": df.columns.tolist(),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the CSV or Excel file.",
                "required": True,
            },
            "target_column": {
                "type": "string",
                "description": "Name of the label/target column if known.",
                "required": False,
            },
        }


# ============================================================
# Tool 2: Data Cleaning — Stage 3
# ============================================================

class CleanDataTool(BaseTool):
    """
    Basic data cleaning (type repair via _read_df) with missing values left
    as NaN unless an imputation strategy is explicitly requested.

    Every downstream tool reads this file. Imputed values are not data: a
    median fill piles fake rows onto one value, shrinks variance and makes
    every test that follows over-confident, so analysis tools handle NaN
    themselves (complete-case per analysis; ML imputes inside its CV
    pipeline). Writes a cleaned copy to disk and returns the new file path —
    downstream tools should use `cleaned_file_path` as their input.
    """

    name = "clean_data"
    description = (
        "Clean a dataset (type repair) and report missing values, duplicate rows "
        "and constant columns. Missing values are left as NaN by default "
        "(strategy 'none'); pass 'mean', 'median', 'mode', 'drop'/'drop_rows' or "
        "'forward_fill' only when imputation is explicitly wanted. "
        "Returns cleaned_file_path for use by subsequent tools."
    )
    uses_cleaned_file = False  # this IS the tool that produces cleaned_file_path
    output_subdir = "data"

    STRATEGIES = frozenset({"none", "mean", "median", "mode", "drop", "drop_rows", "forward_fill"})

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        strategy: str = "none",
        target_column: str | None = None,
        output_dir: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if strategy not in self.STRATEGIES:
            raise ToolExecutionError(
                f"Invalid strategy '{strategy}'. Choose from: {sorted(self.STRATEGIES)}"
            )

        path = Path(file_path)
        df = _read_df(file_path)
        if df.empty:
            raise ToolExecutionError("Dataset has no rows — nothing to clean.")
        if strategy == "drop":
            strategy = "drop_rows"
        original_shape = df.shape
        missing_before = int(df.isnull().sum().sum())
        missing_by_column = {str(c): int(n) for c, n in df.isnull().sum().items() if n > 0}
        duplicate_rows = int(df.duplicated().sum())
        constant_columns = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
        # Numeric columns mostly empty (all-null ones were dropped at read).
        sparse_columns: list[dict[str, Any]] = sorted(
            (
                {"column": str(c), "missing": n, "usable": len(df) - n, "pct_missing": round(100 * n / len(df), 1)}
                for c in df.select_dtypes(include="number").columns
                if (n := int(df[c].isnull().sum())) < len(df) and n > _COVERAGE_GAP_SHARE * len(df)
            ),
            key=lambda d: int(d["missing"]), reverse=True,
        )

        # Exclude target column from imputation
        cols_to_clean = [c for c in df.columns if c != target_column]
        subset = df[cols_to_clean]

        if strategy == "mean":
            subset = subset.fillna(subset.mean(numeric_only=True))
        elif strategy == "median":
            subset = subset.fillna(subset.median(numeric_only=True))
        elif strategy == "mode":
            modes = subset.mode()
            if not modes.empty:
                subset = subset.fillna(modes.iloc[0])
        elif strategy == "drop_rows":
            df = df.dropna()
            if df.empty:
                raise ToolExecutionError(
                    "drop_rows removed every row (every row has at least one "
                    "missing value). Re-run clean_data with strategy 'none' — "
                    "analysis tools handle missing values per analysis."
                )
        elif strategy == "forward_fill":
            subset = subset.ffill()

        if strategy != "drop_rows":
            df[cols_to_clean] = subset

        missing_after = int(df.isnull().sum().sum())
        out_dir = Path(output_dir) if output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        # Read back by every downstream tool (uses_cleaned_file redirection) —
        # NOT a final user-facing export, so no escape_csv_formulas here: a
        # quoted-string repair would break exact-value matching for any
        # tool that re-reads this file expecting the original values.
        out_path = _resolve_write_path(out_dir, f"{path.stem}_cleaned.csv")  # always CSV
        df.to_csv(out_path, index=False)
        # This path may already be in the read cache from an earlier
        # step (a re-planned or retried run rewrites the same name).
        invalidate_read_cache(str(out_path))

        return {
            "summary": (
                (
                    f"Left {missing_before} missing values as NaN (no imputation). "
                    if strategy == "none"
                    else f"Cleaned {missing_before - missing_after} missing values using '{strategy}'. "
                )
                + f"{duplicate_rows} duplicate row(s), {len(constant_columns)} constant column(s). "
                f"Shape: {original_shape} → {df.shape}. Saved to '{out_path}'."
            ),
            "cleaned_file_path": str(out_path),
            "rows_before": original_shape[0],
            "rows_after": df.shape[0],
            "missing_before": missing_before,
            "missing_after": missing_after,
            "strategy_used": strategy,
            "missing_by_column": missing_by_column,
            "duplicate_rows": duplicate_rows,
            "constant_columns": constant_columns,
            "sparse_columns": sparse_columns,
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        """One coverage_gap caveat when any numeric column is mostly empty:
        analyses on it rest on far fewer rows than the dataset has."""
        sparse = output.get("sparse_columns") or []
        rows = output.get("rows_before")
        if not sparse or not rows:
            return []
        worst = sparse[0]
        listed = sparse[:_COVERAGE_GAP_MAX_COLS]
        if len(sparse) == 1:
            headline = (
                f"{worst['column']} is {worst['pct_missing']:.0f}% missing "
                f"({worst['usable']:,} of {rows:,} rows usable); analyses using it rest on far fewer rows"
            )
        else:
            headline = (
                f"{len(sparse)} numeric columns are over {_COVERAGE_GAP_SHARE:.0%} missing "
                f"(worst: {worst['column']}, {worst['pct_missing']:.0f}% missing, "
                f"{worst['usable']:,} of {rows:,} rows usable); analyses using them rest on far fewer rows"
            )
        strategy = output.get("strategy_used")
        handling = (
            "Gaps were left unfilled: correlations use only the rows where both columns have "
            "a value (pairwise deletion) and other analyses use complete cases."
            if strategy == "none"
            else f"Gaps were handled with '{strategy}', so those values are estimates, not observations."
        )
        return [Finding(
            finding_id=f"{self.name}_coverage_gap",
            kind="coverage_gap",
            headline=headline,
            detail="; ".join(
                f"{d['column']}: {d['usable']:,} usable of {rows:,} ({d['pct_missing']:.0f}% missing)" for d in listed
            ) + ". " + handling,
            evidence={"columns": listed, "rows": rows, "n_sparse_columns": len(sparse)},
            source_tool=self.name,
            measure=worst["column"],
            confidence=0.9,
            caveats=[handling],
            layer="analyst",
        )]

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to the dataset file.", "required": True},
            "strategy": {
                "type": "string",
                "description": (
                    "Missing-value handling: none (default — leave NaN; analysis tools "
                    "use complete cases) | mean | median | mode | drop | forward_fill."
                ),
                "required": False,
            },
            "target_column": {
                "type": "string",
                "description": "Column to exclude from imputation (label column).",
                "required": False,
            },
        }


def _iqr_bounds(values: pd.Series) -> tuple[float, float]:
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def _column_outlier_mask(
    series: pd.Series, method: str, threshold: float, is_skewed: bool
) -> tuple[pd.Series, str]:
    """
    Flag outliers in one numeric column, choosing a distribution-aware
    method (7.10 / 6.4).

    Raw IQR/z-score on a right-skewed measure (revenue, tenure, claim
    amounts...) treats the column's natural long tail as anomalous, flagging
    20-40% of rows on real data — a method-fit failure, not a finding. When
    the column is flagged `severe_skew` by the profiler, this uses IQR on
    log-transformed values (only valid when every value is positive) or
    falls back to a MAD-based robust threshold (Iglewicz & Hoaglin's
    modified z-score), which is far less sensitive to a long tail than a
    raw standard deviation or raw quartile spread.

    Returns (boolean mask aligned to `series.index`, method actually used).
    """
    clean = series.dropna()
    empty_mask = pd.Series(False, index=series.index)
    if clean.empty:
        return empty_mask, method

    if is_skewed and method in ("iqr", "zscore"):
        if (clean > 0).all():
            log_vals = np.log(clean)
            lower, upper = _iqr_bounds(log_vals)
            flagged = (log_vals < lower) | (log_vals > upper)
            used = "iqr_log"
        else:
            median = float(clean.median())
            mad = float((clean - median).abs().median())
            if mad == 0:
                flagged = pd.Series(False, index=clean.index)
            else:
                modified_z = 0.6745 * (clean - median) / mad
                flagged = modified_z.abs() > _MAD_MODIFIED_Z_THRESHOLD
            used = "mad"
    elif method == "iqr":
        lower, upper = _iqr_bounds(clean)
        flagged = (clean < lower) | (clean > upper)
        used = "iqr"
    elif method == "zscore":
        from scipy import stats
        z = pd.Series(np.abs(stats.zscore(clean)), index=clean.index)
        flagged = z > threshold
        used = "zscore"
    else:
        raise ValueError(f"Unsupported per-column method '{method}'")

    mask = empty_mask.copy()
    mask.loc[flagged.index[flagged]] = True
    return mask, used


# ============================================================
# Tool 3: Outlier Detection — Stage 3
# ============================================================

class DetectOutliersTool(BaseTool):
    """
    Detect outliers in numerical columns using configurable methods.

    Methods: IQR (default), Z-score, Isolation Forest.

    7.10 (subsumes 6.4) — method-fit guards: flags, ordinals, and
    identifiers are never checked (a 0/1 flag or a small closed scale has no
    "outlier" concept), and a column the profiler flagged `severe_skew` gets
    a distribution-aware method (log-IQR or MAD) instead of raw IQR/z-score,
    which otherwise mistakes a heavy right tail for anomalies. Columns whose
    flag rate still exceeds 20% are reported as method-unsuitable rather
    than as a finding.
    """

    name = "detect_outliers"
    description = (
        "Detect outliers in numerical features. "
        "Methods: 'iqr' (default), 'zscore', 'isolation_forest'. "
        "Returns per-column counts and an outlier-flagged dataset path."
    )
    output_subdir = "data"

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None:
            return 1.0
        return 1.0 if profile.columns_of_kind("numeric") else 0.0

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        """
        Thread the profiler's semantic roles into execute() (7.10).

        execute() has no direct access to the DatasetProfile object — every
        other tool in this codebase that needs profile facts at execute time
        (e.g. train_model's split_strategy in ml_pipeline.py) pulls them from
        `memory.get_context("data_profile")` (the dict form set by the
        controller after profiling) inside prepare_params and injects them
        as extra params, rather than changing execute()'s call contract.
        Same pattern here: role-by-column and the severe-skew set are
        injected as internal `_`-prefixed params.
        """
        params = super().prepare_params(params, memory, output_root)
        profile = memory.get_context("data_profile") or {}
        cols = profile.get("columns") or []
        params["_semantic_roles"] = {c.get("name"): c.get("semantic_role") for c in cols}
        params["_skewed_cols"] = {c.get("name") for c in cols if "severe_skew" in (c.get("flags") or [])}
        return params

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        method: str = "iqr",
        threshold: float = 3.0,
        columns: list[str] | None = None,
        output_dir: str | None = None,
        _semantic_roles: dict[str, str] | None = None,
        _skewed_cols: set[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        path = Path(file_path)
        df = _read_df(file_path)
        num_df = df.select_dtypes(include="number")

        if columns:
            valid = [c for c in columns if c in num_df.columns]
            num_df = num_df[valid]

        role_by_col = _semantic_roles or {}
        skewed_cols = _skewed_cols or set()
        excluded = {
            c: role_by_col[c]
            for c in num_df.columns
            if role_by_col.get(c) in _OUTLIER_EXCLUDED_ROLES
        }
        if excluded:
            num_df = num_df.drop(columns=list(excluded))

        if num_df.empty:
            raise ToolExecutionError(
                "No numerical columns eligible for outlier detection "
                "(remaining columns are flags, ordinals, or identifiers)."
                if excluded
                else "No numerical columns found for outlier detection."
            )

        row_count = len(df)
        report: dict[str, Any] = {"method": method, "columns_excluded": excluded}
        per_column_detail: dict[str, dict[str, Any]] = {}
        per_column_outliers: dict[str, int] = {}

        if method in ("iqr", "zscore"):
            mask = pd.Series(False, index=df.index)
            for col in num_df.columns:
                col_mask, used = _column_outlier_mask(
                    num_df[col], method, threshold, col in skewed_cols
                )
                mask = mask | col_mask
                n_flagged = int(col_mask.sum())
                pct_flagged = round(n_flagged / max(row_count, 1) * 100, 2)
                per_column_outliers[col] = n_flagged
                per_column_detail[col] = {
                    "method_used": used,
                    "n_flagged": n_flagged,
                    "pct_flagged": pct_flagged,
                    "method_unsuitable": pct_flagged > _METHOD_UNSUITABLE_PCT,
                }
            if method == "zscore":
                report["threshold"] = threshold
            report["total_outliers"] = int(mask.sum())

        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            clean = num_df.dropna()
            # Fit on a seeded sample above the size cap, score every row.
            cap = analysis_sample_rows()
            fit_rows = clean if len(clean) <= cap else clean.sample(n=cap, random_state=0)
            preds = model.fit(fit_rows).predict(clean)
            if len(fit_rows) < len(clean):
                report.update(sample_note(len(clean), len(fit_rows)))
            mask = pd.Series(False, index=df.index)
            mask.loc[clean.index[preds == -1]] = True
            total_iso = int((preds == -1).sum())
            # Complete rows only (missing values are no longer imputed
            # upstream), so rates are over the rows actually scored.
            report["n_rows_scored"] = len(clean)
            pct_iso = round(total_iso / max(len(clean), 1) * 100, 2)
            # Isolation Forest is multivariate — it flags rows, not columns —
            # so there is no per-feature decomposition of the count. Every
            # included column reports the same (global) figure, tagged so
            # callers can tell it apart from a true per-column method.
            for col in num_df.columns:
                per_column_outliers[col] = total_iso
                per_column_detail[col] = {
                    "method_used": "isolation_forest",
                    "n_flagged": total_iso,
                    "pct_flagged": pct_iso,
                    "method_unsuitable": pct_iso > _METHOD_UNSUITABLE_PCT,
                }
            report["total_outliers"] = total_iso
            report["contamination"] = 0.05

        else:
            raise ToolExecutionError(f"Unknown method '{method}'. Use: iqr, zscore, isolation_forest")

        total = report.get("total_outliers", 0)
        report["row_count"] = row_count
        report["outlier_percentage"] = round(
            total / max(report.get("n_rows_scored", row_count), 1) * 100, 2
        )
        report["per_column_outliers"] = per_column_outliers
        report["per_column_detail"] = per_column_detail
        report["method_unsuitable_columns"] = [
            c for c, d in per_column_detail.items() if d["method_unsuitable"]
        ]

        # Save flagged dataset. This is a final, user-facing artifact (opened
        # in Excel/Sheets to inspect flagged rows) — never read back by the
        # pipeline itself — so it is the one export in this file that gets
        # escape_csv_formulas as well as the path-escape guard.
        df["_is_outlier"] = mask
        out_dir = Path(output_dir) if output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = _resolve_write_path(out_dir, f"{path.stem}_outliers_flagged.csv")  # always CSV
        escape_csv_formulas(df).to_csv(out_path, index=False)
        # This path may already be in the read cache from an earlier
        # step (a re-planned or retried run rewrites the same name).
        invalidate_read_cache(str(out_path))

        unsuitable_note = (
            f" {len(report['method_unsuitable_columns'])} column(s) flagged "
            ">20% of rows — method unsuitable for their distribution, see "
            "method_unsuitable_columns."
            if report["method_unsuitable_columns"]
            else ""
        ) + (f" Model {report['sample_caveat']}." if "sample_caveat" in report else "")

        return {
            "summary": (
                f"Outlier detection ({method}): {total:,} outliers "
                f"({report['outlier_percentage']}% of data). "
                f"Flagged dataset saved to '{out_path}'.{unsuitable_note}"
            ),
            "flagged_file_path": str(out_path),
            **report,
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        """
        Only method-fit caveats are emitted here (7.10). A reasonable (<20%)
        outlier rate on any one column is not insight-grade on its own — a
        raw count of flagged rows has no meaning without a peer/segment
        comparison, so T5 triviality suppression would drop it anyway.
        Columns where the method itself was unsuitable for the distribution
        are the one thing worth surfacing: as a caveat/limitation, not as an
        anomaly finding, which is why `kind="method_fit"` is exempt from
        that suppression (src.core.findings.is_trivial).
        """
        results: list[Finding] = []
        for col, detail in (output.get("per_column_detail") or {}).items():
            if not detail.get("method_unsuitable"):
                continue
            pct = detail.get("pct_flagged", 0)
            results.append(Finding(
                finding_id=f"{self.name}_unsuitable_{col}",
                kind="method_fit",
                headline=(
                    f"Outlier detection unsuitable for '{col}' — "
                    f"{pct}% flagged, distribution is naturally heavy-tailed"
                ),
                detail=(
                    f"'{detail.get('method_used')}' flagged {detail.get('n_flagged')} "
                    f"of {output.get('total_outliers', '?')} rows in '{col}' ({pct}%). "
                    "At this rate the method is describing the column's natural "
                    "long tail, not detecting genuine anomalies."
                ),
                evidence={"column": col, **detail},
                source_tool=self.name,
                measure=col,
                effect=None,
                confidence=0.6,
                caveats=[
                    f"Outlier counts for '{col}' should not be read as an anomaly "
                    "finding — the underlying distribution is heavy-tailed, and "
                    "the detection method used here isn't a good fit for it."
                ],
                layer="appendix",
            ))
        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to dataset.", "required": True},
            "method": {
                "type": "string",
                "description": "Detection method: iqr | zscore | isolation_forest.",
                "required": False,
            },
            "threshold": {
                "type": "float",
                "description": "Z-score threshold (zscore method only). Default: 3.0.",
                "required": False,
            },
            "columns": {
                "type": "list[string]",
                "description": "Subset of columns to check. Defaults to all numerical.",
                "required": False,
            },
        }


def _correlation_groups(
    corr: pd.DataFrame, skip: set[frozenset[str]],
) -> list[dict[str, Any]]:
    """Connected groups (union-find over |r| >= _CORR_GROUP_R pairs, definitional
    pairs in `skip` ignored) of at least _CORR_GROUP_MIN_COLS columns, in column
    order, each with its member r range and mean |r| over all member pairs."""
    cols = [str(c) for c in corr.columns]
    parent = {c: c for c in cols}

    def find(c: str) -> str:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    values = corr.to_numpy(dtype=float)
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            r = values[i, j]
            if not np.isnan(r) and abs(r) >= _CORR_GROUP_R and frozenset((a, cols[j])) not in skip:
                parent[find(a)] = find(cols[j])
    members: dict[str, list[int]] = {}
    for i, c in enumerate(cols):
        members.setdefault(find(c), []).append(i)

    groups: list[dict[str, Any]] = []
    for idx in members.values():
        if len(idx) < _CORR_GROUP_MIN_COLS:
            continue
        rs = [values[a, b] for k, a in enumerate(idx) for b in idx[k + 1:] if not np.isnan(values[a, b])]
        abs_rs = [abs(r) for r in rs]
        groups.append({
            "columns": [cols[i] for i in idx],
            "min_r": round(float(min(abs_rs)), 4),
            "max_r": round(float(max(abs_rs)), 4),
            "mean_abs_r": round(float(np.mean(abs_rs)), 4),
            "any_negative": any(r < 0 for r in rs),
        })
    groups.sort(key=lambda g: g["mean_abs_r"], reverse=True)
    return groups


def _definitional_pairs(num_df: pd.DataFrame, corr: pd.DataFrame) -> dict[frozenset[str], str]:
    """Column pairs whose correlation is true by construction, mapped to
    the reason: identity-like (|r| >= _IDENTITY_R), or one column being the
    product/sum/ratio of two others (both operands pair with the result)."""
    from itertools import combinations

    cols = [str(c) for c in corr.columns]
    found: dict[frozenset[str], str] = {}
    values_r = corr.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        strong = np.triu(np.abs(values_r) >= _IDENTITY_R, 1)
    for i, j in np.argwhere(strong):
        found[frozenset((cols[i], cols[j]))] = f"|r| >= {_IDENTITY_R}: the same quantity recorded twice"
    if not 3 <= len(cols) <= _MAX_RELATION_COLS:
        return found
    sample = num_df[cols].dropna()
    if len(sample) < _MIN_RELATION_ROWS:
        return found
    if len(sample) > _RELATION_SAMPLE:
        sample = sample.sample(n=_RELATION_SAMPLE, random_state=42)
    values = {c: sample[c].to_numpy(dtype=float) for c in cols}

    for c in cols:
        target = values[c]
        if float(np.std(target)) == 0:
            continue
        tol = 1e-9 + _RELATION_RTOL * np.abs(target)

        def matches(fitted: np.ndarray, target: np.ndarray = target, tol: np.ndarray = tol) -> bool:
            return float(np.mean(np.abs(target - fitted) <= tol)) >= _RELATION_MATCH_SHARE

        for x, y in combinations([o for o in cols if o != c], 2):
            a_, b_ = values[x], values[y]
            # An operand that alone equals the column (qty == 1 almost
            # everywhere) makes any formula "match" — that is identity, above.
            if matches(a_) or matches(b_):
                continue
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                formulas = {
                    f"{c} = {x} * {y}": a_ * b_,
                    f"{c} = {x} + {y}": a_ + b_,
                    f"{c} = {x} / {y}": a_ / b_,
                    f"{c} = {y} / {x}": b_ / a_,
                }
            label = next((lbl for lbl, fitted in formulas.items() if matches(fitted)), None)
            if label:
                found.setdefault(frozenset((c, x)), label)
                found.setdefault(frozenset((c, y)), label)
    return found


# ============================================================
# Tool 4: Correlation Analysis — Stage 3 EDA
# ============================================================

class CorrelationAnalysisTool(BaseTool):
    """
    Compute feature correlation matrix and identify top correlations.

    Supports Pearson, Spearman, Kendall, and "auto" (Pearson, switching to
    Spearman for any pair involving a severely skewed column, where a few
    extreme values would otherwise dominate r). Only measure/ordinal columns
    are used — a correlation with a row ID, a 0/1 flag or an integer-coded
    dimension is not a relationship between quantities. Correlations are pairwise-complete: a
    missing value in one column never drops the row from unrelated pairs.
    Returns both global top pairs (with p-values) and target correlations.
    """

    name = "correlation_analysis"
    description = (
        "Compute correlation matrix for numerical features. "
        "Returns top correlated pairs and, if target_column is given, "
        "correlations of all features with the target. "
        "Methods: 'auto' (default: Pearson, Spearman for skewed columns), "
        "'pearson', 'spearman', 'kendall'."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None:
            return 1.0
        return 1.0 if len(profile.columns_of_kind("numeric")) >= 2 else 0.0

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        method: str = "auto",
        top_n: int = 10,
        target_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        from scipy import stats

        from src.core.profiler import SEVERE_SKEW_THRESHOLD, profile_dataframe

        df = _read_df(file_path)
        num_df = df.select_dtypes(include="number")
        try:
            roles = {c.name: c.semantic_role for c in profile_dataframe(df).columns}
        except Exception:
            roles = {}
        # Only quantities: integer-coded identifiers, flags, years and
        # region codes are numeric in storage but not measures.
        num_df = num_df[[
            c for c in num_df.columns
            if c == target_column or not roles or roles.get(str(c)) in ("measure", "ordinal")
        ]]

        if num_df.shape[1] < 2:
            raise ToolExecutionError("Need at least 2 numerical columns for correlation analysis.")

        # The matrix is exploratory, so above the size cap it is computed on a
        # seeded sample; the p-values below are tests and use every row.
        test_df = num_df
        sampled: dict[str, Any] = {}
        if len(num_df) > analysis_sample_rows():
            num_df = num_df.sample(n=analysis_sample_rows(), random_state=0)
            sampled = sample_note(len(test_df), len(num_df))

        # DataFrame.corr is pairwise-complete, so no global dropna.
        if method == "auto":
            skewed = {
                c for c in num_df.columns
                if abs(float(num_df[c].skew())) >= SEVERE_SKEW_THRESHOLD
            }
            corr = pearson_matrix(num_df)
            if skewed:
                spearman = num_df.corr(method="spearman")
                use_rank = np.array([[a in skewed or b in skewed for b in corr.columns] for a in corr.columns])
                corr = corr.where(~use_rank, spearman)
        else:
            skewed = set()
            corr = pearson_matrix(num_df) if method == "pearson" else num_df.corr(method=method)

        def _pair_method(a: str, b: str) -> str:
            if method != "auto":
                return method
            return "spearman" if a in skewed or b in skewed else "pearson"

        # Top correlated pairs (exclude self-correlations)
        cols = corr.columns.tolist()
        ii, jj = np.triu_indices(len(cols), 1)
        rr = corr.to_numpy(dtype=float)[ii, jj]
        keep = ~np.isnan(rr)
        pairs: list[dict[str, Any]] = [
            {"col_a": cols[i], "col_b": cols[j], "correlation": round(float(v), 4)}
            for i, j, v in zip(ii[keep], jj[keep], rr[keep], strict=True)
        ]
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        # Correlations true by construction (total = qty * price) are
        # reported, tagged, and never promoted to findings.
        definitional = _definitional_pairs(num_df, corr)
        for pair in pairs:
            reason = definitional.get(frozenset((str(pair["col_a"]), str(pair["col_b"]))))
            if reason:
                pair["definitional"] = True
                pair["definitional_reason"] = reason
        # Columns that all move together are one phenomenon, not many pairs.
        corr_groups = _correlation_groups(corr, set(definitional))
        group_of = {c: g["columns"] for g in corr_groups for c in g["columns"]}
        for pair in pairs:
            cols_a = group_of.get(str(pair["col_a"]))
            if cols_a is not None and str(pair["col_b"]) in cols_a and not pair.get("definitional"):
                pair["in_group"] = True
        top_pairs = pairs[:top_n]
        tests = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}
        for pair in top_pairs:
            both = test_df[[pair["col_a"], pair["col_b"]]].dropna()
            pair_method = _pair_method(pair["col_a"], pair["col_b"])
            pair["method"] = pair_method
            pair["n"] = len(both)
            try:
                p_val = float(tests[pair_method](both.iloc[:, 0], both.iloc[:, 1])[1])
                pair["p_value"] = None if np.isnan(p_val) else round(p_val, 6)
            except (ValueError, KeyError):
                pair["p_value"] = None

        # Target correlations
        target_corrs: dict[str, float] = {}
        target_encoded = False
        target_corr_method = method
        positive_class: Any = None
        target_values: pd.Series | None = None
        if target_column and target_column in corr.columns:
            target_values = test_df[target_column]
        if target_column and target_column in corr.columns:
            target_corrs = {
                c: round(float(corr.loc[c, target_column]), 4)
                for c in corr.columns
                if c != target_column and not np.isnan(corr.loc[c, target_column])
            }
            target_corrs = dict(
                sorted(target_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
            )
        elif (
            target_column
            and target_column in df.columns
            and df[target_column].nunique(dropna=True) == 2
        ):
            # Non-numeric binary target (e.g. churn yes/no): encode to 0/1 so
            # feature↔target correlation still works (point-biserial). The
            # positive class is the larger label in sorted order (not first
            # appearance), so the sign is stable and can be named.
            raw_target = df[target_column]
            positive_class = sorted(raw_target.dropna().unique(), key=str)[-1]
            encoded = (raw_target == positive_class).astype("float64").where(raw_target.notna())
            aligned = encoded.loc[test_df.index]
            target_values = aligned
            target_encoded = True
            target_corr_method = "point-biserial"
            for c in num_df.columns:
                val = float(num_df[c].corr(aligned))
                if not np.isnan(val):
                    target_corrs[c] = round(val, 4)
            target_corrs = dict(
                sorted(target_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
            )
        elif (
            target_column
            and target_column in df.columns
            and df[target_column].nunique(dropna=True) > 2
        ):
            # Non-numeric target with 3+ classes: Pearson r is undefined, so
            # this used to leave target_correlations silently empty with no
            # warning (IMPROVEMENTS.md #4). Eta-squared — the ANOVA analogue
            # of R² — measures how much of each numeric feature's variance is
            # explained by target-class membership: bounded [0, 1], same
            # dict shape as a correlation magnitude, comparable across
            # features. Unlike Pearson r it has no sign (there's no single
            # "direction" across 3+ unordered classes).
            for c in num_df.columns:
                feature = num_df[c].dropna()
                classes = df[target_column].loc[feature.index]
                groups = [
                    feature[classes == cls].to_numpy()
                    for cls in classes.dropna().unique()
                ]
                groups = [g for g in groups if len(g) >= 2]
                if len(groups) < 2:
                    continue
                overall_mean = feature.mean()
                ss_total = float(((feature - overall_mean) ** 2).sum())
                if ss_total <= 0:
                    continue
                ss_between = sum(len(g) * (g.mean() - overall_mean) ** 2 for g in groups)
                target_corrs[c] = round(float(ss_between / ss_total), 4)
            target_corrs = dict(
                sorted(target_corrs.items(), key=lambda x: x[1], reverse=True)
            )
            target_encoded = True
            target_corr_method = "eta-squared"

        # n and p-value for the strongest feature<->target correlations
        # (eta-squared has no single-coefficient test here).
        target_stats: dict[str, dict[str, Any]] = {}
        if target_values is not None and target_column:
            for c in list(target_corrs)[:_CORR_TOP_N]:
                both = pd.concat([test_df[c], target_values], axis=1).dropna()
                t_method = "pearson" if target_encoded else _pair_method(c, target_column)
                try:
                    p_val = float(tests[t_method](both.iloc[:, 0], both.iloc[:, 1])[1])
                except (ValueError, KeyError):
                    p_val = float("nan")
                target_stats[c] = {
                    "n": len(both),
                    "p_value": None if np.isnan(p_val) else round(p_val, 6),
                    "method": t_method,
                }

        lead = next((p for p in top_pairs if not p.get("definitional")), None)
        top_summary = (
            f"Top pair: {lead['col_a']} ↔ {lead['col_b']} "
            f"(r={lead['correlation']}, {lead['method']}, p={lead['p_value']})"
            if lead
            else "No non-definitional pairs"
        )
        if definitional:
            top_summary += f". {len(definitional)} definitional pair(s) excluded from findings"

        return {
            **sampled,
            "summary": (
                f"Correlation ({method}) on {len(cols)} features. {top_summary}."
                + (f" Correlation matrix {sampled['sample_caveat']}; p-values use all rows." if sampled else "")
            ),
            "method": method,
            "spearman_columns": sorted(str(c) for c in skewed),
            "top_correlations": top_pairs,
            "correlation_groups": corr_groups,
            "target_correlations": target_corrs,
            "target_correlation_method": target_corr_method,
            "target_encoded_binary": target_encoded,
            "target_positive_class": None if positive_class is None else str(positive_class),
            "target_column": target_column,
            "target_correlation_stats": target_stats,
            "definitional_pairs": [
                {"col_a": a, "col_b": b, "reason": reason}
                for (a, b), reason in ((sorted(k), v) for k, v in definitional.items())
            ],
            "features_analyzed": cols,
            "n_features": len(cols),
            "n_samples": int(test_df.dropna(how="all").shape[0]),
        }

    def findings(
        self,
        output: dict[str, Any],
        profile: DatasetProfile | None,
        metadata: DatasetMetadata | None,
    ) -> list[Finding]:
        """
        Turn the strongest pairwise correlations into Findings.

        Only pairs at or above `_CORR_FINDING_THRESHOLD` are considered —
        below that, r is noise-level on most real datasets and not worth
        promoting to the finding bus (T5 triviality suppression would also
        catch it via TRIVIAL_EFFECT_FLOOR, but that floor is much lower than
        what's actually interesting here). `confidence` scales with sample
        size when it's available: a correlation computed on a few dozen rows
        deserves less trust than the same r on thousands.

        Definitional pairs (a formula or the same quantity twice) are never
        findings. The strongest feature<->target correlations are promoted
        too, since the target is what the analysis is about.
        """
        top_pairs = [p for p in output.get("top_correlations") or [] if not p.get("definitional")]

        n_samples = output.get("n_samples")
        if isinstance(n_samples, int) and n_samples > 0:
            confidence = 0.9 if n_samples >= 500 else 0.75 if n_samples >= 100 else 0.5
        else:
            confidence = 0.6

        measure_names = {c.name for c in profile.measures()} if profile is not None else set()
        method = output.get("method", "pearson")

        results: list[Finding] = []
        covered: set[frozenset[str]] = set()
        # One finding per group of columns that move together, ahead of any
        # pair; the pairs inside a group drop to the appendix.
        for group in (output.get("correlation_groups") or [])[:_CORR_TOP_N]:
            cols = [str(c) for c in group["columns"]]
            shown = cols if len(cols) <= _CORR_GROUP_NAMES else [*cols[:_CORR_GROUP_NAMES - 1], f"{len(cols) - _CORR_GROUP_NAMES + 1} more"]
            names = ", ".join(shown[:-1]) + f" and {shown[-1]}"
            lo, hi = group["min_r"], group["max_r"]
            label = "|r|" if group.get("any_negative") else "r"
            results.append(Finding(
                finding_id=f"{self.name}_group_{'_'.join(cols)}",
                kind="correlation",
                headline=f"{names} move together ({label} between {lo:.2f} and {hi:.2f})",
                detail=(
                    f"{len(cols)} columns are strongly correlated with each other "
                    f"(|r| >= {_CORR_GROUP_R}): {', '.join(cols)}. Mean |r| across their "
                    f"pairs is {group['mean_abs_r']:.2f}."
                ),
                evidence={
                    "columns": cols, "min_r": lo, "max_r": hi,
                    "mean_abs_r": group["mean_abs_r"], "n": n_samples,
                },
                source_tool=self.name,
                measure=cols[0],
                effect=group["mean_abs_r"],
                effect_kind="r",
                confidence=confidence,
                caveats=[
                    "Columns in a group usually reflect one shared driver (a common "
                    "cycle or the same source) rather than independent evidence."
                ],
                layer="analyst",
            ))
        loose = [(i, p) for i, p in enumerate(top_pairs) if not p.get("in_group")]
        grouped = [(i, p) for i, p in enumerate(top_pairs) if p.get("in_group")]
        for i, pair in loose[:_CORR_TOP_N] + grouped[:_CORR_TOP_N]:
            r = pair.get("correlation")
            col_a, col_b = pair.get("col_a"), pair.get("col_b")
            if r is None or col_a is None or col_b is None or abs(r) < _CORR_FINDING_THRESHOLD:
                continue
            both_measures = col_a in measure_names and col_b in measure_names
            pair_method = pair.get("method", method)
            p_value = pair.get("p_value")
            pair_n = pair.get("n", n_samples)
            covered.add(frozenset((str(col_a), str(col_b))))
            results.append(Finding(
                finding_id=f"{self.name}_{i}_{col_a}_{col_b}",
                kind="correlation",
                headline=f"{col_a} and {col_b} move together (r={r:.2f})",
                detail=(
                    f"{str(pair_method).capitalize()} correlation between '{col_a}' and "
                    f"'{col_b}' is r={r:.2f} (n={pair_n if pair_n else 'unknown'}"
                    + (f", p={p_value:.3g}" if p_value is not None else "")
                    + ")."
                ),
                evidence={
                    "col_a": col_a, "col_b": col_b, "correlation": r,
                    "method": pair_method, "p_value": p_value, "n": pair_n,
                },
                source_tool=self.name,
                measure=col_a,
                dimension=None if both_measures else col_b,
                effect=r,
                effect_kind="r",
                p_value=p_value,
                confidence=confidence,
                layer="appendix" if pair.get("in_group") else "analyst",
            ))
        results.extend(self._target_findings(output, covered, confidence))
        return results

    def _target_findings(
        self, output: dict[str, Any], covered: set[frozenset[str]], confidence: float
    ) -> list[Finding]:
        """Top feature<->target correlations (|r| >= 0.2, or eta-squared >=
        0.04 for a multi-class target), skipping definitional pairs and pairs
        already reported above."""
        target = output.get("target_column")
        if not target:
            return []
        is_eta = output.get("target_correlation_method") == "eta-squared"
        floor = _TARGET_CORR_FINDING_THRESHOLD ** 2 if is_eta else _TARGET_CORR_FINDING_THRESHOLD
        definitional = {
            frozenset((str(d["col_a"]), str(d["col_b"]))) for d in output.get("definitional_pairs") or []
        }
        target_stats = output.get("target_correlation_stats") or {}
        positive = output.get("target_positive_class")
        target_label = f"{target}={positive}" if positive is not None else str(target)
        results: list[Finding] = []
        for feature, value in (output.get("target_correlations") or {}).items():
            if len(results) >= _CORR_TOP_N:
                break
            key = frozenset((str(feature), str(target)))
            if value is None or abs(value) < floor or key in definitional or key in covered:
                continue
            fstats = target_stats.get(feature) or {}
            label = "eta²" if is_eta else "r"
            p_value = fstats.get("p_value")
            results.append(Finding(
                finding_id=f"{self.name}_target_{feature}_{target}",
                kind="correlation",
                headline=(
                    f"{feature} explains {value:.0%} of the variance in {target} (eta²={value:.2f})"
                    if is_eta
                    else f"{feature} {'rises' if value > 0 else 'falls'} with {target_label} (r={value:.2f})"
                ),
                detail=(
                    f"{output.get('target_correlation_method')} association between '{feature}' "
                    f"and target '{target}'"
                    + (f" (positive class: {positive})" if positive is not None else "")
                    + f": {label}={value:.2f}"
                    + (f" (n={fstats['n']}" + (f", p={p_value:.3g}" if p_value is not None else "") + ")"
                       if fstats else "")
                    + "."
                ),
                evidence={
                    "feature": feature, "target": target, "correlation": value,
                    "method": output.get("target_correlation_method"),
                    "positive_class": positive,
                    "p_value": p_value, "n": fstats.get("n"),
                },
                source_tool=self.name,
                measure=str(feature),
                dimension=str(target),
                effect=value,
                effect_kind="eta_sq" if is_eta else "r",
                p_value=p_value,
                confidence=confidence,
                layer="analyst",
            ))
        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to dataset.", "required": True},
            "method": {
                "type": "string",
                "description": (
                    "auto | pearson | spearman | kendall. Default: auto (Pearson, "
                    "Spearman for pairs involving a severely skewed column)."
                ),
                "required": False,
            },
            "top_n": {
                "type": "int",
                "description": "Number of top correlation pairs to return. Default: 10.",
                "required": False,
            },
            "target_column": {
                "type": "string",
                "description": "If given, also returns feature↔target correlations.",
                "required": False,
            },
        }
