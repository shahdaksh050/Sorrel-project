"""
Data Processing Tools — Execution Layer.

Stage 1: Dataset Ingestion  (IngestDatasetTool)
Stage 3: Data Cleaning      (CleanDataTool)
Stage 3: Outlier Detection  (DetectOutliersTool)
Stage 3: Correlation EDA    (CorrelationAnalysisTool)

All tools are deterministic and return structured dicts.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.coercion import coerce_types
from src.core.findings import Finding
from src.core.io import DatasetReadError, invalidate_read_cache, read_any
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
    df = _read_raw_df(file_path)
    repaired, _coercions = coerce_types(df)
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
    Handle missing values and basic data cleaning.

    Writes a cleaned copy to disk and returns the new file path —
    downstream tools should use `cleaned_file_path` as their input.
    """

    name = "clean_data"
    description = (
        "Clean a dataset by handling missing values using a specified strategy. "
        "Strategies: 'mean', 'median', 'mode', 'drop_rows', 'forward_fill'. "
        "Returns cleaned_file_path for use by subsequent tools."
    )
    uses_cleaned_file = False  # this IS the tool that produces cleaned_file_path
    output_subdir = "data"

    STRATEGIES = frozenset({"mean", "median", "mode", "drop_rows", "forward_fill"})

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        strategy: str = "median",
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
        original_shape = df.shape
        missing_before = int(df.isnull().sum().sum())

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
                    "missing value). Re-run clean_data with strategy 'median' "
                    "or 'mode' instead."
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
                f"Cleaned {missing_before - missing_after} missing values "
                f"using '{strategy}'. Shape: {original_shape} → {df.shape}. "
                f"Saved to '{out_path}'."
            ),
            "cleaned_file_path": str(out_path),
            "rows_before": original_shape[0],
            "rows_after": df.shape[0],
            "missing_before": missing_before,
            "missing_after": missing_after,
            "strategy_used": strategy,
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to the dataset file.", "required": True},
            "strategy": {
                "type": "string",
                "description": "Imputation strategy: mean | median | mode | drop_rows | forward_fill.",
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
            preds = model.fit_predict(clean)
            mask = pd.Series(False, index=df.index)
            mask.loc[clean.index[preds == -1]] = True
            total_iso = int((preds == -1).sum())
            pct_iso = round(total_iso / max(row_count, 1) * 100, 2)
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
        report["outlier_percentage"] = round(total / max(row_count, 1) * 100, 2)
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
        )

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


# ============================================================
# Tool 4: Correlation Analysis — Stage 3 EDA
# ============================================================

class CorrelationAnalysisTool(BaseTool):
    """
    Compute feature correlation matrix and identify top correlations.

    Supports Pearson, Spearman, and Kendall methods.
    Returns both global top pairs and target-specific correlations.
    """

    name = "correlation_analysis"
    description = (
        "Compute correlation matrix for numerical features. "
        "Returns top correlated pairs and, if target_column is given, "
        "correlations of all features with the target. "
        "Methods: 'pearson' (default), 'spearman', 'kendall'."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None:
            return 1.0
        return 1.0 if len(profile.columns_of_kind("numeric")) >= 2 else 0.0

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        method: str = "pearson",
        top_n: int = 10,
        target_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        num_df = df.select_dtypes(include="number").dropna()

        if num_df.shape[1] < 2:
            raise ToolExecutionError("Need at least 2 numerical columns for correlation analysis.")

        corr = num_df.corr(method=method)

        # Top correlated pairs (exclude self-correlations)
        pairs: list[dict[str, Any]] = []
        cols = corr.columns.tolist()
        for i, ca in enumerate(cols):
            for cb in cols[i + 1 :]:
                val = float(corr.loc[ca, cb])
                if not np.isnan(val):
                    pairs.append({"col_a": ca, "col_b": cb, "correlation": round(val, 4)})
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        top_pairs = pairs[:top_n]

        # Target correlations
        target_corrs: dict[str, float] = {}
        target_encoded = False
        target_corr_method = method
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
            # feature↔target correlation still works (point-biserial).
            raw_target = df[target_column]
            encoded = pd.Series(
                pd.factorize(raw_target)[0], index=df.index, dtype="float64"
            ).where(raw_target.notna())
            aligned = encoded.loc[num_df.index]
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
            classes = df[target_column].loc[num_df.index]
            for c in num_df.columns:
                feature = num_df[c]
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

        top_summary = (
            f"Top pair: {top_pairs[0]['col_a']} ↔ {top_pairs[0]['col_b']} "
            f"(r={top_pairs[0]['correlation']})"
            if top_pairs
            else "No pairs"
        )

        return {
            "summary": (
                f"Correlation ({method}) on {len(cols)} features. {top_summary}."
            ),
            "method": method,
            "top_correlations": top_pairs,
            "target_correlations": target_corrs,
            "target_correlation_method": target_corr_method,
            "target_encoded_binary": target_encoded,
            "features_analyzed": cols,
            "n_features": len(cols),
            "n_samples": int(num_df.shape[0]),
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
        """
        top_pairs = output.get("top_correlations") or []
        if not top_pairs:
            return []

        n_samples = output.get("n_samples")
        if isinstance(n_samples, int) and n_samples > 0:
            confidence = 0.9 if n_samples >= 500 else 0.75 if n_samples >= 100 else 0.5
        else:
            confidence = 0.6

        measure_names = {c.name for c in profile.measures()} if profile is not None else set()
        method = output.get("method", "pearson")

        results: list[Finding] = []
        for i, pair in enumerate(top_pairs[:_CORR_TOP_N]):
            r = pair.get("correlation")
            col_a, col_b = pair.get("col_a"), pair.get("col_b")
            if r is None or col_a is None or col_b is None or abs(r) < _CORR_FINDING_THRESHOLD:
                continue
            both_measures = col_a in measure_names and col_b in measure_names
            results.append(Finding(
                finding_id=f"{self.name}_{i}_{col_a}_{col_b}",
                kind="correlation",
                headline=f"{col_a} and {col_b} move together (r={r:.2f})",
                detail=(
                    f"{method.capitalize()} correlation between '{col_a}' and "
                    f"'{col_b}' is r={r:.2f} (n={n_samples if n_samples else 'unknown'})."
                ),
                evidence={"col_a": col_a, "col_b": col_b, "correlation": r, "method": method},
                source_tool=self.name,
                measure=col_a,
                dimension=None if both_measures else col_b,
                effect=r,
                effect_kind="r",
                confidence=confidence,
                layer="analyst",
            ))
        return results

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to dataset.", "required": True},
            "method": {
                "type": "string",
                "description": "pearson | spearman | kendall. Default: pearson.",
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
