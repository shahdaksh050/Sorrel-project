"""
`dsa` — the toolkit pre-bound into the sandbox namespace (see
_sandbox_worker._build_restricted_globals).

Trusted code: it runs inside the worker subprocess with real builtins, so it
can write scratch files and import the built-in analysis tools on the LLM's
behalf. User code reaches it only through the `dsa` global; it is not
importable from sandboxed code (not in ALLOWED_MODULES).

Tool classes are imported lazily, one per `dsa.run` call — the worker is a
cold subprocess per execution, and importing every tool's sklearn/statsmodels
stack up front would tax every sandboxed call, most of which never use `dsa`.
"""
from __future__ import annotations

import datetime
import difflib
import importlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from src.core.chart_spec import MAX_CHART_ROWS, validate_chart_spec
from src.core.profiler import profile_dataframe
from src.core.security import pii_redaction_enabled

#: Analysis tools `dsa.run` may call — pure analyses only (no ingest/clean,
#: model training, reporting, plotting, or the code-execution tools).
_TOOL_CLASSES: dict[str, tuple[str, str]] = {
    "select_statistical_test": ("src.tools.statistical_analysis", "SelectStatisticalTestTool"),
    "segment_comparison": ("src.tools.segment_comparison", "SegmentComparisonTool"),
    "concentration_analysis": ("src.tools.concentration_analysis", "ConcentrationAnalysisTool"),
    "change_analysis": ("src.tools.change_analysis", "ChangeAnalysisTool"),
    "time_series_analysis": ("src.tools.time_series", "TimeSeriesAnalysisTool"),
    "cohort_analysis": ("src.tools.cohort_analysis", "CohortAnalysisTool"),
    "correlation_analysis": ("src.tools.data_processing", "CorrelationAnalysisTool"),
    "detect_outliers": ("src.tools.data_processing", "DetectOutliersTool"),
    "cluster_data": ("src.tools.clustering", "ClusterDataTool"),
    "dimensionality_analysis": ("src.tools.dimensionality", "DimensionalityAnalysisTool"),
    "text_analysis": ("src.tools.text_analysis", "TextAnalysisTool"),
    "geospatial_analysis": ("src.tools.geospatial", "GeospatialAnalysisTool"),
    "financial_analysis": ("src.tools.financial_analysis", "FinancialAnalysisTool"),
    "workforce_analysis": ("src.tools.workforce_analysis", "WorkforceAnalysisTool"),
}

#: compare_groups tests at most this many levels (the most frequent ones).
_MAX_GROUP_LEVELS = 30


def frame_for_disk(frame: pd.DataFrame) -> pd.DataFrame:
    """Parquet needs string column names, and a groupby result keeps its keys
    in the index — surface them as columns so nothing is silently dropped."""
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    if not all(isinstance(c, str) for c in frame.columns):
        frame = frame.set_axis([str(c) for c in frame.columns], axis=1)
    return frame


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, pd.DataFrame):
        return json.loads(value.head(1000).to_json(orient="records", date_format="iso"))
    if isinstance(value, pd.Series):
        return json.loads(value.head(1000).to_json(date_format="iso"))
    return str(value)


def _json_safe(value: Any) -> Any:
    # skipkeys: default= never sees dict keys, and a Timestamp-keyed pivot
    # should lose that key, not fail the whole dsa.run call.
    return json.loads(json.dumps(value, default=_json_default, skipkeys=True))


def _load_tool(tool_name: str) -> Any:
    if tool_name not in _TOOL_CLASSES:
        raise ValueError(
            f"Unknown tool '{tool_name}'. dsa.run accepts: {', '.join(sorted(_TOOL_CLASSES))}."
        )
    module_name, class_name = _TOOL_CLASSES[tool_name]
    try:
        return getattr(importlib.import_module(module_name), class_name)()
    except ImportError as exc:
        raise ValueError(f"Tool '{tool_name}' is unavailable in this sandbox: {exc}") from None


def _numeric(values: Any) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if series.dtype == bool:
        series = series.astype(float)
    return pd.to_numeric(series, errors="coerce").dropna()


def _finite(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _require_columns(frame: pd.DataFrame, *columns: str) -> None:
    names = [str(c) for c in frame.columns]
    for col in columns:
        if col not in frame.columns:
            close = difflib.get_close_matches(str(col), names, n=5, cutoff=0.5)
            suggestion = f" Did you mean: {close}?" if close else f" Columns: {names[:30]}."
            raise ValueError(f"Column '{col}' not found.{suggestion}")


def _cohens_d(a: pd.Series, b: pd.Series) -> float | None:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    pooled = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0 or not math.isfinite(pooled):
        return None
    return (float(a.mean()) - float(b.mean())) / pooled


# ---- chart builders (dsa.chart.*) ------------------------------------------

def _records(data: Any, x: str) -> list[Any]:
    if isinstance(data, pd.Series):
        data = data.reset_index()
    if isinstance(data, pd.DataFrame):
        if x not in data.columns:
            data = data.reset_index()
        # One row past the cap so validate_chart_spec still flags truncation.
        records: list[Any] = json.loads(
            data.head(MAX_CHART_ROWS + 1).to_json(orient="records", date_format="iso")
        )
        return records
    if isinstance(data, list):
        return data
    raise ValueError("Chart data must be a DataFrame, a Series, or a list of record dicts.")


def _chart(chart_type: str, data: Any, x: str, y: str | None, opts: dict[str, Any]) -> dict[str, Any]:
    spec = {**opts, "type": chart_type, "data": _records(data, x), "x": x}
    if y is not None:
        spec["y"] = y
    clean, error = validate_chart_spec(spec)
    if clean is None:
        raise ValueError(error)
    return clean


def bar(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Bar chart spec. opts: color, title, x_title, y_title, y_format, sort, log_y, caption."""
    return _chart("bar", data, x, y, opts)


def line(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Line chart spec; `color` names the series field."""
    return _chart("line", data, x, y, opts)


def area(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Area chart spec; `color` names the series field."""
    return _chart("area", data, x, y, opts)


def scatter(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Scatter chart spec; `color` names the group field."""
    return _chart("scatter", data, x, y, opts)


def histogram(data: Any, x: str, **opts: Any) -> dict[str, Any]:
    """Histogram of the numeric field `x` (no y)."""
    return _chart("histogram", data, x, None, opts)


def heatmap(data: Any, x: str, y: str, color: str, **opts: Any) -> dict[str, Any]:
    """Heatmap spec: `x` and `y` are the two categorical axes and `color` is
    the QUANTITATIVE value field that fills each cell (chart_spec has no
    separate value key — for heatmaps `color` carries the value)."""
    return _chart("heatmap", data, x, y, {**opts, "color": color})


class Toolkit:
    """The object bound to `dsa` inside the sandbox."""

    chart = SimpleNamespace(
        bar=bar, line=line, area=area, scatter=scatter, histogram=histogram, heatmap=heatmap
    )

    def __init__(self, df: pd.DataFrame, scratch_dir: str | Path) -> None:
        self._df = df
        self._scratch = Path(scratch_dir)
        self._calls = 0
        # Finding dicts from every dsa.run call — returned by the worker as
        # payload["tool_findings"]. Private (the sandbox blocks `_` attributes)
        # so sandboxed code can't append findings that never came from a tool.
        self._findings: list[dict[str, Any]] = []

    def run(self, tool_name: str, df: pd.DataFrame | None = None, **params: Any) -> dict[str, Any]:
        """Run a built-in analysis tool on `df` (default: the loaded dataset).
        Returns {"output": <tool output>, "findings": [finding dicts]}."""
        tool = _load_tool(tool_name)
        frame = self._df if df is None else df
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"dsa.run df must be a pandas DataFrame, got {type(frame).__name__}.")
        frame = frame_for_disk(frame)

        # A distinct file per call: read_any caches on (path, mtime, size),
        # so reusing one filename could serve a stale frame.
        self._calls += 1
        path = self._scratch / f"_dsa_frame_{self._calls}.parquet"
        try:
            frame.to_parquet(path, index=False)
        except Exception:
            path = path.with_suffix(".csv")
            frame.to_csv(path, index=False)

        profile = profile_dataframe(frame)
        # The toolkit, not the sandboxed caller, decides where tools read and
        # write: file paths are pinned to the scratch dir, and `_`-prefixed
        # params are internal injection points, never caller input.
        blocked = sorted(k for k in params if k in ("file_path", "output_dir") or k.startswith("_"))
        if blocked:
            raise ValueError(f"dsa.run does not accept {blocked}; the toolkit sets file locations itself.")
        params["output_dir"] = str(self._scratch / "dsa_out")
        # detect_outliers normally gets these from BaseTool.prepare_params.
        if "_semantic_roles" in inspect.signature(tool.execute).parameters:
            params.setdefault("_semantic_roles", {c.name: c.semantic_role for c in profile.columns})
            params.setdefault("_skewed_cols", {c.name for c in profile.columns if "severe_skew" in c.flags})

        from src.tools.base import ToolExecutionError

        try:
            output = tool.execute(file_path=str(path), **params)
        except ToolExecutionError as exc:
            raise ValueError(f"{tool_name}: {exc}") from None

        try:
            found = [f.to_dict() for f in tool.findings(output, profile, None)]
        except Exception:
            found = []  # the output is the deliverable; findings are a bonus
        found = _json_safe(found)
        self._findings.extend(found)
        return {"output": _json_safe(output), "findings": found}

    def collected_findings(self) -> list[dict[str, Any]]:
        """Copy of the findings every dsa.run call produced (worker-side use)."""
        return [dict(f) for f in self._findings]

    def tools(self) -> dict[str, str]:
        """{tool_name: one-line description + required params} for dsa.run."""
        listing: dict[str, str] = {}
        for name in _TOOL_CLASSES:
            try:
                tool = _load_tool(name)
            except ValueError:
                continue
            required = [
                p for p, s in tool.get_schema().items()
                if s.get("required") and p not in ("file_path", "output_dir")
            ]
            first = tool.description.split(". ")[0].strip().rstrip(".")
            listing[name] = f"{first}. Required params: {', '.join(required) or 'none'}."
        return listing

    def profile(self, df: pd.DataFrame | None = None) -> dict[str, dict[str, Any]]:
        """Compact per-column profile: kind, semantic_role, unit_hint, nunique,
        missing_pct, and key stats (numeric) or top values (categorical)."""
        frame = self._df if df is None else df
        out: dict[str, dict[str, Any]] = {}
        for col in profile_dataframe(frame).columns:
            entry: dict[str, Any] = {
                "kind": col.kind,
                "semantic_role": col.semantic_role,
                "unit_hint": col.unit_hint,
                "nunique": col.nunique,
                "missing_pct": col.missing_pct,
            }
            stats = {k: col.stats[k] for k in ("mean", "std", "min", "median", "max", "skew") if k in col.stats}
            if stats:
                entry["stats"] = stats
            # A personal-data column's values would ride RESULT into the next
            # LLM prompt; say what kind of column it is, not what's in it.
            pii = getattr(col, "pii", None)
            if pii and pii_redaction_enabled():
                entry["pii"] = pii
                entry.pop("stats", None)
            elif col.top_values:
                entry["top_values"] = dict(list(col.top_values.items())[:5])
            out[col.name] = entry
        safe: dict[str, dict[str, Any]] = _json_safe(out)
        return safe

    @staticmethod
    def summarize(series: Any) -> dict[str, Any]:
        """n, missing, mean, median, std, min, max, q1, q3, iqr, skew, zero_share."""
        raw = series if isinstance(series, pd.Series) else pd.Series(series)
        clean = _numeric(raw)
        n = len(clean)
        summary: dict[str, Any] = {"n": n, "missing": int(len(raw) - n)}
        if n == 0:
            return summary
        q1, q3 = (float(v) for v in clean.quantile([0.25, 0.75]))
        summary.update({
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "std": float(clean.std(ddof=1)) if n > 1 else 0.0,
            "min": float(clean.min()),
            "max": float(clean.max()),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "skew": _finite(clean.skew()) if n > 2 else None,
            "zero_share": float((clean == 0).mean()),
        })
        return summary

    @staticmethod
    def effect_size(a: Any, b: Any) -> float:
        """Cohen's d of a vs b (pooled SD): ~0.2 small, 0.5 medium, 0.8 large."""
        d = _cohens_d(_numeric(a), _numeric(b))
        if d is None:
            raise ValueError(
                "effect_size needs >= 2 numeric values per group and non-zero variance."
            )
        return d

    @staticmethod
    def compare_groups(df: pd.DataFrame, measure: str, by: str, agg: str = "mean") -> pd.DataFrame:
        """Each level of `by` vs the rest of the data on `measure`.

        Returns a DataFrame, one row per level (the most frequent 30), with:
        `by` (the level), n, n_rest, mean, median, value / rest_value (per
        `agg`), diff_vs_rest, effect + effect_kind (rate_diff for a 0/1
        measure, else cohens_d), p_value (two-proportion z for 0/1, Welch t
        for agg="mean", Mann-Whitney for agg="median") and p_adjusted
        (Benjamini-Hochberg across all levels)."""
        from scipy import stats

        from src.core.multiple_testing import apply_benjamini_hochberg

        if agg not in ("mean", "median"):
            raise ValueError("agg must be 'mean' or 'median'.")
        _require_columns(df, measure, by)
        work = pd.DataFrame({by: df[by], measure: _numeric(df[measure])}).dropna()
        if work.empty:
            raise ValueError(f"'{measure}' has no numeric values to compare.")
        is_rate = set(work[measure].unique()) <= {0.0, 1.0}

        rows: list[dict[str, Any]] = []
        for level, n in work[by].value_counts().head(_MAX_GROUP_LEVELS).items():
            mask = work[by] == level
            a, b = work.loc[mask, measure], work.loc[~mask, measure]
            if b.empty:
                continue
            value, rest_value = float(a.agg(agg)), float(b.agg(agg))
            effect: float | None = None
            p_value: float | None = None
            if is_rate:
                effect = float(a.mean() - b.mean())
                pooled = (a.sum() + b.sum()) / (len(a) + len(b))
                se = math.sqrt(pooled * (1 - pooled) * (1 / len(a) + 1 / len(b)))
                if se > 0:
                    p_value = float(2 * stats.norm.sf(abs(effect / se)))
            elif len(a) >= 2 and len(b) >= 2:
                effect = _cohens_d(a, b)
                test = (
                    stats.ttest_ind(a, b, equal_var=False) if agg == "mean"
                    else stats.mannwhitneyu(a, b, alternative="two-sided")
                )
                p_value = _finite(test.pvalue)
            rows.append({
                by: level,
                "n": int(n),
                "n_rest": len(b),
                "mean": float(a.mean()),
                "median": float(a.median()),
                "value": value,
                "rest_value": rest_value,
                "diff_vs_rest": value - rest_value,
                "effect": effect,
                "effect_kind": "rate_diff" if is_rate else "cohens_d",
                "p_value": p_value,
            })

        tested = [i for i, r in enumerate(rows) if r["p_value"] is not None]
        corrected = apply_benjamini_hochberg([rows[i] for i in tested])
        for i, row in zip(tested, corrected, strict=True):
            rows[i] = row
        for row in rows:
            row.setdefault("p_adjusted", None)
            row.pop("significant_after_correction", None)
        return pd.DataFrame(rows)
