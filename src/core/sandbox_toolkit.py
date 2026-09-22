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

import ast
import datetime
import difflib
import functools
import importlib
import inspect
import json
import math
import re
from collections.abc import Callable
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
    "regression_analysis": ("src.tools.regression", "RegressionAnalysisTool"),
    "experiment_analysis": ("src.tools.experiment_analysis", "ExperimentAnalysisTool"),
    "anomaly_analysis": ("src.tools.anomaly", "AnomalyAnalysisTool"),
    "survival_analysis": ("src.tools.survival", "SurvivalAnalysisTool"),
    "curve_fit_analysis": ("src.tools.curve_fit", "CurveFitAnalysisTool"),
    "forecast_analysis": ("src.tools.forecast", "ForecastAnalysisTool"),
    "mixed_model_analysis": ("src.tools.mixed_model", "MixedModelAnalysisTool"),
    "basket_analysis": ("src.tools.basket", "BasketAnalysisTool"),
    "price_elasticity_analysis": ("src.tools.elasticity", "PriceElasticityTool"),
    "equity_analysis": ("src.tools.equity", "EquityAnalysisTool"),
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

#: cramers_v refuses column pairs whose contingency table would exceed this
#: many cells (an ID-like column would otherwise build a huge dense table).
_MAX_CONTINGENCY_CELLS = 250_000

#: baseline_accuracy is for classification targets, not continuous columns.
_MAX_CLASSES = 50


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


# ---- dsa.derive: restricted arithmetic over columns --------------------------

_MAX_EXPR_CHARS = 500
#: A larger literal exponent on a Series is a way to hang the worker.
_MAX_EXPONENT = 10.0
_BACKTICKED = re.compile(r"`([^`]+)`")


def _literal_number(node: ast.AST) -> float | None:
    """The value of a numeric literal (optionally signed), else None."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literal_number(node.operand)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub) else inner)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    return None


def _eval_arith(node: ast.AST, env: dict[str, pd.Series]) -> Any:
    """Evaluate a whitelisted arithmetic AST: numeric literals, column names,
    unary +/-, and + - * / ** (literal exponent, |n| <= 10). Nothing else
    parses to a value — no calls, attributes, subscripts or comparisons."""
    if isinstance(node, ast.Expression):
        return _eval_arith(node.body, env)
    literal = _literal_number(node)
    if literal is not None:
        return literal
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval_arith(node.operand, env)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            exponent = _literal_number(node.right)
            if exponent is None or abs(exponent) > _MAX_EXPONENT:
                raise ValueError(f"Exponents must be numeric literals with |n| <= {_MAX_EXPONENT:g}.")
            return _eval_arith(node.left, env) ** exponent
        left, right = _eval_arith(node.left, env), _eval_arith(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(
        "Only column names, numeric literals, + - * / ** and parentheses are allowed in a derive expression."
    )


# ---- chart builders (dsa.chart.*) ------------------------------------------

def _records(data: Any, x: str | None) -> list[Any]:
    if isinstance(data, pd.Series):
        data = data.reset_index()
    if isinstance(data, pd.DataFrame):
        if x is not None and x not in data.columns:
            data = data.reset_index()
        # One row past the cap so validate_chart_spec still flags truncation.
        records: list[Any] = json.loads(
            data.head(MAX_CHART_ROWS + 1).to_json(orient="records", date_format="iso")
        )
        return records
    if isinstance(data, list):
        return data
    raise ValueError("Chart data must be a DataFrame, a Series, or a list of record dicts.")


#: Options the chart validator reads; any other keyword is dropped (and noted).
_CHART_OPTS = frozenset({
    "color", "series", "title", "x_title", "y_title", "y2_title", "y_format", "y2_format", "sort",
    "log_y", "caption", "annotations", "size", "priority", "facet", "y_lower", "y_upper", "scale",
    "order", "note", "target", "y2",
})
#: Helpers where a weak model's `hue=`/`group=`/`by=` safely means `color=` (the series field).
_COLOR_HELPERS = frozenset({"bar", "line", "area", "scatter", "dot_ci"})
_SERIES_ALIASES = ("hue", "group", "by", "series", "color")


def _lenient(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Accept the plotting-library keywords weak models reach for instead of
    raising: hue/group/by -> the helper's series/color field, values -> y,
    xlabel/ylabel/label -> axis/plot titles (when unset), kind/bins dropped.
    Nothing that changes what is plotted in a risky way is mapped (no y2)."""
    names = list(inspect.signature(fn).parameters)
    own = next((p for p in ("series", "group") if p in names), None)

    def given(name: str, args: tuple[Any, ...], kw: dict[str, Any]) -> bool:
        return name in kw or (name in names and names.index(name) < len(args))

    @functools.wraps(fn)
    def wrapper(*args: Any, **kw: Any) -> dict[str, Any]:
        kw.pop("kind", None)
        if fn.__name__ != "histogram":
            kw.pop("bins", None)
        for alias, target in (("label", "title"), ("xlabel", "x_title"), ("ylabel", "y_title")):
            text = kw.pop(alias, None)
            if isinstance(text, str) and target not in kw:
                kw[target] = text
        dest = "y" if "y" in names else "x" if "x" in names else None
        if dest and fn.__name__ != "heatmap" and "values" in kw and not given(dest, args, kw):
            kw[dest] = kw.pop("values")
        goal = own or ("color" if fn.__name__ in _COLOR_HELPERS else None)
        if goal and not given(goal, args, kw):
            found = next((a for a in _SERIES_ALIASES if a != goal and a in kw), None)
            if found:
                kw[goal] = kw.pop(found)
        return fn(*args, **kw)

    return wrapper


def _chart(chart_type: str, data: Any, x: str, y: str | None, opts: dict[str, Any]) -> dict[str, Any]:
    dropped = sorted(k for k in opts if k not in _CHART_OPTS)
    spec = {**{k: v for k, v in opts.items() if k in _CHART_OPTS}, "type": chart_type, "data": _records(data, x), "x": x}
    if y is not None:
        spec["y"] = y
    clean, error = validate_chart_spec(spec)
    if clean is None:
        raise ValueError(error)
    if dropped:
        clean["note"] = " ".join(filter(None, [clean.get("note"), f"ignored: {', '.join(dropped)}"]))[:200]
    return clean


@_lenient
def bar(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Bar chart spec. opts (every helper): color, title, x_title, y_title, y_format, sort,
    log_y, caption, annotations, size, priority, facet."""
    return _chart("bar", data, x, y, opts)


@_lenient
def line(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Line chart spec; `color` names the series field."""
    return _chart("line", data, x, y, opts)


@_lenient
def area(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Area chart spec; `color` names the series field."""
    return _chart("area", data, x, y, opts)


@_lenient
def scatter(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Scatter chart spec; `color` names the group field."""
    return _chart("scatter", data, x, y, opts)


@_lenient
def histogram(data: Any, x: str, **opts: Any) -> dict[str, Any]:
    """Histogram of the numeric field `x` (no y)."""
    return _chart("histogram", data, x, None, opts)


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std()
    if not std or pd.isna(std):
        return values * 0.0
    return (values - values.mean()) / std


def _wide_to_long(data: Any, x: str, columns: list[str], scale: str | None) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        data = data.reset_index()
    if not isinstance(data, pd.DataFrame):
        raise ValueError("A wide heatmap (y=[columns]) needs a DataFrame.")
    if x not in data.columns:
        data = data.reset_index()
    missing = [c for c in [x, *columns] if c not in data.columns]
    if missing:
        raise ValueError(f"heatmap columns not found: {missing}. Available: {list(data.columns)}.")
    if "Measure" in (x, *columns):
        raise ValueError("heatmap cannot melt a column named 'Measure'; rename it first.")
    wide = data[[x, *columns]].copy()
    wide[columns] = wide[columns].apply(pd.to_numeric, errors="coerce")
    if scale == "zscore":
        wide[columns] = wide[columns].apply(_zscore)
    value = "zscore" if scale == "zscore" else "value"
    long = wide.melt(id_vars=x, value_vars=columns, var_name="Measure", value_name=value)
    return long.dropna(subset=[value])


@_lenient
def heatmap(data: Any, x: str, y: str | list[str], color: str | None = None, **opts: Any) -> dict[str, Any]:
    """Heatmap spec. LONG: heatmap(long_df, x, y, color=<value col>) — `x`/`y` are the two
    categorical axes, `color` the QUANTITATIVE value filling each cell. WIDE:
    heatmap(wide_df, x, y=[col, col, ...]) — melts the value columns into rows (y axis =
    "Measure", color = value). scale="zscore" standardises each series independently
    (mixed units become comparable). Time-of-day/month axes: aggregate to one row per x first."""
    scale = opts.pop("scale", None)
    if scale not in (None, "zscore"):
        raise ValueError("heatmap scale must be 'zscore' or omitted.")
    if isinstance(y, (list, tuple)):
        columns = list(y)
        if len(columns) < 2:
            raise ValueError("A wide heatmap needs y=[at least two value columns].")
        long = _wide_to_long(data, x, columns, scale)
        opts.setdefault("y_title", "Measure")
        if scale:
            opts["scale"] = scale
        return _chart("heatmap", long, x, "Measure", {**opts, "color": "zscore" if scale else "value"})
    if color is None:
        raise ValueError("heatmap(data, x, y, color=<value column>) needs `color` (or pass y=[value columns]).")
    if scale:
        if isinstance(data, pd.Series):
            data = data.reset_index()
        if not isinstance(data, pd.DataFrame):
            raise ValueError("heatmap scale='zscore' needs a DataFrame.")
        if x not in data.columns:
            data = data.reset_index()
        data = data.copy()
        data[color] = pd.to_numeric(data[color], errors="coerce")
        data[color] = data.groupby(y, sort=False)[color].transform(_zscore)
        opts["scale"] = scale
    return _chart("heatmap", data, x, y, {**opts, "color": color})


#: corr_heatmap shows at most this many variables (highest mean |r| kept).
_MAX_CORR_COLUMNS = 20


@_lenient
def corr_heatmap(
    data: Any, columns: list[str] | None = None, method: str = "pearson", **opts: Any
) -> dict[str, Any]:
    """Correlation-matrix heatmap of numeric columns (default: all; constant columns dropped,
    capped at 20 by highest mean |r|), ordered by hierarchical clustering so related
    variables sit together. method: pearson | spearman | kendall."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("corr_heatmap needs a DataFrame.")
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError("corr_heatmap method must be pearson, spearman or kendall.")
    frame = data.select_dtypes("number") if columns is None else data[list(columns)]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[:, frame.nunique(dropna=True) > 1]
    if frame.shape[1] < 2:
        raise ValueError("corr_heatmap needs at least two non-constant numeric columns.")
    corr = frame.corr(method=method)
    if corr.shape[0] > _MAX_CORR_COLUMNS:
        strength = corr.abs().fillna(0).where(~np.eye(len(corr), dtype=bool), 0).mean()
        keep = strength.nlargest(_MAX_CORR_COLUMNS).index
        corr = corr.loc[keep, keep]
    dist = (1 - corr.abs().fillna(0)).to_numpy(copy=True)
    dist = np.clip((dist + dist.T) / 2, 0, 1)
    np.fill_diagonal(dist, 0)
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    order = [str(corr.columns[i]) for i in leaves_list(linkage(squareform(dist, checks=False), "average"))]
    long = corr.rename(columns=str, index=str).stack().rename("r").rename_axis(["x", "y"]).reset_index()
    long["r"] = long["r"].round(3)
    opts.setdefault("title", f"Correlation ({method})")
    return _chart("heatmap", long, "x", "y", {**opts, "color": "r", "order": order})


@_lenient
def waterfall(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Waterfall chart spec showing sequential positive and negative contributions."""
    return _chart("waterfall", data, x, y, opts)


@_lenient
def lorenz(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Lorenz curve spec showing cumulative distribution vs equality diagonal."""
    return _chart("lorenz", data, x, y, opts)


@_lenient
def dot_ci(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Dot plot with intervals. opts: y_lower and y_upper (both, or neither), color."""
    return _chart("dot_ci", data, x, y, opts)


@_lenient
def dual_axis(data: Any, x: str, y: str, y2: str | None = None, **opts: Any) -> dict[str, Any]:
    """Dual-axis line spec: `y` on the left axis and `y2` (required) on an
    independent right axis, over a shared `x`. opts also take y2_title, y2_format."""
    if y2 is None:
        raise ValueError("dual_axis needs y2=<the second value column> (plotted on the right axis).")
    return _chart("dual_axis", data, x, y, {**opts, "y2": y2})


@_lenient
def boxplot(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Box plot of the RAW numeric values `y` per group `x` (quartiles are
    computed for you; needs >= 4 rows per group, not pre-aggregated numbers)."""
    if isinstance(data, pd.DataFrame) and len(data) > MAX_CHART_ROWS:
        data = data.sample(MAX_CHART_ROWS, random_state=42)
    return _chart("boxplot", data, x, y, opts)


@_lenient
def stacked_bar(data: Any, x: str, y: str, series: str, **opts: Any) -> dict[str, Any]:
    """Stacked bar spec: segments of `y` per `x`, split by the `series` field."""
    return _chart("stacked_bar", data, x, y, {**opts, "color": series})


@_lenient
def grouped_bar(data: Any, x: str, y: str, series: str, **opts: Any) -> dict[str, Any]:
    """Grouped (side-by-side) bar spec: one bar per `series` value within each `x`."""
    return _chart("grouped_bar", data, x, y, {**opts, "color": series})


@_lenient
def pareto(data: Any, x: str, y: str, **opts: Any) -> dict[str, Any]:
    """Pareto spec: additive `y` per category as sorted bars plus a cumulative-share line."""
    return _chart("pareto", data, x, y, opts)


@_lenient
def slope(data: Any, x: str, y: str, group: str, **opts: Any) -> dict[str, Any]:
    """Slope chart spec: `x` has exactly 2 values (e.g. before/after); one line per `group`."""
    return _chart("slope", data, x, y, {**opts, "color": group})


@_lenient
def bullet(data: Any, x: str, y: str, target: str, **opts: Any) -> dict[str, Any]:
    """Bullet spec: actual `y` (bar) against the `target` column (tick), per category `x`."""
    return _chart("bullet", data, x, y, {**opts, "target": target})


@_lenient
def band(data: Any, x: str, y: str, y_lower: str, y_upper: str, **opts: Any) -> dict[str, Any]:
    """Line with a shaded band between the `y_lower` and `y_upper` columns."""
    return _chart("band", data, x, y, {**opts, "y_lower": y_lower, "y_upper": y_upper})


def vega_lite(spec: dict[str, Any], data: Any = None, **opts: Any) -> dict[str, Any]:
    """Escape hatch: a raw Vega-Lite spec, sanitised (inline rows only, no
    config/params/expressions; every field must exist in the rows). Pass rows
    as `data=<DataFrame|list>` or inline as spec["data"]["values"]. opts: title, caption, size, priority."""
    if data is not None:
        spec = {**spec, "data": {"values": _records(data, None)}}
    clean, error = validate_chart_spec({**opts, "vega_lite": spec})
    if clean is None:
        raise ValueError(error)
    return clean


class Toolkit:
    """The object bound to `dsa` inside the sandbox."""

    chart = SimpleNamespace(
        bar=bar,
        line=line,
        area=area,
        scatter=scatter,
        histogram=histogram,
        heatmap=heatmap,
        corr_heatmap=corr_heatmap,
        waterfall=waterfall,
        lorenz=lorenz,
        dot_ci=dot_ci,
        dual_axis=dual_axis,
        boxplot=boxplot,
        stacked_bar=stacked_bar,
        grouped_bar=grouped_bar,
        pareto=pareto,
        slope=slope,
        bullet=bullet,
        band=band,
        vega_lite=vega_lite,
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

    def relations(self, df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
        """Formula relations between numeric columns (the ones in the prompt's
        "Relations" block), best first: kind (product/ratio/sum/difference/
        part_of_total/cumulative), target, terms, expr, exact, r2, max_rel_err,
        confidence. Empty list when no column is a formula of others."""
        frame = self._df if df is None else df
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"dsa.relations df must be a pandas DataFrame, got {type(frame).__name__}.")
        found: list[dict[str, Any]] = _json_safe(profile_dataframe(frame).relations)
        return found

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

    @staticmethod
    def cramers_v(df: pd.DataFrame, col1: str, col2: str) -> dict[str, Any]:
        """Bias-corrected Cramér's V (0-1) between two categorical columns,
        with the chi-square statistic, p_value, dof and n."""
        from scipy import stats

        _require_columns(df, col1, col2)
        sub = df[[col1, col2]].dropna()
        n = len(sub)
        if n == 0:
            raise ValueError(f"No non-null rows for '{col1}' and '{col2}'.")
        r, k = int(sub[col1].nunique()), int(sub[col2].nunique())
        if r * k > _MAX_CONTINGENCY_CELLS:
            raise ValueError(
                f"'{col1}' x '{col2}' has {r} x {k} levels; Cramér's V needs categorical "
                "columns. Bucket or drop the high-cardinality one first."
            )
        if r < 2 or k < 2:
            return {"cramers_v": 0.0, "chi2": 0.0, "p_value": 1.0, "dof": 0, "n": n}
        chi2, p_value, dof, _ = stats.chi2_contingency(pd.crosstab(sub[col1], sub[col2]), correction=False)
        # Bergsma bias correction: raw V overstates association on small samples.
        phi2 = max(0.0, float(chi2) / n - (r - 1) * (k - 1) / (n - 1)) if n > 1 else 0.0
        r_eff = r - (r - 1) ** 2 / (n - 1) if n > 1 else r
        k_eff = k - (k - 1) ** 2 / (n - 1) if n > 1 else k
        denom = min(r_eff - 1, k_eff - 1)
        v = math.sqrt(phi2 / denom) if phi2 > 0 and denom > 0 else 0.0
        return {
            "cramers_v": round(v, 4),
            "chi2": round(float(chi2), 4),
            "p_value": float(p_value),
            "dof": int(dof),
            "n": n,
        }

    @staticmethod
    def crosstab_shares(
        df: pd.DataFrame, index_col: str, columns_col: str, normalize: str = "index"
    ) -> pd.DataFrame:
        """Cross-tab of two columns as percentages: one row per `index_col`
        level (kept as a column, so RESULT keeps its labels) and one column per
        `columns_col` level. normalize: "index" (row %), "columns" or "all"."""
        _require_columns(df, index_col, columns_col)
        if normalize not in ("index", "columns", "all"):
            raise ValueError("normalize must be 'index', 'columns' or 'all'.")
        sub = df[[index_col, columns_col]].dropna()
        shares = pd.crosstab(sub[index_col], sub[columns_col], normalize=normalize) * 100.0
        shares.columns = [str(c) for c in shares.columns]
        return shares.round(2).reset_index()

    @staticmethod
    def baseline_accuracy(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
        """Accuracy of always predicting the most frequent class of a categorical
        `target_col` — the bar any classifier must beat."""
        _require_columns(df, target_col)
        target = df[target_col].dropna()
        if target.empty:
            raise ValueError(f"Target column '{target_col}' has no non-null values.")
        counts = target.value_counts()
        if len(counts) > _MAX_CLASSES:
            raise ValueError(
                f"'{target_col}' has {len(counts)} distinct values; baseline_accuracy is "
                "for a categorical target (use a mean/median baseline for a numeric one)."
            )
        return {
            "target": target_col,
            "majority_class": str(counts.index[0]),
            "majority_baseline_accuracy": round(float(counts.iloc[0]) / len(target), 4),
            "n_classes": len(counts),
            "n": len(target),
        }

    @staticmethod
    def derive(df: pd.DataFrame, name: str, expr: str) -> pd.DataFrame:
        """New DataFrame = `df` plus column `name` computed from `expr`: column
        names (`backtick` ones with spaces), numeric literals, + - * / ** with a
        literal exponent, and parentheses — e.g. "revenue - cost". Division by
        zero gives NaN. No calls or attribute access; `name` must be new."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("derive needs a non-empty column name.")
        if name in df.columns:
            raise ValueError(f"Column '{name}' already exists; choose a new name.")
        if not isinstance(expr, str) or not expr.strip() or len(expr) > _MAX_EXPR_CHARS:
            raise ValueError(f"expr must be a non-empty string of at most {_MAX_EXPR_CHARS} characters.")
        aliases: dict[str, str] = {}

        def alias(match: re.Match[str]) -> str:
            aliases[f"__dsa_col_{len(aliases)}__"] = match.group(1)
            return f"__dsa_col_{len(aliases) - 1}__"

        try:
            tree = ast.parse(_BACKTICKED.sub(alias, expr.strip()), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"expr is not a valid arithmetic expression: {exc.msg}.") from None
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        _require_columns(df, *(aliases.get(u, u) for u in used))
        env: dict[str, pd.Series] = {}
        for ident in used:
            col = aliases.get(ident, ident)
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' is not numeric.")
            env[ident] = df[col].astype("float64")
        try:
            value = _eval_arith(tree, env)
        except RecursionError:
            raise ValueError("expr is nested too deeply.") from None
        out = df.copy()
        out[name] = pd.Series(value, index=df.index, dtype="float64").replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def share_of_total(df: pd.DataFrame, col: str, by: str | None = None) -> pd.DataFrame:
        """Each row's (by=None) or each `by` level's share of the `col` total,
        as `share_pct`. by=None returns `df` plus `<col>_share_pct`; with `by`,
        one row per level (largest first): by, col (sum), share_pct."""
        _require_columns(df, col, *([by] if by else []))
        values = pd.to_numeric(df[col], errors="coerce")
        total = float(values.sum())
        if total == 0 or not math.isfinite(total):
            raise ValueError(f"'{col}' sums to {total:g}; shares of it are undefined.")
        if by is None:
            out = df.copy()
            out[f"{col}_share_pct"] = (values / total * 100.0).round(2)
            return out
        grouped = values.groupby(df[by], dropna=False, observed=True).sum().sort_values(ascending=False)
        result = grouped.rename(col).reset_index()
        result["share_pct"] = (result[col] / total * 100.0).round(2)
        return result

    @staticmethod
    def contribution_to_change(
        df: pd.DataFrame, col: str, period_col: str, by: str | None = None
    ) -> pd.DataFrame:
        """Period-over-period change in the `col` total (sorted by `period_col`).
        Without `by`: period, value, change, pct_change. With `by`: one row per
        period (after the first) and level: change, share_of_change_pct (of the
        total change) and contribution_pp (points of the previous total's growth)."""
        _require_columns(df, col, period_col, *([by] if by else []))
        work = pd.DataFrame({period_col: df[period_col], col: pd.to_numeric(df[col], errors="coerce")})
        if by is None:
            totals = work.groupby(period_col, observed=True)[col].sum().sort_index()
            return pd.DataFrame({
                period_col: totals.index,
                "value": totals.to_numpy(),
                "change": totals.diff().to_numpy(),
                "pct_change": (totals.pct_change() * 100.0).round(2).to_numpy(),
            })
        work[by] = df[by]
        pivot = work.groupby([period_col, by], observed=True)[col].sum().unstack(fill_value=0).sort_index()
        total = pivot.sum(axis=1)
        delta = pivot.diff().iloc[1:]
        rows = delta.reset_index().melt(id_vars=period_col, var_name=by, value_name="change")
        total_change = rows[period_col].map(total.diff())
        prev_total = rows[period_col].map(total.shift())
        with np.errstate(all="ignore"):
            rows["share_of_change_pct"] = (rows["change"] / total_change.replace(0, np.nan) * 100.0).round(2)
            rows["contribution_pp"] = (rows["change"] / prev_total.replace(0, np.nan) * 100.0).round(2)
        rows["_order"] = rows["change"].abs()
        rows = rows.sort_values([period_col, "_order"], ascending=[True, False]).drop(columns="_order")
        return rows.reset_index(drop=True)

    @staticmethod
    def query_sql(sql: str, **tables: pd.DataFrame) -> pd.DataFrame:
        """Execute in-memory vectorized SQL via DuckDB/Polars across registered DataFrames.
        Disallows file I/O, network access, or disk modifications.
        Example:
            dsa.query_sql("SELECT category, AVG(price) as avg_p FROM df GROUP BY category", df=df)
        """
        from src.core.duckdb_engine import query_dataframe
        return query_dataframe(sql, tables)

