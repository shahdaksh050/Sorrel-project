"""
Visualization Tools — Execution Layer.

Stage 3: declarative chart specs (src.core.chart_spec) for a correlation
heatmap, model feature importances, or numeric distributions. The specs are
returned in the output's "charts" list and rendered by the dashboard
(src.core.dashboard), which themes them — this tool writes no files.

Model evaluation charts (ROC, confusion matrix) are not produced here: they
belong to evaluate_model's held-out results, which the dashboard charts
directly. Computing them over the full file would score the model on the
rows it was trained on.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.chart_spec import validate_chart_spec
from src.core.chart_theme import humanize_label
from src.core.io import DatasetReadError, read_any
from src.tools.base import BaseTool, ToolExecutionError

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata, MemorySystem
    from src.core.profiler import DatasetProfile

#: Columns in a correlation heatmap (n² cells must stay under the spec's row cap).
MAX_HEATMAP_COLUMNS = 20
#: Numeric columns given a distribution chart.
MAX_DISTRIBUTIONS = 6
#: Features shown in an importance chart.
MAX_IMPORTANCES = 20
#: Bins per distribution.
HIST_BINS = 20

#: Chart types that used to be drawn here on training rows; now answered by
#: evaluate_model's held-out results.
_EVALUATION_CHARTS = ("roc_curve", "confusion_matrix")


def _read_df(file_path: str) -> pd.DataFrame:
    """Read a dataset via the unified reader (src.core.io.read_any)."""
    try:
        df, _report = read_any(file_path)
    except DatasetReadError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return df


def _validated(spec: dict[str, Any]) -> dict[str, Any]:
    clean, error = validate_chart_spec(spec)
    if clean is None:
        raise ToolExecutionError(f"Chart spec rejected: {error}")
    return clean


class GenerateVisualizationsTool(BaseTool):
    """
    Build declarative chart specs for the dashboard.

    Supported chart_type values:
      - correlation_heatmap
      - feature_importance
      - distributions
    (roc_curve / confusion_matrix are rejected with a pointer to
    evaluate_model, whose held-out results the dashboard already charts.)
    """

    name = "generate_visualizations"
    description = (
        "Add charts to the dashboard. "
        "chart_type: correlation_heatmap | feature_importance | distributions. "
        "ROC / confusion-matrix charts come from evaluate_model's held-out results instead. "
        "Returns validated chart specs under 'charts'."
    )

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        params = super().prepare_params(params, memory, output_root)
        best_path = memory.get_context("best_model_path")
        if best_path:
            raw_mp = params.get("model_path", "")
            if not raw_mp or not Path(raw_mp).exists():
                params["model_path"] = best_path
        return params

    def default_params(
        self, profile: DatasetProfile | None, metadata: DatasetMetadata | None
    ) -> dict[str, Any]:
        """Choose a chart the data can actually support.

        A correlation heatmap needs two or more numeric columns; below that
        distributions still work. Without this the deterministic planner
        scheduled the tool with no chart_type at all and it failed every run.
        """
        if profile is None:
            return {"chart_type": "distributions"}
        n_numeric = sum(1 for c in profile.columns if c.kind == "numeric")
        if n_numeric >= 2:
            return {"chart_type": "correlation_heatmap"}
        if n_numeric >= 1:
            return {"chart_type": "distributions"}
        return {}

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        chart_type: str,
        target_column: str | None = None,
        model_path: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if chart_type in _EVALUATION_CHARTS:
            raise ToolExecutionError(
                f"'{chart_type}' is not drawn here — it would score the model on its own "
                "training rows. Run evaluate_model: its held-out results are charted on the dashboard."
            )
        df = _read_df(file_path)
        if chart_type == "correlation_heatmap":
            charts = self._heatmap(df)
        elif chart_type == "feature_importance":
            charts = self._feature_importance(df, target_column, model_path)
        elif chart_type == "distributions":
            charts = self._distributions(df)
        else:
            raise ToolExecutionError(
                f"Unknown chart_type '{chart_type}'. "
                "Valid: ['correlation_heatmap', 'feature_importance', 'distributions']"
            )
        return {
            "summary": f"Built {len(charts)} '{chart_type}' chart(s) for the dashboard.",
            "chart_type": chart_type,
            "charts": charts,
        }

    # ------------------------------------------------------------------
    # Chart specs
    # ------------------------------------------------------------------

    def _heatmap(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            raise ToolExecutionError(
                f"correlation_heatmap needs at least 2 numeric columns; this "
                f"dataset has {num_df.shape[1]}. Use 'distributions' instead."
            )
        # Keep the most variable columns when there are too many to show.
        if num_df.shape[1] > MAX_HEATMAP_COLUMNS:
            scaled_var = (num_df / num_df.abs().max().replace(0, 1)).var()
            num_df = num_df[scaled_var.nlargest(MAX_HEATMAP_COLUMNS).index]
        corr = num_df.corr()
        rows = [
            {"feature_x": humanize_label(str(a)), "feature_y": humanize_label(str(b)), "r": round(float(v), 3)}
            for a in corr.columns for b in corr.index
            if pd.notna(v := corr.loc[b, a])
        ]
        return [_validated({
            "type": "heatmap", "data": rows, "x": "feature_x", "y": "feature_y", "color": "r",
            "title": "Feature Correlation Heatmap", "x_title": "Column", "y_title": "Column",
            "caption": "Pearson correlation between every pair of numeric columns (−1 to +1).",
        })]

    def _feature_importance(
        self, df: pd.DataFrame, target_column: str | None, model_path: str | None,
    ) -> list[dict[str, Any]]:
        if not model_path or not Path(model_path).exists():
            raise ToolExecutionError("model_path is required for feature_importance chart.")
        if not target_column:
            raise ToolExecutionError("target_column is required for feature_importance chart.")

        with open(model_path, "rb") as f:
            loaded = pickle.load(f)

        # train_model saves a Pipeline([("prep", ColumnTransformer), ("model",
        # estimator)]) (IMPROVEMENTS.md P0.1/P0.5) — importances live on the
        # "model" step, and the raw df.columns don't match its length once
        # one-hot encoding has expanded the categoricals, so names must come
        # from the fitted preprocessor's post-encoding output, not df.columns.
        if hasattr(loaded, "named_steps") and "model" in loaded.named_steps:
            model = loaded.named_steps["model"]
            feature_cols = [str(c) for c in loaded.named_steps["prep"].get_feature_names_out()]
        else:
            model = loaded
            feature_cols = [str(c) for c in getattr(model, "feature_names_in_", [])]
            if not feature_cols:
                from src.tools.ml_pipeline import _prepare_features
                X, _y, _t = _prepare_features(df, target_column)
                feature_cols = [str(c) for c in X.columns]

        if hasattr(model, "feature_importances_"):
            # Tree-based models (RandomForest, XGBoost)
            raw = np.asarray(model.feature_importances_, dtype=float)
            title, y_title = "Top Feature Importances", "Importance"
        elif hasattr(model, "coef_"):
            # Linear models: absolute coefficient magnitude as the proxy.
            # coef_ is (n_classes, n_features) for multi-class, (n_features,) otherwise.
            coef = np.asarray(model.coef_, dtype=float)
            raw = np.abs(coef[0] if coef.ndim == 2 else coef)
            title, y_title = "Top Feature Importances (|coefficient|)", "|Coefficient|"
        else:
            raise ToolExecutionError(
                "Model does not expose feature_importances_ or coef_. "
                "Feature importance chart requires a tree-based or linear model."
            )

        if len(raw) != len(feature_cols):
            raise ToolExecutionError(
                f"Model expects {len(raw)} features but {len(feature_cols)} "
                f"column names were derived — the dataset passed to "
                f"feature_importance does not match the one the model was "
                f"trained on. Pass the same (cleaned) file used by train_model."
            )

        importances = pd.Series(raw, index=feature_cols).sort_values(ascending=False).head(MAX_IMPORTANCES)
        rows = [{"feature": humanize_label(k), "importance": round(float(v), 4)} for k, v in importances.items()]
        return [_validated({
            "type": "bar", "data": rows, "x": "feature", "y": "importance", "sort": "desc",
            "title": title, "x_title": "", "y_title": y_title,
            "caption": "The model's own (training-time) importance scores; the dashboard's drivers "
                       "chart ranks features on held-out data instead.",
        })]

    def _distributions(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        num_cols = df.select_dtypes(include="number").columns.tolist()[:MAX_DISTRIBUTIONS]
        charts: list[dict[str, Any]] = []
        for col in num_cols:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if values.nunique() < 2:
                continue
            # Pre-binned over every row (aggregate before charting), drawn as
            # a frequency polygon through each bin's midpoint.
            counts, edges = np.histogram(values.to_numpy(dtype=float), bins=min(HIST_BINS, values.nunique()))
            rows = [
                {"value": round(float((edges[i] + edges[i + 1]) / 2), 6), "rows": int(n)}
                for i, n in enumerate(counts)
            ]
            label = humanize_label(str(col))
            charts.append(_validated({
                "type": "area", "data": rows, "x": "value", "y": "rows", "y_format": "count",
                "title": f"Distribution — {label}", "x_title": label, "y_title": "Rows",
            }))
        if not charts:
            raise ToolExecutionError("No numeric column with at least two distinct values to chart.")
        return charts

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Dataset path.", "required": True},
            "chart_type": {
                "type": "string",
                "description": "Type of chart: correlation_heatmap | feature_importance | distributions.",
                "required": True,
            },
            "target_column": {
                "type": "string",
                "description": "Target column (required for feature_importance).",
                "required": False,
            },
            "model_path": {
                "type": "string",
                "description": "Path to .pkl model (required for feature_importance).",
                "required": False,
            },
        }
