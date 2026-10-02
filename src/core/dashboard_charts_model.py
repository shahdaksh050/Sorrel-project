"""
Dashboard Agent — model-result charts (comparison, drivers, confusion matrix, ROC).

Split out of dashboard.py; `src.core.dashboard` re-exports every name here.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.chart_theme import humanize_label
from src.core.dashboard_common import MAX_POINTS, ChartSpec, _num


def _model_comparison_chart(train_output: dict[str, Any] | None) -> ChartSpec | None:
    if not train_output:
        return None
    models = train_output.get("models_trained", {})
    if not isinstance(models, dict) or not models:
        return None
    task = str(train_output.get("task_type", "classification"))
    metric = "accuracy" if task == "classification" else "r2"
    # Accuracy reads as a percentage; R² stays on its own scale (it can be
    # negative, and the y domain fits the data rather than clipping it).
    scale, metric_title = (100, "Accuracy %") if metric == "accuracy" else (1, "R²")

    rows: list[dict[str, Any]] = []
    for name, m in models.items():
        if not isinstance(m, dict):
            continue
        # A metric the tool didn't report is left out — never drawn as 0.
        for label, raw in (
            ("Train", (m.get("train_metrics") or {}).get(metric)),
            ("Test", (m.get("test_metrics") or {}).get(metric)),
            ("CV mean", m.get("cv_mean")),
        ):
            if isinstance(raw, (int, float)) and not isinstance(raw, bool) and np.isfinite(raw):
                rows.append({"model": str(name), "metric": label, "score": round(float(raw) * scale, 4)})
    if not rows:
        return None
    return ChartSpec(
        chart_id="model_comparison",
        title="Model Comparison",
        description=f"Train vs held-out test vs cross-validated {metric}. "
                    "A large train-test gap signals overfitting.",
        spec={
            "data": {"values": rows},
            "mark": {"type": "bar"},
            "height": 280,
            "encoding": {
                "x": {"field": "model", "type": "nominal", "axis": {"labelAngle": 0, "title": None}},
                "xOffset": {"field": "metric"},
                "y": {"field": "score", "type": "quantitative", "title": metric_title},
                "color": {
                    "field": "metric",
                    "scale": {"domain": ["Train", "Test", "CV mean"]},
                    "legend": {"orient": "top", "title": None},
                },
                "tooltip": [{"field": "model"}, {"field": "metric"},
                            {"field": "score", "title": metric_title, "format": ".3~f"}],
            },
        },
    )


def _importance_bounds(driver: dict[str, Any], importance: float) -> tuple[float, float] | None:
    """(low, high) around a driver's importance: its own CI when reported,
    else ±1 std across the permutation repeats, else None."""
    low = _num(driver.get("importance_ci_lower", driver.get("ci_lower")))
    high = _num(driver.get("importance_ci_upper", driver.get("ci_upper")))
    if low is not None and high is not None:
        return low, high
    std = _num(driver.get("importance_std", driver.get("std")))
    return (importance - std, importance + std) if std is not None else None


def _drivers_chart(model_output: dict[str, Any] | None) -> ChartSpec | None:
    """Permutation-importance drivers (evaluate_model's `top_drivers`) — the
    one chart that says *what moves the outcome* after a modelling run. A
    dot plot, with the importance's spread across shuffles as error bars
    whenever the tool reported it."""
    if not model_output:
        return None
    values: list[dict[str, Any]] = []
    for d in model_output.get("top_drivers") or []:
        importance = _num(d.get("importance")) if isinstance(d, dict) else None
        if importance is None or d.get("feature") is None:
            continue
        bounds = _importance_bounds(d, importance)
        values.append({
            "feature": humanize_label(str(d["feature"])),
            "importance": round(importance, 4),
            "low": round(bounds[0], 4) if bounds else None,
            "high": round(bounds[1], 4) if bounds else None,
            "effect": str(d.get("headline") or d.get("direction") or ""),
        })
    if not values:
        return None
    y_enc = {"field": "feature", "type": "nominal", "sort": {"field": "importance", "order": "descending"},
             "title": None}
    x_title = "Permutation importance"
    layers: list[dict[str, Any]] = []
    has_bounds = any(v["low"] is not None for v in values)
    if has_bounds:
        layers.append({"mark": {"type": "rule"}, "encoding": {
            "y": y_enc,
            "x": {"field": "low", "type": "quantitative", "title": x_title},
            "x2": {"field": "high"},
        }})
    layers.append({"mark": {"type": "circle", "size": 90, "opacity": 1}, "encoding": {
        "y": y_enc,
        "x": {"field": "importance", "type": "quantitative", "title": x_title},
        "tooltip": [{"field": "feature", "title": "Feature"},
                    {"field": "importance", "title": "Importance", "format": ".3~f"},
                    *([{"field": "low", "title": "Low", "format": ".3~f"},
                       {"field": "high", "title": "High", "format": ".3~f"}] if has_bounds else []),
                    {"field": "effect", "title": "Effect"}],
    }})
    return ChartSpec(
        chart_id="model_drivers",
        title="What Drives the Outcome — Top Model Drivers",
        description="Permutation importance on held-out data: how much the model's score drops "
                    "when each feature is shuffled. Further right = the model leans on it more."
                    + (" Whiskers show how much that drop varied across repeated shuffles."
                       if has_bounds else ""),
        spec={
            "data": {"values": values},
            "height": max(140, 30 * len(values)),
            "layer": layers,
        },
    )


def _confusion_matrix_chart(eval_output: dict[str, Any] | None) -> ChartSpec | None:
    """Held-out confusion matrix, when evaluate_model reports one (a list of
    rows, actual x predicted, aligned with `class_labels`)."""
    matrix = (eval_output or {}).get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) < 2 or not all(isinstance(r, list) for r in matrix):
        return None
    labels = (eval_output or {}).get("class_labels") or []
    if len(labels) != len(matrix):
        labels = [str(i) for i in range(len(matrix))]
    values = [
        {"actual": str(labels[i]), "predicted": str(labels[j]), "count": int(c)}
        for i, row in enumerate(matrix) for j, c in enumerate(row)
        if j < len(labels) and _num(c) is not None
    ]
    if not values:
        return None
    order = [str(label) for label in labels]
    x_enc = {"field": "predicted", "type": "nominal", "sort": order, "title": "Predicted",
             "axis": {"labelAngle": 0}}
    y_enc = {"field": "actual", "type": "nominal", "sort": order, "title": "Actual"}
    return ChartSpec(
        chart_id="model_confusion_matrix",
        title="Where the Model Gets It Wrong — Confusion Matrix",
        description="Held-out rows only: each cell counts rows of the actual class (row) "
                    "predicted as the column's class. The diagonal is correct predictions.",
        spec={
            "data": {"values": values},
            "height": max(160, 40 * len(labels)),
            "layer": [
                {"mark": {"type": "rect"}, "encoding": {
                    "x": x_enc, "y": y_enc,
                    "color": {"field": "count", "type": "quantitative", "title": "Rows",
                              "scale": {"scheme": "oranges"}},
                    "tooltip": [{"field": "actual", "title": "Actual"},
                                {"field": "predicted", "title": "Predicted"},
                                {"field": "count", "title": "Rows", "format": ",d"}],
                }},
                {"mark": {"type": "text"}, "encoding": {
                    "x": x_enc, "y": y_enc, "text": {"field": "count", "format": ",d"},
                }},
            ],
        },
    )


def _roc_chart(eval_output: dict[str, Any] | None) -> ChartSpec | None:
    """Held-out ROC curve, when evaluate_model reports its points — either
    {"fpr": [...], "tpr": [...]} or a list of {"fpr", "tpr"} records."""
    roc = (eval_output or {}).get("roc_curve")
    pairs: list[tuple[Any, Any]] = []
    if isinstance(roc, dict) and isinstance(roc.get("fpr"), list) and isinstance(roc.get("tpr"), list):
        pairs = list(zip(roc["fpr"], roc["tpr"], strict=False))
    elif isinstance(roc, list):
        pairs = [(p.get("fpr"), p.get("tpr")) for p in roc if isinstance(p, dict)]
    points = [(f, t) for f, t in ((_num(a), _num(b)) for a, b in pairs) if f is not None and t is not None]
    if len(points) < 2:
        return None
    if len(points) > MAX_POINTS:
        points = points[:: -(-len(points) // MAX_POINTS)]
    values = [{"fpr": round(f, 4), "tpr": round(t, 4)} for f, t in points]
    auc = _num((eval_output or {}).get("roc_auc", roc.get("auc") if isinstance(roc, dict) else None))
    axis = {"scale": {"domain": [0, 1]}}
    return ChartSpec(
        chart_id="model_roc",
        title="How Well the Model Ranks Cases — ROC Curve" + (f" (AUC {auc:.2f})" if auc is not None else ""),
        description="Held-out rows only: true-positive rate against false-positive rate across every "
                    "decision threshold. The dashed diagonal is a coin flip; the closer the curve hugs "
                    "the top-left corner, the better.",
        spec={
            "height": 260,
            "layer": [
                {"data": {"values": values}, "mark": {"type": "line"}, "encoding": {
                    "x": {"field": "fpr", "type": "quantitative", "title": "False-positive rate", **axis},
                    "y": {"field": "tpr", "type": "quantitative", "title": "True-positive rate", **axis},
                    "tooltip": [{"field": "fpr", "title": "FPR"}, {"field": "tpr", "title": "TPR"}],
                }},
                {"data": {"values": [{"fpr": 0, "tpr": 0}, {"fpr": 1, "tpr": 1}]},
                 "mark": {"type": "line", "strokeDash": [4, 3]},
                 "encoding": {"x": {"field": "fpr", "type": "quantitative"},
                              "y": {"field": "tpr", "type": "quantitative"}}},
            ],
        },
    )
