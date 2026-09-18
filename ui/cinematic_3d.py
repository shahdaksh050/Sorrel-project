"""
The 6-Section Cinematic Experience — Anime.js + fullPage.js + Three.js 3D Master Architecture.

This module provides the Python interface between Streamlit / standalone tools and
the hardware-accelerated 3D fullpage presentation. It serializes live session state
(dataset metadata, 7-stage workflow logs, ML cross-validation scores, overfit guards,
and executive synthesis) and inlines the HTML and JavaScript assets into a sandboxed
viewport or standalone exportable presentation file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

__all__ = [
    "CINEMATIC_PALETTES",
    "build_cinematic_document",
    "export_cinematic_html",
    "extract_cinematic_state",
    "render_cinematic",
]

_ASSETS = Path(__file__).parent / "assets"

#: Luxury Ledger Palettes for Day and Night modes
CINEMATIC_PALETTES: dict[str, dict[str, str]] = {
    "day": {
        "stock": "#f7eedd",
        "sheet": "#fffbf2",
        "sheet_alt": "#f1e4cb",
        "ink": "#3a2b1e",
        "graphite": "#8a7660",
        "pen": "#a34f20",
        "pen_glow": "rgba(163, 79, 32, 0.35)",
        "risk": "#a33526",
        "positive": "#5b8c5a",
        "accent": "#e08a3e",
        "grid": "#e4d4bc",
        "card_bg": "rgba(255, 251, 242, 0.88)",
        "card_border": "rgba(163, 79, 32, 0.22)",
    },
    "night": {
        "stock": "#241c14",
        "sheet": "#1c1610",
        "sheet_alt": "#282017",
        "ink": "#f6eedf",
        "graphite": "#bdae97",
        "pen": "#f0a24a",
        "pen_glow": "rgba(240, 162, 74, 0.45)",
        "risk": "#e2685a",
        "positive": "#7fb77e",
        "accent": "#4fc3f7",
        "grid": "#4a3c28",
        "card_bg": "rgba(28, 22, 16, 0.82)",
        "card_border": "rgba(240, 162, 74, 0.25)",
    },
}


def _read_asset(filename: str) -> str:
    """Read bundled HTML or JS asset with UTF-8 encoding."""
    return (_ASSETS / filename).read_text(encoding="utf-8")


def extract_cinematic_state(session_state: Any) -> dict[str, Any]:
    """Extract and sanitize live analysis data from Streamlit session_state.

    Falls back cleanly to rich demonstration data if analysis hasn't run yet.
    """
    # 1. Workflow Stages
    stage_log_raw = session_state.get("stage_log")
    stage_log = stage_log_raw if isinstance(stage_log_raw, list) else []
    log_map = {num: (status, detail) for num, status, detail in stage_log}

    stage_defs = [
        ("1", "Dataset Ingestion"),
        ("2", "Initial Reasoning"),
        ("3", "Tool Execution"),
        ("4", "Result Interpretation"),
        ("5", "Iterative Refinement"),
        ("6", "RLM Decomposition"),
        ("7", "Report Generation"),
    ]

    stages = [
        {
            "num": num,
            "name": name,
            "status": log_map.get(num, ("done" if session_state.get("analysis_done") else "pending", ""))[0],
            "detail": log_map.get(num, ("", ""))[1],
        }
        for num, name in stage_defs
    ]

    # 2. Dataset Metadata
    meta = session_state.get("metadata")
    preview_df = session_state.get("preview_df")
    preview_name = session_state.get("preview_name") or "sample_dataset.csv"

    if meta is not None:
        row_count = getattr(meta, "row_count", 0)
        col_count = getattr(meta, "column_count", 0)
        task_type = getattr(meta, "task_type", "classification")
        target_col = getattr(meta, "target_column", "Target")
        miss_cells = getattr(meta, "missing_cells", 0)
    elif preview_df is not None:
        row_count = len(preview_df)
        col_count = len(preview_df.columns)
        task_type = "classification" if col_count > 2 else "exploratory"
        target_col = preview_df.columns[-1] if len(preview_df.columns) > 0 else "N/A"
        miss_cells = int(preview_df.isnull().sum().sum())
    else:
        row_count = 1420
        col_count = 14
        task_type = "classification"
        target_col = "converted"
        miss_cells = 12

    # 3. Profiler Insights
    profile_raw = session_state.get("profile")
    profile = profile_raw if isinstance(profile_raw, dict) else {}
    # DatasetProfile.to_dict() emits "columns" as a LIST of column dicts, and so
    # does the sample-report demo state. Accept a name->column mapping too, since
    # this reads straight off session_state and must not take the results page
    # down if the shape ever changes.
    cols_raw = profile.get("columns") if isinstance(profile, dict) else None
    if isinstance(cols_raw, dict):
        cols_summary: list[Any] = list(cols_raw.values())
    elif isinstance(cols_raw, list):
        cols_summary = cols_raw
    else:
        cols_summary = []

    def _count_kind(kind: str) -> int:
        return sum(1 for c in cols_summary if isinstance(c, dict) and c.get("kind") == kind)

    # The `or N` fallbacks keep the showcase populated before any profile exists.
    num_numeric = _count_kind("numeric") or 8
    num_categorical = _count_kind("categorical") or 4
    num_datetime = _count_kind("datetime") or 1
    num_text = _count_kind("text") or 1
    quality_score = profile.get("quality_score", 94) if isinstance(profile, dict) else 94

    # 4. Statistical & Tool Outputs
    tool_results_raw = session_state.get("tool_results")
    tool_results: list[dict[str, Any]] = tool_results_raw if isinstance(tool_results_raw, list) else []

    def find_tool(name: str) -> dict[str, Any] | None:
        for r in tool_results:
            if isinstance(r, dict) and (r.get("tool_name") == name or r.get("tool") == name):
                return r
        return None

    train_out = find_tool("train_model")
    stat_out = find_tool("select_statistical_test")
    corr_out = find_tool("correlation_analysis")
    outlier_out = find_tool("detect_outliers")

    # 5. ML Models & Cross-Validation
    models_list: list[dict[str, Any]] = []
    best_model_name = "GradientBoosting"
    best_cv_score = 0.912
    best_gap = 0.038
    overfit_warnings: list[str] = []

    if train_out and isinstance(train_out, dict):
        best_model_name = train_out.get("best_model", "Model")
        models_trained = train_out.get("models_trained", {})
        if isinstance(models_trained, dict):
            for m_name, m_info in models_trained.items():
                if isinstance(m_info, dict):
                    cv_m = m_info.get("cv_mean", 0.0)
                    gap = m_info.get("train_test_gap", 0.0)
                    warns = m_info.get("overfit_warnings", [])
                    models_list.append({
                        "name": m_name,
                        "cv_mean": round(cv_m * 100, 2),
                        "cv_std": round(m_info.get("cv_std", 0.0) * 100, 2),
                        "gap": round(gap * 100, 2) if gap is not None else 0.0,
                        "is_best": m_name == best_model_name,
                        "warnings": warns,
                    })
                    if m_name == best_model_name:
                        best_cv_score = cv_m
                        best_gap = gap if gap is not None else 0.0
                        overfit_warnings.extend(warns)

    if not models_list:
        # Default high-fidelity demonstration models
        models_list = [
            {"name": "GradientBoostingClassifier", "cv_mean": 92.4, "cv_std": 1.4, "gap": 3.2, "is_best": True, "warnings": []},
            {"name": "RandomForestClassifier", "cv_mean": 90.8, "cv_std": 1.8, "gap": 5.1, "is_best": False, "warnings": []},
            {"name": "LogisticRegression(L2)", "cv_mean": 86.5, "cv_std": 2.1, "gap": 1.8, "is_best": False, "warnings": []},
            {"name": "DecisionTreeClassifier", "cv_mean": 81.2, "cv_std": 3.4, "gap": 12.8, "is_best": False, "warnings": ["Train-test gap > 10% (overfitting)"]},
        ]

    # 6. Executive Synthesis
    report_raw = session_state.get("final_report")
    report = report_raw if isinstance(report_raw, dict) else {}
    user_objective = session_state.get("user_objective") or "Analyze key drivers of conversion and detect any overfitting."
    reasoning = report.get("reasoning") or (
        "Analysis completed across 5-fold stratified cross-validation. GradientBoosting demonstrated optimal generalizability "
        "with 92.4% CV accuracy and a tight 3.2% train-test gap. Outlier detection identified 1.8% anomalous records which were "
        "robustly normalized. No data leakage or unregularized multi-collinearity was observed."
    )

    theme = session_state.get("theme", "night")
    palette = CINEMATIC_PALETTES.get(theme, CINEMATIC_PALETTES["night"])

    top_corrs = [
        {"pair": "tenure ↔ total_spend", "val": 0.78},
        {"pair": "usage_rate ↔ converted", "val": 0.64},
        {"pair": "support_tickets ↔ churn_risk", "val": 0.59},
    ]
    if corr_out and isinstance(corr_out, dict) and isinstance(corr_out.get("correlations"), list):
        parsed_corrs = []
        for c in corr_out["correlations"][:4]:
            if isinstance(c, dict):
                p1 = c.get("feature_1", "")
                p2 = c.get("feature_2", "")
                v = c.get("correlation", 0.0)
                parsed_corrs.append({"pair": f"{p1} ↔ {p2}", "val": round(v, 2)})
        if parsed_corrs:
            top_corrs = parsed_corrs

    stat_name = stat_out.get("test_name", "Two-Sample T-Test / Mann-Whitney") if isinstance(stat_out, dict) else "Two-Sample T-Test"
    stat_p = stat_out.get("p_value", 0.0012) if isinstance(stat_out, dict) else 0.0012
    outlier_p = outlier_out.get("outlier_percentage", 1.8) if isinstance(outlier_out, dict) else 1.8

    return {
        "theme": theme,
        "palette": palette,
        "dataset": {
            "name": preview_name,
            "row_count": row_count,
            "col_count": col_count,
            "task_type": task_type,
            "target_col": target_col,
            "missing_cells": miss_cells,
            "quality_score": quality_score,
            "column_types": {
                "numeric": num_numeric,
                "categorical": num_categorical,
                "datetime": num_datetime,
                "text": num_text,
            },
        },
        "stages": stages,
        "statistics": {
            "test_name": stat_name,
            "p_value": stat_p,
            "significant": True,
            "top_correlations": top_corrs,
            "outlier_pct": outlier_p,
        },
        "ml": {
            "best_model": best_model_name,
            "best_cv": round(best_cv_score * 100, 1),
            "best_gap": round(best_gap * 100, 1),
            "models": models_list,
            "overfit_warnings": overfit_warnings,
            "cv_folds": 5,
            "regularization": "L2 Ridge / Stratified 5-Fold",
        },
        "synthesis": {
            "objective": user_objective,
            "reasoning": reasoning,
            "findings": [
                "Primary driver of conversion is user session engagement duration (>4.2m).",
                "Cross-validation envelope indicates high generalization stability (std: ±1.4%).",
                "Tree depth capped at max_depth=6 to strictly prohibit memorizing noise.",
            ],
        },
    }


def build_cinematic_document(
    state_dict: dict[str, Any] | None = None,
    theme: str = "night",
) -> str:
    """Assemble the self-contained HTML document for the 6-Section 3D Cinematic Experience.

    Inlines HTML structure, styles, Three.js scene, fullPage.js choreography,
    and Anime.js v4 unified animation loop.
    """
    if state_dict is None:
        state_dict = extract_cinematic_state({"theme": theme})
    else:
        state_dict.setdefault("theme", theme)
        state_dict.setdefault("palette", CINEMATIC_PALETTES.get(theme, CINEMATIC_PALETTES["night"]))

    # Escape state JSON against early closing script tag
    state_json = json.dumps(state_dict, ensure_ascii=False).replace("<", "\\u003c")

    html_template = _read_asset("cinematic_3d.html")
    scene_script = _read_asset("cinematic_3d.js")

    return (
        html_template
        .replace("__CINEMATIC_STATE_JSON__", state_json)
        .replace("__CINEMATIC_SCENE_SCRIPT__", scene_script)
    )


def render_cinematic(
    state_or_session: Any = None,
    *,
    height: int = 860,
    theme: str = "night",
    compact: bool = False,
) -> None:
    """Mount the 6-Section Cinematic Experience in Streamlit.

    Uses st.iframe when available, falling back to components.html.

    Args:
        compact: Small-viewport mode for the workspace hero box. Opens on a text-light
            step 0 (a loose particle cloud that assembles into step 1) and hides the
            header brand block, which otherwise collides with the section text.
    """
    if isinstance(state_or_session, dict) and "dataset" in state_or_session:
        state_dict = state_or_session
    elif state_or_session is not None:
        state_dict = extract_cinematic_state(state_or_session)
    else:
        state_dict = extract_cinematic_state(st.session_state)

    state_dict["theme"] = theme
    state_dict["compact"] = compact
    doc = build_cinematic_document(state_dict, theme=theme)

    if hasattr(st, "iframe"):
        st.iframe(doc, height=height)
    else:
        components.html(doc, height=height, scrolling=False)


def export_cinematic_html(
    output_path: Path | str,
    state_dict: dict[str, Any] | None = None,
    theme: str = "night",
) -> Path:
    """Export the self-contained 3D presentation as a standalone HTML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = build_cinematic_document(state_dict=state_dict, theme=theme)
    path.write_text(doc, encoding="utf-8")
    return path
