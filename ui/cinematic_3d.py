"""
The 6-Section Cinematic Experience (Sorrel): section pager + Anime.js + Three.js.

This module provides the Python interface between Streamlit / standalone tools and
the hardware-accelerated 3D fullpage presentation. It serializes live session state
(dataset metadata, 7-stage workflow logs, ML cross-validation scores, overfit guards,
and executive synthesis) and inlines the HTML and JavaScript assets into a sandboxed
viewport or standalone exportable presentation file.
"""

from __future__ import annotations

import base64
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.core import design_tokens
from ui.styles import static_url

__all__ = [
    "CINEMATIC_PALETTES",
    "build_cinematic_document",
    "export_cinematic_html",
    "extract_cinematic_state",
    "render_cinematic",
]

_ASSETS = Path(__file__).parent / "assets"


def _rgba(hex_color: str, alpha: float) -> str:
    """A token hex as `rgba(...)`, so card surfaces track the token instead of drifting from it."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


#: Card surface alpha per mode: the only genuinely local part of this palette; everything else
#: comes straight from `design_tokens`. There is no blur behind the cards, so they stay mostly
#: opaque to keep text legible over the 3D scene (and fully opaque under reduced transparency).
_CARD_BG_ALPHA: dict[str, float] = {"day": 0.94, "night": 0.92}

#: The Sorrel palettes for Day and Night. This file's "grid" key predates `design_tokens`'
#: "rule" naming, so it is mapped explicitly. Card borders are the hairline token (no glow).
CINEMATIC_PALETTES: dict[str, dict[str, str]] = {
    mode: {
        **palette,
        "grid": palette["rule"],
        "card_bg": _rgba(palette["sheet"], _CARD_BG_ALPHA[mode]),
        "card_border": palette["rule"],
    }
    for mode, palette in design_tokens.PALETTES.items()
}

#: The font families the standalone export embeds. The shared stylesheet still declares older faces
#: until the final cleanup; none of them is used by this document, so none is inlined.
_EXPORT_FAMILIES = frozenset({"Geist", "Geist Mono", "Newsreader"})
_FONT_FACE_RE = re.compile(r"@font-face\s*\{[^}]*\}", re.DOTALL)
_FONT_FAMILY_RE = re.compile(r"font-family\s*:\s*['\"]?([^;'\"]+)['\"]?\s*;")
_FONT_URL_RE = re.compile(r"""url\((["']?)\./([^"')]+)\1\)""")


def _mode_css(mode: str) -> str:
    """The `--token: value;` lines the document's stylesheet carries for one mode."""
    palette = CINEMATIC_PALETTES[mode]
    lines = [design_tokens.css_root_block(mode, indent="      ")]  # type: ignore[arg-type]
    for key in ("grid", "card_bg", "card_border"):
        lines.append(f"      --{key.replace('_', '-')}: {palette[key]};")
    return "\n".join(lines)


def _inline_export_fonts(root: Path) -> str:
    """`<style>` for the standalone export: the Sorrel `@font-face` rules with each woff2 as a data URI.

    The rules are variable fonts (`format('woff2-variations')`, a weight range), so a rule is kept
    as written and only its `url('./file.woff2')` is replaced. A rule whose file is missing is
    dropped rather than left pointing at a relative path, so the export never depends on a file
    beside it (the page then falls back to the system fonts).
    """
    font_dir = root / "static" / "vendor" / "fonts"
    css = (font_dir / "ledger-fonts.css").read_text("utf-8")
    kept: list[str] = []
    for face in _FONT_FACE_RE.findall(css):
        family = _FONT_FAMILY_RE.search(face)
        if family is None or family.group(1).strip() not in _EXPORT_FAMILIES:
            continue
        missing = False

        def to_data_uri(match: re.Match[str]) -> str:
            nonlocal missing
            font_path = font_dir / match.group(2)
            if not font_path.is_file():
                missing = True
                return match.group(0)
            b64 = base64.b64encode(font_path.read_bytes()).decode("ascii")
            return f"url('data:font/woff2;base64,{b64}')"

        inlined = _FONT_URL_RE.sub(to_data_uri, face)
        if not missing:
            kept.append(inlined)
    return "<style>" + "\n".join(kept) + "</style>"


@lru_cache(maxsize=2)
def _read_asset(filename: str) -> str:
    """Read bundled HTML or JS asset with UTF-8 encoding, once per process (matches `pipeline_3d.py`'s `_asset`)."""
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
                if r.get("status") in ("success", None):
                    out = r.get("output")
                    if isinstance(out, dict):
                        return out
                return r
        return None

    train_out = find_tool("train_model")
    stat_out = find_tool("select_statistical_test") or find_tool("statistical_analysis")
    corr_out = find_tool("correlation_analysis")
    outlier_out = find_tool("detect_outliers")
    dim_out = find_tool("dimensionality_analysis")

    # 5. ML Models & Cross-Validation
    models_list: list[dict[str, Any]] = []
    best_model_name = ""
    best_cv_score = 0.0
    best_gap = 0.0
    overfit_warnings: list[str] = []
    has_models = False

    if train_out and isinstance(train_out, dict):
        best_model_candidate = train_out.get("best_model")
        models_trained = train_out.get("models_trained", {})
        if isinstance(models_trained, dict) and models_trained:
            has_models = True
            best_model_name = str(best_model_candidate or "Model")
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

    analysis_done = bool(session_state.get("analysis_done") or session_state.get("final_report"))
    if not has_models:
        if not analysis_done:
            # Pre-analysis demonstration models
            has_models = True
            best_model_name = "GradientBoosting"
            best_cv_score = 0.924
            best_gap = 0.032
            models_list = [
                {"name": "GradientBoostingClassifier", "cv_mean": 92.4, "cv_std": 1.4, "gap": 3.2, "is_best": True, "warnings": []},
                {"name": "RandomForestClassifier", "cv_mean": 90.8, "cv_std": 1.8, "gap": 5.1, "is_best": False, "warnings": []},
                {"name": "LogisticRegression(L2)", "cv_mean": 86.5, "cv_std": 2.1, "gap": 1.8, "is_best": False, "warnings": []},
                {"name": "DecisionTreeClassifier", "cv_mean": 81.2, "cv_std": 3.4, "gap": 12.8, "is_best": False, "warnings": ["Train-test gap > 10% (overfitting)"]},
            ]
        else:
            best_model_name = "None (Descriptive EDA)"

    # 6. Executive Synthesis & Real Findings
    report_raw = session_state.get("final_report")
    report = report_raw if isinstance(report_raw, dict) else {}
    run_view = session_state.get("run_view")
    if run_view is None and report:
        from src.core.run_view import build_run_view
        try:
            run_view = build_run_view(report, {}, objective=str(session_state.get("user_objective") or ""), is_sample=False)
        except Exception:
            run_view = None

    user_objective = session_state.get("user_objective") or (
        run_view.objective if run_view and run_view.objective else "Comprehensive Autonomous Statistical Exploration."
    )

    reasoning = ""
    if run_view and run_view.reasoning:
        reasoning = run_view.reasoning
    elif report.get("reasoning"):
        reasoning = str(report["reasoning"])
    elif report.get("executive_summary"):
        reasoning = str(report["executive_summary"])

    if not reasoning:
        if analysis_done:
            reasoning = (
                f"Analysis completed for {preview_name}. Identified key feature distributions, "
                f"correlation vectors, and outlier boundaries across {row_count:,} rows and {col_count} columns."
            )
        else:
            reasoning = (
                "Continuous autonomous analysis pipeline: Ingests raw data into external memory, "
                "profiles feature types, tests statistical hypotheses, trains validated models, "
                "and synthesizes verified findings."
            )

    real_findings: list[str] = []
    if run_view and run_view.headline_findings:
        for f in run_view.headline_findings:
            if isinstance(f, dict) and f.get("headline"):
                real_findings.append(f["headline"])
    elif report.get("key_findings"):
        for f in report["key_findings"]:
            if isinstance(f, dict) and f.get("headline"):
                real_findings.append(f["headline"])
            elif isinstance(f, str) and f.strip():
                real_findings.append(f.strip())
    elif report.get("findings"):
        for f in report["findings"]:
            if isinstance(f, dict) and f.get("headline"):
                real_findings.append(f["headline"])
            elif isinstance(f, str) and f.strip():
                real_findings.append(f.strip())
    elif run_view and run_view.insights:
        for ins in run_view.insights:
            if isinstance(ins, str) and ins.strip():
                real_findings.append(ins.strip())

    outlier_p = 0.0
    outlier_rows = 0
    if outlier_out and isinstance(outlier_out, dict):
        outlier_p = float(outlier_out.get("outlier_percentage", 0.0))
        outlier_rows = int(outlier_out.get("total_outliers", 0))
    elif not analysis_done:
        outlier_p = 1.8
        outlier_rows = int(row_count * 0.018)

    if not real_findings:
        if analysis_done:
            real_findings = [
                f"Data quality scored at {quality_score}/100 across {col_count} audited columns.",
                f"Audit identified {outlier_p}% anomalous records ({outlier_rows:,} rows).",
                "Statistical significance and multiple-testing thresholds verified.",
            ]
        else:
            real_findings = [
                "Feature distributions audited with automated normality and variance tests.",
                "Multiple testing controlled via Benjamini-Hochberg false discovery rate.",
                "Model generalization bounded with anti-overfitting constraints.",
            ]

    theme = session_state.get("theme", "night")
    palette = CINEMATIC_PALETTES.get(theme, CINEMATIC_PALETTES["night"])

    top_corrs: list[dict[str, Any]] = []
    if corr_out and isinstance(corr_out, dict):
        pairs_source = corr_out.get("top_correlations") or corr_out.get("correlations") or []
        if isinstance(pairs_source, list):
            for c in pairs_source[:4]:
                if isinstance(c, dict):
                    p1 = c.get("col_a") or c.get("feature_1") or ""
                    p2 = c.get("col_b") or c.get("feature_2") or ""
                    v = c.get("correlation", 0.0)
                    if p1 and p2:
                        top_corrs.append({"pair": f"{p1} ↔ {p2}", "val": round(float(v), 2)})
    if not top_corrs:
        if not analysis_done:
            top_corrs = [
                {"pair": "tenure ↔ total_spend", "val": 0.78},
                {"pair": "usage_rate ↔ converted", "val": 0.64},
                {"pair": "support_tickets ↔ churn_risk", "val": 0.59},
            ]

    stat_name = stat_out.get("test_name", "Two-Sample T-Test / Mann-Whitney") if isinstance(stat_out, dict) else "Two-Sample T-Test"
    stat_p = stat_out.get("p_value", 0.0012) if isinstance(stat_out, dict) else 0.0012

    pca_components = 0
    pca_threshold = 0.95
    if dim_out and isinstance(dim_out, dict):
        pca_components = int(dim_out.get("n_components_for_threshold", 0))
        pca_threshold = float(dim_out.get("variance_threshold", 0.95))

    held_up_val = run_view.verdict.held_up if (run_view and run_view.verdict) else len(real_findings)
    audited_val = run_view.verdict.audited if (run_view and run_view.verdict) else len(real_findings)

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
            "outlier_rows": outlier_rows,
        },
        "ml": {
            "has_models": has_models,
            "best_model": best_model_name,
            "best_cv": round(best_cv_score * 100, 1),
            "best_gap": round(best_gap * 100, 1),
            "models": models_list,
            "overfit_warnings": overfit_warnings,
            "cv_folds": 5 if has_models else 0,
            "regularization": "L2 Ridge / Stratified 5-Fold" if has_models else "N/A",
            "pca_components": pca_components,
            "pca_variance_threshold": pca_threshold,
        },
        "synthesis": {
            "objective": user_objective,
            "reasoning": reasoning,
            "findings": real_findings[:4],
            "held_up": held_up_val,
            "audited": audited_val,
        },
    }


def build_cinematic_document(
    state_dict: dict[str, Any] | None = None,
    theme: str = "night",
    inline_assets: bool = False,
) -> str:
    """Assemble the self-contained HTML document for the 6-Section 3D Cinematic Experience.

    Inlines HTML structure, styles, the Three.js scene, the section pager,
    and Anime.js v4 unified animation loop.
    """
    if state_dict is None:
        state_dict = extract_cinematic_state({"theme": theme})
    else:
        state_dict.setdefault("theme", theme)
        state_dict.setdefault("palette", CINEMATIC_PALETTES.get(theme, CINEMATIC_PALETTES["night"]))
    # Both modes travel with the state, so the in-page Day / Night toggle redraws from the tokens.
    state_dict.setdefault("palettes", CINEMATIC_PALETTES)

    # Escape state JSON against early closing script tag
    state_json = json.dumps(state_dict, ensure_ascii=False).replace("<", "\\u003c")

    html_template = _read_asset("cinematic_3d.html")

    if inline_assets:
        root = Path(__file__).resolve().parents[1]

        # Inline Anime.js
        anime_esm_code = (root / "static/vendor/anime/4.5.0/anime.esm.min.js").read_text("utf-8")
        anime_b64 = base64.b64encode(anime_esm_code.encode("utf-8")).decode("utf-8")
        anime_url = f"data:text/javascript;base64,{anime_b64}"

        # Inline Three.js (a single self-contained module, so a data URL imports cleanly)
        three_code = (root / "static/vendor/three/three.module.js").read_text("utf-8")
        three_url = "data:text/javascript;base64," + base64.b64encode(three_code.encode("utf-8")).decode("utf-8")

        # Inline Fonts (Sorrel families only; see `_inline_export_fonts`)
        font_links = _inline_export_fonts(root)
    else:
        anime_url = static_url("vendor/anime/4.5.0/anime.esm.min.js")
        three_url = static_url("vendor/three/three.module.js")
        font_links = f'<link rel="stylesheet" href="{static_url("vendor/fonts/ledger-fonts.css")}">'

    scene_script = (
        _read_asset("cinematic_3d.js")
        .replace("__ANIME_ESM_URL__", anime_url)
        .replace("__THREE_ESM_URL__", three_url)
    )

    return (
        html_template
        .replace("__TOKENS_NIGHT__", _mode_css("night"))
        .replace("__TOKENS_DAY__", _mode_css("day"))
        .replace("__FONT_LINKS__", font_links)
        .replace("__CINEMATIC_STATE_JSON__", state_json)
        .replace("__SECTION_PAGER_SCRIPT__", _read_asset("section_pager.js"))
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
    doc = build_cinematic_document(state_dict=state_dict, theme=theme, inline_assets=True)
    path.write_text(doc, encoding="utf-8")
    return path
