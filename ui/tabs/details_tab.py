"""
Details Tab (Statistical & ML Lab, Ingestion, Profiling, Agent Run Trace, Governance).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.core.plain_language import format_p
from src.core.run_view import RunView
from ui.components.cards import (
    find_tool,
    render_agent_grid,
    render_governance,
    render_handoff_stream,
    render_other_findings,
    safe_df,
)
from ui.components.how_we_got_here import build_how_html


def render_details_tab(
    report: dict[str, Any],
    tool_results: list[dict[str, Any]],
    prof: dict[str, Any] | None,
    run_view: RunView | None = None,
) -> None:
    """Render Tier 4: Statistical & ML Lab, Detailed Audit & Inspection."""
    clean_out = find_tool(tool_results, "clean_data")
    outlier_out = find_tool(tool_results, "detect_outliers")
    stat_out = find_tool(tool_results, "select_statistical_test")
    train_out = find_tool(tool_results, "train_model")
    eval_out = find_tool(tool_results, "evaluate_model")

    if run_view is not None:
        how_html = build_how_html(run_view.how)
        if how_html:
            st.markdown(how_html, unsafe_allow_html=True)

    st.markdown("## Case Log")
    st.caption("A sequential record of technical checks performed on this dataset.")

    profile_status = st.session_state.get("profile_status")
    if isinstance(profile_status, str) and profile_status.startswith("failed"):
        st.warning(
            "Running in degraded mode — profiling failed, so dataset-nature "
            f"tools (time-series, text, geo...) were unavailable. Reason: {profile_status[8:]}"
        )

    read_report = st.session_state.get("read_report")
    coercions = st.session_state.get("coercions")
    if read_report or coercions:
        with st.container(border=True):
            st.markdown("### 1. Data Ingestion & Repair")
            if read_report:
                rr_bits = [
                    f"format `{read_report.get('format')}`",
                    f"encoding `{read_report.get('encoding')}`",
                ]
                if not read_report.get("encoding_confident", True):
                    rr_bits[-1] += " (guessed)"
                if read_report.get("delimiter"):
                    rr_bits.append(
                        f"delimiter `{read_report.get('delimiter')!r}`"
                        + ("" if read_report.get("delimiter_sniffed") else " (from extension)")
                    )
                st.caption("Detected at read time: " + ", ".join(rr_bits) + ".")
                for note in read_report.get("notes", []):
                    st.caption(f"⚠ {note}")
            if coercions:
                st.caption(f"{len(coercions)} column(s) repaired:")
                st.dataframe(
                    safe_df(
                        pd.DataFrame([
                            {
                                "Column": c["column"],
                                "Rule": c["rule"],
                                "Converted": c["n_converted"],
                                "Failed": c["n_failed"],
                            }
                            for c in coercions
                        ])
                    ),
                    width="stretch",
                )

    prof = prof or st.session_state.get("profile")
    if prof:
        with st.container(border=True):
            st.markdown("### 2. Profiling")
            prows = [
                {
                    "Column": c.get("name"),
                    "Kind": c.get("kind"),
                    "Dtype": c.get("dtype"),
                    "Missing %": c.get("missing_pct"),
                    "Unique": c.get("nunique"),
                    "Flags": ", ".join(c.get("flags", [])),
                }
                for c in prof.get("columns", [])
            ]
            st.dataframe(safe_df(pd.DataFrame(prows)), width="stretch")

    if clean_out:
        with st.container(border=True):
            st.markdown("### 3. Data Cleaning")
            c1, c2, c3 = st.columns(3)
            c1.metric("Strategy used", clean_out.get("strategy_used", "—"))
            c2.metric("Missing values before", clean_out.get("missing_before", "—"))
            c3.metric("Missing values after", clean_out.get("missing_after", "—"))

    if outlier_out:
        with st.container(border=True):
            st.markdown("### 4. Outlier Detection")
            c1, c2 = st.columns(2)
            c1.metric("Unusual rows found", outlier_out.get("total_outliers", "—"))
            c2.metric("Share of all rows", f"{outlier_out.get('outlier_percentage','—')}%")
            pc = outlier_out.get("per_column_outliers", {})
            if pc:
                pc_df = pd.DataFrame(
                    [(c, v) for c, v in pc.items() if v > 0],
                    columns=["Column", "Unusual values"],
                ).sort_values("Unusual values", ascending=False)
                if not pc_df.empty:
                    st.dataframe(safe_df(pc_df), width="stretch")

    if stat_out:
        with st.container(border=True):
            st.markdown("### 5. Statistical Tests")
            c1, c2, c3 = st.columns(3)
            c1.metric("Test used", stat_out.get("test_name", "—"))
            c2.metric("p-value", format_p(stat_out.get("p_value")))
            c3.metric("Likely real, not chance", "Yes" if stat_out.get("significant") else "No")
            st.info(stat_out.get("interpretation", "No interpretation recorded."))

    if train_out:
        with st.container(border=True):
            st.markdown("### 6. Model Training & Evaluation")
            mt2 = train_out.get("models_trained", {})
            best2 = train_out.get("best_model", "")
            task2 = train_out.get("task_type", "classification")
            pk2 = "accuracy" if task2 == "classification" else "r2"
            sk2 = "f1_score" if task2 == "classification" else "rmse"

            st.markdown(
                f"**Type of problem:** `{task2}` &nbsp;|&nbsp; **Best model:** `{best2}` "
                f"&nbsp;|&nbsp; **Times each model was tested:** `{train_out.get('n_cv_folds', 5)}` "
                f"&nbsp;|&nbsp; **Held back for testing:** `{int(train_out.get('test_size', 0.2)*100)}%`"
            )
            rows = []
            for nm, m in mt2.items():
                tr2 = m.get("train_metrics", {})
                te2 = m.get("test_metrics", {})
                g = m.get("train_test_gap")
                rows.append({
                    "Model": f"{nm} (best)" if nm == best2 else nm,
                    f"Train {pk2}": f"{tr2.get(pk2,0)*100:.1f}%",
                    f"Test {pk2}": f"{te2.get(pk2,0)*100:.1f}%",
                    "CV mean": f"{m.get('cv_mean',0)*100:.1f}%",
                    "CV std": f"±{m.get('cv_std',0)*100:.1f}%",
                    "Gap": (
                        f"{g*100:.1f}%" + (" [RISK]" if g and g >= 0.10 else "")
                        if g is not None
                        else "—"
                    ),
                    sk2.replace("_", " "): (
                        f"{te2.get(sk2,0)*100:.1f}%"
                        if sk2 != "rmse"
                        else f"{te2.get(sk2,0):.4f}"
                    ),
                })
            st.dataframe(safe_df(pd.DataFrame(rows)), width="stretch")

            if eval_out:
                st.markdown("#### Accuracy by Category")
                cr = eval_out.get("classification_report", {})
                if cr:
                    cr_rows = [
                        {
                            "Class": lbl,
                            "Precision": f"{v.get('precision',0):.3f}",
                            "Recall": f"{v.get('recall',0):.3f}",
                            "F1": f"{v.get('f1-score',0):.3f}",
                            "Support": int(v.get("support", 0)),
                        }
                        for lbl, v in cr.items()
                        if isinstance(v, dict)
                    ]
                    st.dataframe(safe_df(pd.DataFrame(cr_rows)), width="stretch")

    with st.container(border=True):
        st.markdown("### 7. Domain-Specific Analyses")
        st.caption("Segmentation, trends, text, and geography — run when your data called for them.")
        render_other_findings(tool_results)

    with st.expander("8. Run Trace — how the agents worked through this", expanded=False):
        st.caption("Each helper does one job, and hands off to the next.")
        st.markdown(
            render_agent_grid(st.session_state.get("stage_log", []), tool_results, report),
            unsafe_allow_html=True,
        )

        st.markdown("#### What Was Said, Step by Step")
        st.caption("A record of what each helper passed to the next, and when.")
        st.markdown(
            render_handoff_stream(st.session_state.get("progress_lines", []), tool_results),
            unsafe_allow_html=True,
        )

        sub_results = report.get("rlm_sub_results")
        if sub_results:
            st.markdown("#### How the Tricky Parts Were Split Up")
            st.caption("Big questions got broken into smaller ones so nothing got lost.")
            for s_idx, sub in enumerate(sub_results, 1):
                with st.expander(
                    f"Part {s_idx:02d}: {sub.get('task_name', 'Smaller Question')}",
                    expanded=False,
                ):
                    st.json(sub)

    gov = report.get("governance")
    if gov:
        with st.container(border=True):
            st.markdown("### 9. Governance & audit")
            st.caption("What AI-written code ran, where it ran, and what was refused.")
            render_governance(gov)
