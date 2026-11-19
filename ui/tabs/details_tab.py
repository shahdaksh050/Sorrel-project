"""
Details tab: how the answer was reached, in increasing technical depth (data, methods, governance, raw record).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.core.plain_language import format_p
from src.core.run_view import RunView
from ui.components.cards import (
    audit_head_html,
    find_tool,
    render_agent_grid,
    render_governance,
    render_handoff_stream,
    render_other_findings,
    safe_df,
    tab_head,
)
from ui.components.how_we_got_here import build_how_html


def render_details_tab(
    report: dict[str, Any],
    tool_results: list[dict[str, Any]],
    prof: dict[str, Any] | None,
    run_view: RunView | None = None,
) -> None:
    """Render Tier 4: Statistical & ML Lab, Detailed audit and inspection."""
    clean_out = find_tool(tool_results, "clean_data")
    outlier_out = find_tool(tool_results, "detect_outliers")
    stat_out = find_tool(tool_results, "select_statistical_test")
    train_out = find_tool(tool_results, "train_model")
    eval_out = find_tool(tool_results, "evaluate_model")

    tab_head("Details", "Your data and the", "checks")
    if run_view is not None:
        how_html = build_how_html(run_view.how)
        if how_html:
            st.markdown(how_html, unsafe_allow_html=True)

    st.caption("What was done to the file and which analyses ran, from plain summary to technical detail.")

    profile_status = st.session_state.get("profile_status")
    if isinstance(profile_status, str) and profile_status.startswith("failed"):
        st.warning(
            "Running in degraded mode: profiling failed, so dataset-nature "
            f"tools (time-series, text, geo...) were unavailable. Reason: {profile_status[8:]}"
        )

    # The analyses that ran for this data come first: they are what most readers came to see, ahead
    # of how the file was read and repaired.
    with st.container(border=True, key="audit_other"):
        st.markdown(
            audit_head_html(
                "Analyses specific to your data",
                "Segmentation, trends, text and geography, run when your data called for them.",
            ),
            unsafe_allow_html=True,
        )
        render_other_findings(tool_results)

    read_report = st.session_state.get("read_report")
    coercions = st.session_state.get("coercions")
    if read_report or coercions:
        with st.container(border=True, key="audit_read"):
            st.markdown(audit_head_html("How the file was read and repaired"), unsafe_allow_html=True)
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
        with st.container(border=True, key="audit_profile"):
            st.markdown(audit_head_html("Column profile"), unsafe_allow_html=True)
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
        with st.container(border=True, key="audit_cleaning"):
            st.markdown(audit_head_html("Cleaning"), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Strategy used", clean_out.get("strategy_used", "—"))
            c2.metric("Missing values before", clean_out.get("missing_before", "—"))
            c3.metric("Missing values after", clean_out.get("missing_after", "—"))

    if outlier_out:
        with st.container(border=True, key="audit_outliers"):
            st.markdown(audit_head_html("Unusual rows"), unsafe_allow_html=True)
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
        with st.container(border=True, key="audit_stat"):
            st.markdown(audit_head_html("Statistical test"), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Test used", stat_out.get("test_name", "—"))
            c2.metric("p-value", format_p(stat_out.get("p_value")))
            significance = stat_out.get("significant")
            c3.metric("Below test threshold", {True: "Yes", False: "No"}.get(significance, "Unavailable"))
            st.info(stat_out.get("interpretation", "No interpretation recorded."))

    cluster_tool = find_tool(tool_results, "cluster_data")
    is_clustering = (train_out and train_out.get("task_type") == "clustering") or bool(cluster_tool)
    if is_clustering:
        with st.container(border=True, key="audit_clustering"):
            st.markdown(
                audit_head_html("Clustering diagnostics", "Unsupervised fit on analysis frame (no supervised holdout)."),
                unsafe_allow_html=True,
            )
            diag = (train_out.get("clustering_diagnostics") if train_out else None) or (cluster_tool or {})
            fit_pop = (
                diag.get("fit_population")
                or (train_out.get("train_samples") if train_out else None)
                or diag.get("n_samples")
                or "Unavailable"
            )
            sil = diag.get("silhouette_score")
            sil_str = f"{sil:.3f}" if isinstance(sil, (int, float)) else "Unavailable"
            inertia = diag.get("inertia")
            inertia_str = f"{inertia:,.1f}" if isinstance(inertia, (int, float)) else "Unavailable"
            best_mod = (train_out.get("best_model") if train_out else None) or diag.get("best_model") or "Clustering"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fit population", f"{fit_pop:,}" if isinstance(fit_pop, int) else str(fit_pop))
            c2.metric("Best model", str(best_mod))
            c3.metric("Silhouette score", sil_str)
            c4.metric("Inertia", inertia_str)

            c_sizes = diag.get("cluster_sizes") or {}
            c_shares = diag.get("cluster_shares") or {}
            if c_sizes:
                st.markdown("##### Cluster size distribution")
                dist_rows = []
                for cid, sz in c_sizes.items():
                    share = c_shares.get(cid)
                    share_str = f"{share:.1%}" if isinstance(share, (int, float)) else "Unavailable"
                    dist_rows.append({"Cluster": str(cid), "Size (n)": sz, "Share": share_str})
                st.dataframe(safe_df(pd.DataFrame(dist_rows)), width="stretch")
    elif train_out:
        with st.container(border=True, key="audit_models"):
            st.markdown(audit_head_html("Models"), unsafe_allow_html=True)
            task = train_out.get("task_type")
            metric = "f1_score" if task == "classification" else "r2"
            label = "weighted F1" if task == "classification" else "R²"
            val_scheme = train_out.get("cv_scheme") or "Unavailable"
            baseline_val = train_out.get("baseline_cv_mean")
            baseline_str = f"{baseline_val:.3f}" if isinstance(baseline_val, (int, float)) else "Unavailable"
            lift_val = train_out.get("lift_over_baseline")
            lift_str = f"{lift_val:+.3f}" if isinstance(lift_val, (int, float)) else "Unavailable"

            c1, c2, c3 = st.columns(3)
            c1.metric("Validation design", str(val_scheme))
            c2.metric("Baseline score", baseline_str)
            c3.metric("Model lift vs baseline", lift_str)

            rows = []
            for name, model in train_out.get("models_trained", {}).items():
                train_m = (model.get("train_metrics") or {}).get(metric)
                test_m = (model.get("test_metrics") or {}).get(metric)
                cv_m = model.get("cv_mean")
                cv_s = model.get("cv_std")
                gap_m = model.get("train_test_gap")
                rows.append({
                    "Model": name,
                    "Selected": name == train_out.get("best_model"),
                    f"Train {label}": f"{train_m:.3f}" if isinstance(train_m, (int, float)) else "Unavailable",
                    f"Test {label}": f"{test_m:.3f}" if isinstance(test_m, (int, float)) else "Unavailable",
                    f"CV mean {label}": f"{cv_m:.3f}" if isinstance(cv_m, (int, float)) else "Unavailable",
                    "CV std": f"{cv_s:.3f}" if isinstance(cv_s, (int, float)) else "Unavailable",
                    "Train–test gap": f"{gap_m:.3f}" if isinstance(gap_m, (int, float)) else "Unavailable",
                })
            if rows:
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

    coverage = report.get("coverage") or (run_view.coverage if run_view else {})
    if isinstance(coverage, dict) and coverage.get("total"):
        with st.container(border=True, key="audit_questions"):
            st.markdown(
                audit_head_html(
                    "Question Coverage & Open Questions",
                    "Audit of questions addressed by executed analysis steps vs. remaining open.",
                ),
                unsafe_allow_html=True,
            )
            ans = coverage.get("answered", 0)
            tot = coverage.get("total", 0)
            st.write(f"**Business questions answered:** {ans} of {tot}")
            unanswered = coverage.get("unanswered") or []
            if unanswered:
                st.markdown(f"##### Open business questions ({len(unanswered)})")
                q_rows = []
                for u in unanswered:
                    q_rows.append({
                        "Question": u.get("text", ""),
                        "Status": "Blocked" if u.get("blocked") else "Open",
                        "Reason": u.get("reason") or u.get("blocked_reason") or "Not answered by executed steps",
                    })
                st.dataframe(safe_df(pd.DataFrame(q_rows)), width="stretch")

    gov = report.get("governance")
    if gov:
        with st.container(border=True, key="audit_safety"):
            st.markdown(
                audit_head_html("Safety and audit", "What AI-written code ran, where it ran, and what was refused."),
                unsafe_allow_html=True,
            )
            render_governance(gov)

    with st.expander("Step-by-step record (technical)", expanded=False):
        st.caption("Each step does one job and hands off to the next.")
        st.markdown(
            render_agent_grid(st.session_state.get("stage_log", []), tool_results, report),
            unsafe_allow_html=True,
        )

        st.markdown("#### What each step passed on")
        st.caption("A record of what each step passed to the next, and when.")
        st.markdown(
            render_handoff_stream(st.session_state.get("progress_lines", []), tool_results),
            unsafe_allow_html=True,
        )
