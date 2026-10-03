"""
Downloads tab: report, data, models and the optional 3D presentation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from ui.components.cards import md_text


def render_downloads_tab(
    report: dict[str, Any],
    tool_results: list[dict[str, Any]],
    tmp_dir: str,
) -> None:
    """Downloads: a hand-off surface grouped by what the file is."""
    st.markdown("## Downloads")
    st.caption("Everything from this run, ready to keep or share.")

    if tmp_dir:
        out = Path(tmp_dir) / "output"
        rdir = out / "reports"
        mds = sorted(rdir.glob("*.md")) if rdir.exists() else []
        html_file = rdir / "report.html"

        st.markdown("### Report")
        left, right = st.columns(2)
        with left:
            if html_file.exists():
                st.download_button(
                    "Report as a web page (HTML)", html_file.read_bytes(), "report.html",
                    mime="text/html", key="dl_html_vault", width="stretch",
                )
                st.caption("A clean reading view you can open in any browser or print.")
        with right:
            if mds:
                st.download_button(
                    "Report as text (Markdown)", mds[0].read_bytes(), mds[0].name,
                    mime="text/markdown", key="dl_md_vault", width="stretch",
                )
                st.caption("The same report as plain text.")

        st.markdown("### Data")
        st.download_button(
            "All findings as data (JSON)", json.dumps(report, indent=2, default=str), "final_report.json",
            mime="application/json", key="dl_json_vault",
        )
        st.caption("Every finding, number and check from this run, for use in other tools.")

        mdir = out / "models"
        if mdir.exists() and any(mdir.iterdir()):
            st.markdown("### Models")
            for mdl_f in sorted(mdir.iterdir()):
                st.download_button(
                    f"Trained model: {mdl_f.name}", mdl_f.read_bytes(), mdl_f.name,
                    mime="application/octet-stream", key=f"dlm_vault_{mdl_f.name}",
                )
            st.caption("Saved model files, signed so this app can load them again.")

        # The 3D presentation is large and optional, so it is built when asked for, not on every render.
        st.markdown("### Optional: 3D presentation")
        st.caption("A self-contained web page that replays the run in 3D. It works offline.")
        if st.button("Prepare the 3D presentation", key="prep_3d_presentation"):
            from ui.cinematic_3d import build_cinematic_document, extract_cinematic_state

            st.session_state["_cinema_pres_html"] = build_cinematic_document(
                extract_cinematic_state(st.session_state),
                theme=st.session_state.get("theme", "night"),
                inline_assets=True,
            )
        if st.session_state.get("_cinema_pres_html"):
            st.download_button(
                "Download the 3D presentation (HTML)",
                st.session_state["_cinema_pres_html"].encode("utf-8"),
                "dsa_agent_3d_presentation.html",
                mime="text/html",
                key="dl_3d_cinema_standalone",
            )

        st.divider()
        st.markdown("### Report preview")
        if mds:
            st.markdown(mds[0].read_text(encoding="utf-8"))
        else:
            # No Markdown file on disk. Two different reasons look identical
            # to a user staring at a blank Downloads tab, so tell them which
            # one happened instead of dumping the raw final_result dict:
            # (1) GenerateReportTool actually failed — surface its real
            # error_message, recorded on the tool_results list; (2) it never
            # ran / hasn't reached this point yet — render a readable preview
            # from `report` itself instead of a raw JSON tree either way.
            failed = next(
                (
                    r for r in tool_results
                    if r.get("tool_name") == "generate_report" and r.get("status") != "success"
                ),
                None,
            )
            if failed:
                st.error(
                    "Markdown report generation failed: "
                    + str(failed.get("error") or "no error message recorded.")
                )
            else:
                st.caption("No Markdown file was found on disk for this run — showing the raw result instead.")

            reasoning = report.get("reasoning")
            if reasoning:
                st.markdown(md_text(str(reasoning)))
            insights = report.get("insights") or []
            if insights:
                st.markdown("**Insights**")
                for item in insights:
                    text = item.get("headline", item) if isinstance(item, dict) else item
                    st.markdown(f"- {md_text(str(text))}")
            recs = report.get("recommendations") or []
            if recs:
                st.markdown("**Recommendations**")
                for item in recs:
                    st.markdown(f"- {md_text(str(item))}")
            with st.expander("Raw result (JSON)", expanded=False):
                st.json(report)

        with st.expander("Full technical log", expanded=False):
            st.json(tool_results)
