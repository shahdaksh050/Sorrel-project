"""
Downloads Tab (Artifact Vault, Reports, 3D Presentation, Models, Logs).
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
    """Render Tier 6: Artifact Vault & Exports."""
    st.markdown("## Downloads")
    st.caption("Everything from this run, ready to keep or share.")

    if tmp_dir:
        out = Path(tmp_dir) / "output"
        rdir = out / "reports"

        st.markdown(
            "<style>.dossier-card { background: var(--sheet-alt); border: 1px solid var(--rule); "
            "border-radius: var(--radius); padding: 1rem; margin-bottom: 1rem; display: flex; "
            "flex-direction: column; gap: 0.5rem; transition: transform 0.15s, box-shadow 0.15s; } "
            ".dossier-card:hover { transform: translateY(-2px); box-shadow: var(--lift-sm); }</style>",
            unsafe_allow_html=True,
        )

        dl_cols = st.columns(3)
        with dl_cols[0]:
            st.markdown("### Presentations")
            with st.container(border=True):
                st.markdown("**3D Cinematic Journey**")
                st.caption("Standalone HTML with 3D models and animations.")
                from ui.cinematic_3d import build_cinematic_document, extract_cinematic_state

                cinema_pres_html = build_cinematic_document(
                    extract_cinematic_state(st.session_state),
                    theme=st.session_state.get("theme", "night"),
                    inline_assets=True,
                )
                st.download_button(
                    "🎬 Download HTML",
                    cinema_pres_html.encode("utf-8"),
                    "dsa_agent_3d_presentation.html",
                    mime="text/html",
                    key="dl_3d_cinema_standalone",
                    type="primary",
                    use_container_width=True,
                )
            with st.container(border=True):
                st.markdown("**Shareable Report**")
                st.caption("A clean web-page reading view of the report.")
                html_file = rdir / "report.html"
                if html_file.exists():
                    st.download_button(
                        "📄 Download HTML",
                        html_file.read_bytes(),
                        "report.html",
                        mime="text/html",
                        key="dl_html_vault",
                        use_container_width=True,
                    )

        mds = sorted(rdir.glob("*.md")) if rdir.exists() else []

        with dl_cols[1]:
            st.markdown("### Data & Logs")
            with st.container(border=True):
                st.markdown("**Markdown Report**")
                st.caption("The core report in plain text markdown.")
                if mds:
                    st.download_button(
                        "📝 Download Markdown",
                        mds[0].read_bytes(),
                        mds[0].name,
                        mime="text/markdown",
                        key="dl_md_vault",
                        use_container_width=True,
                    )
            with st.container(border=True):
                st.markdown("**Agent Memory Vault**")
                st.caption("The complete raw data of everything the agents found.")
                st.download_button(
                    "💾 Download JSON",
                    json.dumps(report, indent=2, default=str),
                    "final_report.json",
                    mime="application/json",
                    key="dl_json_vault",
                    use_container_width=True,
                )

        with dl_cols[2]:
            st.markdown("### Trained Models")
            mdir = out / "models"
            if mdir.exists() and any(mdir.iterdir()):
                for mdl_f in sorted(mdir.iterdir()):
                    with st.container(border=True):
                        st.markdown(f"**{mdl_f.name}**")
                        st.caption("Pickled model object ready for predictions.")
                        st.download_button(
                            "📦 Download",
                            mdl_f.read_bytes(),
                            mdl_f.name,
                            mime="application/octet-stream",
                            key=f"dlm_vault_{mdl_f.name}",
                            use_container_width=True,
                        )
            else:
                st.caption("No predictive models were saved for this run.")

        st.divider()
        st.markdown("### Report Preview")
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

        with st.expander("Full Technical Log", expanded=False):
            st.json(tool_results)
