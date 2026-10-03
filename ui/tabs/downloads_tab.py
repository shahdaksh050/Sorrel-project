"""
Downloads tab: an artifact shelf. Each file the run produced has a type tag, a plain
purpose, a size and an availability state, with the real download button beside it.

The files and buttons are exactly the ones this tab has always offered (report, data,
models and the optional 3D presentation); only the presentation around them is new.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import streamlit as st

from ui.components.cards import artifact_info_html, md_text


def _shelf_group(title: str) -> None:
    """A group label for the shelf, kept as a level-3 heading for assistive technology."""
    st.markdown(
        f'<div class="shelf-group" role="heading" aria-level="3">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


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

        _shelf_group("Report")
        # HTML report
        with st.container(key="artifact_html"):
            info, action = st.columns([0.72, 0.28], vertical_alignment="center")
            if html_file.exists():
                html_bytes = html_file.read_bytes()
                info.markdown(
                    artifact_info_html(
                        "HTML", "report.html",
                        "A clean reading view you can open in any browser or print.",
                        len(html_bytes), "Ready",
                    ),
                    unsafe_allow_html=True,
                )
                with action:
                    st.download_button(
                        "Report as a web page (HTML)", html_bytes, "report.html",
                        mime="text/html", key="dl_html_vault", width="stretch",
                    )
            else:
                info.markdown(
                    artifact_info_html(
                        "HTML", "Web page report",
                        "A clean reading view you can open in any browser or print.",
                        None, "Not available for this run",
                    ),
                    unsafe_allow_html=True,
                )
        # Markdown report
        with st.container(key="artifact_md"):
            info, action = st.columns([0.72, 0.28], vertical_alignment="center")
            if mds:
                md_bytes = mds[0].read_bytes()
                info.markdown(
                    artifact_info_html(
                        "MD", mds[0].name, "The same report as plain text.", len(md_bytes), "Ready",
                    ),
                    unsafe_allow_html=True,
                )
                with action:
                    st.download_button(
                        "Report as text (Markdown)", md_bytes, mds[0].name,
                        mime="text/markdown", key="dl_md_vault", width="stretch",
                    )
            else:
                info.markdown(
                    artifact_info_html(
                        "MD", "Markdown report", "The same report as plain text.", None, "Not available for this run",
                    ),
                    unsafe_allow_html=True,
                )

        _shelf_group("Data")
        json_text = json.dumps(report, indent=2, default=str)
        with st.container(key="artifact_json"):
            info, action = st.columns([0.72, 0.28], vertical_alignment="center")
            info.markdown(
                artifact_info_html(
                    "JSON", "final_report.json",
                    "Every finding, number and check from this run, for use in other tools.",
                    len(json_text.encode("utf-8")), "Ready",
                ),
                unsafe_allow_html=True,
            )
            with action:
                st.download_button(
                    "All findings as data (JSON)", json_text, "final_report.json",
                    mime="application/json", key="dl_json_vault", width="stretch",
                )

        mdir = out / "models"
        if mdir.exists() and any(mdir.iterdir()):
            _shelf_group("Models")
            for n, mdl_f in enumerate(sorted(mdir.iterdir())):
                mdl_bytes = mdl_f.read_bytes()
                with st.container(key=f"artifact_model_{n}"):
                    info, action = st.columns([0.72, 0.28], vertical_alignment="center")
                    info.markdown(
                        artifact_info_html(
                            mdl_f.suffix.lstrip(".").upper() or "FILE", mdl_f.name,
                            "A saved model file, signed so this app can load it again.",
                            len(mdl_bytes), "Ready",
                        ),
                        unsafe_allow_html=True,
                    )
                    with action:
                        st.download_button(
                            f"Trained model: {mdl_f.name}", mdl_bytes, mdl_f.name,
                            mime="application/octet-stream", key=f"dlm_vault_{mdl_f.name}",
                            width="stretch",
                        )

        # The 3D presentation is large and optional, so it is built when asked for, not on every render.
        _shelf_group("Optional: 3D presentation")
        with st.container(key="artifact_3d"):
            info_slot, action = st.columns([0.72, 0.28], vertical_alignment="center")
            pres_bytes: bytes | None = None
            with action:
                if st.button("Prepare the 3D presentation", key="prep_3d_presentation", width="stretch"):
                    from ui.cinematic_3d import build_cinematic_document, extract_cinematic_state

                    st.session_state["_cinema_pres_html"] = build_cinematic_document(
                        extract_cinematic_state(st.session_state),
                        theme=st.session_state.get("theme", "night"),
                        inline_assets=True,
                    )
                pres_html = st.session_state.get("_cinema_pres_html")
                if pres_html:
                    payload = pres_html.encode("utf-8")
                    pres_bytes = payload
                    st.download_button(
                        "Download the 3D presentation (HTML)",
                        payload,
                        "dsa_agent_3d_presentation.html",
                        mime="text/html",
                        key="dl_3d_cinema_standalone",
                        width="stretch",
                    )
            # Filled after the button, so the state shown is the one this run just produced.
            info_slot.markdown(
                artifact_info_html(
                    "HTML", "3D presentation",
                    "A self-contained web page that replays the run in 3D. It works offline.",
                    len(pres_bytes) if pres_bytes is not None else None,
                    "Ready" if pres_bytes is not None else "Not prepared yet",
                ),
                unsafe_allow_html=True,
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
