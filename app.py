"""
Streamlit UI — Agentic Data Analysis System.

Run:
    streamlit run app.py

Key design points
-----------------
* The analysis runs on a worker thread (ui/run.py). The script never blocks:
  a polling fragment shows progress and Stop and folds the finished result into
  session state. The worker never calls st.* and never reads session state.
* Dataset preview is saved to session_state on file upload and rendered from
  there — no dependency on sidebar scope surviving a rerun.
* OpenRouter support added (any model string, OpenAI-compatible endpoint).
"""
from __future__ import annotations

import html
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

from src.core.io import read_any_bytes
from src.core.run_config import RunConfig
from src.core.security import ALLOWED_EXTENSIONS

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

#: Operator switch for a shared deployment. When set, visitors cannot (a) fall
#: back to a server-side API key, (b) point the server at a URL they typed, or
#: (c) change the safety limits, which come from the server's environment.
_HOSTED = os.getenv("DSA_HOSTED", "false").strip().lower() in ("1", "true", "yes")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Agentic Data Analysis",
    page_icon="🧾",
    layout="wide",
    # "auto": open on a desktop, collapsed on a phone, where an open sidebar covers the page.
    initial_sidebar_state="auto",
)

# ── Landing Page (Phase 2 Narrative) ─────────────────────────────────────────
if not st.session_state.get("entered", False):
    from ui.landing import show_landing_page

    # We overlay a hidden native Streamlit button. The iframe JS will click this directly!
    st.markdown("""
        <style>
            /* Hide the native button so it doesn't float over the 3D scene */
            div.stButton > button {
                opacity: 0;
                position: fixed;
                top: -9999px;
            }
        </style>
    """, unsafe_allow_html=True)

    if st.button("HIDDEN_ENTER", key="hidden_enter") or show_landing_page():
        st.session_state["entered"] = True
        st.rerun()

    st.stop()


# ── Rich stub ─────────────────────────────────────────────────────────────────
def _stub_rich() -> None:
    """Silence rich so src/ imports work without the package installed."""
    for mod_name in [
        "rich", "rich.console", "rich.panel",
        "rich.table", "rich.tree", "rich.progress",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    class _C:
        def print(self, *a: Any, **k: Any) -> None: pass
    class _P:
        def __init__(self, *a: Any, **k: Any): pass
    class _T:
        def __init__(self, *a: Any, **k: Any): pass
        def add_column(self, *a: Any, **k: Any) -> None: pass
        def add_row(self, *a: Any, **k: Any) -> None: pass
    class _Tr:
        def __init__(self, *a: Any, **k: Any): pass
        def add(self, *a: Any, **k: Any) -> _Tr: return self
    class _Pr:
        def __init__(self, *a: Any, **k: Any): pass
        def __enter__(self) -> _Pr: return self
        def __exit__(self, *a: Any) -> None: pass
        def add_task(self, *a: Any, **k: Any) -> int: return 0
        def update(self, *a: Any, **k: Any) -> None: pass
    class _Sp:
        def __init__(self, *a: Any, **k: Any): pass
    class _Tx:
        def __init__(self, *a: Any, **k: Any): pass

    # Assign to the sub-module entries in sys.modules directly —
    # never traverse sys.modules["rich"].tree as an attribute chain.
    sys.modules["rich"].Console = _C                  # type: ignore[attr-defined]
    sys.modules["rich.console"].Console = _C          # type: ignore[attr-defined]
    sys.modules["rich.panel"].Panel = _P              # type: ignore[attr-defined]
    sys.modules["rich.tree"].Tree = _Tr               # type: ignore[attr-defined]
    sys.modules["rich.table"].Table = _T              # type: ignore[attr-defined]
    sys.modules["rich.progress"].Progress = _Pr       # type: ignore[attr-defined]
    sys.modules["rich.progress"].SpinnerColumn = _Sp  # type: ignore[attr-defined]
    sys.modules["rich.progress"].TextColumn = _Tx     # type: ignore[attr-defined]


_stub_rich()


from ui.components.cards import (
    STAGE_DEFS,
)
from ui.components.cards import (
    gauge as _gauge,
)
from ui.components.cards import (
    get_vega_config as _get_vega_config,
)
from ui.components.cards import (
    render_agent_grid as _render_agent_grid,
)
from ui.components.cards import (
    render_datum as _datum,
)
from ui.components.cards import (
    render_steps_list as _render_steps_list,
)
from ui.components.cards import (
    safe_df as _safe_df,
)
from ui.components.cards import (
    section as _section,
)
from ui.components.provisional import build_provisional_html
from ui.run import RUN_DIR_PREFIX, ActiveRun, RunSpec, remove_run_dir
from ui.styles import inject_theme_css as _inject_theme_css
from ui.tabs import (
    render_answers_tab,
    render_charts_tab,
    render_details_tab,
    render_downloads_tab,
)

# Every session opens in Day mode. The sidebar toggle then owns the theme for
# the rest of the session. The browser's own colour scheme is deliberately not
# read: Streamlit's native widgets follow a fixed theme (.streamlit/config.toml),
# so a dark browser must not flip only the custom CSS and leave the widgets behind.
_DEFAULT_THEME = "day"

# ── Session-state initialisation ──────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "theme":          _DEFAULT_THEME,
    "preview_df":     None,   # pd.DataFrame
    "preview_name":   "",     # sanitised filename (safe for filesystem)
    "orig_name":      "",     # exact name as uploaded (change detection)
    "preview_bytes":  None,   # raw bytes
    "stage_log":      [],
    "analysis_done":  False,
    "analysis_error": None,
    "final_report":   None,
    "tool_results":   [],
    "metadata":       None,
    "profile":        None,   # DatasetProfile.to_dict()
    "dashboard":      None,   # list of ChartSpec dicts
    "tmp_dir":        None,
    "progress_lines": [],
    "llm_warning":    None,
    "from_uploader":  False,
    "preview_notes":  [],
    "run_objective":  "",     # the question the finished run was given
    "is_sample":      False,  # the finished run analysed the bundled sample file
    "run_view":       None,   # RunView of the finished run (src/core/run_view.py)
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Inject theme-aware CSS immediately (must run after session_state is ready) ─
_inject_theme_css()


# ── Helpers ───────────────────────────────────────────────────────────────────

#: Prefix of every per-run temp directory; `remove_run_dir` only ever deletes
#: directories that carry it, directly under the system temp root.
_RUN_DIR_PREFIX = RUN_DIR_PREFIX
_remove_run_dir = remove_run_dir


def _active_run() -> ActiveRun | None:
    """The run on a worker thread right now, if any."""
    run: ActiveRun | None = st.session_state.get("_run")
    return run


def _reset_pipeline() -> None:
    run = _active_run()
    if run is not None:
        # A run is still on its thread: signal it and let the polling fragment
        # reset once the worker ends, so a second run can never overlap it. The
        # worker deletes its own directory when it was discarded.
        run.discard_and_stop()
    else:
        _remove_run_dir(st.session_state.get("tmp_dir"))
    st.session_state.pop("current_summary_path", None)
    for k in ("stage_log", "analysis_done", "analysis_error",
              "final_report", "tool_results", "metadata", "profile",
              "dashboard", "tmp_dir", "progress_lines", "llm_warning",
              "preview_notes", "run_view"):
        st.session_state[k] = _DEFAULTS[k]


def _set_stage(num: str, status: str, detail: str = "") -> None:
    log: list[tuple[str, str, str]] = [
        e for e in st.session_state["stage_log"] if e[0] != num
    ]
    log.append((num, status, detail))
    st.session_state["stage_log"] = log


def _fold_run(run: ActiveRun) -> None:
    """Move a finished run's results into session state (script thread only)."""
    snap = run.snapshot()
    st.session_state["stage_log"] = list(snap.stage_log)
    st.session_state["progress_lines"] = list(snap.progress_lines)
    if run.metadata is not None:
        st.session_state["metadata"] = run.metadata
    out = run.outcome
    if out is None:
        st.session_state["analysis_error"] = run.error or "The run ended without a result."
        return
    st.session_state["tool_results"] = out.tool_results
    st.session_state["final_report"] = out.final
    st.session_state["profile"] = out.profile
    st.session_state["read_report"] = out.read_report
    st.session_state["coercions"] = out.coercions
    st.session_state["profile_status"] = out.profile_status
    st.session_state["dashboard"] = out.dashboard
    st.session_state["run_view"] = out.run_view
    st.session_state["llm_warning"] = out.llm_warning
    if out.summary_path:
        st.session_state["current_summary_path"] = out.summary_path
    st.session_state["analysis_done"] = True


@st.fragment(run_every="1s")
def _run_progress() -> None:
    """Poll the worker: show progress and Stop, then fold the result in and rerun."""
    run = _active_run()
    if run is None:
        return
    snap = run.snapshot()
    if snap.state != "running":
        st.session_state.pop("_run", None)
        if snap.discard:
            _reset_pipeline()
        else:
            _fold_run(run)
        st.rerun()

    if snap.discard:
        sub = "Cancelling. Waiting for the step in progress to finish."
    elif snap.stop_requested:
        sub = "Stopping. The step in progress finishes first, then a partial report is written."
    else:
        sub = snap.note or "Usually 1–3 minutes, depending on the dataset and the model."
    st.markdown(
        f'<div class="run-banner">Running the analysis<span class="sub">{html.escape(sub)}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(_render_steps_list(list(snap.stage_log)), unsafe_allow_html=True)
    _prov_html = build_provisional_html(snap.provisional)
    if _prov_html:
        st.markdown(_prov_html, unsafe_allow_html=True)
    if st.button("Stop", key="stop_run", disabled=snap.stop_requested or snap.discard):
        run.request_stop()
        st.rerun(scope="fragment")
    st.caption("Stop is cooperative: a step that is already running finishes before the run ends.")


def _draw_pipeline_rig(slot: Any) -> list[Any]:
    """Render the 3D pipeline rig into `slot`; return the stages it drew.

    Called from two places — the normal position at the end of the script, and
    just before `st.stop()` on an aborted run, since otherwise the hero would
    keep a blank gap where the rig should be.

    Imported lazily to match how `src/` is loaded in this file: after ROOT
    lands on sys.path.
    """
    from ui.pipeline_3d import Stage, StageStatus
    from ui.pipeline_3d import render as render_pipeline

    log = {n: (s, d) for n, s, d in st.session_state["stage_log"]}
    stages = [
        Stage(
            num=num,
            name=name,
            # session_state is untyped; _set_stage only writes StageStatus values.
            status=cast(StageStatus, log.get(num, ("pending", ""))[0]),
            detail=log.get(num, ("pending", ""))[1],
        )
        for num, name in STAGE_DEFS
    ]
    cur_theme = st.session_state.get("theme", "day")
    with slot.container():
        if st.session_state.get("show_cinematic_hero", False):
            from ui.cinematic_3d import render_cinematic
            render_cinematic(st.session_state, height=480, theme=cur_theme, compact=True)
        else:
            render_pipeline(stages, height=420, theme=cur_theme)
    return stages


@st.cache_resource(show_spinner=False)
def _cuda_available() -> bool:
    from src.tools.ml_pipeline import _detect_cuda_gpu

    return _detect_cuda_gpu()


@st.cache_data(ttl=1800, show_spinner=False)
def _get_dynamic_models(
    provider: str,
    api_key: str = "",
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Dynamically fetch available models and capability telemetry from provider."""
    from src.core.model_telemetry import fetch_available_models
    profiles = fetch_available_models(provider, api_key=api_key, base_url=base_url)
    return [
        {
            "model": p.model,
            "label": p.format_dropdown_label(),
            "context_window": p.context_window,
            "rpm_limit": p.rpm_limit,
            "tpm_limit": p.tpm_limit,
            "speed_tag": p.speed_tag,
            "is_free": p.is_free,
        }
        for p in profiles
    ]


def _join_review(uploads: list[Any]) -> dict[int, dict[str, Any]]:
    """Compact review card per related table. Default accepts the proposed
    join (no override); returns {upload_index: override} for Skip / Change key."""
    from src.core.joins import preview_joins

    base = st.session_state.get("preview_df")
    if base is None:
        return {}
    sig = (st.session_state.get("preview_name"), tuple((u.name, u.size) for u in uploads))
    cached = st.session_state.get("_join_prev")
    if not cached or cached[0] != sig:
        idx: list[int] = []
        tables: list[tuple[str, pd.DataFrame]] = []
        for i, u in enumerate(uploads):
            try:
                tables.append((Path(u.name).stem, read_any_bytes(u.getvalue(), u.name)[0]))
                idx.append(i)
            except Exception:
                continue   # unreadable files are reported at run time
        try:
            prev = preview_joins(base, tables, Path(st.session_state.get("preview_name") or "main").stem)
        except Exception:
            prev = []
        cached = (sig, idx, prev)
        st.session_state["_join_prev"] = cached
    _, idx, previews = cached
    out: dict[int, dict[str, Any]] = {}
    for i, p in zip(idx, previews, strict=False):
        plan = p["plan"]
        if plan is None:
            color, body = "var(--risk)", f"No join proposed. {html.escape(p['reason_if_none'])}"
        else:
            cov = float(plan["coverage"])
            color = "var(--positive)" if cov >= 0.9 else "var(--accent)" if cov >= 0.6 else "var(--risk)"
            if plan["cardinality"] == "many_to_many":
                color = "var(--accent)"
            body = (
                f"<b>{html.escape(plan['left_key'])}</b> = <b>{html.escape(plan['right_key'])}</b> · "
                f"{plan['cardinality'].replace('_', '-')} · {cov:.0%} matched"
            )
        st.markdown(
            f'<div style="border:1px solid var(--rule-faint); border-left:3px solid {color}; '
            f'background:var(--sheet); border-radius:var(--radius); padding:.4rem .6rem; '
            f'font-size:13px; margin-top:.5rem;"><b>{html.escape(p["name"])}</b><br>{body}</div>',
            unsafe_allow_html=True,
        )
        opts = ["Use this join", "Skip", "Change key"] if plan else ["Skip", "Change key"]
        mode = st.radio("Join", opts, horizontal=True, key=f"jr_mode_{i}", label_visibility="collapsed")
        if mode == "Skip":
            out[i] = {"skip": True}
        elif mode == "Change key":
            lcols, rcols = p["left_columns"], p["columns"]
            lk = st.selectbox("Main table column", lcols, key=f"jr_l_{i}",
                              index=lcols.index(plan["left_key"]) if plan and plan["left_key"] in lcols else 0)
            rk = st.selectbox("Related table column", rcols, key=f"jr_r_{i}",
                              index=rcols.index(plan["right_key"]) if plan and plan["right_key"] in rcols else 0)
            out[i] = {"left_key": lk, "right_key": rk}
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
# Where each part of the page goes. The containers are created first, in page order, and filled
# later in the script, so the primary task (file, question, run) can sit in the main area while
# the settings stay in the sidebar.
_run_in_flight = st.session_state.get("_run") is not None
_workspace_state = (
    "running" if _run_in_flight
    else "done" if st.session_state.get("analysis_done")
    else "failed" if st.session_state.get("analysis_error")
    else None
)
_hero_box = st.container()
# Before a run the inputs are the page's one task; once there is a run they tuck away.
_inputs_box = (
    st.expander("Your file and question", expanded=False)
    if _workspace_state
    else st.container(border=True)
)

with st.sidebar:
    st.markdown(
        '<div class="side-brand">'
        '<div class="side-title">Agentic Data Analysis</div>'
        '<div class="side-sub">Your friendly assistant for making sense of data.</div></div>',
        unsafe_allow_html=True,
    )

    # ── Theme Selector ────────────────────────────────────────────────────────
    _cur_theme = st.session_state.get("theme", "day")
    theme_sel = st.selectbox(
        "Theme",
        ["Day Mode", "Night Mode"],
        index=0 if _cur_theme == "day" else 1,
        help="Switch between a bright look for daytime and a cozy dark look for night.",
    )
    _new_theme = "night" if theme_sel == "Night Mode" else "day"
    if _new_theme != _cur_theme:
        st.session_state["theme"] = _new_theme
        st.rerun()

    with _inputs_box:
        # ── Upload ────────────────────────────────────────────────────────────────
        st.markdown('<div class="step-head"><span class="step-n">1</span> Add your file</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "CSV, TSV or Excel",
            type=sorted(ext.lstrip(".") for ext in ALLOWED_EXTENSIONS),
            label_visibility="collapsed",
        )

        # Persist to session_state immediately on upload / clear on removal.
        # Every upload passes through src.core.security before touching disk:
        # extension allowlist, size ceiling, magic-byte sniffing, safe filename.
        if uploaded is not None:
            if uploaded.name != st.session_state.get("orig_name", ""):
                _reset_pipeline()
                raw_bytes = uploaded.read()
                st.session_state["orig_name"] = uploaded.name
                from src.core.security import UploadValidationError, validate_upload
                try:
                    safe_name = validate_upload(uploaded.name, raw_bytes)
                except UploadValidationError as _ve:
                    st.session_state["preview_df"]    = None
                    st.session_state["preview_bytes"] = None
                    st.session_state["preview_name"]  = ""
                    st.error(f"Upload rejected. {_ve}")
                else:
                    st.session_state["preview_bytes"] = raw_bytes
                    st.session_state["preview_name"]  = safe_name
                    _fname = safe_name.lower()
                    try:
                        # One reader for every format/encoding/delimiter — a bare
                        # pd.read_csv here previewed a semicolon- or cp1252-encoded
                        # export as a single mangled column while the analysis
                        # behind it was correct.
                        _pdf, _prep = read_any_bytes(raw_bytes, safe_name)
                        st.session_state["preview_df"] = _pdf
                        st.session_state["preview_read_report"] = _prep
                        st.session_state["preview_notes"] = list(_prep.notes)
                    except Exception as _e:
                        st.session_state["preview_df"] = None
                        st.session_state["preview_notes"] = []
                        st.error(f"Could not read file: {_e}")
                    st.session_state["from_uploader"] = True
        else:
            if st.session_state.get("orig_name") and st.session_state.get("from_uploader"):

                _stale_run = _active_run()
                if _stale_run is not None:
                    _stale_run.discard_and_stop()  # its results belong to the removed file
                _theme = st.session_state.get("theme", "day")
                for _k2, _v2 in _DEFAULTS.items():
                    st.session_state[_k2] = _v2
                st.session_state["theme"] = _theme
                st.session_state["from_uploader"] = False

        _notes: list[str] = st.session_state.get("preview_notes") or []
        if _notes:
            _items = "".join(f"<li>{html.escape(str(n))}</li>" for n in _notes)
            st.markdown(
                f'<details class="file-notices"><summary>{len(_notes)} file notice'
                f'{"s" if len(_notes) > 1 else ""} (auto-repaired)</summary><ul>{_items}</ul></details>',
                unsafe_allow_html=True,
            )

        st.caption("Related tables (optional): customers, products... joined automatically on shared IDs.")
        related_uploads = st.file_uploader(
            "Related tables",
            type=sorted(ext.lstrip(".") for ext in ALLOWED_EXTENSIONS),
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        join_choices: dict[int, dict[str, Any]] = (
            _join_review(related_uploads) if related_uploads and st.session_state.get("preview_bytes") else {}
        )

        st.markdown('<div class="step-head"><span class="step-n">2</span> Ask your question</div>', unsafe_allow_html=True)
        objective = st.text_area(
            "What do you want to know? (optional, plain English)",
            placeholder="e.g. What's driving this result? Which rows are the "
                        "outliers, and why?",
            height=90,
            help="Your helpers will prioritise analyses that answer this question "
                 "and address it directly in the final report.",
        )

        target_col = st.text_input(
            "Target column",
            placeholder="e.g. outcome or price (blank = auto)",
        )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    st.markdown('<div class="side-head">LLM Provider</div>', unsafe_allow_html=True)
    provider = st.selectbox(
        "Provider",
        ["openai", "anthropic", "gemini", "groq", "openrouter", "nvidia"]
        + ([] if _HOSTED else ["local"]),
        format_func=lambda p: "Local / offline" if p == "local" else p,
    )

    local_base_url = ""
    if provider == "local":
        local_base_url = st.text_input(
            "Server URL (OpenAI-compatible)",
            value="http://localhost:11434/v1",
            help="Works with Ollama, LM Studio, vLLM, llama.cpp server, "
                 "text-generation-webui, etc. Must be reachable from this "
                 "machine. No data leaves it.",
        )

    _key_label = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "gemini": "Gemini",
        "groq": "Groq",
        "openrouter": "OpenRouter",
        "nvidia": "NVIDIA",
        "local": "Local server",
    }.get(provider, provider)

    key_ph = {
        "openai": "sk-...",
        "anthropic": "sk-ant-...",
        "gemini": "from aistudio.google.com/apikey",
        "groq": "gsk_...",
        "nvidia": "nvapi-...",
        "openrouter": "sk-or-...",
        "local": "usually not required",
    }.get(provider, "sk-...")

    api_key = st.text_input(
        f"{_key_label} API Key" + (" (optional)" if provider == "local" else ""),
        type="password",
        placeholder=key_ph,
    )

    # On a shared server a visitor must supply their own key; the server's
    # environment key is only a convenience for a single-user local run.
    _env_key = "" if _HOSTED else os.getenv(f"{provider.upper()}_API_KEY", "")
    _effective_key = api_key.strip() or _env_key
    _dyn_models = _get_dynamic_models(
        provider,
        api_key=_effective_key,
        base_url=local_base_url if provider == "local" else None,
    )

    free_models = [m for m in _dyn_models if m.get("is_free")]
    paid_models = [m for m in _dyn_models if not m.get("is_free")]

    # Show Free and Paid models separately when both are present
    if free_models and paid_models:
        tier_choice = st.radio(
            "Pricing Tier",
            [f"Free Models ({len(free_models)})", f"Paid Models ({len(paid_models)})", f"All ({len(_dyn_models)})"],
            horizontal=True,
            index=0,
            key=f"tier_filter_{provider}",
        )
        if "Free" in tier_choice:
            active_dyn_models = free_models
        elif "Paid" in tier_choice:
            active_dyn_models = paid_models
        else:
            active_dyn_models = sorted(
                _dyn_models,
                key=lambda m: (not m.get("is_free", False), m["model"]),
            )
    elif free_models and not paid_models:
        st.caption("All models listed below are Free Tier eligible.")
        active_dyn_models = free_models
    else:
        st.caption("Paid API billing applies per token.")
        active_dyn_models = paid_models

    model_options = [m["model"] for m in active_dyn_models]
    model_labels = {m["model"]: m["label"] for m in active_dyn_models}

    if not model_options:
        model_options = ["default"]
        model_labels = {"default": "default"}

    model_sel = st.selectbox(
        "Model",
        model_options,
        format_func=lambda m: model_labels.get(m, m),
    )

    if provider in ("openrouter", "nvidia", "local", "gemini", "groq"):
        custom_m = st.text_input(
            "Custom model string (overrides above)",
            placeholder={
                "gemini": "e.g. gemini-2.5-flash, gemini-flash-latest",
                "groq": "e.g. llama-3.3-70b-versatile, llama-3.1-8b-instant",
                "openrouter": "e.g. cohere/command-r-plus",
                "nvidia": "e.g. nvidia/llama-3.1-nemotron-70b-instruct",
                "local": "e.g. the exact tag your server has pulled/loaded",
            }[provider],
        )
        final_model = custom_m.strip() if custom_m.strip() else model_sel
    else:
        final_model = model_sel

    # ── Reasoning Mode ────────────────────────────────────────────────────────
    reasoning_mode = st.selectbox(
        "Reasoning Mode",
        ["Adaptive (Recommended)", "Fast (Low Reasoning)", "Deep (High Reasoning)"],
        index=0,
        help=(
            "Adaptive: dynamically scales reasoning effort, low during routine exploratory steps, "
            "higher when anomalies or statistical conflicts occur, and thorough for final synthesis.\n"
            "Fast: forces minimal reasoning effort across all cycles for maximum execution speed.\n"
            "Deep: uses full reasoning depth across all cycles."
        ),
    )
    # Passed to the run with its other settings, never through os.environ.
    reasoning_effort = (
        "low" if "Fast" in reasoning_mode else "high" if "Deep" in reasoning_mode else "adaptive"
    )

    # ── Engine ────────────────────────────────────────────────────────────────
    # The two capability switches. Both default on; either can be turned off
    # independently, and the analysis still runs end to end and still writes a
    # full report — that is the point of them.
    st.markdown('<div class="side-head">Engine</div>', unsafe_allow_html=True)
    use_llm = st.toggle(
        "AI narrative (LLM)",
        value=True,
        help=(
            "On: the LLM plans the analysis and writes the narrative.\n"
            "Off: fully deterministic, the plan comes from the data profile "
            "and domain detection, and the report is built from tool output. "
            "No network calls, no API key needed, and much faster."
        ),
    )
    use_ml = st.toggle(
        "Machine learning",
        value=True,
        help=(
            "On: trains models, clusters, and runs PCA.\n"
            "Off: skips every model-fitting step. Statistical tests, "
            "correlations and the domain analyses still run. This is the "
            "single biggest speed-up available: training dominates runtime."
        ),
    )
    if not use_llm and not use_ml:
        st.caption("Fully deterministic, statistics-only mode, fastest.")
    elif not use_llm:
        st.caption("Deterministic planning, ML still on.")
    elif not use_ml:
        st.caption("AI narrative on, no models fitted.")
    tune_hyperparameters = False
    max_depth = 6
    test_pct = 20
    n_cv = 5
    if use_ml:
        # IMPROVEMENTS.md 7.16 — genuinely wired, not just informational.
        # `TrainModelTool.requires_context` (ml_pipeline.py) reads these back
        # from memory context and fills them into the actual train_model
        # call whenever the planner leaves them empty (the same fallback-fill
        # mechanism target_column already used) — so moving these sliders
        # really does change what gets trained, not just what's displayed.
        thorough = st.toggle(
            "Thorough tuning (slower)",
            value=False,
            help=(
                "Off (default): models train with their default hyperparameters, fast.\n"
                "On: searches hyperparameters per model before picking the best "
                "one, meaningfully slower (measured: ~20s extra at 20k rows) "
                "but can improve accuracy."
            ),
        )
        tune_hyperparameters = thorough

        if _cuda_available():
            st.caption("**GPU acceleration**: CUDA detected. XGBoost models will train on GPU (`device='cuda'`).")
        else:
            st.caption("**Compute**: CPU mode (multi-core parallel training).")

        with st.expander("Advanced model settings"):
            max_depth = st.slider("Max tree depth (Random Forest / XGBoost)", 2, 15, 6)
            test_pct = st.slider("Test split %", 10, 40, 20, step=5)
            n_cv = st.slider("CV folds (k)", 3, 10, 5)

    # ── Privacy and code execution ────────────────────────────────────────────
    # Operator controls (AGENTS.md "Operator controls"). On a shared deployment
    # they come from the server's environment and are shown read-only: a
    # visitor must not be able to switch isolation off or loosen the privacy
    # floor from the page. On a local single-user run they stay editable.
    from src.core.governance import code_execution_enabled
    from src.core.privacy import min_cell_size

    st.markdown('<div class="side-head">Privacy</div>', unsafe_allow_html=True)
    min_cell: int = min_cell_size()
    enable_code: bool = True
    max_code_runs: int = 40
    require_isolation: bool = False
    if _HOSTED:
        st.caption(
            f"Groups smaller than {min_cell} are combined so individuals can't be identified. "
            "Set by the server."
        )
        st.markdown('<div class="side-head">Code execution</div>', unsafe_allow_html=True)
        st.caption(
            "AI-written code is "
            + ("allowed in a restricted sandbox." if code_execution_enabled() else "switched off on this server.")
            + " Set by the server."
        )
    else:
        min_cell = int(st.number_input("Minimum group size", min_value=1, max_value=50,
                                       value=min_cell, step=1))
        st.caption("Groups smaller than this are combined so individuals can't be identified")

        st.markdown('<div class="side-head">Code execution</div>', unsafe_allow_html=True)
        enable_code = st.toggle(
            "Allow AI-written code",
            value=True,
            help=(
                "On: the planner may write and run its own analysis code in the sandbox.\n"
                "Off: only the built-in tools run."
            ),
        )
        max_code_runs = int(st.number_input(
            "Max code runs per analysis",
            min_value=0,
            value=40,
            step=1,
            disabled=not enable_code,
            help="Further code steps are refused and logged once the budget is spent.",
        ))
        require_isolation = st.toggle(
            "Require container isolation",
            value=False,
            disabled=not enable_code,
            help=(
                "On: code runs only in the Docker sandbox and is refused if Docker is "
                "unavailable. Turn on whenever the data or question comes from someone "
                "you don't trust."
            ),
        )

    # ── Analysis Settings ─────────────────────────────────────────────────────
    st.markdown('<div class="side-head">Analysis Settings</div>', unsafe_allow_html=True)
    # "Thorough" reproduces the previous default (min=1, max=5); Quick and Deep
    # scale the same min/max iteration knobs the controller already takes.
    thoroughness = st.radio(
        "How thorough",
        ["Quick", "Thorough", "Deep"],
        index=1,
        horizontal=True,
        help="Quick finds the headline answer fast. Thorough double-checks it. Deep explores more before settling.",
    )
    min_iter, max_iter = {
        "Quick":    (1, 2),
        "Thorough": (1, 5),
        "Deep":     (2, 15),
    }[thoroughness]
    enable_rlm = st.toggle(
        "Break hard questions into smaller ones",
        value=False,
        help="Splits a complex question into smaller sub-questions it can tackle one at a time. Turn on for deep exploration; keep off for the fastest runtime.",
    )

    has_file = st.session_state["preview_df"] is not None
    # A key is only needed when the AI narrative is on: the no-AI run is fully
    # deterministic and makes no network call.
    has_key  = bool(api_key.strip()) or provider == "local" or not use_llm
    can_run  = (
        has_file and has_key
        and not st.session_state["analysis_done"]
        and _active_run() is None
    )

with _inputs_box:
    st.markdown('<div class="step-head"><span class="step-n">3</span> Run</div>', unsafe_allow_html=True)
    run_clicked = st.button(
        "Run analysis",
        disabled=not can_run,
        type="primary",
        key="btn_run_analysis",
    )
    if not has_file:
        st.caption("Add a file above, or try the bundled sample below.")
    elif not has_key and not st.session_state["analysis_done"]:
        st.caption(
            "Add your API key in the settings sidebar to enable the run, or switch the AI narrative off."
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — header
# ══════════════════════════════════════════════════════════════════════════════
with _hero_box:
    hero_text, hero_plate = st.columns([0.46, 0.54], gap="large",
                                       vertical_alignment="center")

    # Once there is a file or a run, the page is a workspace, not a landing page: the marketing
    # headline gives way to a compact header so the answer is not pushed below the fold.
    with hero_text:
        if _workspace_state:
            _file_label = html.escape(st.session_state.get("preview_name") or "your file")
            _lead = {"running": "Analysing", "done": "Results for", "failed": "Could not finish"}[_workspace_state]
            st.markdown(
                '<div class="hero compact">'
                '<div class="hero-eyebrow">Agentic Data Analysis</div>'
                f'<h1>{_lead} <span class="hero-file">{_file_label}</span></h1></div>',
                unsafe_allow_html=True,
            )
            if st.button("New analysis", key="btn_new_analysis"):
                _reset_pipeline()
                st.rerun()
        else:
            st.markdown(
                '<div class="hero">'
                '<h1>We check every answer twice.</h1>'
                '<p class="hero-sub">Upload any spreadsheet: sales, survey, sports, science, '
                'whatever you\'ve got. Your assistant studies it, tests its own conclusions, '
                'and tells you which patterns are real, and which are just luck.</p></div>',
                unsafe_allow_html=True,
            )
        _hero_cinema_on = st.session_state.get("show_cinematic_hero", False)
        _hero_btn_txt = "Standard 3D Plate" if _hero_cinema_on else "3D Cinematic Showcase"
        if st.button(_hero_btn_txt, key="btn_toggle_hero_cinema"):
            st.session_state["show_cinematic_hero"] = not _hero_cinema_on
            st.rerun()

    # The plate: a live technical drawing of the run, ruled off the headline and
    # running past the container edge. The pipeline executes further down this same
    # script pass, so the drawing is filled into this placeholder afterwards — that
    # way it shows the state of the run that just happened.
    with hero_plate.container(key="plate"):
        steps_list_slot = st.empty()
        pipeline_slot = st.empty()

    # The datum line under the hero carries the run's readings, filled at the same
    # time as the plate.
    datum_slot = st.empty()


# ══════════════════════════════════════════════════════════════════════════════
# DATASET PREVIEW — always visible once a file is loaded
# ══════════════════════════════════════════════════════════════════════════════
preview_df: pd.DataFrame | None = st.session_state["preview_df"]

if preview_df is not None and not st.session_state["analysis_done"] and not _run_in_flight:
    _section("Dataset preview", st.session_state["preview_name"])
    _miss_cells = int(preview_df.isnull().sum().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_gauge("Rows", f"{len(preview_df):,}"), unsafe_allow_html=True)
    c2.markdown(_gauge("Columns", str(len(preview_df.columns))),
                unsafe_allow_html=True)
    c3.markdown(_gauge("Missing cells", f"{_miss_cells:,}",
                       flag=_miss_cells > 0), unsafe_allow_html=True)
    c4.markdown(
        _gauge("Numeric columns",
               str(len(preview_df.select_dtypes(include="number").columns))),
        unsafe_allow_html=True,
    )

    with st.expander("First 10 rows", expanded=True):
        st.dataframe(_safe_df(preview_df.head(10)), width='stretch')

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Column types and missing values")
        dtype_df = pd.DataFrame(
            [(c, str(t), int(preview_df[c].isnull().sum()))
             for c, t in preview_df.dtypes.items()],
            columns=["Column", "Type", "Missing"],
        )
        st.dataframe(_safe_df(dtype_df), width='stretch', height=200)
    with col_r:
        st.markdown("#### Descriptive statistics")
        st.dataframe(_safe_df(preview_df.describe()), width='stretch', height=200)
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE — started on button click, runs on a worker thread
# ══════════════════════════════════════════════════════════════════════════════
# The click builds a frozen RunSpec, checks the model is reachable, and starts
# an ActiveRun (ui/run.py). The page stays responsive: the polling fragment
# `_run_progress` shows progress and Stop, and folds the finished outcome into
# session state. The worker never calls st.* and never reads session state.
#
# The empty-state "sample" button sets this flag and reruns, so the sample goes
# through exactly the same run path as an uploaded file.
_sample_run: bool = bool(st.session_state.pop("_sample_run", False))
if (run_clicked or _sample_run) and st.session_state.get("_run") is None:
    # The sample is always a no-AI run, so it needs no key and no network.
    run_llm: bool = use_llm and not _sample_run
    run_objective_text: str = "" if _sample_run else objective.strip()
    run_target: str | None = "churn" if _sample_run else (target_col.strip() or None)
    _reset_pipeline()
    for num, _ in STAGE_DEFS:
        _set_stage(num, "pending")

    # Save dataset to a temp file
    tmp = tempfile.mkdtemp(prefix=_RUN_DIR_PREFIX)
    st.session_state["tmp_dir"] = tmp
    dpath  = str(Path(tmp) / st.session_state["preview_name"])
    outdir = str(Path(tmp) / "output")
    with open(dpath, "wb") as _f:
        _f.write(st.session_state["preview_bytes"])
    related_paths: list[str] = []
    join_overrides: dict[str, dict[str, Any]] = {}
    if related_uploads and not _sample_run:
        from src.core.security import UploadValidationError, validate_upload
        for _i, _ru in enumerate(related_uploads):
            _rb = _ru.getvalue()
            try:
                _rname = validate_upload(_ru.name, _rb)
            except UploadValidationError as _ve:
                st.warning(f"Related table {_ru.name} skipped. {_ve}")
                continue
            _rp = Path(tmp) / "related" / str(_i) / _rname
            _rp.parent.mkdir(parents=True, exist_ok=True)
            _rp.write_bytes(_rb)
            related_paths.append(str(_rp))
            if _i in join_choices:
                join_overrides[_rp.stem] = join_choices[_i]

    # Nothing below is published through os.environ: provider, model, key, URL,
    # objective, output directory and reasoning effort all travel with the run
    # (RunConfig / RunSpec), because the environment is shared by every
    # visitor's session.
    run_config = RunConfig(
        provider=provider,
        model=final_model,
        api_key=_effective_key or None,
        base_url=(local_base_url.strip() or None) if provider == "local" else None,
        reasoning_effort=reasoning_effort,
    )
    _picked = next((m for m in _dyn_models if m["model"] == final_model), None)
    if _picked and run_llm:
        from src.core.model_telemetry import get_limiter

        # Model facts (context window, published limits), not per-user state.
        _profile = get_limiter().get_profile(provider, final_model)
        _profile.context_window = _picked["context_window"]
        _profile.rpm_limit = _picked["rpm_limit"] or _profile.rpm_limit
        _profile.tpm_limit = _picked["tpm_limit"] or _profile.tpm_limit
    if not _HOSTED:
        # Local single-user run only: src/core reads these governance and
        # privacy limits from the environment. A shared deployment takes them
        # from the server's own environment and never from the page.
        os.environ["ENABLE_CODE_EXECUTION"] = "true" if enable_code else "false"
        os.environ["MAX_CODE_EXECUTIONS"] = str(max_code_runs)
        os.environ["SANDBOX_REQUIRE_ISOLATION"] = "true" if require_isolation else "false"
        os.environ["DSA_MIN_CELL_SIZE"] = str(min_cell)

    # ── LLM preflight — fail fast with the REAL error instead of running
    #    the whole pipeline on the deterministic fallback ──────────────────
    from src.core.controller import LLMClient

    # In no-LLM mode there is nothing to preflight — the run is fully
    # deterministic, so requiring a reachable model (or any API key) would
    # block the very mode that exists to work without one.
    _ok, _ping_err = (True, "") if not run_llm else LLMClient(run_config).ping()
    if not _ok:
        _set_stage("2", "error", "LLM unreachable")
        st.session_state["analysis_error"] = _ping_err
        st.error(
            f"Could not reach the model, so the analysis did not start. "
            f"Provider `{provider}`, model `{final_model}`."
        )
        st.code(_ping_err, language=None)
        st.info(
            "Check that the model ID exists on this provider, that the API key "
            "is valid, and that the account has credits. Then run it again."
        )
        _draw_pipeline_rig(pipeline_slot)  # the hero slot must not stay empty
        st.stop()

    st.session_state["run_objective"] = run_objective_text
    st.session_state["is_sample"] = _sample_run
    _new_run = ActiveRun(
        RunSpec(
            dataset_path=dpath,
            output_dir=outdir,
            tmp_dir=tmp,
            stage_names=tuple(STAGE_DEFS),
            objective=run_objective_text,
            target=run_target,
            min_iterations=min_iter,
            max_iterations=max_iter,
            enable_rlm=enable_rlm,
            use_llm=run_llm,
            use_ml=use_ml,
            run_config=run_config,
            related_paths=tuple(related_paths),
            join_overrides=join_overrides,
            ml_max_depth=max_depth if use_ml else None,
            ml_test_size=test_pct / 100.0 if use_ml else None,
            ml_n_cv_folds=n_cv if use_ml else None,
            ml_tune=tune_hyperparameters if use_ml else None,
            is_sample=_sample_run,
            keep_summary=not _HOSTED,
        )
    )
    st.session_state["_run"] = _new_run
    _new_run.start()
    # Re-render once so the sidebar (Run disabled, New analysis offered) already
    # reflects the active run; the fragment below takes over from here.
    st.rerun()

if st.session_state.get("_run") is not None:
    _run_progress()


# ══════════════════════════════════════════════════════════════════════════════
# HERO PLATE & DATUM REFRESH
# ══════════════════════════════════════════════════════════════════════════════
_done = sum(1 for _, s, _ in st.session_state["stage_log"] if s == "done")
_errored = any(s == "error" for _, s, _ in st.session_state["stage_log"])
_running = any(s == "active" for _, s, _ in st.session_state["stage_log"])

_run_active = _active_run() is not None
if (st.session_state.get("analysis_done") or _errored or _running) and not _run_active:
    steps_list_slot.markdown(_render_steps_list(st.session_state["stage_log"]), unsafe_allow_html=True)
    with pipeline_slot.container():
        with st.expander("Show how it's working", expanded=False):
            _draw_pipeline_rig(st.empty())
            st.markdown("#### The Team at Work")
            st.markdown(_render_agent_grid(st.session_state["stage_log"]), unsafe_allow_html=True)
else:
    steps_list_slot.empty()
    stages_3d = _draw_pipeline_rig(pipeline_slot)

# The datum line: the same run state as the drawing, in words and figures.
_error_step = next((n for n, s, _ in st.session_state["stage_log"] if s == "error"), None)
_active_step = next((n for n, s, _ in st.session_state["stage_log"] if s == "active"), None)
_status_str = (
    f"Stopped at step {_error_step}" if _error_step else
    "Running" if _run_active else
    "Stopped early" if (st.session_state.get("final_report") or {}).get("stopped") else
    "Done" if st.session_state.get("analysis_done") else
    f"Working on step {_active_step} of 7" if _active_step else
    "Ready"
)

_view_for_cell = st.session_state.get("run_view")
_ran_without_ai = bool(
    (_view_for_cell is not None and _view_for_cell.how.no_ai)
    or (_run_in_flight and (st.session_state.get("is_sample") or not use_llm))
)
_model_cell = "No AI (deterministic)" if _ran_without_ai else (final_model if "final_model" in locals() else "N/A")

_cells = [
    ("State", _status_str),
    ("File", st.session_state.get("preview_name") or "None loaded"),
    ("Model", _model_cell),
]
datum_slot.markdown(_datum(_cells), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS — 5-TAB ARCHITECTURAL DOSSIER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("analysis_done"):
    report: dict[str, Any] = st.session_state.get("final_report") or {}
    tool_results: list[dict[str, Any]] = st.session_state.get("tool_results") or []
    meta: Any = st.session_state.get("metadata")  # DatasetMetadata | None (lazy import)
    tmp_dir: str = st.session_state.get("tmp_dir") or ""
    outdir = str(Path(tmp_dir) / "output") if tmp_dir else ""
    dash: list[dict[str, Any]] | None = st.session_state.get("dashboard")
    profile: dict[str, Any] | None = st.session_state.get("profile")

    # The run's LLM failure reason (set above from memory "llm_error") was
    # stored but never rendered anywhere, so a planning call that silently
    # fell back to the deterministic plan left no visible trace in the UI.
    if st.session_state.get("llm_warning"):
        st.warning(st.session_state["llm_warning"])

    # IMPROVEMENTS.md 7.21 (user-confirmed): 4 top-level tabs, findings-led.
    # "Your Helpers" (the agent grid / handoff stream) folds into Details as
    # a "Run Trace" subsection rather than competing for top-level attention
    # with the findings that answer the user's actual question; the live
    # "3D Cinematic Journey" tab is dropped as a top-level tab since its
    # content already exists as a standalone HTML export in Downloads
    # (`tab_vault`, below) — keeping both was two ways to reach the same
    # experience.
    if report.get("stopped"):
        _n_tools = sum(
            1 for t in tool_results
            if t.get("tool_name") not in ("ingest_dataset", "planner", "generate_report")
        )
        st.warning(
            "This run was stopped before it finished. The results below cover only "
            f"the {_n_tools} analysis step{'s' if _n_tools != 1 else ''} completed up to that point, "
            "so they may be incomplete."
        )
    if st.session_state.get("is_sample"):
        st.info(
            "This is a real analysis of the bundled sample file "
            "(data/sample_customer_churn.csv), run without the AI narrative. "
            "It is not your data."
        )
    (tab_brief, tab_dash, tab_lab, tab_vault) = st.tabs([
        "Answers",
        "Charts",
        "Details",
        "Downloads",
    ])

    vega_cfg = _get_vega_config()

    with tab_brief:
        render_answers_tab(
            report=report,
            tool_results=tool_results,
            meta=meta,
            dash=dash,
            prof=profile,
            preview_df=preview_df,
            vega_cfg=vega_cfg,
            run_view=st.session_state.get("run_view"),
        )

    with tab_dash:
        render_charts_tab(
            dashboard=dash,
            report=report,
            vega_cfg=vega_cfg,
        )

    with tab_lab:
        render_details_tab(
            report=report,
            tool_results=tool_results,
            prof=profile,
            run_view=st.session_state.get("run_view"),
        )

    with tab_vault:
        render_downloads_tab(
            report=report,
            tool_results=tool_results,
            tmp_dir=tmp_dir,
        )


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════
if (preview_df is None
        and not st.session_state["analysis_done"]
        and not st.session_state["stage_log"]):
    st.markdown(
        '<div class="empty">'
        '<h2>Let\'s see what your data shows.</h2>'
        '<p>Add a CSV or Excel file above, tell us what you\'d like to know, and run it. '
        'Your helpers will study the data, test their answers, and double-check '
        'everything before showing you the results. An API key is only needed for the '
        'AI-written summary; the analysis itself runs without one.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Meet Your Helpers")
    st.markdown(_render_agent_grid([]), unsafe_allow_html=True)
    st.write("")
    st.caption(
        "Runs the bundled customer sample through the real analysis, "
        "with no API key and no AI narrative."
    )
    if st.button("Try it with sample data", type="primary"):
        _sample = ROOT / "data" / "sample_customer_churn.csv"
        if _sample.exists():
            _sample_bytes = _sample.read_bytes()
            _sample_df, _sample_report = read_any_bytes(_sample_bytes, _sample.name)
            st.session_state["preview_df"] = _sample_df
            st.session_state["preview_read_report"] = _sample_report
            st.session_state["preview_bytes"] = _sample_bytes
            st.session_state["preview_name"] = _sample.name
            st.session_state["orig_name"] = _sample.name
            st.session_state["from_uploader"] = False
            st.session_state["_sample_run"] = True
            st.rerun()
        else:
            st.error("The bundled sample file is missing from this install (data/sample_customer_churn.csv).")

# ══════════════════════════════════════════════════════════════════════════════
# MICRO-INTERACTIONS (Phase 3)
# ══════════════════════════════════════════════════════════════════════════════
from ui.animations import inject_micro_interactions

inject_micro_interactions()
