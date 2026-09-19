"""
Streamlit UI — Agentic Data Analysis System.

Run:
    streamlit run app.py

Key fixes vs previous version
------------------------------
* NO background thread + st.rerun() loop.  The pipeline runs synchronously
  inside st.status() so Streamlit renders live progress without fighting its
  own execution model.
* Dataset preview is saved to session_state on file upload and rendered from
  there — no dependency on sidebar scope surviving a rerun.
* OpenRouter support added (any model string, OpenAI-compatible endpoint).
"""
from __future__ import annotations

import html
import json
import os
import sys
import tempfile
import traceback
import types
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

from src.core.io import read_any, read_any_bytes
from src.core.plain_language import describe_uncertainty, plainify
from src.core.security import ALLOWED_EXTENSIONS

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Agentic Data Analysis",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
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


def _inject_theme_css() -> None:
    """Inject dynamic Ledger CSS supporting Day and Night modes via Python state."""
    theme = st.session_state.get("theme", "night")

    if theme == "dark" or theme == "night":
        theme_vars = """
        --stock:       #241c14;
        --sheet:       #2f251a;
        --sheet-alt:   #3a2e1f;
        --ink:         #f3e9d8;
        --graphite:    #d0c2a8;
        --pen:         #f0a24a;
        --pen-hover:   #ffb86b;
        --risk:        #e2685a;
        --accent:      #d99a4e;
        --positive:    #7fb77e;
        --rule:        #4a3c28;
        --rule-faint:  #3a2e1f;
        --lift:        0 4px 18px rgba(0,0,0,.35);
        --lift-sm:     0 2px 8px rgba(0,0,0,.3);
        --code-bg:     #2a2015;
        """
    else:
        theme_vars = """
        --stock:       #f7eedd;
        --sheet:       #fffbf2;
        --sheet-alt:   #f1e4cb;
        --ink:         #3a2b1e;
        --graphite:    #8a7660;
        --pen:         #a34f20;
        --pen-hover:   #7e3d18;
        --risk:        #a33526;
        --accent:      #e08a3e;
        --positive:    #5b8c5a;
        --rule:        #e4d4bc;
        --rule-faint:  #eee3cb;
        --lift:        0 4px 14px rgba(58,43,30,.14);
        --lift-sm:     0 2px 8px rgba(58,43,30,.10);
        --code-bg:     #f1e4cb;
        """

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800&family=Mukta:wght@400;500;600;700&display=swap');

:root {{
    {theme_vars}
    --radius:      14px;
    --radius-pill: 999px;

    --sans:    'Mukta', ui-sans-serif, 'Segoe UI', system-ui, sans-serif;
    --heading: 'Baloo 2', 'Mukta', ui-sans-serif, sans-serif;
    --mono:    'Cascadia Code', Consolas, ui-monospace, monospace;
}}


/* ── The page ── */
#MainMenu, footer, .stAppDeployButton {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
/* The native run-status widget (spinner + "Stop") is the one piece of
   stock Streamlit chrome the above rules don't touch — it's the only way
   to interrupt a running analysis, so it stays, just re-themed to match
   the Ledger palette instead of Streamlit's defaults. */
[data-testid="stStatusWidget"] {{
    background: var(--sheet);
    border: 1px solid var(--rule);
    border-radius: var(--radius-pill);
    box-shadow: var(--lift-sm);
    color: var(--ink);
}}
[data-testid="stStatusWidget"] svg {{ color: var(--pen); }}
.stApp {{
    background-color: var(--stock);
    background-image: none;
    transition: background-color 0.2s ease;
}}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; }}
html, body, .stApp, [class*="css"] {{ font-family: var(--sans); color: var(--ink); }}
hr {{ border: none; border-top: 1px solid var(--rule) !important; }}
a {{ color: var(--pen) !important; text-underline-offset: 3px; font-weight: 600; }}

section[data-testid="stSidebar"] {{
    background: var(--sheet);
    border-right: 1px solid var(--rule);
    background-image: none;
}}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] label p {{ font-size: 13.5px; color: var(--graphite); font-weight: 600; }}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: var(--rule); border-radius: 6px; border: 2px solid var(--stock); }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* ── Type: warm & rounded ── */
h1, h2, h3, h4, h5, h6 {{
    font-family: var(--heading) !important;
    letter-spacing: -.01em;
    color: var(--ink);
}}
h1 {{ font-weight: 800 !important; }}
h2 {{ font-weight: 700 !important; font-size: 27px !important; line-height: 1.15; }}
h3 {{ font-weight: 700 !important; font-size: 19px !important; }}
h4 {{ font-weight: 700 !important; font-size: 15.5px !important; letter-spacing: 0; }}
.stMarkdown p, .stMarkdown li {{ font-size: 15.5px; line-height: 1.65; max-width: 72ch; }}
code, kbd, pre, .stCode {{ font-family: var(--mono) !important; }}
[data-testid="stMetricValue"] {{ font-family: var(--heading) !important; font-weight: 700; color: var(--ink) !important; }}
[data-testid="stMetricLabel"] * {{ color: var(--graphite) !important; }}
[data-testid="stFileUploader"] section {{ background: var(--sheet) !important; border: 1px dashed var(--rule) !important; }}
[data-testid="stFileUploader"] section * {{ color: var(--ink) !important; }}
[data-testid="stFileUploader"] small {{ color: var(--graphite) !important; }}
.stExpander {{ border-color: var(--rule) !important; background: var(--sheet) !important; }}
.stExpander summary {{ color: var(--ink) !important; }}

/* ── Quick Facts bar ── */
.datum {{ display: flex; align-items: stretch; flex-wrap: wrap;
         background: var(--sheet); border: 1px solid var(--rule);
         border-radius: var(--radius); box-shadow: var(--lift-sm);
         margin: 0 0 1.5rem; overflow: hidden; }}
.datum .cell {{ padding: .7rem 1.2rem; margin-right: 0;
               border-right: 1px solid var(--rule-faint); }}
.datum .cell:last-child {{ border-right: none; }}
.datum .k {{ font-size: 12px; color: var(--graphite); font-weight: 600; }}
.datum .v {{ font-family: var(--sans); font-size: 14px; font-weight: 700; color: var(--ink); margin-top: 2px; }}

/* ── Section head ── */
.sect {{ margin: 2.2rem 0 1.1rem; border-bottom: 2px solid var(--rule);
        padding-bottom: .5rem; }}
.sect:first-child {{ margin-top: .4rem; }}
.sect h2, .sect h3 {{ margin: 0; padding: 0; }}
.sect .note {{ font-size: 13px; color: var(--graphite); margin-top: .3rem; }}

/* ── Hero ── */
.hero {{ padding: .2rem 0 1rem; }}
.hero h1 {{
    font-size: clamp(36px, 5.6vw, 64px);
    font-weight: 800;
    line-height: 1.04;
    letter-spacing: -.02em;
    margin: 0;
    max-width: 15ch;
    animation: riseIn 650ms cubic-bezier(.16,.84,.34,1) both;
}}
@keyframes riseIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.hero .hero-sub {{
    color: var(--graphite); font-size: 16px; line-height: 1.6;
    margin: 1rem 0 0; max-width: 54ch;
}}
@media (prefers-reduced-motion: reduce) {{ .hero h1 {{ animation: none; }} }}

.st-key-plate {{ padding-left: 16px; margin-right: -2.8rem; }}
@media (max-width: 900px) {{ .st-key-plate {{ margin-right: 0; padding-left: 0; }} }}

/* ── Sidebar masthead & Theme controls ── */
.side-brand {{ margin: .1rem 0 .8rem; }}
.side-title {{ font-family: var(--heading); font-weight: 800; font-size: 18px;
              line-height: 1.15; color: var(--ink); }}
.side-sub {{ font-size: 12.5px; color: var(--graphite); margin-top: 4px;
            max-width: 26ch; line-height: 1.45; }}
.side-head {{ font-family: var(--sans); font-weight: 700; font-size: 12.5px;
             color: var(--graphite); margin: 1.5rem 0 .6rem; }}
.side-head:first-of-type {{ margin-top: .5rem; }}

/* ── Buttons ── */
.stButton button, .stDownloadButton button {{
    font-family: var(--sans); font-weight: 700; font-size: 14.5px;
    border-radius: var(--radius-pill) !important; letter-spacing: 0;
    transition: transform .12s ease, box-shadow .12s ease, background .12s ease;
}}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {{
    background: var(--pen); color: var(--sheet); border: none;
    box-shadow: var(--lift-sm);
}}
.stButton button[kind="primary"]:hover:enabled,
.stDownloadButton button[kind="primary"]:hover:enabled {{
    background: var(--pen-hover); color: var(--sheet);
    transform: translateY(-1px); box-shadow: var(--lift);
}}
.stButton button[kind="primary"]:disabled {{
    background: var(--sheet-alt); color: var(--graphite);
    border: 1px dashed var(--rule); box-shadow: none;
}}
.stButton button[kind="secondary"], .stDownloadButton button[kind="secondary"] {{
    background: var(--sheet); border: 1px solid var(--rule); color: var(--ink);
    box-shadow: var(--lift-sm);
}}
.stButton button[kind="secondary"]:hover:enabled,
.stDownloadButton button[kind="secondary"]:hover:enabled {{
    background: var(--sheet-alt); color: var(--ink); border-color: var(--pen);
    transform: translateY(-1px);
}}
:focus-visible {{ outline: 2px solid var(--pen) !important; outline-offset: 2px; }}
.stButton button:focus-visible, .stDownloadButton button:focus-visible {{
    outline: 2px solid var(--pen) !important; outline-offset: 3px;
}}

/* ── Tabs ── */
.stTabs [role="tablist"] {{
    gap: 0 !important; background: var(--sheet-alt) !important; border-radius: var(--radius-pill) !important;
    padding: 6px !important; overflow-x: auto; border: none; border-bottom: none !important;
    display: flex !important; width: 100% !important;
}}
.stTabs [role="tab"] {{
    flex: 1 1 0px !important; justify-content: center !important; text-align: center !important;
    border-radius: var(--radius-pill) !important; padding: 6px 16px !important; background-color: transparent !important;
    border: none !important; transition: all 0.14s ease; margin: 0 !important; outline: none !important;
}}
.stTabs [role="tab"]:last-child {{ margin-right: 0 !important; }}
.stTabs [role="tab"]:focus, .stTabs [role="tab"]:focus-visible {{ outline: none !important; }}
.stTabs [role="tab"] p {{ font-size: 14.5px; font-weight: 700;
                                 color: var(--graphite) !important; letter-spacing: 0; margin: 0 !important; }}
.stTabs [role="tab"]:hover p {{ color: var(--ink) !important; }}
.stTabs [role="tab"][aria-selected="true"] {{ background-color: var(--pen) !important; }}
.stTabs [role="tab"][aria-selected="true"] p {{ color: var(--sheet) !important; }}
.stTabs [data-testid="stTabIndicator"] {{ display: none !important; }}
.stTabs [data-testid="stTabsContent"] {{ padding-top: 1.5rem; }}

/* ── 3D & Viewport Enhancements ── */
iframe {{
    border-radius: var(--radius);
    border: 1px solid var(--rule) !important;
    background: transparent !important;
    box-shadow: var(--lift-sm);
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
}}
iframe:hover {{
    border-color: var(--pen) !important;
    box-shadow: var(--lift);
}}

/* ── Inputs ── */
[data-testid="stFileUploaderDropzone"] {{
    background: var(--sheet-alt); border: 2px dashed var(--rule); border-radius: var(--radius);
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--pen);
                                               background: var(--sheet); }}
[data-testid="stFileUploader"] button {{
    background: var(--sheet) !important; border: 1px solid var(--rule) !important; color: var(--ink) !important;
}}
[data-testid="stFileUploader"] button:hover {{
    background: var(--sheet-alt) !important; border-color: var(--pen) !important;
}}
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
    border-radius: 10px !important; border-color: var(--rule) !important;
    background: var(--sheet) !important; color: var(--ink) !important;
}}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {{
    color: var(--graphite) !important; opacity: 0.8 !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{ border-color: var(--pen) !important; }}

/* ── Stat tile ── */
.gauge {{ background: var(--sheet); border: 1px solid var(--rule);
         border-radius: var(--radius); box-shadow: var(--lift-sm);
         padding: 1rem 1.1rem; height: 100%; min-height: 96px; position: relative; }}
.gauge .v {{ font-family: var(--heading); font-size: 26px; font-weight: 700;
            line-height: 1.1; letter-spacing: -.01em; color: var(--ink);
            overflow-wrap: anywhere; }}
.gauge.long .v   {{ font-size: 19px; }}
.gauge.longer .v {{ font-size: 14.5px; line-height: 1.25; }}
.gauge .k {{ font-size: 12.5px; color: var(--graphite); margin-top: .4rem; font-weight: 600; }}
.gauge .s {{ font-size: 11.5px; color: var(--graphite); margin-top: 2px; }}
.gauge.flag {{ border-color: var(--risk); background: color-mix(in srgb, var(--risk) 8%, var(--sheet)); }}
.gauge.flag .v {{ color: var(--risk); }}

/* ── Callout cards ── */
.defect-stamp {{
    border: 1px solid var(--risk);
    background: color-mix(in srgb, var(--risk) 8%, var(--sheet));
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    margin: 1.2rem 0;
    box-shadow: var(--lift-sm);
}}
.defect-stamp .stamp-tag {{
    font-family: var(--sans); font-size: 12px; font-weight: 700;
    color: var(--risk); display: block; margin-bottom: 4px;
}}
.defect-stamp .stamp-title {{
    font-family: var(--heading); font-size: 18px; font-weight: 800; color: var(--risk);
    margin-bottom: 6px;
}}
.defect-stamp .stamp-desc {{
    font-size: 14.5px; line-height: 1.58; color: var(--ink); max-width: 68ch;
}}

.cert-stamp {{
    border: 1px solid var(--pen);
    background: color-mix(in srgb, var(--pen) 8%, var(--sheet));
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    margin: 1.2rem 0;
    box-shadow: var(--lift-sm);
}}
.cert-stamp .stamp-tag {{
    font-family: var(--sans); font-size: 12px; font-weight: 700;
    color: var(--pen); display: block; margin-bottom: 4px;
}}
.cert-stamp .stamp-title {{
    font-family: var(--heading); font-size: 18px; font-weight: 800; color: var(--pen);
    margin-bottom: 6px;
}}
.cert-stamp .stamp-desc {{
    font-size: 14.5px; line-height: 1.58; color: var(--ink); max-width: 68ch;
}}

/* ── Finding cards (Answers tab, IMPROVEMENTS.md 7.15) ── */
.finding-card {{
    background: var(--sheet); border: 1px solid var(--rule);
    border-left: 4px solid var(--pen); border-radius: var(--radius);
    box-shadow: var(--lift-sm); padding: 1rem 1.3rem; margin: 0 0 .9rem;
}}
.finding-headline {{
    font-family: var(--heading); font-size: 17px; font-weight: 700;
    color: var(--ink); line-height: 1.4;
}}

/* ── Trust strip (Answers tab) ── */
.trust-strip {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }}
.trust-cell {{
    background: var(--sheet); border: 1px solid var(--rule); border-radius: var(--radius);
    box-shadow: var(--lift-sm); padding: .9rem 1.2rem; flex: 1; min-width: 160px;
}}
.trust-cell .k {{ font-size: 12px; color: var(--graphite); font-weight: 600;
                  text-transform: uppercase; letter-spacing: .5px; }}
.trust-cell .v {{ font-family: var(--heading); font-size: 24px; font-weight: 800;
                  color: var(--ink); margin-top: .3rem; }}

/* ── How the agent read the data (Answers tab) ── */
.du {{
    background: var(--sheet); border: 1px solid var(--rule); border-radius: var(--radius);
    box-shadow: var(--lift-sm); padding: .9rem 1.2rem; margin: 0 0 1.5rem;
    font-size: 14.5px; line-height: 1.6; color: var(--ink); max-width: 74ch;
}}
.du .k {{ font-size: 12px; color: var(--graphite); font-weight: 600;
          text-transform: uppercase; letter-spacing: .5px; margin-right: .4rem; }}
.du .note {{ color: var(--graphite); }}

/* ── Sandbox isolation badge (Details tab) ── */
.iso-badge {{
    display: inline-block; font-family: var(--sans); font-size: 12px; font-weight: 700;
    padding: 3px 8px; border-radius: 4px; margin: 0 0 .75rem;
}}
.iso-badge.ok {{ color: var(--positive); background: color-mix(in srgb, var(--positive) 15%, transparent); }}
.iso-badge.warn {{ color: var(--risk); background: color-mix(in srgb, var(--risk) 15%, transparent); }}

/* ── KPI strip (technical-detail expander) — classes replace what used to be
   ~35 lines of inline style= per tile (IMPROVEMENTS.md 7.22) ── */
.kpi-row {{ display: flex; gap: 1.5rem; margin-bottom: 1.2rem; flex-wrap: wrap; }}
.kpi-gauge-card {{
    background: var(--sheet); padding: 1.5rem; border-radius: var(--radius); box-shadow: var(--lift-sm);
    flex: 1; min-width: 250px; display: flex; flex-direction: column; align-items: center;
    justify-content: center; position: relative; overflow: hidden;
}}
.kpi-gauge-label {{ font-size: 14px; color: var(--graphite); font-weight: 700;
                    margin-bottom: 1.5rem; text-transform: uppercase; letter-spacing: 1px; }}
.kpi-ring-wrap {{ position: relative; width: 140px; height: 140px;
                  display: flex; align-items: center; justify-content: center; }}
.kpi-ring-value {{ display: flex; flex-direction: column; align-items: center;
                   margin-top: 6px; z-index: 10; }}
.kpi-ring-num {{ font-family: var(--heading); font-size: 38px; font-weight: 800;
                 color: var(--ink); line-height: 1; }}
.kpi-ring-sub {{ font-size: 11px; font-weight: 700; color: var(--graphite);
                 text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }}
.kpi-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
              gap: 1rem; flex: 2; min-width: 300px; }}
.kpi-tile {{ background: var(--sheet); padding: 1.5rem; border-radius: var(--radius);
             box-shadow: var(--lift-sm); display: flex; flex-direction: column; justify-content: center; }}
.kpi-tile.flagged {{ border: 1px solid var(--risk); }}
.kpi-tile .k {{ font-size: 12px; color: var(--graphite); font-weight: 600;
               text-transform: uppercase; letter-spacing: .5px; }}
.kpi-tile .v {{ font-family: var(--heading); font-size: 24px; font-weight: 700;
               color: var(--ink); margin-top: 0.5rem; }}
.kpi-tile .v.big {{ font-size: 32px; font-weight: 800; margin-top: 0.2rem; }}
.kpi-tile .v.risk {{ color: var(--risk); }}
.kpi-tile .s {{ font-size: 12px; color: var(--graphite); margin-top: 4px; }}

/* ── Team grid & cards ── */
.agent-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 14px;
    margin: 1.2rem 0;
}}
.agent-card {{
    background: var(--sheet);
    border: 1px solid var(--rule);
    border-radius: var(--radius);
    box-shadow: var(--lift-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}
.agent-card:hover {{
    transform: translateY(-2px);
    box-shadow: var(--lift);
    border-color: var(--pen);
}}
.agent-card.agent-active {{
    border-color: var(--pen);
    background: color-mix(in srgb, var(--pen) 6%, var(--sheet));
    animation: pulseActive 2s infinite cubic-bezier(0.4, 0, 0.2, 1);
}}
@keyframes pulseActive {{
    0% {{ box-shadow: 0 0 0 0 rgba(240, 162, 74, 0.4); }}
    70% {{ box-shadow: 0 0 0 10px rgba(240, 162, 74, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(240, 162, 74, 0); }}
}}
.agent-card.agent-flagged {{
    border-color: var(--risk);
}}
.agent-grid.org-chart-layout {{
    display: grid;
    /* auto-fit/minmax, not a fixed column count + viewport media query —
       this grid can render inside a narrower container (e.g. the run
       progress rail), and a viewport-width breakpoint doesn't know that.
       Matches the same pattern .agent-grid/.kpi-tiles already use above. */
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 20px;
    position: relative;
    padding: 20px 0;
}}
.agent-grid.org-chart-layout::before {{
    content: '';
    position: absolute;
    top: 50%;
    left: 20px;
    right: 20px;
    height: 2px;
    background: var(--rule);
    z-index: 0;
    opacity: 0.5;
}}
.agent-grid.org-chart-layout .agent-card {{
    z-index: 1;
}}

.agent-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    padding-bottom: 7px;
    border-bottom: 1px solid var(--rule-faint);
}}
.agent-role {{
    font-family: var(--heading);
    font-weight: 700;
    font-size: 14.5px;
    color: var(--ink);
}}
.agent-badge {{
    font-family: var(--sans);
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .02em;
    padding: 3px 6px;
    border-radius: 4px;
}}
.agent-badge.done {{ color: var(--positive); background: color-mix(in srgb, var(--positive) 15%, transparent); }}
.agent-badge.running {{ color: var(--pen); background: color-mix(in srgb, var(--pen) 15%, transparent); }}
.agent-badge.error {{ color: var(--risk); background: color-mix(in srgb, var(--risk) 15%, transparent); }}

/* ── Bento Dashboard Layout ── */
.bento-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    align-items: start;
}}
.bento-card {{
    background: var(--sheet);
    border: 1px solid var(--rule);
    border-radius: var(--radius);
    box-shadow: var(--lift-sm);
    padding: 1.2rem;
    transition: box-shadow 0.2s ease;
}}
.bento-card:hover {{
    box-shadow: var(--lift);
}}
.bento-card.full-width {{
    grid-column: 1 / -1;
}}
.agent-desc {{
    font-size: 12.5px;
    line-height: 1.5;
    color: var(--graphite);
    margin: 5px 0 9px;
}}
.agent-metric {{
    font-family: var(--sans);
    font-size: 11.5px;
    font-weight: 600;
    color: var(--graphite);
    background: var(--sheet-alt);
    padding: 4px 8px;
    border-radius: 8px;
    display: inline-block;
}}

/* ── Handoff Stream Feed ── */
.handoff-stream {{
    margin: 1.5rem 0;
    border-left: 2px solid var(--rule);
    padding-left: 1.2rem;
}}
.handoff-item {{
    margin-bottom: 1rem;
    position: relative;
}}
.handoff-item::before {{
    content: "";
    position: absolute;
    left: calc(-1.2rem - 5px);
    top: 5px;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--pen);
}}
.handoff-meta {{
    font-family: var(--sans);
    font-size: 12px;
    color: var(--pen);
    font-weight: 700;
    margin-bottom: 2px;
}}
.handoff-text {{
    font-size: 14.5px;
    line-height: 1.55;
    color: var(--ink);
}}

/* ── Executive Directive ── */
.exec-directive {{
    background: var(--sheet);
    border: 1px solid var(--rule);
    border-left: 4px solid var(--pen);
    border-radius: var(--radius);
    box-shadow: var(--lift-sm);
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.5rem;
}}
.exec-directive .dir-label {{
    font-family: var(--sans);
    font-size: 12px;
    color: var(--pen);
    font-weight: 700;
    margin-bottom: 5px;
}}
.exec-directive .dir-content {{
    font-size: 15.5px;
    line-height: 1.65;
    color: var(--ink);
}}

/* ── Cards ── */
[data-testid="stExpander"] {{
    background: var(--sheet) !important; border: 1px solid var(--rule) !important;
    border-radius: var(--radius); box-shadow: var(--lift-sm);
}}
[data-testid="stExpander"] summary {{ font-weight: 700; font-size: 14.5px; color: var(--ink) !important; }}
[data-testid="stExpander"] summary:hover {{ color: var(--pen) !important; }}
[data-testid="stCode"] pre, pre {{
    background: var(--code-bg) !important; border: 1px solid var(--rule);
    border-radius: 10px; font-size: 12.5px; color: var(--ink) !important;
}}
[data-testid="stAlert"] {{ border-radius: var(--radius); }}
[data-testid="stAlertContainer"] {{
    background: var(--sheet-alt) !important; border-radius: var(--radius);
    border-left: 4px solid var(--graphite);
    padding: .6rem .8rem .6rem 1.1rem; color: var(--ink) !important;
}}
[data-testid="stAlertContainer"] p {{ color: inherit !important; font-size: 14.5px; }}
[data-testid="stAlertContainer"] svg {{ fill: currentColor; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
    border-left-color: var(--positive); color: var(--positive) !important;
}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]),
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
    border-left-color: var(--risk); color: var(--risk) !important;
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border-radius: var(--radius); box-shadow: var(--lift-sm); overflow: hidden;
}}

/* ── Steps list ── */
.sc {{ display: flex; align-items: center; gap: 12px;
      padding: .6rem .2rem; border-bottom: 1px solid var(--rule-faint);
      font-size: 14.5px; color: var(--graphite); }}
.sc .sc-num {{ font-family: var(--sans); font-size: 12px; font-weight: 700; color: var(--ink);
              flex: none; width: 1.8em; height: 1.8em; display: flex; align-items: center;
              justify-content: center; border-radius: 50%; background: var(--rule); border: 1px solid transparent; }}
.sc .nm {{ color: var(--ink); font-weight: 600; }}
.sc .detail {{ margin-left: auto; font-size: 12px;
              color: var(--graphite); text-align: right; padding-left: 1rem; }}
.sc.done  .sc-num {{ background: var(--pen); color: var(--sheet); }}
.sc.active .sc-num {{ background: var(--pen); color: var(--sheet); }}
.sc.active {{ background: color-mix(in srgb, var(--pen) 6%, transparent); border-radius: 10px; }}
.sc.active .nm::after {{ content: " — working"; font-weight: 400;
                        color: var(--pen); font-size: 12.5px; }}
.sc.skip  .sc-num {{ background: var(--rule); color: var(--graphite); }}
.sc.skip .nm {{ color: var(--graphite); font-weight: 400; }}
.sc.err   .sc-num {{ background: var(--risk); }}
.sc.err .nm {{ color: var(--risk); }}

/* ── Annotations ── */
.ic, .rc, .wc {{
    border-left: 3px solid var(--rule); border-radius: 0 10px 10px 0;
    padding: .5rem .8rem .5rem 1rem;
    margin: 0 0 .75rem; font-size: 15px; line-height: 1.6; max-width: 74ch;
    color: var(--ink); background: var(--sheet-alt);
}}
.ic {{ border-left-color: var(--graphite); }}
.rc {{ border-left-color: var(--pen); }}
.wc {{ border-left-color: var(--risk); color: var(--risk); background: color-mix(in srgb, var(--risk) 6%, var(--sheet-alt)); }}
.ic .mk, .rc .mk, .wc .mk {{
    font-family: var(--sans); font-size: 11.5px; font-weight: 700; color: var(--graphite);
    display: block; margin-bottom: 2px;
}}
.rc .mk {{ color: var(--pen); }}
.wc .mk {{ color: var(--risk); }}

.reason {{ background: var(--sheet); border: 1px solid var(--rule);
          border-radius: var(--radius); box-shadow: var(--lift-sm); padding: 1.3rem 1.5rem;
          font-size: 15.5px; color: var(--ink); line-height: 1.72; max-width: 72ch; }}

.run-banner {{ border-radius: var(--radius);
              background: color-mix(in srgb, var(--pen) 10%, var(--sheet)); padding: .8rem 1.1rem;
              color: var(--pen); font-size: 14.5px; font-weight: 700;
              margin: .4rem 0 1.2rem; }}
.run-banner .sub {{ display: block; font-weight: 500; color: var(--graphite);
                   font-size: 13px; margin-top: 2px; }}

.empty {{ padding: 3rem 0 3.5rem; max-width: 58ch; }}
.empty h2 {{ font-size: clamp(28px, 4vw, 42px); font-family: var(--heading);
            font-weight: 800; line-height: 1.05;
            margin: 0 0 1rem; }}
.empty p {{ color: var(--graphite); font-size: 16px; line-height: 1.62; margin: 0; }}
.empty .steps {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 2rem; }}
.empty .steps div {{ background: var(--sheet); border: 1px solid var(--rule);
                     border-radius: var(--radius-pill); padding: .4rem 1rem;
                     font-size: 13px; font-weight: 600; color: var(--graphite); }}

@media (max-width: 600px) {{
    .datum {{ flex-direction: column; }}
    .datum .cell {{ border-right: none; border-bottom: 1px solid var(--rule-faint); padding: .8rem 1.1rem; }}
    .datum .cell:last-child {{ border-bottom: none; }}
    .agent-grid {{ grid-template-columns: 1fr; }}
    .side-head {{ margin: 1.2rem 0 .5rem; }}
}}
</style>
""", unsafe_allow_html=True)


# ── Session-state initialisation ──────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "theme":          "day",
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
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Inject theme-aware CSS immediately (must run after session_state is ready) ─
_inject_theme_css()


# ── Constants ─────────────────────────────────────────────────────────────────
STAGE_DEFS = [
    ("1", "Reading Your File"),
    ("2", "Understanding Your Question"),
    ("3", "Running the Numbers"),
    ("4", "Making Sense of It"),
    ("5", "Double-Checking"),
    ("6", "Solving the Tricky Parts"),
    ("7", "Writing Your Report"),
]

# Shared Vega-Lite config — charts are plotted in the same warm ink palette
# as the rest of the console (DESIGN.md). The terracotta pen is the measured
# series; the warm red is reserved for the series that carries risk.
PLOT_INK = "#3a2b1e"
PLOT_GRAPHITE = "#8a7660"
PLOT_RULE = "#e4d4bc"
PEN_BLUE = "#a34f20"
PEN_RED = "#a33526"

def _get_vega_config() -> dict[str, Any]:
    """Ledger palette for Vega-Lite charts.

    Delegates to `src.core.chart_theme.vega_config` — the single source of
    truth for the palette (DESIGN.md's "Tokens — ink" table) — instead of
    hand-maintaining a second copy of the same hex codes (IMPROVEMENTS.md
    7.17). `dark` follows the same day/night session-state flag the rest of
    the app themes off of.
    """
    from src.core.chart_theme import vega_config as _vega_config_impl
    _dark = st.session_state.get("theme", "day") in ("night", "dark")
    return _vega_config_impl(dark=_dark)

VEGA_PLOT_CONFIG = _get_vega_config()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_pipeline() -> None:
    for k in ("stage_log", "analysis_done", "analysis_error",
              "final_report", "tool_results", "metadata", "profile",
              "dashboard", "tmp_dir", "progress_lines", "llm_warning"):
        st.session_state[k] = _DEFAULTS[k]


def _set_stage(num: str, status: str, detail: str = "") -> None:
    log: list[tuple[str, str, str]] = [
        e for e in st.session_state["stage_log"] if e[0] != num
    ]
    log.append((num, status, detail))
    st.session_state["stage_log"] = log


def _stage_card(num: str, name: str,
                status: str, detail: str = "") -> str:
    """One row of the stage ledger. Numbered: the pipeline is a real sequence."""
    cls = {"done": "done", "active": "active",
           "skipped": "skip", "error": "err"}.get(status, "")
    det = f'<span class="detail">{html.escape(str(detail))}</span>' if detail else ""
    aria = ' aria-live="polite"' if status == "active" else ""
    return (f'<div class="sc {cls}"{aria}>'
            f'<span class="sc-num">{num.zfill(2)}</span>'
            f'<span class="nm">{html.escape(str(name))}</span>{det}</div>')


def _render_steps_list(stage_log: list[tuple[str, str, str]]) -> str:
    """Render the primary, always-visible numbered steps list."""
    log_map = {n: (s, d) for n, s, d in stage_log}
    html_out = ['<div aria-live="polite" aria-atomic="false">']
    for num, name in STAGE_DEFS:
        status, detail = log_map.get(num, ("pending", ""))
        html_out.append(_stage_card(num, name, status, detail))
    html_out.append('</div>')
    return "".join(html_out)


def _datum(cells: list[tuple[str, str]]) -> str:
    """A ruled measurement bar. Each reading gets its own cell and hairline.

    Values can come straight from user input (e.g. the free-text "Custom
    model string" field feeding the "Model" cell) — HTML-escape both key
    and value before interpolating into markup rendered with
    unsafe_allow_html=True, or a value like
    `<img src=x onerror=alert(1)>` executes as-is (IMPROVEMENTS.md #10).
    """
    body = "".join(
        f'<div class="cell"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div></div>'
        for k, v in cells
    )
    return f'<div class="datum">{body}</div>'


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


def _gauge(label: str, value: str, sub: str = "", *, flag: bool = False) -> str:
    """One cell of the instrument readout: the number leads, the label follows.

    `flag` switches the cell to the risk pen — reserved for a measurement the
    reader should not trust, never used for emphasis.

    Values arrive at any length (a percentage, or a model name), so the type
    steps down rather than wrapping mid-word and pulling the strip's rules out
    of alignment.
    """
    value = str(value)
    fit = "" if len(value) <= 11 else " long" if len(value) <= 18 else " longer"
    value, label, sub = html.escape(value), html.escape(str(label)), html.escape(str(sub))
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    return (f'<div class="gauge{fit}{" flag" if flag else ""}">'
            f'<div class="v">{value}</div>'
            f'<div class="k">{label}</div>{sub_html}</div>')


def _gap_is_risky(gap: float) -> bool:
    """A train-test gap above 10 points means the model memorised the split.

    Uses the same threshold the training tool itself warns on
    (`src.tools.ml_pipeline.OVERFIT_THRESHOLD`, and the same strict `>`),
    so the KPI gauge and the comparison table's overfit_warnings can never
    disagree about the same model (IMPROVEMENTS.md #10).
    """
    from src.tools.ml_pipeline import OVERFIT_THRESHOLD
    return gap > OVERFIT_THRESHOLD


def _section(title: str, note: str = "", level: str = "h3") -> None:
    """Section head: the title sits on its own rule, with an optional mono note.

    No tracked-caps label floats above it — the rule is the structure.
    """
    note_html = f'<div class="note">{html.escape(str(note))}</div>' if note else ""
    st.markdown(
        f'<div class="sect"><{level}>{html.escape(str(title))}</{level}>{note_html}</div>',
        unsafe_allow_html=True,
    )


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any datetime/Timestamp columns to strings so PyArrow can serialise them."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)
        elif out[col].dtype == object:
            # Mixed types that may include Timestamps
            try:
                if out[col].dropna().apply(lambda x: hasattr(x, "strftime")).any():
                    out[col] = out[col].astype(str)
            except Exception:
                out[col] = out[col].astype(str)
    return out


def _find_tool(tool_results: list[dict[str, Any]],
               name: str) -> dict[str, Any] | None:
    for r in tool_results:
        if r.get("tool_name") == name and r.get("status") == "success":
            out = r.get("output")
            return out if isinstance(out, dict) else {}
    return None


#: Tool names already given a bespoke section elsewhere in the UI — the
#: generic renderer below only covers what's left, so a successful tool
#: result is never reachable *only* via the raw "Full Technical Log" JSON.
_BESPOKE_RENDERED_TOOLS = frozenset({
    "ingest_dataset", "clean_data", "detect_outliers", "correlation_analysis",
    "select_statistical_test", "train_model", "evaluate_model",
    "generate_report", "generate_visualizations", "planner",
})


def _render_other_findings(tool_results: list[dict[str, Any]]) -> None:
    """
    Fallback card for any successful tool result without a bespoke section
    (cluster_data, time_series_analysis, text_analysis, geospatial_analysis,
    dimensionality_analysis, and any future tool). Each already writes a
    well-formed sentence into output.summary; this surfaces that plus a few
    headline numbers per known shape instead of leaving the finding
    reachable only via the raw JSON log at the bottom of Downloads.
    """
    seen: set[str] = set()
    shown_any = False
    for r in tool_results:
        name = r.get("tool_name", "")
        if name in _BESPOKE_RENDERED_TOOLS or name in seen or r.get("status") != "success":
            continue
        out = r.get("output")
        if not isinstance(out, dict):
            continue
        seen.add(name)
        shown_any = True
        st.markdown(f"#### {name.replace('_', ' ').title()}")
        summary = out.get("summary")
        if summary:
            st.info(str(summary))

        if name == "cluster_data":
            c1, c2, c3 = st.columns(3)
            c1.metric("Clusters found", out.get("n_clusters", "—"))
            c2.metric("Silhouette score", out.get("silhouette_score", "—"))
            c3.metric("Separation", out.get("separation_quality", "—"))
        elif name == "time_series_analysis":
            c1, c2, c3 = st.columns(3)
            c1.metric("Trend", str(out.get("trend_direction", "—")).title())
            c2.metric("Stationary?", "Yes" if out.get("is_stationary") else "No")
            lags = out.get("seasonal_lags_detected") or []
            c3.metric("Seasonal lag(s)", ", ".join(str(x) for x in lags) or "None found")
        elif name == "text_analysis":
            c1, c2, c3 = st.columns(3)
            c1.metric("Vocabulary size", out.get("vocab_size", "—"))
            c2.metric("Avg. words / row", out.get("avg_word_count", "—"))
            top = out.get("top_tokens") or []
            words = [t.get("token", t) if isinstance(t, dict) else t for t in top[:6]]
            if words:
                st.caption("Most frequent words: " + ", ".join(str(w) for w in words))
        elif name == "geospatial_analysis":
            c1, c2 = st.columns(2)
            c1.metric("Points mapped", out.get("n_points", "—"))
            centroid = out.get("centroid") or {}
            if centroid:
                c2.metric("Centroid", f"{centroid.get('lat', '—')}, {centroid.get('lon', '—')}")
        elif name == "dimensionality_analysis":
            c1, c2, c3 = st.columns(3)
            c1.metric("Numeric features", out.get("n_features", "—"))
            threshold = out.get("variance_threshold")
            c2.metric(
                f"Components for {threshold:.0%} variance" if threshold else "Components needed",
                out.get("n_components_for_threshold", "—"),
            )
            pairs = out.get("high_correlation_pairs") or []
            c3.metric("Highly correlated pairs", len(pairs))

    if not shown_any:
        st.caption("No additional analyses ran for this dataset.")


def _render_data_understanding(du: dict[str, Any]) -> str:
    """The planner's first read of the dataset, as one compact card. Every
    value is LLM-written, so all of it is escaped."""
    def _row(label: str, value: Any) -> str:
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        if not value:
            return ""
        return f'<div><span class="k">{html.escape(label)}</span>{html.escape(str(value))}</div>'

    kind = " · ".join(str(du[k]) for k in ("domain", "archetype") if du.get(k))
    rows = "".join([
        _row("Subject", du.get("subject")),
        _row("Kind", kind),
        _row("Measures", du.get("key_measures")),
        _row("Dimensions", du.get("key_dimensions")),
        _row("Time", du.get("time_column")),
    ])
    caveats = "".join(
        f'<div class="note">⚠ {html.escape(str(c))}</div>' for c in du.get("caveats") or []
    )
    if not rows and not caveats:
        return ""
    return f'<div class="du">{rows}{caveats}</div>'


def _read_audit_log(path: str | None) -> list[dict[str, Any]]:
    """Audit records from the append-only JSONL; unreadable lines are skipped."""
    if not path:
        return []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _render_governance(gov: dict[str, Any]) -> None:
    """Governance summary and the code-execution audit trail for this run.
    Audit code is shown with st.code only — it is LLM-authored."""
    budget = gov.get("execution_budget")
    cells = [
        ("Code runs", f"{gov.get('code_executions', 0)} / {budget}" if budget is not None
         else str(gov.get("code_executions", 0)), False),
        ("Failed", str(gov.get("code_failures", 0)), bool(gov.get("code_failures"))),
        ("Refused", str(gov.get("code_refusals", 0)), bool(gov.get("code_refusals"))),
    ]
    if gov.get("llm_calls") is not None:
        cells.append(("LLM calls", f"{gov['llm_calls']:,}", False))
    if gov.get("llm_tokens") is not None:
        cap = gov.get("llm_token_cap") or 0
        cells.append(("LLM tokens", f"{gov['llm_tokens']:,}" + (f" / {cap:,}" if cap else ""),
                      bool(cap) and gov["llm_tokens"] >= cap))
    for col, (label, value, flag) in zip(st.columns(len(cells)), cells, strict=True):
        col.markdown(_gauge(label, value, flag=flag), unsafe_allow_html=True)

    if not gov.get("code_execution_enabled", True):
        st.caption("AI-written code was switched off for this run.")
    backends = [b for b in gov.get("sandbox_backends") or [] if b != "refused"]
    if "subprocess" in backends:
        note = " — some runs used Docker" if "docker" in backends else ""
        st.markdown(
            '<span class="iso-badge warn">Process-level isolation only (subprocess)'
            f'{note}</span>',
            unsafe_allow_html=True,
        )
        st.caption("Set SANDBOX_BACKEND=docker, or turn on \"Require container isolation\", "
                   "for a kernel boundary around AI-written code.")
    elif "docker" in backends:
        st.markdown('<span class="iso-badge ok">Container isolation (Docker)</span>',
                    unsafe_allow_html=True)

    entries = _read_audit_log(gov.get("audit_log"))
    if not entries:
        st.caption("No AI-written code ran or was refused in this run."
                   if not gov.get("audit_log") else "The audit log could not be read.")
        return

    def _ms(value: Any) -> str:
        return f"{value:,.0f}" if isinstance(value, (int, float)) else "—"

    with st.expander(f"Audit log — {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}"):
        st.dataframe(_safe_df(pd.DataFrame([{
            "Time": str(e.get("timestamp") or "")[:19].replace("T", " "),
            "Tool": e.get("tool_name"),
            "Status": e.get("status"),
            "Backend": e.get("backend") or "—",
            "Duration (ms)": _ms(e.get("duration_ms")),
            "SHA-256": (e.get("code_sha256") or "")[:12] or "—",
        } for e in entries])), width='stretch')
        st.caption(f"Full record: {gov.get('audit_log')}")

    for i, e in enumerate(entries, 1):
        sha = (e.get("code_sha256") or "")[:12]
        with st.expander(f"#{i:02d} `{e.get('tool_name', '?')}` · {e.get('status', '?')}"
                         + (f" · `{sha}`" if sha else "")):
            if e.get("code"):
                st.code(e["code"], language="python")
            if e.get("error"):
                st.code(e["error"], language=None)


def _render_defect_stamp(gap_val: float | None) -> str:
    """Render authentic engineering defect stamp or certification seal for generalization."""
    if gap_val is None:
        return ""
    if _gap_is_risky(gap_val):
        return f"""
        <div class="defect-stamp">
            <span class="stamp-tag">⚠ Heads up — this might not hold up</span>
            <div class="stamp-title">THE MODEL MEMORISED THE EXAMPLES ({gap_val*100:.1f}% GAP)</div>
            <div class="stamp-desc">
                It did noticeably better on the data it trained on than on data it hadn't seen —
                a sign it memorised quirks rather than learning the real pattern.
                We've ranked it lower because of this.
            </div>
        </div>
        """
    else:
        return f"""
        <div class="cert-stamp">
            <span class="stamp-tag">✓ Good news — this should hold up</span>
            <div class="stamp-title">THE MODEL PERFORMED CONSISTENTLY ({gap_val*100:.1f}% GAP)</div>
            <div class="stamp-desc">
                It did about as well on new data as on the data it trained on —
                a good sign the pattern it found is real, not a fluke.
            </div>
        </div>
        """


def _find_chart_by_id(dash: list[dict[str, Any]] | None, chart_id: str) -> dict[str, Any] | None:
    """Look up a dashboard panel by its chart_id — lets a surface reuse a
    panel the backend already built instead of re-deriving the same chart
    (IMPROVEMENTS.md 7.17)."""
    for ch in (dash or []):
        if ch.get("chart_id") == chart_id:
            return ch
    return None


def _render_dashboard_chart(ch: dict[str, Any], vega_cfg: dict[str, Any]) -> None:
    """Render one dashboard panel (title, chart, caption/description) — the
    one place a ChartSpec dict becomes Streamlit markup, shared by the
    Charts tab and the Answers tab's technical-detail section so neither
    hand-rolls its own copy (IMPROVEMENTS.md 7.17).

    Prefers the finding's own `caption` (its headline sentence) over the
    chart's generic `description` when both are present — the caption is
    the "narrated dashboard" payoff of the finding-chart binding (7.18).
    """
    with st.container(border=True):
        st.markdown(
            f"<div style='font-family: var(--heading); font-size: 1.1rem; font-weight: 700; "
            f"color: var(--ink); margin-bottom: 0.5rem;'>{html.escape(str(ch.get('title', '')))}</div>",
            unsafe_allow_html=True,
        )
        _spec = dict(ch.get("spec", {}))
        _spec.setdefault("background", "transparent")
        _spec.setdefault("config", vega_cfg)
        st.vega_lite_chart(_spec, width='stretch')
        _caption = ch.get("caption")
        if _caption:
            st.caption(str(_caption))
        elif ch.get("description"):
            st.markdown(
                f"<div style='font-size: 0.9rem; color: var(--graphite); margin-top: 0.5rem; "
                f"line-height: 1.4;'>{html.escape(str(ch['description']))}</div>",
                unsafe_allow_html=True,
            )


def _render_finding_card(finding: dict[str, Any], chart_finding_ids: set[str]) -> str:
    """One full-width sentence card for a top-ranked Finding (IMPROVEMENTS.md
    7.15) — the headline carries the real number, an optional detail/caveat
    line sits underneath, and a light text cross-reference points at the
    chart that visualizes it, if any (7.18) — no anchor-scroll mechanism,
    just an honest pointer to the Charts tab.
    """
    headline = html.escape(plainify(str(finding.get("headline", ""))))
    detail = finding.get("detail")
    caveats = finding.get("caveats") or []
    sub_text = (
        plainify(str(detail)) if detail
        else plainify(str(caveats[0])) if caveats
        else describe_uncertainty(finding) or ""
    )
    sub_html = (
        f'<div style="font-size:13px;color:var(--graphite);margin-top:4px;">{html.escape(sub_text)}</div>'
        if sub_text else ""
    )
    xref = ""
    if finding.get("finding_id") and finding["finding_id"] in chart_finding_ids:
        xref = '<div style="font-size:12px;color:var(--pen);margin-top:6px;font-weight:600;">→ see chart in the Charts tab</div>'
    return (
        '<div class="finding-card">'
        f'<div class="finding-headline">{headline}</div>'
        f'{sub_html}{xref}'
        '</div>'
    )


def _search_findings(query: str, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cheap, deterministic keyword search over the finding bus for the
    'Ask a follow-up question' box (IMPROVEMENTS.md 7.20, scoped down — no
    new tool calls, no LLM call, purely matching fields already on hand)."""
    terms = [t for t in query.lower().split() if t]
    if not terms:
        return []
    scored: list[tuple[int, float, dict[str, Any]]] = []
    for f in findings:
        haystack = " ".join(
            str(f.get(k) or "") for k in ("headline", "measure", "dimension", "detail")
        ).lower()
        hits = sum(1 for t in terms if t in haystack)
        if hits:
            scored.append((hits, float(f.get("importance") or 0.0), f))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [f for _, _, f in scored[:3]]


def _render_agent_grid(
    stage_log: list[tuple[str, str, str]],
    tool_results: list[dict[str, Any]] | None = None,
    report: dict[str, Any] | None = None
) -> str:
    """Render visual architecture cards for the autonomous multi-agent teamwork roster."""
    log_map = {n: s for n, s, _ in stage_log}
    tool_results = tool_results or []
    report = report or {}

    # SVG simple line icons using --ink or --pen styling (currentColor)
    i_compass = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>'
    i_shield = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
    i_ruler = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.3 15.3l-7.6-7.6a2 2 0 0 0-2.8 0l-1.6 1.6a2 2 0 0 0 0 2.8l7.6 7.6c.8.8 2 .8 2.8 0l1.6-1.6a2 2 0 0 0 0-2.8Z"/><path d="m14.5 12.5 2-2"/><path d="m11.5 9.5 2-2"/><path d="m8.5 6.5 2-2"/><path d="m17.5 15.5 2-2"/></svg>'
    i_zap = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'
    i_search = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>'
    i_repeat = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>'
    i_globe = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>'
    i_chart = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>'

    agents = [
        {
            "icon": i_compass,
            "name": "Planner",
            "role": "Plans the approach",
            "desc": "Reads your question and breaks it into a step-by-step plan, then decides when enough checking has been done.",
            "stage": "2",
            "tool": "Reasoning",
        },
        {
            "icon": i_shield,
            "name": "File Checker",
            "role": "Checks your file is safe and healthy",
            "desc": "Makes sure your file is safe to open, figures out what each column means, spots anything unusual, and gives your data a health score out of 100.",
            "stage": "1",
            "tool": "Checks & cleans",
        },
        {
            "icon": i_ruler,
            "name": "Fact-Checker",
            "role": "Tests what's actually true",
            "desc": "Runs the right statistical tests to check whether a pattern is real or could just be chance, and finds which columns move together.",
            "stage": "3",
            "tool": "Statistical tests",
        },
        {
            "icon": i_zap,
            "name": "Model Builder",
            "role": "Builds and tests prediction models",
            "desc": "Trains several different prediction models and tests each one on different slices of your data, so a lucky guess doesn't get mistaken for a good model.",
            "stage": "3",
            "tool": "Model training",
        },
        {
            "icon": i_search,
            "name": "Reality-Checker",
            "role": "Catches models that just memorised",
            "desc": "Compares how each model does on data it trained on versus data it's never seen. If a model only looks good because it memorised the examples, this agent flags it and marks it down.",
            "stage": "4",
            "tool": "Model checking",
        },
        {
            "icon": i_repeat,
            "name": "Double-Checker",
            "role": "Goes back for another pass",
            "desc": "Looks at what's been found so far, and if there are loose ends or your question isn't fully answered yet, sends the work back for another round.",
            "stage": "5",
            "tool": "Another pass",
        },
        {
            "icon": i_globe,
            "name": "Detail Handler",
            "role": "Handles the tricky, many-part questions",
            "desc": "When a question has too many moving parts to answer in one go, this splits it into smaller pieces, solves each one separately, and brings the answers back together.",
            "stage": "6",
            "tool": "Splitting up work",
        },
        {
            "icon": i_chart,
            "name": "Report Writer",
            "role": "Builds your charts and report",
            "desc": "Builds charts that fit your data, then puts everything together into the report you can download and share.",
            "stage": "7",
            "tool": "Charts & report",
        },
    ]

    cards_html = []
    for ag in agents:
        st_val = log_map.get(ag["stage"], "pending")
        if st_val == "done":
            badge_cls = "done"
            badge_txt = "Completed"
            card_cls = "agent-card"
        elif st_val == "active":
            badge_cls = "running"
            badge_txt = "Executing"
            card_cls = "agent-card agent-active"
        elif st_val == "error":
            badge_cls = "error"
            badge_txt = "Flagged"
            card_cls = "agent-card agent-flagged"
        else:
            badge_cls = ""
            badge_txt = "Standby"
            card_cls = "agent-card"

        # Determine dynamic output for the detail panel
        agent_output = ""
        if ag["name"] == "Planner":
            agent_output = report.get("reasoning", "Waiting for a plan.")
        elif ag["name"] == "File Checker":
            _prof = st.session_state.get('profile') or {}
            agent_output = f"Health score {_prof.get('quality_score', '—')}/100. Cleaned up any issues found."
        elif ag["name"] == "Fact-Checker":
            agent_output = "Tests complete: checked which columns move together and whether the differences are real."
        elif ag["name"] == "Model Builder":
            agent_output = f"Best model so far: {report.get('best_model', 'N/A')}. Tested multiple times on different slices of your data."
        elif ag["name"] == "Reality-Checker":
            agent_output = "Checked every model for memorisation. Applied a penalty to any that didn't hold up."
        elif ag["name"] == "Double-Checker":
            agent_output = "Finished reviewing — went back for more passes where needed."
        elif ag["name"] == "Detail Handler":
            agent_output = f"{len(report.get('rlm_sub_results', []))} smaller questions solved separately and combined."
        elif ag["name"] == "Report Writer":
            agent_output = "Report finished, with charts, key findings, and what to do next."

        cards_html.append(f"""
        <div class="{card_cls}" style="position: relative; overflow: hidden; display: flex; flex-direction: column;">
            <details style="padding: 1rem; cursor: pointer; width: 100%;">
                <summary style="list-style: none; display: flex; flex-direction: column; outline: none;">
                    <div class="agent-header" style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                        <span class="agent-role" style="font-weight: 700; color: var(--ink); display: flex; align-items: center; gap: 8px;">{ag['icon']} {ag['name']}</span>
                        <span class="agent-badge {badge_cls}">[{badge_txt}]</span>
                    </div>
                    <div class="agent-desc" style="margin-top: 0.5rem; color: var(--graphite); font-size: 0.95rem;">{ag['desc']}</div>
                    <div class="agent-metric" style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--graphite); font-weight: 600;">Role: {ag['role']} · Tool: {ag['tool']}</div>
                </summary>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px dashed var(--rule); font-size: 0.95rem;">
                    <div style="margin-bottom: 0.5rem;"><strong>Rule:</strong> {ag['desc']}</div>
                    <div style="color: var(--pen); font-weight: 600;"><strong>Found:</strong> {html.escape(str(agent_output))}</div>
                </div>
            </details>
        </div>
        """.strip())

    return f'<div class="agent-grid org-chart-layout">{"".join(cards_html)}</div>'


def _render_handoff_stream(progress_lines: list[str], tool_results: list[dict[str, Any]]) -> str:
    """Render timeline feed of inter-agent messages and handoffs."""
    items_html = []
    if progress_lines:
        for line in progress_lines[:20]:
            if not line.strip():
                continue
            meta = "Note"
            if "[done]" in line:
                meta = "Done"
            elif "[run ]" in line:
                meta = "Started"
            elif "ok" in line:
                meta = "Finished"
            clean_text = line.replace("[done]", "").replace("[run ]", "").replace("[    ]", "").replace("[fail]", "⚠ ").strip()
            items_html.append(f"""
            <div class="handoff-item">
                <div class="handoff-meta">{meta}</div>
                <div class="handoff-text">{html.escape(clean_text)}</div>
            </div>
            """.strip())
    elif tool_results:
        for r in tool_results:
            name = r.get("tool_name", "Step")
            status = r.get("status", "success")
            summary = r.get("output", {}).get("summary", "") or r.get("error", "")
            time_ms = r.get("execution_time_ms", 0)
            items_html.append(f"""
            <div class="handoff-item">
                <div class="handoff-meta">{html.escape(str(name))} · {html.escape(str(status))} · {time_ms:.0f}ms</div>
                <div class="handoff-text">{html.escape(str(summary))}</div>
            </div>
            """.strip())
    else:
        items_html.append("""
        <div class="handoff-item">
            <div class="handoff-meta">Waiting</div>
            <div class="handoff-text">Your helpers are ready. Upload a file to get started.</div>
        </div>
        """.strip())
    return f'<div class="handoff-stream">{"".join(items_html)}</div>'


def _render_agent_deep_dive(agent_name: str, tool_results: list[dict[str, Any]], report: dict[str, Any]) -> None:
    """Render structured details for an inspected agent persona."""
    details = {
        "🧭 Planner": {
            "mission": "Reads your question and turns it into a step-by-step plan — what to check first, what to try next, and when the plan needs adjusting.",
            "directive": "Only works from summaries and statistics, never your raw data rows — the way a manager works from a report rather than the raw ledger.",
            "tools": "Reasoning and planning",
            "output": report.get("reasoning", "Waiting for a plan."),
        },
        "🛡️ File Checker": {
            "mission": "Checks your file is safe to open, figures out what each column means, and gives your data a health score.",
            "directive": "Scores your data 0–100 based on missing values, duplicate rows, and anything that looks off.",
            "tools": "File safety checks, data profiling",
            "output": f"Health score {st.session_state.get('profile', {}).get('quality_score', '—')}/100. Cleaned up any issues found.",
        },
        "📐 Fact-Checker": {
            "mission": "Runs statistical tests to check whether a pattern in your data is real, or could just be chance.",
            "directive": "Checks how your data is shaped before picking which test is fair to use — the right test depends on the shape.",
            "tools": "Statistical tests, correlation checks",
            "output": "Tests complete: checked which columns move together and whether the differences are real.",
        },
        "⚡ Model Builder": {
            "mission": "Trains a few different prediction models on your data and scores each one.",
            "directive": "Tests every model on several different slices of the data, not just one, so a lucky split doesn't make a bad model look good.",
            "tools": "Model training (several approaches, tested against each other)",
            "output": f"Best model so far: {report.get('best_model', 'N/A')}. Tested multiple times on different slices of your data.",
        },
        "🔍 Reality-Checker": {
            "mission": "Compares how each model performs on data it trained on versus data it's never seen.",
            "directive": "If a model does noticeably better on familiar data than new data, it's flagged as having memorised rather than learned — and marked down.",
            "tools": "Model checking",
            "output": "Checked every model for memorisation. Applied a penalty to any that didn't hold up.",
        },
        "🔁 Double-Checker": {
            "mission": "Looks at what's been found so far and decides whether your question has really been answered.",
            "directive": "Sends the work back for another pass if things haven't settled down yet, up to a set limit of tries.",
            "tools": "Review and another pass",
            "output": "Finished reviewing — went back for more passes where needed.",
        },
        "🌐 Detail Handler": {
            "mission": "Splits a big, many-part question into smaller pieces, solves each on its own, then brings the answers back together.",
            "directive": "Keeps each piece small and separate, so a complicated question doesn't overwhelm any single step.",
            "tools": "Splitting up and recombining work",
            "output": f"{len(report.get('rlm_sub_results', []))} smaller questions solved separately and combined.",
        },
        "📊 Report Writer": {
            "mission": "Builds charts that fit your data and puts everything into a report you can download and share.",
            "directive": "Uses the same easy-to-read style for the charts and the report as the rest of the app, and gives you both a written version and a webpage version.",
            "tools": "Charts and report writing",
            "output": "Report finished, with charts, key findings, and what to do next.",
        },
    }
    info = details.get(agent_name, details["🧭 Planner"])
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.markdown(f"**What it does:** {info['mission']}")
        st.markdown(f"**Its rule:** {info['directive']}")
    with c2:
        st.markdown(f"**What it uses:** {info['tools']}")
        st.markdown(f"**What it found:** {info['output']}")


def _load_teamwork_preview() -> None:
    """Populate full autonomous multi-agent teamwork demo with sample customer churn data."""
    _reset_pipeline()
    sample_path = ROOT / "data" / "sample_customer_churn.csv"
    if sample_path.exists():
        raw_bytes = sample_path.read_bytes()
        df, _ = read_any(str(sample_path))
    else:
        import numpy as np
        np.random.seed(42)
        df = pd.DataFrame({
            "tenure": np.random.randint(1, 72, 100),
            "monthly_charges": np.random.uniform(20, 120, 100).round(2),
            "total_charges": np.random.uniform(100, 8000, 100).round(2),
            "contract": np.random.choice(["Month-to-month", "One year", "Two year"], 100),
            "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], 100),
            "payment_method": np.random.choice(["Electronic check", "Mailed check", "Bank transfer"], 100),
            "churn": np.random.choice([0, 1], 100, p=[0.73, 0.27]),
        })
        raw_bytes = df.to_csv(index=False).encode("utf-8")

    st.session_state["preview_df"] = df
    st.session_state["preview_name"] = "sample_customer_churn.csv"
    st.session_state["orig_name"] = "sample_customer_churn.csv"
    st.session_state["preview_bytes"] = raw_bytes
    st.session_state["from_uploader"] = False

    tmp = tempfile.mkdtemp()
    st.session_state["tmp_dir"] = tmp
    out_dir = Path(tmp) / "output"
    rep_dir = out_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    st.session_state["stage_log"] = [
        ("1", "done", "100 rows × 8 cols · task=classification · target=churn"),
        ("2", "done", "5 steps planned by the Planner"),
        ("3", "done", "6 tools executed: clean, outliers, corr, test, train, eval"),
        ("4", "done", "Anti-overfit audit passed: gap 4.2% < 10%"),
        ("5", "done", "Converged in 2 iterations (residual variance resolved)"),
        ("6", "done", "2 RLM sub-tasks offloaded via REPL context"),
        ("7", "done", "analysis_report.md & report.html compiled"),
    ]

    st.session_state["tool_results"] = [
        {
            "tool_name": "clean_data",
            "status": "success",
            "execution_time_ms": 42.0,
            "output": {
                "summary": "Cleaned dataset: 0 missing values found. Handled numeric types and standardized categorical levels.",
                "strategy_used": "median",
                "missing_before": 0,
                "missing_after": 0,
            },
        },
        {
            "tool_name": "detect_outliers",
            "status": "success",
            "execution_time_ms": 58.0,
            "output": {
                "summary": "Detected 4 outlier rows across total_charges using IQR method (3.0 threshold). Kept in dataset.",
                "total_outliers": 4,
                "outlier_percentage": 4.0,
                "per_column_outliers": {"total_charges": 4, "monthly_charges": 0, "tenure": 0},
            },
        },
        {
            "tool_name": "correlation_analysis",
            "status": "success",
            "execution_time_ms": 85.0,
            "output": {
                "summary": "Identified strongest correlation pairs with churn: tenure (-0.35) and monthly_charges (+0.28).",
                "top_correlations": [
                    {"col_a": "tenure", "col_b": "churn", "correlation": -0.352},
                    {"col_a": "monthly_charges", "col_b": "churn", "correlation": 0.284},
                    {"col_a": "monthly_charges", "col_b": "total_charges", "correlation": 0.651},
                    {"col_a": "tenure", "col_b": "total_charges", "correlation": 0.824},
                ],
            },
        },
        {
            "tool_name": "select_statistical_test",
            "status": "success",
            "execution_time_ms": 36.0,
            "output": {
                "summary": "Mann-Whitney U test confirmed statistically significant tenure difference between churners and retainers (p=0.0004).",
                "test_name": "Mann-Whitney U Test",
                "p_value": 0.00041,
                "significant": True,
                "interpretation": "Tenure of churned customers is significantly lower than retained customers (median 10 mos vs 38 mos, p < 0.001).",
            },
        },
        {
            "tool_name": "train_model",
            "status": "success",
            "execution_time_ms": 320.0,
            "output": {
                "summary": "Trained 3 stratified 5-fold models. Random Forest achieved highest CV accuracy (81.0% ± 3.2%).",
                "task_type": "classification",
                "best_model": "RandomForestClassifier",
                "n_cv_folds": 5,
                "test_size": 0.2,
                "overfit_warnings": [],
                "models_trained": {
                    "RandomForestClassifier": {
                        "cv_mean": 0.810,
                        "cv_std": 0.032,
                        "train_metrics": {"accuracy": 0.852, "f1_score": 0.840},
                        "test_metrics": {"accuracy": 0.810, "f1_score": 0.795},
                        "train_test_gap": 0.042,
                    },
                    "LogisticRegression": {
                        "cv_mean": 0.790,
                        "cv_std": 0.028,
                        "train_metrics": {"accuracy": 0.800, "f1_score": 0.772},
                        "test_metrics": {"accuracy": 0.780, "f1_score": 0.760},
                        "train_test_gap": 0.020,
                    },
                    "GradientBoostingClassifier": {
                        "cv_mean": 0.775,
                        "cv_std": 0.035,
                        "train_metrics": {"accuracy": 0.885, "f1_score": 0.871},
                        "test_metrics": {"accuracy": 0.760, "f1_score": 0.735},
                        "train_test_gap": 0.125,
                    },
                },
            },
        },
        {
            "tool_name": "evaluate_model",
            "status": "success",
            "execution_time_ms": 48.0,
            "output": {
                "summary": "Model evaluation complete. Precision 0.82, Recall 0.79 for Retained (0); Precision 0.78, Recall 0.74 for Churned (1).",
                "classification_report": {
                    "Retained (0)": {"precision": 0.824, "recall": 0.795, "f1-score": 0.809, "support": 15},
                    "Churned (1)": {"precision": 0.780, "recall": 0.740, "f1-score": 0.759, "support": 5},
                },
            },
        },
    ]

    st.session_state["profile"] = {
        "quality_score": 92,
        "duplicate_rows": 0,
        "memory_mb": 0.12,
        "column_count": len(df.columns),
        "columns": [
            {"name": c, "kind": "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical",
             "dtype": str(df[c].dtype), "missing_pct": 0.0, "nunique": int(df[c].nunique()), "flags": []}
            for c in df.columns
        ],
        "warnings": [],
    }

    st.session_state["dashboard"] = [
        {
            "chart_id": "model_comparison",
            "title": "Cross-Validation Accuracy vs Generalization Gap",
            "description": "Comparison of models ranking by 5-fold CV score and train-test gap to penalize memorization.",
            "spec": {
                "mark": "bar",
                "data": {"values": [
                    {"Model": "Random Forest", "Metric": "CV Accuracy", "Score": 81.0},
                    {"Model": "Random Forest", "Metric": "Generalization Gap", "Score": 4.2},
                    {"Model": "Logistic Regression", "Metric": "CV Accuracy", "Score": 79.0},
                    {"Model": "Logistic Regression", "Metric": "Generalization Gap", "Score": 2.0},
                    {"Model": "Gradient Boosting", "Metric": "CV Accuracy", "Score": 77.5},
                    {"Model": "Gradient Boosting", "Metric": "Generalization Gap", "Score": 12.5},
                ]},
                "encoding": {
                    "x": {"field": "Model", "type": "nominal", "axis": {"labelAngle": 0}},
                    "xOffset": {"field": "Metric"},
                    "y": {"field": "Score", "type": "quantitative", "title": "Percentage (%)"},
                    "color": {"field": "Metric", "type": "nominal"},
                },
            },
        },
        {
            "chart_id": "top_correlations",
            "title": "Key Drivers: Feature Correlation with Customer Churn",
            "description": "Tenure exhibits strong protective negative correlation (-0.35), while high monthly charges drive churn (+0.28).",
            "spec": {
                "mark": "bar",
                "data": {"values": [
                    {"Feature": "Tenure (Months)", "Correlation": -0.352},
                    {"Feature": "Monthly Charges", "Correlation": 0.284},
                    {"Feature": "Paperless Billing", "Correlation": 0.175},
                    {"Feature": "Total Charges", "Correlation": -0.198},
                ]},
                "encoding": {
                    "y": {"field": "Feature", "type": "nominal", "sort": "-x"},
                    "x": {"field": "Correlation", "type": "quantitative", "scale": {"domain": [-0.5, 0.5]}},
                    "color": {
                        "condition": {"test": "datum.Correlation >= 0", "value": "#a34f20"},
                        "value": "#8a7660",
                    },
                },
            },
        },
    ]

    st.session_state["final_report"] = {
        "best_model": "RandomForestClassifier",
        "reasoning": (
            "Customer churn is primarily driven by tenure length and high monthly billing tiers. "
            "Customers on month-to-month contracts with tenure < 12 months exhibit a 44% higher probability of churning. "
            "The Random Forest model demonstrated superior cross-validated generalization (81.0% accuracy, train-test gap 4.2%), "
            "comfortably satisfying the 10% anti-overfitting safety threshold."
        ),
        "insights": [
            "Early-tenure vulnerability: First 12 months account for 68% of all churn instances.",
            "Billing sensitivity: Accounts paying over $75/mo without fiber reliability churn at 2.3× baseline.",
            "Contractual resilience: Annual and two-year agreements reduce churn by 78% relative to monthly contracts.",
        ],
        "recommendations": [
            "Deploy targeted 90-day onboarding incentives for high-charge month-to-month cohorts.",
            "Offer contract term upgrades with bundled savings prior to the critical 6-month drop-off cliff.",
            "Route at-risk accounts identified by the Random Forest model to proactive retention concierges.",
        ],
        "rlm_sub_results": [
            {
                "task_name": "Tenure Stratification Analysis",
                "query": "Quantify churn hazard rate across 0-6mo, 6-12mo, and 12-24mo cohorts",
                "finding": "Hazard rate peaks at month 4 (31.2% hazard rate), dropping to 4.1% past month 24.",
            },
            {
                "task_name": "Billing Tier Elasticity",
                "query": "Evaluate elasticity between monthly charge increments and churn probability",
                "finding": "Every $10 increase above $65/mo produces an incremental 4.8% churn risk.",
            },
        ],
    }

    st.session_state["progress_lines"] = [
        "[done] Stage 1: Reading Your File  100 rows × 8 cols · task=classification · target=churn",
        "[run ] Iteration 1: model reasoning",
        "[done] Stage 2: Understanding Your Question  5 steps planned by the Planner",
        "       ok  clean_data: median imputation on missing numeric cells",
        "       ok  detect_outliers: 4 anomaly rows flagged via IQR",
        "       ok  correlation_analysis: mapped feature associations with churn",
        "       ok  select_statistical_test: Mann-Whitney U test (p=0.0004)",
        "       ok  train_model: 5-fold CV on RandomForest, LogisticRegression, GradientBoosting",
        "       ok  evaluate_model: Reality-Checker confirmed train-test gap 4.2% < 10% [Certified]",
        "[done] Stage 3: Running the Numbers  6 tools executed successfully",
        "[run ] Iteration 1: interpreting and refining",
        "[done] Stage 4: Making Sense of It  Key drivers tenure and charges synthesized",
        "[done] Stage 5: Double-Checking  Loop converged in 2 iterations",
        "[done] Stage 6: Solving the Tricky Parts  2 sub-tasks offloaded via REPL context",
        "[done] Stage 7: Writing Your Report  Certified markdown and HTML dossier published",
    ]

    md_content = f"# Executive Analytical Dossier: Customer Churn Analysis\\n\\n{st.session_state['final_report']['reasoning']}\\n\\n## Recommendations\\n- " + "\\n- ".join(st.session_state['final_report']['recommendations'])
    (rep_dir / "analysis_report.md").write_text(md_content, encoding="utf-8")
    (rep_dir / "report.html").write_text("<html><body>" + md_content + "</body></html>", encoding="utf-8")
    (rep_dir / "final_report.json").write_text(json.dumps(st.session_state["final_report"], indent=2), encoding="utf-8")

    st.session_state["analysis_done"] = True


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


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
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

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="side-head">Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "CSV, TSV or Excel",
        type=sorted(ext.lstrip(".") for ext in ALLOWED_EXTENSIONS),
        label_visibility="collapsed",
    )

    if uploaded is None and st.session_state["preview_df"] is None:
        if st.button("📂 Try a sample dataset", width='stretch'):
            sample_path = ROOT / "data" / "sample_customer_churn.csv"
            if sample_path.exists():
                st.session_state["preview_df"], _ = read_any(str(sample_path))
                st.session_state["preview_name"] = "sample_customer_churn.csv"
                st.session_state["orig_name"] = "sample_customer_churn.csv"
                st.session_state["preview_bytes"] = sample_path.read_bytes()
                st.session_state["from_uploader"] = False
                st.rerun()

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
                    for _note in _prep.notes:
                        st.caption(f"⚠ {_note}")
                except Exception as _e:
                    st.session_state["preview_df"] = None
                    st.error(f"Could not read file: {_e}")
                st.session_state["from_uploader"] = True
    else:
        if st.session_state.get("orig_name") and st.session_state.get("from_uploader"):

            _theme = st.session_state.get("theme", "day")
            for _k2, _v2 in _DEFAULTS.items():
                st.session_state[_k2] = _v2
            st.session_state["theme"] = _theme
            st.session_state["from_uploader"] = False

    target_col = st.text_input(
        "Target column",
        placeholder="e.g. outcome, price, result  (blank = find groupings)",
    )

    objective = st.text_area(
        "What do you want to know? (optional, plain English)",
        placeholder="e.g. What's driving this result? Which rows are the "
                    "outliers, and why?",
        height=90,
        help="Your helpers will prioritise analyses that answer this question "
             "and address it directly in the final report.",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    st.markdown('<div class="side-head">LLM Provider</div>', unsafe_allow_html=True)
    provider = st.selectbox(
        "Provider",
        ["openai", "anthropic", "gemini", "groq", "openrouter", "nvidia", "local"],
        format_func=lambda p: "Local / offline" if p == "local" else p,
    )

    local_base_url = ""
    if provider == "local":
        local_base_url = st.text_input(
            "Server URL (OpenAI-compatible)",
            value="http://localhost:11434/v1",
            help="Works with Ollama, LM Studio, vLLM, llama.cpp server, "
                 "text-generation-webui, etc. Must be reachable from this "
                 "machine — no data leaves it.",
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

    _env_key = os.getenv(f"{provider.upper()}_API_KEY", "")
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
            [f"🆓 Free Models ({len(free_models)})", f"💳 Paid Models ({len(paid_models)})", f"All ({len(_dyn_models)})"],
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
        st.caption("🟢 All models listed below are Free Tier eligible.")
        active_dyn_models = free_models
    else:
        st.caption("💳 Paid API billing applies per token.")
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
            "Adaptive: dynamically scales reasoning effort — low during routine exploratory steps, "
            "higher when anomalies or statistical conflicts occur, and thorough for final synthesis.\n"
            "Fast: forces minimal reasoning effort across all cycles for maximum execution speed.\n"
            "Deep: uses full reasoning depth across all cycles."
        ),
    )
    if "Adaptive" in reasoning_mode:
        os.environ["LLM_REASONING_EFFORT"] = "adaptive"
    elif "Fast" in reasoning_mode:
        os.environ["LLM_REASONING_EFFORT"] = "low"
    elif "Deep" in reasoning_mode:
        os.environ["LLM_REASONING_EFFORT"] = "high"

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
            "Off: fully deterministic — the plan comes from the data profile "
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
            "single biggest speed-up available — training dominates runtime."
        ),
    )
    if not use_llm and not use_ml:
        st.caption("⚡ Fully deterministic, statistics-only mode — fastest.")
    elif not use_llm:
        st.caption("🔌 Deterministic planning, ML still on.")
    elif not use_ml:
        st.caption("⚡ AI narrative on, no models fitted.")
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
                "Off (default): models train with their default hyperparameters — fast.\n"
                "On: searches hyperparameters per model before picking the best "
                "one — meaningfully slower (measured: ~20s extra at 20k rows) "
                "but can improve accuracy."
            ),
        )
        tune_hyperparameters = thorough

        if _cuda_available():
            st.caption("🚀 **GPU Acceleration**: CUDA detected! XGBoost models will train on GPU (`device='cuda'`).")
        else:
            st.caption("💻 **Compute**: CPU mode (multi-core parallel training).")

        with st.expander("Advanced model settings"):
            max_depth = st.slider("Max tree depth (Random Forest / XGBoost)", 2, 15, 6)
            test_pct = st.slider("Test split %", 10, 40, 20, step=5)
            n_cv = st.slider("CV folds (k)", 3, 10, 5)

    # ── Code execution ────────────────────────────────────────────────────────
    # Operator controls for LLM-authored code (src/core/governance.py);
    # defaults match the governance defaults.
    st.markdown('<div class="side-head">Code execution</div>', unsafe_allow_html=True)
    enable_code = st.toggle(
        "Allow AI-written code",
        value=True,
        help=(
            "On: the planner may write and run its own analysis code in the sandbox.\n"
            "Off: only the built-in tools run."
        ),
    )
    max_code_runs = st.number_input(
        "Max code runs per analysis",
        min_value=0,
        value=40,
        step=1,
        disabled=not enable_code,
        help="Further code steps are refused and logged once the budget is spent.",
    )
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
    min_iter   = st.slider("Min iterations", 1, 5, 1, help="Number of iterative discovery cycles. 1 is fast and recommended for quick analysis; increase for deeper multi-cycle discovery.")
    max_iter   = st.slider("Max iterations", 1, 25, 5)
    enable_rlm = st.toggle("Enable recursive decomposition (Stage 6)", value=False, help="Recursively breaks complex tasks into sub-problems. Turn on for deep exploration; keep off for fastest runtime.")

    st.divider()

    # ── Buttons ───────────────────────────────────────────────────────────────
    has_file = st.session_state["preview_df"] is not None
    has_key  = bool(api_key.strip()) or provider == "local"
    can_run  = has_file and has_key and not st.session_state["analysis_done"]

    run_clicked = st.button(
        "Run Analysis",
        disabled=not can_run,
        width='stretch',
        type="primary",
    )
    if st.session_state["analysis_done"] or st.session_state["analysis_error"]:
        if st.button("New Analysis", width='stretch'):
            _reset_pipeline()
            st.rerun()

    demo_clicked = st.button(
        "⚡ See a Sample Report (Demo)",
        width='stretch',
        help="Instantly load sample data and see what a finished report looks like.",
    )
    if demo_clicked:
        _load_teamwork_preview()
        st.rerun()

    if not has_file:
        st.caption("Upload a CSV/Excel file or click \"See a Sample Report\" above.")
    elif not has_key:
        st.caption("Add your API key to enable the run.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — header
# ══════════════════════════════════════════════════════════════════════════════
hero_text, hero_plate = st.columns([0.46, 0.54], gap="large",
                                   vertical_alignment="center")

with hero_text:
    st.markdown(
        '<div class="hero">'
        '<h1>We check every answer twice.</h1>'
        '<p class="hero-sub">Upload any spreadsheet — sales, survey, sports, science, '
        'whatever you\'ve got. Your assistant studies it, tests its own conclusions, '
        'and tells you which patterns are real — and which are just luck.</p></div>',
        unsafe_allow_html=True,
    )
    _hero_cinema_on = st.session_state.get("show_cinematic_hero", False)
    _hero_btn_txt = "🔬 Standard 3D Plate" if _hero_cinema_on else "🎬 3D Cinematic Showcase"
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

if preview_df is not None and not st.session_state["analysis_done"]:
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
# PIPELINE — runs synchronously inside st.status() on button click
# ══════════════════════════════════════════════════════════════════════════════
if run_clicked:
    _reset_pipeline()
    for num, _ in STAGE_DEFS:
        _set_stage(num, "pending")

    # Save dataset to a temp file
    tmp = tempfile.mkdtemp()
    st.session_state["tmp_dir"] = tmp
    dpath  = str(Path(tmp) / st.session_state["preview_name"])
    outdir = str(Path(tmp) / "output")
    with open(dpath, "wb") as _f:
        _f.write(st.session_state["preview_bytes"])

    # Set env vars before importing src
    os.environ["LLM_PROVIDER"]          = provider
    os.environ["LLM_MODEL"]             = final_model
    _picked = next((m for m in _dyn_models if m["model"] == final_model), None)
    if _picked:
        from src.core.model_telemetry import get_limiter

        _profile = get_limiter().get_profile(provider, final_model)
        _profile.context_window = _picked["context_window"]
        _profile.rpm_limit = _picked["rpm_limit"] or _profile.rpm_limit
        _profile.tpm_limit = _picked["tpm_limit"] or _profile.tpm_limit
    os.environ["MIN_ITERATIONS"]        = str(min_iter)
    os.environ["MAX_ITERATIONS"]        = str(max_iter)
    os.environ["ENABLE_RLM_INFERENCE"]  = "true" if enable_rlm else "false"
    os.environ["ENABLE_LLM"]            = "true" if use_llm else "false"
    os.environ["ENABLE_ML"]             = "true" if use_ml else "false"
    os.environ["ENABLE_CODE_EXECUTION"] = "true" if enable_code else "false"
    os.environ["MAX_CODE_EXECUTIONS"]   = str(int(max_code_runs))
    os.environ["SANDBOX_REQUIRE_ISOLATION"] = "true" if require_isolation else "false"
    os.environ["OUTPUT_DIR"]            = outdir
    if objective.strip():
        os.environ["USER_OBJECTIVE"] = objective.strip()
    else:
        os.environ.pop("USER_OBJECTIVE", None)
    {
        "openai":     lambda: os.environ.__setitem__("OPENAI_API_KEY",     api_key.strip()),
        "anthropic":  lambda: os.environ.__setitem__("ANTHROPIC_API_KEY",  api_key.strip()),
        "gemini":     lambda: os.environ.__setitem__("GEMINI_API_KEY",     api_key.strip()),
        "groq":       lambda: os.environ.__setitem__("GROQ_API_KEY",       api_key.strip()),
        "openrouter": lambda: os.environ.__setitem__("OPENROUTER_API_KEY", api_key.strip()),
        "nvidia":     lambda: os.environ.__setitem__("NVIDIA_API_KEY",     api_key.strip()),
        "local":      lambda: (
            os.environ.__setitem__("LOCAL_LLM_API_KEY", api_key.strip() or "not-needed"),
            os.environ.__setitem__("LOCAL_LLM_BASE_URL", local_base_url.strip() or "http://localhost:11434/v1"),
        ),
    }[provider]()

    # ── Spinner placeholder — replaced after run completes ───────────────
    _spinner_ph = st.empty()
    _spinner_ph.markdown(
        '<div class="run-banner">Running the analysis'
        '<span class="sub">Usually 1–3 minutes, depending on the dataset and '
        'the model.</span></div>',
        unsafe_allow_html=True,
    )

    # ── Collect progress lines into session state (no st.write during run) ─
    _progress_lines: list[str] = []

    def _upd_live_ui() -> None:
        """Update the steps list in the hero placeholder during a run."""
        steps_list_slot.markdown(_render_steps_list(st.session_state["stage_log"]), unsafe_allow_html=True)

    def _upd(num: str, s: str, detail: str = "") -> None:
        _set_stage(num, s, detail)
        _ico = {"done": "[done]", "active": "[run ]", "error": "[fail]",
                "skipped": "[skip]"}.get(s, "[    ]")
        _nm  = next(n for no, n in STAGE_DEFS if no == num)
        _progress_lines.append(f"{_ico} Stage {num}: {_nm}" + (f"  {detail}" if detail else ""))
        _upd_live_ui()

    # Initial live paint so stage progress is immediately visible upon clicking Run
    _set_stage("1", "active", "Initializing run & preflight…")
    _upd_live_ui()

    # Set up the expander right away
    with pipeline_slot.container():
        with st.expander("Show how it's working", expanded=True):
            _draw_pipeline_rig(st.empty())
            st.markdown("#### The Team at Work")
            st.markdown(_render_agent_grid(st.session_state["stage_log"]), unsafe_allow_html=True)

    # ── LLM preflight — fail fast with the REAL error instead of running
    #    the whole pipeline on the deterministic fallback ──────────────────
    from src.core.controller import AgentController, LLMClient

    # In no-LLM mode there is nothing to preflight — the run is fully
    # deterministic, so requiring a reachable model (or any API key) would
    # block the very mode that exists to work without one.
    _ok, _ping_err = (True, "") if not use_llm else LLMClient().ping()
    if not _ok:
        _spinner_ph.empty()
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

    try:
        _upd("1", "active", "ingesting…")
        agent = AgentController(
            min_iterations=min_iter,
            max_iterations=max_iter,
            enable_rlm=enable_rlm,
            use_llm=use_llm,
            use_ml=use_ml,
        )
        if use_ml:
            # IMPROVEMENTS.md 7.16 — TrainModelTool.requires_context reads
            # these back and fills them into the real train_model call
            # whenever the planner leaves them empty.
            agent.memory.set_context("ui_max_depth", max_depth)
            agent.memory.set_context("ui_test_size", test_pct / 100.0)
            agent.memory.set_context("ui_n_cv_folds", n_cv)
            agent.memory.set_context("ui_tune_hyperparameters", tune_hyperparameters)
        meta = agent.load_dataset(
            dpath,
            target_hint=target_col.strip() or None,
            interactive=False,
        )
        st.session_state["metadata"] = meta
        _upd("1", "done",
             f"{meta.row_count:,} rows × {meta.column_count} cols · task={meta.task_type} · target={meta.target_column}")

        _upd("2", "active",
             "calling LLM for analysis plan…" if use_llm
             else "building deterministic plan from the data profile…")
        _upd("3", "pending")
        _upd("4", "pending")
        _upd("5", "pending")
        _upd("6", "pending" if enable_rlm else "skipped",
             "" if enable_rlm else "disabled")
        _upd("7", "pending")

        # ── Lightweight callbacks — only update stage_log, no st.write ────
        def _on_step(tool_name: str, status: str, detail: str) -> None:
            _set_stage("3", "active", detail)
            _progress_lines.append(f"       {'ok  ' if status=='success' else '... '}{detail}")
            _upd_live_ui()
            if tool_name == "train_model":
                _spinner_ph.markdown(
                    '<div class="run-banner">Running the analysis'
                    '<span class="sub">Still working — training models can take a few minutes depending on the data size...</span></div>',
                    unsafe_allow_html=True,
                )

        def _on_iter(iteration: int, stage: str) -> None:
            if "stage2" in stage:
                _set_stage("2", "active", f"iter {iteration} — LLM reasoning…")
                _progress_lines.append(f"[run ] Iteration {iteration}: model reasoning")
            elif "stage4" in stage or "stage5" in stage:
                _set_stage("4", "active", f"iter {iteration} — interpreting results…")
                _set_stage("5", "active", f"iter {iteration} — refining plan…")
                _progress_lines.append(f"[run ] Iteration {iteration}: interpreting and refining")
            _upd_live_ui()
            if iteration > 1:
                _spinner_ph.markdown(
                    '<div class="run-banner">Running the analysis'
                    '<span class="sub">Still working — refining answers can take a few minutes...</span></div>',
                    unsafe_allow_html=True,
                )

        agent.on_step_callback      = _on_step
        agent.on_iteration_callback = _on_iter

        final = agent.analyze()

        # ── Mark all stages done ──────────────────────────────────────────
        _upd("2", "done", "plan generated & executed")
        tool_names_run = list({r.get("tool_name","") for r in [t.to_dict() for t in agent.memory.tool_results]})
        _upd("3", "done", f"{len(agent.memory.tool_results)} tools executed: {', '.join(tool_names_run[:5])}")
        _upd("4", "done", "results interpreted")
        _upd("5", "done", f"{agent.memory.iteration_count} iteration(s)")
        if enable_rlm:
            sub = agent.memory.get_context("rlm_sub_results")
            _upd("6", "done",
                 f"{len(sub)} sub-tasks" if sub else "no decomposition needed")
        _upd("7", "done", "report saved")

        st.session_state["tool_results"]  = [r.to_dict() for r in agent.memory.tool_results]
        st.session_state["final_report"]  = final
        if agent.last_profile is not None:
            st.session_state["profile"] = agent.last_profile.to_dict()
        st.session_state["read_report"] = agent.memory.get_context("read_report")
        st.session_state["coercions"] = agent.memory.get_context("coercions")
        st.session_state["profile_status"] = agent.memory.get_context("profile_status")
        _dash_path = Path(outdir) / "reports" / "dashboard.json"
        if _dash_path.exists():
            try:
                st.session_state["dashboard"] = json.loads(
                    _dash_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                st.session_state["dashboard"] = None
        st.session_state["analysis_done"] = True
        st.session_state["progress_lines"] = _progress_lines
        llm_err = agent.memory.get_context("llm_error")
        if llm_err:
            st.session_state["llm_warning"] = f"Fallback plan was used (LLM issue): {llm_err[:300]}"
        _spinner_ph.empty()

    except Exception:
        err = traceback.format_exc()
        st.session_state["analysis_error"] = err
        st.session_state["progress_lines"] = _progress_lines
        for _n, _s, _d in reversed(st.session_state["stage_log"]):
            if _s == "active":
                _set_stage(_n, "error", "failed")
                break
        _spinner_ph.empty()
        st.error(
            "Something went wrong during the run and it couldn't finish. "
            "This usually means a step in the analysis hit an unexpected "
            "problem with this specific file — see the technical details "
            "below if you want to know exactly what happened."
        )
        with st.expander("Technical details"):
            st.code(err, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# HERO PLATE & DATUM REFRESH
# ══════════════════════════════════════════════════════════════════════════════
_done = sum(1 for _, s, _ in st.session_state["stage_log"] if s == "done")
_errored = any(s == "error" for _, s, _ in st.session_state["stage_log"])
_running = any(s == "active" for _, s, _ in st.session_state["stage_log"])

if st.session_state.get("analysis_done") or _errored or _running:
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
_errored = any(s == "error" for _, s, _ in st.session_state["stage_log"])
_running = any(s == "active" for _, s, _ in st.session_state["stage_log"])
_status_str = (
    "FAILED" if _errored else
    "RUNNING" if _running else
    f"{_done}/7 COMPLETE" if _done > 0 else
    "IDLE"
)

_cur_theme_name = "Day Mode" if st.session_state.get("theme", "day") == "day" else "Night Mode"
_cells = [
    ("State", _status_str),
    ("Theme", _cur_theme_name),
    ("File", st.session_state.get("preview_name") or "None loaded"),
    ("Model", final_model if "final_model" in locals() else "N/A"),
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
    (tab_brief, tab_dash, tab_lab, tab_vault) = st.tabs([
        "Answers",
        "Charts",
        "Details",
        "Downloads",
    ])

    train_out   = _find_tool(tool_results, "train_model")
    eval_out    = _find_tool(tool_results, "evaluate_model")
    corr_out    = _find_tool(tool_results, "correlation_analysis")
    outlier_out = _find_tool(tool_results, "detect_outliers")
    stat_out    = _find_tool(tool_results, "select_statistical_test")
    clean_out   = _find_tool(tool_results, "clean_data")
    vega_cfg    = _get_vega_config()

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 1: EXECUTIVE BRIEFING
    # ═════════════════════════════════════════════════════════════════════════
    with tab_brief:
        # Case Heading
        _user_obj = os.environ.get("USER_OBJECTIVE") or objective.strip()
        if _user_obj and report.get("reasoning"):
            st.markdown(
                f'<div class="exec-directive" style="border-left: 4px solid var(--pen); padding-left: 1rem; margin-bottom: 2rem; background: var(--sheet); border-radius: var(--radius); padding: 1.5rem; box-shadow: var(--lift-sm);">'
                f'<h2 style="font-size: 1.2rem; color: var(--graphite); text-transform: uppercase; letter-spacing: 1px; margin: 0 0 0.5rem 0;">Case Brief</h2>'
                f'<div style="font-size: 1.4rem; font-weight:700; margin-bottom:1rem; color:var(--ink);">Objective: {html.escape(_user_obj)}</div>'
                f'<div class="dir-content" style="font-size: 1.1rem; line-height: 1.6; color: var(--ink);">{html.escape(plainify(str(report["reasoning"])))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif report.get("reasoning"):
            st.markdown(
                f'<div class="exec-directive" style="border-left: 4px solid var(--pen); padding-left: 1rem; margin-bottom: 2rem; background: var(--sheet); border-radius: var(--radius); padding: 1.5rem; box-shadow: var(--lift-sm);">'
                f'<h2 style="font-size: 1.2rem; color: var(--graphite); text-transform: uppercase; letter-spacing: 1px; margin: 0 0 0.5rem 0;">Case Brief</h2>'
                f'<div class="dir-content" style="font-size: 1.1rem; line-height: 1.6; color: var(--ink);">{html.escape(plainify(str(report["reasoning"])))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        _du_html = _render_data_understanding(report.get("data_understanding") or {})
        if _du_html:
            st.markdown("#### How we read your data")
            st.markdown(_du_html, unsafe_allow_html=True)

        best_model   = report.get("best_model") or "N/A"
        best_cv      = "0"
        best_gap_str = "0"
        gap_val: float | None = None

        if train_out:
            _mt_map = train_out.get("models_trained", {})
            _best   = train_out.get("best_model", "")
            if _best and _best in _mt_map:
                _bm      = _mt_map[_best]
                best_cv  = f"{_bm.get('cv_mean', 0)*100:.1f}"
                gap_val  = _bm.get("train_test_gap")
                best_gap_str = f"{gap_val*100:.1f}" if gap_val is not None else "0"

        task_type = (
            (train_out.get("task_type") if train_out else None)
            or (meta.task_type if meta else "—") or "—"
        )

        outlier_pct = str(outlier_out.get('outlier_percentage', '0')) if outlier_out else "0"

        prof: dict[str, Any] = st.session_state.get("profile") or {}
        _q = int(prof.get("quality_score", 0))

        # When there's no ML target, promote whatever analysis actually ran
        if not train_out:
            _dominant = (
                _find_tool(tool_results, "cluster_data")
                or _find_tool(tool_results, "time_series_analysis")
                or _find_tool(tool_results, "geospatial_analysis")
                or _find_tool(tool_results, "text_analysis")
                or _find_tool(tool_results, "dimensionality_analysis")
            )
            if _dominant and _dominant.get("summary"):
                st.info(f"**What we found:** {_dominant['summary']}")

        # ── What we found: the top-ranked Findings, as full-width sentence
        # cards (IMPROVEMENTS.md 7.15) — this is the "invert the Answers tab"
        # fix. Model-internal numbers (best model / CV / gap) move below,
        # behind the technical-detail expander, since they're not answers to
        # the user's question even when a model was trained.
        _all_findings: list[dict[str, Any]] = report.get("findings") or []
        _dash_finding_ids = {c.get("finding_id") for c in (dash or []) if c.get("finding_id")}
        _card_findings = [
            f for f in _all_findings
            if f.get("layer") in ("exec", "analyst") and f.get("kind") not in ("method_fit", "coverage_gap")
        ][:5]

        st.markdown("#### What we found")
        if _card_findings:
            for _f in _card_findings:
                st.markdown(_render_finding_card(_f, _dash_finding_ids), unsafe_allow_html=True)
        elif _all_findings:
            # Findings exist but every one of them is a caveat/appendix-only
            # item for this run — say so rather than showing nothing.
            st.caption("No headline-worthy findings cleared the bar for this run — "
                       "see the Details tab for the full finding list.")
        else:
            # Old cached result (pre-finding-bus) or a totally empty run:
            # fall back to the plain insights list rather than a blank tab.
            _ins_list = report.get("insights", [])
            if _ins_list:
                for _i, _ins in enumerate(_ins_list, start=1):
                    st.markdown(f'<div class="ic"><span class="mk">{_i:02d}</span>{html.escape(plainify(str(_ins)))}</div>', unsafe_allow_html=True)
            else:
                st.caption("No explicit statistical discoveries recorded.")

        st.markdown("#### What to do")
        _rec_list = report.get("recommendations", [])
        if _rec_list:
            for _rec in _rec_list:
                st.markdown(f'<div class="rc"><span class="mk">Do</span>{html.escape(plainify(str(_rec)))}</div>', unsafe_allow_html=True)
        else:
            st.caption("No operational recommendations generated.")

        # ── Trust strip: quality, coverage, caveats — the honest summary of
        # how much to trust this run, including what it declined to answer
        # (the payoff of the 7.5 question agenda: use it, don't let it go
        # unused) ───────────────────────────────────────────────────────────
        st.markdown("#### At a glance")
        _row_count = (
            (meta.row_count if meta else None)
            or (prof.get("row_count") if prof else None)
            or (len(preview_df) if preview_df is not None else None)
        )
        _caveat_count = sum(1 for f in _all_findings if f.get("kind") in ("method_fit", "coverage_gap"))
        _coverage = report.get("coverage") or {}
        _unanswered = _coverage.get("unanswered") or []

        _trust_cells = [
            ("Data quality", f"{_q}/100"),
            ("Rows analyzed", f"{_row_count:,}" if isinstance(_row_count, int) else "—"),
            ("Caveats flagged", str(_caveat_count)),
        ]
        st.markdown(
            '<div class="trust-strip">' + "".join(
                f'<div class="trust-cell"><div class="k">{html.escape(k)}</div>'
                f'<div class="v">{html.escape(v)}</div></div>'
                for k, v in _trust_cells
            ) + '</div>',
            unsafe_allow_html=True,
        )
        if _unanswered:
            st.caption(f"⚠ {len(_unanswered)} question(s) considered, not answered.")
            with st.expander("What wasn't answered, and why"):
                for _u in _unanswered:
                    st.markdown(f"- {_u.get('text', '')}")

        # ── Ask a follow-up question (IMPROVEMENTS.md 7.20, scoped down):
        # a plain keyword search over the finding bus — no new tool calls,
        # no LLM call. ───────────────────────────────────────────────────
        st.markdown("#### Ask a follow-up question")
        _ask_q = st.text_input(
            "Ask a follow-up question",
            placeholder="e.g. does tenure affect churn?",
            label_visibility="collapsed",
            key="ask_data_q",
        )
        if _ask_q.strip():
            _matches = _search_findings(_ask_q, _all_findings)
            if _matches:
                for _m in _matches:
                    st.success(f"Based on what was found: {plainify(str(_m.get('headline', '')))}")
            else:
                st.info("Nothing in this analysis directly answers that — try rephrasing, "
                        "or check the Details tab for full coverage.")

        # ── Technical detail: model-internal numbers, kept but demoted ─────
        with st.expander("Show technical detail"):
            _radial_color = "var(--positive)" if _q >= 80 else ("var(--accent)" if _q >= 60 else "var(--risk)")
            _gap_flag = gap_val is not None and _gap_is_risky(gap_val)
            st.markdown(
                f"""<div class="kpi-row">
<div class="kpi-gauge-card">
<div class="kpi-gauge-label">Data Quality</div>
<div class="kpi-ring-wrap">
<svg width="140" height="140" viewBox="0 0 140 140" style="position: absolute; top: 0; left: 0; transform: rotate(-90deg); overflow: visible; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));">
<circle cx="70" cy="70" r="56" fill="none" stroke="var(--rule-faint)" stroke-width="12" />
<circle cx="70" cy="70" r="56" fill="none" stroke="{_radial_color}" stroke-width="12" stroke-linecap="round" stroke-dasharray="351.86" stroke-dashoffset="{351.86 - (351.86 * _q / 100)}" class="anime-gauge" data-q="{_q}" />
</svg>
<div class="kpi-ring-value">
<div class="kpi-ring-num count-up" data-value="{_q}" data-suffix="">{_q}</div>
<div class="kpi-ring-sub">Score</div>
</div>
</div>
</div>
<div class="kpi-tiles">
<div class="kpi-tile">
<div class="k">Best Model</div>
<div class="v">{html.escape(best_model)}</div>
<div class="s">Task: {html.escape(str(task_type))}</div>
</div>
<div class="kpi-tile">
<div class="k">CV Score</div>
<div class="v big count-up" data-value="{best_cv}" data-suffix="%">{best_cv}%</div>
</div>
<div class="kpi-tile{' flagged' if _gap_flag else ''}">
<div class="k">Train-Test Gap</div>
<div class="v big{' risk' if _gap_flag else ''} count-up" data-value="{best_gap_str}" data-suffix="%">{best_gap_str}%</div>
</div>
</div>
</div>""",
                unsafe_allow_html=True,
            )

            # Generalization Defect / Certification Stamp
            if gap_val is not None:
                st.markdown(_render_defect_stamp(gap_val), unsafe_allow_html=True)

            for _w in (train_out.get("overfit_warnings", []) if train_out else []):
                st.markdown(f'<div class="wc"><span class="mk">Risk</span>{html.escape(str(_w))}</div>', unsafe_allow_html=True)

            # Model comparison & correlation charts — rendered from the
            # dashboard artifact the backend already built, by chart_id,
            # instead of app.py re-deriving the same chart (IMPROVEMENTS.md
            # 7.17). Falls back to a caption rather than crashing when the
            # panel isn't present (e.g. an old cached dashboard.json).
            _mc_chart = _find_chart_by_id(dash, "model_comparison")
            if _mc_chart:
                st.markdown("##### How each model scored (Train vs Test vs CV)")
                _render_dashboard_chart(_mc_chart, vega_cfg)
            elif train_out:
                st.caption("Model comparison chart not available for this run.")

            _tc_chart = _find_chart_by_id(dash, "top_correlations")
            if _tc_chart:
                st.markdown("##### Strongest feature correlations")
                _render_dashboard_chart(_tc_chart, vega_cfg)
            elif corr_out:
                st.caption("Correlation chart not available for this run.")

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 3: DYNAMIC DASHBOARD
    # ═════════════════════════════════════════════════════════════════════════
    with tab_dash:
        dashboard: list[dict[str, Any]] | None = st.session_state.get("dashboard")
        if dashboard:
            st.caption("Built automatically to fit your data — the most important panels lead.")

            # Layout is data-driven, not a hardcoded chart_id set (7.18): a
            # panel explicitly on the exec layer, or among the top 3 by the
            # backend's own `priority` ranking, gets full width; everything
            # else goes in the 2-column grid. `dashboard` already arrives
            # sorted by priority descending (build_dashboard()'s contract),
            # so the first 3 entries *are* the top 3 — no re-sort here.
            _top_priority_ids = {c.get("chart_id") for c in dashboard[:3]}
            _grid_charts: list[dict[str, Any]] = []

            for _ch in dashboard:
                _is_full_width = _ch.get("size") == "wide" or (
                    _ch.get("size") != "half"
                    and (_ch.get("layer") == "exec" or _ch.get("chart_id") in _top_priority_ids)
                )
                if _is_full_width:
                    _render_dashboard_chart(_ch, vega_cfg)
                else:
                    _grid_charts.append(_ch)
            if _grid_charts:
                _dcols = st.columns(2)
                for _i, _ch in enumerate(_grid_charts):
                    with _dcols[_i % 2]:
                        _render_dashboard_chart(_ch, vega_cfg)
        else:
            # Say *why* when the coverage record can tell us — an "everything
            # was declined" run reads very differently from "nothing ran yet"
            # (IMPROVEMENTS.md 7.22).
            _dash_coverage = report.get("coverage") or {}
            if _dash_coverage.get("total", 0) > 0 and _dash_coverage.get("answered", 0) == 0:
                st.info(
                    "No chart-worthy findings were produced for this run — every "
                    "question on the agenda was either declined or came back without "
                    "a chartable pattern. See the Details tab for the full coverage breakdown."
                )
            else:
                st.info(
                    "No dashboard was generated for this run. The run may still be "
                    "processing, or the analysis produced no chartable results."
                )

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 4: STATISTICAL & ML LAB
    # ═════════════════════════════════════════════════════════════════════════
    with tab_lab:
        st.markdown("### Case Log")
        st.caption("A sequential record of technical checks performed on this dataset.")

        _profile_status = st.session_state.get("profile_status")
        if isinstance(_profile_status, str) and _profile_status.startswith("failed"):
            st.warning(
                "Running in degraded mode — profiling failed, so dataset-nature "
                f"tools (time-series, text, geo...) were unavailable. Reason: {_profile_status[8:]}"
            )

        _read_report = st.session_state.get("read_report")
        _coercions = st.session_state.get("coercions")
        if _read_report or _coercions:
            with st.container(border=True):
                st.markdown("#### 1. Data Ingestion & Repair")
                if _read_report:
                    _rr_bits = [f"format `{_read_report.get('format')}`", f"encoding `{_read_report.get('encoding')}`"]
                    if not _read_report.get("encoding_confident", True):
                        _rr_bits[-1] += " (guessed)"
                    if _read_report.get("delimiter"):
                        _rr_bits.append(f"delimiter `{_read_report.get('delimiter')!r}`" + ("" if _read_report.get("delimiter_sniffed") else " (from extension)"))
                    st.caption("Detected at read time: " + ", ".join(_rr_bits) + ".")
                    for _note in _read_report.get("notes", []):
                        st.caption(f"⚠ {_note}")
                if _coercions:
                    st.caption(f"{len(_coercions)} column(s) repaired:")
                    st.dataframe(_safe_df(pd.DataFrame([
                        {"Column": c["column"], "Rule": c["rule"], "Converted": c["n_converted"], "Failed": c["n_failed"]}
                        for c in _coercions
                    ])), width='stretch')

        prof: dict[str, Any] | None = st.session_state.get("profile")
        if prof:
            with st.container(border=True):
                st.markdown("#### 2. Profiling")
                _prows = [{
                    "Column":    c.get("name"),
                    "Kind":      c.get("kind"),
                    "Dtype":     c.get("dtype"),
                    "Missing %": c.get("missing_pct"),
                    "Unique":    c.get("nunique"),
                    "Flags":     ", ".join(c.get("flags", [])),
                } for c in prof.get("columns", [])]
                st.dataframe(_safe_df(pd.DataFrame(_prows)), width='stretch')

        if clean_out:
            with st.container(border=True):
                st.markdown("#### 3. Data Cleaning")
                _c1, _c2, _c3 = st.columns(3)
                _c1.metric("Strategy used", clean_out.get("strategy_used", "—"))
                _c2.metric("Missing values before", clean_out.get("missing_before", "—"))
                _c3.metric("Missing values after",  clean_out.get("missing_after", "—"))

        if outlier_out:
            with st.container(border=True):
                st.markdown("#### 4. Outlier Detection")
                _c1, _c2 = st.columns(2)
                _c1.metric("Unusual rows found", outlier_out.get("total_outliers", "—"))
                _c2.metric("Share of all rows", f"{outlier_out.get('outlier_percentage','—')}%")
                _pc = outlier_out.get("per_column_outliers", {})
                if _pc:
                    _pc_df = pd.DataFrame(
                        [(c, v) for c, v in _pc.items() if v > 0],
                        columns=["Column", "Unusual values"],
                    ).sort_values("Unusual values", ascending=False)
                    if not _pc_df.empty:
                        st.dataframe(_safe_df(_pc_df), width='stretch')

        if stat_out:
            with st.container(border=True):
                st.markdown("#### 5. Statistical Tests")
                _c1, _c2, _c3 = st.columns(3)
                _c1.metric("Test used", stat_out.get("test_name", "—"))
                _c2.metric("p-value", f"{stat_out.get('p_value', 0):.4f}")
                _c3.metric("Likely real, not chance", "Yes" if stat_out.get("significant") else "No")
                st.info(stat_out.get("interpretation", "No interpretation recorded."))

        if train_out:
            with st.container(border=True):
                st.markdown("#### 6. Model Training & Evaluation")
                _mt2   = train_out.get("models_trained", {})
                _best2 = train_out.get("best_model", "")
                _task2 = train_out.get("task_type", "classification")
                _pk2   = "accuracy" if _task2 == "classification" else "r2"
                _sk2   = "f1_score" if _task2 == "classification" else "rmse"

                st.markdown(
                    f"**Type of problem:** `{_task2}` &nbsp;|&nbsp; **Best model:** `{_best2}` "
                    f"&nbsp;|&nbsp; **Times each model was tested:** `{train_out.get('n_cv_folds', 5)}` "
                    f"&nbsp;|&nbsp; **Held back for testing:** `{int(train_out.get('test_size', 0.2)*100)}%`"
                )
                _rows = []
                for _nm, _m in _mt2.items():
                    _tr2 = _m.get("train_metrics", {})
                    _te2 = _m.get("test_metrics",  {})
                    _g   = _m.get("train_test_gap")
                    _rows.append({
                        "Model": f"{_nm} (best)" if _nm == _best2 else _nm,
                        f"Train {_pk2}": f"{_tr2.get(_pk2,0)*100:.1f}%",
                        f"Test {_pk2}":  f"{_te2.get(_pk2,0)*100:.1f}%",
                        "CV mean":  f"{_m.get('cv_mean',0)*100:.1f}%",
                        "CV std":   f"±{_m.get('cv_std',0)*100:.1f}%",
                        "Gap": (f"{_g*100:.1f}%" + (" [RISK]" if _g and _g >= .10 else "")
                                if _g is not None else "—"),
                        _sk2.replace("_", " "): (
                            f"{_te2.get(_sk2,0)*100:.1f}%"
                            if _sk2 != "rmse" else f"{_te2.get(_sk2,0):.4f}"
                        ),
                    })
                st.dataframe(_safe_df(pd.DataFrame(_rows)), width='stretch')

                if eval_out:
                    st.markdown("##### Accuracy by Category")
                    _cr = eval_out.get("classification_report", {})
                    if _cr:
                        _cr_rows = [
                            {"Class": _lbl,
                             "Precision": f"{_v.get('precision',0):.3f}",
                             "Recall":    f"{_v.get('recall',0):.3f}",
                             "F1":        f"{_v.get('f1-score',0):.3f}",
                             "Support":   int(_v.get("support", 0))}
                            for _lbl, _v in _cr.items() if isinstance(_v, dict)
                        ]
                        st.dataframe(_safe_df(pd.DataFrame(_cr_rows)), width='stretch')

        with st.container(border=True):
            st.markdown("#### 7. Domain-Specific Analyses")
            st.caption("Segmentation, trends, text, and geography — run when your data called for them.")
            _render_other_findings(tool_results)

        # IMPROVEMENTS.md 7.21 — "Your Helpers" folded in here as the run
        # trace: still available for anyone curious how the run actually
        # went, but no longer competing with the findings for top-level
        # attention. Kept inside an expander (collapsed by default) since
        # it's supplementary to the numbered case log above, not part of it.
        with st.expander("8. Run Trace — how the agents worked through this", expanded=False):
            st.caption("Each helper does one job, and hands off to the next.")
            st.markdown(_render_agent_grid(st.session_state["stage_log"], tool_results, report), unsafe_allow_html=True)

            st.markdown("##### What Was Said, Step by Step")
            st.caption("A record of what each helper passed to the next, and when.")
            st.markdown(
                _render_handoff_stream(st.session_state.get("progress_lines", []), tool_results),
                unsafe_allow_html=True,
            )

            sub_results = report.get("rlm_sub_results")
            if sub_results:
                st.markdown("##### How the Tricky Parts Were Split Up")
                st.caption("Big questions got broken into smaller ones so nothing got lost.")
                for _s_idx, _sub in enumerate(sub_results, 1):
                    with st.expander(f"Part {_s_idx:02d}: {_sub.get('task_name', 'Smaller Question')}", expanded=False):
                        st.json(_sub)

        _gov = report.get("governance")
        if _gov:
            with st.container(border=True):
                st.markdown("#### 9. Governance & audit")
                st.caption("What AI-written code ran, where it ran, and what was refused.")
                _render_governance(_gov)

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 6: ARTIFACT VAULT & EXPORTS
    # ═════════════════════════════════════════════════════════════════════════
    with tab_vault:
        st.markdown("### Downloads")
        st.caption("Everything from this run, ready to keep or share.")

        if tmp_dir:
            _out = Path(tmp_dir) / "output"
            _rdir = _out / "reports"

            st.markdown(
                "<style>.dossier-card { background: var(--sheet-alt); border: 1px solid var(--rule); border-radius: var(--radius); padding: 1rem; margin-bottom: 1rem; display: flex; flex-direction: column; gap: 0.5rem; transition: transform 0.15s, box-shadow 0.15s; } .dossier-card:hover { transform: translateY(-2px); box-shadow: var(--lift-sm); }</style>",
                unsafe_allow_html=True
            )

            # File Cards Layout using Containers and Grid
            _dl_cols = st.columns(3)
            with _dl_cols[0]:
                st.markdown("#### Presentations")
                with st.container(border=True):
                    st.markdown("**3D Cinematic Journey**")
                    st.caption("Standalone HTML with 3D models and animations.")
                    from ui.cinematic_3d import build_cinematic_document, extract_cinematic_state
                    _cinema_pres_html = build_cinematic_document(
                        extract_cinematic_state(st.session_state),
                        theme=st.session_state.get("theme", "night"),
                    )
                    st.download_button(
                        "🎬 Download HTML",
                        _cinema_pres_html.encode("utf-8"),
                        "dsa_agent_3d_presentation.html",
                        mime="text/html",
                        key="dl_3d_cinema_standalone",
                        type="primary",
                        use_container_width=True,
                    )
                with st.container(border=True):
                    st.markdown("**Shareable Report**")
                    st.caption("A clean web-page reading view of the report.")
                    _html = _rdir / "report.html"
                    if _html.exists():
                        st.download_button(
                            "📄 Download HTML",
                            _html.read_bytes(), "report.html", mime="text/html",
                            key="dl_html_vault",
                            use_container_width=True,
                        )

            with _dl_cols[1]:
                st.markdown("#### Data & Logs")
                with st.container(border=True):
                    st.markdown("**Markdown Report**")
                    st.caption("The core report in plain text markdown.")
                    _mds = sorted(_rdir.glob("*.md")) if _rdir.exists() else []
                    if _mds:
                        st.download_button(
                            "📝 Download Markdown",
                            _mds[0].read_bytes(), _mds[0].name, mime="text/markdown",
                            key="dl_md_vault",
                            use_container_width=True,
                        )
                with st.container(border=True):
                    st.markdown("**Agent Memory Vault**")
                    st.caption("The complete raw data of everything the agents found.")
                    st.download_button(
                        "💾 Download JSON",
                        json.dumps(report, indent=2, default=str),
                        "final_report.json", mime="application/json",
                        key="dl_json_vault",
                        use_container_width=True,
                    )

            with _dl_cols[2]:
                st.markdown("#### Trained Models")
                _mdir = _out / "models"
                if _mdir.exists() and any(_mdir.iterdir()):
                    for _mdl_f in sorted(_mdir.iterdir()):
                        with st.container(border=True):
                            st.markdown(f"**{_mdl_f.name}**")
                            st.caption("Pickled model object ready for predictions.")
                            st.download_button(
                                "📦 Download",
                                _mdl_f.read_bytes(), _mdl_f.name,
                                mime="application/octet-stream",
                                key=f"dlm_vault_{_mdl_f.name}",
                                use_container_width=True,
                            )
                else:
                    st.caption("No predictive models were saved for this run.")

            st.divider()
            st.markdown("#### Report Preview")
            if _mds:
                st.markdown(_mds[0].read_text(encoding="utf-8"))
            else:
                st.json(report)

            with st.expander("Full Technical Log", expanded=False):
                st.json(tool_results)


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════
if (preview_df is None
        and not st.session_state["analysis_done"]
        and not st.session_state["stage_log"]):
    st.markdown(
        '<div class="empty">'
        '<h2>Let\'s see what your data shows.</h2>'
        '<p>Add a CSV or Excel file in the sidebar, tell us what you\'d like to know, '
        'and enter your API key. Your helpers will study it, test their answers, '
        'and double-check everything before showing you the results.</p>'
        '<div class="steps">'
        '<div>1. Read</div><div>2. Understand</div><div>3. Run</div>'
        '<div>4. Explain</div><div>5. Check</div><div>6. Solve</div>'
        '<div>7. Report</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Meet Your Helpers")
    st.markdown(_render_agent_grid([]), unsafe_allow_html=True)
    st.write("")
    if st.button("▶ See a Sample Report (Demo)", type="primary"):
        _load_teamwork_preview()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MICRO-INTERACTIONS (Phase 3)
# ══════════════════════════════════════════════════════════════════════════════
from ui.animations import inject_micro_interactions  # noqa: E402

inject_micro_interactions()
