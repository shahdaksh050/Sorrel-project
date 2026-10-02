"""
Ledger Design System Styles and CSS Injection for Streamlit UI.
"""
from __future__ import annotations

import streamlit as st

from src.core import design_tokens

#: Self-hosted font stylesheet, served by Streamlit's static route
#: (`[server] enableStaticServing = true`). Its @font-face rules point at the
#: .woff2 files beside it by relative URL, so no request leaves for a font CDN.
FONT_CSS_PATH = "app/static/fonts/ledger-fonts.css"


def static_url(path: str) -> str:
    """URL of a file under ./static (served at /app/static), honouring `server.baseUrlPath`."""
    base = str(st.get_option("server.baseUrlPath") or "").strip("/")
    rel = f"app/static/{path.lstrip('/')}"
    return f"/{base}/{rel}" if base else f"/{rel}"


def font_css_url() -> str:
    """URL of the local font stylesheet, honouring `server.baseUrlPath`."""
    return static_url("fonts/ledger-fonts.css")


def inject_theme_css() -> None:
    """Inject dynamic Ledger CSS supporting Day and Night modes via Python state."""
    theme = st.session_state.get("theme", "day")

    mode: design_tokens.Mode = "night" if theme in ("dark", "night") else "day"

    # --lift/--lift-sm are styles.py-local (shadow blur tuned per mode, not part of
    # the shared palette) so they stay hand-typed alongside the imported token lines.
    if mode == "night":
        theme_vars = design_tokens.css_root_block(mode) + """
        --lift:        0 4px 18px rgba(0,0,0,.35);
        --lift-sm:     0 2px 8px rgba(0,0,0,.3);
        """
    else:
        theme_vars = design_tokens.css_root_block(mode) + """
        --lift:        0 4px 14px color-mix(in srgb, var(--ink) 14%, transparent);
        --lift-sm:     0 2px 8px color-mix(in srgb, var(--ink) 10%, transparent);
        """

    st.markdown(f"""
<style>
@import url('{font_css_url()}');

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
/* Section headings inside a tab keep the earlier visual size at their corrected level. */
[data-baseweb="tab-panel"] h2 {{ font-size: 21px !important; }}
[data-baseweb="tab-panel"] h3 {{ font-size: 17px !important; }}
[data-baseweb="tab-panel"] .exec-directive h2 {{ font-size: 27px !important; }}
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
         background: var(--sheet); border-radius: var(--radius);
         border-bottom: 1px solid var(--rule-faint);
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

/* ── Chart panels ── */
.chart-title {{
    font-family: var(--heading); font-size: 1.1rem !important; font-weight: 700 !important;
    color: var(--ink); margin: 0 0 .5rem; padding: 0;
}}
.chart-desc {{ font-size: .9rem; color: var(--graphite); margin-top: .5rem; line-height: 1.4; }}

/* ── Primary task steps (file, question, run) ── */
.step-head {{
    display: flex; align-items: center; gap: .55rem; margin: 1rem 0 .4rem;
    font-family: var(--heading); font-weight: 700; font-size: 1.1rem; color: var(--ink);
}}
.step-head:first-child {{ margin-top: .2rem; }}
.step-n {{
    display: inline-grid; place-items: center; width: 1.6rem; height: 1.6rem; border-radius: 50%;
    background: var(--pen); color: var(--sheet); font-size: .85rem; font-weight: 800;
}}
.file-notices {{ margin: .3rem 0 .7rem; font-size: 13px; color: var(--graphite); }}
.file-notices summary {{ cursor: pointer; font-weight: 600; min-height: 24px; }}
.file-notices ul {{ margin: .3rem 0 0; padding-left: 1.2rem; line-height: 1.5; }}

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
.hero.compact {{ padding: .4rem 0 .6rem; }}
.hero.compact .hero-eyebrow {{
    font-size: 12.5px; font-weight: 700; letter-spacing: .04em; color: var(--pen); margin-bottom: .35rem;
}}
.hero.compact h1 {{
    font-size: clamp(26px, 3.4vw, 40px); line-height: 1.12; max-width: none; letter-spacing: -.01em;
    animation: none; overflow-wrap: anywhere;
}}
.hero.compact .hero-file {{ color: var(--graphite); font-weight: 700; font-size: .62em; display: inline-block; overflow-wrap: anywhere; }}

.st-key-plate {{ padding-left: 16px; margin-right: -2.8rem; }}
/* The 3D plate is an optional extra; on a phone it would push the inputs off the first screen.
   The stage list beside it, and the live progress panel, carry the same information. */
@media (max-width: 768px) {{
    .st-key-plate {{ padding-left: 0; margin-right: 0; }}
    .st-key-plate iframe,
    .st-key-plate .stElementContainer:has(iframe) {{ display: none; }}
}}
@media (max-width: 900px) {{
    .st-key-plate {{ margin-right: 0; padding-left: 0; }}
    [data-testid="stHorizontalBlock"]:has(.st-key-plate) {{ flex-wrap: wrap !important; }}
    [data-testid="stHorizontalBlock"]:has(.st-key-plate) > [data-testid="stColumn"] {{
        min-width: 100% !important; flex: 1 1 100% !important;
    }}
    .datum .cell {{ flex: 1 1 45%; }}
    .stTabs [role="tab"] {{ min-height: 44px; }}
    .stButton button, .stDownloadButton button {{ min-height: 44px; }}
}}

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
/* The plate (FrontendPlan.md 2.8 keeps this one's elevation permanently) — no
   hover-lift, since the iframe is not a link or a Streamlit callback target. */
iframe {{
    border-radius: var(--radius);
    border: 1px solid var(--rule) !important;
    background: transparent !important;
    box-shadow: var(--lift-sm);
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
    color: var(--graphite) !important; opacity: .9 !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{ border-color: var(--pen) !important; }}

/* ── Stat tile ── */
.gauge {{ background: var(--sheet); border: 1px solid transparent;
         border-radius: var(--radius);
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
    background: var(--sheet);
    border-left: 4px solid var(--pen); border-radius: var(--radius);
    padding: 1rem 1.3rem; margin: 0 0 .9rem;
}}
.finding-headline {{
    font-family: var(--heading); font-size: 17px; font-weight: 700;
    color: var(--ink); line-height: 1.4;
}}

/* ── Trust strip (Answers tab) ── */
.trust-strip {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }}
.trust-cell {{
    background: var(--sheet); border-radius: var(--radius);
    padding: .9rem 1.2rem; flex: 1; min-width: 160px;
}}
.trust-cell .k {{ font-size: 12px; color: var(--graphite); font-weight: 600;
                  text-transform: uppercase; letter-spacing: .5px; }}
.trust-cell .v {{ font-family: var(--heading); font-size: 24px; font-weight: 800;
                  color: var(--ink); margin-top: .3rem; }}

/* ── How the agent read the data (Answers tab) ── */
.du {{
    background: var(--sheet); border-radius: var(--radius);
    padding: .9rem 1.2rem; margin: 0 0 1.5rem;
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
    background: var(--sheet); padding: 1.5rem; border-radius: var(--radius);
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
             display: flex; flex-direction: column; justify-content: center; }}
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
    border: 1px solid transparent;
    border-radius: var(--radius);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}
/* Each card is a <details>/<summary> disclosure (render_agent_grid, cards.py) —
   genuinely clickable, so the hover lift stays (FrontendPlan.md 2.8). */
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
    0%   {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--pen) 40%, transparent); }}
    70%  {{ box-shadow: 0 0 0 10px color-mix(in srgb, var(--pen) 0%, transparent); }}
    100% {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--pen) 0%, transparent); }}
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
    font-size: 12px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 999px;
    color: var(--graphite);
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
    border-radius: var(--radius);
    padding: 1.2rem;
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
    border-left: 4px solid var(--pen);
    border-radius: var(--radius);
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
    border-radius: var(--radius); overflow: hidden;
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
.sc.skip  .sc-num {{ background: var(--sheet-alt); color: var(--graphite); border: 1px solid var(--rule); }}
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

.reason {{ background: var(--sheet);
          border-radius: var(--radius); padding: 1.3rem 1.5rem;
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

/* ── Audited-entry check row (FrontendPlan.md section 5) ── */
.check-row {{ display: flex; flex-wrap: wrap; gap: .9rem; margin-top: .6rem; }}
.check {{ font-size: 13px; font-weight: 600; display: inline-flex; align-items: center; gap: 4px; }}
.check.ok {{ color: var(--positive); }}
.check.risk {{ color: var(--risk); }}
.check.note {{ color: var(--graphite); font-weight: 500; }}
@media (prefers-reduced-motion: no-preference) {{
  .check-row.animate .check {{
    opacity: 0; transform: translateY(2px);
    animation: checkIn 240ms cubic-bezier(.16,1,.3,1) forwards;
  }}
  .check-row.animate .check:nth-child(1) {{ animation-delay: 0ms; }}
  .check-row.animate .check:nth-child(2) {{ animation-delay: 60ms; }}
  .check-row.animate .check:nth-child(3) {{ animation-delay: 120ms; }}
  .check-row.animate .check:nth-child(4) {{ animation-delay: 180ms; }}
  .check-row.animate .check:nth-child(5) {{ animation-delay: 240ms; }}
  .check-row.animate .check:nth-child(6) {{ animation-delay: 300ms; }}
}}
@keyframes checkIn {{ to {{ opacity: 1; transform: translateY(0); }} }}
/* ── "How we got here" (Details): hairline-ruled blocks, tokens only ── */
.how-we-got-here {{ margin: 0 0 2rem; max-width: 1100px; }}
.how-title {{ font-family: var(--heading); margin: 0 0 .25rem; }}
.how-lede {{ color: var(--graphite); font-size: 14.5px; line-height: 1.55; margin: 0 0 1rem; }}
.how-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 0 2.5rem; }}
.how-block {{ border-top: 1px solid var(--rule); padding: .9rem 0 1.1rem; min-width: 0; }}
.how-h {{ font-size: 15px; font-weight: 700; margin: 0 0 .15rem; color: var(--ink); }}
.how-note {{ font-size: 13px; color: var(--graphite); line-height: 1.5; margin: 0 0 .6rem; }}
.how-lead {{ font-size: 15px; font-weight: 600; color: var(--ink); line-height: 1.5; margin: 0 0 .4rem; }}
.how-text {{ font-size: 14.5px; color: var(--ink); line-height: 1.6; margin: 0 0 .5rem; overflow-wrap: anywhere; }}
.how-sub {{ font-size: 12.5px; font-weight: 700; color: var(--graphite); margin: .6rem 0 .2rem; }}
.how-list {{ margin: 0 0 .4rem; padding-left: 1.1rem; font-size: 14px; line-height: 1.55; color: var(--ink); overflow-wrap: anywhere; }}
.how-more {{ font-size: 13px; color: var(--graphite); margin: 0; }}
.how-tag {{ font-size: 12px; font-weight: 700; color: var(--graphite); }}
.how-tag.supported {{ color: var(--positive); }}
.how-tag.refuted {{ color: var(--risk); }}
/* ── "Found so far, may change" (live run): hairline block, tokens only ── */
.prov {{ border-top: 1px solid var(--rule); padding: .8rem 0 .4rem; margin: .8rem 0; max-width: 74ch; }}
.prov-h {{ font-size: 15px; font-weight: 700; margin: 0 0 .15rem; color: var(--ink); }}
.prov-note {{ font-size: 13px; color: var(--graphite); line-height: 1.5; margin: 0 0 .5rem; }}
.prov-list {{ margin: 0; padding-left: 1.1rem; font-size: 14px; line-height: 1.55; color: var(--ink); overflow-wrap: anywhere; }}
.prov-kind {{ font-size: 12px; font-weight: 700; color: var(--graphite); }}
.prov-count {{ font-size: 13px; color: var(--graphite); margin: .4rem 0 0; }}
/* ── Native widget text follows the page tokens ──
   Streamlit's native theme is fixed (.streamlit/config.toml, Day palette), so in
   Night mode its own text colours would be dark on a dark page. These rules take
   the colour from the tokens instead; in Day they resolve to the same values. */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: var(--graphite) !important; opacity: 1 !important; }}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {{ color: var(--graphite) !important; }}
[data-testid="stCheckbox"] label *, [data-testid="stRadio"] label *,
[data-testid="stToggle"] label * {{ color: var(--ink) !important; }}
[data-testid="stExpander"] summary * {{ color: var(--ink) !important; }}
[data-testid="stMetricLabel"] * {{ color: var(--graphite) !important; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{ color: var(--ink) !important; }}
[data-testid="stFileUploaderDropzoneInstructions"] * {{ color: var(--graphite) !important; }}
[data-testid="stTooltipIcon"] svg {{ color: var(--graphite) !important; }}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"] {{
    background: var(--sheet) !important; border-color: var(--rule) !important;
}}
[data-testid="stTextInput"] button, [data-testid="stNumberInput"] button {{ color: var(--ink) !important; }}
[data-testid="stNumberInput"] input {{ background: var(--sheet) !important; color: var(--ink) !important; }}
[data-testid="stNumberInput"] button {{ background: var(--sheet) !important; color: var(--ink) !important; }}
</style>
""", unsafe_allow_html=True)
