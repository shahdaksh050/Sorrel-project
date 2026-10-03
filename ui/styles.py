"""
Sorrel workspace styles and CSS injection for the Streamlit UI.

The look is the prototype's workspace (docs/prototypes): warm paper, near-black
ink, one forest-green pen, hairline 4px cards, Geist for the interface and
Newsreader for the italic accent word. The workspace has no gradients, glow,
blur, shadows or transitions, and nothing that cannot be clicked reacts to the
pointer. Colour values are never typed here: they come from
`src.core.design_tokens.css_root_block`, and every rule below reads them as
`var(--token)`.

Colour roles worth knowing before editing a rule:

* `--pen` is the forest-green fill. Night `--pen` is only about 2.5:1 as text or
  as a thin indicator, so text, focus rings and the active-tab underline use
  `--accent-text`; text on a `--pen` fill uses `--accent-ink`.
* `--accent` is the amber: a warning, a failed step, a gauge worth a look.
* `--risk` / `--danger-text` mean one thing only, "this finding or number may not
  hold". They are never a generic error colour.
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
    return static_url("vendor/fonts/ledger-fonts.css")


def theme_vars_for(mode: design_tokens.Mode) -> str:
    """The `:root` custom properties for one mode: the shared tokens, then the few derived ones.

    Derived here, not in `design_tokens`, because they are CSS-only: a control
    border that clears 3:1 on both surfaces, and the "may not hold" text colour.
    """
    rule_strong = (
        "color-mix(in srgb, var(--rule) 50%, white)"
        if mode == "night"
        else "color-mix(in srgb, var(--rule) 60%, black)"
    )
    return (
        design_tokens.css_root_block(mode)
        + f"""
        --rule-strong: {rule_strong};
        --risk-text:   var(--danger-text); /* AA text for "may not hold", in both modes */
        --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
        --dur-fast: 150ms;
        --dur-base: 250ms;
        --dur-slow: 400ms;
        """
    )


def inject_theme_css() -> None:
    """Inject the Sorrel CSS for Day or Night, from session state."""
    theme = st.session_state.get("theme", "day")

    mode: design_tokens.Mode = "night" if theme in ("dark", "night") else "day"
    theme_vars = theme_vars_for(mode)

    st.markdown(f"""
<style>
@import url('{font_css_url()}');

:root {{
    {theme_vars}
    --radius:      4px;
    --radius-control: 4px;
    --radius-pill: 999px;

    --sans:    'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --heading: var(--sans);
    --serif:   'Newsreader', Georgia, serif;
    --mono:    'Geist Mono', 'JetBrains Mono', ui-monospace, monospace;
    --text-xs: .8125rem;
    --text-sm: .9375rem;
    --text-base: 1rem;
    --text-lg: 1.25rem;
    --text-xl: 1.5625rem;
    --text-2xl: 1.9375rem;
    --text-3xl: 2.5rem;
    --text-4xl: 3.5rem;
    --ease-in-out: cubic-bezier(.4, 0, .2, 1);
}}

@media (prefers-reduced-motion: reduce) {{
    :root {{
        --dur-fast: 0ms !important;
        --dur-base: 0ms !important;
        --dur-slow: 0ms !important;
    }}
    *, *::before, *::after {{
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: .01ms !important;
    }}
}}


/* ── Reveal motion ──
   Expand, collapse and fade, nothing else: no hover-lift, no glow, nothing that loops. 250 ms at the
   most, and only when the browser has not asked for less (the block above zeroes every duration, and
   this one is gated too). Disclosures (the expanders and the cards that open) ease their height
   where the browser supports it (Chromium today; elsewhere they simply open at once). */
@media (prefers-reduced-motion: no-preference) {{
    :root {{ interpolate-size: allow-keywords; }}
    details::details-content {{
        block-size: 0; overflow: clip;
        transition: block-size var(--dur-base) var(--ease-out), content-visibility var(--dur-base) allow-discrete;
    }}
    details[open]::details-content {{ block-size: auto; }}
    @keyframes revealIn {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}
    [data-baseweb="tab-panel"]:not([hidden]) {{ animation: revealIn var(--dur-base) var(--ease-out) both; }}
    .stepper .step-dot, .stepper .step::after, .step-state, .step-name {{
        transition: background-color var(--dur-fast) ease, border-color var(--dur-fast) ease, color var(--dur-fast) ease;
    }}
}}

/* ── The page ── */
#MainMenu, footer, .stAppDeployButton {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
/* The native run-status widget (spinner + "Stop") is the one piece of
   stock Streamlit chrome the above rules don't touch: it's the only way
   to interrupt a running analysis, so it stays, re-themed to the tokens. */
[data-testid="stStatusWidget"] {{
    background: var(--sheet);
    border: 1px solid var(--rule);
    border-radius: var(--radius-pill);
    color: var(--ink);
}}
[data-testid="stStatusWidget"] svg {{ color: var(--accent-text); }}
.stApp {{ background-color: var(--stock); background-image: none; }}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; }}
html, body, .stApp, [class*="css"] {{ font-family: var(--sans); color: var(--ink); }}
/* Figures line up in columns: tabular numerals on every metric and table. */
.gauge .v, .datum .v, .kpi-ring-num, .kpi-tile .v, .trust-cell .v,
[data-testid="stMetricValue"], [data-testid="stDataFrame"], .run-banner, .artifact .sz {{
    font-variant-numeric: tabular-nums;
}}
hr {{ border: none; border-top: 1px solid var(--rule) !important; }}
a {{ color: var(--accent-text) !important; text-underline-offset: 3px; font-weight: 500; }}
.sr-only {{
    position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0;
    overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
}}

section[data-testid="stSidebar"] {{
    background: var(--sheet);
    border-right: 1px solid var(--rule);
    background-image: none;
}}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] label p {{ font-size: var(--text-xs); color: var(--graphite); font-weight: 500; }}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: var(--rule); border-radius: 6px; border: 2px solid var(--stock); }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* ── Type: Geist for the interface, Newsreader italic for the one accent word ── */
h1, h2, h3, h4, h5, h6 {{
    font-family: var(--heading) !important;
    letter-spacing: -.01em;
    color: var(--ink);
}}
h1 {{ font-weight: 600 !important; font-size: var(--text-2xl) !important; line-height: 1.15; letter-spacing: -.02em; }}
h2 {{ font-weight: 600 !important; font-size: var(--text-lg) !important; line-height: 1.2; }}
h3 {{ font-weight: 600 !important; font-size: var(--text-base) !important; line-height: 1.25; }}
/* Section headings inside a tab keep one scale: 20px sections, 16px sub-sections. */
[data-baseweb="tab-panel"] h2 {{ font-size: var(--text-lg) !important; }}
[data-baseweb="tab-panel"] h3 {{ font-size: var(--text-base) !important; }}
[data-baseweb="tab-panel"] .exec-directive h2 {{ font-size: var(--text-lg) !important; }}
/* A small square in the accent colour leads each section, so the page has a rhythm without decoration. */
[data-baseweb="tab-panel"] h2::before {{
    content: ""; display: inline-block; width: 8px; height: 8px; margin-right: 10px;
    border-radius: 2px; background: var(--pen); vertical-align: 2px;
}}
[data-baseweb="tab-panel"] .exec-directive h2::before {{ content: none; }}
h4 {{ font-weight: 600 !important; font-size: var(--text-sm) !important; letter-spacing: 0; }}
.stMarkdown p, .stMarkdown li {{ font-size: var(--text-base); line-height: 1.6; max-width: 68ch; }}
code, kbd, pre, .stCode {{ font-family: var(--mono) !important; }}
[data-testid="stMetricValue"] {{ font-family: var(--mono) !important; font-weight: 600; color: var(--ink) !important; }}
[data-testid="stMetricLabel"] * {{ color: var(--graphite) !important; }}
[data-testid="stFileUploader"] section {{ background: var(--stock) !important; border: 1px dashed var(--rule-strong) !important; }}
[data-testid="stFileUploader"] section * {{ color: var(--ink) !important; }}
[data-testid="stFileUploader"] small {{ color: var(--graphite) !important; }}
.stExpander {{ border-color: var(--rule-faint) !important; background: var(--sheet) !important; }}
.stExpander summary {{ color: var(--ink) !important; }}
.eyebrow, .hero-eyebrow {{
    display: block; font-family: var(--mono); font-size: 11px; font-weight: 600; line-height: 1.2;
    letter-spacing: .08em; text-transform: uppercase; color: var(--graphite); margin: 0 0 .5rem;
}}

/* ── Brand: the italic serif wordmark with its small seal ── */
.topbar-name {{ display: flex; align-items: center; gap: 10px; }}
.brand-seal {{
    width: 24px; height: 24px; display: inline-grid; place-items: center; flex: none;
    background: var(--pen); color: var(--accent-ink); border-radius: 3px;
    font-family: var(--serif); font-style: italic; font-weight: 500; font-size: 17px; line-height: 1;
    padding-bottom: 2px;
}}
.brand-name {{
    font-family: var(--serif); font-style: italic; font-weight: 500; font-size: 23px;
    letter-spacing: -.01em; line-height: 1; color: var(--ink);
}}

/* ── Readout: State, File, Analysis ── */
.datum {{
    display: grid; grid-auto-flow: column; grid-auto-columns: minmax(0, 1fr);
    background: var(--sheet); border: 1px solid var(--rule); border-radius: var(--radius);
    margin: 0 0 1.5rem;
}}
.datum .cell {{ padding: 10px 16px; border-left: 1px solid var(--rule); min-width: 0; }}
.datum .cell:first-child {{ border-left: none; }}
.datum .k {{
    font-family: var(--mono); font-size: 10.5px; font-weight: 500; line-height: 1.2;
    letter-spacing: .08em; text-transform: uppercase; color: var(--graphite);
}}
.datum .v {{ font-size: var(--text-sm); font-weight: 600; color: var(--ink); margin-top: 4px; overflow-wrap: anywhere; }}

/* ── Section head: the title, with an optional mono note on the right ── */
.sect {{
    display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap;
    margin: 1.6rem 0 .9rem; padding: 0;
}}
.sect:first-child {{ margin-top: .4rem; }}
.sect h2, .sect h3 {{ margin: 0; padding: 0; font-size: var(--text-lg) !important; letter-spacing: -.01em; }}
.sect .note {{
    font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: var(--graphite); overflow-wrap: anywhere;
}}

/* ── Chart panels ── */
.chart-title {{
    font-family: var(--heading); font-size: var(--text-base) !important; font-weight: 600 !important;
    color: var(--ink); margin: 0 0 .5rem; padding: 0; line-height: 1.4;
}}
.chart-desc {{ font-size: var(--text-sm); color: var(--graphite); margin-top: .5rem; line-height: 1.4; }}
.chart-src {{ font-size: var(--text-xs); color: var(--graphite); margin: .5rem 0 0; }}
/* One line per chart: the verdict as a chip (glyph and word), then the finding it supports. */
.chart-evidence {{ display: flex; align-items: flex-start; gap: 10px; margin: 0 0 .6rem;
    font-size: var(--text-sm); line-height: 1.45; color: var(--ink-2); }}
.chart-evidence .ev-text {{ min-width: 0; overflow-wrap: anywhere; }}
.verdict-chip {{ flex: none; display: inline-block; padding: 1px 8px; border: 1px solid currentColor;
    border-radius: var(--radius-pill); font-family: var(--mono); font-size: 11px; font-weight: 600;
    letter-spacing: .04em; text-transform: uppercase; line-height: 1.5; color: var(--graphite); background: var(--sheet); }}
.verdict-chip.held {{ color: var(--positive); background: color-mix(in srgb, var(--positive) 8%, var(--sheet)); }}
.verdict-chip.needs_more {{ color: var(--danger-text); background: color-mix(in srgb, var(--risk) 8%, var(--sheet)); }}

/* ── Primary task steps (file, question, mode, run) ── */
.step-head {{
    display: flex; align-items: center; gap: 10px; margin: 1.1rem 0 .5rem;
    font-family: var(--heading); font-weight: 600; font-size: var(--text-base); color: var(--ink);
}}
.step-head:first-child {{ margin-top: .2rem; }}
.step-n {{
    display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; flex: none;
    background: var(--pen); color: var(--accent-ink);
    font-family: var(--mono); font-size: 12px; font-weight: 600; line-height: 1;
}}
.file-identity {{
    margin: .6rem 0 .3rem; padding: 10px 14px; font-size: var(--text-sm); overflow-wrap: anywhere;
    background: var(--accent-soft); border: 1px solid var(--pen); border-radius: var(--radius);
    color: var(--accent-text);
}}
.file-identity b {{ color: inherit; font-weight: 600; }}
.file-notices {{ margin: .3rem 0 .7rem; font-size: var(--text-xs); color: var(--graphite); }}
.file-notices summary {{ cursor: pointer; font-weight: 500; min-height: 24px; color: var(--ink-2); }}
.file-notices ul {{ margin: .3rem 0 0; padding-left: 1.2rem; line-height: 1.5; }}

/* The task card holds the one primary task. Only the border colour and corner come from the
   key class, so the card still draws its own native frame. */
.st-key-task_card {{ border-color: var(--rule); border-radius: var(--radius); }}

/* ── Page header ── */
.hero {{ padding: .2rem 0 1rem; }}
.hero h1 {{
    font-size: clamp(1.7rem, 3vw, 2.3rem); font-weight: 600; line-height: 1.15;
    letter-spacing: -.02em; margin: 0; max-width: 18ch;
}}
.hero .hero-sub {{
    color: var(--ink-2); font-size: var(--text-base); line-height: 1.55;
    margin: .6rem 0 0; max-width: 58ch;
}}
.hero.compact {{ padding: .4rem 0 .6rem; }}
.hero.compact h1 {{ max-width: none; overflow-wrap: anywhere; }}
/* The file name is the one italic serif word in the header. */
.hero .hero-file {{
    font-family: var(--serif); font-style: italic; font-weight: 400; color: var(--accent-text);
    overflow-wrap: anywhere;
}}
.workspace-welcome {{ border-bottom: 1px solid var(--rule); margin-bottom: 1.25rem; }}
.st-key-hero_band {{
    background: var(--sheet-alt); border: 1px solid var(--rule); border-radius: var(--radius);
    padding: 1rem 1.4rem 1.2rem; margin-bottom: 1.5rem;
}}
.st-key-hero_band .hero.compact {{ padding: .2rem 0 .4rem; }}
.st-key-hero_actions {{ display: flex; flex-direction: column; align-items: flex-end; gap: .5rem; }}
.st-key-hero_actions .stButton {{ width: auto; }}
.sample-choice {{
    padding: 14px 16px; background: var(--stock); border: 1px solid var(--rule); border-radius: var(--radius);
}}
.sample-choice .sample-title {{ color: var(--ink); font-size: var(--text-sm); font-weight: 600; }}
.sample-choice p {{ color: var(--graphite); font-size: var(--text-xs); line-height: 1.45; margin: .25rem 0 .6rem; }}

/* The 3D plate is an optional extra; on a phone it would push the inputs off the first screen.
   The stage list beside it, and the live progress panel, carry the same information. */
@media (max-width: 768px) {{
    .st-key-plate iframe,
    .st-key-plate .stElementContainer:has(iframe) {{ display: none; }}
}}
@media (max-width: 900px) {{
    [data-testid="stHorizontalBlock"]:has(.st-key-plate) {{ flex-wrap: wrap !important; }}
    [data-testid="stHorizontalBlock"]:has(.st-key-plate) > [data-testid="stColumn"] {{
        min-width: 100% !important; flex: 1 1 100% !important;
    }}
    .datum {{ grid-auto-flow: row; }}
    .datum .cell {{ border-left: none; border-top: 1px solid var(--rule); }}
    .datum .cell:first-child {{ border-top: none; }}
    .stTabs [role="tab"] {{ min-height: 44px; }}
    .stButton button, .stDownloadButton button {{ min-height: 44px; }}
}}

/* ── Sidebar masthead & settings groups ── */
.side-brand {{ margin: .1rem 0 .8rem; }}
.side-word {{
    font-family: var(--serif); font-style: italic; font-weight: 500; font-size: 20px;
    letter-spacing: -.01em; line-height: 1; color: var(--ink); margin-bottom: .6rem;
}}
.side-title {{ font-family: var(--heading); font-weight: 600; font-size: var(--text-base); line-height: 1.15; color: var(--ink); }}
.side-sub {{ font-size: var(--text-xs); color: var(--graphite); margin-top: 4px;
            max-width: 26ch; line-height: 1.45; }}
.side-head {{
    font-family: var(--mono); font-weight: 600; font-size: 11.5px; line-height: 1.3;
    letter-spacing: .06em; text-transform: uppercase; color: var(--ink-2);
    margin: 1.5rem 0 .6rem; padding-top: .8rem; border-top: 1px solid var(--rule-faint);
}}
.side-head:first-of-type {{ margin-top: .5rem; }}

/* ── Buttons ── */
.stButton button, .stDownloadButton button {{
    font-family: var(--sans); font-weight: 500; font-size: var(--text-sm);
    border-radius: var(--radius-control) !important; letter-spacing: 0;
}}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {{
    background: var(--pen); color: var(--accent-ink); border: 1px solid transparent;
}}
.stButton button[kind="primary"]:enabled *, .stDownloadButton button[kind="primary"]:enabled * {{
    color: var(--accent-ink) !important;
}}
.stButton button[kind="primary"]:hover:enabled,
.stDownloadButton button[kind="primary"]:hover:enabled {{
    background: var(--pen-hover); color: var(--accent-ink);
}}
.stButton button[kind="primary"]:disabled {{
    background: var(--sheet-alt); color: var(--graphite);
    border: 1px dashed var(--rule-strong);
}}
.stButton button[kind="secondary"], .stDownloadButton button[kind="secondary"] {{
    background: transparent; border: 1px solid var(--rule-strong); color: var(--ink);
}}
.stButton button[kind="secondary"]:hover:enabled,
.stDownloadButton button[kind="secondary"]:hover:enabled {{
    background: transparent; color: var(--ink); border-color: var(--ink);
}}
:focus-visible {{ outline: 2px solid var(--accent-text) !important; outline-offset: 2px; }}
.stButton button:focus-visible, .stDownloadButton button:focus-visible {{
    outline: 2px solid var(--accent-text) !important; outline-offset: 3px;
}}

/* ── Tabs: an underline marks the active one, not a filled pill bar ── */
[data-testid="stTabs"] [role="tablist"] {{
    gap: 4px !important; background: transparent !important; border-radius: 0 !important;
    padding: 0 !important; overflow-x: auto; border: none; border-bottom: 1px solid var(--rule) !important;
    display: flex !important; width: 100% !important; position: relative;
}}
[data-testid="stTabs"] [role="tab"] {{
    flex: 0 0 auto !important; justify-content: center !important; text-align: center !important;
    border-radius: 0 !important; padding: 12px 18px !important; background-color: transparent !important;
    border: none !important; border-bottom: 2px solid transparent !important; margin: 0 0 -1px !important;
    outline: none !important; z-index: 1; position: relative; white-space: nowrap;
}}
[data-testid="stTabs"] [role="tab"]:last-child {{ margin-right: 0 !important; }}
[data-testid="stTabs"] [role="tab"]:focus {{ outline: none !important; }}
[data-testid="stTabs"] [role="tab"]:focus-visible {{ outline: 2px solid var(--accent-text) !important; outline-offset: -2px; }}
[data-testid="stTabs"] [role="tab"] p {{ font-size: var(--text-sm); font-weight: 500;
                                 color: var(--graphite) !important; letter-spacing: 0; margin: 0 !important; }}
[data-testid="stTabs"] [role="tab"]:hover p {{ color: var(--ink) !important; }}
/* The selected tab paints its own underline: the label never depends on Streamlit's highlight bar,
   whose markup changes between releases. Night `--pen` is too dim for a thin line, so the line uses
   the accent text colour (the same green in Day). */
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ border-bottom-color: var(--accent-text) !important; }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{ color: var(--ink) !important; font-weight: 600; }}
[data-testid="stTabs"] .react-aria-SelectionIndicator,
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none !important; }}
[data-testid="stTabs"] [data-testid="stTabPanel"] {{ padding-top: 1.5rem; }}

/* ── 3D plate ── */
/* The plate is not a link or a Streamlit callback target, so it has a hairline and nothing else. */
.st-key-plate iframe {{
    border-radius: var(--radius);
    border: 1px solid var(--rule) !important;
    background: transparent !important;
}}

/* ── Inputs ── */
[data-testid="stFileUploaderDropzone"] {{
    background: var(--stock); border: 1.5px dashed var(--rule-strong); border-radius: var(--radius);
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--ink); }}
[data-testid="stFileUploader"] button {{
    background: transparent !important; border: 1px solid var(--rule-strong) !important; color: var(--ink) !important;
}}
[data-testid="stFileUploader"] button:hover {{ border-color: var(--ink) !important; }}
/* ── Inputs (Polling-safe, no !important) ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    border-radius: var(--radius-control); border: 1px solid var(--rule-strong);
    background: var(--stock); color: var(--ink);
}}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {{
    color: var(--graphite); opacity: .9;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {{
    border-color: var(--accent-text); box-shadow: 0 0 0 1px var(--accent-text);
}}
[data-testid="stTextInput"] input:hover,
[data-testid="stTextArea"] textarea:hover,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {{
    border-color: var(--graphite);
}}


/* ── Stat tile (dataset preview, governance) ── */
.gauge {{ background: var(--sheet); border: 1px solid var(--rule); border-radius: var(--radius);
         padding: 14px 16px; height: 100%; min-height: 96px; }}
.gauge .v {{ font-family: var(--mono); font-size: var(--text-xl); font-weight: 600;
            line-height: 1.1; color: var(--ink); overflow-wrap: anywhere; }}
.gauge.long .v   {{ font-size: var(--text-lg); }}
.gauge.longer .v {{ font-size: var(--text-sm); line-height: 1.25; }}
.gauge .k {{ font-family: var(--mono); font-size: 10.5px; font-weight: 500; letter-spacing: .08em;
            text-transform: uppercase; color: var(--graphite); margin-top: .5rem; line-height: 1.2; }}
.gauge .s {{ font-size: var(--text-xs); color: var(--graphite); margin-top: 4px; }}
/* A gauge "worth a look" is amber with a shape and words in its note, never red. */
.gauge:not(.flag) .v {{ color: var(--accent-text); }}
.gauge.flag {{ border-color: var(--accent); }}
.gauge.flag .s {{ color: var(--accent); font-weight: 600; }}

/* ── Callout cards: the model may not hold up / should hold up ── */
.defect-stamp {{
    border: 1px solid var(--risk);
    background: color-mix(in srgb, var(--risk) 6%, var(--sheet));
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    margin: 1.2rem 0;
}}
.defect-stamp .stamp-tag {{
    font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .04em;
    color: var(--danger-text); display: block; margin-bottom: 4px;
}}
.defect-stamp .stamp-title {{
    font-family: var(--heading); font-size: var(--text-base); font-weight: 600; color: var(--danger-text);
    margin-bottom: 6px;
}}
.defect-stamp .stamp-desc {{
    font-size: var(--text-sm); line-height: 1.58; color: var(--ink); max-width: 68ch;
}}

.cert-stamp {{
    border: 1px solid var(--pen);
    background: var(--accent-soft);
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    margin: 1.2rem 0;
}}
.cert-stamp .stamp-tag {{
    font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .04em;
    color: var(--accent-text); display: block; margin-bottom: 4px;
}}
.cert-stamp .stamp-title {{
    font-family: var(--heading); font-size: var(--text-base); font-weight: 600; color: var(--accent-text);
    margin-bottom: 6px;
}}
.cert-stamp .stamp-desc {{
    font-size: var(--text-sm); line-height: 1.58; color: var(--ink); max-width: 68ch;
}}

/* ── Finding cards (Answers tab, IMPROVEMENTS.md 7.15) ── */
.finding-card {{
    margin: 0 0 .9rem;
}}
.finding-headline {{
    font-family: var(--heading); font-size: var(--text-base); font-weight: 600;
    color: var(--ink); line-height: 1.4;
}}

/* ── Facts at a glance (Answers tab): the readout again, ruled cells ── */
.trust-strip {{
    display: grid; grid-auto-flow: column; grid-auto-columns: minmax(0, 1fr);
    background: var(--sheet); border: 1px solid var(--rule); border-radius: var(--radius); margin: 0 0 1rem;
}}
.trust-cell {{ padding: 10px 16px; border-left: 1px solid var(--rule); min-width: 0; }}
.trust-cell:first-child {{ border-left: none; }}
.trust-cell .k {{ font-family: var(--mono); font-size: 10.5px; color: var(--graphite); font-weight: 500;
                  text-transform: uppercase; letter-spacing: .08em; line-height: 1.2; }}
.trust-cell .v {{ font-size: var(--text-sm); font-weight: 600; color: var(--accent-text); margin-top: 4px; overflow-wrap: anywhere; }}
@media (max-width: 900px) {{
    .trust-strip {{ grid-auto-flow: row; }}
    .trust-cell {{ border-left: none; border-top: 1px solid var(--rule); }}
    .trust-cell:first-child {{ border-top: none; }}
}}

/* ── How the agent read the data (Answers tab) ── */
.du {{
    background: var(--sheet); border: 1px solid var(--rule); border-radius: var(--radius);
    padding: .9rem 1.2rem; margin: 0 0 1.5rem;
    font-size: var(--text-sm); line-height: 1.6; color: var(--ink-2); max-width: 74ch;
}}
.du .k {{ font-family: var(--mono); font-size: 11px; color: var(--graphite); font-weight: 600;
          text-transform: uppercase; letter-spacing: .06em; margin-right: .4rem; }}
.du .note {{ color: var(--graphite); }}

/* ── Sandbox isolation badge (Details tab): shape, then words ── */
.iso-badge {{
    display: inline-block; font-family: var(--sans); font-size: var(--text-xs); font-weight: 600;
    padding: 3px 8px; border-radius: var(--radius); margin: 0 0 .75rem;
    color: var(--ink); border: 1px solid var(--rule-strong);
}}
.iso-badge.ok {{ border-color: var(--positive); }}
.iso-badge.ok::before {{ content: "\\2713  "; color: var(--positive); }}
.iso-badge.warn {{ border-color: var(--accent); }}
.iso-badge.warn::before {{ content: "!  "; color: var(--accent); font-weight: 700; }}

/* ── KPI strip (technical-detail expander) ── */
.kpi-row {{ display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }}
.kpi-gauge-card {{
    background: var(--sheet); border: 1px solid var(--rule); padding: 1.5rem; border-radius: var(--radius);
    flex: 1; min-width: 250px; display: flex; flex-direction: column; align-items: center;
    justify-content: center; position: relative; overflow: hidden;
}}
.kpi-gauge-label {{ font-family: var(--mono); font-size: 11px; color: var(--graphite); font-weight: 600;
                    margin-bottom: 1.5rem; text-transform: uppercase; letter-spacing: .08em; }}
.kpi-ring-wrap {{ position: relative; width: 140px; height: 140px;
                  display: flex; align-items: center; justify-content: center; }}
.kpi-ring-value {{ display: flex; flex-direction: column; align-items: center;
                   margin-top: 6px; z-index: 10; }}
.kpi-ring-num {{ font-family: var(--mono); font-size: var(--text-3xl); font-weight: 600;
                 color: var(--ink); line-height: 1; }}
.kpi-ring-sub {{ font-family: var(--mono); font-size: 11px; font-weight: 600; color: var(--graphite);
                 text-transform: uppercase; letter-spacing: .08em; margin-top: 2px; }}
.kpi-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
              gap: 1rem; flex: 2; min-width: 300px; }}
.kpi-tile {{ background: var(--sheet); border: 1px solid var(--rule); padding: 1.5rem; border-radius: var(--radius);
             display: flex; flex-direction: column; justify-content: center; }}
.kpi-tile.flagged {{ border-color: var(--risk); }}
.kpi-tile .k {{ font-family: var(--mono); font-size: 10.5px; color: var(--graphite); font-weight: 500;
               text-transform: uppercase; letter-spacing: .08em; }}
.kpi-tile .v {{ font-family: var(--mono); font-size: var(--text-lg); font-weight: 600;
               color: var(--ink); margin-top: 0.5rem; overflow-wrap: anywhere; }}
.kpi-tile .v.big {{ font-size: var(--text-2xl); margin-top: 0.2rem; }}
.kpi-tile .v.risk {{ color: var(--danger-text); }}
.kpi-tile .s {{ font-size: var(--text-xs); color: var(--graphite); margin-top: 4px; }}

/* ── Step-by-step record: one disclosure card per step ── */
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
}}
/* Each card is a <details>/<summary> disclosure (render_agent_grid, cards.py), so it is
   genuinely clickable and may answer the pointer. */
.agent-card:hover {{ border-color: var(--ink); }}
.agent-card.agent-active {{
    border-color: var(--pen);
    background: var(--accent-soft);
}}
.agent-card.agent-flagged {{ border-color: var(--accent); }}
.agent-grid.org-chart-layout {{
    display: grid;
    /* auto-fit/minmax, not a fixed column count + viewport media query:
       this grid can render inside a narrower container (e.g. the run
       progress rail), and a viewport-width breakpoint doesn't know that. */
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 20px;
    position: relative;
    padding: 20px 0;
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
    font-weight: 600;
    font-size: var(--text-sm);
    color: var(--ink);
}}
/* The badge carries a shape and a word; colour only repeats them. */
.agent-badge {{
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: var(--radius-pill);
    border: 1px solid currentColor;
    color: var(--graphite);
}}
.agent-badge.done {{ color: var(--positive); }}
.agent-badge.running {{ color: var(--accent-text); }}
.agent-badge.error {{ color: var(--accent); }}

/* ── Finding and provisional cards ── */
.bento-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
    align-items: start;
}}
.bento-card, .finding-card {{
    background: var(--sheet);
    border: 1px solid var(--rule);
    border-radius: var(--radius);
    padding: 1.2rem;
}}
.bento-card.full-width, .finding-card.full-width {{
    grid-column: 1 / -1;
}}
.finding-card.flagged {{ border-color: var(--risk); }}
.finding-detail {{ color: var(--graphite); font-size: var(--text-xs); line-height: 1.5; margin-top: .4rem; }}
.finding-chart-note {{ color: var(--accent-text); font-size: var(--text-xs); font-weight: 600; margin-top: .55rem; }}
.agent-card {{ position: relative; overflow: hidden; display: flex; flex-direction: column; }}
.agent-details {{ padding: 1rem; cursor: pointer; width: 100%; }}
.agent-summary {{ list-style: none; display: flex; flex-direction: column; }}
.agent-header {{ display: flex; justify-content: space-between; align-items: center; width: 100%; }}
.agent-role {{ font-weight: 600; color: var(--ink); display: flex; align-items: center; gap: 8px; }}
.agent-more {{ margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--rule-faint); font-size: var(--text-sm); }}
.agent-more-rule {{ margin-bottom: .5rem; }}
.agent-more-found {{ color: var(--accent-text); font-weight: 500; }}
.kpi-ring-svg {{ position: absolute; top: 0; left: 0; transform: rotate(-90deg); overflow: visible; }}
.agent-desc {{
    font-size: var(--text-xs);
    line-height: 1.5;
    color: var(--graphite);
    margin: 5px 0 9px;
}}
.agent-metric {{
    font-family: var(--mono);
    font-size: 11.5px;
    font-weight: 500;
    color: var(--graphite);
    background: var(--sheet-alt);
    padding: 4px 8px;
    border-radius: var(--radius);
    display: inline-block;
}}
/* Compact: the team is a roster of small cards, three or four to a row. Each shows its name, its
   status and one line; the full description and what it found open with the card. */
.agent-grid.org-chart-layout {{
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 10px;
    padding: 6px 0 2px;
    margin: .4rem 0 0;
}}
.agent-details {{ padding: .7rem .85rem; }}
.agent-header {{ margin-bottom: 4px; padding-bottom: 0; border-bottom: none; }}
.agent-line {{ font-size: var(--text-xs); color: var(--graphite); line-height: 1.4; }}
.agent-more {{ margin-top: .6rem; padding-top: .6rem; }}
.agent-more .agent-desc {{ margin: 0 0 .5rem; }}

/* ── Handoff Stream Feed ── */
.handoff-stream {{
    margin: 1.5rem 0;
    border-left: 1px solid var(--rule);
    padding-left: 1.2rem;
}}
.handoff-item {{
    margin-bottom: 1rem;
    position: relative;
}}
.handoff-item::before {{
    content: "";
    position: absolute;
    left: calc(-1.2rem - 4px);
    top: 6px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--pen);
}}
.handoff-meta {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .04em;
    text-transform: uppercase;
    color: var(--accent-text);
    font-weight: 600;
    margin-bottom: 2px;
}}
.handoff-text {{
    font-size: var(--text-sm);
    line-height: 1.55;
    color: var(--ink);
}}

/* ── Summary: the question and its answer ── */
.exec-directive {{
    background: var(--accent-soft);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--pen);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
}}
.exec-directive h2 {{ margin: 0 0 .4rem; }}
.exec-directive .dir-label {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--accent-text);
    font-weight: 600;
    margin-bottom: 5px;
}}
.exec-directive .dir-content {{
    font-size: 17px;
    line-height: 1.55;
    color: var(--ink);
    margin: 0;
}}

/* ── Cards ── */
[data-testid="stExpander"] {{
    background: var(--sheet) !important; border: 1px solid var(--rule) !important;
    border-radius: var(--radius); box-shadow: none;
}}
[data-testid="stExpander"] summary {{ font-weight: 500; font-size: var(--text-sm); color: var(--ink) !important; }}
[data-testid="stExpander"] summary:hover {{ color: var(--accent-text) !important; }}
[data-testid="stCode"] pre, pre {{
    background: var(--code-bg) !important; border: 1px solid var(--rule);
    border-radius: var(--radius); font-size: var(--text-xs); color: var(--ink) !important;
}}
[data-testid="stAlert"] {{ border-radius: var(--radius); }}
/* Notes are neutral. A warning or an error is amber (a step stopped, something to look at);
   red is kept for "this may not hold". Each alert keeps its own icon, so shape and words stay. */
[data-testid="stAlertContainer"] {{
    background: var(--sheet-alt) !important; border-radius: var(--radius);
    border: 1px solid var(--rule);
    padding: .7rem 1rem; color: var(--ink) !important;
}}
[data-testid="stAlertContainer"] p {{ color: inherit !important; font-size: var(--text-sm); }}
[data-testid="stAlertContainer"] svg {{ fill: currentColor; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
    border-color: var(--positive); background: color-mix(in srgb, var(--positive) 8%, var(--sheet)) !important;
}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]),
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
    border-color: var(--accent); background: color-mix(in srgb, var(--accent) 10%, var(--sheet)) !important;
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border-radius: var(--radius); overflow: hidden;
}}

/* ── The seven steps: a vertical list beside the 3D plate ──
   A numbered dot, the step's name and a status word with a shape. A hairline rail runs down through the
   dots and is filled behind each finished or skipped step, so progress reads at a glance. */
.stepper {{
    display: flex; flex-direction: column; width: 100%; max-width: 30rem; box-sizing: border-box;
    margin: 0 0 1.25rem; padding: 18px 20px;
    background: var(--sheet); border: 1px solid var(--rule); border-radius: var(--radius);
}}
.stepper .step {{
    position: relative; display: grid; grid-template-columns: 28px minmax(0, 1fr);
    grid-template-areas: "dot name" "dot state" "dot detail"; column-gap: 14px; row-gap: 2px; padding-bottom: 20px;
}}
.stepper .step:last-child {{ padding-bottom: 0; }}
.stepper .step:not(:last-child)::after {{
    content: ""; position: absolute; left: 13px; top: 32px; bottom: 2px; width: 2px; background: var(--rule);
}}
/* The rail is filled past a finished step and past a skipped one (it was passed, just not needed). */
.stepper .step.done:not(:last-child)::after, .stepper .step.skip:not(:last-child)::after {{ background: var(--pen); }}
.step-dot {{
    grid-area: dot; align-self: start; position: relative; z-index: 1; width: 28px; height: 28px;
    display: grid; place-items: center; border-radius: 50%; background: var(--sheet);
    border: 1.5px solid var(--rule-strong); color: var(--graphite);
    font-family: var(--mono); font-size: 12px; font-weight: 600; line-height: 1;
}}
.step.done .step-dot {{ background: var(--pen); border-color: var(--pen); color: var(--accent-ink); }}
.step.active .step-dot {{ border-color: var(--pen); color: var(--accent-text); outline: 3px solid var(--accent-soft); }}
.step.err .step-dot {{ border-color: var(--accent); color: var(--accent); }}
.step.skip .step-dot {{ border-style: dashed; }}
.step-name {{
    grid-area: name; align-self: end; font-size: var(--text-sm); font-weight: 600; color: var(--ink); line-height: 1.3;
}}
.step:not(.done):not(.active):not(.err) .step-name {{ color: var(--graphite); font-weight: 500; }}
.step.active .step-name {{ color: var(--accent-text); }}
.step-state {{
    grid-area: state; align-self: start; font-family: var(--mono); font-size: 11px; font-weight: 600;
    letter-spacing: .06em; text-transform: uppercase; color: var(--graphite);
}}
.step.done .step-state {{ color: var(--positive); }}
.step.active .step-state {{ color: var(--accent-text); }}
.step.err .step-state {{ color: var(--accent); }}
/* What the step reported, in the step's own row: rows read, tools run, iterations. It wraps, never clips. */
.step-detail {{
    grid-area: detail; align-self: start; margin-top: 4px; font-size: var(--text-xs); line-height: 1.45;
    color: var(--graphite); overflow-wrap: anywhere;
}}

/* A found-so-far item that has just appeared carries `is-new` on its markup (it marks one seen once);
   this rule keeps the class styled, and it does not animate. */
.prov-item.is-new {{ animation: none; }}
.check-row.animate .check {{ animation: none; }}

/* ── Annotations: Insight / Do / Risk, hairline cards with a mono tag ── */
.ic, .rc, .wc {{
    display: flex; gap: 12px; align-items: baseline;
    border: 1px solid var(--rule); border-radius: var(--radius);
    padding: 14px 16px; margin: 0 0 .6rem;
    font-size: var(--text-sm); line-height: 1.55; max-width: 74ch;
    color: var(--ink-2); background: var(--sheet);
}}
.ic .mk, .rc .mk, .wc .mk {{
    flex: none; font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .06em;
    text-transform: uppercase; color: var(--graphite);
}}
.rc .mk {{ color: var(--accent-text); }}
.wc .mk {{ color: var(--danger-text); }}
.wc {{ border-color: var(--risk); }}

.reason {{ background: var(--sheet); border: 1px solid var(--rule);
          border-radius: var(--radius); padding: 1.3rem 1.5rem;
          font-size: var(--text-base); color: var(--ink); line-height: 1.7; max-width: 72ch; }}

.run-banner {{ border-radius: var(--radius); border: 1px solid var(--pen);
              background: var(--accent-soft); padding: .9rem 1.1rem;
              color: var(--accent-text); font-size: var(--text-base); font-weight: 600;
              margin: .4rem 0 1.2rem; }}
.run-banner.ok {{ border-color: var(--positive); background: color-mix(in srgb, var(--positive) 8%, var(--sheet)); color: var(--ink); }}
.run-banner.caution {{ border-color: var(--accent); background: color-mix(in srgb, var(--accent) 9%, var(--sheet)); color: var(--ink); }}
.run-banner .sub {{ display: block; font-weight: 500; color: var(--ink-2);
                   font-size: var(--text-sm); margin-top: 4px; }}

.empty {{ padding: 1.5rem 0 2.5rem; max-width: 62ch; }}
.empty h2 {{ font-size: var(--text-lg); font-family: var(--heading);
            font-weight: 600; line-height: 1.15; margin: 0 0 .75rem; }}
.empty p {{ color: var(--graphite); font-size: var(--text-base); line-height: 1.6; margin: 0; }}
.empty .next-steps {{ margin: 0 0 1rem; padding-left: 1.25rem; color: var(--ink-2);
                      font-size: var(--text-base); line-height: 1.6; }}
.empty .next-steps li {{ margin-bottom: .3rem; }}

/* ── Top bar ── */
.topbar-state {{ color: var(--graphite); font-size: var(--text-sm); overflow-wrap: anywhere; }}
.topbar-state b {{ color: var(--ink); font-weight: 600; }}
.topbar-hint {{ display: block; font-size: var(--text-xs); }}

/* ── Join review (one card per related table) ── */
.join-card {{
    border: 1px solid var(--rule-faint); border-left: 3px solid var(--rule-strong);
    background: var(--sheet); border-radius: var(--radius); padding: .4rem .6rem;
    font-size: var(--text-xs); margin-top: .5rem;
}}

/* ── Footer line ── */
.site-foot {{
    margin: 3rem 0 .5rem; padding-top: 1rem; border-top: 1px solid var(--rule);
    font-size: var(--text-xs); color: var(--graphite); max-width: 68ch; line-height: 1.55;
}}

@media (max-width: 600px) {{
    .agent-grid {{ grid-template-columns: 1fr; }}
    .side-head {{ margin: 1.2rem 0 .5rem; }}
}}

/* ── Audited-entry check row (FrontendPlan.md section 5): a shape and words on every check ── */
.check-row {{ display: flex; flex-wrap: wrap; gap: .5rem 1rem; margin-top: .6rem; }}
.check {{ font-size: var(--text-xs); font-weight: 600; display: inline-flex; align-items: baseline; gap: 6px; }}
.check.ok {{ color: var(--positive); }}
.check.risk {{ color: var(--danger-text); }}
.check.note {{ color: var(--graphite); font-weight: 500; }}

/* ── Details for analysts: exact figures, folded away behind the plain sentence ── */
.tech-note {{ margin: .6rem 0 0; font-size: var(--text-xs); color: var(--graphite); }}
.tech-note summary {{
    display: inline-block; cursor: pointer; font-family: var(--mono); font-size: 11.5px; font-weight: 500;
    letter-spacing: .04em; color: var(--ink-2);
}}
.tech-note summary:hover {{ color: var(--ink); }}
.tech-note p {{ margin: .5rem 0 0; font-family: var(--mono); font-size: 11.5px; line-height: 1.55; color: var(--graphite); overflow-wrap: anywhere; }}

/* ── Audit trail (Details): a titled section, then hairline rows ── */
.audit-head {{ margin: 0 0 .6rem; padding-bottom: .5rem; border-bottom: 1px solid var(--rule-faint); }}
.audit-head .audit-title {{ margin: 0; padding: 0; font-family: var(--heading); font-size: var(--text-base) !important; font-weight: 600 !important; color: var(--ink); }}
.audit-head p {{ margin: 2px 0 0; font-size: var(--text-xs); color: var(--graphite); line-height: 1.45; }}
[class*="st-key-audit_"] {{ border-color: var(--rule); border-radius: var(--radius); }}

/* ── Downloads: an artifact shelf. Each file has a type tag, a purpose, a size and a state ── */
.shelf-group {{
    font-family: var(--mono); font-size: 11px; font-weight: 600; line-height: 1.2; letter-spacing: .08em;
    text-transform: uppercase; color: var(--graphite); margin: 1.4rem 0 .5rem;
}}
[class*="st-key-artifact_"] {{
    background: var(--sheet); border: 1px solid var(--rule); border-radius: var(--radius);
    padding: 12px 16px; margin-bottom: .5rem;
}}
.artifact {{ display: grid; grid-template-columns: 56px minmax(0, 1fr) auto auto; gap: 16px; align-items: center; }}
.artifact .ty {{
    font-family: var(--mono); font-size: 11px; font-weight: 600; line-height: 1; letter-spacing: .06em;
    text-align: center; padding: 6px 0; border: 1px solid var(--rule); border-radius: 3px;
    color: var(--ink-2); background: var(--stock);
}}
.artifact .nm {{ font-family: var(--mono); font-size: var(--text-xs); font-weight: 600; line-height: 1.3; color: var(--ink); overflow-wrap: anywhere; }}
.artifact .why {{ margin-top: 3px; font-size: var(--text-xs); color: var(--graphite); line-height: 1.45; }}
.artifact .sz {{ font-family: var(--mono); font-size: 12px; color: var(--graphite); white-space: nowrap; }}
.avail {{
    font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .04em;
    text-transform: uppercase; color: var(--positive); white-space: nowrap;
}}
.avail.off {{ color: var(--graphite); }}
@media (max-width: 900px) {{
    .artifact {{ grid-template-columns: 56px minmax(0, 1fr); }}
    .artifact .sz, .artifact .avail {{ grid-column: 2; }}
}}

/* ── "How we got here" (Details): hairline-ruled blocks, tokens only ── */
/* An audit trail: one row per question, the question on the left and the answer on the right, so the
   rows line up whatever their length. The header strip spans the whole card. */
.how-we-got-here {{ margin: 0 0 2rem; background: var(--sheet); border: 1px solid var(--rule); border-radius: var(--radius); overflow: hidden; }}
.how-head {{ padding: 16px 22px 14px; background: var(--sheet-alt); border-bottom: 1px solid var(--rule); }}
.how-title {{ font-family: var(--heading); margin: 0 0 .2rem; padding: 0; }}
.how-lede {{ color: var(--graphite); font-size: var(--text-xs); line-height: 1.55; margin: 0; padding: 0; }}
.how-grid {{ display: block; padding: 0; }}
.how-block {{ display: grid; grid-template-columns: minmax(180px, .7fr) minmax(0, 2.3fr); gap: 0 2rem;
    border-top: 1px solid var(--rule-faint); padding: 1.1rem 22px 1.2rem; min-width: 0; }}
.how-block:first-child {{ border-top: none; }}
.how-label, .how-body {{ min-width: 0; }}
.how-h {{ font-size: var(--text-sm); font-weight: 600; margin: 0 0 .25rem; color: var(--ink); }}
/* A small square marker in the row's colour: green for a decision, amber for a workaround, brick only
   for numbers that could not be traced (those may not hold). */
.how-h::before {{ content: ""; display: inline-block; width: 8px; height: 8px; margin-right: 8px; border-radius: 2px; background: var(--pen); }}
.how-block.warn .how-h::before {{ background: var(--accent); }}
.how-block.risk .how-h::before {{ background: var(--risk); }}
.how-note {{ font-size: var(--text-xs); color: var(--graphite); line-height: 1.5; margin: 0; }}
@media (max-width: 760px) {{
    .how-block {{ grid-template-columns: 1fr; gap: .5rem; }}
}}
.how-lead {{ font-size: var(--text-sm); font-weight: 600; color: var(--ink); line-height: 1.5; margin: 0 0 .4rem; }}
.how-text {{ font-size: var(--text-sm); color: var(--ink-2); line-height: 1.6; margin: 0 0 .5rem; overflow-wrap: anywhere; }}
.how-sub {{ font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--graphite); margin: .6rem 0 .2rem; }}
.how-list {{ margin: 0 0 .4rem; padding-left: 1.1rem; font-size: var(--text-sm); line-height: 1.55; color: var(--ink-2); overflow-wrap: anywhere; }}
.how-more {{ font-size: var(--text-xs); color: var(--graphite); margin: 0; }}
/* A chip: a glyph and a word, tinted by status. A refuted idea "does not hold", so it is the brick one. */
.how-tag {{ display: inline-block; padding: 1px 8px; margin-right: 6px; border: 1px solid currentColor; border-radius: var(--radius-pill);
    font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
    color: var(--graphite); background: var(--sheet); vertical-align: 1px; }}
.how-tag.supported {{ color: var(--positive); background: color-mix(in srgb, var(--positive) 8%, var(--sheet)); }}
.how-tag.refuted {{ color: var(--danger-text); background: color-mix(in srgb, var(--risk) 8%, var(--sheet)); }}
.how-ideas {{ list-style: none; padding-left: 0; }}
.how-ideas li {{ margin-bottom: .55rem; }}
/* ── "Found so far, may change" (live run): hairline block, tokens only ── */
.prov {{ border-top: 1px solid var(--rule); padding: .8rem 0 .4rem; margin: .8rem 0; max-width: 100%; }}
.prov-h {{ font-size: var(--text-sm); font-weight: 600; margin: 0 0 .15rem; color: var(--ink); }}
.prov-note {{ font-size: var(--text-xs); color: var(--graphite); line-height: 1.5; margin: 0 0 1rem; }}
.prov-list {{ margin: 0; padding-left: 1.1rem; font-size: var(--text-sm); line-height: 1.55; color: var(--ink); overflow-wrap: anywhere; }}
.prov-kind {{ font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--graphite); display: block; margin-bottom: 4px; }}
.prov-count {{ font-size: var(--text-xs); color: var(--graphite); margin: .4rem 0 0; }}
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
    background: var(--stock) !important; border-color: var(--rule-strong) !important;
}}
[data-testid="stTextInput"] button, [data-testid="stNumberInput"] button {{ color: var(--ink) !important; }}
[data-testid="stNumberInput"] input {{ background: var(--stock) !important; color: var(--ink) !important; }}
[data-testid="stNumberInput"] button {{ background: var(--stock) !important; color: var(--ink) !important; }}
/* Segmented controls (Day / Night, "With an AI summary"). The native theme is fixed Day, so in Night its
   selected segment painted a light fill under light text and the label vanished. Both states take their
   colours from the tokens: the selected one is the green pen with the button-label colour (9.6:1 in Day,
   5.3:1 in Night), the other is a card with body text. The words inside inherit, whatever element holds them. */
[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_control"] {{
    background: var(--sheet) !important; color: var(--ink-2) !important; border-color: var(--rule-strong) !important;
}}
[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_control"]:hover {{
    border-color: var(--graphite) !important; color: var(--ink) !important;
}}
[data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {{
    background: var(--pen) !important; color: var(--accent-ink) !important; border-color: var(--pen) !important;
}}
[data-testid="stButtonGroup"] button * {{ color: inherit !important; }}
/* ── Evidence inspector (Answers): ranked findings beside the selected one ── */
.st-key-selected_finding [role="radiogroup"] {{ gap: 8px; }}
.st-key-selected_finding [role="radiogroup"] > label {{
    padding: 14px 16px; margin: 0; align-items: flex-start;
    background: var(--sheet); border: 1px solid var(--rule); border-radius: var(--radius);
}}
/* The finding list is clickable, so it may answer the pointer; the selected one is marked by a bar. */
.st-key-selected_finding [role="radiogroup"] > label:hover {{ border-color: var(--graphite); }}
.st-key-selected_finding [role="radiogroup"] > label:has(input:checked) {{
    border-color: var(--accent-text); box-shadow: inset 3px 0 0 var(--accent-text);
}}
.st-key-selected_finding [role="radiogroup"] p {{ font-size: var(--text-sm); line-height: 1.45; }}
.st-key-evidence_panel {{ background: var(--sheet); border-radius: var(--radius); }}
.evidence-head {{ display: flex; flex-direction: column; align-items: flex-start; gap: .5rem; margin-bottom: .4rem; }}
.evidence-title {{ font-family: var(--heading); font-size: var(--text-lg) !important; font-weight: 600 !important; line-height: 1.3; margin: 0; padding: 0; }}
.evidence-detail {{ color: var(--ink-2); font-size: var(--text-sm); line-height: 1.6; max-width: 68ch; margin: .4rem 0 .6rem; }}
.evidence-none, .evidence-source {{ color: var(--graphite); font-size: var(--text-xs); margin: .4rem 0; }}
.evidence-caveats {{ font-size: var(--text-sm); color: var(--ink); margin: .6rem 0; }}
.evidence-caveats ul {{ margin: .25rem 0 0; padding-left: 1.2rem; color: var(--ink-2); line-height: 1.5; }}
.verdict-mark {{
    display: inline-flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 11px; font-weight: 600;
    letter-spacing: .06em; text-transform: uppercase;
    padding: .15rem .65rem; border-radius: var(--radius-pill); border: 1px solid currentColor; background: var(--sheet);
}}
.verdict-mark.held {{ color: var(--positive); }}
.verdict-mark.needs_more {{ color: var(--risk-text); }}
.verdict-mark.unchecked {{ color: var(--graphite); border-style: dashed; }}

/* ── Accessibility states ── */
/* Touch: every control is at least 44 px tall. */
@media (pointer: coarse) {{
    .stButton button, .stDownloadButton button, [data-testid="stTabs"] [role="tab"],
    [data-testid="stExpander"] summary, [data-testid="stButtonGroup"] button {{ min-height: 44px; }}
}}
/* Higher contrast: heavier outlines, no tinted surfaces standing in for borders. */
@media (prefers-contrast: more) {{
    :focus-visible {{ outline-width: 3px !important; }}
    .finding-card, .bento-card, .gauge, .exec-directive, .datum, .trust-strip, .stepper, .agent-card,
    [data-testid="stExpander"], [class*="st-key-artifact_"] {{ border-color: var(--rule-strong) !important; }}
    .check, .run-banner {{ border: 1px solid currentColor; }}
}}
/* Reduced transparency: nothing translucent sits behind text. */
@media (prefers-reduced-transparency: reduce) {{
    .finding-card, .bento-card, .run-banner, .check, .defect-stamp, .wc {{ background: var(--sheet) !important; }}
    * {{ backdrop-filter: none !important; }}
}}
</style>
""", unsafe_allow_html=True)
