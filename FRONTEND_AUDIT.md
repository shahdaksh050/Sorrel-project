# Frontend audit (Phase 1)

Branch `frontend-overhaul`, HEAD `96c4598`. No code was changed to produce this file.
Where the brief and the code disagree, the code wins. Section 8 lists every disagreement.

**Method.** I read `app.py`, `ui/styles.py`, `ui/landing.py`, `ui/animations.py`, `ui/run.py` (first 200 lines), `ui/components/provisional.py`, the top of `ui/components/cards.py`, `ui/tabs/answers_tab.py`, `ui/pipeline_3d.py` (lines 60+), `src/core/design_tokens.py`, `.streamlit/config.toml`, `.github/workflows/ci.yml`, and the head of `docs/FrontendOverhaulPlan.md`.
I did **not** read in full: `ui/components/cards.py` (below line 180), `how_we_got_here.py`, the Charts/Details/Downloads tabs, `ui/cinematic_3d.py`, the 3D JS assets, `landing_component/index.html` (119 KB), or the second half of `ui/run.py`. Facts about those files below are from grep or directory listings only, and say so.

## 1. File-by-file map

| File | Role |
|---|---|
| `app.py` (1252 lines) | One top-to-bottom script. Routing (landing vs workspace), session-state defaults, sidebar (theme, provider/key/model, engine, privacy, analysis settings), upload and its security checks, run start, polling fragment, hero, results tabs, empty state. |
| `ui/styles.py` | `inject_theme_css()`: one big `<style>` block with all tokens and component classes, Day/Night by Python-side state. |
| `ui/landing.py` + `ui/landing_component/` | v1 `declare_component("landing", path=...)`. A full-screen fixed iframe (`index.html`, own Three.js scene, own CSS/JS, own vendored fonts, anime.js and a private copy of three.module.js). Returns `{"enter", "theme"}`. |
| `ui/animations.py` | `inject_micro_interactions()`: a zero-height `components.html` iframe that runs JS in the *parent* document (count-up, stagger, hover-float, donut gauge) and installs a body-wide `MutationObserver`. |
| `ui/pipeline_3d.py` + `ui/assets/pipeline_3d.{html,js}` | The 7-stage "plate". HTML document built in Python and mounted with `st.iframe` (or `components.html` on old Streamlit). |
| `ui/cinematic_3d.py` + `ui/assets/cinematic_3d.{html,js}` | Optional "3D Cinematic Showcase" hero, same mounting. Also exported as a standalone HTML download. |
| `ui/components/cards.py` | `STAGE_DEFS`, `gauge`, `section`, `stage_card`, `render_steps_list`, `render_datum`, `render_finding_card`, defect/cert stamps, agent grid, chart rendering, `get_vega_config`. All HTML-string builders; dynamic text goes through `html.escape`. |
| `ui/components/provisional.py` | Pure HTML for "Found so far, may change". |
| `ui/components/how_we_got_here.py`, `icons.py` | Details-tab audit trail; inline SVG icons. |
| `ui/tabs/*.py` | `answers_tab`, `charts_tab`, `details_tab`, `downloads_tab`. |
| `ui/run.py` | `RunSpec` (frozen), `ActiveRun` (worker thread, lock, stop event), `RunSnapshot`, `RunOutcome`. Imports no Streamlit. |
| `static/` | Served at `/app/static` (`enableStaticServing = true`). Fonts (Baloo 2, Mukta; many subsets, ~1.1 MB), `vendor/three/` (1.33 MB), `vendor/vega/` (vega, vega-lite, vega-embed). |
| `src/core/design_tokens.py` | **The token source of truth, and it lives under `src/`.** |

Other consumers of the same tokens: `src/core/html_report.py` (shareable report), `src/core/chart_theme.py`, `ui/landing.py` (5 keys), `ui/pipeline_3d.py`, `ui/cinematic_3d.py`, `scripts/sync_streamlit_theme.py` (writes the generated block in `.streamlit/config.toml`). `tests/test_theme_sync.py` guards the sync.

## 2. State, routing, worker/snapshot/fragment, Day/Night

**Routing.** `st.session_state["entered"]` false: the landing runs, `st.stop()`s, and the rest of `app.py` never executes. A hidden `st.button("HIDDEN_ENTER")` is pushed off-screen by a local `<style>`, and the iframe also returns `{"enter": true}`; either one sets `entered` and reruns. On the landing page the injected CSS forces **every** `iframe` to `position: fixed; 100vw x 100vh; z-index: 999999`. Anything else iframe-based placed on the landing page would be pulled full-screen too.

**Session state.** `_DEFAULTS` (about 22 keys) is applied once. Dataset preview (`preview_df`, `preview_bytes`, `preview_name`, `orig_name`) is kept in session state so it survives reruns. `_reset_pipeline()` clears run results but not the preview. Removing the uploaded file resets everything except `theme`.

**Run flow.** Click `Run analysis` (or the sample button, which sets `_sample_run` and reruns). The script builds a frozen `RunSpec`, runs an LLM preflight `ping()` (skipped without an LLM), starts `ActiveRun`, stores it in `session_state["_run"]`, and calls `st.rerun()`. On the next pass `_run_progress()` (an `st.fragment(run_every="1s")`) calls `run.snapshot()` and renders: run banner, steps list, "Found so far", Stop button, caption.
When `snap.state != "running"` the fragment pops `_run`, folds the outcome into session state, and calls a **full** `st.rerun()`. Stop uses `st.rerun(scope="fragment")`.
The worker never touches `st.*`; confirmed in `ui/run.py` docstring and imports.

**What is and is not inside the fragment.** Inside: banner, steps list, provisional block, Stop. Outside (drawn once per script pass, so **not** refreshed each tick): hero, datum bar, the 3D plate, the dataset preview. During a run the plate does not animate stage changes between full reruns; it is effectively ambient already.

**Day/Night.** The sidebar control is a **`st.selectbox("Theme", ["Day Mode","Night Mode"])`**, not a toggle. Changing it writes `session_state["theme"]` and calls `st.rerun()`. `inject_theme_css()` then re-emits the whole `<style>` block with the other token set (`css_root_block(mode)`). So a theme switch is a full script rerun, and the CSS (roughly 35 KB) is re-sent on every rerun. The 3D iframes receive `theme` as an argument and rebuild their document. The landing has its own toggle inside the iframe and writes the choice back through the component value.
Every session starts in Day by design; the browser colour scheme is deliberately ignored.

## 3. `ui/styles.py`

**Tokens** (from `design_tokens.py`; actual names, not the brief's):
`--stock` page, `--sheet` card, `--sheet-alt`, `--ink` text, `--graphite` muted text, `--pen` primary (terracotta), `--pen-hover`, `--risk`, `--accent` (amber), `--positive`, `--rule`, `--rule-faint`, `--margin`, `--code-bg`. Plus styles.py-local `--lift`, `--lift-sm`, `--radius` (14px), `--radius-pill`, `--sans` (Mukta), `--heading` (Baloo 2), `--mono`.

**Component classes** (all token-driven): `.datum`/`.cell`, `.sect`, `.hero`/`.hero.compact`, `.step-head`/`.step-n`, `.side-*`, `.gauge`(+`.flag`), `.defect-stamp`, `.cert-stamp`, `.finding-card`, `.trust-strip`, `.du`, `.iso-badge`, `.kpi-*`, `.agent-*` (+`pulseActive` infinite animation), `.bento-grid`/`.bento-card` (**a CSS-only "bento" already exists**, only partly used), `.handoff-*`, `.exec-directive`, `.sc*` (steps list), `.ic/.rc/.wc`, `.reason`, `.run-banner`, `.empty`, `.check-row`, `.how-*`, `.prov-*`.

**Selectors that depend on Streamlit internals** (these break on a Streamlit upgrade):
`stHeader`, `stStatusWidget`, `stSidebar`, `stFileUploader`/`stFileUploaderDropzone`/`...DropzoneInstructions`, `stMetricValue/Label`, `stExpander`, `stAlert`/`stAlertContainer`/`stAlertContent{Success,Error,Warning}`, `stDataFrame`, `stTable`, `stHorizontalBlock`, `stColumn`, `stCaptionContainer`, `stWidgetLabel`, `stCheckbox`, `stRadio`, `stToggle`, `stTooltipIcon`, `stTextInput`, `stNumberInput`, `stCode`, `stTabIndicator`, `stTabsContent`, `stVegaLiteChart` (animations.py), `stElementContainer`; `data-baseweb` `tab-panel`, `input`, `base-input`, `textarea`, `select`; classes `.stApp`, `.stButton`, `.stDownloadButton`, `.stTabs`, `.stMarkdown`, `.stTextInput`, `.stTextArea`, `.stSelectbox`, `.stSlider`, `.block-container`, `.st-key-plate`; heavy use of `:has()` and `!important`.

**Fonts:** `@import url('app/static/fonts/ledger-fonts.css')`: local, no CDN.

**Gradients:** none anywhere in `styles.py`. The `color-mix(...)` tints are flat fills. (Relevant to the Spotlight/Glow exception.)

### Contrast of the current tokens (computed, WCAG 2.x)

| Pair | Day | Night | Note |
|---|---|---|---|
| ink on stock | 10.75 | 14.21 | ok |
| graphite on stock / sheet / sheet-alt | 5.11 / 5.48 / 4.74 | 7.10 / 6.37 / 5.88 | ok |
| pen on stock / sheet | 4.95 / 5.31 | 7.78 / 6.99 | ok |
| sheet on pen (button text) | 5.31 | 6.99 | ok |
| risk on sheet | 6.34 | **4.46** | Night: 0.04 short of AA text |
| risk on its own 8% tint (`.wc`, `.defect-stamp`) | 5.59 | **4.04** | Night fails AA for risk-coloured body text |
| positive on stock / sheet | **4.49** / 4.81 | 7.00 / 6.29 | Day on stock: 0.01 short |
| **accent on stock / sheet** | **2.32 / 2.49** | 6.77 / 6.08 | Day `--accent` fails even 3:1; unusable for text or as the only cue |
| **rule on stock / sheet** | **1.65 / 1.77** | **1.79 / 1.61** | Used as *input* and dropzone borders: fails 3:1 for UI components (WCAG 1.4.11). Fine for decorative dividers only |

These are existing issues, independent of the overhaul. Fixing `--risk` (Night) or `--accent` (Day) at the source means editing `src/core/design_tokens.py`, which the brief forbids; see risk R1.

## 4. Streamlit, components v2, Node

- Installed Streamlit **1.63.0**; `requirements.txt` pins `streamlit>=1.49.0`.
- `streamlit.components.v2` **is importable** here. Signature: `component(name, *, html, css, js, isolate_styles=True)`. Verified from the installed docstring: `asset_dir`-served files exist **only for components shipped inside an installed package**; for an app-local component the JS/CSS must be passed as raw strings. I have not checked which Streamlit release introduced v2, so the `>=1.49.0` floor does not guarantee it.
- The repo already uses v1 (`streamlit.components.v1.declare_component`, imported as a module, which is the supported form per the v1 docstring).
- **Node v24.5.0, npm 11.5.1** are installed on this machine. Python 3.13.5 in `.venv`.
- **Playwright is not installed** in `.venv` (CI installs it for `scripts/ui_smoke.py`). The IDE's browser tool also fails here (Playwright driver download returns 404), so I cannot drive a browser from this session until that is solved. See R9.

## 5. Inventory of every screen and state

| # | State | Where it renders |
|---|---|---|
| 1 | Landing | `ui/landing.py` iframe; `app.py` stops after it |
| 2 | Workspace, empty (no file) | marketing hero + idle plate + datum "Ready" + bordered input card + "Let's see what your data shows" + "Meet Your Helpers" grid + sample button |
| 3 | File uploaded | adds Dataset preview (4 gauges, "First 10 rows", dtype table, describe table); notices `<details>` if the reader auto-repaired |
| 4 | Upload rejected / unreadable | `st.error` inside the input card |
| 5 | Configured vs. not runnable | `Run analysis` disabled; caption explains missing file or key |
| 6 | Related tables + join review | per-table card with radio (Use this join / Skip / Change key) |
| 7 | LLM unreachable (preflight) | `st.error` + `st.code` + `st.info`, stage 2 marked error, plate drawn, `st.stop()` |
| 8 | Running, stages 1 to 7 | compact hero "Analysing {file}", inputs collapsed, fragment panel; each stage `pending / active / done / skipped / error` |
| 9 | Running: queued behind other runs, stop requested, cancelling | `snap.note` / `sub` text in the banner |
| 10 | Running: "Found so far" | `.prov` block, latest 8 of N |
| 11 | Stopped early | results + warning "stopped before it finished"; datum "Stopped early" |
| 12 | Completed | sample banner (if sample), optional `llm_warning`, 4 tabs: Answers, Charts, Details, Downloads; "Show how it's working" expander |
| 13 | Completed with LLM fallback | `st.warning(llm_warning)` above tabs |
| 14 | Failed (worker produced no outcome) | hero "Could not finish {file}". **`analysis_error` is written but never displayed** in `app.py` (it is only read to pick the hero text), so the reason is not shown. A presentation-only gap. |
| 15 | Night variants of all the above | same markup, other token set |
| 16 | No-WebGL / reduced-motion | handled inside the 3D assets (not re-verified; those JS files were not read) |

## 6. Tests and how to run

- App: `.venv\Scripts\python.exe -m streamlit run app.py` (from the repo root).
- Lint: `ruff check .`. Types: `mypy src/`. Tests: `pytest -m "not slow" -q` (`addopts` already adds `-m "not slow"`). Also `python scripts/validate.py`, `python scripts/dry_run.py`, `python scripts/ui_smoke.py` (Playwright; drives the real sample run, finds the landing CTA as `#hero-enter-btn` **inside the landing iframe**).
- CI (`.github/workflows/ci.yml`): `quality` job on windows+ubuntu (py3.13, plus py3.11 on ubuntu): ruff, mypy, pytest, validate, dry_run, bench; `ui-smoke` job on ubuntu with Playwright. **CI installs only Python.** Any committed `static/islands/` build must work with no Node step, and no CI step may require one.
- UI-relevant tests I found by name: `test_landing`, `test_landing_v2`, `test_pipeline_3d`, `test_cinematic_3d`, `test_provisional`, `test_how_we_got_here`, `test_run_view`, `test_run_worker`, `test_run_summary`, `test_html_report`, `test_chart_theme`, `test_theme_sync`, `test_architecture`.
- `tests/test_architecture.py` has three tests (tools, engine, memory import rules). None constrains `ui/`.

**Baseline on this machine, before any change** (so later failures are attributable):

| Check | Result |
|---|---|
| `ruff check .` | pass |
| `mypy src/` | **fails before checking anything**: `numpy/__init__.pyi:737: Type statement is only supported in Python 3.12 and greater`, because `pyproject.toml` sets `python_version = "3.11"` and the installed numpy stubs use 3.12 syntax. A local environment/config mismatch; not caused by the frontend. |
| `pytest -m "not slow" -q -x` | stopped at the first failure: **659 passed, 1 failed** (`test_duckdb_engine_level_lock_holds_even_without_regex`; `duckdb` is not installed in this venv). Tests after it did not run because of `-x`. |

## 7. Pre-existing runtime calls to third-party hosts

The brief says there must be none. I found these by searching `ui/` for `https://`:

| File | What it loads |
|---|---|
| `ui/animations.py:135` | `anime.js` from `cdnjs.cloudflare.com`, on **every** workspace page, in the parent document |
| `ui/assets/pipeline_3d.html:266-267` | GSAP (cdnjs) and anime.js 4.5 (jsdelivr) |
| `ui/assets/cinematic_3d.html:9-14, 1306` | Google Fonts, fullPage.js CSS and JS (cdnjs) |
| `ui/pipeline_3d.py:77` | Three.js from jsdelivr, but only as a fallback after the local copy |
| `ui/landing_component/index.html` | outbound `<a href>` links to GitHub and a canonical/og URL (links, not loads) |

Under `LOCAL_ONLY=true` or an offline machine, the first three are real violations of constraint 6. They are inside `ui/`, so they are in scope to fix by vendoring. I have not changed anything yet. See the plan's Step 1b.

## 8. Discrepancies between the brief / docs and the code

1. **`DESIGN.md`, `HANDOVER.md`, `PLAN.md` and a root `IMPROVEMENTS.md` do not exist.** Only `docs/IMPROVEMENTS.md`, `docs/FrontendOverhaulPlan.md`, `docs/FrontendImplementationPlan.md`, `docs/FutureScope.md`, `README.md`, `AGENTS.md` do. `AGENTS.md` and `design_tokens.py` still cite `DESIGN.md`; `FrontendOverhaulPlan.md` says `HANDOVER.md` was deleted. "Reconcile DESIGN.md" therefore means **writing it** from `design_tokens.py` + `styles.py`; there was no old one to read. I read `docs/FrontendOverhaulPlan.md` only to its first 70 lines and `README.md` only by headings.
2. **Token names and palette.** The code uses `--stock/--sheet/--ink/--graphite/--pen/...`, not `--bg-color/--text-main/--accent`. The palette is a pale **buff-green** (`#eef1e0`) with **faint blue** rules, not warm cream. The "two accent pens" are `--pen` (terracotta `#a34f20`) and `--accent` (amber `#e08a3e`). `--accent` is *not* the primary; `--pen` is.
3. **Tokens are in `src/`** (`src/core/design_tokens.py`), which constraint 1 forbids touching. New tokens must go in `styles.py` (as `--lift` already does). Changing the palette itself is not possible without an AGENTS.md "Ask First" change, and would desync `html_report.py`, the 3D scenes and `config.toml`.
4. **Day/Night is a selectbox**, plus a separate toggle inside the landing iframe.
5. **"anime.js already vendored"**: it is vendored only inside `ui/landing_component/vendor/`. The workspace loads anime.js from a CDN.
6. **`git diff master...frontend-overhaul`** is 285 files / +184k lines and includes **97 files under `src/`** and most of the tests. The branch had diverged from master long before this task, so "no changes under `src/`" must be measured against the commit I start from (`96c4598`), not master.
7. **Landing is a full-page iframe with its own HTML**, not a Streamlit layout. A React "hero island" cannot be a second component beside it; it has to load *inside* that document (plan section 3).
8. **`.gitignore` already ignores any `build/` and `dist/`** directory anywhere. The island build output must not use those names inside `static/islands/`.
9. A **CSS-only bento grid** (`.bento-grid`, `.bento-card`) already exists in `styles.py`.

## 9. Risk list

| ID | Risk | Impact | Mitigation direction |
|---|---|---|---|
| R1 | Palette-level fixes (Night `--risk`, Day `--accent`) live in `src/` | Cannot fix at the source; local overrides desync the report/3D | Add *derived* overlay tokens in `styles.py` for UI text (e.g. a Night-safe risk text colour); ask before touching `src/core/design_tokens.py` |
| R2 | `--rule` used as input/dropzone border (1.6 to 1.8:1) | WCAG 1.4.11 failure today | New `--rule-strong` overlay token for control borders |
| R3 | Streamlit-internal selectors | A Streamlit upgrade silently breaks styling | Keep the list in section 3; add a smoke check that key `data-testid`s exist |
| R4 | Fragment ticks every 1 s | Any iframe/animation/`<style>` in the fragment can remount, flicker or duplicate | Stable keys; props only from changed snapshot fields; Spike B measures it |
| R5 | `animations.py` runs a body-wide `MutationObserver` and re-animates any new `.gauge`/`.cell`; hover-float is an infinite anime loop | Interacts with anything new added to the DOM; violates "once, outside the fragment" for pointer scripts | Replace in Step 6 with one idempotent, reduced-motion-aware script |
| R6 | Third-party CDN loads (section 7) | Breaks offline / `LOCAL_ONLY`; privacy | Vendor GSAP, anime.js, fullPage.js and the fonts |
| R7 | `iframe { position: fixed ... }` on the landing | Would hijack any island iframe placed on the landing | Island scoped to the landing document, not a second iframe |
| R8 | Hidden-button landing handshake | Fragile; CI depends on `#hero-enter-btn` in the landing iframe | Keep the id and the handshake unchanged |
| R9 | No browser available: IDE browser tool fails, Playwright not installed locally | Cannot do the "watch a live run in Day and Night" checks without a fix | Install `playwright` + chromium locally (dev only), or you run the spikes and send screenshots |
| R10 | `analysis_error` never displayed on a failed run | User sees "Could not finish" with no reason | Display-only fix, in scope |
| R11 | Local `mypy` and one `pytest` failure are environmental | Could mask real regressions | Record this baseline; compare after each step |
| R12 | `st.markdown` content passes through Streamlit's sanitiser | Custom elements, some attributes may be stripped | Prefer classes and inline SVG already proven in `icons.py` |
