# Frontend overhaul plan (Phase 2)

Starts from branch `frontend-overhaul` at `96c4598`. Read `FRONTEND_AUDIT.md` first; this plan relies on its numbers.
**Nothing here has been implemented.** Statements marked *(unverified)* are from memory or inference and are to be checked in the spikes.

## 0. Decisions I need from you

| # | Decision | My recommendation |
|---|---|---|
| D1 | Design direction: A "Evolve Ledger" or B "New identity" (section 1) | **A** |
| D2 | Island mechanism: v1 `declare_component` (iframe) or v2 `st.components.v2` (section 4.1) | **v1** |
| D3 | The hero island loads *inside* the existing landing document, not as a second component (section 4.2) | approve |
| D4 | Vendor the existing CDN loads (GSAP, anime.js, fullPage.js, Google Fonts) in Step 1b; and decide what to do about the **fullPage.js and GSAP licences** (section 7) | approve vendoring; you decide on the licences |
| D5 | 21st.dev nominations are **blocked** (section 3): approve using shadcn/ui (MIT) as the reference for those slots, or send me picks | approve substitution |
| D6 | OK to `pip install playwright` + `playwright install chromium` into `.venv` as a **dev-only** tool (not added to `requirements.txt`), so I can watch real runs in Spike B? The IDE browser tool is broken here | approve |
| D7 | Props are plain display text rendered only as React text nodes, not pre-`html.escape`d (section 4.4) | approve |
| D8 | Largest bento tile = top-ranked finding; the written answer and the "You asked" heading stay native above it (section 4.6) | approve |
| D9 | `src/core/design_tokens.py` is under `src/`, so Night `--risk` text (4.04:1 on its tint) and Day `--accent` (2.3:1) cannot be fixed at source without your OK. Plan: derived overlay tokens in `styles.py` only. Say if you want the source edited instead (AGENTS.md "Ask First") | overlay only |
| D10 | Aceternity licence stance (section 3) | **resolved**: Aceternity is inspiration only; no code used. Using Magic UI (MIT) + original CSS. |

## 1. The two design directions

Both keep: WCAG AA (checked numerically per token pair, in `tests/`), `--risk` meaning only "this number may not hold", plain-language copy, answers first, `prefers-reduced-motion` for every animation including inside islands, keyboard access, and the 3D views' existing `role="img"` summaries, keyboard controls and no-WebGL text.

### A. Evolve Ledger (recommended)

Keep the buff-green paper, faint-blue rules, terracotta `--pen` and amber `--accent`. Add depth and motion on top.

- **New overlay tokens, in `styles.py` only** (never in `src/`): `--rule-strong` (control borders that meet 3:1), `--glow` and `--spot` (made with `color-mix` from `--pen`/`--accent`, translucent), `--risk-text` (Night-safe text colour for `.wc`/`.defect-stamp`), `--lift-lg`, and motion tokens `--ease-out`, `--dur-fast/-base/-slow`.
- **Glow/spotlight colours** come only from `--pen` and `--accent`; `--risk` only on a risk-flagged item. On Day, `--accent` is 2.3:1 on the page, so it is used **only as a decorative tint**, never as text, an icon or the sole signal.
- **Exception to "no gradients"** (to be written into `DESIGN.md`): allowed only for the four ambient effects Spotlight, Glowing Effect, Hover Border Gradient and Card Spotlight. All stops derive from tokens, low alpha, never behind running text at more than a faint tint, never on tables or charts, static (no motion) under `prefers-reduced-motion`.
- Everything else stays flat: no gradients on surfaces, buttons or backgrounds.

### B. New identity

New palette and style. Not possible within your constraints as written: the palette lives in `src/core/design_tokens.py`, which is also read by `html_report.py`, the 3D scenes, the landing mirror, `chart_theme.py` and `.streamlit/config.toml` (via `scripts/sync_streamlit_theme.py`). B needs that one file edited, which constraint 1 forbids and AGENTS.md lists as Ask First.
`DESIGN.md` does not exist, so I cannot list "every rule it supersedes" from it. The rules stated in `design_tokens.py`/`styles.py` that B would replace are: buff-green stock and sheet values; faint-blue hairline rules; the wine `--margin` chosen for hue separation from `--risk`; the two-pen split; the Baloo 2 + Mukta pairing (if you want new type); and the no-gradient rule (superseded under A as well).
If you choose B I need an explicit waiver for `src/core/design_tokens.py`, and I will extend `tests/test_theme_sync.py` so the five consumers cannot drift.

## 2. Reconciling the docs

`DESIGN.md` is created in Step 1 from `design_tokens.py` and `styles.py`, in the same commit as the token overlay. It records: tokens and contrast table, the `--risk` rule, the gradient exception above, motion rules, the island contract. `AGENTS.md` and `design_tokens.py` already point at it, so no other doc edits are needed for that link.

## 3. Tool verification & Design Behaviours

**Aceternity UI is used as design inspiration only.** Its official licence forbids redistribution of source files. No Aceternity code is copied, ported, or adapted.

Replacements are built from **Magic UI** (MIT) or written as **original CSS** from scratch, matching the desired behaviour:

| Desired Behaviour | Implementation |
|---|---|
| **Hero Spotlight** | Original CSS (`radial-gradient` from `--accent`). Static under reduced motion, paused when hidden. |
| **Hero Headline Fade** | Derived from Magic UI `text-animate` (MIT). Trimmed to by-word blurIn only, no Tailwind, `LazyMotion` only. Plays once per session. |
| **Progress RunStepper** | Original CSS + React. Look of a fading step list, controlled by python props, no timers, no loops. |
| **Pointer-following glow** | Tier-2 pure CSS/JS: one shared pointer listener setting `--mx/--my` for a radial gradient on hovered cards. |
| **Hover Border Gradient** | Original Tier-2 CSS (`@property` angle), hover/focus only. |
| **Dropzone styling** | Original Tier-2 CSS: native `stFileUploader` with dashed `--rule-strong` border, hover lift and glow. |
| **Tabs / Sliding pill** | Original Tier-2 CSS: styling the native Streamlit tabs indicator. |

**Magic UI integration:** Each adopted component (e.g. `text-animate`) is read and trimmed: dependencies like `next-themes` and `@radix-ui/react-icons` are removed, hard-coded colours are replaced by tokens, and reduced-motion/accessibility are added. An attribution header, MIT notice, and snapshot date are kept in each adopted file, plus a central note in `THIRD_PARTY_UI.md`.

**Sliding-pill tabs.** `styles.py` currently hides `stTabIndicator` (`display: none`). Streamlit's own indicator already slides between tabs; restyling it as the pill would give the motion with no JS. *(Unverified: needs a check that it renders as a moving absolutely-positioned element in 1.63.)* Fallback if not: a static pill on the selected tab, as today.

**Other named tools**
- **Spline:** not used, as instructed.
- **canvasui.dev, skipper.ui, animmaster, threeui:** not investigated. `motion` and the vendored Three.js/anime.js already cover the need; I will report them as skipped, not as "verified absent".
- **manus.im:** not integrated.
- `design-references/` does not exist yet; nothing to follow.

### 21st.dev nominations: blocked

I could not make exact, licence-checked picks.
- The 21st.dev listing pages are client-rendered: the fetched HTML contains the category links but no component entries or licence fields.
- The IDE browser subagent failed at start-up (Playwright driver download returned 404), so I could not read component detail pages.
- Licences on 21st.dev are set **per component by its author**, and the site's own presentation (demos, previews) is separately owned by 21st Labs. So a nomination without reading the detail page would be a guess.

What I verified: these category pages exist (`/community/components/s/` + `stat`, `alert`, `empty-state`, `badge`, `input`, `button`, `card`, `toggle`, `skeleton`*). *The skeleton and toggle slugs are listed on the site index but I did not open them.
**Proposed substitution (D5):** the nine slots are plain shadcn-style primitives. I recreate them in Tier-2 CSS from **shadcn/ui** (MIT, `https://ui.shadcn.com`), whose licence I can read at source, using the ten category pages above only as visual reference. If you want specific 21st.dev components, send the URLs and I will check each licence on its page.

| Slot | Recreated from (shadcn/ui, MIT) | Lands in |
|---|---|---|
| Sidebar inputs/selects/sliders/toggles | Input, Select, Slider, Switch | `ui/styles.py` (restyles the native widgets) |
| Stat tiles | Card + a stat pattern | `.gauge`, `.trust-cell`, `.kpi-tile` |
| Callout cards (insight / warning / recommendation) | Alert | `.ic/.wc/.rc`, `.defect-stamp`, `.cert-stamp`, `stAlert` |
| Empty states | Card pattern | `.empty` |
| Skeleton loaders | Skeleton | new `.skeleton` class |
| Badges | Badge | `.agent-badge`, `.iso-badge`, `.prov-kind` |
| Day/Night toggle | Switch/Toggle | the native sidebar control (selectbox kept, see 4.5) |

## 4. Island architecture

### 4.1 Mechanism: v1 `declare_component` (D2)

| | v1 (iframe) | v2 (`st.components.v2`) |
|---|---|---|
| Availability | works on every Streamlit the repo allows; already used by the landing | importable on 1.63 (checked); the release that introduced it is unknown, and the pin is `>=1.49.0` |
| Serving a built bundle | `path=` serves a directory by URL; browser-cached | `asset_dir` works only for **installed packages**; app-local components must pass JS/CSS as **raw strings** on every script run |
| Tailwind isolation | separate document: leakage is impossible | shadow DOM (`isolate_styles=True`) isolates CSS, but the JS runs with full page privileges |
| Theming | tokens passed as args (what you asked for) | can read `--st-*` vars, which are the *fixed native* theme, not our Night mode |
| Fragment behaviour | args-identical re-render should not remount; **to be proven in Spike B** | unknown |

I recommend v1: same mechanism as the landing, no version-floor bump, no app-DOM JS, bundles served as cacheable files. Cost: an iframe per island and pointer effects only within the iframe.

### 4.2 Layout of the project

```
ui/islands/                  source (Vite + React 19 + TypeScript strict + Tailwind v4)
  package.json, package-lock.json, vite.config.ts, tsconfig.json, README note
  src/shared/   tokens bridge, useReducedMotion, streamlit bridge (postMessage), cn()
  src/hero/     Hero island (Magic UI TextAnimate derived, original CSS spotlight)
  src/progress/ Progress steps (original CSS + React)
  src/findings/ Findings bento (original CSS)
static/islands/              BUILD OUTPUT, committed
  index.html                 one declare_component entry; `island` arg picks the island
  hero.js                    stable name, loaded by the landing document
  assets/…                   hashed shared and per-island chunks
  manifest.json              what exists, for the Python availability check
```
`node_modules/` is added to `.gitignore` and `.dockerignore`. Output names avoid `build/` and `dist/` (already git-ignored). `npm ci && npm run build` is documented in the README; Node is never needed to run the app. Tailwind has been dropped from Spike A as it was unneeded.

**Hero (D3).** The landing is a self-contained 119 KB HTML document in a full-screen iframe whose injected CSS forces every iframe to full-screen. A second Streamlit component there would be hijacked. Instead the landing document gets one small module script that dynamically imports `/<baseUrlPath>/app/static/islands/hero.js` (the base path is derived from `location.pathname`, since a static file cannot be templated). If that import fails the script does nothing and the existing hero stays exactly as it is. `#hero-enter-btn` and the enter/theme handshake are not touched, so `scripts/ui_smoke.py` keeps working. Theme: the island reads the landing document's own CSS variables and watches its theme class, so it follows the landing's Day/Night switch without a prop.

**Progress and findings** are mounted from Python through one `declare_component("dsa_islands", path="static/islands")` with `island="progress"` or `"findings"`, stable `key`s, and a shared chunk.

### 4.3 Typed Python helpers

`ui/components/islands.py`: frozen dataclasses for each prop set (`StepperProps`, `StepProps`, `FindingTile`, `FindingsProps`), pure builders `stepper_props(snap_stage_log) -> StepperProps` and `findings_props(...)`, and `islands_available() -> bool` (checks `static/islands/index.html` and `manifest.json`, and that every file the manifest lists exists). Props are built **only** from `STAGE_DEFS`, the snapshot and the `RunView`; never from the API key, dataset rows or raw LLM output. All new Python is fully type-hinted and gets tests.

### 4.4 Props contract and escaping (D7)

Props are plain display strings, sanitised and truncated in Python (`plainify`, length caps), and rendered **only as React text nodes**; no `dangerouslySetInnerHTML` anywhere (enforced by a lint rule and a test that greps the source). I deviate from "already-escaped": pre-escaping would double-encode (`&amp;amp;`) because React escapes at the text-node boundary. The effective guarantee (LLM/dataset text is never interpreted as HTML) is the same, and a test feeds `<img onerror>` payloads through the helper and the TS bundle contract.

### 4.5 Theming, in place

`tokens` (both palettes plus the overlay tokens) and `theme` are props. The island root maps them onto CSS custom properties; a theme change updates those variables, no re-mount. Tailwind is v4 with `preflight` omitted and all utilities scoped under the island root class; colours come from `var(--pen)` etc., never literals (a build-time check fails on hex literals in `src/`). The Day/Night control stays the native selectbox (changing it reruns the script; a v1 iframe given a new `theme` arg should update in place: **Spike B confirms**).

### 4.6 Islands and their fallbacks

| Island | Where | Fallback |
|---|---|---|
| **Hero** (lazy) | landing document | existing hero untouched if the import fails or `static/islands/` is absent |
| **Run progress** | inside the polled fragment, **beside** the existing steps list | additive: the native `render_steps_list` stays the primary readable account. If assets are missing, `islands_available()` is false and nothing extra renders; if the iframe fails at runtime its height stays 0 until it reports ready |
| **Findings** (one bundle, `variant="answers"` or `"provisional"`) | Answers tab "What we found"; "Found so far" | the existing `render_finding_card` / `build_provisional_html` HTML. Used whenever `islands_available()` is false. For a runtime load failure the island reports `ready` once via `setComponentValue`; until then the legacy cards render, then are replaced (one extra rerun on first mount, outside the polled fragment) |

Largest bento tile = the top-ranked headline finding (D8). The "You asked …" heading and written answer remain native, so the page keeps one real `<h1>/<h2>` structure and the answer stays selectable text. A finding is risk-flagged exactly when the existing `audited_checks` has a `"fail"` state (to be confirmed against `render_finding_card` in Step 5). Glow is `--risk` for those only; others use `--pen`/`--accent`. No per-word effects on answers.

A test `tests/test_islands_fallback.py` monkeypatches the assets directory to empty and asserts every helper returns the legacy HTML path; a second test asserts a corrupt manifest does the same.

## 5. Tier-2 CSS and the named recreations

- **Pointer glow script, once.** One idempotent script, installed outside the fragment, sets `--mx/--my` on hovered cards through one delegated, `requestAnimationFrame`-throttled `pointermove`. Guarded by a `window.__dsaPointer` flag; no per-element listeners; disabled under reduced motion. It **replaces** the current `animations.py` observer (see R5).
- **Run button.** A `st.container(key="run-cta-<state>")` wrapper gives CSS a state hook (`idle / running / done`) derived from the existing `_workspace_state`. The button keeps `key="btn_run_analysis"` and the same `can_run` guard. The label may change per state; behaviour does not.
- **Hover Border Gradient look**: pure CSS with a `@property` angle, hover/focus-visible only, static ring under reduced motion.
- **Landing CTA**: lives in the landing document's own CSS; same look applied there.
- **Dropzone**: native `stFileUploader`; dashed `--rule-strong` border, hover lift and glow, an explicit drag-over style. The uploader, its checks and `validate_upload` are not touched.
- **Charts**: Vega config themed from tokens (`get_vega_config` already exists); no chart-selection logic changes; no glow on charts or tables.

## 6. Making every animation fragment-poll-safe

| Effect | Trigger | Why it will not restart on a 1 s tick |
|---|---|---|
| Stepper check/slide | `status` transition for a stage id | React compares previous and next `status` per stage id; identical props do nothing |
| Glowing Effect on active step | `current` changes | CSS-driven; only the active step mounts it |
| Findings entrance | a `finding_id` not seen before | `seen` set lives in the island and in `sessionStorage` per run, so a remount does not replay |
| Pointer glow | real pointer movement | not tied to renders |
| Hero text | first visit in the session | `sessionStorage` flag |
| `<style>`/`<script>` | injected once | existing `inject_theme_css()` is emitted once per script pass, outside the fragment; the fragment emits no `<style>` |
| Layout shift | n/a | island iframe height is reserved (`min-height` from the stage count) before mount |

Memoisation: stepper props are rebuilt each tick, but Python skips re-sending when the serialised props are unchanged (cached last-props in the fragment), so the component receives no new args. **Spike B measures** re-mounts, flicker, restarts and shift; if it cannot be made stable I fall back to Tier-2 CSS for live views and tell you.

## 7. Dependencies, licences, sizes

Only dev-time npm deps; only built output ships. Licences below are **from memory** for the well-known ones and will be captured mechanically in Step 0 with `license-checker`, then written to `THIRD_PARTY_UI.md` (source, licence, version for every npm package bundled and every borrowed component).

| Package | Use | Licence (to verify) |
|---|---|---|
| react, react-dom (19) | island runtime | MIT |
| motion | animations (`LazyMotion`/mini `animate` to keep it small) | MIT |
| vite, typescript, @vitejs/plugin-react | build only, not shipped | MIT |

Not used: TailwindCSS, `@tabler/icons-react`, `lucide-react`, `react-dropzone`, `@radix-ui/react-tabs`, `canvas-reveal-effect`/three for islands.

**Bundle budgets** (gzipped; these are targets I will measure in the spikes, not measurements): shared chunk ≤ 70 KB; hero ≤ 12 KB; progress ≤ 10 KB; findings ≤ 18 KB; per-island CSS ≤ 6 KB; **total islands ≤ 110 KB gzipped**. For scale, the repo already ships a 1.33 MB `three.module.js`. If React alone breaks the budget the fallback is a Preact alias; I will not do that without telling you. Landing island lazy-loaded after first paint; progress and findings load only when first needed.

**Pre-existing third-party items to resolve (D4)**
- `ui/animations.py`, `pipeline_3d.html`, `cinematic_3d.html` load from CDNs (audit section 7). Plan: vendor anime.js 3.x, anime.js 4.x (pipeline uses a different major), GSAP and the fonts under `static/vendor/` with licence files.
- **fullPage.js (cinematic export):** *(unverified, from memory)* v4 is GPLv3 or a paid commercial licence with a licence key. If that is right, it conflicts with "free only" and with distributing it in a downloadable HTML file. **GSAP:** *(unverified)* its licence is a custom "no charge" licence, not MIT. I will not vendor either until you decide: keep as is (CDN, so a `LOCAL_ONLY` violation), vendor, or replace. Replacing the cinematic export's scroll engine is large and not in this plan unless you ask.

## 8. Rollout order

0. **Prerequisites:** D6 (local browser), record baseline (done in the audit), `license-checker`.
1. **Spike A (static island):** scaffold `ui/islands/`, build the hero island, load it in the landing document with the failure fallback. Verify with Node absent (committed build), Day and Night, reduced motion, and `scripts/ui_smoke.py`. **STOP for review.**
2. **Spike B (live island):** progress island inside the polled fragment, real run, report remounts / flicker / restarts / shift / page weight. **STOP for review.**
3. Step 1: tokens + overlay + `DESIGN.md` (+ Step 1b vendoring if D4).
4. Step 2: shell (landing, header/hero, sidebar, section heads, spacing).
5. Step 3: Tier-2 primitives.
6. Step 4: run experience.
7. Step 5: tabs, with the findings island.
8. Step 6: micro-interactions (replace the `animations.py` observer).
9. Step 7: responsive pass.
10. Step 8: `report.html` restyle only if it stays self-contained and content-identical; else skipped and reported.
After each step: `ruff check .`, the `mypy`/`pytest` baseline comparison, `scripts/validate.py`; one commit per step on `frontend-overhaul`; no force-push; `master` untouched. `git diff 96c4598..HEAD -- src/` must be empty and `ui/run.py` must gain no `st.*` calls (checked by a test).

**CI.** Existing jobs install only Python and keep working because `static/islands/` is committed. I propose one extra, **non-blocking** job that runs `npm ci && npm run build` and fails if the committed build differs from source; it starts as `continue-on-error` until proven reproducible. The `Dockerfile` handles the backend worker only and correctly ignores `static/islands/`, and `.dockerignore` excludes `ui/islands/node_modules/`.

## 9. Rollback

Each step is one commit, so `git revert <commit>` undoes it. The islands are additive: deleting `static/islands/` (or `islands_available()` returning false) returns every screen to the current components. `styles.py` overlay tokens are additive; reverting Step 1 restores today's CSS. Nothing touches `src/` or the snapshot, so there is no data or backend rollback.

## 10. Limits of this plan

- No browser has been driven yet (D6), so nothing about live behaviour is verified.
- Whether a v1 iframe survives the 1 s fragment tick without remount, and whether a changed `theme` arg updates it in place, is the central assumption; Spike B tests it.
- Bundle sizes and several licences are targets or recollection until measured.
- I did not read the 3D JS, the landing HTML internals, the Charts/Details/Downloads tabs or the second half of `ui/run.py`; Steps 2 to 5 will read them before editing.
