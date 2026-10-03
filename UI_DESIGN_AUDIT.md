# UI design audit (2026-10-03)

Audit only. No code was changed and the app was not run. Everything below comes from reading the
working tree on `frontend-overhaul` plus the committed reference screenshots.

**What this audit is judged against.** Your brief: an agentic data-analysis product for technical and
non-technical people, judged equally by examiners and everyday users, "no AI slop, a purpose behind
everything", and 3D for retention and a wow factor. Each finding says what, where, why, a direction,
and whether it needs a decision from you.

**Limits of the evidence (read these first)**

- The app was not run in initial audit pass. Anything marked *(verify)* was inferred from CSS/Python and needed a browser check.
- **Live run evidence update (2026-10-03, Antigravity)**: Live browser screenshots of `AirQualityUCI.csv` in the Answers tab were analyzed. Verified finding **A3** (confirmed visually: `.finding-card.full-width` failed to span `grid-column: 1 / -1`, leaving an orphaned 3rd slot in row 2). Identified and added new live-run findings **A9 through A14**.
- Night mode and mid-run states are **unverified**. The committed screenshots in
  `design-references/screenshots/step3/` are mislabeled (`01_landing_*` is blank, `02_empty_*` shows
  the landing page, `03_uploaded_night_*` shows Day mode). I used them only for the sidebar and the
  empty workspace, and I did not examine the mobile shots.
- I read in full: `app.py`, `ui/styles.py`, `ui/animations.py`, `ui/landing.py`, `ui/tabs/answers_tab.py`,
  `ui/tabs/charts_tab.py`, `ui/components/provisional.py`, and most of `ui/components/cards.py`.
  I read only parts of `ui/landing_component/index.html` (hero and stage panels), `downloads_tab.py`
  and `ui/pipeline_3d.js`. I did **not** read `details_tab.py`, `how_we_got_here.py`, `cinematic_3d.*`,
  or the middle/bottom of the landing page.
- `FRONTEND_AUDIT.md` and `OVERHAUL_PLAN.md` are partly stale (for example `animations.py` is now only a
  pointer-glow script). I re-verified what I took from them.

Severity: **P0** undermines the product's purpose or the first minute. **P1** clearly hurts. **P2** polish
or consistency. Tags: `[demo]` matters to examiners, `[user]` matters to everyday users.

---

## 1. The three questions that frame everything

**Does each thing have a purpose?** Most of the content layer does (plain stage names, answers-first order,
"may change" honesty). The visual layer mostly does not: hover glows on static text, spinning borders on
stat tiles, a left-border stripe on every callout, 8 agent cards that repeat the 7 stages under different
names. The 3D is the biggest purpose gap (section 4).

**Is it "hardcoded"?** Yes, in the specific sense of values that bypass the system: about 19 distinct
font sizes, half-pixel steps, 74 `!important`, 12 inline-style lines in `cards.py`, hand-synced token copies
in the landing page, two undefined CSS variables. See section 3.

**Is it fighting its platform?** Yes. See the Streamlit section in the chat reply, and section 3.

---

## 2. Findings by user journey

### Landing

- **L1 [demo] P1 — Three parallel navigators for four stages, but the product has seven.**
  Nav pills "01 Raw … 04 Checks", floating dots (`index.html:1385-1388`) and a telemetry bar all
  navigate the same 4 groups. The workspace shows 7 stages (`cards.py:27`) and the Details tab shows
  8 "agents" (`cards.py:230-295`). Three mental models of one pipeline.
  *Direction:* one model of the pipeline everywhere. Decision needed: which one (7 stages is what the
  engine actually runs).
- **L2 [demo] P2 — Template tells, repeated.** One accented word in the headline and in every stage title
  (`index.html:1421, 1463, 1487, 1511, 1535`, CSS `:764`); a pill eyebrow above every heading that repeats
  the heading ("Stage 1: reading your file" over "Your table, read and checked"); "→"/"↓" on CTAs
  (`:1427-1428`); a green tick on all three trust lines. Each is a default, not a choice.
  *Direction:* drop the accent word and the duplicate eyebrows; let the heading do the job.
- **L3 [user] P1 — Pinned scroll-driven 3D track.** *(verify)* Scroll-hijacking is expensive for people who
  just want to start. I could not confirm whether "Open the workspace" is reachable from the pinned
  section (I did not read it). *Direction:* the CTA must stay visible throughout.
- **L4 P2 — Landing is a 121 KB single HTML file with its own hand-synced token copy** (`landing.py:14-28`
  admits this). Drift risk; any design change is made twice.
- **Keep:** the headline claim is specific and verifiable ("We check every answer twice"), and the page says
  honestly that the 3D "is an illustration of this step" (`index.html:1490`).

### Entering the workspace (empty state)

- **W1 [user][demo] P0 — The workspace opens with the marketing headline again.** `app.py:883` repeats
  "We check every answer twice." right after the user pressed "Open the workspace". Below it come the 3D
  plate and the Quick Facts bar, so the upload card starts roughly 540 px down on a 900 px screen
  (see `03_uploaded_night_desktop.png`). The one task on this page is the last thing you reach.
  *Direction:* task-first. The upload and the question are the hero. The headline belongs on the landing.
- **W2 [demo][user] P0 — "Try it with sample data" is at the very bottom** (`app.py:1234`), under an
  8-card grid. It is the fastest route to a result for a new user and the safest live-demo path for an
  examiner. *Direction:* put it next to the upload as an equal first choice ("Use my file" / "Try a
  sample").
- **W3 [user] P1 — Two uploaders on screen, one of them rare.** "Related tables (optional)" gets the same
  weight as the main upload (`app.py:535-541`), and its helper text lists ~25 extensions
  ("DTA, FEATHER, GZ, H5, HDF5, … XPT, ZSAV"). Non-technical users do not know any of those.
  *Direction:* say "Spreadsheets (CSV, Excel)" and put the rest, plus related tables, behind one
  "More formats / add related tables" disclosure.
- **W4 [user] P1 — "Target column" is a free-text box** (`app.py:556`). Nobody types a column name from
  memory. After upload it should be a dropdown of the file's own columns, defaulting to "Let the assistant
  decide".
- **W5 [user] P2 — Disabled Run button reads as a placeholder.** Dashed outline, reason in a caption
  elsewhere (`styles.py:264-267`, `app.py:850-855`). Say why on the button area itself.
- **W6 [user] P2 — Quick Facts bar shows system state.** "State: Ready · File: None loaded · Model: gpt-4o"
  (`app.py:1117-1121`). The model name means nothing to most users and the bar sits between the hero and
  the input, pushing it down. *Direction:* show what the user needs (file name, rows × columns, what
  happens next), or drop it before a file exists.
- **W7 [demo] P1 — "3D Cinematic Showcase" button on the first screen** (`app.py:889-893`), before any data
  exists. It competes with the primary action and shows nothing about the user's data yet.
- **W8 [user] P2 — The empty-state copy and the "Meet Your Helpers" grid repeat the hero**
  (`app.py:1217-1228`). Three blocks say "we study, test and double-check your data".
- **Keep:** the numbered steps 1-2-3 (a real sequence), the sample run being a real analysis labelled
  "It is not your data" (`app.py:1161-1166`), no API key needed for the no-AI path.

### Sidebar and configuration

- **S1 [user] P0 — The sidebar is an engineer's settings panel.** Roughly 25 controls (`app.py:458-830`).
  Theme is the first control (the most valuable spot goes to a preference). Then provider, model with
  labels like "gpt-4o (128k ctx · 10000 RPM · 3…" (truncated in the screenshot), pricing tier, reasoning
  mode, "Thorough tuning", a GPU caption with a code literal `device='cuda'` (`app.py:744`), max tree depth,
  test split, CV folds, minimum group size, "Allow AI-written code", "Max code runs per analysis",
  "Require container isolation", RLM toggle. This is the progressive-disclosure failure the research warns
  about (see sources).
  *Direction (you chose "users bring their own key"):* one clear block, "Connect an AI (only for the
  written summary)", with the key field, a plain "where do I get a key" help, and a connection status.
  Everything else goes under an "Advanced" disclosure with safe defaults. Hosted mode already hides some of
  these; the default view should look like hosted mode.
- **S2 [user] P0 — The best path is hidden.** "AI narrative" defaults on, which disables Run until a key
  is pasted (`app.py:836, 853`), and the caption says "in the settings sidebar", which is collapsed on a
  phone. The deterministic no-AI analysis, which works instantly and is private, hides behind a toggle.
  *Direction:* make "With an AI summary / Without" a visible choice at step 3.
- **S2b [demo] P1 — An environment API key does not enable Run.** `has_key` (`app.py:835`) tests only the
  text box (`api_key.strip()`), while the run itself uses `_effective_key`, which falls back to the
  `<PROVIDER>_API_KEY` environment variable (`app.py:608-609`). A local demo that relies on an env key
  shows Run disabled with "Add your API key…". Verified from the code; not run.
- **S3 [user] P2 — Theme is a two-item dropdown** (`app.py:460`) for a binary choice, and each change
  reruns the whole script and resends ~35 KB of CSS. A switch in the header is the expected control. The
  landing already has its own toggle, so the choice exists twice.
- **S4 P2 — Join review cards use a 3 px left border in `--accent` as the only cue**
  (`app.py:410`). Accent is 2.3:1 on the page in Day (`FRONTEND_AUDIT.md` section 3), below the 3:1 for
  non-text cues (WCAG 1.4.11).

### Running

- **R1 [demo][user] P0 — The 3D plate does not show progress, though progress is its stated job.** The
  plate sits outside the polling fragment (`app.py:895-905, 1079-1096`). It is drawn on full script passes,
  so between those it shows the state from the last pass, while the numbered list beside it (inside the
  fragment, `app.py:270-302`) is what actually moves. After the run it is tucked into a "Show how it's
  working" expander (`app.py:1090`). A 3D scene that duplicates a list but updates less often does not earn
  its place. See section 4.
- **R2 [user] P1 — A failed run does not say why.** `analysis_error` is written (`app.py:254, 1030`) and
  read once, to choose the hero wording (`app.py:439`). A grep of `app.py` and `ui/` finds no place that
  renders it, so the user sees "Could not finish {file}" (`app.py:870`) and nothing else. (The preflight
  failure is the exception: it shows its message directly, `app.py:1031-1039`.)
- **R3 [user] P2 — Preflight errors dump the raw provider message into a code block** (`app.py:1035`).
  The hint text below it is good. Translate the common cases (bad key, no credit, unknown model) into one
  plain sentence and keep the raw text under a disclosure.
- **R4 P2 — "Usually 1–3 minutes" is static** (`app.py:290`). No elapsed time, no per-step progress.
- **R5 P2 — `.sc.active .nm::after { content: " — working" }`** (`styles.py:776`): a spaced em-dash label
  that repeats what the active styling already says.
- **Keep:** "Found so far, may change" and its refusal to show check marks on unaudited findings
  (`provisional.py` docstring) is exactly the right honesty. The `aria-live` stage list is good.

### Results

- **A1 [demo][user] P0 — The verdict is under-designed.** "N of M findings held up" is the product's
  differentiator, and it renders as `.run-banner` (`answers_tab.py:104`), the same small pen-coloured strip
  used for "Running the analysis" (`app.py:292`). It should be the dominant element of the results page.
  (Only shown when audits actually ran: keep that rule.)
- **A2 [user] P1 — A finding and its evidence are in different tabs.** The card says "→ see chart in the
  Charts tab" as unlinked text with inline styles (`cards.py:148`). The product's job is interpretation;
  the number, the sentence and the chart should be together (mini chart or an expand in the card).
- **A3 [demo] P1 — The "primary finding spans the full width" rule is dead.** *(Confirmed visually from live AirQualityUCI run)*:
  `render_finding_card` adds `full-width` (`cards.py:163`) but the CSS only defines `.bento-card.full-width` (`styles.py:659`). The primary finding card sits in column 1 of a 3-column row, leaving row 2 with 2 cards and an orphaned blank 3rd slot.
  *Direction:* Add `.finding-card.full-width { grid-column: 1 / -1; }` in `styles.py` so the lead card spans across the top and the remaining 4 cards form a balanced 2×2 grid.
- **A4 [demo] P1 — The 4 px left stripe is the universal callout.** Finding card (`styles.py:465`), Executive
  directive (`:717`), insight/recommendation/risk notes (`:793-806`), Streamlit alerts (`:745-758`), the
  landing (`index.html:850`). It is one of the most-cited AI-UI tells, and here the same pen colour means
  "finding", "directive" and "recommendation", so colour carries no information.
  *Direction:* give each kind a distinct form (a verdict block, a numbered recommendation list, a risk
  block) and reserve colour for meaning.
- **A5 [user] P1 — Hover effects on things you cannot click.** A pointer-following spotlight on static
  finding cards (`styles.py:631-657`, `animations.py`), and a spinning conic-gradient border on static stat
  tiles (`styles.py:396-404`). Interaction affordance without an interaction. Keep the glow only where it
  encodes meaning (a flagged finding), or remove it.
- **A6 [user] P2 — Jargon in the technical block:** "CV Score", "Train-Test Gap", "Best Model"
  (`answers_tab.py:259-271`). The defect/cert stamp text next to it (`cards.py:192-217`) is the model to
  copy: it explains the number in a sentence.
- **A7 [user] P2 — `st.warning` is mapped to `--risk` red** (`styles.py:755-758`), so "AI was unavailable,
  used the built-in plan" looks like "this number may not hold". `DESIGN.md` reserves `--risk` for the
  second meaning only.
- **A8 P2 — "Search these findings" is a keyword filter** (honestly labelled in code). Users of an "agentic"
  product will expect to ask follow-up questions. Product decision, not a styling one: either set the
  expectation in the UI or build the feature.
- **A9 [demo][user] P0 — Verdict count contradicts card count: "4 of 4 findings held up" directly above 5 cards.**
  *(Identified from live AirQualityUCI run)*: The verdict banner says `4 of 4 findings held up.`, but exactly 5 finding cards are displayed under "What we found".
  *Where:* `run_view.py:206-217` (`compute_verdict`), `answers_tab.py:105-108`.
  *Why:* `compute_verdict` filters `audited = [marks for marks in ... if marks]`. Card 5 (guideline threshold exceedance) had no audit checks attached, so `audited` was 4, not 5. A user seeing 5 cards under a "4 of 4" banner assumes 1 card failed, was skipped, or that the counter is buggy.
  *Direction:* Distinguish audited statistical discoveries from exploratory observations (e.g., `4 of 4 audited findings held up (1 domain observation)`), or include all headline findings in the denominator.
- **A10 [user] P1 — Raw 4-decimal floats leak into card body copy.**
  *(Identified from live AirQualityUCI run)*: Headlines use clean, human-rounded numbers (`19:00 (1,201) and is lowest at 04:00 (631)`, `(17.7)`, `(2.92)`), but card body copy dumps raw unrounded floats: `1201.1968`, `630.8992`, `939.1534`, `0.3807`, `978.4195`, `844.9444`, `17.7356`, `2.9167`, `10.0831`, `0.3322`, `11.2497`, `7.2841`.
  *Where:* `cards.py:134-144`, `src/core/plain_language.py`, finding generator formatting.
  *Why:* Formatters in the engine output raw string representations of floats into `finding["detail"]` without passing them through rounding filters.
  *Direction:* Pass finding detail strings through a number formatter (max 1–2 decimals for averages, 3 decimals for p-values/correlations, integers for counts).
- **A11 [user] P1 — Fluctuating sample sizes across adjacent cards without explanation.**
  *(Identified from live AirQualityUCI run)*: Card 1 displays `✓ 9,326 records`, while Cards 2, 3, 4, 5 display `8,991 records` / `8,991 readings` / `6384 of 8991 readings`.
  *Where:* `findings.py:attach_finding_checks`, `audited_entry.py`.
  *Why:* Pairwise correlation and sensor-error cleaning (`-200` sentinel values) dropped incomplete rows for specific sensor columns, but the user is not told why the dataset record count fluctuates between 9,326 and 8,991.
  *Direction:* Add a brief note or badge tooltip explaining pairwise complete cases vs full table rows.
- **A12 [user] P2 — Robotic title repetition & punctuation inconsistencies in card copy.**
  *(Identified from live AirQualityUCI run)*: Card 2 title: `PT08.S1(CO) and PT08.S5(O3) move together (a very strong positive relationship)`. Card 2 body: `Pearson correlation between 'PT08.S1(CO)' and 'PT08.S5(O3)' is a very strong positive relationship...`. In Card 5, `'C6H6(GT)'` is wrapped in single quotes in the title, while in Cards 1–4, column names are unquoted.
  *Where:* `findings.py`, `plainify()`.
  *Why:* Templates concatenate raw column strings and fixed severity phrases without deduplicating headline statements.
  *Direction:* Strip redundant title lead-ins from body text; normalize column identifier formatting.
- **A13 [demo] P1 — The "N of M held up" banner uses a warning/danger color palette.**
  *(Identified from live AirQualityUCI run)*: In Day mode, `.run-banner` is styled with `color: var(--pen)` (`#a34f20` terracotta) and `background: color-mix(in srgb, var(--pen) 10%, var(--sheet))` (`styles.py:812-815`).
  *Why:* A 100% clean verification pass (`4 of 4 findings held up`) is rendered in a rust/reddish tint that looks like a warning or error banner instead of a success state (`var(--positive)`).
  *Direction:* When `verdict.needs_more == 0`, style `.run-banner` with `var(--positive)` (clean green tint); reserve `var(--pen)` / `var(--risk)` only when findings failed audits or need more data.
- **A14 [user] P2 — Unlabelled Executive Briefing card.**
  *(Identified from live AirQualityUCI run)*: At the very top of Answers, the executive briefing text sits in `.exec-directive` with a left border stripe, but has **no heading** when `view.objective` is empty (`answers_tab.py:83-87`). The section below it has `## What we found`, making the top card look like an unheaded orphan block.
  *Direction:* Render a standard section title (e.g. `Executive Summary` or `Overview`) when `view.objective` is empty.
- **Keep:** the answers-first order (question → verdict → evidence → caveats → action), the plain-language
  headings ("What we found", "Be careful about", "What to do"), and the defect/cert stamp copy.

### Downloads

- **D1 P2 — Emoji on buttons** (`downloads_tab.py:50, 64, 81`) next to a custom SVG icon set
  (`icons.py`), plus ⚠ / ✓ glyphs elsewhere (`cards.py:199, 210`, `answers_tab.py:167`). Pick one icon language.
- **D2 P2 — A `<style>` block is injected on every render for a `.dossier-card` class nothing uses**
  (`downloads_tab.py:28-34`). Leftover from the earlier identity.
- **D3 P2 — Old vocabulary:** "Artifact Vault", "Agent Memory Vault", "Presentations". Say what the file is:
  "Report (web page)", "Report (Markdown)", "All findings as data".
- **D4 P2 — The cinematic export is built on every render of the tab** (`downloads_tab.py:44-48`), with
  inlined assets. Build it on click.

---

## 3. System-level findings

- **Y1 P1 — No type scale.** About 19 distinct pixel sizes (11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5,
  17, 18, 19, 21, 24, 26, 27, 32, 38) plus rem and clamp values. Half-pixel steps are noise, not hierarchy.
  *Direction:* a 6–7 step scale as tokens (that is what "no hardcoded nonsense" looks like in CSS).
- **Y2 P1 — Hard-coded values bypass the tokens.** 12 inline-style lines in `cards.py` (the agent cards carry
  most), and a literal `rgba(0,0,0,0.1)` drop-shadow in `answers_tab.py:248`.
- **Y3 P1 — Two CSS variables are used but never defined for the app page.** `--ease-in-out` (used at
  `styles.py:311`, added in `b98ddb5`) and `--mono` (used at `styles.py:142`). Repo-wide, both are defined
  only inside the two 3D iframe documents (`pipeline_3d.html`, `cinematic_3d.html`), and neither
  `design_tokens.py` nor `styles.py` emits them. A `var()` that resolves to nothing invalidates the
  declaration, so the tab-pill slide transition likely does not animate and code blocks fall back to the
  inherited font instead of `codeFont` from `config.toml`. *(verify visually)*
- **Y4 P1 — Fighting Streamlit.** 74 `!important`; 31 distinct Streamlit `data-testid` hooks; `:has()` selectors
  for layout; a global `iframe { position: fixed; z-index: 999999 }` on the landing (`landing.py:63-71`);
  a hidden Streamlit button as a handshake (`app.py:55-69`); JS injected through a zero-height iframe that
  reaches `window.parent` (`animations.py:54`). Each works; together they are one Streamlit upgrade away from
  breaking.
- **Y5 P2 — Global `iframe` styling hits the zero-height helper iframe** (`styles.py:318-323`,
  `animations.py:54`): probably a 2 px bordered, shadowed sliver at the bottom of the page. *(verify)*
- **Y6 P2 — ALL-CAPS labels in five rules** (`styles.py:479, 490, 510, 518, 525`) while the landing page
  comments that the design system bans them (`index.html:289`). `DESIGN.md` as it exists today does not say
  that. Decide the rule, then make both agree.
- **Y7 P2 — One radius for everything.** Cards, tabs, expanders, alerts and iframes all use the same 14 px
  (plus pill buttons). No hierarchy of shape.
- **Y8 P2 — Dead CSS from the "Tier-2 primitives" step.** `.skeleton*` and `.empty-state*` have no users in
  Python. Skeleton loaders would be useful in the run view; either use them or delete them.
- **Y9 P2 — Two vocabularies in the code.** Class and doc names still use the retired identity ("plate",
  "datum", "dossier"), user-facing text uses "Meet Your Helpers", "The Team at Work", "Vault".
- **Y10 P3 — Reduced motion is a blunt hammer.** `* { animation: none !important; transition: none !important }`
  (`styles.py:83-86`) is a sound intent, but it also removes harmless state fades. Keep opacity and colour
  fades, remove movement and loops.
- **Y11 P2 — Mobile hides the 3D entirely** (`styles.py:223-227`). The wow factor does not exist on a
  phone, and the key-entry hint points to a collapsed sidebar.
- **Y12 P2 — Contrast debts already recorded** in `FRONTEND_AUDIT.md` section 3 (Day `--accent` 2.3:1, Night
  `--risk` text 4.04:1 on its tint, `--rule` 1.65:1 as a control border) are partly handled by overlay tokens;
  confirm they are used everywhere.

---

## 3b. Identity verdict: where "Ledger" lands on the common-default list

You opened the identity ("3rd option": a new UI behind a switch, current UI kept for development). The
frontend-design guidance lists five clusters that AI-generated design converges on. Judged against them:

| Default cluster | Ledger today | Evidence |
| :--- | :--- | :--- |
| 1. Warm paper background with a terracotta accent and a rounded display face | **Hits it.** Buff-green paper is a variant of the warm-paper default; `--pen` terracotta `#a34f20` is the lead accent; Baloo 2 is a rounded display face. | `design_tokens.py`, `.streamlit/config.toml`, landing CTA and accent word |
| 2. Near-black with one acid accent | Not hit in Day. Night is untested here. | not verified |
| 3. Broadsheet layout, hairline rules, zero radius | Not hit (the opposite: 14 px radius everywhere). | `styles.py` `--radius` |
| 4. SaaS card kit: one radius, one soft shadow, gradient washes | **Hits it.** One radius, `--lift` shadow on cards, expanders, alerts; radial and conic gradients as hover decoration. | `styles.py:318-323, 396-404, 631-657` |
| 5. Template chrome: eyebrow labels, middle-dot meta strings, arrows on links, mono data labels | **Hits it.** Pill eyebrows (landing), "→/↓" CTAs, and `·` strings: `cards.py:346` "Role: … · Tool: …", `app.py:406` "… · many-to-one · 90% matched", model labels "128k ctx · 10000 RPM", landing telemetry divider `index.html:1396`, uploader "200MB per file •". | as listed |

Ledger was a deliberate answer for shopkeepers and teachers ("warm and universal"), and the tokens are
well-built (tested contrast, five synchronised consumers). But as a *signature* it matches three of the five
clusters, so it will not read as distinct to an examiner who has seen a lot of AI-built projects.

**What any new identity must keep** (so v2 can be swapped in safely):

- `--risk` means only "this number may not hold".
- WCAG AA for text and 3:1 for non-text cues, checked per token pair (the current gaps are in section 3, Y12).
- The token pipeline: `src/core/design_tokens.py` is read by five consumers (`html_report.py`, the 3D scenes,
  the landing mirror, `chart_theme.py`, `.streamlit/config.toml` via `scripts/sync_streamlit_theme.py`), and
  `tests/test_theme_sync.py` guards them. A v2 palette that lives only in v2 avoids touching `src/`
  (AGENTS.md: "Ask First").
- Plain language, answers first, reduced motion, keyboard access, the 3D views' text equivalents.

**Next deliverable, not done here:** a design plan for the v2 identity, produced with the two-pass process
(token plan: 4–6 named colours, type roles, layout concept with wireframes; then a review of that plan
against the default list above before any code). It needs your answers on the platform question first.

---

## 4. The 3D, judged on purpose

Rule applied: 3D earns its place only when it helps the user answer "what can I do, what just happened,
what state am I in?", and there is an equivalent non-3D path (see the WebGL interaction-budget source).

| Scene | Claimed job | What it does today | Verdict |
| :--- | :--- | :--- | :--- |
| Landing pixel grid + pinned track | Brand and explain the pipeline | Does that, and says it is an illustration | **Keep one.** The strongest "first impression" candidate. |
| Workspace pipeline plate | Show run progress | Updates less often than the list beside it; hidden after the run (R1) | **Does not earn its place as built.** |
| Cinematic Showcase (~3,500 lines of HTML/JS) | Wow / shareable | A third illustration of the same pipeline, toggled from the first screen, also exported as HTML | **Redundant.** Candidate to cut, or to keep only as the export. |

**Candidate direction (needs your call, not a recommendation to build yet):** the wow that also has meaning
is 3D of the user's **own data**: rows as points, columns as axes, flagged outliers marked in `--risk`, the
top finding as an annotation. That connects the effect to the answer and gives examiners something no
generic dashboard has. It is also the one place where a 3D view carries information a table cannot.

The existing accessibility work is good and should carry over: canvas `aria-label`, arrow-key control
(`pipeline_3d.js:875-880`), `prefers-reduced-motion`, no-WebGL text.

---

## 5. Keep list (do not tear out)

Plain-language stage names · answers-first result order · the verdict shown only when audits ran · the
"may change" provisional block · the real sample run, labelled as not the user's data · `--risk` meaning
"this number may not hold" · defect/cert stamp wording · honest "illustration" labelling of the 3D ·
local-only fonts (no CDN) · `html.escape` discipline · `aria-live` on the stage list · 44 px touch targets
on small screens · numbered 1-2-3 input steps (a true sequence).

---

## 6. Suggested order of work (if the platform stays as is)

1. W1, W2, S1, S2: task-first workspace, sample next to upload, one "Connect an AI" block + Advanced.
2. A1, A2, A3, A4, A9–A14: the verdict as the hero of results (resolve "4 of 4" vs 5 cards contradiction, positive green tint for clean passes), .full-width grid fix, float rounding filter, evidence with the finding, distinct callout forms.
3. R1 and the 3D decision (section 4).
4. Y1–Y3: type scale, undefined variables, inline styles into classes.
5. Copy and icon pass (D1–D3, Y6, Y9).

Most of the P0s are layout, hierarchy and copy, not platform limits. A platform change fixes Y4 and R1's
update problem; it does not fix W1, W2, S1, S2, A1.

## 7. Sources

- Progressive disclosure for transparency: https://arxiv.org/pdf/1811.02164
- Communicating uncertainty to non-experts: https://ir.cwi.nl/pub/23433/23433B.pdf
- WebGL interaction budget, equivalence and fallback rules: https://hackernoon.com/the-interaction-budget-keeping-webgl-interfaces-usable-on-real-devices
- AI-design tells (side stripes, nested cards, glow, accent word): https://www.impeccable.style/slop
- Trust calibration and transparency in AI interfaces: https://arxiv.org/pdf/2510.15769
- Streamlit custom components v2 (theming, limits): https://docs.streamlit.io/develop/concepts/custom-components/components-v2
- WCAG 1.4.11 Non-text Contrast and 2.3.3 Animation from Interactions (not fetched; cited from knowledge)

---

## 8. Corrections found while planning the revamp (2026-10-03)

Checked against the working tree before the revamp plan; they supersede the matching items above.

- **Y3 is resolved.** `--ease-in-out` and `--mono` are defined in `ui/styles.py` now.
- **`!important` count:** about 105 in `ui/styles.py` (not 74), plus about 70 `data-testid` and 7 `data-baseweb` selectors.
- **`transition: all`** appeared in exactly two rules (dropzone, text inputs); both now name their properties.
- **Fonts had four sources:** CSS `--heading`, `.streamlit/config.toml` `headingFont`, the cinematic export and the landing page did not agree (Baloo 2 vs Mukta). The workspace now uses Bricolage Grotesque, Public Sans and IBM Plex Mono through both CSS and `config.toml`; the landing, cinematic and HTML report still use Baloo 2 and Mukta until Phase 2.
- **Screenshot labels (the mislabelled step3 shots):** `scripts/capture_screenshots.py` compared `"night"` with `"dark"`, so Night shots were Day. It now switches the app's own theme control and writes a `manifest.json` per shot.
- **Streamlit:** the installed version is 1.57; the dependency floor was raised from 1.49 to 1.57.
- **`st.context.theme` is read-only** and only reports light or dark inferred from the background; nothing in Python can set the native theme, so a native dual theme would mean dropping the in-app Day/Night toggle.
- **GSAP** is free for commercial use under Webflow's Standard License (not open source). See `THIRD_PARTY_UI.md`.
