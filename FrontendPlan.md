# Frontend Plan (revised): production polish and a signature, within Ledger

Revised 2026-09-25. Replaces the 2026-09-24 draft. Scope: the Streamlit console (`app.py`, `ui/`), the shareable HTML report (`src/core/html_report.py`), charts (`src/core/chart_theme.py`), and the landing page (`ui/landing_component/`, `frontend-landing/`).

`DESIGN.md` ("Ledger") stays the source of truth. Where this plan proposes changing a Ledger rule, it says so explicitly and lists it under **Decisions for you**.

Every finding below was verified against the code (file:line) or computed (chart palette validator output). Nothing here is from memory of the codebase alone.

---

## 1. How this plan was built

Nine frontend skills were applied. They disagree with each other in places, and several are written for marketing pages rather than an analysis console for non-experts. Each skill was used where its brief matches and overruled where it conflicts with Ledger or the audience.

| Skill | Applied to | Overruled where |
|---|---|---|
| redesign-existing-projects | Audit method and fix priority for the console | Its "swap to Geist/Satoshi" font advice: Baloo 2 + Mukta are already distinctive and chosen for Devanagari support |
| frontend-design | Brief grounding, "spend boldness in one place", copy rules, the AI-default calibration list | Nothing; this skill drives the central finding in section 3 |
| design-taste-frontend | Landing page (its stated scope), CTA and copy rules, em-dash and emoji bans, empty/loading/error states | Its dashboard content: the skill itself says it is not for dashboards/product UI |
| high-end-visual-design | Custom easing curves, GPU-only animation | Double-bezel cards, glass, pill "island" nav: all contradict Ledger's flat, friendly paper |
| minimalist-ui | Card restraint, whitespace, low-opacity shadows | Its serif headings and pastel tag palette |
| dataviz | Chart palette (validated), mark specs, accessibility of charts | Nothing; its checks are computable and were run |
| stitch-design-taste | Structure for tightening `DESIGN.md` (atmosphere, dials, banned list) | "Perpetual micro-interactions on every component": contradicts Ledger's one-orchestrated-moment motion rule |
| gpt-taste | Landing page only (one pinned section, hover physics) | Everything in the console: scroll-hijack and GSAP everywhere would hurt a working tool |
| industrial-brutalist-ui | Nothing | Rejected outright: it is the retired "Drafting Table" aesthetic (mono, square corners, ruled grid), which `DESIGN.md` "Don't" forbids |

Skipped: the three image-generation skills (mockups, not code), `brandkit` (image boards), `design-taste-frontend-v1` (superseded by v2).

---

## 2. Design read and dials

**Reading this as:** an analysis console for non-experts (shopkeeper, teacher, coach, student) whose one job is to say what the data shows *and how far to trust it*, in a warm, handmade "ledger" language.

| Dial | Console | Landing page | Why |
|---|---|---|---|
| Design variance | 4 | 6 | A working tool needs predictable placement; the landing page can breathe more |
| Motion intensity | 3 | 5 | Ledger allows one orchestrated moment; the landing page can add one pinned section |
| Visual density | 4 | 3 | Results need room; the landing page is a first impression |

---

## 3. The central finding: the palette is generic, so the product must look distinctive some other way

The frontend-design skill lists the most common AI-generated look: *"a warm cream background (near #F4F1EA) with a high-contrast display and a terracotta or warm-clay accent."* The design-taste skill bans the same family: cream `#f5f1ea`-type grounds, clay `#b6553a`-type accents, espresso `#1a1714`-type text.

Ledger Day is `--stock #f7eedd`, `--pen #a34f20`, `--ink #3a2b1e`. That is the same family. The palette alone will not make this product recognisable.

What *is* distinctive already:
- **Baloo 2 + Mukta.** Rounded, warm, and chosen for Devanagari support. Rare in this category; keep them.
- **Plain language.** The seven step names and eight helper roles.
- **The product's actual promise.** "We check every answer twice": it tells you where it fooled itself. No competitor leads with that.

The plan therefore spends its boldness on the third point. The signature is **the audited entry** (section 5): every finding shows the checks it passed, like tick marks in a ledger. The palette question is left as an explicit decision (section 10, decision A).

---

## 4. Verified findings

### 4.1 Tokens and fidelity

| # | Finding | Evidence |
|---|---|---|
| T1 | Case Brief heading renders at 27px, not the intended 1.2rem: the global `h2 {font-size: 27px !important}` beats the inline style | `ui/styles.py:109`, `ui/tabs/answers_tab.py:46,55` |
| T2 | A third pen: night accent `#4fc3f7` (bright blue) in the showcase, and `linear-gradient` in its HTML. Both are banned by Ledger | `ui/cinematic_3d.py:57`, `ui/assets/cinematic_3d.js:116,126` |
| T3 | Night `--graphite` is `#d0c2a8` in code but `#b8a688` in `DESIGN.md` and `chart_theme.py` | `ui/styles.py:19`, `DESIGN.md:49`, `src/core/chart_theme.py:75` |
| T4 | Active-agent pulse hardcodes the night pen `rgba(240,162,74)`, so it glows amber in Day mode too | `ui/styles.py:413-415` |
| T5 | Theme fallback disagrees: session default is `"day"`, but the CSS falls back to `"night"` | `app.py:151` vs `ui/styles.py:11` |
| T6 | Palette hexes are copied in four places (`styles.py`, `landing.py:26-39`, `pipeline_3d.py:46-67`, `cinematic_3d.py:31-62`). T2 and T3 are the drift this causes | as listed |
| T7 | Fonts load from Google on every rerun (`@import`), in the report, and on the landing page. This breaks the `LOCAL_ONLY` privacy promise (the "local" run still contacts Google) and causes a flash of fallback font | `ui/styles.py:52`, `src/core/html_report.py:160-162`, landing `<link>` |

### 4.2 Charts (computed with the dataviz validator, not eyeballed)

The Ledger categorical palette fails in both modes:

```
Day   #a34f20 #a33526 #c08a2e #5b8c5a #8a7660 #b5714a   (surface #fffbf2)
  FAIL chroma floor      #5b8c5a, #8a7660 read as gray
  FAIL CVD separation    #8a7660 vs #5b8c5a  dE 1.5 (deutan)
  FAIL normal vision     #a33526 vs #a34f20  dE 5.7: pen and risk look the same to everyone
  WARN contrast          #c08a2e 2.94:1
Night #f0a24a #e2685a #d9a53e #7fb77e #b8a688 #d99a4e   (surface #2f251a)
  FAIL lightness band    5 of 6 too light
  FAIL CVD separation    #b8a688 vs #7fb77e  dE 1.1 (deutan)
  FAIL normal vision     #d99a4e vs #b8a688  dE 7.4
```

It also breaks Ledger's own rule. `--risk` is category 2 (`chart_theme.py:104`), so a series coloured red means "this may not hold" in the UI but "second category" in charts.

**Replacement, validated (all checks PASS, no warnings, same hue order in both modes):**

| Slot | Hue | Day | Night |
|---|---|---|---|
| 1 | Ledger rust (the pen) | `#a34f20` | `#cc7f34` |
| 2 | Ink blue | `#1f6aa0` | `#4d97cf` |
| 3 | Leaf green | `#3f7f4a` | `#4fa46a` |
| 4 | Plum | `#6d4a8c` | `#9d7fd0` |
| 5 | Ochre | `#9a7418` | `#a8892a` |
| 6 | Rose | `#c4648a` | `#c86e92` |

Notes:
- `--risk` leaves the categorical set and becomes a status colour only.
- Night slot 1 is not the UI pen `#f0a24a`, which is too light for the dark chart band. Chart inks and UI pens are separate token sets.
- Order matters: green must never sit next to rose (they collapse under deuteranopia). A 7th series folds into "Other".
- Re-run to confirm:

  ```
  node <dataviz>/scripts/validate_palette.js "<hexes>" --mode light --surface "#fffbf2"
  node <dataviz>/scripts/validate_palette.js "<hexes>" --mode dark  --surface "#2f251a"
  ```

Other chart items:
- Axes use `domainColor: ink` (`chart_theme.py:214`). The dataviz skill wants axes and grid to recede: use `rule`/`graphite`.
- Legends are square symbols with no direct labels. Series of 4 or fewer should be labelled directly.

### 4.3 The trust signal is buried

- **Model verdict hidden.** The certification/defect stamp, the product's most distinctive element, renders only inside the collapsed "Show technical detail" expander (`answers_tab.py:207-249`). It only covers the *model's* train-test gap, not individual findings.
- **Most checks never shown.** Finding cards show a headline plus either the detail text or the first caveat (`ui/components/cards.py:118-143`). The run's other checks never reach the user: multiple-testing adjustment (`p_adjusted`), sample size, confidence intervals, the new fragility audit, the causal-language guard, and target-leakage alerts.
- **Leakage alerts stop at memory.** They are stored in memory context (`target_leakage_alerts`) but never copied into `final_result`, so the UI cannot show them.

### 4.4 Generic patterns

- **Same card for everything.** ~15 components are the same 14px card with the same `--lift-sm` shadow: `.datum`, `.gauge`, both stamps, `.finding-card`, `.trust-cell`, `.du`, `.kpi-*`, `.agent-card`, `.bento-card`, `.exec-directive`, expanders, dataframes, `.reason` (`ui/styles.py`). Cards even nest (KPI tiles inside an expander card).
- **Hover lift on every card.** `.agent-card:hover`, `.bento-card:hover`, `iframe:hover` all lift, although most aren't clickable (`ui/styles.py:236,402,488`). The frontend-design skill calls this a generated-page tell.
- **All-caps tracked labels, against Ledger's sentence case.** `.trust-cell .k`, `.du .k`, `.kpi-gauge-label`, `.kpi-ring-sub`, `.kpi-tile .k`, `.agent-badge`, the Case Brief eyebrow, the stamp titles ("THE MODEL MEMORISED THE EXAMPLES", `cards.py:171,182`), and the facts-bar state ("FAILED / RUNNING / IDLE", `app.py:1324-1329`).
- **Emoji as UI.** 📂 🆓 💳 🟢 ⚡ 🔌 🚀 💻 🎬 🔬 ▶ in buttons and captions (`app.py:613,759-884,955,985,1432`). A matching SVG set already exists in `ui/components/icons.py`.
- **Em-dashes in visible copy.** 34 in `app.py`, 28 in `cards.py`, and more in the tabs, including the hero sub-line and the `.sc.active` "— working" suffix (`styles.py:610`).
- **Numbered markers on non-sequences.** Insight fallback uses `01, 02, …` (`answers_tab.py:124`).
- **Ring gauge for one number.** The data-quality score is a 140px SVG ring with a black drop-shadow and count-up animation (`answers_tab.py:216-228`); a stat figure says it better.

### 4.5 Information architecture

- **Hero button.** The only hero button toggles the 3D showcase (`app.py:984-988`). The real actions (upload, question, Run, sample report) all live in the sidebar.
- **Duplicate CTA.** "See a Sample Report (Demo)" appears twice, with different emoji (`app.py:955` and `:1432`).
- **Steps shown three times.** The idle page shows the seven steps as the 3D plate, the steps list, and a pill row (`app.py:1422-1426`), plus an eight-card helper grid.
- **Facts-bar noise.** "Theme: Day Mode" is shown as a fact about the run (`app.py:1331-1334`).
- **Sidebar jargon.** "Min iterations", "Max iterations", "Enable recursive decomposition (Stage 6)" (`app.py:932-934`) break the plain-language rule for this audience.
- **Misleading label.** "Ask a follow-up question" is a keyword search over existing finding headlines (`cards.py:146-160`), and the label promises more than it does.

### 4.6 Production

- **No real cancel.** The run executes synchronously inside the button handler (`app.py:1044` onward). Streamlit's Stop kills the script and discards partial results the callbacks already hold (`app.py:1210-1235`).
- **Observer leak.** `inject_micro_interactions()` adds a new `MutationObserver` on `document.body` every rerun and never disconnects the old one (`ui/animations.py:150-153`, called at `app.py:1441`).
- **Uncached assets.** Cinematic assets are re-read from disk every rerun (`ui/cinematic_3d.py:65-67`; `pipeline_3d.py:90-93` shows the cached pattern).
- **Plate re-embedded.** The 3D plate iframe is rebuilt on every rerun, including typing in the question box.
- **One breakpoint.** Only `max-width: 600px`; the two-column hero and four-tab row break between 600 and 900px.
- **Fragile CSS.** The theme targets Streamlit-internal `[data-testid]` selectors with no visual regression check; `streamlit>=1.49,<2` allows minor upgrades that can break it silently.
- **Report needs the internet.** The HTML report needs Google Fonts and the Vega CDN to render, and has no `@media print` styles (`html_report.py:160-169`).
- **Theme ignores OS preference.** The theme is manual only.

### 4.7 Landing page

- **Duplicate file.** `ui/landing_component/index.html` and `frontend-landing/index.html` are byte-identical. `frontend-landing/src/*.ts` is referenced by nothing.
- **Monospace everywhere.** JetBrains Mono is loaded and `var(--mono)` is used 21 times, against Ledger's "never reach for mono".
- **Title Case headings.** "Everything Needed to Trust Your Data", "Everything You Need to Know".
- **SaaS template structure.** Feature grid, "Choose Your Analytical Infrastructure" tiers, FAQ accordion, closing CTA.
- **Jargon.** "Interactive 3D Pipeline Synchronisation".

---

## 5. The signature: the audited entry

**Idea.** A ledger is where you record something and then check it. Every finding becomes an *entry*: a sentence in plain language, followed by a short row of check marks, each one a test the finding actually passed or failed. The run as a whole gets one verdict at the top. The answer and its trustworthiness become one object instead of an answer card plus a hidden technical expander.

```
  You asked: Why did sales drop in March?

  ┌ Verdict ─────────────────────────────────────────────┐
  │ 4 of 5 findings held up. 1 needs more data.          │
  └──────────────────────────────────────────────────────┘

  Weekend sales fell 18% after the price change.
  ✓ Not luck   ✓ 2,340 records   ✓ Holds without extreme days   ○ Pattern, not proof of cause

  North region customers spend 31% more per visit.
  ✓ Not luck   ! Only 42 records   ✓ Holds without extreme days
```

**Check rules** (derived only from data a finding already carries; a check that was not run shows nothing, never a fake tick):

| Mark | Plain label | Source | Pass / fail |
|---|---|---|---|
| Not luck | "Not luck" / "Could be chance" | `p_adjusted` (else `p_value`) | < 0.05 after run-level Benjamini-Hochberg |
| Records | "2,340 records" / "Only 42 records" | `evidence.n` / `n_level` | ≥ `_SMALL_N` in `src/core/plain_language.py` |
| Robust | "Holds without extremes" / "Depends on a few rows" | fragility audit (`src/core/sensitivity.py`) | not fragile |
| Leakage | shown only when flagged: "Too good to be true" | `target_leakage_alerts` | any alert naming the finding's dimension |
| Cause | "Pattern, not proof of cause" | causal-language guard, observational design | neutral note, never a tick |
| Holds on new data | model-driven findings only | `train_test_gap` | `gap_is_risky` is false |

**Design rules:**
- Ticks use `--positive`, the fail mark uses `--risk` (its one meaning), and the neutral note uses `--graphite`.
- Every mark has a text label; colour alone never carries meaning.
- The one orchestrated motion moment: when a run finishes, the ticks ink in left to right (opacity plus a 2px rise, 60ms stagger, `cubic-bezier(.16,1,.3,1)`), gated on `prefers-reduced-motion`. Nothing else in the console animates by itself.
- The verdict replaces the buried model stamp at the top of Answers. The model-level stamp becomes one line of it.
- The same entry and tick component renders in the HTML report, so the shared artifact carries the audit too.

**Backend prerequisites** (small, no schema change):
1. Have the audits write machine-readable results into `Finding.evidence["checks"]`, e.g. `{"fragile": false, "leakage": false}`. `evidence` is already a free dict, so this needs no new field. Otherwise the UI would be parsing caveat strings.
2. Copy `target_leakage_alerts` into `final_result` in `controller.py`.

---

## 6. Phased plan

Effort: S (hours), M (1-3 days), L (a week+). Each item has an acceptance check.

### Phase 1: correctness and quick wins (2-4 days)

| # | What | Where | Effort | Done when |
|---|---|---|---|---|
| 1.1 | Case Brief: drop the eyebrow and inline styles; the objective becomes the heading ("You asked: …") | `answers_tab.py:41-59`, `styles.py` | S | No inline `style=` in the block; heading size matches the h2 scale |
| 1.2 | Remove `#4fc3f7` and gradients from the showcase; use Ledger night tokens | `cinematic_3d.py:31-62`, `cinematic_3d.js:116,126`, `cinematic_3d.html` | S | `grep -c 4fc3f7` = 0; no `linear-gradient` |
| 1.3 | Fix token drift T3, T4, T5 | `styles.py:11,19,413-415` | S | Night graphite matches DESIGN.md; pulse uses `var(--pen)` via `color-mix`; one fallback theme |
| 1.4 | Swap in the validated chart palette; take `--risk` out of the category range; recessive axes | `src/core/chart_theme.py:103-108,214-215` | S | Validator passes both modes; `RISK_*` not in `CATEGORY_RANGE_*` |
| 1.5 | Single MutationObserver: store it on the parent window, disconnect before re-attaching | `ui/animations.py:150-153` | S | Observer count stays 1 after 20 reruns (check via DevTools) |
| 1.6 | `@lru_cache` cinematic asset reads | `ui/cinematic_3d.py:65-67` | S | Matches `pipeline_3d.py` pattern |
| 1.7 | Copy sweep: emoji to `icons.py` SVGs or plain words; em-dashes out of visible strings; sentence case for all labels, stamps and statuses; `01/02` markers removed | `app.py`, `ui/components/cards.py`, `ui/tabs/*`, `styles.py` (`text-transform` rules) | M | `grep` for emoji ranges, `—`, and `text-transform: uppercase` in UI files returns 0 |
| 1.8 | Facts bar says what matters: "Ready" / "Working on step 3 of 7" / "Stopped at step 4" / "Done"; drop the Theme cell | `app.py:1321-1338` | S | No all-caps states; no theme cell |
| 1.9 | One sample-report button, one label ("Try it with sample data"), in the main column | `app.py:954-961,1432-1434` | S | One occurrence of that intent |
| 1.10 | Rename "Ask a follow-up question" to "Search these findings" | `answers_tab.py:187-204` | S | Label matches behaviour |
| 1.11 | Plain-language sidebar settings: "How thorough" (Quick / Thorough / Deep) mapping to iterations; "Break hard questions into smaller ones" | `app.py:930-934` | S | No "iterations" or "Stage 6" in visible text |
| 1.12 | Delete the duplicate landing (`frontend-landing/`) or decide to finish its TS rewrite; don't keep both | `frontend-landing/` | S | One landing source |

### Phase 2: production and structure (1-2 weeks)

| # | What | Where | Effort | Done when |
|---|---|---|---|---|
| 2.1 | One token module (`ui/tokens.py`) imported by styles, landing, plate, showcase and `chart_theme`; a test asserts the hexes equal `DESIGN.md` | new `ui/tokens.py` + 5 importers | M | One definition per hex; test green |
| 2.2 | Main-column entry flow: file, question, and Run live in the main column as the first "entry"; the sidebar keeps provider, key and settings only; the hero button becomes the primary action | `app.py:972-1000`, sidebar block | M | A first-time user never needs the sidebar for the default run |
| 2.3 | Real cancel and partial results: run the controller in a worker thread writing progress to a session-scoped queue; an `st.fragment(run_every=…)` renders progress; "Stop" sets a flag the controller's step callback checks; results so far render as a "Stopped at step N" report | `app.py:1044-1300`, controller callback hook | L | Stopping mid-run shows completed findings; no lost state |
| 2.4 | Isolate the plate in an `st.fragment` so unrelated reruns don't rebuild the iframe | `app.py:196-227` | M | Typing the question does not reload the iframe |
| 2.5 | Self-host Baloo 2 + Mukta (`server.enableStaticServing`, `static/fonts`, `@font-face` + `font-display: swap`) for console, report and landing; drop JetBrains Mono | `.streamlit/config.toml`, `styles.py:52`, `html_report.py:160-162`, landing | M | No request to `fonts.googleapis.com` in the network log; `LOCAL_ONLY` run makes zero third-party requests |
| 2.6 | Offline, printable report: inline the Vega runtime (or fall back to static SVG), add `@media print` (page breaks per section, no shadows, URLs after links) | `src/core/html_report.py` | M | Report renders with the network off; prints cleanly to PDF |
| 2.7 | Follow the OS theme by default (`st.context.theme`, Streamlit ≥1.46), with the manual toggle as an override | `app.py:151,592-601`, `styles.py:11` | S | Fresh session matches the OS setting |
| 2.8 | Fewer cards (needs decision B): keep elevation only for the verdict, the plate, and expanders; stat tiles become figures on a hairline; KPI tiles stop nesting; hover-lift only on clickable elements | `styles.py` card rules, `answers_tab.py`, `cards.py` | M | ≤ 4 components use `--lift-sm`; no nested cards |
| 2.9 | Breakpoint 600-900px for hero columns, tab row and facts bar; 44px touch targets | `styles.py` | M | No horizontal scroll at 375 / 768 / 1024px |
| 2.10 | Visual regression guard: headless screenshots of the four tabs in both themes on a sample run, diffed before any Streamlit upgrade | `scripts/ui_snapshots.py` | M | Script runs in CI or pre-upgrade |
| 2.11 | Loading and empty states: a skeleton of the Answers layout while running; empty tabs explain what to do next, without apologising | `ui/tabs/*` | M | No blank tab and no bare spinner |
| 2.12 | Idle page: one account of the seven steps (keep the plate plus the steps list; drop the pill row) | `app.py:1416-1434` | S | Steps appear at most twice |

### Phase 3: signature and identity (1-2 weeks)

| # | What | Where | Effort | Done when |
|---|---|---|---|---|
| 3.1 | Backend prerequisites for audited entries (section 5): `evidence["checks"]` written by the audits; leakage alerts copied into `final_result` | `src/core/controller.py`, `src/core/sensitivity.py` | S/M | Checks present in `report["findings"][i]["evidence"]` |
| 3.2 | The audited entry component: sentence plus check row, per the rules table | `ui/components/cards.py:118-143` (replaces `render_finding_card`) | M | Every check rendered has a text label; unrun checks render nothing |
| 3.3 | The verdict at the top of Answers; the model stamp moves out of the expander into it | `answers_tab.py`, `cards.py:163-188` | M | Verdict is visible without expanding anything |
| 3.4 | The one motion moment: ticks ink in on completion; remove the perpetual `pulseActive`, count-up and ring animations | `styles.py`, `animations.py`, `answers_tab.py:216-228` | S/M | Under reduced motion, nothing animates; otherwise only the tick reveal |
| 3.5 | The report as the bound ledger: same entries and ticks, the verdict on the first page, the question as the title, printable | `src/core/html_report.py` | M | The report and the console show identical trust marks |
| 3.6 | Charts speak the same language: direct labels for ≤4 series; a chart tied to a finding shows that finding's sentence and ticks as its caption | `src/core/dashboard.py` chart specs, `charts_tab.py` | M | No legend-only charts with ≤4 series |
| 3.7 | Landing rebuilt around the promise: hero "We check every answer twice." with a real audited entry from the sample dataset as the hero visual (a real component preview, not a mock screenshot); one pinned section showing a finding failing a check; no tiers, no FAQ accordion; sentence case; no mono | `ui/landing_component/index.html` | L | Passes the design-taste pre-flight list (section 14 of that skill) |
| 3.8 | Tighten `DESIGN.md`: add the dials, the audited-entry component, the chart palette, the banned list (emoji, em-dash in UI, all-caps labels, hover lift on non-interactive cards) | `DESIGN.md` | S | The doc matches the code |

### Phase 4: optional architecture

Stay on Streamlit. Nothing above needs a rewrite. If Phase 2.3 (threaded run plus fragment polling) proves too coarse for live progress, the next step is a narrow bidirectional custom component for progress and the plate, reusing the `postMessage` pattern `ui/landing_component` already has. Tabs, forms, tables and charts stay native. (L)

---

## 7. The dashboard (Charts tab)

Built from the skills (dataviz above all, plus frontend-design and redesign-existing-projects) and from published guidance, listed under Sources at the end of this section. All current-state claims below were checked in code or in the saved `dashboard.json` files of eight past runs under `output/`.

### 7.1 What it is today

**Worth keeping (it already does these well):**
- **Emphasis.** Finding charts already use the emphasis form: the finding's segment is solid, the rest at 0.45 opacity (e.g. the segment-lift bar spec in `output/runs/1789840808-68c96ce2/reports/dashboard.json`).
- **Honest tooltips.** They carry 95% CI bounds and sample size, and axes/tooltips are unit-aware (`chart_theme.axis_format`).
- **Curation.** Finding-tagged panels lead (`dashboard.py:2673-2681`). Duplicate panels of the same data are dropped (`:2631-2660`). Charts that would render empty never ship (`:2678`).
- **Heatmap scale.** The correlation heatmap uses a diverging scale centred on 0 (`dashboard.py:1448`).

**What holds it back:**

| # | Finding | Evidence |
|---|---|---|
| D1 | **No overview.** The Charts tab opens straight into panel 1. Nothing says what the dashboard concludes, so it fails the "5-second test" (can a reader state the main answer within five seconds?) | `ui/tabs/charts_tab.py:20-43` |
| D2 | **Titles describe the chart, not the finding.** Real titles: "Distribution – AH", "Segments – 2 clusters (silhouette 0.29)", "Where the Model Gets It Wrong – Confusion…", "Relationship – C6H6(GT) vs PT08.S2(NMHC)". Raw column names and jargon (silhouette, ROC, confusion) for a non-expert audience | 8 saved `dashboard.json` files |
| D3 | **The takeaway is buried and technical.** For tagged panels the finding sentence exists, but only as a small caption below the chart, with raw names and ratios: "Apparel product_category customers average amount of $410.97 vs $1,625.02 for everyone else (0.25x, 353 vs 400)" | same file, `caption` field; `cards.py:487-489` |
| D4 | **No structure.** 3-13 panels (8-12 typical) in one flat list: the first three full width, the rest two per row. Charts about the same question are not grouped; distributions sit between findings | `charts_tab.py:21-43` |
| D5 | **No interaction beyond tooltips.** No legend isolation, no cross-filtering, no filter row: zero Vega `params` in `dashboard.py`. Streamlit 1.57 (installed) supports `st.vega_lite_chart(on_select=…, selection_mode=…, key=…)` | `grep -c '"params"' src/core/dashboard.py` = 0 |
| D6 | **No table view.** A value is only reachable by hovering. Fails the dataviz rule "tooltips enhance, never gate" and WCAG for non-mouse users | `cards.py:475-495` |
| D7 | **Screen readers get nothing.** The Vega spec has no top-level `description`; `ChartSpec.description` exists but is only shown as fallback visible text | `cards.py:490-495`; spec dump |
| D8 | **Marks off-spec.** Dashed gridlines `gridDash [2,3]` (a listed anti-pattern), axis domain in full ink, area fill at 0.35 opacity (spec: ~0.10), square legend symbols for line series, no 4px rounded bar ends, no bar-thickness cap, reference line dashed and unlabelled | `chart_theme.py:212-236`; spec dump |
| D9 | **Theme-blind scales.** `blueorange` and `oranges` are Vega built-ins, not Ledger tokens; the pale diverging midpoint glares on the Night sheet | `dashboard.py:1185,1299,1448`; `chart_spec.py:1108-1112` |
| D10 | **Categorical palette fails colour-blindness checks** (section 4.2), and uses `--risk` as a category | `chart_theme.py:103-108` |
| D11 | **Every panel is a bordered card** (`st.container(border=True)`) inside a tab: the same card kit as section 4.4 | `cards.py:477` |

### 7.2 Principles for this dashboard

This is not a monitoring dashboard. It explains one analysis to a non-expert. So the rules are:

1. **Overview first, then details on demand** (Shneiderman's mantra; the "inverted pyramid" in current dashboard guidance). The conclusion sits at the top, the drivers in the middle, the raw data at the bottom.
2. **The top band fits one screen** (Few: "monitored at a glance"). It answers the question with no scrolling and no interaction.
3. **Titles state the takeaway in plain words; the subtitle carries the precise description** (Datawrapper: conversational titles, technical detail in the description).
4. **Label directly; use a legend only where direct labels can't work** (Datawrapper and dataviz). A single series gets no legend.
5. **Emphasis is the default form for a finding**: the thing the finding is about in the rust pen, everything else in de-emphasis graphite (dataviz "choosing a form").
6. **Every chart has a table twin; tooltips never gate a value** (dataviz interaction rules).
7. **One filter row above everything it scopes, never per chart** (dataviz interaction rules).

### 7.3 Proposed layout

```
Charts
┌ The short answer (one screen) ───────────────────────────────────────┐
│  Apparel customers spend about a quarter    │  $1,988   avg per order │
│  of what everyone else does.                │  4 of 5   findings held │
│  ✓ Not luck  ✓ 353 records  ✓ Holds…        │  12,408   orders        │
│  [ hero chart: the top finding, emphasis form, direct-labelled ]      │
└───────────────────────────────────────────────────────────────────────┘
 Filter: Region [All ▾]   Period [All · Last 12 months · Custom]   (only if the data has them)

 Your question                     finding-tagged panels, takeaway titles, check rows
 What drives it                    drivers, model comparison (plain-language titles)
 Over time                         trend, change, waterfall
 How the data is spread            distributions, category counts, heatmap: collapsed by default
```

- **The overview band.** Leads with one hero figure: the top finding's headline effect, ≥48px, in Mukta, not a display face (dataviz rule). Next to it, at most 3 stat tiles, drawn from existing data (row count, findings that held, the key measure's overall value), each with a plain label. Below them, the top finding's chart as the hero chart.
- **Sections are derived deterministically from existing fields,** with no new LLM call:

  | Section | Rule |
  |---|---|
  | Your question | `finding_id` set and finding `layer == "exec"` or `objective_fit` high |
  | What drives it | `chart_id` in `model_drivers`, `model_comparison`, `top_correlations`, `scatter_top_pair` |
  | Over time | `chart_id` starts with `time_series`, `change_`, `waterfall`, or has a temporal x field |
  | How the data is spread | everything else (EDA tier) |

- **Empty sections are omitted,** not shown with a placeholder.
- **"How the data is spread" is collapsed by default.** It is detail on demand; today it takes equal space with the answers.
- **Wide-screen layout.** Findings that carry a chart render full width with text beside the chart. Supporting charts pair two per row only when both are the same form, so heights line up.

### 7.4 Anatomy of one panel

```
 Apparel customers spend about a quarter of what others do     ← takeaway title (finding headline, plainified)
 Average order value by product category, with 95% ranges      ← subtitle: what is plotted + units
 ✓ Not luck   ✓ 353 records   ✓ Holds without extreme orders   ← check row (same component as Answers)
 ┌──────────────────────────────────────────────┐
 │  ▇ Apparel $411        (rust, direct label)  │
 │  ▆ Electronics  ▅ Home  ▅ Toys  (graphite)   │
 │  ── Average $1,988 (solid hairline, labelled)│
 └──────────────────────────────────────────────┘
 Show as table · Based on 12,408 orders                        ← table toggle + source line, small graphite
```

- **Title.** For finding-tagged panels the title is the finding headline through `plainify`. For untagged charts, `chart_spec.describe_chart` already produces a sentence; use it, never "Distribution – X".
- **Subtitle.** Today's descriptive title, humanised (`humanize_label`), with units.
- **Reference lines** are solid graphite hairlines with a direct label ("Average $1,988"), not unlabelled dashes.
- **Table view.** A "Show as table" toggle renders the chart's own rows with `st.dataframe` and `column_config` formats matching the axis formats. The same rows, so the numbers always agree.
- **Accessibility.** Set the top-level Vega `description` to the takeaway plus subtitle, so screen readers announce the finding (Vega-Lite's description is the recommended single-attribute summary).
- **Container.** No bordered card per panel (D11). Panels separate by whitespace and a 2px section rule, as `DESIGN.md` section heads already do. Elevation stays reserved for the overview band.

### 7.5 Mark and theme spec changes (`src/core/chart_theme.py`)

| Setting | Now | Proposed | Why |
|---|---|---|---|
| Categorical range | fails CVD, includes risk | validated 6-slot palette (section 4.2) | colour-blind safe; red keeps one meaning |
| Grid | `gridDash [2,3]`, `rule` colour | solid 1px, `rule-faint` | dashed grid is a listed anti-pattern |
| Axis domain / ticks | `ink` | `rule`, ticks off | axes recede; the data is the loud part |
| Axis and legend titles | Baloo 2 600 | Mukta 600, graphite | headings belong to the page, not the plot |
| Bar | no radius, band fills slot | `cornerRadiusEnd: 4`, thickness capped at ~24px (relative band width) | dataviz mark spec |
| Line | default | `strokeWidth: 2`, round cap/join | mark spec |
| Point | default | size ≈ 64 (r ≈ 4), filled, 2px surface-colour stroke ring | legible where points overlap |
| Area | opacity 0.35 | opacity 0.10 | a wash, not a block |
| Legend symbol | square for all | `stroke` for lines, `square` for bars; hidden for one series | the legend mirrors the mark |
| Sequential scale | built-in `oranges` | a Ledger rust ramp per mode | theme-aware |
| Diverging scale | built-in `blueorange` | ink-blue ↔ rust per mode, neutral midpoint one step off the sheet | no glare in Night mode |

After changing the palette and ramps, run the validator again: categorical with `validate_palette.js`, ramps with `--ordinal`.

### 7.6 Interaction

| Level | What | How | Effort |
|---|---|---|---|
| Keep | Tooltips with CI and n | as today; values lead, labels follow | – |
| Add | Legend click-to-isolate on multi-series charts | Vega-Lite legend-bound param (`select: point, bind: legend`); no rerun | S |
| Add | Table view per panel | 7.4 | S |
| Add | Filter row (region-like dimension, time range) | only when the profile has a low-cardinality dimension or a time axis. Filters rebuild the panels whose data carries that field; panels built from model output say "Not filtered: based on the full data" | L |
| Optional | Cross-filter: click a bar to filter the others | `st.vega_lite_chart(on_select="rerun", selection_mode=…, key=…)`; same slice logic as the filter row | L |

**Honest limit.** Filtering means recomputing chart data on a slice of the dataset. Findings, p-values and models were computed on the full data and must not silently change. So a filtered view labels itself "Exploring a subset: the checks above were run on all the data", and the check rows hide while a filter is active.

### 7.7 Dashboard phases

| # | What | Where | Effort | Done when |
|---|---|---|---|---|
| **Phase 1** | | | | |
| DB1 | Chart theme spec changes (7.5) and the validated palette | `src/core/chart_theme.py` | S | Validator passes; no `gridDash`; bars have rounded ends |
| DB2 | Vega `description` on every spec (takeaway + subtitle) | `render_dashboard_chart`, `cards.py:475` | S | Every rendered spec has `description` |
| DB3 | Takeaway titles: finding headline as title, current title as subtitle; `describe_chart` sentence for untagged panels | `cards.py:475-495`, `dashboard.py:_attach_finding_metadata` | M | No panel title matches `^(Distribution|Relationship|Trend|Segments|Category Counts) –` |
| DB4 | Label reference lines ("Average $1,988"), solid graphite | `dashboard.py` group/segment charts (`:1898-2005`, `:2197-2297`) | S | No unlabelled `strokeDash` rules |
| **Phase 2** | | | | |
| DB5 | Overview band: hero figure, ≤3 stat tiles, hero chart | `charts_tab.py` | M | The main answer is visible without scrolling at 1366×768 |
| DB6 | Sections (7.3 mapping); "How the data is spread" collapsed | `charts_tab.py`, a small `section_for(chart)` helper next to `build_dashboard` | M | Every panel lands in exactly one section; empty sections are omitted |
| DB7 | Table view toggle per panel | `cards.py:render_dashboard_chart` | S | Every panel's values are reachable without hovering |
| DB8 | Drop per-panel bordered cards; whitespace + section rules | `cards.py:477`, `styles.py` | S | No `st.container(border=True)` around charts |
| DB9 | Legend isolation on multi-series charts; direct labels for ≤4 series | `chart_spec.py` builders, `dashboard.py` time-series/cluster charts | M | No legend-only chart with ≤4 series |
| DB10 | Ledger sequential and diverging ramps, per mode | `chart_theme.py`, `dashboard.py:1185,1299,1448`, `chart_spec.py:1108-1112` | S | No built-in scheme names left in specs |
| **Phase 3** | | | | |
| DB11 | Check rows on finding panels (the audited-entry component from section 5) | `cards.py` | S (after 3.2) | Same checks as the Answers tab, same rules |
| DB12 | Filter row with the honest-limit labelling (7.6) | `charts_tab.py`, dashboard rebuild on slice | L | A filtered view never shows full-data checks as if they applied to the slice |
| DB13 | Optional cross-filter via `on_select` | `charts_tab.py` | L | Only after DB12 is stable |
| DB14 | The same layout in the HTML report (overview band, sections, takeaway titles) | `src/core/html_report.py` | M | Report and Charts tab tell the same story in the same order |

### Sources

- [Common Pitfalls in Dashboard Design – Stephen Few (Perceptual Edge)](https://www.perceptualedge.com/articles/Whitepapers/Common_Pitfalls.pdf)
- [Book review: Information Dashboard Design – UXmatters](https://www.uxmatters.com/mt/archives/2007/04/book-review-information-dashboard-design.php)
- [Dashboard design best practices – Domo](https://www.domo.com/learn/article/dashboard-design-examples-best-practices) (inverted pyramid, top-left priority)
- [Effective dashboard design – DataCamp](https://www.datacamp.com/tutorial/dashboard-design-tutorial) (five-second rule)
- [Dashboard design principles – Luzmo](https://www.luzmo.com/blog/dashboard-design)
- [What to consider when using text in data visualizations – Datawrapper](https://www.datawrapper.de/blog/text-in-data-visualizations)
- [Annotations in bar, range and dot charts – Datawrapper](https://www.datawrapper.de/blog/annotations-in-bar-charts)
- [st.vega_lite_chart – Streamlit docs](https://docs.streamlit.io/develop/api-reference/charts/st.vega_lite_chart) (`on_select`, `selection_mode`, `key`)
- [Accessible data-visualisation tooling in 2026 – Disability World](https://www.disabilityworld.org/articles/accessible-data-viz-tooling-2026/) (Vega-Lite `description` for screen readers)
- [Config – Vega docs](https://vega.github.io/vega/docs/config/) (ARIA properties for SVG output)

---

## 8. Considered and rejected

| Idea (source skill) | Why not |
|---|---|
| Glass panels, double-bezel cards, floating pill nav (high-end-visual-design) | Ledger is flat paper. Glass on a warm cream page reads as a template, and costs blur repaints |
| Scroll-hijack, card stacks, text-scrub reveals in the console (gpt-taste) | Users come to read results, and scroll-driven motion slows them down. Allowed once on the landing page only |
| Perpetual pulse, shimmer or typewriter loops (stitch-design-taste) | Contradicts Ledger's single motion moment; the active-agent pulse is being removed for this reason |
| Serif display headings (minimalist-ui) | Baloo 2 is the identity; a serif would move toward the generic cream-and-serif look from section 3 |
| Stock or picsum photography (design-taste, minimalist-ui) | Third-party image requests contradict `LOCAL_ONLY`; a real audited entry is a better hero visual than a photo |
| Brutalist or telemetry aesthetic (industrial-brutalist-ui) | It is the retired Drafting Table system |

---

## 9. Risks

1. **Token drift returns** unless 2.1 lands. T2 and T3 are what four copies produce.
2. **Checks must never look better than the evidence.** An unrun check must render nothing. A tick for a test that was not done would betray the product's promise.
3. **Threaded runs in Streamlit (2.3)** need care: no `st.*` calls from the worker thread, and the controller must hold no thread-unsafe global state. Prototype before committing.
4. **Streamlit upgrades** can break `data-testid` CSS silently until 2.10 exists.
5. **The landing rebuild (3.7)** can absorb unbounded time; timebox it. The literal-iconography 3D plate stays out of scope, as `DESIGN.md` already says.
6. **Dashboard filters (DB12) can mislead.** A slice can show a pattern the full-data checks never tested. The honest-limit labelling in 7.6 is not optional; if it can't be built, ship the dashboard without filters.
7. **Takeaway titles are only as good as the finding headlines.** DB3 surfaces `plainify` output as the title, so any awkward headline becomes a visible title. Review a sample of real headlines before shipping.

---

## 10. Decisions for you

**A. The palette.** Ledger Day sits in the most common AI palette family (section 3). Options:
- **A1 (recommended): keep it** and earn distinctiveness through the audited entry, the type, and the copy. Lowest risk, and consistent with the redesign you approved two weeks ago.
- **A2: shift toward real ledger paper.** Pale buff-green stock, blue feint rules, a red double margin rule. It comes from the metaphor itself and nobody else uses it. But it reopens a settled decision and brings back ruled lines, which `DESIGN.md` rejected (though ledger ruling is not the engineering quadrille).

**B. Fewer cards (2.8).** `DESIGN.md` says "rounded cards instead of ruled plates". Cutting from ~15 carded components to about 4 keeps that spirit (soft, rounded, friendly), but the doc's Structure section needs rewording. Approve or skip.

**C. Landing page.** Rebuild around the promise (3.7, L), or keep the current page with only the Phase 1 fixes.

**D. Dashboard interactivity.** How far to go past the static improvements (7.6):
- **D1 (recommended): stop at Phase 2.** Overview band, sections, takeaway titles, table views, legend isolation. No filters. Every number on screen matches the checked, full-data analysis.
- **D2: add the filter row (DB12).** Useful for exploring, but it needs the "exploring a subset" labelling, and it is the largest dashboard item.
- **D3: add cross-filtering as well (DB13).**

