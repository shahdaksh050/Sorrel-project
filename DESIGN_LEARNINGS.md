# Design learnings (2026-10-03)

What was researched, decided and built for a possible "v2" interface, and what to carry into the
**current** Streamlit UI. The v2 attempt (FastAPI + React, named "Doubly") was **scrapped by the user**
because they did not like the new UI. The design principles are still wanted, applied to the current UI.

## 0. Status and where the scrapped work is

- Everything built for v2 was moved into a git stash, not deleted. List it with `git stash list`; the
  entry is named `v2 attempt (scrapped 2026-10-03)`. Restore it with `git stash pop` (or `git stash apply`
  to keep a copy in the stash). A different, older stash (`WIP on daksh/updates`) is not related.
- The stash holds: `api/` (FastAPI backend), `web/` (React/TypeScript app), the data-truth fixes in `src/`
  (section 7), `V2_DESIGN_PLAN.md`, `V2_TECHNICAL_PLAN.md`, the logo and font specimen, tests.
- Kept in the working tree: this file, `UI_DESIGN_AUDIT.md` (the audit of the current UI, still valid),
  `GEMINI.md` (the other agent's coordination ledger).
- Nothing in the v2 work was ever run: no tests, no app, no browser. Every behaviour claim below that
  concerns v2 is therefore unverified. What *was* measured: contrast ratios, font features, git state.
- Why the user rejected it: **not recorded**. Ask before reusing any of it.

## 1. The product and what the interface must show

- An agentic data-analysis assistant for technical and non-technical people. Judged equally by examiners
  (live demo) and everyday users.
- The differentiator is the **checking**: it audits its own answers and says which held up. The interface
  should make that visible, not just "AI analysis".
- Constraints from the repo: answers first, plain language, `--risk` means only "this number may not
  hold", AA contrast, reduced motion, keyboard access, no third-party hosts at runtime, LLM and dataset
  text escaped (never raw HTML).

## 2. What the research said (sources)

- Progressive disclosure supports non-experts: show the simple thing first, advanced options one level
  deeper (https://arxiv.org/pdf/1811.02164).
- Communicating uncertainty to non-experts needs care, and transparency calibrates trust better than
  reliability alone (https://ir.cwi.nl/pub/23433/23433B.pdf, https://arxiv.org/pdf/2510.15769).
- 3D earns its place only when it answers "what can I do, what just happened, what state am I in?", and a
  non-3D path must complete the same task: interaction budget, capped pixel ratio, fallbacks
  (https://hackernoon.com/the-interaction-budget-keeping-webgl-interfaces-usable-on-real-devices).
- The visible tells of AI-generated design (https://www.impeccable.style/slop): one-sided coloured
  borders on cards, nested cards, identical card grids, decorative glow, glassmorphism, purple-blue
  gradients, "Inter everywhere", pulsing dots, one accented word in a headline, hover effects on things
  that cannot be clicked, em-dash overuse.
- Naming: short, easy-to-say real words read as more trustworthy; a name should not say "trust" itself;
  metaphor words (assay, hallmark, touchstone) suit "tested for quality". (namestation / siegelgale
  results from the search.)
- Streamlit's limits: the rerun model, a restricted set of UI elements, custom components isolated or
  unable to change app CSS (https://docs.streamlit.io/develop/concepts/custom-components/components-v2).
- The frontend-design guidance used: five common AI-default clusters to avoid (warm paper plus terracotta;
  near-black plus one acid accent; broadsheet hairlines and zero radius; the SaaS card kit; template
  chrome such as eyebrow labels, middle-dot strings, arrows on links, mono small labels).
- The apple-design skill (Apple WWDC material), section 6 below.

## 3. Principles that were decided (apply these to the current UI)

### 3.1 Colour carries meaning
- Colour is reserved for verdicts. Everything else is ink on a neutral page. When a person sees colour it
  tells them how far to trust something.
- The engine has **three** states per headline finding, not two:
  `held up` (audited, no check failed), `needs more data` (a check failed), `not checked` (no pass/fail
  audit mark). A finding whose only mark is the neutral "pattern, not proof of cause" note is **not
  checked**. The code lives in `src/core/run_view.py` (`compute_verdict`, `audited_checks`).
- Never show an unchecked finding as failing. "Needs more data" only when a check actually failed.
- The verdict count must equal the number of cards listed. (Audit A9: "4 of 4 held up" over 5 cards.)
- A success state looks like success: a clean pass is not rendered in a rust/red tint (audit A13).
- Measured contrast (WCAG 2.x, computed): a candidate palette of green `#176B55` and crimson `#B3263E` on
  `#F1F3F4` gave about 5.8:1 each, but the two have **the same luminance** (ratio 1.00), so people with
  colour-vision differences cannot tell them apart. Every verdict must also carry a shape and a word:
  held = filled circle + "Held up"; needs more = ringed circle with a cross + "Needs more data"; not
  checked = hollow dotted circle + "Not checked". The dotted ring was too faint at 18 px; use a thicker
  stroke.
- Control borders need 3:1 (WCAG 1.4.11). The current UI's `--rule` is 1.6 to 1.8:1 and Day `--accent` is
  2.3:1 (see `FRONTEND_AUDIT.md` section 3).

### 3.2 Structure
- The first screen is the task (upload, question, run), not a marketing headline. Putting "We check every
  answer twice" on the workspace after the user has already entered pushed the upload card 540 px down.
- A sample run is a first-class choice next to the upload, not a button below an 8-card grid.
- One raised surface at most; sections are unboxed; no cards inside cards; no identical card grids.
- Claims and their evidence belong together: list the findings, show the selected finding's chart, checks
  and caveats beside it. Do not send people to another tab to find the chart.
- Progressive disclosure for settings: one plain block for connecting an AI (users bring their own key),
  everything else (model depth, CV folds, GPU, container isolation, code-run budgets) behind "Advanced"
  with safe defaults. Make "with an AI summary / without" a visible choice instead of a toggle that
  silently disables Run.
- Target column should be a dropdown of the file's own columns, with "let the assistant decide" as the
  default. Do not make people type a column name from memory.
- One name for the pipeline. The current UI has three (7 stages, 8 "agents", 4 groups on the landing).

### 3.3 Type
- A real scale (13, 16, 20, 25, 31, 40, 56 px was used), not about 19 sizes with half-pixel steps.
- Size-specific tracking and leading: display about -0.02em / 1.05, headings -0.01em / 1.2, body 0 / 1.6,
  small +0.01em / 1.45. Sizes in `rem` so the browser's text size scales the layout. Line length under
  68 characters. Sentence case, no all-caps labels.
- Mono only for text that is verbatim from the user's file (column names, cell values).
- Fonts that were shortlisted: Bricolage Grotesque (headings), Public Sans (text), IBM Plex Mono (data).
  Verified with fontTools on the shipped files: Bricolage and Public Sans have a `tnum` (tabular figures)
  feature; Plex Mono's digits are already equal width; all three are OFL-1.1 on npm via
  `@fontsource-variable/*` and `@fontsource/ibm-plex-mono`. A rendered specimen with real UI strings read
  well in Day and Night. Whether they look right in the real app was never checked.

### 3.4 Shape and surfaces
- Radii by role (controls 6 px, one raised surface 12 px, marks round), not one radius everywhere.
- No decorative shadows, no gradients, no glow, no glass.
- Hover and spotlight effects only on things that can be clicked. The current UI has a pointer-following
  glow on static finding cards and a spinning conic border on static stat tiles (audit A5).
- Do not use a coloured left stripe as the universal callout (audit A4); give each kind of content its
  own form.

### 3.5 Copy
- Plain verbs, one name per action across the flow, errors that say what happened and what to do, raw
  technical text behind a disclosure.
- No arrows on links, no emoji, no middle-dot separated strings, no spaced em dashes, no eyebrow labels.
- Specific labels over generic ones. Numbers shown to people are rounded for people (audit A10), and a
  count that differs between two cards must say why (audit A11).
- Examples: "Accuracy on data it had not seen" for CV score; "How much worse it did on new data" for the
  train-test gap; "4 of 4 checked findings held up. 1 finding was not checked."

### 3.6 Motion and interaction (from the apple-design skill)
Adopted: respond on pointer-down (`:active`) not on release; feedback continuous during an interaction;
interruptible motion that animates from the live value; springs (damping 1.0, response 0.3 to 0.4 s;
bounce about 0.8 only after a flick); momentum projection `project(v) = (v/1000)*0.998/(1-0.998)` and
velocity handoff; rubber-banding `(o*d*0.55)/(d+0.55*|o|)`; exit along the path of entry; reduced motion
becomes a short cross-fade; `prefers-reduced-transparency` and `prefers-contrast: more` honoured; inline
validation; wayfinding on every screen.
Not adopted: translucent glass chrome and blur (it is on the AI-tell list and conflicts with the no
decoration rule; only a modal's dimming layer may be translucent), haptics and sound, motion on arrival
beyond one allowed moment, and a system font as default (the custom fonts have a stated reason).
Streamlit note: keyframe animations and CSS transitions cannot be grabbed mid-flight. In Streamlit most
of this can only be approximated (press feedback, reduced-motion handling, type, copy, structure); true
spring and gesture work needs a component or another front end.

### 3.7 3D (the user wanted a wow factor with a purpose)
- The purpose test: the 3D must answer a question a table cannot, never animate anything the pipeline did
  not do, and have a non-3D equivalent that completes the same task.
- The audit judged the three existing scenes: the landing grid (keep, it is honest that it illustrates),
  the workspace plate (updates less often than the numbered list beside it, then hides in an expander),
  the Cinematic Showcase (a third illustration of the same pipeline).
- The idea that was proposed: one object built from the user's own rows (positions from a 3-axis
  projection, axis labels worded honestly as blends, groups shown by outlines not colour, no cell values
  sent). It was never rendered; the first implementation was scrapped, and a design note records the
  decisions.
- three.js lesson: a point with `sizeAttenuation` is `size * (viewHeightPx / 2) / cameraDistance` pixels
  wide, with no field-of-view term, so a "0.05 world units" size draws about 1.6 px. Size from a target
  pixel width instead. A camera at distance 7.5 with fov 30 keeps a rotated unit cube in view.

### 3.8 Accessibility floor
Keyboard focus ring visible (2 px, thicker under high contrast); 44 px targets on coarse pointers;
`prefers-reduced-motion`; `role="img"` plus a text description for any chart or 3D view; polite live
regions for run progress; skip link; one h1 per screen; never colour alone.

## 4. Naming and logo findings

- Chosen (for v2): **Doubly**. Rationale: says the promise ("we check every answer twice") in one plain
  word; starts with D so the logo letter works. A quick web search found no product by that name; nearest
  were Double (assistant app), Dubly.AI (video translation), doubleAI (reasoning models). This was not
  trademark clearance. v1 keeps its current name.
- Rejected after search: Datum (Datumo, Datuum.ai, Datums, DatumFuse.AI are all in this space; DatumFuse
  pitches spreadsheets to insights), Plumb (an AI platform for non-technical users), Assay (Assaytech and
  assay-analysis products), Second Look (SecondLook listed as an analytics competitor).
- Logo concept: a solid "D" with a ringed point cut into the bowl (the "checked data point"), ink only so
  it works in Day and Night, readable at 16 px. A dotted D variant was expressive but fragile below about
  48 px. Two other concepts (an interlocked disc and ring; a small dotted D) were dropped: one looked like
  an eclipse, the other read as a "P". The files are in the stash.

## 5. Streamlit versus alternatives (the platform discussion)

- Most of the P0 problems in the audit are layout, hierarchy and copy, so they would follow the product to
  any framework. The platform matters for two things: live 3D updates, and the amount of CSS that fights
  Streamlit (74 `!important`, 31 `data-testid` hooks, an iframe hack for the landing, a hidden-button
  handshake, JS injected through a zero-height iframe).
- Options discussed: stay in Streamlit and fix hierarchy/copy (cheapest, keeps one process); a full-page
  React component inside Streamlit (like the landing already does); a separate FastAPI + React app linked
  from each UI; NiceGUI or Reflex (Python-only, no rerun model, smaller ecosystem); Gradio, Dash,
  Chainlit (poor fit). The user chose FastAPI + React and then rejected the result, so the open question
  is now "improve the current UI" (this file, section 8).
- The seam that made a separate front end cheap: `ui/run.py` (`ActiveRun`, `RunSpec`, `RunSnapshot`,
  `RunOutcome`) has no Streamlit in it. `app.py` still holds the run-building logic inline
  (about lines 957 to 1073) and sets governance limits through `os.environ`, which is process-global and
  must not be used on a shared server.

## 6. Defects found in the current UI (full list in `UI_DESIGN_AUDIT.md`)

Ranked by effect: workspace repeats the landing headline (W1); sample button at the bottom (W2); sidebar is
an engineer's panel and the no-AI path is hidden (S1, S2); an environment API key does not enable Run
(S2b: `has_key` ignores `_env_key`, `app.py:835`); the 3D plate does not show progress (R1); a failed run
never shows its reason (R2: `analysis_error` is written, never rendered); the verdict is under-designed
and contradicts the card count (A1, A9); the primary-finding full-width rule is dead because the CSS only
defines `.bento-card.full-width` (A3); `--ease-in-out` and `--mono` are undefined for the app page (Y3);
no type scale (Y1); inline styles bypass tokens (Y2).

## 7. Data-truth fixes that were made (they are in the stash and are not UI-specific)

These changed engine behaviour that the current UI also reads. They were written but **never run**; the
unit tests that cover them were written but not executed.
- `run_view.finding_state` and `Verdict.not_checked`: the three-way verdict (section 3.1).
- Verdict banner copy in `ui/tabs/answers_tab.py` ("N of M checked findings held up. 1 finding was not
  checked."), the same rule in `src/core/html_report.py`, and a `.run-banner.ok` style for a clean pass.
- `plain_language.format_number` and use of it in finding text (counts with separators, measures to two
  decimals, correlations to three, p-values as "p<0.001"); unquoted column names in finding headlines.
- "8,991 of 9,471 rows (rows with a gap are left out)" check label, fed by `n_table_rows` from the
  controller.
- `src/core/shape_preview.py` (a deterministic 3-axis PCA projection of a sample of rows).
- `ui/__init__.py` loads its names lazily so importing `ui.run` does not import Streamlit.
Known leftovers: raw statistics still print in headlines in `variable_methods.py`, `regression.py`,
`equity.py`, `ml_evaluate.py`, `financial_analysis.py`; about a handful of quoted non-column values were
left on purpose.

## 8. How to apply this to the current UI (a suggested order)

1. Workspace first screen: remove the repeated headline, put upload and sample side by side, move the
   cinematic toggle out of the first screen (W1, W2, W7).
2. Sidebar: one "Connect an AI" block, a visible with/without-AI choice, everything else under Advanced;
   target column as a dropdown; fix the env-key `has_key` bug (S1, S2, S2b, W4).
3. Results: make the verdict the dominant element, equal to the card count, success-coloured when clean;
   put the chart with its finding; fix the dead full-width rule; give callouts distinct forms (A1 to A4,
   A9, A13).
4. Run view: show a failed run's reason; stop presenting the plate as live progress or make it update
   (R1, R2).
5. Tokens and type: define the missing variables, introduce a type scale, move inline styles into classes,
   remove hover effects from non-interactive cards, honour reduced-motion in a finer way (Y1 to Y3, A5, Y10).
6. Copy and icons: apply section 3.5; one icon language (D1).
   Restore the stash first if the data-truth fixes (section 7) are wanted.

## 9. Process lessons

- Writing a typed contract first (`api/schemas.py`, mirrored TypeScript types) let several agents work in
  parallel without clashing. Disjoint file claims plus a ledger (`GEMINI.md`) worked.
- Cross-package assumptions are where parallel work breaks: a verdict rule that disagreed with the engine,
  a theme attribute named `dark` not `night`, `ui/__init__.py` importing Streamlit, JSON responses that
  reject `NaN`. Each was caught only by reading, not running.
- Do not build a whole UI before anyone sees a slice of it. The user rejected the result without ever
  seeing it run; a clickable single screen would have settled taste early.
- The committed reference screenshots in `design-references/screenshots/step3/` are mislabeled (one blank,
  one the landing under the wrong name, a "night" shot showing day). Re-shoot before relying on them.
- npm facts if v2 is ever resumed: typescript-eslint 8.71 supports TypeScript below 6.1 and jsx-a11y
  supports ESLint up to 9; react 19.3 satisfies the peer ranges of `@react-three/fiber` 9 and drei 10.
