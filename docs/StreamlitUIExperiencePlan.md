# Streamlit UI experience plan

Status: proposed implementation plan. This document turns the repository's current screenshots, audits, and design references into a controlled plan for improving the existing Streamlit application. It does not authorize changes to analysis logic, the RLM engine, MemorySystem, or the tool layer.

## 1. Design read and scope

**Design read:** a desktop-first, evidence-first data-analysis workspace for non-specialists and examiners, with moderate information density, calm confidence, purposeful motion, and a useful but optional spatial explanation.

This is a redesign-preserve project. Preserve the app's current workflow, Ledger identity, analysis semantics, public-facing honesty, and four result destinations. Recompose the workspace around the task and the evidence rather than layering another visual system on top.

| Surface | Variance | Motion | Density | Reason |
| --- | ---: | ---: | ---: | --- |
| Workspace | 4 | 3 | 6 | A daily analysis tool benefits from clear hierarchy and controlled density, not theatrical asymmetry. |
| Results | 5 | 3 | 7 | Findings and evidence need a stronger focal layout with compact supporting detail. |
| Landing | 6 | 4 | 3 | A short product introduction can use an asymmetric composition and one restrained 3D moment. |
| 3D pipeline | 5 | 5 | 3 | Spatial interaction earns a place only when it explains the seven-stage process. |

The values above adapt TasteSkill's dials to a product UI. Its own instructions explicitly exclude dense dashboards and multi-step product UI, so do not import its landing-page patterns such as sticky stacks, horizontal scroll hijacks, fake dashboard previews, testimonial layouts, or decorative marquees into the workspace.

## 2. Current visual baseline

Treat `design-references/screenshots/step3/` as the visual baseline, not as a correct state-labelled test suite. The captured files have material labelling and capture defects:

- `01_landing_*` includes blank or near-blank frames rather than a reliable landing state.
- `02_empty_*` contains the working landing/workspace views despite its name.
- `03_uploaded_*` and `04_midrun_*` show useful portions of the upload and input journey, but are cropped and do not reliably demonstrate Night mode or a true running state.

The screenshots and the current audits agree on the primary product problems:

1. The workspace repeats a large marketing promise after the person has entered the product, pushing the task lower in the viewport.
2. The permanent sidebar exposes implementation settings before the primary task.
3. The long rounded, bordered input surface feels like a nested form card rather than an analysis desk.
4. The 3D plate competes with upload and question entry, but its state is not a trustworthy live-progress indicator.
5. Results need one evidence flow: finding, chart, verification state, caveat, and next action together.
6. Some reference files are blank or stale, so visual changes cannot be judged against filenames alone.

Re-capture the following before changing layout: landing Day/Night desktop and mobile; workspace empty Day/Night desktop and mobile; uploaded Day/Night desktop and mobile; real running Day/Night desktop and mobile; completed Answers, Charts, Details, and Downloads. Record browser size, theme, file, objective, and whether WebGL succeeded.

## 3. Non-negotiable experience principles

### Purpose and honesty

- Make the primary promise visible in the result, not in marketing language: answer the question, show the evidence, and state what remains uncertain.
- Keep `Held up`, `Needs more data`, and `Not checked` as distinct states. A missing check is not a failed check.
- Reserve `--risk` for a finding or number that may not hold. Pair every state with a word and shape, not colour alone.
- Do not invent metrics, progress, sample findings, confidence, or model claims. A visual metaphor must not imply that an analysis actually occurred.
- State the data path and LLM use accurately. Retain no-LLM analysis as a first-class route.

### Simplicity without emptiness

- Give the opening screen one dominant task: add data, ask a question, run analysis.
- Keep model, provider, GPU, budgets, isolation, and other engineering controls available but one level deeper.
- Replace decorative containers with grouping, spacing, and clear section order. Use elevation only to communicate a real layer.
- Make the sample run adjacent to file upload rather than a low-priority fallback.

### Craft and accessibility

- Use one semantic token system for Day and Night. Preserve hierarchy and AA contrast in both.
- Use the current Ledger palette because it is an established identity, not a generic warm-craft default. Do not introduce a competing blue, purple, or glass palette.
- Give buttons immediate pressed feedback, obvious keyboard focus, readable label contrast, and stable layout while the app reruns.
- Support reduced motion, increased contrast, reduced transparency, touch targets, text alternatives for charts/3D, and a usable no-WebGL fallback.

## 4. Target information architecture

### Workspace shell

Replace the current hero-plus-form impression with an analysis desk.

```text
Top bar: DSA Agent | current dataset/run state | Day/Night | Settings

Main task area
  Dataset row: Add data / Run sample | file identity and profile when loaded
  Question row: objective | target-column choice | analysis mode
  Run row: primary Run analysis | concise run expectation

Progress or completed answer replaces the task area's supporting panel
```

Use a compact top bar for wayfinding. Do not rebuild it as an Apple control strip or add decorative nav labels. On a narrow viewport, stack the dataset, question, and run controls in that order; keep the primary button full-width.

### Settings

Keep the Streamlit sidebar as a collapsed configuration space, or make its first state a single `Settings` disclosure. Organise it into four plain-language groups:

1. `Analysis mode`: with AI summary / without AI, machine learning, target-column choice.
2. `Connect an AI`: provider, key, model, reasoning mode. Explain billing and data handling where relevant.
3. `Advanced analysis`: tuning and execution budgets.
4. `Safety`: operator-controlled settings displayed read-only where the hosted app requires that.

Do not make API key entry, provider selection, or CUDA information visually larger than upload and question entry.

### Completed-result layout

Keep the existing Answers, Charts, Details, and Downloads destinations, but make Answers the evidence home.

```text
Answer to your question
  verdict summary: 3 held up | 1 needs more data | 2 not checked

Primary finding                         Evidence inspector
headline + plain-language impact        selected chart
why it matters                          checks that ran
                                       caveats and source/tool link

Remaining findings: compact ranked list with status marks
Recommendations: only when backed by results
```

On wide screens, use a 5/7 or 7/5 grid that connects the selected finding with its evidence. On narrow screens, place the chart and checks immediately below the selected finding. Do not use a three-equal-card grid or a bento grid simply because it looks modern.

## 5. Visual language

### Typography

- Replace the rounded display treatment where it makes the product read as playful rather than precise. Test a self-hosted system-first stack or the already researched Bricolage Grotesque/Public Sans/IBM Plex Mono option in a browser before committing to it.
- Use a deliberate scale: 13, 16, 20, 25, 31, 40, 56 px expressed in `rem`.
- Tighten display tracking/leading; use moderate headings and comfortable body leading. Keep body width under about 68 characters.
- Use tabular figures for metrics. Limit mono to literal dataset values, column names, hashes, and audit IDs.
- Use sentence case. Remove decorative eyebrows, section numbering, hero version labels, decorative scroll cues, and emoji interface icons.

### Surfaces and colour

- Retain the buff-green stock, faint blue rules, terracotta action pen, and semantic positive/risk colours. Document which tokens may appear as text, controls, chart marks, and decorative tints.
- Use one action accent per surface. Semantic status colours may appear only with their labels and marks.
- Define a role-based radius scale: controls 6-8 px, one raised panel 12 px, status marks circular. Reserve full pills for compact filters or the active tab indicator.
- Remove conic-gradient hover borders, pointer glows on static cards, and broad shadows. Keep a small tinted shadow only when a floating layer needs separation.
- Do not use glass or gradients in the workspace. If the landing later uses an ambient effect, it needs a solid fallback and must not sit behind running text.

### Charts and tables

- Keep chart selection, aggregation, and validation in `src/core/` unchanged. UI work only improves framing, labels, captions, and accessible access to the data behind the chart.
- Give every chart a direct relationship to a finding. Do not show a gallery of unexplained charts first.
- Use an adjacent table/disclosure alternative and a concise title that says what changed or differs.
- Apply the same Day/Night tokens to chart text, rules, annotations, and selected state. Do not add glow or 3D effects to analytical charts.

## 6. State-by-state implementation plan

### Phase 0: establish visual truth

**Goal:** make later decisions measurable.

- Run the current app with a real sample CSV and capture the state matrix named in section 2.
- Verify keyboard navigation, 200% text zoom, browser zoom, reduced motion, and no-WebGL fallback.
- Create a small visual-check script or documented repeatable capture process. Do not trust the existing screenshot labels.
- Record layout shift while a run updates, initial load time, and any external runtime network request.

**Likely files:** `scripts/capture_screenshots.py`, existing UI smoke tests, screenshot directory. Do not change the application yet.

### Phase 1: stabilise the system before recomposition

**Goal:** give every subsequent surface the same visual rules.

- Audit `ui/styles.py` for undefined CSS variables, inline-style exceptions, repeated selector overrides, and hover effects on static content.
- Create documented semantic aliases in `ui/styles.py` only when the existing token source lacks a presentation-safe value. Do not edit `src/core/design_tokens.py` without explicit approval.
- Define type, spacing, radius, elevation, focus, state, and motion tokens. Use the same controls in Day and Night.
- Replace unbounded `transition: all` rules with property-specific transitions. Make `prefers-reduced-motion`, `prefers-reduced-transparency`, and `prefers-contrast: more` intentional states.
- Add visual tests for contrast, token parity, focus, reduced motion, and no restarting animations on idle reruns.

**Likely files:** `ui/styles.py`, `.streamlit/config.toml`, `tests/test_theme_sync.py`, new focused UI style tests.

### Phase 2: task-first workspace

**Goal:** make the next useful action visible in the initial viewport.

- Remove the repeated workspace hero. Retain a compact contextual heading only after a file is loaded, for example the dataset name and what is ready.
- Rebuild the first screen as a three-part task flow: dataset, question, run. Place real sample-run access beside upload.
- Move the cinematic trigger out of the initial task path. Place it in Details or Downloads, or make it a quiet optional link after the input flow.
- Turn the file uploader into a clear dropzone with file type/size guidance, loaded-file identity, validation result, and recovery action.
- Change target-column entry from manual text to a data-derived select once a dataset is loaded, keeping an automatic choice as the default.

**Likely files:** `app.py`, `ui/styles.py`, `ui/components/cards.py`, any upload helper tests.

### Phase 3: progressive settings and trustworthy run states

**Goal:** preserve expert control without making configuration the product.

- Restructure the sidebar using the four groups in section 4. Keep tooltips for unfamiliar technical terms.
- Surface the with-AI/without-AI decision in the task area. Do not silently disable Run when no key is present.
- Ensure environment-supplied keys, local-only settings, and hosted operator controls communicate their real availability.
- Add a dedicated failed/stopped state that renders the recorded reason and safe recovery action instead of leaving a blank workspace.
- While analysis runs, show a native text-first stage timeline driven only by real callbacks. Reserve height and animate a new stage once, not on every polling tick.

**Dependency:** any worker/stop/polling or run-view contract work must be treated as a separate, tested backend/UI-foundation change. Do not smuggle it into CSS work.

**Likely files:** `app.py`, `ui/run.py`, `ui/components/provisional.py`, `ui/components/cards.py`, relevant controller callback tests.

### Phase 4: answers as an evidence inspector

**Goal:** make proof more legible than presentation.

- Make the run-level verdict summary match the exact number and state of listed findings.
- Render a ranked finding list with stable IDs, text-and-shape verdict marks, and one selected primary finding.
- Render its chart, check row, caveats, evidence source, and plain-language implication in the same visual region.
- Distinguish recommendations from observations; suppress a recommendation that cannot point to evidence.
- Remove generic card treatments. Use a strong primary finding surface only when it creates a useful focus layer; keep remaining findings as a calm list or low-elevation rows.
- Round numbers for reading but preserve original values in an accessible details disclosure.

**Likely files:** `ui/tabs/answers_tab.py`, `ui/components/cards.py`, `ui/tabs/charts_tab.py`, `src/core/run_view.py` only if a tested read-only display field is genuinely absent.

### Phase 5: charts, details, downloads, and report continuity

**Goal:** make secondary views follow the same evidence hierarchy.

- Start Charts with a short orientation and link each chart to the relevant finding/status. Preserve an overview only when it answers a distinct question.
- Make Details a readable audit trail: data profile, method, tool outcome, degradation, governance, and raw trace in increasing technical depth.
- Make Downloads a clean handoff surface grouped by report, data/visual artefacts, model, and optional cinematic export. Do not use decorative download cards.
- Restyle the HTML report only after the workspace hierarchy is stable. Keep it self-contained, print-ready, offline-safe, content-identical, and visually compatible with the workspace.

**Likely files:** `ui/tabs/charts_tab.py`, `ui/tabs/details_tab.py`, `ui/tabs/downloads_tab.py`, `src/core/html_report.py`.

### Phase 6: disciplined motion and 3D

**Goal:** use movement only to explain state, hierarchy, or direct manipulation.

- Keep native Streamlit motion to press feedback, hover/focus, selection changes, and short opacity/transform transitions.
- Remove page-global or repeated `MutationObserver` animation injection. Never install a listener on every rerun.
- Make Three.js the owner of rendering within its iframe/canvas. Pause or lower work when hidden, cap device pixel ratio, dispose resources, and keep a text timeline with the same information.
- Let GSAP own only sequenced, reversible stage/camera choreography inside one isolated 3D surface. Do not use GSAP for ordinary controls or scroll hijacking.
- Let anime.js own only a single isolated landing/storytelling surface, if it still has a clearly different job. Remove its overlap with GSAP in the pipeline after a verified replacement.
- Eliminate runtime CDN dependencies before presenting `LOCAL_ONLY` as a browser-wide claim. Resolve the fullPage.js licensing/runtime issue before making the cinematic export a core demo feature.
- Do not add drag, springs, velocity, momentum, or rubber-banding unless the user directly manipulates a custom surface. For such a surface, use pointer capture, live-value interruption, and a reduced-motion static/fade fallback.

**Likely files:** `ui/animations.py`, `ui/pipeline_3d.py`, `ui/assets/pipeline_3d.{html,js}`, `ui/cinematic_3d.py`, `ui/assets/cinematic_3d.{html,js}`, `ui/landing_component/index.html`, `THIRD_PARTY_UI.md`.

### Phase 7: responsive, accessibility, and performance pass

**Goal:** let the product retain its meaning beyond the ideal desktop screenshot.

- Test 390 px mobile, 768 px tablet, 1024 px laptop, and wide desktop layouts. Collapse multipane results into one evidence sequence.
- Ensure 44 px touch targets, visible focus, logical tab order, correct headings, live announcements, and nonvisual chart/3D descriptions.
- Verify 200% text zoom, 400% browser zoom, high contrast, reduced motion, and reduced transparency.
- Lazy-load optional 3D/cinematic resources and reserve their layout. Measure LCP, INP, CLS, JavaScript payload, and GPU work.
- Re-capture all states from Phase 0 and compare them against the original baseline with a written regression checklist.

## 7. Motion ownership contract

| Layer | Owner | Allowed work | Forbidden work |
| --- | --- | --- | --- |
| Ordinary Streamlit UI | CSS | Focus, hover, active, selected-state, short opacity/transform transitions | Global DOM scripting, continuous physics, rerun-restarted entrance effects |
| Optional interactive island | GSAP | Reversible, labelled sequences that need a timeline | Scroll hijacking in the workspace, generic card animation |
| 3D canvas | Three.js | Scene/camera/rendering for explanatory spatial state | Essential workflow state, raw data disclosure, unsupported model metaphors |
| Landing-only storytelling | anime.js | One isolated nonessential narrative interaction | Second ownership of pipeline camera/state motion |

All motion must communicate feedback, hierarchy, state, or a direct manipulation. Use `transform` and `opacity`. A reduced-motion mode keeps comprehension with a short fade or static state.

## 8. Acceptance checklist

Do not call a phase complete until all of its relevant checks pass:

- The first workspace viewport answers what to do, how to do it, and what data state exists.
- A person can run the bundled sample without an API key, receive a real result, and understand the no-LLM path.
- Every listed finding is counted exactly once and carries the correct semantic state.
- A chart is available with the finding it supports, and its data has an accessible alternative.
- Day and Night retain contrast, hierarchy, and the same meaning.
- Running/idle reruns do not restart settled animations or shift the layout.
- Reduced motion, no-WebGL, keyboard-only, touch, and narrow-screen paths retain all essential tasks.
- No external runtime asset breaks a local-only statement, and no license issue remains undocumented.
- `ruff check .`, `mypy src/`, `pytest tests/ -v`, the end-to-end sample CSV run, and report output satisfy `AGENTS.md` after code changes.

## 9. Source catalogue

### Project sources

- `AGENTS.md`: architecture boundaries, testing requirements, safety, and protected core changes.
- `README.md`: product promise, seven-stage workflow, current Streamlit surfaces, safety claims.
- `DESIGN.md`: current Ledger identity, tokens, WCAG/accessibility commitments, and 3D fallback rules.
- `DESIGN_LEARNINGS.md`: current UI principles, Apple interaction translation, design decisions, and lessons from the scrapped v2 attempt.
- `UI_DESIGN_AUDIT.md`: the most current screen-level audit, live-run observations, priorities, and screenshot limitations.
- `FRONTEND_AUDIT.md`, `OVERHAUL_PLAN.md`, `docs/FrontendOverhaulPlan.md`, and `docs/FrontendImplementationPlan.md`: historical implementation detail. Revalidate every claim before use.
- `design-references/screenshots/step3/`: current visual baseline. Use only with the capture caveats in section 2.
- `ui/styles.py`, `app.py`, `ui/components/`, `ui/tabs/`, `ui/pipeline_3d.py`, `ui/cinematic_3d.py`: current UI implementation.
- `skills/dsa-interface-design/SKILL.md`: project-local operational design skill created from the references below.

### Skill sources and exact applicability

- [Apple Design reference attachment](C:/Users/daksh/.codex/attachments/f8ce604c-df5a-46e8-95c5-3f7f3aaaaa7a/pasted-text.txt): response on pointer-down, direct manipulation, interruption, springs, velocity handoff, momentum, spatial consistency, accessibility, typography, and the eight Apple principles. Apply its fluid-gesture rules only to genuine custom interactions.
- [TasteSkill implementation](../.agents/skills/design-taste-frontend/SKILL.md): use its brief inference, design read, audit-first redesign protocol, colour/shape/theme locks, dark-mode and performance checks, anti-generic bans, and pre-flight process. Its own section 13 excludes dashboards and multi-step product UI, so do not apply its marketing layouts wholesale.
- [Project DSA interface skill](../skills/dsa-interface-design/SKILL.md): apply the product-specific evidence-first, Streamlit, accessibility, 3D, and motion ownership rules.
- `redesign-existing-projects` skill: use its targeted-upgrade sequence: typography, colour, interaction feedback, layout, components, states, final polish. Preserve working functionality.

### Primary external documentation

- [Apple design principles](https://developer.apple.com/design/human-interface-guidelines/design-principles)
- [Apple HIG foundations](https://developer.apple.com/design/human-interface-guidelines/foundations)
- [Apple: Designing Fluid Interfaces](https://developer.apple.com/videos/play/wwdc2018/803/)
- [Apple: The details of UI typography](https://developer.apple.com/videos/play/wwdc2020/10175/)
- [TasteSkill documentation](https://www.tasteskill.dev/docs)
- [GSAP documentation](https://gsap.com/docs/v3/GSAP/)
- [GSAP licensing/package record](https://www.npmjs.com/package/gsap)
- [Three.js fundamentals](https://threejs.org/manual/pages/fundamentals.html)
- [Three.js animation system](https://threejs.org/manual/pages/animation-system.html)
- [anime.js documentation](https://animejs.com/documentation/)

## 10. Decision gates before implementation

Request explicit approval before any of these choices, because each changes the scope or risk materially:

1. Replacing the current font family or adding new self-hosted fonts.
2. Editing `src/core/design_tokens.py` instead of adding UI-only presentation aliases.
3. Changing the landing's visual identity rather than applying copy and performance fixes.
4. Replacing or relicensing fullPage.js/GSAP, or removing an existing export.
5. Adding a new React/custom-component surface beyond the existing isolated component pattern.
6. Implementing worker threading, Stop, polling, or a new RunView contract as part of the UI work.
