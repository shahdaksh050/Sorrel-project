# Design System

This file documents the frontend design constraints, tokens and aesthetic rules for the `agentic-data-analysis` application. The product's visible name is **Sorrel**, the working name of DSA Agent; package names, modules, widget keys and `DSA_*` environment variables keep their old names.

## 1. Design Direction: "Sorrel"

Warm unbleached paper, near-black ink and one forest-green pen, with hairline rules, 4 px corners, an amber for "look twice" and a brick for "this may not hold". It was designed as a prototype (`docs/prototypes/`, see `SORREL_PORT_SPEC.md` and `WORKSPACE_PORT_NOTES.md`) and ported into the landing page, the Streamlit workspace, the HTML report, the charts and the 3D scenes. One palette module drives all of them.

### Principles:
- **Answers First:** The interface is built for non-experts (shopkeepers, teachers, coaches). Clarity and trust matter more than flash. A plain sentence leads; exact figures stay available in a folded "Details for analysts" note.
- **Calm Tone:** No aggressive or panicked styling.
- **WCAG AA:** Contrast ratios are strictly enforced in both themes (`tests/test_contrast.py`, `tests/test_design_tokens.py`).
- **Never colour alone:** every status is a shape and a word (tick, exclamation, circle, dot, dash).
- **Honest copy:** examples are labelled as examples; figures come from data; no unsupported claims (the landing's banned-phrase tests enforce this).
- **Accessibility:** keyboard navigable, visible focus, screen-reader friendly (3D views keep `role="img"` summaries and a text equivalent).

## 2. Design Tokens

The source of truth is `src/core/design_tokens.py` (Day and Night). `ui/styles.py`, the landing page, the report and the 3D scenes read from it. Key names date from the earlier "Ledger" look and were kept so imports keep working; a few no longer match their colour (`pen` is green, `accent` is amber, `stock` is the page paper).

### Core palette
- `--stock`: page background.
- `--sheet`: card surface. `--sheet-alt`: alternate surface (also the warm tone of the header band and the pipeline card).
- `--ink`: primary text. `--ink-2`: body text on cards. `--graphite`: secondary text. `--ink-4`: quiet or disabled, never text that must be read.
- `--pen`: the forest-green primary accent, used for fills, borders and marks. `--pen-hover`: its deeper hover state.
- `--accent`: amber, the secondary accent: "look twice", alerts, errors, skipped and stopped steps.
- `--risk`: brick. **Rule:** reserved to mean "this number or finding may not hold". Never a generic error or a destructive action.
- `--positive`: green for success and "held up".
- `--rule`: hairline. `--rule-faint`: softer divider.
- `--accent-soft`: tinted surface behind accent text. `--accent-ink`: text on a `--pen` fill.
- `--bg-deep`: the deep forest band. `--paper-elevated`: a raised surface. `--margin`, `--code-bg`: decorative and code backgrounds.

### Text roles (AA-safe)
Raw `--pen` and `--risk` are fills: in Night mode `--pen` on a card is only about 2.5:1 and `--risk` about 3.9:1. When the colour is a glyph or a word, use:
- `--accent-text`: the accent as text (about 10:1 on a Day card, 8:1 on a Night card).
- `--danger-text`: "may not hold" as text.
Day values equal the raw tokens; Night values are lighter.

### Derived tokens (in `ui/styles.py` only)
- `--rule-strong`: control borders, at least 3:1 against the page and cards.
- `--risk-text`: `var(--danger-text)` in both modes.
- Motion tokens: `--ease-out`, `--dur-fast` (150 ms), `--dur-base` (250 ms), `--dur-slow` (reserved; not used for reveals).

## 3. Colour use

Neutral surfaces carry the page; colour is used sparingly and always means something.
- **Green (`--pen`, `--positive`):** the primary accent, "held up", finished steps, the active tab, the small square that leads each section heading, figures in gauges.
- **Amber (`--accent`):** look twice: flagged gauges, alerts and errors, a skipped or stopped step, data-quality warnings.
- **Brick (`--risk`, `--danger-text`):** only "this may not hold": a finding that needs more data, a refuted idea, numbers that could not be traced, an overfitting warning.
- **Tints:** a soft green behind the summary, a warm band behind the header, readout and steps. Chips (verdict, status) use a glyph, a word and a faint tint of their own colour.
- **Charts:** a six-colour categorical range for each theme, validated for colour-blind separation (every pair, normal vision and the three colour-blind modes) and at least 3:1 against the chart surface. Slot 1 is the green pen family, no second green, and `--risk` never appears in a series (`src/core/chart_theme.py`).

## 4. Gradients

Gradients are **forbidden** in the workspace: no glow, spotlight, conic border or pointer-following effect on any surface, and no hover effect on anything that cannot be clicked.

**Exception:** the landing page may draw its hairline coordinate grids with gradients and may use one faint, static ambient tint derived from existing tokens, never behind running text beyond a faint tint.

## 5. Reduced Motion & Accessibility

- `@media (prefers-reduced-motion: reduce)` must be respected universally.
- All animations (hero fade, drifting spotlights, border sweeps) must disable their keyframes or skip motion entirely when reduced motion is preferred.
- **Workspace motion is reveals only** (amended 2026-10-04). Expanding and collapsing a disclosure, and a short fade or settle when a tab or a status changes, are allowed: 250 ms or less, eased, never looping, and only inside `@media (prefers-reduced-motion: no-preference)`. Still forbidden: hover lift, glow, spotlight, gradient, pulse, bounce, and anything that moves without a user action or a state change. `tests/test_presentation_cards.py` enforces this.
- Assistive technologies must have access to the final text state immediately (visually hidden `aria-label` or `sr-only` text blocks). Live progress announces one polite sentence per stage change, not the whole list.
- Also honoured: `prefers-contrast: more`, `prefers-reduced-transparency`, and coarse pointers (44 px targets).

## 6. Typography

Three self-hosted families (SIL OFL, variable woff2, Latin subset) under `static/vendor/fonts/` (app, 3D scenes), `static/fonts/` (the HTML report embeds from here) and `ui/landing_component/fonts/` (landing):
- **Geist**: the interface and headings (600 for headings).
- **Geist Mono**: figures, labels, status words, file and column names.
- **Newsreader** (normal and italic): the wordmark and the one italic accent word in a headline, such as the file name in "Results for *file.csv*".

One scale in `ui/styles.py`: 13, 16, 20, 25, 31, 40, 56 px as `--text-xs` to `--text-4xl` in `rem`, with `--text-sm` (15 px) for dense UI text. Metrics use tabular figures. The stylesheet file names (`ledger-fonts.css`, `landing-fonts.css`) are path contracts that tests and code pin; they keep their names.

## 7. Workspace structure

- **Header band:** on a warm band, the title (with the file name as the italic accent), the New analysis and 3D-view buttons, and a three-cell readout (State, File, Analysis).
- **Steps and plate:** the seven steps as a vertical list (a numbered dot, the step's name, a status word with a shape; a hairline rail runs down through the dots and is filled behind finished and skipped steps), with the 3D plate beside it. The layout is the same during a run and after it; both are drawn from the live stage log, and a change of stage redraws them (the plate carries on, it does not replay its entrance). There is no "how it's working" panel on the home page; the team cards live in Details, under "Step-by-step record".
- **Results:** four tabs with an underline for the active one: Answers (summary, "N of M checked findings held up", ranked findings with evidence, what to be careful about, what to do, at a glance), Charts (evenly framed pairs, one height per panel type, one evidence line each), Details (an audit trail: how we got here, then the analyses that ran for this data, then how the file was read and repaired and the technical checks), Downloads (an artifact shelf with type, purpose, size, state, and the real report previewed on request in the app's theme).
- **Settings** stay in the sidebar, each with a safe default.
- No 3D in Answers or Charts; the answer, evidence, caveat and next action are readable without any spatial interaction.

## 8. Embedded views and exports

Native HTML/CSS is the default for progress, findings and every other element; nothing here is needed to upload, run, read a finding or export.
- The **3D pipeline plate** and the optional **cinematic walkthrough** are iframe documents built from the tokens. Three.js and GSAP load from the app's own files first (a CDN is only a fallback if that fails); data is written with `textContent` or escaped before `innerHTML`; a text equivalent stays on the page.
- The **landing page** is a Streamlit component (`ui/landing_component/index.html`): a classic-script bridge returns `{enter, theme}` and works without the 3D module, which is an ES module on the vendored r160 three.js.
- The **HTML report** is one offline file: its fonts and chart libraries are embedded when the bundled files are present (a CDN tag set is the fallback if one is missing); it follows the viewer's colour scheme when opened on its own, and the app's preview forces the app's own Day/Night theme (`retheme_report_html`).
