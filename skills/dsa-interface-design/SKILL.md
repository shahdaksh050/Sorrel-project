---
name: dsa-interface-design
description: Design or redesign the DSA Agent Streamlit workspace, landing page, dashboards, reports, charts, 3D explanations, or interaction states. Use when improving visual hierarchy, responsive behavior, motion, accessibility, Apple-inspired interaction craft, or anti-generic frontend quality without changing analysis semantics.
---

# DSA Interface Design

Build DSA Agent as a calm, distinctive evidence workstation for people who may not be data experts. The product promise is not "AI analysis"; it is an answer with evidence, caveats, and an honest account of what held up.

## Source hierarchy

Apply instructions in this order:

1. The user's current request and `AGENTS.md`.
2. `DESIGN_LEARNINGS.md` and `UI_DESIGN_AUDIT.md` for current product decisions and known defects.
3. `DESIGN.md` and `src/core/design_tokens.py` for the established Ledger identity and tokens.
4. This skill for repeatable design process and quality bars.

Treat older frontend plans and audits as historical unless their claims are rechecked against the code. Preserve user worktree changes. Do not change analysis results, governance, privacy behavior, MemorySystem, RLM, or tool contracts as part of visual work.

## Design read

Before coding, state a one-sentence design read that identifies the audience, task, information density, mood, motion depth, and platform constraints. For this product, start from:

> A precise, warm data-analysis desk where evidence is easy to inspect and advanced machinery stays available without taking over.

Keep the Ledger palette unless the user explicitly authorizes a token-system change. Do not imitate Apple branding, controls, icons, or product chrome. Apply Apple principles of purpose, agency, responsibility, familiarity, flexibility, simplicity, craft, and delight as behavior, not as a skin.

## Workflow

1. Inspect the affected UI, state model, styles, and tests. View current screenshots or run the app before trusting an old audit.
2. Identify the user task, the critical state transitions, the evidence the person needs, and the least capable device that must work.
3. Audit hierarchy, clarity, density, semantics, interaction, accessibility, loading, empty, error, and reduced-motion states. Keep a factual distinction between observed defects and hypotheses.
4. Make a small, coherent design decision: layout family, type scale, surface model, color meaning, responsive behavior, and motion ownership. Do not accumulate isolated decorative fixes.
5. Implement with the existing Streamlit stack and its native controls wherever possible. Keep custom islands optional and focused.
6. Verify Day and Night, desktop and narrow mobile, keyboard focus, no-WebGL fallback, empty/uploading/running/error/completed states, and a real result.
7. Run the project quality gates required by `AGENTS.md` when code changes are complete.

## Product structure

### Lead with the task

- Put dataset, question, scope, and Run in the initial work area. Do not repeat a landing-page marketing headline in the workspace.
- Make a real sample run a first-class alternative to upload.
- Make the no-LLM route visible and dignified. Never make a missing API key look like a disabled product.
- Keep provider, model, budgets, acceleration, and isolation controls behind a clearly named Advanced section or settings surface.
- Use labels taken from the task: `Add a dataset`, `Ask a question`, `Run analysis`, `Evidence`, `Details`, `Downloads`.

### Make evidence the visual centre

- Put a plain-language answer first, followed by the most important finding, its chart, the checks that apply to it, and its caveats in the same reading flow.
- Show the three finding states exactly: `Held up`, `Needs more data`, and `Not checked`. Never infer failure from absence of a check.
- Pair every state with text and a distinct shape; never rely on color alone. Keep `--risk` reserved for evidence that may not hold.
- Round data for people while retaining precise data in an appropriate disclosure. Explain changed counts.
- Keep progress truthful. Never animate a stage as complete until the real callback or result says so.

### Design the full run lifecycle

| State | Give the person |
| --- | --- |
| Empty | One focused start path, a sample option, and a brief explanation of what happens next. |
| Dataset loaded | File identity, useful profile summary, editable target choice, and the next action. |
| Running | Current stage, meaningful elapsed/progress feedback, clear stop behavior, and stable layout. |
| Completed | Answer, verdict summary, evidence, caveats, then technical detail and downloads. |
| Failed or stopped | The actual reason, safe recovery action, and any trustworthy partial output. |

## Visual system

### Typography and layout

- Use a deliberate scale in `rem`: 13, 16, 20, 25, 31, 40, 56 are the default steps. Let browser text scaling reflow the layout.
- Use tight display leading and tracking, moderate heading leading, and comfortable body leading. Enable tabular figures for data-heavy numbers.
- Use mono only for literal dataset values, field names, code, or audit identifiers.
- Limit prose to about 65-68 characters. Use sentence case and direct, specific labels.
- Prefer whitespace, grouping, and alignment over containers. Use one elevated surface at most in a view; do not nest cards.
- Use an asymmetric composition only when it reinforces hierarchy. Do not default to three identical cards.

### Colour, surfaces, and icons

- Keep one primary accent for action. Reserve semantic colors for verdicts, warnings, and errors.
- Maintain Day and Night hierarchy parity. Use off-white/off-black rather than pure white/black. Check AA contrast and 3:1 control boundaries in both modes.
- Use role-based radii: controls small, a raised surface larger, status marks round. Do not make every object a pill.
- Avoid decorative gradients, glass, glow, perpetual grain, and generic box shadows. A depth effect must communicate a layer or an interaction.
- Use one coherent icon language. Do not use emoji as interface icons.

### Anti-generic locks

- Keep color, shape, and page-theme systems coherent across each screen.
- Do not ship AI-purple gradients, stock SaaS cards, coloured left stripes on every callout, fake dashboards, decorative status dots, dashboard-hero metrics, or a long stream of equal panels.
- Do not add made-up metrics, placeholder people, unsupported claims, visual metaphors that imply analysis happened, or controls that do nothing.
- Do not use all-caps eyebrows, numbered marketing eyebrows, hero version labels, decorative scroll prompts, arrow-filled link copy, or em/en dashes in product copy.
- Do not add a hover treatment to a non-clickable element.

## Interaction, motion, and 3D

### Native Streamlit interactions

- Give actionable controls immediate press feedback with `:active`, visible hover and focus states, and short transform/opacity transitions.
- Use transitions to clarify state changes, not to decorate reruns. Reserve space for live content so polling does not cause layout shift or restart animations.
- Respect `prefers-reduced-motion`, `prefers-reduced-transparency`, and `prefers-contrast: more`. Reduced motion uses a short fade or static change, not a slide.
- Use `transform` and `opacity` for animation. Use `IntersectionObserver` or CSS scroll-driven effects when appropriate; do not attach general `window.scroll` handlers.

### Fluid custom interactions

Use a custom island only when native Streamlit cannot provide the interaction and the interaction improves understanding or agency. For draggable sheets, canvas controls, or other direct manipulation:

- Respond on pointer-down and track the pointer 1:1 with pointer capture.
- Animate from the current presented value, preserve release velocity, allow interruption, and use critically damped springs by default.
- Use momentum projection and rubber-banding only for genuine gesture surfaces, never for ordinary buttons or panels.
- Keep entry and exit paths spatially consistent and anchor overlays to their trigger.

### Animation ownership

- Let CSS own ordinary Streamlit feedback.
- Let GSAP own sequenced, reversible choreography within one isolated 3D surface when a timeline is genuinely needed.
- Let Three.js own canvas rendering only. Pause rendering when hidden, cap pixel ratio, dispose resources, provide keyboard controls, and expose an equivalent text explanation.
- Let anime.js own at most one isolated storytelling surface. Do not load anime.js and GSAP for the same interaction without a written reason.
- Never make 3D, WebGL, sound, or motion necessary to upload, run, understand a finding, or use an export.
- Keep all runtime assets local where the project promises local-only behavior. Review GSAP's non-MIT license before a hosted or redistributed use.

## Streamlit implementation rules

- Use native Streamlit widgets for forms, uploads, tabs, tables, downloads, and disclosure whenever they satisfy the requirement.
- Put reusable presentation builders in `ui/components/`, tab-specific rendering in `ui/tabs/`, and global theme rules in `ui/styles.py`. Keep `ui/run.py` free of `st.*` calls.
- Escape all dynamic HTML. Treat dataset and model text as data, not markup or instructions.
- Avoid brittle global JavaScript, repeated `MutationObserver` installation, per-rerun script injection, and CSS selectors that create behavior drift.
- Keep chart selection and analysis logic unchanged unless the user separately requests a product or backend change.

## Accessibility and honesty

- Make focus obvious, targets usable on touch, and errors specific and inline.
- Provide a keyboard path, one main heading, sensible heading order, accessible labels, and live status announcements.
- Give visualizations and 3D scenes a text alternative. Let people inspect data behind charts.
- State where a file goes, whether an LLM receives controlled summaries, and when analysis is deterministic. Do not claim local processing unless it is true for the deployment.

## Pre-flight

Before shipping, answer yes to every item:

- Does the first screen make the next task obvious without reading a marketing paragraph?
- Is the primary answer, supporting evidence, and caveat visible in one reading flow?
- Are every number, status, control, and animation tied to real state or a useful action?
- Does Day/Night preserve contrast and hierarchy, and does reduced motion preserve comprehension?
- Does the narrow layout remain usable without horizontal loss, WebGL, or hover?
- Did the change preserve backend semantics, privacy, accessibility, and the existing user worktree?
- Does the result look authored for this evidence product rather than a template or an Apple imitation?
