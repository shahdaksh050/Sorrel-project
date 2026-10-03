# Design System

This file documents the frontend design constraints, tokens, and aesthetic rules for the `agentic-data-analysis` application.

## 1. Design Direction: "Evolve Ledger"

The visual identity is based on the "Ledger" aesthetic: buff-green paper backgrounds, faint-blue rules, terracotta pen, and amber accents. 

### Principles:
- **Answers First:** The interface is built for non-experts (shopkeepers, teachers, coaches). Clarity and trust matter more than flash.
- **Calm Tone:** No aggressive or panicked styling.
- **WCAG AA:** Contrast ratios are strictly enforced (tested in `tests/test_theme_sync.py` and `tests/test_islands.py`).
- **Accessibility:** Keyboard navigable, screen-reader friendly (existing 3D views keep their `role="img"` summaries).

## 2. Design Tokens

The source of truth for the core palette is `src/core/design_tokens.py`.
Tokens used by the frontend are exposed as CSS variables (e.g. `var(--accent)`).

### Core Palette:
- `--stock`: Background surface (buff-green / dark).
- `--sheet`: Secondary surface.
- `--sheet-alt`: Tertiary surface.
- `--ink`: Primary text.
- `--graphite`: Secondary text.
- `--pen`: Terracotta primary accent.
- `--pen-hover`: Terracotta interactive state.
- `--risk`: Red-shifted alert color. **Rule:** `--risk` is strictly reserved to mean "this number/finding may not hold". Never use it for generic errors or destructive actions without context.
- `--accent`: Amber secondary accent.
- `--positive`: Green success metric.
- `--rule`: Faint blue layout borders.
- `--rule-faint`: Softer divider.
- `--margin`: Wine margin line.
- `--code-bg`: Code block background.

### Derived Overlay Tokens (in `ui/styles.py` only):
To maintain contrast safety without mutating the backend `src/core/design_tokens.py`, several tokens are derived at the CSS layer:
- `--rule-strong`: For control borders meeting 3:1 contrast.
- `--glow` / `--spot`: Translucent overlays created via `color-mix()` from `--pen` or `--accent`.
- `--risk-text`: A contrast-safe variant of `--risk` for text in Night mode (the raw `--risk` token yields 4.04:1, so the derived token ensures >= 4.5:1).
- `--lift-lg`: Shadow elevation.
- **Motion tokens:** `--ease-out`, `--dur-fast`, `--dur-base`, `--dur-slow`.

### Known Contrast Limitations:
- The `report.html` standalone export and the 3D pipeline scenes currently inherit the Night `--risk` text contrast issue (4.04:1).

## 3. Gradients

Gradients are **forbidden** in the workspace: no glow, spotlight, conic border or pointer-following effect on any surface, and no hover effect on anything that cannot be clicked (amended 2026-10-03; the earlier "Gradient Exception" list is withdrawn).

**Exception:** the landing page may use one ambient gradient effect, derived from existing tokens at low alpha, with a solid fallback, never behind running text beyond a faint tint, and static under `prefers-reduced-motion`.

## 4. Reduced Motion & Accessibility

- `@media (prefers-reduced-motion: reduce)` must be respected universally.
- All animations (hero fade, drifting spotlights, border sweeps) must disable their keyframes or skip motion entirely when reduced motion is preferred.
- **Workspace motion is reveals only** (amended 2026-10-04). Expanding and collapsing a disclosure, and a short fade or settle when a tab or a status changes, are allowed: 250 ms or less, eased, never looping, and only inside `@media (prefers-reduced-motion: no-preference)`. Still forbidden: hover lift, glow, spotlight, gradient, pulse, bounce, and anything that moves without a user action or a state change. `tests/test_presentation_cards.py` enforces this.
- Assistive technologies must have access to the final text state immediately (e.g., visually hidden `aria-label` or `sr-only` text blocks).

## 5. UI Islands Contract

Native HTML/CSS is the default for progress, findings and every other element. Islands are optional and isolated, and none is needed to upload, run, read a finding or export.

- The **Hero** is the approved React island. It receives state as plain text properties from Python, skips rendering if its bundle is missing, and never injects raw HTML (`dangerouslySetInnerHTML` is forbidden; all LLM and dataset text is rendered as text nodes).
- The **pipeline view** is the second approved island (Streamlit Components v2, not React; amendment pending the Phase 6 spike). It is mounted once, receives stage data from Python on each progress tick, and applies deltas without rebuilding the scene. Three.js owns the single render loop, all assets are vendored (no CDN), data is written with `textContent`, and a text timeline with the same information stays on the page.

## 6. Typography (2026-10-03)

Bricolage Grotesque (headings), Public Sans (text) and IBM Plex Mono (literal column names, values, hashes, audit IDs only), self-hosted under `static/vendor/fonts/`. One scale in `ui/styles.py`: 13, 16, 20, 25, 31, 40, 56 px as `--text-xs` to `--text-4xl` in `rem`, with `--text-sm` (15 px) for dense UI text. Metrics use tabular figures. Baloo 2 and Mukta remain only in the landing page, cinematic export and HTML report until those move over.
