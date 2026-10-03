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

## 3. The Gradient Exception

Gradients are generally **forbidden** on surfaces, buttons, and backgrounds. The UI must remain flat to preserve the Ledger aesthetic.

**Exceptions:** Gradients are allowed *only* for the following ambient effects:
1. **Spotlight Beam** (e.g. Hero background).
2. **Glowing Effect** (e.g. Active step in progress, or around a `--risk` finding).
3. **Hover Border Gradient** (e.g. Focus rings).
4. **Card Spotlight** (e.g. Pointer-following glow).

**Constraints on allowed gradients:**
- Must derive from existing tokens (`--pen`, `--accent`, `--risk`) with low alpha (`color-mix` or `rgba`).
- Never placed behind running text at more than a faint tint (ensuring WCAG AA contrast).
- Never applied to data-dense views (tables or charts).
- Must be **static (no motion)** under `@media (prefers-reduced-motion: reduce)`.

## 4. Reduced Motion & Accessibility

- `@media (prefers-reduced-motion: reduce)` must be respected universally.
- All animations (hero fade, drifting spotlights, border sweeps) must disable their keyframes or skip motion entirely when reduced motion is preferred.
- Assistive technologies must have access to the final text state immediately (e.g., visually hidden `aria-label` or `sr-only` text blocks).

## 5. UI Islands Contract

Rich, isolated React components ("islands") are used for display-only enhancements (e.g., Hero, Progress, Findings).
- Islands receive state as plain text properties from Python. 
- Island rendering must gracefully fail or skip rendering if the built bundle is missing.
- Island sources **must never inject raw HTML** (`dangerouslySetInnerHTML` is forbidden). All LLM output and dataset text is rendered as React text nodes to prevent XSS.
