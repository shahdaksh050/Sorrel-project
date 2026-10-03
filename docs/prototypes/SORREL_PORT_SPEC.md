# Sorrel port: shared spec

Every worker on this port reads this file first. It fixes the contracts so parallel work cannot drift.
The prototype is the visual source of truth: `docs/prototypes/verdacert_dsa_preview.html`,
`verdacert_dsa_original_3d.js`, `verdacert_dsa_scroll_fx.js`. Its workspace notes are in
`WORKSPACE_PORT_NOTES.md`. The prototype's static checkers are in `docs/prototypes/checks/`.

## Goal

Keep the original app exactly as it behaves today, and give it the prototype's look, structure and
principles: the Sorrel palette, Geist / Newsreader / Geist Mono type, hairline cards, plain-language
framing with technical figures folded away, the closing-card 3D pipeline track on the landing, and the
prototype's workspace layout. The product is renamed **Sorrel** in everything the user sees.

## Hard rules (every worker)

1. **No behaviour change.** Do not change controller logic, data flow, tool contracts, session-state keys,
   widget keys (`key=...`) or what the analysis computes. Markup and CSS may change.
2. **Keep escaping.** Every interpolated string in HTML stays `html.escape`d. No new `unsafe_allow_html`
   path that interpolates raw data.
3. **Offline.** No CDN hosts, no Google Fonts, no remote scripts, in any shipped file. Fonts and three.js are
   self-hosted. The HTML report stays one standalone offline file.
4. **Accessibility floor.** WCAG AA text contrast in BOTH themes (use the `--accent-text` / `--danger-text`
   tokens for text, never raw `--accent` / `--danger` on tinted surfaces). Visible focus. Every control
   labelled. Shape plus words, never colour alone (tick / exclamation / circle on check rows). Respect
   `prefers-reduced-motion`, `prefers-contrast`, and `prefers-reduced-transparency` as `ui/styles.py` already does.
5. **DESIGN rules for the workspace:** no gradients, glow, blur or spotlight; no hover effect on anything that
   cannot be clicked; `--risk`/`--danger` ONLY means "this finding or number may not hold" (never a generic
   error: use the warning amber for those). No 3D in Answers or Charts.
6. **Honest copy.** Keep the original's honesty rules and tests: the landing's data-path statement, banned
   phrases, "illustration, not your data" labelling. Never invent a figure; numbers shown in a demo or
   example must be labelled as examples or come from data.
7. **Naming.** Visible strings say **Sorrel** (page title and icon, sidebar masthead, hero eyebrow, report
   title and header, footer, landing). Do NOT rename packages, modules, file paths, env vars (`DSA_*`),
   widget keys, or the repo. The footer/about line says Sorrel is the working name of DSA Agent.
8. **Do not commit.** The coordinator commits. List every file you changed and every TEST you changed, with
   the reason, in your final report. Never delete a check to make a test pass: re-pin the same intent to the
   new value. If a test's intent no longer applies, say so and propose the replacement instead of dropping it.
9. **Stay inside your file list.** Files owned by another worker are read-only for you.

## Token mapping (names kept, values changed)

Python keys in `src/core/design_tokens.py` and the CSS variables built from them (`--stock`, `--sheet`,
`--pen` ...) KEEP their names so existing imports and ~dozen consumers keep working. Values change to the
prototype's. Where a name no longer matches its colour (`pen` is now forest green), a comment says so.

| Python key / CSS var | Prototype role | Day | Night |
| :--- | :--- | :--- | :--- |
| `stock` / `--stock` | page background (`--bg`) | `#f7f4ed` | `#141712` |
| `sheet` / `--sheet` | card surface (`--paper`) | `#fffdf7` | `#1c2219` |
| `sheet_alt` | alt surface (`--bg-alt`) | `#efeae0` | `#1a1e17` |
| `ink` | text | `#1a1a17` | `#edf0e4` |
| `graphite` | secondary text (`--ink-3`) | `#66665e` | `#8a927d` |
| `pen` / `--pen` | primary accent (`--accent`) | `#1f4634` | `#326d48` |
| `pen_hover` | accent hover (`--accent-deep`) | `#15311f` | `#3d8257` |
| `risk` / `--risk` | may-not-hold (`--danger`) | `#8a3a2a` | `#cf5544` |
| `accent` | secondary accent (`--warning`, amber) | `#a35a18` | `#d48239` |
| `positive` | success | `#24643c` | `#4ca167` |
| `rule` | hairline | `#d9d3c4` | `#2a3324` |
| `rule_faint` | soft hairline (`--rule-soft`) | `#e6e0d1` | `#20271b` |
| `code_bg` | code background | `#efeae0` | `#1a1e17` |
| `margin` | decorative only | chosen by coordinator | chosen by coordinator |
| NEW `ink_2` | body text | `#3a3a35` | `#c4c9b6` |
| NEW `ink_4` | disabled / quiet | `#9a9a90` | `#5a6150` |
| NEW `accent_soft` | tinted accent surface | `#e4ebe5` | `#1c2b20` |
| NEW `accent_ink` | text on accent | `#f7f4ed` | `#edf0e4` |
| NEW `accent_text` | accent as TEXT (AA) | `#1f4634` | `#7fc79a` |
| NEW `danger_text` | risk as TEXT (AA) | `#8a3a2a` | `#e8826a` |
| NEW `bg_deep` | dark band / deep surface | `#12241a` | `#0d120e` |
| NEW `paper_elevated` | raised surface | `#ffffff` | `#22291e` |

Prototype ratios were checked: every text pair at or above 4.5:1 in both themes (`checks/contrast.py`).

## Type

Families (all self-hosted, SIL OFL, variable woff2, `latin` and `latin-ext` subsets):
`Geist` (UI sans), `Geist Mono` (figures, labels, code), `Newsreader` (serif: wordmark, italic accents; both
normal and *italic*). Weights used: Geist 300-700, Geist Mono 400-600, Newsreader 400-500.
Stacks: `--sans: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;`
`--serif: 'Newsreader', Georgia, serif;` `--mono: 'Geist Mono', 'JetBrains Mono', ui-monospace, monospace;`.
Files live in `static/fonts/` (app, via `ledger-fonts.css`) and `ui/landing_component/fonts/` (landing).
The old Baloo 2 and Mukta files are removed once nothing references them.

## Prototype classes that already match original names

`step-head`, `step-n`, `datum`, `gauge`, `sc` (stage row), `run-banner`, `file-identity`, `trust-strip`,
`check-row` / `check ok|risk|note`, `directive` (original `exec-directive`). Restyle these; do not rename.

## Gates every worker runs on its own files

`ruff check <your files>`, `mypy src/` (if you touched `src/`), the tests that cover your files (named in
your brief), `python -c "import ui.styles"` style import checks, and for CSS inside f-strings: render the
string and check braces balance. No browser, no screenshots, no network.
