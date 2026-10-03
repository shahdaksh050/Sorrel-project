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
| `pen_hover` | accent hover, deeper than `pen` | `#15311f` | `#2b6040` (the prototype's `#3d8257` was only 4.0:1 under button text) |
| `risk` / `--risk` | may-not-hold (`--danger`) | `#8a3a2a` | `#cf5544` |
| `accent` | secondary accent (`--warning`, amber) | `#a35a18` | `#d48239` |
| `positive` | success | `#24643c` | `#4ca167` |
| `rule` | hairline | `#d9d3c4` | `#2a3324` |
| `rule_faint` | soft hairline (`--rule-soft`) | `#e6e0d1` | `#20271b` |
| `code_bg` | code background | `#efeae0` | `#1a1e17` |
| `margin` | decorative only (brass, far in hue from `risk`) | `#9a8650` | `#b3a06a` |
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
**Installed in Phase 1** (do not re-download): `geist-latin-wght-normal.woff2`, `geist-mono-latin-wght-normal.woff2`,
`newsreader-latin-wght-normal.woff2`, `newsreader-latin-wght-italic.woff2`, plus `LICENSE-geist.txt`, `LICENSE-geist-mono.txt`,
`LICENSE-newsreader.txt`, in all three font folders, with `@font-face` rules appended to each stylesheet:
`static/vendor/fonts/ledger-fonts.css` (console, pipeline and cinematic scenes), `static/fonts/ledger-fonts.css` (the HTML report
embeds from here), `ui/landing_component/fonts/landing-fonts.css` (landing). The stylesheet FILE NAMES stay (`ledger-fonts.css`,
`landing-fonts.css`): tests and code pin those paths. The old Baloo 2, Mukta, Bricolage Grotesque, Public Sans and IBM Plex files
and rules are still present and are deleted in the final cleanup phase, once nothing references them. Each owner switches the
families their files USE to the Sorrel stacks; the coordinator removes the old ones afterwards.

## Prototype classes that already match original names

`step-head`, `step-n`, `datum`, `gauge`, `sc` (stage row), `run-banner`, `file-identity`, `trust-strip`,
`check-row` / `check ok|risk|note`, `directive` (original `exec-directive`). Restyle these; do not rename.

## Gates every worker runs on its own files

`ruff check <your files>`, `mypy src/` (if you touched `src/`), the tests that cover your files (named in
your brief), `python -c "import ui.styles"` style import checks, and for CSS inside f-strings: render the
string and check braces balance. No browser, no screenshots, no network.

## File contracts the existing tests pin (from the test inventory)

- `ui/styles.py` must keep these exact marker comments, each followed later by `</style>`: `/* ── Native widget text follows the page tokens`,
  `/* ── "How we got here"`, `/* ── "Found so far, may change"`; plus `--mono:`, `--ease-in-out:`, `.finding-card.full-width`,
  `.st-key-plate iframe`; no `fonts.googleapis` / `fonts.gstatic`; functions `font_css_url()` and `inject_theme_css()`. CSS blocks between
  those markers use `var(--...)` only (no hex). Every class `build_how_html` / `build_provisional_html` emits must be defined in styles.py.
- `ui/landing.py` keeps `LEDGER_TOKENS_DAY` / `LEDGER_TOKENS_NIGHT` (keys stock, sheet, ink, graphite, pen) and the module attribute
  `_landing_component`. `ui/landing_component/index.html` declares `--stock: #hex`, `--pen: #hex`, `--ink: #hex` for both themes (the
  `html.theme-day` selector) with exactly the Python token values (single space after the colon, lowercase hex), keeps `--heading` and
  `--sans` variables, the ids `hero-enter-btn`, `hero-explore-btn`, `enter-btn`, `cta-launch-btn`, `nav-launch-btn`, `btn-theme-toggle`,
  `id="data-path"`, the bridge payloads `setComponentValue({ enter: true, theme: currentTheme })` and `({ enter: false, ... })`,
  `function dsaBridge()` in a classic script BEFORE `<script type="module">`, `window.__dsaModuleReady`, JSON-LD with a
  `SoftwareApplication` node, and a visible line starting `Where your data goes:` with the required phrases.
- Honest-copy banned words (case-insensitive, visible text): swarm, synapse, manifold, undulation, certified, telemetry, "never leaves",
  "runs entirely on your machine", "100% local", "zero cloud", "cloud transmissions", air-gapped, wcag, compliance, cryptographic,
  "formal mathematical verification", "guard status", operational, "rlm v2", "high risk", "no credit card"; also no pricing/FAQ remnants.
- `ui/landing_component/fonts/landing-fonts.css` must not contain the substring `http` anywhere (comments included), and every
  `url('./x.woff2')` must exist beside it.
- Static folder rule: only `.woff2`, `.css`, `.js`, `.json`, and `LICENSE*.txt` / `OFL.txt` under `static/`.
- `static/vendor/three/three.module.js` must stay byte-identical to `ui/landing_component/vendor/three.module.js`.
- Path contracts: `static/vendor/fonts/ledger-fonts.css` (served at `/app/static/vendor/fonts/ledger-fonts.css`).
- The HTML report must stay offline (fonts embedded as base64, no CDN) and `src/core/report_assets.py` pins the Baloo/Mukta rule counts in
  `tests/test_report_assets.py` (re-pin to the Sorrel families, same intent).
- `ui/pipeline_3d.py` keeps `PALETTE`, `Stage`, `_asset`, `build_document`, `render`; `PALETTE[k] == palette("day")[k]`.
