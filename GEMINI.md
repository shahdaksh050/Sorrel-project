# GEMINI.md — Cross-Agent Coordination & Working Memory

This document serves as the shared coordination ledger and working memory between **Google Antigravity (Gemini)** and **Claude Code** (or other AI agents) collaborating on this repository.

---

## 1. Recent Completed Work (Antigravity)

**Branch**: `frontend-overhaul` (merged cleanly from `origin/frontend-overhaul-v2`)  
**Commit**: `805041f` — *fix(ui): eliminate 3D cinematic UI inconsistencies, hallucinated metrics, and visual artifacts*

### A. Data Truth & Unwrapping Fixes ([ui/cinematic_3d.py](file:///d:/(Dev2)DSA_AGENT/ui/cinematic_3d.py))
- **Unwrapped `ToolResult` objects**: `find_tool()` was previously returning raw wrapper dictionaries (`{"tool": ..., "output": ...}`) without extracting `.get("output")`. Lookups for `train_out.get("best_model")` returned `None`, triggering mock SaaS fallbacks on every dataset.
- **Bound Real Correlations**: Mapped `col_a`, `col_b`, and `correlation` from `CorrelationAnalysisTool` outputs (`top_correlations`).
- **Bound Real Headline Findings & Outliers**: Extracted genuine conclusions and findings from `run_view` or `final_report`. Extracted exact outlier counts and percentages from `detect_outliers`.
- **Descriptive EDA vs. Supervised ML Mode (`has_models: bool`)**:
  - When analyzing descriptive datasets without classification/regression targets (e.g. `AirQualityUCI.csv`), the UI automatically pivots:
    - **Section 4**: Displays **Continuous Feature Manifolds** (PCA components, explained variance, continuous dimension counts) rather than hallucinated classifier tournaments.
    - **Section 5**: Displays **Statistical Robustness Envelope** (audited outlier bounds, distribution integrity directives) rather than fake train/test gap warnings.
    - **Bottom Datum Bar**: Displays `MODE: Descriptive EDA`, `QUALITY: XX/100`, and `ANOMALIES: X.X%`.

### B. Visual 3D Refinements ([ui/assets/cinematic_3d.js](file:///d:/(Dev2)DSA_AGENT/ui/assets/cinematic_3d.js))
- **Stage 2 Constellation**: Updated link pairings (`linkPairsMap[2]`) to connect adjacent nodes within the same cluster filament (`cluster = i % 3`), and reduced line opacity in Day mode to `0.28`. Replaced the tangled, dark brown "yarn ball" with a luminous astronomical constellation of correlation vectors.
- **Stage 4 Generalization Envelope**: Adjusted `outerMat.opacity` to `0.16` (`depthWrite: false`), `innerMat.opacity` to `0.10`, and highlighted edges with `opacity: 0.85`. Replaced the opaque olive-green rock with a crystalline, faceted emerald diamond shield where interior data points sparkle through.
- **Stage 5 Executive Dossier**:
  - Re-engineered the 1,440 tablet particles into 16 horizontal ledger ruling lines that neatly wrap around the seal (left margin `-1.45`, wrapping to `0.55` near the seal, extending to `1.35` below it).
  - Formed the 608 seal particles into two concentric rings with an embossed signet core at `cx = 1.18, cy = 0.68`.
  - Refined `sealGeo` (height `0.035`) and `sealMat` (metallic luster `0.92`, roughness `0.18`) into an elegant burnished gold signet stamp.
- **Day Mode Lighting**: Tuned `matterMat.roughness` (`0.32`) and `metalness` (`0.40`) in Day mode so particles catch warm ambient and key lights gracefully without looking like dark charcoal pellets.
- **HUD Pin Clamping**: Adjusted `minPinX` and `pinnedX` clamping bounds to prevent leader line badges from clipping on the right edge of the screen.

### C. Typography & Compact Sizing ([ui/assets/cinematic_3d.html](file:///d:/(Dev2)DSA_AGENT/ui/assets/cinematic_3d.html))
- **Typography**: Removed the cartoonish font `Baloo 2`. Replaced `--heading` with technical editorial sans `'Mukta', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif`.
- **Text Contrast**: Scoped blur text-shadows strictly to `.theme-night`, eliminating dirty brown halos around dark text in Day mode.
- **Audited Claim Badges**: Replaced gimmicky marketing badges (`✓ ZERO HALLUCINATIONS`) with verifiable claim audit counts (`✓ 4 OF 4 CLAIMS VERIFIED` and `✓ REPRODUCIBLE REPL LEDGER`).
- **Compact Hero Box Styling**: Added `.compact` responsive CSS for paddings, font sizing, and cards so the 3D presentation scales cleanly inside the `height=480` Streamlit container without clipping or nested scrollbars.

### D. Self-Contained Standalone HTML Export ([ui/tabs/downloads_tab.py](file:///d:/(Dev2)DSA_AGENT/ui/tabs/downloads_tab.py))
- Enabled `inline_assets=True` so downloaded HTML presentations embed all fonts, Three.js, and Anime.js as base64 data URIs and run offline anywhere without broken `/app/static/` URLs.

### E. Live Run Answers Tab Audit ([UI_DESIGN_AUDIT.md](file:///d:/(Dev2)DSA_AGENT/UI_DESIGN_AUDIT.md))
- **A3 Verified Visually**: Confirmed `.finding-card.full-width` failed to span `grid-column: 1 / -1` due to `.bento-card.full-width` CSS mismatch, causing Card 1 to sit in column 1 and leaving row 2 with an orphaned blank 3rd slot.
- **A9 Added [P0]**: Verdict count contradiction (`4 of 4 findings held up` directly above 5 cards).
- **A10 Added [P1]**: Raw unrounded 4-decimal floats leaking into card body copy (`1201.1968`, `630.8992`, `939.1534`, `17.7356`, `2.9167`, `10.0831`).
- **A11 Added [P1]**: Fluctuating sample sizes across adjacent cards (9,326 vs 8,991) due to pairwise null drops without user explanation.
- **A12 Added [P2]**: Robotic title repetition in body copy and quote formatting inconsistencies (`'C6H6(GT)'` vs unquoted `CO(GT)`).
- **A13 Added [P1]**: Success banner (`4 of 4 findings held up`) styled in terracotta warning tint (`var(--pen)`) instead of green (`var(--positive)`).
- **A14 Added [P2]**: Unlabelled Executive Directive card when no custom prompt objective is passed.

---

## 2. Active Plans & Next Steps

1. **Streamlit Component Hydration & State Sync**:
   - Ensure the `run_view` summary, audit trail, and details tabs in [app.py](file:///d:/(Dev2)DSA_AGENT/app.py) maintain identical metrics with the 3D showcase.
   - Verify that dataset switches in Streamlit clear the cached cinematic state and reload clean profile data.
2. **Performance Monitoring**:
   - Keep 3D canvas render budget strictly at <= 60 FPS across both desktop and compact iframe environments.
   - Preserve zero per-frame heap allocations (Pillar 7 in `cinematic_3d.js`).
3. **Cross-Agent Quality Gates**:
   - Any agent making modifications must maintain:
     1. `ruff check .` with 0 violations.
     2. `mypy src/` with 0 type errors.
     3. `pytest tests/test_cinematic_3d.py tests/test_islands.py tests/test_contrast.py tests/test_architecture.py -v` (all 24 passing).

---

## 3. Collaboration Protocol for Antigravity & Claude Code

To collaborate without race conditions or merge conflicts:

| Practice | Guideline |
| :--- | :--- |
| **Branch Separation** | Never push directly to each other's active work branch. Use feature branches (e.g. `frontend-overhaul` vs `claude-feature`) or worktrees (`.worktrees/`). |
| **Shared Rules** | [AGENTS.md](file:///d:/(Dev2)DSA_AGENT/AGENTS.md) is the primary directive source for both agents. Preserve all 7-stage workflow rules, anti-overfitting requirements, and type hint mandates. |
| **No File Clashes** | Avoid simultaneously editing the exact same file in two terminals. If Antigravity is refactoring `ui/`, Claude Code should focus on `src/` or tests, and vice versa. |
| **Commit Discipline** | Commit atomic units of work with conventional commit messages (`feat:`, `fix:`, `refactor:`, `test:`). |

> **Override, set by the user on 2026-10-03:** we share one working tree on `frontend-overhaul` and coordinate
> through markdown files, not worktrees. Section 4 replaces the "Branch Separation" row for now.

---

## 4. Status after the v2 attempt (2026-10-03)

**The v2 attempt (FastAPI + React, "Doubly") was scrapped by the user** because they did not like the new
UI. All of it is in a git stash named `v2 attempt (scrapped 2026-10-03)`; the working tree is back at
commit `805041f`. Every v2 task (T1 to T4, H1 to H5), file claim and the `V2_*.md` plans are void.
The design principles are wanted, applied to the **current** Streamlit UI.

Read `DESIGN_LEARNINGS.md` for what was learned and decided, and `UI_DESIGN_AUDIT.md` for the defects in
the current UI (IDs W, S, R, A, D, Y). Section 8 of `DESIGN_LEARNINGS.md` is the suggested order of work.

### Roles and working rules (still in force)

- **Claude Code is the lead; Antigravity (Gemini) is the helper** (the user's decision). Both work together
  through markdown files in one working tree on `frontend-overhaul`.
- **Claim before you edit:** add a row to "File claims" below; remove it when done. Never edit a claimed
  file. Commits only when the user asks.
- **No claim without evidence:** reports say what was read or run, with `file:line` or a screenshot path,
  and say what was not verified.
- The user's standing rule: do not run the app or the test suite until they say so.
- `AGENTS.md` gates apply to any code change: `ruff check .`, `mypy src/`, and the pytest list in section 2.

### File claims

| Agent | Files | Since | Purpose |
| :--- | :--- | :--- | :--- |
| Claude Code | `app.py`, `ui/styles.py`, `ui/components/cards.py`, `ui/tabs/answers_tab.py`, `.streamlit/config.toml`, `static/vendor/fonts/`, `scripts/capture_screenshots.py`, `THIRD_PARTY_UI.md` | 2026-10-03 | UI revamp, Phase 0 and Phase 1 (plan: `C:\Users\daksh\.claude\plans\using-the-knowledge-of-cheerful-wombat.md`) |

### Task queue

| ID | Owner | Status | Task |
| :--- | :--- | :--- | :--- |
| UI-1 | Claude Code | code done, unverified (Phases 0 to 5 and the static parts of 6) | Phase 0 and Phase 1 of the UI revamp plan: top bar, mode control in the task area, collapsed Settings, new fonts. Awaiting Checkpoint 1 (needs the user's go-ahead to run the app). |
| UI-2 | Claude Code | waiting for Checkpoint 1 | Phases 2 to 7. |
| UI-3 | Antigravity (Gemini) | Complete (Proposal & Prototype) | Verdacert UI breakdown & adaptation plan: docs/VerdacertInspirationPlan.md, interactive prototype in docs/prototypes/verdacert_dsa_preview.html |

