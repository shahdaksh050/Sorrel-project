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

### F. Domain Reasoning & Causal Inference Pipeline (2026-10-09)
- **Domain Reasoning Engine** ([`src/core/domain_reasoning.py`](file:///d:/(Dev2)DSA_AGENT/src/core/domain_reasoning.py)): Deterministic validation of domain specs (`DomainAnalysisSpec`), sanitizing LLM proposals against catalog, financial, industrial, or scientific profiles.
- **Causal Design & Adversarial Critique** ([`src/core/causal_design.py`](file:///d:/(Dev2)DSA_AGENT/src/core/causal_design.py)): Contract validation for causal proposals, unobserved confounding checks, and deterministic adversarial review.
- **Causal Analysis Tools** ([`src/tools/causal_analysis.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/causal_analysis.py)): Implemented `CausalAnalysisTool` (Inverse Propensity Weighting with positivity/overlap checking) and `DifferenceInDifferencesTool` (cluster-robust SEs, parallel pre-trend testing).
- **Controller & Report Integration**: Injected causal workflows into [`controller.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller.py), [`report_generator.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/report_generator.py), and [`html_report.py`](file:///d:/(Dev2)DSA_AGENT/src/core/html_report.py).

### G. Backend Architecture Pruning & Intent-Steered Planning (2026-10-09)
- **Pruned Memory Pinning**: Deleted `SharedAnalysisContext` and its test, stopping raw DataFrame retention.
- **Pruned Unused Engine Helpers**: Removed `fast_groupby` and `fast_quantiles` from `src/core/duckdb_engine.py` while preserving full SQL execution (`dsa.query_sql`).
- **Eliminated Duplicate Join Engine**: Deleted `src/core/relational_joiner.py` and test, cementing `src/core/joins.py` as canonical.
### H. Analytical Depth, Clean Formatting, & Weak Clustering Gate (2026-10-09) [Task BE-2]
- **Human-Friendly Number Formatting** ([`src/tools/clustering.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/clustering.py)): Implemented `_format_metric_value()` eliminating raw `e+` scientific notation from cluster headlines (e.g. converting `4.85e+04 vs 3.14e+04` to `48,500 vs 31,400`).
- **LLM Planning Categorical Steering** ([`src/core/prompt_manager.py`](file:///d:/(Dev2)DSA_AGENT/src/core/prompt_manager.py), [`src/core/agenda.py`](file:///d:/(Dev2)DSA_AGENT/src/core/agenda.py)): Explicitly instructed planner prompts (`SYSTEM_PROMPT_CORE`, `SYSTEM_PROMPT_COMPACT`, `INITIAL_ANALYSIS_PROMPT`, `ITERATION_PROMPT`) to prioritize group differences using `segment_comparison` when prominent categorical dimensions exist. Boosted expected value and added `[RECOMMENDED]` badges to high-signal segment questions in the agenda block.
- **Weak Clustering Gate & Finding Prioritization** ([`src/core/findings.py`](file:///d:/(Dev2)DSA_AGENT/src/core/findings.py), [`src/tools/clustering.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/clustering.py)): Gated cluster findings with weak silhouette (`< 0.35`) by discounting apparent profile effects by silhouette strength, capping confidence at `0.25`, and updating `rank_findings()` to ensure verified group lifts (`segment_lift`) and drivers always outrank weak cluster artifacts.
- **Tests Added**: Unit tests in [`tests/test_clustering.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_clustering.py) testing `_format_metric_value` and weak clustering gating. Verified with `ruff check .` (0 violations) and `mypy src/` (0 errors across 105 files).

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
| **Shared Rules** | [AGENTS.md](file:///d:/(Dev2)DSA_AGENT/AGENTS.md) is the primary directive source for both agents. Preserve all 5-stage workflow rules, anti-overfitting requirements, and type hint mandates. |
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
| BE-1 | Antigravity (Gemini) | Complete | Pruned orphans (SharedAnalysisContext, fast_groupby/quantiles, relational_joiner); wired question_router into fallback plan. |
| BE-2 | Antigravity (Gemini) | Complete | Deepen data analysis: fix clustering formatting, steer LLM planning to segment comparisons, gate weak clustering (see HANDOVER.md). |
| BE-3 | Antigravity (Gemini) | Complete | Fix sandbox subprocess timeout: fast schema dtype builder (dropped profile_dataframe), cached Win32 path resolution in audit hook, throttled psutil RSS checks (1.0s), bumped default timeout to 60s, steered prompt away from groupby dynamic code towards segment_comparison. |
| BE-4 | Antigravity (Gemini) | Complete | Self-healing dynamic code fallback to segment_comparison, enhanced categorical distribution previews (percentages + n), and unit/dollar-quantified recommendations across LLM & deterministic modes. |
| BE-5 | Antigravity (Gemini) | Complete | Fix sandbox static_check error on .read_csv: AST reader normalization (_ReaderCallTransformer, _normalize_sandboxed_code) safely transforms pd.read_csv to in-memory df references without filesystem I/O. |
| BE-6 | Antigravity (Gemini) | Complete | Dynamic code self-healing & error immunity: AST normalization of hallucinated methods (.reset, .drop_column, etc.), worker compatibility aliases, MultiIndex serialization flattening, and direct groupby execution fallback. |
| BE-7 | Antigravity (Gemini) | Complete | Dynamic chart cluster labels & stale fragment timer cleanup: rich cluster labels in legend/tooltips, on-chart centroid markers & text callouts, axis variance, and Streamlit fragment timer reset & log filter. |
| AI-1 | Antigravity (Gemini) | Complete | Milestone A (External AI Compatibility): AI access policy, origin configuration, public/ discovery files (robots, sitemap, llms.txt, security.txt), docs/public/*.md, surface linter script, and discovery test suite. |
| AI-2 | Antigravity (Gemini) | Complete | Rewire public origin to localhost:8501, landing navbar branding clean-up (removed [s] seal & student badge), and Human/AI view mode segmented switcher with contextual LLM overlay. |
| BE-8 | Antigravity (Gemini) | Complete | Restored 7-stage workflow and RLM feature-group decomposition (RLMEngine.decompose_and_invoke, REPLEnvironment, prompt_manager subtask templates, controller Stage 6, 7-stage UI stepper & Detail Handler agent card, and unit tests). |
| BE-9 | Antigravity (Gemini) | Complete | Senior quantitative pipeline upgrades: Simpson's paradox & confounder auditing, 2-way categorical interaction screening, ScenarioSimulationTool for what-if driver interventions & gap closing, and 2x2 Action Prioritization Matrix in Markdown and HTML reports. |
| BE-10 | Antigravity (Gemini) | Complete | Autonomous unprompted investigation engine: semantic North Star inference, lead hypothesis promotion from Question Agenda, autonomous mission prompt injection, and What-If scenario pre-seeding. |

### I. Sandbox Optimization & Group Comparison Steering (2026-10-09) [Task BE-3]
- **Worker Subprocess Lightening** ([`src/core/_sandbox_worker.py`](file:///d:/(Dev2)DSA_AGENT/src/core/_sandbox_worker.py)): Replaced heavy `profile_dataframe(df)` in `_build_schema` with a lightweight, direct pandas dtype classifier, dropping the costly profiling pass and regex scans inside the child process.
- **Windows Path Resolution Caching** ([`src/core/_sandbox_worker.py`](file:///d:/(Dev2)DSA_AGENT/src/core/_sandbox_worker.py)): Added path resolution cache to `_within()` in the audit hook, avoiding repeated Win32 `GetFinalPathNameByHandle` syscalls during import/file events.
- **Process Polling & RSS Throttling** ([`src/core/sandbox.py`](file:///d:/(Dev2)DSA_AGENT/src/core/sandbox.py)): Throttled `psutil` process tree memory checks to at most once per second (from every 0.2s), eliminating OS process snapshot CPU spikes on Windows. Preserved `_stderr_tail` in `traceback` on timeout. Increased default timeout to 60.0s.
- **Prompt Planner Steering** ([`src/core/prompt_manager.py`](file:///d:/(Dev2)DSA_AGENT/src/core/prompt_manager.py)): Replaced `df.groupby(...)` dynamic code examples with legitimate custom composite calculations; explicitly instructed the planner to always use `segment_comparison` for group comparisons.

### J. Self-Healing Fallbacks, Perception Previews & Quantified Actions (2026-10-09) [Task BE-4]
- **Self-Healing Dynamic Code Fallback** ([`src/core/controller_steps.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_steps.py)): Intercepts failed or timed-out `execute_dynamic_code` steps attempting a `df.groupby(...)`. Extracts dimension and measure columns, validates them against the dataset schema, and transparently executes `segment_comparison` to produce verified statistical findings while recording the code execution for audit compliance.
- **Categorical Perception & Distribution Previews** ([`src/core/profiler.py`](file:///d:/(Dev2)DSA_AGENT/src/core/profiler.py)): Augmented `_level_shares` to format top categorical categories with both percentages and exact sample counts (`[Android 94% (n=963), iOS 5% (n=47)]`), allowing the LLM planner to perceive group sizes and class imbalance directly before planning.
- **Quantified Business Decisions** ([`src/core/prompt_manager.py`](file:///d:/(Dev2)DSA_AGENT/src/core/prompt_manager.py), [`src/core/controller_report.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_report.py)): Enforces concrete, business-actionable recommendations naming affected groups and quantifying expected impact in baseline units/dollars. Deterministic synthesis extracts concrete segment lift and driver actions directly from ranked findings.
- **Frontend Presentation Alignment**: Verified contract compatibility with `ui/tabs/answers_tab.py` (rendered under "What to do" as `<div class="rc">`), `ui/cinematic_3d.py` (Executive Dossier), and standalone HTML exports.

### K. Sandbox Reader AST Normalization (2026-10-09) [Task BE-5]
- **AST Reader Normalization** ([`src/core/sandbox.py`](file:///d:/(Dev2)DSA_AGENT/src/core/sandbox.py)): Implemented `_ReaderCallTransformer(ast.NodeTransformer)` and `_normalize_sandboxed_code()`. Automatically detects and normalizes LLM code expressions invoking dataset file-loading methods (`pd.read_csv(...)`, `pd.read_table(...)`, `from pandas import read_csv`, etc.) into direct references to the pre-loaded in-memory `df` (`df = df` or `data = df`), while preserving scope integrity with `global df` when assigned inside nested functions.
- **Static Check & Multi-Backend Wiring**: Wired `_normalize_sandboxed_code()` into `_static_check()`, `SubprocessSandbox.execute()`, `DockerSandbox.execute()`, and `run_sandboxed()`. Completely eliminates unrecoverable `static_check` rejections on `.read_csv` while maintaining strict security policies against actual arbitrary file/network I/O.
- **Tests Added**: Unit tests added to [`tests/test_sandbox.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_sandbox.py) for static check validation of `pd.read_csv`, `from pandas import read_csv`, chained calls (`pd.read_csv(...).dropna()`), functions, and full end-to-end sandbox execution on pre-loaded datasets. Verified: all 37 sandbox tests pass, `ruff check .` (0 violations), `mypy src/` (0 errors across 105 files).

### L. Dynamic Code Self-Healing & Error Immunity (2026-10-10) [Task BE-6]
- **Root Cause Analysis**:
  1. *Hallucinated Pandas API Methods*: LLMs routinely write `.reset()` instead of `.reset_index()`, `.drop_column()` instead of `.drop(columns=...)`, `.to_dataframe()` instead of `.to_frame()`, `.sort(...)` instead of `.sort_values(...)`, and deprecated `.append()`.
  2. *MultiIndex Tuple Keys Crash JSON Serialization*: Aggregations like `.agg({'rating': ['count', 'mean'], 'price': 'mean'}).reset_index()` create MultiIndex column tuples `('rating', 'count')`. When converted with `.to_dict(orient="records")`, `json.dumps()` crashed with `TypeError: keys must be str, int, float, bool or None, not tuple`.
  3. *Unassigned RESULT Variable*: Scripts writing `agg = df.groupby(...)` without explicit `RESULT = agg` failed top-level validation.
  4. *Cardinality Bounds on Segment Comparison*: `segment_comparison` restricts dimensions to 2-20 levels; datasets with >20 categories (e.g. 25-50 brands) failed the pivot.
- **AST Normalization & Auto-Wiring** ([`src/core/sandbox.py`](file:///d:/(Dev2)DSA_AGENT/src/core/sandbox.py)): Upgraded AST transformers to automatically rewrite `.reset()` -> `.reset_index()`, `.drop_column(...)` -> `.drop(columns=...)`, `.to_dataframe()` -> `.to_frame()`, `.sort(...)` -> `.sort_values(...)`, and append `RESULT = <candidate>` if `RESULT` is omitted.
- **Worker Runtime Compatibility & Serialization Immunity** ([`src/core/_sandbox_worker.py`](file:///d:/(Dev2)DSA_AGENT/src/core/_sandbox_worker.py)):
  - Injected runtime compatibility aliases (`reset`, `to_dataframe`, `drop_column`, `rename_column`, `sort`, `append`).
  - Added `_safe_reset_index` and `_clean_dict_keys` to flatten MultiIndex columns into clean strings (`rating_count`, `rating_mean`, `price_mean`), completely eliminating `TypeError` tuple-key crashes.
  - Implemented execution self-healing retry loop: auto-fixes column name case/typos on `KeyError`, auto-corrects `.reset()` on `AttributeError`, pre-loads standard safe modules (`re`, `json`, `datetime`, `math`), and uses `traceback.format_exception(err)` for robust error reporting.
- **Multi-Measure & Direct Groupby Fallback** ([`src/core/controller_steps.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_steps.py)):
  - Updated `_extract_groupby_columns(code)` to parse multi-measure aggregations (`.agg({'rating': ['count', 'mean'], 'price': 'mean', ...})`).
  - Implemented 2-stage fallback: Attempt 1 pivots to `segment_comparison`; Attempt 2 directly computes grouped statistics, lift vs baseline, registers a verified `Finding(kind="segment_lift")`, and returns success.
- **Tests Added & Verification**:
  - `TestSandboxedCodeSelfHealing` in [`tests/test_sandbox.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_sandbox.py) (4 tests).
  - Updated `test_extract_groupby_columns` and added `test_dynamic_code_fallback_direct_groupby` in [`tests/test_controller_reasoning_steps.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_controller_reasoning_steps.py).
  - Full suite passed: `pytest tests/test_controller_reasoning_steps.py tests/test_sandbox.py -v` (59 passed), `pytest tests/test_sandbox_worker.py -v` (18 passed), `ruff check .` (0 violations), `mypy src/` (0 errors across 105 files).

### M. Dynamic Chart Cluster Labels & Stale Fragment Timer Cleanup (2026-10-10) [Task BE-7]
- **Streamlit Stale Fragment Timer Cleanup** ([`app.py`](file:///d:/(Dev2)DSA_AGENT/app.py)):
  - Root cause: When an active analysis ended, `_run_progress` called `st.rerun(scope="app")`. In subsequent runs `_run` was `None`, so `if st.session_state.get("_run") is not None:` skipped calling `_run_progress()`. Because the fragment was never rendered in the new run, Streamlit server dropped the fragment handler, while the client-side JavaScript interval kept firing every 1,000 ms, spamming the terminal with `The fragment with id <id> does not exist any more`.
  - Fix: Called `_run_progress()` unconditionally with dynamic `run_every` (`1s` when running, `None` when inactive). When inactive, it immediately executes with `run_every=None` and returns, instructing the client browser to halt the interval timer.
  - Added targeted `_SuppressStaleFragmentFilter` logging filter to `logging.getLogger("streamlit")` and `logging.getLogger("streamlit.runtime.fragment")` to silence any transient WebSocket race conditions.
- **Human-Readable Cluster Labels & On-Chart Centroid Callouts** ([`src/tools/clustering.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/clustering.py), [`src/core/dashboard_charts_basic.py`](file:///d:/(Dev2)DSA_AGENT/src/core/dashboard_charts_basic.py)):
  - In `ClusterDataTool`: Synthesizes defining cluster descriptors by evaluating relative deviation against dataset baseline means (e.g. `High price, Low mileage`). Builds `cluster_labels` (`Cluster 1: High price (n=450, 48%)`), `cluster_short_labels` (`Cluster 1 (High price)`), and attaches them directly to `pca_points`.
  - In `_cluster_chart`:
    - Re-engineered the 2-D PCA scatter into a high-legibility layered Vega-Lite visualization:
      1. Layer 1: Semi-transparent scatter points (`circle`, opacity 0.55) colored by `cluster_label`, with a titled, clear legend (`Discovered Segments`) and informative tooltips (`Segment`, `Cluster ID`, `PC 1`, `PC 2`).
      2. Layer 2: Prominent centroid cross markers (`point`, size 130, strokeWidth 2.5) anchored at each cluster's mean `(cx, cy)` position.
      3. Layer 3: Bold on-chart cluster badges (`text`, fontSize 12, dy -12) positioned directly over each centroid anchor.
    - Added PCA explained variance percentages to axis titles (e.g. `PC 1 (55% var)`, `PC 2 (25% var)`).
- **Verification & Quality Gates**:
  - `tests/test_clustering.py` and `tests/test_dashboard.py` updated to verify `cluster_labels`, `cluster_short_labels`, and layered Vega-Lite specs.
  - Full chart lint suite passed: `pytest tests/test_chart_lint.py -v` (341 passed).
  - Clean quality gates: `ruff check .` (0 violations), `mypy src/` (0 errors across 105 files).

### N. Milestone A & Localhost AI Surface (2026-10-12) [Tasks AI-1 & AI-2]
- **Phase 0 Policy Decisions** ([`docs/AI_ACCESS_POLICY.md`](file:///d:/(Dev2)DSA_AGENT/docs/AI_ACCESS_POLICY.md)): Formalized operational policies across credential model (server-managed initial), ephemeral dataset retention (24h default), crawler access matrix, hosted Docker sandboxing, and zero-data model training guarantees.
- **Origin Configuration Engine & Root Asset Serving** ([`src/core/origin_config.py`](file:///d:/(Dev2)DSA_AGENT/src/core/origin_config.py)):
  - Centralized canonical deployment URL generation with `SORREL_PUBLIC_ORIGIN` (defaulting to `http://localhost:8501`), strict origin normalization, and helpers for `rel="alternate"` and `rel="describedby"` discovery tags.
  - Implemented `sync_discovery_assets_to_static()`, synchronizing `public/*` and `docs/public/*` into Streamlit's Starlette root static directory (`file_util.get_static_dir()`). Direct HTTP GET requests to `/robots.txt`, `/sitemap.xml`, `/llms.txt`, `/.well-known/security.txt`, and `/docs/*.md` now return actual machine-readable assets with proper MIME types (HTTP 200) instead of falling back to Streamlit's SPA `index.html`.
- **Root Discovery Assets** (`public/`):
  - [`public/robots.txt`](file:///d:/(Dev2)DSA_AGENT/public/robots.txt): Explicit policy allowing search crawlers (`OAI-SearchBot`, `Googlebot`, `Bingbot`) while disallowing untrusted training scrapers (`GPTBot`, `Google-Extended`, `CCBot`) and strictly protecting private paths (`/api/`, `/runs/`, `/uploads/`). Sitemap rewired to `http://localhost:8501/sitemap.xml`.
  - [`public/sitemap.xml`](file:///d:/(Dev2)DSA_AGENT/public/sitemap.xml): Machine-readable XML index of all canonical public pages rewired to `http://localhost:8501/`.
  - [`public/llms.txt`](file:///d:/(Dev2)DSA_AGENT/public/llms.txt): Standard single-H1 machine-readable index with blockquote summary, navigation sections, and canonical links rewired to `http://localhost:8501/docs/*.md`.
  - [`public/.well-known/security.txt`](file:///d:/(Dev2)DSA_AGENT/public/.well-known/security.txt) & [`public/security.txt`](file:///d:/(Dev2)DSA_AGENT/public/security.txt): RFC 9116 compliant security disclosure declarations rewired to `http://localhost:8501/`.
- **Public Markdown Documentation** (`docs/public/*.md`):
  - In-depth, high-density documentation for external agent ingestion: `overview.md`, `workflow.md`, `safety.md`, `api-quickstart.md`, `mcp.md`, `file-formats.md`, `result-schema.md`, `limits.md`, and `changelog.md`. Canonical headers, cURL commands, and MCP configs rewired to `http://localhost:8501`.
- **Landing Metadata, Header Clean-up & Interactive AI Modal** ([`ui/landing_component/index.html`](file:///d:/(Dev2)DSA_AGENT/ui/landing_component/index.html)):
  - Removed `.brand-seal` (`[s]` icon) and `.brand-badge` ("Student project") from landing page header, leaving a clean typographic wordmark `<span class="brand-name">Sorrel</span>`.
  - Removed redundant `[llms.txt standard]` pill badge from the AI modal header.
  - Added segmented view switcher in navbar (`[ 👤 Human | 🤖 AI / Agent ]`) and quick-jump tab buttons in the modal bar (`/llms.txt`, `/robots.txt`, `/sitemap.xml`, `Public Docs`).
  - Added full interactive tabs inside the modal for `/llms.txt`, `/robots.txt`, `/sitemap.xml`, and an interactive `Public Docs` guide reader (with dropdown selector for all 9 guides, instant syntax-highlighted preview, copy to clipboard, and instant client-side file download).
  - Open raw URL buttons now link to live HTTP endpoints (`/robots.txt`, `/sitemap.xml`, `/llms.txt`) which return real content without redirecting to the landing page.
- **Surface Linter & Discovery Test Suite**:
  - Validated with [`scripts/check_ai_surface.py`](file:///d:/(Dev2)DSA_AGENT/scripts/check_ai_surface.py) passing cleanly.
  - Implemented unit tests in [`tests/test_ai_discovery.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_ai_discovery.py) (16 tests).
- **Quality Gates**:
  - `ruff check .` (0 violations)
  - `mypy src/` (0 errors across 106 source files)
  - `pytest tests/test_ai_discovery.py tests/test_landing.py tests/test_landing_v2.py -v` (36 passed)
  - `pytest tests/test_cinematic_3d.py tests/test_contrast.py tests/test_architecture.py tests/test_ai_discovery.py -v` (39 passed)

### O. 7-Stage Workflow & RLM Decomposition Restoration (2026-10-12) [Task BE-8]
- **RLM Engine & REPL Environment** ([`src/rlm/engine.py`](file:///d:/(Dev2)DSA_AGENT/src/rlm/engine.py), [`src/rlm/__init__.py`](file:///d:/(Dev2)DSA_AGENT/src/rlm/__init__.py)):
  - Restored `REPLEnvironment` and `RLMSubTask` dataclass.
  - Restored `RLMEngine.decompose_and_invoke()` with bounded parallel thread pool (`_MAX_DECOMPOSE_WORKERS = 6`), per-subtask error isolation, token budget retry expansion (`_RETRY_MAX_TOKENS_CEILING = 8192`), and deterministic result ordering.
- **Prompt Manager Subtask Assembly** ([`src/core/prompt_manager.py`](file:///d:/(Dev2)DSA_AGENT/src/core/prompt_manager.py)):
  - Restored `RLM_SUBTASK_SYSTEM`, `RLM_SUBTASK_PROMPT`, `_rlm_block()`, `_column_patterns()`, and `rlm_group_findings()`.
  - Re-injected `_rlm_block()` into iteration and final synthesis prompts so feature-group findings enrich executive conclusions.
- **Controller Stage 6 Decomposition** ([`src/core/controller.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller.py), [`src/core/controller_report.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_report.py)):
  - Restored `_should_decompose()`, `_rlm_skip_reason()`, and `_run_rlm_decomposition()` for datasets with >12 columns.
  - Restored Stage 7 synthesis tags (`stage7:no_steps_synthesis`, `stage7:converged_synthesis`, `stage7:max_iter_synthesis`, `stage7:report`).
- **UI Stepper, Run Worker & Agent Grid** ([`ui/components/cards.py`](file:///d:/(Dev2)DSA_AGENT/ui/components/cards.py), [`ui/run.py`](file:///d:/(Dev2)DSA_AGENT/ui/run.py)):
  - Restored 7-stage definitions: Reading Your File (1), Understanding Your Question (2), Running the Numbers (3), Making Sense of It (4), Double-Checking (5), Solving the Tricky Parts (6), Writing Your Report (7).
  - Restored `Detail Handler` agent card in `render_agent_grid()` and deep dive modal.
  - Fixed `RunSpec` dataclass field order in `ui/run.py` and restored Stage 6 / Stage 7 callbacks and completion logs.
- **Verification & Documentation**:
  - `AGENTS.md`: Restored 7-Stage Workflow table and Success Criteria.
  - Test suites updated and passing: `test_rlm_engine.py` (13 passed), `test_rlm_truncation.py` (6 passed), `test_presentation_cards.py` (22 passed), `test_run_worker.py` (12 passed), `test_prompt_manager.py` (21 passed), `test_controller.py` (29 passed), `test_html_report.py` (14 passed).
  - Quality gates: `ruff check .` (0 violations), `mypy src/` (0 errors across 106 source files).

### P. Senior Quantitative Analytical Upgrades (2026-10-12) [Task BE-9]
- **Simpson's Paradox & Confounder Auditor** ([`src/core/sensitivity.py`](file:///d:/(Dev2)DSA_AGENT/src/core/sensitivity.py), [`src/core/findings.py`](file:///d:/(Dev2)DSA_AGENT/src/core/findings.py), [`src/core/controller.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller.py)):
  - Implemented `audit_simpsons_paradox(df, finding, candidate_stratifiers, min_stratum_n)` checking whether bivariate correlations or categorical segment contrasts invert sign across third-party subgroups.
  - Attached to finding audit checks (`checks["paradox_free"] = not has_paradox`) in `attach_finding_checks()`.
  - Wired into Stage 5 post-hoc audit loop in `AgentController._audit_simpsons_paradox()`.
- **2-Way Categorical Interaction Screening** ([`src/tools/segment_comparison.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/segment_comparison.py)):
  - Implemented `_screen_interactions()` evaluating compound cross-products ($D_1 \times D_2$) vs additive expectations ($\text{Lift}(A) + \text{Lift}(B)$).
  - Emits `Finding(kind="interaction_synergy", ...)` for statistically significant compound effects ($p < 0.05$ and deviation $> 0.08$).
- **What-If Scenario Simulation Tool** ([`src/tools/scenario_simulation.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/scenario_simulation.py), [`src/core/tool_registry.py`](file:///d:/(Dev2)DSA_AGENT/src/core/tool_registry.py)):
  - Implemented `ScenarioSimulationTool(BaseTool)`: models parametric continuous driver shifts (+5%, +10%, +25%, -10%), population-level impact, and 95% confidence intervals, plus segment equalization opportunity. Emits `decision_scenario` findings and Markdown decision tables.
- **Action Prioritization Matrix (2x2)** ([`src/core/findings.py`](file:///d:/(Dev2)DSA_AGENT/src/core/findings.py), [`src/tools/report_generator.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/report_generator.py), [`src/core/html_report.py`](file:///d:/(Dev2)DSA_AGENT/src/core/html_report.py)):
  - Implemented `classify_action_matrix()` sorting findings into 4 quadrants: `quick_wins` (High Impact, High Feasibility), `strategic_bets` (High Impact, High Complexity / Structural), `incremental` (Moderate Impact, High Feasibility), and `watch_caveats` (Confounders / Sensitive / Fragile).
  - Formatted into Markdown decision table in `GenerateReportTool` and responsive styled cards in `generate_html_report()`.
- **Verification & Tests**:
  - Unit tests added: [`tests/test_scenario_simulation.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_scenario_simulation.py) (5 passed), [`tests/test_action_matrix.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_action_matrix.py) (3 passed), plus regression tests in `test_sensitivity.py` (10 passed), `test_segment_discovery.py` (4 passed), `test_report_generator.py` (15 passed), `test_html_report.py` (14 passed), and `test_architecture.py` (3 passed).
  - Quality gates: `ruff check .` (0 violations), `mypy src/` (0 errors across 107 source files).

### Q. Autonomous Unprompted Mode Upgrades (2026-10-12) [Task BE-10]
- **North Star Inference & Autonomous Objective Synthesis** ([`src/core/controller.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller.py)):
  - Implemented `_synthesize_autonomous_objective(profile, agenda)` selecting currency, count, conversion, or top variance measures and key categorical dimensions.
  - Automatically synthesizes and stores `autonomous_objective` in `MemorySystem` when `user_objective` is empty.
  - Automatically promotes the #1 ranked investigative question from `build_agenda()` into `HypothesisTree` as the Lead Hypothesis (`parent_id="H1"`).
- **Prompt Manager Autonomous Mission Injection** ([`src/core/prompt_manager.py`](file:///d:/(Dev2)DSA_AGENT/src/core/prompt_manager.py)):
  - Updated `_objective_block()` to inject `## Analysis Mission (Autonomous Discovery)` when no natural language prompt was provided, giving the LLM planner and synthesizers a grounded, domain-specific North Star instead of drifting aimlessly.
- **Decision Intelligence Pre-Seeding in Fallback Plan** ([`src/core/controller_planning.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_planning.py)):
  - Pre-seeded `scenario_simulation` alongside `segment_comparison` in `_build_fallback_plan()`, ensuring What-If driver shifts and segment gap recovery opportunities are always quantified in unprompted mode.
- **Reporting & Run View Surfacing** ([`src/core/controller_report.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_report.py), [`src/core/run_view.py`](file:///d:/(Dev2)DSA_AGENT/src/core/run_view.py), [`src/tools/report_generator.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/report_generator.py)):
  - Forwarded `autonomous_objective` into `build_run_view()`, Markdown report header, and HTML report header so unprompted runs clearly state the autonomous investigation mission rather than `(none stated)`.
- **Verification & Tests**:
  - Unit tests added in [`tests/test_autonomous_mode.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_autonomous_mode.py) (4 passed).
  - Quality gates: `ruff check .` (0 violations), `mypy src/` (0 errors across 107 source files).


