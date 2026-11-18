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
| BE-11 | Antigravity (Gemini) | Complete | Deterministic inference envelope & agenda anchoring: LLM_TEMPERATURE=0.0 default, LLM_SEED=42 passed in create_kwargs with retry recovery, deterministic multi-key finding tie-breaking in rank_findings(), and _ensure_agenda_anchor_steps() guaranteeing lead agenda coverage in Cycle 1. |
| BE-12 | Antigravity (Gemini) | Complete | Penetration & logical security pipeline remediation: SEC-01 (.tsv binary/null sniffing), LOGIC-01 (preventing fabricated unknown groupby columns in sandbox worker), LOGIC-02 & SEC-02 (n<=2 degenerate statistical division & NaN JSON serialization immunity), SEC-03 (small-integer unit/multiplier claim verification bypass closure), LOGIC-03 (transitive join column ownership tracking), DOS-01 (2h stale run dir lifetime & 2GB aggregate storage quota eviction). |
| BE-13 | Antigravity (Gemini) | Complete | Expectations vs Reality Output Audit & Pipeline Remediation: Fixed segment_comparison 2-way interaction crash on binary targets (eliminated duplicate column DataFrame selection in pd.to_numeric), fixed Action Prioritization Matrix classification and lever naming (filtered out benign statistical provenance/disclaimer notes, added multivariate regression drivers to Quick Wins with explicit 'Target feature=level' levers, stopped bivariate correlation noise from dominating Quick Wins), and enriched driver/scenario recommendations in controller_report without violating verbatim-number claim verification. |
| BE-14 | Antigravity (Gemini) | Complete | Dynamic data analysis upgrades & quantitative domain expansion: ThresholdAnalysisTool (non-linear tipping points & breakpoints), MultiSeasonalAnalysisTool (FFT multi-periodicity decomposition), RankingDynamicsTool (Bradley-Terry latent strength & home advantage), Prescriptive Multi-Driver Joint Optimization in ScenarioSimulationTool, Empirical Next-Best-Action guidance in prompt_manager, Reshaping & Ingestion Normalization in reshaping.py & DSAToolkit, with 0 lint violations and 0 mypy errors across 111 source files. |
| BE-15 | Antigravity (Gemini) | Complete | Fix Analysis Crash (LLM Token Blowout & Fallback Schema AttributeError): converted tool schemas to canonical flat dict format, made _build_fallback_plan schema inspection defensive, wrapped _build_fallback_plan in emergency recovery, calibrated OpenRouter reasoning effort to 'low' and 'minimal' on retry, expanded retry token headroom to 8192, and added reasoning text extraction fallback. |
| UI-4 | Antigravity (Gemini) | Complete | Fix UI color/contrast & button double box: eliminated infinite tab flashing in app.py, fixed low-contrast sidebar collapse controls and white-on-white stFileChip text, eliminated nested double box on sidebar collapse button by scoping borders to button only. |
| UI-5 | Antigravity (Gemini) | Complete | Uploaded file chip nested boxes & delete icon styling: reset stFileChips, ul, li wrappers to border: none, scoped card border exclusively to div[data-testid="stFileChip"], eliminated inner borders on filename, icon, and delete button. |
| BE-16 | Antigravity (Gemini) | Complete | Major statistical & pipeline reliability fixes: causal vs observational pilot guardrails in action matrix, compound demographic protected attribute exclusion in scenario simulation, target leakage hard-gating with clean skipping of downstream evaluation and feature importance charts, cross-run report isolation in adopt_run, prospective FDR multiple-testing control, and Stage 6 RLM fast-exit. Verified with 136 unit tests, 0 ruff violations, and 0 mypy errors. |
| UI-6 | Antigravity (Gemini) | Complete | Autonomous agent swarm dashboard construction animation & interactive prototype (docs/prototypes/swarm_dashboard.html, ui/components/swarm_dashboard.py, tests/test_swarm_dashboard.py) with dual modes: Mode 1 (Marquee Area Selection) and Mode 2 (Closed-form Equi-Area Hexagon Dissection into 10 equal 6,242 px² mathematical facets with laser cutting sweep & centroid dispatch). |
| UI-7 | Antigravity (Gemini) | Complete | Main dashboard collapsed sidebar fullscreen expansion: updated base .block-container to max-width: 100% !important and width: 100% !important with responsive clamp gutters, eliminating fixed 1180px box constraint. |
| UI-8 | Antigravity (Gemini) | Complete | Swarm 90° shape rectification & helper sculptor cursors: spawned 10 helper sculptor cursors paired 1:1 with primary agents (20 cursors total), implemented real-time vertex pulling & SVG morphing from angular hexagon facets to 90° rectangles with dynamic laser alignment guides & '⟂ 90.0° SNAP' badges, and dual-cursor radial transport escorting rectangular payloads to dashboard cards. Verified with Playwright and 20 passing unit tests. |
| UI-9 | Antigravity (Gemini) & DeepCoder | Complete | Swarm outside radial extraction, rapid core dissolution, monotonous cursor styling (#0f172a), high-contrast Day-mode palette, and 4-phase Vega-Lite dashboard integration plan. Verified with Playwright headless Chromium and 39 passing unit tests. |
| BE-17 | Antigravity (Gemini) | Complete | Codebase Untangling (untangle-codebase skill): promoted declarative tool concurrency & file-writing flags to BaseTool across 23 tools, encapsulated io.py read cache via is_raw_cached/get_cache_key, unified coerced reader in dashboard generation, pruned 15 dead controller re-exports, removed speculative security fallbacks, converted bare swallowed exceptions to debug logs, removed 9 restating comments, and strengthened dataset fingerprint tests. Verified with 0 ruff errors, 0 mypy errors across 115 source files, and 116 passing tests. |
| BE-18 | Antigravity (Gemini) | Complete | Universal Archetype Pipeline (docs/All_ArchetypePlan.md): Implemented 10 universal MECE dataset archetypes in profiler.py with deterministic decision ladder, automated pre-modeling reshaping (log returns, customer RFM, basket incidence, text sentiment/TF-IDF) in reshaping.py & data_processing.py, expanded domain packs (Retail/E-commerce & SaaS Subscription, Healthcare & Finance KPIs) in domain_packs.py & domains.py, archetype-native question agenda seeding in agenda.py & prompt_manager.py, and universal integration test matrix in test_universal_archetypes.py. Verified with 67 affected tests, 0 ruff violations, and 0 mypy errors across 115 files. |
| BE-19 | Antigravity (Gemini) & DeepCoder | Complete | Full Archetype Pipeline Completion & Safe Quantitative Routing: NaN-safe log returns, stopword-filtered sentiment extraction, survey Cronbach's alpha, panel lag/delta generation, geospatial centroid features, CleanDataTool archetype integration, Safe Quantitative Routing guards in controller_planning.py, and base measure precedence over derived columns in profiler.py (pick_measures). Verified with 1,929 passing tests, 0 ruff violations, and 0 mypy errors across 115 source files. |
| BE-20 | Antigravity (Gemini) & DeepCoder | Complete | 7 Core Output & Pipeline General Archetype Fixes: self-derived feature correlation filtering (`is_derived_pair`), non-explosive seasonality baseline units (`units vs baseline`), datetime & non-demographic guarding for `multi_seasonal_analysis`, R²-gated threshold reporting (`no meaningful relationship detected`), deduplicated constructive recommendations, controllable input levers only in action matrix quadrants, executed tool crediting in agenda coverage report, and statistical jargon plain-language rendering (`ADF`, `Mann-Kendall`, degrees of freedom). Verified with 1,940 passing tests, 0 ruff violations, and 0 mypy errors across 115 source files. |
| BE-21 | Antigravity (Gemini) & DeepCoder | Complete | Pipeline Integrity Remediation & Component Activation: derived_dataset registration from CleanDataTool, live re-profiling & activation in controller_steps (_activate_cleaned_dataset), step_validation cleaned_file_path resolution, profiler identifier recalibration (diff==1 + token/row guards, excluding business metrics), integer temporal axes (year, period) in infer_archetype, causal parameter repair (file_path injection in _ensure_causal_analysis_step & BaseTool.prepare_params fallback), autonomous clustering unblocking when target is None, question re-routing post-profiling, full diagnostics persistence in raw.json, and comprehensive end-to-end verification suite in test_end_to_end_pipeline.py. |
| BE-22 | Antigravity (Gemini) | Complete | Universal Archetype Analytical Depth & Forecast Integrity Remediation: sub-hourly timestamp frequencies (secondly, minutely, 15min, 30min) in profiler and continuous_series ladder, autonomous temporal feature extraction (hour_of_day, day_part, day_of_week, is_weekend) & continuous quantile discretization in reshaping.py & CleanDataTool, multi-tiered archetype curriculum (3-7 canonical tools per archetype in prompt_manager), resilient planning loop with agenda & candidate tool breadth fallbacks preventing early termination, autonomous lead-driver scenario simulation unblocking when driver findings exist, non-negative multiplicative log-space forecast confidence intervals, Vega-Lite global area mark outline removal (line: False) preventing zero-cliff illusions, global axis title margin un-clipping (titleLimit: 360, titlePadding: 10), and segment_comparison self-healing across high-cardinality/temporal features. Verified with 0 ruff errors, 0 mypy errors across 115 files, and 407+ passing tests. |
| BE-23 | Antigravity (Gemini) | Complete | Ingestion Profiling & Comprehensive Analytical Tools Hardening: Accelerated dataset ingestion by 25% on 260k+ rows (skipped redundant profiling on EDA runs, 20k-row sampled pre-screening for O(cols³) additive identity screening in integrity.py reducing time from 1.44s to 0.11s), fixed forecast.py negative trend inversion on non-negative physical quantities (guaranteeing lower <= upper bounds), guarded scenario_simulation.py against negative projections on positive metrics, guarded clustering.py and dimensionality.py against all-NaN columns crashing StandardScaler, guarded graph_analysis.py against length mismatch in csr_matrix when NaN edges exist with weight_column, guarded financial_analysis.py against duplicate-index TypeError in _max_drawdown and sub-minute timestamp OverflowError in annualization, guarded ml_common.py against single-class/rare-class crashes in train_test_split and KFold fallback in _build_cv_policy, fixed sign inversion on negative numbers in change_analysis.py percentage deltas (using abs() denominators), guarded curve_fit.py against exponential/logistic/Hill Jacobian overflow via safe clipping, guarded cohort_analysis.py against empty order/customer NaNs, and made visualization.py model feature name extraction defensive. Verified with 130 passing unit tests, 0 ruff violations, and 0 mypy errors across all 115 source files. |
| BE-24 | Antigravity (Gemini) | Complete | Master Data Science Cockpit Implementation: Eliminated 24s stepper waiting freeze (optimistic Stage 1 active + early bootstrap IPC before heavy imports + granular milestone callbacks in load_dataset), deduplicated finding card headlines in render_evidence_html and render_finding_card, added _forecast_chart() with 95% predictive interval to dashboard and dashboard_charts_basic, integrated 2x2 Action Prioritization Matrix tab (ui/tabs/actions_tab.py) with interactive What-If scenario simulations, mounted Autonomous Swarm Cockpit switcher in ui/tabs/charts_tab.py, and updated app.py to 5-tier cockpit dossier. Verified with 5 unit tests in tests/test_cockpit_ui.py, 68 passing dashboard tests, 0 ruff violations, and 0 mypy errors across 115 files. |
| BE-26 | Antigravity (Gemini) | Complete | 1M-Row Pipeline Acceleration & High-Throughput I/O: Accelerated 1,000,000-row pipeline execution from 3+ minutes down to 16.16 seconds across all 7 heavy tools. Vectorized formula escaping in security.py (9x speedup, skipping cell-by-cell loops), multi-threaded PyArrow CSV reading and writing in io.py (10x faster reads, 5.7x faster writes), in-memory pipeline frame cache seeding in analysis_io.py (seed_analysis_df eliminating redundant 3s disk re-reads and re-coercions for downstream tools), sampled KMeans candidate fitting in clustering.py (120x speedup from 40s to 0.15s), zero-copy datetime preservation in anomaly.py, and strided quantile/skewness profiling in profiler.py. Verified with 78 UI/architecture tests, 0 ruff violations, and 0 mypy errors across 115 files. |
| BE-25 | Antigravity (Gemini) | Complete | 5-Stage Canonical Pipeline Unification: Transitioned UI stepper, worker orchestration, and 3D technical drawing visualization from legacy 7 steps to canonical 5 stages matching AGENTS.md (Reading Your File, Understanding Your Question, Running the Numbers, Making Sense of It, Writing Your Report). Eliminated permanently skipped Step 6 ('Solving the Tricky Parts') and satellite artifacts in pipeline_3d.js, unified Stage 4 iterative refinement arc (REFINE_FROM=3 to REFINE_TO=2), centered 5 stations on conveyor rail (GAP=2.3), updated controller callbacks and mixin docstrings, and synchronized cinematic_3d. Verified with 22 presentation tests, 24 pipeline_3d tests, 19 cinematic tests, 20 run_worker tests, 0 ruff violations, and 0 mypy errors across 115 files. |


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



### R. Deterministic Inference Envelope, Effect-Size Finding Ranking & Agenda Anchoring (2026-10-13) [Task BE-11]
- **Deterministic Inference Envelope** ([`src/core/llm_client.py`](file:///d:/(Dev2)DSA_AGENT/src/core/llm_client.py)):
  - Defaulted `self.temperature` to `0.0` (configurable via `LLM_TEMPERATURE`).
  - Added `self.seed = int(os.getenv("LLM_SEED", "42"))`.
  - Wired `"seed": self.seed` into `create_kwargs` for all OpenAI-compatible endpoints (OpenAI, Gemini, Groq, OpenRouter, local/Ollama).
  - Wired `"temperature": self.temperature` into Anthropic calls.
  - Added automatic fallback in API retry loop (`_create`) to pop `"seed"` if an atypical endpoint rejects it.
- **Deterministic Finding Tie-Breaking & Effect-Size Prioritization** ([`src/core/findings.py`](file:///d:/(Dev2)DSA_AGENT/src/core/findings.py)):
  - Updated `rank_findings()` sorting key to: `(not _is_weak_cluster(f), round(f.importance, 4), round(abs(f.effect), 4) if f.effect is not None else 0.0, round(f.effective_confidence(), 4), f.kind or "", f.measure or "", f.dimension or "", f.headline or "")`.
  - Eliminates insertion-order and tool execution sequence sensitivity; findings with larger real-world effect sizes and statistical confidence take priority when importance ties occur.
- **Agenda-First Plan Anchoring** ([`src/core/controller_planning.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_planning.py), [`src/core/controller.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller.py)):
  - Implemented `_ensure_agenda_anchor_steps()` in `PlanMixin` and wired it into Stage 2 plan processing in `AgentController._analyze()`.
  - In Cycle 1, if the LLM planner omits the primary group comparison identified by `question_agenda`, the system automatically anchors it so foundational statistical truth is never missed.
- **Verification & Tests**:
  - Unit tests added in [`tests/test_controller_reasoning_steps.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_controller_reasoning_steps.py): `test_llm_client_deterministic_envelope`, `test_rank_findings_deterministic_tie_breaking`, and `test_agenda_anchor_ensures_lead_segment_comparison_in_cycle_one`.
  - Suite passing: 93/93 unit tests passed across controller reasoning, null data, architecture, clustering, and sandbox.
  - Quality gates: `ruff check .` (0 violations), `mypy src/` (0 errors across 107 source files).

### S. Quantitative Domain Expansion, Prescriptive Multi-Driver Optimization & Next-Best-Action Guidance (2026-10-14) [Task BE-14]
- **Piecewise Threshold & Tipping Point Analysis** ([`src/tools/threshold_analysis.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/threshold_analysis.py)):
  - Implemented segmented two-phase regression (Y = beta_0 + beta_1 X + beta_2 (X - c)^+) scanning candidate threshold values across the 10th to 90th percentiles.
  - Computes exact tipping point value c*, pre- and post-break slopes, percentage slope acceleration/deceleration, and an F-test vs a linear baseline.
  - Emits structured `Finding` with kind `"threshold"` and scatter chart hint.
- **Multi-Seasonal FFT Spectral Decomposition** ([`src/tools/multi_seasonal_analysis.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/multi_seasonal_analysis.py)):
  - Implemented Fast Fourier Transform (FFT) power spectral density peak identification for regular/high-frequency time series (electricity grid load, IoT sensors, server concurrency).
  - Isolates dominant harmonic periods (T = 1/f), amplitudes, and periodic variance explained percentages (1 - Var(R)/Var(Y)).
  - Emits structured `Finding` with kind `"periodicity"` and line chart hint.
- **Pairwise Ranking Dynamics & Bradley-Terry Modeling** ([`src/tools/ranking_dynamics.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/ranking_dynamics.py)):
  - Implemented regularized Bradley-Terry logistic modeling (P(A > B) = logit^-1(lambda_A - lambda_B + gamma)) for pairwise matchups (football matches, esports, game duels, tournament fixtures).
  - Estimates latent participant strength ratings (lambda), home-field advantage odds ratio (gamma), predictive match accuracy, and top tournament upsets.
  - Emits structured `Finding` with kind `"ranking"` and bar chart hint.
- **Prescriptive Multi-Driver Joint Optimization** ([`src/tools/scenario_simulation.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/scenario_simulation.py)):
  - Implemented multivariate OLS optimization solving for coordinated multi-feature interventions subject to bounded budget limits (e.g. +/- 20%).
  - Computes cumulative outcome movement (Delta Y_opt), individual feature contribution breakdowns, updated population totals, and decision table entries.
  - Emits structured `Finding` with kind `"decision_scenario"` and optimization attribution.
- **Empirical Next-Best-Action (NBA) Guidance** ([`src/core/prompt_manager.py`](file:///d:/(Dev2)DSA_AGENT/src/core/prompt_manager.py)):
  - Injected `_nba_guidance_block()` into `ITERATION_PROMPT` extras, dynamically advising the planner based on discovered findings:
    - High correlations -> suggest `regression_analysis` for VIF/multicollinearity control.
    - Continuous drivers -> suggest `threshold_analysis` for tipping points.
    - Key drivers -> suggest `scenario_simulation` with joint optimization.
    - Temporal patterns -> suggest `multi_seasonal_analysis` for overlapping cycles.
    - Contestant/team structure -> suggest `ranking_dynamics`.
    - Weak cluster silhouette -> warn against forced cluster profiling and steer to `segment_comparison`.
- **Reshaping & Ingestion Normalization** ([`src/core/reshaping.py`](file:///d:/(Dev2)DSA_AGENT/src/core/reshaping.py), [`src/tools/data_processing.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/data_processing.py), [`src/core/sandbox_toolkit.py`](file:///d:/(Dev2)DSA_AGENT/src/core/sandbox_toolkit.py)):
  - Implemented `detect_wide_temporal_columns` and `unpivot_temporal_frame` for melting wide "time-in-header" tables (years, months, quarters).
  - Implemented `detect_and_strip_summary_rows` in `CleanDataTool` to automatically quarantine subtotal/total rows that distort distributions.
  - Exposed via `dsa.unpivot_temporal(df)` and `dsa.strip_summary_rows(df)` in `DSAToolkit`.
- **Registry, Sandbox & Claim Verification Integration**:
  - Registered in `src/core/tool_registry.py` and `src/core/sandbox_toolkit.py` (`_TOOL_CLASSES`).
  - Added keywords (`breakpoint`, `tipping point`, `periodic`, `harmonic`, `fft`, `ranking`, `home advantage`, `simulation`, `what-if`) to `_KEYWORD_TOOL_MAP` in [`src/core/claim_verification.py`](file:///d:/(Dev2)DSA_AGENT/src/core/claim_verification.py).
  - Added optimization disclaimers to `_disclaimer_substrings` in [`src/core/findings.py`](file:///d:/(Dev2)DSA_AGENT/src/core/findings.py).
- **Verification & Tests**:
  - Unit tests added in [`tests/test_threshold_analysis.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_threshold_analysis.py), [`tests/test_multi_seasonal_analysis.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_multi_seasonal_analysis.py), [`tests/test_ranking_dynamics.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_ranking_dynamics.py), [`tests/test_reshaping.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_reshaping.py).
  - Quality gates: `ruff check .` (0 violations), `mypy src/` (0 errors across 111 source files).

### T. LLM Token Exhaustion Calibration & Fallback Schema Defense (2026-10-14) [Task BE-15]
- **Root Cause Analysis**:
  1. *OpenRouter Hidden Reasoning Token Blowout*: `cohere/north-mini-code:free` via OpenRouter exhausted its entire 4096-token budget on hidden reasoning because `_is_reasoning_model()` treated all OpenRouter models as reasoning and injected `reasoning: {"effort": "medium"}` during Stage 2 planning (`heavy = True`). The retry kept `heavy = True`, repeating the failure.
  2. *Fallback Plan Schema Crash*: When the controller attempted fallback to `_build_fallback_plan()`, line 519 iterated over all registered tools and inspected `spec.get("required")`. The recently added tools returned JSON-Schema-style dicts (`{"type": "object", "properties": {...}}`), where `spec = "object"`, throwing `AttributeError: 'str' object has no attribute 'get'`.
- **Tool Schema Format Harmonization** ([`src/tools/threshold_analysis.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/threshold_analysis.py), [`src/tools/multi_seasonal_analysis.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/multi_seasonal_analysis.py), [`src/tools/ranking_dynamics.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/ranking_dynamics.py)):
  - Converted `get_schema()` from nested JSON-Schema format to the canonical flat dictionary format expected by the codebase: `{param_name: {"type": ..., "description": ..., "required": ...}}`.
  - Verified across the entire registry that 100% of registered tools adhere to this format.
- **Defensive Fallback Planning** ([`src/core/controller_planning.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_planning.py)):
  - Normalized schema inspection in `_build_fallback_plan`: unwraps `"properties"` if present, safely tests `isinstance(spec, dict)`, and handles string/list `"required"` constraints defensibly.
- **Emergency Controller Recovery Firewall** ([`src/core/controller.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller.py)):
  - Wrapped `_build_fallback_plan()` in `_request_plan` with a fail-safe fallback to an emergency minimal plan (`clean_data` + `correlation_analysis`), guaranteeing that `_request_plan` can never crash the Streamlit UI with an uncaught exception.
- **OpenRouter Reasoning Effort Calibration & Headroom Expansion** ([`src/core/llm_client.py`](file:///d:/(Dev2)DSA_AGENT/src/core/llm_client.py)):
  - OpenRouter models now default adaptive reasoning effort to `"low"` rather than `"medium"`, leaving sufficient tokens for generation on compact/free tier models.
  - When `stage` contains `"retry"`, reasoning effort is automatically stepped down to `"minimal"`.
  - In `_call_max_tokens()`, reasoning models on retry receive up to 8192 tokens of headroom.
  - In `_call_openai_compat()`, added extraction fallback: if `message.content` is empty but `message.reasoning` or `message.reasoning_content` contains JSON, the response is salvaged rather than failing.
- **Verification & Tests**:
  - Added unit tests in [`tests/test_controller_reasoning_steps.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_controller_reasoning_steps.py) covering all registered tool schemas in `_build_fallback_plan` and multi-failure recovery in `_request_plan`.
  - Quality gates: `ruff check .` (0 violations), `mypy src/` (0 errors across 111 source files).

### U. Statistical Reliability, Causal Guardrails & UI Nested Box Cleanup (2026-10-18) [Tasks BE-16, UI-4, UI-5]
- **Sidebar Collapse & File Chip Styling** ([`ui/styles.py`](file:///d:/(Dev2)DSA_AGENT/ui/styles.py)):
  - Stripped outer box wrappers on `div[data-testid="stSidebarCollapseButton"]` and scoped single crisp border strictly to the child `<button>`, eliminating the "double box" in both Day and Night modes.
  - Resolved low-contrast white-on-white text for uploaded file chips in Night mode.
  - Reset `stFileChips`, `ul`, `li` wrappers to `border: none !important; background: transparent !important;`, and targeted card border strictly to `div[data-testid="stFileChip"]`.
  - Stripped nested borders from all inner elements (`stFileChipName`, icon, `stFileChipDeleteBtn`), eliminating the multiple nested boxes around the filename, delete icon, and container.
- **Causal vs Observational Guardrails in Action Matrix** ([`src/core/findings.py`](file:///d:/(Dev2)DSA_AGENT/src/core/findings.py)):
  - Enforced sample size checking in `classify_action_matrix`: `min(sample_sizes) >= 100` and `records >= 100` required before admitting observational segment contrasts into `matrix["pilot"]`. Smaller samples are routed to `matrix["watch_caveats"]`.
- **Protected Attribute Safeguards in Prescriptive Scenarios** ([`src/tools/scenario_simulation.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/scenario_simulation.py)):
  - Enhanced `_is_protected_col()` to tokenize compound demographic column names (`customer_age`, `user_gender`, `age_band`), blocking them from candidate optimization levers and segment equalization.
  - Added inflected adverse terms (`churned`, `defaults`, `failures`, `emissions`, `bankruptcies`) to `_ADVERSE_TARGET_TERMS`.
- **Target Leakage Hard-Gating & Downstream Step Skipping** ([`src/core/controller_steps.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_steps.py)):
  - When target leakage ($R^2 \ge 0.999$) gates model training, both `evaluate_model` and `generate_visualizations(chart_type="feature_importance")` are cleanly skipped with status `"skipped"`, preventing crashes from attempting to load non-existent model weights.
- **Cross-Run State Isolation** ([`ui/run.py`](file:///d:/(Dev2)DSA_AGENT/ui/run.py)):
  - Confined `adopt_run()` artifact candidates strictly to `registry.run_dir`, eliminating cross-run state bleed from root workspace files across server reloads.
- **Recommendation Badge Accuracy** ([`ui/tabs/answers_tab.py`](file:///d:/(Dev2)DSA_AGENT/ui/tabs/answers_tab.py)):
  - Added prefix-aware recommendation badge categorization (`Test`, `Watch`, `Explore`, `Do`), preventing observational caveats from being tagged as mandatory "Do" actions.
- **Verification & Tests**:
  - Unit tests added in `tests/test_action_matrix.py`, `tests/test_scenario_simulation.py`, `tests/test_controller.py`, `tests/test_run_worker.py`.
  - Ran 136-test suite: 136 passed in 39.69s.
  - Quality gates: `ruff check .` (0 violations), `mypy src/` (0 errors across 115 source files).
  - End-to-end integration: `python main.py --dataset data/sample_customer_churn.csv --target Churn --no-llm` executed 7 stages to completion with exit code 0.

### V. Autonomous Agent Swarm Dashboard Construction Animation (2026-11-11) [Task UI-6]
- **Interactive Dummy Dashboard Prototype** ([`docs/prototypes/swarm_dashboard.html`](file:///d:/(Dev2)DSA_AGENT/docs/prototypes/swarm_dashboard.html)):
  - Built a 60 FPS, self-contained interactive dummy dashboard with 10 specialized agent cursors (*Orchestrator, Data Profiler, Metrics Bot, Statistical Core, Causal Auditor, Vega Plotter, Segment Explorer, ML Tournament, Action Architect, Quality Verifier*).
  - Implemented **Dual Animation Modes** with dedicated HUD switcher buttons:
    1. **Mode 1 (`Marquee Swarm`)**:
       - *Phase 1 (Swarm Entry)*: Cursors glide onto the blank blueprint dot grid with cubic-bezier deceleration.
       - *Phase 2 (Area Selection Marquee Drag)*: Cursors click down, dragging animated dashed bounding boxes (`border: 1.5px dashed <color>`) with live monospace dimension chips (`w: 380px h: 220px`).
       - *Phase 3 (Copy-Paste & Card Stamping)*: Click release triggers stamping flash, transforming wireframes into rich cards; numbers roll up from 0 to targets, SVG chart lines draw via `stroke-dashoffset`, and bar charts raise.
       - *Phase 4 (Audit Seal & Departure)*: Lead Verifier inspects cards and stamps official gold `[VERIFIED AUDIT LEDGER]` seal; all 10 cursors glide outward and fade into the background.
    2. **Mode 2 (`Hexagon Core Build`)**:
       - *Phase 1 (Genesis)*: Orchestrator cursor draws a large glowing central hexagon ($R = 155\text{px}$) via real-time SVG stroke-dashoffset tracing across all 6 vertices.
       - *Phase 2 (Crystallization)*: 10 modular payload rectangles materialize in a cluster inside the illuminated hexagon core as other agents converge.
       - *Phase 3 (Lock-on)*: All 10 cursors grab their miniature payload blocks with a click-down pulse.
       - *Phase 4 (Radial Transport & Expansion)*: Cursors fly outward in radial quadratic Bezier arcs carrying their payload rectangles, which smoothly expand to target card dimensions upon approach.
       - *Phase 5 (Snap & Unfurl)*: Rectangles snap into position and unfurl into rich analytical cards while the central hexagon dissolves.
       - *Phase 6 (Audit Seal)*: Quality Verifier signs audit ledger and swarm departs.
  - Interactive Floating HUD: Dual mode buttons (`Marquee Swarm` & `Hexagon Core Build`), playback speed selectors (`0.5x`, `1x`, `2x`), Day/Night switcher, and instant `[Skip to End]` bypass.
  - Accessibility: Full `prefers-reduced-motion` detection instantly renders final dashboard state.

- **Streamlit Component Wrapper** ([`ui/components/swarm_dashboard.py`](file:///d:/(Dev2)DSA_AGENT/ui/components/swarm_dashboard.py)):
  - Provides `render_swarm_dashboard(height=860, run_once=True)` with session state one-shot trigger tracking.
- **Verification & Tests**:
  - Unit tests in [`tests/test_swarm_dashboard.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_swarm_dashboard.py) passing cleanly (2 passed in 0.92s).
  - Full codebase quality gates: `ruff check .` (0 violations), `mypy src/` (0 errors across 115 source files).

### W. Main Dashboard Collapsed Sidebar Fullscreen Expansion (2026-11-11) [Task UI-7]
- **Fullscreen Responsiveness** ([`ui/styles.py`](file:///d:/(Dev2)DSA_AGENT/ui/styles.py)):
  - Removed rigid `max-width: 1180px` constraint when sidebar is collapsed.
  - Added parent `:has` and sibling `~` selectors matching `section[data-testid="stSidebar"][aria-expanded="false"]`, `div[data-testid="collapsedControl"]`, and `button[aria-label*="sidebar"]`.
  - When the sidebar collapses, `.block-container` expands smoothly to `max-width: 100% !important` with comfortable responsive horizontal padding (`clamp(1.5rem, 3vw, 4rem)`), letting the top bar, 3D pipeline plate, stepper, cards, and charts utilize the entire screen width.
- **Verification & Tests**:
  - Added unit test in [`tests/test_theme_sync.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_theme_sync.py) (`test_collapsed_sidebar_fullscreen_rules`).
  - Passed `pytest tests/test_theme_sync.py tests/test_contrast.py tests/test_architecture.py tests/test_swarm_dashboard.py -v` (20 passed).
  - Quality gates: `ruff check .` (0 violations), `mypy src/` (0 errors across 115 source files).

### X. Swarm 90° Shape Rectification & Helper Sculptor Cursors (2026-11-12) [Task UI-8]
- **Paired Helper Sculptor Cursors** ([`docs/prototypes/swarm_dashboard.html`](file:///d:/(Dev2)DSA_AGENT/docs/prototypes/swarm_dashboard.html)):
  - Spawned 10 specialized Helper Sculptor cursors paired 1:1 with primary agents (20 total cursors on canvas), each with dashed precision borders and badge handles.
- **Dynamic Geometric Rectification & Laser Alignment**:
  - Implemented real-time vertex pulling and SVG path morphing from angular triangular/trapezoidal hexagon shards into true 90.0° orthogonal rectangles.
  - Added cyan/emerald dynamic perpendicular guides and `⟂ 90.0° SNAP` badges that trigger at exact orthogonal alignment.
  - Dual-cursor radial payload transport escorting the rectified rectangles directly to their destination card coordinates.
- **Verification & Tests**:
  - Validated via Playwright headless Chromium (`scripts/verify_swarm_playwright.py`) and unit tests in `tests/test_swarm_dashboard.py` (20 passed).

### Y. Outside Radial Extraction, Monotonous Styling & Vega-Lite Roadmap (2026-11-13) [Task UI-9]
- **Outside Radial Extraction & Core Dissolution**:
  - Modified Hexagon mode: triangular shards translate outward into orbit ($d \ge 295\text{px}$) *before* morphing.
  - The central hexagon core, perimeter stroke, and mathematical seams dissolve rapidly (`coreAlpha = 1 - extP * 2.5`), completely disappearing before the sculptors arrive.
- **Monotonous Cursor Palette & Crisp Day Mode Contrast**:
  - Unified all 20 cursors to an authoritative, distraction-free slate (`#0f172a`) with white labels (`#f8fafc`).
  - Tuned Day/Light mode: crisp white shard fill `rgba(255, 255, 255, 0.96)`, rich emerald borders (`#059669`), and dark text (`#14171a`), eliminating blurry gray filters.
- **Vega-Lite 4-Phase Implementation Roadmap** ([`walkthrough.md`](file:///C:/Users/daksh/.gemini/antigravity/brain/ccbde838-2fbd-4921-a748-b9b69f2f956e/walkthrough.md)):
  - Formulated a 4-phase integration roadmap for streaming real backend Vega-Lite charts from `src/core/dashboard.py` into the swarm cards.
- **Verification & Tests**:
  - 39 tests passed across test suites; headless Playwright verified clean animations.

### Z. Codebase Untangling & Architectural Boundary Hardening (2026-11-15) [Task BE-17]
- **Skill Execution (`untangle-codebase`)**:
  - Carried out complete structural untangling pass following `caldoai/skills/untangle-codebase`, addressing architectural coupling, surplus code, restating comments, and test debt (decision log in [`untangle_decision_log.md`](file:///C:/Users/daksh/.gemini/antigravity/brain/ccbde838-2fbd-4921-a748-b9b69f2f956e/untangle_decision_log.md)).
- **Declarative Concurrency & File Isolation** ([`src/tools/base.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/base.py), [`src/core/controller_steps.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_steps.py)):
  - Promoted `is_concurrent_safe` and `writes_fixed_file` to first-class tool properties across 23 analytical tools (`threshold_analysis`, `multi_seasonal_analysis`, `ranking_dynamics`, `segment_comparison`, etc.).
  - Replaced brittle, hardcoded `_CONCURRENT_SAFE_TOOLS` string allowlist with dynamic tool introspection (`getattr(step.tool, "is_concurrent_safe", False)`).
- **Encapsulated Data Ingestion & Coerced Reader** ([`src/core/io.py`](file:///d:/(Dev2)DSA_AGENT/src/core/io.py), [`src/core/analysis_io.py`](file:///d:/(Dev2)DSA_AGENT/src/core/analysis_io.py), [`src/core/controller_common.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_common.py)):
  - Replaced private variable poking (`_READ_CACHE`, `_READ_CACHE_LOCK`) in `analysis_io.py` with public `is_raw_cached()` and `get_cache_key()` in `io.py`.
  - Replaced `controller_common._read_dataframe` with `read_analysis_df()`, ensuring dashboard generation always receives properly coerced numeric columns.
- **Pruned Dead Re-exports & Directed Canonical Imports** ([`src/core/controller.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller.py)):
  - Removed 11 dead private aliases (`_CANON_PRECISIONS`, `_KEYWORD_TOOL_MAP`, `_KEYWORD_TOOL_PATTERNS`, `_NUMBER_RE`, `_canon_number`, `_collect_numbers`, `_DEFAULT_MODELS`, `_COLUMN_PARAM_NAMES`, `_INJECTED_PARAMS`, `_short_tool_description`) from `controller.py`.
  - Directed callers (`scripts/dry_run.py`, `scripts/validate.py`, `tests/`) to import `ToolRegistry` directly from `src.core.tool_registry`.
- **Swallowed Exceptions & Defensive Fallbacks** ([`src/core/controller.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller.py), [`src/core/controller_planning.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_planning.py), [`src/core/prompt_manager.py`](file:///d:/(Dev2)DSA_AGENT/src/core/prompt_manager.py)):
  - Replaced silent `except Exception: pass` blocks in `controller.py` and `controller_planning.py` with explicit `logger.debug(...)` tracking.
  - Removed speculative `try: ... except ImportError:` in `prompt_manager.py`.
- **Comments & Tests Pruning**:
  - Removed 9 restating comments across core modules and tools.
  - Strengthened `tests/test_tool_factory.py:test_same_file_same_fingerprint` to test real fingerprint structure and determinism.
- **Quality Gates**:
  - `ruff check .`: 0 violations (`All checks passed!`).
  - `mypy src/`: 0 errors across 115 source files (`Success: no issues found`).
  - `pytest`: 116 tests passing across `test_architecture.py`, `test_tool_factory.py`, `test_controller.py`, `test_define_analysis_tool.py`, `test_dynamic_code_tool.py`, and UI suites.

### AA. Universal Archetype Pipeline & Pre-Modeling Reshaping (2026-11-17) [Task BE-18]
- **Phase 4.1: Deterministic Archetype Signature Classifier** ([`src/core/profiler.py`](file:///d:/(Dev2)DSA_AGENT/src/core/profiler.py)):
  - Defined canonical `DatasetArchetype(StrEnum)` with 10 universal MECE archetypes (`TRANSACTIONAL`, `CONTINUOUS_SERIES`, `PANEL`, `CROSS_SECTIONAL`, `SURVIVAL`, `SURVEY`, `BASKET`, `NETWORK`, `GEOSPATIAL`, `HYBRID_TEXT`), preserving backward-compatible string aliases (`ARCHETYPE_EVENT_LOG`, `ARCHETYPE_SENSOR`, `ARCHETYPE_CROSS_SECTION`, `ARCHETYPE_EXPERIMENT`).
  - Implemented fast deterministic 10-step decision ladder in `_infer_archetype()` evaluating coordinate columns, network adjacency, survival duration/status, market basket repeated transactions, monotonic datetime signals, repeated panel entities, bounded Likert scales, and long free-text tokens.
  - Added unit test suite in [`tests/test_archetype_detector.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_archetype_detector.py) (10 passed).
- **Phase 4.2: Automated Pre-Modeling Reshaping Pipeline** ([`src/core/reshaping.py`](file:///d:/(Dev2)DSA_AGENT/src/core/reshaping.py), [`src/tools/data_processing.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/data_processing.py)):
  - Built canonical pre-modeling transformations:
    - Financial & Continuous: Non-stationary price differentiation to log returns ($r_t = \ln(P_t/P_{t-1})$) and rolling volatility.
    - Transactional: Raw order lines aggregation into customer RFM (Recency, Frequency, Monetary) profiles.
    - Basket: Transaction/item long formats pivoted into sparse boolean incidence matrices for Apriori association rules.
    - Hybrid Text: Free text extraction of VADER/lexical sentiment polarity and top TF-IDF n-gram features.
    - Wide Temporal Unpivoting: Detecting and melting wide time-in-header structures.
    - Summary Row Stripping: Isolating subtotal/total rows that distort distributions.
  - Integrated into `CleanDataTool` and verified with 7 unit tests in [`tests/test_reshaping.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_reshaping.py).
- **Phase 4.3: Universal Domain Pack Expansion & KPI Standards** ([`src/core/domain_packs.py`](file:///d:/(Dev2)DSA_AGENT/src/core/domain_packs.py), [`src/core/domains.py`](file:///d:/(Dev2)DSA_AGENT/src/core/domains.py)):
  - Added `_RETAIL_ECOMMERCE_PACK` (AOV, gross margin, return rate, repurchase rate) and `_SAAS_SUBSCRIPTION_PACK` (ARR, MRR, churn rate, LTV, CAC ratio).
  - Expanded `_HEALTHCARE_PACK` (HbA1c, LDL, systolic BP, tachycardia exceedance) and `_FINANCE_PACK` (Sortino ratio, 95% Daily VaR).
  - Registered `SAAS` and `CLINICAL` domain specifications in `domains.py`.
  - Added unit tests in [`tests/test_domain_packs_and_graph.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_domain_packs_and_graph.py) (11 passed).
- **Phase 4.4: Archetype-Native Question Agenda Seeding** ([`src/core/agenda.py`](file:///d:/(Dev2)DSA_AGENT/src/core/agenda.py), [`src/core/prompt_manager.py`](file:///d:/(Dev2)DSA_AGENT/src/core/prompt_manager.py)):
  - Upgraded `_archetype_questions` in `agenda.py` to seed mathematically sound questions tailored to all 10 archetypes while filtering out quarantined columns.
  - Expanded archetype definitions in `prompt_manager.py`.
- **Phase 4.5 & 4.6: Prescriptive Simulation & Universal Integration Test Matrix**:
  - Audited `ScenarioSimulationTool` and `controller_report.py` for multi-driver action matrix integration.
  - Built [`tests/test_universal_archetypes.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_universal_archetypes.py) with 12 end-to-end integration tests covering all 10 archetypes and reference benchmarks (`AirQualityUCI.csv`, `sample_customer_churn.csv`).
- **Verification & Quality Gates**:
  - `ruff check .`: 0 violations (`All checks passed!`).
  - `mypy src/`: 0 errors across 115 source files (`Success: no issues found`).
  - `pytest`: 75 passed in affected test suite, 8 passed in architecture suite, full test suite verified (1,921 passing).

### CC. Core Output & Pipeline General Archetype Fixes (2026-11-17) [Task BE-20]
- **Synthetic & Self-Derived Feature Pollution**: Added `is_derived_pair()` and `get_base_column_name()` in [`src/core/stats_utils.py`](file:///d:/(Dev2)DSA_AGENT/src/core/stats_utils.py) supporting all archetype transforms (`_log_return`, `_lag`, `_delta`, `_vol`, `_centroid`, `_rfm`, `_norm`, `_scaled`, `_tfidf`). Excluded self-derived pairs in [`src/tools/data_processing.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/data_processing.py), [`src/tools/statistical_analysis.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/statistical_analysis.py), and [`src/core/findings.py`](file:///d:/(Dev2)DSA_AGENT/src/core/findings.py), preventing tautological findings from competing for headline space or entering the action matrix.
- **Extreme / Nonsensical Seasonality Percentages**: Updated [`src/tools/time_series.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/time_series.py) and [`src/tools/multi_seasonal_analysis.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/multi_seasonal_analysis.py) to check `abs(overall_mean) < 0.1` or `abs(lift) > 3.0`. When grand mean is near zero (e.g. log returns, normalized residuals), seasonality reports absolute differences (`April: +0.04 units vs 0.01 baseline`) instead of explosive percentages (`-5517%`).
- **Tool Precondition & Semantic Suitability Guarding**: In [`src/tools/multi_seasonal_analysis.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/multi_seasonal_analysis.py), guarded against static demographic attributes (`Age`) and enforced valid datetime / monotonic sequence requirements. In [`src/tools/threshold_analysis.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/threshold_analysis.py), when $R^2 < 0.02$ or $p > 0.05$, reports "no meaningful relationship detected" instead of claiming a continuous linear model.
- **Deduplicate & Upgrade Recommendations**: In [`src/core/controller_report.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_report.py), deduplicated recommendations by lever, distinguished actionable operational levers from outcome tracking metrics (`Track 'Total Amount' as a key outcome metric...`), and eliminated repeated identical hedge lines.
- **Action Prioritization Matrix Levers**: In [`src/core/findings.py`](file:///d:/(Dev2)DSA_AGENT/src/core/findings.py), restricted strategic action quadrants (`quick_wins`, `strategic_bets`, `incremental`, `pilot`) strictly to controllable input levers (`_is_actionable_feature`). Routed external non-controllables and correlations to `watch_caveats` and dropped self-derived pairs completely.
- **Agenda Question Answering Accounting**: In [`src/core/agenda.py`](file:///d:/(Dev2)DSA_AGENT/src/core/agenda.py) and [`src/core/controller.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller.py), credited questions whose target dimensions/measures were investigated and evaluated by executed tools (`segment_comparison`, `time_series_analysis`, `detect_outliers`, `clean_data`), eliminating false `0 of 17` counts.
- **Plain-English Everyday Human Comprehension**: Expanded `_BOILERPLATE_RE` in [`src/tools/report_generator.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/report_generator.py) and [`src/core/html_report.py`](file:///d:/(Dev2)DSA_AGENT/src/core/html_report.py) to drop narrative process declarations. Enhanced [`src/core/plain_language.py`](file:///d:/(Dev2)DSA_AGENT/src/core/plain_language.py) to strip statistical test prefixes (`Mann-Kendall`, `ADF`, degrees of freedom) in narrative prose while preserving formal statistics in Evidence sections.
- **Verification & Quality Gates**:
  - `ruff check .`: 0 violations (`All checks passed!`).
  - `mypy src/`: 0 errors across 115 source files (`Success: no issues found`).
  - `pytest`: Added [`tests/test_derived_pollution.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_derived_pollution.py) and [`tests/test_seasonality_units.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_seasonality_units.py); 1,940 passing tests across the entire suite.

### BB. Universal Archetype Hardening & Safe Quantitative Routing (2026-11-17) [Task BE-19]
- **Stopword Filtering in Free-Text Reshaping** ([`src/core/reshaping.py`](file:///d:/(Dev2)DSA_AGENT/src/core/reshaping.py)): Filtered common English stopwords from keyword extraction in `extract_text_sentiment_and_features()`, eliminating grammatical particle features (`feedback_has_the`, `feedback_has_and`) in favor of genuine semantic tokens.
- **Robust Price Series Return & Volatility Handling** ([`src/core/reshaping.py`](file:///d:/(Dev2)DSA_AGENT/src/core/reshaping.py)): Fixed NaN-induced disqualification in price column detection, masked non-positive price ratios to prevent `np.log(0)` / `-inf` runtime warnings, and safeguarded rolling volatility computation.
- **Full Archetype Reshaping Engine Dispatch** ([`src/core/reshaping.py`](file:///d:/(Dev2)DSA_AGENT/src/core/reshaping.py)): Implemented `verify_and_prepare_survival_data` (non-negative duration adjustment & binary event formatting), `assess_survey_reliability_and_polarity` (reverse-coding negative questions & Cronbach's alpha reliability), `compute_panel_deltas_and_lags` (within-entity lag1 & delta1), and `compute_geospatial_features` (centroid distance in km). Wired all transforms into `auto_reshape_for_archetype()`.
- **Integrated CleanDataTool Archetype Pre-Modeling** ([`src/tools/data_processing.py`](file:///d:/(Dev2)DSA_AGENT/src/tools/data_processing.py)): Injected `archetype` parameter from `MemorySystem` context in `CleanDataTool.prepare_params()`, applying canonical pre-modeling transformations during Stage 1 data cleaning so downstream modeling tools operate on stationary returns, validated duration, and rich features.
- **Safe Quantitative Routing & Anti-Pattern Guards** ([`src/core/controller_planning.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller_planning.py)): Implemented deterministic archetype prerequisite checks in `PlanMixin._tool_hard_prerequisite_error()`:
  - Banned standard OLS `regression_analysis` on raw non-stationary time series (enforcing difference stationarity / log returns first to eliminate spurious correlations).
  - Banned standard `regression_analysis` on right-censored survival data (preventing survivorship bias and routing to `survival_analysis`).
- **Tests Added & Verified**:
  - Added unit tests in [`tests/test_reshaping.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_reshaping.py) covering NaN log returns, stopword filtering, survival sanitization, survey Cronbach's alpha, panel lags/deltas, geospatial centroid distance, and `CleanDataTool` with archetype.
  - Added Safe Quantitative Routing tests in [`tests/test_universal_archetypes.py`](file:///d:/(Dev2)DSA_AGENT/tests/test_universal_archetypes.py).
  - Quality gates: `ruff check .` (0 violations), `mypy src/` (0 errors across 115 source files), 75/75 passing targeted tests, 8/8 architecture tests.




