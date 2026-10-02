# DSA Agent Frontend Visual Overhaul Plan

Status: active plan. Scope is a full visual overhaul plus a workspace
foundations track (Phase 1b). The baseline is the committed tree on `master`
(commit `1585d9d`, "UI Refinements"); see section 2a. An earlier round of
uncommitted UI changes was reverted by the owner on 2026-10-02 and is not part
of the baseline.

Date: 2026-10-02. Revised the same day after a repository review, and again
after the working tree was reset to the committed state.

Convention: open questions to the owner are in section 17. Decisions already
taken are in section 2b and win over any older text in this file.

### Revision log

Edits made to the original text after reviewing it against the repository and
the owner's decisions. Anything not listed here is unchanged.

| Where | Change | Why |
|---|---|---|
| Header | "Fresh plan" replaced by an active-plan status | Much of the plan's groundwork already exists in the working tree |
| Section 2, protected behavior | Carve-out for config delivery and a read-only `RunView` adapter | The owner chose the structure track first, and the original rules forbade it |
| Section 2, surface table | Landing prototype row now says delete, not "archive or remove" | Owner decision: drop `frontend-landing/` |
| Section 2a, 2b, 2c | New: verified current state, decisions log, findings that drive new work | The original assumed a blank slate |
| Phase 0 | Prototype-only marking replaced by re-point-then-delete; added baseline commit and Streamlit-embedding inventory | As above |
| Phase 1 | Token task rewritten around what exists; added alias-first rename rule and the embedding spike | `design_tokens.py` already exists; modular source cannot be assumed to load inside a Streamlit iframe |
| Phase 1b | New phase: workspace foundations | Config seam, run view, module split, hygiene, UI tests |
| Phase 2, trust item | Replaced "local processing" with a plain statement of the data path | "Local processing" is untrue for a hosted server |
| Phase 1b, Phase 4, Phase 0, section 13, section 14 | Applied the later owner answers: real-run sample, per-session keys with locked operator settings, open palette, recommended usability checks | Owner answers on 2026-10-02 |
| Phase 4 | Added skeleton wiring, run-view consumption, demo labelling | Ties the redesign to Phase 1b and to the no-fabricated-numbers rule |
| Phase 8, section 14 | Baseline commit added; "handover" references replaced | `HANDOVER.md` was deleted |
| Sections 15 to 17 | New: risks, standing rules, open questions | Needed to run the plan |
| Header, section 2a, 2c, Phase 0, 1, 1b, 4, 6, section 2d, risks | Rewritten against the committed tree after the owner reverted an earlier uncommitted UI round: threaded run, Stop, offline report, console fonts, Charts overview and the snapshot script are no longer assumed to exist and are open work again; line references were re-verified | The working tree was reset to commit `1585d9d` on 2026-10-02 |
| Section 2d, decisions log, section 17 | Added a final-year-project scope cut (must, should, defer) and demo-host notes; reduced the retention question | Owner: "it is a final year project" |
| Section 2c items 1, 9, 10, 11; Phase 1b A and D; Phase 4; Phase 0 | Added the hosted gaps found on a second pass: output-dir and key race, shared run history, visitor-typed model URL, shared rate limiter, key gate blocking the sample run, lock-file pin | The hosted decision makes each of them a real exposure |

This plan covers every user-facing frontend surface: the landing and entry
experience, the Streamlit workspace, the embedded workflow plate, the
cinematic 3D export, and the shareable HTML report. It preserves backend
semantics and execution behavior while replacing the visual system,
information hierarchy, motion model, and frontend architecture where needed.

## 1. Objective

Make DSA Agent feel like one coherent, premium, trustworthy data-analysis
instrument rather than several related prototypes. The result must lead with a
plain-language answer, its evidence, and its caveats before exposing technical
machinery. It should be calm enough for a first-time non-technical user,
expressive enough for 3D to clarify the workflow, and fast enough that motion
never feels like it is fighting the browser or the Streamlit rerun model.

The product promise remains:

> Give DSA Agent a dataset and a question. It runs the analysis, checks its
> conclusions, and explains what held up and what still needs more evidence.

The visual overhaul may replace the current visual language, but it must not
invent product capabilities, fabricate metrics, hide uncertainty, or change
the backend's analysis contracts.

## 2. Scope and constraints

### In scope

- First-load landing page and entry CTA.
- Streamlit upload, objective, settings, run-state, preview, and results UI.
- The seven-stage workflow plate shown during analysis.
- The standalone cinematic presentation exported from Downloads.
- The shareable HTML report, including print and offline behavior.
- Theme switching, responsive behavior, keyboard access, reduced motion,
  WebGL fallback, and visual QA.
- An explicit placement and ownership rule for every 3D, scroll, and UI
  animation, so no surface accumulates competing animation engines.
- Removal of duplicate or disconnected frontend implementations after the
  replacement is proven.

### Protected behavior

- Do not modify the RLM engine, task decomposition logic, MemorySystem schema,
  or backend tool behavior as part of the visual redesign.
- Do not change analysis results, statistical logic, governance behavior, or
  privacy guarantees.
- Preserve the existing Streamlit session-state contracts and export data
  contracts unless a frontend adapter is added and tested. Phase 1b is the
  adapter: it narrows session state to a small set of typed keys and is
  covered by its own tests before any tab depends on it.
- Carve-out for Phase 1b. Two kinds of change are in scope and are not
  "analysis contracts":
  - config delivery: a `RunConfig` object passed to `AgentController` and
    `LLMClient` in place of process-wide environment variables, with the same
    defaults and the same effective behavior for a single user;
  - a read-only `RunView` built from `final_result`, memory context and the
    dashboard, which adds no new analysis and changes no tool output.
  Anything that changes what is computed, ranked, or reported stays out of
  scope.
- `AGENTS.md` lists `src/rlm/engine.py`, task decomposition logic, the
  `MemorySystem` schema, deleting source files, and adding a tool as Ask-First.
  `controller.py` and `llm_client.py` are not named there. Phase 1b changes
  only how configuration reaches them, never decomposition or memory schema, so
  it proceeds without a per-edit approval; the diff is still reported in the
  summary. Stop and ask if a change would touch decomposition, the engine, or
  the memory schema. (An earlier revision of this plan wrongly called both
  files Ask-First.)
- Preserve user worktree changes already present when implementation begins.
- Do not delete a source file until the replacement is live, tested, and the
  old file is confirmed to have no remaining imports or export references.
- Do not make WebGL, animation, or spatial interaction a prerequisite for
  understanding an answer, completing a task, or accessing an export.

### Current authoritative surfaces

| Surface | Current entry point | Current issue | Target |
|---|---|---|---|
| Landing | `ui/landing.py` -> `ui/landing_component/index.html` | Large inline monolith, token drift, duplicate narrative, scroll and animation coupling | One maintainable, responsive entry experience with an optional 3D explanation and a complete no-WebGL path |
| Landing prototype | `frontend-landing/index.html` plus unused TypeScript | Stale copy of the pre-rebuild landing (it still has the FAQ section); referenced only by `tests/test_landing_v2.py` | Delete. Owner decision. First re-point `tests/test_landing_v2.py` at `ui/landing_component`, then remove the directory |
| Workspace | `app.py`, `ui/styles.py`, `ui/components/`, `ui/tabs/` | Strong information architecture but repetitive cards, pill-heavy controls, technical density | A clear analysis workspace with progressive disclosure |
| Workflow plate | `ui/pipeline_3d.py`, `ui/assets/pipeline_3d.{html,js}` | Good fallback and interaction, but separate runtime conventions and external dependency loading | Shared 3D runtime contract with a text-first seven-stage timeline and a secondary spatial explanation |
| Cinematic export | `ui/cinematic_3d.py`, `ui/assets/cinematic_3d.{html,js}` | Fullpage.js, large scene, multiple interaction systems, hard-to-verify claims | Optional, standalone native-scroll explanation that uses the same runtime primitives and never blocks the report path |
| Shareable report | `src/core/html_report.py` | Good report structure but remote fallback assets and a separate chrome system | Offline-first, print-ready report sharing the product tokens |

## 2a. Current state (verified 2026-10-02, after the reset)

The baseline is the committed tree. On 2026-10-02 the owner reverted an
earlier, uncommitted round of UI changes (a threaded run with Stop, a
Charts overview band, self-hosted console
fonts, an offline report, a snapshot script) because it left the UI uncertain,
and returned to the last working code. Nothing from that round is assumed to
exist. Where the plan used to cite it, the item is now open work.

None of the current UI has been viewed in a browser during this review; it was
verified by reading code. Phase 0 captures it. `README.md` is being rewritten by
the owner and is not touched by this plan.

| Area | What exists | Where |
|---|---|---|
| Palette source | One Python token module with Day and Night palettes and a CSS `:root` generator, used by `ui/styles.py`. The palette is the buff-green "ledger paper" set (stock `#eef1e0`) | `src/core/design_tokens.py`, `ui/styles.py:20-25` |
| Streamlit theme | **Out of date.** The TOML still carries the old cream and terracotta values and a comment pointing at the deleted `DESIGN.md`, so it disagrees with `design_tokens.py` | `.streamlit/config.toml` |
| Console fonts | Loaded from Google Fonts by an `@import` on every rerun. `static/fonts/` and `ledger-fonts.css` exist and are not used by the console | `ui/styles.py:32`, `static/fonts/` |
| Report | Print CSS exists. Fonts and Vega load from CDNs; the vendored Vega bundles in `static/vendor/vega/` are not used by the report, so it needs the internet to draw charts | `src/core/html_report.py:165-180` |
| Audited-entry checks | Backend writes `evidence["checks"]` and leakage alerts reach `final_result`. One module (`audited_entry.py`) decides which marks show. Answers finding cards and the HTML report render the check row | `src/core/audited_entry.py`, `src/core/findings.py`, `ui/components/cards.py:143`, `src/core/html_report.py:240` |
| Run model | The analysis runs synchronously inside the Run click handler. No Stop, no worker thread, no polling. Callbacks `on_step_callback` and `on_iteration_callback` exist; there is no cooperative-stop hook | `app.py` (run handler near line 1062), `controller.py:286-288` |
| Tabs | Four tabs: Answers, Charts, Details, Downloads. Charts is a flat list of panels with a half and full width rule | `app.py:1377`, `ui/tabs/` |
| Answers | Finding cards with the check row, a run-level "N of M findings held up" banner (only when audits ran), recommendations, a model stamp, a technical expander. The Charts tab panels do not show check rows | `ui/tabs/answers_tab.py:109-135`, `ui/components/cards.py:123` |
| Theme default | First session follows the OS theme | `app.py:152` |
| Landing | Marketing-template landing in one inline file of about 120 KB, still including the FAQ section | `ui/landing.py`, `ui/landing_component/index.html` |
| Landing prototype | `frontend-landing/` is a near copy of the same page plus unused TypeScript | `frontend-landing/` |
| Plate and cinematic | Plate in `ui/pipeline_3d.py` and `ui/assets/pipeline_3d.*`; cinematic export in `ui/cinematic_3d.py` and `ui/assets/cinematic_3d.*` | as listed |

Also true today, and relevant to the rules in this plan:

- Remote loads: the console (Google Fonts), the plate (Three.js from jsDelivr,
  GSAP, Anime.js, Google Fonts), the cinematic export (Three.js, Anime.js v4,
  fullPage.js, Google Fonts), the landing (Three.js, Anime.js 3.2.2, Google
  Fonts) and the report (Google Fonts, Vega from jsDelivr). A run with
  `LOCAL_ONLY` therefore still contacts third parties from the browser.
- `ui/animations.py` is live: `app.py:1445-1447` calls
  `inject_micro_interactions()` on every rerun. It loads Anime.js from a CDN
  and attaches a `MutationObserver` to the page body each time, which this
  plan's runtime rules forbid.
- `ui/pipeline_3d.py:139-172` holds wrapper functions that only forward to
  `ui/cinematic_3d.py`.
- Streamlit is pinned `>=1.49,<2` in `requirements.txt` and `pyproject.toml`;
  1.57 is installed and pinned in `requirements.lock`. The app already uses
  `st.context.theme` and `st.iframe` with a fallback.
- There are no UI tests beyond asset checks in `test_landing*`,
  `test_pipeline_3d` and `test_cinematic_3d`. No test renders a tab.

## 2b. Decisions log

Decisions already taken by the owner. Do not re-ask.

| Date | Decision | Consequence |
|---|---|---|
| 2026-10-02 | Deployment is hosted for a few users, not single-user local | Cross-session leaks are release blockers (section 2c); Streamlit stays; no FastAPI split in this plan |
| 2026-10-02 | First track is structure: `RunConfig`, `RunView`, split `app.py` | Phase 1b runs before the Phase 4 workspace redesign and does not wait for the direction board |
| 2026-10-02 | Keep the 3D plate | Phase 5 migrates it; it stays secondary to the text timeline |
| 2026-10-02 | Keep the cinematic export (reverses an earlier "drop it") | Phase 5 migrates it to native scroll; it stays optional polish |
| 2026-10-02 | Drop `frontend-landing/` | Re-point its test, then delete (Phase 0) |
| 2026-10-02 | Sample demo: first kept as is, then replaced by a real run (later the same day, after the conflict with the no-fabricated-numbers rule was shown) | The sample button runs the deterministic pipeline on `data/sample_customer_churn.csv` and builds a real `RunView`; `_load_teamwork_preview` and its placeholder report files are deleted in Phase 1b |
| 2026-10-02 | Landing trust copy states the data path plainly | No claim of local processing; say where the file goes and that summaries go to the chosen LLM provider, and offer the run-without-AI option |
| 2026-10-02 | API keys are entered per session and never stored; operator safety settings are locked | Code execution, isolation, run caps and `LOCAL_ONLY` come from the server environment and show read-only on the page |
| 2026-10-02 | Palette is open | The direction board may propose a new palette; Phase 1 token work stays palette-agnostic |
| 2026-10-02 | The non-technical usability checks are recommended, not a gate | Definition of done no longer requires them; unresolved findings are still recorded if they are run |
| 2026-10-02 | `DESIGN.md` and the older planning documents deleted | Type scale, stage names and plain-language copy rules now live only in git history; the direction board must re-lock them. The seven stage names have one source, `STAGE_DEFS` in `ui/components/cards.py`; every surface imports it or is generated from it |

Added after the owner's answer on hosting facts:

| Date | Decision | Consequence |
|---|---|---|
| 2026-10-02 | The project is a final-year project, not a commercial product | Scope is cut to what is demonstrable, defensible in a viva, and finishable on a fixed deadline (section 2d). Hosting is a demonstration deployment, not a service with retention or privacy commitments |
| 2026-10-02 | Priority is a demonstration-ready build for a B.Tech CSE final-year project; other questions are left to the implementer's judgement | Section 2d tiers apply as written; the deadline and venue stay open and only change ordering, not scope |
| 2026-10-02 | The earlier uncommitted UI round was reverted; the committed tree is the baseline | Everything it added is open work, not groundwork. The owner's reason: it left the UI uncertain, and an overhaul is planned anyway |
| 2026-10-02 | Live findings: option B, an optional `on_finding_callback` with a provisional "Found so far" list (no check marks, labelled as subject to change). Full `analyze_iter()` stays deferred | Should tier, and it needs the run to move to a worker thread first (the run is synchronous in the committed tree). Provisional findings carry no check row; `RunView.provisional_findings` is empty once the run completes |

## 2d. Scope for a final-year project

The plan above is written to production standard. A final-year project is
judged on a working, honest, well-explained system on a fixed date, not on
completeness. This section ranks the work. Where it conflicts with a phase
below, this section sets the priority and the phase text stays as the detailed
specification.

**Must (the demo cannot be shown or defended without these):**

- Phase 0 in a reduced form: baseline commit, one real browser run of the
  sample CSV (Run, a failed preflight, completed result), baseline
  screenshots, the timed sample run. Confirms the committed tree works in a
  browser before anything else changes.
- Phase 1b A, minimal: per-session keys, no key or objective in `os.environ`,
  output directory and key passed as arguments (removes the click-time race),
  operator-only local-model URL. If the demo is on a public link, this is what
  stops one visitor using another's key or seeing another's results.
- Phase 1b B: `RunView`, because it is the typed contract between backend and UI
  and the strongest architecture story for the report and viva.
- The real sample run (replaces the fabricated demo) and the run gate
  `has_key or not use_llm`, so a visitor can see a real result with no key.
- Phase 2 copy cleanup: remove every claim the system cannot prove, and state
  the data path plainly.
- Phase 4 core: answer, evidence, caveat, next action first (the check row and
  the verdict banner already exist on Answers; reorder so they lead and add the
  check row to Charts panels); "How we got here" from the hidden outputs; a
  loading skeleton; one primary action per screen.
- Phase 6 core: the report prints cleanly (print CSS exists; verify it) and
  states what it needs the network for.
- Documentation for examiners: refresh `README.md` (it is outdated), one
  architecture diagram of the final design, before and after screenshots, and a
  short "known limitations" section. Honest limits score better than hidden
  ones.

**Should (do if time allows, in this order):**

- Phase 1 token closure (bring `.streamlit/config.toml` back in line with
  `design_tokens.py` and add an equality test, generated CSS for landing, plate
  and cinematic export), self-host the console fonts (the files exist in
  `static/fonts/`), and the embedding spike.
- Phase 6 offline report: inline the vendored Vega bundles and Latin fonts from
  `static/`, and give each chart a no-script data-table fallback.
- Run in a worker thread with a cooperative Stop and a polling fragment (the
  committed run is synchronous). Needed before the provisional "Found so far"
  list (`on_finding_callback`) can exist.
- Phase 1b C: split `app.py` and `cards.py`; delete the dead code.
- Phase 1b D and E: per-session run directories and cleanup; AppTest smoke tests
  and the session-isolation test.
- Phase 3 landing: static-first hero and no scroll hijacking, reusing the
  existing 3D scene rather than rewriting it.
- Phase 5 plate: a text-first timeline with the existing scene as the secondary
  view.

**Defer (state them as future work in the report):**

- One shared Three.js runtime across landing, plate and cinematic export, and
  the performance-tier measurements on a real phone.
- Migrating the cinematic export to native scroll (it stays optional polish).
- The full viewport and accessibility matrix; run a reduced one instead: three
  viewports (1440, 768, 390), both themes, keyboard-only through the main flow,
  reduced motion, no WebGL.
- A formal usability study. An informal check with three to five classmates
  or family members on the five tasks is enough, and its results (including
  failures) go in the report.
- `analyze_iter` and any filter or cross-filter work on the Charts tab.
- Renaming tokens to the new semantic names, unless the direction board needs
  it.

**Deployment notes for a demonstration host:**

- A public demo link means unknown visitors run the app. Set
  `ENABLE_CODE_EXECUTION=false` unless the host can run the Docker sandbox, and
  do not expose `SANDBOX_REQUIRE_ISOLATION=false` as a page toggle.
  `AGENTS.md` says to require isolation whenever data or objectives come from
  untrusted users.
- Many free hosts have an ephemeral filesystem and no Docker. Assume the
  subprocess sandbox only, and run directories that vanish on restart.
- Landing copy for this context, pending the owner's confirmation of where it
  is hosted: files are processed on the server for the length of the session
  and removed when it ends (Phase 1b D must make that true); summaries of the
  data go to the chosen LLM provider unless AI is switched off; this is an
  academic project and should not be given sensitive data. Do not promise
  retention periods or compliance.
- Bring a fallback for the viva: a pre-recorded run, and the bundled sample file
  runnable offline with the LLM off, in case the network or a free API tier
  fails on the day.


## 2c. Findings that drive new work

Verified in code against the committed tree. Each maps to a task in Phase 1b or
Phase 4.

1. **Configuration travels through `os.environ`.** The run handler
   (`app.py:1087-1110`) sets about fifteen process-wide variables on Run, and
   reasoning effort is set at `app.py:826-830`. On a shared server this leaks
   between sessions:
   - `app.py:749-750`: `_effective_key = api_key or os.getenv(...)` falls back
     to a key in the server's environment (used for the model-list fetch). The
     run itself only starts when a key was typed (`app.py:961`), but that typed
     key is then written to the process environment and read lazily by the
     LLM client, so a second visitor's Run click can replace the key a first
     visitor's run is about to use;
   - `ui/tabs/answers_tab.py:43` reads `USER_OBJECTIVE` from the environment
     before the widget value, so a second session shows the first session's
     question;
   - three analysis tools read the objective straight from the environment
     (`src/tools/anomaly.py:288`, `change_analysis.py:158`,
     `time_series.py:199`), so another visitor's question can steer measure
     selection in a different visitor's run. `time_series` runs on a thread
     pool, so a per-thread fix must carry its context into the pool;
   - `app.py:1091-1097` mutates the global rate-limiter profile, and
     `get_limiter()` is one process-wide object keyed by provider and model.
     The values set are model facts, so this is accepted for the demo;
   - the UI writes `OUTPUT_DIR` and the provider key into the environment at
     click time, and the controller reads them a moment later. Because Streamlit
     serves sessions on separate threads, a second Run click in that gap can
     send the first run's artifacts to the wrong directory or use the wrong key.
2. **The backend produces results the console never shows.** Compared against
   every string in `app.py` and `ui/`:

   | Produced | Today |
   |---|---|
   | `analysis_decision` (describe versus model, and why) | Reports only |
   | `degradations` (what fell back) | Reports only; console shows one `llm_warning` line |
   | `unverified_claims` | Reports only |
   | `deliverable_audit` (asked-for outputs delivered, missing, repaired) | No user surface |
   | `hypothesis_tree` | Persisted and fed to the prompt; shown nowhere |
   | `llm_usage`, `api_telemetry` (tokens, cost, rate headroom) | HTML report only |
   | `question_agenda`, `question_routing` | Only the `coverage` summary reaches the UI |
   | `evidence["checks"]` on each finding | Rendered on Answers cards and in the report; not on Charts panels |
   | `target_leakage_alerts` | Folded into the finding checks; no standalone alert surface |

3. **No typed contract between backend and UI.** Tabs read loose dicts;
   `find_tool(tool_results, "train_model")` (`ui/components/cards.py:463`)
   reaches into `models_trained` and `classification_report`; the dashboard
   comes back through `dashboard.json` on disk; `app.py` is a 1,447-line
   script with `session_state` read and written throughout.
4. **Presentation logic is split across two packages.** `ui/` holds Streamlit
   code, while `src/core/dashboard.py` (2.8k lines), `chart_spec.py`,
   `html_report.py`, `chart_theme.py`, `plain_language.py` and
   `audited_entry.py` also decide what the user sees. This is an asset for
   any future frontend because it is framework-free.
5. **Dead and duplicate code.** `ui/animations.py` is live but violates the
   runtime rules (CDN load, observer per rerun); the wrapper block in
   `ui/pipeline_3d.py` only forwards; `frontend-landing/` is a near duplicate.
6. **The sample demo fabricates results.** `_load_teamwork_preview`
   (`app.py:235` to about `492`) writes invented statistics and findings into
   session state and writes placeholder `report.html`, `analysis_report.md` and
   `final_report.json` into a temp directory that the Downloads tab then
   serves. The cinematic export binds to the same session state through
   `extract_cinematic_state`. It has no `findings`, so it can never show the
   audited entry, which is the product's signature.
7. **Run directories are not managed.** `tempfile.mkdtemp()` at `app.py:262`
   and `app.py:1062` is never removed; the UI also writes `output/runs/ui-*`
   relative to the working directory (`app.py:1283`).
8. **Presentation hacks that depend on Streamlit internals.** `_stub_rich()`
   (`app.py:71`) replaces `rich` with fakes in `sys.modules` and behaves
   differently depending on import order; the landing hides a native button
   at `top: -9999px` and forces its iframe to `z-index: 999999`.
9. **Run comparison crosses sessions.** `app.py:1283` copies each run's
   `summary.json` into the shared `output/runs/ui-*`, and
   `render_run_compare` (`ui/components/cards.py:694-701`) calls `list_runs()`,
   which globs `output/runs/*/reports/summary.json` for every run in the tree,
   including other visitors' and CLI runs. Each summary carries the dataset
   name and findings, so a visitor can see and diff other visitors' results.
10. **A visitor-supplied URL is fetched by the server.** The sidebar's "Server
    URL (OpenAI-compatible)" field (`app.py:715`) is passed to
    `_get_dynamic_models` (`app.py:751`, `model_telemetry.py:362,434`), which
    fetches it on every rerun, and LLM calls go to it too. On a hosted server
    that lets any visitor make the server send requests to `localhost` or an
    internal or cloud-metadata address.
11. **A real sample run is blocked by the key gate.** `can_run` requires a key
    even when the LLM is off (`app.py:961-962`); the fake demo only worked
    because it bypassed the run entirely.
12. **The wait shows stages, not answers.** The run is synchronous, so the page
    is blocked until it ends, with no Stop. Streaming findings during the wait
    needs both a worker thread and an event or callback from the controller;
    `analyze()` is a single blocking method of about 470 lines with untyped
    callbacks and no stop hook.

## 3. Design direction gate

Before implementation, create a small visual direction board and approve one
direction. The recommended starting point is **Guided Evidence**:

- a plain-language, evidence-first data story rather than a technical control
  room;
- one calm focal 3D object only when it explains transformation, checking, or
  report formation;
- evidence, sources, uncertainty, and actions treated as first-class visual
  material;
- generous reading space and short, concrete copy;
- a restrained action accent, with positive and risk colors reserved for real
  result states;
- no generic glassmorphism, neon dashboard styling, decorative telemetry, or
  continuous particle ambience;
- coherent light and dark themes, with no arbitrary section-level inversions.

Design read: this is a trust-first data-analysis product for everyday users,
leaning toward editorial data storytelling with a bounded Three.js workflow
artifact, not an immersive 3D showcase.

The direction board must show the landing hero, a workspace result, one 3D
transition, a chart panel, and a printed report excerpt. It must explicitly
lock these decisions before code:

1. Palette and theme relationship.
2. Heading, body, numeric, and code typography.
3. Spacing scale and content width.
4. Radius, border, shadow, and surface rules.
5. Accent, positive, risk, and neutral semantics.
6. Icon and status-mark language.
7. 3D material language, lighting, and focal geometry.
8. Motion dials for marketing, workspace, and reduced-motion modes.

Recommended design dials:

| Dial | Landing | Workspace | 3D/export |
|---|---:|---:|---:|
| Visual variance | 5/10 | 3/10 | 5/10 |
| Information density | 4/10 | 6/10 | 4/10 |
| Motion intensity | 4/10 | 1/10 | 3/10 |
| Decorative detail | 2/10 | 1/10 | 2/10 |

### Source-backed design principles

- Borrow visible sources, configuration-specific sharing, full-screen
  exploration, and progressive controls from [Our World in Data's Grapher
  redesign](https://ourworldindata.org/redesigning-our-interactive-data-visualizations).
- Borrow the rule that each visual chapter answers one question from [The
  Pudding's storytelling guide](https://pudding.cool/process/how-to-make-dope-shit-part-3/).
- Borrow chart annotations, alternative descriptions, data access, and
  color-safe presentation from [Datawrapper's accessibility
  approach](https://www.datawrapper.de/academy/how-we-make-sure-our-charts-maps-and-tables-are-accessible).
- Use [Three.js rendering on demand](https://threejs.org/manual/pages/rendering-on-demand.html)
  and its [responsive rendering guidance](https://threejs.org/manual/pages/responsive.html)
  as the performance baseline for every scene.
- Use [GSAP ScrollTrigger](https://gsap.com/docs/v3/Plugins/ScrollTrigger/)
  only when scroll reveals a necessary narrative step. It is not a reason to
  pin or animate ordinary content.
- Treat every meaningful 3D scene, chart, and workflow diagram as a complex
  image with an equivalent text path, following [W3C's complex-image
  guidance](https://www.w3.org/WAI/tutorials/images/complex/).

### 3D placement and runtime ownership

| Surface | 3D role | Required non-3D path | Runtime owner |
|---|---|---|---|
| Landing | Optional explanation of the checked-analysis workflow, loaded after the stable hero | Plain promise, workflow summary, and entry CTA | Three.js, with GSAP only for a bounded narrative sequence if required |
| In-progress workflow | Secondary spatial reflection of the active stage | Live text timeline, current-stage explanation, and cancel state | Three.js plus CSS or Web Animations API for labels |
| Answers and charts | None by default | Answer, evidence, caveat, chart, and next action are the primary interface | CSS or Web Animations API only |
| HTML report | Static visual or poster only | Fully readable, printable report with chart descriptions and data tables | No live animation runtime |
| Cinematic export | Optional shareable process explanation | Native-scroll, text-first story and static poster | Three.js, with GSAP only for necessary scroll narration |

Anime.js is not an initial dependency. It overlaps with GSAP and would create
a third animation owner. If it is ever introduced, it must replace rather than
run beside the chosen UI-motion layer on a given surface.

The visual direction gate is not permission to add unsupported product claims.
All examples must use real or clearly labelled demonstration data.

## 4. Phase 0: baseline and inventory

### Tasks

- Capture the current landing, empty workspace, uploaded-data workspace,
  running state, completed Answers, Charts, Details, Downloads, cinematic
  export, and HTML report.
- Capture day and night themes at desktop, tablet, and mobile widths.
- Record current load timing, bundle sizes, frame rate, render pixel ratio,
  WebGL availability, and Streamlit rerun behavior.
- Build a surface map of every CSS class, iframe, CDN dependency, theme token,
  and DOM selector used by the frontend.
- Trace the live import path from `app.py` to each rendered surface.
- Build a 3D-placement and runtime-owner inventory from the table above. Mark
  any existing scene, ticker, event listener, or library that violates it.
- Define five task-based, think-aloud usability checks for non-technical
  participants: explain the product promise, start an analysis, identify the
  answer, find the caveat, and export or share the report.
- Run them with five to seven representative non-technical users if the owner
  can arrange it. This is recommended, not a gate (section 2b). It is an
  acceptance activity, not permission to fabricate user-research results; if
  it is not run, the status log says so.
- Commit the current working tree (the staged documentation deletions and this
  plan) before anything else changes, on a branch, and commit after each later
  step. An earlier uncommitted round of UI work was lost to a reset; commits
  are the only protection. The "preserve worktree changes" rule needs a commit
  to preserve them against. No co-author trailer.
- Run the app once with a real sample CSV and exercise a run to completion, a
  failed preflight (bad key or model), and "New analysis". The run is
  synchronous in the committed tree, so there is no Stop to test. Confirm the
  four tabs in a browser. This is the first time this review sees them.
- Capture baseline screenshots of each state in both themes. A small Chrome
  DevTools screenshot script is worth writing if time allows (Should); a manual
  set is enough for the Must tier. An earlier draft of such a script was lost
  in the reset.
- Time the deterministic sample run end to end (LLM off, ML on and off), because
  it becomes the first impression once the hard-coded demo is replaced. Record
  the report's byte size here too.
- Confirm the Streamlit pin: `requirements.lock:172` has `streamlit==1.57.0`,
  while `requirements.txt` and `pyproject.toml` allow `>=1.49`. Decide whether
  the hosted deployment installs from the lock file, and raise the lower bound
  if the code needs `st.context.theme`.
- Inventory every remote load by surface (the list in section 2a is the
  starting point) and every place a hex literal still appears outside
  `src/core/design_tokens.py` and `.streamlit/config.toml`.
- Inventory Streamlit embedding: for each iframe, record whether it is
  `st.iframe`, `components.html` or `declare_component`, its origin, and
  whether relative URLs resolve inside it (see the Phase 1 spike).
- Re-point `tests/test_landing_v2.py` from `frontend-landing/` to
  `ui/landing_component/`, confirm nothing else references `frontend-landing/`,
  then delete the directory. This is the one implementation change allowed
  inside Phase 0 because it is a deletion of a verified duplicate, and it is
  made in its own commit.
- Add a short frontend acceptance checklist to the implementation branch.

### Exit criteria

- The working tree is committed; Phase 0 starts from a clean tree.
- Every user-facing screen has a baseline screenshot.
- Every surface has one identified source of truth.
- Every animation surface has a recorded purpose, non-motion fallback, and
  single runtime owner.
- Unsupported claims and demo-only values are listed for copy cleanup.
- No implementation file is changed during the baseline capture, other than
  the landing-prototype removal described above.

## 5. Phase 1: shared design system and frontend contracts

### Tasks

- `src/core/design_tokens.py` already is the canonical Python source and
  `css_root_block()` already feeds the console CSS and the report. The work is
  to finish it:
  - bring the one manual copy, `.streamlit/config.toml`, back in line (it still
    carries the old cream hexes and cites the deleted `DESIGN.md`) with a small
    generator script or a test that asserts the TOML hexes equal
    `design_tokens.py` (TOML cannot import Python);
  - stop loading Baloo 2 and Mukta from Google Fonts in `ui/styles.py:32` and
    serve the files already in `static/fonts/` through Streamlit static serving
    (`enableStaticServing` in the config);
  - give the landing HTML, the plate, and the cinematic export a validated
    token payload or generated CSS block at the Python boundary instead of
    hex literals in inline HTML and JavaScript. `ui/landing.py` already pulls
    five keys from the module; extend that pattern to the rest.
- Define semantic tokens rather than component-specific colors:
  `canvas`, `surface`, `surface-raised`, `ink`, `muted`, `rule`, `action`,
  `positive`, `risk`, and `focus`. The current names are `stock`, `sheet`,
  `sheet_alt`, `ink`, `graphite`, `pen`, `risk`, `positive`, `rule`. Renaming
  touches `ui/styles.py`, `src/core/html_report.py`, the landing, the plate,
  the cinematic export, and the `LEDGER_TOKENS_*` imports in
  `tests/test_landing.py` and `tests/test_landing_v2.py`. Add the new names as
  aliases first, migrate consumers one surface at a time, and remove the old
  names only when a repository search finds no user.
- The palette is open (section 2b): the direction board may propose a new one.
  The token work above is palette-agnostic and can proceed before the board is
  approved. If the board changes the palette, update `design_tokens.py` and
  regenerate the TOML copy; no other file should need a hex edited.
- Define typography roles for display, body, numeric, code, and labels.
- Define a small spacing scale, content widths, breakpoint rules, control
  heights, focus rings, and minimum touch targets.
- Define one radius family and one shadow family. Do not let every card invent
  its own treatment.
- Define icon and status conventions that work without emoji or color alone.
- Self-host all required fonts and bundle or locally serve approved frontend
  libraries. Keep the page functional with no network access.
- Establish `Three.js` as the sole WebGL runtime. Use GSAP only on the
  landing or cinematic export when an approved narrative needs scroll-bound
  sequencing. Use CSS or the Web Animations API for ordinary workspace motion.
- Do not add Anime.js while GSAP is in use. Reassess only if the selected
  motion owner is deliberately replaced and the bundle, lifecycle, and
  fallback implications are documented.
- Create a shared frontend contract for:
  - theme changes;
  - reduced-motion state;
  - analysis stage state;
  - finding/check/verdict state;
  - 3D progress state;
  - WebGL fallback state;
  - export/report data.

### Recommended architecture

The repository has no frontend package build today. Prefer modular native ES
modules and locally served assets first, with a build tool introduced only if
it materially improves testing or bundle control. Do not keep adding code to a
single 3,000-line HTML file.

**Embedding spike (do this before committing to the module layout).** The
preference above rests on an assumption that has not been tested here. The
plate is built as one HTML string in Python and handed to `st.iframe` (with a
`components.html` fallback), so it has no directory beside it for relative
`import` paths to resolve against. The cinematic export must be a single
self-contained offline file. The landing is the exception: `declare_component`
serves a real directory. Prototype and record the result for each surface:

- whether an iframe created from an HTML string can load modules from
  Streamlit's static path (`/app/static/...`, including the configured
  `baseUrlPath`), or must inline them;
- how a modular source becomes one inlined document for the export. Today's
  pattern already inlines at the Python boundary
  (`.replace("__SCENE_SCRIPT__", ...)` in `ui/pipeline_3d.py`); a small
  bundling or inlining step in Python may be the simplest answer;
- the size cost of vendoring Three.js locally under `static/vendor/`, and
  whether the cinematic export can share one copy with the plate or must embed
  its own.

If relative modules do not work inside a string-built iframe, the build-tool
question is answered by evidence and not by preference. Record the outcome in
this section before Phase 3 starts.

Proposed boundaries:

```text
ui/frontend/
  shared/
    tokens.js
    theme.js
    motion.js
    accessibility.js
    lifecycle.js
    chart_descriptions.js
  three/
    runtime.js
    formations.js
    materials.js
    camera.js
    interaction.js
    fallback.js
  landing/
    landing.js
    landing.css
  plate/
    plate.js
    plate.css
  cinematic/
    cinematic.js
    cinematic.css
```

The exact directory may change during implementation, but every surface must
use shared runtime and token primitives rather than forks of the same logic.

### Exit criteria

- One token payload renders identically in both themes across all surfaces.
- A network-disabled browser still loads fonts, icons, 3D libraries, and the
  report assets from local files.
- The frontend contract is typed or validated at every Python-to-browser
  boundary.
- Each result chart and each meaningful 3D state has a tested short summary,
  detailed description or data-table path, and visible source or provenance
  when that information exists.
- No new design code depends on Streamlit's private CSS class names unless a
  documented adapter is unavoidable.

## 5b. Phase 1b: workspace foundations

Added by the structure-first decision (section 2b). This phase does not depend
on the direction board and runs in parallel with it. It must finish before the
Phase 4 workspace redesign, because that redesign consumes the run view and the
module split. It changes how configuration reaches the backend and how results
reach the UI; it does not change what is computed.

### Tasks

**A. `RunConfig`: replace the environment channel (closes findings 2c.1).**

- Add a frozen `RunConfig` dataclass (suggested home `src/core/run_config.py`):
  provider, model, base URL, API key, reasoning effort, min and max iterations,
  recursive decomposition, LLM and ML switches, code-execution switches,
  minimum group size, objective, target hint, output directory, and the four
  ML overrides (`max_depth`, `test_size`, `n_cv_folds`, tuning). The API key
  field must be excluded from `repr` and from any log, audit or export.
- Provide `RunConfig.from_env()` so `main.py` and tests keep working with the
  same defaults, and so a single-user CLI run behaves exactly as before.
- Pass it to `AgentController` and `LLMClient`. Keys are never written to
  `os.environ` by the UI.
- First audit every environment read in `src/` (roughly 47 sites across the
  controller, LLM client, governance, privacy and sandbox) and classify each as
  per-run (move to `RunConfig`) or operator-level (stays in the environment).
  Some modules read lazily at call time, and the controller runs tools on
  thread pools, so a context variable set in the worker thread does not reach
  those pool threads unless the context is copied. Choose explicit parameters
  or a copied context from the audit, not by assumption.
- Separate user-level from operator-level settings in the sidebar. Operator
  controls in `AGENTS.md` (`ENABLE_CODE_EXECUTION`, `SANDBOX_REQUIRE_ISOLATION`,
  `MAX_CODE_EXECUTIONS`, `LOCAL_ONLY`, token caps) are documented as
  environment controls. On a hosted server a user must not be able to switch
  isolation off from the page. Show them as read-only status when the operator
  has set them. Decided (section 2b): users edit provider, model, thoroughness,
  and the ML options; the operator owns code execution, isolation, run caps,
  token caps and `LOCAL_ONLY`.
- API keys are entered per session and never persisted, written to
  `os.environ`, logged, or included in any export. Delete the
  `api_key or os.getenv(...)` fallthrough at `app.py:749-750`; a server-side key
  is not used for visitors. A single-user local or CLI run may still read
  `<PROVIDER>_API_KEY` from the environment through `RunConfig.from_env()`.
- Make controller console output quiet through a `RunConfig` flag instead of
  replacing `rich` in `sys.modules` (`app.py:71`).
- Do not mutate the global rate-limiter profile from the UI
  (`app.py:1091-1097`); pass the model's limits into the client. Scope limiter
  state per session, or per hash of the key, so one visitor's usage does not
  throttle another's.
- Pass `output_dir` and the key to `AgentController` and `LLMClient` as
  arguments, never through the environment, which also removes the click-time
  race in 2c.1.
- The local-model base URL, and the "local" provider as a whole, are
  operator-only on the hosted deployment: the server must never fetch or call a
  URL a visitor typed. The operator may preset one in the environment; the page
  shows it read-only or hides the provider.
- Make the run gate `has_key or not use_llm` and let the sample button force
  the LLM off regardless of the sidebar (see Phase 4).

**B. `RunView`: one typed object between backend and UI (closes 2c.2 and 2c.3).**

- Add a framework-free module (suggested `src/core/run_view.py`) with frozen
  dataclasses and one builder, `build_run_view(...)`, that takes `final_result`,
  the memory context, the dashboard specs, the profile and the read report, and
  returns the single object every surface reads.
- Contents: question and verdict; ranked findings with their check marks;
  coverage (answered and unanswered questions); recommendations; charts with
  panel metadata; model summary (best model, cross-validated score, train-test
  gap, per-class report); data understanding; ingestion and repair notes;
  profile; governance summary; and the outputs that are hidden today:
  `analysis_decision`, `degradations`, `unverified_claims`, `deliverable_audit`,
  `llm_usage` with cost, rate headroom, and a hypothesis-tree summary. Absent
  data is represented as absent, never as an empty tick.
- It is read-only: it derives from existing results and adds no analysis. A
  `to_dict()` form is the contract for the Python-to-browser payloads in
  Phase 1 and for the cinematic export.
- Tabs, the cinematic export and the demo take a `RunView`. `find_tool` and the
  `dashboard.json` disk read disappear from the UI. The dashboard specs travel
  in the view; the file stays on disk as a download only.
- `st.session_state` shrinks to a small set of typed keys, for example
  `config`, `run`, `view`, and UI preferences. Document them in one place.

**C. Module split (closes 2c.5, part of 2c.3).**

- Split `app.py` into `ui/sidebar.py` (returns a `RunConfig`), `ui/run.py`
  (worker, `_Run`, polling fragment), `ui/idle.py` (hero, upload, empty state),
  `ui/results.py` (tab host), and `ui/state.py` (typed session keys). `app.py`
  becomes a thin entry that wires them.
- Split `ui/components/cards.py` into pure HTML builders with no `st.*` and no
  `session_state` (testable without Streamlit) and Streamlit renderers that
  call them.
- Remove, in separate commits: `ui/animations.py` and its call at
  `app.py:1445-1447` (it loads Anime.js from a CDN and adds a `MutationObserver`
  per rerun); the `pipeline_3d.py` wrapper block; `_load_teamwork_preview`
  (replaced by the real sample run, Phase 4). Each removal is preceded by a
  repository search for importers.
- Replace remaining inline `style=` attributes in finding cards
  (`ui/components/cards.py`, in `render_finding_card` near line 123) with classes.

**D. Hosted hygiene (closes 2c.7).**

- One run directory per session and run, under a configurable root, not under
  the working directory. Remove it on "New analysis", on a discarded run, and
  by a startup sweep of directories older than a set age. Replace both
  `tempfile.mkdtemp()` calls and the `output/runs/ui-*` copy at `app.py:1283`.
- Never write one session's artifacts where another session's Downloads tab
  can list them.
- Scope "Compare with a previous run" to the current session's own runs (a
  per-session runs directory passed to `list_runs`), or turn it off on the
  hosted deployment. Stop copying summaries into the shared `output/runs`
  (`app.py:1283`). `controller.py:231-235` writes `output/latest.txt` only when no
  output directory is given; the UI always gives one, but confirm that on the
  hosted path.

**E. UI tests.**

- Build a fixture `RunView` (from a real deterministic run on the sample CSV,
  stored as JSON) and add `streamlit.testing.v1.AppTest` smoke tests that
  render each of the four tabs from it without a network or a model.
- Add a session-isolation test: two configs with different keys and objectives
  never read each other's values.
- Unit-test `build_run_view` for each hidden output, including the absent case.
- Add the token equality test from Phase 1 here if it is not already in place.

**F. Deferred and optional.**

- `AgentController.analyze_iter()` as an event generator (stage, step and
  finding events) to stream findings during the wait. Highest user-visible
  payoff in the progress experience, and the largest blast radius in
  `analyze()`. Not part of this overhaul unless the owner promotes it; Phase 4
  must work without it, using stage and step events only. If it is promoted,
  define the event types in `RunView`'s package first so a later API layer can
  reuse them.

### Exit criteria

- No UI code writes API keys, objectives, output paths or model settings to
  `os.environ`; a repository search for `os.environ` in `app.py` and `ui/`
  (including `.pop` and `.setdefault`) returns nothing.
- Two concurrent sessions cannot read each other's key, objective, results,
  run history, or rate-limit budget, and a visitor cannot make the server fetch
  a URL they typed.
- Every tab renders from a `RunView`; no tab calls `find_tool` or reads
  `dashboard.json`.
- `app.py` is a thin entry point; no module in `ui/` exceeds a size the owner
  agrees on (suggested ceiling: 500 lines).
- The dead code listed in task C is gone and nothing imports it.
- Run directories are created under the configured root and removed on reset.
- AppTest smoke tests for the four tabs pass; `ruff check .`, `mypy src/` and
  `pytest tests/` pass.
- The diff to `controller.py` and `llm_client.py` is reported to the owner and
  touches neither decomposition logic nor the memory schema.

## 6. Phase 2: information architecture and copy overhaul

### Landing narrative

Replace the current marketing-template stack with a tighter narrative:

1. **Promise:** one clear sentence about checked answers.
2. **Input:** dataset plus plain-English question.
3. **Proof:** one clearly labelled audited finding with its evidence and caveat.
4. **Method:** the seven-stage process in plain language.
5. **3D explanation:** an optional living object explains one transformation,
   check, or report-formation step. It never becomes a decorative system
   diagram or the only explanation.
6. **Trust:** reproducibility, uncertainty, and governance, plus a plain
   statement of the data path. The original text led with "local processing",
   which is untrue on a hosted server: the file is uploaded to that server and,
   unless the run uses no LLM or a local model, summaries of it go to the chosen
   LLM provider. The page states exactly that, names the run-without-AI option,
   and makes no privacy claim stronger than the deployment can prove (section
   2b). The owner supplies the retention period and hosting facts before the
   copy is final.
7. **Action:** one primary entry CTA and one secondary documentation action.

Remove or rewrite:

- duplicate hero and workflow copy;
- all-caps telemetry and fake system-status chrome;
- unsupported exact claims such as guaranteed percentages or universal
  safety labels;
- jargon such as `swarm synapses`, `concentric manifold`, and `certified safe`
  when the product does not generate that exact evidence;
- pricing, enterprise, or testimonial sections unless the product actually
  supports them;
- decorative stage dots that do not add navigational value.

### Workspace information architecture

- Make the first screen answer three questions: what should I provide, what
  will happen, and what will I receive.
- Put file upload, objective, and Run in one calm primary task area.
- Keep advanced provider, model, privacy, sandbox, and thoroughness settings
  available through a clear advanced section without making them look required.
- During a run, show one progress narrative with current stage, elapsed state,
  cancel behavior, and what remains.
- After a run, keep `Answers`, `Charts`, `Details`, and `Downloads`, but make
  their hierarchy and labels consistent with the user's task.
- Fix the completed-result hierarchy as **answer, evidence, caveat, next
  action**, then technical metrics and trace details.
- Surface the verdict and evidence checks before technical metrics.
- Keep technical details available through progressive disclosure, not hidden
  behind unexplained jargon.
- Remove the misleading implication that keyword search is a conversational
  follow-up feature unless true follow-up execution is implemented.

### Report information architecture

- Lead with objective, verdict, headline findings, and evidence checks.
- Follow with supporting charts and methodology.
- Put governance, run trace, and raw technical output at the end.
- Make the report readable without JavaScript and usable when printed.
- Pair every chart with a plain-language takeaway, a source or provenance
  statement when available, and an accessible description or data table.

### Exit criteria

- Each surface has one primary action.
- Each stage and finding has plain-language copy.
- All numeric claims are either data-bound or explicitly labelled as examples.
- No duplicate section repeats the same message without a new purpose.

## 7. Phase 3: landing page rebuild

### Tasks

- Replace the inline monolith in `ui/landing_component/index.html` with the
  modular landing architecture.
- Keep Streamlit component messaging for Enter, theme, and keyboard actions.
- Make native document scroll the default. Do not use fullPage.js or hijack
  wheel/touch scrolling.
- Ship a stable static hero, promise, and entry CTA before loading any 3D
  module. Progressively enhance with one 3D stage only where it adds
  explanatory value. Let the rest of the page scroll normally.
- Use GSAP only if a reviewed narrative sequence cannot be expressed with CSS
  or the Web Animations API. Do not add Anime.js to the landing.
- Build responsive layouts for small phone, large phone, tablet, laptop, and
  wide desktop instead of relying on one breakpoint.
- Ensure all copy and controls remain available when WebGL is unavailable,
  blocked, or disabled by reduced motion.
- Give the hero a stable first paint before loading the 3D module.
- Use one CTA path into the workspace. Keep the hidden Streamlit bridge
  invisible but robust and keyboard accessible.
- Add semantic landmarks, heading order, focus states, skip navigation, and
  readable link labels.
- Remove duplicate prototype behavior from `frontend-landing` only after the
  new page is the verified live source.

### Landing motion contract

- One renderer clock drives 3D motion.
- Scroll progress is sampled through `IntersectionObserver`, CSS scroll-driven
  animation, or a bounded GSAP ScrollTrigger. Do not attach an uncontrolled
  global scroll render path.
- One continuous progress scalar drives camera, formation, material emphasis,
  and relevant DOM state.
- UI transitions use CSS or the Web Animations API. GSAP may own an approved
  scroll narrative but must not run a second, competing scene ticker.
- Transitions are interruptible and reversible.
- Reduced motion disables ambient movement, parallax, autoplay, and camera
  travel while leaving all content and navigation available.
- Any autoplay longer than five seconds has a visible pause control.

### Exit criteria

- Landing communicates the product in one viewport without requiring 3D.
- The 3D scene enhances the story instead of obscuring text.
- No scroll hijacking, dual clock, third animation engine, or unbounded
  animation remains.
- The live component and standalone preview use the same source and behavior.

## 8. Phase 4: Streamlit workspace redesign

### Tasks

- Recompose the empty state as a clear first-run workspace, not a marketing
  continuation of the landing page.
- Redesign the upload/objective/run area with clear grouping and stronger
  visual affordance for the next action.
- Simplify the sidebar into sections with progressive disclosure and plain
  labels. Keep privacy and execution controls prominent enough to be trusted.
- Rebuild the run state around a readable stage timeline and explicit cancel
  behavior. Do not animate static cards as if they were interactive.
- Replace repeated generic cards with a small set of purpose-specific surfaces:
  input, evidence, verdict, chart, trace, warning, and export.
- Redesign tabs so the active state is clear without a full-width pill bar.
- Use a responsive layout that remains useful at 768px and below without
  relying on fragile Streamlit internal selectors.
- Keep charts and tables as data products. Improve surrounding framing,
  captions, empty states, and drill-down controls without changing chart data.
- Make the first Answer view lead with the audited verdict, evidence checks,
  uncertainty, and recommended next action.
- Keep 3D out of the default Answers and Charts experience. The answer,
  evidence, caveat, chart annotation, and next action must be readable without
  any spatial interaction.
- Redesign Details as an inspectable audit trail rather than a wall of
  technical containers.
- Make Downloads feel like a deliberate artifact shelf with clear file type,
  purpose, and availability state.
- Ensure all HTML interpolation remains escaped and all controls have labels.
- Build Details from the `RunView` (Phase 1b) as an audit trail with a "How we
  got here" section: the mode decision and the options it rejected, what fell
  back and why, how many numbers the verbatim-citation guard could not trace,
  which requested outputs were delivered or missing, which hypotheses were
  tested and refuted, and a usage footer (time, tokens, cost) when the run had
  an LLM. Plain language only; technical terms keep a companion sentence.
  None of this appears today in the console.
- Keep the audited check row and the "N of M findings held up" banner that
  Answers already renders (`ui/tabs/answers_tab.py:109-135`), make them the
  first thing on the tab, and add the same check row to Charts panels. An
  absent check renders nothing, never a tick.
- During a run, show a skeleton shaped like the Answers layout so the page
  holds its layout. It must not animate. (Add it; no skeleton exists in the
  committed tree.)
- Replace the hard-coded sample demo with a real run (section 2b). The run gate
  today needs an API key even with the LLM off, so change it to
  `has_key or not use_llm`, and have the sample button force the LLM off
  whatever the sidebar says. Whether ML is on for the sample is decided from the
  Phase 0 timing: on if the timed run is short enough to be a good first
  impression, off otherwise (the model verdict is part of the audited-entry
  story, so prefer on). The sample button starts the deterministic pipeline (no
  LLM, no key) on
  `data/sample_customer_churn.csv` through the same worker as any other run, and
  the result is a real `RunView`. Delete `_load_teamwork_preview` and its
  placeholder report files. Label the run as the bundled sample file so nobody
  mistakes it for their own data. Because it is a real run it also exercises the
  audited entry and the Downloads artifacts, and a second cached copy of a
  real run may serve as the AppTest fixture (Phase 1b E).
- Limit workspace motion to CSS or Web Animations API feedback such as state
  changes, loading skeletons, and carefully bounded reveals. Do not load GSAP,
  Anime.js, or a continuous WebGL scene into ordinary workspace views.

### Exit criteria

- A first-time user can upload, ask, and run without opening advanced settings.
- A completed run makes the answer, confidence/caveat, and evidence visible
  before deep technical detail.
- A completed run presents the same answer, evidence, caveat, and next action
  in keyboard, no-motion, and no-WebGL modes.
- Every interactive element has a visible focus state and a keyboard path.
- No static card receives a lift animation that implies it is clickable.

## 9. Phase 5: 3D runtime and visual overhaul

### Shared runtime invariants

- One clock and one lifecycle owner per scene.
- No scene shares animation ownership with Anime.js. GSAP may feed a bounded
  landing or cinematic progress value, but it never owns the render loop.
- Frame-rate-independent damping using `dt`.
- No per-frame object or array allocations in hot loops.
- No per-frame layout reads followed by writes.
- No direct DOM text churn every frame unless the value actually changed.
- Camera matrices are updated before projection.
- Canvas remains outside transformed DOM subtrees.
- Renderer pixel ratio is capped and selected by device tier.
- Render loop pauses when offscreen, hidden, unsupported, or not dirty.
- Visibility changes reset the clock delta to prevent jumps on return.
- Every listener, observer, renderer, geometry, material, and texture has a
  teardown path.

### Visual system

- Replace the current token cloud with one meaningful visual metaphor selected
  during the direction gate: evidence being sorted, checked, compared, or
  bound into a report. The semantic explanation begins in DOM text and the
  scene only reinforces it.
- Use stable instance identity across formations so morphs read as continuity,
  not disappearance and reappearance.
- Use a restrained material set with declared color management, controlled
  roughness/metalness, and a simple key/fill/rim lighting rig.
- Treat risk and positive colors as semantic states, not ambient decoration.
- Remove unnecessary grids, glows, line noise, and perpetual particle motion.
- Prefer one focal object and supporting geometry over many competing objects.

### Pipeline plate

- Preserve the seven workflow stages and stage statuses from Python.
- Make the seven stages readable in text first and reflected in the 3D state.
- Keep a visible textual timeline synchronized with the scene at all times;
  keyboard focus and screen-reader navigation operate on that timeline, not on
  opaque canvas geometry.
- Keep CAD perspective controls only if user testing shows they help. Otherwise
  replace them with one simple inspect/reset interaction.
- Make the fallback list fully equivalent in meaning to the 3D scene.
- Keep pointer interaction local to the canvas, coalesce pointer state, and do
  raycasting in the render loop or on a controlled dirty path.

### Cinematic export

- Treat the cinematic export as optional polish after the upload-to-answer
  path works. It must not become the primary way to understand results or
  obtain the report.
- Replace fullPage.js section hijacking with native scroll and accessible
  section navigation.
- Share the same scene formations, camera contract, themes, and reduced-motion
  behavior as the pipeline plate.
- Make export state data-bound. Never display preview numbers as if they came
  from the current run.
- Add a static poster/fallback and a text-only path for print and no-WebGL use.
- Keep standalone export self-contained and offline-capable.

### Performance tiers

Define and measure at least:

- full desktop;
- standard laptop;
- mobile GPU;
- low-power or no-WebGL fallback.

Per tier, select instance count, DPR, antialiasing, postprocessing, ambient
motion, and interaction detail. Establish budgets from measurement and record
them in the scene documentation. Do not claim smoothness without testing a
real laptop and phone.

### Exit criteria

- The scene remains visually continuous through forward, reverse, and
  interrupted transitions.
- 3D motion is smooth at the chosen budgets on target devices.
- Reduced motion, no WebGL, hidden tab, offscreen, and slow-network paths are
  complete experiences rather than error states.
- Memory and GPU resources are released when an iframe or export is torn down.
- The text-first timeline and fallback communicate the same active-stage and
  process meaning as the scene.

## 10. Phase 6: report and export redesign

### Tasks

- Share the canonical tokens and typography with the Streamlit UI.
- Remove reliance on Google Fonts and CDN Vega for the normal offline path.
- Keep the report functional when scripts fail: headings, findings, tables,
  and methodology must still render. Today fonts and Vega load from CDNs and
  every chart is drawn client-side, so with scripts or the network off a chart
  is blank. Inline the vendored bundles in `static/vendor/vega/` and the
  Latin-subset fonts from `static/fonts/` as data URIs, with the CDN tags as
  the fallback if `static/` is missing. Render
  each chart's data table (or a static SVG) server-side inside the chart
  container and let the script replace it, so the no-script path is the same
  content, not an empty box.
- Keep the inlined Vega runtime's size in view: it is embedded in every report.
  Record the report's byte size in Phase 0 and set a budget.
- Improve print CSS: page breaks, contrast, table repetition, hidden controls,
  and readable URLs only where useful.
- Make charts responsive without clipping and preserve accessible descriptions.
- Make the HTML export visually related to the report without copying the
  cinematic chrome into a document that should be read.
- Use static 3D posters only when they convey a real process concept, and pair
  them with a concise caption plus the full text path. Do not embed a live 3D
  runtime in the standard report.
- Add explicit export states for missing reports, unavailable models, and
  incomplete runs.

### Exit criteria

- A report can be opened from a disconnected machine with all required local
  assets.
- A printed report is readable and does not expose interactive-only chrome.
- Exported values match the run state and hostile text remains escaped.

## 11. Phase 7: accessibility, content, and interaction quality

### Accessibility checks

- Keyboard-only navigation through landing, workspace, tabs, dialogs,
  controls, stage navigation, and downloads.
- Visible focus at normal and high-contrast settings.
- Correct landmarks, heading hierarchy, labels, descriptions, and live-region
  announcements for run progress and errors.
- No information conveyed by color alone.
- Contrast checked for both themes, including charts and disabled controls.
- 200% zoom and narrow viewport checks without hidden essential content.
- Reduced-motion behavior tested by changing the media query at runtime.
- Screen-reader path for the 3D scene that exposes the same stage information.
- Short and long descriptions, annotations, and data-table alternatives for
  charts, diagrams, and meaningful 3D scenes.
- Touch targets at least 44px where applicable.

### Content checks

- Sentence case throughout user-facing UI.
- Plain verbs and short explanations before technical terms.
- No emoji as the primary status or navigation icon.
- No fake precision, unsupported pricing, fake certification, or absolute
  privacy claims that the runtime cannot prove.
- One naming system for the seven stages across landing, console, plate,
  cinematic export, and report.
- No user-facing explanation depends on spatial reasoning, animation, color,
  or a technical term without a plain-language companion.

## 12. Phase 8: implementation order and migration strategy

0. Commit the current working tree as the baseline (two commits), then run
   Phase 0. Phase 0 runs before any other implementation change.
1. Create the visual direction board, 3D-placement matrix, and acceptance
   checklist. In parallel, and not blocked by the board, run Phase 1b
   (`RunConfig`, `RunView`, module split, hygiene, UI tests), and the Phase 1
   embedding spike.
2. Add shared token serialization, frontend contract validation, and chart or
   scene description contracts.
3. Build the shared motion/lifecycle/accessibility utilities and enforce one
   runtime owner per surface.
4. Build the new landing shell with a static/no-WebGL version first.
5. Add the new 3D runtime behind the landing shell and validate it in a
   standalone local preview.
6. Switch `ui/landing.py` to the new verified source.
7. Redesign the Streamlit empty, input, running, and completed states. Phase
   1b must be complete first; this step consumes `RunView` and the split
   modules.
8. Migrate the pipeline plate to the shared runtime.
9. Migrate the cinematic export to the same runtime and native scroll model.
10. Migrate report tokens and offline assets.
11. Remove stale duplicate behavior only after import, test, and preview
    searches prove it is unused. Preserve an archive or migration note when
    deletion is not necessary.
12. Recommended: run five to seven non-technical, task-based usability checks.
    Feed observed comprehension failures back into the content and interaction
    design. Not a gate.
13. Run the complete verification matrix. Record remaining intentional
    tradeoffs in `IMPROVEMENTS.md` (one line per closed or deferred item, the
    ledger convention that file already uses) and keep a dated status log at
    the end of this plan. `HANDOVER.md` no longer exists.

## 13. Testing and verification matrix

### Static and repository checks

- `ruff check .`
- `mypy src/`
- `pytest tests/ -v`
- no API keys, `.env` files, or large datasets in tracked files;
- architecture import checks remain clean;
- frontend contract tests cover all Python-to-browser state payloads.

### Browser and visual checks

Test with the real Streamlit app and standalone exports at:

- 1440 x 900 desktop;
- 1280 x 800 laptop;
- 1024 x 768 tablet landscape;
- 768 x 1024 tablet portrait;
- 390 x 844 phone;
- 320px minimum content width where practical.

For each viewport, test day/night, empty state, uploaded state, running state,
completed state, error state, no-WebGL fallback, reduced motion, keyboard
navigation, and 200% zoom. Capture before/after screenshots and inspect the
middle of every 3D transition, reverse transition, and interrupted transition.

### Runtime checks

- no console errors or unhandled promise rejections;
- no duplicate observers after Streamlit reruns;
- no iframe rebuild for unrelated state changes unless required;
- no animation loop while hidden or offscreen;
- stable frame pacing on target desktop and mobile devices;
- no layout shift after fonts, 3D, or report assets load;
- teardown releases WebGL and event-listener resources.
- no surface loads more than its designated animation/runtime owners.

### Product checks

- seven-stage workflow remains intact and accurately labelled;
- all findings and metrics shown in the UI come from the current run;
- unfinished checks never render as passed;
- local-only and sandbox status are truthful;
- generated report exists in `output/reports/` after a sample run;
- the end-to-end sample CSV workflow completes without uncaught exceptions.
- recommended: five to seven representative non-technical participants can
  explain the promise, start a run, find the answer and caveat, and export a
  report without facilitator rescue. Record failures as design issues, not as
  user error. Not required for completion.

## 14. Definition of done

The overhaul is complete only when:

- all five frontend surfaces use one coherent visual system;
- the live landing no longer diverges from its standalone preview;
- the 3D runtime is smooth, interruptible, lifecycle-safe, and accessible;
- content remains useful with no WebGL, reduced motion, no network, or print;
- the console answers the user's question before exposing technical machinery;
- three-dimensional interaction is present only where it clarifies the
  workflow and is never needed to understand an answer or report;
- all visible claims are data-bound or clearly labelled as examples;
- visual screenshots have been reviewed at desktop, tablet, and mobile sizes;
- `ruff`, `mypy`, and the full test suite pass;
- the seven-stage sample workflow and report generation are verified;
- if the recommended non-technical usability checks were run, their unresolved
  findings are documented; if they were not run, the status log says so;
- `IMPROVEMENTS.md` and the status log in this plan record remaining intentional
  tradeoffs;
- Phase 1b exit criteria are met, including session isolation on the hosted
  deployment;
- no surface loads a third-party script, font or stylesheet at runtime.

## 15. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Modular ES source cannot load inside a string-built iframe | Phase 1 architecture is wrong; the plate and landing diverge again | Embedding spike in Phase 1, result recorded before Phase 3 |
| Phase 1b touches `controller.py` and `llm_client.py` | A regression in the files every run depends on | Keep defaults identical; `RunConfig.from_env()` for the CLI; report the diff; static gates after each change |
| Environment reads are lazy and tools run on thread pools | A per-run setting silently falls back to the process default in a pool thread | Audit all read sites first; prefer explicit parameters; add a test that checks a pool-thread tool sees the run's value |
| Token rename touches six surfaces and two test files | Breakage across surfaces at once | Aliases first, one surface at a time, old names removed last |
| Streamlit minor upgrades change internal selectors | Theme breaks silently | A screenshot baseline captured in Phase 0 (script optional) and compared before any Streamlit bump; `requirements.lock:172` pins 1.57.0, but `requirements.txt` and `pyproject.toml` allow `>=1.49`, so deploy from the lock file |
| Run history and rate limits are shared across visitors | One visitor sees another's summaries or is throttled by another's usage | Per-session runs directory and per-session limiter scope (Phase 1b A and D) |
| A visitor-typed model URL is fetched by the server | Server-side request forgery from a hosted page | Local base URL is operator-only (Phase 1b A) |
| Hosted users edit operator controls from the page | Isolation or code-execution limits bypassed | Read-only operator status on the hosted deployment (Phase 1b A) |
| The sample button depends on a deterministic run | If the run fails, first impression is an error | Keep the sample file in the repo; Phase 0 exercises this path; show a plain error with the technical details collapsed |
| Usability checks are skipped | Comprehension failures go unseen | They stay recommended; the status log records that they were not run |
| 3D scope grows again | The text path and the report slip | The 3D placement matrix; cinematic export stays optional polish; Phase 4 never depends on it |
| The committed UI is unconfirmed in a browser during this review | Phase 0 may find defects the plan did not predict | Phase 0 runs before any new implementation |
| Uncommitted work is lost again | An earlier round was wiped by a reset | Work on a branch; commit after every step |

## 16. Standing rules for whoever implements this plan

- Follow `AGENTS.md`: type hints on every function, reasoning in `src/core/` and
  execution in `src/tools/`, no raw data in prompts, no keys in source.
- No testing or running of the app is done unless the owner asks; the Phase 0
  run and the browser checks are the places to ask.
- Quality gates for any code change: `ruff check .` and `mypy src/`, plus the
  tests the change touches. The full gate list is in section 13.
- One concern per commit; no co-author trailer.
- Do not delete a file until a repository search shows no importers and the
  replacement is live.
- Do not pick a visual direction. Present options and wait for approval.
- When this plan and a decision in section 2b disagree, the decision wins and
  the plan is edited.

## 17. Open questions for the owner

Resolved on 2026-10-02 (recorded in section 2b): the sample demo is replaced by
a real run; the landing states the data path plainly; keys are per session and
operator safety settings are locked; the palette is open; the usability checks
are recommended, not a gate.

Also resolved: the project is a final-year project (section 2d), so the
retention-period question is dropped; the landing states that files are removed
when the session ends and carries an academic-project notice.

Still open:

1. **Where and when (blocks the Must and Should cut in section 2d).** The
   deadline and the demo date; whether the project is shown live from a laptop,
   from a public free host, or from a college server; whether the examiners
   will be given a link to use themselves; and what is assessed (working demo,
   written report, code, viva). The tiers in section 2d are a proposal until
   these are known.
2. **Operator settings list (blocks Phase 1b A).** Confirm the split in the
   decisions log: users edit provider, model, thoroughness and ML options; the
   operator locks code execution, isolation, run caps, token caps and
   `LOCAL_ONLY`. Recommendation, not a neutral option: also lock the local-model
   base URL and the "local" provider to the operator, because a visitor-typed
   URL is fetched by the server (finding 2c.10). Name anything else that should
   move to the other side.
3. **`analyze_iter` (blocks nothing, shapes Phase 4).** Promote the event
   stream into this overhaul, or leave it deferred and design the progress view
   for stage and step events only? The plan assumes deferred.

## 18. Status log

| Date | Entry |
|---|---|
| 2026-10-02 | Plan authored by a first agent; reviewed against the repository and extended with sections 2a to 2c, Phase 1b and sections 15 to 17. Decisions in section 2b recorded. No code changed. Nothing in section 2a has been viewed in a browser. |
| 2026-10-02 | Owner stated this is a final-year project. Added section 2d (must, should, defer tiers and demo-host notes), recorded the decision, and replaced the retention question with a deadline and demo-context question. |
| 2026-10-02 | Owner answered the first four questions (demo, trust claim, keys and settings, palette and usability gate); answers applied to the phases and decisions log. Open: hosting facts, operator settings confirmation, `analyze_iter`. |
| 2026-10-02 | The owner reverted an earlier uncommitted UI round (the working tree was reset to `1585d9d`). Sections 2a and 2c, Phase 0, 1, 1b, 4, 6, section 2d and the risks were rewritten against the committed tree; threaded run, Stop, offline report, console fonts, Charts overview and the snapshot script are open work again. |
| 2026-10-02 | Phase 1b A implemented on the committed tree (static gates only: `ruff check .` and `mypy src/` pass; not run in a browser or under pytest). Added `src/core/run_config.py` (`RunConfig`) and `src/core/run_context.py` (per-thread objective, carried into the controller's tool thread pool). `LLMClient` and `AgentController` accept a `RunConfig`; three tools stopped reading `USER_OBJECTIVE` from the environment. `app.py` stopped writing provider, model, key, URL, objective, output directory, iterations and switches to `os.environ`; a `DSA_HOSTED` operator flag hides the local provider, ignores the server key, and shows privacy and code-execution limits read-only; the run gate is `has_key or not use_llm`; the fabricated demo is replaced by a real no-AI sample run (`_sample_run`) labelled as the bundled file; per-run temp directories are prefixed and removed on reset; run comparison is local-only. Review fixes the same day: the objective now lives in `objective_scope` around `load_dataset` and `analyze` (it no longer outlives the run), and the tool thread pool runs each call in its own context copy (one context object cannot be entered by several threads). Known small deviation: the RLM engine's `llm_concurrency()` no longer sees the UI-selected provider, so a local-model run with decomposition on uses the cloud default of 4 parallel calls instead of 1 (the engine is Ask-First, so it was left alone). **Status: written and statically checked only. It has not been run, and the concurrent tool batch in particular needs one real run before this item is called done.** Remaining Phase 1b: `RunView`, module split, session-isolation and AppTest tests. |
