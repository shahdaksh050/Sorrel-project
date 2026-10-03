# DSA Agent — Next Steps Implementation Plan

**Status:** proposed  
**Basis:** read-only audit of `frontend-overhaul` against `master` (2026-10-02)  
**Scope:** reliability, deployment safety, user experience, analytical trust, and documentation.  
**Out of scope:** replacing the analysis architecture, adding new analysis tools, or redesigning the RLM engine.

---

## 1. Goal

Make the current frontend-overhaul branch safe to release and easier to use without sacrificing the evidence, audit trail, or safety controls that distinguish DSA Agent from a generic dashboard.

The desired default experience is:

```text
Upload dataset → ask a question → run → read a plain-language answer
                                      ↳ inspect evidence, charts, and downloads when needed
```

The product should remain useful in deterministic/no-LLM mode, should preserve the distinction between partial and complete analyses, and should make advanced controls available without making them the primary user journey.

---

## 2. Current strengths to retain

- Typed, deterministic execution tools remain separate from LLM planning.
- No-LLM and no-ML paths remain first-class, not degraded fallback paths.
- Per-run LLM configuration and `ContextVar` objective isolation prevent cross-run prompt/key leakage.
- Background runs, live progress, cooperative stop, and partial-result reporting improve perceived responsiveness.
- Findings, caveats, evidence checks, multiple-testing controls, and degradation notes remain visible.
- Model pickle HMAC signatures and output-path confinement remain mandatory.
- The normal HTML report stays self-contained and has a data-table fallback for charts.
- The primary workspace stays answers-first; technical detail belongs in Details and Downloads.

---

## 3. Delivery order

| Phase | Outcome | Release gate |
| --- | --- | --- |
| 0 | Clean, reproducible branch | Quality commands and smoke test pass |
| 1 | Safe concurrent execution | No user-run configuration crosses run boundaries |
| 2 | Frictionless default workflow | A new user can complete a no-LLM and LLM run without configuration confusion |
| 3 | Trustworthy, understandable results | Each answer explains evidence, limits, and next action |
| 4 | Honest distribution and documentation | Install, deployment, export, and roadmap claims match reality |

Phases 0 and 1 are release blockers. Phases 2–4 are the highest-value product follow-up work.

---

## 4. Phase 0 — Quality and reproducibility

### 0.1 Fix patch hygiene

**Problem**

`git diff --check master...frontend-overhaul` reports trailing blank lines in `src/core/controller.py` and whitespace in two vendored Three.js copies.

**Work**

1. Remove trailing blank lines from first-party Python files.
2. Treat vendored JavaScript as third-party source:
   - either preserve it byte-for-byte and exclude it from whitespace enforcement, or
   - replace it with an upstream version whose checksum/version is recorded.
3. Add a small CI check that fails on first-party whitespace errors but does not ask formatters to rewrite vendor assets.

**Acceptance criteria**

- `git diff --check master...frontend-overhaul` has no first-party findings.
- `ruff check .` passes with zero violations.
- Vendored asset version and licence files are retained.

### 0.2 Restore a repeatable local development environment

**Problem**

The audit environment could not start the registered Python 3.13 interpreter, and `ruff` was unavailable on PATH. The repository should not depend on a machine-specific WindowsApps launcher state.

**Work**

1. Document one supported bootstrap flow using `.venv`.
2. Add a short environment verification command/script that reports Python version and required tools before tests run.
3. Decide whether `requirements.lock` is the canonical reproducible install or whether it should be regenerated/removed; do not leave two ambiguous dependency sources.

**Acceptance criteria**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ruff check .
python -m mypy src/
python -m pytest tests/ -v
python scripts/validate.py
python scripts/dry_run.py
```

All commands complete successfully on a clean Windows and Linux checkout.

### 0.3 Add an explicit release command

Create a non-destructive `scripts/release_check.py` (or documented task runner) that invokes the existing quality sequence and reports the first actionable failure. It must not silently skip browser smoke, type checking, or the dry run.

---

## 5. Phase 1 — Correct run isolation and lifecycle

### 1.1 Make all user-selected run settings per-run

**Problem**

The UI passes provider/model/key through `RunConfig`, but writes code-execution, sandbox-isolation, budget, and minimum-cell-size choices into process-wide `os.environ`. With concurrent analyses enabled, a local/shared deployment can apply one visitor's settings to another visitor's run.

**Work**

1. Extend the immutable per-run configuration object with:
   - `enable_code_execution: bool`
   - `max_code_executions: int`
   - `sandbox_require_isolation: bool`
   - `min_cell_size: int`
2. Pass that configuration from `app.py` → `RunSpec` → `AgentController`.
3. Make governance, sandbox, and privacy readers prefer explicit run configuration over environment defaults.
4. Preserve environment variables as operator defaults only.
5. Do not modify the `MemorySystem` schema as part of this work.

**Architectural constraint**

The controller may orchestrate the configuration. Tools should receive already-resolved values through their normal parameters/context and must not mutate global execution policy.

**Acceptance criteria**

- Two concurrent `ActiveRun` instances with opposite safety settings retain their own values throughout execution.
- A hosted deployment remains locked to operator-defined settings.
- Existing CLI environment-variable behavior remains backward compatible.
- New configuration helper functions have unit tests and PEP 484 annotations.

### 1.2 Fix retained-summary collisions

**Problem**

Retained UI summaries are written to `output/runs/ui-YYYYMMDD-HHMMSS/reports`. Two runs completing in the same second can overwrite each other.

**Work**

1. Use the run/session ID (or a UUID generated at run creation) in the retained path.
2. Store a minimal manifest beside the summary: run ID, creation time, dataset display name, objective, and completion state.
3. Keep the output write atomic where practical: write to a temporary file, then replace.

**Acceptance criteria**

- Concurrent completions never share a report directory.
- Run comparison lists the correct dataset/objective for each stored summary.
- A unit test forces same-timestamp completions and proves no overwrite occurs.

### 1.3 Complete lifecycle controls

**Work**

1. Add a visible **Delete this run** action after completion.
2. Explain the difference between:
   - Stop: finish the current tool and produce a partial report.
   - New analysis: discard the active run when it reaches a safe checkpoint.
   - Delete: remove retained local artefacts for the displayed run.
3. When run slots are full, show queue position if it can be calculated safely; otherwise show an honest queued state without an invented ETA.
4. Add a small retention statement to the UI and documentation. Closed-tab temporary directories are currently swept later, not immediately.

**Acceptance criteria**

- A user can remove a completed run without deleting anything outside its resolved run directory.
- Stop, discard, and delete each have dedicated tests.
- No path deletion accepts arbitrary user-controlled paths.

---

## 6. Phase 2 — Improve the primary user experience

### 2.1 Make the default path opinionated and short

The first screen after the landing page should foreground only:

1. Dataset upload.
2. Optional plain-English question.
3. Run analysis.

Keep a visible no-LLM option, but move provider/model, RLM, ML tuning, model depth, sandbox settings, related-table overrides, and privacy controls into an **Advanced settings** expander or a configuration drawer.

**Acceptance criteria**

- A first-time user can run the bundled sample without an API key or provider setup.
- A user with a valid `.env` key can run without pasting it into the UI.
- A user who switches AI narrative off is never blocked by model discovery or API-key requirements.
- The browser smoke test covers both the default sample route and an `.env`-key-equivalent UI-state test.

### 2.2 Fix API-key run eligibility

**Problem**

The UI resolves `_effective_key` from user input or the local environment, but enables the Run button only when the textbox itself is populated.

**Work**

Use the resolved value for eligibility in non-hosted mode. Never expose whether a server key exists to hosted visitors.

**Acceptance criteria**

- Local `.env` configuration enables the Run button.
- Hosted mode requires a user-supplied key and does not disclose server credentials.
- Tests cover local/hosted behavior for every supported provider.

### 2.3 Keep visual polish subordinate to task completion

The landing page and workflow plate are useful orientation aids, but not the work product.

**Work**

1. Keep the text workflow/status fallback permanently available.
2. Honor reduced-motion preferences across landing, plate, and cinematic views.
3. Defer or lazy-load expensive 3D content until it is requested.
4. Measure initial interactive time on a typical laptop and phone-width layout.
5. Do not add more decorative surfaces until the default run path meets the reliability gates above.

**Acceptance criteria**

- Upload and sample-run controls are visible without needing to interact with 3D content.
- Keyboard navigation reaches all essential controls.
- A failure to load Three.js/animation libraries never hides the textual workflow state.

---

## 7. Phase 3 — Make answers easier to trust and act on

### 3.1 Standardise the Answer tab hierarchy

Every completed or partial result should answer, in this order:

1. **What we found** — up to three directly relevant findings.
2. **How strong is the evidence?** — sample size, effect/metric, uncertainty, and audit/check status.
3. **What could change the conclusion?** — data-quality, leakage, confounding, or stop/degradation notes.
4. **What to do next** — one to three actions, labelled as analysis recommendations rather than facts.

Do not show a synthetic certainty score when no relevant check ran. The current distinction between audited and unaudited findings should remain explicit.

### 3.2 Explain analysis selection

Add a compact, deterministic **Why this analysis?** panel containing:

- declared/inferred target and task type;
- selected methods and why they applied;
- excluded methods and why they did not apply;
- row count after cleaning and important treatments;
- whether the result used deterministic mode, an LLM, RLM decomposition, ML, or a partial run.

This is derived from existing profile, plan, degradation, and governance data; it should not require a new LLM call.

### 3.3 Tighten ML validity messaging

**Work**

1. Make five folds the default and minimum for ordinary i.i.d. data, as declared by repository policy.
2. For small, chronological, or grouped datasets where fewer splits are unavoidable, show the actual split count and an explicit reliability warning.
3. Avoid presenting train/test performance as deployment readiness. Keep CV as the primary ranking signal.
4. Add tests for the minimum-fold rule and each split strategy.

**Acceptance criteria**

- The UI cannot silently request three-fold i.i.d. validation while claiming the five-fold policy.
- Every model result states split strategy, fold count, hold-out size, train/test gap, and overfit warning status.

---

## 8. Phase 4 — Documentation, packaging, and exports

### 4.1 Make documentation match the product

1. Update README wording that says the frontend overhaul is planned; it is substantially implemented on this branch.
2. Split the roadmap into **delivered**, **in progress**, and **future** sections.
3. Add a deployment matrix:

| Mode | Intended audience | Key source | Code execution | Isolation |
| --- | --- | --- | --- | --- |
| Local single user | Developer / analyst | `.env` or UI | User-configurable | Optional |
| Hosted/shared | External visitors | Visitor-supplied | Off by default | Required by default |

4. State realistic temporary-data retention behavior, including stale-run cleanup.

### 4.2 Align installation metadata

1. Decide whether DuckDB is optional or required.
2. If required, declare it in both `pyproject.toml` and `requirements.txt`.
3. If optional, put it in a named extra and make the UI/tooling disclose when it is unavailable.
4. Verify the package with both `pip install .` and `pip install -r requirements.txt`.

### 4.3 Make export claims precise

The normal analytical HTML report is designed to work offline. The cinematic export currently relies on external fonts and JavaScript CDNs.

Choose one path:

- vendor/version/pin all cinematic dependencies and make it genuinely offline; or
- retain CDNs, label the export **network-enabled**, and provide a static/text fallback.

---

## 9. Test plan

### Unit tests

Add or extend tests for:

- per-run safety configuration isolation;
- environment-key run eligibility in local and hosted modes;
- same-second run-summary collision avoidance;
- retained-run deletion confinement;
- ML fold policy and split-strategy exceptions;
- documentation/export mode flags where behavior is contractual.

### Integration tests

Extend the existing browser smoke suite to verify:

1. Landing → no-LLM sample → Answers/Charts/Details/Downloads.
2. Stop during a deliberately slowed tool and verify partial-report language.
3. Narrow/mobile viewport with no horizontal overflow.
4. 3D library failure still leaves a usable textual workflow.

### Non-functional checks

- No API keys in tracked files.
- `ruff check .` has zero first-party violations.
- `mypy src/` has zero errors.
- `pytest tests/ -v` passes.
- `scripts/validate.py` and `scripts/dry_run.py` pass.
- A sample CSV completes end-to-end and produces populated Markdown, JSON, and HTML reports.

---

## 10. Suggested milestone definition

### Release candidate

Ship only after Phases 0 and 1 complete, plus the default sample flow passes browser smoke.

### User-experience release

Ship Phases 2 and 3 together. The measure of success is not more interface surface area; it is a shorter path from upload to a cautious, useful answer.

### Follow-up release

Complete Phase 4 once packaging and documentation can be verified from a clean clone and exports describe their real connectivity requirements.

---

## 11. Guardrails during implementation

- Do not alter `src/rlm/engine.py`, task decomposition logic, or the `MemorySystem` schema without explicit approval.
- Keep controller reasoning and tool execution separate.
- Continue using `BaseTool.run()` from orchestration code.
- Every new utility or configuration helper needs typed signatures and focused tests.
- Do not place raw data rows in prompts; use existing metadata/context mechanisms.
- Treat deployment safety defaults as behavior, not cosmetic UI state.

