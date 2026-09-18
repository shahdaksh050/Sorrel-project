# Improvement Plan — next round

Status at time of writing (2026-09-18): the backend overhaul (sandbox `dsa`
toolkit, `DF_OUT`/`CHART`/`FINDING` pass-through, lightweight-LLM prompts,
profiler semantics, finding ranking, tool correctness fixes, chart rendering)
and the AI-safety/governance layer are written and pass `ruff` + `mypy`, but
**none of the new runtime paths have executed**. Everything below is ordered
by what unblocks the rest.

Recommended sequence: **Phase 0 → Phase 2's evaluation harness → Phase 1
(guided by harness results) → Phases 3–4.**

---

## Phase 0 — Verify & stabilise (before any new features)

1. **Fix the tests the last round broke.**
   - `tests/test_sandbox.py`, `tests/test_sandbox_worker.py`: new
     `ALLOWED_MODULES` (duckdb/polars removed), extended
     `BLOCKED_BUILTIN_NAMES`, `_worker_env(scratch_dir)` signature.
   - Dynamic-code / generated-tool tests: failures now raise
     `ToolExecutionError`; the old `status`/`error_type`/`traceback`/`hint`
     output keys are gone.
   - Then run the full `pytest tests/`.
2. **Exercise the runtime audit hook first** — highest-risk unrun code: it
   cannot be uninstalled, so a false positive on a legitimate library file
   access kills the worker mid-analysis. Run a real `execute_dynamic_code`
   step using `dsa.run(...)`, `CHART = dsa.chart...`, `DF_OUT` + `save_as`,
   and a `dsa.run("cluster_data", ...)` (KMeans/threadpoolctl path).
3. **Live runs with a free OpenRouter / NVIDIA model** on three datasets:
   the churn sample, the planted-effects transactional set, and a
   non-business set (sensor readings or a survey). Check that:
   - `data_understanding` is sensible and uses real columns;
   - the draft plan is edited, not copied verbatim;
   - open agenda questions get answered;
   - `final_result["governance"]` and the audit JSONL are populated.
4. **Open the dashboard and HTML report**: confirm LLM charts, log-scale
   histograms, scree dual axis, drivers chart and "Other" buckets render.
5. **Rebuild the Docker image** (it copies `src/`, so new modules are picked
   up) and run once with `SANDBOX_BACKEND=docker` and
   `SANDBOX_REQUIRE_ISOLATION=true`.
6. **Regression tests that lock in the last round:**
   - security corpus — known escapes that must stay blocked (dunder
     traversal, `getattr`, format-string walks, `read_*`, `fetch_*`,
     `dsa.run` path params, private toolkit state);
   - legitimate-idiom corpus that must keep passing (the 18 snippets
     already checked against `_static_check`);
   - planted-effects cases for sum-vs-mean aggregation, partial-final-period
     trimming, attrition polarity, concentration uniform baseline.

## Phase 1 — Analysis correctness (audit items not yet fixed)

- **Entity-level aggregation (pseudo-replication).** When
  `rows_per_entity > 1`, tests treat every row as independent evidence.
  Aggregate to one value per entity before inference; `profile.entity_col`
  is detected but no tool consumes it.
- **Simpson's-paradox check.** The prompt asks for it; no tool can do it.
  Add an optional stratification dimension to `segment_comparison` and
  report whether the effect holds within strata.
- **Run-wide multiple-testing correction.** p-values are corrected within
  each tool only. Pool across tools and write the run-level `p_adjusted`
  back onto findings before ranking.
- **Imputation.** Stop feeding median-imputed data into inferential tools
  (complete-case per analysis instead). For ML, impute inside the CV
  pipeline so the test fold doesn't leak into training.
- **Time series.**
  - per-entity series instead of one summed series (panel data);
  - explicit gap handling and calendar-aligned lags;
  - trend significance (Mann-Kendall) and STL seasonality instead of R²;
  - multiple-testing correction on month-of-year tests.
- **Column relationships.** Detect definitional pairs (total = qty × price,
  sum columns) and suppress them as findings; promote target correlations
  to findings; flag functional dependencies/hierarchies (city → country).
- **Data archetypes, not just domains.** `domains.py` knows three domains
  (financial, transactional, workforce). Add generic archetypes that drive
  the agenda: event log, panel, cross-section, survey/Likert, sensor time
  series, experiment/A-B test — so science, health and sports data get
  proper plans.
- **Remaining audit items:** no post-hoc test for k>2 groups; float group
  columns silently binned into quartiles; clustering on repeated rows;
  ML leakage threshold misfiring on ≥99%-majority targets.

## Phase 2 — Excellent results from lightweight LLMs

- **Evaluation harness (top priority of this phase).** Planted-effects
  datasets × models (free OpenRouter/NVIDIA tiers, later Ollama). Score:
  planted effects recovered, false findings, unverified claims, wasted
  steps, tokens, latency. Without it, prompt changes are guesswork.
- **Plan validator before execution.** Deterministic check that columns
  exist and parameters match each tool's schema; errors go back to the LLM
  in the same cycle instead of burning a tool run.
- **Constrained output.** JSON-schema / tool-calling modes where providers
  support them; grammar-constrained JSON (Ollama `format`, llama.cpp GBNF)
  for offline models, replacing after-the-fact repair.
- **Context budgeting.** Measure prompt size per provider/model and compact
  adaptively (digest length, findings count, profile columns) so 8k-context
  models never truncate.
- **Generated-tool library.** Persist and reuse generated tools across runs
  keyed on column-schema compatibility rather than exact file fingerprint;
  tools reused ≥ N times are queued for review and promotion to built-ins.

## Phase 3 — Charts & reports

- **A chart for every top finding kind** via deterministic `chart_hint`s:
  segment lift as bars with CIs, concentration as a Lorenz curve, change as
  a waterfall by segment, drivers as a dot plot.
- **Uncertainty display** — error bars/bands wherever a CI exists.
- **Headless spec validation** (`vl-convert`) in the test suite so an
  invalid Vega-Lite spec fails CI instead of rendering a blank panel.
- **Report sections** for `data_understanding` and the governance/audit
  summary in both the Markdown and HTML reports.
- **Delete or repurpose `src/tools/visualization.py`** — its PNGs reach no
  consumer; ROC/confusion matrix there are computed on training rows.

## Phase 4 — Safety & governance, next steps

- **Data egress to LLM providers.** The profile sends category values and
  stats to an external API. Add PII detection/redaction and a local-only
  mode (offline model, nothing leaves the machine).
- **Review mode.** Human-in-the-loop approval of code-executing steps in
  the UI before they run.
- **Audit viewer** in the Details tab; hashed log of every LLM call
  (prompt/response) alongside the code-execution log.
- **Per-run cost/token caps** using the existing usage tracking.
- **Docker as deployment default** with a seccomp profile and an image
  build in CI.
- **Decision to revisit:** duckdb/polars were removed from the sandbox
  allowlist (native I/O invisible to the audit hook). Restore behind the
  Docker backend only if SQL ergonomics prove valuable in the harness.

## Maintenance

- Split `src/core/controller.py` (~2,600 lines) into planner loop, step
  execution, and claim verification modules.
- Move `measure_aggregation` / `is_partial_final_period` from
  `src/tools/time_series.py` (imported by sibling tools) into the profiler
  or a shared stats module.
- `is_high_dimensional` no longer counts integer-coded dimensions — confirm
  the Stage 6 RLM decomposition trigger still fires where it should.
