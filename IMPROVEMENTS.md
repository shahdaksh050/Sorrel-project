# Backend Improvements — open backlog

**What this file is.** The open engineering backlog for the backend and the UI,
plus the current improvement plan (**Round 8**, live). **Items that have
already landed are not here in full** — Round 7 and its carried-forward
predecessors are commented out in place (`<!-- ... -->`, so the reasoning
stays readable in the raw file/git history without cluttering the live
backlog) and given one line each in the [closed ledger](#closed-ledger) at
the end. Everything visible in the body of this file is either Round 8 (live)
or a still-open item carried forward from an earlier round.

**Gate state, measured 2026-09-18 (end of the live-LLM testing pass, after the
`_loads_lenient` fix and its regression test).**
`pytest tests/ -q` (equivalently, the project's own default `-m "not slow"`
addopts) → **473 passed, 1 deselected, 0 failed** (472 plus the new
`test_literal_newline_in_string_value_still_parses`). The
`register_and_persist` cleanup that removed the stale `hasattr(memory,
"add_generated_tool")` guard (see the live-LLM pass note below) initially left
5 `test_tool_factory.py` tests red — their local `_FakeMemory` stubs predated
the guard's removal and didn't implement `add_generated_tool`; fixed by
updating the stubs, not the source, since `MemorySystem.add_generated_tool`
was already real (`memory.py:481`). Round 8 initially shipped
with no test files (an explicit interim user directive, "no testing, I'll
test later") — that directive was then reversed the same day ("do the tests,
make sure everything works well together"), and this gate state reflects the
follow-up pass: full pytest coverage written for every Round 8 module (138
new tests, listed under Round 8's own status note below), plus a real,
independent finding from actually *running* `tests/test_planted_effects.py`
for the first time — it had been written in the Round 7 session but never
executed. `ruff check .` and `mypy src/` are both clean on every file this
session touched (Round 8's own files, `controller.py`, `memory.py`,
`segment_comparison.py`, `workforce_analysis.py`); a small, pre-existing
type-hygiene debt remains elsewhere (12 mypy errors across 9 files nobody
touched this session — `statistical_analysis.py`, `report_generator.py`,
`text_analysis.py`, `geospatial.py`, `financial_analysis.py`,
`cohort_analysis.py`, `change_analysis.py`, `clustering.py`,
`dimensionality.py` — and ~41 pre-existing ruff findings, mostly a
consistent-but-nonstandard `UPPER_CASE` local-variable style in
`tests/fixtures/planted_effects.py` and unrelated issues in `ui/animations.py`)
not addressed here since fixing them was out of this pass's scope.

**2026-09-18.** Round 7 (7.1–7.22, the finding bus, semantic layer, insight
library, method-fit guards, analysis-mode decision, dashboard story layer,
layered reports, planted-effect harness, UI restructure) plus Q1, P2.2, P3.1,
P3.3, P3.4, R3.1, 6.1, 6.3, 6.4, P1.2, P1.6, P2.3, P2.4, P2.7, U1.2, U1.5 all
landed and are now commented out below — see `HANDOVER.md` §1–§2c for what was
verified and how. **Same day, later:** Round 8 was opened — LLM Sandbox mode
(dynamic tool creation/modification) and chart-format intelligence — per a
fresh `/goal` directive. See the Round 8 section immediately below for the
live plan; "Still open from Round 7" (just above "Carried-forward open items")
lists what Round 7 deliberately left for later (7.12–7.14, parts of
7.18–7.20/7.22) so none of it got lost in the comment-out pass.

**Reading order.** Round 10 (immediately below) is the live plan. Round 9 is
its baseline (all green, see its own status note). The carried-forward items
below both keep their original numbering (`6.x`, `U1.x`, `P1.x`…) so older
notes, commits and conversations still resolve.

---

---

# Round 10 — Close the orphan-wiring gap, harden performance, make cost measurable

**Directive (2026-09-23/24, via `/goal` then `/advisor`):** the user asked for
a full quality/correctness pass over everything merged since Round 9 —
FutureScope Phases 0–7 (`31e29f1`) plus Phases 9–12, which existed on an
unmerged branch (`daksh/updatesv2`, commit `d11c115`) and have now been
fast-forward merged into `master` — followed by an `/advisor`-guided plan for
further quality and speed. This section is that plan. Executed via 8 parallel
review forks (one per phase-cluster, disjoint files) plus the lead handling
`controller.py` (the contention point, per every prior round's own lesson).

## Baseline (verified this session, 2026-09-23/24)

- `ruff check .` / `mypy src/`: clean, 84 source files.
- `pytest tests/ -q`: 1339 passed (one, `test_survival.py::test_large_frame_is_fast`,
  is a 6s wall-clock budget assertion that flakes under concurrent system
  load — passed standalone in 2.14s both times it was checked; not a code
  regression, see D7 below).
- `scripts/validate.py` 68/68, `scripts/dry_run.py` 40/40.
- A **real, exploitable sandbox vulnerability** was found and closed:
  `src/core/duckdb_engine.py`'s SQL-safety check was a regex denylist with a
  word-boundary bug (`\bread_csv\b` doesn't match `read_csv_auto`) plus no
  engine-level lock, letting sandboxed/LLM-authored code read arbitrary host
  files (`SELECT * FROM read_csv_auto('C:/secret.txt')`, even a bare
  `SELECT * FROM 'C:/secret.csv'`). Fixed with DuckDB's own
  `enable_external_access=false` + `lock_configuration=true` on every
  connection — the actual authoritative boundary now, regex is defense in
  depth only. Regression tests added (`test_duckdb_regex_evasions_rejected`,
  `test_duckdb_engine_level_lock_holds_even_without_regex` — the latter
  bypasses the regex entirely and confirms the engine-level lock still
  holds).
- **Nine correctness bugs found and fixed** across the review forks (each
  with a planted + null regression test, `ruff`/`mypy` clean):
  `domain_packs.py`'s WHO/EU limit regexes never matched AirQualityUCI's own
  real column names (`CO(GT)` — the `(` breaks a `\b` boundary; 100% miss
  rate on the project's own reference dataset, contradicting Phase 5's exit
  gate); `graph_analysis.py` false-positived on ordinary business tables with
  `marketing_source`/`sales_target` columns; `causal_guard.py`'s
  `classify_study_design` misclassified an observational medical dataset as
  a randomized experiment purely because it had a column named `treatment`
  (silently disabling the causal-language guard on real data); a
  false-positive "missing deliverable" in `deliverable_contract.py` on
  generic phrasing like "explain what happened here"; a dead/backwards
  profile-informed forecast boost in `question_router.py`; dead code with
  five hardcoded hex colors left in `app.py` after the UI extraction (the
  project's Q1 work was supposed to have a single color source of truth);
  and three bugs of my own introduced while wiring `dependence.py` (below) —
  overwriting `final_result["findings"]` with the raw, unranked list
  (silently defeated the no-skill-driver suppression in
  `findings.rank_findings`), `check_missingness_mechanism` skipping `r=±1`
  (the single strongest possible MAR signal) as if it were a degenerate
  case, and a Simpson's-paradox finding headline that claimed "reverses
  within every stratum" when the underlying check fires on a majority, not
  unanimity.
- **Three real performance bugs found and fixed**, all in code that runs on
  every dataset load with no gate to skip it:
  - `src/core/io.py`'s `detect_and_exclude_subtotals` used `df.iterrows()` —
    profiled at **114 of 118 seconds** reading a 500k-row CSV. Vectorized
    with `.str.strip().str.lower()` + `.isin()`/`.str.endswith()`; confirmed
    correct against the io-fork's own "Allison Corp must not be excluded"
    test. **~10-13s → ~1.1s** on the same file.
  - `src/core/integrity.py`'s `discover_logical_constraints` ran a full
    `i != j` sweep over every numeric-column pair, each paying for a
    `dropna()` + diff + sum — unbounded on a wide table (300 columns ≈
    90,000 pairs). Fixed by checking the (free) name-hint regex first and
    only paying for the dataframe work on a hinted pair, or when the table
    is under a bounded column-count cap; also replaced the pandas-level
    per-pair overhead with numpy indexing on one pre-extracted matrix
    (same O(rows) per pair, far less overhead — profiled at 0.455 of 0.541s
    on a 15-column, 9.5k-row table before this half of the fix).
  - Same file's `check_missingness_mechanism` ran one
    `scipy.stats.pointbiserialr` Python call per (missing_col, other_col)
    pair with no multiple-testing correction (25.7% false "systematic bias"
    rate on genuinely MCAR 30-column data, empirically measured) — the io
    fork fixed the correction (Benjamini-Hochberg, → 1.3%) and vectorized
    the pairwise scan with `DataFrame.corrwith()`, capped at
    `_MAX_OTHER_COLS_FOR_MISSINGNESS` for wide tables.
- `scripts/bench.py --check` is **not independently confirmed green this
  session** — every re-run after the first (which showed exactly one
  regression, `airquality.total` at +36%/0.6s absolute, since fixed) got
  progressively *worse* across code paths nobody touched (`train_model`,
  `find_relations`, `profile_dataframe`…), which cannot be a code effect;
  Windows Defender (`MsMpEng.exe`, ~4.3GB resident) was confirmed actively
  churning, almost certainly scanning the ~13 skill packages installed
  mid-session. **10.0 below is re-running it clean.** The four fixes above
  are each independently confirmed via isolated, reproducible before/after
  timing on the actual function, immune to this noise — that evidence
  stands regardless of what a noisy full-suite run shows.

## Design decisions (resolved via `/advisor`)

- **D1 — The defect class, not the individual bugs, is the finding.** Every
  phase-0-12 correctness bug this session was either "code that's written
  and unit-tested but nothing in the pipeline calls it" (`dependence.py` —
  fixed this session; `sensitivity.py`'s two functions; `question_router`'s
  output; `relational_joiner.py`; `ReadReport.extra_tables`; the Phase 10
  interrupt signal that nothing raises; `variable_methods.py`'s discarded
  CLR transform) or a regex/threshold that never got run against the
  project's own real data. A per-item fix doesn't prevent the next one.
  10.1/10.2 below are the structural guards; everything else is the current
  backlog of specific instances.
- **D2 — Wiring an orphan into `controller.py` is Ask-First, not a default
  yes.** AGENTS.md flags `controller.py` and `MemorySystem` schema changes
  explicitly. The user pre-cleared exactly one instance this session
  (`dependence.py`) after being shown the gap. The rest (10.4-10.8) are
  listed as **Open questions**, not committed work — don't wire them without
  asking, even though the pattern and the fix are now well understood.
- **D3 — Consolidate the post-hoc audit passes, don't add a fifth one.**
  `controller.py`'s end-of-run block already has three independent
  "re-read the dataframe, run a check, reassign
  `final_result["findings"]`" audits (Deliverable Contract, Dependence &
  Confounding, Causal Claim Guard) — that duplication is exactly what
  produced the raw-findings bug above (one of the three blocks bypassed
  `ranked_findings()`). Any new audit (sensitivity, a consolidated
  leakage check) joins this pass, not a new one.
- **D4 — Make "~0 cost when absent" a bench stage, not a claim.**
  `scripts/bench.py` already reports per-tool and some per-stage timings
  (`stage.profile_dataframe`, `stage.find_relations`,
  `stage.build_dashboard`…) but the Phase 0-7/9-12 per-load systems
  (`evaluate_data_integrity`, `detect_domain_pack`, `classify_study_design`,
  `route_question`, `SharedAnalysisContext`, the dependence audit) aren't
  individually broken out — their cost is invisible, bundled into whichever
  tool happens to run near them. Every phase's Generality Rule claim is
  unverifiable until it has its own line in the bench table.
- **D5 — DuckDB for built-in tools is profile-gated, not a rollout.** Pandas
  groupby is already C-level; `dsa.query_sql` exists for LLM-authored code,
  not as a mandate to rewrite existing tools. Only take this up once D4's
  bench stages point at a specific hot spot DuckDB would actually help.
- **D6 — `relational_joiner.py` vs `joins.py`: one system, not two.**
  `joins.py` (Round 9) is already wired end-to-end (`controller.py`'s
  `_join_related_files`, the UI's join-review step). `relational_joiner.py`
  (Phase 11) is a second, unconnected foreign-key-discovery/star-schema
  implementation that duplicates it. The actual gap Phase 11 was meant to
  close — `ReadReport.extra_tables` (extra sheets in a multi-tab Excel
  upload) is silently dropped, never reaching `joins.py` either — should be
  closed by routing `extra_tables` through the *existing* wired system, not
  by wiring the second one. Deleting `relational_joiner.py` is Ask-First
  (AGENTS.md: deleting existing source files).
- **D7 — Bench noise from background load is a real operational risk, not
  just today's bad luck.** A session that installs anything (`pip install`,
  a skill package, even a large `git checkout`) right before running
  `bench --check` will see exactly what happened tonight. `10.0` covers
  today's re-run; a longer-term fix (a system-load sanity check bench.py
  runs before measuring, or a median-of-N mode) is left to the user's
  judgment in Open Questions, not implemented speculatively.

## Work items

### Batch A — offline, no LLM (do first)

- **10.0 Bench re-verification.** Re-run `scripts/bench.py --check` with no
  other heavy process running (confirm via `tasklist`/`wmic cpu get
  loadpercentage` first). *Exit:* 0 regressions, or any real one is
  root-caused the same way 10.0's predecessor items were (isolate the
  specific function, profile it, fix it, verify in isolation before
  trusting the full-suite number again).
- **10.1 AST reachability check.** Extend `tests/test_architecture.py`
  (same no-execution AST-walk pattern already there) with a check: every
  public function/class defined in `src/core/*.py` must be referenced
  somewhere outside its own file and outside `tests/`, or be in an explicit
  allowlist (for genuine standalone exports — types, constants meant for
  external re-import). This is the mechanical version of D1 — it would have
  caught `dependence.py`, `sensitivity.py`'s two functions, and
  `relational_joiner.py` on the day they were written, not one session
  later. *Exit:* the check runs clean against the current tree once the
  Open Questions below are resolved one way or the other (wired, or
  explicitly allowlisted with a one-line reason).
- **10.2 Consolidate the post-hoc audit passes (D3).** One ordered pass in
  `controller.py`: load the dataframe once, run
  dependence → (sensitivity, if 10.4 is approved) → deliverable contract →
  causal claim guard last (it rewrites `Finding` text in place, so it must
  see the final set), then project `final_result["findings"]` through
  `ranked_findings()` exactly once at the end. *Exit:* the raw-findings bug
  class is structurally impossible, not just fixed at today's three call
  sites.
- **10.3 Bench stages for D4.** Add explicit timed stages to
  `scripts/bench.py` for `evaluate_data_integrity`, `detect_domain_pack` +
  `evaluate_domain_pack`, `classify_study_design`, `route_question`,
  `SharedAnalysisContext` construction, and the dependence audit — mirroring
  the existing `stage.*` pattern. *Exit:* every Phase 0-7/9-12 per-load
  system has its own line in the bench table; `bench_baseline.json` updated
  once 10.0 is clean.

### Batch A — gated on Open Questions (each is Ask-First; see below)

- **10.4 Wire `sensitivity.py`.** `audit_finding_sensitivity` (jackknife
  fragility — verified correct, empirically matches an independent
  `scipy.stats.pearsonr` recomputation) as a post-hoc pass over the top
  ranked findings within 10.2's consolidated audit; `detect_target_leakage`
  early, alongside target detection — it catches semantic/naming leaks
  (`_POST_OUTCOME_PREFIXES`, a column literally derived from the target
  name) that the pre-existing `ml_pipeline.py::_detect_target_leakage`
  cannot (that one only sees already-encoded `X`/`y` at training time via a
  purity/group-determinism test) — the two are complementary, not
  duplicates; keep both.
- **10.5 Route `extra_tables` through `joins.py` (D6).** Read
  `read_report.extra_tables` in `controller.py` (currently captured, never
  consumed — confirmed empirically: a real 2-sheet Excel upload silently
  analyzes only the first sheet) and hand it to the already-wired
  `join_related`/`preview_joins` path instead of building a second join
  engine. Decide `relational_joiner.py`'s fate as part of the same
  question — delete it (its useful ideas, if any, fold into `joins.py`), or
  leave it deliberately unwired with a comment explaining why it exists
  (an AST-reachability allowlist entry either way, per 10.1).
- **10.6 Give the Phase 10 interrupt a trigger.** `ToolInterruptSignal` +
  the halt/hypothesis-tree/prompt-injection pipeline are fully wired and
  integration-tested (`tests/test_dynamic_interrupts.py` drives it through
  the real `tool_registry`) — but zero built-in tools ever raise it.
  `stats_utils.py` already computes `is_zero_inflated` (zero_prop > 0.25)
  and only uses it to pick a model *name*; have the tool that reads that
  recommendation raise the interrupt instead of silently switching models
  when the inflation is severe, matching FutureScope's own example.
- **10.7 Wire `question_router.py`'s output.** `route_question()` is called
  and stored (`self.question_routing` / `memory.set_context`) but nothing
  reads it back — confirmed distinct from `agenda.py` (agenda answers "what
  should we ask of this *data*", the router answers "what is the user's
  objective *asking for*" and recommends tools) and from
  `ground_objective_column` (grounds a column name, not a tool list) — not
  redundant, genuinely unwired. Let `recommended_tools` influence the
  deterministic fallback plan's step ordering.
- **10.8 Wire the discarded CLR transform.** `variable_methods.py` computes
  a correct compositional centered-log-ratio transform and then only reads
  the resulting column *names*, never the transformed values — no
  correlation anywhere in the codebase is compositional-aware, so Phase 3's
  own exit-gate claim ("compositional shares do not produce spurious
  negative Pearson correlations") is unverified in practice. Wire the CLR
  values into `statistical_analysis.py`'s correlation path when the
  compositional flag is set.

### Batch B — one live LLM run, after Batch A is green

- **10.9 Verify Round 9's queued LLM-robustness items actually landed.**
  Grepped this session: none of them did — no circuit breaker after
  repeated unusable replies, `max_retries=2` still set on the Anthropic
  client (SDK auto-retries a timeout instead of one controlled retry), no
  "Here's a thinking process" prose-prefix stripping. Every prior live-LLM
  run in this project's history (Round 9's Batch B, twice) produced no
  usable plan for exactly these reasons. Implement them before spending
  another live run on anything else.
- **10.10 One live run exercising 10.4-10.8** (whichever were approved),
  end to end, on a real dataset with a real objective — the same pattern
  every prior round used to validate wiring, not just unit tests of the
  pure functions.

### Batch C — features (gated on the user; not started speculatively)

- **10.11 DuckDB for built-in tools (D5).** Only after 10.3's bench stages
  point at a specific hot spot pandas' own C-level groupby doesn't already
  cover — no blind adoption.

## Verification

Batch A: 10.0-10.3 are script/pytest-checkable directly. 10.4-10.8 each ship
with a planted-case + null-case test the same way `dependence.py`'s wiring
did this session (a real `AgentController.analyze(use_llm=False)` run, not
just a call to the standalone function) — that distinction is *the* lesson
of this round. Batch B: judged by whether a real model's plan reaches the
newly-wired tools and produces findings whose numbers trace back to them.
Batch C: ships with before/after bench numbers on the specific hot spot that
justified it.

## Sequencing

10.0 → 10.1 ‖ 10.3 (independent) → **user decides Open Questions** →
10.2 (needs to know which audits it's consolidating) → 10.4 ‖ 10.5 ‖ 10.6 ‖
10.7 ‖ 10.8 (disjoint files, parallelizable) → 10.9 → 10.10 → (further user
decision) → 10.11.

## Open questions for the user

1. **10.4** — wire `sensitivity.py` (jackknife fragility + semantic target
   leakage)? Touches `controller.py` and (for the leakage check) the target
   auto-detection call site.
2. **10.5** — route `extra_tables` through `joins.py`, and delete
   `relational_joiner.py` or leave it explicitly unwired? Touches
   `controller.py`; deletion is Ask-First on its own.
3. **10.6** — raise `ToolInterruptSignal` on severe zero-inflation instead of
   silently switching models? Touches whichever tool consumes
   `stats_utils`'s model recommendation (not yet identified precisely —
   `regression.py` or `statistical_analysis.py`).
4. **10.7** — wire `question_router`'s `recommended_tools` into the
   deterministic plan? Touches `controller.py`'s plan-building path.
5. **10.8** — wire the CLR transform into `statistical_analysis.py`'s
   correlation logic?
6. **Priority** — Batch A structural/perf items (10.0-10.3, no controller.py
   risk) first as recommended, or go straight to whichever of 10.4-10.8 the
   user cares most about?

---

---

# Round 9 — Verify, harden, extend (PLANNED — nothing started)

**Baseline.** Commit `abd080a` (clean tree). Last measured gates, 2026-09-19/20: `ruff check .` and `mypy src/` clean; `pytest tests/` 629 passed, 1 deselected; deterministic run of `AirQualityUCI.csv --no-llm` exit 0 in 6 s, 8 charts all with data, 0 placeholder values left, `AH` 0.18–2.23. **Never run:** the seven tools registered late in Round 8 follow-up work (`survival_analysis`, `curve_fit_analysis`, `mixed_model_analysis`, `forecast_analysis`, `basket_analysis`, `price_elasticity_analysis`, `equity_analysis`), the chart-design LLM pass, the join path, and the project's own `scripts/validate.py` / `scripts/dry_run.py` gates.

**Why this round exists.** The last sessions added ~5,000 unexecuted lines. Every real defect found so far came from *running something*, not from reading: a read-only-array crash swallowed by a `try/except` (relation detection had never worked), a Vega `test` condition the sanitiser rejects (correlation heatmap could never render), a dotted field name that blanked a scatter, a decimal-comma regression. The plan is therefore ordered by "how much can be proven without spending LLM calls", with live runs batched at the end because of the session limit.

## Design decisions (resolved via `/advisor`)

- **D1 — Offline first.** Batch A needs no API key and no dataset beyond synthetic fixtures. Live-LLM work is one batched run (Batch B). Batch C (features) starts only after A and B are green *and* the user has answered the open questions below.
- **D2 — Run the project's own gates first.** `scripts/validate.py` (68 checks) and `scripts/dry_run.py` (40 checks) exercise registry, schemas and parameter resolution without an LLM. They have not been run since the new tools were registered.
- **D3 — RLM truncation is diagnosed from code before any re-run.** Do not scope it as "raise the budget". Three suspects, two changed this session: (a) `llm_client.py` NVIDIA path no longer falls back to `reasoning_content` and raises on empty content, so a reasoning model whose chain-of-thought lands in `content` now behaves differently; (b) "repaired but truncated" now raises instead of returning a half-formed plan (correct, but turns "bad plan runs" into "sub-task lost"); (c) the sub-task prompt goes through `_fit` while `_metadata()` is still unbudgeted. Fix shape: retry once with a larger budget plus a JSON-only reminder, and a per-sub-task partial-failure path so one failed group never drops its siblings.
- **D4 — Do not trust the "pre-existing failure" claim on the planted-effects fixture.** A subagent widened `tests/fixtures/planted_effects.py` (2 → 4 years, n 4,000 → 16,000, region per customer) to clear a Benjamini-Hochberg gate, asserting the test "fails identically at HEAD". Unverified. If wrong, the harness built to catch regressions was widened to hide one caused by the sentinel/profiler/privacy changes. **RESOLVED 2026-09-20 (9.2, done by the lead):** at a clean `bf32b32` checkout the transactional test fails too (West `segment_lift` observed -0.5475 vs planted +0.25), and today's code run against the ORIGINAL fixture fails with the identical number, so the code did not regress. Cause: the original fixture drew `region` per order while the tools measure per customer, so a customer's orders spread across regions and the planted West premium was diluted and flipped sign. Assigning region per customer (plus 4 years / n=16,000 so month findings survive BH correction) is a legitimate fixture fix. 629-green result stands on this point.
- **D5 — Charts get an automated correctness guard.** All chart defects this session were "spec looks valid, renders nothing or breaks". Add a spec lint (JSON-schema validity of the emitted Vega-Lite + `chart_has_data` + field-name safety) as a test over every chart builder and every `dsa.chart.*` helper, using no new runtime dependency. An optional headless render check may sit behind the `slow` marker if `vl-convert-python` is acceptable as a dev-only dependency.
- **D6 — Ideas are ranked by value/cost and gated on the user (see Open questions).** Nothing in Batch C is built speculatively.

## Repro record (keep; `/tmp` will not survive)

Live run, 2026-09-20: `main.py --dataset AirQualityUCI.csv --output-dir output/aq_llm --max-iterations 6 --objective "Which pollutants move together, show a heatmap of monthly average per pollutant, and what drives CO levels?"`. Killed by the user mid-run; partial output in `output/aq_llm/` (keep until 9.3 is done). Failure lines:

```
⚠ sub-task 'numerical_group_2' failed: LLM reply was truncated at
max_tokens=4096 (finish_reason=length) and could not be repaired: LLM returned
non-JSON: Here's a thinking process:
⚠ sub-task 'categorical_group' failed: (same message, same prefix)
✓ RLM decomposition complete: 1 sub-task(s) resolved.
```
Two of three sub-tasks lost; the reply began with prose ("Here's a thinking process: 1. **Analyze User Input:** …") instead of JSON.

## Work items

### Batch A — offline, no LLM (do first)
- **9.0 Gates.** Rerun `ruff check .`, `mypy src/`, `pytest tests/ -q`, then `scripts/validate.py` and `scripts/dry_run.py`. Fix registry/schema/param breakage from the seven new tools. *Exit:* all five green.
- **9.1 Smoke tests for the seven new tools.** One synthetic fixture per tool with a planted, checkable answer (e.g. exponential decay with known half-life; two survival groups with known hazard ratio; grouped data with known ICC; seasonal series with a known period; planted co-purchase pair; log-log data with known elasticity; planted pay gap) **plus a negative test per tool that `applies_to` returns 0.0 on unrelated data** — that gate is what keeps them out of the planner prompt, so a misfire regresses prompt size and tool choice everywhere. Also assert each tool's wall time on 50k rows (< 3 s) and that it is safe to run concurrently. *Exit:* new tests in the suite, planted values recovered within tolerance.
- **9.2 Verify the planted-effects change.** Check out `bf32b32` in a separate worktree, run `tests/test_planted_effects.py` with the *original* fixture. If it passes there, the failure is a regression from this session (suspects: sentinel nulling, `has_identifier_name_hint`/`is_identifier_like`, small-cell folding in segment_comparison/workforce, coercion changes) — fix production code and revert the fixture. If it fails there too, keep the fixture change and record why in the fixture's docstring.
- **9.3 RLM truncation.** Implement D3 with a scripted-LLM unit test that reproduces prose-then-truncate and asserts: one retry with a larger budget, siblings survive, run still completes. Add a per-call `max_tokens` for sub-tasks and a JSON-only instruction. Budget `_metadata()` through `_fit`. *Exit:* the reproduction test fails before the fix and passes after.
- **9.4 Chart lint (D5).** Test that walks every builder in `dashboard.py`, `visualization.py`, `chart_designer.py` and every `dsa.chart.*` helper with dotted/bracketed/unicode column names and asserts: schema-valid, no unsafe field names, `chart_has_data` true, no expression referencing a raw column. *Exit:* lint runs in `pytest`.
- **9.5 Small leftovers.** NVIDIA path token-usage capture; replace the private `_rlm_engine._record_usage` coupling in `controller._design_charts` with a public method; `distributions` skipping balanced 0/1 columns (draw them as bars); survival agenda question missing its group column; findings *headlines* naming a level with n < `DSA_MIN_CELL_SIZE` (suppression currently covers tables and charts only — audit text paths).

### Batch B — one live run (LLM), after A is green
- **9.6 End-to-end on `AirQualityUCI.csv`.** Same objective as the repro. Record per stage: wall time, tokens (from `llm_usage`), LLM call count. Check: design pass adds/drops (log them), month × pollutant heatmap renders and uses the wide → long path, `Drew …` critique lines appear in tool summaries and the LLM reacts, coverage report is not all-unanswered, RESULT-only dynamic runs show the "No FINDING declared" hint and the auto-chart fallback fires. Save the run's `dashboard.json` and screenshot as the new reference. *Exit:* a written observation list; any defect becomes a Batch A-style item.

### Batch C — features (each needs a yes from the user; see Open questions)
- **9.7 LLM-assisted column-role mapping.** The planner's `data_understanding` proposes roles (`duration`, `event`, `price`, `quantity`, `order_id`, `item`, `group`, `outcome`, `protected_attribute`, `x_dose`, …) for unusual or non-English names; a deterministic validator checks each against the data (dtype, uniqueness, value set) and only validated roles are stored as profile overrides read by every `applies_to`/`default_params` through one helper. No LLM ⇒ today's token matching unchanged. *Risk:* a wrong role silently mis-routes a tool — hence validate-then-store and a visible "roles inferred" note in the report.
- **9.8 Vocabulary consolidation.** Each tool currently carries its own token tuples. Move them into one `src/core/vocab.py` (English + es/fr/de/pt synonyms), used by profiler and all `applies_to`. Pure refactor + coverage tests; do this *before* 9.7 so role mapping has one place to write to.
- **9.9 Join review.** Auto-join uses name + value-overlap heuristics (≥ 60% containment, refuses many-to-many). Add a review step in the UI showing the proposed keys, cardinality and match rate with accept/reject/change-key; default accept so headless runs behave as today. Add a hard test that a shared bare `id` between unrelated tables is not silently joined.
- **9.10 Performance budget + CI.** `scripts/bench.py`: per-stage timings for the deterministic run on (a) AirQualityUCI, (b) a 500k-row synthetic, (c) a 300-column synthetic; record baselines; fail on > 25% regression. CI job running ruff, mypy, `pytest -m "not slow"`, `validate.py`, `dry_run.py`, bench. Time-box: relations scan, sentinel scan, `applies_to` across all tools, design-call latency.
- **9.11 UI: suppression setting and run comparison.** Sidebar control for `DSA_MIN_CELL_SIZE` with a visible "small groups combined" indicator; a two-run comparison view (findings added/removed, charts changed) fed by a `summary.json` per run.
- **9.12 Docs.** `AGENTS.md`/README/HANDOVER: the seven tools and their triggers; env vars `CHART_DESIGN`, `DSA_MIN_CELL_SIZE`, `DSA_MAX_ROWS`; the 60-chart safety ceiling; optional installs (`pyreadstat`, `pytables`, `xarray`); join behaviour.

## Verification
Batch A: automated (all of 9.0–9.5 are pytest/script-checkable). Batch B: the observation list plus saved artifacts. Batch C: each item ships with tests and, for 9.7/9.9, a negative test proving the failure mode it introduces is guarded. No item is "done" on a static gate alone — the lesson of this round.

## Sequencing
9.0 → 9.1 ‖ 9.2 ‖ 9.3 ‖ 9.4 (independent files, parallelisable with disjoint ownership) → 9.5 → **9.6** → (user decisions) → 9.8 → 9.7 → 9.9 → 9.10 → 9.11 → 9.12. Agents: 9.1 splits by tool (three agents, disjoint test files); 9.3 and 9.4 are one agent each; 9.0, 9.2 and 9.6 are done by the lead because they need judgement on the results.

## Open questions for the user
1. **9.7 role mapping:** yes to LLM-proposed roles (with validation), or keep matching deterministic and only widen the vocabulary (9.8)?
2. **9.9 join review:** interactive confirmation in the UI, or keep auto-join with disclosure?
3. **Which domain matters most** for real data (environment, sales, HR, clinical)? It decides which of the seven tools get real-dataset validation first.
4. **Chart ceiling:** keep the 60-chart safety ceiling, or remove it entirely?

### Batch A results (2026-09-20) and 9.13 (added)
- **9.0 gates — done.** ruff, mypy (68 files), pytest 629 at the time, `scripts/dry_run.py` 40/40, `scripts/validate.py` 68/68 after fixing one stale check (it expected image files from the visualization tool, which now returns Vega-Lite specs).
- **9.1 smoke tests — done for all 7 tools (7 new test files).** New tests: `test_survival`, `test_curve_fit` (15), `test_mixed_model`, `test_forecast` (11), `test_basket`, `test_elasticity`, `test_equity` (14). Real bugs fixed in `mixed_model.py` (L-BFGS stall at the zero-variance boundary fell back to OLS; ICC computed on raw y diluted by the fixed effect — 0.53 vs true 0.9; a per-row-unique group column accepted; `applies_to` fired on any entity column). `curve_fit_analysis.applies_to` also fixed (lead): generic axis names (`age`, `time`, `day`, `hour`, `distance`) no longer offer the tool on business tables — they count only for sensor/experiment archetypes or narrow all-numeric tables; verified HR/churn -> 0.0, dose-response -> 0.4.
- **9.2 planted-effects — resolved** (see D4).
- **9.3 RLM truncation — done.** Real causes: 4096-token budget with no retry + a model writing prose in `content`; the removed `reasoning_content` fallback and the fail-on-truncation only exposed it. Now: one sequential retry at 2x max_tokens (cap 8192), low effort, JSON-only reminder; siblings unaffected; `record_usage()` public; `_metadata()` budgeted. 5 tests in `test_rlm_truncation.py`.
- **9.4 chart lint — done.** `tests/test_chart_lint.py`, 341 cases over ~67 producers x 10 adversarial names. One defect fixed (raw `vega_lite` sanitiser rejected columns with a backslash). `distributions` now draws 2-value/uniform columns as bars.
- **9.5 leftovers:** done except *headlines naming a level with n < `DSA_MIN_CELL_SIZE`* (still open).

### 9.13 LLM robustness for lightweight/local models — code landed 2026-09-20, NOT YET TESTED (user: "test later")
Risks identified and fixed: fixed prompt overflowing 8k windows (system ~2.1k tok + 13-tool block ~3.6k tok before any data); Ollama's real server context (2-8k) vs the table's native window (128k); cloud-speed timeout/max_tokens on CPU inference; parallel RLM sub-tasks queueing on one local model; `<think>` text leaking into JSON; plan JSON drifting from the contract; string-typed params; a 400 "context length exceeded" being retried as a rejected JSON mode.
- `llm_client.py`: `<think>/<thinking>/<reasoning>` stripping (incl. unterminated), BOM, smart-quote/trailing-comma repair only after strict parse fails, first object with `status`/`steps` wins; context clamp (`max_tokens <= window - prompt - 256`) and `LLMContextError` (not retried); local defaults (window = `LOCAL_LLM_CONTEXT` else min(table, 8192), timeout 600 s, max_tokens 2048, `max_retries` 0); `BoundedSemaphore` (`LLM_MAX_CONCURRENCY`, 1 local / 4 cloud) around the network call; per-stage max_tokens (blank `LLM_MAX_TOKENS` in `.env.example` enables it; reasoning models keep 4096); bare top-level list of steps wrapped as `{"steps": ...}`.
- `prompt_manager.py`/`tool_registry.py`: compact tier below a 12k budget (or `PROMPT_COMPACT=1/0`): system ~830 tok, tool block <= ~700 tok (top 10 + footer; `clean_data`, `execute_dynamic_code` always kept); controller now passes `compact_tool_descriptions`.
- `controller.py`/`step_validation.py`: tolerant plan parsing (aliases for tool/params/step keys, nested/list/dict-keyed plans, fuzzy tool names, reply-status repair), parameter type coercion from the tool schema, 8-step cap per cycle (profile-built plans exempt), duplicate-step drop, chart-design pass skipped below a 16k window, a cycle-1 "complete" with no tool run now falls back to the deterministic plan.
- Sandbox: hints for wrong `dsa.*` names/kwargs (`did you mean`), unavailable `plt/sns/sklearn`; chart helpers accept `hue/group/by/xlabel/ylabel/kind/values` aliases and drop unknown options with an "ignored: ..." note.
**Test checklist:** (1) 8k-context Ollama run on AirQualityUCI (`LOCAL_LLM_CONTEXT=8192`), compact prompt engaged, no context error; (2) a reply wrapped in `<think>...</think>` and one with an unterminated `<think>`; (3) plan drift examples: `{"tool":"clean_dat","params":"{\"file_path\":\"x\"}"}`, `{"plan":{"1":{...}}}`, a top-level list, `"status":"IN PROGRESS"`, `columns:"a, b"`, `n_clusters:"3"`; (4) a cycle-1 `"complete"` with fake `key_metrics`; (5) sandbox `dsa.chart.barplot(...)`, `hue=`, `plt.plot`; (6) `LLM_MAX_CONCURRENCY=1` with RLM decomposition; (7) full pytest (the suites edited this round have not been re-run together) then `validate.py`/`dry_run.py` again.

### Wave 1-2 results (2026-09-20/21)
Decisions taken (user said "perform the rounds", the four open questions stayed unanswered, recommended defaults used): LLM-proposed roles WITH deterministic validation; auto-join with a default-accept review step; synthetic data for every domain; 60-chart safety ceiling kept.
- **9.14 analysis quality — landed, verified on `AirQualityUCI.csv` (deterministic run).** Hour-of-day and day-of-week profiles for sub-daily data (CO(GT) peaks 19:00 at 3.73, trough 05:00 at 0.71, 5.2x; weekends -30%; only measures with >= 50% completeness; the tool's primary measure carries the exec finding). Seasonality-aware change findings (AH +51% Feb->Mar 2005 was -1% vs March 2004 -> rewritten "a normal seasonal pattern", appendix layer, confidence 0.6 -> 0.35). Seasonality-aware TREND findings (under two seasonal cycles a visibly seasonal series is headlined "may be the season, not a trend", appendix, confidence halved). Correlated columns collapse into ONE group finding (4 columns, r 0.88-0.98). `coverage_gap` finding for columns > 50% missing (NMHC(GT) 90%). Cluster wording per Kaufman & Rousseeuw (weak/none), PCA finding in the appendix, `format_p` (`<0.001`, never `0.0`). Stable objective-aware `pick_measures`; `time_series` no longer chooses its default measure by raw variance (unit-dependent); "what drives CO levels" grounds to `CO(GT)` via `ground_objective_column` and driver phrasing counts as prediction intent. Minimal RLM sub-task prompts (~7k -> ~0.7k tokens typical) and `RLM_DECOMPOSE=auto|on|off`. Small-group suppression extended to finding text (`Finding.__post_init__`, `redact_small_level`). Known leftover: `cohort_analysis` RFM finding names a segment without an n check.
- **9.8 vocabulary — landed.** `src/core/vocab.py`: one multilingual role vocabulary (en/es/fr/de/pt), whole-token only, accent-insensitive; profiler and the seven specialist tools refactored; ambiguous words deliberately omitted (`data`, `tempo`, `dia`, `tag`, `genre`, `prime`, `lot`, `charge`, `probe`, `stuck`, `handicap`).
- **9.7 validated LLM roles — landed, controller wiring untested.** `roles.validate_roles` (16 roles, data checks, <= 12 proposals) -> `DatasetProfile.role_overrides` -> `vocab.column_role`/`roles_for` used by the specialist tools; planner `data_understanding.roles` (optional); tool blocks refreshed; report line "Column roles inferred (validated against the data)". No LLM run has exercised `_apply_column_roles` yet.
- **9.9 join review / 9.11 UI — landed, UI untested visually.** `joins.preview_joins`, `join_related(..., overrides)`, bare `id/index/code/key` keys need >= 90% containment, `load_dataset(..., join_overrides)`; sidebar group-size input (`DSA_MIN_CELL_SIZE`); `summary.json` per run + `run_compare` + "Compare with a previous run" expander. Streamlit rendering has NOT been looked at.
- **9.10 benchmark + CI — landed.** `scripts/bench.py`, `scripts/bench_baseline.json`, `.github/workflows/ci.yml` (overwrote an older Ubuntu-only 3.11/3.12 workflow; now windows+ubuntu, Python 3.13; bench step non-blocking).
- **9.15 speed on large data — landed.** Seconds before -> after: big500k 127.7 -> 49.5 (train 44 -> 22, outliers/correlation/time-series ~22 -> ~7), wide300 22.3 -> 15.5, mixed60 25.4 -> 6.2, AirQualityUCI 1.73 -> 1.32. Causes: every tool re-ran `coerce_types` on the whole file (now cached per file), per-pair `.loc` lookups (now numpy), string coercion once per distinct value. Exploratory statistics on > `DSA_ANALYSIS_SAMPLE_ROWS` (200,000) rows use a seeded sample with `sampled_from/sampled_to` + caveat; p-values use every row. The `train_model` 100k/200k refit was deliberately not added (existing 50k cap already bounds it).
- **Null-data + rubric tests — landed.** `tests/test_null_data.py`, `tests/test_quality_rubric.py`. Found and fixed: a no-skill model still emitted permutation-importance "driver" findings (importance 0.61) — now dropped in `rank_findings` when the model's lift < 0.05.
- **Not yet done:** Batch B live LLM run (needs a completed run to judge narrative quality; last attempt was killed), 9.12 docs (AGENTS.md/README/HANDOVER), the RFM small-group leftover, and a visual check of the new Streamlit UI pieces.

### Batch B live run 2 (2026-09-20 16:01-16:33, stopped by the user) — result: the LLM path produced nothing usable
`main.py --dataset AirQualityUCI.csv --output-dir output/aq_llm2 --max-iterations 3 --objective "Which pollutants move together, show a heatmap of monthly average per pollutant, and what drives CO levels?"`, model `nvidia/nemotron-3.5-lightning:free` via OpenRouter, `.env` has `LLM_MAX_TOKENS=4096`. Partial output kept in `output/aq_llm2/`.
- **Worked:** objective grounding — log line `Auto-detected 'CO(GT)' as the target (confidence 80%)`, task type regression (run 1 had candidate `AH` at 25% and "describe"). The deterministic fallback plan then ran `regression_analysis`, `train_model`, `evaluate_model` for CO. Cost-aware RLM decomposition skipped itself ("findings already cover all 2 numeric column groups"). The truncation retry path fired as designed.
- **Failed:** every LLM reply was chain-of-thought prose ("Let me first understand ...") that consumed the whole 4,096-token budget before any JSON, so both plan attempts were unusable and the run fell back to the deterministic plan; iteration 2 returned `empty content (finish_reason='error')` after 11 min.
- **Timing (audit log):** calls finished 16:05:33 (~4 min), 16:14:53 (+561 s), 16:25:54 (+660 s). 4,096 tokens in 561 s = ~7 tok/s (run 1 the same model did ~68 tok/s, ~60 s per call): free-tier congestion. Tools total ~25 s.
- **Causes on our side:** (1) no circuit breaker: after 2 unusable replies the run keeps retrying every iteration (~10 min each) although the deterministic fallback already gives the same result; (2) client `timeout` 120 s with `max_retries=2` on cloud providers means one slow call can take 6-10 min (SDK retries timeouts); (3) `.env` `LLM_MAX_TOKENS=4096` disables the per-stage token defaults added in 9.13; (4) `<think>` stripping does not catch the "Here's a thinking process" prose style.
- **Not verified because no usable plan ever arrived:** chart-design pass, the month-by-pollutant heatmap request, dynamic-code findings reaching the report, the coverage report on an LLM run, `_apply_column_roles`.
- **Fixes queued (not done):** run-level LLM circuit breaker (2 consecutive unusable replies or a call over a wall-clock limit -> deterministic for the rest of the run, with a one-line note); no SDK retry on timeouts and a per-call wall-clock cap; startup JSON probe that marks a chatty model and switches to a compact plan schema or structured-output mode; document removing `LLM_MAX_TOKENS` from `.env`; recognise "thinking process" prose prefixes when stripping reasoning text.

**Status: Batch A and Batch C landed; Batch B run twice with no usable LLM result (see above); verification pass green (1229 tests, validate 68/68, dry_run 40/40, ruff and mypy clean). Updated 2026-09-21.**

---

# Round 8 — LLM Sandbox Mode: dynamic tool creation & modification, chart intelligence

**Directive (2026-09-18, via `/goal`):** implement the LLM Sandbox mode completely
— the agent must be able to *create new tools and modify existing ones at
runtime* for better analysis/interpretation, plus make charts that a normal
human (not just an analyst) can look at and understand. **No testing** (user's
explicit instruction, overriding AGENTS.md's "one unit test per tool" rule for
this round only — verification is by direct script invocation and reading the
code, the same pattern HANDOVER.md's Round 7 session used for its background-
agent output, not by writing pytest files). Executed via parallel Claude Code
subagents, one per file-owning boundary, briefed against the fixed interface
contracts below so they don't need to coordinate live. **Ask First flags below
are noted for the record, not re-litigated** — the `/goal` directive is the
pre-clearance, same posture Round 7's constraints took for its own scope.

**What already exists (sub-project 1 of the "Universal Dynamic Analyst" plan,
`docs/superpowers/specs/2026-09-11-isolated-compute-sandbox-design.md`):**
`src/core/sandbox.py` + `src/core/_sandbox_worker.py` run one-shot LLM-authored
Python in an isolated subprocess (import allowlist, restricted builtins, no
network/file access, timeout + soft memory cap), and `src/tools/dynamic_code.py`
wraps it as `execute_dynamic_code`, a `BaseTool` the planner can already select.
That sub-project explicitly scoped out 2 (self-correction retry loop), 3
(question→code translation) and 4 (formatting results into reports/charts) as
separate work. **Round 8 supersedes all three**: instead of one-shot code, the
agent can register a *named, described, re-callable tool* that persists for
the rest of the run (and optionally across runs on the same dataset), and can
*revise* one that errored — which is what turns sub-projects 2–4 from
"someday" into "now," because a tool that failed once can be fixed and called
again by name instead of being a dead end.

**Why not just let `execute_dynamic_code` do this already:** it is
intentionally single-shot and un-named — every call is independent, so there
is nothing to reuse, nothing to schedule twice, and nothing a later iteration
can refer back to. It stays exactly as-is; Round 8 is additive.

## Design decisions (resolved via `/advisor` before writing this plan)

1. **The system prompt stays static and cached.** `PromptManager.get_system_prompt()`
   is built once per `analyze()` call and handed to `RLMEngine` at construction
   (`controller.py:1066-1071`); `LLMClient._dispatch`'s Anthropic branch marks
   it `cache_control: ephemeral` (P1.6(b)). A tool created in iteration 3 must
   still be plainly *callable* in iteration 4 — the answer is **not** to rebuild
   the system prompt (which would kill the cache and re-bill the prefix every
   cycle), but to list newly-created tools in `get_iteration_user_prompt()`,
   which is already per-iteration and uncached. The system prompt gains exactly
   one new *static* paragraph: what `define_analysis_tool` is, the `df`/
   `SCHEMA`/`RESULT`/`FINDING` contract, and the allowed-module list — all of
   which is already true for every dataset, so it costs nothing to cache.
2. **Parameters reach generated-tool code as data, never as templated source.**
   `run_sandboxed` gains an `extra_globals: dict[str, Any] | None` parameter,
   threaded through the existing `input.json` handoff into
   `_sandbox_worker._build_restricted_globals`, JSON-validated before the
   subprocess spawns. No string concatenation into the code body, ever — a
   parameter value with a newline or a quote in it must stay a value.
3. **Tool registration is controller's job, not the tool's own `execute()`.**
   AGENTS.md's layer rule ("tools/* must never drive memory/engine/controller
   behaviour") already implies this, and it has a precedent: `clean_data`
   doesn't call `MemorySystem.set_context("cleaned_file_path", ...)` itself —
   the controller does, after the step succeeds (see the "Always Do" row in
   AGENTS.md). `DefineAnalysisToolTool.execute()` only *validates and smoke-
   tests* a proposed tool (pure, no side effects on the registry); the
   controller, which already owns `tool_registry` and `memory`, performs the
   actual `register()` + persistence once the step reports success.
4. **Collision and origin guards.** `ToolRegistry.register()` (`controller.py:589`)
   currently overwrites silently — `self._registry[tool.name] = tool` with no
   check. A generated tool named `clean_data` would replace the real cleaner
   for the rest of the run. New rule: a name already held by a *built-in* tool
   is always rejected; a name already held by a *previously generated* tool is
   allowed to overwrite **only** when the new spec is itself LLM-generated
   (i.e. this is treated as "modification," versioned, old code kept in
   history) — this is the mechanism that makes self-correction possible.
5. **Generated tools are invisible to the deterministic (`--no-llm`) planner.**
   Same pattern `DynamicCodeExecutionTool` already uses: `requires_llm = True`.
   `ToolRegistry.candidate_tools()` already excludes `requires_llm` tools when
   `use_llm=False` (`controller.py:634-635`), so no change to the fallback
   sweep is needed and the R3.1 regression (a tool re-added itself to the
   deterministic plan through the generic `applies_to` sweep) cannot recur
   here by construction.
6. **The static pre-check's RESULT rule must relax, or self-correction burns
   its budget on false failures.** `_static_check` (`sandbox.py:111-115`)
   requires a **top-level** `ast.Assign` to `RESULT`. One-shot code from a
   single prompt rarely branches; a reusable, parameterized tool body
   plausibly does (`if grain == "month": RESULT = ... else: RESULT = ...`).
   Fix: walk the whole tree for *any* assignment target named `RESULT`
   (`ast.walk` instead of `tree.body`), accepting `AnnAssign` too. The runtime
   check (`RESULT_VAR_NAME not in restricted_globals` in `_sandbox_worker.py`)
   remains the authoritative gate either way, so this only removes a source of
   spurious rejections, not the safety net.
7. **`FINDING` numbers must also land in the tool's own `output` dict.**
   `_flag_unverified_claims` (`controller.py`) checks narrative numbers
   against tool outputs — HANDOVER §1 documents a false-positive here from
   rounding mismatch, already fixed with multi-precision canonicalisation.
   A generated tool's `FINDING` evidence numbers need to appear verbatim in
   `output` too, or every AI-generated insight gets stamped "unverified" on
   sight. `GeneratedTool.execute()` copies `finding.evidence` values into its
   returned `output` dict for this reason — not a workaround, a requirement.
8. **Persistence is keyed to the dataset, never auto-reloaded across
   datasets.** A tool generated for file A references A's columns; loading it
   automatically against file B is a guaranteed `KeyError`. Persist to
   `<output_root>/generated_tools/<name>.json` per run (audit + reproducibility)
   with the dataset fingerprint (`src/core/io.py`'s existing path+mtime+size
   cache key) recorded in the spec; reloading across sessions is opt-in and
   fingerprint-checked, not automatic.
9. **A hard cap on generated tools per run** (`MAX_GENERATED_TOOLS = 6`)
   bounds tool sprawl the same way `MAX_ITERATIONS` bounds reasoning cycles.

## Interface contracts (fixed so agents can build against them without syncing)

```python
# src/core/sandbox.py — extended, not replaced
def run_sandboxed(
    code: str,
    dataset_ref: str,
    extra_globals: dict[str, Any] | None = None,   # NEW — JSON-primitive values only
    timeout_s: float = 20.0,
    memory_limit_mb: int = 512,
) -> SandboxResult: ...

@dataclass
class SandboxResult:
    status: Literal["ok", "error"]
    result: Any | None
    finding: dict[str, Any] | None   # NEW — from an optional top-level `FINDING = {...}`,
                                      #        same JSON-conversion rules as RESULT, never required
    stdout: str
    error_type: str | None
    traceback: str | None
    hint: str | None
    duration_ms: float
```

```python
# src/core/tool_factory.py — NEW
@dataclass
class GeneratedToolSpec:
    name: str                       # snake_case, validated, not a built-in tool name
    description: str
    params_schema: dict[str, Any]   # exact BaseTool.get_schema() shape
    code: str
    version: int
    created_at: str                 # ISO timestamp
    dataset_fingerprint: str        # reuse src.core.io's path+mtime+size cache key
    source: str = "llm_generated"

def validate_spec(
    name: str, description: str, params_schema: dict[str, Any], code: str,
    existing_tool_names: list[str], existing_generated: dict[str, GeneratedToolSpec],
) -> list[str]:
    """Returns a list of human-readable error strings; [] means valid.
    Rejects: bad snake_case, collision with a *built-in* name, malformed
    schema, code failing the relaxed static check. Allows overwrite of a name
    already in `existing_generated` (that's a modification, bumps version)."""

def register_and_persist(
    spec: GeneratedToolSpec, tool_registry: Any, memory: Any, output_root: str,
) -> None:
    """Builds a GeneratedTool from spec, tool_registry.register()s it,
    appends/updates it on memory's generated-tools list, writes
    <output_root>/generated_tools/<name>.json (with prior versions under a
    `history` key, not overwritten)."""

def load_persisted_tools(output_root: str, dataset_fingerprint: str) -> list[GeneratedToolSpec]:
    """Opt-in reload for a matching dataset fingerprint only — never auto-runs
    across datasets."""
```

```python
# src/tools/generated_tool.py — NEW
class GeneratedTool(BaseTool):
    requires_llm = True   # invisible to the --no-llm deterministic planner (decision 5)
    def __init__(self, spec: GeneratedToolSpec) -> None: ...
    def get_schema(self) -> dict[str, Any]: return self.spec.params_schema
    def execute(self, file_path: str, **kwargs: Any) -> dict[str, Any]:
        """extra_globals = kwargs (minus file_path), JSON-validated; delegates
        to run_sandboxed(self.spec.code, file_path, extra_globals=...). Copies
        sandbox_result.finding's evidence numbers into the returned output
        dict (decision 7) before returning."""
    def findings(self, output, profile, metadata) -> list[Finding]:
        """Builds one Finding from output["finding_payload"] if present;
        tags caveats=["AI-generated analysis tool"] for the trust surface
        (T8); never raises — malformed payload means no finding, not a crash."""

# src/tools/define_analysis_tool.py — NEW
class DefineAnalysisToolTool(BaseTool):
    requires_llm = True
    name = "define_analysis_tool"
    def execute(self, tool_name, description, params_schema, code,
                example_params=None, file_path=None, **_) -> dict[str, Any]:
        """PURE — no registry/memory mutation (decision 3). Relaxed-static-
        checks `code`, optionally smoke-tests it once via run_sandboxed with
        example_params against file_path, returns {"summary", "status":
        "ready"|"error", "spec": {...}, "hint": ...}. Controller reads
        output["status"]=="ready" to decide whether to call
        tool_factory.register_and_persist()."""
```

## Work items

| # | Item | Owner | Files | Flags |
| :-- | :--- | :--- | :--- | :--- |
| **8.1** | `extra_globals` threading + relaxed static check + optional `FINDING` | Agent A | `src/core/sandbox.py`, `src/core/_sandbox_worker.py` | additive, no existing behaviour changes for `execute_dynamic_code` |
| **8.2** | Tool factory + `GeneratedTool` + meta-tool | Agent B | `src/core/tool_factory.py` (NEW), `src/tools/generated_tool.py` (NEW), `src/tools/define_analysis_tool.py` (NEW) | **Ask First** (new tool contract) — pre-cleared by `/goal` |
| **8.3** | Controller/memory/prompt wiring: register meta-tool, post-step registration hook, `generated_tools` on `MemorySystem`, iteration-prompt block, one static system-prompt paragraph | **Main session** (highest blast radius — `controller.py` is the contention point per every prior round) | `src/core/controller.py`, `src/core/memory.py`, `src/core/prompt_manager.py` | **Ask First** (`MemorySystem` schema) — pre-cleared |
| **8.4** | Chart intelligence: wire `unit_hint` → Vega `axis`/tooltip `format` + human axis titles, sort categorical bars by value, one-line plain-language captions on un-bound EDA filler panels | Agent C | `src/core/dashboard.py`, `src/core/chart_theme.py` | Load the `dataviz` skill first. **Do not** re-touch panel selection/suppression/cross-filtering/theme tokens/tab structure — 7.8/7.15/7.17/7.18 already landed those (HANDOVER §2c); re-auditing them is out of scope and wastes the round |

### 8.4 detail — the actual gap, not a dashboard rewrite

`unit_hint` (`profiler.py`'s semantic layer) is read today only for measure
*selection* (`dashboard.py:307`) and sum-vs-mean aggregation choice
(`dashboard.py:600`) — never for how a value is *displayed*. A revenue chart's
axis shows `269.3`, not `$269`; a churn-rate chart shows `0.17`, not `17%`.
That is the concrete "make it make sense to a normal human" gap, and it's
narrow and mechanical:

- `chart_theme.py` gains `axis_format(unit_hint: str | None) -> dict[str, str]`
  returning e.g. `{"format": "$,.0f"}` for `"currency"`, `{"format": ".0%"}`
  for `"percent"`, `{"format": ",d"}` for `"count"`, `{}` otherwise — theme-
  adjacent, not per-chart hardcoded, matching how `vega_config()` is already
  the one place colour lives.
- Every `ChartSpec` builder in `dashboard.py` that knows a column's
  `unit_hint` (histogram, time-series, category/bar, box plot) applies it to
  the relevant `encoding.<channel>.axis` and tooltip format, and gives the
  axis a human title (`"Amount ($)"` not `"amount"`) rather than leaving the
  raw column name as the label.
- Category/segment bar charts get `"sort": "-y"` (or `-x`) wherever order
  doesn't already carry meaning (i.e. not already time-ordered), so the
  tallest bar reads first instead of alphabetical column order.
- EDA filler panels — the histograms/category-count charts with **no** bound
  finding (everything with a finding already gets a caption from 7.18) —
  get a one-line plain-language caption computed from data already on hand
  (`_five_number_summary`, category counts): "Most values fall between X and
  Y" / "Three categories cover N% of rows." Small, bounded, not a new
  analysis.

## Verification (no pytest, per directive)

Each agent verifies its own slice by **direct invocation**, not by writing
test files — the same pattern Round 7's session used to validate background-
agent output (HANDOVER.md §1, §2c):
- Agent A: a scratch script calling `run_sandboxed(code, ref, extra_globals={"k": "v"})`
  directly and printing the result — confirms the value round-trips and the
  relaxed static check still rejects a genuinely missing `RESULT`.
- Agent B: a scratch script building a `GeneratedToolSpec` by hand, calling
  `GeneratedTool(spec).run(file_path=..., **params)` directly against a real
  fixture CSV — confirms `execute()`/`get_schema()`/`findings()` all round-trip
  without the controller in the loop yet.
- Main session (8.3): once A and B land, a full `main.py --no-llm` smoke run
  first (must stay green — generated tools must not appear, per decision 5),
  then a targeted script that constructs `AgentController`, calls
  `tool_registry.register(GeneratedTool(spec))` directly (simulating what the
  controller hook will do), and confirms the tool is callable by name and its
  `findings()` output reaches `memory.ranked_findings()`.
- Agent C: re-run the existing `out_test` smoke-run reports and visually
  confirm (by reading the emitted `dashboard.json`/`report.html` chart specs)
  that a currency measure's axis now carries a `$` format and a percent
  measure carries `%` — no visual/browser check needed since the artifact is
  plain JSON.

## Sequencing

```
8.1 (Agent A) ─┐
8.2 (Agent B) ─┼─→ 8.3 (main session, integrates all three) ─→ done
8.4 (Agent C) ─┘        (chart work is independent of 8.1/8.2)
```

8.1 and 8.2 must both land before 8.3's controller hook can be wired
end-to-end (the hook calls `tool_factory.register_and_persist`, which builds
on `GeneratedTool`, which calls the extended `run_sandboxed`), but both agents
can build against the interface contracts above without waiting on each
other. 8.4 has no dependency on 8.1–8.3 and runs fully in parallel.

## Status: landed 2026-09-18, verified end-to-end

All four items (8.1–8.4) landed via three parallel subagents plus the main
session's own controller/memory/prompt wiring, exactly against the interface
contracts above — no deviations reported by any agent except two the agents
caught and fixed themselves before handback: a non-deterministic `finding_id`
(was using Python's randomized `hash()`, replaced with a deterministic slug —
`BaseTool`'s documented determinism contract requires this) and
`validate_spec` not originally rejecting a declared parameter named `df`/
`SCHEMA`/`RESULT`/`FINDING` (would have collided with the sandbox's reserved
globals on every call — now rejected at definition time).

**Verified together, end-to-end, by direct invocation** (a throwaway script,
deleted after use — no pytest, per directive): `define_analysis_tool` called
directly on the churn dataset defined a real tool (`high_charge_share`,
sharing customers above mean `monthly_charges`, with a `FINDING` payload);
`_maybe_register_generated_tool` registered it into the live `ToolRegistry`
and persisted its spec to `<run>/generated_tools/high_charge_share.json`;
the tool was then called by name, returned `status="ok"`, and its
`findings()` produced a `Finding` (`"58% of customers pay above the 50
average"`) that reached `memory.ranked_findings()` — the full loop the
finding bus (Round 7) was built to support. Also verified: the generated
tool and the meta-tool are both **absent** from `candidate_tools(use_llm=False)`
(decision 5 holds), and a collision attempt naming a generated tool
`clean_data` was rejected with the real `CleanDataTool` left untouched
(decision 4 holds). The full `main.py --no-llm` pipeline was re-run twice
during this work (once after the controller/memory/prompt wiring, once after
all three agents' changes were combined) and stayed green both times — no
regression to the existing deterministic path.

**Not verified this round:** an actual LLM-driven run exercising
`define_analysis_tool` from a real model's own plan (everything above called
the tool directly, bypassing the planner); opening the Streamlit app to see a
generated-tool finding/chart rendered in the UI; the `load_persisted_tools`
cross-session reload path (built and unit-verified by Agent B in isolation,
not exercised from a second `analyze()` run in this session). None of these
block calling Round 8 done — they're the honest list of what a live/UI pass
would still confirm.

### Follow-up testing pass (2026-09-18, same day)

The "no testing" directive above was reversed later the same day ("do the
tests, make sure everything works well together"). What that pass added,
beyond the manual verification already described:

- **138 new pytest tests**, real (no mocking of the code under test):
  `tests/test_sandbox.py`/`tests/test_sandbox_worker.py` (extended —
  `extra_globals`/`FINDING`/the relaxed static check), `tests/test_tool_factory.py`,
  `tests/test_generated_tool.py`, `tests/test_define_analysis_tool.py`
  (all new), `tests/test_round8_tool_creation.py` (new — the full
  define→register→call→finding-bus→collision-guard loop, formalizing the
  manual script above as a repeatable test), `tests/test_chart_theme.py` and
  `tests/test_dashboard_chart_formatting.py` (new — 8.4's unit-format work,
  exercised through real `build_dashboard()` calls, not just the theme
  functions in isolation).
- **Two pre-existing stale tests fixed** (predating Round 8, surfaced by
  actually running the suite): `tests/test_sandbox.py`'s
  `test_result_assignment_nested_in_if_not_recognized` asserted the exact
  behavior decision 6 deliberately relaxed — rewritten to assert the new
  contract instead of reverting it; `tests/test_data_shapes.py`'s
  `test_json_records_rejects_with_clear_error` asserted JSON support didn't
  exist, which stopped being true when U1.2 landed in Round 7 — rewritten to
  assert successful parsing.
- **`tests/test_planted_effects.py` was run for the first time** (written in
  the Round 7 session per its own harness design, per HANDOVER.md never
  actually executed before now) and found **5 real, pre-existing failures**
  — none caused by Round 8, all latent since Round 7. Diagnosed and fixed,
  one finding bus/pipeline bug each except where noted:
  - **Concentration finding missing on a transactional fixture** — the
    fixture's own construction was too uniform (13.73% top-decile share,
    below both `cohort_analysis`/`concentration_analysis`'s shared 15%
    triviality floor) to produce a real concentration signal; fixed the
    *fixture* (added a genuine top-decile-spends-3x plant), not the tools.
  - **`segment_comparison` never tested a classification target against any
    dimension at all** — its measure-selection helper only included binary
    flags (like `churn`) as a last resort when no continuous measure
    existed, and `default_params()` separately pinned the sweep to a
    currency-flavoured column instead of the dataset's actual target. Real
    pipeline bug, fixed in `src/tools/segment_comparison.py`.
  - **Drawdown fixture's background noise organically exceeded its own
    planted drawdown** — `financial_analysis.py`'s max-drawdown calculation
    was verified correct; the fixture's unrelated random walk produced a
    larger, unplanted decline elsewhere in the series. Fixed the fixture
    (capped incidental drawdowns below the planted one).
  - **No pay-gap finding possible without a gender/sex column** — the only
    "unadjusted pay gap" comparison implemented was gender/sex-specific,
    even though the tool already computed per-department medians right next
    to it. Real coverage gap, fixed in `src/tools/workforce_analysis.py`
    (generalized to a reusable `_pay_gap_between()` helper, called for both
    department and gender/sex independently).
  - **A datetime column got auto-selected as an ML target at 20%
    confidence** — two related real bugs: `DatasetMetadata.detect_target_with_confidence()`
    (`src/core/memory.py`) never excluded datetime-typed columns from
    candidacy, and `_decide_analysis_mode()` (`src/core/controller.py`)
    recorded `mode: "model"` for any non-None candidate regardless of
    confidence, disconnected from the confidence bands its own caller
    actually acts on. Both fixed — a real "decline to model" bug independent
    of the specific fixture.
- **Two mypy regressions caught in review, not by the diagnostic agent's own
  run** (it doesn't run mypy): a variable named `_` in
  `workforce_analysis.py` shadowed `execute()`'s own `**_: Any` kwargs
  catch-all parameter (renamed); a stale `# type: ignore[override]` on
  `findings()` was carried over unnecessarily during the same edit (removed).
  Full suite re-confirmed green (472 passed) after both fixes.
- **Final state**: `pytest tests/ -q` → 472 passed, 1 deselected, 0 failed.
  `ruff`/`mypy` clean on every file this pass touched. Two more `main.py
  --no-llm` smoke runs, both clean.

### Live-LLM testing pass (2026-09-18, same day)

Everything above (the 472-test suite plus the manual verification script) called
`define_analysis_tool`/`GeneratedTool` directly — this pass was the first time
Round 8 ran through an actual LLM's own plan, per the explicit gap the prior
pass flagged as unverified. Provider: `openrouter` (real key confirmed present,
never printed), model `nvidia/nemotron-3.5-lightning:free` — a small, free-tier
model, not a frontier one; that fact matters for what this pass could and
couldn't establish (see "Not established" below).

`main.py` was run against `data/sample_customer_churn.csv` with an objective
designed so no built-in tool answers it directly (a composite "service value
score" across six add-on columns, asked once at a 0.5 cutoff and once at a
stricter 0.3 cutoff) — specifically to see whether the model would define one
parameterized tool and call it twice, per the design intent this round was
built for.

- **Real, deterministic bug found and fixed, independent of the specific
  model**: iteration 1's LLM response was structurally complete — every brace
  balanced, a `"reasoning"` field, presumably a `"steps"` array — but
  contained a literal, unescaped newline inside the `"reasoning"` string
  instead of an escaped `\n`. `LLMClient._parse_json` (`controller.py`)
  called `json.loads` in strict mode at every repair stage (the initial
  parse, the optional `json_repair` library, the manual truncation-repair,
  and the brace-extraction fallback) — strict mode rejects any control
  character inside a string with "Invalid control character," so **all four
  repair layers failed identically** on input that was otherwise perfectly
  parseable. The response was discarded, iteration 1 fell back to the
  deterministic 9-tool plan, and the run's remaining iterations were spent
  without ever giving the LLM a clean first look at the objective. Small/
  free-tier models are exactly the ones most likely to skip `\n`-escaping in
  natural-language fields, making this a real, recurring blocker to Round 8
  ever being exercised live, not a one-off fluke of this specific response.
  **Fix**: added `LLMClient._loads_lenient` (`controller.py`) — tries
  `json.loads` strictly first, falls back to `strict=False` (which permits
  raw control characters inside strings, the one thing wrong with this class
  of response, without weakening any other validation) — and used it at all
  four `json.loads` call sites inside `_parse_json`. Regression test added:
  `tests/test_controller.py::TestLLMClientParseJson::test_literal_newline_in_string_value_still_parses`,
  reproducing the exact live failure (a complete JSON object with a raw
  newline embedded in a string field) and asserting it now parses instead of
  raising.
- **Real cleanup, not a behavior bug**: `tool_factory.register_and_persist`
  still carried a `hasattr(memory, "add_generated_tool")` guard and a
  `TODO(controller-integration)` comment claiming `MemorySystem` "does not
  yet expose a generated-tools list" — stale from before the controller-
  wiring pass landed `add_generated_tool`/`list_generated_tools` on
  `MemorySystem` (`memory.py:481,494`). The guard was always true given the
  current code (so no behavior changed), but the comment actively
  misdescribed the persistence path as a no-op. Removed the guard and the
  stale TODO; the call is now unconditional, matching what actually happens.
- **Verified by reading the code, not by observing it live** (the run never
  got far enough to exercise these paths before the JSON-parsing bug above
  cut the first iteration short): the collision/version-bump contract
  (design decision 4) — `ToolRegistry.register()` itself is an unguarded
  overwrite, but `_maybe_register_generated_tool` calls `validate_spec` first
  with `existing_tool_names` already excluding previously-generated names, so
  a built-in-name collision is rejected before `register()` is ever called
  and a same-name redefinition (self-correction) is correctly treated as a
  version bump, not a collision — matches the interface contract exactly, no
  bug found here despite it being the most likely place for one per the
  original task brief.
- **Not established this pass** (the run was stopped, by direction, before a
  second iteration completed, once the JSON-parsing root cause above was
  identified and fixed at the code level): whether a live model — including
  a stronger one than this free-tier default — actually reaches for
  `define_analysis_tool` for this objective, parameterizes the cutoff instead
  of redefining the tool per call, and produces a final answer whose numbers
  trace back to the tool's own `FINDING` evidence rather than being flagged
  by `_flag_unverified_claims`. That live behavioral verification is
  explicitly left to a follow-up run (by the user, not this session) now that
  the JSON-parsing blocker is fixed.

<!--

Closed 2026-09-18 — see HANDOVER.md §1-§2c for the validated implementation
and the Closed Ledger at the end of this file for one-line pointers per item.
Kept here, commented out, for the historical reasoning behind each item.

# Round 7 — Autonomy & Insight-Quality Audit

**Question asked:** the goal is an app that is *fully autonomous on any field of
data* and returns the *best dynamic report, dashboard, charts and real human
insights* possible. What would that take, and where is the current backend
against it?

**Method.** Read the whole backend (`src/core/*`, `src/rlm/engine.py`,
`src/tools/*`, `main.py`, the report/dashboard builders, and `app.py` where it
constrains the backend API), then **ran the pipeline end-to-end twice** and
judged the artifacts it produced, rather than reasoning from source alone:

| Run | Dataset | Command | Time |
| :-- | :--- | :--- | :--- |
| A | `data/sample_customer_churn.csv` (1,500 × 21, classification) | `main.py --no-llm` | 60 s |
| B | synthetic transactional CSV (4,000 × 8: `order_id`, `order_date`, `customer_id`, `region`, `product_category`, `quantity`, `amount`, `discount_pct`) | `main.py --no-llm` | 25 s |

Run B was built with three **planted effects** so the output could be graded
rather than admired: a +25% revenue premium in `region == "West"`, a ×1.4
November/December seasonal lift, and 10 orders per customer (repeat-purchase
structure). Artifacts inspected: `*_report.md`, `*_raw.json`,
`dashboard.json`, `report.html`, `final_report.json`, console output.

**Scope note.** Rounds 2–4 audited *correctness of the ML path*; Round 5
audited *generality of the data path*; Round 6 was domain-layer residue. **None
of them asked whether the output is any good.** That is this round. Where an
item here is already logged elsewhere it is cross-referenced, not restated.

**Constraints confirmed with the user before writing this:**

1. **LLM-on is the primary mode**; deterministic is a fallback that must be
   decent, not equal. *Caveat on this whole round: both runs were `--no-llm`,
   so the plan below is graded against the deterministic path. Findings 7.1–7.7
   and 7.10 are structural and hit both paths; 7.12 is where the LLM path's own
   gaps are collected and is explicitly ungraded until a provider is run.*
2. **Audience is "all types of people"** → the report must be *layered*, not
   re-pitched: one executive surface, one analyst surface, one appendix.
3. **Structural change is allowed**, including new core modules, new tools, a
   changed `MemorySystem` schema, and changes to the agent loop. AGENTS.md's
   "Ask First" gates are therefore pre-cleared for this round's items, but each
   one below still carries its flag so the blast radius stays visible.
4. **Dashboard stack: a recommendation was requested.** See
   [Appendix A](#appendix-a--recommended-dashboard-configuration). Short
   version: keep Vega-Lite as the spec language, change *what* is put in the
   specs and how they are assembled.

---

## Phase 1 — Target state: what the goal actually requires

Ten capabilities. Each is phrased as a property of the *output*, because that is
what the goal is about; the code implication follows.

**T1 — Semantic type & grain layer.** The system must know that a column is a
*measure* (revenue, charges), a *dimension* (region, contract), a *flag* (0/1
senior_citizen), an *ordinal* (rating 1–5), an *identifier*, or a *time axis* —
and what one row *is* (one order? one customer? one customer-month?). Structural
kind (`numeric`/`categorical`) is not enough: almost every defect in Phase 2
traces back to a flag or a measure being treated as generic numeric.

**T2 — An explicit analysis-mode decision.** Before planning, the system should
decide and *record* whether this dataset calls for description, explanation,
prediction, forecasting, segmentation, or comparison — and be willing to answer
"this is a descriptive dataset; there is nothing here worth predicting."
Autonomy includes the autonomy to decline to model.

**T3 — A question agenda, not a tool list.** A human analyst starts from
questions ("which segments churn most?", "is revenue growing?", "what
concentrates?") and then picks methods. The agenda should be generated from the
semantic layer + objective, ranked by expected value, and *then* mapped to tool
calls — so coverage is judged in question space, where a user's "why" lives.

**T4 — One finding bus.** Every analysis should emit structured `Finding`
objects (claim, the numbers behind it, effect size, direction, confidence,
surprise, audience-facing sentence, caveat, chart binding). Insights, the
Markdown report, the HTML report, the dashboard and the UI should all be
*projections of the same ranked finding list*. Any output surface that
re-derives its own narrative from raw tool JSON will drift — and today three of
them do, each differently.

**T5 — An insight library that matches what humans call insight.** Concretely:
segment-vs-baseline rate/mean comparison with lift; driver direction *at level
granularity* ("month-to-month contracts churn 3.1× the base rate"), not
"`contract` is a categorical feature"; concentration (Pareto); change over time
on the correct grain; cohort behaviour; anomalies and regime breaks;
missingness that correlates with the target; relationship strength with
practical meaning. Plus the inverse — **triviality suppression**: a histogram of
a 0/1 flag, a uniform category count, an outlier scan that flags 39% of rows are
noise and must not be printed as findings.

**T6 — Layered narrative.** Headline → so-what → evidence → method, generated
by the LLM *from findings only* (the verbatim-number enforcement in
`_flag_unverified_claims` already exists and is the right mechanism to build
on). Non-technical readers get lift and money; analysts get effect sizes and
diagnostics; the appendix carries methodology and caveats.

**T7 — Dashboard as a story, not a gallery.** Panels chosen because a finding
needs them, ordered by that finding's rank, each captioned with what it shows;
cross-filtering within a section; low-value panels suppressed; aggregate-first
data so specs stay small and unbiased.

**T8 — Trust surfaces.** Effect size before p-value — Round 5 item 4 landed
this properly (the stats tool leads with the effect, and the BH table reaches
the reports' limitations section), so the remaining gap is *ranking*: findings
are still ordered by tool execution order, not by effect size, so a negligible
result can sit above a large one. Plus method-fit checks (don't run IQR on flags, don't log1p a negatively-skewed
binary); sufficiency gates that actually gate rather than warn; a quality score
that discriminates.

**T9 — Time to first insight.** A profile + cheap findings within seconds, deep
work streamed behind it. Run A spent 45 s of 60 s inside `train_model` with
tuning on by default. For an interactive app that is a product problem, not
just the perf item P1.2 files it as.

**T10 — Measurable insight quality.** Planted-effect datasets plus an assertion
harness: "on this file, the system must report the West premium, the Q4 lift and
the repeat-purchase concentration." Without this, "best insights possible" is
unfalsifiable and every later change is a matter of taste. Run B was a one-off
version of exactly this, and the system failed all three plants.

---

## Phase 2 — What the runs actually produced

### Run A — churn (classification, no domain match)

| Observed in the artifacts | Why it matters |
| :--- | :--- |
| Key Insight #5 was `Mann-Whitney U` on **`feature_column='senior_citizen'`, `group_column='gender'`** — p=0.0846, "negligible effect" | Nobody asked whether a 0/1 senior-citizen flag differs by gender. A non-question became a headline finding, and then the **only row** in the Benjamini-Hochberg table in both reports. |
| `treatments_applied`: **"Applied log1p to `phone_service` (skew=-2.94 — heavy tail compressed)"** | `phone_service` is a 0/1 flag, and log1p is wrong for *negative* skew in any case. Reported to the user as a considered decision. |
| **39.47%** of rows flagged as outliers (IQR), printed as Key Insight #4 | At 39% the honest finding is "IQR does not fit this data" (flags + skewed charges). No per-column breakdown is emitted, so the number can't be interrogated. |
| `evaluate_model` computed `driver_narrative`: **`contract` #1, `tenure_months` #2, `internet_service` #3** | The single most insight-like output of the run. It reaches `report.html` and nothing else — absent from `insights`, the Markdown Key Insights, `final_report.json` and the CLI summary. And `contract`'s entry reads "a categorical feature", with no direction and no level named. |
| Quality score **100/100** with 11 missing cells, an ID column and the above | The score does not discriminate. |
| Dashboard: 13 panels including `Distribution — senior_citizen` (a binary), `Category Counts — gender` (uniform by construction) | Panels are chosen by column kind and cardinality, not by whether they carry a finding. `contract` — the #1 driver — gets **no panel at all**. |
| Ingest log line says `task=clustering` while the run trained a classifier | `IngestDatasetTool` runs before target auto-detection; the tool log contradicts the run. Cosmetic, but it is in the report. |

### Run B — transactional (domain matched, planted effects)

| Observed | Why it matters |
| :--- | :--- |
| Domain inference: `transactional`, **confidence 1.00**, roles fully resolved, evidence "10.0 rows per customer" | The domain layer works. This is the foundation the rest of the plan builds on. |
| `cohort_analysis` summary: *"4,000 transactions totalling 704,945.38 …; 399 customers, 99.8% repeat; top 10% of customers drive 18.9% of revenue"* | The best human sentence the whole system produced — and it appears in **no** insight list, **no** chart, and not in the Markdown report's "Additional Analyses" (which covers `cluster_data`, `time_series_analysis`, `text_analysis`, `geospatial_analysis`, `dimensionality_analysis` only — cf. item 6.1, which logged the *chart* half of this gap). |
| `time_series_analysis` ran over **4,000 raw transaction rows** in row order → *"No clear trend (R²=0.0006)… no strong seasonal signal"* | The planted ×1.4 Q4 lift was **missed**. A transaction log is not a series; it must be aggregated to a grain (daily/weekly/monthly **sum** of the measure) before trend/seasonality means anything. |
| Trend chart: **"monthly mean `quantity`"** | Wrong measure (`quantity`, not `amount`) and wrong aggregate (mean, not sum) for a revenue question. Chosen by `_rank_numeric_features` variance ranking, which has no notion of a measure. |
| `train_model` fitted a **regression on `amount`** (R² 0.81, `split=time_series`) | No one asked for a prediction. `amount` was auto-selected purely because the name heuristic lists it (`_NUMERIC_TARGET_NAMES` in `memory.py`, 0.85 ≥ the 0.75 autonomy threshold), then 4 models were trained to predict revenue from quantity and category — near-tautological, and it consumed most of the run. |
| `select_statistical_test`: **`quantity` by `region`** → p=0.0574, negligible | The planted effect is `amount` by `region` (+25%). The tool tested the neighbouring column and reported "no effect", i.e. an actively misleading answer. |
| Planted effects found: **0 of 3** (West premium, Q4 lift, revenue concentration — the last computed but never surfaced) | This is the headline result of the round. |
| Dashboard: 8 panels, including `Relationship — quantity vs discount_pct` | No revenue-over-time, no revenue-by-region, no revenue-by-category panel. For a sales file, the three charts anyone would want are all absent. |

---

## Phase 3 — Gap analysis against the current codebase

| Target | What exists today | Gap |
| :-- | :--- | :--- |
| T1 semantic layer | `profiler.py:263-341` — kinds: numeric / categorical / datetime / boolean / identifier / constant / text; flags: `severe_skew`, `high_cardinality`, `high_missing`, `id_like`. `domains.py` adds semantic *roles* but only for 3 registered domains | No measure/dimension/flag/ordinal distinction, no units, no row-grain fact. 0/1 ints are plain `numeric`, which is the root of the log1p, IQR, histogram and test-pairing defects |
| T2 analysis mode | `metadata.infer_task_type()` (`memory.py:184-205`) + confidence-gated target detection (`controller.py:722-785`) | Only ever chooses classification / regression / clustering / eda. No "describe, don't model" outcome; a name-matched numeric column becomes a regression target unconditionally |
| T3 question agenda | `ToolRegistry.candidate_tools` scores tools via `applies_to` (`controller.py:557-598`); `_build_fallback_plan` sweeps them by score | Planning is in *tool* space. Nothing represents the question a tool is meant to answer, so parameter choice is arbitrary (`statistical_analysis.py:219-240` takes `numeric[0]`/`groups[0]` positionally) and coverage is never assessed against the objective |
| T4 finding bus | `ToolResult.output` dicts + three independent narrators: `_deterministic_final` (`controller.py:1320-1432`), `report_generator._format_additional_analyses` (`report_generator.py:183-238`), `html_report.build_html_report` (`html_report.py:159+`) | Each surface hardcodes a different subset of tools. `_deterministic_final` knows 5 tools (train, correlation, outliers, stat-test, cluster); drivers, cohort, financial, workforce, time-series, geo, PCA never become insights. This is one bug repeated per surface, and it is why Run B's best sentence vanished |
| T5 insight library | Real analysis exists in `statistical_analysis.py`, `cohort_analysis.py`, `financial_analysis.py`, `workforce_analysis.py`, `ml_pipeline._explain_drivers` | No segment-vs-baseline comparison anywhere (the core of "why"); driver direction stops at "a categorical feature"; no concentration/Pareto as a general tool; no triviality suppression — everything computed is printed |
| T6 layered narrative | `PromptManager` Form 2 (`insights` / `recommendations` / `key_metrics`); `_flag_unverified_claims` (`controller.py:1690-1734`) enforces verbatim numbers | Flat lists, one register for all readers. The grounding mechanism is good and should become the *contract* for a narrative written from findings |
| T7 dashboard story | `dashboard.py` — 11 builders, fixed candidate order (`dashboard.py:638-651`), `MAX_POINTS=1000` random row sample inlined per chart | Column-driven, not finding-driven; no suppression; no cross-filter; no captions tied to a finding; `cohort`/`financial`/`workforce` unpanelled (item 6.1); raw rows inlined per chart (P2.7) |
| T8 trust | Effect sizes + CIs + sample-size notes in the stats tool; `_detect_target_leakage`; `_TREND_MIN_R_SQUARED`; `degradations.py`; BH correction at report time | No method-fit gate (IQR on flags — item 6.4; `_SkewLog1pTransformer.fit` at `ml_pipeline.py:139-156` uses `abs(skew)` and only skips `min < 0`, so a negatively-skewed 0/1 flag is log1p'd); `is_sufficient` warns but does not gate; quality score saturates at 100 |
| T9 speed | `--no-ml`, `_step_cache`, read cache, `n_jobs=1` (measured, P1.3) | Tuning on by default = 45 s of a 60 s run (P1.2); `analyze()` is blocking with callbacks only (P2.5); no progressive "cheap findings first" stage |
| T10 measurable quality | `tests/` (232 passing), `scripts/validate.py`, `scripts/dry_run.py`, `tests/test_data_shapes.py` | Tests assert *mechanism* ("a chart spec is JSON-serialisable"), never *insight recovery*. Nothing would have failed when Run B missed all three planted effects |

---

## Phase 4 — Prioritised improvement plan

Ranked by **"does this change what the user reads"**, not by effort.

| # | Item | Impact | Effort | Depends on | Flags |
| :-- | :--- | :--- | :--- | :--- | :--- |
| **7.1** | Finding bus: one `Finding` type, all surfaces project it | **Highest** | 2 d | — | **Ask First** (memory schema) |
| **7.2** | Insight library: segment comparison, level-granular drivers, concentration | **Highest** | 2.5 d | 7.1 | new tools |
| **7.3** | Semantic layer: measure / dimension / flag / ordinal + row grain | **Highest** | 1.5 d | — | touches profiler API |
| **7.4** | Analysis-mode decision + "don't model that" target guard | High | 1 d | 7.3 | behaviour change |
| **7.5** | Question agenda between profile and plan | High | 2 d | 7.3, 7.1 | controller loop |
| **7.6** | Relevance-driven statistical testing (family, not one arbitrary pair) | High | 1 d | 7.3, 7.5 | |
| **7.7** | Grain-aware time series (aggregate before diagnosing) | High | 1 d | 7.3 | |
| **7.8** | Dashboard story layer + suppression + finding→panel binding | High | 2 d | 7.1, 7.2 | artifact schema |
| **7.9** | Layered report (exec / analyst / appendix) | Medium-High | 1.5 d | 7.1, 7.6 | |
| **7.10** | Method-fit guards: outliers, skew sign/cardinality, quality score | Medium-High | 1 d | 7.3 | subsumes 6.4 |
| **7.11** | Planted-effect evaluation harness | **Highest** (as a gate) | 1.5 d | — | do first |
| **7.12** | LLM path: grounded narration contract, ask-your-data, cost accounting | High | 2.5 d | 7.1 | ungraded so far |
| **7.13** | Time-to-first-insight: tuning budget, progressive stages, streaming | Medium | 2 d | 7.5 | P1.2 / P2.5 |
| **7.14** | Dashboard/report service contract + run history | Medium | 2 d | 7.8 | P2.4 / P2.5 |
| **7.15** | Summary tab leads with findings, not model diagnostics | **Highest** | 1 d | 7.1 | UI |
| **7.16** | Wire or delete the three dead ML sliders; expose the tuning cost | High | 0.5 d | — | UI, trust |
| **7.17** | One chart source, one theme module (kill the duplicate charts) | High | 1 d | — | UI, precedes 7.8 |
| **7.18** | Panel metadata drives layout; finding→chart links; cross-filter | Medium-High | 1.5 d | 7.8 | UI |
| **7.19** | Stream findings during the run instead of stage chips | Medium-High | 1 d | 7.13 | UI |
| **7.20** | Ask-your-data box, what-if form, run history | High | 2.5 d | 7.1, 7.12 | UI, new surfaces |
| **7.21** | Tab restructure + owned empty/error states | Medium | 1 d | 7.15 | **your call** |
| **7.22** | `app.py` decomposition, styles into the sheet, a11y pass | Medium | 2 d | 7.15–7.19 | UI |

### Sequencing

```
7.11 (harness, red)  →  7.3 (semantics)  →  7.1 (finding bus)  →  7.2 (insights)
                                    ↘  7.4, 7.7, 7.10  ↗
                                       7.5 (agenda) → 7.6
                                       7.8 (dashboard) → 7.9 → 7.12 → 7.13 → 7.14
```

**If only one thing gets done: 7.11 then 7.1 + 7.2 as one push.** The harness
makes the goal falsifiable; the finding bus plus the insight library is the
difference between "correlation r=0.44 on quantity↔amount" and "West-region
orders run 25% above the rest, and Q4 lifts revenue 40%."

---

### 7.11 — Planted-effect evaluation harness *(do this first; it fails today)*

`tests/fixtures/` already has a factory (Round 5 item 1) for *shape* edge cases.
Add a **semantic** corpus: generated files with known, documented effects, and
tests that assert the effects are *recovered in the findings*, not that a tool
ran.

Corpus (5–7 files, each with a `plants` manifest): transactional with a region
premium + Q4 seasonality + revenue concentration (Run B, kept); churn with a
dominant categorical driver (`contract`) and a weak numeric one; a price series
with a drawdown; an HR roster with a pay gap by department; a panel/grouped
file; a pure-description file with **no** plantable target (asserting the system
*declines* to model — the T2 behaviour); a text-column file.

Assertion style, so it grades output rather than mechanism:

```python
findings = run_pipeline(path, use_llm=False).findings
assert_finding(findings, kind="segment_lift", dimension="region",
               level="West", measure="amount", min_rank=3)
assert_no_finding(findings, kind="outlier_scan", min_pct=30)   # method-fit
```

Score per dataset: plants recovered / plants present, plus a **noise count**
(findings that are trivially true). Print a table. That table is the metric the
rest of this round optimises, and today it reads 0/3 on Run B.

### 7.1 — Finding bus

New `src/core/findings.py`:

```python
@dataclass
class Finding:
    finding_id: str
    kind: str                  # segment_lift | driver | trend | concentration | ...
    headline: str              # audience-facing, numbers embedded
    detail: str                # analyst-facing
    evidence: dict[str, Any]   # the exact numbers, traceable to a ToolResult
    source_tool: str
    measure: str | None; dimension: str | None; level: str | None
    effect: float | None; effect_kind: str | None   # lift | cohens_d | r | eta_sq
    p_value: float | None; p_adjusted: float | None
    confidence: float          # evidence strength × sufficiency
    surprise: float            # distance from the base rate / prior expectation
    importance: float          # ranking key = f(effect, confidence, surprise, objective fit)
    caveats: list[str]
    chart_hint: dict[str, Any] | None   # what would show this
```

Tools keep returning their current `output` dicts (no tool rewrite), and gain an
optional `findings() -> list[Finding]` derived from that output — so the
migration is per-tool and incremental. `MemorySystem` grows a `findings` list
(the **Ask First** schema change) with `add_findings` / `ranked_findings`.

Then delete the three parallel narrators: `_deterministic_final`,
`report_generator._format_additional_analyses` and `html_report`'s section
picker all become projections of `ranked_findings()`. **This alone fixes Run A's
orphaned drivers and Run B's orphaned cohort insight, and structurally prevents
the next tool from being orphaned** — the failure mode Round 3 fixed once by
hand with `_render_other_findings` and which recurred anyway for the three
domain tools.

Objective fit enters `importance` here, which is how "prioritise analyses that
answer the objective" becomes a mechanism instead of a prompt sentence.

### 7.2 — Insight library

Three new tools plus one upgrade. These are the analyses that produce sentences
a human would actually say:

1. **`segment_comparison`** — for each dimension × measure (or target rate),
   compute per-level mean/rate vs the overall baseline, with lift, CI, n, and a
   significance test; emit `segment_lift` findings ranked by |lift| × n.
   Recovers Run B's West premium and Run A's `contract` story: *"Month-to-month
   customers churn at 42% vs a 27% baseline (1.6×, n=812)."* Guard against
   testing every level of every dimension by capping cardinality and correcting
   across the family (reuse `multiple_testing.apply_benjamini_hochberg`).
2. **`concentration_analysis`** — Pareto/Gini over a measure by an entity
   dimension. Generalises the one good line `cohort_analysis` already produces
   (top-10% revenue share) to any file with a measure and an entity.
3. **`change_analysis`** — period-over-period movement on the correct grain
   (which period changed, by how much, which segment drove it). This is the
   "what happened" question no current tool answers.
4. **Upgrade `_explain_drivers`** (`ml_pipeline.py:1138-1210`) — for a
   categorical driver, report *which level* pushes which way (per-level mean
   prediction or target rate), so `contract` stops being "a categorical
   feature". Pair permutation importance with the `segment_comparison` result
   for the same column so the model-based and descriptive views agree.

Plus **triviality suppression** as a shared predicate: no findings from flags
with no variance, uniform categories, near-duplicate measures, or a scan that
flags >20% of rows (that becomes a *method-fit* caveat instead — 7.10).

### 7.3 — Semantic layer

Extend `ColumnProfile` with `semantic_role` (`measure` | `dimension` | `flag` |
`ordinal` | `identifier` | `time` | `text` | `constant`) and `unit_hint`
(currency / percent / count — `coercion.py` already detects currency and percent
at read time and throws that knowledge away after converting). Add
`DatasetProfile.grain` (the column set that makes a row unique, and the
`rows_per_entity` fact `domains._rows_per_unique` already computes for domain
checks).

Rules that matter most: an integer column with `nunique == 2` is a **flag**, not
numeric; a small-cardinality integer with order is **ordinal**; a numeric column
that is summable and not a flag/ordinal/ID is a **measure**. Then:

- `dashboard._rank_numeric_features` ranks measures for measure questions
  (fixes "monthly mean quantity" → "monthly total revenue").
- `_histogram_charts` skips flags (fixes `Distribution — senior_citizen`).
- `DetectOutliersTool` skips flags (part of the 39% number).
- `_SkewLog1pTransformer` skips flags and negative skew (fixes the
  `phone_service` log1p treatment).
- `select_statistical_test` gets typed candidates instead of `numeric[0]`.

This is one change with six downstream fixes, which is why it outranks its own
size.

### 7.4 — Analysis-mode decision + target guard

Add `decide_analysis_mode(profile, objective) -> AnalysisDecision` (mode, target
or None, rationale, alternatives rejected) and record it in memory and in the
report's methodology. Rules: an auto-detected target needs *both* a name signal
**and** a plausible modelling shape; a measure at transaction grain in a
domain-matched transactional file is a **descriptive** subject, not a regression
target unless the objective asks for prediction; `_AUTODETECT_HIGH` stops being
sufficient on its own. Stating "no prediction target — this is a descriptive
sales log, so here is what it describes" is a better answer than R²=0.81 on
`amount`, and it also returns Run B's 25 s of model fitting to the user.

### 7.5 — Question agenda

New `src/core/agenda.py`: `build_agenda(profile, decision, domains, objective)
-> list[Question]` where a `Question` carries text, kind, the columns it
concerns, an expected-value score, and the tool call(s) that would answer it.
The planner prompt then presents the **agenda** alongside the tool list, and the
deterministic planner walks the agenda instead of sweeping `applies_to` scores —
so parameters come from the question ("does `amount` differ by `region`?")
rather than from positional defaults. Unanswered questions become a reported
coverage gap, which is also the honest place for "I could not answer X."

Keeps `applies_to` as the capability filter; adds intent above it.

### 7.6 — Relevance-driven statistical testing

`select_statistical_test` currently answers one arbitrary pairing per call.
Change its contract to a **family**: given a measure (or target) and the
candidate dimensions, test all admissible pairings, return them ranked by effect
size with BH correction applied *inside* the family, and emit one finding per
survivor. `requires_context`'s `target_column → group_column` injection is
currently defeated by `default_params` filling `group_column` first
(`base.py:124-128` only fills empty params) — the fix is for the agenda (7.5) to
supply both, and for `default_params` to become "propose", not "decide".

**Ordering constraint, or this regresses:** `default_params` exists to make
tools *schedulable* on no-LLM runs (`base.py:81-97`), and
`_build_fallback_plan` drops any tool whose required schema params are still
unfilled (`controller.py:1185-1200`). So the agenda must populate parameters
**before** that runnability check runs. Demote `default_params` without doing
that and the deterministic planner stops scheduling `select_statistical_test`
(and `generate_visualizations`) altogether — the exact failure
`default_params` was added to fix.

### 7.7 — Grain-aware time series

Before diagnosing, resample to the natural grain: group by day/week/month and
aggregate the **measure** appropriately (sum for additive money/count measures,
mean for rates), then run trend/ADF/seasonality on that series. Report the grain
in the output and in the chart title. Seasonality should use the calendar
(month-of-year, day-of-week factors), not just autocorrelation lags over row
order — which is what missed the Q4 lift in Run B. `financial_analysis` already
does grain reasoning (`_infer_periods_per_year`); reuse it rather than
re-deriving.

### 7.8 — Dashboard story layer

`build_dashboard` becomes `build_dashboard(findings, df, profile, …)`: each
top-ranked finding requests its panel via `chart_hint`, EDA panels fill the
remainder, and suppression drops panels with no story. Panels carry
`finding_id`, `priority`, `layer` (exec / analyst / appendix) and a caption
taken from the finding's `headline` — so a chart always says what it shows.
Includes the 6.1 panels (cohort RFM + revenue-by-month, financial drawdown,
workforce pay/tenure) as the first concrete customers of the new signature.
Artifact schema and interaction config: [Appendix A](#appendix-a--recommended-dashboard-configuration).

### 7.9 — Layered report

Both reports gain a fixed skeleton driven by finding rank: **Headline** (3–5
findings in plain language with money/lift) → **What to do** → **Evidence**
(effect sizes, tests, model metrics) → **How it was analysed** (agenda,
rationales, decision record from 7.4) → **Limits** (degradations, caveats, BH
table, unverified claims). Same content, three depths; the executive layer never
contains a p-value and the appendix never hides one.

### 7.10 — Method-fit guards

Distribution-aware outlier detection (item **6.4**, still open): choose IQR /
z-score / modified z-score (MAD) / isolation forest by skew and kind, skip flags
and IDs, emit per-column counts, and when >20% of rows flag, report it as
"method unsuitable" rather than as a finding. `_SkewLog1pTransformer`: skip
flags/ordinals, require positive skew, and prefer `yeo-johnson` where skew is
negative. Quality score: make it discriminate (ID columns, unusable target,
sufficiency, coercion damage should all move it off 100).

### 7.12 — LLM path *(the part this round could not grade)*

Both runs were deterministic, so the LLM path is unaudited output-wise. Once a
provider is configured, three things belong here regardless: (a) the narrative
prompt should take **findings, not raw tool JSON** (shorter context, grounded by
construction, and `_flag_unverified_claims` becomes a near-no-op rather than a
net); (b) an "ask your data" turn that answers a follow-up question from the
finding bus plus one optional tool call (PLAN.md Tier 2 item 8 — highest demo
value and now cheap, since findings are structured); (c) token/cost/latency
accounting (**P3.1**) and a per-run budget, because iterating a 15-cycle loop
against a paid provider without accounting is how a bill happens. Also worth a
check on the iteration prompt re-sending the whole result set each cycle
(**P1.6**) once findings replace raw JSON.

### 7.13 — Time to first insight

Tuning off by default with an explicit `tuning_budget_s` (**P1.2**); a
`profile → cheap findings → deep analysis` staging so the UI can show real
findings in ~2 s; `analyze()` gains a generator/event interface (**P2.4/P2.5**)
so Streamlit and a future API consume the same stream instead of callbacks.

### 7.14 — Service contract + history

Freeze a versioned artifact bundle per run (`profile.json`, `findings.json`,
`dashboard.json`, `report.html`, `report.md`) under a per-run output directory
(**P2.4**), a thin FastAPI layer over the event stream, and a SQLite run history
(PLAN.md Tier 2 item 10). This is what makes the backend UI-agnostic; it is last
because everything above changes the artifact shape.

---

## Phase 5 — UI recommendations (`app.py`, 2,693 lines)

Graded the same way as the backend: by what the interface *says* to the person
reading it, against `DESIGN.md`'s own stated audience — "anyone with a
spreadsheet and a question… a shopkeeper checking sales… they don't know what
'generalisation' or 'train-test gap' means, and they shouldn't have to."

Current shape: a sidebar (upload, objective, provider/model, toggles, ML
sliders, run button), a hero with a 7-stage pipeline drawing, then six result
tabs — **Summary · Your Helpers · Charts · Full Details · 3D Cinematic Journey ·
Downloads** (`app.py:2136-2143`).

### 7.15 — The Summary tab opens with model diagnostics instead of answers

The first screen after a run is a **Data Quality** radial gauge plus three
tiles: **Best Model**, **CV Score**, **Train-Test Gap** (`app.py:2217-2251`).
For the stated audience, two of those three are meaningless and the third is a
model-internal number. The *answers* — `insights` / `recommendations` — sit
below them under "Key Discoveries" (`app.py:2260-2278`).

Worse on descriptive data: Run B had no legitimate prediction target, so the
same three tiles render **`N/A`**, **`0%`**, **`0%`** (defaults at
`app.py:2177-2189`), and a fourth of the first screen becomes a report on a
model nobody wanted. The `if not train_out` branch does try to compensate with
`st.info("**What we found:** …")` (`app.py:2201-2211`), but it prints *one*
tool summary, chosen from a hardcoded five-tool list that excludes
`cohort_analysis` — which is exactly why Run B's best sentence never appeared in
the UI either.

**Recommendation.** Invert the tab. Top: the three highest-`importance`
findings from the bus (7.1) as full-width sentence cards, each with its number
in the sentence. Then "What to do". Then a single **trust strip** (quality
score, row count, caveat count, "1 result may not hold" in `--risk` when
applicable). Model metrics move down into an analyst layer that a "Show
technical detail" toggle expands in place — one surface, two depths, which is
how "all types of people" gets served without writing two products.

### 7.16 — Three sidebar controls do nothing

`max_depth` (`app.py:1774`), `test_pct` (`:1776`) and `n_cv` (`:1777`) are
assigned and never read again — no env var, no constructor argument, no tool
parameter (`grep` for each: one hit at the slider, plus an unrelated
`train_out.get('n_cv_folds', 5)` display at `:2506`). A user who sets "Test
split % = 30" gets a 20% split and is told nothing. `enable_rlm`, by contrast,
is wired properly (`:1904`, `:1983`) — so the pattern exists, these three just
missed it.

**Recommendation.** Wire them through `train_model`'s parameters (they are all
real schema params) or delete them. A control that silently does nothing costs
more trust than a missing feature, and it is a ~20-line fix either way. While
there: the tuning cost from 7.13 belongs here as a visible choice — "Thorough
(slower)" vs "Quick" — rather than a hidden `tune_hyperparameters: bool = True`
default (`ml_pipeline.py:480`) that turned Run A into a 60-second wait.

### 7.17 — Charts are built twice, themed twice, and drift

`dashboard.py` builds the model-comparison and correlation panels; the Summary
tab then builds *its own* versions of the same two charts inline
(`app.py:2297-2319`, `:2328-2346`). Two implementations of one chart, and they
already differ — the inline pair reference `PEN_BLUE`/`PLOT_INK`
(`app.py:731-735`) while the dashboard builders carry hardcoded blueprint-era
hexes (`#12467e`, `#8aa6c2` in `dashboard.py:441`, `:513`, `:594`) that belong
to the "Drafting Table" identity `DESIGN.md` explicitly retired. The Vega
config exists twice as well: `app._get_vega_config` (`:737-763`) and
`html_report._VEGA_PLOT_CONFIG`.

**Recommendation.** One `src/core/chart_theme.py` exporting the Ledger tokens
and a `vega_config(theme)` function; `dashboard.py` emits **no** colours
(theme injected at render, per Appendix A item 4); `app.py` renders panels from
the artifact and builds no chart of its own. This is 7.8's prerequisite — a
story dashboard cannot have a second, differently-themed copy of two of its
panels living upstream of it.

### 7.18 — Panels are static, and nothing connects a finding to its chart

Panel width is decided by a hardcoded id whitelist — `_full_width_ids =
{"model_comparison", "top_correlations", "scatter_top_pair", "time_series"}`
(`app.py:2388`) — so every panel added later silently renders half-width. There
is no cross-filtering, no drill-down, and no link from an insight card to the
chart that evidences it.

**Recommendation.** Take width/order/layer from the panel's own fields in the
artifact (7.8), not from a set in the view. Add three interactions, in this
order of value: (a) each finding card carries a "see the chart" anchor to its
bound panel; (b) one composed spec per section so a click on a region/segment
cross-filters its neighbours (Appendix A item 3); (c) a "rows behind this"
expander on a finding, showing the actual rows its evidence cites — the single
most trust-building thing an analysis UI can offer a sceptical reader.

### 7.19 — The wait shows stages, not findings

A run blocks for 1–3 minutes (`app.py:1928`) behind stage chips and a spinner
whose copy escalates to "Still working…" (`:2012-2016`, `:2028-2032`). Nothing
real appears until everything finishes, because results are written to
`session_state` in one batch at the end (`:2051-2066`).

**Recommendation.** Once `analyze()` yields events (7.13 / P2.5), stream
findings into the Summary as they land — "Found: West-region orders run 25%
above the rest" at second 3, not at second 60. Same total runtime, a completely
different felt experience, and it makes the deep phases interruptible: a "good
enough, stop here" button becomes possible because partial findings are already
on screen.

### 7.20 — The highest-value surfaces are still missing

Three things the backend can now almost support, none of them present: an **Ask
your data** box (a follow-up question answered from the finding bus plus at most
one extra tool call — PLAN.md Tier 2 item 8, and cheap once 7.1 exists); a
**what-if** form built from the feature schema against the saved pipeline (Tier
2 item 9 — the Round 4 refactor made saved models self-contained precisely so
this is possible); and **run history** (Tier 2 item 10) so a user can compare
today's file with last month's.

Also missing, and smaller: an "explain this" affordance on any number, which is
the cheapest way to serve the non-technical half of the audience without
diluting the analyst half.

### 7.21 — Two of six tabs are showmanship; the empty states are unowned

**Your Helpers** (agent grid + handoff stream) and **3D Cinematic Journey**
(`ui/cinematic_3d.py`, 353 lines + 1,881 lines of JS) occupy a third of the top
level. That is a legitimate choice for a demo or a thesis defence, and I am not
calling it a defect — but it competes for the same attention as the findings,
and it is where a large share of `app.py`'s size lives.

**Recommendation** (a judgement call for you, not a fix): promote insight
surfaces to the top level — **Answers · Charts · Details · Downloads** — fold
the helper grid into Details as the run trace, and keep the cinematic export in
Downloads where it already exists as a standalone HTML. Then spend the reclaimed
tab on 7.20's "Ask your data".

Either way, the empty and error states need owners: `st.info("No dashboard was
generated for this run.")` (`:2412`) tells the user nothing about why, and the
failure path prints a raw Python traceback (`:2081-2082`) to an audience
`DESIGN.md` defines as non-technical. Each should say what happened, what it
means, and what to do — and 7.5's coverage gaps give you the honest version of
the first: "here is what I could not answer, and why."

### 7.22 — Structure: one 2,693-line module, styled inline

38 `unsafe_allow_html` blocks, most carrying hardcoded inline CSS — the
Summary tab's KPI strip alone is ~35 lines of inline `style="…"` with literal
`px` sizes and `var(--token)` references mixed together (`app.py:2217-2251`).
`_inject_theme_css()` already exists (`:116`) as the right mechanism, so the
inline styling is habit rather than necessity — and it means `DESIGN.md`'s
tokens are authoritative in the stylesheet but optional everywhere else.

**Recommendation.** Move every style into the injected sheet, leave only class
names in the f-strings, then split the module: `ui/pages/<tab>.py` for layout,
`ui/components/` for the cards/tiles/strips, `app.py` as wiring and session
state only. Add the accessibility pass `DESIGN.md` promises for tokens but
nothing enforces for components: keyboard focus and `aria` labels on the
click-to-expand helper cards, a text label beside every `--risk` colour cue
(colour is currently the only carrier in the defect stamp and the gap tile),
and a check that the two-column grids stack rather than scroll on a narrow
window.

---

## Appendix A — Recommended dashboard configuration

**Keep Vega-Lite.** It is the right choice here and switching would cost more
than it returns: specs are plain JSON (an LLM can write and a test can assert
them), the same spec renders in Streamlit (`st.vega_lite_chart`), in the
self-contained HTML report (`vega-embed`), and in any future JS frontend, and
theming is a single injected `config` object. The problems in Phase 2 are not
Vega-Lite's — they are *what we put in the specs*. Five changes:

1. **Aggregate-first data, not sampled rows.** This is **P2.7**'s fix, still
   open; it already specifies bin counts for histograms, five-number summaries
   for box plots, and a lower scatter cap, with the resampled time series as
   the model. Round 7 adds only two arguments to it: the sampling is
   *statistically* wrong as well as heavy (a 1,000-row sample of
   `dashboard.MAX_POINTS` hides exactly the tails an outlier or skew finding is
   about), and the aggregates should be emitted as named datasets (next item)
   rather than per-spec copies. Send pre-computed
   bins, per-level aggregates and resampled series; keep raw rows only where the
   mark needs them (scatter, boxplot), and then say so in the caption.
2. **Named datasets, referenced once.** Emit
   `{"datasets": {"revenue_by_month": [...], "by_region": [...]}, "panels": [...]}`
   with panels using `"data": {"name": "revenue_by_month"}`. One copy per
   dataset instead of one per chart, and two panels over the same rows can then
   be cross-filtered.
3. **Cross-filtering inside a section.** Compose a section as one spec with
   `vconcat`/`hconcat` + a shared `params` selection (`"select": {"type":
   "point", "fields": ["region"]}`) and `filter` transforms on the dependants,
   plus `"bind": "scales"` for zoom on time axes. In Streamlit, prefer one
   composed spec per section over N separate `st.vega_lite_chart` calls (which
   cannot cross-filter); use `on_select="rerun"` only where a selection must
   drive Python.
4. **One theme module.** `html_report._VEGA_PLOT_CONFIG` and `app._get_vega_config`
   are two copies of the same intent and will drift — move to
   `src/core/chart_theme.py`, keyed to the Ledger tokens in `DESIGN.md`, and
   inject at render time (never bake theme into a stored spec, so the same
   artifact can render light/dark).
5. **Escape hatches, deliberately narrow.** Real basemaps → pydeck/deck.gl or
   Plotly (Vega-Lite has no tile layer; the current "geospatial" panel is a
   lat/lon scatter); 3-D/cinematic → the existing Three.js surfaces;
   >100k points → server-side aggregation or a data URL served by the API
   (7.14) rather than inline values.

With LLM-on as the primary mode, add one thing Vega-Lite cannot give you: a
per-panel **caption written from the finding** (7.8), which is what turns a
gallery into a narrated dashboard.

## Appendix B — Reproducing the runs in this round

```bash
# Run A
.venv/Scripts/python.exe main.py --dataset data/sample_customer_churn.csv \
    --no-llm --output-dir out_a

# Run B: generate the planted-effect transactional file first (see 7.11 —
# this generator becomes tests/fixtures/), then
.venv/Scripts/python.exe main.py --dataset tx.csv --no-llm --output-dir out_b
```

Graded artifacts: `out_*/reports/*_report.md`, `*_raw.json` (tool outputs,
`driver_narrative`, `treatments_applied`), `dashboard.json` (panel titles),
`final_report.json` (insights actually delivered). `PYTHONIOENCODING=utf-8` when
running through Bash on Windows; the `?`/`�` in console chart titles is the
cp1252 console, not a defect in the artifacts.

-->

## Still open from Round 7 (the commented block above has the full reasoning)

| Item | What's still missing |
| :-- | :--- |
| **7.12** | LLM path: findings-grounded narration, ask-your-data, cost accounting — backlog itself calls this "ungraded" pending a live provider run |
| **7.13** | Progressive `profile → cheap findings → deep analysis` staging + a visible tuning-budget toggle; blocked on P2.5 |
| **7.14** | Versioned artifact bundle, FastAPI layer, run history — sequenced last by design |
| **7.18** (partial) | Cross-filtering between charts in a section, and a "rows behind this" expander on a finding — the simpler "→ see chart" text link is done |
| **7.19** | Streaming findings into the UI during a run instead of static stage chips — blocked on P2.5 |
| **7.20** (partial) | What-if form and run history were scoped out; the "ask your data" keyword search over findings is done |
| **7.22** (partial) | `ui/pages/`/`ui/components/` module split not attempted (by design, HANDOVER §3) — `app.py` is still one file |

# Carried-forward open items

Everything below was raised in an earlier round and is **still open**, verified
against the current code while writing Round 7 (call-site claims by `grep -rn`
across `src/`, `app.py`, `main.py`, `scripts/` and `tests/`).

## Failing gates

<!-- Closed 2026-09-18 — Q1 fixed, see HANDOVER.md §2 and Closed Ledger.

### Q1 — Two tests assert a palette that no longer exists

`pytest tests/ -q` → 338 passed, **2 failed** (228 s):

- `tests/test_landing.py::test_landing_day_and_night_themes`
- `tests/test_landing_v2.py::test_v2_warm_ledger_tokens`

Both assert `"--stock: #130f0b" in content`. `DESIGN.md`'s Ledger tokens are
`--stock` = `#f7eedd` (Day) / `#241c14` (Night); `#130f0b` belongs to neither,
and the rendered landing page ships `class="theme-day"`. So the tests encode a
palette that predates the current design system — the same staleness Round 2's
§0 fixed for the plate tests, recurring in the landing suite.

**Fix:** give the palette a single source of truth and have both the code and
the tests import it, rather than re-asserting hexes. This is the third instance
of the same root cause — Round 2's §0 fixed it for the plate tests, Round 7's
7.17 fixes it for the two Vega configs and the blueprint-era hexes still in
`dashboard.py` — so **do Q1 and 7.17 together**, or the palette gets
de-duplicated twice and re-diverges a third time. Restoring a green suite also
restores the baseline every other item in this file is validated against.

-->

## Regressed since it was fixed

<!-- Closed 2026-09-18 — R3.1 fixed, see HANDOVER.md §2 (`generate_visualizations`
excluded from the deterministic sweep) and Closed Ledger.

### R3.1 — `generate_visualizations`' PNGs still reach nobody, and the fallback plan schedules it again

Round 3 removed `generate_visualizations` from the deterministic plan because
nothing consumed its PNGs. Both halves are back:

- **No consumers.** `grep` for `.png` / `chart_path` / `image_path` across
  `app.py`, `src/core/html_report.py` and `src/tools/report_generator.py`
  returns nothing. The files are written and never read.
- **It is scheduled again.** The Round 3 fix deleted a hardcoded step, but
  `_build_fallback_plan`'s profile sweep (`src/core/controller.py:1156-1211`)
  re-adds any tool scoring ≥ `_FALLBACK_MIN_SCORE`;
  `GenerateVisualizationsTool` keeps `applies_to`'s 1.0 default and is absent
  from `_FALLBACK_EXCLUDED_TOOLS`. Confirmed in Round 7's Run A: step 5,
  *"profile-driven selection scored 'generate_visualizations' at 1.00"*.

**Fix:** decide what the PNGs are *for* — a static export for the Markdown
report and the Downloads tab (then wire them there and reference them from
`report_generator`), or nothing (then add the tool to
`_FALLBACK_EXCLUDED_TOOLS` and leave it for the LLM planner to call
deliberately). Either is fine; the current state pays for them and shows nobody.
The general lesson is Round 7's 7.1: a hardcoded per-tool fix in one surface
does not survive a generic mechanism added later.

-->

## Round 6 residue

### Prioritised items

| # | Item | Impact | Effort | Depends on | Flags |
| :-- | :--- | :--- | :--- | :--- | :--- |
| **6.1** | Chart panels for `cohort_analysis` / `financial_analysis` / `workforce_analysis` | **Highest** | 0.5 d | — | user-requested |
| **6.2** | Tests for the domain layer, date coercion, leakage detector, toggles, read cache | High | 1 d | — | |
| **6.3** | Surface `date_ambiguous` as an explicit warning | Medium | 1 h | — | cheapest |
| **6.4** | Distribution-aware outlier detection | Medium | 0.5 d | — | **Ask First** |

<!-- Closed 2026-09-18 — 6.1 fixed, see HANDOVER.md §2 (`dashboard.py` 6.1 row:
cohort/financial/workforce panels added) and Closed Ledger.

### 6.1 — Three domain tools compute results that are never charted  *(session "P1")*

`build_dashboard` resolves exactly six tool outputs
(`src/core/dashboard.py:631-636`): `train_model`, `correlation_analysis`,
`cluster_data`, `time_series_analysis`, `geospatial_analysis`,
`dimensionality_analysis`. `cohort_analysis`, `financial_analysis` and
`workforce_analysis` are absent, so RFM segments and drawdown curves reach the
reports as prose and tables and never become a chart.

Ranked top because it is the gap the user named directly ("real charts with real
value"), and because the analysis behind the charts already exists and is
verified — this is presentation wiring, not new computation.

One `_*_chart` builder per panel, appended to `candidates`: drawdown area +
cumulative-return line (financial), RFM segment bar + revenue-by-month line
(cohort), tenure histogram + headcount-by-department bar (workforce). Cap rows
at `MAX_POINTS` and mind P2.7 — dashboard specs inline raw rows, so each new
panel adds to artifact size.

-->

### 6.2 — The new code has no tests at all  *(session "P2")*

The suite is green (**326 passed, 1 deselected, 1 warning**, 69 s) and covers
none of the last two sessions' work. Evidence that does not rot with the count:
grepping `tests/` for
`domains|infer_domains|_detect_target_leakage|date_dayfirst|read_cache|use_ml|use_llm`
matches **zero files**. No `tests/test_domains.py` exists;
`tests/test_coercion.py` predates the date work and never mentions a convention.

Risk order: date-convention detection (silently rewrites data), the leakage
detector (needs a true positive *and* a true negative), domain inference
(structural discriminators + the `resolve_column` exclusion order that fixed
AOV), the capability toggles (`use_ml=False` must exclude every `requires_ml`
tool), and `invalidate_read_cache` firing on rewrite — the Windows
mtime-granularity trap has no test holding it shut.

**Do not merge this with P4.1–P4.2.** Those cover the older untested tools
(`time_series`, `text_analysis`, `geospatial`, `dimensionality`). 6.2 is
new-code coverage; the two are separate debts with separate scopes.

*Verified still open:* no test in `tests/` references `infer_domains`,
`cohort_analysis`, `financial_analysis`, `workforce_analysis`,
`geospatial_analysis`, `text_analysis` or `time_series_analysis` (the only
matches are incidental mentions in `test_dashboard.py`, `test_data_shapes.py`
and `test_ml_enhancements.py`). Build this inside Round 7's 7.11 harness rather
than as a parallel suite.

<!-- Closed 2026-09-18 — 6.3 fixed, see HANDOVER.md §2 (`degradations.py` 6.3
row: explicit `date_ambiguous` warning) and Closed Ledger.

### 6.3 — `date_ambiguous` is computed, then rendered as if it were a success  *(session "P3")*

`_detect_date_convention` returns `"date_ambiguous"` (`src/core/coercion.py:172`)
when no day in the column exceeds 12 — dd/mm and mm/dd are indistinguishable
from the data, and the parser picks one silently. That verdict reaches the user
only through the generic coercion line in `src/core/degradations.py:46-50`:
`Column 'x' repaired from string to datetime (date_ambiguous rule): N converted, 0 left unparsed`
— which reads as a clean repair. Nothing warns that the dates may be wrong.

Fix: a dedicated branch in the degradation log for datetime coercions carrying
the ambiguous rule, naming the column and stating the convention could not be
determined. Worst failure mode in this round (a confidently wrong date axis on
every chart) against the smallest fix.

*Verified still open:* `date_ambiguous` appears only where it is produced
(`src/core/coercion.py:172`) — no reader in `degradations.py`, in either
report, or in the UI.

-->

<!-- Closed 2026-09-18 — 6.4 fixed, see HANDOVER.md §2 (`data_processing.py`
6.4 row: distribution-aware outlier detection, verified 0.67% vs 39.47%) and
Closed Ledger.

### 6.4 — Outlier detection ignores the skew flag the profiler already sets  *(session "P4")*

`detect_outliers` (`src/tools/data_processing.py:286`) applies IQR, z-score or
isolation-forest with no reference to the column's distribution. On the retail
fixture it flagged **~22% of revenue rows** — revenue is right-skewed by nature,
so the tail is the business, not an anomaly.

The profiler already computes what is needed: skewness per column and a
`severe_skew` flag (`src/core/profiler.py:317-318`, `SEVERE_SKEW_THRESHOLD`),
surfaced in `to_prompt_string` (`profiler.py:182-184`). `detect_outliers` never
reads it.

Fix: for a `severe_skew` column, apply IQR to log-transformed values or switch
to a robust alternative (MAD-based, or asymmetric fences), and report which rule
was used per column. **Ask First** — the default changes reported outlier counts
on existing datasets.

*Scope note:* Round 7's 7.10 generalises this into a method-fit gate (skip
flags and IDs, report per-column counts, declare the method unsuitable above a
20% flag rate). Keep 6.4's skew-specific reasoning — it is the concrete half of
that item.

-->

## Round 5 residue

<!-- Closed 2026-09-18 — U1.2 and U1.5 fixed, see HANDOVER.md §2 (`io.py` row:
json/jsonl/parquet/gz/zip with depth-capped flatten; row cap + reservoir
sampling, one choke point) and Closed Ledger.

### U1.2 — Format coverage is narrow (roadmap item 8)


Supported: `.csv`, `.tsv`, `.xlsx`, `.xls`. Unsupported: JSON,
JSONL, Parquet, SQL, compressed CSV, nested/semi-structured data of any kind.

The supported set is also *declared inconsistently* across three places:

| Site | Accepts |
| :--- | :--- |
| `security.ALLOWED_EXTENSIONS:25` | `.csv`, `.xlsx`, `.xls` |
| Streamlit uploader (`app.py:1413`) | `csv`, `xlsx`, `xls` |
| tools' `_read_df` | `.csv`, **`.tsv`**, `.xlsx`, `.xls` |
| `controller._read_dataframe:42` | `.csv`, `.xlsx`, `.xls` |

`.tsv` is a phantom format: reachable by direct tool/CLI invocation, rejected
by upload validation, unprofileable, and corrupt when it does load.

*Verified still open:* `src/core/io.py:33` —
`SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}`. `.tsv` is no
longer a phantom format (the unified reader handles it end to end), but
JSON/JSONL, Parquet and compressed CSV remain unsupported.

**8. Format expansion** — JSON/JSONL (with `json_normalize` flattening for
nested records, depth-capped), Parquet, `.gz`/`.zip` CSV. Needs a decision
from you on nested data: flatten, or reject with a clear message? **Ask First**
— this expands the product's supported-input promise, which is a product
decision, not a code one.

### U1.5 — Scale is unbounded, and the read cache only half-closes it (roadmap item 9)


Measured on this machine (read + profile):

| Rows | File | `pd.read_csv` | `profile_dataframe` | In-memory |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.6 MB | 0.03 s | 0.01 s | 0.5 MB |
| 200,000 | 11.6 MB | 0.15 s | 0.09 s | 9.9 MB |
| 1,000,000 | 58.0 MB | 0.64 s | 0.57 s | 49.6 MB |

Profiling is **not** a bottleneck at these sizes and needs no optimisation.
Two architectural risks remain, both unmeasured beyond 1M rows and stated here
as risks rather than defects:

1. **No row cap, no chunking, no sampling policy.** Every read is a full load.
   Memory is the binding constraint and nothing degrades gracefully when it
   binds.
2. **Every tool re-reads the file from disk independently.** A 9-tool run on a
   1M-row file pays the ~0.64 s read nine times and holds nine transient
   copies. `dashboard.py` has a `MAX_POINTS` cap for rendering, but nothing
   equivalent governs analysis input.

*Partly closed:* `src/core/io.py` now caches reads on path+mtime+size
(`_cache_key`, `read_any`), so the nine-reads-per-run waste is gone. **Still
open:** there is no row cap and no reported sampling above it, so a file an
order of magnitude past 1M rows has no defined behaviour.

**9. Scale policy** — a configurable row cap with *reported* reservoir
sampling above it, and a read-once cache keyed on path+mtime so a 9-tool run
reads once rather than nine times. **Ask First** — sampling changes results,
so whether that is acceptable (and the default threshold) is your call.
Note the measurements in U1.5: this is about robustness beyond 1M rows and
wasted I/O, not a current performance problem.

-->

---

# P1 — Speed

### Measured baseline

50,000 rows × 16 columns (12 numeric, 2 categorical, 1 datetime, 1 binary
target), 13.2 MB CSV, warm OS cache, single run:

| Step | Wall time |
| :--- | ---: |
| `_read_df` — one CSV parse | 0.12 s |
| `profile_dataframe` | 0.08 s |
| `clean_data` | 0.53 s |
| `detect_outliers` (iqr) | 0.55 s |
| `correlation_analysis` | 0.11 s |
| **`train_model`** (tuning **off**) | **5.71 s** |
| `evaluate_model` (permutation importance) | 2.34 s |
| `cluster_data` (k search 2..8) | 3.60 s |

Per-model breakdown at 20,000 rows (the threshold below which tuning is
**on by default** — `src/tools/ml_pipeline.py:227`):

| Model | `.fit()` | `cross_val_score` n_jobs=1 | `cross_val_score` n_jobs=-1 | `_tune` (n_iter ≤ 8) |
| :--- | ---: | ---: | ---: | ---: |
| random_forest | 0.20 s | 1.21 s | **2.66 s** | **14.03 s** |
| xgboost | 0.25 s | 0.92 s | **1.92 s** | **6.99 s** |
| logistic_regression | 0.01 s | 0.09 s | **1.24 s** | 0.30 s |

Two results drive everything below.

<!-- Closed 2026-09-18 (the "on by default" headline complaint) — see
HANDOVER.md §2 (`ml_pipeline.py` row: "P1.2 (tuning off by default)") and
Closed Ledger. The remaining sub-fixes here (successive halving, an adaptive/
visible budget) are now tracked live under Round 8-preceding item **7.13**
(still open, see "Still open from Round 7" above), not restated twice.

### P1.2 — Hyperparameter tuning is ~95% of model-training time, and it is on by default

At 20k rows, tuning costs **21.3 s** against **0.46 s** of actual fitting —
a 46× multiplier. `do_tune = tune_hyperparameters and len(X) <= 20_000`
(`src/tools/ml_pipeline.py:227`) means the default path for any dataset a
user is likely to upload interactively pays it. (A replanning cycle no
longer pays it twice — the `_step_cache` added in Round 2 serves an identical
re-planned step from cache.)

The arithmetic: `n_iter=min(8, n_combos)` × `n_splits=5` = up to **40 fits
per model** in `_tune` (`src/tools/ml_pipeline.py:492-500`), *plus* 1
final fit, *plus* 5 more in the separate `cross_val_score` at `:288`.
**46 fits to report one model.** Three models → ~138 fits.

Three fixes, in order of value:

1. ~~**Delete the redundant CV entirely when tuning ran.**~~ **Done in
   Round 2** — `_tune` now returns `cv_mean`/`cv_std` from
   `search.best_score_`/`cv_results_`, and the separate `cross_val_score` is
   skipped when tuning ran. The fit arithmetic above therefore reads ~41 fits
   per tuned model, not 46, and the two remaining fixes below are what is
   left of this item.
2. **Switch to successive halving.**
   `sklearn.model_selection.HalvingRandomSearchCV` evaluates many
   configurations on small data subsets and promotes only survivors,
   typically reaching the same optimum in 3–5× less time on this search
   space shape.
3. **Make the tuning budget explicit and adaptive.** `n_iter=8` on a
   3-parameter random-forest grid of 27 combinations is a coin flip
   dressed as a search. Either raise it and accept the cost knowingly, or
   drop to a 2-point grid for the interactive path and expose
   `tune_hyperparameters` in the UI. Right now the user pays 21 s for a
   search they cannot see or control.

*Still the default:* `tune_hyperparameters: bool = True`
(`src/tools/ml_pipeline.py:480`). Measured again in Round 7 — Run A spent 45 s
of its 60 s total inside `train_model`. Round 7's 7.13 and 7.16 carry the
product half of this (a visible "Quick vs Thorough" choice, and a budget); this
item stays for the defaults and the grid size.

-->

<!-- Closed 2026-09-18 — P1.6 fixed, see HANDOVER.md §2 (`memory.py`:
`get_results_summary_digest()`; `prompt_manager.py`: "P1.6(a): iteration
prompt uses the digest") and Closed Ledger.

### P1.6 — The iteration prompt re-sends the entire accumulated result set every cycle

`get_iteration_user_prompt` (`src/core/prompt_manager.py:310-360`) calls
`memory.get_results_summary()` with the default
`max_chars_per_result=1200` (`src/core/memory.py:367`), which serialises
**every** tool result accumulated so far. By iteration 8 with 6 tools per
cycle, that is up to 48 entries — tens of thousands of tokens re-sent on
every call, growing linearly, with `MAX_ITERATIONS` defaulting to 15.

There is also no token accounting anywhere (see P3.1), so this cost is
invisible.

**Fix:** two cheap changes. (a) Pass only the *latest* iteration's results
in full and a one-line-per-tool digest for older ones — the LLM's job on
iteration N is to react to what just happened. (b) On providers that
support it, mark the static system prompt for caching (Anthropic
`cache_control`, OpenAI automatic prefix caching) — the tool-description
block from `get_system_prompt` (`src/core/prompt_manager.py:238-239`) is
byte-identical across all 15 calls and is currently re-billed each time.

-->

### P1.8 — The test suite takes 124 s, which is why it stops being run

199 passing tests at ~0.6 s each is dominated by real sklearn fits on
generated frames. At two minutes, the suite falls outside the
edit-run-edit loop and gets skipped locally — which is a plausible reason
the five stale failures in §0 survived.

**Fix:** mark the genuinely slow model-fitting tests
`@pytest.mark.slow`, add `-m "not slow"` to the default `addopts` in
`pyproject.toml:62`, and run the full set in CI. Shrink the synthetic
frames in the ML tests — they exist to check plumbing and output shape,
not to measure accuracy. Target: under 15 s for the default suite.

---

# P2 — Architecture and extensibility

<!-- Closed 2026-09-18 — P2.2 resolved via option (b), see HANDOVER.md §2b
(AGENTS.md's Layer Rules table corrected + `tests/test_architecture.py` added
as a drift-check) and Closed Ledger.

### P2.2 — AGENTS.md's layer rules are violated by the code they describe

AGENTS.md states: `tools/*` may depend on `base.py`, stdlib and data libs,
and must **not** import `memory`. But:

- `src/tools/base.py:20` — `from src.core.memory import MemorySystem, ToolResult`
  at module level.
- `src/tools/data_processing.py:19` — `from src.core.memory import DatasetMetadata`.
- `src/tools/base.py:23-25` — `DatasetMetadata` and `DatasetProfile` under
  `TYPE_CHECKING`, which is the honest version of the same dependency.

The imports aren't wrong — `prepare_params` genuinely needs `MemorySystem`
and `ToolResult` is genuinely the tool-layer return type. The *document*
is stale, and a stale architecture doc is worse than none: it stops being
checked.

**Fix — pick one and commit:**
(a) Extract `ToolResult`, `DatasetMetadata`, `AnalysisStep`, `DatasetProfile`
into `src/core/contracts.py` that both layers may import, leaving
`memory.py` as behaviour only. This makes the stated rule true again and
is the better end state.
(b) Amend the AGENTS.md table to permit `memory` *types* (not the
`MemorySystem` instance) in the tool layer.

Either way, add a CI check — `import-linter` with a contract file, or a
ten-line `tests/test_architecture.py` walking the AST — so the rules can't
drift again silently.

*Still open, and note the irony:* `src/tools/base.py` imports `MemorySystem`
from `src/core/memory.py` (line 20) — exactly what AGENTS.md's layer table
forbids for `tools/*`.

-->

<!-- Closed 2026-09-18 — P2.3 fixed, see HANDOVER.md §2 (`base.py` row: "P2.3
output_dir validation via resolve_output_path() in prepare_params()") and
Closed Ledger.

### P2.3 (second half) — the output-root guard is still dead code

The first half landed in Round 2: `clean_data` and `detect_outliers` now
declare `output_subdir = "data"`. What remains:

Meanwhile `src/core/security.py:197-212` defines `resolve_output_path`,
whose entire purpose is "refuse any escape from the output root." Grepping
every call site: `tests/test_security.py` only. The same is true of
`escape_csv_formulas` (`src/core/security.py:179-194`) — tested, never
called, so every CSV the pipeline writes is still formula-injectable when
opened in Excel.

Both derived files also flow to the LLM as `file_path` values for
downstream tools, and `output_dir` is an LLM-supplied parameter, so the
planner currently chooses where the pipeline writes.

**Fix:** give both tools `output_subdir = "data"`, and route *every*
tool's file write through `resolve_output_path(output_root, ...)`. Apply
`escape_csv_formulas` to user-facing CSV exports only — its own docstring
correctly warns not to apply it to files the pipeline reads back.

*Verified still open:* `resolve_output_path` is defined
(`src/core/security.py:203`) and called from nowhere in `src/`.

-->

<!-- Closed 2026-09-18 — P2.4 fixed, see HANDOVER.md §2 ("P2.4: session-scoped
output dir default + output/latest.txt") and Closed Ledger.

### P2.4 — Runs share one output directory, so concurrent runs corrupt each other

Every run writes to fixed paths: `output/models/random_forest.pkl`
(`src/tools/ml_pipeline.py:303`), `output/reports/dashboard.json`
(`src/core/controller.py:962`), `output/reports/report.html`
(`src/core/controller.py:993`). Two analyses in flight — two Streamlit
sessions, or a CLI run alongside the app — overwrite each other's models
and reports mid-flight, and `evaluate_model` can load a `.pkl` written by
the other run.

**Fix:** `MemorySystem` already generates `self.session_id`
(`src/core/memory.py:303`). Make the controller's `_output_dir` default to
`output/runs/{session_id}/` and symlink or copy `output/latest`. This is a
prerequisite for anything multi-user, and it gives run history for free.

-->

### P2.5 — `AgentController.analyze()` has no non-blocking or streaming interface

`analyze()` (`src/core/controller.py:649-797`) runs the whole pipeline —
up to 15 LLM round trips and every tool execution — in one synchronous
call, driving a Rich `Progress` bar it owns (`:675-679`). The only
extension points are two fire-and-forget callbacks,
`on_step_callback` / `on_iteration_callback`
(`src/core/controller.py:507-509`).

That forces every non-CLI caller to block. `app.py:1707` calls
`agent.analyze()` inline in the Streamlit script run, so the UI freezes
for the full duration, the callbacks can only append to a list that is
rendered afterwards (`app.py:946-987`), and there is no way to cancel a
run. An HTTP API in front of this would have the same problem.

This is a backend API gap, not UI work: the controller offers no way to
observe or interrupt a run in progress.

**Fix:** add `analyze_iter()` as a generator yielding structured progress
events (`stage`, `iteration`, `tool`, `status`, `payload`), and implement
`analyze()` as `deque(self.analyze_iter(), maxlen=0)` plus a return value.
Accept an optional `cancel_token` checked at iteration and step
boundaries. Move the Rich `Progress` out of the controller and into
`main.py`, where the CLI owns its own presentation — the controller
currently imports `rich.progress` and prints emoji directly
(`src/core/controller.py:29-30`, `:675-679`), which is presentation logic in
the orchestration layer.

*Partly addressed:* `on_step_callback` / `on_iteration_callback` exist
(`src/core/controller.py:672-675`) and `app.py` drives the stage chips from
them. **Still open:** `analyze()` is a single blocking call, so nothing partial
can be rendered — this is the dependency Round 7's 7.19 needs.

<!-- Closed 2026-09-18 — P2.7 fixed, see HANDOVER.md §2 (`dashboard.py` row:
"P2.7 (aggregate-first, named datasets)") and Closed Ledger.

### P2.7 — Dashboard specs inline raw rows, so artifact size scales with chart count

`_records` inlines up to `MAX_POINTS = 1_000` rows per chart
(`src/core/dashboard.py:36`, `:92-101`), and `build_dashboard` can emit
4 histograms + 3 category charts + scatter + box + time-series + results
charts (`src/core/dashboard.py:489-511`). Each of those Vega-Lite specs
carries its own copy of the data, and `build_html_report` embeds all of
them into a single `report.html`. Existing artifacts already show the
shape of this: `output/reports/*_raw.json` are ~124 KB each, dominated by
`cluster_data`'s 1,000 `pca_points` (`src/tools/clustering.py:170-177`).

**Fix:** pre-aggregate server-side rather than shipping rows —
histograms become bin counts, box plots become five-number summaries,
the time series is already resampled (`src/core/dashboard.py:341-347`) and
should be the model for the rest. Scatter is the one chart that genuinely
needs points; cap it lower (250–400 is visually indistinguishable at
typical opacity). Expect a 5–10× reduction in `report.html` size and a
correspondingly faster first paint.

-->

---

# P3 — Observability, cost control, reproducibility

<!-- Closed 2026-09-18 — P3.1 fixed, see HANDOVER.md §2b (real `response.usage`
captured for Anthropic/OpenAI-compatible providers; `usage_summary()` reports
real token counts/cost) and Closed Ledger.

### P3.1 — No token, cost, or latency accounting

`RLMEngine` records per-call latency and 120-character snippets
(`src/rlm/engine.py:62-69`, `:141-150`), which is useful for a trace table
and nothing else. Nowhere does the system read `response.usage` from the
provider SDK, so there is no record of prompt tokens, completion tokens,
or spend — for a loop that can make 15+ calls with a linearly growing
prompt (P1.6), that is the one number an operator most wants.

**Fix:** capture `usage` in `LLMClient._dispatch` (all three branches
expose it), accumulate it on the engine's trace entries, expose
`RLMEngine.usage_summary()`, and surface tokens + estimated cost in the
report footer and the trace table. Add an optional `max_total_tokens`
budget that ends the loop gracefully via `_deterministic_final` rather
than by exhausting iterations.

*Raised in priority by Round 7's constraints:* with LLM-on as the primary mode
and a 15-iteration default loop, no token accounting is a financial risk, not
just an observability gap. Round 7's 7.12 carries it.

-->

### P3.2 — `print`-based diagnostics via Rich, no structured logging

The orchestration layer writes user-facing prose with emoji directly to a
module-level `Console` (`src/core/controller.py:38` and ~40 `console.print`
calls; same pattern in `src/core/memory.py:24`, `src/rlm/engine.py:12`).
There is no `logging` usage anywhere in `src/`. Consequences: output can't
be redirected, filtered by level, or captured as JSON lines; a failed run
leaves no artifact to diagnose from; and `scripts/validate.py:52-66` has to
stub out `rich` entirely just to import the modules under test.

**Fix:** `logging.getLogger(__name__)` for diagnostics, Rich only in the
presentation layer (`main.py`, `app.py`) via `RichHandler`. Write a
`run.log` alongside each run's outputs (pairs naturally with P2.4).

<!-- Closed 2026-09-18 — P3.3 fixed (removed, the unused-dependency option),
see HANDOVER.md §2 (`rlm/engine.py` + deps row: "P3.3 (removed unused dep)")
and Closed Ledger.

### P3.3 — `arize-phoenix` is a declared dependency with zero imports

`requirements.txt:44` pins `arize-phoenix>=3.0.0` under "Observability."
Grepping `src/`, `app.py`, `main.py` and `scripts/` for `phoenix`: no
hits. It is a large dependency tree (FastAPI, SQLAlchemy, Alembic,
OpenTelemetry — all visible in `.venv/Scripts/`) that every install and
every CI run pays for and nothing uses.

**Fix:** either wire it up — it is a genuinely good fit for P3.1, since
the RLM trace is already structured for it — or remove it. Do not leave it
declared and unused.

*Verified still open:* `arize-phoenix>=3.0.0` is in `requirements.txt`, and
`grep` for `phoenix` across `src/`, `app.py` and `main.py` returns nothing.

-->

<!-- Closed 2026-09-18 — P3.4 fixed, see HANDOVER.md §2 (deps + lockfile,
`requirements.lock` now in the repo) and Closed Ledger.

### P3.4 — No dependency lockfile, and `pyproject.toml` declares no dependencies at all

`pyproject.toml:6-12` has no `[project.dependencies]`; `requirements.txt`
is 100% `>=` constraints with no upper bounds and no lock. CI installs
whatever PyPI serves that morning across a 3.11/3.12 matrix
(`.github/workflows/ci.yml:29-31`).

This matters more here than in most projects because behaviour is
version-sensitive in ways the codebase already knows about: comments at
`src/tools/ml_pipeline.py:89` and `src/tools/statistical_analysis.py:114`
both work around pandas 3's `str` dtype, `n_init="auto"`
(`src/tools/ml_pipeline.py:407`) is sklearn-version-dependent, and
`src/tools/ml_pipeline.py:436-442` documents a real sklearn truthiness
trap. A silent minor-version bump can change a reported metric with no
test failure.

**Fix:** move the runtime list into `[project.dependencies]` with sensible
upper bounds, generate `requirements.lock` via `uv pip compile` or
`pip-tools`, install from the lock in CI, and keep the loose file for
development. Add a scheduled job that re-resolves and runs the suite, so
upstream drift surfaces as a PR rather than as a wrong number.

---

*Verified still open (stale — see closure note above):* no lockfile or
constraints file in the repo, and `pyproject.toml` still declares no
`dependencies`.

-->

---

# P4 — Test coverage

### P4.1 — Four tools ship with no tests at all

`tests/` has no `test_time_series.py`, `test_text_analysis.py`,
`test_geospatial.py` or `test_dimensionality.py`, though all four tools
are registered and reachable by the planner
(`src/core/controller.py:413-416`). AGENTS.md's tool contract requires
"at least one unit test in `tests/`" for every tool.

These four are the *most* likely to need tests: each auto-detects its own
input columns from heuristics
(`src/tools/time_series.py:35-55`, `src/tools/text_analysis.py:37-58`,
`src/tools/geospatial.py:31-54`) and each is gated by an `applies_to`
score that decides whether the planner ever sees it. A silent regression
in a detector makes the tool invisible rather than broken — the failure
mode no one notices.

**Fix:** one test per tool covering (a) the happy path on a small
synthetic frame, (b) `applies_to` returning 0.0 on unsuitable data and
1.0 on suitable, (c) the auto-detection path with the column omitted.

*Worse than logged:* it is **seven** untested tools now, not four —
`time_series`, `text_analysis`, `geospatial`, `dimensionality`,
`cohort_analysis`, `financial_analysis`, `workforce_analysis` (the last three
postdate this item). Overlaps 6.2; do them together inside Round 7's 7.11.

### P4.2 — No coverage measurement

Nothing in `pyproject.toml` or `ci.yml` measures coverage, so the gap in
P4.1 is invisible to CI and the next one will be too.

**Fix:** `pytest-cov` with `--cov=src --cov-report=term-missing`, and a
`--cov-fail-under` floor set just below today's actual number so it
ratchets up rather than blocking immediately.

---

# Appendix — Measurement method (carried-forward P1 items)

All timings from this machine (Windows 11, Python 3.13 in `.venv`), warm
OS file cache, single run each — treat them as order-of-magnitude, not
benchmarks.

- **Gates:** `ruff check .`, `mypy src/`, `pytest tests/ -q`,
  `python scripts/validate.py`, each timed end to end.
- **Pipeline table (P1):** synthetic frame, 50,000 rows × 16 columns
  (12 `np.random.normal` numerics, 2 categoricals at cardinality 5 and 20,
  1 hourly datetime, 1 binary target derived from `num_0` plus noise),
  seed 0, written to CSV and driven through the real tool classes via
  `BaseTool.run()` in pipeline order.
- **Per-model table (P1.2/P1.3):** first 20,000 rows of the cleaned frame
  — deliberately at the `do_tune` threshold
  (`src/tools/ml_pipeline.py:227`) — through `_prepare_features` /
  `_encode_target`, then each of `fit`, `cross_val_score` at `n_jobs=1`
  and `n_jobs=-1`, and `TrainModelTool._tune`, timed separately with the
  same `StratifiedKFold(5, shuffle=True, random_state=42)` and
  `f1_weighted` scorer the tool uses.
- **Call-site claims** ("never called", "no readers") are from
  `grep -rn` across `src/`, `app.py`, `main.py`, `scripts/` and `tests/`,
  excluding `.venv/`, `.py/` and `__pycache__/`.

Benchmark scripts were written to the session scratchpad and are not part
of the repository; the parameters above are sufficient to reproduce them.

---


---

# Closed ledger

One line per item removed from this file. The full original text of each is in
git history (`git log -p IMPROVEMENTS.md`).

| Item | Closed by | What changed |
| :--- | :--- | :--- |
| P0.1 / P0.5 / P0.6 | Round 4 | `Pipeline([("prep", ColumnTransformer), ("model", …)])` fit on the training fold only; saved models self-contained; one-hot for linear models, ordinal for trees |
| P0.2 | Round 2 | `cross_val_score` fits `X_train`/`y_train`, never the full frame |
| P0.3 | Round 2 | Splitter reads `is_time_series` / `panel_group_cols`; chronological and `GroupShuffleSplit` splits, shared with `evaluate_model` |
| P0.4 | Round 2 | Datetime columns expanded (year/month/day/dow/hour/is_weekend/days_since_min) instead of dropped |
| P0.7 | Round 2 | `_flag_unverified_claims` annotates any numeric literal in the synthesis that no tool result supports |
| P0.8 | Round 2 | An explicit `max_iterations` argument wins over the environment |
| P0.9 / P1.4 | Round 2 | `_step_cache` keyed on tool + resolved params + input mtime/size |
| P1.1 / P1.3 | Round 2 (decided, no action) | CSV reads are not the bottleneck; `n_jobs=1` kept — measured 4× faster than `n_jobs=-1` on 16 cores |
| P1.2 (part) | Round 2 | Redundant `cross_val_score` after tuning dropped; `cv_mean`/`cv_std` read from the search results |
| P1.5 | later session | `decompose_and_invoke` runs sub-tasks on a bounded `ThreadPoolExecutor` (`src/rlm/engine.py:197-201`) |
| P1.7 | Round 2 | SDK client built once, lazily, and reused across calls |
| P2.1 | Round 5 item 2 | All five `_read_df` copies are now thin wrappers over `src.core.io.read_any` |
| P2.3 (first half) | Round 2 | `clean_data` / `detect_outliers` declare `output_subdir = "data"` |
| P2.6 | Round 6 session | `BaseTool.default_params` lets a tool propose its own required parameters |
| P4.3 | Round 2 | Invariant tests for the anti-leakage / anti-overfitting behaviour |
| U0.1 / U0.2 / U1.1 | Round 5 item 2 | `src/core/io.py` — one reader with encoding detection, delimiter sniffing, duplicate-header detection and a `ReadReport` |
| U0.3 / U1.3 | Round 5 item 5 | ID-guard fixed; `is_sufficient` / `sufficiency_reason` and a row-count floor on the quality score |
| U0.4 | Round 5 item 4 | Effect sizes, confidence intervals, sample-size notes, practical-vs-statistical verdicts, Benjamini-Hochberg across a run |
| U0.5 | Round 5 item 7 | `profile_status` in memory; degraded mode stated in both reports and the UI |
| U0.6 | Round 5 item 6 | Markdown report gained a data overview, methodology (planner rationales) and limitations |
| U0.7 / U1.4 | Round 5 item 3 | `src/core/coercion.py` — currency/percent/thousands/bool/date repair, reported per column |
| U1.5 (part) | Round 5 item 9 | Read-once cache keyed on path+mtime+size |
| U1.6 (part) | Round 5 item 1 | `tests/fixtures/` + `tests/test_data_shapes.py` exist. **The finding itself is still open:** the suite is still organised by module, not by data shape — see 6.2 / P4.1, which Round 7's 7.11 closes |
| Round 5 item 10 | Round 5 | `src/core/degradations.py` — structured degradation log, rendered in both reports |
| Round 6 bugs (4) | Round 6 (`1bcb739`) | dd/mm/yyyy convention detection; `_detect_target_leakage`; `_TREND_MIN_R_SQUARED`; sequential domain-role claiming |
| Round 3 item 2 | Round 3 | `_render_other_findings` in `app.py`; the time-series chart reads the tool's own columns and findings |
| Round 2 §0 | Round 2 | Stale tests re-pointed at the Ledger palette; `_parse_steps` calls bound to an instance |
| Q1 | Round 7 session (2026-09-18) | `LEDGER_TOKENS_DAY/NIGHT` imported by the two landing tests instead of re-typed hex literals |
| R3.1 | Round 7 session | `generate_visualizations` excluded from the deterministic fallback sweep |
| 6.1 | Round 7 session | Cohort/financial/workforce chart panels added to `dashboard.py` |
| 6.3 | Round 7 session | Explicit `date_ambiguous` warning in `degradations.py` |
| 6.4 | Round 7 session | Distribution-aware outlier detection (skew/flag-aware); 39.47% → 0.67% on the churn fixture |
| U1.2 | Round 7 session | `io.py` — JSON/JSONL/Parquet/`.gz`/`.zip`, depth-capped `json_normalize` flatten |
| U1.5 | Round 7 session | `io.py` — configurable row cap + reported reservoir sampling, one choke point |
| P1.2 (remainder) | Round 7 session | `tune_hyperparameters` default flipped off; successive-halving/adaptive-budget sub-fixes carried forward as 7.13 |
| P1.6 | Round 7 session | `get_results_summary_digest()` — latest iteration in full, one-line digest for older ones; Anthropic system prompt marked `cache_control` |
| P2.2 | Round 7 session | AGENTS.md's Layer Rules table corrected to match reality; `tests/test_architecture.py` added as an AST-based drift check |
| P2.3 (second half) | Round 7 session | Every tool write routed through `resolve_output_path()` in `BaseTool.prepare_params()` |
| P2.4 | Round 7 session | Session-scoped output directory default + `output/latest.txt` |
| P2.7 | Round 7 session | Dashboard specs pre-aggregate (bins, five-number summaries, resampled series) instead of inlining raw rows |
| P3.1 | Round 7 session | Real `response.usage` captured for Anthropic/OpenAI-compatible providers; `usage_summary()` reports real tokens/cost |
| P3.3 | Round 7 session | Unused `arize-phoenix` dependency removed |
| P3.4 | Round 7 session | `requirements.lock` generated; runtime deps given upper bounds |
