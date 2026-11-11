# Expectations vs. Reality Remediation Plan

**Plan date:** 2026-10-16  
**Audit source:** `docs/EXPECTATIONS_VS_REALITY_AUDIT_2026-10-14.md`  
**Verified baseline revision:** `6f7bff1f83ae4e77a7747c6f01294c9e960575af`  
**Scope:** EVR-01 through EVR-23  
**Primary objective:** make Sorrel safe and internally honest as local analytical
decision support before expanding its interface surface.

## 1. Outcome and ordering principles

The work should be delivered in this order:

1. Stop unsafe recommendations and correct public claims immediately.
2. Establish one typed evidence-to-recommendation contract.
3. Make audits gate synthesis rather than annotate an already-published answer.
4. Make statuses and artifacts mechanically truthful and strictly serializable.
5. Restore the full quality gate and architecture/security boundaries.
6. Improve objective-led planning only after explicit approval.
7. Treat REST/MCP as a separate product epic, not as incidental audit cleanup.

The system is not decision-ready until Phases 0-4 and their release gates are
complete. Phase 5 improves relevance and cost but is approval-gated. Phase 6 is
specification consolidation. Phase 7 is optional product expansion.

## 2. Decisions and constraints

### 2.1 Decisions adopted by this plan

- **The canonical workflow remains seven stages.** That is the current
  `AGENTS.md` contract. The older five-stage direction in `HANDOVER.md` is stale
  and must not drive an engine rewrite. A future workflow change requires a
  separate approved architecture decision.
- **Public REST/MCP pages become roadmap material immediately.** Implementing a
  secure asynchronous multi-tenant service is too large to mix with analytical
  correctness remediation.
- **Public run status is separate from the LLM/controller protocol.** Internal
  planning replies may retain `status="complete"`; public artifacts receive a
  distinct typed terminal status. This avoids conflating orchestration messages
  with run outcomes.
- **Ambiguous scenario targets are non-prescriptive.** They default to
  `sensitivity_only`; there is no implicit maximize fallback.
- **Observational evidence cannot produce direct action language.** It may
  produce a monitoring signal, hypothesis, or pilot proposal. Only identified
  causal or experimental evidence can produce a `causal_action`.
- **Failed verification quarantines a claim.** `[unverified]` is not permitted
  in a primary insight or recommendation.
- **Audit semantics are corrected before being strengthened.** Rename
  `audit_integrity_ok` to the accurate `audit_write_ok` now; cryptographic/WORM
  guarantees are a separate enhancement.
- **Model-signing secrets are operator state, not artifacts.** They must be
  stored outside model/export/run directories. Asymmetric signing can be a
  later deployment option.

### 2.2 Approval gates

The following work must not start without explicit user approval:

1. Changes to task decomposition or deterministic planning logic, including
   EVR-09.
2. Any change to `src/rlm/engine.py`, including work proposed for EVR-15.
3. Any `MemorySystem` schema change. Prefer a final-result contract that does
   not require one.
4. Adding a new execution tool. This plan does not require a new tool.
5. Building the advertised REST API/MCP service rather than marking it as
   roadmap.

### 2.3 Delivery rules

- Keep reasoning/orchestration in `src/core/` and execution in `src/tools/`.
- Controllers continue to call tools only through `ToolRegistry` and
  `BaseTool.run()`.
- Every changed or new function is fully typed.
- Every new utility or contract receives unit tests.
- Preserve backward compatibility only through explicit adapters; do not keep
  ambiguous fields with two meanings.
- Use deterministic seeds (`random_state=42`) in all statistical/model tests.
- Do not edit or delete the source audit document.

## 3. Target contracts

These contracts should be agreed and tested before broad refactoring.

### 3.1 Objective contract

Every scenario result carries:

```text
target
target_goal: minimize | maximize | hold | sensitivity_only
goal_source: user | domain_policy | inferred_ambiguous
goal_rationale
actionable_levers[]
excluded_levers[{column, reason}]
model_family
bounded_target: bool
```

Rules:

- Explicit user direction wins unless it violates a hard validation rule.
- Known adverse targets (for example churn, fraud, default, mortality, failure,
  emissions, and named pollutants) map to `minimize`.
- Known beneficial targets (for example revenue or yield) may map to
  `maximize` only with a recorded policy match.
- Unknown or mixed-semantics targets map to `sensitivity_only`.
- `hold` reports feasible stability ranges and never an optimum.
- Binary targets use bounded probability estimates; an unconstrained linear
  projection cannot be labelled optimization.
- Non-controllable, post-outcome, identifier, timestamp, sensor-reading, text
  token, highly collinear, or target-leaking columns cannot be levers.

### 3.2 Evidence and recommendation contract

Every finding receives a machine-readable disposition:

```text
evidence_tier: observation | hypothesis | pilot | causal_action
primary_eligible: bool
eligibility_reasons[]
checks: {
  claim_verified,
  robust,
  leakage_free,
  paradox_free,
  causal_language_safe,
  model_converged
}
```

Every primary recommendation is a structured object rather than an untracked
string:

```text
recommendation_id
finding_ids[]
disposition: monitor | investigate | pilot | act
action
rationale
claims[{metric, value, unit, derivation, evidence_path, verified}]
limitations[]
```

Eligibility policy:

- `act` requires identified causal/experimental evidence, controllability, and
  all mandatory checks passing.
- `pilot` may use strong observational evidence but must say what would be
  tested and how success would be measured.
- `monitor` and `investigate` are the only allowed primary dispositions for
  descriptive associations, anomalies, sensors, timestamps, tokens, and
  ambiguous scenarios.
- A failed mandatory check sets `primary_eligible=false`; the item remains only
  in limitations, watch, or audit surfaces.

### 3.3 Public terminal-status contract

Use a dedicated enum for public artifacts:

| Status | Meaning |
| --- | --- |
| `succeeded` | Required work completed; no material warning or failed required step. |
| `succeeded_with_warnings` | Result is usable with explicitly surfaced quality, coverage, convergence, or numerical warnings. |
| `partial` | One or more objective-critical steps failed or the run produced only partial evidence. |
| `failed` | No valid final result can be produced. |
| `cancelled` | User or operator stopped the run. |

Every public artifact also includes `execution_issues`, `coverage`,
`verification_summary`, and `schema_version`. Coverage remains a documented
heuristic and is never used alone to claim literal completeness.

## 4. Phased implementation plan

## Phase 0 — Containment and baseline

**Purpose:** prevent fresh unsafe output while structural remediation is in
progress.

### WP0.1 — Freeze reproducible fixtures

**Covers:** all EVRs indirectly.  
**Changes:**

- Preserve the churn and air-quality commands from the audit as golden-run
  scripts or parametrized integration fixtures using temporary output paths.
- Add compact deterministic fixtures for fraud/default, mortality/failure,
  revenue, ambiguous target, text-only, chronological, and grouped/panel data.
- Capture the current full-suite failures in a machine-readable baseline; do
  not bless them as expected failures.

**Tests/exit:** each fixture has a declared target direction, expected evidence
tier, and prohibited recommendation language. The baseline is reproducible on
the supported Python/OS matrix.

### WP0.2 — Immediate recommendation kill switches

**Covers:** EVR-01, EVR-02, EVR-03, EVR-04.  
**Files:** `src/tools/scenario_simulation.py`, `src/core/findings.py`,
`src/core/controller_report.py`, related tests.  
**Changes:**

- Change the scenario default to `sensitivity_only` and expose `target_goal` in
  `get_schema()`.
- Until the full evidence contract lands, exclude every scenario,
  observational association, failed-robustness finding, and unverified claim
  from Quick Wins, Strategic Bets, and primary recommendations.
- Keep quarantined items visible in an explicitly non-actionable audit/watch
  section.

**Tests/exit:** churn and pollution cannot recommend an increase; dates,
sensors, tokens, anomalies, and observational segment gaps cannot become Quick
Wins; primary reports contain no `[unverified]` recommendation.

### WP0.3 — Correct public discovery claims

**Covers:** EVR-05, EVR-17, EVR-19, EVR-20, EVR-23.  
**Files:** `docs/public/*`, `README.md`, `public/.well-known/security.txt`,
landing documentation registry, discovery tests.  
**Changes:**

- Label API, MCP, result-schema, tenancy, quotas, and retention pages as
  unshipped roadmap proposals or remove them from released discovery indexes.
- Replace “immutable audit trail” with the actual mutable local JSONL guarantee.
- Remove localhost and workstation-local canonical/security URLs.
- Describe the current seven-stage workflow consistently.
- State actual runtime limits until a single typed source is implemented.

**Tests/exit:** the public-surface checker fails when an unimplemented interface
is described as available; no public artifact contains `file:///`, a drive
path, or a localhost canonical URL.

## Phase 1 — Evidence-safe synthesis

**Purpose:** create one post-audit source of truth for every user-facing surface.

### WP1.1 — Implement the objective contract

**Covers:** EVR-01.  
**Files:** `src/tools/scenario_simulation.py`, `src/core/findings.py` or another
architecture-approved neutral contract module, schema/provenance renderers.  
**Changes:**

- Add the typed goal values and validation rules in Section 3.1.
- Add a conservative, testable domain policy for common adverse and beneficial
  targets; unmatched targets remain sensitivity-only.
- Replace binary-target least-squares optimization with bounded probability
  sensitivity, or refuse prescription when the model is unsuitable.
- Filter candidate levers using explicit controllability metadata, temporal
  ordering, leakage rules, and collinearity diagnostics.
- Persist direction, source, rationale, exclusions, and model family in the
  finding and report provenance.

**Tests/exit:** golden tests cover churn, default, fraud, failure, mortality,
pollution, revenue, and an ambiguous target. Direction is present in the tool
schema and every output surface.

### WP1.2 — Add evidence maturity and actionability gates

**Covers:** EVR-02.  
**Files:** `src/core/findings.py`, `src/core/causal_design.py`,
`src/core/controller_report.py`, action-matrix/report tests.  
**Changes:**

- Replace name-only actionability inference with explicit evidence tier plus
  controllability and temporal-order metadata.
- Remove the rule that treats observational caveats as benign for action
  promotion.
- Generate `monitor`, `investigate`, or `pilot` language for non-causal
  findings; reserve direct intervention verbs for `causal_action`.
- Make the positive-control path an RCT or a validated identified causal design.

**Tests/exit:** negative controls include tokens, dates, sensors, anomalies,
unadjusted segment gaps, and observational scenarios; positive controls include
an RCT and a validated difference-in-differences design.

### WP1.3 — Refactor finalization into audit then project then render

**Covers:** EVR-03, EVR-04, EVR-07, EVR-08, parts of EVR-12/13.  
**Files:** `src/core/controller.py`, `src/core/controller_report.py`,
`src/core/claim_verification.py`, report/dashboard builders.  
**Changes:**

1. Collect candidate findings and execution issues.
2. Run dependence, leakage, robustness, paradox, causal-language, convergence,
   and claim-verification checks.
3. Compute eligibility and quarantine failed items.
4. Rank the audited projection exactly once.
5. Build insights, recommendations, action matrix, summary, dashboard, Markdown,
   HTML, and JSON only from that projection.
6. Run a final cross-surface contract audit without mutating the result.

Create one typed projection object or pure function rather than allowing each
renderer to reinterpret raw findings. Do not change task decomposition in this
work package.

**Tests/exit:** an injected high-effect fragile finding is absent from every
primary surface and present in limitations/watch; a deliberately unsupported
number is absent from primary prose and recorded in audit metadata; all
surfaces agree on finding eligibility and rank.

### WP1.4 — Make claim provenance structural

**Covers:** EVR-04.  
**Files:** `src/core/claim_verification.py`, recommendation synthesis and report
renderers.  
**Changes:**

- Represent numeric recommendation claims with value, unit, derivation, source
  finding, and evidence path before prose rendering.
- For binary outcomes, carry base rate, comparison rate, absolute percentage
  point change, relative risk/lift, and sample sizes as distinct fields.
- Verify structured claims, render only verified values, and retain rejected
  claims only in audit metadata.

**Tests/exit:** rounding is deterministic; absolute and relative effects cannot
be confused; release-mode primary content contains zero unverified claims.

## Phase 2 — Truthful status and artifact integrity

### WP2.1 — Public result envelope and status resolver

**Covers:** EVR-07.  
**Files:** final-result assembly, summary/dashboard/report renderers, UI status
mapping, result contract tests.  
**Changes:**

- Introduce the public status enum from Section 3.3 without changing internal
  LLM completion messages.
- Resolve status from cancellation, required-step failures, usable findings,
  convergence/numerical warnings, verification failures, and documented
  objective coverage.
- Add structured issues with `code`, `severity`, `stage`, `tool`, `message`, and
  `finding_ids`.
- Put the same status, issue counts, and top objective-critical coverage gaps in
  raw result, `summary.json`, dashboard, Markdown, and HTML.
- Rename “failed runs” governance metrics to explicitly say sandbox/code
  execution failures where that is what they count.

**Tests/exit:** stopped runs are `cancelled`; injected required-step failures
are `partial`; usable runs with convergence/numerical warnings are
`succeeded_with_warnings`; clean runs are `succeeded`.

### WP2.2 — One strict JSON boundary

**Covers:** EVR-11.  
**Files:** a shared core serialization utility plus every public JSON writer.  
**Changes:**

- Add a typed recursive normalizer for mappings, sequences, NumPy/pandas
  scalars, datetimes, nullable values, and non-finite floats.
- Normalize non-finite values to `null` and add a data-quality issue when the
  value is analytically meaningful.
- Serialize every public artifact with `allow_nan=False`.
- Prohibit renderer-local `default=str` fallbacks for public contracts.

**Tests/exit:** nested `NaN`, positive/negative infinity, NumPy scalars,
timestamps, and pandas nullable dtypes round-trip through a strict parser.

### WP2.3 — Correct ingestion population accounting

**Covers:** EVR-10.  
**Files:** `src/core/io.py`, `ReadReport`, ingestion/profile tests.  
**Changes:**

- After retaining data columns, drop rows for which every retained value is
  null.
- Record the removed-row count and reason in `ReadReport`.
- Ensure profiling, coverage, report denominators, and downstream tools use the
  cleaned population.

**Tests/exit:** CSV and TSV fixtures with more than 15 delimiter-only trailing
rows are corrected; partially empty rows remain; AirQualityUCI reports 9,357
real records.

### WP2.4 — Chart roles and ranking policy

**Covers:** EVR-12, EVR-13.  
**Files:** `src/core/dashboard_common.py`, dashboard builders, finding
serialization, README/schema tests.  
**Changes:**

- Require every chart to have either `chart_role="evidence"` plus `finding_id`,
  or `chart_role="context"` plus a stable `context_id`.
- Add `display_rank`, `raw_importance_rank`, and `ranking_policy` to findings,
  or expose separate scored and curated lists. Prefer explicit ranks to hidden
  reordering.

**Tests/exit:** zero orphan charts; ranks are deterministic; machine consumers
can reconstruct both importance order and curated display order.

### WP2.5 — Plain-language correctness

**Covers:** EVR-18.  
**Files:** `src/core/plain_language.py`, Markdown/HTML rendering tests.  
**Changes:** preserve surrounding delimiters during confidence-interval
rewrites and make adjusted-p phrasing conditional on available sentence
context.

**Tests/exit:** exact sentence snapshots cover bracketed, parenthesized, and
bare intervals plus adjusted and unadjusted p-values; Markdown and HTML have
balanced punctuation.

## Phase 3 — Model validity and deterministic quality

### WP3.1 — Candidate-specific model diagnostics

**Covers:** EVR-06, EVR-08.  
**Files:** `src/tools/ml_pipeline.py`, `src/tools/ml_common.py`, result synthesis,
ML/UI tests.  
**Changes:**

- Capture fit and cross-validation warnings per candidate.
- Persist `converged`, `n_iter`, warning category/message, retry history, and
  selected-model status.
- Retry logistic regression with scaling, a suitable solver, and a bounded
  higher iteration limit; downgrade an unresolved candidate.
- Generate recommendations for the selected model only unless the output is an
  explicit candidate comparison.
- Keep overfit warnings keyed by model; never translate a rejected tree model's
  warning into advice about selected logistic regression.

**Tests/exit:** a non-selected overfit candidate does not trigger the primary
warning; an unconverged selected candidate produces a visible downgrade and
`succeeded_with_warnings` or selects a sound alternative.

### WP3.2 — Enforce split/fold policy

**Covers:** EVR-06.  
**Files:** `src/tools/ml_common.py`, `src/tools/ml_pipeline.py`, validator and
schema tests.  
**Changes:**

- Enforce five folds for i.i.d. classification/regression when data permits.
- Keep `TimeSeriesSplit` for chronological data and `GroupKFold` for panels.
- Return a structured exception rationale when group/time/sample constraints
  force fewer folds.
- Give `n_cv_folds` a schema minimum consistent with the normal contract.
- Relabel `scripts/validate.py` as smoke validation and align its overfit
  assertion with the product threshold (`0.10`).

**Tests/exit:** split strategy, fold count, leakage protection, and exception
rationale are verified for i.i.d., time-series, and grouped fixtures.

### WP3.3 — Restore the test and dependency contract

**Covers:** EVR-14.  
**Files:** affected tools/tests, `pyproject.toml`, `requirements.txt`, lockfile,
CI matrix.  
**Changes:**

- Decide and document canonical field names for multi-seasonal, ranking, and
  threshold outputs; use explicit deprecated aliases only if compatibility is
  required. Do not silently swallow unknown aliases through `**_`.
- Update summary-row detection for pandas `StringDtype`.
- Narrow the placeholder test to sentinel values in evidence/source fields.
- Bisect suite order and isolate the mixed-model global-state leak; add a test
  that runs the trigger sequence in one process.
- Align pytest constraints and regenerate the supported lock/environment.

**Tests/exit:** the complete suite passes in one process and in randomized
order; supported OS/Python jobs use the same declared dependency policy.

## Phase 4 — Architecture and security boundaries

### WP4.1 — Expand architecture enforcement

**Covers:** EVR-15.  
**Files:** `tests/test_architecture.py`, controller/report/agenda/dashboard
dependencies.  
**Changes:**

- Add AST checks for both import directions, direct LLM-client calls from
  controllers, core-to-tool behavior imports, and private cross-layer helpers.
- Route chart reasoning through the injected RLM abstraction without directly
  calling `llm_client.call()`.
- Move applicability and dashboard helpers to neutral contracts or access them
  through `ToolRegistry`.
- Any required edit to `src/rlm/engine.py` is split into a separately approved
  change.

**Tests/exit:** deliberately planted forbidden imports/calls fail architecture
tests; controller uses no direct tool import or raw LLM call.

### WP4.2 — Remove signing secrets from artifacts

**Covers:** EVR-16.  
**Files:** `src/core/model_io.py`, export/retention/download code, tests.  
**Changes:**

- Load or create the HMAC key in an operator-owned configuration directory
  outside run and artifact roots; fail safely when a verification key is
  unavailable.
- Never include dotfiles, key material, or undeclared files in artifact
  manifests/downloads.
- Define migration behavior for already-signed local models; never silently
  accept unsigned or unverifiable pickles.

**Tests/exit:** exported/retained bundles contain no key; possession of a bundle
alone cannot create a valid replacement model; invalid signatures are rejected
before unpickling.

### WP4.3 — Make audit guarantees accurate

**Covers:** EVR-17.  
**Files:** `src/core/governance.py`, result/report fields, public docs/tests.  
**Changes:**

- Rename the current flag to `audit_write_ok` and report separate code/LLM audit
  paths and event counts.
- Emit a run manifest even when there are zero events.
- Record observed memory/RSS only if it is actually measured.
- Design hash chaining, signed manifests, and durable/WORM storage as a later
  deployment hardening proposal, not as an implied current property.

**Tests/exit:** zero-event, successful-write, and failed-write cases have honest
semantics; no field named “integrity” exists without verification.

### WP4.4 — Frontend message and secret hygiene

**Covers:** EVR-21, EVR-22, EVR-23.  
**Files:** landing component, `.gitignore`, CI/pre-commit configuration, public
assets/tests.  
**Changes:**

- Validate `event.source === window.parent`, expected origin, and the complete
  `postMessage` schema.
- Expand ignore rules for `.env.*`, Streamlit secrets, private keys, and model
  keys while explicitly allowing safe examples.
- Add a tracked-secret scanner and non-local-origin public-build check.

**Tests/exit:** hostile sibling/iframe messages are ignored; secret fixtures are
detected; `.env.example` remains allowed; public builds contain no local path or
localhost canonical URL.

## Phase 5 — Objective-led planning (approval required)

### WP5.1 — Question-first tool selection

**Covers:** EVR-09.  
**Files:** controller planning/decomposition modules, agenda/question router,
tool registry contracts, planning tests.  
**Precondition:** explicit user approval because this changes task
decomposition.

**Changes:**

- Score a tool only after hard prerequisites pass.
- Rank candidates by objective/question fit, expected decision value, evidence
  gap closed, cost, and redundancy—not generic applicability alone.
- Give each objective-critical question a tool/time budget.
- Stop when declared critical coverage and evidence-quality thresholds are met,
  or when further tools have low expected value.
- Prevent numeric-only specialists on text-only data and prevent generic
  scenario/threshold analyses without objective fit.

**Tests/exit:** churn and air plans omit irrelevant specialists; text-only data
does not schedule numeric tests/clustering; identical inputs produce identical
fallback plans; stopping rationale is recorded.

## Phase 6 — One source for product specifications

### WP6.1 — Canonical workflow specification

**Covers:** EVR-19.  
**Changes:** define the seven stage identifiers, names, descriptions, and entry
points once, then generate or validate README, public docs, UI, events, report
labels, and docstrings from it. A five-phase explainer is allowed only when
labelled as macro phases and mapped to all seven canonical stages.

**Tests/exit:** a consistency test fails on missing, reordered, or renamed
stages across every shipped surface.

### WP6.2 — Typed runtime limits and rendered documentation

**Covers:** EVR-20.  
**Changes:** define upload bytes, expansion ratio, uncompressed bytes, row and
column caps, sandbox timeout/memory, run duration, and retention behavior in one
typed configuration source. Runtime reads it; documentation is rendered from
or tested against it. Do not document retention, tenancy, quotas, symlink
handling, or archive rules until implemented and tested.

**Tests/exit:** boundary tests cover each limit immediately below, at, and above
the threshold; runtime, README, public docs, and environment-variable help agree.

## Phase 7 — Optional REST/MCP product epic

**Covers:** long-term EVR-05.  
**Precondition:** explicit product decision and threat model.  
**Recommendation:** defer until Phases 0-6 are green.

If approved, begin with a versioned adapter over the existing controller and a
formal schema. The epic must include authentication, authorization, tenant
isolation, upload lifecycle, idempotency, cancellation, SSE, retention jobs,
artifact authorization, quotas/rate limiting, MCP transport, and black-box
contract tests. Do not publish a partial endpoint under the current promises.

**Exit:** every documented REST route and MCP tool passes black-box tests and
schema validation, including cross-tenant denial, cancellation, expiry, and
artifact authorization. Otherwise the roadmap labeling remains.

## 5. Proposed change-set sequence

Each row should be a reviewable PR/change set. Do not combine later structural
work into the immediate containment patches.

| Order | Change set | Depends on | Approval | Primary EVRs |
| ---: | --- | --- | --- | --- |
| 1 | Golden fixtures and baseline | none | no | all |
| 2 | Scenario/recommendation kill switches | 1 | no | 01-04 |
| 3 | Public truth correction | none | no | 05, 17, 19, 20, 23 |
| 4 | Objective contract | 1-2 | no | 01 |
| 5 | Evidence/actionability contract | 1-2 | no | 02 |
| 6 | Audit-first final projection | 4-5 | no | 03-04, 08, 12-13 |
| 7 | Public status/result envelope | 6 | no | 07 |
| 8 | Strict JSON and ingestion fixes | 1 | no | 10-11 |
| 9 | Chart/rank/grammar fixes | 6 | no | 12-13, 18 |
| 10 | Model diagnostics and CV policy | 1, 7 | no | 06, 08 |
| 11 | Test-contract and dependency recovery | 1 | no | 14 |
| 12 | Architecture enforcement | 6 | engine edit: ask first | 15 |
| 13 | Signing, audit, frontend/secret hardening | none | no | 16-17, 21-23 |
| 14 | Objective-led planning | 4-7 | **yes** | 09 |
| 15 | Canonical specs and limits | 3, 7-13 | no | 19-20 |
| 16 | REST/MCP implementation | all prior | **yes** | 05 |

Change sets 4-7 form one conceptual migration but should remain separately
reviewable. During that migration, compatibility adapters may read old finding
dictionaries; all newly written artifacts must use the new schema.

## 6. Test strategy

### 6.1 Unit and contract tests

- Objective classification and ambiguity fallback.
- Evidence-tier/actionability matrix.
- Structured claim derivation and verification.
- Eligibility projection and cross-surface identity.
- Public status resolution.
- Strict JSON normalization.
- All-null row accounting.
- Candidate convergence/overfit attribution and split selection.
- Chart role and ranking serialization.
- Audit/write semantics and artifact allowlisting.
- Architecture AST rules, origin checks, and public-spec consistency.

### 6.2 Golden end-to-end matrix

| Fixture | Required assertion |
| --- | --- |
| Customer churn | Churn is minimized or sensitivity-only; no fragile/unverified primary claim. |
| Air quality | Pollutants are not maximized; row count is 9,357. |
| Fraud/default/failure/mortality | Adverse target never increases by default. |
| Revenue | Explicit policy may maximize; direction provenance is present. |
| Ambiguous target | No optimum or direct action is emitted. |
| Text-only | No numeric-only specialist; tokens are not levers. |
| Time series | `TimeSeriesSplit`; no shuffled leakage. |
| Grouped/panel | `GroupKFold`; entity leakage is blocked. |
| Non-finite nested output | Every JSON artifact parses with a strict parser. |
| Partial tool failure | Status and issue are present on every surface. |
| Fragile high-effect finding | Watch/limitation only; absent from primary recommendation. |
| Valid causal design | Positive-control `causal_action` path works. |

### 6.3 Mandatory gates for every change set

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe src\
.\.venv\Scripts\python.exe -m pytest tests\ -v
.\.venv\Scripts\python.exe scripts\validate.py
.\.venv\Scripts\python.exe scripts\check_ai_surface.py
```

In addition, run the golden churn and air-quality analyses after any change to
scenario logic, findings, audits, synthesis, status, reporting, or ingestion.
The release gate is based on one-process full-suite results, not isolated test
files.

## 7. Release gates

### Gate A — Safe local exploratory release

All must pass:

1. No implicit adverse-outcome maximization.
2. No observational association presented as a direct action.
3. No unverified or failed-audit primary claim.
4. All renderers derive from the same post-audit projection.
5. Public docs describe only shipped local capabilities.
6. Full lint, type, unit, validator, and golden-run gates are green.

### Gate B — Trustworthy decision-support release

Gate A plus:

1. Public statuses and issues truthfully represent partial work and warnings.
2. All public JSON validates strictly against one versioned schema.
3. Selected models expose convergence and model-specific diagnostics.
4. Every chart and displayed rank has explicit provenance/policy.
5. Exported models contain no signing secret and audit terminology matches the
   implemented guarantee.
6. Objective-led planning has been approved, implemented, and validated, or the
   product is explicitly scoped as analyst-guided exploration.

### Gate C — REST/MCP release

Gate B plus black-box conformance for every advertised API/MCP, authentication,
tenant isolation, lifecycle, retention, quota, cancellation, SSE, and artifact
authorization behavior.

## 8. Risk register and rollback

| Risk | Mitigation | Rollback |
| --- | --- | --- |
| New evidence gates suppress too much | Positive controls and eligibility reason codes; measure quarantine rates. | Disable individual gate by versioned policy, never restore unsafe default. |
| Status rename breaks UI/tests | Separate internal and public status types; adapter at artifact boundary. | Revert adapter while retaining issue metadata. |
| Finding schema migration is broad | Pure projection and compatibility reader; version all new artifacts. | Continue reading v1; stop writing v1. |
| Model-key relocation invalidates old local models | Explicit migration/import command or documented re-train path; never bypass verification. | Restore old model directory read-only for migration, not export. |
| Audit-first refactor changes ordering | Snapshot raw rank and display rank separately; deterministic tests. | Revert projection change set as one unit. |
| Planner relevance change reduces coverage | Approval gate, shadow planning comparison, fixed budgets and golden fixtures. | Keep current planner behind a configuration flag during evaluation. |
| API/MCP scope crowds out correctness | Keep it as Phase 7 with a separate go/no-go. | Retain roadmap-only docs. |

## 9. Completion definition

The audit can be closed only when:

- every EVR has a linked change set, test, and artifact demonstrating closure;
- the entire quality gate passes in one supported environment and CI matrix;
- churn, air-quality, text-only, time-series, grouped, non-finite, fragile, and
  partial-failure golden runs pass;
- no release surface overstates causality, completion, verification, audit
  integrity, retention, tenancy, or interface availability; and
- approval-gated items are either explicitly approved and complete or clearly
  documented as deferred scope with no contradictory product claim.

## 10. EVR traceability matrix

| EVR | Primary work package(s) | Closure evidence |
| --- | --- | --- |
| 01 | WP0.2, WP1.1 | Direction golden tests and provenance |
| 02 | WP0.2, WP1.2 | Negative/positive actionability controls |
| 03 | WP1.3 | Fragile-finding cross-surface test |
| 04 | WP1.3, WP1.4 | Zero unverified primary claims |
| 05 | WP0.3, Phase 7 | Roadmap truth or black-box implementation |
| 06 | WP3.1, WP3.2 | Convergence and fold tests |
| 07 | WP2.1 | Status/issue cross-surface tests |
| 08 | WP3.1 | Selected-model warning test |
| 09 | WP5.1 | Objective-fit planning golden tests |
| 10 | WP2.3 | Air-quality and delimiter-row tests |
| 11 | WP2.2 | Strict JSON test matrix |
| 12 | WP2.4 | Zero orphan charts |
| 13 | WP2.4 | Explicit deterministic ranks/policy |
| 14 | WP3.3 | Full suite green in one process/random order |
| 15 | WP4.1 | Bidirectional AST architecture tests |
| 16 | WP4.2 | Secret-free export/forgery tests |
| 17 | WP0.3, WP4.3 | Honest audit flags and zero-event manifest |
| 18 | WP2.5 | Sentence and renderer snapshots |
| 19 | WP0.3, WP6.1 | Canonical seven-stage consistency test |
| 20 | WP0.3, WP6.2 | Runtime/doc boundary tests |
| 21 | WP4.4 | Hostile-message tests |
| 22 | WP4.4 | Secret scanner/ignore tests |
| 23 | WP0.3, WP4.4 | Non-local public build validation |
