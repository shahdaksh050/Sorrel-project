# Expectations vs. Reality Audit

**Date:** 2026-10-14  
**Repository revision:** `6f7bff1f83ae4e77a7747c6f01294c9e960575af` (`Improved Quality`)  
**Scope:** source code, current saved outputs, fresh temporary outputs, analytical semantics, report contracts, architecture, tests, and security posture  
**Assessment:** **not ready to be presented as trustworthy decision support or as the documented REST/MCP product**. It is usable as a local exploratory-analysis prototype if its recommendations are treated as hypotheses rather than actions.

## Executive verdict

The repository can ingest data, execute the seven-stage local workflow, train models, and generate populated Markdown/HTML/JSON reports. Lint and type checks pass, both fresh end-to-end runs completed without an uncaught exception, and the standalone workflow validator passed 68/68 checks.

The largest gaps are above the execution layer:

1. Scenario simulation defaults to **maximizing** any target, including adverse outcomes. Fresh runs presented as “optimization”/top findings projections of `+54.5%` churn and `+39.7%` carbon-monoxide pollution.
2. Observational associations and linear sensitivities can be promoted to “Quick Wins,” “Strategic Bets,” and intervention recommendations without causal or controllability evidence.
3. Robustness and claim-verification audits run after primary synthesis, but their failures do not rebuild or quarantine the primary insights and recommendations.
4. Public documentation advertises an authenticated asynchronous REST API, an MCP server, tenant isolation, retention controls, and a stable result schema for which no implementation or conformance adapter exists in the inspected repository.
5. The required full test gate fails: `1771 passed, 8 failed`. Four failures are deterministic product/test contract drift, one is a false-positive test, and three are suite-order/global-state-sensitive mixed-model failures whose root cause remains unisolated.

No confirmed remote code-execution, credential leak, or DOM-XSS vulnerability was found. There are nevertheless two medium security-design gaps: model-signing keys are exported with the signed models, and the documented “immutable” audit trail is a mutable JSONL write log whose `audit_integrity_ok` flag only means that writes did not fail.

## Audit method and evidence base

The review used the installed `security-best-practices` skill for Python/web security triage and then independently checked the findings against source and runtime artifacts.

Evidence inspected:

- Current saved artifacts under `output/runs/`: 8 non-empty run summaries/raw reports, their Markdown/HTML reports, dashboards, models, and execution logs.
- Fresh churn run under `C:\tmp\dsa-agent-audit-6f7bff1-churn`.
- Fresh air-quality run under `C:\tmp\dsa-agent-audit-6f7bff1-air`.
- A direct non-finite JSON reproduction under `C:\tmp\dsa-json-nan-repro`.
- Source, tests, public documentation, README, UI, and dependency declarations at the revision above.
- Independent read-only cross-checks of the high-severity analytical, architecture, presentation, output-disclosure, and security findings.

The `output/` tree was modified by another process during the audit. All saved-output counts below refer to the final observed snapshot; the two named `C:\tmp` runs are isolated fresh reproductions. No product source was changed by this audit.

## Validation results

| Check | Result | Interpretation |
| --- | ---: | --- |
| `ruff check .` | Pass | Zero lint violations. |
| `mypy src/` | Pass | Zero type errors across 111 source files. |
| `pytest tests/ -v` | **Fail** | `1771 passed, 8 failed, 1 deselected`; see EVR-14. |
| Focused rerun of failing files | Partial | 14 passed, 5 failed; all 3 mixed-model tests passed in isolation. |
| `scripts/validate.py` | Pass | 68/68 seven-stage dry-run checks passed and a populated report was produced. |
| `scripts/check_ai_surface.py` | Pass | Documentation/link presence passed; this checker does not test advertised API/MCP implementations. |
| Fresh churn CLI run | Pass mechanically | Exit 0, 14 successful tools, populated reports; analytical safety failures remain. |
| Fresh air-quality CLI run | Pass mechanically | Exit 0, 9 successful tools, populated reports; analytical safety failures remain. |
| Current saved JSON strict parsing | Pass | 24/24 current JSON artifacts parsed strictly. A reachable `NaN` serialization defect was separately reproduced. |
| Tracked-secret scan | Pass | No high-confidence API key/private-key pattern and no tracked `.env`/secret file found. |

### Fresh-output snapshot

| Property | Churn | Air quality |
| --- | ---: | ---: |
| Final status | `complete` | `complete` |
| Findings | 38 | 28 |
| Recommendations | 4 | 2 |
| Internal coverage ledger | 3 of 26 answered | 2 of 12 answered |
| Charts | 25 | 12 |
| Charts without `finding_id` | 13 | 6 |
| Default scenario goal | `maximize` | `maximize` |
| Scenario projection | churn `+54.51%` | `CO(GT) +39.72%` |
| Claim-verifier flags in primary recommendations | 2 | 0 |
| Audit log path | `null` | `null` |
| `audit_integrity_ok` | `true` | `true` |

### Current saved-output snapshot

- 8 non-empty summaries/raw reports were inspected.
- 73 dashboard charts exist; 39 have no `finding_id`.
- 6 of 8 `summary.json` finding arrays are not monotonically ordered by `importance`.
- 3 reports contain 6 `[unverified]` markers in primary recommendations.
- 3 runs contain 4 failed analysis steps. Full Markdown/HTML/raw dossiers disclose them, but `summary.json` and dashboards omit them and the top-level result remains `complete`.

## Expectations vs. reality

| Expected contract | Observed reality |
| --- | --- |
| Evidence-first recommendations | Descriptive associations and unconstrained linear sensitivities become intervention language and action quadrants. |
| Adverse outcomes are minimized | `target_goal` defaults to `maximize`; callers cannot set it through the advertised tool schema. |
| Post-hoc audits protect the final answer | Audits update `findings`, but primary insights/recommendations remain the pre-audit versions. |
| Unverified numeric assertions are rejected | They are annotated and still printed as recommendations. |
| “Complete” means the objective is adequately covered | Runs can be `complete` with 3/26 or 2/12 agenda questions matched, convergence warnings, or failed planned steps. |
| Every chart supports a finding | 39/73 current saved charts and 19/37 fresh charts are unlinked. |
| Ranked findings are a clear order | A diversity pass intentionally reorders them, but artifacts expose neither a display rank nor the ranking policy. |
| Public API/MCP/schema are available | Only their documentation is present in the inspected repository. |
| Immutable audit trail | Append-only-by-convention JSONL with no hash chain/signature/WORM guarantee. |
| Full quality gate is green | Ruff/mypy pass; the full pytest gate fails. |

## Critical findings

None confirmed. No evidence established an immediately exploitable remote compromise or destructive data-corruption path. The high-severity analytical findings are nevertheless release blockers because they can invert the requested business/health objective.

## High findings

### EVR-01 — Scenario optimization maximizes adverse outcomes

**Evidence.** `ScenarioSimulationTool.applies_to()` assigns broad datasets a score of `0.85` (`src/tools/scenario_simulation.py:46-60`). `default_params()` supplies only the target (`:62-73`), while `execute()` defaults `target_goal="maximize"` (`:75-84`). The least-squares joint package changes direction only for the exact string `minimize` (`:177-219`), labels the result “Multi-Driver Joint Optimization,” and gives it `objective_fit=0.90` (`:382-407`). `get_schema()` does not expose `target_goal` (`:412-439`).

The fresh churn report projects churn from `0.222` to `0.343`, a `+54.51%` increase (`C:\tmp\dsa-agent-audit-6f7bff1-churn\reports\sample_customer_churn_report.md:155-162`). The fresh air report recommends a scenario that increases `CO(GT)` by `39.72%` (`C:\tmp\dsa-agent-audit-6f7bff1-air\reports\AirQualityUCI_report.md:108-110,171-177`). Both runs end with `status="complete"`.

**Impact.** The system can reverse the user's intent and present harm as optimization. This is a decision-safety failure, not a wording issue.

**Fix.** Disable “optimization” unless the target's direction is explicit and validated. Add a typed objective contract (`minimize`, `maximize`, `hold`, `sensitivity_only`) and safe domain mappings for common adverse targets such as churn, fraud, default, mortality, failure, emissions, and pollutants. Ambiguous targets must produce neutral sensitivity analysis, never a prescriptive optimum. Binary targets need bounded probability models; candidate levers need actionability and collinearity checks.

**Acceptance tests.** Golden tests for churn, default, fraud, failure rate, mortality, pollution, revenue, and an ambiguous target; no adverse target may be increased by default, and `target_goal` must be visible in the schema and final provenance.

### EVR-02 — Observational evidence is promoted as operational action

**Evidence.** The action matrix treats “effects are adjusted associations,” “observational linear extrapolation,” “observational cohort difference,” and “linear optimization” as benign disclaimers (`src/core/findings.py:609-623`). It then places qualifying rows in Quick Wins or Strategic Bets and says “robust statistical confidence” or “potential upside” (`:643-668`). Deterministic synthesis similarly emits “Prioritize,” “Target,” “Optimize,” and “Evaluate intervention” (`src/core/controller_report.py:251-280`).

Runtime examples include:

- Air-quality Quick Wins to simulate sensor/pollutant interventions and Strategic Bets to “Optimize NOx(GT)” spikes (`C:\tmp\dsa-agent-audit-6f7bff1-air\reports\AirQualityUCI_report.md:69-83`).
- A saved text-only run turns the most frequent word into `Target review_text='terrible'` with “High potential upside” (`output/runs/1791963563-bb4cef47/reports/text_column_report.md:22-38`).
- A saved price series recommends “Optimize price,” “Optimize max_drawdown,” and “Optimize total_return” (`output/runs/1791963560-113523a4/reports/price_series_report.md:52-59`).

**Impact.** Statistical confidence is being confused with intervention validity. Users can mistake correlations, labels, sensors, timestamps, anomalies, or text tokens for controllable causal levers.

**Fix.** Separate `observation`, `hypothesis`, `pilot`, and `causal_action` outputs. Causal/observational caveats must be material. Only validated experimental or identified causal evidence may enter an action quadrant; otherwise phrase the result as a monitoring signal or experiment proposal. Require explicit controllability metadata rather than inferring it mainly from a column name.

**Acceptance tests.** Text tokens, dates, sensors, anomaly values, observational segment gaps, and unadjusted scenarios must never become Quick Wins. An RCT or validated causal design should be the positive-control case.

### EVR-03 — Post-hoc robustness failures do not rebuild primary synthesis

**Evidence.** Importance scoring has no audit/fragility penalty (`src/core/findings.py:183-194`). Deterministic insights and recommendations are created from pre-audit ranked findings without checking robustness (`src/core/controller_report.py:244-276`). Finalization later attaches checks and refreshes only `final_result["findings"]` (`src/core/controller.py:1499-1514`); it does not regenerate insights or recommendations.

In the fresh churn output, the top Month-to-month result is recommended at `989.8% higher churn`, while its audit marks `robust=false` and reports a `99.2%` effect shift after removing high-leverage rows (`C:\tmp\dsa-agent-audit-6f7bff1-churn\reports\sample_customer_churn_report.md:36,121-125`). The later action matrix correctly places it in Risk & Watch (`:50`), proving that different report surfaces disagree.

**Impact.** The strongest user-facing recommendation can ignore the audit meant to qualify or downgrade it, leaving primary recommendations and the later action matrix in conflict. The observed trimming result preserves the effect's sign and remains large, so this example proves a robustness-label/synthesis inconsistency, not that the association itself is false.

**Fix.** Make post-hoc audits a gate before final ranking and synthesis, or rebuild every derived surface from one post-audit finding projection. A failed robustness, leakage, paradox, claim-verification, or causal-language check must lower rank and prevent primary recommendation status.

**Acceptance tests.** Inject a high-effect fragile finding and assert that it appears only in limitations/watch surfaces; primary insight, recommendation, dashboard priority, and summary must agree.

### EVR-04 — Unverified recommendations ship despite a strict-rejection promise

**Evidence.** Public documentation says unverified assertions are “strictly rejected” (`docs/public/overview.md:21-24`). The implementation annotates them in place (`src/core/claim_verification.py:196-203,246-272`), logs them, and continues (`src/core/controller.py:1327-1333`). The recommendation generator derives new percentages from effect values (`src/core/controller_report.py:251-264`), and report generation prints those recommendations directly (`src/tools/report_generator.py:833-838`).

The fresh churn report prominently recommends `989.8%` and `116.9%` values with `[unverified]` annotations (`...sample_customer_churn_report.md:35-38`). In the current saved snapshot, 3 of 8 reports contain 6 such primary recommendation markers.

**Impact.** The claim verifier detects a provenance failure but does not enforce the advertised boundary.

**Nuance.** The observed percentages are rounded derivations from finding effects, not evidence of fabricated raw data. The confirmed defect is that provenance-unverified quantities remain actionable and contradict the public contract.

**Fix.** Either derive recommendation numbers into a structured, auditable evidence object before verification, or omit/quarantine any recommendation that fails. Keep rejected text only in an audit section. For binary outcomes, carry explicit absolute percentage-point difference, relative risk/lift, base rates, and sample sizes instead of formatting one generic `effect` value as a headline percentage. Change the public promise only if annotation is deliberately the intended policy.

**Acceptance tests.** Release-mode reports must contain zero unverified primary claims; a deliberately unsupported number must be absent from insights/recommendations and present in structured audit metadata.

### EVR-05 — Advertised REST API, MCP server, and stable result schema are absent

**Evidence.** The product overview advertises REST and remote MCP interfaces (`docs/public/overview.md:30-36`). The API guide specifies bearer authentication, `/v1/datasets`, `/v1/runs`, SSE, cancellation, and artifacts (`docs/public/api-quickstart.md:7-27`). The MCP guide names a streamable endpoint and seven tools (`docs/public/mcp.md:7-42`). The result specification promises a versioned schema with `schema_version`, `run_id`, `status="succeeded"`, typed recommendations, limitations, governance, and artifacts (`docs/public/result-schema.md:7-39,85-96`).

An exhaustive tracked-source search found no FastAPI router, `/v1` routes, MCP server, or named MCP tools; only the two documentation files matched API/MCP implementation-style filenames. The implemented app is Streamlit. The actual result built at `src/core/controller_report.py:327-339` uses `status="complete"`, recommendation strings, `finding_id`, and a materially different top-level shape. `scripts/check_ai_surface.py:142-164` validates documentation presence/headers but not server or schema conformance, so it passes despite the gap.

**Impact.** External integrators are promised endpoints, authentication, tenancy, lifecycle behavior, and a stable contract that this repository cannot provide.

**Scope caution.** An independently deployed, untracked service cannot be ruled out. The confirmed statement is that no implementation or conformance adapter exists in the inspected repository.

**Fix.** Immediate: label these pages as design proposals/roadmap or remove them from the released discovery surface. Long term: implement one versioned adapter with authentication, tenancy, cancellation, retention, SSE, MCP, artifact authorization, and schema conformance tests. Do not expose a partial endpoint under the current promises.

**Acceptance tests.** Either (a) HTTP/MCP black-box tests exercise every documented route/tool and validate the published schema, or (b) no public surface claims they exist.

## Medium findings

### EVR-06 — The selected logistic model did not converge, but the result does not say so

`LogisticRegression` is capped at `max_iter=500` (`src/tools/ml_pipeline.py:1194-1197`). Fit and cross-validation run without capturing `ConvergenceWarning` (`:959-987`), and best-model selection has no convergence criterion. The fresh churn run emitted six warnings; the saved estimator reported `n_iter_=[500]`, exactly the cap, yet logistic regression was selected and no structured convergence state or convergence warning was recorded for that model. The run-level `overfit_warnings` list was nonempty for XGBoost, which is a different condition and candidate.

The minimum-five-fold contract is also not enforced. `n_cv_folds` has no schema minimum (`src/tools/ml_pipeline.py:1395-1397`) and is passed directly into `StratifiedKFold`/`KFold` (`src/tools/ml_common.py:183-207`). The standalone validator explicitly requests 3 folds and still labels its section “anti-overfitting” (`scripts/validate.py:153-164`). It also considers a `0.172` train/test gap “sane” because its smoke threshold is `<0.5` (`:300-311`), even though the product warning threshold is `0.10`.

Failure to converge does not prove the predictions are unusable, but it makes coefficient stability and optimum quality unknown. Capture fit/CV warnings per candidate, retry with scaled data/appropriate solver/higher limit, persist `converged`, `n_iter`, and warning evidence, and do not promote an unconverged candidate without a visible downgrade. Enforce at least five folds when the data permits, with explicit documented exceptions for insufficient time/group splits. Add convergence and fold-count tests, and keep the 68-check script labeled as a smoke test rather than a release certification.

### EVR-07 — `complete` overstates coverage and partial failures

The controller always returns `status="complete"` from deterministic synthesis (`src/core/controller_report.py:327-339`). Coverage is attached later (`src/core/controller.py:1314-1320`) using a deliberately conservative heuristic (`src/core/agenda.py:522-585`), but the primary Markdown report does not render it. The fresh runs report `complete` while their own ledgers say only 3/26 and 2/12 questions matched.

The same status is deliberate for a user-stopped partial run: `tests/test_controller_stop.py:47-60` asserts both `stopped=true` and `status="complete"`, even when no analysis tool ran. The UI correctly labels later stages skipped and the report partial, but machine consumers still receive a successful terminal word.

Three of eight retained runs contain four failed analysis steps. The full dossier discloses them under Tool Execution Log and Limitations, which is good; however, raw results still say `complete`, and compact `summary.json`/dashboard artifacts lose the error state. “Failed runs 0” in Governance counts sandbox code failures, not failed analysis steps, but that scope is not labeled.

The fresh air run also emitted repeated NumPy `invalid value encountered in divide` warnings that were not represented in its final artifacts. This does not by itself invalidate a result, but numerical warnings need structured attribution rather than disappearing with console output.

Add terminal states such as `succeeded`, `succeeded_with_warnings`, `partial`, `failed`, and `cancelled`; include coverage, convergence, unverified claims, and `execution_issues` in every machine-facing artifact. Render the top objective-critical gaps in the primary report. Label sandbox failures separately. Because coverage is heuristic, show both its method and denominator rather than claiming literal analytical completeness.

### EVR-08 — Overfit advice is triggered by a non-selected model

`src/core/controller_report.py:227-242` emits one generic recommendation if **any** candidate has an overfit warning. In the fresh churn run, logistic regression was best with `train_test_gap=-0.0075`; only XGBoost exceeded the gap threshold, yet the first recommendation says to reduce model complexity/lower `max_depth`, advice that does not describe the selected logistic model.

Attach warnings to model names, distinguish selected-model risk from rejected-candidate diagnostics, and generate model-specific remediation. The primary recommendations should discuss the chosen model only unless explicitly comparing candidates.

### EVR-09 — Deterministic planning is broad rather than objective-led

Fallback routing largely uses generic applicability scores. Scenario simulation scores almost any dataset with two measures at `0.85` (`src/tools/scenario_simulation.py:46-60`). The churn plan ran PCA, threshold, survival, equity, scenario, and variable-scale analyses; the air plan ran an adverse scenario and a weakly related threshold analysis. A text-only saved run attempted a statistical test and clustering even though no numeric measures existed, producing two disclosed errors (`output/runs/1791963563-bb4cef47/reports/text_column_report.md:66-98`).

Add objective/question-fit, prerequisites, expected decision value, and a per-question/tool budget. Specialist applicability should be necessary but not sufficient. Stop when objective-critical coverage and evidence quality reach a declared threshold. **Changing task-decomposition/planning logic requires user approval under `AGENTS.md` before implementation.**

### EVR-10 — All-null records inflate the air-quality population

The bundled `data/AirQualityUCI.csv` contains 114 delimiter-only rows. The reader drops all-null columns (`src/core/io.py:874-885`) but not all-null rows; trailing cleanup examines at most the final 15 lines and does not classify equal-width delimiter-only rows as metadata (`:372-394`). The fresh report says `9471` rows, while conversions consistently show only `9357` real records (`...AirQualityUCI_report.md:10,31-37`).

Drop rows for which every retained data column is null before profiling, record the count in `ReadReport`, and use the corrected denominator everywhere. Add CSV/TSV tests with more than 15 delimiter-only trailing rows and a control case where partially empty rows must remain.

### EVR-11 — JSON writers can emit non-standard bare `NaN`

Report, summary, dashboard, and CLI serializers use `json.dumps(..., default=str)` without `allow_nan=False` or deep normalization (`src/tools/report_generator.py:912-935`, `src/core/controller_report.py:155-171,488-490`, `src/core/dashboard_common.py:80-82`, `main.py:142-146`). A direct reachable reproduction wrote `"value": NaN` at `C:\tmp\dsa-json-nan-repro\nan_probe_raw.json:14`, which strict JSON parsers reject.

Create one shared deep JSON-normalization utility for NumPy/pandas scalars and non-finite floats, choose `null` plus an optional data-quality note, and serialize every public artifact with `allow_nan=False`. Add nested `NaN`, `Infinity`, NumPy scalar, timestamp, and nullable-dtype tests.

### EVR-12 — Chart-to-evidence linkage does not match the product promise

README says each chart is tied to the finding it supports (`README.md:711`), while `ChartSpec.finding_id` is optional (`src/core/dashboard_common.py:50-76`). In the final saved snapshot, 39/73 charts are unlinked; fresh churn and air dashboards have 13/25 and 6/12 unlinked respectively.

Either change the promise and add `chart_role: evidence|context`, with a `context_id` for baseline EDA, or require `finding_id` for analytical charts. Add dashboard schema and orphan-chart tests.

### EVR-13 — “Ranked findings” are intentionally reordered without an exposed display policy

`rank_findings()` sorts by importance and then applies source diversity and deduplication (`src/core/findings.py:432-482`). That can be a reasonable presentation choice, but JSON retains only the underlying `importance`; 6/8 current summaries therefore appear unsorted.

Add `display_rank`, `raw_importance_rank`, and `ranking_policy`, or keep machine JSON in raw score order and expose a separate curated presentation list. Test deterministic ordering and document the distinction.

### EVR-14 — The required test gate fails and dependency declarations disagree

The full suite has eight failures:

1. Three mixed-model tests fail in the full suite with “truth value of a Series is ambiguous” but pass together in isolation. This is a suite-order/global-state-sensitive defect whose root cause is still unisolated; it nevertheless breaks CI confidence.
2. `multi_seasonal_analysis` tests expect `period_steps` and `periodic_variance_explained_pct`, while output exposes `period` and `primary_variance_explained_pct` (`tests/test_multi_seasonal_analysis.py:31-37`; `src/tools/multi_seasonal_analysis.py:194-229`).
3. `ranking_dynamics` tests pass `*_col`, but the tool schema accepts `*_column` and silently absorbs aliases through `**_`; output also returns `rankings`/`home_advantage_odds` instead of expected `ratings`/`home_advantage_odds_ratio` (`tests/test_ranking_dynamics.py:38-55`; `src/tools/ranking_dynamics.py:75-124,173-189,276-288`).
4. `detect_and_strip_summary_rows` checks only `dtype == "object"`, missing pandas 3 `StringDtype` and leaving `Grand Total` (`tests/test_reshaping.py:43-52`; `src/core/reshaping.py:81-104`).
5. `threshold_analysis` tests expect `threshold`, `f_pvalue`, `pre_slope`, `post_slope`; output returns `breakpoint`, `p_value`, `pre_breakpoint_slope`, `post_breakpoint_slope` (`tests/test_threshold_analysis.py:25-33`; `src/tools/threshold_analysis.py:231-246`).
6. The raw-placeholder test's regex matches a legitimate scenario shift near `-200`, not a sentinel value, so this failure is a test false positive (`tests/test_quality_rubric.py:73,89-93`).

Environment drift compounds this: `pyproject.toml:44-49` requires pytest `<9`, `requirements.txt:36-40` does not cap it, and the environment snapshot pins pytest `9.0.3` (`requirements.lock:145-146`). Align one supported environment, fix the four real contracts, narrow the sentinel assertion to evidence/source values, and isolate the mixed-model state leak. The gate is not complete until the full suite passes in one process on every supported OS/Python matrix.

### EVR-15 — Architectural boundaries have drifted beyond their tests

- Chart design calls `self.llm_client.call()` directly (`src/core/controller_report.py:104-133`), violating the repository rule that LLM calls go through `RLMEngine.invoke()`.
- Core agenda code dynamically imports and instantiates `src.tools.*` to call tool behavior (`src/core/agenda.py:451-463`) instead of using `ToolRegistry`/neutral contracts.
- Dashboard core imports a private tool helper (`src/core/dashboard.py:698-703`).
- `src/rlm/engine.py:15` imports concurrency policy from `src.core.llm_client`, although the engine boundary says it should depend on an injected callable and REPL environment.
- `tests/test_architecture.py:44-72` tests only selected forbidden directions and therefore misses these reverse dependencies and direct-call behavior.

Route chart calls through the RLM layer, move applicability/benchmark contracts to neutral modules or the registry, and expand AST architecture tests. **Any change to `src/rlm/engine.py` requires explicit user approval first.**

### EVR-16 — The model-signing secret is exported beside the models

`src/core/model_io.py:4-14,26,34-61` stores `.model_key` in the same model directory as the pickle and signature. The Downloads UI iterates every file in that directory without filtering (`ui/tabs/downloads_tab.py:114-134`), and retained runs copy the full tree (`ui/run.py:60-80`). The fresh churn model directory contains a 32-byte `.model_key` alongside all model artifacts.

Anyone receiving that bundle can forge a valid HMAC signature. No current model-upload route was found, so this is not a confirmed present remote-RCE path; it defeats the integrity property if artifacts are shared or later re-imported. Store the secret outside artifact roots/operator-side, or use asymmetric signatures with a public verification key. Filter secrets/dotfiles from downloads and retention manifests; test that export bundles cannot sign replacements.

### EVR-17 — “Immutable audit trail” and `audit_integrity_ok` overstate the control

Public safety docs promise an immutable audit trail containing each script, hash, duration, and memory footprint (`docs/public/safety.md:19-29`). Governance appends ordinary mutable JSONL (`src/core/governance.py:210-231`); its entry includes duration but no observed memory/RSS field. `audit_integrity_ok` only checks whether append operations failed (`:233-255`); on fresh zero-event runs it is `true` while both audit paths are `null`.

Either rename the field to `audit_write_ok` and document the scope, or add a hash chain, signed run manifest, durable append-only/WORM storage, and verification step. If a reproducible ledger is promised, emit a zero-event manifest rather than equating no write error with integrity.

### EVR-18 — Plain-language rewriting corrupts report punctuation and statistical grammar

The CI regex can consume the closing parenthesis of surrounding prose (`src/core/plain_language.py:170-173,222-230`). Fresh churn output consequently contains unmatched text such as `(... odds ratio 0.14, likely between 0.09 and 0.22.` (`...sample_customer_churn_report.md:25-26,137,146`). `_adjusted_phrase` also emits anaphoric “still so after correcting...” without ensuring a preceding raw-p statement (`src/core/plain_language.py:155-162`), producing a malformed executive sentence at report line 31.

Preserve matched delimiters, make adjusted-p phrasing context-sensitive, and add exact sentence-level plus Markdown/HTML rendering tests for bracketed, parenthesized, and bare intervals. Numbers are not altered, so this is product-trust/presentation risk rather than a changed statistical result.

### EVR-19 — The canonical seven-stage workflow is documented as five stages in several surfaces

`AGENTS.md` and runtime UI define seven stages, but README calls it a five-stage workflow (`README.md:34,258-266`), public workflow docs enumerate only stages 1-5 (`docs/public/workflow.md:11-53`), and the embedded landing registry repeats that model while nearby copy says seven stages. Stale report/component docstrings also label Stage 7 work as Stage 5.

Create one canonical stage specification and generate/assert every UI/public rendering from it. A simplified five-phase or four-scene explainer is acceptable only if labeled “macro phases/scenes” and explicitly mapped to canonical stages. Add a consistency test; the current discovery checker only verifies file/header presence.

### EVR-20 — Published and internal limits, retention, and configuration facts drift

- Docs promise a 250 MB upload limit and 10:1 compression ceiling (`docs/public/file-formats.md:22-38`); code defaults are 200 MB and 100:1 (`src/core/security.py:41-48,81-100`).
- The same page says archives with sub-folders, symlinks, or executables are rejected. The safety pass checks entry count, expanded size, and ratio (`src/core/security.py:203-236`); the reader enforces exactly one non-directory member but does not inspect its path/type metadata (`src/core/io.py:275-293`). This is a documentation/control mismatch, not a demonstrated extraction exploit—the code streams the member rather than extracting it.
- Docs promise a 500-column cap and 24-hour retained-dataset purge (`docs/public/limits.md:34-38`); no 500-column gate or retained `output/runs` expiry was found. `sweep_stale_run_dirs()` cleans only `dsa-run-*` temporary directories after two hours (`ui/run.py:99-170`), while completed artifacts are copied durably (`:60-80`).
- Internal/default documentation is inconsistent: `run_sandboxed()` documents a 45-second default but executes with 60 seconds (`src/core/sandbox.py:1291-1297`); README says 45 while public docs and runtime say 60.

Put limits in one typed configuration source and render documentation/tests from it. Add boundary tests for upload bytes, expansion ratio, row/column caps, timeout, and retention jobs. Do not claim tenant/retention guarantees until an implemented deployment layer enforces them.

## Low findings

### EVR-21 — Landing-page `postMessage` receivers do not validate origin/source

Two listeners accept the first `streamlit:render`-shaped message without checking `event.source` or `event.origin` (`ui/landing_component/index.html:1915-1921,3948-3955`). Current effect appears limited to theme spoofing/first-message denial, not data exfiltration. Require `event.source === window.parent`, validate the expected origin, and validate message schema. Test hostile sibling/iframe messages.

### EVR-22 — Secret-ignore policy is too narrow

`.gitignore:13-15` ignores only the exact `.env` name, not common variants such as `.env.local`, `.streamlit/secrets.toml`, private-key formats, or `.model_key`. No tracked secret was found. Expand patterns carefully and add a CI secret scanner/pre-commit check; allow explicit safe examples such as `.env.example`.

### EVR-23 — Local filesystem/localhost links leak into public security surfaces

`docs/public/safety.md:50` and `docs/AI_ACCESS_POLICY.md:95` contain a developer-local `file:///d:/...` link; `public/.well-known/security.txt:3-4` uses localhost canonical/policy URLs. The embedded landing documentation copies these values. Use deployment-origin-derived/root-relative URLs and validate public builds with a non-local origin. This is information leakage and broken-contact hygiene, not a credential disclosure.

## Confirmed strengths

- Ruff and mypy are clean.
- Both fresh CLI analyses and the 68-check standalone seven-stage validation completed and produced populated reports.
- Model outputs do contain cross-validation means/stds and train/test gaps; the defect is in convergence and how warnings are attributed, not absence of anti-overfit metrics.
- Failed tools are visible in the full human dossier and limitations rather than silently swallowed.
- The action matrix does quarantine at least some fragile findings; the contradiction is that primary synthesis remains stale.
- All 24 current saved JSON artifacts happened to be strict-valid.
- No high-confidence credential/private-key pattern was found in tracked files, and no `.env` was tracked.
- No customer ID values searched from the churn sample appeared in generated reports.
- No confirmed DOM XSS, unsafe remote script dependency, `eval`, or raw secret storage was found in the reviewed frontend.

## Prioritized remediation plan

### P0 — Stop unsafe recommendations and correct public truth claims

1. **Target-direction policy:** make scenario direction explicit; default ambiguous/adverse targets to sensitivity-only; temporarily exclude scenario findings from recommendations/action matrices.
2. **Actionability/causality gate:** observations become hypotheses or monitoring signals; only validated causal/experimental results become actions.
3. **Audit-first synthesis:** run claim, fragility, leakage, paradox, causal-language, and convergence gates before ranking; regenerate every output surface once.
4. **Primary-claim quarantine:** reject unverified and failed-robustness items from insights/recommendations; expose them only in audit/limitations.
5. **Public contract correction:** mark REST/MCP/schema/tenant/retention pages as roadmap immediately, or hide them until implementation.

**Primary modules:** `scenario_simulation.py`, `findings.py`, `controller.py`, `controller_report.py`, `claim_verification.py`, `report_generator.py`, public docs.  
**Exit gate:** the churn and pollution golden runs cannot recommend increasing harm; zero unverified/failed-robustness primary claims; public discovery describes only shipped interfaces.

### P1 — Make completion and artifacts mechanically trustworthy

1. Add typed terminal status plus structured `execution_issues`, convergence state, and objective coverage to final, summary, dashboard, Markdown, and HTML artifacts.
2. Add a shared strict JSON normalizer and `allow_nan=False` everywhere.
3. Drop/report all-null rows before profiling.
4. Repair plain-language punctuation/adjusted-p grammar.
5. Expose chart roles/linkage and display-ranking policy.
6. Make model warnings candidate-specific and selected-model-specific.

**Exit gate:** strict schema validation for every artifact; all-null-row fixture reports the correct population; zero charts lack either `finding_id` or an explicit context role/`context_id`; status becomes `succeeded_with_warnings` for injected partial failures; report grammar snapshots pass.

### P2 — Restore contracts, architecture, and security boundaries

1. Fix the four deterministic tool/test contract drifts; fix the placeholder false-positive test and mixed-model state leak.
2. Align pytest declarations and produce a real resolved lock/environment matrix.
3. Route chart LLM calls through `RLMEngine`; remove core-to-tool behavior imports; expand architecture tests.
4. Keep signing keys outside export roots or adopt asymmetric signatures; filter secret artifacts.
5. Rename or cryptographically strengthen audit-integrity semantics.
6. Validate `postMessage` source/origin and broaden secret hygiene.

**Approval boundary:** changes to `src/rlm/engine.py` or task-decomposition logic require explicit user approval under `AGENTS.md`. No new tool is required for this plan; if one becomes desirable, its Execution Layer contract must be approved first.  
**Exit gate:** full pytest/ruff/mypy green; architecture tests cover both dependency directions and direct LLM calls; exported model bundles contain no signing secret; audit fields match their actual guarantee.

### P3 — Consolidate public/runtime specifications

1. Generate workflow labels from one seven-stage spec.
2. Generate limits/schema documentation from typed code constants.
3. If API/MCP remain in scope, implement one authenticated adapter and black-box conformance suite; otherwise keep them explicitly non-shipped.
4. Add non-local-origin public-build checks and remove workstation paths.

**Exit gate:** documentation, UI, runtime events, result schemas, limits, and discovery tests all derive from or validate against the same contracts.

## Recommended release gates

Do not describe the system as decision-ready until all are true:

1. No adverse-outcome optimization without explicit direction.
2. No observational association is labeled an action without causal/actionability qualification.
3. No unverified, fragile, leaking, reversed, or unconverged result appears as a primary recommendation.
4. Every terminal status truthfully represents partial failures and objective coverage.
5. Every public JSON artifact validates under strict JSON and one versioned schema.
6. Public REST/MCP/retention/tenancy claims have black-box implementations, or the claims are removed.
7. `ruff check .`, `mypy src/`, and the complete `pytest tests/ -v` suite pass in the supported environment.
8. Golden end-to-end tests cover churn, pollution/adverse targets, text-only data, time series, grouped/panel data, non-finite values, and partial tool failure.
9. Exported models do not include their signing secret; audit guarantees are accurately named and tested.

## Residual limitations of this audit

- No live external LLM provider was called, so provider-specific prompt behavior, token limits, rate limiting, and LLM-authored synthesis were not exercised.
- Hosted Docker isolation/seccomp and a real multi-tenant deployment were not executed.
- No interactive browser session or assistive-technology pass was run; frontend review was static.
- No deployed REST/MCP service was supplied. The absence finding is repository-scoped.
- Coverage matching is intentionally heuristic, so `3/26` and `2/12` are the product's own ledger values, not a claim that the analyses literally answered only that many concepts.
- Runtime outputs are ignored/volatile; the two isolated fresh output directories should be preserved separately if they are needed after temporary-storage cleanup.

## Reproduction commands

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe src\
.\.venv\Scripts\python.exe -m pytest tests\ -v
.\.venv\Scripts\python.exe scripts\validate.py
.\.venv\Scripts\python.exe scripts\check_ai_surface.py

.\.venv\Scripts\python.exe main.py `
  --dataset data\sample_customer_churn.csv `
  --target churn `
  --objective "What factors are most associated with customer churn, and which findings are reliable enough to act on?" `
  --no-llm `
  --output-dir C:\tmp\dsa-agent-audit-6f7bff1-churn

.\.venv\Scripts\python.exe main.py `
  --dataset data\AirQualityUCI.csv `
  --objective "What changed in air quality over time, which pollutants move together, and which conclusions are reliable?" `
  --no-llm --no-ml `
  --output-dir C:\tmp\dsa-agent-audit-6f7bff1-air
```
