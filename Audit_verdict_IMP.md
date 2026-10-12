Audit verdict
The project has a strong foundation—clear layering intentions, extensive tooling, evidence-oriented reporting, good visual identity, and unusually broad tests—but it is not release-ready yet.
The largest risk is not visual polish. It is that Sorrel can present observational associations as actionable interventions, promote leaked models, and describe post-selection statistics as FDR-controlled. The current unfinished diff also introduces five concrete regressions.
I performed a read-only review using the requested gpt-taste, review-agent, and redesign-existing-projects guidance, three parallel sub-agents, static inspection, current-diff review, and desktop/mobile browser runs. No source files were modified.
Release blockers
1. Observational analysis is presented as causal opportunity
[scenario_simulation.py (line 103)](D:/(Dev2\)DSA_AGENT/src/tools/scenario_simulation.py:103) selects the most correlated variable, fits univariate OLS, and describes the extrapolation as “projected impact.” Segment mode selects the lowest group and calls the entire difference “recoverable,” with hard-coded confidence values.
In the live sample run this produced claims such as a -10% tenure_months shift changing churn and a “recoverable” gender gap. These are not defensible intervention estimates.
Improve it by:
Renaming this output “descriptive sensitivity.”
Removing “recoverable,” “impact,” and intervention language unless a validated causal design exists.
Requiring an explicit controllable lever.
Accounting for confounding, residual uncertainty, driver selection, and fit-population coverage.
Refusing sensitive-attribute action recommendations.
2. Multiple-testing correction happens after selection
[CorrelationAnalysisTool (line 1095)](D:/(Dev2\)DSA_AGENT/src/tools/data_processing.py:1095) calculates significance only for selected top correlations. [Statistical testing (line 935)](D:/(Dev2\)DSA_AGENT/src/tools/statistical_analysis.py:935) creates findings only after raw significance filtering, and [run-level BH (line 74)](D:/(Dev2\)DSA_AGENT/src/core/multiple_testing.py:74) then adjusts that already-selected set.
This does not control run-level false discovery rate.
Improve it by registering every attempted hypothesis before filtering, defining test families prospectively, adjusting the complete family, and only then creating or ranking findings.
3. Controller bypasses the execution-layer contract
[controller_steps.py (line 348)](D:/(Dev2\)DSA_AGENT/src/core/controller_steps.py:348) performs pandas analysis directly after a dynamic-code failure, creates a finding, fabricates execution timing, and returns a successful ToolResult.
This violates the repository’s mandatory BaseTool.run() boundary and weakens auditability.
Move this fallback through an existing registered tool, probably segment_comparison. Adding a new tool would require explicit approval under AGENTS.md.
4. Suspected target leakage does not block model promotion
[ml_pipeline.py (line 740)](D:/(Dev2\)DSA_AGENT/src/tools/ml_pipeline.py:740) warns about leakage, but the model can still be crowned, persisted, and emitted as a positive headline. The leakage caveat is separated from the performance finding.
Hard-gate saving and promotion when leakage is suspected. Retrain without suspect columns and attach leakage status directly to every model-performance result.
5. Hosted security defaults can fail open
[app.py (line 779)](D:/(Dev2\)DSA_AGENT/app.py:779) relies on DSA_HOSTED=true for hosted protections. Network users may inherit server-side credentials, while UI configuration mutates process-wide environment state across sessions.
Use secure defaults whenever binding beyond loopback, isolate configuration per session/run, never expose server keys to browser sessions, and add concurrent-session isolation tests.
6. Public documentation advertises infrastructure absent from this repository
The public documentation claims an authenticated asynchronous REST API, SSE, MCP server, tenant isolation, quotas, and automatic 24-hour deletion:
[API quickstart (line 24)](D:/(Dev2\)DSA_AGENT/docs/public/api-quickstart.md:24)
[MCP documentation (line 7)](D:/(Dev2\)DSA_AGENT/docs/public/mcp.md:7)
[Safety policy (line 42)](D:/(Dev2\)DSA_AGENT/docs/public/safety.md:42)
[Runtime limits (line 7)](D:/(Dev2\)DSA_AGENT/docs/public/limits.md:7)
No corresponding API/MCP server implementation was found in this repository; only the documentation exists. Either implement and test these contracts or clearly mark them as proposals. Claims such as “tenant-isolated,” “strictly rejected,” and “immutable audit trail” should not be published without enforcement tests.
Regressions in the current unfinished diff
These findings are specifically attributable to the current 12-file working-tree diff.
[report_generator.py (line 69)](D:/(Dev2\)DSA_AGENT/src/tools/report_generator.py:69) and [html_report.py (line 63)](D:/(Dev2\)DSA_AGENT/src/core/html_report.py:63) discard an entire substantive executive summary if any sentence matches a boilerplate phrase. Filter sentence-by-sentence instead.

[visualization.py (line 124)](D:/(Dev2\)DSA_AGENT/src/tools/visualization.py:124) falls back from a heatmap to distributions even when there are zero usable numeric columns, then reports success with zero charts. Require a usable nonconstant numeric column or return an explicit failure.

[scenario_simulation.py (line 44)](D:/(Dev2\)DSA_AGENT/src/tools/scenario_simulation.py:44) now considers any profile dimension sufficient, while execution only accepts specific nonnumeric segment columns with adequate group sizes. Numeric-coded and boolean dimensions can produce content-free runs.

[report_generator.py (line 400)](D:/(Dev2\)DSA_AGENT/src/tools/report_generator.py:400) and [html_report.py (line 262)](D:/(Dev2\)DSA_AGENT/src/core/html_report.py:262) can render clean_data coverage gaps in both “Data Integrity” and “Limitations.” Deduplicate by finding ID and keep caveat kinds in one section.

[findings.py (line 218)](D:/(Dev2\)DSA_AGENT/src/core/findings.py:218) classifies findings using broad headline prefixes such as “missing values in.” A valid analytical headline like “Missing values in income predict churn” can disappear as hygiene. Prefer structured kind and source_tool values.

Statistical and analytical weaknesses
[Action matrix classification (line 495)](D:/(Dev2\)DSA_AGENT/src/core/findings.py:495) normalizes Finding objects but compares raw effects for serialized dictionaries. Reports pass dictionaries, so incompatible effect measures are ranked together. Scenario findings are also biased toward “Quick Wins.”
[Dataset row limiting (line 843)](D:/(Dev2\)DSA_AGENT/src/core/io.py:843) materializes the complete dataset before applying a random row cap. This does not prevent ingest OOM and destroys time-series continuity and panel histories.
[Causal IPW bootstrap (line 203)](D:/(Dev2\)DSA_AGENT/src/tools/causal_analysis.py:203) resamples precomputed weights without refitting the propensity model. DiD silently turns categorical covariates into zeros and uses poorly scaled raw datetime values.
[Seasonality tests (line 369)](D:/(Dev2\)DSA_AGENT/src/tools/time_series.py:369) apply independent-observation tests to autocorrelated time points.
[Integrity inference (line 77)](D:/(Dev2\)DSA_AGENT/src/core/integrity.py:77) promotes empirical numeric orderings into “logical constraints,” treats negative revenue-like values as impossible, examines flatlines in existing row order, and labels missingness MCAR_likely even though MNAR cannot be ruled out.
[CV construction (line 183)](D:/(Dev2\)DSA_AGENT/src/tools/ml_common.py:183) does not adequately preflight small classes, single-group panels, or tiny temporal datasets.
[Model persistence (line 955)](D:/(Dev2\)DSA_AGENT/src/tools/ml_pipeline.py:955) lacks a signed manifest binding the model to dataset fingerprint, feature schema, class mapping, split design, dependency versions, and run ID.
[Report JSON generation (line 913)](D:/(Dev2\)DSA_AGENT/src/tools/report_generator.py:913) persists complete tool outputs and evidence. Prompt PII redaction does not guarantee report-artifact redaction.
Statistical-test selection uses Shapiro and Levene pretests to choose later tests, which distorts operating characteristics. Prefer robust defaults such as Welch tests and use diagnostics as caveats.
Weighted F1 is the universal classification objective, which can conceal minority-class failure. Select macro-F1, balanced accuracy, or PR-AUC based on class balance and user objective.
Interface and design audit
Accessibility
The landing overlay declares an accessible modal but does not trap focus, make background content inert, or restore focus: [index.html (line 1737)](D:/(Dev2\)DSA_AGENT/ui/landing_component/index.html:1737).
Its tabs lack complete ARIA relationships, roving focus, hidden state, and arrow-key controls: [index.html (line 1997)](D:/(Dev2\)DSA_AGENT/ui/landing_component/index.html:1997).
At mobile widths, navigation links simply disappear while the remaining controls stay in one crowded row: [index.html (line 1206)](D:/(Dev2\)DSA_AGENT/ui/landing_component/index.html:1206).
Reduced-motion mode still forces users through a 560vh pinned narrative: [index.html (line 1248)](D:/(Dev2\)DSA_AGENT/ui/landing_component/index.html:1248).
Progress text lacks a stable live region, copy actions are not announced, and the chart table alternative is capped at 20 rows.
Product comprehension
The product tells incompatible workflow stories:
Seven stages in marketing metadata.
Four stages in the animated landing sequence.
Five stages in the workspace footer at [app.py (line 1415)](D:/(Dev2\)DSA_AGENT/app.py:1415).
Seven actual STAGE_DEFS in [cards.py (line 29)](D:/(Dev2\)DSA_AGENT/ui/components/cards.py:29).
Use one canonical seven-stage definition. If four stages are shown for simplicity, label them explicitly as four user-facing phases mapped to seven internal stages.
Other issues:
AI mode is the default, but required provider settings are hidden in a collapsed sidebar. The disabled Run button tells users to find a Streamlit chrome arrow.
All result tabs render eagerly at [app.py (line 1345)](D:/(Dev2\)DSA_AGENT/app.py:1345), including hidden charts and large tables.
Non-ML runs display 0% CV and gap metrics rather than “not run”: [answers_tab.py (line 99)](D:/(Dev2\)DSA_AGENT/ui/tabs/answers_tab.py:99).
The result surface has excessive nested borders and cards. Flatten tab panels and reserve emphasis for findings, warnings, and artifacts.
The landing overuses numbered/meta labels—“Stage 1 of 4,” section numbers, uppercase eyebrow text—which the gpt-taste review identifies as adding interface noise.
Marketing content and CTAs repeat too often; the landing is longer than its narrative requires.
Canonical and Open Graph URLs are hard-coded to localhost: [index.html (line 14)](D:/(Dev2\)DSA_AGENT/ui/landing_component/index.html:14).
A very large duplicate documentation registry is embedded directly into the 4,000-line landing HTML, ensuring documentation drift.
The footer has no direct privacy, security, or terms links.
The optional cinematic pager globally hijacks scrolling and navigation keys: [section_pager.js (line 31)](D:/(Dev2\)DSA_AGENT/ui/assets/section_pager.js:31).
Browser-run evidence
The desktop and mobile sample run completed, but browser logging accumulated many Vega warnings, including infinite extents, conflicting sort properties, and regressions with more parameters than observations. Several charts are therefore receiving empty or non-finite domains and may render blank or misleadingly.
Add a chart-spec validation boundary that rejects:
Empty series.
Non-finite domains.
Insufficient regression observations.
Confidence intervals with missing bounds.
Charts without usable axis units or data-table exports.
Testing and repository health
The last complete run reported:
1,693 tests passed.
35 tests failed.
33 failures were associated with the broken local Python/subprocess environment.
Two direct product failures remained:App-render test expects five workflow steps while the product now has seven.
Night chart-grid contrast is 1.217 against the required 1.3; the token is [chart_theme.py (line 119)](D:/(Dev2\)DSA_AGENT/src/core/chart_theme.py:119).

The checked-in .venv points to a Microsoft Store interpreter path that is currently unusable. Rebuild the environment from a pinned dependency definition and ensure subprocess tests inherit an installed package rather than relying on PYTHONPATH.
Current automated UI tests explicitly verify structure rather than visual behavior. Add Playwright coverage for 390, 768, 1024, and 1440px; keyboard-only operation; modal/tab semantics; reduced motion; axe accessibility; and visual regressions.
Recommended repair order
Stop causal/action language from scenario outputs and prevent those findings from becoming Quick Wins.
Rebuild multiple-testing around a complete hypothesis registry.
Remove controller-side data execution and route everything through BaseTool.run().
Hard-block leaked-model promotion and persistence.
Secure hosted/session configuration and artifact-level PII handling.
Fix the five current-diff regressions and add adversarial tests for each.
Make the documented product surface match what actually exists.
Repair modal, tabs, reduced-motion behavior, mobile navigation, and first-run configuration.
Unify workflow-stage copy and eliminate zero-valued “not run” metrics.
Add model manifests, design-aware sampling, robust time-series/causal inference, and byte-aware caches.
Restore a reproducible Python environment and require the complete green quality gate before release.
The most important principle is: Sorrel should make fewer claims, but make each claim demonstrably true. The interface already has enough visual character; trustworthiness and analytical restraint are now the highest-leverage improvements.