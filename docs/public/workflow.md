# Analysis Workflow

Version: 1.0.0  
Last Updated: 2026-10-12  
Canonical: http://localhost:8501/docs/workflow.md  

Sorrel executes all analyses through a strict, multi-stage governed reasoning and execution workflow. External clients (REST, MCP, and UI) observe progress across these stages through typed events and run handles.

---

## The Workflow Stages

```text
[Stage 1: Ingestion & Profiling]
            │
            ▼
[Stage 2: Hypothesis & Plan Generation]
            │
            ▼
[Stage 3: Tool Execution & Sandbox Verification] ◄──┐ (Iterative Loop)
            │                                       │
            ▼                                       │
[Stage 4: Iterative Refinement & Adversarial Review]─┘
            │
            ▼
[Stage 5: Synthesis, Post-Hoc Audits & Dossier Generation]
```

### Stage 1: Dataset Ingestion & Profiling
- **Parsing**: Uploaded tables are safely coerced into uniform columnar representations.
- **Profiling**: Columns are categorized into continuous, discrete, categorical, datetime, or text modalities.
- **Statistical Baselines**: Calculates quantile spreads, missingness rates, skewness, variance, entropy, and prominent categorical distributions.
- **Integrity Checks**: Detects ID-like unique keys, constant columns, high missingness traps, and invalid sentinel encodings.

### Stage 2: Hypothesis & Plan Generation
- **Domain Framing**: Evaluates dataset schemas against domain packs (e.g. e-commerce, healthcare, finance, industrial).
- **Hypothesis Formulation**: Formulates prioritized testable propositions aligned with the user objective or discovered distributions.
- **Step Decomposition**: Constructs a directed analytical plan consisting of specific tool invocations (e.g., correlations, segment comparisons, statistical tests).

### Stage 3: Tool Execution & Sandbox Verification
- **Execution Layer**: Invocations execute through `BaseTool.run()`. Tools include `CorrelationAnalysisTool`, `SegmentComparisonTool`, `StatisticalTestingTool`, `ClusterDataTool`, `TrainModelTool`, and `CausalAnalysisTool`.
- **Sandbox Quarantining**: Custom calculations or dynamic scripts run inside an audited sandbox with strict timeouts (60s) and memory ceilings (1024 MB).
- **Finding Extraction**: Each tool execution registers candidate `Finding` objects capturing effect sizes, p-values, sample sizes, and baseline metrics.

### Stage 4: Iterative Refinement & Adversarial Review
- **Multiple Testing Guard**: Corrects raw p-values across all executed hypotheses using the Benjamini-Hochberg False Discovery Rate (FDR) procedure.
- **Adversarial Critique**: Subject candidate findings to robustness tests, confounding checks, and sensitivity checks (e.g., verifying whether segment lifts persist after controlling for covariates).
- **Adaptive Re-planning**: If initial hypotheses yield weak signals or high uncertainty, generates follow-up steps.

### Stage 5: Synthesis, Post-Hoc Audits & Dossier Generation
- **Claim Verification**: The `ClaimVerification` engine cross-examines every numeric quantity in narrative summaries against the audited execution ledger. Unbacked assertions are stripped or flagged.
- **Dossier Compilation**: Produces structured findings, charts, concrete unit/dollar-quantified recommendations, and explicit analytical limitations.
- **Report Rendering**: Generates downloadable executive reports in Markdown, JSON, and standalone HTML formats.

---

## Finding Structure & Evidence

Every finding produced by the workflow includes:
- **`headline`**: High-signal, human-readable summary of the observed phenomenon.
- **`kind`**: Finding category (e.g. `segment_lift`, `correlation`, `model_performance`, `anomaly`).
- **`importance`**: Computed relevance score (0.0 to 1.0) based on effect magnitude and sample coverage.
- **`confidence`**: Audited confidence incorporating p-values, cross-validation stability, and FDR bounds.
- **`evidence`**: Audited metric values, baseline comparisons, sample counts, and passed verification tests.
