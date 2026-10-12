# Result Schema Specification

Version: 1.0.0  
Last Updated: 2026-10-12  
Canonical: http://localhost:8501/docs/result-schema.md  

This document specifies the public, version-stable JSON schema returned by `GET /v1/runs/{run_id}/result` and the `get_analysis_result` MCP tool.

---

## 1. Schema Overview

The result payload is an authoritative, typed dictionary containing audited findings, verified numbers, business recommendations, and governance metadata:

```json
{
  "schema_version": "1.0",
  "run_id": "run_3d8e9a2b5c1f",
  "status": "succeeded",
  "objective": "Identify the primary drivers of customer churn.",
  "analysis_mode": "supervised_ml",
  "dataset_summary": {
    "rows": 10000,
    "columns": 14,
    "numeric_columns": 8,
    "categorical_columns": 6,
    "missingness_pct": 0.4
  },
  "findings": [
    {
      "id": "find_01j9a8b7c6",
      "kind": "segment_lift",
      "headline": "Month-to-month contracts exhibit 3.2x higher churn rate than annual contracts",
      "explanation": "Customers on month-to-month terms churn at 42.7% compared to the 13.3% baseline.",
      "importance": 0.94,
      "confidence": 0.98,
      "p_value": 0.0001,
      "effect_size": 3.21,
      "evidence": {
        "metric": "churn_rate",
        "segment": "Month-to-month",
        "segment_mean": 0.427,
        "baseline_mean": 0.133,
        "sample_size": 4200,
        "checks_passed": ["fdr_bh_correction", "chi_square_independence", "sample_size_adequate"]
      }
    }
  ],
  "recommendations": [
    {
      "action": "Incentivize annual contract migration for new sign-ups",
      "target_segment": "Month-to-month",
      "quantified_impact": "Estimated 14% reduction in overall churn volume (~$320,000 ARR preservation)",
      "confidence": "high"
    }
  ],
  "limitations": [
    "Unobserved tenure confounding: contract type and tenure are strongly correlated.",
    "Dataset lacks payment failure telemetry."
  ],
  "governance": {
    "code_executions_count": 6,
    "pii_redacted_columns": ["email", "phone_number"],
    "claims_verified": 4,
    "claims_unverified": 0
  },
  "artifacts": [
    {
      "name": "analysis-report",
      "media_type": "text/markdown",
      "bytes": 42100,
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "download_url": "/v1/runs/run_3d8e9a2b5c1f/artifacts/analysis-report"
    }
  ],
  "created_at": "2026-10-12T12:00:05Z",
  "completed_at": "2026-10-12T12:01:22Z"
}
```

---

## 2. Field Definitions

| Field | Type | Description |
| :--- | :--- | :--- |
| `schema_version` | string | Constant `"1.0"`. Breaking changes will increment the major version. |
| `run_id` | string | Unique, opaque analysis handle. |
| `status` | string | Terminal status: `"succeeded"` or `"failed"`. |
| `findings` | array | Prioritized, evidence-backed findings checked against the execution ledger. |
| `findings[].importance` | float | Relative importance score normalized from `0.0` to `1.0`. |
| `findings[].confidence` | float | Audited statistical confidence accounting for FDR correction and sample size. |
| `recommendations` | array | Plain-language, unit/dollar-quantified business recommendations. |
| `limitations` | array | Explicit analytical constraints, confounding risks, or missing features. |
| `governance` | object | Summary of security controls, sandboxing runs, and claim verification outcomes. |
| `artifacts` | array | Manifest of downloadable reports, Vega-Lite charts, and signed models. |
