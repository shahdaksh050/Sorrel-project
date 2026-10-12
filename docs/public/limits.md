# Runtime Limits and Quotas

Version: 1.0.0  
Last Updated: 2026-10-12  
Canonical: http://localhost:8501/docs/limits.md  

To ensure system stability, multi-tenant isolation, and predictable resource utilization, Sorrel enforces operational limits across all API and agent interactions.

---

## 1. Concurrency and Quotas

| Dimension | Default Limit | Behavior on Exceeding |
| :--- | :--- | :--- |
| **Concurrent Analyses** | 2 active runs per tenant | Returns HTTP `429 Too Many Requests` with `Retry-After` header. |
| **Daily Run Quota** | 50 analyses per day | Returns HTTP `429 Too Many Requests` (`quota_exceeded`). |
| **API Request Rate** | 60 requests / minute | Returns HTTP `429 Too Many Requests`. |
| **Dataset Storage** | 5 GB cumulative storage | Refuses new uploads until older datasets are removed or expire. |

---

## 2. Analysis Execution Ceilings

- **Maximum Run Duration**: 600 seconds (10 minutes) total wall-clock time per analysis. Runs exceeding this ceiling transition to state `"failed"` with error `timeout_exceeded`.
- **Maximum Planning Iterations**: 10 iterative refinement steps (`max_iterations`).
- **Sandbox Subprocess Timeout**: 60 seconds per code block.
- **Sandbox Memory RSS Limit**: 1024 MB per subprocess tree.
- **Maximum Sandboxed Code Invocations**: 40 executions per analysis. Further execution attempts are refused.

---

## 3. Upload & Tabular Data Ceilings

- **Maximum Upload File Size**: 250 MB.
- **Maximum Uncompressed Size**: 500 MB.
- **Maximum Rows**: 1,000,000 observations.
- **Maximum Columns**: 500 dimensions.
- **Dataset Retention**: Ephemeral datasets expire and are automatically purged after 24 hours.

---

## 4. Backpressure and Retry Semantics

When rate limits or concurrency ceilings are reached:
- Sorrel emits standard HTTP `429 Too Many Requests` status codes.
- Responses include a `Retry-After: <seconds>` header indicating the wait interval before retrying.
- Non-idempotent operations must include an `Idempotency-Key` header to safely retry requests without duplicate billing or execution.
