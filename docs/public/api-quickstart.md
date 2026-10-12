# API Quickstart

Version: 1.0.0  
Last Updated: 2026-10-12  
Canonical: http://localhost:8501/docs/api-quickstart.md  

The Sorrel REST API enables automated systems to run evidence-first analyses. The API is asynchronous and handle-based: analysis requests return immediately with a `run_id`, allowing clients to poll status or stream Server-Sent Events (SSE).

---

## 1. Authentication

All requests to the `/v1/` API require a Bearer token in the `Authorization` header:

```bash
Authorization: Bearer sorrel_live_xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 2. The Four-Step Workflow

```text
1. Upload Table ───────► POST /v1/datasets ───────► dataset_id
2. Start Analysis ─────► POST /v1/runs ───────────► run_id (HTTP 202)
3. Poll / Stream ──────► GET /v1/runs/{id} ───────► status: running | succeeded
4. Retrieve Results ───► GET /v1/runs/{id}/result ─► Audited findings & evidence
```

### Step 1: Upload a Dataset

Upload your tabular file (`.csv`, `.parquet`, `.json`, `.tsv`) as multipart form data:

```bash
curl -X POST http://localhost:8501/v1/datasets \
  -H "Authorization: Bearer $SORREL_API_KEY" \
  -F "file=@customer_churn.csv"
```

**Response (HTTP 201 Created):**
```json
{
  "dataset_id": "ds_9f8b2c1a4e7d",
  "filename": "customer_churn.csv",
  "bytes": 1048576,
  "rows": 10000,
  "columns": 14,
  "created_at": "2026-10-12T12:00:00Z",
  "expires_at": "2026-10-13T12:00:00Z"
}
```

### Step 2: Start an Analysis

Initiate the governed analysis run. Pass the `dataset_id`, analytical objective, and optional configuration:

```bash
curl -X POST http://localhost:8501/v1/runs \
  -H "Authorization: Bearer $SORREL_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: churn-q4-2026-analysis" \
  -d '{
    "dataset_id": "ds_9f8b2c1a4e7d",
    "objective": "Identify the primary drivers of customer churn and quantify retention levers.",
    "target": "churn",
    "options": {
      "use_ml": true,
      "max_iterations": 8
    }
  }'
```

**Response (HTTP 202 Accepted):**
```json
{
  "run_id": "run_3d8e9a2b5c1f",
  "status": "queued",
  "location": "/v1/runs/run_3d8e9a2b5c1f",
  "created_at": "2026-10-12T12:00:05Z"
}
```

### Step 3: Monitor Run Progress

Poll the run status endpoint:

```bash
curl -X GET http://localhost:8501/v1/runs/run_3d8e9a2b5c1f \
  -H "Authorization: Bearer $SORREL_API_KEY"
```

**Response (HTTP 200 OK):**
```json
{
  "run_id": "run_3d8e9a2b5c1f",
  "status": "running",
  "current_stage": 3,
  "stage_name": "Tool Execution & Sandbox Verification",
  "progress_pct": 55,
  "active_step": "Comparing segments across contract_type"
}
```

Alternatively, stream real-time updates via Server-Sent Events (SSE):
```bash
curl -N -H "Authorization: Bearer $SORREL_API_KEY" \
  http://localhost:8501/v1/runs/run_3d8e9a2b5c1f/events
```

### Step 4: Retrieve Structured Results & Artifacts

Once `status` reaches `"succeeded"`, fetch the final audited result:

```bash
curl -X GET http://localhost:8501/v1/runs/run_3d8e9a2b5c1f/result \
  -H "Authorization: Bearer $SORREL_API_KEY"
```

List and download generated artifacts (reports, charts, serialized models):
```bash
# List available artifacts
curl -X GET http://localhost:8501/v1/runs/run_3d8e9a2b5c1f/artifacts \
  -H "Authorization: Bearer $SORREL_API_KEY"

# Download the executive markdown report
curl -X GET http://localhost:8501/v1/runs/run_3d8e9a2b5c1f/artifacts/analysis-report \
  -H "Authorization: Bearer $SORREL_API_KEY" \
  -o executive_report.md
```

---

## 3. Cancelling a Run

Runs can be cancelled cooperatively at any time:

```bash
curl -X POST http://localhost:8501/v1/runs/run_3d8e9a2b5c1f/cancel \
  -H "Authorization: Bearer $SORREL_API_KEY"
```

---

## 4. Error Envelope

All non-2xx responses follow a uniform error shape:

```json
{
  "error": {
    "code": "quota_exceeded",
    "message": "Maximum concurrent analyses limit reached for this tenant.",
    "request_id": "req_8b7a6c5d4e",
    "retry_after_s": 30
  }
}
```
