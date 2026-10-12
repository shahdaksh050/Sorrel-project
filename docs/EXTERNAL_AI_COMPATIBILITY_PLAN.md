# External AI Compatibility Plan

Status: proposed implementation plan  
Product: Sorrel / DSA Agent  
Scope: public web discovery, AI-readable documentation, REST/OpenAPI, remote MCP,
browser-agent integration, authentication, governance, and interoperability testing.

## 1. Outcome

Sorrel should support four distinct ways an external AI system can use the product:

1. **Discover Sorrel** through search crawlers, `robots.txt`, `sitemap.xml`, canonical
   metadata, and structured data.
2. **Understand Sorrel** through `/llms.txt`, concise Markdown documentation, and stable
   machine-readable descriptions.
3. **Call Sorrel** through an authenticated, versioned REST API with an OpenAPI contract.
4. **Use Sorrel as an agent tool** through a remote MCP server, with an optional WebMCP
   layer for browser-resident agents.

These are separate capabilities. `llms.txt` helps an agent find and interpret content; it
does not authorize an agent, upload a dataset, start an analysis, or replace an API/MCP
contract. `robots.txt` expresses crawler policy; it is not authentication. WebMCP helps a
browser agent operate the visible site; it does not replace a remote MCP server.

The target architecture is:

```text
Public web pages -- llms.txt / sitemap / Markdown --> search and browsing agents

Human browser -------------------------------------> Streamlit UI
Browser agent --------------- WebMCP --------------> Streamlit UI actions

External application ----- REST/OpenAPI ---+
                                           +--> Run Service --> AgentController
ChatGPT/Claude/Codex/etc. -- remote MCP ----+                         |
                                                                      +--> 7-stage workflow
```

`AgentController` remains the only analysis orchestrator. Neither REST nor MCP may call
`BaseTool.execute()` or individual analysis tools directly.

## 2. Current baseline

The repository already contains most of the internal primitives needed for an MVP:

- `AgentController.load_dataset(..., interactive=False)` supports non-interactive ingestion.
- `AgentController.analyze()` returns a structured dictionary.
- `ui/run.py` supplies `RunSpec`, `ActiveRun`, `RunSnapshot`, `RunOutcome`, cancellation,
  provisional findings, and a process-wide concurrency limit without importing Streamlit.
- `src/core/run_view.py` converts completed output into a framework-independent read model.
- `src/core/run_config.py` isolates LLM provider configuration and secrets per run.
- `src/core/security.py` has filename, extension, upload-size, compression, and signature
  validation that the API must reuse.
- Runs already produce reports, dashboards, summaries, models, and audit logs.

Important gaps:

- There is no root-level `robots.txt`, `sitemap.xml`, or `llms.txt` deployment contract.
- Streamlit serves repository assets under `/app/static`, not at origin-root paths.
- Landing metadata is inside an iframe and currently uses `https://dsa-agent.local/` as its
  canonical URL, so it is not valid production origin metadata.
- There is no stable public documentation site or Markdown alternative for product/API pages.
- There is no HTTP service, OpenAPI contract, API authentication, durable run registry, or
  remote MCP endpoint.
- `ActiveRun` is process-local and thread-backed. It cannot survive a process restart or be
  safely shared by multiple web workers.
- Artifact schemas are useful but not yet frozen as a versioned public contract.

## 3. Compatibility principles

### 3.1 One backend, multiple adapters

REST, MCP, WebMCP, CLI, and Streamlit must all delegate to the same run-service contract.
There must not be separate analysis implementations for different clients.

### 3.2 Preserve the seven-stage workflow

External clients may configure a run and observe it, but may not skip governance, invoke
`BaseTool.execute()` directly, or bypass `AgentController`. The controller must continue to
use `BaseTool.run()` and all existing safety checks.

### 3.3 Long-running work uses handles

Starting an analysis returns a `run_id`; it does not keep one HTTP/MCP request open until the
analysis finishes. Clients poll or subscribe to events and fetch the final result later.

### 3.4 Public content and private data are different trust zones

Marketing pages, docs, `llms.txt`, and API schemas are public. Uploaded datasets, objectives,
results, reports, models, and audit records are private and tenant-scoped by default.

### 3.5 Structured output is authoritative

External clients consume versioned JSON fields and artifact manifests. Human prose is a
presentation, not the only copy of a finding.

### 3.6 Agent tools are narrow and truthful

Tool names and descriptions must state side effects, limits, prerequisites, and whether an
operation is asynchronous. Do not expose a generic `run_python`, raw tool registry, arbitrary
path, or arbitrary URL-fetch capability.

### 3.7 Capability maturity is explicit

- REST/OpenAPI: production contract.
- Remote MCP: production agent protocol after REST stabilizes.
- WebMCP: progressive enhancement while browser support remains less universal.
- `llms.txt`: curated discovery aid, not a promise that every model vendor consumes it.

## 4. Phase 0 — product and policy decisions

Record these decisions before implementation because they change authentication and data
retention design:

1. **Public versus account-only use.** Recommended: public docs; authenticated analyses.
2. **Credential model.** Recommended: Sorrel owns server-side LLM credentials initially.
   Add bring-your-own-key only after secret lifecycle and support policy are defined.
3. **Retention.** Recommended default: uploads and derived artifacts expire after 24 hours;
   account owners may explicitly retain a run.
4. **Crawler policy.** Decide search discovery independently from model-training permission.
5. **Hosted code execution.** Recommended: `DSA_HOSTED=true`, code execution disabled by
   default; if enabled, require Docker isolation with no network.
6. **Anonymous quota.** Recommended: no anonymous analysis endpoint. A demo can run only the
   bundled sample in deterministic mode.
7. **API availability.** Decide whether this is a public developer API, private beta, or an
   internal protocol used only by first-party integrations.

Deliverable: `docs/AI_ACCESS_POLICY.md`, reviewed by the product and security owners.

## 5. Phase 1 — root-level discovery and crawlability

### 5.1 Add origin-root files

Serve these from the actual production origin:

```text
/robots.txt
/sitemap.xml
/llms.txt
/.well-known/security.txt
/openapi.json               once the API launches
/docs/                      public HTML documentation
/docs/*.md                  clean Markdown alternatives
```

Do not assume that adding them to the existing `static/` directory is enough. Streamlit is
configured to expose that directory under `/app/static`. Use one of:

- a reverse proxy/CDN rule serving a dedicated `public/` directory at the origin root;
- a small public FastAPI/ASGI layer mounted in front of Streamlit; or
- separate `www` and `app` hosts, with `www` serving discovery/docs and `app` serving the UI.

Recommended production host layout:

```text
https://sorrel.example/          public landing and docs
https://app.sorrel.example/      Streamlit workspace
https://api.sorrel.example/      REST, OpenAPI, MCP
```

One origin is acceptable for an MVP, but root paths must still be under deployment control.

### 5.2 Define crawler policy deliberately

Start from a documented policy rather than copying a broad `User-agent: *` block. For example:

```text
User-agent: *
Allow: /
Disallow: /api/
Disallow: /runs/
Disallow: /uploads/

User-agent: OAI-SearchBot
Allow: /

User-agent: GPTBot
Disallow: /

User-agent: Google-Extended
Disallow: /

Sitemap: https://sorrel.example/sitemap.xml
```

This example allows ordinary/search discovery while declining explicitly documented training
uses. It is a policy example, not a mandatory choice. Revisit crawler tokens quarterly.
Private content must remain behind authentication; `robots.txt` is not a security boundary.

### 5.3 Fix canonical and indexable metadata

- Replace the hard-coded `https://dsa-agent.local/` canonical and Open Graph URL with the real
  deployment origin supplied through configuration.
- Put the canonical metadata in the top-level indexable response, not only an iframe.
- Ensure the top-level response has a stable title, description, canonical URL, language,
  Open Graph data, and social preview image.
- Keep JSON-LD aligned with visible content. Use `SoftwareApplication` or `WebApplication` for
  the product, `Organization` only when a real organization identity exists, and `WebAPI`
  documentation when the public API exists.
- Never mark uploaded user datasets or private run artifacts as indexable.

### 5.4 Publish a real sitemap

Include only canonical public pages: product overview, how it works, safety/privacy, supported
data formats, API quickstart, API reference, MCP setup, limits, changelog, and status/contact.
Exclude login pages, temporary artifacts, API operations, uploaded files, and run URLs.

### Phase 1 acceptance

- Root discovery files return `200` without JavaScript, cookies, or authentication.
- Correct `Content-Type` and UTF-8 charset are present.
- No production metadata contains `.local`, localhost, or temporary tunnel URLs.
- Search crawlers can read public content; private routes return authorization errors.
- CI validates `robots.txt`, parses the sitemap, and checks that every listed URL is canonical.

## 6. Phase 2 — `llms.txt` and AI-readable documentation

### 6.1 Create a concise `/llms.txt`

Follow the current proposal: one H1, a short blockquote summary, optional explanatory text,
then H2 sections containing described links. Keep it curated and small.

Proposed content shape:

```markdown
# Sorrel

> Sorrel is an evidence-first data-analysis service. It accepts tabular datasets,
> executes a governed seven-stage workflow, and returns checked findings, charts,
> reports, limitations, and an audit trail.

Use the API or MCP endpoint for actions. Public documentation never grants access to
uploaded datasets. Analysis runs are asynchronous and return a run identifier.

## Start Here

- [Product overview](https://sorrel.example/docs/overview.md): What Sorrel does and does not do.
- [Analysis workflow](https://sorrel.example/docs/workflow.md): The seven stages and their outputs.
- [Safety and privacy](https://sorrel.example/docs/safety.md): Data handling, sandboxing, PII, and retention.

## Integrate

- [API quickstart](https://sorrel.example/docs/api-quickstart.md): Upload, start, poll, and fetch a result.
- [OpenAPI schema](https://api.sorrel.example/openapi.json): Versioned HTTP contract.
- [MCP setup](https://sorrel.example/docs/mcp.md): Remote MCP endpoint, tools, and authentication.

## Reference

- [Supported files](https://sorrel.example/docs/file-formats.md): Accepted formats and upload limits.
- [Result schema](https://sorrel.example/docs/result-schema.md): Findings, evidence, caveats, and artifacts.
- [Limits](https://sorrel.example/docs/limits.md): Runtime, concurrency, quotas, and model behavior.

## Optional

- [Changelog](https://sorrel.example/docs/changelog.md): Contract and product changes.
```

Do not place secrets, private endpoints, internal paths, system prompts, operational hostnames,
or unsupported marketing claims in this file.

### 6.2 Publish clean Markdown alternatives

Every key documentation page should have a clean Markdown representation, preferably at the
same path with `.md` appended or substituted. HTML responses should advertise it with:

```html
<link rel="alternate" type="text/markdown" href="/docs/overview.md">
<link rel="describedby" href="/llms.txt">
```

Equivalent HTTP `Link` headers are acceptable and are preferable for non-HTML responses.

Write pages for retrieval quality:

- Put the direct answer and product limits first.
- Use one concept per heading and stable anchor IDs.
- Define terminology once and use it consistently.
- Include compact request/response examples.
- State version and last-updated date.
- Use absolute canonical links in `llms.txt`.
- Avoid navigation boilerplate, animated-only explanations, and claims visible only in images.

### 6.3 Generate, do not hand-drift

Create a build/check script that:

- validates the `llms.txt` grammar and linked URLs;
- ensures Markdown alternatives exist;
- ensures every public API operation appears in the API reference;
- rejects localhost, `.local`, tunnel, or staging URLs in production assets;
- reports broken anchors and redirects;
- optionally derives link lists from a small reviewed manifest.

The descriptions themselves remain human-reviewed. Fully generating them from route names
would produce low-quality agent context.

### Phase 2 acceptance

- An agent given only `/llms.txt` can answer a fixed evaluation set about capabilities,
  limits, privacy, supported inputs, API startup, and MCP setup.
- The agent does not infer that browsing docs grants access to private data.
- Every non-optional link resolves directly to useful Markdown or a machine-readable schema.
- `/llms.txt` remains below a documented context-size budget.

## 7. Phase 3 — framework-neutral run service

Before exposing HTTP, move reusable run behavior out of the `ui` namespace. This is not a new
analysis tool and must not modify the RLM engine or task decomposition logic.

Proposed modules:

```text
src/service/
├── __init__.py
├── contracts.py       public dataclasses/enums and schema versions
├── run_service.py     create, inspect, cancel, and finalize runs
├── run_worker.py      AgentController adapter and callbacks
├── repository.py      run/dataset metadata persistence interface
├── artifact_store.py  tenant-safe artifact lookup and retention
└── events.py          structured event contract
```

Responsibilities:

- Create cryptographically random, opaque `dataset_id` and `run_id` values.
- Validate ownership on every lookup.
- Allocate a unique output directory per run.
- Reuse `validate_upload()` before any file parser sees bytes.
- Construct `RunConfig` without logging credentials.
- Call `AgentController.load_dataset(..., interactive=False)` and then `analyze()`.
- Convert callbacks into typed events.
- Snapshot `RunView`, tool results, governance, usage, and artifact metadata.
- Support cooperative cancellation.
- Persist state transitions: `queued`, `running`, `succeeded`, `failed`, `cancelled`, `expired`.
- Make terminal states immutable.

### 7.1 Durable state

MVP: SQLite metadata plus filesystem/object-storage artifacts, one worker process.  
Production: PostgreSQL metadata, object storage, and a worker queue. Avoid binding the public
contract to a particular queue implementation.

Required records:

- `principals`: account/service identity and status.
- `api_credentials`: hashed credential identifier, scopes, timestamps; never the plaintext key.
- `datasets`: owner, safe filename, media type, byte size, checksum, state, expiry.
- `runs`: owner, dataset ID, objective, options, state, timestamps, schema version, error code.
- `run_events`: sequence number, type, stage, timestamp, safe payload.
- `artifacts`: run ID, logical name, media type, byte size, checksum, storage key, expiry.

Do not store the per-run LLM API key in these tables.

### 7.2 Event contract

Each event should have:

```json
{
  "schema_version": "1.0",
  "run_id": "run_...",
  "sequence": 17,
  "time": "2026-10-12T12:34:56Z",
  "type": "tool.completed",
  "stage": 3,
  "iteration": 2,
  "status": "running",
  "message": "Statistical comparison completed",
  "data": {}
}
```

Do not put raw rows, complete DataFrames, LLM prompts, secrets, stack traces, or generated code
in externally visible events.

### 7.3 Artifact manifest

Freeze a manifest rather than exposing filesystem paths:

```json
{
  "schema_version": "1.0",
  "run_id": "run_...",
  "artifacts": [
    {
      "name": "analysis-report",
      "media_type": "text/markdown",
      "bytes": 48122,
      "sha256": "...",
      "download_url": "/v1/runs/run_.../artifacts/analysis-report"
    }
  ]
}
```

### Phase 3 acceptance

- Existing Streamlit behavior can be driven through the service without importing Streamlit.
- Two simultaneous runs cannot read or overwrite each other's inputs, models, or reports.
- A process restart preserves discoverable run state; interrupted runs become a documented
  terminal/recoverable state.
- State-transition and tenant-isolation tests cover every method.

## 8. Phase 4 — REST API and OpenAPI

Use FastAPI as a thin transport adapter over `src/service/`. FastAPI/Pydantic models may validate
HTTP shapes but must not contain analysis logic.

Proposed layout:

```text
src/api/
├── app.py
├── dependencies.py
├── errors.py
├── schemas.py
└── routes/
    ├── health.py
    ├── datasets.py
    ├── runs.py
    └── artifacts.py
```

### 8.1 Versioned endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Process health only; no secrets or dependency detail |
| `GET` | `/readyz` | Worker/store readiness for infrastructure |
| `POST` | `/v1/datasets` | Multipart dataset upload; returns `dataset_id` |
| `GET` | `/v1/datasets/{dataset_id}` | Upload state and safe metadata |
| `DELETE` | `/v1/datasets/{dataset_id}` | Delete an unused/expired owned dataset |
| `POST` | `/v1/runs` | Validate request, enqueue analysis, return `202` |
| `GET` | `/v1/runs/{run_id}` | State, stages, progress, and timestamps |
| `GET` | `/v1/runs/{run_id}/events` | Server-Sent Events with resumable sequence ID |
| `GET` | `/v1/runs/{run_id}/result` | Final structured result; `409` until terminal success |
| `GET` | `/v1/runs/{run_id}/artifacts` | Artifact manifest |
| `GET` | `/v1/runs/{run_id}/artifacts/{name}` | Authorized artifact download |
| `POST` | `/v1/runs/{run_id}/cancel` | Cooperative cancellation; idempotent |
| `GET` | `/v1/capabilities` | Formats, limits, schema versions, feature switches |

Do not accept a server-side filesystem path or arbitrary remote URL in public request bodies.
Remote URL ingestion creates SSRF, DNS-rebinding, credential-forwarding, and unbounded-download
risks. Add it only with a separate fetcher, allowlist/policy, redirect checks, private-address
blocking, size limits, and content verification.

### 8.2 Start-run request

```json
{
  "dataset_id": "ds_...",
  "objective": "Identify the strongest evidence-backed drivers of churn.",
  "target": "churn",
  "options": {
    "use_llm": true,
    "use_ml": true,
    "enable_rlm": true,
    "max_iterations": 8
  },
  "idempotency_key": "client-generated-unique-value"
}
```

Operator policy must clamp all options. A client cannot enable a capability disabled by the
server, increase a limit beyond its quota, choose an arbitrary output directory, or override
sandbox policy.

### 8.3 Response rules

- `202 Accepted` for queued work, with `Location: /v1/runs/{run_id}`.
- One error envelope: `code`, `message`, `request_id`, and optional safe `details`.
- Stable machine error codes such as `unsupported_file`, `quota_exceeded`,
  `dataset_expired`, `run_not_ready`, and `isolation_unavailable`.
- No Python class names, local paths, stack traces, SQL, prompts, or credentials in responses.
- `Idempotency-Key` support for upload finalization, run creation, and cancellation.
- Pagination and maximum page size for every collection.
- UTC ISO-8601 timestamps and explicit schema versions.

### 8.4 Authentication and authorization

Private beta:

- Generate high-entropy API keys once; store only a salted/peppered hash and a short key ID.
- Accept `Authorization: Bearer ...` only over HTTPS.
- Scopes: `datasets:write`, `datasets:read`, `runs:write`, `runs:read`, `runs:cancel`,
  `artifacts:read`.
- Resolve the authenticated principal once and apply ownership filtering inside repositories,
  not only route handlers.

Public/multi-user product:

- Add OAuth 2.1/OIDC authorization-code flow with PKCE.
- Use short-lived access tokens and refresh-token rotation.
- Keep service accounts separate from end-user identities.
- Require explicit user consent for third-party agent access to private data.

### 8.5 OpenAPI quality

- Pin the public schema version and check it into source control as a generated artifact.
- Give every operation a stable `operationId` suitable for generated clients and LLM tools.
- Include realistic examples and all terminal error responses.
- Mark write/cancel endpoints clearly; do not disguise side effects as `GET`.
- Generate at least one typed client in CI to catch invalid schema changes.
- Run an OpenAPI breaking-change check on pull requests.

### Phase 4 acceptance

- A client can upload, start, observe, cancel, fetch a result, and download an artifact without
  browser automation.
- OpenAPI validation and generated-client smoke tests pass.
- Cross-tenant access returns the same safe not-found response used for nonexistent IDs.
- Load tests prove queue and concurrency limits; excess work receives backpressure rather than
  starting unlimited threads.

## 9. Phase 5 — remote MCP server

Build the remote MCP endpoint only after service and REST schemas stabilize. Use the official
Python MCP SDK to match the Python backend and expose Streamable HTTP at a stable `/mcp` URL.

Proposed module:

```text
src/mcp_server/
├── app.py
├── auth.py
├── schemas.py
├── tools.py
└── resources.py
```

This directory contains protocol adapters, not Execution Layer tools from `src/tools/`.

### 9.1 Initial MCP tools

| Tool | Behavior | Side effect |
| --- | --- | --- |
| `get_sorrel_capabilities` | Formats, limits, enabled features, schema versions | None |
| `create_dataset_upload` | Creates a short-lived, one-use upload target | Allocates upload |
| `start_data_analysis` | Starts the governed seven-stage run for a `dataset_id` | Creates run/cost |
| `get_analysis_status` | Returns stage/status and recent safe events | None |
| `get_analysis_result` | Returns final structured findings and caveats | None |
| `list_analysis_artifacts` | Returns typed artifact resources | None |
| `cancel_data_analysis` | Requests cooperative cancellation | Changes run state |

Avoid passing dataset bytes as large base64 strings in normal MCP JSON. Prefer a short-lived
upload URL or a host-supported file/resource handoff. Validate the bytes after upload exactly as
the website does.

Do not initially expose:

- individual analysis tools;
- dynamic code execution;
- generated-tool definition;
- filesystem paths;
- arbitrary SQL or Python;
- arbitrary URL fetches;
- raw audit logs containing code or prompt text;
- deletion of retained runs.

### 9.2 MCP resources

Use read-only resources for completed artifacts and documentation, for example:

```text
sorrel://docs/integration
sorrel://runs/{run_id}/result
sorrel://runs/{run_id}/report.md
sorrel://runs/{run_id}/dashboard.json
```

Resource access must perform the same authorization check as REST. Resource URIs are opaque
references, not proof of permission.

### 9.3 Tool schema and instructions

- Use full input and output schemas; avoid unstructured JSON blobs where a stable model exists.
- Put shared workflow instructions in server initialization instructions, kept concise.
- State that start/cancel tools have side effects.
- State that provisional findings can change until the run succeeds.
- Return `run_id` as the explicit handle passed to later calls.
- Keep tool list order deterministic and tool names version-stable.
- Provide structured content plus a compact human-readable summary.
- Use safe annotations/hints supported by the negotiated MCP version.

### 9.4 MCP authentication

- Development/private use may use a scoped bearer token.
- Public third-party use should implement OAuth 2.1 with PKCE and per-user consent.
- Validate issuer, audience, expiry, scopes, and redirect URIs.
- Never treat the MCP client identity as the end-user identity without an explicit delegation.
- Keep ChatGPT/Codex-specific extensions optional; the core server must work with other compliant
  MCP hosts.

### 9.5 MCP conformance and evaluations

- Validate with MCP Inspector.
- Run protocol conformance tests for initialization, discovery, tools, resources, errors,
  cancellation, authentication failure, and schema validation.
- Test at least ChatGPT/Codex and one non-OpenAI MCP host.
- Maintain direct, indirect, adversarial, ambiguous, and out-of-scope prompt cases.
- Verify the model does not call `start_data_analysis` when the user only asks what Sorrel does.
- Verify cancellation and cost-producing actions receive the host's expected confirmation.

### Phase 5 acceptance

- A fresh MCP client can discover the server, authenticate, upload through the approved flow,
  start a run, poll it, and retrieve evidence-backed results.
- No MCP operation can exceed REST permissions or server-side policy.
- Removing MCP leaves the REST API and Streamlit UI unaffected.

## 10. Phase 6 — WebMCP progressive enhancement

WebMCP targets agents operating inside a user's browser session. Treat it as an enhancement,
not the primary integration surface, until browser support and the specification are stable.

Good first browser tools:

- `select_sample_dataset`
- `set_analysis_objective`
- `configure_analysis_options`
- `start_visible_analysis`
- `get_visible_run_status`
- `open_result_section`
- `request_visible_run_cancel`

Rules:

- Declarative tools should wrap ordinary accessible forms wherever possible.
- Imperative tools must call the same UI/service actions as human controls.
- The visible page remains the source of user context; do not silently operate another account
  or hidden run.
- Starting/cancelling an analysis must create clear visible feedback.
- Never expose API keys, local file paths, hidden widget state, or arbitrary JavaScript execution.
- Preserve keyboard, screen-reader, and non-agent operation.
- Feature-detect WebMCP; the page must work when it is absent.

Because the current landing page lives in a Streamlit component iframe, prototype WebMCP first
in the workspace surface where the actual form and session exist. Confirm origin and iframe
permission behavior before committing to an iframe-based implementation.

### Phase 6 acceptance

- A supporting browser agent can perform the same documented workflow as a user without DOM
  guesswork.
- Every agent action is visible, attributable, and governed by the active user session.
- Standard browser and accessibility tests remain unchanged when WebMCP is unavailable.

## 11. Phase 7 — optional ChatGPT/MCP Apps UI

After the tool-only MCP server is reliable, optionally add a portable MCP Apps component for:

- upload progress;
- seven-stage run progress;
- finding cards with evidence state;
- charts and report links;
- cancellation and retry affordances.

Keep it optional. The MCP tools must remain fully usable without custom UI. Start with the open
MCP Apps bridge and use vendor-specific extensions only for capabilities the portable standard
cannot provide. Apply a strict component CSP and allow only required origins.

Do not embed the full Streamlit application inside the MCP app. Build a small, task-specific UI
from the same structured result contract.

## 12. Phase 8 — security, privacy, and abuse controls

### 12.1 Upload boundary

- Reuse `src/core/security.py`; do not implement weaker API-only validation.
- Stream uploads to a bounded quarantine file; do not read unbounded request bodies into memory.
- Enforce compressed and expanded size, file count, nesting, parse time, and row/column limits.
- Verify extension, MIME type, signature, and parser compatibility.
- Compute a checksum and malware-scan where hosting policy requires it.
- Use storage-generated paths only; never use a client filename as a path.
- Reject archives with paths, links, devices, or multiple ambiguous datasets.

### 12.2 Agent and prompt boundary

- Treat objective text, column names, file contents, MCP arguments, and fetched docs as untrusted
  data.
- Continue using metadata-only prompts, PII masking, and `sanitize_for_prompt`.
- Do not let `llms.txt`, Markdown docs, or MCP descriptions contain instructions that override
  user consent or server policy.
- Keep generated code inside the existing sandbox and execution governor.
- Refuse code execution if hosted isolation is required but unavailable.

### 12.3 Authorization boundary

- Authorize every dataset, run, event stream, and artifact access.
- Use opaque identifiers with no owner information.
- Never trust a storage URL supplied by a client.
- Short-lived download URLs must be bound to an artifact, principal, and expiry.
- Prevent browser CSRF when cookie authentication is introduced; bearer-only API clients should
  not use ambient cookies.

### 12.4 Resource controls

- Per-principal request, upload-byte, stored-byte, concurrent-run, LLM-token, sandbox-execution,
  and daily-cost quotas.
- Global emergency kill switches for LLM calls, code execution, uploads, and new runs.
- Queue backpressure with clear retry metadata.
- Hard timeouts and memory limits inherited by workers and subprocesses.
- No automatic retries of non-idempotent, cost-producing operations without an idempotency key.

### 12.5 Audit and observability

Record:

- request ID, principal ID, credential ID, operation, run ID, outcome, latency, bytes, and quota;
- run state changes and cancellation source;
- MCP initialization/tool calls without secret arguments;
- authentication failures and authorization denials;
- artifact downloads;
- schema/client version for interoperability debugging.

Never record plaintext API keys, authorization headers, signed URLs, raw datasets, PII values,
full prompts by default, or secrets embedded in errors. Maintain operator audit logs separately
from user-downloadable analysis audits.

## 13. Public result contract

The first public result version should contain:

```text
schema_version
run_id
status
objective
dataset_summary       counts and safe metadata, never raw rows
analysis_mode
findings[]            ID, kind, headline, explanation, importance, confidence
evidence[]            method, values, checks, caveats, provenance
recommendations[]
coverage
limitations[]
governance            safe summary of LLM/code/privacy controls
usage                 calls/tokens/cost estimate when available
artifacts[]            logical references, not filesystem paths
created_at/completed_at
```

Versioning rules:

- Additive optional fields are allowed within v1.
- Renames, changed meanings, required-field changes, or enum removals require a new major version.
- Finding and artifact IDs remain stable for the life of a run.
- Unknown enum values must not crash generated clients.
- Do not expose internal `ToolResult` dictionaries as the public contract; translate them.

## 14. Repository work map

Expected additions:

```text
public/
├── robots.txt
├── sitemap.xml
├── llms.txt
└── security.txt

docs/public/
├── overview.md
├── workflow.md
├── safety.md
├── api-quickstart.md
├── mcp.md
├── file-formats.md
├── result-schema.md
└── limits.md

src/service/             shared run service
src/api/                 REST/OpenAPI adapter
src/mcp_server/          remote MCP adapter
tests/test_service_*.py
tests/test_api_*.py
tests/test_mcp_*.py
tests/test_ai_discovery.py
tests/contract/          checked OpenAPI/MCP fixtures and compatibility tests
scripts/check_ai_surface.py
```

Likely modifications:

- deployment/reverse-proxy configuration for origin-root files and HTTPS routes;
- `ui/run.py`, after service extraction, to import rather than own backend run execution;
- landing metadata and production-origin configuration;
- dependency files for optional `api` and `mcp` extras;
- README integration instructions and environment variable documentation.

Ask first before modifying `src/rlm/engine.py`, task decomposition, or the `MemorySystem` schema.
This plan does not require any of those changes.

## 15. Testing strategy

### 15.1 Static discovery tests

- Parse `llms.txt` and require the documented order and one H1.
- HEAD/GET every linked resource and validate media type.
- Parse `sitemap.xml`; reject private, duplicate, redirected, noncanonical, and staging URLs.
- Parse `robots.txt` and assert the chosen crawler matrix.
- Validate canonical, Markdown alternate, and `describedby` links.
- Validate JSON-LD syntax and correspondence with visible content.

### 15.2 Service tests

- Unit-test every new utility/function as required by repository policy.
- State-machine transitions, idempotency, cancellation, expiry, restart recovery.
- Parallel run isolation and artifact ownership.
- Upload validation parity with Streamlit.
- No secret values in repr, logs, events, errors, or persisted records.
- Deterministic offline end-to-end run over the bundled sample CSV.

### 15.3 API tests

- OpenAPI schema validation and breaking-change detection.
- Authentication and every scope combination.
- Cross-tenant object reference tests.
- Multipart size/format/decompression adversarial cases.
- SSE ordering, reconnect with last event ID, and terminal close behavior.
- Rate limits, quota exhaustion, backpressure, and safe errors.
- Generated Python and TypeScript client smoke tests.

### 15.4 MCP tests

- Protocol initialization and capability negotiation.
- Deterministic tool discovery and schema validation.
- Tool happy paths, invalid arguments, unauthorized access, wrong-owner handles, expired uploads,
  unfinished results, cancellation, and server failures.
- MCP Inspector and official conformance suite where available.
- Prompt-based evals for tool selection, no-op informational queries, confirmation-sensitive
  actions, prompt injection, and out-of-scope requests.

### 15.5 Browser-agent tests

- Progressive feature detection.
- WebMCP action parity with visible controls.
- Authentication/session boundaries and visible feedback.
- Playwright coverage for ordinary human operation, browser-agent operation, and unsupported
  browsers.
- Accessibility checks before and after tool registration.

### 15.6 Repository-wide gates

Every implementation phase must finish with:

```text
ruff check .
mypy src/
pytest tests/ -v
```

Before release, also run a seven-stage sample analysis and verify a populated Markdown report in
the per-run artifact directory. Scan tracked files for credential patterns.

## 16. Rollout sequence

### Milestone A — AI-readable public web

- Production origin configuration.
- Top-level crawlable landing/docs shell.
- Correct canonical metadata and JSON-LD.
- `robots.txt`, `sitemap.xml`, `/llms.txt`, and Markdown docs.
- Discovery CI and evaluation set.

This is low-risk and can ship before the callable API.

### Milestone B — private REST beta

- Shared run service and durable IDs/state.
- Authenticated upload/run/result/artifact endpoints.
- OpenAPI and generated-client smoke test.
- One worker, explicit concurrency and retention limits.

Invite a small set of API users; do not advertise public MCP yet.

### Milestone C — remote MCP beta

- Tool-only MCP server over the shared service.
- Bearer auth for private testers, then OAuth for third-party user delegation.
- Two-host interoperability testing and agent eval suite.

### Milestone D — browser and embedded agent experience

- WebMCP progressive enhancement for visible workspace actions.
- Optional MCP Apps result/progress component.
- Broader crawler policy and documentation review.

### Milestone E — production hardening

- Multi-worker queue and production metadata/object stores.
- Full quotas, billing/cost policy, incident controls, retention jobs, SLIs/SLOs, backup/restore,
  and documented deprecation process.
- Public integration docs and changelog.

## 17. Metrics

Track metrics that distinguish discovery from successful use:

- `/llms.txt` requests, linked Markdown fetches, status codes, and user agents.
- Public documentation referrals and citations where observable.
- API time to first successful run and integration failure reasons.
- MCP initialization success, tool-call success, schema failures, and authentication failures.
- Correct tool-selection rate on the eval set.
- Median queue time, time to first provisional finding, total runtime, cancellations, and failures.
- Cross-client compatibility pass rate.
- Security signals: rejected uploads, quota blocks, authorization denials, and isolation failures.

Do not use crawler volume alone as a success metric. The meaningful outcome is an external agent
correctly understanding Sorrel's limits and completing an authorized analysis.

## 18. Definition of done

Sorrel is externally AI-compatible when all of the following are true:

1. Public agents can discover and understand the product from stable root-level and Markdown
   resources without running the Streamlit UI.
2. Crawler/search and training-use policies are explicit and independently configurable.
3. A non-browser client can complete the full upload-to-result workflow through a versioned,
   authenticated REST contract.
4. At least two independent MCP hosts can complete the same workflow through a stable remote MCP
   endpoint.
5. REST, MCP, WebMCP, CLI, and Streamlit all reach the same `AgentController` workflow and safety
   controls.
6. Every private object is tenant-scoped, every state-changing action is auditable, and secrets
   never enter results or logs.
7. Long-running jobs survive ordinary web request lifetimes and expose typed progress,
   cancellation, terminal errors, and durable artifacts.
8. Static discovery, unit, contract, security, end-to-end, lint, and type-check gates pass.

## 19. Explicit non-goals for the first release

- Making every internal analysis tool independently callable by external agents.
- Accepting arbitrary remote dataset URLs.
- Allowing anonymous users to run paid LLM or sandbox workloads.
- Giving an MCP client raw filesystem, SQL, Python, prompt, or model-pickle access.
- Treating `llms.txt` as an SEO guarantee or crawler permission mechanism.
- Depending on WebMCP as the only agent interface.
- Rewriting the RLM engine, task decomposition, or MemorySystem schema.

## 20. First implementation slice

The safest useful first pull request should contain only Milestone A:

1. production-origin configuration;
2. corrected top-level canonical/metadata ownership;
3. `public/robots.txt`, `public/sitemap.xml`, and `public/llms.txt`;
4. six to eight reviewed Markdown documentation pages;
5. `rel="alternate"` and `rel="describedby"` delivery;
6. `scripts/check_ai_surface.py`;
7. unit/contract tests for every discovery artifact;
8. deployment notes explaining how root files are served ahead of Streamlit.

The second pull request should extract the framework-neutral run service without changing
analysis behavior. The third should add the private REST beta. MCP begins only after that contract
has passed integration testing.
