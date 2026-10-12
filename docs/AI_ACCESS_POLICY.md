# Sorrel — AI Access, Crawler, and Security Policy

Status: Approved Operational Policy  
Version: 1.0.0  
Last Updated: 2026-10-12  
Product: Sorrel (Evidence-First Autonomous Data Analysis)  

---

## 1. Purpose and Scope

This document defines the formal access rules, privacy boundaries, credential governance,
and crawler directives for external AI agents, automated indexers, and developer clients
interacting with Sorrel.

Sorrel is designed to be easily discoverable and understandable by autonomous AI agents
while strictly isolating private user data, private datasets, and computational workflows
behind authenticated boundaries.

---

## 2. Policy Decisions

### 2.1 Public Discovery vs. Authenticated Workflows
- **Public Surface**: Product documentation, workflow descriptions, API schemas, `/llms.txt`,
  and sitemaps are completely public. They require no authentication, cookies, or JavaScript.
- **Private Surface**: Uploading tabular datasets, scheduling analysis runs, streaming
  execution events, retrieving statistical findings, downloading models, and accessing audit
  logs require explicit tenant authentication.
- **Zero Ambiguity Principle**: Public documentation describes capabilities and interfaces;
  browsing or ingesting public documentation grants zero privileges over private datasets.

### 2.2 Credential Governance
- **Server-Managed Credentials (Phase 1–3)**: In the initial deployment, Sorrel manages
  server-side LLM provider credentials internally. No external API client is required to
  supply their own LLM keys.
- **Tenant API Keys**: Client access to Sorrel is mediated by high-entropy, salted/hashed API
  keys (`Bearer sorrel_...`). Secret keys are displayed once at creation and never logged or
  stored in plaintext.
- **Bring-Your-Own-Key (BYOK, Future Scope)**: BYOK integration will be enabled only under
  formal tenant secret lifecycle management with per-run memory flushing.

### 2.3 Data Retention and Lifecycle
- **Default 24-Hour Expiry**: Uploaded datasets, temporary artifacts, and intermediate run
  scratchpads are automatically deleted 24 hours after run completion.
- **Explicit Retention**: Account owners may mark specific runs as retained for audit and
  reproducibility purposes.
- **No Data Training**: Customer datasets, queries, and generated analysis findings are
  **never used to train machine learning or AI models**, internally or externally.

### 2.4 Web Crawler & AI Scraper Directives
- **Search and Retrieval Agents**: Search engine crawlers (Googlebot, Bingbot, Slurp) and
  agentic search retrieval bots (`OAI-SearchBot`) are welcomed on all public paths (`/`, `/docs/*`).
- **AI Model Training Crawlers**: Crawlers that harvest web content for LLM pre-training or
  fine-tuning without attribution (`GPTBot`, `Google-Extended`) are explicitly disallowed.
- **Private Routes**: `/api/*`, `/runs/*`, and `/uploads/*` are disallowed to all automated
  crawlers in `robots.txt`.

### 2.5 Hosted Code Execution & Sandboxing
- When operating in hosted multi-tenant mode (`DSA_HOSTED=true`), code execution is disabled by
  default (`ENABLE_CODE_EXECUTION=false`).
- If code execution is enabled by tenant policy, execution is strictly quarantined inside
  rootless Docker containers with zero network access, memory RSS ceilings (1024 MB), wall-clock
  timeouts (60 s), and process-tree limits (`src/core/sandbox.py`).

### 2.6 Anonymous Quotas and Demos
- No anonymous analysis runs are permitted. Unauthenticated requests to analysis creation
  endpoints return HTTP `401 Unauthorized`.
- Public demonstrations operate strictly against pre-audited, bundled sample datasets
  (e.g., `AirQualityUCI.csv`) in offline, deterministic mode.

### 2.7 API & Protocol Availability
- Phase 1 (Milestone A): Public web discovery, `/llms.txt`, and clean Markdown documentation.
- Phase 2 (Milestone B): Private REST Beta with OpenAPI 3.1 specification.
- Phase 3 (Milestone C): Remote Model Context Protocol (MCP) server for agent tool calling.

---

## 3. Crawler Directives Matrix

| User-Agent | Public Web (`/`, `/docs/*`) | Discovery (`/llms.txt`, `sitemap.xml`) | Private (`/api/`, `/runs/`, `/uploads/`) |
| :--- | :--- | :--- | :--- |
| `*` (Default Search) | **Allow** | **Allow** | **Disallow** |
| `OAI-SearchBot` | **Allow** | **Allow** | **Disallow** |
| `GPTBot` | **Disallow** | **Disallow** | **Disallow** |
| `Google-Extended` | **Disallow** | **Disallow** | **Disallow** |
| `CCBot` | **Disallow** | **Disallow** | **Disallow** |

---

## 4. Security & Responsible Disclosure

If you identify a security vulnerability, data leakage, or sandboxing escape in Sorrel:
- Do not file public GitHub issues.
- Report all vulnerabilities according to [security.txt](file:///d:/(Dev2)DSA_AGENT/public/.well-known/security.txt) to `security@sorrel.ai`.
- PGP encryption keys and security advisories are maintained at `http://localhost:8501/.well-known/security.txt`.
