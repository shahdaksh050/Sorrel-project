# Safety, Privacy, and Governance

Version: 1.0.0  
Last Updated: 2026-10-12  
Canonical: http://localhost:8501/docs/safety.md  

Sorrel is designed around strict computational and data-handling guardrails. Untrusted datasets, external agent instructions, and LLM reasoning steps operate within fortified trust zones.

---

## 1. Zero Raw Data in Prompts

- **Metadata-Only Prompts**: Raw data rows and tabular contents are never injected into LLM prompts. The LLM receives only anonymized schema definitions, statistical summaries (quantiles, missingness rates), and tool result summaries.
- **Automatic PII Redaction**: Detected personal identifying information (e.g. emails, phone numbers, credit card tokens, social security IDs, IP addresses) is masked before metadata profiles reach the prompt layer (`REDACT_PII=true`).
- **Prompt Sanitization**: All user objectives and column headers undergo strict input sanitization to neutralize prompt injection or delimiter breakout attempts.

---

## 2. Sandboxed Code Execution

When dynamic data manipulation or exploratory calculations are required, code authored by the LLM is subject to multi-layered sandboxing:

- **AST Normalization & Static Analysis**: Before execution, code passes through an AST security scanner that rejects forbidden imports (`os`, `sys`, `socket`, `subprocess`, `requests`), filesystem calls, and private attribute access.
- **Kernel-Level Isolation**: In hosted environments (`DSA_HOSTED=true`), code executes inside non-root Docker containers with seccomp filtering and no network access.
- **Resource Ceilings**:
  - Wall-clock timeout: 60 seconds (`SANDBOX_TIMEOUT_S=60`).
  - Memory ceiling: 1024 MB RSS (`SANDBOX_MEMORY_MB=1024`).
  - Max executions per analysis: 40 runs (`MAX_CODE_EXECUTIONS=40`).
- **Immutable Audit Trail**: Every executed script, SHA-256 hash, execution duration, and memory footprint is logged to `output/audit/code_executions.jsonl`.

---

## 3. Machine Learning Governance

- **Anti-Overfitting Contract**: All model training pipelines enforce 5-fold cross-validation (`cv_mean`), track train/test performance gaps, and raise automated warnings if `train_test_gap > 0.10`.
- **HMAC Signed Artifacts**: Serialized model files (`.pkl`) are signed using HMAC-SHA256 upon creation. Pickles with missing or invalid signatures are rejected upon load, preventing arbitrary code execution during deserialization.

---

## 4. Data Privacy & Retention Policy

- **Ephemeral Retention**: Datasets uploaded via the REST API or MCP server, intermediate scratchpads, and execution artifacts expire and are purged after **24 hours** by default.
- **Zero Model Training**: Customer datasets and derived findings are strictly quarantined and never used to train or fine-tune proprietary or foundation AI models.
- **Tenant Isolation**: Every analysis allocates an isolated, cryptographically random workspace directory (`output/runs/<run_id>/`). Cross-tenant file lookups return safe `404 Not Found` responses.

---

## 5. Security Inquiries and Vulnerability Reporting

To report potential vulnerabilities, consult [security.txt](file:///d:/(Dev2)DSA_AGENT/public/.well-known/security.txt) or email `security@sorrel.ai`. Responsible disclosures are reviewed promptly by the security engineering team.
