# Safety, Privacy, and Governance

Version: 1.0.1
Last Updated: 2026-10-16
Canonical: /docs/safety.md

## Local-use boundary

This repository ships a local Streamlit/CLI application, not a hosted
multi-tenant service. The operator is responsible for the workstation,
configured output directory, access controls, backups, and deletion of local
datasets and artefacts. There is no implemented tenant-isolation layer or
automatic 24-hour retention/purge policy.

## Prompt and execution safeguards

- Raw data rows and full DataFrames are not placed in LLM prompts; the system
  uses metadata and result summaries, with configurable PII redaction.
- LLM-authored code uses the sandbox and configured timeout, memory, and
  execution-count controls. Docker isolation can be required by the operator.
- Model files are HMAC-signed before the application's model loader accepts
  them.

## Audit logs

Code-execution and LLM-call audits are local JSONL write logs. They record
events and write outcomes, but they are mutable local files: they are not an
immutable ledger, hash chain, signature service, or WORM storage guarantee.

## Security reporting

For vulnerability reports, use [security.txt](/.well-known/security.txt) or
email `security@sorrel.ai`.
