# Local Runtime Limits

Version: 1.0.1
Last Updated: 2026-10-16
Canonical: /docs/limits.md

These are defaults for the local application and CLI. They are configurable by
the operator and are not hosted-service quotas.

| Control | Default |
| --- | --- |
| Dataset read row cap | 1,000,000 (`DSA_MAX_ROWS`) |
| Maximum reasoning iterations | 15 (`MAX_ITERATIONS`) |
| Sandboxed code executions per analysis | 40 (`MAX_CODE_EXECUTIONS`) |
| Sandboxed code timeout | 45 seconds (`SANDBOX_TIMEOUT_S`) |
| Sandboxed code memory limit | 1024 MB (`SANDBOX_MEMORY_MB`) |

The repository does not currently implement per-tenant concurrency limits,
daily quotas, API rate limits, HTTP `429` responses, or automatic 24-hour data
retention. Local files remain where the operator configures them until the
operator removes them.
