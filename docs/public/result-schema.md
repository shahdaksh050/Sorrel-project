# Result Format Status

Version: unreleased
Last Updated: 2026-10-16
Canonical: /docs/result-schema.md

Sorrel writes local analysis artefacts such as Markdown, JSON, HTML, charts, and
audit logs under the configured output directory. Their structure is an internal
implementation detail and is not a versioned remote result schema.

There is no `GET /v1/runs/{run_id}/result` endpoint and no MCP
`get_analysis_result` tool in this release. Consumers should use the local UI or
CLI outputs and should not depend on field names or terminal-status vocabulary
as a stable external contract.

A stable remote schema remains roadmap work and will be published with its
implementation and compatibility tests.
