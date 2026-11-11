# Sorrel Overview

Version: 1.0.1
Last Updated: 2026-10-16
Canonical: /docs/overview.md

Sorrel is a local exploratory data-analysis application for tabular datasets.
Run it through Streamlit or the CLI. It profiles data, plans and executes
governed tools, and writes local reports and charts. Findings are analytical
evidence and hypotheses, not operational instructions or professional advice.

## Shipped interfaces

- Local Streamlit workspace: `streamlit run app.py`
- Local CLI: `python main.py --dataset <path>`

## Not shipped

This repository does not currently provide a REST API, MCP server, stable
remote result schema, remote authentication, tenant isolation, hosted quotas,
or automatic 24-hour retention. Those subjects remain roadmap work; see the
status pages for [REST](/docs/api-quickstart.md), [MCP](/docs/mcp.md), and
[result format](/docs/result-schema.md).

## Seven-stage workflow

1. Dataset ingestion and profiling
2. Initial reasoning
3. Tool execution and sandbox verification
4. Result interpretation
5. Iterative refinement and adversarial review
6. RLM workflow management
7. Synthesis, post-hoc audits, and dossier generation
