# Product Overview

Version: 1.0.0  
Last Updated: 2026-10-12  
Canonical: http://localhost:8501/docs/overview.md  

Sorrel is an evidence-first, autonomous data analysis engine. It ingests tabular datasets, executes a governed, iterative seven-stage analytical loop, and returns audited statistical findings, interactive charts, executive dossiers, and a reproducible computation ledger.

---

## What Sorrel Does

1. **Ingests and Profiles Tables**: Safely loads CSV, Parquet, TSV, and JSON tables; validates schema integrity, column distributions, outliers, missingness, and cardinality.
2. **Generates Empirical Hypotheses**: Uses domain reasoning and structured objectives to formulate verifiable hypotheses.
3. **Executes Sandboxed Tools**: Calls deterministic, type-safe Python tools for exploratory analysis, statistical hypothesis testing, machine learning pipelines, and causal inference.
4. **Checks Its Own Findings**: Automatically applies multiple-testing corrections (Benjamini-Hochberg FDR), statistical sensitivity analysis, adversarial counter-checks, and verbatim number verification.
5. **Produces Audited Dossiers**: Generates an executive summary, ranked evidence cards with clear confidence scores, quantified action items, and transparent limitations.

---

## What Sorrel Does Not Do

- **Does Not Hallucinate Claims**: Every reported number in a finding must trace back to audited tool execution output in memory. Unverified assertions are strictly rejected by the claim verifier.
- **Does Not Expose Raw Rows to LLMs**: Datasets are never sent in prompts. LLMs receive only schema profiles, aggregate statistics, and tool execution summaries.
- **Does Not Train on Customer Data**: Uploaded datasets, intermediate state, and reports are private, tenant-isolated, and never used for model training.
- **Does Not Run Unbounded Code**: Sandboxed execution enforces strict wall-clock timeouts (60 seconds), memory ceilings (1024 MB), and restricted module allowlists.

---

## Integration Interfaces

Sorrel is accessible across multiple interfaces that all converge on the same governed workflow:

- **Interactive Workspace**: Streamlit browser interface for human analysts.
- **REST API (Beta)**: Versioned JSON endpoints for asynchronous run creation, status polling, and artifact retrieval (`http://localhost:8501/docs/api-quickstart.md`).
- **Remote Model Context Protocol (MCP)**: Streamable HTTP tools for external autonomous agents such as ChatGPT, Claude, and Codex (`http://localhost:8501/docs/mcp.md`).
- **AI-Readable Index**: `/llms.txt` and clean Markdown documentation for autonomous discovery and reasoning.
