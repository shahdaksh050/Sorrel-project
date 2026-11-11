# Seven-Stage Analysis Workflow

Version: 1.0.1
Last Updated: 2026-10-16
Canonical: /docs/workflow.md

Sorrel's local analysis workflow has seven stages:

1. **Dataset ingestion and profiling** — load, coerce, and profile the dataset.
2. **Initial reasoning** — produce an initial analysis plan using the configured
   LLM or deterministic fallback.
3. **Tool execution and sandbox verification** — validate and run registered
   execution tools.
4. **Result interpretation** — interpret tool output through the RLM engine.
5. **Iterative refinement and adversarial review** — refine the plan and review
   evidence.
6. **RLM workflow management** — manage recursive invocation and execution
   state.
7. **Synthesis, post-hoc audits, and dossier generation** — compile local
   reports, charts, limitations, and governance details.

The workflow is local to the application process. It does not expose remote
run handles, typed progress events, REST endpoints, or MCP tools.
