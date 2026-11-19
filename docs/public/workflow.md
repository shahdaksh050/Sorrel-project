# Five-Stage Analysis Workflow

Version: 1.0.1
Last Updated: 2026-11-19
Canonical: /docs/workflow.md

Sorrel analyzes tabular data in five stages.

1. **Dataset ingestion and profiling** — load, coerce, and profile the dataset.
2. **Initial reasoning** — produce an initial analysis plan using the configured
   LLM or deterministic fallback.
3. **Tool execution and sandbox verification** — validate and run registered
   execution tools.
4. **Interpretation and iterative refinement** — review the evidence, revisit
   open questions, and execute follow-up analyses where the data permits.
5. **Synthesis, post-hoc audits, and dossier generation** — compile local
   reports, charts, limitations, and governance details.

The Streamlit workspace starts a local analysis worker and reads its progress
events. The CLI runs the controller directly. A successful tool execution means
the calculation ran. A finding's audit describes the check that was applied to
its evidence. Question coverage records what the run could answer, and the
report lists remaining questions and execution issues.
