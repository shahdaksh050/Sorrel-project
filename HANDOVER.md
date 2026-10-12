# HANDOVER.md — Next Session Directives for Antigravity

**Date / Timestamp**: 2026-10-10T00:38:00+05:30  
**Branch**: `master` (shared working tree on `frontend-overhaul`)  
**Status**: Clean gates (`ruff`: 0 violations, `mypy`: 0 errors across 105 files, all 37 sandbox tests passing)

---

## 1. Context & Completed Work

Over the recent sessions, backend data quality, execution resilience, and sandbox stability were overhauled:
- **BE-1**: Pruned dead orphans (`SharedAnalysisContext`, `fast_groupby`/`quantiles`, `relational_joiner`); wired `question_router` into fallback plan.
- **BE-2**: Fixed cluster metric formatting (`e+` notation eliminated); steered LLM planner towards `segment_comparison`; added weak clustering silhouette gate (`< 0.35`).
- **BE-3**: Lightened sandbox worker subprocess (fast dtype builder, cached Win32 path resolution in audit hook, throttled psutil RSS checks, bumped default timeout to 60s).
- **BE-4**: Built self-healing fallback for failed sandbox groupby code to `segment_comparison`; enhanced categorical distribution previews with sample counts; enforced quantified actions.
- **BE-5**: Solved the `static_check: .read_csv is not allowed` issue via AST reader normalization (`_ReaderCallTransformer`, `_normalize_sandboxed_code`), transparently rewriting `pd.read_csv(...)` to pre-loaded in-memory `df` references.

---

## 2. Next Session Core Directives

The user reviewed the spec-to-code comparison and decided on two key initiatives for the next session:
1. **De-bloat the "RLM" architecture**: Scrap the pseudo-academic Stage 6 feature decomposition, consolidate token/cost telemetry into the LLM client, and streamline into a clean 5-stage pipeline.
2. **Execute the Top Backend & Security Audit Fixes** (held from earlier): Harden `file_path` containment, patch decompression bomb vectors, and close audit gaps.

---

## 3. Detailed Work Packages

### Phase 1: RLM De-bloating & Pipeline Streamlining

#### Problem Analysis
- "RLM" (Recursive Language Model) was designed around 2023-era 4k context windows.
- In this codebase, **Stage 6 (`_run_rlm_decomposition`)** chunks numeric columns into groups of 8 and fires parallel sub-calls (`RLMSubTask`) without tools, asking the LLM to speculate on column names.
- In practice, `_should_decompose()` auto-skips Stage 6 on ~90% of real runs because Stage 3 already covered the columns. When it does run, it burns extra tokens and latency for minimal analytical value.
- `REPLEnvironment` in `src/rlm/engine.py` is simply a plain dictionary (`self.variables = {}`), not an actual execution environment.

#### Actions
1. **Deprecate Stage 6 Feature Decomposition**:
   - In [`src/core/controller.py`](file:///d:/(Dev2)DSA_AGENT/src/core/controller.py): Remove `_run_rlm_decomposition()`, `_should_decompose()`, and `_rlm_skip_reason()`.
   - In [`src/core/prompt_manager.py`](file:///d:/(Dev2)DSA_AGENT/src/core/prompt_manager.py): Remove `_rlm_block()`, `RLM_SUBTASK_SYSTEM`, and subtask prompt templates.
2. **Consolidate Telemetry & Tracing**:
   - Preserve token accounting (`usage_summary()`), cost estimates (`COST_PER_1K_TOKENS`), and call traces (`_TraceEntry`).
   - Move or re-export these cleanly from `src/core/llm_client.py` or streamline `src/rlm/engine.py` into a lightweight `InferenceManager`.
   - Drop the unused `REPLEnvironment` key-value wrapper.
3. **Formalize the 5-Stage Workflow**:
   - Update [`AGENTS.md`](file:///d:/(Dev2)DSA_AGENT/AGENTS.md) and [`README.md`](file:///d:/(Dev2)DSA_AGENT/README.md) to replace the 7-stage waterfall with the honest 5-stage pipeline:
     - **Stage 1**: Dataset Ingestion & Profiling
     - **Stage 2**: Hypothesis & Action Plan Generation
     - **Stage 3**: Tool Execution & Sandbox Verification
     - **Stage 4**: Iterative Refinement & Adversarial Review
     - **Stage 5**: Synthesis, Post-Hoc Audits & Dossier Generation

---

### Phase 2: Top Audit Security & Correctness Fixes (from `docs/BACKEND_AUDIT_2026-10-04.md`)

| ID | Issue | Target File(s) | Fix Summary |
| :--- | :--- | :--- | :--- |
| **H-1** | Planner `file_path` is not strictly confined | `src/tools/base.py`, `src/core/controller_steps.py` | In `_resolve_file_path`, strictly enforce that `file_path` matches the loaded dataset, cleaned file, or a registered derived path; reject arbitrary paths. |
| **H-2 & H-3** | Decompression bomb and unvalidated related-table uploads | `src/core/io.py`, `src/core/security.py`, `app.py` | Add output size cap to `_decompress()` (e.g. 50MB); run `validate_upload()` on `.gz`/`.zip` archives and related-table joins. |
| **H-4** | `DSA_HOSTED=true` ignored in Streamlit | `app.py` | Add `load_dotenv()` at the very top of `app.py` before `_HOSTED` evaluation. |
| **H-5** | Sandbox static policy bypass via pandas expression evaluators | `src/core/sandbox.py`, `src/core/_sandbox_worker.py` | Add `"eval"`, `"query"`, `"pipe"` to `_BLOCKED_ATTRS`; deny `ctypes.call_function` in audit hook. |
| **H-7 & H-8** | Unsanitized prompt injection & PII leakage | `src/core/prompt_manager.py` | Wrap `_hypotheses_block` and chart-design category samples in `_sp()`, `_redact()`, and check `col.pii`. |
| **H-9 & H-10** | Controller completion guard & audit timing | `src/core/controller.py`, `src/core/controller_report.py` | Fix `controller.py:938-948` guard so premature completions trigger fallback planning; ensure contract audit runs after dashboard generation. |

---

## 4. Operational Rules & Constraints

1. **Execution Restrictions**:
   - **Do NOT run `streamlit run app.py` or full `pytest tests/` unless explicitly requested by the user.**
   - You may run specific unit tests (`pytest tests/test_sandbox.py`, `pytest tests/test_controller.py`) and `.venv\Scripts\ruff check .` / `.venv\Scripts\mypy src/`.
2. **Quality Gates**:
   - `ruff check .` must pass with **0 violations**.
   - `mypy src/` must pass with **0 errors across all source files**.
   - Line endings: convert files to LF (`\n`) before editing.
3. **Collaboration Protocol**:
   - Claude Code owns frontend files (`app.py`, `ui/styles.py`, `ui/tabs/answers_tab.py`). For H-4 in `app.py`, coordinate or make the single 1-line `load_dotenv()` fix with atomic precision.

---

## 5. Verification Checklist for Next Session

1. [ ] Stage 6 RLM feature decomposition pruned without breaking `AgentController` analysis loop.
2. [ ] Token/cost accounting and call tracing preserved and fully functional.
3. [ ] Audit fixes H-1, H-2, H-3, H-4, H-5, H-7, H-8, H-9, H-10 implemented with unit tests.
4. [ ] Documentation updated in `AGENTS.md`, `README.md`, and `GEMINI.md`.
5. [ ] `ruff check .` and `mypy src/` clean.

