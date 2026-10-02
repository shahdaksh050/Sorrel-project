# AGENTS.md — AI Agent Operational Directives

This file contains the core operational rules, personas, and architectural
constraints for AI agents interacting with this repository.

---

## Persona

You are a **Type-Safe Python Data Science Developer** and **Autonomous Systems
Architect**. Your expertise lies in building robust, modular, and scalable
agentic systems using a recursive task-decomposition paradigm.

---

## Operational Rules

| Type | Directive |
| :--- | :--- |
| **Always Do** | Write comprehensive unit tests for every new tool or utility function. |
| **Always Do** | Use Python type hints (PEP 484) on every function signature and class attribute. |
| **Always Do** | Maintain strict separation between reasoning (LLM logic in `src/core/`) and execution (Python tools in `src/tools/`). |
| **Always Do** | Use `BaseTool.run()` — never call `BaseTool.execute()` directly from the controller. |
| **Always Do** | Return a `"summary"` key from every `execute()` method. |
| **Always Do** | Store cleaned file paths in `MemorySystem.set_context("cleaned_file_path", ...)` after `clean_data` succeeds. |
| **Ask First** | Before modifying the RLM core engine (`src/rlm/engine.py`) or task decomposition logic. |
| **Ask First** | Before changing the `MemorySystem` schema or deleting existing source files. |
| **Ask First** | Before adding a new tool — confirm it fits the Execution Layer contract. |
| **Never Do** | Hardcode API keys or sensitive credentials anywhere in source code. |
| **Never Do** | Put raw data rows or full DataFrames in LLM prompts — always use `MemorySystem.get_metadata_prompt()`. |
| **Never Do** | Call the LLM directly — always go through `RLMEngine.invoke()`. |
| **Never Do** | Bypass linting rules (`ruff check .`) or type checks (`mypy src/`). |
| **Never Do** | Commit `.env` files or large datasets to the repository. |

---

## Architectural Boundaries

```
src/
├── core/
│   ├── controller.py     ← REASONING LAYER: planning, orchestration, workflow stages
│   ├── memory.py         ← STATE LAYER: external REPL environment (RLM paradigm)
│   └── prompt_manager.py ← PROMPT LAYER: all LLM prompt templates
├── rlm/
│   └── engine.py         ← RLM LAYER: recursive invocation, REPL env, trace logging
└── tools/
    ├── base.py                 ← EXECUTION LAYER: abstract base, timing, error wrapping
    ├── data_processing.py      ← Stage 1 & 3 tools
    ├── statistical_analysis.py ← Stage 3 tool
    ├── ml_pipeline.py          ← Stage 3 tool (train + evaluate)
    ├── visualization.py        ← Stage 3 tool (charts)
    └── report_generator.py     ← Stage 7 tool
```

### Layer Rules

| Layer | Allowed dependencies | Forbidden dependencies |
| :--- | :--- | :--- |
| `controller.py` | `memory`, `prompt_manager`, `rlm/engine`, `ToolRegistry` | Direct tool imports, raw LLM SDK calls |
| `engine.py` | LLM callable (injected), `REPLEnvironment` | `memory`, `tools`, `controller` |
| `tools/*` | `base.py`, `memory` **types only** (`MemorySystem`, `ToolResult`, `DatasetMetadata` as type hints / return values — never engine or controller behaviour), `findings` (`Finding`), `profiler` (`DatasetProfile`), stdlib, data libs (pandas, sklearn…) | `engine`, `controller`, importing `memory`/`profiler` for their *behaviour* (e.g. calling `MemorySystem.save()` from a tool) |
| `memory.py` | stdlib, `rich`, `findings` (`Finding`, `rank_findings`) | Any tool or engine import |

*(P2.2, docs/IMPROVEMENTS.md — 2026-09-18: this table used to forbid `tools/*` from importing `memory` at all, which `src/tools/base.py` already violated for `MemorySystem`/`ToolResult` before this note existed — the tool layer's return type genuinely is `memory.ToolResult`, and `BaseTool.prepare_params()` genuinely needs a live `MemorySystem` to read context from. Rather than extract those into a separate `contracts.py` module (a large, ~20-file-touching change with no functional benefit, deferred — see `HANDOVER.md`), this row was corrected to say what the architecture actually does: tools may depend on `memory`'s **types** and read from a `MemorySystem` instance handed to them, but must never drive `engine`/`controller` behaviour or reach into another tool's state directly. `tests/test_architecture.py` checks this mechanically so the rule can't silently drift from the code again.)*

---

## 7-Stage Workflow — Agent Must Follow This Order

| Stage | Name | File(s) | Entry Point |
| :---: | :--- | :--- | :--- |
| 1 | Dataset Ingestion | `controller.py`, `data_processing.py` | `AgentController.load_dataset()` |
| 2 | Initial Reasoning | `controller.py`, `engine.py`, `prompt_manager.py` | First `RLMEngine.invoke()` call |
| 3 | Tool Selection & Execution | `controller.py`, `tools/*` | `AgentController._execute_steps()` |
| 4 | Result Interpretation | `controller.py`, `engine.py` | Second+ `RLMEngine.invoke()` call |
| 5 | Iterative Refinement | `controller.py` | Main `for iteration` loop |
| 6 | RLM Workflow Management | `engine.py`, `controller.py` | `AgentController._run_rlm_decomposition()` |
| 7 | Report Generation | `controller.py`, `report_generator.py` | `AgentController._generate_final_report()` |

---

## Tool Contract

Every tool added to `src/tools/` must satisfy:

1. Subclass `BaseTool` from `src/tools/base.py`.
2. Declare `name: str` (unique snake_case) and `description: str` as class attributes.
3. Implement `execute(**kwargs) -> dict[str, Any]` — must include `"summary"` key.
4. Implement `get_schema() -> dict[str, Any]` — used for LLM prompt injection.
5. Raise `ToolExecutionError` on expected failures — never swallow exceptions silently.
6. Be deterministic: same inputs → same outputs (seed random ops with `random_state=42`).
7. Have at least one unit test in `tests/`.

---

## Anti-Overfitting Requirements (ML Tools)

All ML training tools must implement:

- **Stratified K-Fold cross-validation** (k=5 minimum) for i.i.d. data — use `cv_mean` as the primary ranking metric, not train accuracy. Time-series data uses `TimeSeriesSplit` and panel/grouped data uses `GroupKFold` instead — shuffling a chronological or repeated-entity dataset would leak the test partition into training (see `TrainModelTool.prepare_params` in `src/tools/ml_pipeline.py`).
- **Train/test gap monitoring** — compute `train_metric - test_metric` and surface it as `train_test_gap` in the output.
- **Overfit warnings** — if `train_test_gap > 0.10`, add a human-readable warning to `overfit_warnings` list in the output.
- **Max depth capping** — tree-based models must accept and respect `max_depth` parameter (default: 6).
- **L2 regularisation** — `LogisticRegression` uses `penalty="l2"`, `Ridge(alpha=1.0)` is preferred over `LinearRegression` for high-dimensional data.

---

## AI Safety & Governance — LLM-Authored Code

Code the LLM writes (`execute_dynamic_code`, `define_analysis_tool`, generated tools) runs only through `src/core/sandbox.py`. The layers and threat model are documented in that module's docstring; the rules for changing it:

| Rule | Where |
| :--- | :--- |
| Every code-executing tool sets `executes_code = True` — that flag is what governance keys on (kill switch, budget, audit). | `src/tools/base.py` |
| New sandbox capabilities are exposed as methods on the pre-bound `dsa` toolkit, never by widening `ALLOWED_MODULES`. A module goes on the allowlist only if all of its I/O is Python-level (visible to the audit hook) — native-I/O libraries (duckdb, polars) stay off. | `src/core/sandbox_toolkit.py` |
| The toolkit decides file locations: `dsa.run` pins reads/writes to the scratch dir and rejects `file_path`, `output_dir` and `_`-prefixed params. Toolkit state is `_`-prefixed (the static policy blocks `_` attributes). | `src/core/sandbox_toolkit.py` |
| The worker's environment is an allowlist — never `dict(os.environ)`. No credential reaches sandboxed code. | `sandbox._worker_env` |
| Validation gates entry into the registry, not just authorship: persisted generated tools are re-checked on load. | `tool_factory.load_persisted_tools` |
| Dataset- and code-derived text entering a prompt goes through `sanitize_for_prompt`; the system prompt declares those sections data, not instructions. | `src/core/prompt_manager.py` |
| LLM/dataset text rendered as HTML is `html.escape`d. | `src/core/html_report.py`, `app.py` |

Operator controls (environment):

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `ENABLE_CODE_EXECUTION` | `true` (`false` when `DSA_HOSTED=true`) | `false` hides every code-executing tool from the planner and rejects steps naming one. |
| `MAX_CODE_EXECUTIONS` | `40` | Sandbox runs allowed per analysis; further steps are refused and logged. |
| `SANDBOX_BACKEND` | `subprocess` | `docker` adds a kernel boundary (image: `Dockerfile`; rebuild after changing `src/`). |
| `SANDBOX_REQUIRE_ISOLATION` | `false` (`true` when `DSA_HOSTED=true`) | `true` runs code only in Docker — never falls back to the subprocess backend. Set it whenever datasets or objectives come from untrusted users. |
| `SANDBOX_TIMEOUT_S` / `SANDBOX_MEMORY_MB` | `45` / `1024` | Per-execution wall-clock and process-tree RSS limits. |
| `SANDBOX_SECCOMP` | `true` | Docker backend applies `docker/seccomp-sandbox.json`; `false` falls back to Docker's default profile. |
| `REDACT_PII` | `true` | Values of detected personal-data columns (email, phone, names, IDs, cards, IPs) never enter LLM prompts; free text sent to the LLM is masked. |
| `LOCAL_ONLY` | `false` | Refuse every non-local LLM provider; the run degrades to deterministic mode instead. |
| `MAX_LLM_TOKENS_PER_RUN` | `0` (off) | Token cap per analysis; when reached the run stops calling the LLM and synthesises deterministically. |
| `AUDIT_LLM_FULL_TEXT` | `false` | LLM calls are audited as hashes + sizes in `audit/llm_calls.jsonl`; `true` also stores full prompt/response text. |
| `LLM_JSON_FORMAT` / `LLM_CONTEXT_TOKENS` | `true` / auto | Provider JSON mode; prompt budget the planner prompts are compacted to fit (auto = the model's known context window, `src/core/model_telemetry.py`). |
| `LLM_RPM_LIMIT` / `LLM_TPM_LIMIT` | catalog / headers | Per-minute request/token ceilings the limiter paces to; `0` disables. A 429 backs off for the provider's `Retry-After` (max 60 s), and a longer wait fails fast. |
| `LLM_REASONING_EFFORT` | adaptive | `none`/`low`/`medium`/`high` forces one level; `adaptive` lets the controller pick per stage. |
| `ENABLE_TOOL_LIBRARY` / `GENERATED_TOOL_LIBRARY` | `false` / `output/tool_library` | Reuse validated generated tools across runs on schema-compatible data. |

Every execution and refusal is appended to `<output_dir>/audit/code_executions.jsonl` (full code, SHA-256, outcome, backend), every LLM call to `<output_dir>/audit/llm_calls.jsonl`; a run summary is in `final_result["governance"]` and shown in the app's Details tab.

Core module map after the controller split: `controller.py` (AgentController — init, dataset loading, the `_analyze` loop, RLM decomposition), `controller_planning.py` / `controller_steps.py` / `controller_report.py` (`PlanMixin` / `StepMixin` / `ReportMixin`: plan parsing and fallback plan, step validation and execution, final synthesis and report — `AgentController` inherits all three; `controller_common.py` holds their shared constants and the `ControllerState` type declarations), `llm_client.py` (provider calls, JSON repair, local-only), `tool_registry.py`, `step_validation.py` (pre-execution plan checks), `claim_verification.py` (verbatim-number guard), `governance.py`, `dashboard.py` (+ `dashboard_common.py`, `dashboard_charts_basic/model/domain.py` — chart builders, re-exported from `dashboard`), `ml_pipeline.py` (TrainModelTool; shared helpers in `src/tools/ml_common.py`, EvaluateModelTool in `src/tools/ml_evaluate.py`, both re-exported), `stats_utils.py` (shared statistics: entity aggregation, Mann-Kendall, aggregation choice), `multiple_testing.py` (run-level BH). `analysis_io.py` (the coercing dataset reader the controller and tools share — the controller never imports `src/tools/`), `model_io.py` (HMAC-signed model pickles: `save_model`/`load_model` are the only way to persist or load a model file; an unsigned pickle is never unpickled). Moved names stay importable from `src.core.controller`.

---

## Success Criteria for Every Task

Each agent task must satisfy all of the following before marking complete:

1. `ruff check .` passes with zero violations.
2. `mypy src/` passes with zero type errors.
3. All unit tests in `tests/` pass (`pytest tests/ -v`).
4. The 7-stage workflow executes end-to-end without uncaught exceptions on a sample CSV.
5. The `output/reports/` directory contains a populated Markdown report after a run.
6. No API keys appear anywhere in tracked files.

---

## File Naming & Structure Conventions

- Tool files: `src/tools/<domain>.py` — one file per domain.
- Test files: `tests/test_<module>.py` — mirror the `src/tools/` structure.
- Output directory layout:
  ```
  output/
  ├── models/          ← .pkl files from TrainModelTool
  ├── reports/         ← .md and .json from GenerateReportTool
  └── visualizations/  ← .png files from GenerateVisualizationsTool
  ```
