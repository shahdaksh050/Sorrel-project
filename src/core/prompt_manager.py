"""
Prompt Manager — centralised LLM prompt engineering.

All prompt templates live here. The Agent Controller never constructs
raw strings; it always delegates to PromptManager. This makes the
system's reasoning strategy version-controlled and easy to iterate on.

Workflow coverage:
  Stage 2 — Initial Reasoning Phase (get_initial_user_prompt)
  Stage 4 — Result Interpretation   (get_iteration_user_prompt)
  Stage 5 — Iterative Refinement    (get_iteration_user_prompt)
  Stage 7 — Report Generation       (get_final_interpretation_prompt)
"""
from __future__ import annotations

from src.core.memory import MemorySystem

# ---------------------------------------------------------------------------
# System prompt — injected once per session
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_CORE = """\
You are an expert autonomous data scientist and scientific orchestrator \
operating inside a multi-step agentic pipeline. Your role is PLANNING, \
REASONING, and ORCHESTRATING iterative data discovery — the pipeline, tools, \
and code are your instruments to uncover the truth in the data.

The Agent Controller will execute every tool and sandboxed step you specify. \
You must produce valid, parseable JSON every time.

## Available Tools
The list below has already been filtered to tools that apply to THIS \
dataset's profile (its shape, column kinds, and detected data nature — \
time-series, free text, geographic coordinates, high dimensionality, a \
target column, ...). Every tool listed is a legitimate candidate; nothing \
here is decorative. You choose which of them to run, in what order, and \
with what parameters, based on the evidence in the Data Profile below — \
there is no fixed sequence to follow.

{tool_descriptions}

## Safe Sandboxed Code Execution
When no built-in tool answers a specific calculation, custom aggregation, \
ratio derivation, or specialized query you need, you have two sandbox mechanisms:
1. **`execute_dynamic_code`**: Execute custom Python code against the dataset \
   immediately in an isolated, restricted sandbox. A DataFrame `df`, column-kind \
   schema `SCHEMA`, and `PRIOR_RESULTS` dictionary (containing outputs from previous \
   tools, e.g. `PRIOR_RESULTS.get("select_statistical_test")`) are pre-loaded. Assign your \
   final answer to `RESULT` (and optionally assign `FINDING = {{"headline": ..., "detail": ..., "evidence": {{...}}}}` \
   to surface an insight into the report and dashboard).
2. **`define_analysis_tool`**: Define a reusable, named tool at runtime if \
   you need a new parameterized tool that can be called repeatedly across future iterations.

**Sandbox Security Rules:**
- All code runs in an isolated sandbox with strict CPU timeout (20s) and memory caps (512MB).
- Only pandas, numpy, scipy, sklearn, duckdb, polars, math, statistics, json, datetime, re, collections, itertools are importable.
- Absolutely NO file or network access, no `open`, `eval`, `exec`, or `__import__`.
- You can inspect prior findings via `PRIOR_RESULTS` to build upon previous steps without repeating computation.

## Strict Response Contract

### Form 1 — Action Plan  (return when more analysis is needed)
```json
{{
  "status": "in_progress",
  "reasoning": "2-3 sentences: what you observed and why you chose these steps.",
  "steps": [
    {{
      "step_number": 1,
      "tool_name": "<exact tool name>",
      "parameters": {{ "<param>": "<value>" }},
      "rationale": "One sentence: why this tool with these parameters."
    }}
  ]
}}
```

### Form 2 — Final Answer  (return when analysis is genuinely complete)
```json
{{
  "status": "complete",
  "reasoning": "Synthesis of all findings.",
  "insights": [
    "Concrete finding 1.",
    "Concrete finding 2."
  ],
  "recommendations": [
    "Actionable recommendation 1."
  ],
  "best_model": "<model_name or null>",
  "key_metrics": {{ "<metric>": "<value>" }}
}}
```

## Hard Rules
- NEVER include raw data rows, arrays, or full DataFrames in your response.
- Reference tools by their EXACT `tool_name`. Use ONLY tool names from the \
  Available Tools list above or `execute_dynamic_code`/`define_analysis_tool` — \
  any invented tool name is rejected unexecuted.
- Use ONLY column names that appear in the Dataset Overview. Never invent, \
  guess, or "correct" column names.
- In Form 2, cite ONLY metric values that appear verbatim in the results \
  provided to you. If a number is not in the results, do not state it — \
  never estimate, extrapolate, or invent values.
- Provide a `rationale` for EVERY step — this is a research-grade system.
- If a tool failed in the previous iteration, adapt the plan. \
  Do not repeat the identical call that already errored.
- When a `cleaned_file_path` appears in results, use it as the input \
  `file_path` for all subsequent tools that read data.
- Respond with ONLY the JSON object — no markdown fences, no preamble.
"""


# ---------------------------------------------------------------------------
# Stage 2 — Initial reasoning (first LLM call)
# ---------------------------------------------------------------------------

INITIAL_ANALYSIS_PROMPT = """## Dataset Overview
{dataset_metadata}

## Your Task
You are an Autonomous Data Science Orchestrator in Phase 1: Foundation & Reconnaissance.
Your goal is to begin the investigation of this dataset.

Ground your plan in the evidence above: the dataset's column kinds,
warnings, detected nature, and distribution ranges tell you what this data actually is.

Invariants (the only fixed rules):
1. Clean data before running any other analysis on it — start with
   clean_data (strategy "median" is the safe default) on the original
   file_path, then use the cleaned_file_path it returns as `file_path`
   for every subsequent tool that reads data.
2. Include `file_path` in every tool call that requires it.
3. Do not plan a tool that isn't in the Available Tools list.
4. If the user objective names a goal, keep it as your central guiding target.
5. **Phase 1 Cadence**: Produce 1–3 targeted foundational steps (e.g. clean_data \
   to resolve missingness and types, followed by initial distribution or \
   correlation analysis). Do NOT schedule downstream predictive modeling, \
   deep segmentation, or visualizations yet — you will examine the concrete \
   empirical results of Phase 1 in the next cycle, and use that evidence to \
   orchestrate subsequent steps.

Respond in Form 1 (Action Plan).
"""


# ---------------------------------------------------------------------------
# Stage 4+5 — Result interpretation & iterative refinement
# ---------------------------------------------------------------------------

ITERATION_PROMPT = """\
## Dataset Overview
{dataset_metadata}

## Analysis Results So Far (Stage 4 — Result Interpretation)
{results_summary}

## Pending Steps
{pending_steps}

## Failed Steps (if any)
{failed_steps}

## Your Task
You are an Autonomous Data Science Orchestrator in Cycle {iteration} of iterative data discovery.

1. **Review and Interpret**: Review the empirical results from previous cycles carefully. \
What patterns, anomalies, correlations, or surprising distributions emerged?
2. **Formulate Hypotheses**: Formulate follow-up hypotheses or investigative questions based \
on what you observed.
3. **Select Instruments**: Choose the next tools or sandboxed code to test those hypotheses.
   - If an anomaly or strong relationship appeared, investigate it further with targeted \
statistical tests, segment comparisons, or custom calculations (`execute_dynamic_code`).
   - If predictive modeling is warranted, select appropriate model architectures and \
features informed by the correlations and distributions observed.
   - If no built-in tool answers a specific calculation, use `execute_dynamic_code` to run \
custom Python in the isolated sandbox.
4. **Plan Next Actions**: Plan 1–3 focused steps in Form 1 (Action Plan) for this cycle.
5. **Completion Criteria**: Return Form 2 (Final Answer) ONLY when you have thoroughly explored \
multiple analytical angles, tested hypotheses, and validated findings against the user objective. \
Do not conclude prematurely after just running basic exploratory tools.

Remember: always use `cleaned_file_path` as the source for downstream tools \
when cleaning has already been performed.

Respond in the correct JSON form.
"""


# ---------------------------------------------------------------------------
# Stage 7 — Final report synthesis
# ---------------------------------------------------------------------------

FINAL_INTERPRETATION_PROMPT = """\
## Dataset Overview
{dataset_metadata}

## Complete Analysis Results (Stage 7 — Report Generation)
{results_summary}

## Your Task
Synthesise ALL results into the final report. Be specific — reference actual \
metric values, column names, and model names from the results above.

Your insights must be data-driven and actionable. \
Recommendations must be concrete and implementable.

Respond in Form 2 (Final Answer). This is your last call.
"""


# ---------------------------------------------------------------------------
# Stage 6 — RLM sub-task prompt (used by RLMEngine.decompose_and_invoke)
# ---------------------------------------------------------------------------

RLM_SUBTASK_PROMPT = """\
## Sub-Task: {task_id}
{description}

## Sub-Context (from REPL environment)
{context_summary}

## Dataset Overview
{dataset_metadata}

## Results So Far
{results_summary}

## Your Task
You are a focused sub-analyst (Stage 6 RLM decomposition). Analyse ONLY the \
feature group above using the results already available — do NOT propose new \
tool calls; the tools for this data have already run.

Respond with ONLY this JSON shape:
{{
  "status": "complete",
  "task_id": "{task_id}",
  "insights": ["Concrete finding about this feature group.", "..."],
  "recommendations": ["Optional recommendation tied to these features."]
}}

Cite only values that appear in the results above. If the results contain \
nothing about these features, return an empty insights list — do not invent findings.
"""


# ---------------------------------------------------------------------------
# PromptManager
# ---------------------------------------------------------------------------

class PromptManager:
    """Assembles structured LLM prompts from memory state and tool descriptions."""

    def __init__(self, memory: MemorySystem, tool_descriptions: str) -> None:
        self.memory = memory
        self.tool_descriptions = tool_descriptions

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT_CORE.format(tool_descriptions=self.tool_descriptions)

    def _objective_block(self) -> str:
        """User's natural-language goal, injected into every reasoning prompt."""
        objective = self.memory.get_context("user_objective")
        if not objective:
            return ""
        return (
            f"\n## User Objective\n"
            f'The user asked: "{objective}"\n'
            f"Prioritise analyses that answer this objective. "
            f"Your final insights MUST directly address it.\n"
        )

    def _profile_block(self) -> str:
        """Compact data-profile summary produced at ingestion, when available."""
        summary = self.memory.get_context("data_profile_summary")
        if not summary:
            return ""
        return f"\n## Data Profile (automated first look)\n{summary}\n"

    def _rlm_block(self) -> str:
        """
        Stage 6 sub-task findings, fed back into later reasoning cycles so the
        decomposed analyses actually inform the final synthesis.
        """
        sub_results = self.memory.get_context("rlm_sub_results")
        if not isinstance(sub_results, dict) or not sub_results:
            return ""
        lines: list[str] = []
        for task_id, res in sub_results.items():
            if not isinstance(res, dict):
                continue
            insights = res.get("insights") or res.get("reasoning") or ""
            if isinstance(insights, list):
                insights = " ".join(str(i) for i in insights[:3])
            insights = str(insights).strip()
            if insights:
                lines.append(f"- [{task_id}] {insights[:300]}")
        if not lines:
            return ""
        return (
            "\n## RLM Sub-Analysis Findings (Stage 6 feature-group deep dives)\n"
            + "\n".join(lines)
            + "\n"
        )

    def _generated_tools_block(self) -> str:
        """
        Round 8 — tools the LLM defined earlier in THIS run via
        define_analysis_tool. The system prompt (static, cached) explains
        the contract once; this block is the dynamic, per-iteration list of
        what's actually callable now, so a tool created in iteration N shows
        up here from iteration N+1 onward without ever touching the cached
        system prompt (see IMPROVEMENTS.md Round 8, decision 1).
        """
        tools = self.memory.list_generated_tools()
        if not tools:
            return ""
        lines = [
            "\n## Tools You Created During This Run (callable now)",
            "These are in addition to the tools listed in the system prompt:",
        ]
        for spec in tools:
            name = spec.get("name", "?")
            desc = spec.get("description", "")
            params = spec.get("params_schema", {}) or {}
            param_names = ", ".join(params.keys()) if isinstance(params, dict) else ""
            lines.append(f"- `{name}` (params: {param_names or 'none'}) — {desc}")
        return "\n".join(lines) + "\n"

    def get_initial_user_prompt(self) -> str:
        meta = self.memory.dataset_metadata
        file_path  = meta.file_path  if meta else "UNKNOWN_PATH"
        target_col = meta.target_column if meta else None
        task_type  = meta.task_type if meta else "eda"

        concrete_instructions = (
            f"\n## Concrete Values for Your Plan\n"
            f"- original_file_path  = \"{file_path}\"\n"
            f"- target_column       = \"{target_col}\"\n"
            f"- task_type           = \"{task_type}\"\n"
            f"- Use these EXACT string values in your JSON parameters.\n"
            f"- After clean_data runs, the controller will automatically "
            f"substitute cleaned_file_path for file_path in subsequent tools.\n"
        )
        return (
            INITIAL_ANALYSIS_PROMPT.format(
                dataset_metadata=self.memory.get_metadata_prompt(),
            )
            + self._profile_block()
            + self._objective_block()
            + concrete_instructions
        )

    def get_iteration_user_prompt(self) -> str:
        meta = self.memory.dataset_metadata
        pending = self.memory.get_pending_steps()
        failed = self.memory.get_failed_steps()
        # Inject cleaned path if available
        cleaned = self.memory.get_context("cleaned_file_path")
        best_model_path = self.memory.get_context("best_model_path")
        best_model_name = self.memory.get_context("best_model_name")

        pending_str = (
            "\n".join(
                f"Step {s.step_number}: {s.tool_name}({json_compact(s.parameters)})"
                for s in pending
            )
            or "None — re-evaluate whether more analysis is needed."
        )
        failed_lines: list[str] = [
            f"Step {s.step_number}: {s.tool_name} → "
            f"{s.result.error_message if s.result else 'unknown error'} "
            f"(retries: {s.retry_count})"
            for s in failed
        ]
        if not failed_lines:
            failed_lines = [
                f"{r.tool_name} → {r.error_message or 'unknown error'}"
                for r in self.memory.tool_results
                if r.status == "error"
            ]
        failed_str = "\n".join(failed_lines) or "None."

        concrete = ""
        if cleaned:
            concrete += f"\n## Current File Paths\n- cleaned_file_path = \"{cleaned}\"\n"
        if best_model_path:
            concrete += f"- best_model_path = \"{best_model_path}\"\n"
            concrete += f"- best_model_name = \"{best_model_name}\"\n"
        if meta and meta.target_column:
            concrete += f"- target_column = \"{meta.target_column}\"\n"

        return (
            ITERATION_PROMPT.format(
                iteration=self.memory.iteration_count,
                dataset_metadata=self.memory.get_metadata_prompt(),
                # P1.6 — full detail for the results the LLM hasn't reacted
                # to yet, a digest for everything earlier, instead of
                # re-sending every accumulated result on every iteration.
                # Results are tagged with the iteration that PRODUCED them
                # (controller._execute_steps sets result.iteration from
                # memory.iteration_count, which this prompt builder itself
                # bumped to the CURRENT cycle before this call — see
                # analyze()'s loop) — so "iteration_count" here always names
                # a bucket that's still empty; the results the LLM is
                # actually seeing for the first time are the PREVIOUS
                # cycle's, hence "- 1".
                results_summary=self.memory.get_results_summary_digest(self.memory.iteration_count - 1),
                pending_steps=pending_str,
                failed_steps=failed_str,
            )
            + self._profile_block()
            + self._rlm_block()
            + self._objective_block()
            + self._generated_tools_block()
            + concrete
        )

    def get_final_interpretation_prompt(self) -> str:
        return (
            FINAL_INTERPRETATION_PROMPT.format(
                dataset_metadata=self.memory.get_metadata_prompt(),
                results_summary=self.memory.get_results_summary(),
            )
            + self._rlm_block()
            + self._objective_block()
        )

    def get_rlm_subtask_prompt(
        self,
        task_id: str,
        description: str,
        context_summary: str,
    ) -> str:
        return RLM_SUBTASK_PROMPT.format(
            task_id=task_id,
            description=description,
            context_summary=context_summary,
            dataset_metadata=self.memory.get_metadata_prompt(),
            results_summary=self.memory.get_results_summary(),
        )


def json_compact(d: dict) -> str:  # type: ignore[type-arg]
    import json
    return json.dumps(d, default=str)
