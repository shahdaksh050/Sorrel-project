"""
Dynamic Code Execution Tool — Execution Layer.

Wraps src.core.sandbox.run_sandboxed as a BaseTool so the existing
profile-driven planner can select it like any other tool. Additive:
use only for questions no other tool in the registry answers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.sandbox import run_sandboxed
from src.tools.base import BaseTool, ToolExecutionError

if TYPE_CHECKING:
    from src.core.memory import MemorySystem


class DynamicCodeExecutionTool(BaseTool):
    """Execute custom Python code against the dataset in an isolated subprocess."""

    requires_llm = True
    uses_cleaned_file = True

    name = "execute_dynamic_code"
    description = (
        "Execute custom Python code against the dataset when no other tool "
        "answers the question. A DataFrame `df`, column-kind `SCHEMA`, and "
        "prior tool outputs `PRIOR_RESULTS` dict (e.g. PRIOR_RESULTS.get('select_statistical_test')) "
        "are ALREADY PRE-LOADED; assign the answer to RESULT. Do NOT call "
        "pd.read_csv() or open() — filesystem access is strictly blocked. "
        "Runs in an isolated, restricted sandbox — no network access, "
        "no imports outside pandas/numpy/scipy/sklearn/duckdb/polars/math/"
        "statistics/json/datetime/re/collections/itertools. Use only for "
        "questions the other analysis tools cannot address."
    )

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        params = super().prepare_params(params, memory, output_root)
        prior: dict[str, Any] = {}
        for tr in memory.tool_results:
            if tr.status == "success" and isinstance(tr.output, dict):
                clean_output: dict[str, Any] = {}
                for k, v in tr.output.items():
                    try:
                        json.dumps(v)
                        clean_output[k] = v
                    except (TypeError, OverflowError):
                        continue
                prior[tr.tool_name] = clean_output
        params["prior_results"] = prior
        return params

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        code: str,
        prior_results: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not code or not code.strip():
            raise ToolExecutionError("No code was provided to execute.")
        if not Path(file_path).exists():
            raise ToolExecutionError(f"Dataset file not found: {file_path}")

        sandbox_result = run_sandboxed(
            code=code, dataset_ref=file_path, prior_results=prior_results
        )

        if sandbox_result.status == "ok":
            return {
                "summary": f"Executed successfully in {sandbox_result.duration_ms:.0f} ms.",
                "status": "ok",
                "result": sandbox_result.result,
                "stdout": sandbox_result.stdout,
                "error_type": None,
                "traceback": None,
                "hint": None,
                "duration_ms": sandbox_result.duration_ms,
            }

        return {
            "summary": (
                f"Execution failed ({sandbox_result.error_type}): "
                f"{sandbox_result.hint or 'see traceback for details.'}"
            ),
            "status": "error",
            "result": None,
            "stdout": sandbox_result.stdout,
            "error_type": sandbox_result.error_type,
            "traceback": sandbox_result.traceback,
            "hint": sandbox_result.hint,
            "duration_ms": sandbox_result.duration_ms,
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the (cleaned) dataset.",
                "required": True,
            },
            "code": {
                "type": "string",
                "description": (
                    "Python code to execute against the dataset. Must assign "
                    "the final answer to a variable named RESULT at the top "
                    "level. A pandas DataFrame `df`, `SCHEMA` dict, and `PRIOR_RESULTS` dict "
                    "(outputs from prior executed tools) are ALREADY PRE-LOADED — do NOT "
                    "call pd.read_csv() or open(); filesystem access is blocked. "
                    "Only pandas, numpy, scipy, sklearn, duckdb, polars, math, "
                    "statistics, json, datetime, re, collections, itertools are available."
                ),
                "required": True,
            },
        }
