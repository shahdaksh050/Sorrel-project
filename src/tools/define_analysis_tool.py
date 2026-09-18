"""
Meta-tool: lets the LLM planner define a new, named, reusable analysis
tool at runtime (Round 8, item 8.2 — the "define_analysis_tool" contract
from IMPROVEMENTS.md's Round 8 section).

Pure validation/smoke-test layer, matching design decision 3 (AGENTS.md's
layer rule): this tool's execute() never registers anything on
ToolRegistry, never touches MemorySystem, and never calls
src.core.tool_factory.register_and_persist. It only answers the question
"is this proposed tool spec runnable?" so the controller (which owns the
registry and memory) can decide what to do with a "ready" answer.

This is also the self-correction loop's feedback path: when the smoke test
fails, the sandbox's own hint/error_type comes back in this tool's output
so the LLM can revise `code` and call define_analysis_tool again for the
same tool_name.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.sandbox import ALLOWED_MODULES_TEXT, _static_check, run_sandboxed
from src.tools.base import BaseTool


class DefineAnalysisToolTool(BaseTool):
    """Validate (and optionally smoke-test) a proposed new analysis tool."""

    requires_llm = True

    name = "define_analysis_tool"
    description = (
        "Define a new, named, reusable analysis tool when no existing tool "
        "answers a question you need to answer repeatedly or with different "
        "parameters. This does NOT run the analysis itself and does NOT "
        "register the tool — it validates your proposed code and, if a "
        "dataset is given, smoke-tests it once, then reports whether the "
        "tool is ready to be created. Code contract: a pandas DataFrame "
        "`df` and a `SCHEMA` dict (column name -> semantic kind) are "
        "pre-loaded; any parameter you declare in params_schema is "
        "available as a plain variable of that name; you must assign the "
        "final answer to a variable named RESULT. You may also assign "
        "FINDING = {'headline': ..., 'detail': ..., 'evidence': {...}} to "
        "surface a real, human-readable insight into the report and "
        "dashboard — use this whenever your tool discovers something "
        "worth reporting, not just a computed value. Only "
        f"{ALLOWED_MODULES_TEXT} may be imported; no file or network "
        "access, no eval/exec/open. After this tool reports "
        "status == 'ready', the tool becomes callable by tool_name on a "
        "later step."
    )

    def get_schema(self) -> dict[str, Any]:
        return {
            "tool_name": {
                "type": "string",
                "description": (
                    "snake_case name for the new tool (e.g. "
                    "'average_order_value_by_region'). Must not collide with "
                    "an existing built-in tool name."
                ),
                "required": True,
            },
            "description": {
                "type": "string",
                "description": "Human-readable description of what this tool computes, for future planning steps.",
                "required": True,
            },
            "params_schema": {
                "type": "object",
                "description": (
                    "Parameters this tool accepts beyond file_path, in the same "
                    "shape as any tool's get_schema(): "
                    "{param_name: {'type':..., 'description':..., 'required': bool}}. "
                    "Pass {} if the tool needs no parameters. May be given as a "
                    "JSON object or as a JSON-encoded string."
                ),
                "required": True,
            },
            "code": {
                "type": "string",
                "description": (
                    "Python code implementing the tool. Must assign RESULT at "
                    "the top level; may optionally assign FINDING. Any name "
                    "declared in params_schema is available as a plain variable."
                ),
                "required": True,
            },
            "example_params": {
                "type": "object",
                "description": (
                    "Optional example values for the declared parameters, used "
                    "only to smoke-test the code once against file_path."
                ),
                "required": False,
            },
            "file_path": {
                "type": "string",
                "description": (
                    "Path to the (cleaned) dataset to smoke-test the code "
                    "against before reporting the tool as ready."
                ),
                "required": True,
            },
        }

    def execute(  # type: ignore[override]
        self,
        tool_name: str,
        description: str,
        params_schema: Any,
        code: str,
        example_params: dict[str, Any] | None = None,
        file_path: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        # params_schema may arrive as a dict or as a JSON-encoded string —
        # accept either so a planner that serializes all object-typed
        # arguments to text doesn't fail here for a formatting reason.
        if isinstance(params_schema, str):
            try:
                params_schema = json.loads(params_schema) if params_schema.strip() else {}
            except json.JSONDecodeError as exc:
                return {
                    "summary": f"'{tool_name}' is not ready: params_schema is not valid JSON.",
                    "status": "error",
                    "spec": {
                        "name": tool_name,
                        "description": description,
                        "params_schema": params_schema,
                        "code": code,
                    },
                    "hint": f"params_schema JSON parse error: {exc}",
                }

        spec = {
            "name": tool_name,
            "description": description,
            "params_schema": params_schema,
            "code": code,
        }

        if not code or not code.strip():
            return {
                "summary": f"'{tool_name}' is not ready: no code was provided.",
                "status": "error",
                "spec": spec,
                "hint": "Provide code that assigns the final answer to RESULT.",
            }

        static_error = _static_check(code)
        if static_error is not None:
            error_type, hint = static_error
            return {
                "summary": f"'{tool_name}' is not ready: code failed the static check ({error_type}).",
                "status": "error",
                "spec": spec,
                "hint": hint,
            }

        if not isinstance(params_schema, dict):
            return {
                "summary": f"'{tool_name}' is not ready: params_schema must be an object.",
                "status": "error",
                "spec": spec,
                "hint": f"Expected a dict for params_schema, got {type(params_schema).__name__}.",
            }

        if file_path and Path(file_path).exists():
            sandbox_result = run_sandboxed(
                code=code,
                dataset_ref=file_path,
                extra_globals=example_params or {},
                timeout_s=10.0,
            )
            if sandbox_result.status != "ok":
                return {
                    "summary": (
                        f"'{tool_name}' is not ready: the smoke test failed "
                        f"({sandbox_result.error_type})."
                    ),
                    "status": "error",
                    "spec": spec,
                    "hint": sandbox_result.hint or "See traceback for details.",
                }
            return {
                "summary": (
                    f"'{tool_name}' passed validation and a smoke test in "
                    f"{sandbox_result.duration_ms:.0f} ms — ready to be created."
                ),
                "status": "ready",
                "spec": spec,
                "hint": None,
            }

        return {
            "summary": f"'{tool_name}' passed static validation (no smoke test run) — ready to be created.",
            "status": "ready",
            "spec": spec,
            "hint": None,
        }
