"""
Governance for LLM-authored code — the controls an operator sets, and the
record of what the agent actually ran.

  - Kill switch: ENABLE_CODE_EXECUTION=false removes every code-running
    tool (execute_dynamic_code, define_analysis_tool, generated tools) from
    the planner's tool list and rejects any step that names one anyway.
  - Budget: MAX_CODE_EXECUTIONS (default 40) caps sandbox runs per analysis,
    so a planner stuck in a fix-and-retry loop cannot burn unbounded compute.
  - Audit log: every sandbox execution — and every refusal — is appended to
    <output_dir>/audit/code_executions.jsonl with the full code, its SHA-256,
    the outcome and the backend that ran it. Append-only JSONL, one record
    per line, so a reviewer can reconstruct exactly what executed.

Pure: stdlib only, no memory/controller/tool imports.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AUDIT_SUBDIR = "audit"
AUDIT_FILENAME = "code_executions.jsonl"

#: Parameters recorded verbatim in the audit log; everything else is logged
#: by name only (prior results and large payloads don't belong in an audit).
_AUDITED_PARAM_KEYS = ("tool_name", "save_as", "description", "params_schema", "example_params")


def code_execution_enabled() -> bool:
    return os.getenv("ENABLE_CODE_EXECUTION", "true").strip().lower() not in ("0", "false", "no")


def max_code_executions() -> int:
    try:
        return max(0, int(os.getenv("MAX_CODE_EXECUTIONS", "40")))
    except ValueError:
        return 40


@dataclass
class CodeGovernor:
    """Per-run state: how many code executions happened, and where the
    audit log lives."""

    output_dir: str
    session_id: str
    executions: int = 0
    refusals: int = 0
    failures: int = 0
    backends: set[str] = field(default_factory=set)

    @property
    def audit_path(self) -> Path:
        return Path(self.output_dir) / AUDIT_SUBDIR / AUDIT_FILENAME

    def refusal_reason(self) -> str | None:
        """Why the next code execution must not run, or None if it may."""
        if not code_execution_enabled():
            return "Code execution is disabled for this deployment (ENABLE_CODE_EXECUTION=false)."
        if self.executions >= max_code_executions():
            return (
                f"The code-execution budget for this run ({max_code_executions()}) is spent. "
                "Use the built-in tools, or finish the analysis."
            )
        return None

    def record(
        self,
        *,
        tool_name: str,
        code: str,
        params: dict[str, Any],
        status: str,
        iteration: int,
        step_number: int,
        output: dict[str, Any] | None = None,
        error: str | None = None,
        refused: bool = False,
    ) -> None:
        """Append one audit record. Never raises — auditing must not be the
        reason an analysis fails."""
        if refused:
            self.refusals += 1
        else:
            self.executions += 1
            if status != "success":
                self.failures += 1
        output = output or {}
        backend = str(output.get("backend") or ("refused" if refused else ""))
        if backend:
            self.backends.add(backend)
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "iteration": iteration,
            "step_number": step_number,
            "tool_name": tool_name,
            "status": "refused" if refused else status,
            "backend": backend or None,
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest() if code else None,
            "code": code,
            "params": {k: params[k] for k in _AUDITED_PARAM_KEYS if k in params},
            "param_names": sorted(k for k in params if not k.startswith("_") and k != "prior_results"),
            "duration_ms": output.get("duration_ms"),
            "derived_dataset": (output.get("derived_dataset") or {}).get("path"),
            "error": (error or "")[:2000] or None,
        }
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass

    def summary(self) -> dict[str, Any]:
        return {
            "code_execution_enabled": code_execution_enabled(),
            "code_executions": self.executions,
            "code_failures": self.failures,
            "code_refusals": self.refusals,
            "execution_budget": max_code_executions(),
            "sandbox_backends": sorted(self.backends),
            "audit_log": str(self.audit_path) if (self.executions or self.refusals) else None,
        }
