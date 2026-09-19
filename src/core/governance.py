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

LLM egress controls, same shape:
  - LLM audit: every LLM call is appended to
    <output_dir>/audit/llm_calls.jsonl — provider, model, SHA-256 of the
    prompt and the response, sizes, token usage. Prompt/response text only
    with AUDIT_LLM_FULL_TEXT=true.
  - Token cap: MAX_LLM_TOKENS_PER_RUN (default 0 = unlimited).
  - LOCAL_ONLY=true: only an offline provider (local/ollama) may be called.

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
LLM_AUDIT_FILENAME = "llm_calls.jsonl"

#: Providers that keep every prompt on this machine.
LOCAL_PROVIDERS = frozenset({"local", "ollama"})

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


def _env_true(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in ("1", "true", "yes")


def local_only() -> bool:
    return _env_true("LOCAL_ONLY")


def max_llm_tokens_per_run() -> int:
    """0 means unlimited."""
    try:
        return max(0, int(os.getenv("MAX_LLM_TOKENS_PER_RUN", "0")))
    except ValueError:
        return 0


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_llm_call(
    audit_dir: str | Path,
    *,
    stage: str | None,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response: str | None,
    usage: dict[str, Any] | None,
    error: str | None = None,
) -> None:
    """Append one LLM-call record to <audit_dir>/llm_calls.jsonl. Hashes,
    not text, unless AUDIT_LLM_FULL_TEXT=true. Never raises."""
    prompt = system_prompt + "\n" + user_prompt
    entry: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "stage": stage,
        "provider": provider,
        "model": model,
        "prompt_sha256": _sha256(prompt),
        "response_sha256": _sha256(response) if response is not None else None,
        "prompt_chars": len(prompt),
        "response_chars": len(response) if response is not None else 0,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "error": (error or "")[:500] or None,
    }
    if _env_true("AUDIT_LLM_FULL_TEXT"):
        entry["system_prompt"] = system_prompt
        entry["user_prompt"] = user_prompt
        entry["response"] = response
    try:
        path = Path(audit_dir) / LLM_AUDIT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass


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

    def summary(self, llm_usage: dict[str, Any] | None = None) -> dict[str, Any]:
        """`llm_usage` is RLMEngine.usage_summary() for this run, if any."""
        llm_usage = llm_usage or {}
        llm_audit = Path(self.output_dir) / AUDIT_SUBDIR / LLM_AUDIT_FILENAME
        return {
            "llm_calls": int(llm_usage.get("call_count", 0) or 0),
            "llm_tokens": int(llm_usage.get("total_tokens", 0) or 0),
            "llm_token_cap": max_llm_tokens_per_run(),
            "local_only": local_only(),
            "llm_audit_log": str(llm_audit) if llm_audit.exists() else None,
            "code_execution_enabled": code_execution_enabled(),
            "code_executions": self.executions,
            "code_failures": self.failures,
            "code_refusals": self.refusals,
            "execution_budget": max_code_executions(),
            "sandbox_backends": sorted(self.backends),
            "audit_log": str(self.audit_path) if (self.executions or self.refusals) else None,
        }
