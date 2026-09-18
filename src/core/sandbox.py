"""
Isolated Compute Sandbox — safe execution of LLM-generated Python code
against the current dataset.

Additive to the existing profile-driven tool architecture (see
docs/superpowers/specs/2026-09-11-isolated-compute-sandbox-design.md):
the agent reaches for this when no BaseTool in the registry answers a
question, not as a replacement for the deterministic tool library.

Process-level trust, not a kernel/VM boundary — the threat model is
buggy or wasteful LLM-generated code, not a deliberate sandbox-escape
adversary. Revisit with a container/VM boundary before ever executing
code from untrusted external users.

Pure orchestration in this file; the restricted execution itself
happens in a subprocess running _sandbox_worker.py.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psutil

#: Modules the sandboxed code is allowed to import. Enforced both
#: statically (_static_check, below) and at runtime inside the worker's
#: import hook (src/core/_sandbox_worker.py) as defense in depth.
ALLOWED_MODULES: frozenset[str] = frozenset({
    "pandas", "numpy", "scipy", "sklearn", "duckdb", "polars",
    "math", "statistics", "json", "datetime", "re", "collections",
    "itertools",
})

ALLOWED_MODULES_TEXT: str = ", ".join(sorted(ALLOWED_MODULES))

#: Builtins referenced by name are blocked even before exec() — matches
#: the restricted __builtins__ dict the worker installs.
BLOCKED_BUILTIN_NAMES: frozenset[str] = frozenset({
    "open", "eval", "exec", "compile", "__import__", "input", "exit", "quit",
})

#: Result-variable contract the generated code must satisfy.
RESULT_VAR_NAME: str = "RESULT"

#: Captured stdout is capped so it never balloons a future LLM prompt.
STDOUT_CAP_CHARS: int = 8_000

#: Names `extra_globals` may never define — they're owned by the worker's
#: own execution namespace. Checked here (before the subprocess spawns) and
#: again, defensively, inside _sandbox_worker._build_restricted_globals.
RESERVED_GLOBAL_NAMES: frozenset[str] = frozenset(
    {"df", "SCHEMA", "RESULT", "FINDING", "__builtins__"}
)


@dataclass
class SandboxResult:
    """Outcome of one sandboxed code execution."""

    status: Literal["ok", "error"]
    result: Any | None
    #: From an optional top-level `FINDING = {...}` in the sandboxed code,
    #: JSON-converted the same way RESULT is. Never required; None if the
    #: code didn't set one or if it failed to JSON-convert.
    finding: dict[str, Any] | None
    stdout: str
    #: "static_check" | "import_blocked" | "syntax" | "runtime"
    #: | "timeout" | "memory" | "output_invalid"
    error_type: str | None
    traceback: str | None
    hint: str | None
    duration_ms: float


def _static_check(code: str) -> tuple[str, str] | None:
    """
    Validate `code` before any subprocess is spawned.

    Returns (error_type, hint) if the code fails validation, else None.
    Catches the mistakes a lightweight LLM makes most often — a missing
    RESULT assignment or an import outside the allowlist — in
    milliseconds, without burning a timeout cycle on a subprocess.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return "syntax", f"Code has a syntax error: {exc.msg} (line {exc.lineno})."

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                if top_level not in ALLOWED_MODULES:
                    return (
                        "static_check",
                        f"Only {ALLOWED_MODULES_TEXT} are available. "
                        f"Remove the import for '{alias.name}'.",
                    )
        elif isinstance(node, ast.ImportFrom):
            top_level = (node.module or "").split(".")[0]
            if top_level not in ALLOWED_MODULES:
                return (
                    "static_check",
                    f"Only {ALLOWED_MODULES_TEXT} are available. "
                    f"Remove the import for '{node.module}'.",
                )
        elif isinstance(node, ast.Name) and node.id in BLOCKED_BUILTIN_NAMES:
            return (
                "static_check",
                f"'{node.id}' is not available in the sandbox. "
                "Use only pandas/numpy/scipy/sklearn/duckdb/polars operations.",
            )

    # Walk the whole tree, not just tree.body — a reusable, parameterized
    # tool body plausibly assigns RESULT inside an if/else, a loop, or a
    # function rather than only at the top level. This only removes a
    # source of spurious *static* rejections; the runtime check in
    # _sandbox_worker.py (RESULT_VAR_NAME not in restricted_globals after
    # exec) remains the authoritative gate on what's actually required.
    has_result_assignment = any(
        (
            isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == RESULT_VAR_NAME for t in stmt.targets)
        )
        or (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == RESULT_VAR_NAME
        )
        for stmt in ast.walk(tree)
    )
    if not has_result_assignment:
        return (
            "static_check",
            f"Your code must assign the final answer to a variable named "
            f"{RESULT_VAR_NAME} at the top level.",
        )
    return None


_WORKER_SCRIPT = Path(__file__).with_name("_sandbox_worker.py")

#: Repo root, so the worker subprocess can `import src.core...` despite
#: running with cwd set to a scratch temp directory (see _worker_env below).
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: How often the parent polls the child's RSS / checks for completion.
_POLL_INTERVAL_S = 0.2


def _worker_env() -> dict[str, str]:
    """Child process environment with the repo root prepended to
    PYTHONPATH, so `src.core.io` etc. import cleanly regardless of the
    worker's cwd — without any sys.path manipulation inside the worker
    script itself."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(_REPO_ROOT) if not existing else f"{_REPO_ROOT}{os.pathsep}{existing}"
    )
    return env


def _validate_extra_globals(
    extra_globals: dict[str, Any] | None,
) -> tuple[str, str] | None:
    """
    Validate `extra_globals` before any subprocess is spawned.

    Returns (error_type, hint) if invalid, else None. Every value must be
    JSON-primitive — checked with a json.dumps round-trip, since that's
    exactly the boundary the worker itself must cross the value back over —
    and no key may collide with a name the worker's own execution namespace
    reserves for itself.
    """
    if not extra_globals:
        return None
    for key, value in extra_globals.items():
        if key in RESERVED_GLOBAL_NAMES:
            return (
                "static_check",
                f"extra_globals key '{key}' collides with a reserved name "
                f"({', '.join(sorted(RESERVED_GLOBAL_NAMES))}).",
            )
        try:
            json.dumps(value)
        except TypeError:
            return (
                "static_check",
                f"extra_globals['{key}'] must be a JSON-primitive value "
                "(str/int/float/bool/None/list/dict) — got "
                f"{type(value).__name__}.",
            )
    return None


def run_sandboxed(
    code: str,
    dataset_ref: str,
    extra_globals: dict[str, Any] | None = None,
    timeout_s: float = 20.0,
    memory_limit_mb: int = 512,
) -> SandboxResult:
    """
    Execute `code` against the dataset at `dataset_ref` in an isolated
    subprocess and return a structured result.

    `extra_globals`, when given, is JSON-validated and threaded into the
    worker's restricted execution namespace as additional top-level names —
    parameters reach generated code as data, never as templated source.

    Never raises for a failure of the *sandboxed code* — syntax errors,
    blocked imports, timeouts, and runtime exceptions all come back as
    SandboxResult(status="error", ...).
    """
    t0 = time.perf_counter()

    static_error = _static_check(code)
    if static_error is not None:
        error_type, hint = static_error
        return SandboxResult(
            status="error", result=None, finding=None, stdout="",
            error_type=error_type, traceback=None, hint=hint,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    extra_globals_error = _validate_extra_globals(extra_globals)
    if extra_globals_error is not None:
        error_type, hint = extra_globals_error
        return SandboxResult(
            status="error", result=None, finding=None, stdout="",
            error_type=error_type, traceback=None, hint=hint,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    # ignore_cleanup_errors: the scratch dir is the killed child's cwd, and on
    # Windows a rmtree over a directory whose handle a just-terminated process
    # still holds raises PermissionError out of __exit__ — which would escape
    # run_sandboxed and break the "never raises" contract above. Leaking a temp
    # directory is the better failure mode.
    with tempfile.TemporaryDirectory(
        prefix="sandbox_", ignore_cleanup_errors=True
    ) as scratch_dir:
        input_path = Path(scratch_dir) / "input.json"
        result_path = Path(scratch_dir) / "result.json"
        input_path.write_text(
            json.dumps(
                {
                    "code": code,
                    "dataset_ref": dataset_ref,
                    "extra_globals": extra_globals,
                }
            ),
            encoding="utf-8",
        )

        proc = subprocess.Popen(
            [sys.executable, str(_WORKER_SCRIPT), str(input_path), str(result_path)],
            cwd=scratch_dir,
            env=_worker_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            ps_proc: psutil.Process | None = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            ps_proc = None

        killed_as: str | None = None
        while True:
            try:
                proc.wait(timeout=_POLL_INTERVAL_S)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - t0
                if elapsed > timeout_s:
                    killed_as = "timeout"
                elif ps_proc is not None:
                    try:
                        rss_mb = ps_proc.memory_info().rss / (1024 * 1024)
                        if rss_mb > memory_limit_mb:
                            killed_as = "memory"
                    except psutil.NoSuchProcess:
                        pass
                if killed_as is not None:
                    proc.kill()
                    proc.wait()
                    break

        duration_ms = (time.perf_counter() - t0) * 1000

        if killed_as == "timeout":
            return SandboxResult(
                status="error", result=None, finding=None, stdout="",
                error_type="timeout",
                traceback=None,
                hint=(
                    f"Execution exceeded {timeout_s:.0f}s. Avoid unbounded "
                    "loops; operate on df directly instead of iterating rows."
                ),
                duration_ms=duration_ms,
            )
        if killed_as == "memory":
            return SandboxResult(
                status="error", result=None, finding=None, stdout="",
                error_type="memory",
                traceback=None,
                hint=(
                    f"Execution used more than {memory_limit_mb}MB. Avoid "
                    "materializing full copies of large intermediate results."
                ),
                duration_ms=duration_ms,
            )

        if not result_path.exists():
            return SandboxResult(
                status="error", result=None, finding=None, stdout="",
                error_type="runtime",
                traceback=f"Worker exited with code {proc.returncode} and wrote no result.",
                hint="The sandboxed process crashed before producing a result.",
                duration_ms=duration_ms,
            )

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return SandboxResult(
                status="error", result=None, finding=None, stdout="",
                error_type="runtime",
                traceback=str(exc),
                hint="The sandboxed process produced an unreadable result.",
                duration_ms=duration_ms,
            )

        return SandboxResult(
            status=payload["status"],
            result=payload.get("result"),
            finding=payload.get("finding"),
            stdout=payload.get("stdout", ""),
            error_type=payload.get("error_type"),
            traceback=payload.get("traceback"),
            hint=payload.get("hint"),
            duration_ms=duration_ms,
        )
