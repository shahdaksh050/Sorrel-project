"""
Isolated Compute Sandbox — safe execution of LLM-generated Python code
against the current dataset.

Additive to the existing profile-driven tool architecture (see
docs/superpowers/specs/2026-09-11-isolated-compute-sandbox-design.md):
the agent reaches for this when no BaseTool in the registry answers a
question, not as a replacement for the deterministic tool library.

Security model — layered, each layer covering the previous one's gaps:

  1. Static policy (`_static_check`, here, before any process starts):
     imports limited to ALLOWED_MODULES; no introspection builtins
     (getattr/vars/globals...); no `_`-prefixed attributes (closes the
     `x.__class__.__subclasses__()` / `f.__globals__` family, including
     traversal off the pre-bound `dsa` toolkit's functions); no frame or
     module-handle attributes (`gi_frame`, `.os`, `.sys`...); no file,
     network or pickle APIs (`read_*`, `to_parquet`, `np.load`, `fetch_*`);
     no format strings that walk attributes. Fast, and every rejection
     carries a hint the LLM can act on.
  2. Runtime policy (`_sandbox_worker._install_runtime_policy`): a
     `sys.addaudithook` hook installed immediately before exec — it cannot
     be removed — that blocks process creation, sockets, raw memory access
     (`ctypes.cdata`), environment mutation, and any file access outside
     the scratch dir (reads also allowed from the interpreter's own
     library directories and `src/`). It sees Python-level escapes that
     slipped past layer 1.
  3. Nothing worth stealing: the child gets an allowlisted environment (no
     API keys) and its TEMP is the scratch dir.
  4. Resource limits on the whole process tree: wall-clock timeout, RSS
     cap, thread caps, result-size cap.
  5. `SANDBOX_BACKEND=docker` adds a kernel boundary (no network,
     read-only root, dropped capabilities, non-root user, pid/memory/cpu
     cgroups). `SANDBOX_REQUIRE_ISOLATION=true` refuses to run code at all
     rather than fall back to the subprocess backend.

Threat model: the subprocess backend is designed against buggy, wasteful
or prompt-injected LLM code (the realistic case — the planner reads
dataset-derived text). Layers 1-4 are not a substitute for a kernel
boundary against a determined adversary exploiting native-code bugs in
numpy/pandas; use the Docker backend with SANDBOX_REQUIRE_ISOLATION=true
whenever datasets or objectives come from untrusted users.

Pure orchestration in this file; the restricted execution itself
happens in a subprocess running _sandbox_worker.py.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import psutil

#: Modules the sandboxed code is allowed to import. Enforced both
#: statically (_static_check, below) and at runtime inside the worker's
#: import hook (src/core/_sandbox_worker.py) as defense in depth.
#: duckdb and polars are deliberately absent: their file/network I/O runs in
#: native code the runtime audit hook cannot see, and pandas covers the
#: same analytical ground.
ALLOWED_MODULES: frozenset[str] = frozenset({
    "pandas", "numpy", "scipy", "sklearn",
    "math", "statistics", "json", "datetime", "re", "collections",
    "itertools",
})

ALLOWED_MODULES_TEXT: str = ", ".join(sorted(ALLOWED_MODULES))

#: Submodules of allowed packages that exist to do I/O, spawn processes,
#: download data, or load native code.
_BLOCKED_SUBMODULES: tuple[str, ...] = (
    "pandas.io", "pandas.compat", "pandas.util", "pandas.testing",
    "numpy.ctypeslib", "numpy.f2py", "numpy.distutils", "numpy.testing", "numpy.lib.npyio",
    "numpy.lib.format", "scipy.io", "scipy.datasets", "sklearn.datasets._base",
    "sklearn.externals", "sklearn.utils._testing",
)

#: Builtins referenced by name are blocked even before exec() — matches
#: the restricted __builtins__ dict the worker installs. The introspection
#: builtins are here because each is a way around the attribute policy
#: below (`getattr(x, "__cl" + "ass__")`).
BLOCKED_BUILTIN_NAMES: frozenset[str] = frozenset({
    "open", "eval", "exec", "compile", "__import__", "input", "exit", "quit",
    "getattr", "setattr", "delattr", "vars", "globals", "locals",
    "breakpoint", "help", "memoryview",
})

#: Attribute names sandboxed code may not touch: handles to the OS,
#: interpreter internals (frames, code objects), serialization that executes
#: code, and file/network I/O on the allowed libraries. `_`-prefixed
#: attributes are blocked separately and wholesale.
_BLOCKED_ATTRS: frozenset[str] = frozenset({
    # module handles libraries re-export as attributes
    "os", "sys", "io", "subprocess", "shutil", "socket", "builtins", "importlib",
    "pathlib", "tempfile", "glob", "urllib", "http", "ctypes", "ctypeslib",
    "pickle", "cPickle", "marshal", "compat", "f2py", "distutils", "testing",
    # process / environment
    "system", "popen", "spawn", "environ", "getenv", "putenv",
    # frames, code and loader internals
    "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code",
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "tb_frame", "tb_next",
    "co_code", "func_globals", "loader", "exec_module", "modules",
    # numpy / pandas / pyarrow file I/O not covered by the prefixes below
    "load", "loadtxt", "save", "savez", "savez_compressed", "savetxt", "genfromtxt",
    "fromfile", "tofile", "memmap", "DataSource", "fromregex",
    "to_pickle", "to_parquet", "to_feather", "to_orc", "to_hdf", "to_sql",
    "to_clipboard", "to_excel", "to_stata", "ExcelWriter", "ExcelFile", "HDFStore",
    "dump_svmlight_file", "load_svmlight_file", "load_svmlight_files",
})

#: Attribute-name prefixes blocked for the same reason: every pandas/pyarrow
#: reader (`read_csv`, `read_pickle`, `read_html` — which fetches URLs), every
#: sklearn downloader (`fetch_openml`), and pyarrow's native file openers.
_BLOCKED_ATTR_PREFIXES: tuple[str, ...] = ("_", "read_", "fetch_", "open_")

#: A str.format field that walks attributes or indexes ("{0.x}", "{0[k]}") —
#: format strings resolve attributes without any attribute node in the AST.
_FORMAT_FIELD_WALK_RE = re.compile(r"\{[^{}:!]*[.\[][^{}]*\}")


def _blocked_attr(name: str) -> bool:
    return name in _BLOCKED_ATTRS or name.startswith(_BLOCKED_ATTR_PREFIXES)


def _blocked_module(module: str) -> bool:
    return module.split(".")[0] not in ALLOWED_MODULES or any(
        module == m or module.startswith(m + ".") for m in _BLOCKED_SUBMODULES
    )

#: Result-variable contract the generated code must satisfy.
RESULT_VAR_NAME: str = "RESULT"

#: Longest code body accepted — an analysis step, not a program.
MAX_CODE_CHARS: int = 12_000

#: Largest serialized worker result accepted back from the sandbox — RESULT
#: flows into later LLM prompts and reports, so aggregate, don't dump rows.
MAX_RESULT_BYTES: int = 2_000_000

#: Captured stdout is capped so it never balloons a future LLM prompt.
STDOUT_CAP_CHARS: int = 8_000

#: Names `extra_globals` may never define — they're owned by the worker's
#: own execution namespace. Checked here (before the subprocess spawns) and
#: again, defensively, inside _sandbox_worker._build_restricted_globals.
RESERVED_GLOBAL_NAMES: frozenset[str] = frozenset(
    {"df", "SCHEMA", "RESULT", "FINDING", "PRIOR_RESULTS", "CHART", "DF_OUT", "dsa", "__builtins__"}
)

#: Scratch-dir filename the worker writes DF_OUT to, before the parent
#: copies it to the caller's `derived_dest`.
_DERIVED_FILENAME: str = "derived_out.parquet"


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
    #: Clean spec from an optional top-level `CHART` (src/core/chart_spec.py),
    #: or the reason it was rejected — a bad chart never fails the run.
    chart: dict[str, Any] | None = None
    chart_error: str | None = None
    #: Where an optional `DF_OUT` DataFrame was copied (the caller's
    #: `derived_dest`), with its shape, or why it wasn't saved.
    derived_path: str | None = None
    derived_rows: int | None = None
    derived_columns: list[str] | None = None
    derived_error: str | None = None
    #: Finding dicts produced by `dsa.run(...)` calls inside the code.
    tool_findings: list[dict[str, Any]] = field(default_factory=list)
    #: Which backend ran the code ("subprocess" | "docker" | "refused") —
    #: recorded in the governance audit log.
    backend: str = ""


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

    if len(code) > MAX_CODE_CHARS:
        return (
            "static_check",
            f"Code is {len(code):,} characters; the limit is {MAX_CODE_CHARS:,}. "
            "Split the work into smaller steps.",
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _blocked_module(alias.name):
                    return (
                        "static_check",
                        f"Only {ALLOWED_MODULES_TEXT} are available (without their "
                        f"I/O submodules). Remove the import for '{alias.name}'.",
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level or _blocked_module(node.module or ""):
                return (
                    "static_check",
                    f"Only {ALLOWED_MODULES_TEXT} are available (without their "
                    f"I/O submodules). Remove the import from '{node.module}'.",
                )
            for alias in node.names:
                if alias.name == "*" or _blocked_attr(alias.name):
                    return (
                        "static_check",
                        f"Importing '{alias.name}' from {node.module} is not allowed in the "
                        "sandbox (file/network I/O, private names and wildcard imports are blocked).",
                    )
        elif isinstance(node, ast.Name) and (
            node.id in BLOCKED_BUILTIN_NAMES or node.id.startswith("__")
        ):
            return (
                "static_check",
                f"'{node.id}' is not available in the sandbox. Use df, pandas/numpy/"
                "scipy/sklearn operations and the dsa toolkit instead.",
            )
        elif isinstance(node, ast.Attribute) and _blocked_attr(node.attr):
            return (
                "static_check",
                f"Attribute '.{node.attr}' is not allowed in the sandbox (private names, "
                "interpreter internals and file/network I/O are blocked). df is already "
                "loaded; use DF_OUT with save_as to persist a derived dataset.",
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and (
            "__" in node.value or "._" in node.value
            or _FORMAT_FIELD_WALK_RE.search(node.value)
        ):
            return (
                "static_check",
                "String literals may not contain '__', '._' or format fields that access "
                "attributes/indexes (e.g. '{0.x}'). Use f-strings with plain variables.",
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


#: Host environment variables the worker inherits — an allowlist, so an API
#: key (or any secret added later) is never reachable from sandboxed code.
#: Windows needs SYSTEMROOT/WINDIR/PATH to start Python and load DLLs.
_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "LANG", "LC_ALL", "TZ",
    "VIRTUAL_ENV",
)


def _worker_env(scratch_dir: str) -> dict[str, str]:
    """Minimal child environment: allowlisted host variables, the repo root
    on PYTHONPATH (so `src.core.io` imports from a scratch cwd), TEMP pointed
    at the scratch dir, one thread per numeric library (so the RSS cap and
    timeout bound the real work), and no bytecode writes."""
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    env.update({
        "PYTHONPATH": str(_REPO_ROOT),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TEMP": scratch_dir, "TMP": scratch_dir, "TMPDIR": scratch_dir,
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "LOKY_MAX_CPU_COUNT": "1",
    })
    return env


def _tree_rss_mb(proc: psutil.Process) -> float:
    """RSS of the worker plus every descendant — a library that forks a
    helper must not escape the memory cap."""
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            continue
    return total / (1024 * 1024)


def _kill_tree(proc: psutil.Process | None, popen: subprocess.Popen[bytes]) -> None:
    """Kill the worker and all its descendants (Popen.kill alone orphans them)."""
    victims: list[psutil.Process] = []
    if proc is not None:
        try:
            victims = proc.children(recursive=True)
        except psutil.Error:
            victims = []
    popen.kill()
    for child in victims:
        try:
            child.kill()
        except psutil.Error:
            continue
    popen.wait()


def _env_number(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _result_from_payload(
    payload: dict[str, Any],
    duration_ms: float,
    scratch_derived: Path,
    derived_dest: str | None,
) -> SandboxResult:
    """Build the SandboxResult for a worker payload, copying a DF_OUT file out
    of the scratch dir first — it is deleted with the TemporaryDirectory, so
    callers must invoke this inside that block."""
    derived_path = None
    derived_error = payload.get("derived_error")
    if payload.get("status") == "ok" and derived_dest and scratch_derived.exists():
        try:
            Path(derived_dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(scratch_derived, derived_dest)
            derived_path = str(derived_dest)
        except OSError as exc:
            derived_error = f"Could not save the derived dataset: {exc}"
    return SandboxResult(
        status=payload["status"],
        result=payload.get("result"),
        finding=payload.get("finding"),
        stdout=payload.get("stdout", ""),
        error_type=payload.get("error_type"),
        traceback=payload.get("traceback"),
        hint=payload.get("hint"),
        duration_ms=duration_ms,
        chart=payload.get("chart"),
        chart_error=payload.get("chart_error"),
        derived_path=derived_path,
        derived_rows=payload.get("derived_rows") if derived_path else None,
        derived_columns=payload.get("derived_columns") if derived_path else None,
        derived_error=derived_error,
        tool_findings=payload.get("tool_findings") or [],
    )


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


class SandboxBackend(ABC):
    """Abstract strategy for executing code in an isolated sandbox."""

    @abstractmethod
    def execute(
        self,
        code: str,
        dataset_ref: str,
        extra_globals: dict[str, Any] | None = None,
        prior_results: dict[str, Any] | None = None,
        timeout_s: float = 20.0,
        memory_limit_mb: int = 512,
        derived_dest: str | None = None,
    ) -> SandboxResult:
        """Execute code against dataset_ref and return a structured result."""
        ...


class SubprocessSandbox(SandboxBackend):
    """
    Subprocess sandbox backend.

    Executes code in a separate Python subprocess running `_sandbox_worker.py`
    with restricted builtins, AST static validation, memory capping, and
    execution timeouts. Zero external dependencies.
    """

    def execute(
        self,
        code: str,
        dataset_ref: str,
        extra_globals: dict[str, Any] | None = None,
        prior_results: dict[str, Any] | None = None,
        timeout_s: float = 20.0,
        memory_limit_mb: int = 512,
        derived_dest: str | None = None,
    ) -> SandboxResult:
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
            scratch_derived = Path(scratch_dir) / _DERIVED_FILENAME
            input_path.write_text(
                json.dumps(
                    {
                        "code": code,
                        "dataset_ref": dataset_ref,
                        "extra_globals": extra_globals,
                        "prior_results": prior_results or {},
                        "derived_output_path": str(scratch_derived) if derived_dest else None,
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.Popen(
                # -s: no user site-packages. Not -I, which would also drop the
                # PYTHONPATH the worker needs to import src.*.
                [sys.executable, "-s", str(_WORKER_SCRIPT), str(input_path), str(result_path)],
                cwd=scratch_dir,
                env=_worker_env(scratch_dir),
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
                            if _tree_rss_mb(ps_proc) > memory_limit_mb:
                                killed_as = "memory"
                        except psutil.NoSuchProcess:
                            pass
                    if killed_as is not None:
                        _kill_tree(ps_proc, proc)
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

            if result_path.stat().st_size > MAX_RESULT_BYTES:
                return SandboxResult(
                    status="error", result=None, finding=None, stdout="",
                    error_type="output_invalid", traceback=None,
                    hint=(
                        f"The result is larger than {MAX_RESULT_BYTES // 1_000_000} MB. "
                        "Aggregate before assigning RESULT (group, summarise, head)."
                    ),
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

            return _result_from_payload(payload, duration_ms, scratch_derived, derived_dest)


class DockerSandbox(SandboxBackend):
    """
    Hardened container sandbox backend using Docker.

    Provides true OS-level isolation (namespaces, cgroups, network=none,
    read-only filesystem, non-root user).
    Ready for future deployment or production servers.
    """

    def __init__(self, image_tag: str = "dsa-sandbox:latest") -> None:
        self.image_tag = image_tag

    @staticmethod
    def is_available() -> bool:
        """Return True if docker executable exists and daemon is responsive."""
        docker_bin = shutil.which("docker")
        if not docker_bin:
            return False
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            return res.returncode == 0
        except Exception:
            return False

    def execute(
        self,
        code: str,
        dataset_ref: str,
        extra_globals: dict[str, Any] | None = None,
        prior_results: dict[str, Any] | None = None,
        timeout_s: float = 20.0,
        memory_limit_mb: int = 512,
        derived_dest: str | None = None,
    ) -> SandboxResult:
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

        if not Path(dataset_ref).exists():
            return SandboxResult(
                status="error", result=None, finding=None, stdout="",
                error_type="runtime", traceback=None,
                hint=f"Dataset file not found: {dataset_ref}",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        with tempfile.TemporaryDirectory(
            prefix="docker_sandbox_", ignore_cleanup_errors=True
        ) as scratch_dir:
            scratch_path = Path(scratch_dir)
            dataset_source = Path(dataset_ref)
            dataset_target = scratch_path / f"dataset{dataset_source.suffix}"
            try:
                shutil.copy2(dataset_source, dataset_target)
            except Exception as exc:
                return SandboxResult(
                    status="error", result=None, finding=None, stdout="",
                    error_type="runtime", traceback=str(exc),
                    hint="Failed to prepare dataset for Docker sandbox.",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )

            input_path = scratch_path / "input.json"
            result_path = scratch_path / "result.json"

            container_input = "/scratch/input.json"
            container_result = "/scratch/result.json"
            container_data = f"/scratch/{dataset_target.name}"
            scratch_derived = scratch_path / _DERIVED_FILENAME

            input_path.write_text(
                json.dumps(
                    {
                        "code": code,
                        "dataset_ref": container_data,
                        "extra_globals": extra_globals,
                        "prior_results": prior_results or {},
                        "derived_output_path": (
                            f"/scratch/{_DERIVED_FILENAME}" if derived_dest else None
                        ),
                    }
                ),
                encoding="utf-8",
            )

            cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "--read-only",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "--env", "TMPDIR=/tmp",
                "--tmpfs", "/tmp:rw,size=100m",
                "-v", f"{scratch_path.resolve()}:/scratch:rw",
                f"--memory={memory_limit_mb}m",
                "--cpus=1.0",
                "--pids-limit=50",
                self.image_tag,
                container_input,
                container_result,
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=timeout_s,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(
                    status="error", result=None, finding=None, stdout="",
                    error_type="timeout",
                    traceback=None,
                    hint=f"Docker container execution exceeded {timeout_s:.0f}s.",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
            except Exception as exc:
                return SandboxResult(
                    status="error", result=None, finding=None, stdout="",
                    error_type="runtime",
                    traceback=str(exc),
                    hint=f"Failed to run Docker sandbox: {exc}",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )

            duration_ms = (time.perf_counter() - t0) * 1000

            if not result_path.exists():
                return SandboxResult(
                    status="error", result=None, finding=None, stdout="",
                    error_type="runtime",
                    traceback=proc.stderr or proc.stdout,
                    hint="The Docker container exited without writing a result.",
                    duration_ms=duration_ms,
                )

            if result_path.stat().st_size > MAX_RESULT_BYTES:
                return SandboxResult(
                    status="error", result=None, finding=None, stdout="",
                    error_type="output_invalid", traceback=None,
                    hint=(
                        f"The result is larger than {MAX_RESULT_BYTES // 1_000_000} MB. "
                        "Aggregate before assigning RESULT (group, summarise, head)."
                    ),
                    duration_ms=duration_ms,
                )
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                return SandboxResult(
                    status="error", result=None, finding=None, stdout="",
                    error_type="runtime",
                    traceback=str(exc),
                    hint="The Docker container produced an unreadable result.",
                    duration_ms=duration_ms,
                )

            return _result_from_payload(payload, duration_ms, scratch_derived, derived_dest)


def isolation_required() -> bool:
    """SANDBOX_REQUIRE_ISOLATION=true: LLM code may only run behind the
    Docker kernel boundary — never on the subprocess fallback."""
    return os.environ.get("SANDBOX_REQUIRE_ISOLATION", "").strip().lower() in ("1", "true", "yes")


def get_sandbox_backend(backend_name: str | None = None) -> SandboxBackend | None:
    """
    Return the active sandbox backend strategy.

    Resolution:
    1. Explicit backend_name argument ('docker' or 'subprocess').
    2. Environment variable SANDBOX_BACKEND ('docker' or 'subprocess').
    3. If 'docker' is requested, validates Docker daemon is available.
       Defaults to SubprocessSandbox (zero external dependencies).

    Returns None when isolation is required (`isolation_required()`) but
    Docker is unavailable — the caller must refuse to execute, not fall back.
    """
    target = (backend_name or os.environ.get("SANDBOX_BACKEND", "")).strip().lower()
    if (target == "docker" or isolation_required()) and DockerSandbox.is_available():
        return DockerSandbox()
    if isolation_required():
        return None
    return SubprocessSandbox()


def run_sandboxed(
    code: str,
    dataset_ref: str,
    extra_globals: dict[str, Any] | None = None,
    prior_results: dict[str, Any] | None = None,
    timeout_s: float | None = None,
    memory_limit_mb: int | None = None,
    backend: SandboxBackend | None = None,
    derived_dest: str | None = None,
) -> SandboxResult:
    """
    Execute `code` against the dataset at `dataset_ref` in an isolated
    sandbox and return a structured result.

    Delegates to the active SandboxBackend (SubprocessSandbox or DockerSandbox).
    `extra_globals`, when given, is JSON-validated and threaded into the
    worker's restricted execution namespace as additional top-level names.
    `prior_results`, when given, is made available inside the sandbox as
    a global dictionary named PRIOR_RESULTS. A `dsa` toolkit
    (src/core/sandbox_toolkit.py) is always pre-bound. If the code sets
    DF_OUT and `derived_dest` is given, the DataFrame is saved there as
    parquet. Unset limits default to env SANDBOX_TIMEOUT_S (45) and
    SANDBOX_MEMORY_MB (1024).
    """
    if timeout_s is None:
        timeout_s = _env_number("SANDBOX_TIMEOUT_S", 45.0)
    if memory_limit_mb is None:
        memory_limit_mb = int(_env_number("SANDBOX_MEMORY_MB", 1024))
    active_backend = backend or get_sandbox_backend()
    if active_backend is None:
        return SandboxResult(
            status="error", result=None, finding=None, stdout="",
            error_type="isolation_unavailable", traceback=None,
            hint=(
                "Code execution is disabled: SANDBOX_REQUIRE_ISOLATION is set and the "
                "Docker sandbox is unavailable. Use the built-in tools instead."
            ),
            duration_ms=0.0, backend="refused",
        )
    result = active_backend.execute(
        code=code,
        dataset_ref=dataset_ref,
        extra_globals=extra_globals,
        prior_results=prior_results,
        timeout_s=timeout_s,
        memory_limit_mb=memory_limit_mb,
        derived_dest=derived_dest,
    )
    result.backend = "docker" if isinstance(active_backend, DockerSandbox) else "subprocess"
    return result
