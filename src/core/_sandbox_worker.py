"""
Sandbox worker — runs as a standalone subprocess (see src/core/sandbox.py
run_sandboxed). Loads the dataset, builds a restricted execution
namespace, execs the caller's code, and writes a JSON result.

Not imported by anything outside tests and sandbox.py's subprocess
spawn — this is the untrusted-code boundary, kept minimal on purpose.
"""
from __future__ import annotations

import builtins as _builtins_module
import contextlib
import datetime
import decimal
import difflib
import io
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from src.core.chart_spec import validate_chart_spec
from src.core.io import read_any
from src.core.profiler import profile_dataframe
from src.core.sandbox import (
    ALLOWED_MODULES_TEXT,
    BLOCKED_BUILTIN_NAMES,
    RESERVED_GLOBAL_NAMES,
    RESULT_VAR_NAME,
    STDOUT_CAP_CHARS,
    _blocked_module,
)
from src.core.sandbox_toolkit import Toolkit, frame_for_disk

#: Optional companion to RESULT — a top-level `FINDING = {...}` in the
#: sandboxed code, JSON-converted the same way RESULT is. Never required.
FINDING_VAR_NAME: str = "FINDING"

#: Optional declarative chart (src/core/chart_spec.py) and optional derived
#: DataFrame to persist — both never required.
CHART_VAR_NAME: str = "CHART"
DF_OUT_VAR_NAME: str = "DF_OUT"

#: `src.*` is importable here because run_sandboxed (src/core/sandbox.py)
#: sets PYTHONPATH to the repo root on this subprocess's environment —
#: no sys.path manipulation needed in this file. The subprocess itself
#: still runs with cwd set to a scratch temp directory.

#: DataFrame/Series results are capped at this many rows when converted
#: to RESULT — matches the spec's proposed default.
RESULT_ROW_CAP = 1_000


def _cap(text: str) -> str:
    if len(text) <= STDOUT_CAP_CHARS:
        return text
    hidden = len(text) - STDOUT_CAP_CHARS
    return text[:STDOUT_CAP_CHARS] + f"\n...[truncated, {hidden} more characters]"


def _build_schema(df: pd.DataFrame) -> dict[str, str]:
    profile = profile_dataframe(df)
    return {col.name: col.kind for col in profile.columns}


def _convert_result(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        truncated = len(value) > RESULT_ROW_CAP
        head = value.head(RESULT_ROW_CAP)
        if not isinstance(head.index, pd.RangeIndex):
            # A groupby/value_counts result keeps its keys in the index.
            head = head.reset_index(allow_duplicates=True)
        records = head.to_dict(orient="records")
        return {"__truncated__": True, "rows": records} if truncated else records
    if isinstance(value, pd.Series):
        return _convert_result(value.to_frame())
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if value is pd.NA or isinstance(
        value, (pd.Timedelta, pd.Period, pd.Interval, decimal.Decimal)
    ):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _restricted_import(
    real_import: Any,
) -> Any:
    # Only the sandboxed code's own `import` statements come through here —
    # library-internal imports resolve against each library's real builtins.
    def _import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level or _blocked_module(name) or any(
            str(item).startswith("_") or _blocked_module(f"{name}.{item}") for item in fromlist or ()
        ):
            raise ImportError(
                f"Only {ALLOWED_MODULES_TEXT} are available (without their I/O "
                f"submodules). Import of '{name}' is not permitted in the sandbox."
            )
        return real_import(name, globals, locals, fromlist, level)

    return _import


class SandboxPolicyError(PermissionError):
    """Raised by the runtime audit hook. A PermissionError so library code
    that already tolerates an unwritable path (e.g. bytecode caching) keeps
    working, while the sandboxed code sees a clear, fixable message."""


#: Audit events denied outright once sandboxed code starts: process
#: creation, networking, raw memory access, environment mutation, registry.
_DENIED_EVENTS: frozenset[str] = frozenset({
    "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
    "os.startfile", "os.kill", "os.killpg", "os.putenv", "os.unsetenv",
    "subprocess.Popen", "_winapi.CreateProcess", "pty.spawn",
    "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname",
    "socket.gethostbyaddr", "socket.sendto", "socket.sendmsg", "urllib.Request",
    "webbrowser.open", "ctypes.cdata", "ctypes.cdata/buffer", "ctypes.string_at",
    "ctypes.wstring_at",
})
# Not denied, deliberately: sys._getframe (pandas warnings, namedtuple),
# object.__setattr__ (lazy library imports patch classes) and ctypes.dlopen
# (threadpoolctl inspects BLAS) are routine library behaviour; the static
# `_`-attribute ban already keeps sandboxed code itself away from them.
_DENIED_PREFIXES: tuple[str, ...] = ("winreg.", "msvcrt.", "_posixsubprocess.")

#: Filesystem-mutating events: allowed only inside the scratch dir.
_PATH_WRITE_EVENTS: frozenset[str] = frozenset({
    "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown",
    "os.link", "os.symlink", "os.truncate", "os.utime", "shutil.rmtree",
    "shutil.copyfile", "shutil.copytree", "shutil.move", "shutil.make_archive",
})
#: Directory-listing events: allowed wherever reads are.
_PATH_READ_EVENTS: frozenset[str] = frozenset({"os.listdir", "os.scandir", "glob.glob"})

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


#: Audit hooks cannot be removed, so the hook is installed once per process
#: and gated by these. `_policy_enforced` is a plain flag, not thread-local:
#: a thread that sandboxed code starts (via joblib/sklearn callbacks) must
#: stay under the policy too.
_policy_installed: bool = False
_policy_enforced: bool = False
_policy_scratch: Path = Path(".")


@contextlib.contextmanager
def _policy_enforcement(scratch_dir: Path) -> Iterator[None]:
    """Enforce the runtime policy for the duration of the block — sandboxed
    code, the libraries it calls, dsa.run tools, and result serialisation."""
    global _policy_enforced, _policy_scratch
    _policy_scratch = scratch_dir.resolve()
    _install_runtime_policy()
    _policy_enforced = True
    try:
        yield
    finally:
        _policy_enforced = False


def _install_runtime_policy() -> None:
    """
    Install the sandbox's runtime policy as a sys.addaudithook hook.

    Reads are allowed from the scratch dir, the interpreter's own library
    directories (lazy imports, tz data) and the project's src/ package
    (dsa.run imports tools lazily); writes only inside the scratch dir.
    """
    global _policy_installed
    if _policy_installed:
        return
    _policy_installed = True

    read_roots = {Path(__file__).resolve().parents[1]}  # src/
    for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}:
        read_roots.add(Path(prefix).resolve())

    def _within(raw: Any, roots: set[Path]) -> bool:
        if isinstance(raw, int):
            return True  # an already-open descriptor; opening it was checked
        try:
            path = Path(os.fsdecode(raw))
            path = (path if path.is_absolute() else _policy_scratch / path).resolve()
        except (TypeError, ValueError, OSError):
            return False
        return any(path == root or root in path.parents for root in roots)

    busy = threading.local()

    def _hook(event: str, args: tuple[Any, ...]) -> None:
        if not _policy_enforced:
            return
        # Path resolution below can raise audit events of its own; checking
        # those recursively would loop, and they only serve this check.
        if getattr(busy, "active", False):
            return
        busy.active = True
        try:
            _check(event, args)
        finally:
            busy.active = False

    def _check(event: str, args: tuple[Any, ...]) -> None:
        if event in _DENIED_EVENTS or event.startswith(_DENIED_PREFIXES):
            raise SandboxPolicyError(f"Sandbox policy: '{event}' is not permitted.")
        if event == "open":
            path, mode, flags = (*args, None, None, None)[:3]
            writes = (
                any(c in str(mode) for c in "wax+") if isinstance(mode, str)
                else bool(isinstance(flags, int) and flags & _WRITE_FLAGS)
            )
            if not _within(path, {_policy_scratch} if writes else read_roots | {_policy_scratch}):
                raise SandboxPolicyError(
                    f"Sandbox policy: {'writing' if writes else 'reading'} files outside the "
                    "sandbox is not permitted. df is already loaded; use DF_OUT to save data."
                )
        elif event in _PATH_WRITE_EVENTS:
            if not all(_within(a, {_policy_scratch}) for a in args[:2] if isinstance(a, (str, bytes, os.PathLike))):
                raise SandboxPolicyError(f"Sandbox policy: '{event}' outside the sandbox is not permitted.")
        elif event in _PATH_READ_EVENTS and args and not _within(args[0], read_roots | {_policy_scratch}):
            raise SandboxPolicyError(f"Sandbox policy: '{event}' outside the sandbox is not permitted.")

    sys.addaudithook(_hook)


def _build_restricted_builtins() -> dict[str, Any]:
    safe = {
        name: obj
        for name, obj in vars(_builtins_module).items()
        if name not in BLOCKED_BUILTIN_NAMES
    }
    safe["__import__"] = _restricted_import(_builtins_module.__import__)
    return safe


def _build_restricted_globals(
    df: pd.DataFrame,
    schema: dict[str, str],
    extra_globals: dict[str, Any] | None = None,
    prior_results: dict[str, Any] | None = None,
    toolkit: Toolkit | None = None,
) -> dict[str, Any]:
    restricted: dict[str, Any] = {
        "df": df,
        "pd": pd,
        "np": np,
        "scipy": scipy,
        "stats": stats,
        "math": math,
        "SCHEMA": schema,
        "PRIOR_RESULTS": prior_results if prior_results is not None else {},
    }
    if toolkit is not None:
        restricted["dsa"] = toolkit
    if extra_globals:
        # The parent process (run_sandboxed) already rejects any key
        # colliding with RESERVED_GLOBAL_NAMES before this subprocess is
        # even spawned; this skip is defense in depth, not the primary gate.
        for key, value in extra_globals.items():
            if key in RESERVED_GLOBAL_NAMES:
                continue
            restricted[key] = value
    restricted["__builtins__"] = _build_restricted_builtins()
    return restricted


def _error_payload(
    error_type: str,
    tb: str | None,
    hint: str | None,
    t0: float,
    stdout: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "result": None,
        "stdout": _cap(stdout),
        "error_type": error_type,
        "traceback": tb,
        "hint": hint,
        "duration_ms": (time.perf_counter() - t0) * 1000,
    }


def _runtime_hint(exc: Exception, code: str, df: pd.DataFrame) -> str:
    """Error type + message, the failing line of the user's code, and for a
    KeyError the closest real column names — small models fix code far more
    reliably when told where it broke and what the column is actually called."""
    message = str(exc)
    msg_snippet = f": {message[:180]}…" if len(message) > 180 else (f": {message}" if message else "")
    hint = f"{type(exc).__name__}{msg_snippet}"

    # <sandboxed_code> has no linecache entry, so read the line from `code`.
    lineno = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<sandboxed_code>":
            lineno = frame.lineno
    lines = code.splitlines()
    if lineno is not None and 0 < lineno <= len(lines):
        hint += f" (line {lineno}: {lines[lineno - 1].strip()[:160]})"

    if isinstance(exc, KeyError) and exc.args:
        columns = [str(c) for c in df.columns]
        key = str(exc.args[0])
        names = re.findall(r"'([^']+)'", key) if "not in index" in key else [key]
        for name in names:
            if name in columns:
                continue
            close = difflib.get_close_matches(name, columns, n=5, cutoff=0.5)
            hint += (
                f" '{name}' is not a column of df; closest: {close}." if close
                else f" df columns are: {columns[:30]}."
            )
    return hint


def _save_derived(value: Any, derived_output_path: str | None) -> dict[str, Any]:
    """Persist DF_OUT for the parent to copy out. The worker writes it — user
    code never gets file access."""
    if not isinstance(value, pd.DataFrame):
        return {"derived_error": f"DF_OUT must be a pandas DataFrame, got {type(value).__name__}."}
    if not derived_output_path:
        return {"derived_error": "DF_OUT was set but this call does not save derived data (pass save_as)."}
    frame = frame_for_disk(value)
    try:
        frame.to_parquet(derived_output_path, index=False)
    except Exception as exc:
        return {"derived_error": f"Could not save DF_OUT as parquet: {exc}"}
    return {"derived_rows": len(frame), "derived_columns": [str(c) for c in frame.columns]}


def _execute(
    code: str,
    dataset_ref: str,
    extra_globals: dict[str, Any] | None = None,
    prior_results: dict[str, Any] | None = None,
    scratch_dir: str | None = None,
    derived_output_path: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()

    try:
        df, _report = read_any(dataset_ref)
    except Exception:
        return _error_payload(
            "runtime", traceback.format_exc(), "Failed to load the dataset.", t0, ""
        )

    schema = _build_schema(df)
    toolkit = Toolkit(df, scratch_dir or Path.cwd())
    restricted_globals = _build_restricted_globals(
        df, schema, extra_globals, prior_results, toolkit
    )
    with _policy_enforcement(Path(scratch_dir or Path.cwd())):
        return _run_restricted(
            code, df, restricted_globals, toolkit, derived_output_path, t0
        )


def _run_restricted(
    code: str,
    df: pd.DataFrame,
    restricted_globals: dict[str, Any],
    toolkit: Toolkit,
    derived_output_path: str | None,
    t0: float,
) -> dict[str, Any]:
    """Exec the caller's code and package its outputs; runs under the runtime
    policy, which also covers serialising objects the code produced."""
    stdout_buf = io.StringIO()

    try:
        compiled = compile(code, "<sandboxed_code>", "exec")
        with contextlib.redirect_stdout(stdout_buf):
            exec(compiled, restricted_globals)
    except ImportError as exc:
        return _error_payload(
            "import_blocked", traceback.format_exc(), str(exc), t0, stdout_buf.getvalue()
        )
    except SyntaxError as exc:
        return _error_payload(
            "syntax",
            traceback.format_exc(),
            f"Code has a syntax error: {exc.msg} (line {exc.lineno}).",
            t0,
            stdout_buf.getvalue(),
        )
    except Exception as exc:
        # Every failure should carry an actionable hint, not just a raw
        # traceback — lightweight models are markedly worse at
        # self-diagnosing tracebacks unassisted.
        return _error_payload(
            "runtime",
            traceback.format_exc(),
            _runtime_hint(exc, code, df),
            t0,
            stdout_buf.getvalue(),
        )
    except BaseException as exc:
        # SystemExit and friends must not skip the worker's own result write.
        return _error_payload(
            "runtime",
            traceback.format_exc(),
            f"{type(exc).__name__} is not allowed; assign the answer to RESULT instead.",
            t0,
            stdout_buf.getvalue(),
        )

    if RESULT_VAR_NAME not in restricted_globals:
        return _error_payload(
            "static_check",
            None,
            f"Your code must assign the final answer to a variable named "
            f"{RESULT_VAR_NAME} at the top level.",
            t0,
            stdout_buf.getvalue(),
        )

    raw_result = restricted_globals[RESULT_VAR_NAME]
    converted = _convert_result(raw_result)
    try:
        json.dumps(converted, default=_json_default)
    except TypeError:
        return _error_payload(
            "output_invalid",
            None,
            f"RESULT must be a plain value (number, string, list, dict) or a "
            f"DataFrame/array — got {type(raw_result).__name__}.",
            t0,
            stdout_buf.getvalue(),
        )

    # FINDING is an optional companion to RESULT — never required, and a
    # missing or unconvertible FINDING is never an execution error, just an
    # absent one.
    finding_payload = None
    if FINDING_VAR_NAME in restricted_globals:
        try:
            candidate = _convert_result(restricted_globals[FINDING_VAR_NAME])
            json.dumps(candidate, default=_json_default)
            finding_payload = candidate
        except Exception:
            finding_payload = None

    # CHART and DF_OUT are optional too: a bad one is reported back as
    # chart_error / derived_error, never turned into an execution failure.
    chart, chart_error = None, None
    raw_chart = restricted_globals.get(CHART_VAR_NAME)
    if raw_chart is not None:
        chart, chart_error = validate_chart_spec(raw_chart)
        if chart is not None and isinstance(raw_chart, dict) and raw_chart.get("truncated"):
            chart["truncated"] = True  # a dsa.chart spec was already capped once

    derived: dict[str, Any] = {}
    if restricted_globals.get(DF_OUT_VAR_NAME) is not None:
        derived = _save_derived(restricted_globals[DF_OUT_VAR_NAME], derived_output_path)

    return {
        "status": "ok",
        "result": converted,
        "finding": finding_payload,
        "chart": chart,
        "chart_error": chart_error,
        "tool_findings": toolkit.collected_findings(),
        **derived,
        "stdout": _cap(stdout_buf.getvalue()),
        "error_type": None,
        "traceback": None,
        "hint": None,
        "duration_ms": (time.perf_counter() - t0) * 1000,
    }


def main() -> None:
    input_path, result_path = sys.argv[1], sys.argv[2]
    payload_in = json.loads(Path(input_path).read_text(encoding="utf-8"))
    # input.json lives in the scratch dir under both backends (host temp dir
    # for SubprocessSandbox, /scratch for DockerSandbox, whose cwd is /app).
    result = _execute(
        payload_in["code"],
        payload_in["dataset_ref"],
        payload_in.get("extra_globals"),
        payload_in.get("prior_results"),
        scratch_dir=str(Path(input_path).parent),
        derived_output_path=payload_in.get("derived_output_path"),
    )
    Path(result_path).write_text(
        json.dumps(result, default=_json_default), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
