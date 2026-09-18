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
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.io import read_any
from src.core.profiler import profile_dataframe
from src.core.sandbox import (
    ALLOWED_MODULES,
    ALLOWED_MODULES_TEXT,
    BLOCKED_BUILTIN_NAMES,
    RESERVED_GLOBAL_NAMES,
    RESULT_VAR_NAME,
    STDOUT_CAP_CHARS,
)

#: Optional companion to RESULT — a top-level `FINDING = {...}` in the
#: sandboxed code, JSON-converted the same way RESULT is. Never required.
FINDING_VAR_NAME: str = "FINDING"

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
        records = value.head(RESULT_ROW_CAP).to_dict(orient="records")
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
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _restricted_import(
    real_import: Any,
) -> Any:
    def _import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        top_level = name.split(".")[0]
        if top_level not in ALLOWED_MODULES:
            raise ImportError(
                f"Only {ALLOWED_MODULES_TEXT} are available. "
                f"Import of '{name}' is not permitted in the sandbox."
            )
        return real_import(name, globals, locals, fromlist, level)

    return _import


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
) -> dict[str, Any]:
    restricted: dict[str, Any] = {
        "df": df,
        "SCHEMA": schema,
    }
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


def _execute(
    code: str, dataset_ref: str, extra_globals: dict[str, Any] | None = None
) -> dict[str, Any]:
    t0 = time.perf_counter()

    try:
        df, _report = read_any(dataset_ref)
    except Exception:
        return _error_payload(
            "runtime", traceback.format_exc(), "Failed to load the dataset.", t0, ""
        )

    schema = _build_schema(df)
    restricted_globals = _build_restricted_globals(df, schema, extra_globals)
    stdout_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(compile(code, "<sandboxed_code>", "exec"), restricted_globals)
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
        # self-diagnosing tracebacks unassisted. Fall back to the
        # exception's own message when it's short enough to stand alone;
        # a very long message is more noise than help as a one-line hint.
        message = str(exc)
        hint = message if message and len(message) <= 200 else None
        return _error_payload(
            "runtime", traceback.format_exc(), hint, t0, stdout_buf.getvalue()
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

    return {
        "status": "ok",
        "result": converted,
        "finding": finding_payload,
        "stdout": _cap(stdout_buf.getvalue()),
        "error_type": None,
        "traceback": None,
        "hint": None,
        "duration_ms": (time.perf_counter() - t0) * 1000,
    }


def main() -> None:
    input_path, result_path = sys.argv[1], sys.argv[2]
    payload_in = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = _execute(
        payload_in["code"],
        payload_in["dataset_ref"],
        payload_in.get("extra_globals"),
    )
    Path(result_path).write_text(
        json.dumps(result, default=_json_default), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
