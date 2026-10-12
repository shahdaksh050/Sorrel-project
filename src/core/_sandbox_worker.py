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
import inspect
import io
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from src.core.chart_spec import validate_chart_spec
from src.core.io import read_any
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


# ---- Pandas LLM Compatibility Aliases ----
def _safe_reset_index(self: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
    res = self.reset_index(*args, **kwargs)
    if isinstance(res.columns, pd.MultiIndex):
        flat_cols = []
        for col in res.columns.values:
            parts = [str(p).strip() for p in col if str(p).strip() != ""]
            flat_cols.append("_".join(parts) if parts else "col")
        res.columns = pd.Index(flat_cols)
    return res


def _df_append(self: pd.DataFrame, other: Any, ignore_index: bool = False, **kwargs: Any) -> pd.DataFrame:
    if isinstance(other, dict):
        other = pd.DataFrame([other])
    elif isinstance(other, list):
        other = pd.DataFrame(other)
    elif isinstance(other, pd.Series):
        other = other.to_frame().T
    return pd.concat([self, other], ignore_index=ignore_index)


_PANDAS_COMPAT_ALIASES: list[tuple[type, str, Any]] = [
    (pd.DataFrame, "reset", _safe_reset_index),
    (pd.Series, "reset", pd.Series.reset_index),
    (pd.DataFrame, "to_dataframe", lambda self, *a, **k: self),
    (pd.Series, "to_dataframe", pd.Series.to_frame),
    (pd.DataFrame, "drop_column", lambda self, col, **k: self.drop(columns=[col] if isinstance(col, str) else col, **k)),
    (pd.DataFrame, "drop_columns", lambda self, cols, **k: self.drop(columns=cols, **k)),
    (pd.DataFrame, "rename_column", lambda self, mapping, **k: self.rename(columns=mapping, **k)),
    (pd.DataFrame, "rename_columns", lambda self, mapping, **k: self.rename(columns=mapping, **k)),
    (
        pd.DataFrame,
        "sort",
        lambda self, *a, **k: self.sort_values(*a, **k) if not a or isinstance(a[0], (str, list)) else self.sort_index(*a, **k),
    ),
    (pd.Series, "sort", pd.Series.sort_values),
    (pd.DataFrame, "append", _df_append),
]

for _cls, _attr_name, _impl in _PANDAS_COMPAT_ALIASES:
    if not hasattr(_cls, _attr_name):
        setattr(_cls, _attr_name, _impl)


def _build_schema(df: pd.DataFrame) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for col, dtype in df.dtypes.items():
        name = str(col)
        if pd.api.types.is_bool_dtype(dtype):
            kinds[name] = "boolean"
        elif pd.api.types.is_numeric_dtype(dtype):
            kinds[name] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            kinds[name] = "datetime"
        else:
            kinds[name] = "categorical"
    return kinds


def _clean_column_name(col: Any) -> str:
    if isinstance(col, tuple):
        parts = [str(p).strip() for p in col if str(p).strip() != ""]
        return "_".join(parts) if parts else "col"
    return str(col)


def _clean_dict_keys(val: Any) -> Any:
    if isinstance(val, dict):
        return {_clean_column_name(k): _clean_dict_keys(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_clean_dict_keys(item) for item in val]
    return val


def _convert_result(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        df_copy = value.copy()
        if isinstance(df_copy.columns, pd.MultiIndex):
            df_copy.columns = pd.Index([_clean_column_name(c) for c in df_copy.columns])
        truncated = len(df_copy) > RESULT_ROW_CAP
        head = df_copy.head(RESULT_ROW_CAP)
        if not isinstance(head.index, pd.RangeIndex):
            # A groupby/value_counts result keeps its keys in the index.
            head = head.reset_index(allow_duplicates=True)
            if isinstance(head.columns, pd.MultiIndex):
                head.columns = pd.Index([_clean_column_name(c) for c in head.columns])
        records = head.to_dict(orient="records")
        clean_records = _clean_dict_keys(records)
        return {"__truncated__": True, "rows": clean_records} if truncated else clean_records
    if isinstance(value, pd.Series):
        return _convert_result(value.to_frame())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (dict, list)):
        return _clean_dict_keys(value)
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
                f"submodules). Import of '{name}' is not permitted. Plots and models "
                "come from `dsa` (dsa.chart.*, dsa.run); pd, np, scipy, stats, math are pre-imported."
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
    "ctypes.wstring_at", "ctypes.call_function",
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

    _resolved_cache: dict[Any, Path | None] = {}

    def _within(raw: Any, roots: set[Path]) -> bool:
        if isinstance(raw, int):
            return True  # an already-open descriptor; opening it was checked
        path = _resolved_cache.get(raw)
        if path is None:
            try:
                p = Path(os.fsdecode(raw))
                path = (p if p.is_absolute() else _policy_scratch / p).resolve()
                if len(_resolved_cache) < 2048:
                    _resolved_cache[raw] = path
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
        "re": re,
        "json": json,
        "datetime": datetime,
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
        # df[[...]] -> "['a'] not in index"; .agg({...}) -> "Column(s) ['a'] do not exist";
        # groupby(...)['a'] -> "Column not found: a"; df['a'] / groupby(['a']) -> the bare key.
        names = (
            re.findall(r"'([^']+)'", key)
            if "not in index" in key or "do not exist" in key
            else [key.removeprefix("Column not found: ")]
        )
        for name in names:
            if name in columns:
                continue
            close = difflib.get_close_matches(name, columns, n=5, cutoff=0.5)
            hint += (
                f" '{name}' is not a column of df; closest: {close}." if close
                else f" df columns are: {columns[:30]}."
            )
    return hint + _dsa_hint(exc)


#: Names a weak model reaches for that the sandbox deliberately does not provide.
_UNAVAILABLE_NAMES = frozenset({"plt", "sns", "sklearn", "px", "go", "open", "requests"})


def _dsa_hint(exc: Exception) -> str:
    """Extra guidance (<= 300 chars) for the usual `dsa` mistakes: a wrong helper
    name, a wrong keyword, or a plotting/file library that is not installed."""
    if isinstance(exc, AttributeError):
        obj, bad = getattr(exc, "obj", None), str(getattr(exc, "name", "") or "")
        chart = vars(Toolkit.chart)
        if obj is Toolkit.chart:
            scope, names = "dsa.chart", sorted(chart)
        elif isinstance(obj, Toolkit):
            scope = "dsa"
            names = sorted(n for n in dir(Toolkit) if not n.startswith("_") and n != "collected_findings")
        else:
            return ""
        close = [f"dsa.chart.{bad}"] if scope == "dsa" and bad in chart else [
            f"{scope}.{n}" for n in difflib.get_close_matches(bad, names, n=1, cutoff=0.5)
        ]
        return f" `{scope}` has no `{bad}`. Available: {', '.join(names)}." + (
            f" Did you mean `{close[0]}`?" if close else ""
        )
    if isinstance(exc, NameError) and getattr(exc, "name", None) in _UNAVAILABLE_NAMES:
        return (
            f" `{exc.name}` is not available: no plotting, file or network libraries here. Charts: "
            "CHART = dsa.chart.bar/line/scatter/histogram(df, x, y, title=...). Models and tests: "
            "dsa.run(tool_name, df=frame, ...). df is already loaded."
            + (" sklearn needs an explicit `from sklearn.x import y`." if exc.name == "sklearn" else "")
        )
    if isinstance(exc, TypeError):
        m = re.match(r"([\w.]+)\(\) (?:got an unexpected keyword argument '(\w+)'|missing|takes)", str(exc))
        if not m:
            return ""
        qual, bad = m.group(1), m.group(2)
        last = qual.rsplit(".", 1)[-1]
        fn = getattr(Toolkit, last, None) if qual.startswith("Toolkit.") else getattr(Toolkit.chart, last, None)
        if not callable(fn):
            return ""
        params = [p for p in inspect.signature(fn).parameters.values() if p.name != "self"]
        names = [p.name for p in params if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
        close = difflib.get_close_matches(bad, names, n=1, cutoff=0.4) if bad else []
        more = " (+ options such as title, caption, color)" if any(p.kind is p.VAR_KEYWORD for p in params) else ""
        return f" `{last}` accepts: {', '.join(names)}{more}." + (f" Did you mean `{close[0]}`?" if close else "")
    return ""


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

    def _try_exec(exec_code: str) -> tuple[BaseException | None, str]:
        buf = io.StringIO()
        try:
            compiled = compile(exec_code, "<sandboxed_code>", "exec")
            with contextlib.redirect_stdout(buf):
                exec(compiled, restricted_globals)
            return None, buf.getvalue()
        except BaseException as exc:
            return exc, buf.getvalue()

    err, out_text = _try_exec(code)
    stdout_buf.write(out_text)

    # Self-heal on runtime error if possible
    if err is not None:
        healed_code: str | None = None
        # 1. KeyError auto-heal: try matching close column in df
        if isinstance(err, KeyError) and err.args:
            key_name = str(err.args[0]).removeprefix("Column not found: ").strip()
            df_cols = [str(c) for c in df.columns]
            matches = [c for c in df_cols if c.lower() == key_name.lower()] or difflib.get_close_matches(key_name, df_cols, n=1, cutoff=0.6)
            if matches:
                target_col = matches[0]
                healed_code = re.sub(rf"(['\"]){re.escape(key_name)}\1", rf"\1{target_col}\1", code)
        # 2. AttributeError on .reset():
        elif isinstance(err, AttributeError) and "reset" in str(err):
            healed_code = code.replace(".reset()", ".reset_index()")
        # 3. NameError on standard modules:
        elif isinstance(err, NameError) and getattr(err, "name", None) in ("re", "json", "datetime", "math"):
            mod_name = str(err.name)
            restricted_globals[mod_name] = __import__(mod_name)
            healed_code = code

        if healed_code and healed_code != code:
            retry_err, retry_out = _try_exec(healed_code)
            if retry_err is None:
                err = None
                code = healed_code
                stdout_buf.truncate(0)
                stdout_buf.seek(0)
                stdout_buf.write(retry_out)

    if err is not None:
        tb_str = "".join(traceback.format_exception(err))
        if isinstance(err, ImportError):
            return _error_payload(
                "import_blocked", tb_str, str(err), t0, stdout_buf.getvalue()
            )
        if isinstance(err, SyntaxError):
            return _error_payload(
                "syntax",
                tb_str,
                f"Code has a syntax error: {err.msg} (line {err.lineno}).",
                t0,
                stdout_buf.getvalue(),
            )
        if not isinstance(err, Exception):
            return _error_payload(
                "runtime",
                tb_str,
                f"{type(err).__name__} is not allowed; assign the answer to RESULT instead.",
                t0,
                stdout_buf.getvalue(),
            )
        return _error_payload(
            "runtime",
            tb_str,
            _runtime_hint(err, code, df),
            t0,
            stdout_buf.getvalue(),
        )

    # Auto-resolve RESULT if omitted
    if RESULT_VAR_NAME not in restricted_globals:
        for cand in (
            "result", "res", "output", "summary", "agg", "table", "ans",
            "answer", "df_out", "findings", "finding", "metrics", "data", "report",
        ):
            val = restricted_globals.get(cand)
            if val is not None and not isinstance(val, (types.ModuleType, Toolkit)):
                restricted_globals[RESULT_VAR_NAME] = val
                break
        else:
            if FINDING_VAR_NAME in restricted_globals or CHART_VAR_NAME in restricted_globals:
                restricted_globals[RESULT_VAR_NAME] = {"status": "ok", "message": "Analysis completed."}
            elif stdout_buf.getvalue().strip():
                restricted_globals[RESULT_VAR_NAME] = {"stdout": stdout_buf.getvalue().strip()}

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
