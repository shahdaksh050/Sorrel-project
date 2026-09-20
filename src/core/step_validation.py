"""
Plan validation — deterministic checks of a planned step against its
tool's schema and the dataset's columns, before the step spends a tool run.

Split out of ``src/core/controller.py``; ``AgentController`` keeps thin
``_is_column_param`` / ``_columns_for`` / ``_validate_step`` methods that
delegate here.
"""
from __future__ import annotations

import difflib
import inspect
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.tool_registry import _INJECTED_PARAMS

if TYPE_CHECKING:
    from src.core.memory import MemorySystem

#: Column-valued parameter names that don't end in _column/_col/_columns.
_COLUMN_PARAM_NAMES = frozenset({"columns", "stratify_by"})


def is_column_param(key: str) -> bool:
    return key.endswith(("_column", "_col", "_columns")) or key in _COLUMN_PARAM_NAMES


def columns_for(memory: MemorySystem, file_path: Any) -> list[str] | None:
    """Columns of the dataset a step will read: a registered derived
    dataset's own columns when file_path is one, else the loaded dataset's."""
    derived = memory.get_context("derived_datasets") or {}
    if isinstance(file_path, str) and file_path and derived:
        try:
            target = Path(file_path).resolve()
            for info in derived.values():
                if isinstance(info, dict) and info.get("path") and Path(info["path"]).resolve() == target:
                    cols = info.get("columns")
                    return [str(c) for c in cols] if isinstance(cols, list) and cols else None
        except OSError:
            pass
    meta = memory.dataset_metadata
    return list(meta.columns) if meta else None


_LIST_TYPES = frozenset({"array", "list", "list[string]", "list[str]"})
_INT_TYPES = frozenset({"int", "integer"})
_FLOAT_TYPES = frozenset({"float", "number"})
_BOOL_TYPES = frozenset({"bool", "boolean"})
_STR_TYPES = frozenset({"string", "str"})
_DROP = object()   # sentinel: the value is a placeholder for "not given"


def _unquote(text: str) -> str:
    """Strip stray backticks and a matching pair of quotes around a name."""
    text = text.strip().strip("`").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    return text


def _coerce_value(kind: str, value: Any, column: bool) -> Any:
    """`value` converted to the declared type `kind`; unchanged when it
    already fits or no safe conversion exists; _DROP for null-like strings."""
    if isinstance(value, str):
        low = value.strip().lower()
        # "none" is a real choice for a free-string option (strategy="none").
        if low in ("", "null") or (low == "none" and (column or kind not in _STR_TYPES)):
            return _DROP
    if kind in _LIST_TYPES:
        if isinstance(value, str):
            text = value.strip()
            items: Any = None
            if text.startswith("["):
                try:
                    items = json.loads(text)
                except ValueError:
                    items = None
            if not isinstance(items, list):
                items = [p.strip() for p in re.split(r"[,;\n]", text.strip("[]")) if p.strip()]
            value = items
        elif isinstance(value, (int, float)):   # bool is an int: a scalar too
            value = [value]
        if column and isinstance(value, list):
            value = [_unquote(v) if isinstance(v, str) else v for v in value]
        return value
    if kind in _STR_TYPES:
        if isinstance(value, list) and value and isinstance(value[0], str):
            value = value[0]
        if column and isinstance(value, str):
            value = _unquote(value)
        return value
    if isinstance(value, str):
        text = value.strip()
        low = text.lower()
        try:
            if kind in _INT_TYPES:
                number = float(text)
                return int(number) if number == int(number) else value
            if kind in _FLOAT_TYPES:
                return float(text)
        except (ValueError, OverflowError):
            return value
        if kind in _BOOL_TYPES:
            if low in ("true", "yes", "y"):
                return True
            if low in ("false", "no", "n"):
                return False
    return value


def _coerce_params(
    schema: dict[str, Any], values: dict[str, Any], skip: frozenset[str] | set[str]
) -> tuple[dict[str, Any], list[str]]:
    """Coerce `values` to the types `schema` declares. Returns the new dict
    and the keys that changed; a value that already has the right type is
    never touched."""
    out = dict(values)
    changed: list[str] = []
    for key, value in values.items():
        spec = schema.get(key)
        if key in skip or not isinstance(spec, dict):
            continue
        fixed = _coerce_value(str(spec.get("type", "")).strip().lower(), value, is_column_param(key))
        if fixed is _DROP:
            del out[key]
            changed.append(key)
        elif fixed != value or type(fixed) is not type(value):
            out[key] = fixed
            changed.append(key)
    return out, changed


def validate_step(
    memory: MemorySystem, tool: Any, raw: dict[str, Any], params: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[str], str]:
    """
    Check a step against its tool's schema before it runs. `raw` is the
    planner's parameters, `params` the prepared ones (injection done).
    Returns (params without unknown keys, dropped keys, errors, note); any
    error means the step must not run. Unknown parameters are dropped,
    not fatal. Values are coerced to the declared types (a comma string for
    a list, "12" for an int); the note says how many were normalised. A
    tool without a schema is not validated (fail open).
    """
    try:
        schema = tool.get_schema()
    except Exception:
        return params, [], [], ""
    if not isinstance(schema, dict) or not schema:
        return params, [], [], ""
    injected = _INJECTED_PARAMS | set(getattr(tool, "requires_context", {}).values())
    try:
        accepted = {
            name for name, p in inspect.signature(tool.execute).parameters.items()
            if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        }
    except (TypeError, ValueError):
        accepted = set()
    known = set(schema) | injected | accepted
    dropped = sorted(k for k in raw if k not in known)
    params = {k: v for k, v in params.items() if k not in dropped}

    # Coerce the planner's values, then carry each fix onto the prepared
    # params — unless prepare_params already rewrote that value itself.
    original_raw = raw
    raw, changed = _coerce_params(schema, {k: v for k, v in raw.items() if k not in dropped}, injected)
    for key in changed:
        if key in params and params[key] == original_raw[key]:
            if key in raw:
                params[key] = raw[key]
            else:
                del params[key]
    note = f"normalised {len(changed)} parameter{'s' if len(changed) != 1 else ''}" if changed else ""

    errors: list[str] = []
    missing = [
        k for k, spec in schema.items()
        if isinstance(spec, dict) and spec.get("required") and k not in injected
        and (params.get(k) is None or params.get(k) == "" or params.get(k) == [])
    ]
    if missing:
        errors.append(f"Missing required parameter(s): {', '.join(missing)}.")

    # LLM-authored code may build its own columns — only built-in tools'
    # column parameters are checked against the data.
    columns = None if getattr(tool, "executes_code", False) else columns_for(memory, params.get("file_path"))
    if columns:
        column_set = set(columns)
        for key, value in raw.items():
            if key in dropped or not is_column_param(key):
                continue
            for name in value if isinstance(value, list) else [value]:
                if not isinstance(name, str) or not name or name in column_set:
                    continue
                matches = [c for c in columns if c.lower() == name.lower()] or difflib.get_close_matches(
                    name, columns, n=3, cutoff=0.6
                )
                hint = f" Did you mean {' or '.join(repr(m) for m in matches)}?" if matches else ""
                errors.append(f"{key}={name!r} is not a column of this dataset.{hint}")
    return params, dropped, errors, note
