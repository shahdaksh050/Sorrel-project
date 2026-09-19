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


def validate_step(
    memory: MemorySystem, tool: Any, raw: dict[str, Any], params: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    """
    Check a step against its tool's schema before it runs. `raw` is the
    planner's parameters, `params` the prepared ones (injection done).
    Returns (params without unknown keys, dropped keys, errors); any
    error means the step must not run. Unknown parameters are dropped,
    not fatal. A tool without a schema is not validated (fail open).
    """
    try:
        schema = tool.get_schema()
    except Exception:
        return params, [], []
    if not isinstance(schema, dict) or not schema:
        return params, [], []
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
    return params, dropped, errors
