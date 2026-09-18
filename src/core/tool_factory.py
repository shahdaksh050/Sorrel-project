"""
Tool factory — turns a validated LLM-proposed tool spec into a persisted,
re-callable `GeneratedTool` (src/tools/generated_tool.py).

Round 8 item 8.2. This module is pure plumbing around three concerns:
  1. `GeneratedToolSpec` — the on-disk/in-memory shape of a generated tool.
  2. `validate_spec` — reject bad names/schemas/code *before* anything is
     built or persisted (called by the controller after
     `DefineAnalysisToolTool` reports `status == "ready"`, and again just
     before registration since the controller is the only thing that knows
     `existing_tool_names`/`existing_generated`).
  3. `register_and_persist` / `load_persisted_tools` — the controller's own
     side effects (AGENTS.md layer rule: tools stay pure, the controller
     performs registration/persistence). Nothing in this module is called
     from `GeneratedTool.execute()` or `DefineAnalysisToolTool.execute()`.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.sandbox import RESERVED_GLOBAL_NAMES, _static_check

#: Hard cap on generated tools per run (design decision 9) — the
#: controller is expected to enforce this before ever calling
#: register_and_persist again; validate_spec does not count existing
#: tools itself since it has no notion of "this run" vs "all runs".
MAX_GENERATED_TOOLS = 6

#: snake_case: lowercase letters, digits, underscores; must start with a
#: letter; no leading/trailing/double underscores enforced loosely (good
#: enough to catch "MyTool", "my-tool", "2cool", "tool name").
_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Keys a single params_schema entry may use, mirroring BaseTool.get_schema's
#: documented shape: {param_name: {"type": ..., "description": ..., "required": bool}}.
_SCHEMA_ENTRY_REQUIRED_KEYS = {"type", "description", "required"}


@dataclass
class GeneratedToolSpec:
    """On-disk/in-memory representation of one LLM-defined tool."""

    name: str                       # snake_case, validated, not a built-in tool name
    description: str
    params_schema: dict[str, Any]   # exact BaseTool.get_schema() shape
    code: str
    version: int
    created_at: str                 # ISO timestamp
    dataset_fingerprint: str        # reuse src.core.io's path+mtime+size cache key
    source: str = "llm_generated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GeneratedToolSpec:
        return cls(
            name=payload["name"],
            description=payload["description"],
            params_schema=payload.get("params_schema", {}),
            code=payload["code"],
            version=payload.get("version", 1),
            created_at=payload.get("created_at", ""),
            dataset_fingerprint=payload.get("dataset_fingerprint", ""),
            source=payload.get("source", "llm_generated"),
        )


def compute_dataset_fingerprint(file_path: str) -> str:
    """
    Reuse src.core.io's (resolved path, mtime, size) cache key, flattened to
    a single string so it can live in a JSON spec / dataclass field. Falls
    back to the raw path if the file cannot be stat'd (matches io._cache_key
    returning None in that case) — a fallback fingerprint still lets
    same-run comparisons work; it only degrades cross-run reload safety,
    which load_persisted_tools already treats as opt-in.
    """
    from src.core.io import _cache_key  # local import: keep io's private helper internal

    key = _cache_key(file_path)
    if key is None:
        return f"unstat:{file_path}"
    resolved_path, mtime_ns, size = key
    return f"{resolved_path}:{mtime_ns}:{size}"


def _validate_name(name: str, existing_tool_names: list[str]) -> list[str]:
    errors: list[str] = []
    if not name or not isinstance(name, str):
        errors.append("Tool name must be a non-empty string.")
        return errors
    if not _SNAKE_CASE_RE.match(name):
        errors.append(
            f"Tool name '{name}' must be snake_case (lowercase letters, digits, "
            "underscores, starting with a letter) — e.g. 'average_order_value'."
        )
    if "__" in name:
        errors.append(f"Tool name '{name}' must not contain a double underscore.")
    if name.endswith("_"):
        errors.append(f"Tool name '{name}' must not end with an underscore.")
    if name in existing_tool_names:
        errors.append(
            f"'{name}' is already a built-in tool name and cannot be overwritten "
            "by a generated tool."
        )
    return errors


def _validate_params_schema(params_schema: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(params_schema, dict):
        errors.append("params_schema must be a dict of {param_name: {...}}.")
        return errors
    for param_name, entry in params_schema.items():
        if not isinstance(param_name, str) or not param_name:
            errors.append(f"params_schema has a non-string/empty key: {param_name!r}.")
            continue
        if param_name in RESERVED_GLOBAL_NAMES:
            errors.append(
                f"params_schema['{param_name}'] collides with a reserved sandbox "
                f"name ({', '.join(sorted(RESERVED_GLOBAL_NAMES))}) and would "
                "permanently fail at call time — rename the parameter."
            )
        if not isinstance(entry, dict):
            errors.append(f"params_schema['{param_name}'] must be a dict, got {type(entry).__name__}.")
            continue
        missing = _SCHEMA_ENTRY_REQUIRED_KEYS - entry.keys()
        if missing:
            errors.append(
                f"params_schema['{param_name}'] is missing keys: {sorted(missing)} "
                "(expected 'type', 'description', 'required')."
            )
        elif not isinstance(entry.get("required"), bool):
            errors.append(f"params_schema['{param_name}']['required'] must be a bool.")
    return errors


def validate_spec(
    name: str,
    description: str,
    params_schema: dict[str, Any],
    code: str,
    existing_tool_names: list[str],
    existing_generated: dict[str, Any],
) -> list[str]:
    """
    Returns a list of human-readable error strings; [] means valid.

    Rejects: bad snake_case, collision with a *built-in* name, malformed
    schema, code failing the relaxed static check (src.core.sandbox._static_check).
    Allows overwrite of a name already in `existing_generated` (that's a
    modification/self-correction, not a collision) — enforced by the caller
    excluding already-generated names from `existing_tool_names` before
    calling this, not by this function reading `existing_generated`'s values
    (accepted as `Any` since callers may hold either GeneratedToolSpec
    instances or their plain-dict serialisation, e.g. from
    MemorySystem.list_generated_tools()); it exists on the signature so a
    future check against prior versions has a natural place to live.
    """
    errors: list[str] = []

    errors.extend(_validate_name(name, existing_tool_names))

    if not description or not description.strip():
        errors.append("description must be a non-empty string.")

    errors.extend(_validate_params_schema(params_schema))

    if not code or not code.strip():
        errors.append("code must be a non-empty string.")
    else:
        static_error = _static_check(code)
        if static_error is not None:
            error_type, hint = static_error
            errors.append(f"Code failed static check ({error_type}): {hint}")

    return errors


def register_and_persist(
    spec: GeneratedToolSpec,
    tool_registry: Any,
    memory: Any,
    output_root: str,
) -> None:
    """
    Builds a GeneratedTool from `spec`, registers it, appends/updates it on
    memory's generated-tools list, and persists it to
    `<output_root>/generated_tools/<name>.json` (prior versions kept under a
    `history` list key, never overwritten).

    Pure side-effect function — the controller's job per design decision 3
    (AGENTS.md layer rule: tools/* must never drive memory/engine/controller
    behaviour). Not called by GeneratedTool.execute() or
    DefineAnalysisToolTool.execute() themselves.
    """
    from src.tools.generated_tool import GeneratedTool

    tool = GeneratedTool(spec)
    tool_registry.register(tool)

    # MemorySystem.add_generated_tool persists the spec the same way
    # memory.add_findings persists findings (see memory.py).
    memory.add_generated_tool(spec.to_dict())

    _persist_spec_to_disk(spec, output_root)


def _persist_spec_to_disk(spec: GeneratedToolSpec, output_root: str) -> None:
    generated_dir = Path(output_root) / "generated_tools"
    generated_dir.mkdir(parents=True, exist_ok=True)
    spec_path = generated_dir / f"{spec.name}.json"

    history: list[dict[str, Any]] = []
    if spec_path.exists():
        try:
            existing_payload = json.loads(spec_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_payload = None
        if existing_payload is not None:
            history = existing_payload.get("history", [])
            # The previously-current spec (minus its own nested history)
            # becomes the newest history entry, oldest-last is not assumed —
            # callers reading history should treat it as an ordered log,
            # append-only, newest appended last.
            prior_current = {k: v for k, v in existing_payload.items() if k != "history"}
            if prior_current.get("name"):
                history.append(prior_current)

    payload = spec.to_dict()
    payload["history"] = history
    spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_persisted_tools(output_root: str, dataset_fingerprint: str) -> list[GeneratedToolSpec]:
    """
    Opt-in reload for a matching dataset fingerprint only — never auto-runs
    across datasets (design decision 8). Reads every
    `<output_root>/generated_tools/*.json`, returning only the specs whose
    persisted `dataset_fingerprint` matches the one passed in.
    """
    generated_dir = Path(output_root) / "generated_tools"
    if not generated_dir.is_dir():
        return []

    specs: list[GeneratedToolSpec] = []
    for spec_path in sorted(generated_dir.glob("*.json")):
        try:
            payload = json.loads(spec_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("dataset_fingerprint") != dataset_fingerprint:
            continue
        try:
            specs.append(GeneratedToolSpec.from_dict(payload))
        except KeyError:
            continue
    return specs


def next_version(existing_generated: dict[str, GeneratedToolSpec], name: str) -> int:
    """Convenience for callers building a GeneratedToolSpec for a
    modification: 1 for a brand-new name, existing version + 1 for an
    overwrite."""
    prior = existing_generated.get(name)
    return 1 if prior is None else prior.version + 1


def make_spec(
    name: str,
    description: str,
    params_schema: dict[str, Any],
    code: str,
    dataset_fingerprint: str,
    existing_generated: dict[str, GeneratedToolSpec] | None = None,
) -> GeneratedToolSpec:
    """Convenience constructor: stamps created_at/version consistently so
    callers (controller integration) don't hand-roll timestamps."""
    existing_generated = existing_generated or {}
    return GeneratedToolSpec(
        name=name,
        description=description,
        params_schema=params_schema,
        code=code,
        version=next_version(existing_generated, name),
        created_at=datetime.now(UTC).isoformat(),
        dataset_fingerprint=dataset_fingerprint,
    )
