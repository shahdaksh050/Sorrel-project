"""
Architecture drift check (IMPROVEMENTS.md P2.2).

AGENTS.md's Layer Rules table describes which modules each layer may import.
That table went stale once (`tools/*` documented as forbidden from
importing `memory` while `src/tools/base.py` already did, for a genuine
reason — see AGENTS.md's P2.2 note) and nobody noticed because nothing
checked it. This file is that check: it walks each layer's imports via the
AST (no execution, no side effects) and asserts the *forbidden* edges never
appear. It intentionally does not try to enforce the *allowed* list
precisely — the goal is catching an accidental new dependency on `engine`
or `controller` from the wrong layer, not policing every import.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _imported_modules(path: Path) -> set[str]:
    """Every dotted module name this file imports (from `import x.y` and
    `from x.y import z`), as the raw dotted string — good enough to check
    for a forbidden prefix without needing real import resolution."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _forbidden(modules: set[str], *forbidden_prefixes: str) -> set[str]:
    return {
        m for m in modules
        if any(m == p or m.startswith(p + ".") for p in forbidden_prefixes)
    }


def test_tools_never_import_engine_or_controller() -> None:
    """`tools/*` may depend on `memory`'s types (AGENTS.md's P2.2-amended
    rule), but must never reach into the reasoning/orchestration layers
    directly — that would let a tool drive the agent loop instead of being
    driven by it."""
    offenders: dict[str, set[str]] = {}
    for path in sorted((SRC / "tools").glob("*.py")):
        bad = _forbidden(_imported_modules(path), "src.rlm.engine", "src.core.controller")
        if bad:
            offenders[str(path.relative_to(ROOT))] = bad
    assert not offenders, f"tools/* importing engine/controller directly: {offenders}"


def test_engine_never_imports_memory_tools_or_controller() -> None:
    """`src/rlm/engine.py` takes an injected `llm_callable` and stays
    ignorant of pipeline state — it must not reach into `memory`, `tools`,
    or `controller` to get it."""
    path = SRC / "rlm" / "engine.py"
    bad = _forbidden(_imported_modules(path), "src.core.memory", "src.tools", "src.core.controller")
    assert not bad, f"rlm/engine.py importing forbidden modules: {bad}"


def test_memory_never_imports_tools_engine_or_controller() -> None:
    """`src/core/memory.py` is the state layer — it may define types other
    layers import, but must never import a tool, the engine, or the
    controller itself (that would be a dependency cycle by construction)."""
    path = SRC / "core" / "memory.py"
    bad = _forbidden(_imported_modules(path), "src.tools", "src.rlm.engine", "src.core.controller")
    assert not bad, f"memory.py importing forbidden modules: {bad}"
