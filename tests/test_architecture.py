"""Static enforcement of the repository's architectural layer boundaries.

These tests inspect source via :mod:`ast`; importing the application to test
its architecture would itself execute optional provider and UI dependencies.
The allowlist for core-to-tool imports is deliberately small and named: the
registry owns tool construction, the sandbox toolkit owns its controlled
dynamic dispatch, and the generated-tool factory owns generated-tool loading.
Every other core module must use a neutral contract rather than a concrete
tool implementation.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CORE_TOOL_IMPORT_ALLOWLIST = frozenset({
    "sandbox_toolkit.py",
    "tool_factory.py",
    "tool_registry.py",
})
CONTROLLER_MODULES = tuple(sorted((SRC / "core").glob("controller*.py")))


def _tree(path: Path) -> ast.Module:
    """Parse *path* without importing application modules."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(path: Path) -> set[str]:
    """Return raw dotted module strings imported by *path*."""
    modules: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _forbidden(modules: set[str], *prefixes: str) -> set[str]:
    """Select module names equal to, or below, one of *prefixes*."""
    return {
        module
        for module in modules
        if any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes)
    }


def _attribute_name(node: ast.AST) -> str | None:
    """Resolve a simple dotted attribute expression without evaluating it."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _dynamic_tool_imports(path: Path) -> list[int]:
    """Return lines dynamically importing a ``src.tools`` module.

    Dynamic imports are execution-layer dependencies just as much as normal
    imports are.  Checking the AST catches the common f-string form without
    having to execute the interpolation.
    """
    lines: list[int] = []
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call) or _attribute_name(node.func) != "importlib.import_module":
            continue
        if not node.args:
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            is_tool_import = argument.value.startswith("src.tools")
        elif isinstance(argument, ast.JoinedStr):
            is_tool_import = any(
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and "src.tools" in value.value
                for value in argument.values
            )
        else:
            is_tool_import = False
        if is_tool_import:
            lines.append(node.lineno)
    return lines


def _private_tool_imports(path: Path) -> list[str]:
    """Return private symbols imported directly from a tool module."""
    symbols: list[str] = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src.tools"):
            symbols.extend(alias.name for alias in node.names if alias.name.startswith("_"))
    return symbols


def test_tools_never_import_reasoning_or_orchestration_layers() -> None:
    """Tools are driven by the controller; they never drive it themselves."""
    offenders: dict[str, set[str]] = {}
    for path in sorted((SRC / "tools").glob("*.py")):
        bad = _forbidden(_imported_modules(path), "src.rlm.engine", "src.core.controller")
        if bad:
            offenders[str(path.relative_to(ROOT))] = bad
    assert not offenders, f"tools importing engine/controller directly: {offenders}"


def test_engine_never_imports_pipeline_state_or_llm_policy() -> None:
    """The engine receives the callable and concurrency policy by injection."""
    path = SRC / "rlm" / "engine.py"
    bad = _forbidden(
        _imported_modules(path),
        "src.core.memory",
        "src.tools",
        "src.core.controller",
        "src.core.llm_client",
    )
    assert not bad, f"rlm/engine.py importing forbidden dependencies: {bad}"


def test_memory_never_imports_execution_or_reasoning_layers() -> None:
    """State objects remain independent of their consumers."""
    path = SRC / "core" / "memory.py"
    bad = _forbidden(_imported_modules(path), "src.tools", "src.rlm.engine", "src.core.controller")
    assert not bad, f"memory.py importing forbidden dependencies: {bad}"


def test_controller_modules_do_not_call_llm_client_directly() -> None:
    """Controllers invoke the LLM through ``RLMEngine.invoke()``, never ``.call()``."""
    offenders: dict[str, list[int]] = {}
    for path in CONTROLLER_MODULES:
        lines = [
            node.lineno
            for node in ast.walk(_tree(path))
            if isinstance(node, ast.Call) and _attribute_name(node.func) == "self.llm_client.call"
        ]
        if lines:
            offenders[str(path.relative_to(ROOT))] = lines
    assert not offenders, f"controller modules call llm_client.call directly: {offenders}"


def test_core_tool_dependencies_are_limited_to_documented_boundary_adapters() -> None:
    """Core code may not instantiate or call concrete tools outside adapters."""
    offenders: dict[str, dict[str, object]] = {}
    for path in sorted((SRC / "core").glob("*.py")):
        if path.name in CORE_TOOL_IMPORT_ALLOWLIST:
            continue
        imports = _forbidden(_imported_modules(path), "src.tools")
        dynamic_lines = _dynamic_tool_imports(path)
        if imports or dynamic_lines:
            offenders[str(path.relative_to(ROOT))] = {
                "imports": sorted(imports),
                "dynamic_import_lines": dynamic_lines,
            }
    assert not offenders, f"core modules importing concrete tool behavior: {offenders}"


def test_core_never_imports_private_tool_helpers() -> None:
    """Private tool helpers are not cross-layer contracts; move them to core."""
    offenders: dict[str, list[str]] = {}
    for path in sorted((SRC / "core").glob("*.py")):
        private_symbols = _private_tool_imports(path)
        if private_symbols:
            offenders[str(path.relative_to(ROOT))] = private_symbols
    assert not offenders, f"core modules importing private tool helpers: {offenders}"
