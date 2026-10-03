"""Hosted-server defaults for code execution (governance.py, sandbox.py)."""
from __future__ import annotations

import pytest


def test_a_hosted_server_defaults_to_no_code_and_required_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.governance import code_execution_enabled
    from src.core.sandbox import isolation_required

    for var in ("ENABLE_CODE_EXECUTION", "SANDBOX_REQUIRE_ISOLATION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("DSA_HOSTED", raising=False)
    assert code_execution_enabled() and not isolation_required()

    monkeypatch.setenv("DSA_HOSTED", "true")
    assert not code_execution_enabled() and isolation_required()

    # The operator can still opt in explicitly.
    monkeypatch.setenv("ENABLE_CODE_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_REQUIRE_ISOLATION", "false")
    assert code_execution_enabled() and not isolation_required()


def _clear_sandbox_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ENABLE_CODE_EXECUTION", "SANDBOX_REQUIRE_ISOLATION", "SANDBOX_BACKEND", "DSA_HOSTED"):
        monkeypatch.delenv(var, raising=False)


def test_code_runs_in_the_subprocess_sandbox_unless_docker_is_demanded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.governance import code_execution_available, code_execution_blocker
    from src.core.sandbox import DockerSandbox

    _clear_sandbox_env(monkeypatch)
    monkeypatch.setattr(DockerSandbox, "is_available", staticmethod(lambda: False))
    # Nothing demands Docker, so its absence must not block anything.
    assert code_execution_blocker() is None and code_execution_available()


def test_a_demanded_but_missing_docker_blocks_code_and_names_the_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.governance import CodeGovernor, code_execution_available, code_execution_blocker
    from src.core.sandbox import DockerSandbox, isolation_demand, run_sandboxed

    _clear_sandbox_env(monkeypatch)
    monkeypatch.setattr(DockerSandbox, "is_available", staticmethod(lambda: False))

    monkeypatch.setenv("SANDBOX_REQUIRE_ISOLATION", "true")
    assert "Require container isolation" in (isolation_demand() or "")
    blocker = code_execution_blocker() or ""
    assert "Docker" in blocker and "SANDBOX_REQUIRE_ISOLATION" in blocker
    assert not code_execution_available()
    assert "unavailable" in (CodeGovernor("out", "s").refusal_reason() or "")
    refused = run_sandboxed("RESULT = 1\n", "unused.csv")
    assert refused.error_type == "isolation_unavailable" and "SANDBOX_REQUIRE_ISOLATION" in (refused.hint or "")

    monkeypatch.delenv("SANDBOX_REQUIRE_ISOLATION")
    monkeypatch.setenv("SANDBOX_BACKEND", "docker")
    assert isolation_demand() == "SANDBOX_BACKEND=docker"

    monkeypatch.delenv("SANDBOX_BACKEND")
    monkeypatch.setenv("DSA_HOSTED", "true")
    monkeypatch.setenv("ENABLE_CODE_EXECUTION", "true")
    assert isolation_demand() == "DSA_HOSTED=true"


def test_a_demanded_docker_that_is_running_leaves_code_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.governance import code_execution_available
    from src.core.sandbox import DockerSandbox

    _clear_sandbox_env(monkeypatch)
    monkeypatch.setenv("SANDBOX_REQUIRE_ISOLATION", "true")
    monkeypatch.setattr(DockerSandbox, "is_available", staticmethod(lambda: True))
    assert code_execution_available()
