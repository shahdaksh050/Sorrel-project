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
