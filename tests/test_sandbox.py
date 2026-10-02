"""Tests for src/core/sandbox.py."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.core import sandbox
from tests.fixtures import single_column


class TestStaticCheck:
    def test_valid_code_passes(self) -> None:
        code = "RESULT = 1\n"
        assert sandbox._static_check(code) is None

    def test_syntax_error_caught(self) -> None:
        code = "RESULT = (\n"
        result = sandbox._static_check(code)
        assert result is not None
        error_type, hint = result
        assert error_type == "syntax"
        assert "syntax error" in hint.lower()

    def test_missing_result_assignment_caught(self) -> None:
        code = "x = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        error_type, hint = result
        assert error_type == "static_check"
        assert "RESULT" in hint

    def test_result_assignment_nested_in_if_is_recognized(self) -> None:
        # Round 8 (IMPROVEMENTS.md decision 6): a reusable, parameterized
        # generated tool plausibly branches (`if grain == "month": RESULT =
        # ... else: ...`), so the static check now walks the whole tree
        # (ast.walk) instead of only tree.body — a RESULT assignment
        # anywhere in the source passes the static check. This is a static
        # pre-check only; an assignment that's syntactically present but
        # never actually *executed* (e.g. inside `if False:`) is still
        # caught at runtime — see test_sandbox_worker.py's
        # test_result_never_assigned_at_runtime, which is what that case
        # exercises now instead of this test.
        code = "if True:\n    RESULT = 1\n"
        assert sandbox._static_check(code) is None

    def test_result_assignment_via_annassign_recognized(self) -> None:
        # Round 8: an annotated assignment (`RESULT: dict = {...}`) is also
        # accepted, not just a plain ast.Assign.
        code = "RESULT: int = 1\n"
        assert sandbox._static_check(code) is None

    def test_missing_result_anywhere_still_caught(self) -> None:
        # The relaxation (ast.walk instead of tree.body) must not become
        # "accept anything" — code with no RESULT assignment at all, in any
        # branch, is still rejected.
        code = "if True:\n    x = 1\nelse:\n    x = 2\n"
        result = sandbox._static_check(code)
        assert result is not None
        assert result[0] == "static_check"
        assert "RESULT" in result[1]

    def test_disallowed_import_caught(self) -> None:
        code = "import os\nRESULT = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        error_type, hint = result
        assert error_type == "static_check"
        assert "os" in hint

    def test_disallowed_import_from_caught(self) -> None:
        code = "from socket import socket\nRESULT = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        assert result[0] == "static_check"

    def test_allowed_import_passes(self) -> None:
        code = "import numpy as np\nRESULT = 1\n"
        assert sandbox._static_check(code) is None

    def test_blocked_builtin_name_caught(self) -> None:
        code = "f = open('x.txt')\nRESULT = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        error_type, hint = result
        assert error_type == "static_check"
        assert "open" in hint


class TestValidateExtraGlobals:
    """Round 8 — parameters reach generated-tool code as data, never as
    templated source (IMPROVEMENTS.md decision 2)."""

    def test_none_is_valid(self) -> None:
        assert sandbox._validate_extra_globals(None) is None

    def test_empty_dict_is_valid(self) -> None:
        assert sandbox._validate_extra_globals({}) is None

    def test_json_primitive_values_are_valid(self) -> None:
        assert sandbox._validate_extra_globals(
            {"a": 1, "b": "x", "c": 1.5, "d": True, "e": None, "f": [1, 2], "g": {"k": "v"}}
        ) is None

    def test_non_json_value_rejected(self) -> None:
        result = sandbox._validate_extra_globals({"bad": open})
        assert result is not None
        assert result[0] == "static_check"

    def test_reserved_name_collision_rejected(self) -> None:
        for reserved in ("df", "SCHEMA", "RESULT", "FINDING", "__builtins__"):
            result = sandbox._validate_extra_globals({reserved: 1})
            assert result is not None, f"{reserved} should be rejected"
            assert result[0] == "static_check"


class TestRunSandboxed:
    def test_success_end_to_end(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        result = sandbox.run_sandboxed(
            "RESULT = float(df['value'].mean())\n", str(dataset)
        )
        assert result.status == "ok"
        assert isinstance(result.result, float)
        assert result.error_type is None
        assert result.duration_ms > 0

    def test_static_check_failure_short_circuits_before_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = single_column(tmp_path)

        def _fail_if_called(*args: object, **kwargs: object) -> None:
            raise AssertionError("subprocess.Popen should not be called")

        monkeypatch.setattr(sandbox.subprocess, "Popen", _fail_if_called)
        result = sandbox.run_sandboxed("x = 1\n", str(dataset))
        assert result.status == "error"
        assert result.error_type == "static_check"

    def test_timeout_kills_process(self, tmp_path: Path) -> None:
        # timeout_s comfortably exceeds worker startup (which imports pandas),
        # so the kill provably interrupts the busy loop rather than the
        # interpreter's own import phase.
        dataset = single_column(tmp_path)
        code = "RESULT = 0\nwhile True:\n    pass\n"
        result = sandbox.run_sandboxed(code, str(dataset), timeout_s=8.0)
        assert result.status == "error"
        assert result.error_type == "timeout"

    def test_memory_limit_kills_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeMemInfo:
            rss = 999 * 1024 * 1024

        class _FakePsProcess:
            def __init__(self, pid: int) -> None:
                pass

            def memory_info(self) -> _FakeMemInfo:
                return _FakeMemInfo()

            def children(self, recursive: bool = True) -> list[Any]:
                return []

        monkeypatch.setattr(sandbox.psutil, "Process", _FakePsProcess)
        dataset = single_column(tmp_path)
        # Long-running so the parent's poll fires before natural completion.
        code = "RESULT = 0\ntotal = 0\nfor i in range(10**9):\n    total += i\nRESULT = total\n"
        result = sandbox.run_sandboxed(code, str(dataset), timeout_s=30.0, memory_limit_mb=50)
        assert result.status == "error"
        assert result.error_type == "memory"

    def test_worker_crash_without_result_file_reports_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        crashing_worker = tmp_path / "crashing_worker.py"
        crashing_worker.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
        monkeypatch.setattr(sandbox, "_WORKER_SCRIPT", crashing_worker)

        dataset = single_column(tmp_path)
        result = sandbox.run_sandboxed("RESULT = 1\n", str(dataset))
        assert result.status == "error"
        assert result.error_type == "runtime"

    def test_extra_globals_reach_the_code_as_values(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        result = sandbox.run_sandboxed(
            "RESULT = extra_val * 2\n", str(dataset), extra_globals={"extra_val": 21}
        )
        assert result.status == "ok"
        assert result.result == 42

    def test_extra_globals_never_templated_into_source(self, tmp_path: Path) -> None:
        # A value containing Python-looking text must stay inert data, never
        # get executed — this is the core guarantee of decision 2.
        dataset = single_column(tmp_path)
        result = sandbox.run_sandboxed(
            "RESULT = payload\n",
            str(dataset),
            extra_globals={"payload": "RESULT = 999\nimport os"},
        )
        assert result.status == "ok"
        assert result.result == "RESULT = 999\nimport os"

    def test_invalid_extra_globals_short_circuits_before_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = single_column(tmp_path)

        def _fail_if_called(*args: object, **kwargs: object) -> None:
            raise AssertionError("subprocess.Popen should not be called")

        monkeypatch.setattr(sandbox.subprocess, "Popen", _fail_if_called)
        result = sandbox.run_sandboxed(
            "RESULT = 1\n", str(dataset), extra_globals={"df": [1, 2, 3]}
        )
        assert result.status == "error"
        assert result.error_type == "static_check"

    def test_finding_round_trips_when_present(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        result = sandbox.run_sandboxed(
            "RESULT = 1\nFINDING = {'headline': 'test'}\n", str(dataset)
        )
        assert result.status == "ok"
        assert result.finding == {"headline": "test"}

    def test_finding_is_none_when_absent(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        result = sandbox.run_sandboxed("RESULT = 1\n", str(dataset))
        assert result.status == "ok"
        assert result.finding is None

    def test_result_assigned_only_inside_branch_now_executes(self, tmp_path: Path) -> None:
        # Round 8 relaxation, end to end: a genuinely reachable branch
        # assignment is no longer rejected by the static pre-check.
        dataset = single_column(tmp_path)
        code = "x = 1\nif x:\n    RESULT = 'a'\nelse:\n    RESULT = 'b'\n"
        result = sandbox.run_sandboxed(code, str(dataset))
        assert result.status == "ok"
        assert result.result == "a"

    def test_prior_results_accessible_in_sandbox(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        code = "RESULT = PRIOR_RESULTS.get('stat_test', {}).get('p_value', 0.0)\n"
        result = sandbox.run_sandboxed(
            code,
            str(dataset),
            prior_results={"stat_test": {"p_value": 0.042}},
        )
        assert result.status == "ok"
        assert result.result == 0.042

    def test_prior_results_defaults_to_empty_dict(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        code = "RESULT = isinstance(PRIOR_RESULTS, dict) and len(PRIOR_RESULTS) == 0\n"
        result = sandbox.run_sandboxed(code, str(dataset))
        assert result.status == "ok"
        assert result.result is True

    def test_docker_sandbox_availability_check(self) -> None:
        avail = sandbox.DockerSandbox.is_available()
        assert isinstance(avail, bool)

    def test_get_sandbox_backend_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend_subp = sandbox.get_sandbox_backend("subprocess")
        assert isinstance(backend_subp, sandbox.SubprocessSandbox)

        monkeypatch.setenv("SANDBOX_BACKEND", "subprocess")
        assert isinstance(sandbox.get_sandbox_backend(), sandbox.SubprocessSandbox)

        monkeypatch.setattr(sandbox.DockerSandbox, "is_available", lambda: False)
        # An explicit docker request with the daemon down is refused, never
        # silently downgraded to the unisolated subprocess backend.
        assert sandbox.get_sandbox_backend("docker") is None


class TestWorkerCrashDiagnostics:
    def test_stderr_tail_surfaces_when_worker_dies_before_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        crasher = tmp_path / "crasher.py"
        crasher.write_text(
            "import sys\nsys.stderr.write('ImportError: no module named boom')\nsys.exit(1)\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(sandbox, "_WORKER_SCRIPT", crasher)
        result = sandbox.run_sandboxed("RESULT = 1\n", str(single_column(tmp_path)))
        assert result.status == "error"
        assert result.traceback is not None
        assert "no module named boom" in result.traceback

    def test_stderr_flood_is_stopped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flooder = tmp_path / "flooder.py"
        flooder.write_text(
            "import sys\nwhile True:\n    sys.stderr.write('x' * 65536)\n    sys.stderr.flush()\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(sandbox, "_WORKER_SCRIPT", flooder)
        monkeypatch.setattr(sandbox, "_STDERR_MAX_BYTES", 1024 * 1024)
        result = sandbox.run_sandboxed("RESULT = 1\n", str(single_column(tmp_path)), timeout_s=20)
        assert result.status == "error"
        assert "stderr" in (result.hint or "")
