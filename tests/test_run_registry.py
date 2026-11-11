"""Tests for the safe, atomic UI run registry."""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.core.run_registry import RunRegistry, find_run


def test_registry_writes_atomic_safe_status_and_terminal_artifacts(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "dsa-run-one")
    created = registry.create("a" * 32)

    assert created.status == "running"
    assert registry.path.is_file()
    registry.heartbeat("3:active")
    identified = registry.set_process_identity(os.getpid())
    completed = registry.finish("succeeded", ("reports/summary.json", "../secret.txt"))

    raw = json.loads(registry.path.read_text(encoding="utf-8"))
    assert completed.status == "succeeded"
    assert identified.process_id == os.getpid()
    assert identified.process_create_time is not None
    assert raw["stage"] == "3:active"
    assert raw["artifacts"] == ["reports/summary.json"]
    assert not list(registry.run_dir.glob(".run-status-*.json"))


def test_registry_excludes_secrets_and_raw_rows_by_construction(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "dsa-run-two")
    registry.create("b" * 32)
    registry.heartbeat("1:active")

    content = registry.path.read_text(encoding="utf-8")
    assert "api_key" not in content
    assert "objective" not in content
    assert "dataset" not in content
    assert "raw" not in content


def test_find_run_rejects_invalid_ids_and_finds_only_matching_record(tmp_path: Path) -> None:
    run_dir = tmp_path / "dsa-run-live"
    registry = RunRegistry(run_dir)
    registry.create("c" * 32)

    assert find_run("../bad", tmp_path) is None
    found = find_run("c" * 32, tmp_path)
    assert found is not None and found.path == registry.path


def test_abandoned_requires_expired_heartbeat_and_missing_exact_process(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "dsa-run-dead")
    registry.create("d" * 32)
    current = registry.load()
    assert current is not None

    # The real process identity is still present: an old timestamp alone must
    # not declare an active worker dead.
    assert registry.is_abandoned(0.0, now=10**12) is False
    data = json.loads(registry.path.read_text(encoding="utf-8"))
    data["process_id"] = 999_999_999
    data["process_create_time"] = 0.0
    data["heartbeat_at"] = "2000-01-01T00:00:00+00:00"
    registry.path.write_text(json.dumps(data), encoding="utf-8")
    assert registry.is_abandoned(1.0) is True


def test_registry_rejects_malformed_or_extra_fields(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "dsa-run-bad")
    registry.run_dir.mkdir()
    registry.path.write_text(json.dumps({"run_id": "e" * 32, "secret": "nope"}), encoding="utf-8")

    assert registry.load() is None
