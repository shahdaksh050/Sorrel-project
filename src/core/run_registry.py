"""Durable, deliberately small status records for UI analysis runs.

The registry is an operational record, not a run cache.  In particular it
never accepts objectives, provider settings, prompts, data values, or errors:
those may contain credentials or user supplied rows.  Records are written with
``os.replace`` so a reconnecting browser sees either the old complete JSON
document or the new complete JSON document, never a partially written file.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import psutil

RunTerminalStatus = Literal["succeeded", "partial", "cancelled", "failed"]
RunLifecycleStatus = Literal["running", "succeeded", "partial", "cancelled", "failed"]

_RUN_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_STATUS_FILE = "run_status.json"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _process_create_time(pid: int) -> float | None:
    try:
        return float(psutil.Process(pid).create_time())
    except psutil.Error:
        return None


@dataclass(frozen=True)
class RunStatus:
    """The safe subset of a run that may survive browser reconnects."""

    run_id: str
    status: RunLifecycleStatus
    stage: str | None
    created_at: str
    updated_at: str
    heartbeat_at: str
    process_id: int | None
    process_create_time: float | None
    artifacts: tuple[str, ...] = field(default_factory=tuple)


class RunRegistry:
    """Own the atomic ``run_status.json`` record located in one run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir / _STATUS_FILE

    @staticmethod
    def new_run_id() -> str:
        """Return a URL-safe opaque identifier with no user-derived content."""
        return uuid.uuid4().hex

    @staticmethod
    def valid_run_id(run_id: str | None) -> bool:
        return bool(run_id and _RUN_ID_RE.fullmatch(run_id))

    def create(self, run_id: str | None = None) -> RunStatus:
        """Create the initial running record before any dataset is read."""
        value = run_id or self.new_run_id()
        if not self.valid_run_id(value):
            raise ValueError("run_id must be a 32-character lowercase hex UUID")
        now = _timestamp()
        pid = os.getpid()
        status = RunStatus(
            run_id=value,
            status="running",
            stage=None,
            created_at=now,
            updated_at=now,
            heartbeat_at=now,
            process_id=pid,
            process_create_time=_process_create_time(pid),
        )
        self._write(status)
        return status

    def load(self) -> RunStatus | None:
        """Return a validated status record, or ``None`` for absent/corrupt records."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or set(raw) != set(RunStatus.__dataclass_fields__):
                return None
            record = RunStatus(**raw)
            if not self.valid_run_id(record.run_id) or record.status not in {
                "running", "succeeded", "partial", "cancelled", "failed"
            }:
                return None
            if not isinstance(record.artifacts, tuple):
                record = RunStatus(**{**asdict(record), "artifacts": tuple(record.artifacts)})
            if any(not _safe_artifact_path(item) for item in record.artifacts):
                return None
            return record
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def heartbeat(self, stage: str | None = None) -> RunStatus:
        """Update liveness and optionally the safe, human-readable stage label."""
        current = self._require_current()
        return self._replace(current, stage=stage if stage is not None else current.stage)

    def set_process_identity(self, process_id: int) -> RunStatus:
        """Bind the record to the exact spawned worker process identity."""
        create_time = _process_create_time(process_id)
        if create_time is None:
            raise RuntimeError("analysis worker process identity is unavailable")
        return self._replace(
            self._require_current(),
            process_id=process_id,
            process_create_time=create_time,
        )

    def finish(self, status: RunTerminalStatus, artifacts: list[str] | tuple[str, ...] = ()) -> RunStatus:
        """Record a terminal status and relative allowlisted artifact paths."""
        cleaned = tuple(item for item in artifacts if _safe_artifact_path(item))
        return self._replace(self._require_current(), status=status, artifacts=cleaned)

    def is_abandoned(self, max_heartbeat_age_s: float, now: float | None = None) -> bool:
        """True only for an expired heartbeat whose exact process identity is gone.

        Matching both PID and process creation time prevents a recycled PID from
        being mistaken for the original worker.
        """
        record = self.load()
        if record is None or record.status != "running":
            return False
        try:
            heartbeat = datetime.fromisoformat(record.heartbeat_at).timestamp()
        except ValueError:
            return True
        if (time.time() if now is None else now) - heartbeat <= max_heartbeat_age_s:
            return False
        if record.process_id is None or record.process_create_time is None:
            return True
        actual = _process_create_time(record.process_id)
        return actual is None or abs(actual - record.process_create_time) > 0.01

    def _require_current(self) -> RunStatus:
        current = self.load()
        if current is None:
            raise RuntimeError("run status has not been created")
        return current

    def _replace(self, current: RunStatus, **changes: object) -> RunStatus:
        now = _timestamp()
        values = asdict(current)
        values.update(changes)
        values["updated_at"] = now
        values["heartbeat_at"] = now
        updated = RunStatus(**values)
        self._write(updated)
        return updated

    def _write(self, status: RunStatus) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".run-status-", suffix=".json", dir=self.run_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(asdict(status), handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def find_run(run_id: str, root: str | Path | None = None) -> RunRegistry | None:
    """Find a validated local run by ID without trusting a user-supplied path."""
    if not RunRegistry.valid_run_id(run_id):
        return None
    search_root = Path(tempfile.gettempdir() if root is None else root).resolve()
    try:
        direct = search_root / _STATUS_FILE
        if direct.is_file():
            registry = RunRegistry(search_root)
            record = registry.load()
            if record is not None and record.run_id == run_id:
                return registry
        candidates = list(search_root.glob(f"dsa-run-*/{_STATUS_FILE}"))
        candidates.extend(search_root.glob(f"*/{_STATUS_FILE}"))
        for path in candidates:
            registry = RunRegistry(path.parent)
            record = registry.load()
            if record is not None and record.run_id == run_id:
                return registry
    except OSError:
        return None
    return None


def _safe_artifact_path(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and not path.parts[0].startswith(".")
