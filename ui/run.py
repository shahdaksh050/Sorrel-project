"""
Background analysis run: the parent monitor and the state the page polls.

The Streamlit script starts one `ActiveRun` per analysis and keeps it in session
state; a polling fragment reads `ActiveRun.snapshot()` once a second. Everything
here runs on, or is read from, a thread that is not the script thread, so:

* this module never imports Streamlit and never touches `st.session_state`;
* every setting the worker needs arrives in a frozen `RunSpec`;
* the worker is normally a spawned process, so a native-library crash or an
  interrupted Streamlit session cannot take the analysis down with it;
* the parent monitor owns `ActiveRun` state and the script thread folds the
  finished `RunOutcome` into session state itself.

Stop starts cooperatively: `request_stop` sets an event that the controller
polls at its checkpoints (`AgentController.should_stop`).  If the worker does
not honour it within a bounded grace period, only that known child process is
terminated.  ``RUNNER_BACKEND=thread`` remains available for constrained
debugging, but production defaults to ``process``.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import pickle
import queue
import shutil
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import psutil

from src.core.findings import Finding, is_trivial
from src.core.run_config import RunConfig
from src.core.run_registry import RunRegistry, RunStatus, RunTerminalStatus, find_run
from src.core.run_view import (
    RUN_VIEW_CONTEXT_KEYS,
    ProvisionalFinding,
    RunView,
    build_run_view,
    is_headline_finding,
)

#: Prefix of every per-run temp directory; `remove_run_dir` only ever deletes
#: directories that carry it, directly under the system temp root.
RUN_DIR_PREFIX = "dsa-run-"


def _retained_run_dir() -> Path:
    """Reserve a unique local directory for the durable copy of one UI run."""
    root = Path("output") / "runs"
    root.mkdir(parents=True, exist_ok=True)
    # Microseconds avoid ordinary collisions; the suffix closes the small
    # remaining race between concurrent browser sessions.
    stamp = datetime.now().strftime("ui-%Y%m%d-%H%M%S-%f")
    for suffix in range(1000):
        candidate = root / (stamp if suffix == 0 else f"{stamp}-{suffix}")
        if not candidate.exists():
            return candidate
    raise OSError("could not allocate a directory for retained run artifacts")


def retain_run_artifacts(output_dir: str) -> str | None:
    """Copy a completed run's safe, user-facing artifacts into ``output/runs``.

    Retention is intentionally an allowlist, not a snapshot of the worker
    tree.  Dotfiles, symlinks, signing keys, scratch files, and unknown future
    artifacts never cross this boundary.
    """
    source = Path(output_dir).resolve()
    if not source.is_dir() or source.is_symlink():
        return None
    try:
        destination = _retained_run_dir().resolve()
        # A configuration pointing the run output at output/runs could make a
        # recursive copy.  Refuse rather than copying an unbounded tree.
        if destination.is_relative_to(source):
            return None
        for relative in _retained_artifacts(source):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
        summary = destination / "reports" / "summary.json"
        return str(summary) if summary.is_file() else None
    except OSError:
        return None


_RETAINED_SUFFIXES: dict[str, frozenset[str]] = {
    "reports": frozenset({".html", ".json", ".md"}),
    "audit": frozenset({".jsonl"}),
    "models": frozenset({".pkl", ".sig"}),
    "visualizations": frozenset({".html", ".pdf", ".png", ".svg"}),
    "data": frozenset({".csv", ".json", ".parquet", ".tsv"}),
}


def _retained_artifacts(source: Path) -> list[Path]:
    """Return safe artifact paths relative to ``source`` in stable order."""
    retained: list[Path] = []
    for category, suffixes in _RETAINED_SUFFIXES.items():
        directory = source / category
        if not directory.is_dir() or directory.is_symlink():
            continue
        for candidate in directory.rglob("*"):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            relative = candidate.relative_to(source)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if candidate.suffix.lower() not in suffixes:
                continue
            if "key" in candidate.name.lower():
                continue
            retained.append(relative)
    return sorted(retained)

RunState = Literal["running", "done", "error"]
StageEntry = tuple[str, str, str]  # (stage number, status, detail)

_STATUS_ICON = {"done": "[done]", "active": "[run ]", "error": "[fail]", "skipped": "[skip]"}


def remove_run_dir(path: str | None) -> None:
    """Delete a run's temp directory (uploads, reports, models)."""
    if not path:
        return
    target = Path(path).resolve()
    if target.name.startswith(RUN_DIR_PREFIX) and target.parent == Path(tempfile.gettempdir()).resolve():
        shutil.rmtree(target, ignore_errors=True)


#: Run directories older than this are leftovers from a closed tab or a crashed process.
try:
    STALE_RUN_DIR_AGE_S = float(os.getenv("STALE_RUN_DIR_AGE_S", str(2 * 3600)))
except ValueError:
    STALE_RUN_DIR_AGE_S = 2.0 * 3600

try:
    MAX_TOTAL_RUN_BYTES = int(os.getenv("MAX_TOTAL_RUN_BYTES", str(2 * 1024 * 1024 * 1024)))
except ValueError:
    MAX_TOTAL_RUN_BYTES = 2 * 1024 * 1024 * 1024


def sweep_stale_run_dirs(
    max_age_s: float = STALE_RUN_DIR_AGE_S,
    now: float | None = None,
    max_total_bytes: int = MAX_TOTAL_RUN_BYTES,
) -> int:
    """Delete run directories nobody can still be using; returns how many were removed.

    A run directory is removed when it carries `RUN_DIR_PREFIX`, sits directly under
    the system temp root, and:
    1. Has not been modified for `max_age_s`, OR
    2. Aggregate storage exceeds `max_total_bytes` (oldest removed first).
    """
    root = Path(tempfile.gettempdir()).resolve()
    current_time = time.time() if now is None else now
    cutoff = current_time - max_age_s
    removed = 0
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0

    valid_dirs: list[tuple[Path, float]] = []
    for entry in entries:
        try:
            if (
                entry.name.startswith(RUN_DIR_PREFIX)
                and entry.is_dir()
                and not entry.is_symlink()
            ):
                mtime = entry.stat().st_mtime
                if mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
                else:
                    valid_dirs.append((entry, mtime))
        except OSError:
            continue

    if max_total_bytes > 0 and valid_dirs:
        valid_dirs.sort(key=lambda x: x[1])
        total_bytes = 0
        dir_sizes: list[tuple[Path, int]] = []
        for d, _ in valid_dirs:
            sz = 0
            try:
                for p in d.rglob("*"):
                    if p.is_file() and not p.is_symlink():
                        sz += p.stat().st_size
            except OSError:
                pass
            dir_sizes.append((d, sz))
            total_bytes += sz

        idx = 0
        while total_bytes > max_total_bytes and idx < len(dir_sizes):
            d, sz = dir_sizes[idx]
            shutil.rmtree(d, ignore_errors=True)
            total_bytes -= sz
            removed += 1
            idx += 1

    return removed


def _max_concurrent_runs() -> int:
    """`MAX_CONCURRENT_RUNS` analyses at once across all visitors (default 4, minimum 1)."""
    try:
        return max(1, int(os.getenv("MAX_CONCURRENT_RUNS", "4")))
    except ValueError:
        return 4


#: Shared by every session in this process, so one busy server cannot be asked to run
#: an unbounded number of analyses at once. Extra runs wait their turn and say so.
_RUN_SLOTS = threading.BoundedSemaphore(_max_concurrent_runs())
_SLOT_POLL_S = 0.5
_HEARTBEAT_INTERVAL_S = 5.0


def _runner_backend() -> Literal["process", "thread"]:
    """Return the supported runner backend, safely defaulting to isolation."""
    return "thread" if os.getenv("RUNNER_BACKEND", "process").strip().lower() == "thread" else "process"


def _stop_grace_s() -> float:
    """Return the cancellation grace period without accepting a negative wait."""
    try:
        return max(0.0, float(os.getenv("RUNNER_STOP_GRACE_S", "10")))
    except ValueError:
        return 10.0


def _worker_cpu_threads() -> int:
    """Bound native numerical libraries inside one analysis worker."""
    try:
        return max(1, int(os.getenv("RUNNER_CPU_THREADS", "2")))
    except ValueError:
        return 2


def _positive_limit(name: str, default: float) -> float:
    """Read a non-negative operational limit; zero disables the limit."""
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _process_tree_rss_bytes(process_id: int) -> int | None:
    """Return current RSS for an exact worker and its living descendants."""
    try:
        process = psutil.Process(process_id)
        members = [process, *process.children(recursive=True)]
        return sum(int(member.memory_info().rss) for member in members if member.is_running())
    except psutil.Error:
        return None


def _cap_worker_cpu_threads() -> None:
    """Apply a conservative thread cap before importing the controller stack."""
    cap = _worker_cpu_threads()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "XGBOOST_NUM_THREADS",
    ):
        try:
            existing = int(os.environ.get(key, str(cap)))
        except ValueError:
            existing = cap
        os.environ[key] = str(max(1, min(cap, existing)))


def _configure_worker_streams() -> None:
    """Make Rich/controller output safe on Windows non-UTF-8 consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

# Kept outside Streamlit session state so a replacement browser session can
# adopt a still-running local worker by its validated durable run ID.
_ACTIVE_RUNS: dict[str, ActiveRun] = {}
_ACTIVE_RUNS_LOCK = threading.Lock()
_FORGOTTEN_RUNS: set[str] = set()


@dataclass(frozen=True)
class RunSpec:
    """Everything one run needs, fixed at click time."""

    dataset_path: str
    output_dir: str
    tmp_dir: str
    stage_names: tuple[tuple[str, str], ...]
    objective: str
    target: str | None
    min_iterations: int
    max_iterations: int
    use_llm: bool
    use_ml: bool
    run_config: RunConfig = field(repr=False)
    enable_rlm: bool = True
    related_paths: tuple[str, ...] = ()
    join_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    ml_max_depth: int | None = None
    ml_test_size: float | None = None
    ml_n_cv_folds: int | None = None
    ml_tune: bool | None = None
    compute_backend: str = "auto"
    is_sample: bool = False
    #: Persist the completed artifact tree where "Compare with a previous run"
    #: looks. Off on a shared deployment, where that folder would mix visitors'
    #: runs.  The name is retained for configuration compatibility.
    keep_summary: bool = True


@dataclass(frozen=True)
class RunOutcome:
    """A finished run's results, ready for the script thread to fold into session state."""

    final: dict[str, Any]
    tool_results: list[dict[str, Any]]
    profile: dict[str, Any] | None
    read_report: Any
    coercions: Any
    profile_status: Any
    dashboard: list[dict[str, Any]] | None
    run_view: RunView
    llm_warning: str | None
    summary_path: str | None
    metadata: Any
    stopped: bool


@dataclass(frozen=True)
class RunSnapshot:
    """A consistent copy of the run's live state, safe to render."""

    state: RunState
    stage_log: tuple[StageEntry, ...]
    progress_lines: tuple[str, ...]
    note: str
    stop_requested: bool
    discard: bool
    #: Headline-worthy findings seen so far, in the order they were found.
    provisional: tuple[ProvisionalFinding, ...] = ()


class _WorkerRun:
    """Minimal, pickle-safe ``ActiveRun`` facade used inside a spawned child.

    The analysis code only needs this small callback surface.  It deliberately
    contains no locks, registry, Streamlit state, or parent object reference;
    all state changes travel over the queue and are applied by the monitor.
    """

    def __init__(self, spec: RunSpec, events: Any, stop_event: Any) -> None:
        self.spec = spec
        self._events = events
        self._stop_event = stop_event

    @property
    def should_stop(self) -> bool:
        return bool(self._stop_event.is_set())

    def _send(self, kind: str, *payload: Any) -> None:
        message = (kind, *payload)
        # ``Queue.put`` serialises on a feeder thread; force that check here
        # so a non-pickleable controller value becomes a visible worker error
        # instead of a silent lost completion event.
        pickle.dumps(message)
        self._events.put(message)

    def set_stage(self, num: str, status: str, detail: str = "") -> None:
        self._send("stage", num, status, detail)

    def advance(
        self,
        num: str,
        detail: str,
        *,
        closing: dict[str, str] | None = None,
        reopen: tuple[str, ...] = (),
    ) -> None:
        self._send("advance", num, detail, closing, reopen)

    def add_progress(self, line: str) -> None:
        self._send("progress", line)

    def add_finding(self, finding: Finding) -> None:
        self._send("finding", finding)

    def set_note(self, note: str) -> None:
        self._send("note", note)

    def set_metadata(self, meta: Any) -> None:
        self._send("metadata", meta)

    def tool_event(self, tool_name: str, status: str) -> None:
        self._send("tool", tool_name, status)

    def _complete(self, outcome: RunOutcome) -> None:
        self._send("complete", outcome)


def _worker_entry(spec: RunSpec, events: Any, stop_event: Any) -> None:
    """Spawn target: cap native threads, execute, and return a safe error trace."""
    _configure_worker_streams()
    _cap_worker_cpu_threads()
    worker = _WorkerRun(spec, events, stop_event)
    try:
        _execute(worker)
    except Exception:
        # The parent, not this process, decides the user-visible terminal
        # state. Tracebacks are vital for operators; a hard process exit is
        # detected separately by the parent monitor.
        events.put(("error", traceback.format_exc()))


class ActiveRun:
    """One analysis with a parent monitor. Create, `start()`, then poll it."""

    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._discard = False
        self._state: RunState = "running"
        self._stage_log: list[StageEntry] = [(num, "pending", "") for num, _ in spec.stage_names]
        self._progress_lines: list[str] = []
        self._note = ""
        self._provisional: list[ProvisionalFinding] = []
        self._outcome: RunOutcome | None = None
        self._error: str | None = None
        self._metadata: Any = None
        self._thread: threading.Thread | None = None
        self._registry = RunRegistry(spec.tmp_dir)
        self._run_id: str | None = None
        self._registry_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._worker: Any = None
        self._worker_stop: Any = None
        self._active_tool: tuple[str, float] | None = None

    @property
    def run_id(self) -> str | None:
        """The durable opaque identifier assigned when ``start`` is called."""
        return self._run_id

    @property
    def registry_status(self) -> RunStatus | None:
        """The safe durable status, suitable for a reconnecting UI."""
        return self._registry.load()

    # ── control (script thread) ──────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("run already started")
        record = self._registry.create()
        self._run_id = record.run_id
        with _ACTIVE_RUNS_LOCK:
            _ACTIVE_RUNS[record.run_id] = self
        self._thread = threading.Thread(target=self._main, name="dsa-run", daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()
        if self._worker_stop is not None:
            self._worker_stop.set()

    def discard_and_stop(self) -> None:
        """Abandon the run: stop it and drop its results once the worker ends."""
        with self._lock:
            self._discard = True
        self._stop_event.set()

    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    # ── reads (any thread) ───────────────────────────────────────────────────

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._state != "running"

    @property
    def outcome(self) -> RunOutcome | None:
        with self._lock:
            return self._outcome

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def metadata(self) -> Any:
        with self._lock:
            return self._metadata

    def snapshot(self) -> RunSnapshot:
        with self._lock:
            return RunSnapshot(
                state=self._state,
                stage_log=tuple(self._stage_log),
                progress_lines=tuple(self._progress_lines),
                note=self._note,
                stop_requested=self._stop_event.is_set(),
                discard=self._discard,
                provisional=tuple(self._provisional),
            )

    # ── writes (worker thread) ───────────────────────────────────────────────

    def set_stage(self, num: str, status: str, detail: str = "") -> None:
        with self._lock:
            self._stage_log = [e for e in self._stage_log if e[0] != num] + [(num, status, detail)]
            self._stage_log.sort(key=lambda e: e[0])
        self._update_registry(stage=f"{num}:{status}")

    def advance(
        self,
        num: str,
        detail: str,
        *,
        closing: dict[str, str] | None = None,
        reopen: tuple[str, ...] = (),
    ) -> None:
        """Make `num` the one active stage, in a single locked update.

        Any other active stage is closed as done (so two stages never read "Working" at once);
        `closing` names stages to close with a final detail, and `reopen` puts stages that an
        earlier cycle finished back to waiting, because a new cycle is about to repeat them.
        """
        closing = closing or {}
        with self._lock:
            log = {n: (s, d) for n, s, d in self._stage_log}
            for n, (status, old_detail) in list(log.items()):
                if n == num:
                    continue
                if n in closing:
                    log[n] = ("done", closing[n])
                elif status == "active":
                    log[n] = ("done", old_detail)
            for n in reopen:
                if log.get(n, ("", ""))[0] == "done":
                    log[n] = ("pending", "")
            log[num] = ("active", detail)
            self._stage_log = sorted(((n, s, d) for n, (s, d) in log.items()), key=lambda e: e[0])
        self._update_registry(stage=f"{num}:active")

    def add_progress(self, line: str) -> None:
        with self._lock:
            self._progress_lines.append(line)

    def add_finding(self, finding: Finding) -> None:
        """Record a new finding as provisional, if it would be a headline candidate."""
        if not is_headline_finding(finding.layer, finding.kind) or is_trivial(finding):
            return
        with self._lock:
            self._provisional.append(
                ProvisionalFinding(finding.finding_id, finding.kind, finding.headline)
            )

    def set_note(self, note: str) -> None:
        with self._lock:
            self._note = note

    def set_metadata(self, meta: Any) -> None:
        with self._lock:
            self._metadata = meta

    def tool_event(self, tool_name: str, status: str) -> None:
        """Track one controller tool deadline without changing tool contracts."""
        with self._lock:
            if status == "running":
                self._active_tool = (tool_name, time.monotonic())
            elif self._active_tool is not None and self._active_tool[0] == tool_name:
                self._active_tool = None

    def _complete(self, outcome: RunOutcome) -> None:
        status: RunTerminalStatus = "partial" if outcome.stopped else "succeeded"
        self._finish_registry(status, _public_artifact_paths(self.spec.output_dir))
        with self._lock:
            self._outcome = outcome
            self._state = "done"

    def _fail(self, error: str, terminal_status: RunTerminalStatus = "failed") -> None:
        self._finish_registry(terminal_status)
        with self._lock:
            self._error = error
            self._state = "error"
            # The stage that was running is the one that failed.
            for num, status, _ in reversed(self._stage_log):
                if status == "active":
                    self._stage_log = [e for e in self._stage_log if e[0] != num] + [
                        (num, "error", "failed")
                    ]
                    self._stage_log.sort(key=lambda e: e[0])
                    break

    def _update_registry(self, stage: str | None = None) -> None:
        """Best-effort status persistence must never hide the real run result."""
        try:
            with self._registry_lock:
                self._registry.heartbeat(stage)
        except (OSError, RuntimeError):
            pass

    def _finish_registry(
        self, status: RunTerminalStatus, artifacts: tuple[str, ...] = ()
    ) -> None:
        try:
            with self._registry_lock:
                self._registry.finish(status, artifacts)
        except (OSError, RuntimeError):
            pass

    def _heartbeat_main(self) -> None:
        while not self._heartbeat_stop.wait(_HEARTBEAT_INTERVAL_S):
            self._update_registry()

    def _acquire_slot(self) -> bool:
        """Wait for a free run slot; False if the run was stopped while it waited."""
        if _RUN_SLOTS.acquire(blocking=False):
            return True
        self.set_note("Waiting for a free slot: other analyses are running on this server.")
        while not self._stop_event.is_set():
            if _RUN_SLOTS.acquire(timeout=_SLOT_POLL_S):
                self.set_note("")
                return True
        return False

    def _main(self) -> None:
        slot = False
        heartbeater = threading.Thread(target=self._heartbeat_main, name="dsa-run-heartbeat", daemon=True)
        heartbeater.start()
        try:
            slot = self._acquire_slot()
            if slot:
                if _runner_backend() == "thread":
                    _execute(self)
                else:
                    self._run_in_process()
            else:
                self._fail("Stopped before the analysis started.")
        except Exception:
            self._fail(traceback.format_exc())
        finally:
            self._heartbeat_stop.set()
            heartbeater.join(timeout=_HEARTBEAT_INTERVAL_S + 1.0)
            if slot:
                _RUN_SLOTS.release()
            with self._lock:
                discarded = self._discard
                if self._state == "running":  # defensive: never leave the page polling forever
                    self._state = "error"
                    self._error = self._error or "The run ended without a result."
            if discarded and self._run_id is not None:
                with _ACTIVE_RUNS_LOCK:
                    if _ACTIVE_RUNS.get(self._run_id) is self:
                        _ACTIVE_RUNS.pop(self._run_id, None)
            if discarded:
                remove_run_dir(self.spec.tmp_dir)

    def _run_in_process(self) -> None:
        """Run one analysis child and project its IPC events onto this parent."""
        context = multiprocessing.get_context("spawn")
        events = context.Queue()
        stop_event = context.Event()
        if self._stop_event.is_set():
            stop_event.set()
        worker = context.Process(
            target=_worker_entry,
            args=(self.spec, events, stop_event),
            name="dsa-analysis-worker",
            daemon=False,
        )
        self._worker = worker
        self._worker_stop = stop_event
        worker.start()
        if worker.pid is None:
            raise RuntimeError("analysis worker started without a process ID")
        with self._registry_lock:
            self._registry.set_process_identity(worker.pid)
        stop_started: float | None = time.monotonic() if self._stop_event.is_set() else None
        run_started = time.monotonic()
        limit_reason: str | None = None
        force_stopped = False
        try:
            while True:
                if self._stop_event.is_set() and stop_started is None:
                    stop_started = time.monotonic()
                    stop_event.set()
                self._drain_worker_events(events, timeout=0.15)
                if self.finished:
                    break
                if stop_started is None:
                    limit_reason = self._resource_limit_reason(worker.pid, run_started)
                    if limit_reason is not None:
                        self.add_progress(f"[limit] {limit_reason}; requesting a partial result")
                        self.set_note(limit_reason)
                        self.request_stop()
                        stop_started = time.monotonic()
                if (
                    stop_started is not None
                    and worker.is_alive()
                    and time.monotonic() - stop_started >= _stop_grace_s()
                ):
                    worker.terminate()
                    force_stopped = True
                    worker.join(timeout=5.0)
                    self._fail(
                        (
                            f"{limit_reason}; the worker did not return a partial result within "
                            "the cancellation grace period."
                            if limit_reason is not None
                            else "Stopped after the worker did not respond within the cancellation grace period."
                        ),
                        terminal_status="cancelled",
                    )
                    break
                if not worker.is_alive():
                    # A Queue feeder can finish after the child exits; drain
                    # every already-available event before declaring it lost.
                    self._drain_worker_events(events, timeout=0.0)
                    if not self.finished:
                        if self._stop_event.is_set():
                            self._fail("Stopped before the worker returned a partial result.", "cancelled")
                        else:
                            self._fail(
                                f"Analysis worker exited unexpectedly (exit code {worker.exitcode})."
                            )
                    break
            if worker.is_alive():
                worker.join(timeout=5.0)
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5.0)
                if not self.finished and not force_stopped:
                    self._fail("Analysis worker was terminated during cleanup.")
            events.close()
            events.join_thread()
            self._worker = None
            self._worker_stop = None

    def _resource_limit_reason(self, worker_pid: int | None, run_started: float) -> str | None:
        """Return a plain-language limit breach for cooperative cancellation."""
        now = time.monotonic()
        tool_timeout = _positive_limit("TOOL_TIMEOUT_S", 120.0)
        with self._lock:
            active_tool = self._active_tool
        if tool_timeout and active_tool is not None and now - active_tool[1] >= tool_timeout:
            return f"Tool '{active_tool[0]}' exceeded its {tool_timeout:g}-second limit"
        wall_timeout = _positive_limit("RUNNER_WALL_TIMEOUT_S", 3600.0)
        if wall_timeout and now - run_started >= wall_timeout:
            return f"Analysis exceeded its {wall_timeout:g}-second wall-clock limit"
        rss_limit_mb = _positive_limit("RUNNER_MEMORY_MB", 4096.0)
        if worker_pid is not None and rss_limit_mb:
            rss = _process_tree_rss_bytes(worker_pid)
            if rss is not None and rss > int(rss_limit_mb * 1024 * 1024):
                return f"Analysis exceeded its {rss_limit_mb:g} MB memory limit"
        return None

    def _drain_worker_events(self, events: Any, timeout: float) -> None:
        """Apply available child events while treating malformed IPC as a failure."""
        first = True
        while True:
            try:
                if first and timeout > 0:
                    event = events.get(timeout=timeout)
                else:
                    event = events.get_nowait()
            except queue.Empty:
                return
            first = False
            if not isinstance(event, tuple) or not event or not isinstance(event[0], str):
                self._fail("Analysis worker sent an invalid progress event.")
                return
            self._apply_worker_event(event)
            if self.finished:
                return

    def _apply_worker_event(self, event: tuple[Any, ...]) -> None:
        """Validate a child's small event protocol before mutating parent state."""
        kind, *payload = event
        if kind == "stage" and len(payload) == 3 and all(isinstance(item, str) for item in payload):
            self.set_stage(payload[0], payload[1], payload[2])
        elif kind == "advance" and len(payload) == 4:
            num, detail, closing, reopen = payload
            if isinstance(num, str) and isinstance(detail, str) and (
                closing is None or isinstance(closing, dict)
            ) and isinstance(reopen, tuple):
                self.advance(num, detail, closing=closing, reopen=reopen)
            else:
                self._fail("Analysis worker sent an invalid stage transition.")
        elif kind == "progress" and len(payload) == 1 and isinstance(payload[0], str):
            self.add_progress(payload[0])
        elif kind == "finding" and len(payload) == 1 and isinstance(payload[0], Finding):
            self.add_finding(payload[0])
        elif kind == "note" and len(payload) == 1 and isinstance(payload[0], str):
            self.set_note(payload[0])
        elif kind == "metadata" and len(payload) == 1:
            self.set_metadata(payload[0])
        elif kind == "tool" and len(payload) == 2 and all(
            isinstance(item, str) for item in payload
        ):
            self.tool_event(payload[0], payload[1])
        elif kind == "complete" and len(payload) == 1 and isinstance(payload[0], RunOutcome):
            self._complete(payload[0])
        elif kind == "error" and len(payload) == 1 and isinstance(payload[0], str):
            self._fail(payload[0])
        else:
            self._fail("Analysis worker sent an invalid event.")


def adopt_run(run_id: str, search_root: str | Path | None = None) -> ActiveRun | None:
    """Return a local run when its durable registry record validates.

    Completed runs deliberately remain adoptable until the user starts a new
    analysis.  This lets a replacement Streamlit session fold an outcome that
    finished while its original browser tab was disconnected.
    """
    if run_id is None or not RunRegistry.valid_run_id(run_id):
        return None
    with _ACTIVE_RUNS_LOCK:
        if run_id in _FORGOTTEN_RUNS:
            return None
        candidate = _ACTIVE_RUNS.get(run_id)
    if candidate is not None:
        status = candidate.registry_status
        return candidate if status is not None and status.run_id == run_id else None

    # Inspect on-disk run_status.json and artifact directories if missing from _ACTIVE_RUNS
    registry = find_run(run_id, root=search_root)
    if registry is None:
        return None
    status = registry.load()
    if status is None or status.run_id != run_id:
        return None

    dummy_spec = RunSpec(
        dataset_path="",
        output_dir=str(registry.run_dir),
        tmp_dir=str(registry.run_dir),
        stage_names=(),
        objective="",
        target=None,
        min_iterations=1,
        max_iterations=1,
        use_llm=False,
        use_ml=False,
        run_config=RunConfig(),
    )
    adopted = ActiveRun(dummy_spec)
    adopted._run_id = run_id
    adopted._registry = registry

    if status.status == "succeeded":
        adopted._state = "done"
    elif status.status in ("failed", "cancelled"):
        adopted._state = "error"
        adopted._error = f"Run status: {status.status}"
    else:
        if registry.is_abandoned(60.0):
            adopted._state = "error"
            adopted._error = "Process terminated or abandoned."
        else:
            adopted._state = "running"

    final_report_path: Path | None = None
    candidates = [
        registry.run_dir / "reports" / "final_report.json",
        registry.run_dir / "reports" / "data_raw.json",
        registry.run_dir / "out" / "reports" / "final_report.json",
        registry.run_dir / "out" / "reports" / "data_raw.json",
        registry.run_dir / "output" / "reports" / "final_report.json",
        registry.run_dir / "output" / "reports" / "data_raw.json",
        registry.run_dir / "final_report.json",
    ]
    for art in status.artifacts:
        p = Path(art)
        if not p.is_absolute():
            for base in (registry.run_dir, registry.run_dir / "out", registry.run_dir / "output"):
                cand = base / art
                if cand.name in ("final_report.json", "data_raw.json") and cand.is_file():
                    candidates.insert(0, cand)
        elif p.name in ("final_report.json", "data_raw.json") and p.is_file():
            candidates.insert(0, p)

    for c in candidates:
        if c.is_file():
            final_report_path = c
            break

    if final_report_path is not None:
        try:
            final_data = json.loads(final_report_path.read_text(encoding="utf-8"))
            if isinstance(final_data, dict):
                dash_path = final_report_path.parent / "dashboard.json"
                dash_data = None
                if dash_path.is_file():
                    try:
                        dash_data = json.loads(dash_path.read_text(encoding="utf-8"))
                    except Exception:
                        dash_data = None

                view = build_run_view(
                    final_data,
                    {},
                    objective=final_data.get("objective", ""),
                    is_sample=False,
                )
                adopted._outcome = RunOutcome(
                    final=final_data,
                    tool_results=final_data.get("tool_results", []),
                    profile=final_data.get("profile"),
                    read_report=None,
                    coercions=None,
                    profile_status=None,
                    dashboard=dash_data,
                    run_view=view,
                    llm_warning=None,
                    summary_path=str(final_report_path),
                    metadata=final_data.get("metadata"),
                    stopped=status.status == "cancelled",
                )
        except Exception:
            pass

    with _ACTIVE_RUNS_LOCK:
        _ACTIVE_RUNS[run_id] = adopted
    return adopted


def forget_run(run_id: str | None) -> None:
    """Release one retained run after its owner explicitly starts over."""
    if run_id is None or not RunRegistry.valid_run_id(run_id):
        return
    with _ACTIVE_RUNS_LOCK:
        _FORGOTTEN_RUNS.add(run_id)
        _ACTIVE_RUNS.pop(run_id, None)


def _public_artifact_paths(output_dir: str) -> tuple[str, ...]:
    """Return only already allowlisted public artifacts relative to output."""
    return tuple(path.as_posix() for path in _retained_artifacts(Path(output_dir)))


def _execute(run: ActiveRun | _WorkerRun) -> None:
    """The analysis itself: the work the Run click handler used to do inline."""
    # Imported here, like the script did: `src` is loaded after the console stub.
    from src.core.controller import AgentController

    spec = run.spec
    names = dict(spec.stage_names)

    def upd(num: str, status: str, detail: str = "") -> None:
        run.set_stage(num, status, detail)
        run.add_progress(
            f"{_STATUS_ICON.get(status, '[    ]')} Stage {num}: {names.get(num, num)}"
            + (f"  {detail}" if detail else "")
        )

    upd("1", "active", "ingesting…")
    agent = AgentController(
        min_iterations=spec.min_iterations,
        max_iterations=spec.max_iterations,
        enable_rlm=spec.enable_rlm,
        use_llm=spec.use_llm,
        use_ml=spec.use_ml,
        objective=spec.objective,
        output_dir=spec.output_dir,
        run_config=spec.run_config,
    )
    agent.memory.set_context("ui_compute_backend", spec.compute_backend)
    if spec.use_ml:
        # IMPROVEMENTS.md 7.16: TrainModelTool.requires_context reads these back
        # and fills them into the real train_model call when the planner leaves
        # them empty.
        agent.memory.set_context("ui_max_depth", spec.ml_max_depth)
        agent.memory.set_context("ui_test_size", spec.ml_test_size)
        agent.memory.set_context("ui_n_cv_folds", spec.ml_n_cv_folds)
        agent.memory.set_context("ui_tune_hyperparameters", spec.ml_tune)
    agent.should_stop = lambda: run.should_stop

    meta = agent.load_dataset(
        spec.dataset_path,
        target_hint=spec.target,
        interactive=False,
        related_files=list(spec.related_paths) or None,
        join_overrides=spec.join_overrides or None,
    )
    run.set_metadata(meta)
    upd(
        "1",
        "done",
        f"{meta.row_count:,} rows × {meta.column_count} cols · task={meta.task_type} · target={meta.target_column}",
    )
    plan_detail = (
        "calling LLM for analysis plan…"
        if spec.use_llm
        else "building deterministic plan from the data profile…"
    )
    upd("2", "active", plan_detail)
    for num in ("3", "4", "5"):
        upd(num, "pending")
    upd("6", "skipped", "not needed")
    upd("7", "pending")

    # What the controller reports maps onto one moving "you are here": plan (2), run the planned
    # steps (3), and on later cycles interpret and refine (4, 5) before running again; the
    # decomposition (6) and the report (7) follow. `advance` keeps exactly one stage active.
    cycle = {"iteration": 1, "planned": False, "ran": 0}

    def on_step(tool_name: str, status: str, detail: str) -> None:
        run.tool_event(tool_name, status)
        iteration = cycle["iteration"]
        label = detail if iteration == 1 else f"iter {iteration} · {detail}"
        if not cycle["planned"]:
            # The first tool of a cycle means its plan arrived: the reasoning stages are over.
            cycle["planned"] = True
            closing = (
                {"2": "plan ready"}
                if iteration == 1
                else {"4": f"iter {iteration}: results interpreted", "5": f"iter {iteration}: plan refined"}
            )
            run.advance("3", label, closing=closing)
        else:
            run.set_stage("3", "active", label)
        if status != "running":
            cycle["ran"] += 1
        run.add_progress(f"       {'ok  ' if status == 'success' else '... '}{detail}")
        if tool_name == "train_model":
            run.set_note("Still working. Training models can take a few minutes depending on the data size.")

    def on_iter(iteration: int, stage: str) -> None:
        if "stage6" not in stage and "stage7" not in stage:
            cycle["iteration"] = iteration
            cycle["planned"] = False
        ran = f"{cycle['ran']} step(s) run"
        if "stage7" in stage:
            run.advance("7", "writing the report…", closing={"3": ran})
            run.add_progress("[run ] Writing the report")
        elif "stage6" in stage:
            run.advance("6", "splitting the question into sub-tasks…", closing={"3": ran})
            run.add_progress("[run ] Splitting the question into sub-tasks")
        elif "stage2" in stage:
            run.advance("2", plan_detail if iteration == 1 else f"iter {iteration}: LLM reasoning…")
            run.add_progress(f"[run ] Iteration {iteration}: model reasoning")
        elif "stage4" in stage or "stage5" in stage:
            run.advance(
                "4",
                f"iter {iteration}: interpreting results and refining the plan…",
                closing={"3": ran},
                reopen=("5",),
            )
            run.add_progress(f"[run ] Iteration {iteration}: interpreting and refining")
        if iteration > 1 and "stage7" not in stage:
            run.set_note("Still working. Refining answers can take a few minutes.")

    agent.on_step_callback = on_step
    agent.on_iteration_callback = on_iter
    agent.on_finding_callback = run.add_finding

    final = agent.analyze()

    # The controller is gone once the page reruns, so everything RunView needs
    # from memory is snapshotted into a plain dict here.
    view = build_run_view(
        final,
        {k: agent.memory.get_context(k) for k in RUN_VIEW_CONTEXT_KEYS},
        objective=spec.objective,
        is_sample=spec.is_sample,
    )
    tool_results = [r.to_dict() for r in agent.memory.tool_results]
    stopped = bool(final.get("stopped"))

    # Say what actually happened to each stage: a stopped run did not do all of it.
    tool_names = list(dict.fromkeys(r.get("tool_name", "") for r in tool_results))
    upd("2", "done", "plan generated" if stopped else "plan generated & executed")
    if stopped:
        upd("3", "done", f"stopped early: {len(tool_results)} tools executed")
        upd("4", "skipped", "stopped")
        upd("5", "skipped", "stopped")
        upd("6", "skipped", "stopped")
        upd("7", "done", "partial report saved")
    else:
        upd("3", "done", f"{len(tool_results)} tools executed: {', '.join(tool_names[:5])}")
        upd("4", "done", "results interpreted")
        upd("5", "done", f"{agent.memory.iteration_count} iteration(s)")
        upd("6", "skipped", "not needed")
        upd("7", "done", "report saved")

    dashboard: list[dict[str, Any]] | None = None
    dash_path = Path(spec.output_dir) / "reports" / "dashboard.json"
    if dash_path.exists():
        try:
            dashboard = json.loads(dash_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            dashboard = None

    summary_path: str | None = None
    if spec.keep_summary:
        summary_path = retain_run_artifacts(spec.output_dir)

    llm_err = agent.memory.get_context("llm_error")
    run._complete(
        RunOutcome(
            final=final,
            tool_results=tool_results,
            profile=agent.last_profile.to_dict() if agent.last_profile is not None else None,
            read_report=agent.memory.get_context("read_report"),
            coercions=agent.memory.get_context("coercions"),
            profile_status=agent.memory.get_context("profile_status"),
            dashboard=dashboard,
            run_view=view,
            llm_warning=f"Fallback plan was used (LLM issue): {llm_err[:300]}" if llm_err else None,
            summary_path=summary_path,
            metadata=meta,
            stopped=stopped,
        )
    )
