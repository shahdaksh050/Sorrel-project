"""
Background analysis run: the worker thread and the state the page polls.

The Streamlit script starts one `ActiveRun` per analysis and keeps it in session
state; a polling fragment reads `ActiveRun.snapshot()` once a second. Everything
here runs on, or is read from, a thread that is not the script thread, so:

* this module never imports Streamlit and never touches `st.session_state`;
* every setting the worker needs arrives in a frozen `RunSpec`;
* the worker writes only to the `ActiveRun`, under its lock, and the script
  thread folds the finished `RunOutcome` into session state itself.

Stop is cooperative. `request_stop` sets an event that the controller polls at
its checkpoints (`AgentController.should_stop`), so a tool that is already
running finishes first. Nothing here kills a thread.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from src.core.findings import Finding
from src.core.run_config import RunConfig
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
    enable_rlm: bool
    use_llm: bool
    use_ml: bool
    run_config: RunConfig = field(repr=False)
    related_paths: tuple[str, ...] = ()
    join_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    ml_max_depth: int | None = None
    ml_test_size: float | None = None
    ml_n_cv_folds: int | None = None
    ml_tune: bool | None = None
    is_sample: bool = False
    #: Copy the run's summary where "Compare with a previous run" looks. Off on
    #: a shared deployment, where that folder would mix visitors' runs.
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


class ActiveRun:
    """One analysis on its own thread. Create, `start()`, then poll `snapshot()`."""

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

    # ── control (script thread) ──────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("run already started")
        self._thread = threading.Thread(target=self._main, name="dsa-run", daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()

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

    def add_progress(self, line: str) -> None:
        with self._lock:
            self._progress_lines.append(line)

    def add_finding(self, finding: Finding) -> None:
        """Record a new finding as provisional, if it would be a headline candidate."""
        if not is_headline_finding(finding.layer, finding.kind):
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

    def _complete(self, outcome: RunOutcome) -> None:
        with self._lock:
            self._outcome = outcome
            self._state = "done"

    def _fail(self, error: str) -> None:
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

    def _main(self) -> None:
        try:
            _execute(self)
        except Exception:
            self._fail(traceback.format_exc())
        finally:
            with self._lock:
                discarded = self._discard
                if self._state == "running":  # defensive: never leave the page polling forever
                    self._state = "error"
                    self._error = self._error or "The run ended without a result."
            if discarded:
                remove_run_dir(self.spec.tmp_dir)


def _execute(run: ActiveRun) -> None:
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
    upd(
        "2",
        "active",
        "calling LLM for analysis plan…"
        if spec.use_llm
        else "building deterministic plan from the data profile…",
    )
    for num in ("3", "4", "5"):
        upd(num, "pending")
    upd("6", "pending" if spec.enable_rlm else "skipped", "" if spec.enable_rlm else "disabled")
    upd("7", "pending")

    def on_step(tool_name: str, status: str, detail: str) -> None:
        run.set_stage("3", "active", detail)
        run.add_progress(f"       {'ok  ' if status == 'success' else '... '}{detail}")
        if tool_name == "train_model":
            run.set_note("Still working. Training models can take a few minutes depending on the data size.")

    def on_iter(iteration: int, stage: str) -> None:
        if "stage2" in stage:
            run.set_stage("2", "active", f"iter {iteration}: LLM reasoning…")
            run.add_progress(f"[run ] Iteration {iteration}: model reasoning")
        elif "stage4" in stage or "stage5" in stage:
            run.set_stage("4", "active", f"iter {iteration}: interpreting results…")
            run.set_stage("5", "active", f"iter {iteration}: refining plan…")
            run.add_progress(f"[run ] Iteration {iteration}: interpreting and refining")
        if iteration > 1:
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
        if spec.enable_rlm:
            upd("6", "skipped", "stopped")
        upd("7", "done", "partial report saved")
    else:
        upd("3", "done", f"{len(tool_results)} tools executed: {', '.join(tool_names[:5])}")
        upd("4", "done", "results interpreted")
        upd("5", "done", f"{agent.memory.iteration_count} iteration(s)")
        if spec.enable_rlm:
            sub = agent.memory.get_context("rlm_sub_results")
            upd("6", "done", f"{len(sub)} sub-tasks" if sub else "no decomposition needed")
        upd("7", "done", "report saved")

    dashboard: list[dict[str, Any]] | None = None
    dash_path = Path(spec.output_dir) / "reports" / "dashboard.json"
    if dash_path.exists():
        try:
            dashboard = json.loads(dash_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            dashboard = None

    summary_path: str | None = None
    sum_path = Path(spec.output_dir) / "reports" / "summary.json"
    if sum_path.exists() and spec.keep_summary:
        try:  # keep a copy where "Compare with a previous run" looks
            keep = Path("output") / "runs" / f"ui-{datetime.now():%Y%m%d-%H%M%S}" / "reports"
            keep.mkdir(parents=True, exist_ok=True)
            (keep / "summary.json").write_bytes(sum_path.read_bytes())
            summary_path = str(keep / "summary.json")
        except OSError:
            summary_path = None

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
