"""ui/run.py: the background run, driven with the real controller and the LLM off."""
from __future__ import annotations

import ast
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ui.run import RUN_DIR_PREFIX, ActiveRun, RunSpec, remove_run_dir

from src.core.run_config import RunConfig

ROOT = Path(__file__).resolve().parents[1]
STAGES = tuple((str(i), f"Stage {i}") for i in range(1, 8))


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHART_DESIGN", "false")
    monkeypatch.chdir(ROOT)  # the run summary copy is relative to the working directory


def _csv(tmp_path: Path) -> str:
    rng = np.random.default_rng(42)
    n = 120
    group = rng.choice(["a", "b", "c"], n)
    df = pd.DataFrame(
        {
            "group": group,
            "value": rng.normal(10, 2, n) + np.where(group == "a", 4.0, 0.0),
            "other": rng.normal(0, 1, n),
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def _spec(tmp_path: Path, dataset: str | None = None, **over: object) -> RunSpec:
    base = RunSpec(
        dataset_path=dataset or _csv(tmp_path),
        output_dir=str(tmp_path / "out"),
        tmp_dir=str(tmp_path),
        stage_names=STAGES,
        objective="Does value differ by group?",
        target=None,
        min_iterations=1,
        max_iterations=3,
        enable_rlm=False,
        use_llm=False,
        use_ml=False,
        run_config=RunConfig(),
        keep_summary=False,
    )
    values = {**base.__dict__, **over}
    return RunSpec(**values)  # type: ignore[arg-type]


def _wait(run: ActiveRun, seconds: float = 120.0) -> None:
    deadline = time.monotonic() + seconds
    while not run.finished and time.monotonic() < deadline:
        time.sleep(0.05)
    assert run.finished, "the run did not finish in time"


def test_a_run_completes_with_a_view_and_truthful_stages(tmp_path: Path) -> None:
    run = ActiveRun(_spec(tmp_path))
    run.start()
    _wait(run)

    snap = run.snapshot()
    assert snap.state == "done"
    out = run.outcome
    assert out is not None and out.stopped is False
    assert out.final["status"] == "complete"
    assert out.run_view.objective == "Does value differ by group?"
    assert out.run_view.how.no_ai is True
    assert out.tool_results
    assert {e[0]: e[1] for e in snap.stage_log}["7"] == "done"
    assert snap.progress_lines, "progress lines feed the Details run trace"


def test_provisional_findings_are_collected_while_running_and_match_the_final_ones(tmp_path: Path) -> None:
    run = ActiveRun(_spec(tmp_path))
    run.start()
    _wait(run)

    snap = run.snapshot()
    out = run.outcome
    assert out is not None
    assert snap.provisional, "the planted group effect should surface at least one headline finding"
    final_ids = {f["finding_id"] for f in out.run_view.findings}
    assert {p.finding_id for p in snap.provisional} <= final_ids
    # Only headline candidates are shown while running.
    by_id = {f["finding_id"]: f for f in out.run_view.findings}
    for p in snap.provisional:
        assert by_id[p.finding_id]["layer"] in ("exec", "analyst")
        assert by_id[p.finding_id]["kind"] not in ("method_fit", "coverage_gap")
    # And the finished view itself holds none.
    assert out.run_view.provisional_findings == ()


def test_stop_requested_up_front_gives_a_stopped_partial_result(tmp_path: Path) -> None:
    run = ActiveRun(_spec(tmp_path))
    run.request_stop()
    run.start()
    _wait(run)

    out = run.outcome
    assert out is not None and out.stopped is True
    assert out.final["stopped"] is True
    stages = {e[0]: e[1] for e in run.snapshot().stage_log}
    assert stages["3"] == "done"
    assert stages["4"] == "skipped" and stages["5"] == "skipped"
    assert "stopped" in {e[0]: e[2] for e in run.snapshot().stage_log}["3"]
    assert any("stopped" in n.lower() for n in out.run_view.how.fallbacks)


def test_a_failing_run_reports_the_error_and_marks_the_stage(tmp_path: Path) -> None:
    run = ActiveRun(_spec(tmp_path, dataset=str(tmp_path / "missing.csv")))
    run.start()
    _wait(run)

    assert run.snapshot().state == "error"
    assert run.outcome is None
    assert run.error and "Traceback" in run.error
    assert any(status == "error" for _, status, _ in run.snapshot().stage_log)


def test_discarding_removes_the_run_directory_when_the_worker_ends() -> None:
    tmp = tempfile.mkdtemp(prefix=RUN_DIR_PREFIX)
    try:
        csv_dir = Path(tmp)
        spec = _spec(csv_dir)
        run = ActiveRun(spec)
        run.discard_and_stop()
        run.start()
        _wait(run)
        assert run.snapshot().discard is True
        deadline = time.monotonic() + 10
        while Path(tmp).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not Path(tmp).exists()
    finally:
        remove_run_dir(tmp)


def test_remove_run_dir_only_touches_prefixed_dirs_under_the_temp_root(tmp_path: Path) -> None:
    keep = tmp_path / "important"
    keep.mkdir()
    remove_run_dir(str(keep))  # no prefix: untouched
    assert keep.exists()
    remove_run_dir(None)
    remove_run_dir("")


def test_a_run_cannot_be_started_twice(tmp_path: Path) -> None:
    run = ActiveRun(_spec(tmp_path))
    run.start()
    with pytest.raises(RuntimeError):
        run.start()
    _wait(run)


def test_two_runs_keep_their_own_objectives(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = ActiveRun(_spec(tmp_path / "a", objective="Question A?"))
    b = ActiveRun(_spec(tmp_path / "b", objective="Question B?"))
    a.start()
    b.start()
    _wait(a)
    _wait(b)
    assert a.outcome is not None and b.outcome is not None
    assert a.outcome.run_view.objective == "Question A?"
    assert b.outcome.run_view.objective == "Question B?"


def test_the_worker_module_never_imports_streamlit() -> None:
    tree = ast.parse((ROOT / "ui" / "run.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not {m for m in imported if m == "streamlit" or m.startswith("streamlit.")}
    # No attribute access to session state either (the docstring may mention it).
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "session_state"]


# ── stale run directories and the shared run limiter ─────────────────────────


def test_sweep_removes_only_old_prefixed_dirs_under_the_temp_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from ui.run import sweep_stale_run_dirs

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    old = tmp_path / f"{RUN_DIR_PREFIX}old"
    fresh = tmp_path / f"{RUN_DIR_PREFIX}fresh"
    other = tmp_path / "somebody-elses-dir"
    for d in (old, fresh, other):
        (d / "output").mkdir(parents=True)
    long_ago = time.time() - 3 * 24 * 3600
    for d in (old, other):
        os.utime(d, (long_ago, long_ago))

    assert sweep_stale_run_dirs(max_age_s=24 * 3600) == 1
    assert not old.exists()
    assert fresh.exists() and other.exists()


def test_a_run_waits_for_a_free_slot_and_can_be_stopped_while_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import ui.run as run_mod

    monkeypatch.setattr(run_mod, "_RUN_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(run_mod, "_SLOT_POLL_S", 0.05)
    assert run_mod._RUN_SLOTS.acquire(blocking=False)  # the only slot is taken

    run = ActiveRun(_spec(tmp_path))
    run.start()
    deadline = time.time() + 5
    while "Waiting for a free slot" not in run.snapshot().note and time.time() < deadline:
        time.sleep(0.05)
    assert "Waiting for a free slot" in run.snapshot().note

    run.request_stop()
    while not run.finished and time.time() < deadline + 5:
        time.sleep(0.05)
    assert run.finished and run.error is not None
    assert "Stopped before the analysis started" in run.error
    # The waiting run never held a slot, so it must not have released one.
    run_mod._RUN_SLOTS.release()
    with pytest.raises(ValueError):
        run_mod._RUN_SLOTS.release()
