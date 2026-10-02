"""
Per-run context: values that belong to ONE analysis run, not to the process.

Why this exists: the Streamlit app used to hand the user's objective to the
analysis tools through ``os.environ["USER_OBJECTIVE"]``. The environment is
process-wide, so on a shared server a second visitor's run could read the first
visitor's question (and steer measure selection with it). This module carries
the same value in a ``contextvars.ContextVar``, which is scoped to the thread
that set it.

Rules:
  * ``AgentController.load_dataset`` and ``analyze`` run inside
    :func:`objective_scope`, which sets the objective and restores the previous
    value on exit, so nothing outlives the run in that thread.
  * Tools read it through :func:`current_objective`. When nothing was set (CLI,
    tests, scripts) it falls back to the ``USER_OBJECTIVE`` environment
    variable, so single-user behaviour is unchanged.
  * Work handed to a thread pool must be wrapped with :func:`in_run_context`;
    pool threads do not inherit the caller's context on their own.
"""
from __future__ import annotations

import contextvars
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

_T = TypeVar("_T")

_objective: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "dsa_run_objective", default=None
)


@contextmanager
def objective_scope(text: str | None) -> Iterator[None]:
    """Make ``text`` this thread's objective for the duration of the block."""
    token = _objective.set((text or "").strip() or None)
    try:
        yield
    finally:
        _objective.reset(token)


def current_objective() -> str:
    """This run's objective, else the ``USER_OBJECTIVE`` env var, else ``""``."""
    value = _objective.get()
    if value is not None:
        return value
    return os.environ.get("USER_OBJECTIVE", "").strip()


def in_run_context(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Wrap ``fn`` so it runs inside a copy of the caller's current context.

    Call this in the submitting thread, before handing ``fn`` to an executor.
    Each call runs in its own copy: one ``Context`` object cannot be entered by
    several threads at once, and an executor runs the same wrapper on many.
    """
    ctx = contextvars.copy_context()

    def _runner(*args: object, **kwargs: object) -> _T:
        return ctx.copy().run(fn, *args, **kwargs)

    return _runner
