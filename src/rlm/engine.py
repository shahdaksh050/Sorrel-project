"""Recursive inference layer — task decomposition and context offloading."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

#: Bounded worker pool for parallel sub-task invocation in
#: decompose_and_invoke(). These are I/O-bound HTTP calls, so threads are
#: correct and the GIL is irrelevant; bounded by policy (not hardware) so a
#: wide decomposition doesn't slam the provider's own rate limits.
_MAX_DECOMPOSE_WORKERS = 6

# ---------------------------------------------------------------------------
# P3.1 — Token / cost accounting
# ---------------------------------------------------------------------------
#
# NOTE ON SCOPE: LLMClient._dispatch/_call_anthropic/_call_openai_compat
# (src/core/controller.py) currently discard the provider SDK's `response`
# object after pulling out the text — `.usage` never survives to the dict
# LLMClient.call() returns, and LLMClient.call() is the `llm_callable` this
# engine is built with. Wiring the provider's real token counts through
# requires editing controller.py, which is out of scope for this change (a
# concurrent edit is in flight there). So this engine reads usage from an
# optional, documented convention instead: a callable *may* attach usage
# info to its returned dict under one of `_USAGE_RESPONSE_KEYS`, e.g.
# ``{"...": ..., "_rlm_usage": {"prompt_tokens": 123, "completion_tokens": 45,
# "provider": "anthropic"}}``. Until LLMClient.call() is updated to attach
# that, usage_summary() will correctly report zeros — this is plumbing for a
# later pass, not a claim that accounting is live end-to-end today.

#: Keys checked (in order) on a callable's response dict for a usage payload.
_USAGE_RESPONSE_KEYS = ("_rlm_usage", "_usage", "usage")

#: Field name aliases accepted inside the usage payload — different SDKs
#: (OpenAI vs Anthropic vs Gemini) name these differently.
_PROMPT_TOKEN_KEYS = ("prompt_tokens", "input_tokens")
_COMPLETION_TOKEN_KEYS = ("completion_tokens", "output_tokens")

#: Approximate USD per 1,000 tokens, prompt/completion, keyed by provider.
#: These are rough, hand-maintained figures for budget-tracking purposes
#: only — not billing-accurate. Update as provider pricing changes.
COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "openai": {"prompt": 0.0025, "completion": 0.010},       # gpt-4o-ish
    "anthropic": {"prompt": 0.003, "completion": 0.015},      # claude-sonnet-ish
    "gemini": {"prompt": 0.00015, "completion": 0.0006},      # gemini-flash-ish
    "openrouter": {"prompt": 0.003, "completion": 0.015},     # varies by model; sonnet-ish default
    "nvidia": {"prompt": 0.0002, "completion": 0.0002},
    "local": {"prompt": 0.0, "completion": 0.0},
    "ollama": {"prompt": 0.0, "completion": 0.0},
    "unknown": {"prompt": 0.0, "completion": 0.0},
}


def _extract_usage(response: Any) -> tuple[int, int, str | None]:
    """Best-effort extraction of (prompt_tokens, completion_tokens, provider)
    from a callable's response, using the ``_USAGE_RESPONSE_KEYS`` convention.

    Returns (0, 0, None) when no usage payload is present — which is the
    normal case today (see module note above).
    """
    if not isinstance(response, dict):
        return 0, 0, None
    payload: Any = None
    for key in _USAGE_RESPONSE_KEYS:
        if key in response and isinstance(response[key], dict):
            payload = response[key]
            break
    if payload is None:
        return 0, 0, None

    def _first_int(keys: tuple[str, ...]) -> int:
        for k in keys:
            v = payload.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    prompt_tokens = _first_int(_PROMPT_TOKEN_KEYS)
    completion_tokens = _first_int(_COMPLETION_TOKEN_KEYS)
    provider = payload.get("provider")
    return prompt_tokens, completion_tokens, provider if isinstance(provider, str) else None


# ---------------------------------------------------------------------------
# REPL Environment — external state store (RLM paradigm)
# ---------------------------------------------------------------------------

class REPLEnvironment:
    """
    Persistent key-value store representing the external REPL environment
    from the recursive-decomposition paradigm.

    The controller writes sub-task context here so each recursive invocation
    can read prior results without inflating the main LLM context window.
    """

    def __init__(self) -> None:
        self.variables: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def summary(self) -> str:
        if not self.variables:
            return "(empty)"
        lines = [f"  {k}: {str(v)[:80]}" for k, v in self.variables.items()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RLMSubTask — unit of decomposed work for Stage 6
# ---------------------------------------------------------------------------

@dataclass
class RLMSubTask:
    """A single decomposed reasoning unit passed to decompose_and_invoke()."""

    task_id: str
    description: str
    context: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reasoning trace entry
# ---------------------------------------------------------------------------

@dataclass
class _TraceEntry:
    iteration: int
    depth: int
    stage: str
    user_prompt_snippet: str
    response_snippet: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# RLM Engine
# ---------------------------------------------------------------------------

class RLMEngine:
    """
    Recursive inference engine.

    Wraps a provider-agnostic LLM callable and adds:
      - Depth-bounded recursive invocation (max_depth guard)
      - External REPL environment for cross-call state sharing
      - Full reasoning trace for post-run inspection
      - Sub-task decomposition (Stage 6)

    The engine never imports memory, tools, or the controller — it only
    depends on the injected llm_callable.
    """

    def __init__(
        self,
        llm_callable: Callable[[str, str], dict[str, Any]],
        system_prompt: str,
        max_depth: int = 5,
        default_provider: str = "unknown",
    ) -> None:
        self._llm = llm_callable
        self._system_prompt = system_prompt
        self.max_depth = max_depth
        self.repl_env = REPLEnvironment()
        self._trace: list[_TraceEntry] = []
        self._iteration: int = 0
        #: Provider used for cost lookup when a call's usage payload doesn't
        #: name one (see module note on P3.1 above).
        self._default_provider = default_provider
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost_usd: float = 0.0
        #: Guards the running totals above — decompose_and_invoke() may
        #: accumulate them from multiple worker threads concurrently.
        self._usage_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_iteration(self, iteration: int) -> None:
        """Tell the engine which outer reasoning cycle we are in."""
        self._iteration = iteration

    @property
    def trace(self) -> list[_TraceEntry]:
        """Read-only copy of the reasoning trace (one entry per LLM call)."""
        return list(self._trace)

    def invoke(
        self,
        user_prompt: str,
        depth: int = 0,
        stage: str = "",
    ) -> dict[str, Any]:
        """
        Call the LLM with the injected system prompt and the given user prompt.

        Args:
            user_prompt: The fully-formed user message.
            depth:       Current recursion depth (0 = top-level call).
            stage:       Human-readable label for the reasoning trace.

        Returns:
            Parsed JSON dict from the LLM, or an error dict on failure.
        """
        if depth > self.max_depth:
            raise RecursionError(
                f"RLM max recursion depth ({self.max_depth}) exceeded at stage '{stage}'."
            )

        t0 = time.perf_counter()
        response = self._llm(self._system_prompt, user_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        prompt_tokens, completion_tokens, provider = _extract_usage(response)
        rates = COST_PER_1K_TOKENS.get(
            provider or self._default_provider, COST_PER_1K_TOKENS["unknown"]
        )
        cost_usd = (prompt_tokens / 1000) * rates["prompt"] + (
            completion_tokens / 1000
        ) * rates["completion"]

        with self._usage_lock:
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._total_cost_usd += cost_usd

        self._trace.append(
            _TraceEntry(
                iteration=self._iteration,
                depth=depth,
                stage=stage,
                user_prompt_snippet=user_prompt[:120],
                response_snippet=str(response)[:120],
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
            )
        )
        return response

    def usage_summary(self) -> dict[str, Any]:
        """
        Running token/cost totals across every invoke() call made so far.

        Cost is a rough estimate from ``COST_PER_1K_TOKENS`` (approximate,
        hand-maintained per-provider rates) — not a billing-accurate figure.
        Token counts are 0 until the injected ``llm_callable`` attaches a
        usage payload to its response dict (see the P3.1 module note above);
        that wiring lives in ``LLMClient.call`` (src/core/controller.py) and
        is not part of this change.
        """
        with self._usage_lock:
            return {
                "total_prompt_tokens": self._total_prompt_tokens,
                "total_completion_tokens": self._total_completion_tokens,
                "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
                "estimated_cost_usd": round(self._total_cost_usd, 6),
                "is_estimate": True,
                "call_count": len(self._trace),
            }

    def decompose_and_invoke(
        self,
        sub_tasks: list[RLMSubTask],
        prompt_builder: Callable[[RLMSubTask], str],
        depth: int = 1,
        max_total_tokens: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Stage 6: run one LLM call per sub-task and aggregate results.

        Sub-tasks are independent — prompt_builder only ever reads a
        sub-task's own task_id/description/context, never a sibling's
        REPL-stored result — so they run concurrently on a small bounded
        thread pool instead of one LLM round trip at a time. Each
        sub-task's context is stored in the REPL environment under
        ``subtask_ctx_<task_id>`` before any call starts, and every
        result under ``subtask_result_<task_id>`` once all calls finish
        (written back in ``sub_tasks`` order, not completion order, so the
        REPL env's final state is deterministic regardless of which
        thread happened to finish first).

        Args:
            sub_tasks:        List of RLMSubTask instances to process.
            prompt_builder:   Callable that turns an RLMSubTask into a prompt string.
            depth:            Recursion depth to pass to invoke().
            max_total_tokens: Optional running-total token budget (prompt +
                completion, across this engine's whole lifetime, per
                usage_summary()). Once exceeded, remaining sub-tasks are
                skipped and the loop ends gracefully — it returns results
                for whatever sub-tasks it already completed, the same as it
                would on normal exhaustion, rather than raising. Enforcing
                this requires checking the budget *between* calls, so a
                budgeted run processes sub-tasks sequentially instead of
                on the concurrent thread pool (bounded by policy anyway —
                see _MAX_DECOMPOSE_WORKERS — so this trades some latency
                for the ability to stop mid-decomposition).

        Returns:
            Dict mapping task_id -> LLM response dict, for the sub-tasks
            actually run. Sub-tasks skipped by the token budget or that
            raised (a transient LLM/provider failure on that one call) are
            both simply absent from the returned dict — one bad sub-task
            must not discard the others that already succeeded, same
            graceful-degradation rationale as the caller's own try/except
            around this whole method.
        """
        for sub_task in sub_tasks:
            self.repl_env.set(f"subtask_ctx_{sub_task.task_id}", sub_task.context)

        def _run(sub_task: RLMSubTask) -> dict[str, Any]:
            prompt = prompt_builder(sub_task)
            return self.invoke(
                prompt, depth=depth, stage=f"stage6:decompose:{sub_task.task_id}"
            )

        def _run_safe(sub_task: RLMSubTask) -> tuple[RLMSubTask, dict[str, Any] | None, Exception | None]:
            try:
                return sub_task, _run(sub_task), None
            except Exception as exc:  # per-sub-task isolation, see docstring
                return sub_task, None, exc

        def _budget_exceeded() -> bool:
            return (
                max_total_tokens is not None
                and self.usage_summary()["total_tokens"] >= max_total_tokens
            )

        completed: list[tuple[RLMSubTask, dict[str, Any]]] = []

        def _record(sub_task: RLMSubTask, response: dict[str, Any] | None, exc: Exception | None) -> None:
            if exc is not None:
                console.print(f"[yellow]  ⚠ sub-task '{sub_task.task_id}' failed: {exc}[/]")
                return
            assert response is not None
            completed.append((sub_task, response))

        if max_total_tokens is not None or len(sub_tasks) <= 1:
            for sub_task in sub_tasks:
                if _budget_exceeded():
                    break
                _record(*_run_safe(sub_task))
        else:
            with ThreadPoolExecutor(max_workers=min(_MAX_DECOMPOSE_WORKERS, len(sub_tasks))) as pool:
                for sub_task, response, exc in pool.map(_run_safe, sub_tasks):
                    _record(sub_task, response, exc)

        results: dict[str, dict[str, Any]] = {}
        for sub_task, response in completed:
            results[sub_task.task_id] = response
            self.repl_env.set(f"subtask_result_{sub_task.task_id}", response)

        return results

    def print_reasoning_trace(self) -> None:
        """Render the full reasoning trace as a Rich table."""
        if not self._trace:
            console.print("[dim]No reasoning trace recorded.[/]")
            return

        table = Table(
            title="RLM Reasoning Trace",
            show_lines=True,
            highlight=True,
        )
        table.add_column("Iter", style="cyan", width=4)
        table.add_column("Depth", width=5)
        table.add_column("Stage", style="magenta", width=24)
        table.add_column("Latency (ms)", width=12)
        table.add_column("Response snippet", style="dim")

        for entry in self._trace:
            table.add_row(
                str(entry.iteration),
                str(entry.depth),
                entry.stage,
                f"{entry.latency_ms:.0f}",
                entry.response_snippet,
            )

        console.print(table)
