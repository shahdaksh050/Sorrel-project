"""Lightweight inference manager with tracing and token/cost accounting."""
from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

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
    Provider-agnostic inference manager.

    Wraps a provider-agnostic LLM callable and adds:
      - Depth guard retained for compatibility with existing callers
      - Full reasoning trace for post-run inspection

    The engine never imports memory, tools, or the controller — it only
    depends on the injected llm_callable.
    """

    def __init__(
        self,
        llm_callable: Callable[[str, str], dict[str, Any]],
        system_prompt: str,
        max_depth: int = 5,
        default_provider: str = "unknown",
        base_max_tokens: int = 4096,
    ) -> None:
        self._llm = llm_callable
        # Retained in the public constructor for configuration compatibility.
        _ = base_max_tokens
        #: Whether the callable accepts per-call `max_tokens` / `reasoning_effort`
        #: (LLMClient.call does; a plain two-argument callable does not).
        try:
            params = inspect.signature(llm_callable).parameters
            self._llm_takes_overrides = "max_tokens" in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            self._llm_takes_overrides = False
        self._system_prompt = system_prompt
        self.max_depth = max_depth
        self._trace: list[_TraceEntry] = []
        self._iteration: int = 0
        #: Provider used for cost lookup when a call's usage payload doesn't
        #: name one (see module note on P3.1 above).
        self._default_provider = default_provider
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost_usd: float = 0.0
        #: Guards totals because callers may record auxiliary LLM usage from
        #: worker threads (for example, chart design).
        self._usage_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_iteration(self, iteration: int) -> None:
        """Tell the engine which outer reasoning cycle we are in."""
        self._iteration = iteration

    def set_system_prompt(self, system_prompt: str) -> None:
        """Swap the injected system prompt (its tool block was refreshed)."""
        self._system_prompt = system_prompt

    @property
    def trace(self) -> list[_TraceEntry]:
        """Read-only copy of the reasoning trace (one entry per LLM call)."""
        return list(self._trace)

    def _record_usage(
        self, prompt_tokens: int, completion_tokens: int, provider: str | None
    ) -> tuple[int, int, float]:
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
        return prompt_tokens, completion_tokens, cost_usd

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        """Count a token-usage payload (``{"prompt_tokens", "completion_tokens",
        "provider"}``, e.g. a reply's ``_rlm_usage``) against this run's totals —
        for LLM calls made outside invoke() (chart design) or failed ones."""
        self._record_usage(*_extract_usage({"_rlm_usage": usage}))

    def invoke(
        self,
        user_prompt: str,
        depth: int = 0,
        stage: str = "",
        *,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Call the LLM with the injected system prompt and the given user prompt.

        Args:
            user_prompt: The fully-formed user message.
            depth:       Current recursion depth (0 = top-level call).
            stage:       Human-readable label for the reasoning trace.
            system_prompt: Replaces the injected one for this call only (a
                         small sub-task needs no tool catalog).

        Returns:
            Parsed JSON dict from the LLM, or an error dict on failure.
        """
        if depth > self.max_depth:
            raise RecursionError(
                f"RLM max recursion depth ({self.max_depth}) exceeded at stage '{stage}'."
            )

        t0 = time.perf_counter()
        try:
            overrides: dict[str, Any] = {}
            if self._llm_takes_overrides:
                if max_tokens is not None:
                    overrides["max_tokens"] = max_tokens
                if reasoning_effort is not None:
                    overrides["reasoning_effort"] = reasoning_effort
            response = self._llm(system_prompt or self._system_prompt, user_prompt, **overrides)
        except Exception as exc:
            # A call that failed after the provider replied (parse failure)
            # still spent tokens — count them against the budget.
            self.record_usage(getattr(exc, "rlm_usage", None))
            raise
        latency_ms = (time.perf_counter() - t0) * 1000

        prompt_tokens, completion_tokens, cost_usd = self._record_usage(*_extract_usage(response))

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

    def print_reasoning_trace(self) -> None:
        """Render the full reasoning trace as a Rich table."""
        if not self._trace:
            console.print("[dim]No reasoning trace recorded.[/]")
            return

        table = Table(
            title="Inference Trace",
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
