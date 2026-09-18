"""
Agent Controller — Orchestration and Reasoning Layer.

Implements the full 7-stage agent workflow:

  Stage 1 — Dataset Ingestion          load_dataset()
  Stage 2 — Initial Reasoning Phase    analyze() → first RLM invoke
  Stage 3 — Tool Selection & Execution analyze() → execution loop
  Stage 4 — Result Interpretation      analyze() → iteration prompt
  Stage 5 — Iterative Refinement       analyze() → loop until complete
  Stage 6 — RLM Workflow Management    _run_rlm_decomposition()
  Stage 7 — Report Generation          _generate_final_report()

ARCHITECTURAL BOUNDARY:
  - This file handles PLANNING and ORCHESTRATION only.
  - Raw data never enters LLM prompts — only summaries via PromptManager.
  - Tools are always called through ToolRegistry, never directly.
  - The LLM is always called through RLMEngine, never directly.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.core.agenda import Question, build_agenda, coverage_report
from src.core.coercion import coerce_types
from src.core.dashboard import build_dashboard, dashboard_to_json
from src.core.degradations import collect_degradations
from src.core.domains import infer_domains
from src.core.findings import Finding
from src.core.io import read_any
from src.core.memory import AnalysisStep, DatasetMetadata, MemorySystem, ToolResult
from src.core.profiler import DatasetProfile, profile_dataframe
from src.core.prompt_manager import PromptManager
from src.rlm.engine import RLMEngine, RLMSubTask

console = Console()


def _read_dataframe(file_path: str) -> pd.DataFrame:
    """Load a CSV/TSV/Excel dataset for dashboard generation."""
    df, _report = read_any(file_path)
    return df

# Max retries before abandoning a failed step
MAX_STEP_RETRIES = 2

# ---------------------------------------------------------------------------
# Verbatim-metric validation (P0.7) — SYSTEM_PROMPT_CORE tells the LLM to
# cite only numbers that appear in tool results, but nothing checked that
# rule. These turn it into a mechanism: any numeric literal the LLM's
# synthesis states that cannot be traced back to an actual tool result is
# flagged, not trusted silently.
# ---------------------------------------------------------------------------

#: Matches numeric literals (integers, decimals, negatives, comma-formatted) in free text.
_NUMBER_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

#: Single-digit integers are almost always counts ("3 models", "top 5
#: features") rather than cited metrics, and are cheap to satisfy by
#: coincidence — excluding them keeps the flag meaningful.
_UNVERIFIABLE_SKIP_ABS_INT = 9

# ---------------------------------------------------------------------------
# Attribution-aware verification — a number existing ANYWHERE in the
# pooled tool output isn't enough: "cleaning handled 503 missing values"
# verified as long as 503 appeared in ANY tool's output, even when the real
# clean_data result said "cleaned 0 missing values" and 503 was actually
# ingest_dataset's row count. When an insight/recommendation sentence names
# a specific kind of analysis, its numbers must come from the tool(s) that
# analysis maps to, not merely from the run somewhere.
# ---------------------------------------------------------------------------

#: Keyword -> the tool name(s) whose own output pool a sentence containing
#: that keyword must be checked against. Each key is a regex matched with a
#: leading word boundary against the lower-cased claim text (so "chi-square"
#: matches but "which" doesn't, and a prefix like "correlat" still covers
#: "correlation"/"correlated"); a sentence can match several keywords/tools
#: at once, in which case the union of their pools is used. Deliberately
#: specific — a generic word ("test", "segment" alone) would pull ordinary
#: prose into the stricter per-tool check and flag correct numbers.
_KEYWORD_TOOL_MAP: dict[str, tuple[str, ...]] = {
    r"clean": ("clean_data",),
    r"imput": ("clean_data",),
    r"outlier": ("detect_outliers",),
    r"correlat": ("correlation_analysis",),
    r"pca\b": ("dimensionality_analysis",),
    r"principal component": ("dimensionality_analysis",),
    r"dimensionality": ("dimensionality_analysis",),
    r"cluster": ("cluster_data",),
    r"silhouette": ("cluster_data",),
    r"concentrat": ("concentration_analysis",),
    r"gini": ("concentration_analysis",),
    r"segments?\b": ("segment_comparison", "cluster_data"),
    r"trend": ("time_series_analysis", "change_analysis"),
    r"seasonal": ("time_series_analysis", "change_analysis"),
    r"accuracy": ("train_model", "evaluate_model"),
    r"f1\b": ("train_model", "evaluate_model"),
    r"auc\b": ("train_model", "evaluate_model"),
    r"r2\b": ("train_model", "evaluate_model"),
    r"statistical test": ("select_statistical_test", "segment_comparison"),
    r"p-?value": ("select_statistical_test", "segment_comparison"),
    r"anova": ("select_statistical_test",),
    r"chi-?squared?\b": ("select_statistical_test",),
    r"mann-whitney": ("select_statistical_test",),
    r"kruskal": ("select_statistical_test",),
}
_KEYWORD_TOOL_PATTERNS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = tuple(
    (re.compile(r"\b" + kw), tools) for kw, tools in _KEYWORD_TOOL_MAP.items()
)

#: Appended to a reasoning prompt when the first reply was unusable
#: (non-JSON, truncated at max_tokens, or empty) — see analyze().
_COMPACT_RETRY_NOTE = (
    "\n\n## Retry Note\nYour previous reply could not be used (it was not "
    "complete, valid JSON — possibly cut off by the output length limit). "
    "Reply again with ONLY the JSON object: at most 8 steps, one short "
    "sentence per rationale, no markdown fences, no text outside the JSON.\n"
)


def _canon_number(value: Any, precision: int = 4) -> str:
    """Normalise a number to a fixed-precision canonical string for
    set-membership comparison, so '0.8', '0.80' and 0.7999999999999999
    (float round-trip noise) all match."""
    try:
        if isinstance(value, str):
            value = value.replace(",", "")
        f = float(value)
        if f.is_integer() and abs(f) < 1e15:
            return str(int(f))
        return f"{round(f, precision):g}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


#: Finding headlines (7.1) round for readability (e.g. "r=0.81") while the
#: tool output they're traced back to often carries more decimals
#: ("correlation=0.8109") — the verified pool indexes several roundings of
#: each source number so a claim's own (looser) precision still matches
#: without weakening the check itself (a genuinely wrong number still fails
#: at every precision).
_CANON_PRECISIONS = (4, 3, 2, 1, 0)


def _collect_numbers(obj: Any, into: set[str]) -> None:
    """Recursively flatten every numeric leaf/substring in a JSON-like
    structure into canonical form, at several roundings (see
    `_CANON_PRECISIONS`)."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        for p in _CANON_PRECISIONS:
            into.add(_canon_number(obj, p))
        # Finding headlines and narrative text display rates/fractions as
        # percentages (e.g. evidence["level_value"]=0.08596 renders as
        # "8.6%"), while the raw tool output stores the fraction — a unit
        # mismatch, not a rounding one, since _CANON_PRECISIONS already
        # covers rounding (that's why "r=0.81" verifies against a stored
        # 0.8109 but "8.6%" didn't verify against a stored 0.08596). This
        # function sees only values, not keys, so it can't check the
        # evidence dict's own `is_rate` flag — a numeric range guard is the
        # generic equivalent: only fraction-range values get a *100 form,
        # so this can't turn an unrelated large number into a false
        # verification.
        if -1.0 <= obj <= 1.0:
            for p in _CANON_PRECISIONS:
                into.add(_canon_number(obj * 100, p))
    elif isinstance(obj, str):
        for match in _NUMBER_RE.finditer(obj):
            val = match.group().replace(",", "")
            for p in _CANON_PRECISIONS:
                into.add(_canon_number(val, p))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, into)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, into)

# Target auto-detection confidence thresholds
_AUTODETECT_HIGH = 0.75   # proceed autonomously above this
_AUTODETECT_LOW  = 0.40   # prompt user (CLI) or best-guess (UI) above this


# ---------------------------------------------------------------------------
# LLM Client — thin, provider-agnostic wrapper
# ---------------------------------------------------------------------------

#: Fallback model per provider, used only when LLM_MODEL is unset.
_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-flash-latest",
    "openrouter": "openai/gpt-4o",
    "nvidia": "openai/gpt-oss-120b",
    "local": "llama3.1",
    "ollama": "llama3.1",
}


class LLMClient:
    """
    Thin wrapper around LLM provider APIs.

    Supports OpenAI, Anthropic, Google Gemini, OpenRouter, NVIDIA NIM, and
    any offline/self-hosted OpenAI-compatible server (Ollama, LM Studio,
    vLLM, llama.cpp server, ...) via provider="local". Credentials come
    from environment variables only — never hardcoded.
    """

    def __init__(self) -> None:
        self.provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model: str = os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(self.provider, "gpt-4o")
        self.temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        self.timeout: float = float(os.getenv("LLM_TIMEOUT", "120"))
        # Built lazily on first call and reused — the SDK clients are
        # long-lived and thread-safe, and re-pooling per call was costing
        # every invocation a fresh TCP+TLS handshake (~100-300ms).
        self._client: Any = None
        # P3.1 — usage from the most recent _dispatch() call, if the
        # provider branch captured one. `call()` attaches it to the parsed
        # response under `_rlm_usage` so RLMEngine's `_extract_usage` (which
        # already looks for that key) reports real token counts instead of
        # the zeros its own docstring warns about. Not every branch sets
        # this (the NVIDIA streaming path doesn't request usage in-stream —
        # left as a smaller follow-up), so it stays best-effort by design.
        #
        # Round 8 hardening — this LLMClient instance is shared across RLM
        # worker threads (RLMEngine.decompose_and_invoke runs concurrent
        # `call()`s on a ThreadPoolExecutor for Stage 6 decomposition), so a
        # single `self._last_usage` instance attribute was a data race: one
        # thread's dispatch could overwrite it between another thread's
        # dispatch and its read, attaching the wrong call's usage (or
        # nothing) to a response. threading.local() gives each thread its
        # own slot, so `call()` always reads back exactly the usage its own
        # `_dispatch()` just set, however many threads are calling this
        # instance concurrently.
        self._usage_local = threading.local()

    def ping(self) -> tuple[bool, str]:
        """
        Cheap connectivity + model-validity check (a small token budget).

        Returns (True, "") on success, (False, "<ExceptionType>: <detail>")
        on any failure — so callers can fail fast with the real reason
        instead of running a whole analysis on the deterministic fallback.
        """
        saved = self.max_tokens
        # Reasoning/"thinking" models (Gemini 3.x, NVIDIA gpt-oss, o-series-
        # style models) spend part of the budget on hidden reasoning tokens
        # before any visible output — 16 was enough for plain chat models
        # but silently starved thinking models into empty content. 200 is
        # still a negligible cost for a connectivity check.
        self.max_tokens = 200
        try:
            self._dispatch("You are a connectivity check. Reply with OK.", "Reply with OK.")
            return True, ""
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        finally:
            self.max_tokens = saved

    def call(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Call the configured LLM and return the parsed JSON response.

        Raises:
            ValueError: If the response cannot be parsed as JSON.
        """
        self._usage_local.value = None
        self._usage_local.truncated = False
        raw = self._dispatch(system_prompt, user_prompt)
        try:
            parsed = self._parse_json(raw)
        except ValueError as exc:
            # A reply cut off at max_tokens is the usual reason a long plan
            # fails to parse — say so, instead of only "non-JSON", so the
            # fix (raise LLM_MAX_TOKENS / lower reasoning effort) is obvious.
            if getattr(self._usage_local, "truncated", False):
                raise ValueError(
                    f"LLM reply was truncated at max_tokens={self.max_tokens} "
                    f"(finish_reason=length) and could not be repaired: {exc}"
                ) from exc
            raise
        if not isinstance(parsed, dict):
            # json.loads happily returns a list/str/number; every caller
            # (and the `_rlm_usage` attach below) needs an object.
            raise ValueError(
                f"LLM returned JSON that is not an object ({type(parsed).__name__})."
            )
        usage = getattr(self._usage_local, "value", None)
        if usage:
            parsed["_rlm_usage"] = usage
        return parsed

    def _dispatch(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        return self._call_openai_compat(system_prompt, user_prompt)

    def _call_openai_compat(self, system_prompt: str, user_prompt: str) -> str:
        """
        OpenAI, OpenRouter, NVIDIA NIM, Google Gemini, and any offline/
        self-hosted OpenAI-compatible server all use the OpenAI SDK — only
        the base_url and api_key differ.
        """
        from openai import OpenAI
        if self.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            base_url: str | None = "https://openrouter.ai/api/v1"
            extra_headers: dict[str, str] = {
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/agentic-data-analysis"),
                "X-Title": "Agentic Data Analysis",
            }
        elif self.provider == "nvidia":
            api_key = os.getenv("NVIDIA_API_KEY", "")
            base_url = "https://integrate.api.nvidia.com/v1"
            extra_headers = {}
        elif self.provider == "gemini":
            # Google's OpenAI-compatible endpoint — no separate SDK needed.
            # https://ai.google.dev/gemini-api/docs/openai
            api_key = os.getenv("GEMINI_API_KEY", "")
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            extra_headers = {}
        elif self.provider in ("local", "ollama"):
            # Offline / self-hosted OpenAI-compatible server: Ollama, LM
            # Studio, vLLM, llama.cpp server, text-generation-webui, etc.
            # No cloud API key required — most local servers accept any
            # non-empty string, so default to a placeholder.
            api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed")
            base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
            extra_headers = {}
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")
            base_url = None
            extra_headers = {}
        if not api_key:
            _key_names = {
                "openrouter": "OPENROUTER_API_KEY",
                "nvidia": "NVIDIA_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }
            raise ValueError(
                f"No API key set for provider '{self.provider}'. "
                f"Set {_key_names.get(self.provider, 'OPENAI_API_KEY')}."
            )
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.timeout,
            "max_retries": 2,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers
        if self._client is None:
            self._client = OpenAI(**client_kwargs)
        client = self._client
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.provider in ("openai", "gemini"):
            # Both support the OpenAI JSON-mode contract; local/offline
            # servers vary too widely, so JSON there relies on _parse_json's
            # fence-stripping and repair fallback instead.
            create_kwargs["response_format"] = {"type": "json_object"}
        if self.provider == "gemini":
            # Gemini 2.5+/3.x "thinking" models spend a large, variable, and
            # otherwise invisible share of max_tokens on hidden reasoning
            # before writing any visible answer. Left uncapped, a normal
            # max_tokens budget can be entirely consumed by thinking, so the
            # JSON answer gets truncated or never starts at all.
            effort = os.getenv("GEMINI_REASONING_EFFORT", "").strip().lower()
            if not effort and any(k in self.model.lower() for k in ("thinking", "2.5", "3.")):
                effort = "low"
            if effort and effort not in ("none", "off"):
                create_kwargs["reasoning_effort"] = effort
        if self.provider == "openrouter":
            # Same failure mode as Gemini above, on OpenRouter's many
            # reasoning-capable (often free) models: hidden reasoning tokens
            # count against max_tokens, so a long Form 1 plan gets truncated
            # or never starts. OpenRouter's unified `reasoning` parameter caps
            # that; models without reasoning ignore it. "none"/"off" omits it.
            effort = os.getenv("OPENROUTER_REASONING_EFFORT", "low").strip().lower()
            if effort not in ("", "none", "off"):
                create_kwargs["extra_body"] = {"reasoning": {"effort": effort}}
        if self.provider == "nvidia":
            # NVIDIA NIM requires streaming; gpt-oss-120b also emits
            # reasoning_content chunks (chain-of-thought) before the answer.
            create_kwargs["stream"] = True
            create_kwargs["top_p"] = 1
            stream = client.chat.completions.create(**create_kwargs)
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    err = getattr(chunk, "error", None)
                    if err:
                        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                        raise ValueError(f"nvidia error for model '{self.model}': {msg}")
                    continue
                delta = chunk.choices[0].delta
                # gpt-oss-120b is a reasoning model: the chain-of-thought
                # arrives in reasoning_content; the final answer arrives in
                # content. Collect both — content is preferred; if it ends up
                # empty (some reasoning-only models), fall back to reasoning.
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning is not None:
                    reasoning_parts.append(reasoning)
                text = getattr(delta, "content", None)
                if text is not None:
                    content_parts.append(text)
            content = "".join(content_parts).strip() or "".join(reasoning_parts).strip()
            if not content:
                raise ValueError(
                    f"NVIDIA returned empty content for model '{self.model}'. "
                    "Check that the model ID is correct and your account has access."
                )
            return content

        try:
            resp = client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            # If Google or any endpoint rejects reasoning_effort for this model, retry without it
            if "reasoning_effort" in str(exc) and "reasoning_effort" in create_kwargs:
                create_kwargs.pop("reasoning_effort", None)
                resp = client.chat.completions.create(**create_kwargs)
            else:
                raise
        # OpenRouter can return HTTP 200 with an error body instead of raising
        # (invalid model slug, moderation, no credits). The SDK then yields
        # choices=None — surface the real message instead of a TypeError.
        err = getattr(resp, "error", None)
        if err:
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise ValueError(
                f"{self.provider} error for model '{self.model}': {msg}"
            )
        if not getattr(resp, "choices", None):
            raise ValueError(
                f"{self.provider} returned no completion for model '{self.model}' — "
                f"the model ID may be invalid, unavailable, or blocked by your "
                f"account's data policy."
            )
        content = resp.choices[0].message.content
        finish_reason = getattr(resp.choices[0], "finish_reason", None)
        self._usage_local.truncated = finish_reason == "length"
        if not content or not str(content).strip():
            # Reasoning models (many OpenRouter free models) can spend the
            # whole max_tokens budget on hidden reasoning and return no
            # visible answer at all — name that case instead of a bare
            # "None content", since the fix is budget/effort, not the prompt.
            if finish_reason == "length":
                raise ValueError(
                    f"{self.provider} model '{self.model}' used the whole "
                    f"max_tokens={self.max_tokens} budget without producing an answer "
                    "(likely hidden reasoning) — raise LLM_MAX_TOKENS or lower "
                    "OPENROUTER_REASONING_EFFORT."
                )
            raise ValueError(
                f"LLM returned empty content (finish_reason={finish_reason!r})."
            )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            # Thread-local (see __init__) — this dispatch may be running on
            # one of several concurrent RLM worker threads.
            self._usage_local.value = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "provider": self.provider,
            }
        return str(content)

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        import anthropic
        from anthropic.types import TextBlock

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key set for provider 'anthropic'. Set ANTHROPIC_API_KEY.")
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout, max_retries=2)
        client = self._client
        # P1.6(b) — the tool-description/system block is byte-identical
        # across all ~15 calls in a run; mark it for prompt caching so it's
        # billed once instead of on every iteration.
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
        )
        usage = getattr(msg, "usage", None)
        if usage is not None:
            # Thread-local (see __init__) — this dispatch may be running on
            # one of several concurrent RLM worker threads.
            self._usage_local.value = {
                "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
                "provider": "anthropic",
            }
        for block in msg.content:
            if isinstance(block, TextBlock):
                return block.text
        raise ValueError("Anthropic response contained no text block.")

    @staticmethod
    def _loads_lenient(text: str) -> dict[str, Any]:
        """
        json.loads with a fallback to strict=False.

        Small/free-tier models routinely emit literal, unescaped newlines
        (and other control characters) inside JSON string values — e.g. a
        multi-line "reasoning" sentence — instead of the required `\\n`
        escape. That is otherwise a *complete, well-formed* response (every
        brace balanced, every field present); strict json.loads rejects it
        anyway with "Invalid control character", which used to fall all the
        way through to the truncation-repair/extraction paths below and
        often still fail there too, discarding a perfectly good plan for a
        cosmetic escaping mistake. strict=False accepts raw control
        characters inside strings (the one thing wrong here) while still
        rejecting genuinely malformed JSON.
        """
        try:
            return cast(dict[str, Any], json.loads(text))
        except json.JSONDecodeError:
            return cast(dict[str, Any], json.loads(text, strict=False))

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        # Strip markdown fences if present
        for fence in ("```json", "```"):
            if fence in raw:
                raw = raw.split(fence)[1].split("```")[0]
                break

        cleaned = raw.strip()

        try:
            return LLMClient._loads_lenient(cleaned)
        except json.JSONDecodeError:
            pass

        # Try json-repair library if installed (handles all edge cases)
        try:
            from json_repair import repair_json
            candidate = repair_json(cleaned, return_objects=False)
            if candidate:
                return LLMClient._loads_lenient(candidate)
        except (ImportError, json.JSONDecodeError):
            pass

        # Manual repair: close open strings/structures and fix trailing : or ,
        repaired = LLMClient._repair_truncated_json(cleaned)
        try:
            return LLMClient._loads_lenient(repaired)
        except json.JSONDecodeError:
            pass

        # Prose wrapped around the object: small models very often answer
        # "Looking at the results, I think... {...}" instead of bare JSON,
        # which used to abort the whole iteration. Pull out the first
        # balanced {...} and try again — this is what makes a lightweight
        # model usable at all, and it costs nothing when the reply was
        # already clean.
        extracted = LLMClient._extract_json_object(cleaned)
        if extracted is not None:
            for candidate in (extracted, LLMClient._repair_truncated_json(extracted)):
                try:
                    return LLMClient._loads_lenient(candidate)
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"LLM returned non-JSON: {raw[:300]}")

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """
        First balanced {...} in `text`, or None.

        Brace counting is string-aware: a `{` or `}` inside a JSON string
        value (or escaped) must not change the depth, or a reply containing
        a brace in prose — or in an analysis rationale — truncates at the
        wrong place and produces something worse than no match.
        """
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        # Unbalanced: hand back the tail so the truncation repair can try.
        return text[start:]

    @staticmethod
    def _repair_truncated_json(s: str) -> str:
        """Close any open strings and bracket structures left by a truncated LLM response."""
        stack: list[str] = []
        in_string = False
        escape_next = False

        for ch in s:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ("{", "["):
                stack.append("}" if ch == "{" else "]")
            elif ch in ("}", "]"):
                if stack and stack[-1] == ch:
                    stack.pop()

        result = s

        # 1. Close any open string literal
        if in_string:
            result += '"'

        if not stack:
            return result

        # 2. Examine the last meaningful (non-whitespace) character to decide
        #    what padding is needed before the closing brackets.
        tail = result.rstrip()
        last_ch = tail[-1] if tail else ""

        if last_ch == ":":
            # Truncated right after a colon — value never started
            result = tail + " null"
        elif last_ch == ",":
            # Trailing comma — remove it so the structure closes cleanly
            result = tail[:-1]

        # 3. Close all open brackets/braces in innermost-first order
        result += "".join(reversed(stack))
        return result


# ---------------------------------------------------------------------------
# Tool Registry — discovers and maps all tools
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Registry of all available execution-layer tools.

    The LLM references tools by name; this class resolves them to
    callable BaseTool instances. Tools register themselves generically
    (register()) rather than the registry hardcoding an exhaustive import
    list — new tools (built-in or, in future, generated) plug in the same
    way the built-ins do.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Any] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        from src.tools.change_analysis import ChangeAnalysisTool
        from src.tools.clustering import ClusterDataTool
        from src.tools.cohort_analysis import CohortAnalysisTool
        from src.tools.concentration_analysis import ConcentrationAnalysisTool
        from src.tools.data_processing import (
            CleanDataTool,
            CorrelationAnalysisTool,
            DetectOutliersTool,
            IngestDatasetTool,
        )
        from src.tools.define_analysis_tool import DefineAnalysisToolTool
        from src.tools.dimensionality import DimensionalityAnalysisTool
        from src.tools.dynamic_code import DynamicCodeExecutionTool
        from src.tools.financial_analysis import FinancialAnalysisTool
        from src.tools.geospatial import GeospatialAnalysisTool
        from src.tools.ml_pipeline import EvaluateModelTool, TrainModelTool
        from src.tools.report_generator import GenerateReportTool
        from src.tools.segment_comparison import SegmentComparisonTool
        from src.tools.statistical_analysis import SelectStatisticalTestTool
        from src.tools.text_analysis import TextAnalysisTool
        from src.tools.time_series import TimeSeriesAnalysisTool
        from src.tools.visualization import GenerateVisualizationsTool
        from src.tools.workforce_analysis import WorkforceAnalysisTool

        for tool in (
            IngestDatasetTool(),
            CleanDataTool(),
            DetectOutliersTool(),
            CorrelationAnalysisTool(),
            SelectStatisticalTestTool(),
            TrainModelTool(),
            EvaluateModelTool(),
            ClusterDataTool(),
            GenerateVisualizationsTool(),
            GenerateReportTool(),
            TimeSeriesAnalysisTool(),
            TextAnalysisTool(),
            DimensionalityAnalysisTool(),
            GeospatialAnalysisTool(),
            DynamicCodeExecutionTool(),
            # Round 8 — LLM Sandbox mode: define_analysis_tool lets the agent
            # register a new, named, reusable tool at runtime (registration
            # itself happens in _maybe_register_generated_tool, called from
            # _execute_steps after this tool's own pure validate/smoke-test
            # step succeeds — see AGENTS.md's layer rule).
            DefineAnalysisToolTool(),
            FinancialAnalysisTool(),
            CohortAnalysisTool(),
            WorkforceAnalysisTool(),
            # 7.2 — insight library (segment comparison, concentration,
            # period-over-period change): the "why"/"what happened"
            # questions no prior tool answered directly.
            SegmentComparisonTool(),
            ConcentrationAnalysisTool(),
            ChangeAnalysisTool(),
        ):
            self.register(tool)

    def register(self, tool: Any) -> None:
        """Add (or replace) a tool in the registry, keyed by its `name`."""
        self._registry[tool.name] = tool

    def get(self, name: str) -> Any:
        if name not in self._registry:
            raise KeyError(
                f"Unknown tool '{name}'. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def has(self, name: str) -> bool:
        return name in self._registry

    def names(self) -> list[str]:
        return list(self._registry.keys())

    def get_all_descriptions(self) -> str:
        """Descriptions for every registered tool, gating aside. Used by
        offline scripts (validate/dry_run) that have no DatasetProfile."""
        return "\n\n".join(t.to_prompt_description() for t in self._registry.values())

    def candidate_tools(
        self,
        profile: Any | None,
        metadata: Any | None,
        use_ml: bool = True,
        use_llm: bool = True,
    ) -> list[Any]:
        """
        Tools relevant to this dataset, ranked by applies_to() score
        (highest first). A tool scoring 0.0 is excluded entirely — this
        IS the dynamic-selection mechanism: what the planner sees is
        already filtered to what fits the data's nature.

        `use_ml`/`use_llm` drop the tools that declare they need those
        capabilities (BaseTool.requires_ml / requires_llm). Filtering here
        rather than at plan time means a disabled capability is invisible
        everywhere at once: the planner never sees the tool, the
        deterministic plan never schedules it, and the prompt never
        describes it.
        """
        scored = [(t, t.applies_to(profile, metadata)) for t in self._registry.values()]
        relevant = [
            (t, s)
            for t, s in scored
            if s > 0.0
            and not (getattr(t, "requires_ml", False) and not use_ml)
            and not (getattr(t, "requires_llm", False) and not use_llm)
        ]
        relevant.sort(key=lambda ts: ts[1], reverse=True)
        return [t for t, _ in relevant]

    def get_candidate_descriptions(
        self,
        profile: Any | None,
        metadata: Any | None,
        use_ml: bool = True,
        use_llm: bool = True,
    ) -> str:
        tools = self.candidate_tools(profile, metadata, use_ml=use_ml, use_llm=use_llm)
        if not tools:
            return self.get_all_descriptions()
        return "\n\n".join(t.to_prompt_description() for t in tools)


# ---------------------------------------------------------------------------
# Agent Controller — the top-level orchestrator
# ---------------------------------------------------------------------------

class AgentController:
    """
    Top-level orchestrator implementing the full 7-stage workflow.

    Reasoning and execution are strictly separated:
      - Reasoning: LLMClient → RLMEngine → PromptManager
      - Execution: ToolRegistry → BaseTool.run()
    """

    def __init__(
        self,
        max_iterations: int | None = None,
        enable_rlm: bool | None = None,
        memory_persist_path: str | None = None,
        use_llm: bool | None = None,
        use_ml: bool | None = None,
        min_iterations: int | None = None,
    ) -> None:
        # Capability switches. Both default on, and both are honest about
        # what they cost: with use_llm off the run is fully deterministic
        # (no network, no narrative synthesis, plans come from the profile);
        # with use_ml off no model is fitted, which is the single largest
        # time saving available since training dominates every run.
        self.use_llm = (
            use_llm
            if use_llm is not None
            else os.getenv("ENABLE_LLM", "true").strip().lower() == "true"
        )
        self.use_ml = (
            use_ml
            if use_ml is not None
            else os.getenv("ENABLE_ML", "true").strip().lower() == "true"
        )
        self.min_iterations = (
            min_iterations
            if min_iterations is not None
            else int(os.getenv("MIN_ITERATIONS", "1"))
        )
        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else int(os.getenv("MAX_ITERATIONS", "15"))
        )
        self.enable_rlm = (
            enable_rlm
            if enable_rlm is not None
            else os.getenv("ENABLE_RLM_INFERENCE", "true").lower() == "true"
        )
        self.memory = MemorySystem(persist_path=memory_persist_path)
        self.llm_client = LLMClient()
        self.tool_registry = ToolRegistry()
        self._rlm_engine: RLMEngine | None = None
        self._prompt_manager: PromptManager | None = None
        # P2.4 — concurrent runs must not corrupt each other's models/reports.
        # An explicit OUTPUT_DIR (the Streamlit app always sets one, into a
        # fresh tempdir per run) is honoured as-is; with no override, default
        # to a session-scoped subdirectory rather than a fixed "output" path,
        # so two unattended CLI runs in flight never share a models/ or
        # reports/ directory. A pointer file (not a symlink — no elevated
        # rights needed on Windows) records the most recent run for humans
        # poking at the output tree by hand.
        if os.getenv("OUTPUT_DIR"):
            self._output_dir: str = os.environ["OUTPUT_DIR"]
        else:
            self._output_dir = str(Path("output") / "runs" / self.memory.session_id)
            try:
                Path(self._output_dir).mkdir(parents=True, exist_ok=True)
                Path("output").mkdir(parents=True, exist_ok=True)
                Path("output", "latest.txt").write_text(self._output_dir, encoding="utf-8")
            except OSError:
                pass
        # Natural-language analysis objective supplied by the user (optional).
        self.objective: str = os.getenv("USER_OBJECTIVE", "").strip()
        if self.objective:
            self.memory.set_context("user_objective", self.objective)
        # Most recent dataset profile (set during load_dataset).
        self.last_profile: DatasetProfile | None = None
        # The coerced dataframe from load_dataset's first profiling pass,
        # held only long enough to re-profile once the target is known
        # (7.4 needs profile-before-target; class-imbalance warnings need
        # target-before-profile). Cleared at the end of load_dataset.
        self._pending_df: pd.DataFrame | None = None
        # Charts from the most recent dashboard build (for the HTML report).
        self._last_charts: list[dict[str, Any]] = []
        self._rlm_decomposed: bool = False   # run decomposition at most once per session
        # Failures per tool across iterations — LLM-replanned steps are new
        # objects each cycle, so retry budgets must be tracked here.
        self._tool_failure_counts: dict[str, int] = {}
        # Cache of successful step results, keyed on (tool_name, resolved
        # params, input-file mtime+size) — the LLM replans from scratch each
        # iteration, so an identical step (same clean_data call, same
        # correlation_analysis params) would otherwise re-execute in full,
        # burning time and — for train_model — silently overwriting an
        # already-referenced model file with a fresh random draw.
        self._step_cache: dict[str, ToolResult] = {}
        # Optional callback fired after each tool: (tool_name, status, detail) -> None
        self.on_step_callback: Any = None
        # Optional callback fired after each LLM iteration: (iteration, stage) -> None
        self.on_iteration_callback: Any = None

    # ------------------------------------------------------------------
    # 7.4 — Analysis-mode decision
    # ------------------------------------------------------------------

    #: Objective keywords that count as "the user actually asked for a
    #: prediction" — without one of these, a name-matched numeric target at
    #: transaction grain is treated as a descriptive subject (see below).
    _PREDICTION_INTENT_KEYWORDS = (
        "predict", "forecast", "model", "classif", "regress",
        "estimate", "will churn", "likely to", "propensity",
    )

    def _decide_analysis_mode(self, col: str | None, confidence: float) -> dict[str, Any]:
        """
        Decide — and return a recorded rationale for — whether prediction is
        even the right mode, before `load_dataset`'s confidence-banded logic
        commits to training on `col`. `_AUTODETECT_HIGH` alone used to be
        sufficient to start training a regressor on any name-matched numeric
        column (`_NUMERIC_TARGET_NAMES` in memory.py); this adds one veto: a
        measure at transaction grain in a domain-matched dataset is a
        descriptive subject, not a modelling target, unless the objective
        actually asks for prediction. Autonomy includes the autonomy to
        decline to model (T2).
        """
        profile = self.last_profile
        objective_l = self.objective.lower()
        wants_prediction = any(kw in objective_l for kw in self._PREDICTION_INTENT_KEYWORDS)

        # A candidate below the autonomy floor (_AUTODETECT_LOW) is not a
        # real target — it's exactly the "very low confidence" band that
        # load_dataset's own confidence-banded logic (below) leaves
        # metadata.target_column unset for and defaults to EDA. Without this
        # check, this function unconditionally recorded `mode: "model"` for
        # ANY name-matched or positional-fallback column, however weak the
        # signal — including a column the caller never actually models.
        if col and confidence < _AUTODETECT_LOW:
            return {
                "mode": "describe",
                "target": None,
                # "candidate" (Round 8) — a column WAS found, just too weak
                # a signal to model on; kept distinct from `target` (whose
                # None means "not modelling", and load_dataset relies on
                # that) so downstream text (agenda.py) can still name the
                # column it declined rather than saying "None".
                "candidate": col,
                "rationale": (
                    f"Candidate target '{col}' reached only {confidence:.0%} confidence "
                    f"(below the {_AUTODETECT_LOW:.0%} autonomy floor) — too weak a signal "
                    "to commit to modelling; proceeding with descriptive EDA instead."
                ),
                "alternatives_rejected": [
                    f"model on '{col}' (confidence {confidence:.0%}, below autonomy floor)"
                ],
            }

        if col and profile and not wants_prediction:
            col_profile = next((c for c in profile.columns if c.name == col), None)
            is_transactional = bool(profile.domains) or (
                profile.entity_col is not None and (profile.rows_per_entity or 0) > 1.5
            )
            if col_profile is not None and col_profile.semantic_role == "measure" and is_transactional:
                return {
                    "mode": "describe",
                    "target": None,
                    "candidate": col,
                    "rationale": (
                        f"'{col}' is a descriptive measure at transaction grain "
                        f"({'domain: ' + profile.domains[0].domain if profile.domains else 'repeat-row entity structure'}), "
                        "and no prediction was requested — describing it (totals, "
                        "concentration, trend) is a better answer than training a "
                        "regressor on a name-matched column."
                    ),
                    "alternatives_rejected": [
                        f"regression on '{col}' (name-matched only, confidence {confidence:.2f})"
                    ],
                }

        if col:
            return {
                "mode": "model",
                "target": col,
                "candidate": col,
                "rationale": f"Auto-detected target '{col}' (confidence {confidence:.0%}).",
                "alternatives_rejected": [],
            }
        return {
            "mode": "describe",
            "target": None,
            "candidate": None,
            "rationale": "No plausible target column found — proceeding with descriptive EDA.",
            "alternatives_rejected": [],
        }

    # ------------------------------------------------------------------
    # Stage 1 — Dataset Ingestion
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        file_path: str,
        target_hint: str | None = None,
        interactive: bool = True,
    ) -> DatasetMetadata:
        """
        Stage 1: Ingest the dataset and store metadata in Memory.

        The LLM never sees raw data — only the compact metadata string.

        Args:
            file_path:   Path to the CSV or Excel file.
            target_hint: Optional column name to use as the ML target.
            interactive: When True and auto-detection confidence is low,
                         prompt the user via stdin. Set to False in
                         non-interactive environments (Streamlit, API).

        Returns:
            DatasetMetadata stored in the Memory System.
        """
        console.print(
            Panel(
                f"[bold]Stage 1 — Dataset Ingestion[/]\nFile: {file_path}",
                title="[bold green]Agent Controller",
                border_style="green",
            )
        )

        # Override target from CLI hint or environment
        target = target_hint or os.getenv("TARGET_COLUMN_HINT")

        ingest_tool = self.tool_registry.get("ingest_dataset")
        result = ingest_tool.run(file_path=file_path, target_column=target)

        if result.status == "error":
            raise RuntimeError(f"Dataset ingestion failed: {result.error_message}")

        raw_meta = result.output["metadata"]
        metadata = DatasetMetadata(**raw_meta)

        # ---- Data profiling runs before target auto-detection (7.4) — the
        # analysis-mode decision below needs the semantic/domain picture
        # (is this a transaction log? is the candidate column a measure?),
        # not just a column name match. Failure here is non-fatal: the
        # decision function below degrades to name-only reasoning when
        # self.last_profile is still None. ----
        try:
            df, read_report = read_any(file_path)
            df, coercions = coerce_types(df, delimiter=read_report.delimiter)
            self._pending_df = df
            profile = profile_dataframe(df, target_column=None)
            try:
                profile.domains = infer_domains(df, profile)
            except Exception as exc:
                profile.domains = []
                console.print(f"  [yellow]⚠ Domain inference skipped: {exc}[/]")
            self.last_profile = profile
            self.memory.set_context("data_profile", profile.to_dict())
            self.memory.set_context("data_profile_summary", profile.to_prompt_string())
            self.memory.set_context(
                "read_report",
                {
                    "path": read_report.path,
                    "format": read_report.format,
                    "encoding": read_report.encoding,
                    "encoding_confident": read_report.encoding_confident,
                    "delimiter": read_report.delimiter,
                    "delimiter_sniffed": read_report.delimiter_sniffed,
                    "duplicate_headers": read_report.duplicate_headers,
                    "notes": read_report.notes,
                },
            )
            self.memory.set_context("coercions", [c.to_dict() for c in coercions])
            self.memory.set_context("profile_status", "ok")
            self.memory.set_context(
                "degradations",
                collect_degradations(
                    self.memory.get_context("read_report"),
                    self.memory.get_context("coercions"),
                    profile.to_dict(),
                    "ok",
                ),
            )
            console.print(
                f"  [cyan]🔬 Profile: quality {profile.quality_score}/100, "
                f"{len(profile.warnings)} warning(s).[/]"
            )
            if coercions:
                console.print(
                    f"  [cyan]🔧 Repaired {len(coercions)} column(s): "
                    + ", ".join(f"{c.column} ({c.rule})" for c in coercions) + "[/]"
                )
        except Exception as exc:
            profile_status = f"failed: {exc}"
            self.memory.set_context("profile_status", profile_status)
            self.memory.set_context(
                "degradations", collect_degradations(None, None, None, profile_status)
            )
            console.print(
                f"  [yellow]⚠ Data profiling failed (non-fatal): {exc}. "
                "Running in degraded mode — dataset-nature tools (time-series, "
                "text, geo...) are unavailable without a profile.[/]"
            )

        # Auto-detect target column when user hasn't provided one
        if not metadata.target_column:
            col, confidence = metadata.detect_target_with_confidence()
            decision = self._decide_analysis_mode(col, confidence)
            self.memory.set_context("analysis_decision", decision)

            if decision["mode"] == "describe" and col is not None:
                # 7.4 veto: a name-matched numeric column that would
                # otherwise auto-model itself is, in this dataset's context,
                # a descriptive subject — record why and skip modelling
                # rather than training a near-tautological regressor.
                console.print(
                    f"  [cyan]📋 Analysis-mode decision: describe, not model — "
                    f"{decision['rationale']}[/]"
                )
                metadata.task_type = "eda"

            elif col and confidence >= _AUTODETECT_HIGH:
                # High confidence — proceed autonomously
                metadata.target_column = col
                console.print(
                    f"  [green]Auto-detected '{col}' as the target "
                    f"(confidence {confidence:.0%}). Proceeding with analysis...[/]"
                )
                metadata.task_type = metadata.infer_task_type()

            elif col and confidence >= _AUTODETECT_LOW:
                # Medium confidence — prompt if interactive, else use best guess
                if interactive:
                    console.print(
                        f"\n[yellow]Low-confidence target detection: '{col}' "
                        f"(confidence {confidence:.0%}).[/]"
                    )
                    user_input = input(
                        f"  Press ENTER to accept '{col}', or type another column name: "
                    ).strip()
                    chosen = user_input if user_input else col
                    if chosen in metadata.columns:
                        metadata.target_column = chosen
                        metadata.task_type = metadata.infer_task_type()
                    else:
                        console.print(f"  [yellow]Column '{chosen}' not found. Using EDA mode.[/]")
                        metadata.task_type = "eda"
                else:
                    console.print(
                        f"  [yellow]Using best-guess target '{col}' "
                        f"(confidence {confidence:.0%}). Pass target_hint to override.[/]"
                    )
                    metadata.target_column = col
                    metadata.task_type = metadata.infer_task_type()

            else:
                # Very low confidence / no viable candidate
                if interactive:
                    console.print("\n[yellow]Could not confidently determine the target column.[/]")
                    avail = ", ".join(list(metadata.columns.keys())[:15])
                    console.print(f"  Available columns: {avail}")
                    user_input = input(
                        "  Please input the target column name"
                        " (or press ENTER for EDA mode): "
                    ).strip()
                    if user_input and user_input in metadata.columns:
                        metadata.target_column = user_input
                        metadata.task_type = metadata.infer_task_type()
                    else:
                        metadata.task_type = "eda"
                else:
                    console.print("  [dim]No target column detected. Defaulting to EDA mode.[/]")
                    metadata.task_type = "eda"

        else:
            # Target was already known (explicit hint). IngestDatasetTool now
            # delegates to infer_task_type() itself, but recompute here too —
            # this is the one place a divergence would silently break
            # training (see IMPROVEMENTS.md #1), so don't gate it on
            # `not metadata.task_type` ever again.
            metadata.task_type = metadata.infer_task_type()

        self.memory.store_dataset_metadata(metadata)
        # Generic context store, not just the DatasetMetadata field — lets
        # BaseTool.requires_context declarations (e.g. select_statistical_test's
        # group_column) fill themselves in without controller-side special-casing.
        self.memory.set_context("target_column", metadata.target_column)
        self.memory.append_tool_result(result)

        # Re-profile with the now-known target column so class-imbalance
        # warnings (which need a target to check) are included — the first
        # pass above ran before the target was decided, deliberately, since
        # the analysis-mode decision needed the profile first. Cheap next to
        # the read/coerce already done; non-fatal, and only runs when the
        # first pass actually succeeded.
        if self._pending_df is not None and self.last_profile is not None:
            try:
                reprofiled = profile_dataframe(self._pending_df, target_column=metadata.target_column)
                reprofiled.domains = self.last_profile.domains
                reprofiled.grain = self.last_profile.grain
                reprofiled.entity_col = self.last_profile.entity_col
                reprofiled.rows_per_entity = self.last_profile.rows_per_entity
                self.last_profile = reprofiled
                self.memory.set_context("data_profile", reprofiled.to_dict())
                self.memory.set_context("data_profile_summary", reprofiled.to_prompt_string())
            except Exception:
                pass  # keep the pre-target profile rather than lose it
        self._pending_df = None

        # 7.5 — question agenda: what a human analyst would ask of this
        # data, generated once profiling and the analysis-mode decision are
        # both settled. Stored in context so the report's methodology
        # section can show real coverage ("N questions, M answered") rather
        # than only ever listing which tools ran.
        try:
            agenda = build_agenda(
                self.last_profile, self.memory.get_context("analysis_decision"), self.objective
            )
            self.memory.set_context("question_agenda", [q.to_dict() for q in agenda])
        except Exception as exc:
            console.print(f"  [yellow]⚠ Question agenda build skipped (non-fatal): {exc}[/]")

        return metadata

    # ------------------------------------------------------------------
    # Stages 2-7 — Full autonomous analysis pipeline
    # ------------------------------------------------------------------

    def analyze(self) -> dict[str, Any]:
        """
        Run the complete autonomous analysis pipeline (Stages 2-7).

        Returns:
            Final analysis report as a structured dict.
        """
        if not self.memory.dataset_metadata:
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")

        # Initialise PromptManager and RLMEngine. Tool descriptions are
        # filtered/ranked against the dataset's profile — the planner only
        # ever sees tools that actually apply to this data's nature.
        tool_desc = self.tool_registry.get_candidate_descriptions(
            self.last_profile,
            self.memory.dataset_metadata,
            use_ml=self.use_ml,
            use_llm=self.use_llm,
        )
        self._prompt_manager = PromptManager(self.memory, tool_desc)
        self._rlm_engine = RLMEngine(
            llm_callable=self.llm_client.call,
            system_prompt=self._prompt_manager.get_system_prompt(),
            max_depth=int(os.getenv("RLM_MAX_DEPTH", "5")),
        )

        console.print("\n[bold magenta]🚀 Starting Autonomous Analysis — Stages 2-7[/]\n")
        final_result: dict[str, Any] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task_id = progress.add_task("Reasoning…", total=None)

            for iteration in range(1, self.max_iterations + 1):
                self.memory.iteration_count = iteration
                self._rlm_engine.set_iteration(iteration)
                progress.update(
                    task_id,
                    description=f"Stage 2/4/5 — Reasoning cycle {iteration}/{self.max_iterations}",
                )

                # ---- Stage 2 / 4+5: Reasoning Phase ----
                if iteration == 1:
                    user_prompt = self._prompt_manager.get_initial_user_prompt()
                    stage_label = "stage2:initial_reasoning"
                else:
                    user_prompt = self._prompt_manager.get_iteration_user_prompt()
                    stage_label = f"stage4-5:iteration_{iteration}"

                if self.on_iteration_callback:
                    self.on_iteration_callback(iteration, stage_label)

                # ---- Deterministic mode: no LLM, by choice ----
                # Distinct from the failure path below. Nothing is "degraded"
                # here — the user asked for a deterministic run, so the
                # profile-driven plan executes once and the report is
                # synthesised from tool output without any network call.
                if not self.use_llm:
                    if iteration == 1:
                        console.print(
                            "[cyan]🔌 LLM disabled — running the deterministic "
                            "profile-driven plan.[/]"
                        )
                        llm_response = self._build_fallback_plan()
                        llm_response["reasoning"] = (
                            "LLM disabled for this run — plan selected from the "
                            "dataset profile and domain inference."
                        )
                    steps = self._parse_steps(llm_response)
                    if steps:
                        self.memory.store_analysis_plan(steps)
                        progress.update(
                            task_id, description=f"Stage 3 — Executing {len(steps)} tool(s)…"
                        )
                        self._execute_steps(steps)
                        self.memory.save()
                    final_result = self._deterministic_final()
                    break

                # ---- Reasoning with graceful degradation ----
                # An LLM/API failure must never abort a running analysis:
                # iteration 1 falls back to a deterministic plan, later
                # iterations synthesise a final answer from existing results.
                try:
                    try:
                        llm_response = self._rlm_engine.invoke(
                            user_prompt, depth=0, stage=stage_label
                        )
                    except ValueError as first_exc:
                        # A malformed/truncated/empty reply (ValueError from
                        # LLMClient.call) is usually a one-off — one retry
                        # asking for a compact plan is far cheaper than
                        # discarding the LLM planner for the whole first
                        # cycle. Transport errors (timeouts, rate limits) are
                        # already retried by the SDK and fall straight
                        # through to the fallback below.
                        console.print(
                            f"[yellow]⚠ Unusable LLM reply on iteration {iteration} "
                            f"({first_exc}) — retrying once with a compact-JSON reminder.[/]"
                        )
                        llm_response = self._rlm_engine.invoke(
                            user_prompt + _COMPACT_RETRY_NOTE,
                            depth=0,
                            stage=f"{stage_label}:retry",
                        )
                    if llm_response.get("status") == "error":
                        raise RuntimeError(
                            str(llm_response.get("error", "Unknown LLM error"))
                        )
                except Exception as exc:
                    self.memory.set_context("llm_error", f"{type(exc).__name__}: {exc}")
                    console.print(
                        f"[yellow]⚠ LLM failure on iteration {iteration}: {exc}[/]"
                    )
                    if iteration == 1:
                        console.print("[yellow]  → Using deterministic fallback plan.[/]")
                        # Record it in the degradations log both reports
                        # render — otherwise, when a later cycle's LLM call
                        # succeeds and supplies the final answer, nothing in
                        # the report says the plan itself wasn't the LLM's.
                        degradations = list(self.memory.get_context("degradations") or [])
                        degradations.append(
                            "LLM planning call failed on the first reasoning cycle "
                            f"({type(exc).__name__}: {str(exc)[:200]}) — the analysis plan "
                            "was chosen by the deterministic profile-driven fallback."
                        )
                        self.memory.set_context("degradations", degradations)
                        llm_response = self._build_fallback_plan()
                    else:
                        console.print("[yellow]  → Synthesising final report from results.[/]")
                        final_result = self._deterministic_final()
                        break

                # ---- Check for completion (Stage 7 trigger) ----
                if llm_response.get("status") == "complete":
                    if iteration < self.min_iterations:
                        console.print(
                            f"[yellow]ℹ LLM signalled complete on iteration {iteration} "
                            f"(< min_iterations {self.min_iterations}) — prompting for deeper hypothesis exploration.[/]"
                        )
                        continue_prompt = (
                            user_prompt
                            + f"\n\n[SYSTEM DIRECTIVE: Minimum exploration cycles not yet reached "
                            f"(currently iteration {iteration} of minimum {self.min_iterations}). "
                            "Do NOT return Form 2 yet. Formulate a specific follow-up hypothesis, anomaly check, "
                            "or deeper investigation, and return Form 1 (Action Plan) with 1–3 steps. "
                            "Use execute_dynamic_code if you need a custom calculation.]"
                        )
                        try:
                            reprompt_res = self._rlm_engine.invoke(
                                continue_prompt, depth=0, stage=f"{stage_label}:deepen_exploration"
                            )
                            if reprompt_res.get("status") != "complete":
                                llm_response = reprompt_res
                            else:
                                console.print("\n[bold green]✅ LLM confirmed analysis complete.[/]")
                                final_result = reprompt_res
                                break
                        except Exception as exc:
                            console.print(f"[yellow]⚠ Exploration reprompt failed ({exc}) — accepting completion.[/]")
                            final_result = llm_response
                            break
                    else:
                        console.print("\n[bold green]✅ LLM signalled analysis complete.[/]")
                        final_result = llm_response
                        break

                # ---- Parse plan steps (tolerant of malformed entries) ----
                steps = self._parse_steps(llm_response)
                if not steps:
                    console.print(
                        f"[yellow]⚠ No valid steps on iteration {iteration} — synthesising final report.[/]"
                    )
                    final_result = self._deterministic_final()
                    break
                self.memory.store_analysis_plan(steps)

                # ---- Stage 3: Tool Selection & Execution ----
                progress.update(task_id, description=f"Stage 3 — Executing {len(steps)} tool(s)…")
                self._execute_steps(steps)

                # ---- Stage 6: RLM Decomposition (if enabled & many features, once only) ----
                if self.enable_rlm and not self._rlm_decomposed and self._should_decompose():
                    progress.update(task_id, description="Stage 6 — RLM task decomposition…")
                    self._run_rlm_decomposition()
                    self._rlm_decomposed = True

                self.memory.save()

            else:
                # Max iterations reached
                console.print("[yellow]⚠ Max iterations reached — generating final report.[/]")
                final_prompt = self._prompt_manager.get_final_interpretation_prompt()
                try:
                    final_result = self._rlm_engine.invoke(
                        final_prompt, depth=0, stage="stage7:max_iter_synthesis"
                    )
                    if final_result.get("status") == "error":
                        raise RuntimeError(str(final_result.get("error", "Unknown LLM error")))
                except Exception as exc:
                    self.memory.set_context("llm_error", f"{type(exc).__name__}: {exc}")
                    final_result = self._deterministic_final()

        # P3.1 — `final_result` is, on the "complete"/max-iteration paths,
        # the raw LLM response dict returned by RLMEngine.invoke(), which
        # LLMClient.call() may have stamped with `_rlm_usage` (this run's
        # own token/cost accounting — see LLMClient.call). That's plumbing
        # for RLMEngine's running totals, not an analytical result, and
        # must not leak into the persisted report/raw JSON. The engine's
        # own running totals are the real place for this to live.
        final_result.pop("_rlm_usage", None)
        if self._rlm_engine is not None:
            self.memory.set_context("llm_usage", self._rlm_engine.usage_summary())

        # Every path through this loop must leave `findings` on the result —
        # the LLM-complete and max-iteration branches return the raw LLM
        # response dict, which doesn't carry the bus. Backfilling here means
        # the reports/dashboard can always read final_result["findings"]
        # regardless of which branch produced the final answer.
        if "findings" not in final_result:
            final_result["findings"] = [f.to_dict() for f in self.memory.ranked_findings()]
        try:
            agenda = self.memory.get_context("question_agenda") or []
            final_result["coverage"] = coverage_report(
                [Question(**q) for q in agenda], final_result["findings"]
            )
        except Exception:
            pass

        # ---- Verbatim-metric validation: enforce "cite only verbatim
        # metrics" as a mechanism, not just a prompt instruction ----
        unverified = self._flag_unverified_claims(final_result)
        if unverified:
            self.memory.set_context("unverified_claims", unverified)
            console.print(
                f"[yellow]⚠ {len(unverified)} unverified metric claim(s) in the "
                f"final synthesis — see memory context 'unverified_claims'.[/]"
            )

        # ---- Stage 7: Report Generation ----
        self._generate_final_report(final_result)
        self._generate_dashboard()
        self._generate_html_report(final_result)

        # Print reasoning trace
        if self._rlm_engine:
            console.print()
            self._rlm_engine.print_reasoning_trace()

        return final_result

    # ------------------------------------------------------------------
    # Resilience helpers — plan parsing and LLM-failure fallbacks
    # ------------------------------------------------------------------

    def _parse_steps(self, llm_response: dict[str, Any]) -> list[AnalysisStep]:
        """
        Parse LLM plan steps, skipping malformed entries instead of crashing.

        Anti-hallucination guard: steps naming tools that do not exist in the
        ToolRegistry are rejected here (never executed), and a planner note is
        recorded so the next reasoning cycle sees the correction.
        """
        steps: list[AnalysisStep] = []
        rejected: list[str] = []
        raw_steps = llm_response.get("steps", [])
        if not isinstance(raw_steps, list):
            return steps
        for idx, s in enumerate(raw_steps, 1):
            if not isinstance(s, dict):
                continue
            tool_name = s.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            if not self.tool_registry.has(tool_name):
                rejected.append(tool_name)
                continue
            parameters = s.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}
            step_number = s.get("step_number")
            steps.append(
                AnalysisStep(
                    step_number=step_number if isinstance(step_number, int) else idx,
                    tool_name=tool_name,
                    parameters=parameters,
                    rationale=str(s.get("rationale", "")),
                )
            )
        if rejected:
            console.print(
                f"  [yellow]⚠ Rejected {len(rejected)} hallucinated tool name(s): "
                f"{', '.join(rejected)}[/]"
            )
            self.memory.append_tool_result(
                ToolResult(
                    tool_name="planner",
                    status="skipped",
                    output={
                        "summary": (
                            f"Rejected unknown tool name(s): {', '.join(sorted(set(rejected)))}. "
                            f"Only these tools exist: {', '.join(self.tool_registry.names())}."
                        )
                    },
                )
            )
        return steps

    #: Nature-driven tools included in the fallback plan when applies_to()
    #: scores them at full confidence (1.0) — same tools, same gating logic
    #: the LLM planner sees, so there's one source of truth for "what suits
    #: this data" (controller._should_decompose folds in the same way).
    #: Tools the deterministic plan sequences explicitly (or never runs from
    #: the profile sweep): pipeline control, the supervised branch decided
    #: below on the target, and code execution, which needs an LLM to write
    #: the code and is meaningless without one.
    _FALLBACK_EXCLUDED_TOOLS = frozenset({
        "ingest_dataset", "clean_data", "generate_report",
        "train_model", "evaluate_model", "cluster_data",
        "execute_dynamic_code",
        # R3.1: its PNGs reach no consumer (grep for .png/chart_path/image_path
        # across app.py, html_report.py, report_generator.py returns nothing) —
        # excluded from the deterministic sweep rather than wired up, so the
        # no-LLM path doesn't spend time writing files nobody reads. Left
        # available to the LLM planner, which can still call it deliberately.
        "generate_visualizations",
    })

    #: applies_to score a tool must reach to earn a slot in the deterministic
    #: plan. Below 1.0 so a domain matched on partial evidence (0.65 for a
    #: ticker+price file with no OHLC) still contributes its analysis.
    _FALLBACK_MIN_SCORE = 0.6

    def _build_fallback_plan(self) -> dict[str, Any]:
        """
        Deterministic analysis plan used when the LLM is unreachable on the
        first reasoning cycle. Always: clean → outliers → correlation, plus
        whichever nature-specific tools the profile-driven gating says
        apply at full confidence, plus (train + evaluate) when a target
        exists or (cluster + visualise) otherwise.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            raise RuntimeError("No dataset loaded — cannot build a fallback plan.")
        fp = meta.file_path
        profile = self.last_profile
        steps: list[dict[str, Any]] = [
            {
                "step_number": 1,
                "tool_name": "clean_data",
                "parameters": {
                    "file_path": fp,
                    "strategy": "median",
                    "target_column": meta.target_column,
                },
                "rationale": "Fallback plan: impute missing values before analysis.",
            },
        ]

        # detect_outliers/correlation_analysis are near-universal but not
        # unconditional — e.g. correlation_analysis needs 2+ numeric columns
        # — so they go through the same applies_to gate as everything else
        # rather than being hardcoded past it (a single-numeric-column
        # dataset would otherwise error out here every time).
        for name, params, rationale in (
            ("detect_outliers", {"file_path": fp, "method": "iqr"}, "Fallback plan: flag anomalous rows."),
            ("correlation_analysis", {"file_path": fp, "target_column": meta.target_column}, "Fallback plan: quantify feature relationships."),
        ):
            if self.tool_registry.has(name) and self.tool_registry.get(name).applies_to(profile, meta) > 0.0:
                steps.append({
                    "step_number": len(steps) + 1,
                    "tool_name": name,
                    "parameters": params,
                    "rationale": rationale,
                })

        # Every registered tool the profile says fits, ranked by its own
        # applies_to score — not a hardcoded name list. This is what makes the
        # no-LLM path a real analyst rather than a stub: a tool registered
        # after this function was written (the domain tools, anything added
        # later) is planned automatically, and a dataset recognised as
        # transactional gets its cohort analysis without an LLM ever being
        # reachable. The previous hardcoded tuple silently excluded every
        # tool it predated.
        already = {s["tool_name"] for s in steps} | self._FALLBACK_EXCLUDED_TOOLS
        for tool in self.tool_registry.candidate_tools(
            profile, meta, use_ml=self.use_ml, use_llm=self.use_llm
        ):
            name = getattr(tool, "name", "")
            if name in already:
                continue
            score = tool.applies_to(profile, meta)
            if score < self._FALLBACK_MIN_SCORE:
                continue

            tool_params: dict[str, Any] = {"file_path": fp}
            try:
                tool_params.update(tool.default_params(profile, meta) or {})
            except Exception:
                pass

            # Never schedule a step that cannot run. file_path and output_dir
            # are injected by BaseTool.prepare_params, and requires_context
            # entries are filled from memory, so only genuinely unfilled
            # required parameters disqualify a tool.
            try:
                schema = tool.get_schema()
            except Exception:
                schema = {}
            injected = {"file_path", "output_dir"} | set(
                getattr(tool, "requires_context", {}).values()
            )
            unfilled = [
                key
                for key, spec in schema.items()
                if spec.get("required")
                and key not in tool_params
                and key not in injected
            ]
            if unfilled:
                continue

            steps.append({
                "step_number": len(steps) + 1,
                "tool_name": name,
                "parameters": tool_params,
                "rationale": (
                    f"Fallback plan: profile-driven selection scored '{name}' "
                    f"at {score:.2f} for this dataset."
                ),
            })
            already.add(name)

        if not self.use_ml:
            # No model-fitting branch at all: no supervised training and no
            # clustering fallback. The profile-driven analyses above already
            # ran, so the plan is complete and genuinely ML-free.
            pass
        elif meta.target_column and meta.task_type in ("classification", "regression"):
            steps += [
                {
                    "step_number": len(steps) + 1,
                    "tool_name": "train_model",
                    "parameters": {
                        "file_path": fp,
                        "target_column": meta.target_column,
                        "task_type": meta.task_type,
                    },
                    "rationale": "Fallback plan: train baseline models with CV.",
                },
                {
                    "step_number": len(steps) + 2,
                    "tool_name": "evaluate_model",
                    "parameters": {
                        "file_path": fp,
                        "target_column": meta.target_column,
                        "task_type": meta.task_type,
                    },
                    "rationale": "Fallback plan: evaluate the best model on held-out data.",
                },
            ]
        else:
            steps += [
                {
                    "step_number": len(steps) + 1,
                    "tool_name": "cluster_data",
                    "parameters": {"file_path": fp},
                    "rationale": "Fallback plan: no target — discover natural segments.",
                },
            ]
        return {
            "status": "in_progress",
            "reasoning": "LLM unavailable — executing deterministic fallback plan.",
            "steps": steps,
        }

    def _generate_dashboard(self) -> None:
        """
        Build the dynamic dashboard from the final state of the analysis and
        save it as JSON next to the reports. Non-fatal on any failure.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            return
        source = str(self.memory.get_context("cleaned_file_path") or meta.file_path)
        try:
            df = _read_dataframe(source)
            profile = profile_dataframe(df, target_column=meta.target_column)
            charts = build_dashboard(
                df,
                profile,
                target_column=meta.target_column,
                task_type=meta.task_type,
                tool_results=[r.to_dict() for r in self.memory.tool_results],
                findings=[f.to_dict() for f in self.memory.ranked_findings()],
            )
            out_path = Path(self._output_dir) / "reports" / "dashboard.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(dashboard_to_json(charts), encoding="utf-8")
            self._last_charts = [c.to_dict() for c in charts]
            self.memory.set_context("dashboard_path", str(out_path))
            console.print(
                f"  [green]📊 Dashboard generated: {len(charts)} chart(s) → {out_path}[/]"
            )
        except Exception as exc:
            console.print(f"  [yellow]⚠ Dashboard generation failed (non-fatal): {exc}[/]")

    def _generate_html_report(self, llm_final: dict[str, Any]) -> None:
        """
        Write the self-contained HTML report (summary + interactive dashboard).
        Non-fatal on any failure.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            return
        try:
            from src.core.html_report import build_html_report

            html_doc = build_html_report(
                dataset_name=Path(meta.file_path).stem,
                llm_insights=llm_final,
                tool_results=[r.to_dict() for r in self.memory.tool_results],
                charts=self._last_charts,
                objective=self.objective,
                profile=self.memory.get_context("data_profile"),
                read_report=self.memory.get_context("read_report"),
                coercions=self.memory.get_context("coercions"),
                plan_rationales=self.memory.get_context("plan_rationales"),
                statistical_test_pvalues=self.memory.get_context("statistical_test_pvalues"),
                unverified_claims=self.memory.get_context("unverified_claims"),
                profile_status=self.memory.get_context("profile_status"),
                degradations=self.memory.get_context("degradations"),
                findings=[f.to_dict() for f in self.memory.ranked_findings()],
                analysis_decision=self.memory.get_context("analysis_decision"),
            )
            out_path = Path(self._output_dir) / "reports" / "report.html"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(html_doc, encoding="utf-8")
            self.memory.set_context("html_report_path", str(out_path))
            console.print(f"  [green]🌐 HTML report generated → {out_path}[/]")
        except Exception as exc:
            console.print(f"  [yellow]⚠ HTML report generation failed (non-fatal): {exc}[/]")

    def _deterministic_final(self) -> dict[str, Any]:
        """
        Synthesise a final report dict, projected from the finding bus (7.1)
        rather than re-deriving a narrative from raw tool output per tool.
        `best_model`/`key_metrics` stay derived directly from train_model's
        output since they're model-internal facts, not audience-facing
        findings; every analytical claim ("X differs by Y", "Z is trending")
        comes from `MemorySystem.ranked_findings()`, which is the same list
        every other surface (both reports, the dashboard) reads — a tool
        that gains a findings() implementation reaches all of them at once
        instead of needing a hardcoded case in each narrator.
        """
        recommendations: list[str] = []
        key_metrics: dict[str, Any] = {}
        best_model = self.memory.get_context("best_model_name")

        train = self.memory.get_last_result_for("train_model")
        if train and train.status == "success":
            models_trained = train.output.get("models_trained", {})
            best = train.output.get("best_model")
            if best and best in models_trained:
                best_model = best
                metrics = models_trained[best]
                key_metrics["cv_mean"] = metrics.get("cv_mean")
                key_metrics["cv_std"] = metrics.get("cv_std")
                key_metrics["train_test_gap"] = metrics.get("train_test_gap")
            warnings = train.output.get("overfit_warnings", [])
            if warnings:
                recommendations.append(
                    "Reduce model complexity (lower max_depth) or add regularisation "
                    "to close the train-test gap."
                )

        ranked = self.memory.ranked_findings()
        insights = [f.headline for f in ranked[:12]]
        if not insights:
            insights.append("Analysis produced no tool results to synthesise.")
        # Chosen deterministic mode and a mid-run LLM failure produce the same
        # findings but are not the same event, and saying so matters: telling
        # someone to "re-run with a reachable provider" when they deliberately
        # switched the LLM off reads as a malfunction rather than the mode
        # working as asked.
        if not self.use_llm:
            reasoning = (
                "Deterministic run: the LLM was switched off, so the plan came "
                "from the dataset profile and domain inference and these "
                "findings were compiled directly from tool output."
            )
            if not recommendations:
                recommendations.append(
                    "Turn the AI narrative on to get these same findings "
                    "interpreted and prioritised in plain language."
                )
        else:
            if not recommendations:
                recommendations.append(
                    "Re-run with a reachable LLM provider for narrative interpretation "
                    "of these deterministic findings."
                )
            reasoning = (
                "Deterministic synthesis: the LLM became unreachable mid-run, so "
                "findings were compiled directly from tool outputs."
            )
            llm_error = self.memory.get_context("llm_error")
            if llm_error:
                reasoning += f" (LLM error: {llm_error})"

        if not self.use_ml:
            recommendations.append(
                "Machine learning was switched off for this run — no model was "
                "fitted. Turn it on for predictive modelling and clustering."
            )
        if self.objective:
            reasoning = f"User objective: {self.objective}\n{reasoning}"

        return {
            "status": "complete",
            "llm_fallback": not self.use_llm or bool(self.memory.get_context("llm_error")),
            "deterministic_mode": not self.use_llm,
            "ml_enabled": self.use_ml,
            "reasoning": reasoning,
            "insights": insights,
            "recommendations": recommendations,
            "best_model": best_model,
            "key_metrics": key_metrics,
            "findings": [f.to_dict() for f in ranked],
        }

    # ------------------------------------------------------------------
    # Stage 3 execution helper
    # ------------------------------------------------------------------

    @staticmethod
    def _step_cache_key(tool_name: str, params: dict[str, Any]) -> str:
        """
        Content-address a step: same tool, same resolved params, same input
        file content → same key. File-valued params are stamped with
        mtime+size (not just path) so an in-place edit still invalidates.
        """
        parts = [tool_name]
        for key in sorted(params):
            value = params[key]
            parts.append(f"{key}={value!r}")
            if isinstance(value, str):
                candidate = Path(value)
                if candidate.is_file():
                    stat = candidate.stat()
                    parts.append(f"{key}.stat={stat.st_mtime_ns}:{stat.st_size}")
        return "|".join(parts)

    def _execute_steps(self, steps: list[AnalysisStep]) -> None:
        """
        Stage 3 — execute each tool in the plan with retry budgets.

        Retry logic: failures are counted per tool across iterations
        (the LLM re-plans with fresh step objects each cycle). Once a
        tool has failed MAX_STEP_RETRIES times it is skipped instead of
        executed again, so one broken tool can never stall the pipeline.
        """
        total_steps = len(steps)
        for idx, step in enumerate(steps, 1):
            if self._tool_failure_counts.get(step.tool_name, 0) >= MAX_STEP_RETRIES:
                console.print(
                    f"  [yellow]⏭ Step {step.step_number}: {step.tool_name} skipped "
                    f"(exceeded {MAX_STEP_RETRIES} retries).[/]"
                )
                skip_result = ToolResult(
                    tool_name=step.tool_name,
                    status="skipped",
                    output={
                        "summary": (
                            f"Skipped: '{step.tool_name}' already failed "
                            f"{MAX_STEP_RETRIES} times. Do not plan it again."
                        )
                    },
                )
                self.memory.append_tool_result(skip_result)
                self.memory.mark_step_complete(step.step_number, skip_result)
                continue

            console.print(
                f"  [cyan]→ Step {step.step_number}: {step.tool_name}[/] "
                f"[dim]{step.rationale[:60]}[/]"
            )
            # Fire pre-execution callback
            if self.on_step_callback:
                self.on_step_callback(step.tool_name, "running",
                                      f"{idx}/{total_steps} — {step.tool_name}…")
            # Defense in depth: _parse_steps filters unknown tools, but never
            # let a registry miss crash the whole pipeline.
            try:
                tool = self.tool_registry.get(step.tool_name)
            except KeyError as exc:
                result = ToolResult(
                    tool_name=step.tool_name,
                    status="error",
                    output={},
                    error_message=str(exc),
                )
                self.memory.append_tool_result(result)
                self.memory.mark_step_complete(step.step_number, result)
                continue

            # Generic parameter resolution, driven by each tool's own
            # declarations (BaseTool.prepare_params) — cleaned_file_path
            # redirection, output_dir injection, and any bespoke overrides
            # (best_model_path, forced test_size, report result injection)
            # all live on the tool itself instead of growing this if-ladder
            # every time a new tool needs to plug into the pipeline.
            params = tool.prepare_params(step.parameters, self.memory, self._output_dir)

            cache_key = self._step_cache_key(step.tool_name, params)
            cached = self._step_cache.get(cache_key)
            if cached is not None:
                console.print(
                    f"  [dim]↺ Step {step.step_number}: {step.tool_name} — "
                    f"identical to a prior successful step, reusing its result.[/]"
                )
                # Reuse the cached ToolResult object directly
                result = cached
            else:
                result = tool.run(**params)
                if result.status == "success":
                    self._step_cache[cache_key] = result
            result.iteration = self.memory.iteration_count
            self.memory.append_tool_result(result)
            self.memory.mark_step_complete(step.step_number, result)

            # Finding bus (7.1) — project this tool's own output into the
            # shared Finding list right after it succeeds, so every surface
            # (deterministic synthesis, both reports, the dashboard) reads
            # one ranked list instead of each re-deriving its own narrative
            # from raw tool JSON. A tool with no findings() override (data
            # prep tools) contributes nothing here, which is correct.
            if result.status == "success":
                try:
                    new_findings: list[Finding] = tool.findings(result.output, self.last_profile, self.memory.dataset_metadata)
                except Exception as exc:
                    new_findings = []
                    console.print(f"  [yellow]⚠ findings() failed for {step.tool_name} (non-fatal): {exc}[/]")
                if new_findings:
                    for i, finding in enumerate(new_findings):
                        if not finding.finding_id:
                            finding.finding_id = f"{step.tool_name}_{step.step_number}_{i}"
                        if not finding.source_tool:
                            finding.source_tool = step.tool_name
                    self.memory.add_findings(new_findings)

            # Round 8 — dynamic tool creation. define_analysis_tool's own
            # execute() is pure (per AGENTS.md's layer rule, it never touches
            # the registry or memory) — it only validates and smoke-tests a
            # proposed tool and reports output["status"] == "ready"/"error".
            # Registration is the controller's job, same precedent as
            # "clean_data succeeds -> controller sets cleaned_file_path".
            if step.tool_name == "define_analysis_tool" and result.status == "success":
                self._maybe_register_generated_tool(result)

            # Item 6 (report restructure): the planner is required to give a
            # rationale for every step (prompt_manager.py), but it was only
            # ever shown truncated in a console panel and then discarded.
            # Accumulate it here so the report's Methodology section can
            # pair each executed tool with why it was chosen.
            rationales = self.memory.get_context("plan_rationales") or []
            rationales.append({
                "step_number": step.step_number,
                "tool_name": step.tool_name,
                "rationale": step.rationale,
            })
            self.memory.set_context("plan_rationales", rationales)

            # Item 4 (statistical rigor): Benjamini-Hochberg correction needs
            # every p-value produced in this run — accumulate them here so
            # the report (item 6) can correct at report time rather than
            # each hypothesis test correcting itself in isolation.
            if step.tool_name == "select_statistical_test" and result.status == "success":
                pvalue_tests = self.memory.get_context("statistical_test_pvalues") or []
                family = result.output.get("family_results")
                if family:
                    # Family mode (7.6) ran one test per eligible dimension,
                    # not just the single strongest pairing surfaced at the
                    # top level — recording only that one (the old
                    # behaviour) under-counted how many tests this run
                    # actually performed, which understates the multiple-
                    # comparison correction at report time. Record every
                    # pairing, with its own (raw, uncorrected) p_value.
                    for entry in family:
                        feature = entry.get("feature_column") or result.output.get("feature_column")
                        group = entry.get("group_column")
                        pvalue_tests.append({
                            "step_number": step.step_number,
                            "feature_column": f"{feature} by {group}" if group else feature,
                            "test_name": entry.get("test_name"),
                            "p_value": entry.get("p_value"),
                        })
                elif "p_value" in result.output:
                    pvalue_tests.append({
                        "step_number": step.step_number,
                        "feature_column": result.output.get("feature_column"),
                        "test_name": result.output.get("test_name"),
                        "p_value": result.output["p_value"],
                    })
                self.memory.set_context("statistical_test_pvalues", pvalue_tests)

            # segment_comparison (7.2) runs its own family of per-level
            # tests (proportions z-test for a rate measure, Welch's t-test
            # otherwise) — those belong in the same run-wide correction pool
            # as select_statistical_test's, or a run that only ever calls
            # this tool would report zero corrected tests despite having
            # run many.
            if step.tool_name == "segment_comparison" and result.status == "success":
                pvalue_tests = self.memory.get_context("statistical_test_pvalues") or []
                for c in result.output.get("comparisons", []):
                    if c.get("p_value") is None:
                        continue
                    test_name = "proportions_ztest" if c.get("is_rate") else "welch_ttest"
                    pvalue_tests.append({
                        "step_number": step.step_number,
                        "feature_column": f"{c.get('measure')} by {c.get('dimension')}={c.get('level')}",
                        "test_name": test_name,
                        "p_value": c["p_value"],
                    })
                self.memory.set_context("statistical_test_pvalues", pvalue_tests)

            if result.status == "error":
                self._tool_failure_counts[step.tool_name] = (
                    self._tool_failure_counts.get(step.tool_name, 0) + 1
                )
                self.memory.increment_retry(step.step_number)

            # Store important outputs in memory context for downstream tools
            if step.tool_name == "train_model" and result.status == "success":
                best = result.output.get("best_model", "")
                mt = result.output.get("models_trained", {})
                if best and best in mt:
                    model_path = mt[best].get("model_path", "")
                    if model_path:
                        self.memory.set_context("best_model_path", model_path)
                        self.memory.set_context("best_model_name", best)
                trained_test_size = result.output.get("test_size")
                if trained_test_size is not None:
                    self.memory.set_context("train_test_size", trained_test_size)
                # evaluate_model must recreate the exact same partition —
                # persist the strategy train_model actually resolved to
                # (may differ from what was requested if a column was
                # missing) so evaluate isn't left shuffling data that was
                # split chronologically or by group.
                trained_split_strategy = result.output.get("split_strategy")
                if trained_split_strategy in ("random", "time_series", "panel"):
                    self.memory.set_context("split_strategy", trained_split_strategy)
                    self.memory.set_context("split_time_column", result.output.get("time_column"))
                    self.memory.set_context("split_group_column", result.output.get("group_column"))

            # If cleaning produced a cleaned file, store it for downstream tools
            if step.tool_name == "clean_data" and result.status == "success":
                cleaned_path = result.output.get("cleaned_file_path")
                if cleaned_path:
                    self.memory.set_context("cleaned_file_path", cleaned_path)
                    console.print(
                        f"  [dim]Cleaned file stored → {cleaned_path}[/]"
                    )
            # Fire post-execution callback
            if self.on_step_callback:
                summary = result.output.get("summary", "")[:80] if result.status == "success" else result.error_message
                self.on_step_callback(step.tool_name, result.status, f"{idx}/{total_steps} done — {summary}")

    def _maybe_register_generated_tool(self, result: ToolResult) -> None:
        """
        Round 8 — validate and, if it passes, register+persist a tool the
        LLM proposed via define_analysis_tool.

        define_analysis_tool.execute() only validates/smoke-tests (it must
        never mutate the registry or memory itself, per AGENTS.md's layer
        rule); this is where that proposal actually becomes a callable tool.
        A validation failure rewrites `result` in place to status="error" so
        the LLM sees exactly what to fix on its next iteration, the same way
        any other tool failure is surfaced.
        """
        if result.output.get("status") != "ready":
            return
        spec_dict = result.output.get("spec") or {}
        name = spec_dict.get("name")
        description = spec_dict.get("description", "")
        params_schema = spec_dict.get("params_schema", {}) or {}
        code = spec_dict.get("code", "")
        if not name:
            result.status = "error"
            result.error_message = "define_analysis_tool returned status='ready' with no tool name."
            return

        from src.core.tool_factory import (
            GeneratedToolSpec,
            compute_dataset_fingerprint,
            register_and_persist,
            validate_spec,
        )

        existing_generated = {g["name"]: g for g in self.memory.list_generated_tools() if g.get("name")}
        errors = validate_spec(
            name=name,
            description=description,
            params_schema=params_schema,
            code=code,
            existing_tool_names=[n for n in self.tool_registry.names() if n not in existing_generated],
            existing_generated=existing_generated,
        )
        if errors:
            result.status = "error"
            result.error_message = (
                f"'{name}' was not registered: " + "; ".join(errors)
            )
            console.print(f"  [yellow]⚠ Generated tool '{name}' rejected: {result.error_message}[/]")
            return

        prior = existing_generated.get(name)
        version = (prior.get("version", 0) + 1) if prior else 1
        # GeneratedTool inherits uses_cleaned_file=True (the default), so it
        # always runs against the cleaned dataset once one exists — fingerprint
        # that same path, or a later opt-in reload (load_persisted_tools)
        # would never match.
        runtime_path = (
            self.memory.get_context("cleaned_file_path")
            or (self.memory.dataset_metadata.file_path if self.memory.dataset_metadata else "")
        )
        fingerprint = compute_dataset_fingerprint(runtime_path) if runtime_path else ""
        spec = GeneratedToolSpec(
            name=name,
            description=description,
            params_schema=params_schema,
            code=code,
            version=version,
            created_at=datetime.now(UTC).isoformat(),
            dataset_fingerprint=fingerprint,
        )
        register_and_persist(spec, self.tool_registry, self.memory, self._output_dir)
        console.print(
            f"  [bold green]✓ Registered generated tool[/] '{name}' "
            f"(v{version}, callable starting next iteration)."
        )

    # ------------------------------------------------------------------
    # Stage 6 — RLM decomposition
    # ------------------------------------------------------------------

    def _should_decompose(self) -> bool:
        """Trigger decomposition for wide datasets — the same structural
        fact (is_high_dimensional) that gates dimensionality_analysis, so
        there's one source of truth for "this data has a lot of features"."""
        if self.last_profile is not None:
            return self.last_profile.is_high_dimensional
        meta = self.memory.dataset_metadata
        return meta is not None and meta.column_count > 15

    def _run_rlm_decomposition(self) -> None:
        """
        Stage 6: Decompose the feature space into sub-groups and run
        targeted LLM sub-calls on each group.

        This is the core RLM innovation: instead of one monolithic context,
        each feature group gets its own focused reasoning call.
        """
        if self._rlm_engine is None or self._prompt_manager is None:
            return

        meta = self.memory.dataset_metadata
        if meta is None:
            return

        num_cols = meta.numerical_cols
        cat_cols = meta.categorical_cols

        # Partition numerical columns into groups of ≤8
        groups: dict[str, list[str]] = {}
        chunk_size = 8
        for i in range(0, len(num_cols), chunk_size):
            groups[f"numerical_group_{i // chunk_size + 1}"] = num_cols[i: i + chunk_size]
        if cat_cols:
            groups["categorical_group"] = cat_cols[:10]

        sub_tasks = [
            RLMSubTask(
                task_id=gid,
                description=f"Analyse {len(cols)}-feature group: {', '.join(cols[:5])}…",
                context={"columns": cols, "group_id": gid},
            )
            for gid, cols in groups.items()
        ]

        if not sub_tasks:
            return

        def build_prompt(task: RLMSubTask) -> str:
            assert self._prompt_manager is not None
            ctx_summary = json.dumps(task.context)
            return self._prompt_manager.get_rlm_subtask_prompt(
                task_id=task.task_id,
                description=task.description,
                context_summary=ctx_summary,
            )

        # Graceful degradation, same rationale as the stage-2 planning call
        # (line ~1184): decomposition results are optional context for final
        # synthesis, not a required step, so a transient LLM/API failure here
        # (rate limit, timeout, provider outage) must not abort the run —
        # it should just mean synthesis proceeds without the extra detail.
        try:
            sub_results = self._rlm_engine.decompose_and_invoke(
                sub_tasks=sub_tasks,
                prompt_builder=build_prompt,
                depth=1,
            )
        except Exception as exc:
            self.memory.set_context("rlm_decomposition_error", f"{type(exc).__name__}: {exc}")
            console.print(f"[yellow]⚠ RLM decomposition failed: {exc}[/]")
            console.print("[yellow]  → Continuing without sub-task decomposition.[/]")
            return

        # Store sub-results in memory context for final synthesis
        self.memory.set_context("rlm_sub_results", sub_results)
        console.print(
            f"  [green]✓ RLM decomposition complete: "
            f"{len(sub_results)} sub-task(s) resolved.[/]"
        )

    # ------------------------------------------------------------------
    # Stage 7 — Report generation
    # ------------------------------------------------------------------

    def _verified_number_pool(self) -> set[str]:
        """Every numeric literal that actually appears in accumulated tool
        results, canonicalised for verbatim-citation checking."""
        pool, _per_tool = self._verified_number_pools()
        return pool

    def _verified_number_pools(self) -> tuple[set[str], dict[str, set[str]]]:
        """Global pool (unchanged — every numeric literal across all tool
        results) plus a pool per tool_name, so a claim keyword-attributed to
        a specific kind of analysis (`_KEYWORD_TOOL_MAP`) can be checked
        against only that tool's OWN output rather than the whole run's
        pooled numbers — a row count from ingest_dataset's summary must not
        verify a claim about what clean_data did."""
        global_pool: set[str] = set()
        per_tool: dict[str, set[str]] = {}
        for r in self.memory.tool_results:
            tool_pool = per_tool.setdefault(r.tool_name, set())
            _collect_numbers(r.to_dict(), tool_pool)
            global_pool |= tool_pool
        if self.memory.dataset_metadata:
            meta_pool: set[str] = set()
            _collect_numbers(self.memory.dataset_metadata.__dict__, meta_pool)
            global_pool |= meta_pool
        return global_pool, per_tool

    def _flag_unverified_claims(self, final_result: dict[str, Any]) -> list[str]:
        """
        Enforce SYSTEM_PROMPT_CORE's "cite only verbatim metrics" rule.

        Any numeric literal in `insights`/`recommendations`/`key_metrics`
        that doesn't trace back to a real tool result is annotated
        in-place with `[unverified: ...]` (never silently trusted) and
        returned so the caller can log/report the hallucination rate.

        Round 8 hardening — a number existing ANYWHERE in the pooled tool
        output used to be enough to "verify" it, which passed claims that
        cited the right number from the WRONG tool (e.g. citing
        ingest_dataset's row count as clean_data's missing-value count,
        since both numbers were in the pool). When a sentence names a
        specific kind of analysis (`_KEYWORD_TOOL_MAP`) and that analysis's
        tool(s) actually ran, its numbers are checked against only that
        tool's own output; a sentence with no such keyword keeps the
        original global-pool check.
        """
        verified, per_tool_pools = self._verified_number_pools()
        ran_tools = {r.tool_name for r in self.memory.tool_results}
        flagged: list[str] = []

        meta = self.memory.dataset_metadata
        meta_pool: set[str] = set()
        if meta:
            for p in _CANON_PRECISIONS:
                meta_pool.add(_canon_number(meta.row_count, p))
                meta_pool.add(_canon_number(meta.column_count, p))

        for field in ("insights", "recommendations"):
            items = final_result.get(field)
            if not isinstance(items, list):
                continue
            for i, item in enumerate(items):
                if not isinstance(item, str):
                    continue
                claimed = [
                    m.group() for m in _NUMBER_RE.finditer(item)
                    if not (
                        "." not in m.group()
                        and abs(int(m.group().replace(",", ""))) <= _UNVERIFIABLE_SKIP_ABS_INT
                    )
                ]
                if not claimed:
                    continue

                item_l = item.lower()
                attributed_tools = sorted({
                    tool
                    for pattern, tools in _KEYWORD_TOOL_PATTERNS
                    if pattern.search(item_l)
                    for tool in tools
                    if tool in ran_tools
                })
                if attributed_tools:
                    pool = set().union(*(per_tool_pools.get(t, set()) for t in attributed_tools))
                    valid_pool = pool | meta_pool
                    bad = sorted({n for n in claimed if _canon_number(n) not in valid_pool})
                    if bad:
                        attribution = "/".join(attributed_tools)
                        items[i] = (
                            f"{item} [unverified: {', '.join(bad)} "
                            f"(not in {attribution} output)]"
                        )
                        flagged.append(
                            f"{field}[{i}]: {', '.join(bad)} (attributed to {attribution})"
                        )
                    continue

                bad = sorted({n for n in claimed if _canon_number(n) not in verified})
                if bad:
                    items[i] = f"{item} [unverified: {', '.join(bad)}]"
                    flagged.append(f"{field}[{i}]: {', '.join(bad)}")

        key_metrics = final_result.get("key_metrics")
        if isinstance(key_metrics, dict):
            for k, v in key_metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    nums = [str(v)]
                elif isinstance(v, str):
                    nums = [m.group() for m in _NUMBER_RE.finditer(v)]
                else:
                    continue
                bad = sorted({n for n in nums if _canon_number(n) not in verified})
                if bad:
                    flagged.append(f"key_metrics.{k}={v!r} [unverified: {', '.join(bad)}]")

        return flagged

    def _generate_final_report(self, llm_final: dict[str, Any]) -> None:
        """
        Stage 7: Invoke GenerateReportTool to produce the Markdown/JSON report.

        The report tool is called with the serialised tool results and the
        LLM's final interpretation so it can produce a complete document.
        """
        meta = self.memory.dataset_metadata
        dataset_name = Path(meta.file_path).stem if meta else "dataset"

        tool_results_json = json.dumps(
            [r.to_dict() for r in self.memory.tool_results], default=str
        )

        report_tool = self.tool_registry.get("generate_report")
        result = report_tool.run(
            dataset_name=dataset_name,
            tool_results_json=tool_results_json,
            llm_insights=llm_final,
            output_dir=str(Path(self._output_dir) / "reports"),
            data_profile=self.memory.get_context("data_profile"),
            read_report=self.memory.get_context("read_report"),
            coercions=self.memory.get_context("coercions"),
            plan_rationales=self.memory.get_context("plan_rationales"),
            statistical_test_pvalues=self.memory.get_context("statistical_test_pvalues"),
            unverified_claims=self.memory.get_context("unverified_claims"),
            profile_status=self.memory.get_context("profile_status"),
            degradations=self.memory.get_context("degradations"),
            findings=[f.to_dict() for f in self.memory.ranked_findings()],
            analysis_decision=self.memory.get_context("analysis_decision"),
        )

        self.memory.append_tool_result(result)

        if result.status == "success":
            console.print(
                Panel(
                    f"[bold green]Stage 7 — Report Generated[/]\n"
                    f"Markdown: {result.output.get('markdown_path')}\n"
                    f"JSON:     {result.output.get('json_path')}",
                    border_style="green",
                )
            )
        else:
            console.print(
                f"[yellow]⚠ Report generation failed: {result.error_message}[/]"
            )
