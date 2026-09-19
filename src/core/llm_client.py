"""
LLM Client — thin, provider-agnostic wrapper around the LLM provider SDKs.

Split out of ``src/core/controller.py``; re-exported from there, so
``from src.core.controller import LLMClient`` keeps working.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, cast

from src.core.governance import LOCAL_PROVIDERS, local_only, record_llm_call
from src.core.model_telemetry import get_limiter

#: Fallback model per provider, used only when LLM_MODEL is unset.
_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-flash-latest",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "openai/gpt-4o",
    "nvidia": "openai/gpt-oss-120b",
    "local": "llama3.1",
    "ollama": "llama3.1",
}


class LocalOnlyError(RuntimeError):
    """LOCAL_ONLY=true and the configured provider would send data off-machine."""


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
        #: Where llm_calls.jsonl is written (set by AgentController); None
        #: disables the LLM audit (offline scripts, bare pings).
        self.audit_dir: str | None = None
        #: Reasoning stage of the current call, for the audit log — set by
        #: the controller before each invoke; best-effort under RLM threads.
        self.stage: str | None = None
        #: OpenAI-compatible JSON mode; switched off for the rest of the run
        #: the first time an endpoint rejects it.
        self._json_mode = os.getenv("LLM_JSON_FORMAT", "true").strip().lower() != "false"

    def ping(self) -> tuple[bool, str]:
        """
        Cheap connectivity + model-validity check (a small token budget).

        Returns (True, "") on success, (False, "<ExceptionType>: <detail>")
        on any failure — so callers can fail fast with the real reason
        instead of running a whole analysis on the deterministic fallback.
        """
        # Reasoning/"thinking" models (Gemini 3.x, NVIDIA gpt-oss, o-series-
        # style models) spend part of the budget on hidden reasoning tokens
        # before any visible output — 16 was enough for plain chat models
        # but silently starved thinking models into empty content. 200 is
        # still a negligible cost for a connectivity check.
        self._usage_local.max_tokens = 200
        self._usage_local.effort = None
        try:
            self._dispatch("You are a connectivity check. Reply with OK.", "Reply with OK.")
            return True, ""
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        finally:
            self._usage_local.max_tokens = None

    def _call_max_tokens(self) -> int:
        """This thread's max_tokens for the current call (per-call override or the default)."""
        return getattr(self._usage_local, "max_tokens", None) or self.max_tokens

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Call the configured LLM and return the parsed JSON response.

        Raises:
            ValueError: If the response cannot be parsed as JSON.
        """
        self._usage_local.value = None
        self._usage_local.truncated = False
        # Per-call overrides live in thread-local state: this instance is shared
        # across RLM worker threads, so instance attributes would race.
        self._usage_local.effort = reasoning_effort
        self._usage_local.max_tokens = max_tokens
        try:
            raw = self._dispatch(system_prompt, user_prompt)
        except Exception as exc:
            self._audit(system_prompt, user_prompt, None, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._usage_local.max_tokens = None
        self._audit(system_prompt, user_prompt, raw, None)
        try:
            parsed = self._parse_json(raw)
        except ValueError as exc:
            # A reply cut off at max_tokens is the usual reason a long plan
            # fails to parse — say so, instead of only "non-JSON", so the
            # fix (raise LLM_MAX_TOKENS / lower reasoning effort) is obvious.
            if getattr(self._usage_local, "truncated", False):
                raise ValueError(
                    f"LLM reply was truncated at max_tokens={self._call_max_tokens()} "
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

    def _audit(self, system_prompt: str, user_prompt: str, response: str | None, error: str | None) -> None:
        if self.audit_dir:
            record_llm_call(
                self.audit_dir, stage=self.stage, provider=self.provider, model=self.model,
                system_prompt=system_prompt, user_prompt=user_prompt, response=response,
                usage=getattr(self._usage_local, "value", None), error=error,
            )

    def _dispatch(self, system_prompt: str, user_prompt: str) -> str:
        if local_only() and self.provider not in LOCAL_PROVIDERS:
            raise LocalOnlyError(
                f"LOCAL_ONLY=true forbids sending prompts to provider '{self.provider}'. "
                "Use LLM_PROVIDER=local or ollama, or unset LOCAL_ONLY."
            )
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
        elif self.provider == "groq":
            # Groq Cloud's ultra-fast OpenAI-compatible endpoint
            # https://console.groq.com/docs/openai
            api_key = os.getenv("GROQ_API_KEY", "")
            base_url = "https://api.groq.com/openai/v1"
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
                "groq": "GROQ_API_KEY",
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
        estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4 + self._call_max_tokens()
        get_limiter().acquire(self.provider, self.model, estimated_tokens=estimated_tokens)

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self._call_max_tokens(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._json_mode:
            # Constrained output: the OpenAI JSON-mode contract, which
            # OpenAI, Gemini, OpenRouter, NVIDIA NIM and Ollama accept. An
            # endpoint that rejects it is retried without it (_create), and
            # _parse_json's lenient repair still backs every reply.
            create_kwargs["response_format"] = {"type": "json_object"}
            if self.provider in LOCAL_PROVIDERS:
                # Ollama's grammar-constrained JSON (`format`), stronger than
                # JSON mode for small offline models.
                create_kwargs["extra_body"] = {"format": "json"}
        # Reasoning effort: per-call override > LLM_REASONING_EFFORT >
        # provider-specific env > adaptive (deeper for planning/synthesis).
        provider_env = {"gemini": "GEMINI_REASONING_EFFORT", "openrouter": "OPENROUTER_REASONING_EFFORT"}.get(
            self.provider, ""
        )
        resolved_effort = (
            getattr(self._usage_local, "effort", None)
            or os.getenv("LLM_REASONING_EFFORT")
            or (os.getenv(provider_env) if provider_env else None)
            or "adaptive"
        ).strip().lower()
        if resolved_effort in ("off", "0"):
            resolved_effort = "none"
        if resolved_effort == "adaptive":
            stage_str = (self.stage or "").lower()
            heavy = any(k in stage_str for k in ("stage2", "stage7", "initial", "synthesis", "report"))
            resolved_effort = "medium" if heavy else "low"

        model_l = self.model.lower()
        if self.provider == "gemini" and any(k in model_l for k in ("thinking", "2.5", "3.")):
            # Gemini 2.5+/3.x "thinking" models spend a large, variable, and
            # otherwise invisible share of max_tokens on hidden reasoning
            # before writing any visible answer. Left uncapped, a normal
            # max_tokens budget can be entirely consumed by thinking, so the
            # JSON answer gets truncated or never starts at all.
            effort = resolved_effort
            if effort == "none" and not ("2.5" in model_l and "flash" in model_l):
                effort = "low"  # only 2.5 Flash can switch thinking off entirely
            if effort in ("none", "low", "medium", "high"):
                create_kwargs["reasoning_effort"] = effort
        if self.provider == "openrouter":
            # Same failure mode as Gemini above, on OpenRouter's many
            # reasoning-capable (often free) models: hidden reasoning tokens
            # count against max_tokens, so a long Form 1 plan gets truncated
            # or never starts. OpenRouter's unified `reasoning` parameter caps
            # that; models without reasoning ignore it. "none" asks for the
            # least reasoning ("minimal").
            effort = "minimal" if resolved_effort == "none" else resolved_effort
            if effort not in ("max", "xhigh", "high", "medium", "low", "minimal"):
                effort = "low"
            create_kwargs.setdefault("extra_body", {})["reasoning"] = {"effort": effort}
        if self.provider == "openai" and re.match(r"(o\d|gpt-5)", model_l):
            # o-series / GPT-5: reasoning models reject `temperature` and
            # `max_tokens` (they take max_completion_tokens) and cannot go below "low"/"minimal".
            create_kwargs.pop("temperature", None)
            create_kwargs["max_completion_tokens"] = create_kwargs.pop("max_tokens")
            effort = resolved_effort or "medium"
            if effort == "none":
                effort = "minimal" if model_l.startswith("gpt-5") else "low"
            if effort in ("minimal", "low", "medium", "high"):
                create_kwargs["reasoning_effort"] = effort
        if self.provider == "nvidia":
            # NVIDIA NIM requires streaming; gpt-oss-120b also emits
            # reasoning_content chunks (chain-of-thought) before the answer.
            create_kwargs["stream"] = True
            create_kwargs["top_p"] = 1
            stream = self._create(client, create_kwargs)
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

        resp = self._create(client, create_kwargs)
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
                    f"max_tokens={self._call_max_tokens()} budget without producing an answer "
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

    @staticmethod
    def _is_rate_limit(status: Any, msg: str) -> bool:
        return status == 429 or (status is None and "rate limit" in msg)

    def _create(self, client: Any, create_kwargs: dict[str, Any]) -> Any:
        """chat.completions.create, retried without an optional parameter
        the endpoint rejects (reasoning_effort, JSON mode) rather than
        failing the call. Also handles 429 rate limit backoff adaptively.
        Each retry removes a key or sleeps backoff, so this terminates."""
        dropped_json = False
        max_429_retries = 3
        attempt_429 = 0
        is_stream = bool(create_kwargs.get("stream", False))

        while True:
            try:
                if hasattr(client.chat.completions, "with_raw_response") and not is_stream:
                    raw_resp = client.chat.completions.with_raw_response.create(**create_kwargs)
                    resp = raw_resp.parse()
                    headers = getattr(raw_resp, "headers", None)
                    if headers:
                        get_limiter().update_from_headers(self.provider, self.model, headers)
                else:
                    resp = client.chat.completions.create(**create_kwargs)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                msg = str(exc).lower()

                if self._is_rate_limit(status, msg) and attempt_429 < max_429_retries:
                    attempt_429 += 1
                    backoff = get_limiter().handle_429(self.provider, self.model, exc)
                    if backoff is None:
                        raise
                    time.sleep(backoff)
                    continue

                # Google (or any endpoint) rejecting reasoning_effort for this model.
                if "reasoning_effort" in create_kwargs and "reasoning_effort" in msg:
                    create_kwargs.pop("reasoning_effort")
                    continue
                if "response_format" in create_kwargs and (
                    getattr(exc, "status_code", None) in (400, 422)
                    # OpenRouter: 404 "No endpoints found that can handle the requested parameters".
                    or any(k in msg for k in ("response_format", "json_object", "json mode", "requested parameters"))
                ):
                    create_kwargs.pop("response_format")
                    extra = create_kwargs.get("extra_body")
                    if isinstance(extra, dict):
                        extra.pop("format", None)
                        if not extra:
                            create_kwargs.pop("extra_body")
                    dropped_json = True
                    continue
                raise
            if dropped_json:
                self._json_mode = False
            return resp

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        import anthropic
        from anthropic.types import TextBlock

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key set for provider 'anthropic'. Set ANTHROPIC_API_KEY.")
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout, max_retries=2)
        client = self._client

        estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4 + self._call_max_tokens()
        get_limiter().acquire(self.provider, self.model, estimated_tokens=estimated_tokens)

        # P1.6(b) — the tool-description/system block is byte-identical
        # across all ~15 calls in a run; mark it for prompt caching so it's
        # billed once instead of on every iteration.
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._call_max_tokens(),
            "system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_prompt}],
        }

        max_429_retries = 3
        attempt_429 = 0
        while True:
            try:
                if hasattr(client.messages, "with_raw_response"):
                    raw_resp = client.messages.with_raw_response.create(**create_kwargs)
                    msg = raw_resp.parse()
                    headers = getattr(raw_resp, "headers", None)
                    if headers:
                        get_limiter().update_from_headers(self.provider, self.model, headers)
                else:
                    msg = client.messages.create(**create_kwargs)
                break
            except Exception as exc:
                if self._is_rate_limit(getattr(exc, "status_code", None), str(exc).lower()) and attempt_429 < max_429_retries:
                    attempt_429 += 1
                    backoff = get_limiter().handle_429(self.provider, self.model, exc)
                    if backoff is None:
                        raise
                    time.sleep(backoff)
                    continue
                raise

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

    def get_context_window(self) -> int:
        """Context window for the active model; LLM_CONTEXT_TOKENS, when set, wins."""
        raw = os.getenv("LLM_CONTEXT_TOKENS", "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        return get_limiter().get_profile(self.provider, self.model).context_window

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
