"""
Model Telemetry & Dynamic Governance Engine.

Tracks, discovers, and enforces rate limits, token quotas, context windows,
and capability profiles dynamically per (provider, model) pair.

Workflow coverage:
- Dynamic model discovery from provider APIs (/v1/models, Gemini catalog, Ollama tags)
- Proactive HTTP response header inspection (x-ratelimit-*, anthropic-ratelimit-*, retry-after)
- Sliding-window rate limiter (RPM & TPM) with env overrides
- Bounded, shared 429 cooldown
- Context window calibration for PromptManager
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass
class ModelCapabilityProfile:
    """Capability and quota telemetry for a specific provider/model."""

    provider: str
    model: str
    context_window: int = 16_000
    max_output_tokens: int = 4_096
    rpm_limit: int | None = None
    rpm_remaining: int | None = None
    rpm_reset_seconds: float | None = None
    tpm_limit: int | None = None
    tpm_remaining: int | None = None
    tpm_reset_seconds: float | None = None
    daily_requests_remaining: int | None = None
    description: str = ""
    speed_tag: str = "Standard"  # "Fast" | "Ultra Fast" | "Deep" | "Standard"
    is_free: bool = False

    def format_dropdown_label(self) -> str:
        """Format human-readable label with context and limits for UI dropdowns."""
        parts: list[str] = []
        if self.context_window >= 1_000_000:
            parts.append(f"{self.context_window // 1_000_000}M ctx")
        elif self.context_window >= 1_000:
            parts.append(f"{self.context_window // 1_000}k ctx")

        if self.rpm_limit is not None:
            parts.append(f"{self.rpm_limit} RPM")
        if self.tpm_limit is not None:
            if self.tpm_limit >= 1_000_000:
                parts.append(f"{self.tpm_limit // 1_000_000}M TPM")
            elif self.tpm_limit >= 1_000:
                parts.append(f"{self.tpm_limit // 1_000}k TPM")

        if self.speed_tag in ("Fast", "Ultra Fast"):
            parts.append(f"⚡ {self.speed_tag}")
        elif self.speed_tag == "Deep":
            parts.append(f"🧠 {self.speed_tag}")

        if self.is_free or ":free" in self.model.lower():
            parts.append("🆓 Free")
        elif self.provider in ("openai", "anthropic") or (self.provider == "openrouter" and not self.is_free):
            parts.append("💳 Paid")

        suffix = f"  ({ ' · '.join(parts) })" if parts else ""
        return f"{self.model}{suffix}"


# Pre-seeded baseline catalog for zero-config startup and offline fallback
DEFAULT_PROFILES: dict[tuple[str, str], ModelCapabilityProfile] = {
    # Gemini
    ("gemini", "gemini-2.5-flash"): ModelCapabilityProfile(
        provider="gemini",
        model="gemini-2.5-flash",
        context_window=1_048_576,
        max_output_tokens=8_192,
        rpm_limit=15,
        tpm_limit=1_000_000,
        speed_tag="Fast",
        is_free=True,
        description="Google's flagship fast multimodal & reasoning model with 1M context",
    ),
    ("gemini", "gemini-flash-latest"): ModelCapabilityProfile(
        provider="gemini",
        model="gemini-flash-latest",
        context_window=1_048_576,
        max_output_tokens=8_192,
        rpm_limit=15,
        tpm_limit=1_000_000,
        speed_tag="Fast",
        is_free=True,
    ),
    ("gemini", "gemini-2.5-pro"): ModelCapabilityProfile(
        provider="gemini",
        model="gemini-2.5-pro",
        context_window=2_097_152,
        max_output_tokens=8_192,
        rpm_limit=2,
        tpm_limit=32_000,
        speed_tag="Deep",
        is_free=True,
        description="Google's highest-intelligence model with deep thinking (2 RPM free tier)",
    ),
    ("gemini", "gemini-3.8-flash"): ModelCapabilityProfile(
        provider="gemini",
        model="gemini-3.8-flash",
        context_window=1_048_576,
        max_output_tokens=8_192,
        rpm_limit=15,
        speed_tag="Fast",
        is_free=True,
    ),
    # Groq (LPU Inference)
    ("groq", "llama-3.3-70b-versatile"): ModelCapabilityProfile(
        provider="groq",
        model="llama-3.3-70b-versatile",
        context_window=128_000,
        max_output_tokens=4_096,
        rpm_limit=30,
        tpm_limit=6_000,
        speed_tag="Fast",
        is_free=True,
        description="Meta's 70B state-of-the-art model on Groq LPUs",
    ),
    ("groq", "llama-3.1-8b-instant"): ModelCapabilityProfile(
        provider="groq",
        model="llama-3.1-8b-instant",
        context_window=128_000,
        max_output_tokens=4_096,
        rpm_limit=30,
        tpm_limit=20_000,
        speed_tag="Ultra Fast",
        is_free=True,
        description="Meta's lightweight 8B model on Groq LPUs (ultra fast)",
    ),
    ("groq", "mixtral-8x7b-32768"): ModelCapabilityProfile(
        provider="groq",
        model="mixtral-8x7b-32768",
        context_window=32_768,
        max_output_tokens=4_096,
        rpm_limit=30,
        tpm_limit=5_000,
        speed_tag="Fast",
        is_free=True,
        description="Mistral 8x7B MoE on Groq LPUs",
    ),
    # OpenAI
    ("openai", "gpt-4o"): ModelCapabilityProfile(
        provider="openai",
        model="gpt-4o",
        context_window=128_000,
        max_output_tokens=4_096,
        rpm_limit=10_000,
        tpm_limit=30_000_000,
        speed_tag="Deep",
        description="OpenAI's flagship omni model",
    ),
    ("openai", "gpt-4o-mini"): ModelCapabilityProfile(
        provider="openai",
        model="gpt-4o-mini",
        context_window=128_000,
        max_output_tokens=4_096,
        rpm_limit=10_000,
        tpm_limit=50_000_000,
        speed_tag="Fast",
        description="OpenAI's fast, affordable small model",
    ),
    ("openai", "gpt-4-turbo"): ModelCapabilityProfile(
        provider="openai",
        model="gpt-4-turbo",
        context_window=128_000,
        max_output_tokens=4_096,
        speed_tag="Deep",
    ),
    ("openai", "gpt-3.5-turbo"): ModelCapabilityProfile(
        provider="openai",
        model="gpt-3.5-turbo",
        context_window=16_385,
        max_output_tokens=4_096,
        speed_tag="Fast",
    ),
    # Anthropic
    ("anthropic", "claude-sonnet-4-6"): ModelCapabilityProfile(
        provider="anthropic",
        model="claude-sonnet-4-6",
        context_window=200_000,
        max_output_tokens=8_192,
        speed_tag="Deep",
        description="Anthropic's balanced frontier reasoning model",
    ),
    ("anthropic", "claude-opus-4-8"): ModelCapabilityProfile(
        provider="anthropic",
        model="claude-opus-4-8",
        context_window=200_000,
        max_output_tokens=8_192,
        speed_tag="Deep",
        description="Anthropic's maximum-depth intelligence model",
    ),
    ("anthropic", "claude-haiku-4-5-20251001"): ModelCapabilityProfile(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        context_window=200_000,
        max_output_tokens=8_192,
        speed_tag="Fast",
        description="Anthropic's fastest lightweight model",
    ),
    # NVIDIA NIM
    ("nvidia", "openai/gpt-oss-120b"): ModelCapabilityProfile(
        provider="nvidia",
        model="openai/gpt-oss-120b",
        context_window=128_000,
        max_output_tokens=4_096,
        speed_tag="Deep",
    ),
    ("nvidia", "meta/llama-3.3-70b-instruct"): ModelCapabilityProfile(
        provider="nvidia",
        model="meta/llama-3.3-70b-instruct",
        context_window=128_000,
        max_output_tokens=4_096,
        speed_tag="Fast",
    ),
    ("nvidia", "meta/llama-3.1-70b-instruct"): ModelCapabilityProfile(
        provider="nvidia",
        model="meta/llama-3.1-70b-instruct",
        context_window=128_000,
        max_output_tokens=4_096,
        speed_tag="Fast",
    ),
    ("nvidia", "deepseek-ai/deepseek-r1"): ModelCapabilityProfile(
        provider="nvidia",
        model="deepseek-ai/deepseek-r1",
        context_window=64_000,
        max_output_tokens=8_192,
        speed_tag="Deep",
    ),
    # OpenRouter defaults — Free models
    ("openrouter", "nvidia/nemotron-3.5-lightning:free"): ModelCapabilityProfile(
        provider="openrouter",
        model="nvidia/nemotron-3.5-lightning:free",
        context_window=262_144,
        speed_tag="Ultra Fast",
        is_free=True,
        description="NVIDIA Nemotron 3.5 Lightning (Free Tier on OpenRouter)",
    ),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"): ModelCapabilityProfile(
        provider="openrouter",
        model="meta-llama/llama-3.3-70b-instruct:free",
        context_window=128_000,
        speed_tag="Fast",
        is_free=True,
        description="Meta Llama 3.3 70B Instruct (Free Tier on OpenRouter)",
    ),
    ("openrouter", "google/gemini-2.0-flash-exp:free"): ModelCapabilityProfile(
        provider="openrouter",
        model="google/gemini-2.0-flash-exp:free",
        context_window=1_048_576,
        speed_tag="Fast",
        is_free=True,
        description="Google Gemini 2.0 Flash Exp (Free Tier on OpenRouter)",
    ),
    ("openrouter", "qwen/qwen-2.5-72b-instruct:free"): ModelCapabilityProfile(
        provider="openrouter",
        model="qwen/qwen-2.5-72b-instruct:free",
        context_window=32_768,
        speed_tag="Fast",
        is_free=True,
        description="Qwen 2.5 72B Instruct (Free Tier on OpenRouter)",
    ),
    # OpenRouter defaults — Paid models
    ("openrouter", "anthropic/claude-3.7-sonnet"): ModelCapabilityProfile(
        provider="openrouter",
        model="anthropic/claude-3.7-sonnet",
        context_window=200_000,
        speed_tag="Deep",
        is_free=False,
        description="Anthropic's hybrid reasoning model (Paid)",
    ),
    ("openrouter", "openai/gpt-4o"): ModelCapabilityProfile(
        provider="openrouter",
        model="openai/gpt-4o",
        context_window=128_000,
        speed_tag="Deep",
        is_free=False,
    ),
    ("openrouter", "google/gemini-2.5-flash"): ModelCapabilityProfile(
        provider="openrouter",
        model="google/gemini-2.5-flash",
        context_window=1_048_576,
        speed_tag="Fast",
        is_free=False,
    ),
    ("openrouter", "deepseek/deepseek-chat"): ModelCapabilityProfile(
        provider="openrouter",
        model="deepseek/deepseek-chat",
        context_window=64_000,
        speed_tag="Fast",
        is_free=False,
    ),
    # Local defaults (100% Free)
    ("local", "llama3.1"): ModelCapabilityProfile(
        provider="local", model="llama3.1", context_window=128_000, speed_tag="Fast", is_free=True
    ),
    ("local", "llama3.2"): ModelCapabilityProfile(
        provider="local", model="llama3.2", context_window=128_000, speed_tag="Fast", is_free=True
    ),
    ("local", "mistral"): ModelCapabilityProfile(
        provider="local", model="mistral", context_window=32_768, speed_tag="Fast", is_free=True
    ),
    ("local", "qwen2.5"): ModelCapabilityProfile(
        provider="local", model="qwen2.5", context_window=32_768, speed_tag="Fast", is_free=True
    ),
    ("local", "phi4"): ModelCapabilityProfile(
        provider="local", model="phi4", context_window=16_384, speed_tag="Fast", is_free=True
    ),
}


def _infer_speed_tag(model_id: str) -> str:
    lower = model_id.lower()
    is_mini = "mini" in lower and "gemini" not in lower
    if is_mini or any(k in lower for k in ("instant", "8b", "flash-8b", "haiku")):
        return "Ultra Fast"
    if any(k in lower for k in ("flash", "fast", "3b", "7b", "8x7b", "chat")):
        return "Fast"
    if any(k in lower for k in ("r1", "pro", "opus", "70b", "120b", "405b", "deepseek")):
        return "Deep"
    return "Standard"


def _infer_context_window(model_id: str, default: int = 16_000) -> int:
    lower = model_id.lower()
    if "gemini" in lower:
        return 2_097_152 if "pro" in lower else 1_048_576
    if any(k in lower for k in ("llama-3.3", "llama-3.1", "gpt-4o", "gpt-4-turbo", "gpt-4.1", "gpt-5", "gpt-oss")):
        return 128_000
    if any(k in lower for k in ("claude-3", "claude-sonnet", "claude-opus", "claude-haiku")):
        return 200_000
    if "deepseek" in lower:
        return 64_000
    if any(k in lower for k in ("32k", "32768")):
        return 32_768
    if any(k in lower for k in ("16k", "16384")):
        return 16_384
    if any(k in lower for k in ("8k", "8192")):
        return 8_192
    return default


def fetch_available_models(
    provider: str,
    api_key: str = "",
    base_url: str | None = None,
    timeout_s: float = 4.0,
) -> list[ModelCapabilityProfile]:
    """
    Dynamically discover available models from a provider's API.

    Returns a list of ModelCapabilityProfile objects populated with discovered
    model IDs, context windows, and rate limit hints. Falls back gracefully
    to DEFAULT_PROFILES if offline or unauthenticated.
    """
    if provider == "gemini":
        if not api_key:
            return [p for (pr, _), p in DEFAULT_PROFILES.items() if pr == "gemini"]
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            req = urllib.request.Request(url, headers={"User-Agent": "dsa-agent/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models_data = data.get("models", [])
            profiles: list[ModelCapabilityProfile] = []
            for m in models_data:
                name = m.get("name", "").removeprefix("models/")
                # Filter to chat/generateContent models
                methods = m.get("supportedGenerationMethods", [])
                if methods and "generateContent" not in methods:
                    continue
                if not any(k in name for k in ("gemini", "flash", "pro")):
                    continue
                in_limit = m.get("inputTokenLimit") or _infer_context_window(name, 1_048_576)
                out_limit = m.get("outputTokenLimit") or 8_192
                known = DEFAULT_PROFILES.get(("gemini", name))
                rpm = known.rpm_limit if known else (2 if "pro" in name else 15)
                profiles.append(
                    ModelCapabilityProfile(
                        provider="gemini",
                        model=name,
                        context_window=in_limit,
                        max_output_tokens=out_limit,
                        rpm_limit=rpm,
                        speed_tag=_infer_speed_tag(name),
                        is_free=True,
                        description=m.get("description", ""),
                    )
                )
            if profiles:
                # Sort: flash models first, then pro
                profiles.sort(key=lambda p: (0 if "flash" in p.model else 1, p.model))
                return profiles
        except Exception:
            pass
        return [p for (pr, _), p in DEFAULT_PROFILES.items() if pr == "gemini"]

    # OpenAI-compatible /v1/models endpoints (Groq, OpenAI, OpenRouter, NVIDIA, Local)
    endpoint = ""
    auth_header = {}
    if provider == "groq":
        endpoint = "https://api.groq.com/openai/v1/models"
        if api_key:
            auth_header = {"Authorization": f"Bearer {api_key}"}
    elif provider == "openai":
        endpoint = "https://api.openai.com/v1/models"
        if api_key:
            auth_header = {"Authorization": f"Bearer {api_key}"}
    elif provider == "openrouter":
        endpoint = "https://openrouter.ai/api/v1/models"
        if api_key:
            auth_header = {"Authorization": f"Bearer {api_key}"}
    elif provider == "nvidia":
        endpoint = "https://integrate.api.nvidia.com/v1/models"
        if api_key:
            auth_header = {"Authorization": f"Bearer {api_key}"}
    elif provider in ("local", "ollama"):
        clean_base = (base_url or "http://localhost:11434/v1").rstrip("/")
        endpoint = f"{clean_base}/models"

    if endpoint and (api_key or provider in ("local", "ollama") or provider == "openrouter"):
        try:
            req = urllib.request.Request(
                endpoint,
                headers={"User-Agent": "dsa-agent/1.0", **auth_header},
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = data.get("data", [])
            profiles = []
            for item in items:
                model_id = item.get("id") or item.get("name")
                if not model_id or not isinstance(model_id, str):
                    continue
                # For OpenAI, only keep chat models
                if provider == "openai" and not any(k in model_id for k in ("gpt-4", "gpt-3.5", "o1", "o3")):
                    continue
                ctx = item.get("context_length") or _infer_context_window(model_id)
                known = DEFAULT_PROFILES.get((provider, model_id))
                rpm = known.rpm_limit if known else None
                tpm = known.tpm_limit if known else None

                # Determine pricing tier (free vs paid)
                if provider in ("groq", "local", "ollama"):
                    is_free = True
                elif provider == "openrouter":
                    pricing = item.get("pricing", {})
                    is_free_price = (
                        str(pricing.get("prompt", "1")) == "0"
                        and str(pricing.get("completion", "1")) == "0"
                    )
                    is_free = is_free_price or ":free" in model_id.lower()
                else:
                    is_free = False

                profiles.append(
                    ModelCapabilityProfile(
                        provider=provider,
                        model=model_id,
                        context_window=ctx,
                        rpm_limit=rpm,
                        tpm_limit=tpm,
                        speed_tag=_infer_speed_tag(model_id),
                        is_free=is_free,
                        description=item.get("description", ""),
                    )
                )
            if profiles:
                # Group free models first for openrouter, then sort alphabetically
                if provider == "openrouter":
                    profiles.sort(key=lambda p: (0 if p.is_free else 1, p.model))
                else:
                    profiles.sort(key=lambda p: p.model)
                return profiles
        except Exception:
            pass

    # Fallback to defaults
    return [p for (pr, _), p in DEFAULT_PROFILES.items() if pr == provider]


#: Longest single wait the limiter will impose. A provider asking for more
#: (daily quota exhausted) is surfaced as an error instead of stalling a run.
MAX_WAIT_S = 60.0

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)?")
_UNIT_SECONDS: dict[str | None, float] = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, None: 1.0, "": 1.0}


def _parse_duration(text: Any) -> float | None:
    """Seconds from '5', '26.5s', '1m26.4s', '20ms' (headers and error text)."""
    parts = _DURATION_PART.findall(str(text or "").strip().lower())
    if not parts:
        return None
    return sum(float(num) * _UNIT_SECONDS[unit] for num, unit in parts)


class DynamicRateAndTokenLimiter:
    """
    Thread-safe rate limiter over sliding 60-second RPM/TPM windows.

    Limits come from, in order: LLM_RPM_LIMIT / LLM_TPM_LIMIT (0 disables),
    provider response headers, the seeded catalog. A 429 puts the model on a
    shared cooldown instead of permanently lowering its ceiling.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._request_history: dict[tuple[str, str], deque[float]] = {}
        self._token_history: dict[tuple[str, str], deque[tuple[float, int]]] = {}
        self._cooldown_until: dict[tuple[str, str], float] = {}
        self._profiles: dict[tuple[str, str], ModelCapabilityProfile] = dict(DEFAULT_PROFILES)

    def get_profile(self, provider: str, model: str) -> ModelCapabilityProfile:
        """Get or create capability profile for a provider/model."""
        with self._lock:
            key = (provider, model)
            if key not in self._profiles:
                self._profiles[key] = ModelCapabilityProfile(
                    provider=provider,
                    model=model,
                    # Unknown hosted models are almost all >=32k; unknown local
                    # ones are often small, so stay conservative there.
                    context_window=_infer_context_window(
                        model, 16_000 if provider in ("local", "ollama") else 32_000
                    ),
                    speed_tag=_infer_speed_tag(model),
                )
            return self._profiles[key]

    @staticmethod
    def _env_limit(name: str) -> int | None:
        raw = os.getenv(name, "").strip()
        return int(raw) if raw.isdigit() else None

    def _caps(self, profile: ModelCapabilityProfile) -> tuple[int | None, int | None]:
        """Effective (rpm, tpm); an env value of 0 means unlimited."""
        rpm = self._env_limit("LLM_RPM_LIMIT")
        tpm = self._env_limit("LLM_TPM_LIMIT")
        return (
            (rpm or None) if rpm is not None else profile.rpm_limit,
            (tpm or None) if tpm is not None else profile.tpm_limit,
        )

    def acquire(self, provider: str, model: str, estimated_tokens: int = 500) -> float:
        """
        Pre-flight check: waits (at most MAX_WAIT_S) for RPM/TPM headroom or an
        active 429 cooldown, then books the request. Returns seconds slept.
        """
        key = (provider, model)
        rpm_cap, tpm_cap = self._caps(self.get_profile(provider, model))
        if tpm_cap:
            # A request larger than the whole budget can never fit; don't wait for it.
            estimated_tokens = min(estimated_tokens, tpm_cap)

        with self._lock:
            now = time.time()
            req_deq = self._request_history.setdefault(key, deque())
            tok_deq = self._token_history.setdefault(key, deque())
            while req_deq and now - req_deq[0] > 60.0:
                req_deq.popleft()
            while tok_deq and now - tok_deq[0][0] > 60.0:
                tok_deq.popleft()

            wait = max(0.0, self._cooldown_until.get(key, 0.0) - now)
            if rpm_cap and len(req_deq) >= rpm_cap:
                wait = max(wait, 60.0 - (now - req_deq[0]) + 0.15)
            if tpm_cap and tok_deq and sum(t[1] for t in tok_deq) + estimated_tokens > tpm_cap:
                wait = max(wait, 60.0 - (now - tok_deq[0][0]) + 0.15)
            wait = min(wait, MAX_WAIT_S)
            # Book the slot at the time we will actually fire, so concurrent
            # threads queue behind this one instead of all waking together.
            fire_at = now + wait
            req_deq.append(fire_at)
            tok_deq.append((fire_at, estimated_tokens))

        if wait > 0:
            time.sleep(wait)
        return wait

    def update_from_headers(
        self,
        provider: str,
        model: str,
        headers: Any,
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Refresh limits/remaining quota from OpenAI-style or Anthropic rate-limit headers."""
        if not headers or not hasattr(headers, "get"):
            return

        with self._lock:
            profile = self.get_profile(provider, model)
            get = headers.get
            if get("anthropic-ratelimit-requests-limit"):
                req_limit = get("anthropic-ratelimit-requests-limit")
                req_rem = get("anthropic-ratelimit-requests-remaining")
                tok_limit = get("anthropic-ratelimit-tokens-limit")
                tok_rem = get("anthropic-ratelimit-tokens-remaining")
            else:
                req_limit = get("x-ratelimit-limit-requests")
                req_rem = get("x-ratelimit-remaining-requests")
                tok_limit = get("x-ratelimit-limit-tokens")
                tok_rem = get("x-ratelimit-remaining-tokens")

            # Groq reports its *daily* request quota in x-ratelimit-limit-requests,
            # which is not an RPM ceiling.
            if req_limit and str(req_limit).isdigit() and provider != "groq":
                profile.rpm_limit = int(req_limit)
            if req_rem and str(req_rem).isdigit():
                profile.rpm_remaining = int(req_rem)
            if tok_limit and str(tok_limit).isdigit():
                profile.tpm_limit = int(tok_limit)
            if tok_rem and str(tok_rem).isdigit():
                profile.tpm_remaining = int(tok_rem)

            profile.rpm_reset_seconds = _parse_duration(get("x-ratelimit-reset-requests"))
            profile.tpm_reset_seconds = _parse_duration(get("x-ratelimit-reset-tokens"))

    def handle_429(self, provider: str, model: str, exc: Exception) -> float | None:
        """
        Register a 429 and return the seconds to wait before retrying, or None
        when the provider asks for longer than MAX_WAIT_S (quota exhausted —
        retrying cannot help). The wait is a cooldown shared by all threads.
        """
        msg = str(exc)
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None) if resp is not None else None
        wait: float | None = None
        if headers is not None and hasattr(headers, "get"):
            wait = _parse_duration(headers.get("retry-after"))
        if wait is None:
            m = re.search(r"(?:try again|retry) in\s+([0-9.hms]+)", msg, re.IGNORECASE)
            wait = _parse_duration(m.group(1)) if m else None
        m_rpm = re.search(r"quota:\s*(\d+)\s*RPM", msg, re.IGNORECASE)
        if m_rpm:
            with self._lock:
                self.get_profile(provider, model).rpm_limit = int(m_rpm.group(1))
        wait = 2.0 if wait is None else wait + 0.3
        if wait > MAX_WAIT_S:
            return None
        with self._lock:
            key = (provider, model)
            self._cooldown_until[key] = max(self._cooldown_until.get(key, 0.0), time.time() + wait)
        return wait


# Global singleton instance
_GLOBAL_LIMITER: DynamicRateAndTokenLimiter | None = None
_LIMITER_LOCK = RLock()


def get_limiter() -> DynamicRateAndTokenLimiter:
    """Retrieve or initialize the global DynamicRateAndTokenLimiter singleton."""
    global _GLOBAL_LIMITER
    with _LIMITER_LOCK:
        if _GLOBAL_LIMITER is None:
            _GLOBAL_LIMITER = DynamicRateAndTokenLimiter()
        return _GLOBAL_LIMITER
