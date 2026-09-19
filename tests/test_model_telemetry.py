"""
Unit tests for Model Telemetry & Dynamic Governance Engine.

Tests:
- ModelCapabilityProfile formatting
- Dynamic model discovery fallbacks and parsing
- DynamicRateAndTokenLimiter rate and token window acquisition
- Response header telemetry extraction (OpenAI, Groq, Anthropic)
- Adaptive 429 backoff calculation and rate ceiling reduction
- LLMClient dynamic context window calibration
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.core.llm_client import LLMClient
from src.core.model_telemetry import (
    DynamicRateAndTokenLimiter,
    ModelCapabilityProfile,
    _infer_context_window,
    _infer_speed_tag,
    _parse_duration,
    fetch_available_models,
)
from src.core.prompt_manager import PromptManager


class TestModelCapabilityProfile:
    def test_format_dropdown_label(self) -> None:
        p1 = ModelCapabilityProfile(
            provider="gemini",
            model="gemini-2.5-flash",
            context_window=1_048_576,
            rpm_limit=15,
            tpm_limit=1_000_000,
            speed_tag="Fast",
        )
        label1 = p1.format_dropdown_label()
        assert "gemini-2.5-flash" in label1
        assert "1M ctx" in label1
        assert "15 RPM" in label1
        assert "1M TPM" in label1
        assert "⚡ Fast" in label1

        p2 = ModelCapabilityProfile(
            provider="openai",
            model="gpt-4o",
            context_window=128_000,
            speed_tag="Deep",
        )
        label2 = p2.format_dropdown_label()
        assert "128k ctx" in label2
        assert "🧠 Deep" in label2

    def test_infer_speed_tag(self) -> None:
        assert _infer_speed_tag("llama-3.1-8b-instant") == "Ultra Fast"
        assert _infer_speed_tag("gemini-2.5-flash") == "Fast"
        assert _infer_speed_tag("deepseek-r1") == "Deep"
        assert _infer_speed_tag("unknown-model") == "Standard"

    def test_infer_context_window(self) -> None:
        assert _infer_context_window("gemini-2.5-pro") == 2_097_152
        assert _infer_context_window("gemini-2.5-flash") == 1_048_576
        assert _infer_context_window("gpt-4o") == 128_000
        assert _infer_context_window("claude-sonnet-4-6") == 200_000
        assert _infer_context_window("custom-32k") == 32_768
        assert _infer_context_window("custom-unknown", default=16000) == 16000


class TestModelDiscovery:
    def test_fetch_available_models_fallback(self) -> None:
        models = fetch_available_models("gemini", api_key="")
        assert len(models) >= 2
        assert any(m.model == "gemini-2.5-flash" for m in models)

        groq_models = fetch_available_models("groq", api_key="")
        assert len(groq_models) >= 1
        assert any("llama" in m.model for m in groq_models)
        assert all(m.is_free for m in groq_models)

        openrouter_models = fetch_available_models("openrouter", api_key="")
        assert any(m.is_free for m in openrouter_models)
        assert any(not m.is_free for m in openrouter_models)


class TestDynamicRateAndTokenLimiter:
    def test_acquire_unthrottled(self) -> None:
        limiter = DynamicRateAndTokenLimiter()
        # Initial request should not sleep
        slept = limiter.acquire("openai", "gpt-4o", estimated_tokens=100)
        assert slept == 0.0

    def test_acquire_rpm_throttling(self) -> None:
        limiter = DynamicRateAndTokenLimiter()
        # Set a tiny RPM cap
        profile = limiter.get_profile("test_provider", "test_model")
        profile.rpm_limit = 2

        slept1 = limiter.acquire("test_provider", "test_model", estimated_tokens=10)
        slept2 = limiter.acquire("test_provider", "test_model", estimated_tokens=10)
        assert slept1 == 0.0
        assert slept2 == 0.0

        # Third request should see exhausted RPM and simulate or compute sleep
        # We patch time.sleep to avoid actually waiting 60s
        with patch("time.sleep") as mock_sleep:
            slept3 = limiter.acquire("test_provider", "test_model", estimated_tokens=10)
            assert slept3 > 0.0
            mock_sleep.assert_called_once()

    def test_update_from_headers_openai(self) -> None:
        limiter = DynamicRateAndTokenLimiter()
        headers = {
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-remaining-requests": "495",
            "x-ratelimit-limit-tokens": "100000",
            "x-ratelimit-remaining-tokens": "95000",
            "x-ratelimit-reset-requests": "2.5s",
        }
        limiter.update_from_headers("openai", "gpt-4o", headers)
        profile = limiter.get_profile("openai", "gpt-4o")
        assert profile.rpm_limit == 500
        assert profile.rpm_remaining == 495
        assert profile.tpm_limit == 100000
        assert profile.tpm_remaining == 95000
        assert profile.rpm_reset_seconds == 2.5

    def test_update_from_headers_anthropic(self) -> None:
        limiter = DynamicRateAndTokenLimiter()
        headers = {
            "anthropic-ratelimit-requests-limit": "100",
            "anthropic-ratelimit-requests-remaining": "98",
            "anthropic-ratelimit-tokens-limit": "50000",
            "anthropic-ratelimit-tokens-remaining": "48000",
        }
        limiter.update_from_headers("anthropic", "claude-sonnet-4-6", headers)
        profile = limiter.get_profile("anthropic", "claude-sonnet-4-6")
        assert profile.rpm_limit == 100
        assert profile.rpm_remaining == 98
        assert profile.tpm_limit == 50000
        assert profile.tpm_remaining == 48000

    def test_handle_429_retry_after(self) -> None:
        limiter = DynamicRateAndTokenLimiter()
        mock_resp = MagicMock()
        mock_resp.headers = {"retry-after": "5"}
        exc = Exception("429 Too Many Requests")
        exc.response = mock_resp  # type: ignore[attr-defined]

        backoff = limiter.handle_429("gemini", "gemini-2.5-flash", exc)
        assert backoff >= 5.0

    def test_handle_429_regex_parsing(self) -> None:
        limiter = DynamicRateAndTokenLimiter()
        exc = Exception("Rate limit exceeded. Try again in 3.5s. Resource has quota: 15 RPM")
        backoff = limiter.handle_429("groq", "llama-3.3-70b-versatile", exc)
        assert backoff >= 3.5
        profile = limiter.get_profile("groq", "llama-3.3-70b-versatile")
        # Should adaptively adjust RPM
        assert profile.rpm_limit is not None and profile.rpm_limit <= 15


    def test_handle_429_does_not_lower_ceiling_permanently(self) -> None:
        limiter = DynamicRateAndTokenLimiter()
        profile = limiter.get_profile("openai", "gpt-4o")
        before = profile.rpm_limit
        limiter.handle_429("openai", "gpt-4o", Exception("429 Rate limit reached"))
        assert profile.rpm_limit == before

    def test_handle_429_gives_up_when_quota_exhausted(self) -> None:
        limiter = DynamicRateAndTokenLimiter()
        exc = Exception("Rate limit reached. Please try again in 1h2m3s.")
        assert limiter.handle_429("groq", "llama-3.3-70b-versatile", exc) is None

    def test_parse_duration(self) -> None:
        assert _parse_duration("5") == 5.0
        assert _parse_duration("1m26.4s") == 86.4
        assert _parse_duration("20ms") == 0.02
        assert _parse_duration("") is None


class TestLLMClientContextWindow:
    def test_get_context_window(self) -> None:
        client = LLMClient()
        client.provider = "gemini"
        client.model = "gemini-2.5-flash"
        ctx = client.get_context_window()
        assert ctx == 1_048_576

    def test_prompt_manager_dynamic_context(self) -> None:
        mem = MagicMock()
        mem.get_context.return_value = None
        pm = PromptManager(
            memory=mem,
            tool_descriptions="desc",
            context_tokens=1_000_000,
        )
        assert pm._budget_tokens() == 1_000_000 * 0.75
