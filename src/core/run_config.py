"""
RunConfig: the LLM connection settings for ONE run, passed explicitly.

The Streamlit app used to publish these through ``os.environ`` (provider,
model, API key, local-server URL, reasoning effort). On a shared server that
leaks between sessions: one visitor's key could end up used for another
visitor's run. A ``RunConfig`` is built per run and handed to ``LLMClient``
directly, so nothing about it is process-wide.

Every field is optional. A field left as ``None`` falls back to the same
environment variable ``LLMClient`` always read, which keeps the CLI, tests and
scripts working unchanged.

The API key is excluded from ``repr`` so it cannot reach a log line or a
traceback through string formatting.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RunConfig:
    """LLM connection settings for one run. ``None`` means "use the environment"."""

    provider: str | None = None
    model: str | None = None
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    reasoning_effort: str | None = None
