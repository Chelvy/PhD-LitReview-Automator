"""
Shared LLM factory for all pipeline agents.

claude-opus-4.x models do not accept a `temperature` parameter (extended-thinking
models use a fixed internal temperature). All other current Claude models do.
Centralising the build here avoids silent 400 errors across every agent.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from config.settings import settings

# Models that reject the temperature parameter
_NO_TEMPERATURE_PREFIXES = ("claude-opus-4",)


def build_llm(
    temperature: float = 0.2,
    max_tokens: int = 4096,
    model: str | None = None,
) -> ChatAnthropic:
    """Return a ChatAnthropic instance, omitting temperature for unsupported models."""
    _model = model or settings.anthropic_model
    kwargs: dict = {
        "model": _model,
        "api_key": settings.anthropic_api_key,
        "max_tokens": max_tokens,
    }
    if not any(_model.startswith(p) for p in _NO_TEMPERATURE_PREFIXES):
        kwargs["temperature"] = temperature
    return ChatAnthropic(**kwargs)
