"""
Shared LLM access for all pipeline agents.

- build_llm() returns a ResilientLLM: ChatAnthropic for the configured model plus a
  fallback model, used when the primary call raises (unknown model ID, unsupported
  parameter, outage after the SDK's own retries) or ends with stop_reason="refusal".
- message_text() extracts the reply text. Claude 5 models think by default, so
  AIMessage.content is a list of thinking + text blocks rather than a plain string.
- parse_json() pulls the JSON payload out of a reply (bare, fenced, or embedded in prose).

Request parameters differ by model generation:
- temperature: rejected with a 400 by every current model (Opus 4.7+, Sonnet 5.x,
  Haiku 5.5, Fable). Only Opus/Sonnet 4.6 and older accept it, so it is sent to those only.
- effort (output_config.effort): supported from Opus 4.5 / Sonnet 4.6 onward; Haiku 4.5
  and older reject it, so it is sent only to known-supporting models.
- max_tokens: the anthropic SDK refuses non-streaming requests above ~21k tokens.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence

import anthropic
import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage

from config.settings import settings

logger = structlog.get_logger(__name__)

_TEMPERATURE_MODEL_PREFIXES = (
    "claude-3",
    "claude-opus-4-6", "claude-sonnet-4-6",
    "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5",
    "claude-opus-4-1", "claude-opus-4-0", "claude-sonnet-4-0",
    "claude-opus-4-2025", "claude-sonnet-4-2025",
)

# Effort levels each model family accepts (xhigh arrived with Opus 4.7). Models not
# listed (Haiku 4.5 and older) reject the parameter, so it is not sent to them.
_ALL_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_EFFORT_LEVELS = (
    ("claude-opus-4-5", ("low", "medium", "high")),
    ("claude-opus-4-6", ("low", "medium", "high", "max")),
    ("claude-sonnet-4-6", ("low", "medium", "high", "max")),
    ("claude-opus-4-7", _ALL_EFFORTS),
    ("claude-opus-4-8", _ALL_EFFORTS),
    ("claude-opus-5", _ALL_EFFORTS),
    ("claude-sonnet-5", _ALL_EFFORTS),
    ("claude-haiku-5", _ALL_EFFORTS),
    ("claude-fable-5", _ALL_EFFORTS),
    ("claude-mythos-5", _ALL_EFFORTS),
)

MAX_NONSTREAMING_TOKENS = 20000

# After this many consecutive primary failures the primary model is skipped for the
# rest of the process, so a bad model ID doesn't double every call's latency.
_STICKY_FAILURE_THRESHOLD = 3
_primary_failures: dict[str, int] = {}


def supports_temperature(model: str) -> bool:
    return model.startswith(_TEMPERATURE_MODEL_PREFIXES)


def effort_for(model: str, effort: Optional[str]) -> Optional[str]:
    """The requested effort if the model supports it, else the nearest lower supported level."""
    if not effort or effort not in _ALL_EFFORTS:
        return None
    levels = next((lv for prefix, lv in _EFFORT_LEVELS if model.startswith(prefix)), None)
    if not levels:
        return None
    for candidate in reversed(_ALL_EFFORTS[: _ALL_EFFORTS.index(effort) + 1]):
        if candidate in levels:
            return candidate
    return None


def reset_failure_counts() -> None:
    """Forget primary-model failures (called at the start of every pipeline run)."""
    _primary_failures.clear()


def _make_chat_model(
    model: str,
    temperature: Optional[float],
    max_tokens: int,
    effort: Optional[str],
) -> ChatAnthropic:
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": settings.anthropic_api_key,
        "max_tokens": min(max_tokens, MAX_NONSTREAMING_TOKENS),
        "max_retries": 3,
        "default_request_timeout": 600,
    }
    if temperature is not None and supports_temperature(model):
        kwargs["temperature"] = temperature
    model_effort = effort_for(model, effort)
    if model_effort:
        kwargs["reasoning_effort"] = model_effort
    return ChatAnthropic(**kwargs)


def _stop_reason(response: Any) -> Optional[str]:
    metadata = getattr(response, "response_metadata", None) or {}
    return metadata.get("stop_reason")


class ResilientLLM:
    """ChatAnthropic wrapper that falls back to a second model on errors and refusals."""

    def __init__(self, primary: ChatAnthropic, fallback: Optional[ChatAnthropic]) -> None:
        self.primary = primary
        self.fallback = fallback

    @property
    def model(self) -> str:
        return self.primary.model

    def _use_fallback(self, messages: Sequence[BaseMessage], reason: str, **kwargs: Any) -> Any:
        logger.warning(
            "llm_fallback",
            primary=self.primary.model,
            fallback=self.fallback.model,
            reason=reason[:300],
        )
        return self.fallback.invoke(messages, **kwargs)

    def invoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
        primary_model = self.primary.model
        if self.fallback is not None and _primary_failures.get(primary_model, 0) >= _STICKY_FAILURE_THRESHOLD:
            return self.fallback.invoke(messages, **kwargs)

        try:
            response = self.primary.invoke(messages, **kwargs)
        except anthropic.AuthenticationError:
            raise  # same API key for both models: falling back cannot help
        except Exception as e:
            if self.fallback is None:
                raise
            response = self._use_fallback(messages, f"{type(e).__name__}: {e}", **kwargs)
            # Only a failure the fallback did not share points at the primary model itself
            _primary_failures[primary_model] = _primary_failures.get(primary_model, 0) + 1
            if _primary_failures[primary_model] == _STICKY_FAILURE_THRESHOLD:
                logger.error(
                    "llm_primary_disabled",
                    model=primary_model,
                    fallback=self.fallback.model,
                    hint="Check ANTHROPIC_MODEL and API access; using the fallback model for the rest of this run",
                )
            return response

        _primary_failures[primary_model] = 0
        if _stop_reason(response) == "refusal" and self.fallback is not None:
            return self._use_fallback(messages, "stop_reason=refusal", **kwargs)
        if _stop_reason(response) == "max_tokens":
            logger.warning("llm_output_truncated", model=primary_model, max_tokens=self.primary.max_tokens)
        return response


def build_llm(
    temperature: Optional[float] = 0.2,
    max_tokens: int = 16000,
    effort: Optional[str] = "high",
    model: Optional[str] = None,
) -> ResilientLLM:
    """Build the primary model client (settings.anthropic_model) with a fallback model."""
    primary_model = model or settings.anthropic_model
    effort = settings.anthropic_effort or effort
    primary = _make_chat_model(primary_model, temperature, max_tokens, effort)

    fallback = None
    fallback_model = settings.anthropic_fallback_model
    if fallback_model and fallback_model != primary_model:
        fallback = _make_chat_model(fallback_model, temperature, max_tokens, effort)
    return ResilientLLM(primary, fallback)


def message_text(message: Any) -> str:
    """Return the text of a chat model reply, ignoring thinking and other non-text blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "".join(parts).strip()
    return str(content or "").strip()


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json(text: str) -> Any:
    """
    Parse the JSON payload of a model reply.
    Tries the whole reply, then each ``` fenced block, then the first decodable
    object/array embedded in surrounding prose. Raises json.JSONDecodeError if none.
    """
    candidates = [text.strip()] + [m.strip() for m in _FENCE_RE.findall(text)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # JSON embedded in prose: take the largest top-level value, so a bracketed
    # reference like "[1]" in the prose is not mistaken for the payload.
    decoder = json.JSONDecoder()
    best, best_len, i = None, 0, 0
    while i < len(text):
        if text[i] in "{[":
            try:
                obj, end = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if end - i > best_len:
                best, best_len = obj, end - i
            i = end
        else:
            i += 1
    if best_len:
        return best
    raise json.JSONDecodeError("No JSON object or array found in model reply", text, 0)


class TruncatedReplyError(ValueError):
    """The reply stopped at max_tokens, so its JSON is incomplete."""


def invoke_json(llm: Any, messages: Sequence[BaseMessage]) -> Any:
    """Invoke the model and parse its reply as JSON. A truncated reply is an error, not partial data."""
    response = llm.invoke(messages)
    if _stop_reason(response) == "max_tokens":
        raise TruncatedReplyError("Model reply was cut off at max_tokens; its JSON is incomplete")
    return parse_json(message_text(response))


_SYSTEMIC_MARKERS = ("authentication_error", "permission_error", "not_found_error", "invalid x-api-key", "401", "403")


def is_systemic_error(message: str) -> bool:
    """Errors that will hit every call (bad key, no access to the model), not one paper."""
    lowered = message.lower()
    return any(marker in lowered for marker in _SYSTEMIC_MARKERS)


def text_field(record: dict[str, Any], key: str, default: str = "N/A", limit: Optional[int] = None) -> str:
    """
    Read a text field for prompt building. LLM extractions and some APIs store
    explicit nulls, so record.get(key, default) can still return None.
    """
    value = record.get(key)
    if value is None or value == "":
        value = default
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
    return value[:limit] if limit else value


def list_field(record: dict[str, Any], key: str) -> list[Any]:
    """Read a list field, tolerating explicit nulls and scalar values."""
    value = record.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
