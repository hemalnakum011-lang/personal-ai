"""Minimal Nemotron client for OpenAI-compatible APIs (NVIDIA build, Nebius Token Factory)."""

import logging
from functools import lru_cache
from typing import Any

from openai import APIStatusError, OpenAI

from agent.config import Settings

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 4096
FALLBACK_ALIAS = "super"
# 404: model not enabled for this account. 503: model overloaded.
FALLBACK_STATUS_CODES = {404, 503}

# Models that returned 404 are skipped for the rest of the process.
_unavailable_models: set[str] = set()
_logged_fallbacks: set[tuple[str, int]] = set()


class Reply(str):
    """The assistant's final answer as a plain string.

    Reasoning is kept on `.reasoning` and must never be shown to the user.
    """

    reasoning: str
    tool_calls: list[Any]
    finish_reason: str | None
    model: str

    def __new__(
        cls,
        content: str,
        *,
        reasoning: str = "",
        tool_calls: list[Any] | None = None,
        finish_reason: str | None = None,
        model: str = "",
    ) -> "Reply":
        reply = super().__new__(cls, content)
        reply.reasoning = reasoning
        reply.tool_calls = tool_calls or []
        reply.finish_reason = finish_reason
        reply.model = model
        return reply


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    settings = get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


def resolve_model(model: str | None) -> str:
    """Map an alias ("nano", "super", "ultra") to its model ID; pass full IDs through."""
    settings = get_settings()
    aliases = {
        "nano": settings.nemotron_nano_model,
        "super": settings.nemotron_super_model,
        "ultra": settings.nemotron_ultra_model,
    }
    name = model or settings.default_model
    return aliases.get(name.lower(), name)


def _split_reasoning(message: Any) -> tuple[str, str]:
    """Return (answer, reasoning), pulling reasoning out of fields or <think> tags."""
    extra = message.model_extra or {}
    reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""
    content = message.content or ""
    if "</think>" in content:
        # Inline reasoning precedes the answer; the opening tag is sometimes omitted.
        head, content = content.rsplit("</think>", 1)
        inline = head.replace("<think>", "").replace("</think>", "").strip()
        reasoning = "\n".join(part for part in (reasoning, inline) if part)
    return content.strip(), reasoning


def _log_fallback_once(model_id: str, status: int, fallback_id: str) -> None:
    if (model_id, status) in _logged_fallbacks:
        return
    _logged_fallbacks.add((model_id, status))
    log.warning("Model %s returned %s; falling back to %s", model_id, status, fallback_id)


def chat(
    model: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Reply:
    """Send one chat completion request and return the final answer.

    `model` is an alias ("nano", "super", "ultra"), a full model ID, or None for
    NEMOTRON_DEFAULT_MODEL. If the model returns 404 or 503, the call is retried
    on the "super" model. Extra kwargs (temperature, tool_choice, ...) go
    straight to the API.
    """
    kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    if tools:
        kwargs["tools"] = tools

    model_id = resolve_model(model)
    fallback_id = resolve_model(FALLBACK_ALIAS)
    if model_id in _unavailable_models:
        model_id = fallback_id

    client = get_client()
    try:
        response = client.chat.completions.create(model=model_id, messages=messages, **kwargs)
    except APIStatusError as err:
        if err.status_code not in FALLBACK_STATUS_CODES or model_id == fallback_id:
            raise
        if err.status_code == 404:
            _unavailable_models.add(model_id)
        _log_fallback_once(model_id, err.status_code, fallback_id)
        model_id = fallback_id
        response = client.chat.completions.create(model=model_id, messages=messages, **kwargs)

    choice = response.choices[0]
    if choice.finish_reason == "length":
        log.warning(
            "Reply from %s hit max_tokens=%s and is truncated", model_id, kwargs["max_tokens"]
        )
    content, reasoning = _split_reasoning(choice.message)
    return Reply(
        content,
        reasoning=reasoning,
        tool_calls=choice.message.tool_calls or [],
        finish_reason=choice.finish_reason,
        model=model_id,
    )
