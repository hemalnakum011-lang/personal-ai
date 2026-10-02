"""Offline tests for agent.llm using a fake OpenAI client."""

import logging
from types import SimpleNamespace

import openai
import pytest
from openai.types.chat import ChatCompletion

from agent import llm
from agent.config import PROVIDERS

PRESET = PROVIDERS["nvidia_build"]


def make_completion(content, *, finish_reason="stop", reasoning=None, model="m"):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        }
    )


def fake_response(status):
    # APIStatusError only reads .request, .status_code and .headers.
    return SimpleNamespace(request=None, status_code=status, headers={})


def status_error(status):
    cls = {400: openai.BadRequestError, 404: openai.NotFoundError, 503: openai.InternalServerError}
    return cls[status]("nope", response=fake_response(status), body=None)


class FakeClient:
    """Plays back queued results; records the model of each call."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, **kwargs})
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def fake(clean_env):
    clean_env.setenv("PROVIDER", "nvidia_build")
    clean_env.setenv("LLM_API_KEY", "test")

    def install(*results):
        client = FakeClient(results)
        clean_env.setattr(llm, "get_client", lambda: client)
        return client

    return install


MSG = [{"role": "user", "content": "hi"}]


def test_reply_is_answer_only_with_reasoning_attribute(fake):
    fake(make_completion("The answer.", reasoning="secret chain of thought"))
    reply = llm.chat("super", MSG)
    assert reply == "The answer."
    assert isinstance(reply, str)
    assert reply.reasoning == "secret chain of thought"
    assert "secret" not in reply


def test_inline_think_tags_are_stripped(fake):
    fake(make_completion("<think>step 1\nstep 2</think>\n\nFinal."))
    reply = llm.chat("super", MSG)
    assert reply == "Final."
    assert reply.reasoning == "step 1\nstep 2"


def test_think_without_opening_tag_is_stripped(fake):
    fake(make_completion("pondering...</think>Final."))
    reply = llm.chat("super", MSG)
    assert reply == "Final."
    assert reply.reasoning == "pondering..."


def test_default_max_tokens_is_4096(fake):
    client = fake(make_completion("ok"))
    llm.chat("super", MSG)
    assert client.calls[0]["max_tokens"] == 4096


def test_length_finish_reason_logs_warning(fake, caplog):
    fake(make_completion("cut off mid", finish_reason="length"))
    with caplog.at_level(logging.WARNING, logger="agent.llm"):
        reply = llm.chat("super", MSG, max_tokens=10)
    assert reply.finish_reason == "length"
    assert "truncated" in caplog.text


@pytest.mark.parametrize("status", [404, 503])
def test_nano_falls_back_to_super(fake, caplog, status):
    client = fake(status_error(status), make_completion("from super"))
    with caplog.at_level(logging.WARNING, logger="agent.llm"):
        reply = llm.chat("nano", MSG)
    assert reply == "from super"
    assert reply.model == PRESET.super_model
    assert [c["model"] for c in client.calls] == [PRESET.nano_model, PRESET.super_model]
    assert f"returned {status}" in caplog.text


def test_fallback_logged_once_and_404_model_skipped_after(fake, caplog):
    client = fake(status_error(404), make_completion("a"), make_completion("b"))
    with caplog.at_level(logging.WARNING, logger="agent.llm"):
        llm.chat("nano", MSG)
        llm.chat("nano", MSG)
    assert caplog.text.count("falling back") == 1
    # Second call goes straight to super; nano is not retried.
    assert [c["model"] for c in client.calls] == [
        PRESET.nano_model,
        PRESET.super_model,
        PRESET.super_model,
    ]


def test_503_is_retried_on_next_call(fake, caplog):
    client = fake(status_error(503), make_completion("a"), make_completion("b"))
    with caplog.at_level(logging.WARNING, logger="agent.llm"):
        llm.chat("nano", MSG)
        llm.chat("nano", MSG)
    assert caplog.text.count("falling back") == 1
    assert [c["model"] for c in client.calls] == [
        PRESET.nano_model,
        PRESET.super_model,
        PRESET.nano_model,
    ]


def test_super_failure_is_not_swallowed(fake):
    fake(status_error(503))
    with pytest.raises(openai.InternalServerError):
        llm.chat("super", MSG)


def test_other_errors_do_not_fall_back(fake):
    client = fake(status_error(400))
    with pytest.raises(openai.BadRequestError):
        llm.chat("nano", MSG)
    assert len(client.calls) == 1
