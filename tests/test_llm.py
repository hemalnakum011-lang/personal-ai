"""Live test against the configured provider. Run with: uv run pytest -m live -s"""

import pytest

from agent.llm import chat, get_settings

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def require_api_key():
    if not get_settings().llm_api_key:
        pytest.skip("LLM_API_KEY not set in .env")


def test_chat_real_call():
    settings = get_settings()
    reply = chat(
        None,
        [{"role": "user", "content": "Reply with one short sentence: what model are you?"}],
    )
    print(
        f"\n[{settings.provider} | {reply.model} | finish={reply.finish_reason} "
        f"| reasoning={len(reply.reasoning)} chars hidden]\n{reply}"
    )
    assert reply.strip()
    assert reply.finish_reason == "stop"
    assert "<think>" not in reply and "</think>" not in reply
