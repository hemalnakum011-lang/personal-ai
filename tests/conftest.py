import pytest

from agent import llm

LLM_ENV_VARS = [
    "PROVIDER",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "NEBIUS_API_KEY",
    "NEBIUS_BASE_URL",
    "NEMOTRON_NANO_MODEL",
    "NEMOTRON_SUPER_MODEL",
    "NEMOTRON_ULTRA_MODEL",
    "NEMOTRON_DEFAULT_MODEL",
]


def _reset_llm_state(get_settings, get_client):
    get_settings.cache_clear()
    get_client.cache_clear()
    llm._unavailable_models.clear()
    llm._logged_fallbacks.clear()


@pytest.fixture
def clean_env(monkeypatch):
    """Isolate a test from the real .env and from cached settings/clients."""
    # Keep the real cached functions: tests may monkeypatch llm.get_client.
    originals = (llm.get_settings, llm.get_client)
    for key in LLM_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("agent.config.load_dotenv", lambda *a, **k: None)
    _reset_llm_state(*originals)
    yield monkeypatch
    _reset_llm_state(*originals)
