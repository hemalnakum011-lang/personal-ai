import os
from pathlib import Path

import pytest

from agent.config import PROJECT_ROOT, PROVIDERS, Settings, load_dotenv


def test_project_root_contains_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_load_dotenv_does_not_override(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nFOO_TEST=from_file\nBAR_TEST='quoted'\n", encoding="utf-8")
    monkeypatch.setenv("FOO_TEST", "from_env")
    monkeypatch.delenv("BAR_TEST", raising=False)

    load_dotenv(env_file)

    assert os.environ["FOO_TEST"] == "from_env"
    assert os.environ["BAR_TEST"] == "quoted"


def test_relative_memory_dir_resolves_under_project_root(clean_env):
    clean_env.setenv("MEMORY_DIR", "memory/data")
    assert Settings.from_env().memory_dir == PROJECT_ROOT / "memory" / "data"


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_preset_sets_matching_url_and_ids(clean_env, provider):
    clean_env.setenv("PROVIDER", provider)
    s = Settings.from_env()
    preset = PROVIDERS[provider]
    assert s.provider == provider
    assert s.llm_base_url == preset.base_url
    assert (s.nemotron_nano_model, s.nemotron_super_model, s.nemotron_ultra_model) == (
        preset.nano_model,
        preset.super_model,
        preset.ultra_model,
    )


def test_provider_inferred_from_base_url(clean_env):
    clean_env.setenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    s = Settings.from_env()
    assert s.provider == "nvidia_build"
    assert s.nemotron_nano_model == PROVIDERS["nvidia_build"].nano_model


def test_mismatched_provider_and_url_is_rejected(clean_env):
    clean_env.setenv("PROVIDER", "nebius")
    clean_env.setenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    with pytest.raises(ValueError, match="belongs to nvidia_build"):
        Settings.from_env()


def test_unknown_provider_is_rejected(clean_env):
    clean_env.setenv("PROVIDER", "openai")
    with pytest.raises(ValueError, match="Unknown PROVIDER"):
        Settings.from_env()


def test_llm_vars_take_precedence_over_nebius_fallbacks(clean_env):
    clean_env.setenv("NEBIUS_API_KEY", "nebius-key")
    assert Settings.from_env().llm_api_key == "nebius-key"
    clean_env.setenv("LLM_API_KEY", "llm-key")
    assert Settings.from_env().llm_api_key == "llm-key"


def test_unknown_base_url_keeps_explicit_provider(clean_env):
    clean_env.setenv("PROVIDER", "nvidia_build")
    clean_env.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    s = Settings.from_env()
    assert s.llm_base_url == "http://localhost:8000/v1"
    assert s.nemotron_super_model == PROVIDERS["nvidia_build"].super_model
