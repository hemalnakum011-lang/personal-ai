"""Project paths and settings loaded from the environment and an optional .env file."""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str
    nano_model: str
    super_model: str
    ultra_model: str


# Each preset pairs a base URL with that provider's spelling of the Nemotron 3 IDs,
# so the URL and model IDs always match. IDs checked 2026-10-02.
PROVIDERS: dict[str, ProviderPreset] = {
    # From GET /models on integrate.api.nvidia.com.
    "nvidia_build": ProviderPreset(
        base_url="https://integrate.api.nvidia.com/v1",
        nano_model="nvidia/nemotron-nano-3-30b-a3b",
        super_model="nvidia/nemotron-3-super-120b-a12b",
        ultra_model="nvidia/nemotron-3-ultra-550b-a55b",
    ),
    # From the Nebius Token Factory model catalog. Serverless models use the
    # global endpoint; regional hosts are only for dedicated endpoints.
    "nebius": ProviderPreset(
        base_url="https://api.tokenfactory.nebius.com/v1/",
        nano_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        super_model="nvidia/nemotron-3-super-120b-a12b",
        ultra_model="nvidia/Nemotron-3-Ultra-550b-a55b",
    ),
}
DEFAULT_PROVIDER = "nebius"


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from a .env file without overriding existing env vars."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _env(*keys: str, default: str = "") -> str:
    """Return the first non-empty value among `keys`; empty values count as unset."""
    for key in keys:
        if value := os.environ.get(key):
            return value
    return default


def _provider_for_url(url: str) -> str | None:
    host = urlparse(url).hostname
    for name, preset in PROVIDERS.items():
        if urlparse(preset.base_url).hostname == host:
            return name
    return None


def resolve_provider(provider: str, base_url: str) -> tuple[str, ProviderPreset]:
    """Pick the preset, inferring it from the base URL when PROVIDER is unset.

    Raises ValueError if PROVIDER and the base URL point at different known providers.
    """
    url_provider = _provider_for_url(base_url) if base_url else None
    name = (provider or url_provider or DEFAULT_PROVIDER).lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown PROVIDER {name!r}; expected one of {sorted(PROVIDERS)}")
    if url_provider and url_provider != name:
        raise ValueError(
            f"PROVIDER={name} but the base URL {base_url} belongs to {url_provider}; "
            "model IDs would not match. Fix PROVIDER or LLM_BASE_URL."
        )
    return name, PROVIDERS[name]


@dataclass(frozen=True)
class Settings:
    provider: str
    llm_api_key: str
    llm_base_url: str
    nemotron_nano_model: str
    nemotron_super_model: str
    nemotron_ultra_model: str
    default_model: str
    memory_dir: Path
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        base_url_override = _env("LLM_BASE_URL", "NEBIUS_BASE_URL")
        provider, preset = resolve_provider(_env("PROVIDER"), base_url_override)
        memory_dir = Path(_env("MEMORY_DIR", default="memory/data"))
        if not memory_dir.is_absolute():
            memory_dir = PROJECT_ROOT / memory_dir
        return cls(
            provider=provider,
            llm_api_key=_env("LLM_API_KEY", "NEBIUS_API_KEY"),
            llm_base_url=base_url_override or preset.base_url,
            # NEMOTRON_*_MODEL are escape hatches for IDs newer than the presets.
            nemotron_nano_model=_env("NEMOTRON_NANO_MODEL", default=preset.nano_model),
            nemotron_super_model=_env("NEMOTRON_SUPER_MODEL", default=preset.super_model),
            nemotron_ultra_model=_env("NEMOTRON_ULTRA_MODEL", default=preset.ultra_model),
            default_model=_env("NEMOTRON_DEFAULT_MODEL", default="super"),
            memory_dir=memory_dir,
            log_level=_env("LOG_LEVEL", default="INFO"),
        )
