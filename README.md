# personal-ai

A personal AI agent built in Python, powered by NVIDIA Nemotron models running on Nebius.

> **Status:** skeleton — sections marked _TODO_ are still to be written.

## Project layout

| Path | Purpose |
|---|---|
| `agent/` | Core agent loop, prompting, tool dispatch |
| `memory/` | Long-term memory storage and retrieval |
| `skills/` | Self-contained capabilities the agent can call |
| `integrations/` | Connectors to external services |
| `deploy/` | Dockerfile, compose files, hosting config |
| `tests/` | pytest suite |

## Quick start (hosted)

_TODO: link to the hosted instance and describe sign-in._

1. Open the hosted app: `<URL>`
2. Sign in and …

## Self-host (Docker)

Requirements: Docker with Compose v2.

```bash
git clone <repo-url> personal-ai
cd personal-ai
cp .env.example .env        # then fill in NEBIUS_API_KEY etc.
docker compose -f deploy/compose.yaml up --build
```

_TODO: add `deploy/compose.yaml` and document ports and volumes._

### Local development (without Docker)

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`, and dependency versions are pinned in `uv.lock`.

```bash
uv sync --locked            # install exactly what's in uv.lock
uv run pytest               # run the tests
```

To add a dependency, run `uv add <package>`. Commit the updated `uv.lock` along with it.

## Troubleshooting

### Windows

- _TODO_: Use Docker Desktop with the WSL 2 backend. Clone the repo inside the WSL filesystem, not `C:\`, so file watching works and I/O stays fast.
- _TODO_: Line endings — set `git config core.autocrlf input` if containers fail with `\r` errors.

### macOS

- _TODO_: On Apple Silicon, if an image has no `arm64` variant, build with `--platform linux/amd64`.
- _TODO_: If `uv` isn't found after installing it, restart the shell or add `~/.local/bin` to `PATH`.

## Nemotron usage

All inference goes through an OpenAI-compatible API (`agent/llm.py`). Set `PROVIDER` in `.env`. Each preset sets both the base URL and that provider's spelling of the model IDs, so the two always match:

| Alias | `PROVIDER=nebius` (Token Factory) | `PROVIDER=nvidia_build` (build.nvidia.com) |
|---|---|---|
| base URL | `https://api.tokenfactory.nebius.com/v1/` | `https://integrate.api.nvidia.com/v1` |
| `nano` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | `nvidia/nemotron-nano-3-30b-a3b` |
| `super` | `nvidia/nemotron-3-super-120b-a12b` | `nvidia/nemotron-3-super-120b-a12b` |
| `ultra` | `nvidia/Nemotron-3-Ultra-550b-a55b` | `nvidia/nemotron-3-ultra-550b-a55b` |

```python
from agent.llm import chat

reply = chat("super", [{"role": "user", "content": "Hello"}])
print(reply)                  # final answer only (a str)
reply.reasoning               # model's reasoning: internal, never show to users
reply.tool_calls              # tool calls, if any
reply.finish_reason           # "length" means truncated (also logged as a warning)
```

- `max_tokens` defaults to 4096. Reasoning tokens count toward it.
- **Nano is optional.** If a model returns 404 (not enabled for your key) or 503 (overloaded), the call is retried on `super`, and a warning is logged once. A model that returns 404 is skipped for the rest of the process.
- `LLM_BASE_URL` can override the preset URL (e.g. a Nebius dedicated endpoint). It must belong to the same provider, or startup fails.
- `NEBIUS_API_KEY` / `NEBIUS_BASE_URL` are still accepted as fallbacks for `LLM_API_KEY` / `LLM_BASE_URL`.
- Live check: `uv run pytest -m live -s`. A plain `uv run pytest` makes no API calls.
- _TODO_: prompting conventions, tool-calling format.

## Nebius services used

_TODO: confirm and fill in._

| Service | Used for |
|---|---|
| Nebius Token Factory (serverless) | Serving Nemotron models via `https://api.tokenfactory.nebius.com/v1/` |
| _TODO_ | _e.g. compute / object storage / managed DB_ |

## License

[MIT](LICENSE)
