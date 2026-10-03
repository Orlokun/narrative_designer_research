# cybersyn-backend

Research Engine and agent pipeline for *Playing with revolutionary machines:
A game-based simulation approach towards Project Cybersyn using LLMs*
(MPhil Digital Humanities, Cambridge).

**The project wiki ([`wiki/index.html`](wiki/index.html)) is the source of truth** for
architecture, agents, data model and rules — open it in a browser. This README covers
running the code.

---

## What's here

```
agents/
├── lib/                   # shared: config, schemas, Ollama client, logging
├── research_engine/
│   ├── gatekeeper/        # RE-GK: rate-limited HTTP proxy  (port 8001)
│   ├── propositor/        # RE-1:  gap detection + mission generation (CLI)
│   └── chronicler/        # RE-2:  speech acts, characters, places (WIP)
├── admin/                 # dashboard FastAPI + single-page UI (port 8080)
├── pipeline/              # agents 1–8: archive build (stubs)
├── runtime/               # agents 9–14: game session (stubs)
├── analysis/              # agents 15–16: post-hoc evaluation (stubs)
├── finetune/              # QLoRA training script (stub)
├── tests/
└── data/                  # SQLite, ChromaDB, JSONL — gitignored, volume-mounted
```

---

## Option A — Local (uv)

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Install dependencies
make install
# or: uv sync

# Copy env template (edit if Ollama runs elsewhere)
cp .env.example .env

# Run services
make gatekeeper          # port 8001
make admin               # port 8080  →  http://localhost:8080

# Run propositor (mission generation — no Ollama needed)
make propositor
make propositor-status

# Tests
make test
make test-cov
```

---

## Option B — Docker / WSL

Requires Docker Desktop with WSL2 integration enabled.

```bash
cp .env.example .env     # edit if needed

make docker-build        # build image (~2 min first time)
make docker-up           # start gatekeeper + admin in background

# Check status
make docker-ps
make docker-logs

# Stop
make docker-down
```

Services after `docker-up`:
| Service | URL |
|---|---|
| Admin dashboard | http://localhost:8080 |
| Gatekeeper API | http://localhost:8001/health |
| Gatekeeper docs | http://localhost:8001/docs |

The `data/` directory is bind-mounted — SQLite, cache, and JSONL files persist
on the host even after `docker-down`.

---

## Option C — Ubuntu WSL (without Docker)

```bash
# Inside Ubuntu-24.04 WSL terminal
cd /path/to/cybersyn-backend

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
cp .env.example .env
make gatekeeper          # or make admin
```

---

## Ollama (optional — needed for LLM features)

```bash
# Install: https://ollama.com
ollama pull gemma4:e4b           # NPC inference, Propositor query reformulation
ollama pull gemma4:26b           # Director + Evaluador (optional)
ollama pull bge-m3               # Embeddings (Indexador)
```

Without Ollama, the Propositor runs with `--no-llm` (rule-based queries only),
and NPC/Director features are unavailable.

---

## Conventions

- Central config: `lib/config.py` (pydantic-settings, reads `.env`)
- Cross-agent types: `lib/schemas.py` — if you're passing a `dict`, stop
- Logging: `lib/logging_setup.get_logger()` — never `print()` in non-test code
- Dependencies: `pyproject.toml` only — no stray `requirements.txt`
- Tests: `uv run pytest` — all tests run without Ollama or external services

## Author

Orlando Guerrero · oguerrerofarias@gmail.com
