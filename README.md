# Narrative Designer Research Engine

A reusable, multi-agent research engine for **historical narrative design**. Pick a historical
moment, let an agent propose the research grid (themes, turning points, institutions,
characters, places), let another agent find which open archives and APIs actually hold material
for it, then run a harvesting pipeline that fills a theme × genre × month coverage matrix with
verified period documents, extracts the cast and the places, and exposes it all through a
knowledge API a **game director** can consult while the player acts.

It is the generalised translation of the research engine built for a game-based simulation of
[Project Cybersyn](projects/cybersyn/project.yaml) (Allende-era Chile, 1969–1973, MPhil in Digital
Humanities, Cambridge). Cybersyn ships as the reference project; the engine behaves identically
under it and now runs under any other project spec.

## How a new moment becomes a project

```bash
uv sync                                  # install
cp .env.example .env                     # choose LLM_PROVIDER (ollama | gemini) and keys

# 1. Propose the research grid for a moment (writes projects/<slug>/project.yaml)
uv run grid-proposer propose \
  --moment "Weimar Berlin from the 1929 crash to Hitler's chancellorship" \
  --start 1929-10 --end 1933-03 --region Germany --language de --slug weimar

# 2. Edit projects/weimar/project.yaml — it is written for a historian to correct.

# 3. Find which APIs have material for it (Gatekeeper running: make gatekeeper)
uv run source-scout run --project weimar --apply

# 4. Activate the project and run the pipeline exactly as for Cybersyn
echo RESEARCH_PROJECT=weimar >> .env
make pipeline
```

The game director then asks the archive about the player's situation:

```python
from lib.knowledge import DirectorQuery, KnowledgeBase

ctx = KnowledgeBase().context(
    DirectorQuery(month_iso="1972-10", theme_ids=[11, 4], character_names=["Allende"], limit=8)
)
print(ctx.as_prompt())   # evidence block for the director's LLM prompt
```

## What is in the box

| Layer | Where | Role |
|---|---|---|
| Project spec | `lib/project.py`, `projects/<slug>/project.yaml` | One declarative file per historical moment: period, themes, critical months, source tiers, vocabulary, relevance / classification rules, seeds. |
| Grid Proposer | `research_engine/grid_proposer/` | Moment → validated project spec (LLM-drafted, deterministically completed; `--no-llm` skeleton). |
| Source Scout | `research_engine/source_scout/`, `lib/sources.py` | Connector catalogue + live probing through the Gatekeeper → recommended source tiers. |
| Pipeline | `research_engine/`, `pipeline/` | Propositor → Gatekeeper → Archivero → Verificator → Mapper → Cast / Location managers → Cast Director → Document Curator. |
| LLM providers | `lib/llm.py` | `LLM_PROVIDER=ollama` (local) or `gemini` (Google AI Studio API serving Gemma on a free tier). |
| Knowledge API | `lib/knowledge.py` | Read-only consultation for the Director: documents, character dossiers, relations, places, ranked for a game situation. |
| Admin | `admin/` | Ops-room style dashboard over the archive (heatmap, documents, cast, locations, runs). |

## Source of truth

The wiki at [`wiki/index.html`](wiki/index.html) is the project's rules and documentation; start at
[`wiki/platform.html`](wiki/platform.html) for the reusable system and
[`wiki/rules.html`](wiki/rules.html) for the working protocol (read before, update after, commit after).

```bash
uv run pytest -v                 # offline test suite
uv run ruff check . && uv run ruff format .
```
