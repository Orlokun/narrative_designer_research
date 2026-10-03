# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reusable multi-agent research engine for historical narrative design. A historical moment is
described by a project spec (`projects/<slug>/project.yaml`, drafted by the Grid Proposer and
source-routed by the Source Scout); the pipeline harvests period documents into a theme × genre ×
month coverage matrix, extracts cast and places, and the Knowledge API lets a game Director
consult the archive. Translated from the Project Cybersyn engine (Allende-era Chile, 1969–1973,
MPhil in Digital Humanities, Cambridge), which ships as the reference project.

## ▶ Source of truth: the wiki (read before, update after)

The **project wiki is the ultimate source of truth and the rules of this project.** It
lives at [`wiki/index.html`](wiki/index.html) — a cross-linked HTML knowledge base.

**This is a hard requirement, not a suggestion:**

1. **Before any work**, read [`wiki/rules.html`](wiki/rules.html) and the wiki page(s)
   covering the area you are about to touch.
2. **After work is done**, update the affected wiki page(s) so the wiki never lags the
   code. A change is not finished until the wiki reflects it. Update the specific page
   *and* any index that lists the thing you changed (Agents / Commands / Data Model).
3. **Commit after every relevant change** — bundle the code change and its wiki update in
   the same commit, with a clear message, and push so the remote is a real backup. A task
   isn't done until it's committed. (Background: ~4 weeks of unpushed work was lost once.)

All detailed architecture, conventions, schemas and command references now live in the
wiki — kept in one place to avoid drift. Do not duplicate that detail back into this file.

## Wiki map — where to read

| Topic | Page |
|---|---|
| Engineering principles, conventions, the working protocol | `wiki/rules.html` |
| The reusable platform: project spec, Grid Proposer, Source Scout, LLM providers, Knowledge API | `wiki/platform.html` |
| Data flow, SQLite-as-truth, source routing, mission logic | `wiki/architecture.html` |
| Every agent, ID, role and build status | `wiki/agents.html` (+ per-agent pages) |
| Propositor / Archivero / Verificator / Mapper / Gatekeeper / Director | `wiki/agent-*.html`, `wiki/gatekeeper.html`, `wiki/director.html` |
| SQLite tables + Pydantic schemas | `wiki/data-model.html` |
| The 16×13×48 coverage matrix: categories, genres, plausibility, source tiers | `wiki/coverage-matrix.html` |
| CLI entry points + every Makefile target | `wiki/commands.html` |
| Admin dashboard routers & pages | `wiki/admin.html` |
| Domain terms, key dates, speech-act theory | `wiki/glossary.html` |
| Rebuild status & deployment goal | `wiki/roadmap.html` |

## Bootstrap commands

```bash
uv sync            # install all deps (including dev group)
uv run pytest -v   # full test suite (offline)
uv run ruff check . && uv run ruff format .
```

Full command reference (services, pipeline cycles, maintenance, Docker): see
[`wiki/commands.html`](wiki/commands.html).
