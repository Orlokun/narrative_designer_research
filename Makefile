# CyberSyn Backend — common dev tasks
# Works on Linux/WSL and Windows (Git Bash / uv)
.PHONY: install test test-cov lint fmt \
        gatekeeper admin propositor propositor-status \
        archivero archivero-status archivero-export \
        verificator verificator-status verify-queue \
        cleaner cleaner-dry-run \
        mapper mapper-status mapper-all remap-all \
        cleaner-mapper \
        cast-manager cast-manager-status cast-manager-all cast-remine cast-enrich cast-analyze cast-score cast-politics \
        location-manager location-manager-status location-enrich location-analyze location-score \
        cast-clean cast-clean-apply \
        cast-director cast-director-status \
        curate curate-all curate-status curate-recurate \
        new-project project-show scout scout-apply \
        pipeline pipeline-loop pipeline-characters drain drain-loop \
        reset \
        docker-build docker-up docker-down docker-logs docker-ps

# ── Local dev ──────────────────────────────────────────────────────────────────

install:
	uv sync

test:
	uv run pytest -v

test-cov:
	uv run pytest --cov=lib --cov=admin --cov=research_engine --cov=pipeline --cov-report=term-missing

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

# ── Platform: set up a new historical moment ───────────────────────────────────
# make new-project MOMENT="Weimar Berlin 1929-33" START=1929-10 END=1933-03 REGION=Germany LANG=de SLUG=weimar
# then edit projects/$(SLUG)/project.yaml, run `make scout-apply P=$(SLUG)` (Gatekeeper up),
# and set RESEARCH_PROJECT=$(SLUG) in .env.

LANG ?= en
P ?= cybersyn

new-project:
	uv run grid-proposer propose --moment "$(MOMENT)" --start $(START) --end $(END) --region "$(REGION)" --language $(LANG) --slug $(SLUG)

project-show:
	uv run grid-proposer show $(P)

scout:
	uv run source-scout run --project $(P)

scout-apply:
	uv run source-scout run --project $(P) --apply

# ── Run services locally ───────────────────────────────────────────────────────

gatekeeper:
	uv run gatekeeper

admin:
	uv run admin

propositor:
	uv run propositor run --limit 100

propositor-status:
	uv run propositor status

archivero:
	uv run archivero run --batch 50 --missions 200

archivero-status:
	uv run archivero status

archivero-export:
	uv run archivero export

verificator:
	uv run verificator run --batch 200

verificator-status:
	uv run verificator status

cleaner:
	uv run verificator clean

cleaner-dry-run:
	@echo "Showing rejected documents that would be deleted..."
	uv run verificator clean --dry-run

mapper:
	uv run mapper run --batch 50

mapper-status:
	uv run mapper status

mapper-all:
	@echo "Procesando TODOS los documentos verificados (mapeados + no mapeados)..."
	uv run mapper run --all --batch 100
	uv run mapper status

remap-all:
	@echo "Re-mapeando TODOS los documentos ya mapeados (verificación de mapping)..."
	uv run mapper run --remap --batch 100
	uv run mapper status

cleaner-mapper:
	@echo "Cleaning rejected documents and mapping verified documents..."
	uv run verificator clean
	uv run mapper run --batch 50

cast-manager:
	uv run cast-manager run --batch 50

cast-manager-status:
	uv run cast-manager status

cast-manager-all:
	@echo "Extrayendo personas de TODOS los documentos verificados+mapeados..."
	uv run cast-manager run --all --batch 100
	uv run cast-manager status

cast-remine:
	@echo "Re-minando documentos ya procesados (aplica extracción mejorada)..."
	uv run cast-manager run --re-extract --batch 100
	uv run cast-manager status

cast-enrich:
	@echo "Vinculando personajes a Wikidata + Wikipedia (requiere Gatekeeper)..."
	uv run cast-manager enrich --limit 200

cast-analyze:
	@echo "Analizando Wikidata + Wikipedia de personajes linkeados (requiere Gatekeeper + ollama serve)..."
	uv run cast-manager analyze --limit 50

cast-score:
	@echo "Recalculando la completitud de todos los personajes (métrica estricta)..."
	uv run cast-manager score

# ── Location Manager (Ag-4) — lugares y locaciones ────────────────────────────

location-manager:
	uv run location-manager run --batch 50

location-manager-status:
	uv run location-manager status

location-enrich:
	@echo "Vinculando lugares a Wikidata + Wikipedia (requiere Gatekeeper)..."
	uv run location-manager enrich --limit 100

location-analyze:
	@echo "Analizando lugares linkeados: GPS, arquitecto, artículo (Gatekeeper + ollama serve)..."
	uv run location-manager analyze --limit 30

location-score:
	@echo "Recalculando la completitud de todos los lugares..."
	uv run location-manager score

cast-politics:
	@echo "Ubicando personajes en la brújula política (requiere ollama serve)..."
	uv run cast-manager politics --limit 200

cast-clean:
	@echo "Personajes implausibles que se eliminarían (dry-run)..."
	uv run cast-manager clean

cast-clean-apply:
	@echo "Eliminando personajes implausibles (boilerplate, títulos, meses)..."
	uv run cast-manager clean --apply

# ── Document Curator (Ag-9) — utility flags per document ─────────────────────────

curate:
	uv run doc-curator run --batch 50

curate-all:
	@echo "Curando TODOS los documentos verificados+mapeados (bucle autoterminado)..."
	uv run doc-curator run --all --batch 100
	uv run doc-curator status

curate-status:
	uv run doc-curator status

curate-recurate:
	@echo "Re-curando el corpus existente (aplica un prompt mejorado)..."
	uv run doc-curator run --re-curate --batch 100
	uv run doc-curator status

verify-queue:
	@echo "Verificando cola hasta vaciar (Ctrl+C para detener)..."
	@while true; do \
		echo ""; \
		uv run verificator status; \
		echo "Procesando siguiente batch..."; \
		result=$$(uv run verificator run --batch 200 2>&1); \
		if echo "$$result" | grep -q "nothing_to_score"; then \
			echo "✓ Cola verificada completamente"; \
			break; \
		fi; \
		sleep 1; \
	done

# ── Reset — wipe all harvested data ───────────────────────────────────────────
# Deletes the SQLite database, generated JSON files, logs, and the Gatekeeper
# disk cache. Source code and .env are not touched.
# Usage: make reset CONFIRM=yes

reset:
	uv run python scripts/reset_data.py $(if $(filter yes,$(CONFIRM)),--yes,)

# ── Pipeline (one full cycle with LLM verification and mapping) ──────────────────
# Flow: generate missions → fetch documents → verify with Gemma4 → map to cells → export
# Requires: make gatekeeper (in another terminal) + ollama serve

# Each run mints one CYBERSYN_RUN_ID, exported so all stages share it; the agents
# record per-stage timing/counts against it (Admin Performance/Results pages).

pipeline:
	@RUN_ID=$$(date -u +%Y%m%dT%H%M%SZ)-$$(date +%N | head -c 6); \
	export CYBERSYN_RUN_ID=$$RUN_ID; \
	echo "Pipeline run: $$CYBERSYN_RUN_ID"; \
	uv run pipeline-run start --trigger pipeline; \
	uv run propositor run --limit 100; \
	uv run archivero run --batch 50 --missions 200; \
	uv run verificator run --batch 200; \
	uv run verificator clean; \
	uv run mapper run --batch 50; \
	uv run cast-manager run --batch 50; \
	uv run cast-manager enrich --limit 50; \
	uv run cast-manager analyze --limit 10; \
	uv run cast-director run --batch 50; \
	uv run location-manager run --batch 50; \
	uv run location-manager enrich --limit 50; \
	uv run location-manager analyze --limit 10; \
	uv run doc-curator run --batch 50; \
	uv run archivero export; \
	uv run pipeline-run finish

# ── Pipeline characters — completar personajes incompletos ────────────────────
# El Propositor genera SOLO misiones entity (los personajes menos completos
# primero, bajo el umbral 0.85 de la métrica estricta); la cadena estándar
# cosecha, y el Cast Manager extrae, enlaza y analiza Wikidata/Wikipedia.
# Requires: make gatekeeper (in another terminal) + ollama serve

pipeline-characters:
	@RUN_ID=$$(date -u +%Y%m%dT%H%M%SZ)-$$(date +%N | head -c 6); \
	export CYBERSYN_RUN_ID=$$RUN_ID; \
	echo "Pipeline characters run: $$RUN_ID"; \
	uv run pipeline-run start --trigger characters; \
	uv run propositor run --focus characters --entity-limit 30; \
	uv run archivero run --batch 50 --missions 100; \
	uv run verificator run --batch 200; \
	uv run verificator clean; \
	uv run mapper run --batch 50; \
	uv run cast-manager run --batch 50; \
	uv run cast-manager enrich --limit 100; \
	uv run cast-manager analyze --limit 30; \
	uv run cast-manager score; \
	uv run cast-director run --batch 50; \
	uv run archivero export; \
	uv run pipeline-run finish

# ── Pipeline loop — continuous with LLM verification and mapping ──────────────
# Generates new missions only when pending queue drops below 50.
# Requires: make gatekeeper (in another terminal) + ollama serve

pipeline-loop:
	@echo "Iniciando pipeline continuo. Ctrl+C para detener."
	@while true; do \
		RUN_ID=$$(date -u +%Y%m%dT%H%M%SZ)-$$(date +%N | head -c 6); \
		export CYBERSYN_RUN_ID=$$RUN_ID; \
		echo "Pipeline run: $$CYBERSYN_RUN_ID"; \
		uv run pipeline-run start --trigger pipeline-loop; \
		uv run propositor run --limit 100 --min-queue 50; \
		uv run archivero run --batch 50 --missions 200; \
		uv run verificator run --batch 200; \
		uv run verificator clean; \
		uv run mapper run --batch 50; \
		uv run cast-manager run --batch 50; \
		uv run cast-manager enrich --limit 50; \
		uv run cast-manager analyze --limit 10; \
		uv run cast-director run --batch 50; \
		uv run location-manager run --batch 50; \
		uv run location-manager enrich --limit 50; \
		uv run location-manager analyze --limit 10; \
		uv run doc-curator run --batch 50; \
		uv run archivero export; \
		uv run pipeline-run finish; \
		echo ""; \
		echo "Ciclo completado. Próximo en 30s... (Ctrl+C para detener)"; \
		sleep 30; \
	done

# ── Drain — vacía la cola pendiente sin generar misiones nuevas ───────────────
# Útil cuando hay acumulación de misiones pending/running.
# Requiere: ollama serve corriendo para verificar y mapear documentos

drain:
	@RUN_ID=$$(date -u +%Y%m%dT%H%M%SZ)-$$(date +%N | head -c 6); \
	export CYBERSYN_RUN_ID=$$RUN_ID; \
	echo "Drain run: $$CYBERSYN_RUN_ID"; \
	uv run pipeline-run start --trigger drain; \
	uv run archivero run --batch 50 --missions 200; \
	uv run verificator run --batch 200; \
	uv run verificator clean; \
	uv run mapper run --batch 50; \
	uv run cast-manager run --batch 50; \
	uv run cast-director run --batch 50; \
	uv run doc-curator run --batch 50; \
	uv run archivero export; \
	uv run pipeline-run finish

drain-loop:
	@echo "Vaciando cola (sin Propositor). Ctrl+C para detener."
	@while true; do \
		RUN_ID=$$(date -u +%Y%m%dT%H%M%SZ)-$$(date +%N | head -c 6); \
		export CYBERSYN_RUN_ID=$$RUN_ID; \
		echo "Drain run: $$CYBERSYN_RUN_ID"; \
		uv run pipeline-run start --trigger drain-loop; \
		uv run archivero run --batch 50 --missions 200; \
		uv run verificator run --batch 200; \
		uv run verificator clean; \
		uv run mapper run --batch 50; \
		uv run cast-manager run --batch 50; \
		uv run cast-director run --batch 50; \
		uv run doc-curator run --batch 50; \
		uv run archivero export; \
		uv run pipeline-run finish; \
		echo ""; \
		echo "Ciclo completado. Próximo en 30s... (Ctrl+C para detener)"; \
		sleep 30; \
	done

# ── Docker ─────────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-ps:
	docker compose ps
