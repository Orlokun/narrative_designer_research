# Plan — Investigación de Personajes: completitud estricta, análisis externo y modo dedicado

**Rama del plan:** `claude/character-research-pipeline-5h2kq8` · **Fecha:** 2026-08-01 ·
**Estado:** propuesta v2 para aprobación

> **Base de implementación: `feat/cast-manager`** (70655a5, ~40 commits sin mergear a
> `main`). La v1 de este plan se escribió contra `main`, donde Cast Manager y Cast
> Director figuran como "Rebuild" — pero en `feat/cast-manager` ya están construidos.
> Esta v2 audita esa rama y planifica solo lo que falta.

---

## 0. Prerrequisito de housekeeping: mergear `feat/cast-manager` a `main` — ✅ hecho

**Completado el 2026-08-01**: `feat/cast-manager` se mergeó fast-forward a `main`
(`7bf333d → 70655a5`) y esta rama quedó rebasada sobre el nuevo `main`. Nota de la
verificación previa al merge: la suite corre 546/547 en verde; el único fallo
(`test_gatekeeper.py::test_rate_limit_escalation_decays_with_time_served`) es
ambiental — asume `time.monotonic()` > 1200s (uptime de máquina > 20 min) y falla en
contenedores recién arrancados. Arreglarlo entra en la F0.

## 1. Lo que YA existe en `feat/cast-manager` (auditado)

| Pieza | Estado | Dónde |
|---|---|---|
| **Cast Manager (Ag-7)** — extracción LLM por chunks (docs largos sin truncar), coreferencia segura por subsecuencia de tokens, menciones con `mentioned_by`, dedup de hechos, filtro de validez, biografías (`enrich`), brújula política (`politics`), limpieza (`clean`) | ✅ Real, con tests | `pipeline/cast_manager/__init__.py` (2 387 líneas), `tests/test_cast_manager.py` |
| **Cast Director (Ag-8)** — relaciones entre personajes + grafo en admin | ✅ Real | `pipeline/cast_director/`, `admin` |
| **Tablas** `characters`, `character_timeline` (kind: `role/event/affiliation/other`), `character_mentions` | ✅ | `cast_manager/__init__.py:1168-1229` |
| **Loop de investigación entity-driven** — Propositor emite misiones `kind='entity'` desde la cola `needs_research`; iterativo (cooldown 3 días, máx. 3 intentos, `should_research()`); queries sembradas con aliases + afiliación; sin distorsión de la matriz de cobertura | ✅ | `research_engine/propositor/__init__.py`, `cast_manager/__init__.py:301-341` |
| **Wikidata/Wikipedia — enlace** (`resolve_wikidata`: QID, wikipedia_url, descripción, `wikidata_checked_at`) | ✅ solo **link** | `cast_manager/__init__.py:1760-1830` |
| **`make pipeline`** ya incluye `cast-manager run` + `cast-director run` | ✅ | `Makefile:171-190` |
| Admin: páginas Cast Manager (ficha, life-line SVG) y Cast Director (red) | ✅ | `admin/routers/cast.py`, static |

## 2. Brechas contra el objetivo (lo que este plan implementa)

1. **La métrica de completitud es laxa** — `compute_completeness` (`:288`):
   `0.3·bio + 0.4·min(1, facts/5) + 0.3·min(1, menciones/3)`. Con bio + 5 hechos +
   3 menciones → 1.0, sin tocar Wikidata/Wikipedia ni diversidad de timeline. Y
   `_RESEARCH_THRESHOLD = 0.6` deja de investigar personajes al 60%.
2. **Wikidata/Wikipedia se linkean pero no se analizan** — el QID y la URL se guardan;
   nadie extrae los claims (cargos, fechas, partido) ni el artículo hacia el
   perfil/timeline.
3. **El timeline no distingue enunciados ni rumores** — `kind` cubre
   role/event/affiliation/other (≈ información/acciones), sin `statement` (enunciado,
   con acto de habla) ni `rumor` (afirmación no verificada, con quién la propaga).
4. **No hay modo de arranque enfocado en personajes** — las misiones entity van
   mezcladas en el `propositor run` normal; no existe `make pipeline-characters` que
   priorice a los incompletos.

## 3. Diseño

### 3.1 Métrica estricta (reemplaza `compute_completeness`)

Función pura, misma firma ampliada; desglose persistido en una nueva columna
`completeness_detail` (JSON) para que el admin muestre *por qué* no llega.

**Suma ponderada (componentes 0–1):**

| Componente | Peso | Cálculo |
|---|---|---|
| Identidad | 0.15 | ⅓ bio + ⅓ roles en timeline + ⅓ fechas vitales (birth/death) |
| Base documental | 0.20 | `min(1, menciones_en_docs_verificados / 8)` (sube de 3 → 8 el objetivo) |
| Enlace externo | 0.25 | Wikidata: 0.125 (0.06 solo-link, 0.125 **analizado**) + Wikipedia: ídem |
| Riqueza del timeline | 0.30 | `0.5·min(1, hechos/12)` + `0.35·(grupos_de_kind_presentes/4)` + `0.15·dispersión temporal` (meses 1969–1973 con ≥1 hecho) |
| Paso por el pipeline | 0.10 | 0.5 extracción sobre ≥1 doc + 0.5 ≥1 misión `entity` completada (`done`) |

Los 4 grupos de kind para la diversidad: **acciones** (`event`), **enunciados**
(`statement`), **información** (`role`/`affiliation`/`other`), **rumores** (`rumor`).

**Techos duros tras la suma:**

- Sin Wikidata **ni** Wikipedia analizados → score ≤ **0.60**
- Menos de 3 de los 4 grupos de kind en el timeline → score ≤ **0.75**
- Ninguna misión `entity` completada → score ≤ **0.80**

**"Completo" = ≥ 0.85.** `_RESEARCH_THRESHOLD` sube de 0.6 → **0.85** y
`_RESEARCH_MAX_ATTEMPTS` de 3 → **5** (el loop ya es acotado por cooldown, así que
subir el umbral no lo desboca). Constantes nombradas, calibrables.

### 3.2 Análisis de Wikidata y Wikipedia (nuevo subciclo `cast-manager analyze`)

Sobre los personajes ya linkeados (`wikidata_id NOT NULL`), por lotes, vía Gatekeeper:

1. **Wikidata**: `wbgetclaims`/entity JSON → P39 (cargos, con fechas), P102 (partido),
   P106 (ocupación), P569/P570 (vitales) → hechos de timeline `kind='role'/'affiliation'`
   con `date_iso`, y birth/death si faltan. Procedencia: el fetch se archiva como
   documento (`source='wikidata'`) y los hechos apuntan a su `doc_id`.
2. **Wikipedia**: fetch del artículo exacto (extract completo, no búsqueda) → se
   archiva como documento normal (pasa por Verificator/Mapper como cualquier otro) →
   LLM extrae bio de un párrafo (si falta o es peor) + hechos fechados.
3. Stamps nuevos: `wikidata_analyzed_at`, `wikipedia_analyzed_at` (migración por
   `ALTER TABLE` idempotente, el idioma de la casa). La métrica premia el análisis,
   no el link.

Homónimos: `resolve_wikidata` ya existe; se le añade verificación de periodo/contexto
antes de analizar (si la descripción del QID contradice 1969–73/Chile → se des-linkea y
se loggea, nunca se analiza a ciegas).

### 3.3 Timeline más rico: enunciados y rumores

- `_FACT_KINDS` += `statement`, `rumor`. Migración de `character_timeline`: columnas
  `speech_act TEXT NULL` (enum `SpeechAct` existente de `lib/schemas.py` — Searle/Flores,
  ya documentado en el glosario) y `reported_by TEXT NULL`, `confidence REAL NULL`.
- El prompt de extracción del Cast Manager gana dos instrucciones: capturar **citas y
  declaraciones** del personaje (→ `statement` + acto de habla) y **afirmaciones no
  verificadas o de oídas sobre él** (→ `rumor` + quién lo afirma + confidence baja).
  El parser puro (`_clean_fact`) valida los campos nuevos.
- Backfill barato: el flag `--re-extract` ya existe para re-minar docs procesados.
- El detalle de personaje en admin muestra los kinds con estilo propio (cita para
  enunciados, marca de incertidumbre para rumores).

### 3.4 Modo de arranque "personajes incompletos"

```
make pipeline-characters
  pipeline-run start --trigger characters
  propositor run --focus characters --entity-limit 30   # sin misiones de celda
  archivero run … / verificator run … / mapper run …
  cast-manager run --batch 50        # extracción de lo nuevo
  cast-manager analyze --limit 30    # wikidata/wikipedia pendientes
  cast-manager enrich --limit 100    # bios faltantes
  cast-director run --batch 50
  pipeline-run finish
```

- `propositor --focus {cells|characters|mixed}` (default `cells` = comportamiento
  actual; `characters` omite las misiones de celda y selecciona personajes por
  `completeness_score ASC` bajo 0.85; `mixed` = hoy). El trigger `characters` queda en
  `pipeline_runs.trigger` (visible en Performance/Results).
- `make pipeline` normal: se añade `cast-manager analyze --limit 10` tras `run` para
  que el análisis externo avance también en ciclos normales, acotado.

## 4. Fases (cada una: tests offline primero → código → wiki → commit → push)

| Fase | Entregable | Wiki |
|---|---|---|
| **F0 — ✅ hecha** | Migraciones (`completeness_detail`, `speech_act`, `reported_by`, `confidence`, `*_analyzed_at`) + nueva `compute_completeness` con techos y desglose + `should_research` recalibrado (0.85, 5 intentos) + fix del test ambiental del Gatekeeper. Tests: 557 en verde | `data-model.html`, `agent-cast-manager.html`, `agent-propositor.html`, `roadmap.html`, `glossary.html` |
| **F1 — ✅ hecha** | `cast-manager analyze`: claims de Wikidata (P39/P102/P106 → hechos, P569/P570 → fechas vitales, deslinkeo de homónimos nacidos > 1973) + artículo de Wikipedia vía LLM (bio de un párrafo + hechos fechados), stamps `*_analyzed_at`, `make cast-analyze`. Nota: los hechos externos llevan `doc_id NULL` (la procedencia es el propio link del personaje), en vez de archivar el fetch como documento | `agent-cast-manager.html`, `sources.html`, `commands.html`, `roadmap.html` |
| **F2 — ✅ hecha** | Kinds `statement`/`rumor` en prompt + parser (`speech_act` validado contra el enum; rumor sin `reported_by` se rechaza; confidence default 0.3) + dedup que no colapsa narrativa con información + admin (citas «» y rumores atenuados/atribuidos). Backfill: `cast-manager run --re-extract` | `agent-cast-manager.html`, `admin.html`, `roadmap.html` (glosario ya cubierto en F0) |
| **F3 — ✅ hecha** | `propositor --focus {mixed,cells,characters}` (default `mixed` = comportamiento actual; `characters` = solo misiones entity, orden `order_by_need`: menos completos primero) + `make pipeline-characters` (trigger `characters`, cadena completa con enrich/analyze) + enrich (50) / analyze (10) acotados en `make pipeline` y `pipeline-loop` | `agent-propositor.html`, `architecture.html`, `commands.html`, `roadmap.html` |
| **F4 — ✅ hecha (tooling)** | `cast-manager score` / `make cast-score`: re-score masivo bajo las constantes vigentes (obligatorio tras cambiar la métrica; corre al final de `pipeline-characters`) + vista de calibración en el admin (`/api/cast/stats`: histograma de 10 buckets, conteo de completos ≥0.85, personajes por techo activo) + desglose por componente en la ficha. El ajuste fino de constantes queda para cuando corras el pipeline sobre el corpus real, con esta evidencia a la vista | `agent-cast-manager.html`, `commands.html`, `admin.html`, `roadmap.html` |

## 5. Riesgos

- **Deflación de scores existente**: al endurecer la métrica, personajes hoy "1.0"
  caerán a ~0.4–0.6 y `needs_research` se reactivará en masa. Es deseable (es el
  objetivo), pero el primer `pipeline-characters` debe correr con `--entity-limit`
  acotado para no inundar la cola de misiones. El loop acotado (intentos+cooldown) ya
  protege contra el runaway.
- **Rumores y procedencia**: un `rumor` sin `reported_by` ni doc no se inserta — regla
  dura en el parser, para que la búsqueda de diversidad no fabrique ruido.
- **Homónimos Wikidata**: verificación de contexto pre-análisis + des-link loggeado.
- **Compatibilidad**: `kind` nuevos son aditivos; la métrica es una función pura con
  regresión completa; el modo `--focus` default no cambia el comportamiento actual.
