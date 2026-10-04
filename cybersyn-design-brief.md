# Project Cybersyn — Design Brief

> A single, self-contained synthesis of the project wiki, written to be handed to a
> design collaborator (e.g. a fresh Claude session focused on game / narrative design).
> It explains **what is being built, why, how the knowledge base is produced, what that
> knowledge base contains, and what the game it feeds is meant to become.**
>
> Source of truth: the project wiki (`wiki/index.html`). This document is a derived
> snapshot — if the two disagree, the wiki wins.

---

## 1. What this is, in one paragraph

A **multi-agent research pipeline** that builds a **historically-grounded knowledge base**
about **Project Cybersyn** and Allende-era Chile (window: **October 1969 – September 1973**).
The pipeline harvests period documents, scores them for relevance, files them against a
**16 categories × 48 months coverage matrix**, and — in its planned end state — extracts the
**people, relationships and events** needed to drive an **LLM-powered NPC dialogue / simulation
game**. Built for an **MPhil in Digital Humanities at Cambridge**, it serves two purposes at once:

1. **The game** — organise narrative events and ground NPC dialogue in real period sources.
2. **The research** — a durable, queryable archive of everything gathered during the MPhil.

The aesthetic and the conceptual frame are **Project Cybersyn (Synco)** itself: Stafford Beer's
1971–73 cybernetic system for steering Chile's nationalised economy in real time, centred on the
**Operations Room (Opsroom)**. The admin dashboard deliberately evokes that dark/amber Ops-Room UI.

---

## 2. Why it exists / design intent

- **Narrative events for the game.** The matrix + the extracted cast/relations are the raw
  material from which narrative events, scenarios and branching dialogue are assembled. The
  knowledge base is the "world bible," generated from primary sources rather than invented.
- **Research archive.** Everything collected during the MPhil lives in one queryable store, so
  the academic work and the game share a single substrate.
- **Grounded NPCs.** NPC dialogue is meant to be anchored (RAG) to actual period documents, so
  characters "know" what their historical counterparts could plausibly have known.
- **A theory of conversation.** Dialogue is modelled on **Searle's speech acts** and the
  **Flores–Winograd "conversation for action" / commitment cycle** — fitting, since Fernando
  Flores was a central Cybersyn figure. This is a first-class part of the data model, not an
  afterthought.

---

## 3. The historical canvas

**Window:** Oct 1969 – Sep 1973 (48 months → the matrix's time axis).

**Key dates / critical months** (priority-boosted in the pipeline):

| Month | Event |
|---|---|
| 1970-09 / 1970-11 | Allende elected (Sep); inaugurated (Nov 3). |
| 1971-07 | Nationalisation of copper. |
| **1972-10** | *Paro de camioneros* — the truckers' strike (category 11, Transporte). |
| **1973-09** | The coup d'état (Sep 11). End of the project window. |

**Recurring figures / terms the world is built around:** Salvador Allende & the Unidad Popular
government; Stafford Beer (designer of Cybersyn, author of the Viable System Model); Fernando
Flores; CORFO (the production-development corporation central to nationalised industry).

---

## 4. The 16×48 coverage matrix (the spine of the knowledge base)

**16 thematic categories × 48 months = 768 cells.** The pipeline's job is to fill every cell
with enough verified period sources. A cell is "100% covered" at **5 verified documents**
(`coverage_score = min(1.0, docs_found / 5)`).

### The 16 categories

| ID | Category | Scope |
|---|---|---|
| 1 | Política Nacional | Government ministries, parliament, parties, civil unrest. |
| 2 | Salud | Healthcare, public-health policy, medical services. |
| 3 | Economía Nacional | GDP, inflation, economic policy, fiscal matters. |
| 4 | Industria Nacional | State industries, nationalisation, CORFO, production. |
| 5 | Industria Privada | Private enterprises, commerce, business. |
| 6 | Estados Unidos | US relations, CIA, ITT, economic blockade, US diplomacy. |
| 7 | Unión Soviética | Soviet relations, Cuba, communist support. |
| 8 | Política Internacional | International relations beyond USSR/USA, trade, diplomacy. |
| 9 | Educación | Education policy, schools, universities, reform. |
| 10 | Ciencia y Tecnología | Scientific research, technology, innovation. |
| 11 | Transporte | Ports, railways, aviation, trucks (1972 trucker strike). |
| 12 | Urbanismo y Megaproyectos | Urban planning, large construction, housing. |
| 13 | Política Militar | Military, armed forces, defence, the 1973 coup. |
| 14 | Artes y Cultura | Arts, literature, music, cultural events. |
| 15 | Deporte | Sports, athletes, sporting events. |
| 16 | Macroeconomía | Overall economic indicators, monetary policy, banking. |

### Source tiers (which sources are queried per category)

| Tier | Categories | Sources queried |
|---|---|---|
| `critica` | Economía, Industria, Transporte, Macroeconomía | archive.org, openalex, crossref, wikisource_es, frus, semantic_scholar, chronicling_america, marxists, web_serp |
| `alta` | Política Nacional, Industria Privada, EEUU, Ciencia, Política Militar | archive.org, openalex, crossref, wikisource_es, semantic_scholar, marxists, wikipedia_es |
| `media` | Salud, URSS, Política Int., Educación, Urbanismo | archive.org, openalex, wikisource_es, wikipedia_es |
| `baja` | Artes, Deporte | openalex, wikisource_es, archive.org, wikipedia_es |

---

## 5. How the knowledge base is produced (the pipeline)

A **linear harvesting pipeline** backed by a **single SQLite database**
(`data/archivo/archivo.sqlite`), with all outbound HTTP funnelled through one gatekeeper.
Agents are **CLI tools, not long-running services** — only the Gatekeeper and the Admin
dashboard stay up.

```
Propositor → Gatekeeper → Archivero → Verificator → Cleaner → Mapper → Cast Manager → Cast Director → …
```

| # | Step | Type | Role | Status |
|---|---|---|---|---|
| 1 | **Propositor** (RE-1) | Agent · Gemma | Reads the matrix, scores coverage gaps, emits ranked search missions. | Real |
| 2 | **Gatekeeper** | Software / class | Fronts ALL HTTP: per-domain rate limits, circuit breaker, 7-day disk cache (:8001). | Real |
| 3 | **Archivero** (Ag-1) | Agent | Fetches docs per mission, extracts clean text, dedups by SHA-256, writes `documents`. | Real |
| 4 | **Verificator** (Ag-2) | Agent · Gemma | Quality-scores each doc [0.0–1.0] for Cybersyn/Allende/Flores relevance. | Real |
| 5 | **Cleaner** | Logic (in Verificator) | Drops docs below the relevance threshold. | Real |
| 6 | **Mapper** (Ag-3) | Agent · Gemma | Assigns each verified doc to a precise matrix cell (1–3 ranked categories + month). | Real |
| 7 | **Cast Manager** | Agent | Extracts people mentioned, keeps a per-character timeline, feeds entity-driven missions back to the Propositor. | **Rebuild** |
| 8 | **Cast Director** | Agent | Extracts complex relations: affective, familial, political, professional, enemies. | **Rebuild** |
| … | *more steps* | | To be added as the design is finalised. | Planned |

**Data flow detail:**

```
Propositor ──queries──▶ Gatekeeper ──HTTP──▶ APIs ──raw──▶ Archivero
                                                              │
                                                              ▼
                                                  documents (dedup by SHA-256)
                                                              │
                          ┌─score < .7─▶ Cleaner ──drop       ▼
  Verificator (Gemma) ◀───┤              (delete row)
                          └─score ≥ .7─▶ Mapper (Gemma) ──category + month
                                                              │
                                                              ▼
                                              coverage table (16×48 heatmap)
                                                              │
                                                              ▼
                                      Cast Manager ──people + timelines
                                              ├──▶ feeds Propositor (entity-driven missions)
                                              ▼
                                      Cast Director ──relations, affiliations, …
                                                              │
                                                              ▼
                                                       (runtime / game layer)
```

**Two mission drivers (intended end state):** today the Propositor's queue is fed by
**matrix coverage gaps**. Once the Cast Manager exists, it is *also* fed by **newly-surfaced
characters** ("go research this person") — i.e. **entity-driven missions** alongside gap-driven ones.

**Relevance threshold (canonical):** score scale **0.0–1.0**, keep if **≥ 0.7**. *(Current code
still defaults to `min_score = 0.3`; raising it to 0.7 is a queued change.)*

**Mission outcomes:** DONE (≥1 doc found) · PENDING-reset (network-only failure, retried) ·
FAILED (sources reachable but no extractable content).

---

## 6. What the knowledge base contains (data model)

**SQLite is the single source of truth** (`data/archivo/archivo.sqlite`). JSON exports are
derived views — never inputs. Three core tables today:

| Table | Written by | Role |
|---|---|---|
| `missions` | Propositor | Gap-ranked search tasks. Lifecycle `pending → running → done \| failed`. |
| `documents` | Archivero (+ Verificator, Mapper stamps) | All fetched documents, deduped by SHA-256. |
| `coverage` | Archivero / Mapper | Per-cell score for the 16×48 matrix. |

**`documents` columns of note:** `doc_id` / `sha256` (identity), `title` / `text` / `provenance`,
`quality_score` (Verificator), `verified_at`, `mapped_category_id` (primary 1–16),
`mapped_categories` (JSON array of 1–3 ranked ids), `mapped_month_iso`,
`doc_tag` (`'context'` = outside the Oct 1969–Sep 1973 window).

**Cast tables (to be designed during rebuild):** `characters` + `character_timeline` (Cast
Manager) and a `character_relations` layer (Cast Director), each relation pointing back to the
source document(s) that justify it.

### Pydantic schemas (`lib/schemas.py`) — the intended end state

These describe where the project is going; many consuming agents are still stubs.

- **Archive & missions:** `Document`, `SourceKind` (PRESS, GOVERNMENT, ACADEMIC, ARCHIVE,
  INTERVIEW, CORRESPONDENCE, OTHER), `Mission`, `FetchRequest`, `FetchResponse`, `DomainStatus`.
- **Dialogue & speech acts (Searle / Flores–Winograd):**
  - `SpeechAct` — ASSERTIVE, DIRECTIVE, COMMISSIVE, EXPRESSIVE, DECLARATIVE.
  - `CommitmentAction` — REQUEST, OFFER, PROMISE, COUNTER_OFFER, DECLINE, CANCEL,
    DECLARE_COMPLETE, DECLARE_SATISFIED, WITHDRAW.
  - `CommitmentStage` — PREPARATION → NEGOTIATION → PERFORMANCE → ACCEPTANCE →
    CLOSED_SATISFIED / CLOSED_BROKEN.
  - `Utterance`, `Commitment`, `DialogueExample` (an SFT/QLoRA training example).
- **Narrative & game:**
  - `NPCPersona` — biography, speech_style, political_position, relationships, rag_anchors,
    portrait_asset.
  - `NarrativeNode`, `NarrativeBranch` — branching story graph.
  - `Scenario`, `Indicator`, `MetricSpec`, `GameRules`, `EventTemplate` — Operations-Room
    metrics and scenario definitions.

---

## 7. Sources the pipeline harvests

**Active connectors:** OpenAlex, CrossRef, Internet Archive, Wikisource ES, FRUS (US State Dept
declassified docs, via archive.org), DuckDuckGo (last-resort search), BN Digital Chile (wired but
untiered), plus four newer ones — **Semantic Scholar, Chronicling America, Wikipedia ES,
Marxists.org (Allende archive)**.

**Deferred (selected, pending a design call):** SciELO, HathiTrust, BCN (Congreso Nacional),
Wikidata SPARQL (to be built when the Cast Manager — its consumer — exists).

Each "connection" has three halves: a rate-limit policy in the Gatekeeper, a URL builder +
extractor in the Archivero, and a tier entry in the Propositor.

---

## 8. The game / runtime layer (where this is heading)

Distinct from the research pipeline, the **runtime layer** is the eventual game loop.

- **Director** (Real) — a custom, framework-free orchestrator: one mutable `WorldState` flows
  through a sequence of registered async nodes, with hooks for transitions, errors and
  checkpoints. No LangGraph / CrewAI by design — the loop stays explicit and debuggable.
- **Still to build (stubs):** Event Generator (tension-calibrated narrative events), NPC Pool
  (persona management), Ledger / Logger / Registrador (runtime bookkeeping), plus analysis
  agents (Comparador, Evaluador) and design agents (Diseñador Narrativo, Game Designer, Level
  Designer, Curador Floresiano, Validador Histórico).
- **Experiment switch:** `EXPERIMENT_CONDITION` = `baseline` or `floresian` — flips NPCs to a
  QLoRA-adapted model when available (to test whether Floresian dialogue theory changes play).

**The throughline:** documents → matrix coverage → cast & relations → NPC personas + narrative
nodes/branches → scenarios with metrics → an Ops-Room-styled, source-grounded simulation.

---

## 9. Current status & what's next

- **Built (Real):** Propositor, Gatekeeper, Archivero, Verificator + Cleaner, Mapper, Director,
  baseline Admin dashboard.
- **Rebuild targets (confirmed):** Cast Manager (step 7) and Cast Director (step 8). Context:
  ~4 weeks of work — around 6 new agents plus heavy Admin enhancements, just before a planned web
  deployment — were lost when an unpushed local repo was reformatted. The roadmap is the recovery
  queue.
- **Open design questions to resolve:**
  - Cast schema (characters / timelines / relations) and whether extraction is LLM-driven or
    rule-based first; coreference / de-duplication of the same person named different ways.
  - How the Propositor ingests new-character signals (push table vs. polled query vs. new
    mission type).
  - Conflict resolution when sources disagree (the classic historiography problem) and
    provenance for every extracted relation.
- **Planned additions (newly queued):**
  - **Admin → Performance page.** Every full pipeline run (`make pipeline`) mints a shared
    **run id**; each agent/section stamps its timing against it, so the page can show how long
    each session took and break it down per agent and per pipeline section.
  - **Admin → Results page.** Shows what the *latest* pipeline instance produced: the newly
    **verified documents** and the newly **added characters** (keyed by that run's id).
  - **Pipeline step: key-gated APIs.** A dedicated step / config point for adding sources whose
    APIs require **manually-supplied API keys** (kept in `.env` / `lib.config.settings`),
    distinct from the current keyless connectors.
- **Goal:** restore to the pre-loss state and deploy to the web.

---

## 10. Glossary (quick reference)

| Term | Meaning |
|---|---|
| Project Cybersyn (Synco) | Stafford Beer's real-time cybernetic management system for Allende's economy (1971–73); the Operations Room is its icon. |
| Mission | A search task for one matrix cell, generated by the Propositor. |
| Coverage cell | One (category, month) pair; 768 in total. |
| Source tier | critica / alta / media / baja — decides which sources a category queries. |
| Context tag | A document outside Oct 1969–Sep 1973, tagged `doc_tag='context'`. |
| FRUS | Foreign Relations of the United States — declassified US State Dept documents. |
| Speech act (Searle) | Assertive / Directive / Commissive / Expressive / Declarative. |
| Commitment cycle (Flores–Winograd) | Preparation → Negotiation → Performance → Acceptance → Closed. |
| EXPERIMENT_CONDITION | `baseline` vs `floresian` NPC dialogue model. |
