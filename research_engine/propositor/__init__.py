"""
RE-1 Propositor — detects gaps in the theme × genre × month coverage matrix and generates search missions.

Operation cycle:
1. Read coverage scores from SQLite (table `coverage`); absent cells default to 0.0.
2. Rank all 9,984 (theme × genre × month) cells by gap × thematic relevance ×
   genre plausibility, with a boost on critical months.
3. Take the configured top fraction, skipping cells already queued (any status).
4. Generate queries with period-appropriate vocabulary; optionally reformulate with Ollama.
5. Poll the Cast Manager's `needs_research` character queue and mint entity missions
   ("research that person") alongside the gap missions.
6. Persist missions in SQLite (table `missions`) and data/missions.json.

CLI:
    uv run propositor run [--top-pct 0.2] [--limit 50] [--entity-limit 20] [--no-llm]
                          [--focus mixed|cells|characters]
    uv run propositor status

--focus characters (the `make pipeline-characters` mode) skips gap missions and
mints entity missions only, chasing the least-complete characters first.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncIterator

from lib.config import settings
from lib.genres import GENRES, Genre, genre_by_id
from lib.llm import LLMClient, LLMError
from lib.logging_setup import get_logger
from lib.project import project
from lib.run_tracker import current_run_id, record_stage
from lib.schemas import Mission, MissionKind, MissionStatus
from pipeline.cast_manager import character_key as _character_key
from pipeline.cast_manager import fetch_characters_needing_research
from pipeline.location_manager import fetch_locations_needing_research

log = get_logger("propositor")

# ── Constants ──────────────────────────────────────────────────────────────────

_LLM_TEMPERATURE: float = 0.4
_LLM_NUM_PREDICT: int = 256
_CRITICAL_PRIORITY_BOOST: float = 0.1


# ── Research grid (from the active project spec) ───────────────────────────────
#
# Themes, critical months, source tiers and period vocabulary are declared in the
# project YAML (see lib/project.py). The module-level names below are kept so the
# rest of this module (and its tests) read exactly as before.


@dataclass(frozen=True)
class Category:
    """A theme row of the matrix, as the Propositor ranks it."""

    id: int
    name: str
    weight: float
    tier: str


_PROJECT = project()

_CATEGORIES: list[Category] = [
    Category(theme.id, theme.name, theme.weight, theme.tier) for theme in _PROJECT.themes
]

_CRITICAL_MONTHS: dict[str, str] = _PROJECT.critical_month_events()

# tier → ordered connector list; genre-first routing tops up from here.
_SOURCES: dict[str, list[str]] = {tier: list(srcs) for tier, srcs in _PROJECT.source_tiers.items()}

# Period-appropriate search terms per theme.
_BASE_TERMS: dict[int, list[str]] = _PROJECT.base_terms()

# Region keyword appended to cell queries ("Chile").
_REGION: str = _PROJECT.region

_MONTHS: list[str] = _PROJECT.months()


# ── SQLite helpers ─────────────────────────────────────────────────────────────


def _ensure_missions_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS missions (
            mission_id            TEXT PRIMARY KEY,
            category              TEXT NOT NULL,
            category_id           INTEGER NOT NULL,
            month_iso             TEXT NOT NULL,
            priority              REAL NOT NULL,
            coverage_score_before REAL NOT NULL,
            search_queries        TEXT NOT NULL,
            target_sources        TEXT NOT NULL,
            rationale             TEXT NOT NULL,
            deadline              TEXT,
            status                TEXT NOT NULL DEFAULT 'pending',
            created_at            TEXT NOT NULL,
            updated_at            TEXT NOT NULL,
            llm_reformulated      INTEGER NOT NULL DEFAULT 0,
            kind                  TEXT NOT NULL DEFAULT 'gap',
            character_id          TEXT,
            genre_id              INTEGER
        )
    """)
    _migrate_missions_schema(conn)
    conn.commit()


def _migrate_missions_schema(conn: sqlite3.Connection) -> None:
    """Add the kind/character_id columns to a pre-entity-missions table (idempotent)."""
    for ddl in (
        "ALTER TABLE missions ADD COLUMN kind TEXT NOT NULL DEFAULT 'gap'",
        "ALTER TABLE missions ADD COLUMN character_id TEXT",
        "ALTER TABLE missions ADD COLUMN genre_id INTEGER",
        "ALTER TABLE missions ADD COLUMN location_id TEXT",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # Column already present


def _read_coverage(conn: sqlite3.Connection) -> dict[tuple[int, int, str], float]:
    """Read (category_id, genre_id, month_iso) → score. Tolerates the legacy 2D
    table (rows count as genre 0) so a cycle never crashes mid-migration."""
    try:
        rows = conn.execute(
            "SELECT category_id, genre_id, month_iso, coverage_score FROM coverage"
        ).fetchall()
        return {(int(r[0]), int(r[1]), str(r[2])): float(r[3]) for r in rows}
    except Exception:
        try:
            rows = conn.execute(
                "SELECT category_id, month_iso, coverage_score FROM coverage"
            ).fetchall()
            return {(int(r[0]), 0, str(r[1])): float(r[2]) for r in rows}
        except Exception:
            return {}


def _read_queued_cells(conn: sqlite3.Connection) -> set[tuple[int, int, str]]:
    """Return (category_id, genre_id, month_iso) triples with a gap mission of any status.

    Legacy gap missions without a genre map to genre 0 (they block no real genre
    cell). Entity missions carry a nominal cell but must not block gap generation.
    """
    try:
        rows = conn.execute(
            "SELECT DISTINCT category_id, COALESCE(genre_id, 0), month_iso "
            "FROM missions WHERE kind = 'gap'"
        ).fetchall()
        return {(int(r[0]), int(r[1]), str(r[2])) for r in rows}
    except Exception:
        return set()


def _read_queued_characters(conn: sqlite3.Connection) -> set[str]:
    """Return character_ids with an entity mission still **in flight** (pending/running).

    Only in-flight missions suppress re-minting. A character whose earlier entity
    mission is done/failed can be researched again once the Cast Manager re-flags it
    (a later attempt with a fresh mission) — that is how the character research loop
    iterates instead of firing exactly once per character.
    """
    try:
        rows = conn.execute(
            "SELECT DISTINCT character_id FROM missions "
            "WHERE kind = 'entity' AND character_id IS NOT NULL "
            "AND status IN ('pending', 'running')"
        ).fetchall()
        return {str(r[0]) for r in rows}
    except Exception:
        return set()


def _upsert_missions(conn: sqlite3.Connection, missions: list[Mission]) -> None:
    now = datetime.now(UTC).isoformat()
    conn.executemany(
        """
        INSERT OR REPLACE INTO missions
            (mission_id, category, category_id, month_iso, priority,
             coverage_score_before, search_queries, target_sources,
             rationale, deadline, status, created_at, updated_at, llm_reformulated,
             kind, character_id, genre_id, location_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                m.mission_id,
                m.category,
                m.category_id,
                m.month_iso,
                m.priority,
                m.coverage_score_before,
                json.dumps(m.search_queries, ensure_ascii=False),
                json.dumps(m.target_sources, ensure_ascii=False),
                m.rationale,
                m.deadline,
                m.status.value,
                m.created_at.isoformat(),
                now,
                int(m.llm_reformulated),
                m.kind.value,
                m.character_id,
                m.genre_id,
                m.location_id,
            )
            for m in missions
        ],
    )
    conn.commit()


# ── Missions snapshot (human-readable JSON) ────────────────────────────────────


def _write_missions_snapshot(path: Path, conn: sqlite3.Connection) -> None:
    """Rewrite the full missions JSON from the current DB state, sorted by priority."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        """SELECT mission_id, category, category_id, month_iso, priority,
                  coverage_score_before, search_queries, target_sources,
                  rationale, status, created_at, updated_at, llm_reformulated,
                  kind, character_id, genre_id
           FROM missions ORDER BY priority DESC, month_iso ASC"""
    ).fetchall()
    missions = [
        {
            "mission_id": r[0],
            "category": r[1],
            "category_id": r[2],
            "month_iso": r[3],
            "priority": r[4],
            "coverage_score_before": r[5],
            "search_queries": json.loads(r[6]),
            "target_sources": json.loads(r[7]),
            "rationale": r[8],
            "status": r[9],
            "created_at": r[10],
            "updated_at": r[11],
            "llm_reformulated": bool(r[12]),
            "kind": r[13],
            "character_id": r[14],
            "genre_id": r[15],
        }
        for r in rows
    ]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(missions, fh, ensure_ascii=False, indent=2)


# ── Duplicate cleanup ──────────────────────────────────────────────────────────


def _reset_failed_missions(conn: sqlite3.Connection) -> int:
    """Reset all failed missions back to pending so they are retried on the next cycle."""
    cur = conn.execute(
        "UPDATE missions SET status='pending', updated_at=? WHERE status='failed'",
        (datetime.now(UTC).isoformat(),),
    )
    conn.commit()
    return cur.rowcount


def _cleanup_duplicates(conn: sqlite3.Connection) -> int:
    """Remove duplicate missions, keeping the best per logical target.

    Gap missions are duplicates per (category_id, month_iso); entity missions per
    character_id — two characters may share the same nominal cell without colliding.
    Gap retention: done > pending > running > failed. **Entity** retention: newest
    wins (created_at DESC), so a fresh research attempt supersedes the character's
    earlier done mission and the loop can iterate.
    Returns number of rows deleted.
    """
    conn.execute("""
        WITH best AS (
            SELECT mission_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY CASE
                           WHEN kind = 'entity'
                               THEN 'entity:' || COALESCE(character_id, mission_id)
                           WHEN kind = 'location'
                               THEN 'location:' || COALESCE(location_id, mission_id)
                           ELSE 'gap:' || category_id || ':'
                                || COALESCE(genre_id, 0) || ':' || month_iso
                       END
                       ORDER BY CASE
                           WHEN kind IN ('entity', 'location') THEN 0
                           WHEN status = 'done'    THEN 0
                           WHEN status = 'pending' THEN 1
                           WHEN status = 'running' THEN 2
                           WHEN status = 'failed'  THEN 3
                       END ASC, created_at DESC
                   ) AS rn
            FROM missions
        )
        DELETE FROM missions WHERE mission_id NOT IN (
            SELECT mission_id FROM best WHERE rn = 1
        )
    """)
    # cursor.rowcount is -1 for WITH-prefixed DELETEs; ask SQLite directly.
    deleted = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    return int(deleted)


# ── Mission construction ───────────────────────────────────────────────────────


def _base_queries(cat_id: int, genre: Genre, month_iso: str) -> list[str]:
    """Three queries for a (theme, genre, month) cell: genre+theme, theme alone,
    and a month-specific genre query so APIs return unique results per cell."""
    terms = _BASE_TERMS.get(cat_id, [])
    cat_name = next((c.name for c in _CATEGORIES if c.id == cat_id), "")
    genre_term = genre.search_terms[0]

    queries: list[str] = []
    if terms:
        queries.append(f"{genre_term} {terms[0]}")
        if len(terms) > 1:
            queries.append(terms[1])
    event = _CRITICAL_MONTHS.get(month_iso)
    if event:
        queries.append(f"{genre_term} {cat_name} {_REGION} {month_iso} {event.split(';')[0].strip()}")
    else:
        queries.append(f"{genre_term} {cat_name} {_REGION} {month_iso}")
    return queries or [f"{_REGION} {month_iso}"]


# Cap mission sources so a single mission never fans out across every connector.
_MAX_MISSION_SOURCES: int = 6


def _mission_sources(genre: Genre, tier: str) -> list[str]:
    """Route sources genre-first, topped up from the category's tier list."""
    sources = list(genre.sources)
    for source in _SOURCES[tier]:
        if source not in sources:
            sources.append(source)
        if len(sources) >= _MAX_MISSION_SOURCES:
            break
    return sources[:_MAX_MISSION_SOURCES]


def _priority(weight: float, plausibility: float, score: float, month_iso: str) -> float:
    """Cell priority: gap × thematic weight × genre plausibility (+ critical-month boost)."""
    p = (1.0 - score) * weight * plausibility
    if month_iso in _CRITICAL_MONTHS:
        p = min(1.0, p + _CRITICAL_PRIORITY_BOOST)
    return round(p, 4)


def _rationale(cat_name: str, genre_name: str, month_iso: str, score: float) -> str:
    event = _CRITICAL_MONTHS.get(month_iso)
    suffix = f" — {event}" if event else ""
    return f"{cat_name} · {genre_name} · {month_iso}{suffix}; cobertura actual {score:.2f}."


# ── Entity missions (Cast Manager wiring) ─────────────────────────────────────

# Entity missions are person-driven, not cell-driven. They still need a nominal
# (category, month) to satisfy the missions schema; the Archivero skips the
# coverage-matrix update for kind='entity' so this cell is never distorted.
_ENTITY_CATEGORY_ID: int = _PROJECT.entity.nominal_theme_id  # nominal only
_ENTITY_MONTH_ISO: str = _PROJECT.entity.nominal_month_iso  # nominal only
_ENTITY_SOURCES: list[str] = list(_PROJECT.entity.sources)
_ENTITY_MENTION_SATURATION: int = 5


def _row_value(row: object, key: str) -> object | None:
    """Read ``key`` from a sqlite3.Row or a plain dict, None if absent."""
    try:
        return row[key]  # type: ignore[index]
    except (KeyError, IndexError):
        return None


def _row_aliases(row: object) -> list[str]:
    """Parse a character row's ``aliases`` JSON column into a list of strings."""
    raw = _row_value(row, "aliases")
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [alias for alias in decoded if isinstance(alias, str)]


def _fact_fragment(fact: str | None) -> str | None:
    """Condense a timeline fact into a short, quotable search fragment, or None."""
    if not fact:
        return None
    words = str(fact).strip().split()
    fragment = " ".join(words[:6]).strip(" .,;:")
    return fragment or None


def _distinct_alias(name: str, aliases: list[str]) -> str | None:
    """The first alias that names a genuinely different surface form, or None."""
    name_key = _character_key(name)
    for alias in aliases:
        if not isinstance(alias, str):
            continue
        alias = alias.strip()
        if alias and _character_key(alias) != name_key:
            return alias
    return None


def entity_queries(
    name: str,
    aliases: list[str] | None = None,
    salient_fact: str | None = None,
) -> list[str]:
    """Build up to three search queries seeding on one character. Deterministic.

    Beyond the exact quoted name, the queries draw on what the cast already knows:
    a **salient fact** (the character's affiliation/role) sharpens one query toward
    the right person (e.g. `"Donald Kendall" president of Pepsi-Cola`), and a
    **distinct alias** broadens recall. Names are exact-match terms, so no LLM
    reformulation — the entity path stays fully offline.
    """
    name = name.strip()
    entity = _PROJECT.entity
    queries = [f'"{name}" {entity.query_suffix}']

    fragment = _fact_fragment(salient_fact)
    if fragment:
        queries.append(f'"{name}" {fragment}')
    else:
        queries.append(f'"{name}" {entity.context_suffix}')

    alias = _distinct_alias(name, aliases or [])
    if alias:
        queries.append(f'"{alias}" {entity.query_suffix}')
    else:
        queries.append(f"{name} {entity.biography_suffix}")

    return queries[:3]


def entity_priority(mention_count: int, completeness_score: float) -> float:
    """Priority of researching a character, comparable to gap priorities (0 to 1).

    Grows with how often the character is mentioned (saturating at
    _ENTITY_MENTION_SATURATION) and shrinks with how complete their record already is.
    """
    mention_weight = min(mention_count, _ENTITY_MENTION_SATURATION) / _ENTITY_MENTION_SATURATION
    gap = 1.0 - max(0.0, min(1.0, completeness_score))
    return round(gap * (0.5 + 0.5 * mention_weight), 4)


def _entity_rationale(name: str, mention_count: int, completeness_score: float) -> str:
    return (
        f"Personaje «{name}» — misión de entidad; "
        f"{mention_count} menciones, completitud {completeness_score:.2f}."
    )


def order_by_need(characters: list) -> list:
    """Reorder a research queue least-complete-first (ties: most-mentioned first).

    The ``--focus characters`` mode chases the thinnest profiles instead of the
    most prominent ones — the default queue order (mention_count DESC) favours
    protagonists; this one favours the characters furthest from complete.
    """
    return sorted(
        characters,
        key=lambda row: (
            float(row["completeness_score"] or 0.0),
            -int(row["mention_count"] or 0),
        ),
    )


def build_entity_missions(
    characters: list,
    queued_character_ids: set[str],
    limit: int,
) -> list[Mission]:
    """Turn needs_research characters into entity missions, preserving input order.

    Characters that already have an entity mission (any status) are skipped.
    Rows beyond `limit` are left for the next cycle. Returns the new missions.
    """
    missions: list[Mission] = []
    for row in characters:
        if len(missions) >= limit:
            break
        character_id = row["character_id"]
        if character_id in queued_character_ids:
            continue
        name = row["name"]
        mention_count = int(row["mention_count"])
        completeness = float(row["completeness_score"])
        aliases = _row_aliases(row)
        salient = _row_value(row, "salient_fact")
        salient_fact = salient if isinstance(salient, str) else None
        missions.append(
            Mission(
                mission_id=f"e-{uuid.uuid4().hex[:8]}",
                category="Política Nacional",
                category_id=_ENTITY_CATEGORY_ID,
                month_iso=_ENTITY_MONTH_ISO,
                priority=entity_priority(mention_count, completeness),
                coverage_score_before=0.0,
                search_queries=entity_queries(name, aliases, salient_fact),
                target_sources=list(_ENTITY_SOURCES),
                rationale=_entity_rationale(name, mention_count, completeness),
                status=MissionStatus.PENDING,
                kind=MissionKind.ENTITY,
                character_id=character_id,
            )
        )
    return missions


def _clear_needs_research(conn: sqlite3.Connection, character_ids: list[str]) -> None:
    """Unflag characters whose research mission now exists (queue consumed)."""
    if not character_ids:
        return
    now = datetime.now(UTC).isoformat()
    conn.executemany(
        "UPDATE characters SET needs_research = 0, updated_at = ? WHERE character_id = ?",
        [(now, character_id) for character_id in character_ids],
    )
    conn.commit()


def _record_research_attempt(conn: sqlite3.Connection, character_ids: list[str]) -> None:
    """Count a research attempt per newly-minted character (loop bound + cooldown).

    Each entity mission minted is one attempt: bump ``research_attempts`` and stamp
    ``last_research_at``. The Cast Manager's ``should_research`` uses both to cap the
    number of attempts and space them out. Tolerates DBs predating the columns.
    """
    if not character_ids:
        return
    now = datetime.now(UTC).isoformat()
    try:
        conn.executemany(
            "UPDATE characters SET research_attempts = COALESCE(research_attempts, 0) + 1, "
            "last_research_at = ? WHERE character_id = ?",
            [(now, character_id) for character_id in character_ids],
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass  # legacy DB without the research-tracking columns


# ── Location missions (Location Manager wiring) ────────────────────────────────

# Location missions research a PLACE: architectural documentation, plans, the
# building's history, GPS-worthy data and trivia. Same nominal-cell trick as
# entity missions — the Archivero never writes their cell to the coverage matrix.
_LOCATION_SOURCES: list[str] = [
    "wikipedia_es",
    "archive.org",
    "wikisource_es",
    "openalex",
]

# Query flavour per location kind — what "researching this place" means.
_LOCATION_KIND_TERMS: dict[str, str] = {
    "building": "arquitectura planos historia",
    "factory": "industria producción historia",
    "residence": "casa historia fotografías",
    "city": "historia urbana fotografías",
    "region": "historia geografía",
    "street": "historia fotografías",
    "office": "historia institución",
    "other": "historia fotografías",
}


def location_queries(name: str, kind: str = "other", aliases: list[str] | None = None) -> list[str]:
    """Search queries for a place: identity, kind-flavoured research, alias variant."""
    queries = [f'"{name}" {_REGION}']
    terms = _LOCATION_KIND_TERMS.get(kind, _LOCATION_KIND_TERMS["other"])
    queries.append(f'"{name}" {terms}')
    alias = _distinct_alias(name, aliases or [])
    if alias:
        queries.append(f'"{alias}" {_REGION} historia')
    return queries[:3]


def _read_queued_locations(conn: sqlite3.Connection) -> set[str]:
    """location_ids with a location mission still in flight (pending/running)."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT location_id FROM missions "
            "WHERE kind = 'location' AND location_id IS NOT NULL "
            "AND status IN ('pending', 'running')"
        ).fetchall()
        return {str(r[0]) for r in rows}
    except sqlite3.OperationalError:
        return set()


def build_location_missions(
    locations: list,
    queued_location_ids: set[str],
    limit: int,
) -> list[Mission]:
    """Turn needs_research locations into location missions, preserving input order."""
    missions: list[Mission] = []
    for row in locations:
        if len(missions) >= limit:
            break
        location_id = row["location_id"]
        if location_id in queued_location_ids:
            continue
        name = row["name"]
        mention_count = int(_row_value(row, "mention_count") or 0)
        completeness = float(_row_value(row, "completeness_score") or 0.0)
        kind = str(_row_value(row, "kind") or "other")
        missions.append(
            Mission(
                mission_id=f"l-{uuid.uuid4().hex[:8]}",
                category="Política Nacional",
                category_id=_ENTITY_CATEGORY_ID,
                month_iso=_ENTITY_MONTH_ISO,
                priority=entity_priority(mention_count, completeness),
                coverage_score_before=0.0,
                search_queries=location_queries(name, kind, _row_aliases(row)),
                target_sources=list(_LOCATION_SOURCES),
                rationale=(
                    f"Investigar el lugar {name}: {mention_count} menciones, "
                    f"completitud {completeness:.2f}."
                ),
                status=MissionStatus.PENDING,
                kind=MissionKind.LOCATION,
                location_id=location_id,
            )
        )
    return missions


def _clear_location_research(conn: sqlite3.Connection, location_ids: list[str]) -> None:
    if not location_ids:
        return
    now = datetime.now(UTC).isoformat()
    try:
        conn.executemany(
            "UPDATE locations SET needs_research = 0, updated_at = ? WHERE location_id = ?",
            [(now, location_id) for location_id in location_ids],
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass


def _record_location_attempt(conn: sqlite3.Connection, location_ids: list[str]) -> None:
    if not location_ids:
        return
    now = datetime.now(UTC).isoformat()
    try:
        conn.executemany(
            "UPDATE locations SET research_attempts = COALESCE(research_attempts, 0) + 1, "
            "last_research_at = ? WHERE location_id = ?",
            [(now, location_id) for location_id in location_ids],
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass


# ── LLM reformulation ─────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = _PROJECT.reformulation.system


async def _reformulate(
    client: LLMClient, cat_name: str, month_iso: str, queries: list[str]
) -> tuple[list[str], bool]:
    prompt = (
        f"Categoría: {cat_name}\nMes: {month_iso}\n"
        f"Queries base:\n{json.dumps(queries, ensure_ascii=False)}\n\n"
        "Reescribe con vocabulario de época y añade una tercera query específica. "
        "Devuelve JSON array de strings (máximo 3 elementos)."
    )
    try:
        raw = await client.generate(
            model=settings.ollama_model_npc,
            prompt=prompt,
            system=_LLM_SYSTEM_PROMPT,
            temperature=_LLM_TEMPERATURE,
            num_predict=_LLM_NUM_PREDICT,
        )
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if 0 <= start < end:
            parsed = json.loads(raw[start:end])
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed) and parsed:
                return parsed[:3], True
    except (LLMError, json.JSONDecodeError, Exception) as exc:
        log.debug("propositor.llm_skip", reason=str(exc)[:120])
    return queries, False


# ── Null async context (for use_llm=False path) ───────────────────────────────


@asynccontextmanager
async def _no_client() -> AsyncIterator[None]:
    yield None


# ── Propositor ─────────────────────────────────────────────────────────────────


class Propositor:
    def __init__(
        self,
        db_path: Path | None = None,
        missions_json: Path | None = None,
        use_llm: bool = True,
    ) -> None:
        self.db_path = db_path or settings.archive_db
        self.missions_json = missions_json or (settings.data_dir / "missions.json")
        self.use_llm = use_llm

    async def _build_missions(
        self,
        cells: list[tuple[float, int, int, str]],
        coverage: dict[tuple[int, int, str], float],
        client: LLMClient | None,
    ) -> list[Mission]:
        cat_lookup = {cat.id: cat for cat in _CATEGORIES}
        missions: list[Mission] = []

        for priority, cat_id, genre_id, month_iso in cells:
            cat = cat_lookup[cat_id]
            genre = genre_by_id(genre_id)
            score = coverage.get((cat_id, genre_id, month_iso), 0.0)
            queries = _base_queries(cat_id, genre, month_iso)
            llm_ok = False

            if client is not None:
                queries, llm_ok = await _reformulate(client, cat.name, month_iso, queries)

            missions.append(
                Mission(
                    mission_id=f"m-{uuid.uuid4().hex[:8]}",
                    category=cat.name,
                    category_id=cat_id,
                    month_iso=month_iso,
                    priority=priority,
                    coverage_score_before=score,
                    search_queries=queries,
                    target_sources=_mission_sources(genre, cat.tier),
                    rationale=_rationale(cat.name, genre.name, month_iso, score),
                    status=MissionStatus.PENDING,
                    llm_reformulated=llm_ok,
                    genre_id=genre_id,
                )
            )
        return missions

    def _mint_entity_missions(
        self, conn: sqlite3.Connection, limit: int, neediest_first: bool = False
    ) -> list[Mission]:
        """Consume the Cast Manager's needs_research queue into entity missions.

        Persists the new missions, then unflags every character that now has a
        mission (newly minted or already queued). Characters beyond `limit` stay
        flagged for the next cycle. Returns the new missions. With
        ``neediest_first`` (the ``--focus characters`` mode) the queue is
        reordered least-complete-first instead of most-mentioned-first.
        """
        if limit <= 0:
            return []
        characters = fetch_characters_needing_research(conn)
        if not characters:
            return []
        if neediest_first:
            characters = order_by_need(characters)

        queued = _read_queued_characters(conn)
        entity_missions = build_entity_missions(characters, queued, limit)
        if entity_missions:
            _upsert_missions(conn, entity_missions)

        minted = {m.character_id for m in entity_missions}
        consumed = [
            row["character_id"]
            for row in characters
            if row["character_id"] in minted or row["character_id"] in queued
        ]
        _clear_needs_research(conn, consumed)
        # One minted mission = one research attempt (bounds and paces the loop).
        _record_research_attempt(conn, [m.character_id for m in entity_missions])

        log.info(
            "propositor.entity_missions",
            queue=len(characters),
            minted=len(entity_missions),
            already_queued=len(consumed) - len(minted),
        )
        return entity_missions

    def _mint_location_missions(self, conn: sqlite3.Connection, limit: int) -> list[Mission]:
        """Consume the Location Manager's needs_research queue into location missions.

        Same lifecycle as entity missions: mint up to ``limit``, unflag every
        location that now has a mission, count one research attempt per mint.
        """
        if limit <= 0:
            return []
        locations = fetch_locations_needing_research(conn)
        if not locations:
            return []

        queued = _read_queued_locations(conn)
        location_missions = build_location_missions(locations, queued, limit)
        if location_missions:
            _upsert_missions(conn, location_missions)

        minted = {m.location_id for m in location_missions}
        consumed = [
            row["location_id"]
            for row in locations
            if row["location_id"] in minted or row["location_id"] in queued
        ]
        _clear_location_research(conn, consumed)
        _record_location_attempt(conn, [m.location_id for m in location_missions])

        log.info(
            "propositor.location_missions",
            queue=len(locations),
            minted=len(location_missions),
        )
        return location_missions

    async def run_cycle(
        self,
        top_fraction: float = 0.2,
        max_new_missions: int = 50,
        retry_failed: bool = False,
        min_queue: int = 0,
        max_entity_missions: int = 20,
        max_location_missions: int = 10,
        focus: str = "mixed",
    ) -> list[Mission]:
        """Run one discovery cycle. Returns list of newly created missions
        (gap missions from the coverage matrix + entity missions from the
        Cast Manager's needs_research queue).

        If min_queue > 0, skips all generation when there are already that many
        pending missions — lets the queue drain before adding more work.
        max_entity_missions = 0 disables the entity path entirely.

        ``focus`` selects the mission mix: ``mixed`` (default — matrix gaps +
        entity queue, today's behaviour), ``cells`` (gap missions only) or
        ``characters`` (entity missions only, queue reordered
        least-complete-first — the `make pipeline-characters` mode).
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # character rows are read by column name
        try:
            _ensure_missions_table(conn)

            if min_queue > 0:
                pending_count = conn.execute(
                    "SELECT COUNT(*) FROM missions WHERE status='pending'"
                ).fetchone()[0]
                if pending_count >= min_queue:
                    log.info(
                        "propositor.skip_min_queue",
                        pending=pending_count,
                        threshold=min_queue,
                    )
                    _write_missions_snapshot(self.missions_json, conn)
                    return []

            if retry_failed:
                reset = _reset_failed_missions(conn)
                if reset:
                    log.info("propositor.failed_missions_reset", count=reset)

            deleted = _cleanup_duplicates(conn)
            if deleted:
                log.info("propositor.duplicates_removed", count=deleted)

            missions: list[Mission] = []
            if focus != "characters":
                coverage = _read_coverage(conn)
                queued = _read_queued_cells(conn)

                # Rank all uncovered (theme, genre, month) cells by priority descending.
                # Plausibility weights keep structurally-empty pairings at the bottom.
                ranked: list[tuple[float, int, int, str]] = []
                for cat in _CATEGORIES:
                    for genre in GENRES:
                        plausibility = genre.plausibility(cat.id)
                        for month in _MONTHS:
                            if (cat.id, genre.id, month) in queued:
                                continue
                            score = coverage.get((cat.id, genre.id, month), 0.0)
                            ranked.append(
                                (
                                    _priority(cat.weight, plausibility, score, month),
                                    cat.id,
                                    genre.id,
                                    month,
                                )
                            )
                ranked.sort(reverse=True)

                n_total = len(_CATEGORIES) * len(GENRES) * len(_MONTHS)
                n_target = max(1, int(n_total * top_fraction))
                cells = ranked[: min(n_target, max_new_missions)]

                log.info(
                    "propositor.cycle_start",
                    focus=focus,
                    total_cells=n_total,
                    queued=len(queued),
                    new_target=len(cells),
                    use_llm=self.use_llm,
                )

                ctx = LLMClient() if self.use_llm else _no_client()
                async with ctx as client:
                    missions = await self._build_missions(cells, coverage, client)

                if missions:
                    _upsert_missions(conn, missions)
                    log.info("propositor.missions_saved", count=len(missions))
            else:
                log.info("propositor.cycle_start", focus=focus, use_llm=self.use_llm)

            entity_missions = (
                self._mint_entity_missions(
                    conn, max_entity_missions, neediest_first=focus == "characters"
                )
                if focus != "cells"
                else []
            )
            # Location missions ride along in mixed mode only — the characters
            # focus stays people-only and the cells focus stays matrix-only.
            location_missions = (
                self._mint_location_missions(conn, max_location_missions)
                if focus == "mixed"
                else []
            )

            _write_missions_snapshot(self.missions_json, conn)
        finally:
            conn.close()

        return missions + entity_missions + location_missions

    def mission_summary(self) -> dict[str, int]:
        """Return {status: count} for all missions in the database."""
        if not self.db_path.exists():
            return {}
        conn = sqlite3.connect(str(self.db_path))
        try:
            _ensure_missions_table(conn)
            rows = conn.execute("SELECT status, COUNT(*) FROM missions GROUP BY status").fetchall()
            return {r[0]: r[1] for r in rows}
        finally:
            conn.close()


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    import traceback

    from lib.logging_setup import configure_logging, current_log_file

    configure_logging()

    ap = argparse.ArgumentParser(
        prog="propositor",
        description="RE-1 Propositor — gap detection & mission generation",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run one discovery cycle")
    run_p.add_argument(
        "--top-pct",
        type=float,
        default=0.2,
        help="Fraction of cells to target (default: 0.20 = 20%%)",
    )
    run_p.add_argument(
        "--limit", type=int, default=50, help="Max new missions per run (default: 50)"
    )
    run_p.add_argument("--no-llm", action="store_true", help="Skip Ollama query reformulation")
    run_p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reset all failed missions back to pending before this cycle",
    )
    run_p.add_argument(
        "--min-queue",
        type=int,
        default=0,
        metavar="N",
        help="Skip generation if N or more pending missions already exist (default: 0 = always generate)",
    )
    run_p.add_argument(
        "--entity-limit",
        type=int,
        default=20,
        metavar="N",
        help="Max new entity (character-research) missions per run (default: 20; 0 disables)",
    )
    run_p.add_argument(
        "--location-limit",
        type=int,
        default=10,
        metavar="N",
        help="Max new location (place-research) missions per run (default: 10; 0 disables; mixed focus only)",
    )
    run_p.add_argument(
        "--focus",
        choices=("mixed", "cells", "characters"),
        default="mixed",
        help="Mission mix: mixed = matrix gaps + entity queue (default), cells = gaps only, "
        "characters = entity missions only, least-complete characters first",
    )

    sub.add_parser("status", help="Print mission queue summary")

    args = ap.parse_args()

    if args.cmd == "run":
        log.info("propositor.main_start", cmd="run", limit=args.limit, use_llm=not args.no_llm)
        try:
            propositor = Propositor(use_llm=not args.no_llm)
            t0 = datetime.now(UTC)
            missions = asyncio.run(
                propositor.run_cycle(
                    top_fraction=args.top_pct,
                    max_new_missions=args.limit,
                    retry_failed=args.retry_failed,
                    min_queue=args.min_queue,
                    max_entity_missions=args.entity_limit,
                    max_location_missions=args.location_limit,
                    focus=args.focus,
                )
            )
            entity_count = sum(1 for m in missions if m.kind == MissionKind.ENTITY)
            rid = current_run_id()
            if rid:
                record_stage(
                    rid,
                    "propositor",
                    t0,
                    datetime.now(UTC),
                    {
                        "missions_created": len(missions),
                        "entity_missions": entity_count,
                    },
                )
            log.info(
                "propositor.main_done",
                missions_generated=len(missions),
                entity_missions=entity_count,
            )
        except Exception as exc:
            log.error("propositor.main_crash", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            log_file = current_log_file()
            if log_file:
                print(f"\n  Log: {log_file}")

        if not missions and args.min_queue > 0:
            print(
                f"\nGeneración omitida: ya hay misiones pendientes suficientes (umbral: {args.min_queue})."
            )
        print(f"\n{len(missions)} misiones generadas ({entity_count} de entidad).")
        for mission in missions[:8]:
            llm_tag = " [LLM]" if mission.llm_reformulated else ""
            if mission.kind == MissionKind.ENTITY:
                print(f"  [{mission.priority:.3f}] entidad: {mission.character_id}")
            else:
                genre = genre_by_id(mission.genre_id) if mission.genre_id else None
                genre_name = genre.name if genre else "(sin género)"
                print(
                    f"  [{mission.priority:.3f}] {mission.category:<24} "
                    f"{genre_name:<28} {mission.month_iso}{llm_tag}"
                )
            print(f"          {mission.mission_id}  ->  {mission.search_queries[0]!r}")
        if len(missions) > 8:
            print(f"  … y {len(missions) - 8} más.")

    elif args.cmd == "status":
        propositor = Propositor()
        summary = propositor.mission_summary()
        if not summary:
            print("Sin misiones en la base de datos.")
        else:
            total = sum(summary.values())
            for status, count in sorted(summary.items()):
                print(f"  {status:<12} {count:>4}  ({count / total * 100:.0f}%)")
            print(f"  {'TOTAL':<12} {total:>4}")


if __name__ == "__main__":
    main()
