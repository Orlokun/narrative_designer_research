"""
Ag-8 Cast Director — extracts the relational layer between characters.

Step 8 of the canonical pipeline. Runs after the Cast Manager: where the Cast
Manager records hard facts (who appears, when, in which document), the Cast
Director extracts the harder, inferential information — how the people relate to
one another: family ties, alliances, collegial links, rivalries, enemies,
shared political affiliation, and the directed mentor / superior links.

Strategy: **per document**, mirroring the Cast Manager. Each verified, mapped
document is read once (Gemma, with a ``--no-llm`` no-op fallback) and the
relations it states are extracted, grounded in the people the Cast Manager has
already linked to that document. Every relation cites the document(s) that
justify it (``provenance``); corroboration across documents raises its
``confidence``. Endpoints are resolved to existing characters with the Cast
Manager's coreference (``resolve_character_key``) — a relation is only stored
when both people are already in the cast.

CLI:
    uv run cast-director run [--batch 50] [--all] [--no-llm]
    uv run cast-director status
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from lib.config import settings
from lib.llm import LLMClient, LLMError
from lib.logging_setup import get_logger
from lib.politics import refine_relation_kind
from lib.run_tracker import current_run_id, record_stage
from pipeline.cast_manager import (
    _load_character_roster,
    character_key,
    resolve_character_key,
)

log = get_logger("cast_director")

# ── Constants ──────────────────────────────────────────────────────────────────

_LLM_TEMPERATURE: float = 0.2
_LLM_NUM_PREDICT: int = 512
_MAX_RELATIONS_PER_DOC: int = 30
_CONFIDENCE_SATURATION: int = 3  # documents corroborating a relation → confidence 1.0

# Canonical relation kinds. Symmetric kinds collapse A-B / B-A into one ordered
# row; directed kinds keep their direction (source → target).
_SYMMETRIC_KINDS: frozenset[str] = frozenset(
    {"familia", "aliado", "colega", "rival", "enemigo", "afiliacion", "contraparte", "otro"}
)
_DIRECTED_KINDS: frozenset[str] = frozenset({"superior", "mentor"})

# Bilingual surface form → (canonical kind, invert direction?). Inverse kinds
# (subordinate, student) map to their directed counterpart with the pair flipped.
_RELATION_KIND_MAP: dict[str, tuple[str, bool]] = {
    "family": ("familia", False), "familia": ("familia", False),
    "relative": ("familia", False), "pariente": ("familia", False),
    "ally": ("aliado", False), "allies": ("aliado", False), "aliado": ("aliado", False),
    "alliance": ("aliado", False), "alianza": ("aliado", False), "friend": ("aliado", False),
    "amigo": ("aliado", False),
    "colleague": ("colega", False), "coworker": ("colega", False), "colega": ("colega", False),
    "collaborator": ("colega", False), "colaborador": ("colega", False),
    "rival": ("rival", False),
    "enemy": ("enemigo", False), "enemigo": ("enemigo", False),
    "adversary": ("enemigo", False), "adversario": ("enemigo", False), "opponent": ("enemigo", False),
    "political": ("afiliacion", False), "party": ("afiliacion", False),
    "afiliacion": ("afiliacion", False), "ideological": ("afiliacion", False),
    "comrade": ("afiliacion", False), "camarada": ("afiliacion", False),
    "counterpart": ("contraparte", False), "contraparte": ("contraparte", False),
    "opposite": ("contraparte", False), "counterparts": ("contraparte", False),
    "superior": ("superior", False), "boss": ("superior", False), "jefe": ("superior", False),
    "supervisor": ("superior", False),
    "subordinate": ("superior", True), "subordinado": ("superior", True),
    "employee": ("superior", True), "empleado": ("superior", True),
    "mentor": ("mentor", False), "teacher": ("mentor", False), "maestro": ("mentor", False),
    "student": ("mentor", True), "alumno": ("mentor", True),
    "mentee": ("mentor", True), "disciple": ("mentor", True), "discipulo": ("mentor", True),
}


# ── Pure functions ──────────────────────────────────────────────────────────────


def normalize_relation_kind(kind: str) -> tuple[str, bool]:
    """Map a raw relation kind to (canonical_kind, invert). Unknown → ("otro", False)."""
    return _RELATION_KIND_MAP.get((kind or "").strip().lower(), ("otro", False))


def normalize_relation(
    source_key: str, target_key: str, kind: str
) -> tuple[str, str, str] | None:
    """Canonicalise a relation into (source, target, kind).

    Inverse kinds flip the pair; symmetric kinds order the pair so A-B and B-A
    collapse to one row; directed kinds keep their direction. Returns None for a
    self-relation (same character on both ends).
    """
    canonical, invert = normalize_relation_kind(kind)
    if invert:
        source_key, target_key = target_key, source_key
    if source_key == target_key:
        return None
    if canonical in _SYMMETRIC_KINDS:
        source_key, target_key = sorted((source_key, target_key))
    return source_key, target_key, canonical


@dataclass
class ExtractedRelation:
    """One relation between two named people, extracted from a single document."""

    source: str
    target: str
    kind: str
    description: str = ""


def parse_relations_response(response: str) -> list[ExtractedRelation] | None:
    """Parse the LLM's JSON into ExtractedRelation list.

    Expected shape: ``{"relations": [{"source": str, "target": str, "kind": str,
    "description": str}]}``. Lenient per-entry (an entry missing either endpoint
    is skipped); returns None only when the top-level shape is wrong.
    """
    try:
        data = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("relations"), list):
        return None

    relations: list[ExtractedRelation] = []
    for raw in data["relations"][:_MAX_RELATIONS_PER_DOC]:
        if not isinstance(raw, dict):
            continue
        source = raw.get("source")
        target = raw.get("target")
        if not isinstance(source, str) or not source.strip():
            continue
        if not isinstance(target, str) or not target.strip():
            continue
        kind = raw.get("kind")
        description = raw.get("description")
        relations.append(ExtractedRelation(
            source=source.strip(),
            target=target.strip(),
            kind=kind.strip() if isinstance(kind, str) else "",
            description=description.strip() if isinstance(description, str) else "",
        ))
    return relations


def _confidence(mention_count: int) -> float:
    """Relation confidence rises with corroboration, saturating at 1.0."""
    return round(min(1.0, mention_count / _CONFIDENCE_SATURATION), 3)


# ── Schema ──────────────────────────────────────────────────────────────────────


def ensure_relation_tables(conn: sqlite3.Connection) -> None:
    """Create character_relations and add documents.relations_extracted_at (idempotent)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS character_relations (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            source_character_id TEXT NOT NULL,
            target_character_id TEXT NOT NULL,
            kind                TEXT NOT NULL,
            description         TEXT,
            confidence          REAL NOT NULL DEFAULT 0.0,
            mention_count       INTEGER NOT NULL DEFAULT 0,
            provenance          TEXT NOT NULL DEFAULT '[]',
            first_seen_at       TEXT NOT NULL,
            updated_at          TEXT NOT NULL,
            UNIQUE(source_character_id, target_character_id, kind)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_relations_source "
        "ON character_relations(source_character_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_relations_target "
        "ON character_relations(target_character_id)"
    )
    try:
        conn.execute("ALTER TABLE documents ADD COLUMN relations_extracted_at TEXT")
    except sqlite3.OperationalError:
        pass  # column already present (or documents table absent in some tests)
    conn.commit()


# ── DB helpers ────────────────────────────────────────────────────────────────


def _fetch_unprocessed_documents(conn: sqlite3.Connection, batch_size: int) -> list[sqlite3.Row]:
    """Verified, mapped documents not yet mined for relations."""
    return conn.execute(
        """
        SELECT doc_id, title, text
        FROM documents
        WHERE verified_at IS NOT NULL
          AND mapped_category_id IS NOT NULL
          AND relations_extracted_at IS NULL
        LIMIT ?
        """,
        (batch_size,),
    ).fetchall()


def _doc_known_characters(conn: sqlite3.Connection, doc_id: str) -> list[str]:
    """Names of characters the Cast Manager already linked to this document."""
    try:
        rows = conn.execute(
            """
            SELECT c.name
            FROM character_mentions m
            JOIN characters c ON c.character_id = m.character_id
            WHERE m.doc_id = ?
            ORDER BY c.name
            """,
            (doc_id,),
        ).fetchall()
        return [row["name"] for row in rows]
    except sqlite3.OperationalError:
        return []


def _resolve(name: str, roster: dict[str, list[str]]) -> str | None:
    """Resolve a surface name to an existing character_id, or None."""
    key = character_key(name)
    if key in roster:
        return key
    return resolve_character_key(name, roster)


def _load_political_positions(
    conn: sqlite3.Connection,
) -> dict[str, tuple[float, float]]:
    """{character_id: (economic, social)} for characters placed on the compass."""
    try:
        rows = conn.execute(
            "SELECT character_id, pol_economic, pol_social FROM characters "
            "WHERE pol_economic IS NOT NULL AND pol_social IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {r["character_id"]: (r["pol_economic"], r["pol_social"]) for r in rows}


def _upsert_relation(
    conn: sqlite3.Connection,
    roster: dict[str, list[str]],
    source_name: str,
    target_name: str,
    kind: str,
    description: str,
    doc_id: str,
    positions: dict[str, tuple[float, float]] | None = None,
) -> str:
    """Persist one relation, accumulating evidence. Returns a status string.

    "new" / "updated" — stored; "unresolved" — an endpoint is not a known
    character; "self" — both endpoints are the same person.

    When ``positions`` (character_id → compass position) is given, a claimed
    ``afiliacion`` between two ideologically distant figures is corrected to
    ``contraparte`` (see ``lib.politics.refine_relation_kind``).
    """
    source_key = _resolve(source_name, roster)
    target_key = _resolve(target_name, roster)
    if source_key is None or target_key is None:
        return "unresolved"

    normalized = normalize_relation(source_key, target_key, kind)
    if normalized is None:
        return "self"
    src, tgt, canonical = normalized

    if positions:
        pos_a, pos_b = positions.get(src), positions.get(tgt)
        if pos_a and pos_b:
            canonical = refine_relation_kind(
                canonical, pos_a[0], pos_a[1], pos_b[0], pos_b[1]
            )

    now = datetime.now(UTC).isoformat()

    existing = conn.execute(
        "SELECT id, description, mention_count, provenance FROM character_relations "
        "WHERE source_character_id=? AND target_character_id=? AND kind=?",
        (src, tgt, canonical),
    ).fetchone()

    if existing is None:
        conn.execute(
            """
            INSERT INTO character_relations
                (source_character_id, target_character_id, kind, description,
                 confidence, mention_count, provenance, first_seen_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (src, tgt, canonical, description or None, _confidence(1),
             json.dumps([doc_id]), now, now),
        )
        return "new"

    provenance = list(dict.fromkeys(json.loads(existing["provenance"] or "[]")))
    if doc_id not in provenance:
        provenance.append(doc_id)
    mention_count = len(provenance)
    # Keep the richest description seen.
    new_description = existing["description"] or ""
    if description and len(description) > len(new_description):
        new_description = description
    conn.execute(
        "UPDATE character_relations SET description=?, confidence=?, mention_count=?, "
        "provenance=?, updated_at=? WHERE id=?",
        (new_description or None, _confidence(mention_count), mention_count,
         json.dumps(provenance), now, existing["id"]),
    )
    return "updated"


def _stamp_relations_extracted(conn: sqlite3.Connection, doc_id: str) -> None:
    conn.execute(
        "UPDATE documents SET relations_extracted_at=? WHERE doc_id=?",
        (datetime.now(UTC).isoformat(), doc_id),
    )


# ── LLM extraction ────────────────────────────────────────────────────────────

_LLM_SYSTEM = (
    "You are a prosopographer for a Digital Humanities project at Cambridge University, "
    "mapping how people in Allende-era Chile (1969-1973) relate to one another. "
    "You respond ONLY with a JSON object. No explanation, no markdown."
)

_LLM_PROMPT = """\
From this document, extract RELATIONS between identifiable people. For each
relation return:
  - "source": the first person's full name
  - "target": the second person's full name
  - "kind": one of family, ally, colleague, rival, enemy, political, counterpart,
    superior, subordinate, mentor, student   (use the relation the text supports)
  - "description": one concise clause justifying it (e.g. "served in Allende's cabinet")

RULES:
- Only relations the document actually states or clearly implies. Do not invent.
- Both endpoints must be real, named people (skip organisations and places).
- Prefer the people already known in this document (listed below) when they match.
- Use "political" only for people who SHARE an alignment; two ideological
  opponents facing each other are "counterpart" (or "enemy" if openly hostile),
  never "political".
- "superior"/"subordinate" and "mentor"/"student" are directed: source is the
  superior / mentor.

KNOWN PEOPLE IN THIS DOCUMENT: {known}

Respond with EXACTLY this shape:
{{"relations": [{{"source": "...", "target": "...", "kind": "ally", "description": "..."}}]}}

TITLE: {title}
TEXT (first 1800 chars):
{text}"""


async def _llm_extract_relations(
    text: str, title: str, known: list[str], client: LLMClient
) -> list[ExtractedRelation] | None:
    """Call Gemma to extract relations from a document. None on error/parse failure."""
    prompt = _LLM_PROMPT.format(
        known=", ".join(known) if known else "(none recorded yet)",
        title=title[:200] or "(no title)",
        text=text[:1800],
    )
    try:
        raw = await client.chat(
            model=settings.ollama_model_npc,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=_LLM_TEMPERATURE,
            num_predict=_LLM_NUM_PREDICT,
            think=False,
        )
        log.debug("cast_director.llm_raw_response", raw=raw.strip()[:120])
        relations = parse_relations_response(raw)
        if relations is None:
            log.warning("cast_director.llm_unparseable", raw=raw.strip()[:150], title=title[:60])
        return relations
    except LLMError as exc:
        log.warning("cast_director.llm_error", reason=str(exc)[:120])
        return None
    except Exception as exc:
        log.warning("cast_director.llm_unexpected_error", reason=str(exc)[:120])
        return None


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class CastDirectorResult:
    """Result of one Cast Director run cycle."""

    processed: int = 0            # documents mined
    relations_new: int = 0        # new relation rows
    relations_updated: int = 0    # existing relations corroborated
    relations_unresolved: int = 0  # endpoints not (yet) in the cast


# ── Cast Director agent ─────────────────────────────────────────────────────────


class CastDirector:
    """Ag-8: Extracts the character relational layer from verified, mapped documents.

    Reads documents with ``verified_at`` and ``mapped_category_id`` set but no
    ``relations_extracted_at``, extracts the relations stated in each (via Gemma),
    and upserts them into ``character_relations`` with provenance and a
    corroboration-based confidence. ``--no-llm`` mode stamps documents without
    extracting (relations are inherently inferential).
    """

    def __init__(self, db_path: Path | None = None, use_llm: bool = True) -> None:
        self.db_path = db_path or settings.archive_db
        self.use_llm = use_llm

    async def run_cycle(self, batch_size: int = 50, all_docs: bool = False) -> CastDirectorResult:
        """Mine verified+mapped documents for character relations.

        Default: one batch. With ``all_docs=True`` it loops batch-by-batch until
        none remain — self-terminating, since each processed document is stamped
        ``relations_extracted_at``.
        """
        if not self.db_path.exists():
            log.info("cast_director.db_not_found", path=str(self.db_path))
            return CastDirectorResult()

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        result = CastDirectorResult()

        try:
            ensure_relation_tables(conn)

            client_ctx: LLMClient | None = None
            if self.use_llm:
                client_ctx = await self._open_ollama()
                if client_ctx is None:
                    return result

            try:
                batch_num = 0
                while True:
                    batch_num += 1
                    rows = _fetch_unprocessed_documents(conn, batch_size)
                    if not rows:
                        if batch_num == 1:
                            log.info("cast_director.nothing_to_extract")
                        else:
                            log.info("cast_director.session_done", batches=batch_num - 1)
                        break

                    log.info(
                        "cast_director.cycle_start",
                        batch=len(rows),
                        batch_num=batch_num,
                        use_llm=self.use_llm,
                    )
                    await self._process_batch(rows, conn, client_ctx, result)
                    conn.commit()

                    if not all_docs:
                        break
            finally:
                if client_ctx is not None:
                    await client_ctx.__aexit__(None, None, None)
        finally:
            conn.close()

        log.info(
            "cast_director.cycle_done",
            processed=result.processed,
            relations_new=result.relations_new,
            relations_updated=result.relations_updated,
            relations_unresolved=result.relations_unresolved,
        )
        return result

    async def _open_ollama(self) -> LLMClient | None:
        """Open an Ollama client and verify the model is present, else None with guidance."""
        client = LLMClient()
        await client.__aenter__()
        try:
            models = await client.list_models()
            if not any(settings.ollama_model_npc in m.get("name", "") for m in models):
                log.error("cast_director.model_not_found", model=settings.ollama_model_npc)
                print(
                    f"\n  ERROR: Model {settings.ollama_model_npc} not found in Ollama.\n"
                    "  Start Ollama with: ollama serve\n"
                    "  Or use --no-llm (stamps documents without extracting).\n"
                )
                await client.__aexit__(None, None, None)
                return None
        except Exception as exc:
            log.error("cast_director.ollama_unreachable", reason=str(exc)[:120])
            print(
                "\n  ERROR: Ollama is not running.\n"
                "  Start it with: ollama serve\n"
                "  Or use --no-llm (stamps documents without extracting).\n"
            )
            await client.__aexit__(None, None, None)
            return None
        return client

    async def _process_batch(
        self,
        rows: list[sqlite3.Row],
        conn: sqlite3.Connection,
        client: LLMClient | None,
        result: CastDirectorResult,
    ) -> None:
        """Extract and persist relations for a batch of documents."""
        roster = _load_character_roster(conn)
        positions = _load_political_positions(conn)
        for row in rows:
            doc_id = row["doc_id"]
            if self.use_llm and client is not None:
                known = _doc_known_characters(conn, doc_id)
                relations = await _llm_extract_relations(
                    row["text"] or "", row["title"] or "", known, client
                )
                for relation in relations or []:
                    status = _upsert_relation(
                        conn, roster, relation.source, relation.target,
                        relation.kind, relation.description, doc_id,
                        positions=positions,
                    )
                    if status == "new":
                        result.relations_new += 1
                    elif status == "updated":
                        result.relations_updated += 1
                    elif status == "unresolved":
                        result.relations_unresolved += 1
            _stamp_relations_extracted(conn, doc_id)
            result.processed += 1

    def status(self) -> dict[str, int]:
        """Return relation counts: total relations, by kind, pending documents."""
        if not self.db_path.exists():
            return {}
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            ensure_relation_tables(conn)

            def _count(sql: str) -> int:
                try:
                    return conn.execute(sql).fetchone()[0]
                except sqlite3.OperationalError:
                    return 0

            return {
                "relations": _count("SELECT COUNT(*) FROM character_relations"),
                "high_confidence": _count(
                    "SELECT COUNT(*) FROM character_relations WHERE confidence >= 0.66"
                ),
                "pending_docs": _count(
                    "SELECT COUNT(*) FROM documents "
                    "WHERE verified_at IS NOT NULL AND mapped_category_id IS NOT NULL "
                    "AND relations_extracted_at IS NULL"
                ),
            }
        finally:
            conn.close()


# ── CLI ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    import traceback

    from lib.logging_setup import configure_logging, current_log_file

    configure_logging()

    ap = argparse.ArgumentParser(
        prog="cast-director",
        description="Ag-8 Cast Director — extract character relations from documents",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Extract relations from verified+mapped documents")
    run_p.add_argument("--batch", type=int, default=50, help="Documents per batch (default: 50)")
    run_p.add_argument("--all", action="store_true",
                       help="Process all eligible docs in a self-terminating loop")
    run_p.add_argument("--no-llm", action="store_true",
                       help="Stamp documents without extracting (no Ollama)")

    sub.add_parser("status", help="Print relation counts")

    args = ap.parse_args()

    if args.cmd == "run":
        use_llm = not args.no_llm
        log.info("cast_director.main_start", batch=args.batch, use_llm=use_llm, all_docs=args.all)
        try:
            director = CastDirector(use_llm=use_llm)
            t0 = datetime.now(UTC)
            result = asyncio.run(director.run_cycle(batch_size=args.batch, all_docs=args.all))
            rid = current_run_id()
            if rid:
                record_stage(rid, "cast_director", t0, datetime.now(UTC), {
                    "processed": result.processed,
                    "relations_new": result.relations_new,
                    "relations_updated": result.relations_updated,
                    "relations_unresolved": result.relations_unresolved,
                })
            log.info("cast_director.main_done", processed=result.processed,
                     relations_new=result.relations_new)
            print(
                f"  Processed: {result.processed} docs\n"
                f"  Relations: +{result.relations_new} new, "
                f"{result.relations_updated} corroborated\n"
                f"  Unresolved endpoints: {result.relations_unresolved}"
            )
        except Exception as exc:
            log.error("cast_director.main_error", error=str(exc)[:200])
            print(f"\n  ERROR: {exc}\n")
            traceback.print_exc()
        finally:
            log_file = current_log_file()
            if log_file:
                print(f"\n  Log: {log_file}")

    elif args.cmd == "status":
        director = CastDirector()
        status = director.status()
        if not status:
            print("Sin base de datos o sin relaciones.")
        else:
            print(f"  Relaciones:        {status['relations']}")
            print(f"  Alta confianza:    {status['high_confidence']}")
            print(f"  Docs pendientes:   {status['pending_docs']}")


if __name__ == "__main__":
    main()
