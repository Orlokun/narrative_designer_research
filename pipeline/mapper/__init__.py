"""
Ag-3 Mapper — assigns verified documents to coverage matrix cells using Gemma LLM.

The Mapper reads each verified document and determines which cell (category_id,
genre_id, month_iso) in the 16×13×48 coverage matrix it belongs to. This is more
precise than the coarser mission-level counting done by Archivero. Genre (letter,
speech, decree, poem…) is classified by the LLM, falling back to a deterministic
source/title heuristic (``genre_from_source``).

Two modes:
  --no-llm  : Assigns category/month from the document's parent mission and genre
              from the heuristic (useful for fast testing or when Ollama is down).
  default   : Calls Gemma4 to classify each document. Requires Ollama running.

CLI:
    uv run mapper run [--batch 50] [--no-llm]
    uv run mapper status
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from lib.config import settings
from lib.coverage import ensure_coverage_table, upsert_coverage_score
from lib.genres import GENRES, VALID_GENRE_IDS
from lib.llm import LLMClient, LLMError
from lib.logging_setup import get_logger
from lib.project import project
from lib.run_tracker import current_run_id, record_stage

log = get_logger("mapper")

# Re-exported for callers/tests that used the Mapper's own coverage upsert.
_upsert_coverage_score = upsert_coverage_score

# ── Constants ──────────────────────────────────────────────────────────────────

_LLM_TEMPERATURE: float = 0.3
_LLM_NUM_PREDICT: int = 128  # JSON response with up to 3 category_ids + genre_id

_PROJECT = project()

# The matrix's time axis (all ISO months of the project period).
_VALID_MONTHS: set[str] = set(_PROJECT.months())
_FALLBACK_MONTH: str = _PROJECT.period.middle_month()

_CATEGORY_NAMES: dict[int, str] = _PROJECT.theme_names()

# ── LLM prompt ────────────────────────────────────────────────────────────────

_LLM_SYSTEM = (
    _PROJECT.classification.persona
    + " You respond ONLY with a JSON object containing category_ids (array, 1-3 items), genre_id, month_iso, and confidence. No explanation."
)

_GENRE_PROMPT_BLOCK = "\n".join(
    f"{genre.id}. {genre.name} — {genre.description}" for genre in GENRES
)

_LLM_PROMPT_TEMPLATE = """\
Classify this document about the history of {region}. Read the FULL text carefully before
deciding. Then assign EXACTLY THREE categories ranked by relevance (primary first)
and the document's WRITING DATE.

READING RULE (most common error to avoid):
{reading_rule}

CATEGORIES (ID 1-{n_categories}):
{categories}

CATEGORY RULES:
- Always return 3 ids. Rank by relevance: primary = MAIN subject, secondary and
  tertiary = clearly co-relevant aspects.
- The 3 ids must be distinct.
- The primary should reflect what the document IS, not a setting it MENTIONS.
{category_examples}

GENRES (ID 1-13) — what the document IS as a textual form:
{genres}

GENRE RULES:
- Pick exactly ONE genre_id: the document's FORM, not its topic. A news article
  about a speech is Prensa informativa (6), not Discurso (1). A printed speech
  transcript is Discurso (1).
- Scholarly analyses (with citations, written after the fact) = Artículo académico (9).
- If the form is genuinely unidentifiable, omit genre_id.

PERIOD: {period_label} ONLY.
If the document is primarily about {region} after {end_month} or before {start_month}, or about another
country/period, set confidence to 0.0 (it will be tagged as out-of-scope context).

DATE EXTRACTION (very important — read carefully):
- Use the document's WRITING/AUTHORSHIP date, NOT the date of events it describes or plans.
- Look in this order:
    1. Explicit date in the document header (e.g. "March 22-23 talks", memo date, signed date)
    2. Source citation / footnotes (e.g. "Drafted by ... cleared by ...", archival metadata)
    3. References to "next month", "last week", "the recent X" — anchored to known events
- A memo discussing planned action "in early June" but written in late March = month 03,
  not 06. The action month is metadata; the writing month is what we want.
- If a span is given (March 22-23), pick the END month of authorship.
- Use ISO format YYYY-MM, range {start_month} through {end_month}.
- If truly undatable, use {fallback_month} as fallback.

DOCUMENT:
Title: {title}
Text (first 1500 chars):
{text}

Respond with ONLY a JSON object (no markdown, no prose):
{{"category_ids": [<id1>, <id2?>, <id3?>], "genre_id": <1-13>, "month_iso": "YYYY-MM", "confidence": <0.0-1.0>}}

Examples:
{json_examples}"""


def _render_prompt_template() -> str:
    """Fill the project-specific parts of the prompt, leaving {title}/{text}/{genres} for later."""
    guidance = _PROJECT.classification
    examples = "\n".join(f"- {line}" for line in guidance.category_examples)
    json_examples = "\n".join(line.replace("{", "{{").replace("}", "}}") for line in guidance.json_examples)
    return (
        _LLM_PROMPT_TEMPLATE.replace("{region}", _PROJECT.region)
        .replace("{reading_rule}", guidance.reading_rule)
        .replace("{n_categories}", str(len(_PROJECT.themes)))
        .replace("{categories}", _PROJECT.category_prompt_block())
        .replace("{category_examples}", examples)
        .replace("{period_label}", _PROJECT.period_label())
        .replace("{start_month}", _PROJECT.period.start_month)
        .replace("{end_month}", _PROJECT.period.end_month)
        .replace("{fallback_month}", _FALLBACK_MONTH)
        .replace("{json_examples}", json_examples)
    )


_LLM_PROMPT = _render_prompt_template()

# ── Pure functions ────────────────────────────────────────────────────────────

def parse_llm_response(response: str) -> tuple[list[int], str, float] | None:
    """Parse LLM JSON response to extract (category_ids, month_iso, confidence).

    Accepts both the new multi-category shape ``"category_ids": [<int>, ...]`` and
    the legacy single-category shape ``"category_id": <int>``. Returned list has
    1–3 unique items, each in 1–16, ordered as the LLM ranked them (primary first).

    Returns None if the response is malformed, contains invalid category_ids /
    month_iso, or confidence is outside [0.0, 1.0].
    """
    try:
        data = json.loads(response.strip())

        # New shape: category_ids array; fall back to legacy single category_id
        raw_ids = data.get("category_ids")
        if raw_ids is None:
            legacy = data.get("category_id")
            raw_ids = [legacy] if legacy is not None else None

        month_iso = data.get("month_iso")
        confidence = data.get("confidence")

        if not isinstance(raw_ids, list) or not raw_ids:
            return None
        if not isinstance(month_iso, str) or not isinstance(confidence, (int, float)):
            return None

        # Validate each id and dedupe while preserving order
        seen: set[int] = set()
        category_ids: list[int] = []
        for cid in raw_ids:
            if not isinstance(cid, int) or cid < 1 or cid > 16:
                return None
            if cid in seen:
                continue
            seen.add(cid)
            category_ids.append(cid)

        if not 1 <= len(category_ids) <= 3:
            return None

        if month_iso not in _VALID_MONTHS:
            return None

        if confidence < 0.0 or confidence > 1.0:
            return None

        return (category_ids, month_iso, float(confidence))

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _month_out_of_range(response: str) -> bool:
    """Return True if JSON is valid but month_iso falls outside Oct 1969–Sep 1973.

    This distinguishes "LLM assigned a valid-format month outside the project window"
    from other failures (bad JSON, wrong types, missing fields, etc.).
    """
    try:
        data = json.loads(response.strip())
        month_iso = data.get("month_iso", "")
        return bool(re.match(r"^\d{4}-\d{2}$", month_iso)) and month_iso not in _VALID_MONTHS
    except (json.JSONDecodeError, TypeError):
        return False


def _is_irrelevant_signal(response: str) -> bool:
    """Return True if LLM explicitly signaled 'irrelevant' via confidence=0.0.

    The system prompt instructs the LLM to assign confidence=0.0 for documents
    outside the period or about other countries. We treat this as a context tag
    rather than a parse failure.
    """
    try:
        data = json.loads(response.strip())
        confidence = data.get("confidence")
        return isinstance(confidence, (int, float)) and confidence <= 0.0
    except (json.JSONDecodeError, TypeError):
        return False


def parse_genre_response(response: str) -> int | None:
    """Extract a valid genre_id (1-13) from the LLM's JSON response.

    Returns None when the field is missing, out of range or the wrong type —
    callers then fall back to ``genre_from_source``.
    """
    try:
        data = json.loads(response.strip())
        genre_id = data.get("genre_id")
        if isinstance(genre_id, int) and genre_id in VALID_GENRE_IDS:
            return genre_id
        return None
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


# Title keywords checked first (a speech edition on a scholarly connector is still
# a speech); FRUS state-department memoranda match before generic correspondence.
_TITLE_GENRE_PATTERNS: list[tuple[int, tuple[str, ...]]] = [
    (3,  ("telegram", "cable", "airgram", "memorandum from", "memorandum prepared",
          "memorandum of conversation", "intelligence")),
    (1,  ("discurso", "speech", "alocución", "address by")),
    (2,  ("carta ", "telex", "correspondencia", "memorándum")),
    (4,  ("decreto", "ley ", "acta ", "resolución", "decree")),
    (5,  ("sesión", "transcripción", "transcript", "debate parlamentario")),
    (7,  ("editorial", "columna", "opinión")),
    (8,  ("informe", "report", "memoria anual")),
    (10, ("testimonio", "entrevista", "interview", "memorias de", "memoir")),
    (11, ("poema", "canción", "poem", "cancionero", "song")),
    (12, ("novela", "cuento", "teatro", "novel")),
    (13, ("manifiesto", "panfleto", "programa de gobierno", "manifesto", "pamphlet")),
]

# Connector → most-likely genre, used when the title gives no signal.
_SOURCE_GENRE_PRIORS: dict[str, int] = {
    "openalex":            9,   # Artículo académico
    "crossref":            9,
    "semantic_scholar":    9,
    "wikipedia_es":        9,   # encyclopedic reference — closest register
    "frus":                3,   # Cable diplomático
    "foia_chile":          3,   # Chile Declassification Project — cables/memos/intel
    "chronicling_america": 6,   # Prensa informativa
    "marxists":            1,   # mostly speeches/writings by political figures
}


def genre_from_source(source_id: str | None, title: str) -> int | None:
    """Deterministic genre guess from title keywords, then connector prior.

    Used by the --no-llm path and as a fallback when the LLM omits genre_id.
    Returns None when neither signal applies (stored as NULL → unclassified).
    """
    title_lower = (title or "").lower()
    if title_lower:
        for genre_id, needles in _TITLE_GENRE_PATTERNS:
            if any(needle in title_lower for needle in needles):
                return genre_id
    return _SOURCE_GENRE_PRIORS.get(source_id or "")


# ── SQLite helpers ────────────────────────────────────────────────────────────

def _migrate_documents_schema(conn: sqlite3.Connection) -> None:
    """Add mapper columns (mapped_category_id, mapped_month_iso, mapped_at, doc_tag,
    mapped_categories, mapped_genre_id) if not present."""
    for column_def in (
        "mapped_category_id INTEGER",
        "mapped_month_iso TEXT",
        "mapped_at TEXT",
        "doc_tag TEXT",
        "mapped_categories TEXT",
        "mapped_genre_id INTEGER",
    ):
        try:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {column_def}")
        except sqlite3.OperationalError:
            pass  # Column already present
    conn.commit()


def _fetch_unmapped_documents(conn: sqlite3.Connection, batch_size: int) -> list[sqlite3.Row]:
    """Fetch verified documents that have not yet been mapped."""
    return conn.execute(
        """
        SELECT doc_id, title, text, source_id
        FROM documents
        WHERE verified_at IS NOT NULL AND mapped_category_id IS NULL
        LIMIT ?
        """,
        (batch_size,),
    ).fetchall()


def _fetch_all_verified_documents(
    conn: sqlite3.Connection,
    batch_size: int,
    before_ts: str | None = None,
) -> list[sqlite3.Row]:
    """Fetch verified documents (including already-mapped) for re-mapping.

    If before_ts is provided, only fetch docs whose mapped_at is NULL or older
    than before_ts. This makes batched processing self-terminating: once every
    doc has been touched in the current session, the result set becomes empty.
    """
    if before_ts is None:
        return conn.execute(
            """
            SELECT doc_id, title, text, source_id
            FROM documents
            WHERE verified_at IS NOT NULL
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
    return conn.execute(
        """
        SELECT doc_id, title, text, source_id
        FROM documents
        WHERE verified_at IS NOT NULL
          AND (mapped_at IS NULL OR mapped_at < ?)
        LIMIT ?
        """,
        (before_ts, batch_size),
    ).fetchall()


def _fetch_mapped_documents(
    conn: sqlite3.Connection,
    batch_size: int,
    before_ts: str | None = None,
) -> list[sqlite3.Row]:
    """Fetch already-mapped documents for re-classification (--remap mode).

    If before_ts is provided, only fetch docs whose mapped_at is older than
    before_ts, making batched processing self-terminating.
    """
    if before_ts is None:
        return conn.execute(
            """
            SELECT doc_id, title, text, source_id
            FROM documents
            WHERE mapped_category_id IS NOT NULL
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
    return conn.execute(
        """
        SELECT doc_id, title, text, source_id
        FROM documents
        WHERE mapped_category_id IS NOT NULL
          AND mapped_at < ?
        LIMIT ?
        """,
        (before_ts, batch_size),
    ).fetchall()


def _save_mapping(
    conn: sqlite3.Connection,
    doc_id: str,
    category_ids: list[int],
    month_iso: str,
    genre_id: int | None = None,
) -> None:
    """Save mapping for a document.

    Writes the primary category to ``mapped_category_id`` (used by the coverage
    matrix and admin filters), the full ranked list as a JSON array to
    ``mapped_categories``, and the genre to ``mapped_genre_id`` (NULL =
    unclassified). ``category_ids[0]`` is treated as primary.
    """
    if not category_ids:
        raise ValueError("category_ids must contain at least one id")
    mapped_at = datetime.now(UTC).isoformat()
    conn.execute(
        """
        UPDATE documents
        SET mapped_category_id=?, mapped_categories=?, mapped_month_iso=?,
            mapped_genre_id=?, mapped_at=?
        WHERE doc_id=?
        """,
        (category_ids[0], json.dumps(category_ids), month_iso, genre_id, mapped_at, doc_id),
    )


def _tag_as_context(conn: sqlite3.Connection, doc_id: str) -> None:
    """Mark document as out-of-scope context (outside Oct 1969–Sep 1973).

    Sets doc_tag='context' and stamps mapped_at so the document is not retried.
    """
    mapped_at = datetime.now(UTC).isoformat()
    conn.execute(
        """
        UPDATE documents
        SET doc_tag=?, mapped_at=?
        WHERE doc_id=?
        """,
        ("context", mapped_at, doc_id),
    )


def _stamp_skipped(conn: sqlite3.Connection, doc_id: str) -> None:
    """Stamp mapped_at on a skipped document without changing its mapping.

    In --all/--remap looping modes, the session-timestamp filter excludes docs
    whose mapped_at is in the future relative to session_start. Without this
    stamp, a doc that the LLM consistently fails on would be re-fetched on
    every batch, causing an infinite loop.
    """
    mapped_at = datetime.now(UTC).isoformat()
    conn.execute(
        "UPDATE documents SET mapped_at=? WHERE doc_id=?",
        (mapped_at, doc_id),
    )


def _get_parent_mission_cell(conn: sqlite3.Connection, doc_id: str) -> tuple[int, str] | None:
    """For --no-llm fallback: retrieve (category_id, month_iso) from parent mission.

    Documents have provenance.mission_id which links to the missions table.
    """
    # Extract mission_id from provenance JSON
    row = conn.execute(
        """
        SELECT json_extract(provenance, '$.mission_id') as mission_id
        FROM documents
        WHERE doc_id = ?
        """,
        (doc_id,),
    ).fetchone()

    if not row or not row["mission_id"]:
        return None

    mission_id = row["mission_id"]

    # Look up the mission to get category_id and month_iso
    mission = conn.execute(
        "SELECT category_id, month_iso FROM missions WHERE mission_id = ?",
        (mission_id,),
    ).fetchone()

    if mission:
        return (mission["category_id"], mission["month_iso"])

    return None


# ── LLM scoring ────────────────────────────────────────────────────────────

async def _llm_classify(
    text: str,
    title: str,
    client: LLMClient,
) -> tuple[list[int], str, int | None, float] | None:
    """Call Gemma4 to classify a document into (category_ids, month_iso, genre_id, confidence).

    genre_id is None when the LLM omits it or returns an invalid id — callers fall
    back to ``genre_from_source``. Returns ([-1], "context", None, 0.0) if the JSON
    is valid but the month is out of range or the LLM signals irrelevance via
    confidence=0.0. Returns None if the call fails or the response cannot be
    parsed otherwise.
    """
    prompt = _LLM_PROMPT.format(
        genres=_GENRE_PROMPT_BLOCK,
        title=title[:200] or "(no title)",
        text=text[:1500],
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
        log.debug("mapper.llm_raw_response", raw=raw.strip()[:100])

        result = parse_llm_response(raw)
        if result is None:
            if _month_out_of_range(raw) or _is_irrelevant_signal(raw):
                return ([-1], "context", None, 0.0)
            log.warning(
                "mapper.llm_unparseable",
                raw=raw.strip()[:150],
                title=title[:60],
            )
            return None
        category_ids, month_iso, confidence = result
        return (category_ids, month_iso, parse_genre_response(raw), confidence)

    except LLMError as exc:
        log.warning("mapper.llm_error", reason=str(exc)[:120])
        return None
    except Exception as exc:
        log.warning("mapper.llm_unexpected_error", reason=str(exc)[:120])
        return None


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class MappingResult:
    """Result of one mapper run cycle."""
    processed: int = 0  # documents classified
    mapped: int = 0     # successfully mapped
    context: int = 0    # tagged as out-of-scope (month outside Oct 1969–Sep 1973)
    skipped: int = 0    # failed to classify (LLM error or fallback returned None)


# ── Mapper agent ──────────────────────────────────────────────────────────────

class Mapper:
    """Ag-3: Assigns verified documents to coverage matrix cells using Gemma LLM.

    Reads documents with verified_at IS NOT NULL and mapped_category_id IS NULL,
    calls the LLM to classify each into a (category_id, month_iso) cell,
    then refreshes the coverage table with document-level counts.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        use_llm: bool = True,
    ) -> None:
        self.db_path = db_path or settings.archive_db
        self.use_llm = use_llm

    async def run_cycle(
        self,
        batch_size: int = 50,
        all_verified: bool = False,
        remap_only: bool = False,
    ) -> MappingResult:
        """Classify verified documents and assign them to coverage matrix cells.

        Modes:
          default          : Process only unmapped verified documents (one batch).
          all_verified=True: Process all verified documents (mapped + unmapped),
                             looping internally in batches until done.
          remap_only=True  : Re-process only already-mapped documents, looping
                             internally in batches until done.

        In all_verified/remap_only modes, a session timestamp is captured at the
        start; each batch fetches docs whose mapped_at is older than the session
        start (or NULL). This makes the run self-terminating: once every doc has
        been touched, the result set becomes empty and the loop exits.

        If use_llm=True, calls Gemma4 for each document. If Ollama is unavailable,
        the run aborts with an error.
        If use_llm=False, assigns category/month from the parent mission (fallback).
        """
        if not self.db_path.exists():
            log.info("mapper.db_not_found", path=str(self.db_path))
            return MappingResult()

        if all_verified and remap_only:
            raise ValueError("all_verified and remap_only are mutually exclusive")

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        result = MappingResult()
        looped_mode = all_verified or remap_only
        session_start = datetime.now(UTC).isoformat() if looped_mode else None

        try:
            _migrate_documents_schema(conn)

            client_ctx: LLMClient | None = None
            if self.use_llm:
                client_ctx = LLMClient()
                await client_ctx.__aenter__()
                try:
                    models = await client_ctx.list_models()
                    if not any(settings.ollama_model_npc in m.get("name", "") for m in models):
                        log.error("mapper.model_not_found", model=settings.ollama_model_npc)
                        print(
                            f"\n  ERROR: Model {settings.ollama_model_npc} not found in Ollama.\n"
                            "  Start Ollama with: ollama serve\n"
                            "  Or use --no-llm fallback.\n"
                        )
                        await client_ctx.__aexit__(None, None, None)
                        return result
                except Exception as exc:
                    log.error("mapper.ollama_unreachable", reason=str(exc)[:120])
                    print(
                        "\n  ERROR: Ollama is not running.\n"
                        "  Start it with: ollama serve\n"
                        "  Or use --no-llm fallback.\n"
                    )
                    await client_ctx.__aexit__(None, None, None)
                    return result

            try:
                batch_num = 0
                while True:
                    batch_num += 1
                    rows = self._fetch_batch(conn, batch_size, all_verified, remap_only, session_start)

                    if not rows:
                        if batch_num == 1:
                            log.info("mapper.nothing_to_map")
                        else:
                            log.info("mapper.session_done", batches=batch_num - 1)
                        break

                    log.info(
                        "mapper.cycle_start",
                        batch=len(rows),
                        batch_num=batch_num,
                        use_llm=self.use_llm,
                        all_verified=all_verified,
                        remap_only=remap_only,
                    )

                    await self._process_batch(rows, conn, client_ctx, result)
                    conn.commit()

                    # Single-batch mode: stop after one pass
                    if not looped_mode:
                        break
            finally:
                if client_ctx is not None:
                    await client_ctx.__aexit__(None, None, None)

            # Update coverage table with document-level counts
            self._refresh_coverage(conn)
            conn.commit()

        finally:
            conn.close()

        log.info(
            "mapper.cycle_done",
            processed=result.processed,
            mapped=result.mapped,
            context=result.context,
            skipped=result.skipped,
        )
        return result

    @staticmethod
    def _fetch_batch(
        conn: sqlite3.Connection,
        batch_size: int,
        all_verified: bool,
        remap_only: bool,
        before_ts: str | None,
    ) -> list[sqlite3.Row]:
        """Pick the appropriate fetch strategy for the current run mode."""
        if remap_only:
            return _fetch_mapped_documents(conn, batch_size, before_ts=before_ts)
        if all_verified:
            return _fetch_all_verified_documents(conn, batch_size, before_ts=before_ts)
        return _fetch_unmapped_documents(conn, batch_size)

    async def _process_batch(
        self,
        rows: list[sqlite3.Row],
        conn: sqlite3.Connection,
        client: LLMClient | None,
        result: MappingResult,
    ) -> None:
        """Process a single batch of documents, updating result counters in place."""
        if self.use_llm and client is not None:
            for row in rows:
                category_ids, month_iso, genre_id, confidence = await self._classify_with_llm(
                    row, client, conn
                )
                if category_ids == [-1] and month_iso == "context":
                    _tag_as_context(conn, row["doc_id"])
                    result.context += 1
                    result.processed += 1
                    log.info(
                        "mapper.doc_tagged_context",
                        doc_id=row["doc_id"],
                        title=row["title"][:60],
                    )
                elif category_ids:
                    if genre_id is None:
                        genre_id = genre_from_source(row["source_id"], row["title"])
                    _save_mapping(conn, row["doc_id"], category_ids, month_iso, genre_id)
                    result.mapped += 1
                    result.processed += 1
                    log.info(
                        "mapper.doc_mapped",
                        doc_id=row["doc_id"],
                        category_ids=category_ids,
                        genre_id=genre_id,
                        month_iso=month_iso,
                        confidence=confidence,
                    )
                else:
                    _stamp_skipped(conn, row["doc_id"])
                    result.skipped += 1
                    log.warning(
                        "mapper.doc_skipped_llm_failed",
                        doc_id=row["doc_id"],
                        title=row["title"][:60],
                    )
        else:
            # --no-llm fallback: cell from parent mission, genre from heuristic
            for row in rows:
                cell = _get_parent_mission_cell(conn, row["doc_id"])
                if cell:
                    category_id, month_iso = cell
                    genre_id = genre_from_source(row["source_id"], row["title"])
                    _save_mapping(conn, row["doc_id"], [category_id], month_iso, genre_id)
                    result.mapped += 1
                    result.processed += 1
                    log.info(
                        "mapper.doc_mapped_fallback",
                        doc_id=row["doc_id"],
                        category_id=category_id,
                        genre_id=genre_id,
                        month_iso=month_iso,
                    )
                else:
                    _stamp_skipped(conn, row["doc_id"])
                    result.skipped += 1
                    log.warning(
                        "mapper.doc_skipped_no_parent_mission",
                        doc_id=row["doc_id"],
                    )

    async def _classify_with_llm(
        self,
        row: sqlite3.Row,
        client: LLMClient,
        conn: sqlite3.Connection,
    ) -> tuple[list[int], str, int | None, float]:
        """Classify a single document using LLM.

        Returns (category_ids, month_iso, genre_id, confidence) on success,
        ([-1], "context", None, 0.0) for out-of-period docs, or ([], "", None, 0.0)
        on failure.
        """
        result = await _llm_classify(row["text"], row["title"], client)
        if result:
            return result
        return [], "", None, 0.0

    def _refresh_coverage(self, conn: sqlite3.Connection) -> None:
        """After mapping a batch, recompute coverage from per-(theme, genre, month) counts.

        Documents without a genre (mapped before the genre axis, or unclassifiable)
        are aggregated under genre 0.
        """
        ensure_coverage_table(conn)
        cells = conn.execute(
            """
            SELECT mapped_category_id,
                   COALESCE(mapped_genre_id, 0) AS genre_id,
                   mapped_month_iso,
                   COUNT(*) as n
            FROM documents
            WHERE verified_at IS NOT NULL AND mapped_category_id IS NOT NULL
            GROUP BY mapped_category_id, genre_id, mapped_month_iso
            """
        ).fetchall()

        for row in cells:
            upsert_coverage_score(
                conn,
                category_id=row["mapped_category_id"],
                genre_id=row["genre_id"],
                month_iso=row["mapped_month_iso"],
                docs_found=row["n"],
            )
        conn.commit()

    def status(self) -> dict[str, int]:
        """Return counts of documents by mapping state.

        Keys:
          "mapped"   — mapped_category_id IS NOT NULL
          "context"  — doc_tag='context'
          "pending"  — verified_at IS NOT NULL AND mapped_category_id IS NULL
        """
        if not self.db_path.exists():
            return {}
        conn = sqlite3.connect(str(self.db_path))
        try:
            _migrate_documents_schema(conn)
            mapped = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE mapped_category_id IS NOT NULL"
            ).fetchone()[0]
            context = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE doc_tag='context'"
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE verified_at IS NOT NULL AND mapped_category_id IS NULL"
            ).fetchone()[0]
            return {"mapped": mapped, "context": context, "pending": pending}
        except Exception:
            return {}
        finally:
            conn.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import traceback

    from lib.logging_setup import configure_logging, current_log_file

    configure_logging()

    ap = argparse.ArgumentParser(
        prog="mapper",
        description="Ag-3 Mapper — assign verified documents to coverage matrix cells",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Classify verified documents")
    run_p.add_argument(
        "--batch", type=int, default=50,
        help="Documents per batch (default: 50). With --all/--remap, the run loops "
             "internally in batches until all matching docs are processed.",
    )
    run_p.add_argument(
        "--all", action="store_true",
        help="Process ALL verified documents in a self-terminating loop "
             "(includes already-mapped). Mutually exclusive with --remap.",
    )
    run_p.add_argument(
        "--remap", action="store_true",
        help="Re-classify ALL already-mapped documents in a self-terminating loop. "
             "Mutually exclusive with --all.",
    )
    run_p.add_argument(
        "--no-llm", action="store_true",
        help="Skip Gemma4 — use parent mission cell as fallback",
    )

    sub.add_parser("status", help="Print mapping status (mapped vs pending)")

    args = ap.parse_args()

    if args.cmd == "run":
        if args.all and args.remap:
            print("\n  ERROR: --all and --remap are mutually exclusive.\n")
            return
        use_llm = not args.no_llm
        all_verified = args.all
        remap_only = args.remap
        log.info(
            "mapper.main_start",
            batch=args.batch,
            use_llm=use_llm,
            all_verified=all_verified,
            remap_only=remap_only,
        )
        try:
            mapper = Mapper(use_llm=use_llm)
            t0 = datetime.now(UTC)
            result = asyncio.run(
                mapper.run_cycle(
                    batch_size=args.batch,
                    all_verified=all_verified,
                    remap_only=remap_only,
                )
            )
            rid = current_run_id()
            if rid:
                record_stage(rid, "mapper", t0, datetime.now(UTC), {
                    "processed": result.processed,
                    "mapped": result.mapped,
                    "context": result.context,
                    "skipped": result.skipped,
                })
            log.info(
                "mapper.main_done",
                processed=result.processed,
                mapped=result.mapped,
                context=result.context,
                skipped=result.skipped,
            )
        except Exception as exc:
            log.error("mapper.main_error", error=str(exc)[:200])
            print(f"\n  ERROR: {exc}\n")
            traceback.print_exc()

    elif args.cmd == "status":
        try:
            mapper = Mapper()
            status = mapper.status()
            if status:
                print(f"  Mapped:  {status.get('mapped', 0)}")
                print(f"  Context: {status.get('context', 0)}")
                print(f"  Pending: {status.get('pending', 0)}")
            else:
                print("Database not found or empty.")
        except Exception as exc:
            log.error("mapper.status_error", error=str(exc)[:200])
            print(f"\n  ERROR: {exc}\n")


if __name__ == "__main__":
    main()
