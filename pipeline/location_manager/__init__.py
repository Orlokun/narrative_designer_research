"""Ag-4 Location Manager — extracts places and locations from verified documents.

Reads verified+mapped documents and documents the places where events happen —
a flour factory in Temuco, a minister's house, La Moneda — keeping per-location
facts: descriptions, appreciations (who says what about the place), events that
happened there and hard data (capacity, address, coordinates). Locations feed
back into the Propositor as location research missions (architectural docs,
plans, GPS data, trivia), mirroring the Cast Manager's character loop.

Inputs:  documents table (verified_at + mapped_category_id set)
Outputs: locations / location_mentions / location_facts tables
Stack:   Gemma (chunked extraction), Wikidata + Wikipedia via the Gatekeeper
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import urllib.parse
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from lib.config import settings
from lib.llm import LLMClient, LLMError
from lib.logging_setup import get_logger
from lib.project import project
from lib.run_tracker import current_run_id, record_stage

# The Location Manager deliberately mirrors the Cast Manager: same slugging,
# same coreference-by-subsequence, same chunking, same Wikidata/Wikipedia
# plumbing — one vocabulary of mechanisms across both entity agents.
from pipeline.cast_manager import (
    _LLM_TEMPERATURE,
    _build_gatekeeper_fetch,
    _name_tokens,
    _parse_wikidata_time,
    _tokens_subsume,
    character_key,
    chunk_text,
    parse_wikidata_labels,
    parse_wikipedia_extract,
    resolve_wikidata,
    should_research,
    wikidata_labels_url,
    wikipedia_extract_url,
)
from pipeline.cast_manager import wikidata_claims_url as _wikidata_claims_url

log = get_logger("location_manager")

_MAX_LOCATIONS_PER_DOC: int = 15
_MAX_FACTS_PER_LOCATION: int = 8

_LOCATION_KINDS: frozenset[str] = frozenset(
    {"building", "factory", "residence", "city", "region", "street", "office", "other"}
)
_LOCATION_FACT_KINDS: frozenset[str] = frozenset({"description", "appreciation", "event", "data"})

# Completeness weighting — leaner than the character metric but the same shape:
# weighted components + a hard external-analysis gate.
_LOC_MENTIONS_TARGET: int = 4
_LOC_FACTS_TARGET: int = 6
_LOC_WEIGHT_DOCUMENTS: float = 0.25
_LOC_WEIGHT_FACTS: float = 0.30
_LOC_EXTERNAL_ANALYZED_CREDIT: float = 0.15  # per source (wikidata, wikipedia)
_LOC_EXTERNAL_LINKED_CREDIT: float = 0.07
_LOC_WEIGHT_COORDINATES: float = 0.15
_LOC_CAP_NO_EXTERNAL_ANALYSIS: float = 0.60
_LOC_COMPLETE_THRESHOLD: float = 0.85

# Wikidata properties mined for a location.
_WD_COORDINATES: str = "P625"
_WD_INCEPTION: str = "P571"
_WD_ARCHITECT: str = "P84"
_WD_TERRITORY: str = "P131"
_WD_IMAGE: str = "P18"  # Commons filename(s) — the location's image gallery seed

_MAX_IMAGES_PER_LOCATION: int = 6
_IMAGE_THUMB_WIDTH: int = 640
_COMMONS_FILE_PATH: str = "https://commons.wikimedia.org/wiki/Special:FilePath/"
_COMMONS_FILE_PAGE: str = "https://commons.wikimedia.org/wiki/File:"

_LOC_DESCRIPTION_MAX_CHARS: int = 1200
_LOC_ARTICLE_MAX_CHARS: int = 6000


# ── Pure functions (offline-testable) ──────────────────────────────────────────


def location_key(name: str) -> str:
    """Normalise a place name into a stable slug used as the dedup key."""
    return character_key(name)


def resolve_location_key(name: str, roster: dict[str, list[str]]) -> str | None:
    """Resolve a surface form to an existing location, or None if it is new.

    ``roster`` maps location_id → known surface forms (canonical name + aliases).
    Exact normalised match always wins. A multi-token form also matches when it
    is an ordered token subsequence of a known form ("La Moneda" ⊂ "Palacio de
    La Moneda") — but single-token names match only exactly, so "Temuco" (the
    city) never collapses into "Fábrica de harina de Temuco". Ambiguity (two
    candidates) never merges.
    """
    new_tokens = _name_tokens(name)
    if not new_tokens:
        return None
    new_slug = location_key(name)

    candidates: set[str] = set()
    for key, forms in roster.items():
        for form in forms:
            if location_key(form) == new_slug:
                return key  # exact form — unambiguous
            form_tokens = _name_tokens(form)
            if len(new_tokens) >= 2 and len(form_tokens) >= 2:
                short, long = sorted((new_tokens, form_tokens), key=len)
                if _tokens_subsume(short, long):
                    candidates.add(key)
                    break
    if len(candidates) == 1:
        return candidates.pop()
    return None  # unknown, or ambiguous — never guess


def _clean_location_fact(raw: object) -> dict | None:
    """Validate and normalise one location fact; None if unusable.

    ``appreciation`` facts are subjective by definition, so ``reported_by``
    (who voices the opinion) is kept whenever the extractor provides it.
    """
    if not isinstance(raw, dict):
        return None
    detail = raw.get("detail")
    if not isinstance(detail, str) or not detail.strip():
        return None
    kind = raw.get("kind")
    kind = kind if isinstance(kind, str) and kind in _LOCATION_FACT_KINDS else "data"
    date_iso = raw.get("date_iso")
    date_iso = date_iso if isinstance(date_iso, str) and date_iso.strip() else None
    reported_by = raw.get("reported_by")
    reported_by = (
        reported_by.strip() if isinstance(reported_by, str) and reported_by.strip() else None
    )
    return {
        "kind": kind,
        "detail": detail.strip(),
        "date_iso": date_iso,
        "reported_by": reported_by,
    }


@dataclass
class ExtractedLocation:
    """One place extracted from a single document."""

    name: str
    kind: str = "other"
    associated_character: str | None = None  # e.g. "casa de Salvador Allende"
    facts: list[dict] = field(default_factory=list)


def parse_locations_response(response: str) -> list[ExtractedLocation] | None:
    """Parse the LLM's JSON into ExtractedLocation objects.

    Expected: ``{"locations": [{"name", "kind", "associated_character",
    "facts": [{"kind", "detail", "date_iso", "reported_by"}]}]}``. Lenient per
    location; None only when the top-level shape is wrong.
    """
    try:
        data = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    raw_locations = data.get("locations")
    if not isinstance(raw_locations, list):
        return None

    locations: list[ExtractedLocation] = []
    for raw in raw_locations[:_MAX_LOCATIONS_PER_DOC]:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip() or not location_key(name):
            continue
        kind = raw.get("kind")
        kind = kind if isinstance(kind, str) and kind in _LOCATION_KINDS else "other"
        associated = raw.get("associated_character")
        associated = (
            associated.strip() if isinstance(associated, str) and associated.strip() else None
        )
        facts = [
            cleaned
            for f in (raw.get("facts") or [])[:_MAX_FACTS_PER_LOCATION]
            if (cleaned := _clean_location_fact(f)) is not None
        ]
        locations.append(
            ExtractedLocation(
                name=name.strip(), kind=kind, associated_character=associated, facts=facts
            )
        )
    return locations


def merge_extracted_locations(
    chunks: list[list[ExtractedLocation]],
) -> list[ExtractedLocation]:
    """Union per-chunk extractions by location key (facts concatenated, deduped)."""
    merged: dict[str, ExtractedLocation] = {}
    for chunk in chunks:
        for loc in chunk:
            key = location_key(loc.name)
            if key not in merged:
                merged[key] = loc
                continue
            existing = merged[key]
            if len(loc.name) > len(existing.name):
                existing.name = loc.name
            if existing.kind == "other" and loc.kind != "other":
                existing.kind = loc.kind
            existing.associated_character = existing.associated_character or (
                loc.associated_character
            )
            seen = {f["detail"] for f in existing.facts}
            existing.facts.extend(f for f in loc.facts if f["detail"] not in seen)
    return list(merged.values())


@dataclass(frozen=True)
class LocationCompletenessResult:
    """The 0-1 score plus its breakdown (persisted for the admin)."""

    score: float
    detail: dict


def compute_location_completeness(
    *,
    mention_count: int,
    fact_count: int,
    wikidata_linked: bool,
    wikidata_analyzed: bool,
    wikipedia_linked: bool,
    wikipedia_analyzed: bool,
    has_coordinates: bool,
) -> LocationCompletenessResult:
    """Strict 0-1 completeness for a location, with breakdown.

    Components: documentary base (0.25, mentions/4), facts (0.30, facts/6),
    external linking (0.30 — Wikidata 0.15 + Wikipedia 0.15, linked-only earns
    0.07 each) and coordinates (0.15 — the GPS datum the game needs). Hard gate:
    neither source analyzed → score ≤ 0.60. Complete = ≥ 0.85. Pure; 3 decimals.
    """

    def _source(linked: bool, analyzed: bool) -> float:
        if analyzed:
            return _LOC_EXTERNAL_ANALYZED_CREDIT
        if linked:
            return _LOC_EXTERNAL_LINKED_CREDIT
        return 0.0

    documents = _LOC_WEIGHT_DOCUMENTS * min(1.0, mention_count / _LOC_MENTIONS_TARGET)
    facts = _LOC_WEIGHT_FACTS * min(1.0, fact_count / _LOC_FACTS_TARGET)
    external = _source(wikidata_linked, wikidata_analyzed) + _source(
        wikipedia_linked, wikipedia_analyzed
    )
    coordinates = _LOC_WEIGHT_COORDINATES if has_coordinates else 0.0

    raw = documents + facts + external + coordinates
    caps: list[str] = []
    score = min(1.0, raw)
    if not (wikidata_analyzed or wikipedia_analyzed):
        caps.append("no_external_analysis")
        score = min(score, _LOC_CAP_NO_EXTERNAL_ANALYSIS)
    score = round(score, 3)

    detail = {
        "documents": round(documents, 3),
        "facts": round(facts, 3),
        "external": round(external, 3),
        "coordinates": round(coordinates, 3),
        "raw": round(raw, 3),
        "caps": caps,
        "score": score,
    }
    return LocationCompletenessResult(score=score, detail=detail)


# ── Schema ─────────────────────────────────────────────────────────────────────


def ensure_location_tables(conn: sqlite3.Connection) -> None:
    """Create the locations, location_mentions and location_facts tables (idempotent)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS locations (
            location_id        TEXT PRIMARY KEY,
            name               TEXT NOT NULL,
            aliases            TEXT NOT NULL DEFAULT '[]',
            kind               TEXT NOT NULL DEFAULT 'other',
            description        TEXT,
            latitude           REAL,
            longitude          REAL,
            character_id       TEXT,
            completeness_score REAL NOT NULL DEFAULT 0.0,
            completeness_detail TEXT,
            mention_count      INTEGER NOT NULL DEFAULT 0,
            needs_research     INTEGER NOT NULL DEFAULT 1,
            research_attempts  INTEGER NOT NULL DEFAULT 0,
            last_research_at   TEXT,
            wikidata_id        TEXT,
            wikipedia_url      TEXT,
            wikidata_desc      TEXT,
            wikidata_checked_at  TEXT,
            wikidata_analyzed_at TEXT,
            wikipedia_analyzed_at TEXT,
            run_id             TEXT,
            first_seen_at      TEXT NOT NULL,
            updated_at         TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS location_mentions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id TEXT NOT NULL,
            doc_id      TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            UNIQUE(location_id, doc_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS location_facts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id TEXT NOT NULL,
            doc_id      TEXT,
            date_iso    TEXT,
            kind        TEXT NOT NULL DEFAULT 'data',
            detail      TEXT NOT NULL,
            reported_by TEXT,
            created_at  TEXT NOT NULL,
            UNIQUE(location_id, doc_id, detail)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS location_images (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id TEXT NOT NULL,
            filename    TEXT NOT NULL,
            url         TEXT NOT NULL,
            page_url    TEXT,
            caption     TEXT,
            source      TEXT NOT NULL DEFAULT 'wikidata',
            created_at  TEXT NOT NULL,
            UNIQUE(location_id, filename)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_locfacts_location ON location_facts(location_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_locmentions_location ON location_mentions(location_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_locimages_location ON location_images(location_id)"
    )
    conn.commit()


def _migrate_documents_schema(conn: sqlite3.Connection) -> None:
    """Add the columns the Location Manager stamps/reads on documents (idempotent)."""
    for column in ("locations_extracted_at TEXT", "seed_location_id TEXT"):
        try:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {column}")
        except sqlite3.OperationalError:
            pass  # column already present


# ── SQLite helpers ─────────────────────────────────────────────────────────────


def _load_location_roster(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """All known surface forms per location: canonical name + aliases."""
    roster: dict[str, list[str]] = {}
    for row in conn.execute("SELECT location_id, name, aliases FROM locations"):
        try:
            aliases = json.loads(row["aliases"] or "[]")
        except json.JSONDecodeError:
            aliases = []
        roster[row["location_id"]] = [row["name"], *aliases]
    return roster


def _upsert_location(
    conn: sqlite3.Connection,
    key: str,
    loc: ExtractedLocation,
    run_id: str | None,
) -> bool:
    """Insert a new location or refresh an existing one. True when newly created."""
    now = datetime.now(UTC).isoformat()
    existing = conn.execute(
        "SELECT name, aliases, kind, character_id FROM locations WHERE location_id=?",
        (key,),
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO locations (location_id, name, kind, needs_research, run_id, "
            "first_seen_at, updated_at) VALUES (?,?,?,1,?,?,?)",
            (key, loc.name, loc.kind, run_id, now, now),
        )
        return True

    try:
        aliases = json.loads(existing["aliases"] or "[]")
    except json.JSONDecodeError:
        aliases = []
    known_forms = {location_key(existing["name"]), *(location_key(a) for a in aliases)}
    if location_key(loc.name) not in known_forms:
        aliases.append(loc.name)
    kind = existing["kind"] if existing["kind"] != "other" else loc.kind
    conn.execute(
        "UPDATE locations SET aliases=?, kind=?, updated_at=? WHERE location_id=?",
        (json.dumps(aliases, ensure_ascii=False), kind, now, key),
    )
    return False


def _link_associated_character(conn: sqlite3.Connection, key: str, character_name: str) -> None:
    """Link a location to a character when the roster resolves the name uniquely."""
    try:
        from pipeline.cast_manager import _load_character_roster, resolve_character_key

        roster = _load_character_roster(conn)
    except sqlite3.OperationalError:
        return
    character_id = resolve_character_key(character_name, roster)
    if character_id:
        conn.execute(
            "UPDATE locations SET character_id=? WHERE location_id=? AND character_id IS NULL",
            (character_id, key),
        )


def _add_location_mention(conn: sqlite3.Connection, key: str, doc_id: str) -> int:
    cur = conn.execute(
        "INSERT OR IGNORE INTO location_mentions (location_id, doc_id, created_at) VALUES (?,?,?)",
        (key, doc_id, datetime.now(UTC).isoformat()),
    )
    return cur.rowcount


def _add_location_facts(
    conn: sqlite3.Connection, key: str, doc_id: str | None, facts: list[dict]
) -> int:
    now = datetime.now(UTC).isoformat()
    added = 0
    for fact in facts:
        cur = conn.execute(
            "INSERT OR IGNORE INTO location_facts "
            "(location_id, doc_id, date_iso, kind, detail, reported_by, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                key,
                doc_id,
                fact.get("date_iso"),
                fact.get("kind", "data"),
                fact["detail"],
                fact.get("reported_by"),
                now,
            ),
        )
        added += cur.rowcount
    return added


def _add_location_images(conn: sqlite3.Connection, key: str, images: list[dict]) -> int:
    """Insert gallery images, deduped by normalised filename. Returns added count.

    Each image dict: ``{"filename", "url", "page_url", "caption", "source"}``.
    The UNIQUE(location_id, filename) key means the Wikidata P18 image and the
    Wikipedia article's lead image (same file, different URLs) store once.
    """
    now = datetime.now(UTC).isoformat()
    added = 0
    for image in images:
        cur = conn.execute(
            "INSERT OR IGNORE INTO location_images "
            "(location_id, filename, url, page_url, caption, source, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                key,
                _image_filename(image["filename"]),
                image["url"],
                image.get("page_url"),
                image.get("caption"),
                image.get("source", "wikidata"),
                now,
            ),
        )
        added += cur.rowcount
    return added


def wikipedia_extract_images_url(article_url: str) -> str | None:
    """The extract URL extended to also return the article's lead image.

    One fetch serves both: ``prop=extracts|pageimages`` adds the page's original
    image (and a thumbnail) to the same response the text extract comes from.
    """
    base = wikipedia_extract_url(article_url)
    if base is None:
        return None
    return base.replace(
        "prop=extracts",
        f"prop=extracts%7Cpageimages&piprop=original%7Cthumbnail&pithumbsize={_IMAGE_THUMB_WIDTH}",
    )


def parse_wikipedia_page_image(body: str) -> str | None:
    """The article's lead-image URL from a ``pageimages`` response, or None."""
    try:
        pages = json.loads(body)["query"]["pages"]
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        return None
    for page in pages.values():
        if not isinstance(page, dict):
            continue
        for prop in ("original", "thumbnail"):
            source = (page.get(prop) or {}).get("source")
            if isinstance(source, str) and source.strip():
                return source.strip()
    return None


def _recompute_location_completeness(conn: sqlite3.Connection, key: str) -> None:
    """Regather the metric inputs for a location and persist score + detail."""
    profile = conn.execute(
        "SELECT latitude, longitude, wikidata_id, wikipedia_url, "
        "wikidata_analyzed_at, wikipedia_analyzed_at FROM locations WHERE location_id=?",
        (key,),
    ).fetchone()
    if profile is None:
        return
    mentions = conn.execute(
        "SELECT COUNT(*) FROM location_mentions WHERE location_id=?", (key,)
    ).fetchone()[0]
    facts = conn.execute(
        "SELECT COUNT(*) FROM location_facts WHERE location_id=?", (key,)
    ).fetchone()[0]
    result = compute_location_completeness(
        mention_count=mentions,
        fact_count=facts,
        wikidata_linked=bool(profile["wikidata_id"]),
        wikidata_analyzed=bool(profile["wikidata_analyzed_at"]),
        wikipedia_linked=bool(profile["wikipedia_url"]),
        wikipedia_analyzed=bool(profile["wikipedia_analyzed_at"]),
        has_coordinates=profile["latitude"] is not None and profile["longitude"] is not None,
    )
    conn.execute(
        "UPDATE locations SET mention_count=?, completeness_score=?, "
        "completeness_detail=?, updated_at=? WHERE location_id=?",
        (
            mentions,
            result.score,
            json.dumps(result.detail, ensure_ascii=False),
            datetime.now(UTC).isoformat(),
            key,
        ),
    )


def refresh_location_research_flags(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    """Re-evaluate every location's needs_research flag (same loop as characters)."""
    now = now or datetime.now(UTC)
    try:
        rows = conn.execute(
            "SELECT location_id, needs_research, completeness_score, "
            "research_attempts, last_research_at FROM locations"
        ).fetchall()
    except sqlite3.OperationalError:
        return 0
    changed = 0
    stamp = now.isoformat()
    for row in rows:
        want = should_research(
            float(row["completeness_score"] or 0.0),
            int(row["research_attempts"] or 0),
            row["last_research_at"],
            now,
            threshold=_LOC_COMPLETE_THRESHOLD,
        )
        if int(row["needs_research"] or 0) != int(want):
            conn.execute(
                "UPDATE locations SET needs_research=?, updated_at=? WHERE location_id=?",
                (int(want), stamp, row["location_id"]),
            )
            changed += 1
    if changed:
        conn.commit()
    return changed


def fetch_locations_needing_research(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Locations flagged needs_research — the Propositor's location-mission queue."""
    try:
        return conn.execute(
            "SELECT location_id, name, kind, aliases, mention_count, completeness_score "
            "FROM locations WHERE needs_research = 1 "
            "ORDER BY mention_count DESC, location_id ASC"
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def rescore_locations(conn: sqlite3.Connection) -> dict[str, int]:
    """Recompute every location's completeness under the current constants."""
    ensure_location_tables(conn)
    keys = [row[0] for row in conn.execute("SELECT location_id FROM locations").fetchall()]
    for key in keys:
        _recompute_location_completeness(conn, key)
    conn.commit()
    requeued = refresh_location_research_flags(conn)
    complete = conn.execute(
        "SELECT COUNT(*) FROM locations WHERE completeness_score >= ?",
        (_LOC_COMPLETE_THRESHOLD,),
    ).fetchone()[0]
    return {"rescored": len(keys), "complete": complete, "requeued": requeued}


# ── LLM extraction ─────────────────────────────────────────────────────────────

_LLM_SYSTEM = (
    "You are a historical geographer for a Digital Humanities project, cataloguing "
    f"the places where events happen in documents about {project().entity_prompts.era_label}. "
    "You respond ONLY with a JSON object. No explanation, no markdown."
)

_LLM_PROMPT = """\
Extract every identifiable PLACE OR LOCATION where events in this document happen
or that the document describes — buildings, factories, offices, residences,
streets, cities, regions. For each place return:
  - "name": the place's name as written (e.g. "Palacio de La Moneda",
    "Fábrica de harina de Temuco", "casa de Salvador Allende en Tomás Moro")
  - "kind": one of "building", "factory", "residence", "city", "region",
    "street", "office", "other"
  - "associated_character": the full name of the person the place belongs to or
    is strongly tied to (e.g. the owner of a residence), else null
  - "facts": 0-{max_facts} facts the document states about the place, each:
      - "kind": "description" (physical/objective description), "appreciation"
        (a subjective opinion or feeling about the place), "event" (something
        that happened there), or "data" (hard datum: address, capacity, output)
      - "detail": one concise clause
      - "date_iso": "YYYY-MM" or "YYYY-MM-DD" if the text dates it, else null
      - "reported_by": for "appreciation", who voices the opinion, if stated

RULES:
- Only real, named or clearly identifiable places. Skip vague references
  ("el norte", "la ciudad") unless the document names them.
- A person's home counts when the document treats it as a place ("casa de X").
- Do not invent facts; a place merely named returns an empty "facts" list.

Respond with EXACTLY this shape:
{{"locations": [{{"name": "...", "kind": "building", "associated_character": null, "facts": [{{"kind": "event", "detail": "...", "date_iso": null, "reported_by": null}}]}}]}}

TITLE: {title}

DOCUMENT:
{text}"""


async def _llm_extract_locations(
    text: str, title: str, client: LLMClient
) -> list[ExtractedLocation] | None:
    """Chunked extraction over one document; per-chunk results unioned by key."""
    chunks = chunk_text(text)
    per_chunk: list[list[ExtractedLocation]] = []
    for chunk in chunks:
        prompt = _LLM_PROMPT.format(
            title=title[:200] or "(no title)",
            text=chunk,
            max_facts=_MAX_FACTS_PER_LOCATION,
        )
        try:
            raw = await client.chat(
                model=settings.ollama_model_npc,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=_LLM_TEMPERATURE,
                num_predict=700,
                think=False,
            )
        except LLMError as exc:
            log.warning("location_manager.llm_error", reason=str(exc)[:120])
            return None
        parsed = parse_locations_response(raw)
        if parsed is None:
            log.warning("location_manager.llm_unparseable", title=title[:60])
            continue
        per_chunk.append(parsed)
    if not per_chunk:
        return None
    return merge_extracted_locations(per_chunk)


# ── Analyze: Wikidata claims + Wikipedia article ───────────────────────────────


def _image_filename(value: str) -> str:
    """Normalise a Commons filename or upload URL into a dedup key.

    "File:Palacio de La Moneda.jpg" and
    "https://upload.wikimedia.org/.../Palacio_de_la_moneda.JPG" both resolve to
    the same key, so the Wikidata P18 image and the article's lead image never
    duplicate each other in a location's gallery.
    """
    name = value.rsplit("/", 1)[-1]
    name = urllib.parse.unquote(name)
    if name.lower().startswith("file:"):
        name = name[5:]
    return name.replace(" ", "_").lower()


def commons_image_url(filename: str, width: int = _IMAGE_THUMB_WIDTH) -> str:
    """A directly-loadable Commons URL for a P18 filename (server-side thumbnail)."""
    quoted = urllib.parse.quote(filename.replace(" ", "_"))
    return f"{_COMMONS_FILE_PATH}{quoted}?width={width}"


def commons_page_url(filename: str) -> str:
    """The Commons File: page for a filename — the attribution/licence link."""
    return _COMMONS_FILE_PAGE + urllib.parse.quote(filename.replace(" ", "_"))


def parse_location_claims(body: str, qid: str) -> dict | None:
    """Parse a location entity's claims: coordinates, inception, architect,
    territory and P18 images (Commons filenames)."""
    try:
        claims = json.loads(body)["entities"][qid]["claims"]
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        return None
    if not isinstance(claims, dict):
        return None

    def _first_value(prop: str) -> dict | None:
        for statement in claims.get(prop) or []:
            value = ((statement.get("mainsnak") or {}).get("datavalue") or {}).get("value")
            if value is not None:
                return value if isinstance(value, dict) else None
        return None

    coords = _first_value(_WD_COORDINATES)
    latitude = longitude = None
    if coords and isinstance(coords.get("latitude"), int | float):
        latitude = float(coords["latitude"])
        longitude = float(coords.get("longitude", 0.0))

    inception = None
    inception_value = _first_value(_WD_INCEPTION)
    if inception_value:
        inception = _parse_wikidata_time(inception_value)

    entity_refs: list[dict] = []
    for prop, label_prefix in ((_WD_ARCHITECT, "Arquitecto"), (_WD_TERRITORY, "Ubicación")):
        value = _first_value(prop)
        value_qid = value.get("id") if isinstance(value, dict) else None
        if isinstance(value_qid, str) and value_qid:
            entity_refs.append({"prop": prop, "value_qid": value_qid, "prefix": label_prefix})

    images: list[str] = []
    for statement in claims.get(_WD_IMAGE) or []:
        value = ((statement.get("mainsnak") or {}).get("datavalue") or {}).get("value")
        if isinstance(value, str) and value.strip():
            images.append(value.strip())
        if len(images) >= _MAX_IMAGES_PER_LOCATION:
            break

    return {
        "latitude": latitude,
        "longitude": longitude,
        "inception": inception,
        "entity_refs": entity_refs,
        "images": images,
    }


_ANALYZE_SYSTEM = (
    f"Eres un historiador urbano especializado en {project().entity_prompts.era_label_local}. "
    "Respondes SOLO con JSON válido, sin texto adicional."
)

_ANALYZE_PROMPT = """Lee este artículo de Wikipedia sobre el lugar "{name}" y devuelve JSON:

{{"description": "un solo párrafo (máx. 3 frases) describiendo el lugar y su papel
en el periodo 1969-1973 chileno",
 "facts": [{{"kind": "description|event|data", "detail": "hecho concreto — incluye
 datos arquitectónicos, cifras, curiosidades", "date_iso": "YYYY-MM o YYYY o null"}}]}}

Reglas: máximo {max_facts} hechos; prioriza arquitectura, historia del edificio,
datos curiosos y el periodo 1969-1973; no inventes nada que el artículo no diga.

ARTÍCULO:
{article}"""


def parse_location_analysis(raw: str) -> dict | None:
    """Parse the article-analysis JSON: a description + cleaned facts."""
    try:
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    if not isinstance(data, dict):
        return None
    description = data.get("description")
    description = (
        description.strip()[:_LOC_DESCRIPTION_MAX_CHARS]
        if isinstance(description, str) and description.strip()
        else None
    )
    facts = [
        cleaned
        for raw_fact in (data.get("facts") or [])[:_MAX_FACTS_PER_LOCATION]
        if (cleaned := _clean_location_fact(raw_fact)) is not None
    ]
    if description is None and not facts:
        return None
    return {"description": description, "facts": facts}


async def _llm_analyze_location(name: str, article: str, client: LLMClient) -> dict | None:
    """Ask Gemma to mine a Wikipedia article about a place."""
    prompt = _ANALYZE_PROMPT.format(
        name=name[:120], max_facts=_MAX_FACTS_PER_LOCATION, article=article[:_LOC_ARTICLE_MAX_CHARS]
    )
    try:
        raw = await client.chat(
            model=settings.ollama_model_npc,
            messages=[
                {"role": "system", "content": _ANALYZE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=_LLM_TEMPERATURE,
            num_predict=700,
            think=False,
        )
        return parse_location_analysis(raw)
    except LLMError as exc:
        log.warning("location_manager.analyze_llm_error", reason=str(exc)[:120])
        return None
    except Exception as exc:
        log.warning("location_manager.analyze_unexpected_error", reason=str(exc)[:120])
        return None


def enrich_locations(
    conn: sqlite3.Connection,
    fetch,
    limit: int = 50,
) -> dict[str, int]:
    """Resolve Wikidata/Wikipedia links for locations not yet checked.

    Same contract as the Cast Manager's enrich: every processed row is stamped
    ``wikidata_checked_at`` (matched or not) so runs self-terminate.
    """
    ensure_location_tables(conn)
    rows = conn.execute(
        "SELECT location_id, name FROM locations WHERE wikidata_checked_at IS NULL "
        "ORDER BY completeness_score DESC, mention_count DESC LIMIT ?",
        (limit,),
    ).fetchall()
    checked = resolved = 0
    for row in rows:
        now = datetime.now(UTC).isoformat()
        info = resolve_wikidata(row["name"], fetch)
        if info is not None:
            conn.execute(
                "UPDATE locations SET wikidata_id=?, wikipedia_url=?, wikidata_desc=?, "
                "wikidata_checked_at=?, updated_at=? WHERE location_id=?",
                (
                    info["wikidata_id"],
                    info["wikipedia_url"],
                    info["description"],
                    now,
                    now,
                    row["location_id"],
                ),
            )
            resolved += 1
            _recompute_location_completeness(conn, row["location_id"])
        else:
            conn.execute(
                "UPDATE locations SET wikidata_checked_at=?, updated_at=? WHERE location_id=?",
                (now, now, row["location_id"]),
            )
        conn.commit()
        checked += 1
    return {"checked": checked, "resolved": resolved}


# ── Result dataclass + agent ───────────────────────────────────────────────────


@dataclass
class LocationResult:
    """Result of one Location Manager run cycle."""

    processed: int = 0
    locations_new: int = 0
    locations_updated: int = 0
    mentions_new: int = 0
    facts_new: int = 0
    research_requeued: int = 0


class LocationManager:
    """Ag-4: Extracts places from verified, mapped documents into the location tables.

    Reads documents with ``verified_at`` and ``mapped_category_id`` set but no
    ``locations_extracted_at``, extracts the places in each via Gemma, and
    upserts locations, mentions and facts. A document harvested by a location
    mission (``seed_location_id``) is force-attributed to its seed, closing the
    research loop exactly like the Cast Manager's character loop.
    """

    def __init__(self, db_path: Path | None = None, use_llm: bool = True) -> None:
        self.db_path = db_path or settings.archive_db
        self.use_llm = use_llm

    async def run_cycle(self, batch_size: int = 50, all_docs: bool = False) -> LocationResult:
        """Mine one batch (or with ``all_docs`` every batch) of unprocessed docs."""
        result = LocationResult()
        if not self.db_path.exists():
            log.info("location_manager.db_not_found", path=str(self.db_path))
            return result

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        run_id = current_run_id()
        try:
            ensure_location_tables(conn)
            _migrate_documents_schema(conn)

            client: LLMClient | None = None
            if self.use_llm:
                client = await self._open_ollama()
                if client is None:
                    return result
            try:
                while True:
                    rows = conn.execute(
                        "SELECT doc_id, title, text, seed_location_id FROM documents "
                        "WHERE verified_at IS NOT NULL AND mapped_category_id IS NOT NULL "
                        "AND locations_extracted_at IS NULL LIMIT ?",
                        (batch_size,),
                    ).fetchall()
                    if not rows:
                        break
                    await self._process_batch(rows, conn, client, run_id, result)
                    conn.commit()
                    if not all_docs:
                        break
            finally:
                if client is not None:
                    await client.__aexit__(None, None, None)

            result.research_requeued = refresh_location_research_flags(conn)
        finally:
            conn.close()

        log.info(
            "location_manager.cycle_done",
            processed=result.processed,
            locations_new=result.locations_new,
            mentions_new=result.mentions_new,
            facts_new=result.facts_new,
        )
        return result

    async def _open_ollama(self) -> LLMClient | None:
        """Open an Ollama client and verify the model is present, else None."""
        client = LLMClient()
        await client.__aenter__()
        try:
            models = await client.list_models()
            if not any(settings.ollama_model_npc in m.get("name", "") for m in models):
                log.error("location_manager.model_not_found", model=settings.ollama_model_npc)
                await client.__aexit__(None, None, None)
                return None
        except Exception as exc:
            log.error("location_manager.ollama_unreachable", reason=str(exc)[:120])
            print("\n  ERROR: Ollama is not running. Start it with: ollama serve\n")
            await client.__aexit__(None, None, None)
            return None
        return client

    async def _process_batch(
        self,
        rows: list[sqlite3.Row],
        conn: sqlite3.Connection,
        client: LLMClient | None,
        run_id: str | None,
        result: LocationResult,
    ) -> None:
        for row in rows:
            extracted: list[ExtractedLocation] = []
            if client is not None:
                extracted = (
                    await _llm_extract_locations(row["text"] or "", row["title"] or "", client)
                    or []
                )
            roster = _load_location_roster(conn)
            extracted_keys: set[str] = set()
            for loc in extracted:
                key = resolve_location_key(loc.name, roster) or location_key(loc.name)
                if not key:
                    continue
                if _upsert_location(conn, key, loc, run_id):
                    result.locations_new += 1
                else:
                    result.locations_updated += 1
                result.mentions_new += _add_location_mention(conn, key, row["doc_id"])
                result.facts_new += _add_location_facts(conn, key, row["doc_id"], loc.facts)
                if loc.associated_character:
                    _link_associated_character(conn, key, loc.associated_character)
                _recompute_location_completeness(conn, key)
                extracted_keys.add(key)

            # Seed attribution: a doc harvested for a location always counts as
            # evidence for it, even if the LLM did not re-name the seed.
            seed = self._row_seed(row)
            if seed and seed not in extracted_keys:
                exists = conn.execute(
                    "SELECT 1 FROM locations WHERE location_id=?", (seed,)
                ).fetchone()
                if exists:
                    result.mentions_new += _add_location_mention(conn, seed, row["doc_id"])
                    _recompute_location_completeness(conn, seed)

            conn.execute(
                "UPDATE documents SET locations_extracted_at=? WHERE doc_id=?",
                (datetime.now(UTC).isoformat(), row["doc_id"]),
            )
            result.processed += 1

    @staticmethod
    def _row_seed(row: sqlite3.Row) -> str | None:
        try:
            return row["seed_location_id"] or None
        except (IndexError, KeyError):
            return None

    async def analyze(self, fetch, limit: int = 30) -> dict[str, int]:
        """Mine linked locations: Wikidata claims (coords, inception, architect,
        territory — no LLM) and the Wikipedia article (Gemma → description +
        facts, architectural data and trivia included). Stamps the
        ``*_analyzed_at`` gates; failures leave no stamp (retried next run)."""
        counts = {
            "analyzed": 0,
            "wikidata_done": 0,
            "wikipedia_done": 0,
            "facts_new": 0,
            "descriptions_set": 0,
            "images_new": 0,
        }
        if not self.db_path.exists():
            return counts

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            ensure_location_tables(conn)
            rows = conn.execute(
                "SELECT location_id, name, description, wikidata_id, wikipedia_url, "
                "wikidata_analyzed_at, wikipedia_analyzed_at FROM locations "
                "WHERE wikidata_id IS NOT NULL "
                "AND (wikidata_analyzed_at IS NULL "
                "     OR (wikipedia_url IS NOT NULL AND wikipedia_analyzed_at IS NULL)) "
                "ORDER BY completeness_score DESC, mention_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
            if not rows:
                return counts

            client = await self._open_ollama() if self.use_llm else None
            try:
                for row in rows:
                    if row["wikidata_analyzed_at"] is None and not self._analyze_wikidata(
                        conn, row, fetch, counts
                    ):
                        counts["analyzed"] += 1
                        continue
                    if (
                        client is not None
                        and row["wikipedia_url"]
                        and row["wikipedia_analyzed_at"] is None
                    ):
                        await self._analyze_wikipedia(conn, row, fetch, client, counts)
                    counts["analyzed"] += 1
                    conn.commit()
            finally:
                if client is not None:
                    await client.__aexit__(None, None, None)
        finally:
            conn.close()

        log.info("location_manager.analyze_done", **counts)
        return counts

    @staticmethod
    def _analyze_wikidata(conn, row, fetch, counts) -> bool:
        """One location's Wikidata side. False on fetch/parse failure (no stamp)."""
        key = row["location_id"]
        body = fetch(_wikidata_claims_url(row["wikidata_id"]))
        if body is None:
            return False
        parsed = parse_location_claims(body, row["wikidata_id"])
        if parsed is None:
            return False
        now = datetime.now(UTC).isoformat()

        if parsed["latitude"] is not None:
            conn.execute(
                "UPDATE locations SET latitude=?, longitude=?, updated_at=? WHERE location_id=?",
                (parsed["latitude"], parsed["longitude"], now, key),
            )
        facts: list[dict] = []
        if parsed["inception"]:
            facts.append(
                {
                    "kind": "data",
                    "detail": "Construcción/fundación",
                    "date_iso": parsed["inception"],
                }
            )
        if parsed["entity_refs"]:
            qids = sorted({ref["value_qid"] for ref in parsed["entity_refs"]})
            labels_body = fetch(wikidata_labels_url(qids))
            labels = parse_wikidata_labels(labels_body) if labels_body else {}
            for ref in parsed["entity_refs"]:
                label = labels.get(ref["value_qid"])
                if label:
                    facts.append({"kind": "data", "detail": f"{ref['prefix']}: {label}"})
        counts["facts_new"] += _add_location_facts(conn, key, None, facts)
        counts["images_new"] += _add_location_images(
            conn,
            key,
            [
                {
                    "filename": filename,
                    "url": commons_image_url(filename),
                    "page_url": commons_page_url(filename),
                    "caption": None,
                    "source": "wikidata",
                }
                for filename in parsed["images"]
            ],
        )
        conn.execute(
            "UPDATE locations SET wikidata_analyzed_at=?, updated_at=? WHERE location_id=?",
            (now, now, key),
        )
        counts["wikidata_done"] += 1
        _recompute_location_completeness(conn, key)
        conn.commit()
        return True

    @staticmethod
    async def _analyze_wikipedia(conn, row, fetch, client, counts) -> None:
        key = row["location_id"]
        api_url = wikipedia_extract_images_url(row["wikipedia_url"])
        if api_url is None:
            return
        body = fetch(api_url)
        article = parse_wikipedia_extract(body) if body is not None else None
        if not article:
            return
        analysis = await _llm_analyze_location(row["name"], article, client)
        if analysis is None:
            return
        now = datetime.now(UTC).isoformat()
        if analysis["description"] and not (row["description"] and row["description"].strip()):
            conn.execute(
                "UPDATE locations SET description=?, updated_at=? WHERE location_id=?",
                (analysis["description"], now, key),
            )
            counts["descriptions_set"] += 1
        counts["facts_new"] += _add_location_facts(conn, key, None, analysis["facts"])
        lead_image = parse_wikipedia_page_image(body)
        if lead_image:
            counts["images_new"] += _add_location_images(
                conn,
                key,
                [
                    {
                        "filename": lead_image,
                        "url": lead_image,
                        "page_url": commons_page_url(_image_filename(lead_image)),
                        "caption": "Imagen principal del artículo",
                        "source": "wikipedia",
                    }
                ],
            )
        conn.execute(
            "UPDATE locations SET wikipedia_analyzed_at=?, updated_at=? WHERE location_id=?",
            (now, now, key),
        )
        counts["wikipedia_done"] += 1
        _recompute_location_completeness(conn, key)
        conn.commit()

    def status(self) -> dict[str, int]:
        """Location counts: locations, mentions, facts, needs_research, pending docs."""
        if not self.db_path.exists():
            return {}
        conn = sqlite3.connect(str(self.db_path))
        try:
            ensure_location_tables(conn)
            _migrate_documents_schema(conn)

            def _count(sql: str) -> int:
                try:
                    return conn.execute(sql).fetchone()[0]
                except sqlite3.OperationalError:
                    return 0

            return {
                "locations": _count("SELECT COUNT(*) FROM locations"),
                "mentions": _count("SELECT COUNT(*) FROM location_mentions"),
                "facts": _count("SELECT COUNT(*) FROM location_facts"),
                "needs_research": _count("SELECT COUNT(*) FROM locations WHERE needs_research=1"),
                "pending_docs": _count(
                    "SELECT COUNT(*) FROM documents WHERE verified_at IS NOT NULL "
                    "AND mapped_category_id IS NOT NULL AND locations_extracted_at IS NULL"
                ),
            }
        finally:
            conn.close()


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    import traceback

    from lib.logging_setup import configure_logging

    configure_logging()

    ap = argparse.ArgumentParser(
        prog="location-manager",
        description="Ag-4 Location Manager — extract places from verified, mapped documents",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Extract places from verified+mapped documents")
    run_p.add_argument("--batch", type=int, default=50, help="Documents per batch (default: 50)")
    run_p.add_argument(
        "--all", action="store_true", help="Process ALL pending documents (self-terminating)"
    )

    enrich_p = sub.add_parser("enrich", help="Link locations to Wikidata + Wikipedia")
    enrich_p.add_argument("--limit", type=int, default=100)
    analyze_p = sub.add_parser(
        "analyze", help="Mine linked Wikidata claims (GPS, architect) + Wikipedia articles"
    )
    analyze_p.add_argument("--limit", type=int, default=30)
    analyze_p.add_argument(
        "--no-llm", action="store_true", help="Wikidata only — skip the article analysis"
    )
    sub.add_parser("score", help="Recompute every location's completeness")
    sub.add_parser("status", help="Print location counts")

    args = ap.parse_args()

    if args.cmd == "run":
        try:
            manager = LocationManager(use_llm=True)
            t0 = datetime.now(UTC)
            result = asyncio.run(manager.run_cycle(batch_size=args.batch, all_docs=args.all))
            rid = current_run_id()
            if rid:
                record_stage(
                    rid,
                    "location_manager",
                    t0,
                    datetime.now(UTC),
                    {
                        "processed": result.processed,
                        "locations_new": result.locations_new,
                        "mentions_new": result.mentions_new,
                        "facts_new": result.facts_new,
                    },
                )
            print(
                f"  Procesados: {result.processed} docs\n"
                f"  Lugares: +{result.locations_new} nuevos, {result.locations_updated} actualizados\n"
                f"  Menciones: +{result.mentions_new}\n"
                f"  Hechos:    +{result.facts_new}"
            )
        except Exception as exc:
            log.error("location_manager.main_error", error=str(exc)[:200])
            print(f"\n  ERROR: {exc}\n")
            traceback.print_exc()

    elif args.cmd == "enrich":
        conn = sqlite3.connect(str(settings.archive_db))
        conn.row_factory = sqlite3.Row
        try:
            result = enrich_locations(conn, _build_gatekeeper_fetch(), limit=args.limit)
            print(f"  Revisados: {result['checked']} lugares")
            print(f"  Vinculados a Wikidata: {result['resolved']}")
        finally:
            conn.close()

    elif args.cmd == "analyze":
        manager = LocationManager(use_llm=not args.no_llm)
        t0 = datetime.now(UTC)
        result = asyncio.run(manager.analyze(_build_gatekeeper_fetch(), limit=args.limit))
        rid = current_run_id()
        if rid:
            record_stage(rid, "location_analyze", t0, datetime.now(UTC), result)
        print(f"  Analizados: {result['analyzed']} lugares")
        print(f"  Wikidata:   {result['wikidata_done']} (coordenadas, arquitecto, fundación)")
        print(f"  Imágenes:   +{result['images_new']} (Commons)")
        print(
            f"  Wikipedia:  {result['wikipedia_done']} artículos → {result['descriptions_set']} descripciones"
        )
        print(f"  Hechos:     +{result['facts_new']}")

    elif args.cmd == "score":
        conn = sqlite3.connect(str(settings.archive_db))
        conn.row_factory = sqlite3.Row
        try:
            result = rescore_locations(conn)
            print(f"  Re-evaluados: {result['rescored']} lugares")
            print(f"  Completos (≥ {_LOC_COMPLETE_THRESHOLD}): {result['complete']}")
            print(f"  Re-encolados: {result['requeued']}")
        finally:
            conn.close()

    elif args.cmd == "status":
        status = LocationManager().status()
        if status:
            for key, value in status.items():
                print(f"  {key}: {value}")
        else:
            print("Database not found or empty.")


if __name__ == "__main__":
    main()
