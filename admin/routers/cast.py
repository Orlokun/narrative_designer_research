"""Cast Manager admin endpoints — the character list and per-character detail.

Backs the /cast-manager page: a searchable character list with completeness
scores, and a detail view with the biography, the dated timeline of facts and
the documents where the character is mentioned (Cast tables, see
wiki/data-model.html#cast).
"""

from __future__ import annotations

import json
import math
import re
import sqlite3

from fastapi import APIRouter, HTTPException

from lib.config import settings
from pipeline.cast_director import _DIRECTED_KINDS as _DIRECTED_RELATION_KINDS
from pipeline.cast_manager import _COMPLETE_THRESHOLD as COMPLETE_THRESHOLD

router = APIRouter(tags=["cast"])

# The Cybersyn period is always highlighted on every character's life-line
# (years 1970, 1971, 1972, 1973 — the band spans [1970, 1974)).
HIGHLIGHT_START: int = 1970
HIGHLIGHT_END: int = 1974

_YEAR_RE = re.compile(r"(\d{4})")
_YEAR_MONTH_RE = re.compile(r"^\s*\d{4}-(\d{2})")
_YEAR_MONTH_DAY_RE = re.compile(r"^\s*\d{4}-\d{2}-(\d{2})")


def _year_fraction(date_iso: str | None) -> float | None:
    """Parse an ISO-ish date into a fractional year, or None if it has no year.

    "1908" → 1908.0, "1970-10" → 1970.75, "1973-09-11" → ~1973.69. Tolerates the
    corpus's quirks: a range like "1970-1973" takes the first year, "1970-00"
    (month 0) and "c. 1908" resolve to the year start.
    """
    if not date_iso:
        return None
    match = _YEAR_RE.search(date_iso)
    if not match:
        return None
    year = int(match.group(1))
    fraction = 0.0
    month_match = _YEAR_MONTH_RE.match(date_iso)
    if month_match:
        month = int(month_match.group(1))
        if 1 <= month <= 12:
            fraction += (month - 1) / 12.0
            day_match = _YEAR_MONTH_DAY_RE.match(date_iso)
            if day_match:
                day = int(day_match.group(1))
                if 1 <= day <= 31:
                    fraction += (day - 1) / 31.0 / 12.0
    return year + fraction


def _timeline_span(
    birth_date: str | None,
    death_date: str | None,
    fact_dates: list[str | None],
) -> tuple[int, int]:
    """Return (start_year, end_year) for a character's life-line x-axis.

    Bounded by birth/death when known, otherwise stretched to the earliest and
    latest dated fact — but always wide enough to contain the 1970-1973
    highlight band.
    """
    fact_years = [y for y in (_year_fraction(d) for d in fact_dates) if y is not None]
    lows = [y for y in (_year_fraction(birth_date), *fact_years) if y is not None]
    highs = [y for y in (_year_fraction(death_date), *fact_years) if y is not None]
    start = math.floor(min(*lows, HIGHLIGHT_START)) if lows else HIGHLIGHT_START
    end = math.ceil(max(*highs, HIGHLIGHT_END)) if highs else HIGHLIGHT_END
    return start, end


def _connect() -> sqlite3.Connection | None:
    if not settings.archive_db.exists():
        return None
    conn = sqlite3.connect(str(settings.archive_db))
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/cast/characters")
async def character_list(search: str = "") -> dict:
    """Characters ordered most-complete first; `search` filters name + aliases."""
    conn = _connect()
    if conn is None:
        return {"available": False, "total": 0, "characters": []}

    try:
        params: tuple = ()
        where = ""
        if search.strip():
            where = "WHERE LOWER(name) LIKE ? OR LOWER(aliases) LIKE ?"
            needle = f"%{search.strip().lower()}%"
            params = (needle, needle)
        rows = conn.execute(
            f"""
            SELECT character_id, name, completeness_score, mention_count, needs_research
            FROM characters
            {where}
            ORDER BY completeness_score DESC, mention_count DESC, name ASC
            """,
            params,
        ).fetchall()
        return {
            "available": True,
            "total": len(rows),
            "characters": [dict(row) for row in rows],
        }
    except Exception:
        return {"available": False, "total": 0, "characters": []}
    finally:
        conn.close()


@router.get("/cast/stats")
async def cast_stats() -> dict:
    """Score distribution + active completeness gates — the calibration view.

    Ten histogram buckets ([0,0.1) … [0.9,1.0]), the complete count against the
    strict threshold, and how many characters each hard gate is currently
    capping (parsed from the persisted ``completeness_detail`` breakdowns).
    """
    empty = {
        "available": False,
        "total": 0,
        "complete": 0,
        "threshold": COMPLETE_THRESHOLD,
        "histogram": [0] * 10,
        "caps": {},
    }
    conn = _connect()
    if conn is None:
        return empty

    try:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(characters)").fetchall()}
        has_detail = "completeness_detail" in columns
        select = (
            "SELECT completeness_score"
            + (", completeness_detail" if has_detail else "")
            + " FROM characters"
        )
        histogram = [0] * 10
        caps: dict[str, int] = {}
        total = complete = 0
        for row in conn.execute(select).fetchall():
            total += 1
            score = float(row["completeness_score"] or 0.0)
            if score >= COMPLETE_THRESHOLD:
                complete += 1
            histogram[min(9, int(score * 10))] += 1
            if has_detail and row["completeness_detail"]:
                try:
                    for cap in json.loads(row["completeness_detail"]).get("caps") or []:
                        caps[cap] = caps.get(cap, 0) + 1
                except json.JSONDecodeError:
                    pass
        return {
            "available": True,
            "total": total,
            "complete": complete,
            "threshold": COMPLETE_THRESHOLD,
            "histogram": histogram,
            "caps": caps,
        }
    except Exception:
        return empty
    finally:
        conn.close()


@router.get("/cast/characters/{character_id}")
async def character_detail(character_id: str) -> dict:
    """One character: bio + aliases, dated timeline, and mentioning documents."""
    conn = _connect()
    if conn is None:
        raise HTTPException(status_code=404, detail="database not found")

    try:
        # birth_date/death_date are recent columns; tolerate a DB whose Cast
        # Manager hasn't migrated yet (this reader never mutates the schema).
        columns = {r[1] for r in conn.execute("PRAGMA table_info(characters)").fetchall()}
        optional = [
            c
            for c in (
                "birth_date",
                "death_date",
                "wikidata_id",
                "wikipedia_url",
                "wikidata_desc",
                "pol_economic",
                "pol_social",
                "pol_label",
                "completeness_detail",
            )
            if c in columns
        ]
        select_cols = (
            "character_id, name, aliases, biography, completeness_score, "
            "mention_count, needs_research, first_seen_at, updated_at"
            + ("".join(f", {c}" for c in optional))
        )
        row = conn.execute(
            f"SELECT {select_cols} FROM characters WHERE character_id = ?",
            (character_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character not found")

        character = dict(row)
        for field in (
            "birth_date",
            "death_date",
            "wikidata_id",
            "wikipedia_url",
            "wikidata_desc",
            "pol_economic",
            "pol_social",
            "pol_label",
            "completeness_detail",
        ):
            character.setdefault(field, None)
        try:
            character["aliases"] = json.loads(character.get("aliases") or "[]")
        except json.JSONDecodeError:
            character["aliases"] = []
        if character["completeness_detail"]:
            try:
                character["completeness_detail"] = json.loads(character["completeness_detail"])
            except json.JSONDecodeError:
                character["completeness_detail"] = None

        # speech_act/reported_by/confidence are recent columns; same tolerance.
        timeline_columns = {
            r[1] for r in conn.execute("PRAGMA table_info(character_timeline)").fetchall()
        }
        narrative = [
            c for c in ("speech_act", "reported_by", "confidence") if c in timeline_columns
        ]
        timeline = []
        for fact in conn.execute(
            f"""
            SELECT date_iso, kind, description, doc_id
                   {"".join(f", {c}" for c in narrative)}
            FROM character_timeline
            WHERE character_id = ?
            ORDER BY COALESCE(date_iso, '9999') ASC, id ASC
            """,
            (character_id,),
        ).fetchall():
            item = dict(fact)
            for field in ("speech_act", "reported_by", "confidence"):
                item.setdefault(field, None)
            item["year"] = _year_fraction(fact["date_iso"])
            timeline.append(item)

        start, end = _timeline_span(
            character["birth_date"],
            character["death_date"],
            [fact["date_iso"] for fact in timeline],
        )

        documents = [
            dict(doc)
            for doc in conn.execute(
                """
                SELECT m.doc_id,
                       COALESCE(d.title, m.doc_id) AS title,
                       m.mentioned_by
                FROM character_mentions m
                LEFT JOIN documents d ON d.doc_id = m.doc_id
                WHERE m.character_id = ?
                ORDER BY m.created_at ASC
                """,
                (character_id,),
            ).fetchall()
        ]

        return {
            "character": character,
            "timeline": timeline,
            "documents": documents,
            "span": {"start": start, "end": end},
            "highlight": {"start": HIGHLIGHT_START, "end": HIGHLIGHT_END},
        }
    finally:
        conn.close()


@router.get("/cast/characters/{character_id}/relations")
async def character_relations(character_id: str) -> dict:
    """A character's relations (Cast Director), both directions, richest first.

    Each entry names the other endpoint (linkable), the kind, confidence, the
    description and the documents that justify it. Directed kinds (superior,
    mentor) expose whether this character is the source (outgoing) or target.
    """
    conn = _connect()
    if conn is None:
        return {"relations": []}

    try:
        rows = conn.execute(
            """
            SELECT r.source_character_id, r.target_character_id, r.kind,
                   r.description, r.confidence, r.mention_count, r.provenance,
                   sc.name AS source_name, tc.name AS target_name
            FROM character_relations r
            LEFT JOIN characters sc ON sc.character_id = r.source_character_id
            LEFT JOIN characters tc ON tc.character_id = r.target_character_id
            WHERE r.source_character_id = ? OR r.target_character_id = ?
            ORDER BY r.confidence DESC, r.mention_count DESC
            """,
            (character_id, character_id),
        ).fetchall()
    except sqlite3.OperationalError:
        return {"relations": []}  # character_relations table not created yet

    relations = []
    for row in rows:
        outgoing = row["source_character_id"] == character_id
        other_id = row["target_character_id"] if outgoing else row["source_character_id"]
        other_name = (row["target_name"] if outgoing else row["source_name"]) or other_id
        try:
            provenance = json.loads(row["provenance"] or "[]")
        except json.JSONDecodeError:
            provenance = []
        relations.append(
            {
                "other_id": other_id,
                "other_name": other_name,
                "kind": row["kind"],
                "direction": "outgoing" if outgoing else "incoming",
                "description": row["description"] or "",
                "confidence": row["confidence"],
                "provenance": provenance,
            }
        )
    conn.close()
    return {"relations": relations}


@router.get("/cast/graph")
async def cast_graph() -> dict:
    """The whole relation network: nodes (characters in ≥1 relation) + edges.

    Backs the Cast Director page's network graph. Nodes carry their degree
    (edge count) for sizing; edges carry kind + confidence for styling.
    """
    conn = _connect()
    if conn is None:
        return {"available": False, "nodes": [], "edges": []}

    try:
        rows = conn.execute(
            """
            SELECT r.source_character_id, r.target_character_id, r.kind, r.confidence,
                   r.description, r.provenance,
                   sc.name AS source_name, tc.name AS target_name
            FROM character_relations r
            LEFT JOIN characters sc ON sc.character_id = r.source_character_id
            LEFT JOIN characters tc ON tc.character_id = r.target_character_id
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return {"available": False, "nodes": [], "edges": []}
    finally:
        conn.close()

    names: dict[str, str] = {}
    degree: dict[str, int] = {}
    edges = []
    for row in rows:
        src, tgt = row["source_character_id"], row["target_character_id"]
        names[src] = row["source_name"] or src
        names[tgt] = row["target_name"] or tgt
        degree[src] = degree.get(src, 0) + 1
        degree[tgt] = degree.get(tgt, 0) + 1
        try:
            provenance = json.loads(row["provenance"] or "[]")
        except json.JSONDecodeError:
            provenance = []
        edges.append(
            {
                "source": src,
                "target": tgt,
                "kind": row["kind"],
                "confidence": row["confidence"],
                "directed": row["kind"] in _DIRECTED_RELATION_KINDS,
                "description": row["description"] or "",
                "provenance": provenance,
            }
        )

    nodes = [
        {"id": cid, "name": names[cid], "degree": degree[cid]}
        for cid in sorted(names, key=lambda c: (-degree[c], c))
    ]
    return {"available": True, "nodes": nodes, "edges": edges}
