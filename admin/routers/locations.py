"""Location Manager admin endpoints — the location list, detail and map data.

Backs the /locations page: a searchable location list with completeness scores,
an SVG coordinate map (every located place is plotted by latitude/longitude),
and a detail view with the description, per-kind facts (descriptions,
appreciations, events, data), the linked character and the mentioning documents
(Location tables, see wiki/data-model.html#locations).
"""

from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, HTTPException

from lib.config import settings
from pipeline.location_manager import _LOC_COMPLETE_THRESHOLD as COMPLETE_THRESHOLD

router = APIRouter(tags=["locations"])


def _connect() -> sqlite3.Connection | None:
    if not settings.archive_db.exists():
        return None
    conn = sqlite3.connect(str(settings.archive_db))
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/locations")
async def location_list(search: str = "") -> dict:
    """Locations ordered most-complete first, plus the map/summary numbers.

    `search` filters name + aliases. Every row carries latitude/longitude so
    the page can plot the located ones without a second request.
    """
    empty = {
        "available": False,
        "total": 0,
        "complete": 0,
        "with_coordinates": 0,
        "threshold": COMPLETE_THRESHOLD,
        "locations": [],
    }
    conn = _connect()
    if conn is None:
        return empty

    try:
        params: tuple = ()
        where = ""
        if search.strip():
            where = "WHERE LOWER(name) LIKE ? OR LOWER(aliases) LIKE ?"
            needle = f"%{search.strip().lower()}%"
            params = (needle, needle)
        rows = conn.execute(
            f"""
            SELECT location_id, name, kind, completeness_score, mention_count,
                   needs_research, latitude, longitude
            FROM locations
            {where}
            ORDER BY completeness_score DESC, mention_count DESC, name ASC
            """,
            params,
        ).fetchall()
        locations = [dict(row) for row in rows]
        return {
            "available": True,
            "total": len(locations),
            "complete": sum(
                1 for loc in locations if loc["completeness_score"] >= COMPLETE_THRESHOLD
            ),
            "with_coordinates": sum(
                1
                for loc in locations
                if loc["latitude"] is not None and loc["longitude"] is not None
            ),
            "threshold": COMPLETE_THRESHOLD,
            "locations": locations,
        }
    except Exception:
        return empty
    finally:
        conn.close()


@router.get("/locations/{location_id}")
async def location_detail(location_id: str) -> dict:
    """One location: profile + linked character, per-kind facts, documents."""
    conn = _connect()
    if conn is None:
        raise HTTPException(status_code=404, detail="database not found")

    try:
        row = conn.execute(
            """
            SELECT l.*, c.name AS character_name
            FROM locations l
            LEFT JOIN characters c ON c.character_id = l.character_id
            WHERE l.location_id = ?
            """,
            (location_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="location not found")

        location = dict(row)
        location.setdefault("character_name", None)
        try:
            location["aliases"] = json.loads(location.get("aliases") or "[]")
        except json.JSONDecodeError:
            location["aliases"] = []
        if location.get("completeness_detail"):
            try:
                location["completeness_detail"] = json.loads(location["completeness_detail"])
            except json.JSONDecodeError:
                location["completeness_detail"] = None

        facts = [
            dict(fact)
            for fact in conn.execute(
                """
                SELECT date_iso, kind, detail, reported_by, doc_id
                FROM location_facts
                WHERE location_id = ?
                ORDER BY COALESCE(date_iso, '9999') ASC, id ASC
                """,
                (location_id,),
            ).fetchall()
        ]

        documents = [
            dict(doc)
            for doc in conn.execute(
                """
                SELECT m.doc_id, COALESCE(d.title, m.doc_id) AS title
                FROM location_mentions m
                LEFT JOIN documents d ON d.doc_id = m.doc_id
                WHERE m.location_id = ?
                ORDER BY m.created_at ASC
                """,
                (location_id,),
            ).fetchall()
        ]

        try:
            images = [
                dict(image)
                for image in conn.execute(
                    """
                    SELECT url, page_url, caption, source
                    FROM location_images
                    WHERE location_id = ?
                    ORDER BY id ASC
                    """,
                    (location_id,),
                ).fetchall()
            ]
        except sqlite3.OperationalError:
            images = []  # DB predating the gallery table

        return {
            "location": location,
            "facts": facts,
            "documents": documents,
            "images": images,
        }
    finally:
        conn.close()
