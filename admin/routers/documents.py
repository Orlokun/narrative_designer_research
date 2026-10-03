"""Document browser — paginated access to verified documents for the carousel."""

from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, HTTPException, Query

from admin.routers.heatmap import CATEGORIES
from lib.config import settings
from lib.doc_flags import FLAGS, flag_by_slug

router = APIRouter(tags=["documents"])

# 1-indexed map: id 1..16 → display name
_CATEGORY_BY_ID: dict[int, str] = {i + 1: name for i, name in enumerate(CATEGORIES)}
_ID_BY_CATEGORY: dict[str, int] = {name: i + 1 for i, name in enumerate(CATEGORIES)}


def _db_conn() -> sqlite3.Connection | None:
    if not settings.archive_db.exists():
        return None
    conn = sqlite3.connect(str(settings.archive_db))
    conn.row_factory = sqlite3.Row
    return conn


def _has_doc_flags(conn: sqlite3.Connection) -> bool:
    """True if the doc_flags table exists (older DBs predate the Curator)."""
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='doc_flags'"
        ).fetchone()
        is not None
    )


def _flags_for(conn: sqlite3.Connection, doc_id: str) -> list[dict]:
    """Return a document's utility flags (slug, name, confidence, rationale), best first."""
    if not _has_doc_flags(conn):
        return []
    rows = conn.execute(
        "SELECT flag, confidence, rationale FROM doc_flags "
        "WHERE doc_id = ? ORDER BY confidence DESC, flag ASC",
        (doc_id,),
    ).fetchall()
    out: list[dict] = []
    for row in rows:
        meta = flag_by_slug(row["flag"])
        out.append(
            {
                "slug": row["flag"],
                "name": meta.name if meta else row["flag"],
                "kind": meta.kind if meta else "good-for",
                "confidence": row["confidence"],
                "rationale": row["rationale"],
            }
        )
    return out


def _category_names(raw: str | None, primary_id: int | None) -> list[str]:
    """Decode mapped_categories JSON list into display names.

    Falls back to [primary_id] when the JSON column is empty or malformed.
    """
    ids: list[int] = []
    if raw:
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, list):
                ids = [int(x) for x in decoded if isinstance(x, int)]
        except (json.JSONDecodeError, TypeError, ValueError):
            ids = []
    if not ids and primary_id is not None:
        ids = [int(primary_id)]
    return [_CATEGORY_BY_ID[i] for i in ids if i in _CATEGORY_BY_ID]


@router.get("/documents")
async def get_documents(
    page: int = Query(default=1, ge=1),
    verified_only: bool = Query(default=True),
    category: str | None = Query(default=None),
    month: str | None = Query(default=None),
    flag: str | None = Query(default=None),
) -> dict:
    """
    Return one document at position `page` (1-based) plus total count.

    Filtering uses the LLM-assigned mapping (`mapped_category_id`,
    `mapped_month_iso`) and falls back to the parent mission's category/month
    for documents that have not yet been mapped. The category filter matches
    against the **primary** mapped category (the first id in mapped_categories).
    The `flag` filter restricts to documents carrying that utility flag (Ag-9).
    """
    conn = _db_conn()
    if conn is None:
        return {"total": 0, "page": 1, "doc": None}

    try:
        where_parts = []
        params: list = []

        if verified_only:
            where_parts.append("d.verified_at IS NOT NULL")
        if flag:
            if not _has_doc_flags(conn):
                return {"total": 0, "page": 1, "doc": None}
            where_parts.append(
                "EXISTS (SELECT 1 FROM doc_flags f WHERE f.doc_id = d.doc_id AND f.flag = ?)"
            )
            params.append(flag)
        if category:
            cid = _ID_BY_CATEGORY.get(category)
            if cid is not None:
                # Match either the LLM mapping OR the mission's category for unmapped docs
                where_parts.append("(COALESCE(d.mapped_category_id, m.category_id) = ?)")
                params.append(cid)
            else:
                where_parts.append("m.category = ?")
                params.append(category)
        if month:
            where_parts.append("(COALESCE(d.mapped_month_iso, m.month_iso) = ?)")
            params.append(month)

        where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        total_row = conn.execute(
            f"""
            SELECT COUNT(*) FROM documents d
            LEFT JOIN missions m ON m.mission_id = json_extract(d.provenance, '$.mission_id')
            {where}
        """,
            params,
        ).fetchone()
        total = total_row[0] if total_row else 0

        if total == 0:
            return {"total": 0, "page": 1, "doc": None}

        page = max(1, min(page, total))

        row = conn.execute(
            f"""
            SELECT d.doc_id, d.title, d.text, d.quality_score,
                   d.source_id, d.source_kind, d.ingested_at, d.verified_at,
                   d.is_complete,
                   d.mapped_category_id, d.mapped_categories, d.mapped_month_iso,
                   d.doc_tag,
                   m.category AS mission_category, m.month_iso AS mission_month
            FROM documents d
            LEFT JOIN missions m ON m.mission_id = json_extract(d.provenance, '$.mission_id')
            {where}
            ORDER BY d.quality_score DESC, d.ingested_at DESC, d.doc_id ASC
            LIMIT 1 OFFSET ?
        """,
            params + [page - 1],
        ).fetchone()

        if row is None:
            return {"total": total, "page": page, "doc": None}

        return {
            "total": total,
            "page": page,
            "doc": _doc_payload(row, _flags_for(conn, row["doc_id"])),
        }
    except Exception as exc:
        return {"total": 0, "page": 1, "doc": None, "error": str(exc)}
    finally:
        conn.close()


def _doc_payload(row: sqlite3.Row, flags: list[dict] | None = None) -> dict:
    """Build the reader's document object (mapped > mission fallback for cell info)."""
    primary_id = row["mapped_category_id"]
    categories = _category_names(row["mapped_categories"], primary_id)
    if not categories:
        categories = [row["mission_category"]] if row["mission_category"] else []
    month_iso = row["mapped_month_iso"] or row["mission_month"] or "—"
    return {
        "doc_id": row["doc_id"],
        "title": row["title"],
        "text": row["text"],
        "quality_score": row["quality_score"],
        "source_id": row["source_id"],
        "source_kind": row["source_kind"],
        "category": categories[0] if categories else "—",
        "categories": categories,
        "mapped": primary_id is not None,
        "doc_tag": row["doc_tag"],
        "flags": flags or [],
        "month_iso": month_iso,
        "ingested_at": row["ingested_at"],
        "verified": row["verified_at"] is not None,
        "is_complete": bool(row["is_complete"]) if row["is_complete"] is not None else True,
    }


@router.get("/documents/by-id/{doc_id}")
async def get_document_by_id(doc_id: str) -> dict:
    """One document by id, plus its 1-based position in the UNFILTERED carousel
    order — every "doc mentioned" link across the admin resolves here, and the
    reader lands on that position with prev/next still working.
    """
    conn = _db_conn()
    if conn is None:
        raise HTTPException(status_code=404, detail="database not found")

    try:
        row = conn.execute(
            """
            SELECT d.doc_id, d.title, d.text, d.quality_score,
                   d.source_id, d.source_kind, d.ingested_at, d.verified_at,
                   d.is_complete,
                   d.mapped_category_id, d.mapped_categories, d.mapped_month_iso,
                   d.doc_tag,
                   m.category AS mission_category, m.month_iso AS mission_month
            FROM documents d
            LEFT JOIN missions m ON m.mission_id = json_extract(d.provenance, '$.mission_id')
            WHERE d.doc_id = ?
        """,
            (doc_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="document not found")

        position_row = conn.execute(
            """
            SELECT pos, total FROM (
                SELECT doc_id,
                       ROW_NUMBER() OVER (
                           ORDER BY quality_score DESC, ingested_at DESC, doc_id ASC
                       ) AS pos,
                       COUNT(*) OVER () AS total
                FROM documents
            ) WHERE doc_id = ?
        """,
            (doc_id,),
        ).fetchone()

        return {
            "doc": _doc_payload(row, _flags_for(conn, row["doc_id"])),
            "position": position_row["pos"],
            "total": position_row["total"],
        }
    finally:
        conn.close()


@router.get("/documents/categories")
async def document_categories() -> list[str]:
    """Return the canonical 16 categories for the filter dropdown."""
    return list(CATEGORIES)


@router.get("/documents/flags")
async def document_flags() -> list[dict]:
    """Return the utility-flag vocabulary + how many documents carry each (Ag-9).

    The full registry is always returned so the reader can render the filter even
    before curation; `count` is 0 for flags nothing carries yet (or if the Curator
    has never run and the table is absent).
    """
    counts: dict[str, int] = {}
    conn = _db_conn()
    if conn is not None:
        try:
            if _has_doc_flags(conn):
                counts = {
                    row["flag"]: row["n"]
                    for row in conn.execute(
                        "SELECT flag, COUNT(*) AS n FROM doc_flags GROUP BY flag"
                    ).fetchall()
                }
        finally:
            conn.close()
    return [
        {
            "slug": f.slug,
            "name": f.name,
            "kind": f.kind,
            "description": f.description,
            "count": counts.get(f.slug, 0),
        }
        for f in FLAGS
    ]
