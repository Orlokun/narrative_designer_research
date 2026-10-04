"""SQLite archive statistics for the dashboard."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter

from lib.config import settings

router = APIRouter(tags=["database"])

_TABLES = ["documents", "characters", "locations", "coverage", "missions"]


@router.get("/database")
async def database_status() -> dict:
    """Return table row counts, file size and 10 most-recent documents."""
    db_path = settings.archive_db

    if not db_path.exists():
        return {
            "exists": False,
            "path": str(db_path),
            "size_mb": 0.0,
            "tables": {t: None for t in _TABLES},
            "recent_docs": [],
            "source_distribution": {},
        }

    size_mb = round(db_path.stat().st_size / (1024 * 1024), 3)
    tables: dict[str, int | None] = {}
    recent_docs: list[dict] = []
    source_dist: dict[str, int] = {}

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        for table in _TABLES:
            try:
                tables[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                tables[table] = None

        try:
            rows = conn.execute(
                "SELECT doc_id, title, date_iso, source_kind, source_id "
                "FROM documents ORDER BY ingested_at DESC LIMIT 10"
            ).fetchall()
            recent_docs = [dict(r) for r in rows]
        except Exception:
            pass

        try:
            rows = conn.execute(
                "SELECT source_kind, COUNT(*) as n FROM documents GROUP BY source_kind"
            ).fetchall()
            source_dist = {r["source_kind"]: r["n"] for r in rows}
        except Exception:
            pass

        conn.close()

    except Exception as exc:
        return {"exists": True, "path": str(db_path), "error": str(exc)}

    return {
        "exists": True,
        "path": str(db_path),
        "size_mb": size_mb,
        "tables": tables,
        "recent_docs": recent_docs,
        "source_distribution": source_dist,
    }
