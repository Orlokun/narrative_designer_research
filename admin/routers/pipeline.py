"""Pipeline agent status — mission queue breakdown and Archivero stats."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter

from lib.config import settings

router = APIRouter(tags=["pipeline"])

_STATUS_KEYS = ("pending", "running", "done", "failed")


@router.get("/pipeline")
async def pipeline_status() -> dict:
    """
    Returns mission queue counts by status plus Archivero document stats.
    Safe to call even before the database exists.
    """
    db_path = settings.archive_db

    empty = {
        "missions": {k: 0 for k in _STATUS_KEYS} | {"total": 0},
        "archivero": {"documents_fetched": 0, "last_run": None},
    }

    if not db_path.exists():
        return empty

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        missions: dict[str, int] = {k: 0 for k in _STATUS_KEYS}
        try:
            for row in conn.execute(
                "SELECT status, COUNT(*) AS n FROM missions GROUP BY status"
            ).fetchall():
                if row["status"] in missions:
                    missions[row["status"]] = row["n"]
        except Exception:
            pass
        missions["total"] = sum(missions[k] for k in _STATUS_KEYS)

        docs_fetched: int = 0
        last_run: str | None = None
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n, MAX(ingested_at) AS last "
                "FROM documents WHERE source_id LIKE 'archivero/%'"
            ).fetchone()
            if row:
                docs_fetched = row["n"] or 0
                last_run     = row["last"]
        except Exception:
            pass

        conn.close()

    except Exception:
        return empty

    return {
        "missions": missions,
        "archivero": {
            "documents_fetched": docs_fetched,
            "last_run": last_run,
        },
    }
