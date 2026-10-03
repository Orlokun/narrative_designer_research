"""Per-agent detailed statistics for the individual admin pages."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter

from admin.routers.heatmap import CATEGORIES
from lib.config import settings
from lib.genres import GENRE_NAMES
from pipeline.verificator import _DEFAULT_MIN_SCORE

router = APIRouter(tags=["agents"])

_STATUSES = ("pending", "running", "done", "failed")
_SCORE_BUCKETS = [
    ("0.0-0.2", 0.0, 0.2),
    ("0.2-0.4", 0.2, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.01),
]


def _connect() -> sqlite3.Connection | None:
    if not settings.archive_db.exists():
        return None
    conn = sqlite3.connect(str(settings.archive_db))
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/agents/propositor")
async def propositor_stats() -> dict:
    """Mission queue detail: counts by status, by category, and top coverage gaps."""
    conn = _connect()
    if conn is None:
        return {"available": False}

    empty_status = {s: 0 for s in _STATUSES}

    try:
        # Status totals
        by_status = dict(empty_status)
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM missions GROUP BY status"
        ).fetchall():
            if row["status"] in by_status:
                by_status[row["status"]] = row["n"]

        # Per-category breakdown
        by_cat: dict[str, dict] = {}
        for row in conn.execute("""
            SELECT category, status, COUNT(*) AS n
            FROM missions
            GROUP BY category, status
            ORDER BY category
        """).fetchall():
            cat = row["category"]
            if cat not in by_cat:
                by_cat[cat] = {s: 0 for s in _STATUSES}
            if row["status"] in by_cat[cat]:
                by_cat[cat][row["status"]] = row["n"]

        by_category = [
            {"category": cat, **counts}
            for cat, counts in sorted(
                by_cat.items(),
                key=lambda x: x[1].get("failed", 0),
                reverse=True,
            )
        ]

        # Coverage gaps (cells with score < 1.0 or no entry)
        covered = conn.execute(
            "SELECT COUNT(*) FROM coverage WHERE coverage_score > 0"
        ).fetchone()[0]

        top_gaps = []
        try:
            rows = conn.execute("""
                SELECT m.category, m.month_iso, m.priority,
                       COALESCE(c.coverage_score, 0.0) AS coverage_score
                FROM missions m
                LEFT JOIN coverage c
                       ON c.category_id = m.category_id AND c.month_iso = m.month_iso
                WHERE COALESCE(c.coverage_score, 0.0) < 1.0
                  AND m.status = 'pending'
                ORDER BY m.priority DESC
                LIMIT 20
            """).fetchall()
            top_gaps = [dict(r) for r in rows]
        except Exception:
            pass

        return {
            "available": True,
            "missions_by_status": by_status,
            "missions_by_category": by_category,
            "top_gaps": top_gaps,
            "total_cells": 768,
            "covered_cells": covered,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    finally:
        conn.close()


def _category_name(category_id: int | None) -> str:
    if category_id is None or not 1 <= category_id <= len(CATEGORIES):
        return "—"
    return CATEGORIES[category_id - 1]


def _genre_name(genre_id: int | None) -> str:
    if genre_id is None or genre_id == 0:
        return "Sin clasificar"
    return GENRE_NAMES.get(genre_id, "—")


@router.get("/agents/mapper")
async def mapper_stats() -> dict:
    """Mapping detail: mapped/pending/context counts, genre & category
    distributions of mapped documents, and the most recent mappings."""
    conn = _connect()
    if conn is None:
        return {"available": False}

    try:
        mapped = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE mapped_category_id IS NOT NULL"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM documents "
            "WHERE verified_at IS NOT NULL AND mapped_category_id IS NULL "
            "  AND (doc_tag IS NULL OR doc_tag != 'context')"
        ).fetchone()[0]
        context = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE doc_tag = 'context'"
        ).fetchone()[0]

        by_genre = [
            {"genre_id": row[0], "genre": _genre_name(row[0]), "n": row[1]}
            for row in conn.execute("""
                SELECT COALESCE(mapped_genre_id, 0), COUNT(*) AS n
                FROM documents
                WHERE mapped_category_id IS NOT NULL
                GROUP BY COALESCE(mapped_genre_id, 0)
                ORDER BY n DESC
            """).fetchall()
        ]

        by_category = [
            {"category_id": row[0], "category": _category_name(row[0]), "n": row[1]}
            for row in conn.execute("""
                SELECT mapped_category_id, COUNT(*) AS n
                FROM documents
                WHERE mapped_category_id IS NOT NULL
                GROUP BY mapped_category_id
                ORDER BY n DESC
            """).fetchall()
        ]

        recent = [
            {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "category": _category_name(row["mapped_category_id"]),
                "genre": _genre_name(row["mapped_genre_id"]),
                "month_iso": row["mapped_month_iso"],
                "mapped_at": row["mapped_at"],
            }
            for row in conn.execute("""
                SELECT doc_id, title, mapped_category_id, mapped_genre_id,
                       mapped_month_iso, mapped_at
                FROM documents
                WHERE mapped_category_id IS NOT NULL
                ORDER BY mapped_at DESC
                LIMIT 15
            """).fetchall()
        ]

        return {
            "available": True,
            "mapped": mapped,
            "pending": pending,
            "context": context,
            "by_genre": by_genre,
            "by_category": by_category,
            "recent": recent,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    finally:
        conn.close()


@router.get("/agents/archivero")
async def archivero_stats() -> dict:
    """Document fetch stats: counts, source breakdown with avg quality, recent docs."""
    conn = _connect()
    if conn is None:
        return {"available": False}

    try:
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        verified = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE verified_at IS NOT NULL"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE quality_score IS NULL"
        ).fetchone()[0]

        last_row = conn.execute(
            "SELECT MAX(ingested_at) FROM documents"
        ).fetchone()
        last_ingested_at = last_row[0] if last_row else None

        # By source with avg quality
        by_source: list[dict] = []
        for row in conn.execute("""
            SELECT source_id,
                   COUNT(*) AS doc_count,
                   ROUND(AVG(COALESCE(quality_score, 0)), 3) AS avg_quality,
                   SUM(CASE WHEN verified_at IS NOT NULL THEN 1 ELSE 0 END) AS verified_count
            FROM documents
            GROUP BY source_id
            ORDER BY doc_count DESC
        """).fetchall():
            # Shorten "archivero/openalex" → "openalex"
            label = row["source_id"].split("/")[-1] if row["source_id"] else "unknown"
            by_source.append({
                "source":         row["source_id"],
                "label":          label,
                "doc_count":      row["doc_count"],
                "avg_quality":    row["avg_quality"],
                "verified_count": row["verified_count"],
            })

        # By kind
        by_kind: dict[str, int] = {}
        for row in conn.execute(
            "SELECT source_kind, COUNT(*) AS n FROM documents GROUP BY source_kind"
        ).fetchall():
            by_kind[row["source_kind"]] = row["n"]

        # Recent docs
        recent = []
        for row in conn.execute("""
            SELECT doc_id, title, source_id, source_kind,
                   quality_score, ingested_at, verified_at
            FROM documents
            ORDER BY ingested_at DESC
            LIMIT 20
        """).fetchall():
            recent.append({
                "doc_id":        row["doc_id"],
                "title":         (row["title"] or "")[:80],
                "source_id":     row["source_id"],
                "source_kind":   row["source_kind"],
                "quality_score": row["quality_score"],
                "ingested_at":   row["ingested_at"],
                "verified":      row["verified_at"] is not None,
            })

        return {
            "available":       True,
            "total_documents": total,
            "verified":        verified,
            "pending":         pending,
            "last_ingested_at": last_ingested_at,
            "by_source":       by_source,
            "by_kind":         by_kind,
            "recent_documents": recent,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    finally:
        conn.close()


@router.get("/agents/verificator")
async def verificator_stats() -> dict:
    """Verification stats: score distribution, category breakdown."""
    conn = _connect()
    if conn is None:
        return {"available": False}

    try:
        total_scored = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE quality_score IS NOT NULL"
        ).fetchone()[0]
        verified = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE verified_at IS NOT NULL"
        ).fetchone()[0]
        rejected = total_scored - verified
        pending = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE quality_score IS NULL"
        ).fetchone()[0]

        avg_row = conn.execute(
            "SELECT ROUND(AVG(quality_score), 3) FROM documents WHERE quality_score IS NOT NULL"
        ).fetchone()
        avg_score = avg_row[0] if avg_row else 0.0

        # Score distribution
        distribution: dict[str, int] = {}
        for label, lo, hi in _SCORE_BUCKETS:
            row = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE quality_score >= ? AND quality_score < ?",
                (lo, hi),
            ).fetchone()
            distribution[label] = row[0] if row else 0

        # Verified by category
        by_category: list[dict] = []
        try:
            rows = conn.execute("""
                SELECT m.category,
                       SUM(CASE WHEN d.verified_at IS NOT NULL THEN 1 ELSE 0 END) AS verified,
                       SUM(CASE WHEN d.quality_score IS NOT NULL
                                 AND d.verified_at IS NULL THEN 1 ELSE 0 END) AS rejected,
                       ROUND(AVG(d.quality_score), 3) AS avg_score
                FROM documents d
                LEFT JOIN missions m ON m.mission_id = json_extract(d.provenance, '$.mission_id')
                WHERE d.quality_score IS NOT NULL AND m.category IS NOT NULL
                GROUP BY m.category
                ORDER BY verified DESC
            """).fetchall()
            by_category = [dict(r) for r in rows]
        except Exception:
            pass

        return {
            "available":          True,
            "total_scored":       total_scored,
            "verified":           verified,
            "rejected":           rejected,
            "pending":            pending,
            "avg_score":          avg_score,
            "threshold":          _DEFAULT_MIN_SCORE,
            "score_distribution": distribution,
            "by_category":        by_category,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    finally:
        conn.close()
