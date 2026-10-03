"""
Shared 3D coverage table — theme × genre × month, the matrix's persistence layer.

Both the Archivero (mission-level counts) and the Mapper (document-level counts)
write coverage scores; this module owns the schema and the upsert so the two
agents can never drift. The Propositor and the Admin heatmap read it.

A cell is (category_id 1-16, genre_id 0-13, month_iso). genre_id 0
(``GENRE_UNCLASSIFIED``) buckets scores recorded before the genre axis existed
or for documents the classifier could not type.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from lib.genres import GENRE_UNCLASSIFIED

# Verified documents per (theme, genre, month) cell that count as full coverage.
FULL_COVERAGE_DOCS: int = 5

_CREATE_COVERAGE = """
    CREATE TABLE IF NOT EXISTS coverage (
        category_id    INTEGER NOT NULL,
        genre_id       INTEGER NOT NULL DEFAULT 0,
        month_iso      TEXT    NOT NULL,
        coverage_score REAL    NOT NULL DEFAULT 0.0,
        updated_at     TEXT    NOT NULL,
        PRIMARY KEY (category_id, genre_id, month_iso)
    )
"""


def ensure_coverage_table(conn: sqlite3.Connection) -> None:
    """Create the 3D coverage table; migrate a legacy 2D table if found.

    Legacy rows (theme × month, no genre) are preserved under genre_id 0 so no
    coverage history is lost. Idempotent.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(coverage)").fetchall()}
    if columns and "genre_id" not in columns:
        conn.execute("ALTER TABLE coverage RENAME TO coverage_legacy_2d")
        conn.execute(_CREATE_COVERAGE)
        conn.execute(
            """
            INSERT INTO coverage (category_id, genre_id, month_iso, coverage_score, updated_at)
            SELECT category_id, ?, month_iso, coverage_score, updated_at
            FROM coverage_legacy_2d
            """,
            (GENRE_UNCLASSIFIED,),
        )
        conn.execute("DROP TABLE coverage_legacy_2d")
    else:
        conn.execute(_CREATE_COVERAGE)
    conn.commit()


def upsert_coverage_score(
    conn: sqlite3.Connection,
    *,
    category_id: int,
    month_iso: str,
    docs_found: int,
    genre_id: int = GENRE_UNCLASSIFIED,
) -> None:
    """Write the score for one (theme, genre, month) cell.

    Score = min(1.0, docs_found / FULL_COVERAGE_DOCS). Does not commit — callers
    batch their own transactions.
    """
    score = min(1.0, round(docs_found / FULL_COVERAGE_DOCS, 4))
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO coverage (category_id, genre_id, month_iso, coverage_score, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(category_id, genre_id, month_iso) DO UPDATE SET
            coverage_score = excluded.coverage_score,
            updated_at     = excluded.updated_at
        """,
        (category_id, genre_id, month_iso, score, now),
    )
