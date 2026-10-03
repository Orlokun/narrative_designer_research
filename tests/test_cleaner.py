"""Tests for the document cleaner (removes rejected documents from the DB)."""

import sqlite3

import pytest

from pipeline.verificator import _clean_rejected, _migrate_documents_schema


@pytest.fixture
def db():
    """In-memory SQLite database with documents table and schema migration."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Create documents table (mimics the archivero schema)
    conn.execute(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            lang TEXT NOT NULL DEFAULT 'es',
            date_iso TEXT,
            authors TEXT NOT NULL DEFAULT '[]',
            places TEXT NOT NULL DEFAULT '[]',
            source_kind TEXT NOT NULL,
            source_id TEXT NOT NULL,
            provenance TEXT NOT NULL DEFAULT '{}',
            rights TEXT,
            sha256 TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        )
        """
    )

    # Apply verificator schema migration to add quality_score, verified_at, is_complete
    _migrate_documents_schema(conn)

    yield conn
    conn.close()


def test_clean_rejected_deletes_scored_not_verified(db):
    """Documents with quality_score set but verified_at NULL should be deleted."""
    # Insert a rejected document (scored at 0.2, below min_score 0.3)
    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at,
            quality_score, verified_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "doc001",
            "Test Title",
            "This is a test document",
            "ACADEMIC",
            "source1",
            "sha256_abc123",
            "2026-04-25T10:00:00Z",
            0.2,  # scored but below threshold
            None,  # not verified
        ),
    )
    db.commit()

    deleted = _clean_rejected(db)
    assert deleted == 1

    # Verify it's gone
    remaining = db.execute(
        "SELECT COUNT(*) FROM documents WHERE doc_id = 'doc001'"
    ).fetchone()[0]
    assert remaining == 0


def test_clean_rejected_keeps_unscored(db):
    """Documents with quality_score NULL (unscored) should be kept."""
    # Insert an unscored document
    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "doc002",
            "Unscored",
            "This doc has not been scored yet",
            "PRESS",
            "source2",
            "sha256_def456",
            "2026-04-25T11:00:00Z",
        ),
    )
    db.commit()

    deleted = _clean_rejected(db)
    assert deleted == 0

    # Verify it still exists
    remaining = db.execute(
        "SELECT COUNT(*) FROM documents WHERE doc_id = 'doc002'"
    ).fetchone()[0]
    assert remaining == 1


def test_clean_rejected_keeps_verified(db):
    """Documents with quality_score set AND verified_at set should be kept."""
    # Insert a verified document
    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at,
            quality_score, verified_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "doc003",
            "Verified Doc",
            "This document passed verification",
            "GOVERNMENT",
            "source3",
            "sha256_ghi789",
            "2026-04-25T12:00:00Z",
            0.8,  # scored and approved
            "2026-04-25T13:00:00Z",  # verified at this timestamp
        ),
    )
    db.commit()

    deleted = _clean_rejected(db)
    assert deleted == 0

    # Verify it still exists
    remaining = db.execute(
        "SELECT COUNT(*) FROM documents WHERE doc_id = 'doc003'"
    ).fetchone()[0]
    assert remaining == 1


def test_clean_rejected_returns_count(db):
    """_clean_rejected should return the exact number of rows deleted."""
    # Insert three rejected documents
    for i in range(3):
        db.execute(
            """
            INSERT INTO documents (
                doc_id, title, text, source_kind, source_id, sha256, ingested_at,
                quality_score, verified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"doc{i:03d}",
                f"Rejected {i}",
                f"Text {i}",
                "ACADEMIC",
                f"source{i}",
                f"sha256_{i}",
                "2026-04-25T10:00:00Z",
                0.1,  # all rejected (below threshold)
                None,
            ),
        )
    db.commit()

    # Also insert one verified doc (should not be counted)
    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at,
            quality_score, verified_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "doc_verified",
            "Verified",
            "This one is fine",
            "PRESS",
            "source_v",
            "sha256_verified",
            "2026-04-25T14:00:00Z",
            0.7,
            "2026-04-25T14:30:00Z",
        ),
    )
    db.commit()

    deleted = _clean_rejected(db)
    assert deleted == 3  # Only the rejected ones, not the verified one


def test_clean_rejected_mixed_scenario(db):
    """Comprehensive test with unscored, rejected, and verified documents."""
    # Unscored
    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "unscored1",
            "Pending",
            "Not yet processed",
            "ACADEMIC",
            "src_u",
            "sha_u",
            "2026-04-25T10:00:00Z",
        ),
    )

    # Rejected
    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at,
            quality_score, verified_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "rejected1",
            "Bad doc",
            "Too short",
            "PRESS",
            "src_r",
            "sha_r",
            "2026-04-25T11:00:00Z",
            0.0,
            None,
        ),
    )

    # Verified
    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at,
            quality_score, verified_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "verified1",
            "Good doc",
            "Full length document",
            "GOVERNMENT",
            "src_v",
            "sha_v",
            "2026-04-25T12:00:00Z",
            0.9,
            "2026-04-25T12:30:00Z",
        ),
    )
    db.commit()

    deleted = _clean_rejected(db)
    assert deleted == 1  # Only rejected1 should be deleted

    # Verify counts
    total = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert total == 2  # unscored1 + verified1

    unscored_count = db.execute(
        "SELECT COUNT(*) FROM documents WHERE quality_score IS NULL"
    ).fetchone()[0]
    assert unscored_count == 1

    verified_count = db.execute(
        "SELECT COUNT(*) FROM documents WHERE verified_at IS NOT NULL"
    ).fetchone()[0]
    assert verified_count == 1
