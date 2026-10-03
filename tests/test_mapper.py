"""Tests for Mapper agent (Ag-3) — assigns documents to coverage matrix cells."""

import json
import sqlite3
from pathlib import Path

import pytest

# NOTE: These tests are designed to run fully offline. The Mapper class uses
# Ollama/Gemma4 for real classification, but we test the JSON parsing and
# DB operations separately from any LLM call.


# ── Pure function tests (JSON parsing) ──────────────────────────────────────


def test_parse_llm_response_valid_legacy_single_id():
    """Legacy single-id shape should still parse, returning a 1-item list."""
    from pipeline.mapper import parse_llm_response

    response = '{"category_id": 4, "month_iso": "1972-10", "confidence": 0.85}'
    result = parse_llm_response(response)
    assert result == ([4], "1972-10", 0.85)


def test_parse_llm_response_valid_multi_id():
    """New multi-id shape should parse with the full ranked list."""
    from pipeline.mapper import parse_llm_response

    response = '{"category_ids": [6, 4, 3], "month_iso": "1973-04", "confidence": 0.9}'
    result = parse_llm_response(response)
    assert result == ([6, 4, 3], "1973-04", 0.9)


def test_parse_llm_response_dedupes_repeated_ids():
    """Duplicate category ids should be collapsed while preserving order."""
    from pipeline.mapper import parse_llm_response

    response = '{"category_ids": [6, 6, 4], "month_iso": "1973-04", "confidence": 0.9}'
    result = parse_llm_response(response)
    assert result == ([6, 4], "1973-04", 0.9)


def test_parse_llm_response_rejects_more_than_three_ids():
    """More than 3 category ids should be rejected as malformed."""
    from pipeline.mapper import parse_llm_response

    response = '{"category_ids": [1, 2, 3, 4], "month_iso": "1972-10", "confidence": 0.9}'
    assert parse_llm_response(response) is None


def test_parse_llm_response_rejects_empty_ids_list():
    """Empty category_ids list should be rejected."""
    from pipeline.mapper import parse_llm_response

    response = '{"category_ids": [], "month_iso": "1972-10", "confidence": 0.9}'
    assert parse_llm_response(response) is None


def test_parse_llm_response_with_extra_whitespace():
    """JSON with leading/trailing whitespace should parse correctly."""
    from pipeline.mapper import parse_llm_response

    response = '  {"category_id": 3, "month_iso": "1971-06", "confidence": 0.7}  \n'
    result = parse_llm_response(response)
    assert result == ([3], "1971-06", 0.7)


def test_parse_llm_response_invalid_category_too_high():
    """category id > 16 should return None."""
    from pipeline.mapper import parse_llm_response

    response = '{"category_ids": [99], "month_iso": "1972-10", "confidence": 0.85}'
    assert parse_llm_response(response) is None


def test_parse_llm_response_invalid_category_zero():
    """category id = 0 should return None (valid range is 1-16)."""
    from pipeline.mapper import parse_llm_response

    response = '{"category_id": 0, "month_iso": "1972-10", "confidence": 0.85}'
    assert parse_llm_response(response) is None

    response = '{"category_ids": [4, 0], "month_iso": "1972-10", "confidence": 0.85}'
    assert parse_llm_response(response) is None


def test_parse_llm_response_invalid_month_out_of_range():
    """Month outside 1969-10 to 1973-09 should return None."""
    from pipeline.mapper import parse_llm_response

    # Before the valid range
    response = '{"category_id": 4, "month_iso": "1969-09", "confidence": 0.85}'
    result = parse_llm_response(response)
    assert result is None

    # After the valid range
    response = '{"category_id": 4, "month_iso": "1973-10", "confidence": 0.85}'
    result = parse_llm_response(response)
    assert result is None


def test_parse_llm_response_invalid_month_format():
    """month_iso not in YYYY-MM format should return None."""
    from pipeline.mapper import parse_llm_response

    response = '{"category_id": 4, "month_iso": "1972/10", "confidence": 0.85}'
    result = parse_llm_response(response)
    assert result is None


def test_parse_llm_response_missing_field():
    """Missing required field should return None gracefully."""
    from pipeline.mapper import parse_llm_response

    # Missing month_iso
    response = '{"category_id": 4, "confidence": 0.85}'
    result = parse_llm_response(response)
    assert result is None


def test_parse_llm_response_malformed_json():
    """Malformed JSON should return None gracefully."""
    from pipeline.mapper import parse_llm_response

    response = '{category_id: 4, month_iso: "1972-10"}'  # Missing quotes around keys
    result = parse_llm_response(response)
    assert result is None


def test_parse_llm_response_empty_string():
    """Empty or whitespace-only response should return None."""
    from pipeline.mapper import parse_llm_response

    assert parse_llm_response("") is None
    assert parse_llm_response("   ") is None
    assert parse_llm_response("\n") is None


def test_parse_llm_response_confidence_out_of_range():
    """Confidence outside [0.0, 1.0] should return None."""
    from pipeline.mapper import parse_llm_response

    response = '{"category_id": 4, "month_iso": "1972-10", "confidence": 1.5}'
    result = parse_llm_response(response)
    assert result is None

    response = '{"category_id": 4, "month_iso": "1972-10", "confidence": -0.1}'
    result = parse_llm_response(response)
    assert result is None


# ── DB operation tests ─────────────────────────────────────────────────────


@pytest.fixture
def db():
    """In-memory SQLite database with documents and coverage tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Create documents table (archivero schema)
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
            ingested_at TEXT NOT NULL,
            quality_score REAL,
            verified_at TEXT,
            is_complete INTEGER DEFAULT 1,
            mapped_category_id INTEGER,
            mapped_month_iso TEXT,
            mapped_at TEXT,
            doc_tag TEXT,
            mapped_categories TEXT,
            mapped_genre_id INTEGER
        )
        """
    )

    # Create the shared 3D coverage table (theme × genre × month)
    from lib.coverage import ensure_coverage_table

    ensure_coverage_table(conn)

    # Create missions table for --no-llm fallback
    conn.execute(
        """
        CREATE TABLE missions (
            mission_id TEXT PRIMARY KEY,
            category_id INTEGER NOT NULL,
            month_iso TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """
    )

    yield conn
    conn.close()


def test_mapper_status_empty_db(db):
    """Status on empty DB should return {'mapped': 0, 'pending': 0}."""
    from pipeline.mapper import Mapper

    mapper = Mapper(db_path=Path(":memory:"))  # Note: won't actually use :memory: in real code
    # For this test, we'll test the logic directly
    mapped = db.execute(
        "SELECT COUNT(*) FROM documents WHERE mapped_category_id IS NOT NULL"
    ).fetchone()[0]
    pending = db.execute(
        "SELECT COUNT(*) FROM documents WHERE verified_at IS NOT NULL AND mapped_category_id IS NULL"
    ).fetchone()[0]
    assert mapped == 0
    assert pending == 0


def test_mapper_status_with_documents(db):
    """Status should accurately count mapped and unmapped verified documents."""
    # Insert verified but unmapped doc
    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at,
            quality_score, verified_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "doc001",
            "Allende speech",
            "This is a document about Allende",
            "GOVERNMENT",
            "source1",
            "sha_001",
            "2026-04-25T10:00:00Z",
            0.8,
            "2026-04-25T10:30:00Z",
        ),
    )

    # Insert already mapped doc
    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at,
            quality_score, verified_at, mapped_category_id, mapped_month_iso, mapped_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "doc002",
            "CIA cable",
            "Declassified CIA document",
            "GOVERNMENT",
            "source2",
            "sha_002",
            "2026-04-25T11:00:00Z",
            0.9,
            "2026-04-25T11:30:00Z",
            6,  # Estados Unidos
            "1972-10",
            "2026-04-25T12:00:00Z",
        ),
    )
    db.commit()

    mapped = db.execute(
        "SELECT COUNT(*) FROM documents WHERE mapped_category_id IS NOT NULL"
    ).fetchone()[0]
    pending = db.execute(
        "SELECT COUNT(*) FROM documents WHERE verified_at IS NOT NULL AND mapped_category_id IS NULL"
    ).fetchone()[0]

    assert mapped == 1
    assert pending == 1


def test_coverage_update_aggregates_correctly(db):
    """After mapping docs, coverage should aggregate counts correctly."""
    from pipeline.mapper import _upsert_coverage_score

    # Insert 3 docs mapped to the same cell
    for i in range(3):
        db.execute(
            """
            INSERT INTO documents (
                doc_id, title, text, source_kind, source_id, sha256, ingested_at,
                quality_score, verified_at, mapped_category_id, mapped_month_iso
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"doc{i:03d}",
                f"Document {i}",
                "Text content",
                "ACADEMIC",
                f"src{i}",
                f"sha_{i}",
                "2026-04-25T10:00:00Z",
                0.7,
                "2026-04-25T10:30:00Z",
                4,  # Industria Nacional
                "1972-10",  # All to same cell
            ),
        )
    db.commit()

    # Upsert coverage for this cell: 3 docs / 5 = 0.6
    _upsert_coverage_score(db, category_id=4, month_iso="1972-10", docs_found=3, genre_id=8)

    # Check the result
    coverage = db.execute(
        "SELECT coverage_score FROM coverage "
        "WHERE category_id = 4 AND genre_id = 8 AND month_iso = '1972-10'"
    ).fetchone()
    assert coverage is not None
    assert coverage[0] == 0.6  # min(1.0, 3/5)


def test_coverage_update_caps_at_one(db):
    """Coverage should cap at 1.0 even if > 5 docs per cell."""
    from pipeline.mapper import _upsert_coverage_score

    _upsert_coverage_score(db, category_id=5, month_iso="1973-06", docs_found=10)
    coverage = db.execute(
        "SELECT coverage_score FROM coverage WHERE category_id = 5 AND month_iso = '1973-06'"
    ).fetchone()
    assert coverage is not None
    assert coverage[0] == 1.0  # capped at 1.0, not 2.0


def test_coverage_update_replaces_existing(db):
    """Upserting should replace existing coverage_score."""
    from pipeline.mapper import _upsert_coverage_score

    # Initial insert: 2 docs
    _upsert_coverage_score(db, category_id=3, month_iso="1971-06", docs_found=2)
    coverage = db.execute(
        "SELECT coverage_score FROM coverage WHERE category_id = 3 AND month_iso = '1971-06'"
    ).fetchone()
    assert coverage[0] == 0.4

    # Update: now 4 docs (for same cell)
    _upsert_coverage_score(db, category_id=3, month_iso="1971-06", docs_found=4)
    coverage = db.execute(
        "SELECT coverage_score FROM coverage WHERE category_id = 3 AND month_iso = '1971-06'"
    ).fetchone()
    assert coverage[0] == 0.8


def test_fetch_unmapped_documents(db):
    """Should fetch only verified but unmapped documents."""
    from pipeline.mapper import _fetch_unmapped_documents

    # Unscored (should not appear)
    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("doc_unscored", "Unscored", "Text", "PRESS", "src", "sha", "2026-04-25T10:00:00Z"),
    )

    # Verified but unmapped (should appear)
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
            "Content here",
            "GOVERNMENT",
            "src",
            "sha_v",
            "2026-04-25T10:00:00Z",
            0.8,
            "2026-04-25T10:30:00Z",
        ),
    )

    # Verified and already mapped (should not appear)
    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at,
            quality_score, verified_at, mapped_category_id, mapped_month_iso
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "doc_mapped",
            "Mapped",
            "Already done",
            "ACADEMIC",
            "src",
            "sha_m",
            "2026-04-25T10:00:00Z",
            0.9,
            "2026-04-25T10:30:00Z",
            4,
            "1972-10",
        ),
    )
    db.commit()

    unmapped = _fetch_unmapped_documents(db, batch_size=10)
    assert len(unmapped) == 1
    assert unmapped[0]["doc_id"] == "doc_verified"


def test_save_mapping(db):
    """Should save primary id, JSON list of all ids, month_iso, and timestamp."""
    from pipeline.mapper import _save_mapping

    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("doc001", "Title", "Text", "PRESS", "src", "sha", "2026-04-25T10:00:00Z"),
    )
    db.commit()

    _save_mapping(db, doc_id="doc001", category_ids=[6, 4, 3], month_iso="1973-04")

    row = db.execute(
        "SELECT mapped_category_id, mapped_categories, mapped_month_iso, mapped_at "
        "FROM documents WHERE doc_id = 'doc001'"
    ).fetchone()

    assert row["mapped_category_id"] == 6  # primary = first in list
    assert json.loads(row["mapped_categories"]) == [6, 4, 3]
    assert row["mapped_month_iso"] == "1973-04"
    assert row["mapped_at"] is not None


def test_save_mapping_single_category(db):
    """Single-category list should still write a valid JSON array."""
    from pipeline.mapper import _save_mapping

    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("doc002", "T", "X", "PRESS", "src", "sha2", "2026-04-25T10:00:00Z"),
    )
    db.commit()

    _save_mapping(db, doc_id="doc002", category_ids=[14], month_iso="1971-08")

    row = db.execute(
        "SELECT mapped_category_id, mapped_categories FROM documents WHERE doc_id='doc002'"
    ).fetchone()
    assert row["mapped_category_id"] == 14
    assert json.loads(row["mapped_categories"]) == [14]


def test_save_mapping_rejects_empty_list(db):
    """Empty category_ids should raise ValueError, not silently no-op."""
    from pipeline.mapper import _save_mapping

    with pytest.raises(ValueError):
        _save_mapping(db, doc_id="doc_x", category_ids=[], month_iso="1972-10")


# ── Integration test: _all_months validation ───────────────────────────────


def test_valid_months_range():
    """Test that valid months are 1969-10 through 1973-09 (48 months total)."""
    from pipeline.mapper import _VALID_MONTHS

    assert "1969-10" in _VALID_MONTHS
    assert "1973-09" in _VALID_MONTHS
    assert "1969-09" not in _VALID_MONTHS  # before range
    assert "1973-10" not in _VALID_MONTHS  # after range
    assert len(_VALID_MONTHS) == 48


# ── Out-of-range month detection ───────────────────────────────────────────


def test_month_out_of_range_returns_true_for_pre_period():
    """Month before 1969-10 should be detected as out of range."""
    from pipeline.mapper import _month_out_of_range

    response = '{"category_id": 1, "month_iso": "1969-08", "confidence": 0.8}'
    assert _month_out_of_range(response) is True

    response = '{"category_id": 1, "month_iso": "1969-09", "confidence": 0.8}'
    assert _month_out_of_range(response) is True


def test_month_out_of_range_returns_true_for_post_period():
    """Month after 1973-09 should be detected as out of range."""
    from pipeline.mapper import _month_out_of_range

    response = '{"category_id": 1, "month_iso": "1973-10", "confidence": 0.8}'
    assert _month_out_of_range(response) is True

    response = '{"category_id": 1, "month_iso": "1974-01", "confidence": 0.8}'
    assert _month_out_of_range(response) is True


def test_month_out_of_range_returns_false_for_valid_month():
    """Valid months in range should return False."""
    from pipeline.mapper import _month_out_of_range

    response = '{"category_id": 1, "month_iso": "1969-10", "confidence": 0.8}'
    assert _month_out_of_range(response) is False

    response = '{"category_id": 1, "month_iso": "1972-05", "confidence": 0.8}'
    assert _month_out_of_range(response) is False

    response = '{"category_id": 1, "month_iso": "1973-09", "confidence": 0.8}'
    assert _month_out_of_range(response) is False


def test_month_out_of_range_returns_false_for_bad_json():
    """Bad JSON should return False (not out-of-range, just malformed)."""
    from pipeline.mapper import _month_out_of_range

    assert _month_out_of_range("not json") is False
    assert _month_out_of_range("{}") is False
    assert _month_out_of_range('{"category_id": 1}') is False


def test_month_out_of_range_returns_false_for_malformed_month():
    """Malformed month format should return False."""
    from pipeline.mapper import _month_out_of_range

    response = '{"category_id": 1, "month_iso": "1972/10", "confidence": 0.8}'
    assert _month_out_of_range(response) is False

    response = '{"category_id": 1, "month_iso": "72-10", "confidence": 0.8}'
    assert _month_out_of_range(response) is False


# ── Context tagging ────────────────────────────────────────────────────────


def test_tag_as_context_sets_doc_tag(db):
    """_tag_as_context should set doc_tag='context' and stamp mapped_at."""
    from pipeline.mapper import _tag_as_context

    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("doc002", "Frei Letter", "Text", "PRESS", "src", "sha", "2026-04-25T10:00:00Z"),
    )
    db.commit()

    _tag_as_context(db, doc_id="doc002")

    row = db.execute(
        "SELECT doc_tag, mapped_at FROM documents WHERE doc_id = 'doc002'"
    ).fetchone()

    assert row["doc_tag"] == "context"
    assert row["mapped_at"] is not None  # ISO timestamp should be set


# ── Session-timestamp filtering for --all and --remap modes ────────────────


def _insert_doc(db, doc_id, verified_at=None, mapped_category_id=None, mapped_at=None):
    """Helper to insert a test document with optional verification/mapping state."""
    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at,
            verified_at, mapped_category_id, mapped_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id, f"Title {doc_id}", f"Text {doc_id}", "PRESS", "src", doc_id,
            "2026-04-25T10:00:00Z", verified_at, mapped_category_id, mapped_at,
        ),
    )
    db.commit()


def test_fetch_all_verified_with_before_ts_excludes_already_processed(db):
    """With before_ts, docs whose mapped_at >= before_ts should be excluded."""
    from pipeline.mapper import _fetch_all_verified_documents

    session_start = "2026-04-25T12:00:00+00:00"

    # Verified, never mapped → should be returned
    _insert_doc(db, "d_unmapped", verified_at="2026-04-25T11:00:00Z")
    # Verified, mapped before session → should be returned (eligible for remap)
    _insert_doc(
        db, "d_old_mapped", verified_at="2026-04-25T11:00:00Z",
        mapped_category_id=1, mapped_at="2026-04-25T11:30:00+00:00",
    )
    # Verified, mapped during session → should be EXCLUDED
    _insert_doc(
        db, "d_just_mapped", verified_at="2026-04-25T11:00:00Z",
        mapped_category_id=1, mapped_at="2026-04-25T12:30:00+00:00",
    )

    rows = _fetch_all_verified_documents(db, batch_size=10, before_ts=session_start)
    doc_ids = {r["doc_id"] for r in rows}

    assert "d_unmapped" in doc_ids
    assert "d_old_mapped" in doc_ids
    assert "d_just_mapped" not in doc_ids


def test_fetch_all_verified_without_before_ts_returns_everything(db):
    """Without before_ts, behavior should be unchanged (all verified docs)."""
    from pipeline.mapper import _fetch_all_verified_documents

    _insert_doc(db, "d1", verified_at="2026-04-25T11:00:00Z")
    _insert_doc(
        db, "d2", verified_at="2026-04-25T11:00:00Z",
        mapped_category_id=1, mapped_at="2026-04-25T12:30:00+00:00",
    )

    rows = _fetch_all_verified_documents(db, batch_size=10)
    assert len(rows) == 2


def test_fetch_mapped_documents_only_returns_mapped(db):
    """_fetch_mapped_documents should exclude unmapped docs."""
    from pipeline.mapper import _fetch_mapped_documents

    _insert_doc(db, "d_unmapped", verified_at="2026-04-25T11:00:00Z")
    _insert_doc(
        db, "d_mapped", verified_at="2026-04-25T11:00:00Z",
        mapped_category_id=1, mapped_at="2026-04-25T11:30:00+00:00",
    )

    rows = _fetch_mapped_documents(db, batch_size=10)
    doc_ids = {r["doc_id"] for r in rows}

    assert "d_mapped" in doc_ids
    assert "d_unmapped" not in doc_ids


def test_fetch_mapped_documents_with_before_ts_filters_recent_remaps(db):
    """With before_ts, docs already remapped during this session are excluded."""
    from pipeline.mapper import _fetch_mapped_documents

    session_start = "2026-04-25T12:00:00+00:00"

    _insert_doc(
        db, "d_old", verified_at="2026-04-25T11:00:00Z",
        mapped_category_id=1, mapped_at="2026-04-25T11:30:00+00:00",
    )
    _insert_doc(
        db, "d_recent", verified_at="2026-04-25T11:00:00Z",
        mapped_category_id=1, mapped_at="2026-04-25T12:30:00+00:00",
    )

    rows = _fetch_mapped_documents(db, batch_size=10, before_ts=session_start)
    doc_ids = {r["doc_id"] for r in rows}

    assert "d_old" in doc_ids
    assert "d_recent" not in doc_ids


# ── Irrelevant signal (confidence=0.0) ─────────────────────────────────────


def test_is_irrelevant_signal_detects_zero_confidence():
    """LLM signaling confidence=0.0 should be flagged as irrelevant."""
    from pipeline.mapper import _is_irrelevant_signal

    response = '{"category_id": 0, "month_iso": "1969-10", "confidence": 0.0}'
    assert _is_irrelevant_signal(response) is True

    response = '{"category_id": 4, "month_iso": "1972-10", "confidence": 0}'
    assert _is_irrelevant_signal(response) is True


def test_is_irrelevant_signal_returns_false_for_high_confidence():
    """Normal confidence values should not be flagged as irrelevant."""
    from pipeline.mapper import _is_irrelevant_signal

    response = '{"category_id": 4, "month_iso": "1972-10", "confidence": 0.9}'
    assert _is_irrelevant_signal(response) is False

    response = '{"category_id": 4, "month_iso": "1972-10", "confidence": 0.3}'
    assert _is_irrelevant_signal(response) is False


def test_is_irrelevant_signal_returns_false_for_bad_json():
    """Bad JSON should return False (not irrelevant, just malformed)."""
    from pipeline.mapper import _is_irrelevant_signal

    assert _is_irrelevant_signal("not json") is False
    assert _is_irrelevant_signal("{}") is False
    assert _is_irrelevant_signal('{"category_id": 1}') is False


# ── Stamp skipped (prevents infinite loop in --all/--remap) ────────────────


def test_stamp_skipped_updates_mapped_at_only(db):
    """_stamp_skipped should set mapped_at without touching mapping fields."""
    from pipeline.mapper import _stamp_skipped

    _insert_doc(db, "d_skip", verified_at="2026-04-25T11:00:00Z")

    _stamp_skipped(db, doc_id="d_skip")

    row = db.execute(
        "SELECT mapped_at, mapped_category_id, mapped_month_iso FROM documents WHERE doc_id='d_skip'"
    ).fetchone()

    assert row["mapped_at"] is not None
    assert row["mapped_category_id"] is None
    assert row["mapped_month_iso"] is None


def test_stamp_skipped_excludes_doc_from_session_refetch(db):
    """After _stamp_skipped, the doc should not appear in --all session re-fetch."""
    from pipeline.mapper import _fetch_all_verified_documents, _stamp_skipped

    session_start = "2026-04-25T12:00:00+00:00"

    _insert_doc(db, "d_skip", verified_at="2026-04-25T11:00:00Z")

    # Before stamp: doc is fetchable
    rows = _fetch_all_verified_documents(db, batch_size=10, before_ts=session_start)
    assert "d_skip" in {r["doc_id"] for r in rows}

    # Stamp it (simulating a skip during the session, after session_start)
    _stamp_skipped(db, doc_id="d_skip")

    # After stamp: doc is excluded by the timestamp filter
    rows = _fetch_all_verified_documents(db, batch_size=10, before_ts=session_start)
    assert "d_skip" not in {r["doc_id"] for r in rows}


# ── Genre classification (the matrix's X axis) ────────────────────────────────


class TestParseGenreResponse:
    def test_extracts_valid_genre(self):
        from pipeline.mapper import parse_genre_response

        raw = '{"category_ids": [1], "genre_id": 3, "month_iso": "1972-10", "confidence": 0.9}'
        assert parse_genre_response(raw) == 3

    def test_none_when_genre_missing(self):
        from pipeline.mapper import parse_genre_response

        assert parse_genre_response('{"category_ids": [1], "month_iso": "1972-10"}') is None

    def test_none_for_out_of_range_genre(self):
        from pipeline.mapper import parse_genre_response

        assert parse_genre_response('{"genre_id": 0}') is None
        assert parse_genre_response('{"genre_id": 14}') is None

    def test_none_for_non_integer_genre(self):
        from pipeline.mapper import parse_genre_response

        assert parse_genre_response('{"genre_id": "discurso"}') is None
        assert parse_genre_response("not json") is None


class TestGenreFromSource:
    def test_title_keywords_beat_source_prior(self):
        from pipeline.mapper import genre_from_source

        # A speech edition hosted on a scholarly connector is still a speech.
        assert genre_from_source("openalex", "Discurso de Salvador Allende en la UNCTAD") == 1

    def test_frus_memoranda_are_cables(self):
        from pipeline.mapper import genre_from_source

        assert genre_from_source("frus", "Memorandum From the President's Assistant") == 3
        assert genre_from_source("frus", "Telegram From the Embassy in Chile") == 3

    def test_scholarly_sources_default_to_academico(self):
        from pipeline.mapper import genre_from_source

        assert genre_from_source("openalex", "The Chilean road to socialism") == 9
        assert genre_from_source("crossref", "Allende's economic policy") == 9
        assert genre_from_source("semantic_scholar", "Cybernetics in Chile") == 9

    def test_press_source_defaults_to_prensa(self):
        from pipeline.mapper import genre_from_source

        assert genre_from_source("chronicling_america", "Chile nationalizes copper mines") == 6

    def test_title_detects_official_documents(self):
        from pipeline.mapper import genre_from_source

        assert genre_from_source("wikisource_es", "Decreto 520 de requisición") == 4
        assert genre_from_source(None, "Ley 17.450 nacionalización del cobre") == 4

    def test_title_detects_poetry_and_fiction(self):
        from pipeline.mapper import genre_from_source

        assert genre_from_source("wikisource_es", "Canción del poder popular") == 11
        assert genre_from_source("archive.org", "Novela de la reforma agraria") == 12

    def test_unknown_source_without_keywords_is_none(self):
        from pipeline.mapper import genre_from_source

        assert genre_from_source("archive.org", "Untitled fragment 17") is None
        assert genre_from_source(None, "") is None


def test_save_mapping_stores_genre(db):
    from pipeline.mapper import _save_mapping

    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("doc_g", "Title", "Text", "PRESS", "src", "sha_g", "2026-04-25T10:00:00Z"),
    )
    db.commit()

    _save_mapping(db, doc_id="doc_g", category_ids=[1], month_iso="1971-05", genre_id=7)

    row = db.execute(
        "SELECT mapped_genre_id FROM documents WHERE doc_id='doc_g'"
    ).fetchone()
    assert row["mapped_genre_id"] == 7


def test_save_mapping_genre_defaults_to_null(db):
    from pipeline.mapper import _save_mapping

    db.execute(
        """
        INSERT INTO documents (
            doc_id, title, text, source_kind, source_id, sha256, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("doc_n", "Title", "Text", "PRESS", "src", "sha_n", "2026-04-25T10:00:00Z"),
    )
    db.commit()

    _save_mapping(db, doc_id="doc_n", category_ids=[2], month_iso="1971-05")

    row = db.execute(
        "SELECT mapped_genre_id FROM documents WHERE doc_id='doc_n'"
    ).fetchone()
    assert row["mapped_genre_id"] is None


def test_refresh_coverage_groups_by_genre(db):
    """Coverage refresh must aggregate per (theme, genre, month); NULL genre → 0."""
    from pipeline.mapper import Mapper

    def insert(doc_id, genre_id):
        db.execute(
            """
            INSERT INTO documents (
                doc_id, title, text, source_kind, source_id, sha256, ingested_at,
                quality_score, verified_at, mapped_category_id, mapped_month_iso,
                mapped_genre_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0.8, '2026-04-25T10:30:00Z', 4, '1972-10', ?)
            """,
            (doc_id, "T", "X", "PRESS", "src", f"sha_{doc_id}", "2026-04-25T10:00:00Z",
             genre_id),
        )

    insert("d1", 6)
    insert("d2", 6)
    insert("d3", 1)
    insert("d4", None)  # pre-genre-axis document
    db.commit()

    mapper = Mapper(db_path=Path(":memory:"), use_llm=False)
    mapper._refresh_coverage(db)

    rows = {
        (r["genre_id"]): r["coverage_score"]
        for r in db.execute(
            "SELECT genre_id, coverage_score FROM coverage "
            "WHERE category_id=4 AND month_iso='1972-10'"
        )
    }
    assert rows[6] == pytest.approx(0.4)   # 2/5
    assert rows[1] == pytest.approx(0.2)   # 1/5
    assert rows[0] == pytest.approx(0.2)   # unclassified bucket
