"""Tests for the document browser endpoints — by-id deep-linking. Fully offline."""

from __future__ import annotations

import sqlite3

import pytest
from httpx import ASGITransport, AsyncClient

from admin.app import app
from lib.config import settings


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "archivo.sqlite"
    monkeypatch.setattr(settings, "archive_db", path)
    return path


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _seed(db_path, rows):
    """rows: (doc_id, quality_score, verified)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'Documento',
            text TEXT NOT NULL DEFAULT 'contenido',
            quality_score REAL,
            source_id TEXT DEFAULT 'src',
            source_kind TEXT DEFAULT 'archive',
            provenance TEXT NOT NULL DEFAULT '{}',
            sha256 TEXT NOT NULL DEFAULT '',
            ingested_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
            verified_at TEXT,
            is_complete INTEGER DEFAULT 1,
            doc_tag TEXT,
            mapped_category_id INTEGER,
            mapped_categories TEXT,
            mapped_month_iso TEXT
        )
    """)
    conn.execute(
        "CREATE TABLE missions (mission_id TEXT PRIMARY KEY, category TEXT, category_id INTEGER, month_iso TEXT)"
    )
    for doc_id, score, verified in rows:
        conn.execute(
            "INSERT INTO documents (doc_id, title, sha256, quality_score, verified_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                doc_id,
                f"Título {doc_id}",
                doc_id,
                score,
                "2026-01-01T00:00:00Z" if verified else None,
            ),
        )
    conn.commit()
    conn.close()


class TestDocumentById:
    async def test_missing_db_404(self, client, temp_db):
        r = await client.get("/api/documents/by-id/nope")
        assert r.status_code == 404

    async def test_unknown_doc_404(self, client, temp_db):
        _seed(temp_db, [("d1", 0.9, True)])
        r = await client.get("/api/documents/by-id/nope")
        assert r.status_code == 404

    async def test_returns_doc_with_carousel_position(self, client, temp_db):
        # Carousel order: quality_score DESC, so d-high is page 1, d-low page 2.
        _seed(temp_db, [("d-low", 0.4, True), ("d-high", 0.9, True)])
        r = await client.get("/api/documents/by-id/d-low")
        assert r.status_code == 200
        body = r.json()
        assert body["doc"]["doc_id"] == "d-low"
        assert body["doc"]["title"] == "Título d-low"
        assert body["position"] == 2
        assert body["total"] == 2

        # The position must agree with the unfiltered carousel.
        carousel = (await client.get("/api/documents?page=2&verified_only=false")).json()
        assert carousel["doc"]["doc_id"] == "d-low"

    async def test_unverified_docs_are_reachable(self, client, temp_db):
        # Links from the Archivero recent list may point at not-yet-verified docs.
        _seed(temp_db, [("d-pending", None, False), ("d-ok", 0.9, True)])
        r = await client.get("/api/documents/by-id/d-pending")
        assert r.status_code == 200
        body = r.json()
        assert body["doc"]["verified"] is False
        # NULL score sorts last in the unfiltered ordering
        assert body["position"] == 2


def _add_flags(db_path, entries):
    """entries: (doc_id, flag, confidence). Creates doc_flags on first call."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doc_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL, flag TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0, rationale TEXT,
            assigned_by TEXT NOT NULL DEFAULT 'curator',
            assigned_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
            UNIQUE(doc_id, flag)
        )
    """)
    for doc_id, flag, conf in entries:
        conn.execute(
            "INSERT INTO doc_flags (doc_id, flag, confidence, rationale) VALUES (?, ?, ?, ?)",
            (doc_id, flag, conf, "because"),
        )
    conn.commit()
    conn.close()


class TestDocumentFlags:
    async def test_flags_registry_endpoint(self, client, temp_db):
        _seed(temp_db, [("d1", 0.9, True)])
        _add_flags(temp_db, [("d1", "military-logic", 0.9)])
        r = await client.get("/api/documents/flags")
        assert r.status_code == 200
        body = {f["slug"]: f for f in r.json()}
        assert "military-logic" in body
        assert body["military-logic"]["count"] == 1
        assert body["diplomacy"]["count"] == 0  # in registry, uncarried

    async def test_document_payload_includes_flags(self, client, temp_db):
        _seed(temp_db, [("d1", 0.9, True)])
        _add_flags(temp_db, [("d1", "military-logic", 0.9), ("d1", "high-value", 0.5)])
        body = (await client.get("/api/documents/by-id/d1")).json()
        slugs = [f["slug"] for f in body["doc"]["flags"]]
        assert slugs == ["military-logic", "high-value"]  # confidence DESC
        assert body["doc"]["flags"][0]["name"] == "Lógica militar"

    async def test_flag_filter_restricts_carousel(self, client, temp_db):
        _seed(temp_db, [("d1", 0.9, True), ("d2", 0.8, True)])
        _add_flags(temp_db, [("d1", "diplomacy", 0.9)])
        r = await client.get("/api/documents?flag=diplomacy")
        body = r.json()
        assert body["total"] == 1
        assert body["doc"]["doc_id"] == "d1"

    async def test_flag_filter_missing_table_is_empty(self, client, temp_db):
        _seed(temp_db, [("d1", 0.9, True)])  # no doc_flags table created
        r = await client.get("/api/documents?flag=diplomacy")
        assert r.json()["total"] == 0

    async def test_payload_without_flags_table(self, client, temp_db):
        _seed(temp_db, [("d1", 0.9, True)])  # no doc_flags table
        body = (await client.get("/api/documents/by-id/d1")).json()
        assert body["doc"]["flags"] == []
