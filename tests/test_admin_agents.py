"""Tests for the Mapper admin endpoint (/api/agents/mapper). Fully offline."""

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
    """rows: (doc_id, verified, category_id, genre_id, month, doc_tag, mapped_at)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'Documento',
            text TEXT NOT NULL DEFAULT 'x',
            sha256 TEXT NOT NULL DEFAULT '',
            ingested_at TEXT NOT NULL DEFAULT '2026-01-01',
            quality_score REAL,
            verified_at TEXT,
            doc_tag TEXT,
            mapped_category_id INTEGER,
            mapped_genre_id INTEGER,
            mapped_month_iso TEXT,
            mapped_at TEXT
        )
    """)
    for doc_id, verified, cat, genre, month, tag, mapped_at in rows:
        conn.execute(
            """INSERT INTO documents
               (doc_id, sha256, verified_at, mapped_category_id, mapped_genre_id,
                mapped_month_iso, doc_tag, mapped_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, doc_id, "2026-01-01T00:00:00Z" if verified else None,
             cat, genre, month, tag, mapped_at),
        )
    conn.commit()
    conn.close()


class TestMapperStats:
    async def test_missing_db(self, client, temp_db):
        r = await client.get("/api/agents/mapper")
        assert r.status_code == 200
        assert r.json()["available"] is False

    async def test_counts_and_distributions(self, client, temp_db):
        _seed(temp_db, [
            ("d1", True, 1, 1, "1970-11", None, "2026-01-02T10:00:00Z"),
            ("d2", True, 1, 6, "1972-10", None, "2026-01-02T11:00:00Z"),
            ("d3", True, 4, None, "1972-10", None, "2026-01-02T12:00:00Z"),  # sin género
            ("d4", True, None, None, None, None, None),                     # pendiente
            ("d5", True, None, None, None, "context", "2026-01-02T09:00:00Z"),
        ])
        r = await client.get("/api/agents/mapper")
        body = r.json()
        assert body["available"] is True
        assert body["mapped"] == 3
        assert body["pending"] == 1
        assert body["context"] == 1

        genres = {g["genre"]: g["n"] for g in body["by_genre"]}
        assert genres["Discurso"] == 1
        assert genres["Prensa informativa"] == 1
        assert genres["Sin clasificar"] == 1

        categories = {c["category"]: c["n"] for c in body["by_category"]}
        assert categories["Política Nacional"] == 2
        assert categories["Industria Nacional"] == 1

    async def test_recent_ordered_by_mapped_at_desc(self, client, temp_db):
        _seed(temp_db, [
            ("d1", True, 1, 1, "1970-11", None, "2026-01-02T10:00:00Z"),
            ("d2", True, 2, 6, "1971-01", None, "2026-01-03T10:00:00Z"),
        ])
        r = await client.get("/api/agents/mapper")
        recent = r.json()["recent"]
        assert [d["doc_id"] for d in recent] == ["d2", "d1"]
        assert recent[0]["category"] == "Salud"
        assert recent[0]["genre"] == "Prensa informativa"
        assert recent[0]["month_iso"] == "1971-01"


class TestVerificatorThresholdReflectsSystem:
    async def test_endpoint_reports_the_real_default(self, client, temp_db):
        """The admin must never hardcode the threshold — it reads the Verificator's
        own constant, so UI and pipeline can't diverge again."""
        from pipeline.verificator import _DEFAULT_MIN_SCORE

        _seed(temp_db, [("d1", True, 1, 1, "1970-11", None, "2026-01-02T10:00:00Z")])
        r = await client.get("/api/agents/verificator")
        body = r.json()
        assert body["threshold"] == _DEFAULT_MIN_SCORE
        assert body["threshold"] == 0.6
