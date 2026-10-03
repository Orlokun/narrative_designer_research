"""Tests for the Admin heatmap router — the theme × genre projection endpoint.

Fully offline: httpx ASGITransport against the FastAPI app, settings.archive_db
monkeypatched to a temp SQLite file.
"""

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


def _seed_documents(db_path, rows):
    """rows: list of (doc_id, category_id, genre_id, is_complete)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 't',
            text TEXT NOT NULL DEFAULT 'x',
            provenance TEXT NOT NULL DEFAULT '{}',
            sha256 TEXT NOT NULL DEFAULT '',
            ingested_at TEXT NOT NULL DEFAULT '2026-01-01',
            verified_at TEXT,
            is_complete INTEGER DEFAULT 1,
            doc_tag TEXT,
            mapped_category_id INTEGER,
            mapped_month_iso TEXT,
            mapped_genre_id INTEGER
        )
    """)
    for doc_id, category_id, genre_id, is_complete in rows:
        conn.execute(
            """
            INSERT INTO documents
                (doc_id, sha256, verified_at, is_complete,
                 mapped_category_id, mapped_month_iso, mapped_genre_id)
            VALUES (?, ?, '2026-01-01T00:00:00Z', ?, ?, '1972-10', ?)
            """,
            (doc_id, doc_id, is_complete, category_id, genre_id),
        )
    conn.commit()
    conn.close()


class TestGenreHeatmap:
    async def test_missing_db_returns_unavailable(self, client, temp_db):
        r = await client.get("/api/heatmap/genres")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is False
        assert len(body["categories"]) == 16
        assert len(body["genres"]) == 13
        assert len(body["cells"]) == 16
        assert all(len(row) == 13 for row in body["cells"])

    async def test_documents_counted_in_their_cells(self, client, temp_db):
        _seed_documents(temp_db, [
            ("d1", 4, 8, 1),   # Industria × Informe técnico, complete
            ("d2", 4, 8, 1),
            ("d3", 4, 8, 0),   # incomplete counts fractionally
            ("d4", 1, 1, 1),   # Política × Discurso
        ])
        r = await client.get("/api/heatmap/genres")
        body = r.json()
        assert body["available"] is True
        # rows are category_id-1, cols genre_id-1
        assert body["doc_counts"][3][7] == 3
        assert body["doc_counts"][0][0] == 1
        # score: (2 complete + 0.2×1 incomplete)/20 = 0.11
        assert body["cells"][3][7] == pytest.approx(0.11)

    async def test_unclassified_documents_bucket_separately(self, client, temp_db):
        _seed_documents(temp_db, [
            ("d1", 4, None, 1),
            ("d2", 4, None, 1),
        ])
        r = await client.get("/api/heatmap/genres")
        body = r.json()
        assert body["unclassified"][3] == 2
        assert all(n == 0 for row in body["doc_counts"] for n in row)

    async def test_context_docs_excluded(self, client, temp_db):
        _seed_documents(temp_db, [("d1", 4, 8, 1)])
        conn = sqlite3.connect(str(temp_db))
        conn.execute("UPDATE documents SET doc_tag='context' WHERE doc_id='d1'")
        conn.commit()
        conn.close()
        r = await client.get("/api/heatmap/genres")
        assert r.json()["doc_counts"][3][7] == 0

    async def test_plausibility_matrix_included(self, client, temp_db):
        r = await client.get("/api/heatmap/genres")
        body = r.json()
        plausibility = body["plausibility"]
        assert len(plausibility) == 16
        assert all(len(row) == 13 for row in plausibility)
        # Spot-check the registry: EEUU × Cable = 1.0, Deporte × Cable = 0.1
        assert plausibility[5][2] == 1.0
        assert plausibility[14][2] == pytest.approx(0.1)

    async def test_summary_weighted_by_plausibility(self, client, temp_db):
        _seed_documents(temp_db, [(f"d{i}", 6, 3, 1) for i in range(20)])
        r = await client.get("/api/heatmap/genres")
        summary = r.json()["summary"]
        assert summary["total_cells"] == 16 * 13
        assert summary["filled_cells"] == 1
        assert 0.0 < summary["coverage_pct"] <= 100.0


class TestGenreMonthStrip:
    async def test_strip_for_empty_db(self, client, temp_db):
        r = await client.get("/api/heatmap/genres/months?category_id=4&genre_id=8")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is False
        assert body["category"] == "Industria Nacional"
        assert body["genre"] == "Informe técnico"
        assert len(body["months"]) == 48
        assert body["months"][0] == "1969-10"
        assert body["months"][-1] == "1973-09"
        assert body["cells"] == [0.0] * 48
        assert "critical_months" in body

    async def test_strip_counts_documents_per_month(self, client, temp_db):
        _seed_documents(temp_db, [
            ("d1", 4, 8, 1),   # all seeded at 1972-10
            ("d2", 4, 8, 1),
            ("d3", 4, 8, 0),   # incomplete → fractional
            ("d4", 4, 1, 1),   # other genre — must not count
            ("d5", 1, 8, 1),   # other theme — must not count
        ])
        r = await client.get("/api/heatmap/genres/months?category_id=4&genre_id=8")
        body = r.json()
        idx = body["months"].index("1972-10")
        assert body["doc_counts"][idx] == 3
        assert body["cells"][idx] == pytest.approx(0.11)  # (2 + 0.2) / 20
        assert sum(body["doc_counts"]) == 3

    async def test_strip_genre_zero_shows_unclassified(self, client, temp_db):
        _seed_documents(temp_db, [("d1", 4, None, 1)])
        r = await client.get("/api/heatmap/genres/months?category_id=4&genre_id=0")
        body = r.json()
        assert body["genre"] == "Sin clasificar"
        assert sum(body["doc_counts"]) == 1

    async def test_strip_rejects_out_of_range(self, client, temp_db):
        assert (await client.get("/api/heatmap/genres/months?category_id=0&genre_id=1")).status_code == 422
        assert (await client.get("/api/heatmap/genres/months?category_id=1&genre_id=14")).status_code == 422
