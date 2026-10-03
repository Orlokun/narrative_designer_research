"""Tests for the Location Manager admin endpoints (/api/locations/*). Fully offline."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from admin.app import app
from lib.config import settings
from pipeline.cast_manager import ensure_cast_tables
from pipeline.location_manager import ensure_location_tables


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "archivo.sqlite"
    monkeypatch.setattr(settings, "archive_db", path)
    return path


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _seed(db_path):
    conn = sqlite3.connect(str(db_path))
    ensure_location_tables(conn)
    ensure_cast_tables(conn)
    conn.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            text TEXT NOT NULL DEFAULT 'x',
            sha256 TEXT NOT NULL DEFAULT '',
            ingested_at TEXT NOT NULL DEFAULT '2026-01-01'
        )
    """)
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO documents (doc_id, title, sha256) VALUES ('doc1', 'Informe La Moneda', 's1')"
    )
    conn.execute(
        "INSERT INTO characters (character_id, name, first_seen_at, updated_at) "
        "VALUES ('salvador-allende', 'Salvador Allende', ?, ?)",
        (now, now),
    )
    detail = json.dumps({"caps": [], "score": 0.9})
    conn.execute(
        "INSERT INTO locations (location_id, name, aliases, kind, description, "
        "latitude, longitude, character_id, completeness_score, completeness_detail, "
        "mention_count, needs_research, first_seen_at, updated_at) VALUES "
        "('la-moneda', 'La Moneda', '[\"Palacio de La Moneda\"]', 'building', "
        "'Sede del gobierno.', -33.443, -70.654, 'salvador-allende', 0.9, ?, 3, 0, ?, ?)",
        (detail, now, now),
    )
    conn.execute(
        "INSERT INTO locations (location_id, name, kind, completeness_score, "
        "mention_count, needs_research, first_seen_at, updated_at) VALUES "
        "('fabrica-temuco', 'Fábrica de harina de Temuco', 'factory', 0.2, 1, 1, ?, ?)",
        (now, now),
    )
    conn.execute(
        "INSERT INTO location_mentions (location_id, doc_id, created_at) "
        "VALUES ('la-moneda', 'doc1', ?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO location_facts (location_id, doc_id, date_iso, kind, detail, "
        "reported_by, created_at) VALUES ('la-moneda', 'doc1', '1972-10', "
        "'appreciation', 'Un edificio solemne', 'el embajador', ?)",
        (now,),
    )
    conn.commit()
    conn.close()


class TestLocationList:
    async def test_missing_db(self, client, temp_db):
        r = await client.get("/api/locations")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is False
        assert body["locations"] == []

    async def test_lists_locations_with_summary(self, client, temp_db):
        _seed(temp_db)
        r = await client.get("/api/locations")
        body = r.json()
        assert body["available"] is True
        assert body["total"] == 2
        assert body["complete"] == 1
        assert body["with_coordinates"] == 1
        assert body["threshold"] == 0.85
        first = body["locations"][0]
        assert first["location_id"] == "la-moneda"  # most complete first
        assert first["latitude"] == -33.443
        assert body["locations"][1]["needs_research"] == 1

    async def test_search_filters_by_name_and_alias(self, client, temp_db):
        _seed(temp_db)
        r = await client.get("/api/locations?search=temuco")
        assert [loc["name"] for loc in r.json()["locations"]] == ["Fábrica de harina de Temuco"]
        r = await client.get("/api/locations?search=palacio")
        assert [loc["location_id"] for loc in r.json()["locations"]] == ["la-moneda"]


class TestLocationDetail:
    async def test_unknown_location_404(self, client, temp_db):
        _seed(temp_db)
        r = await client.get("/api/locations/ninguna-parte")
        assert r.status_code == 404

    async def test_detail_includes_facts_documents_and_character(self, client, temp_db):
        _seed(temp_db)
        r = await client.get("/api/locations/la-moneda")
        body = r.json()
        loc = body["location"]
        assert loc["name"] == "La Moneda"
        assert loc["aliases"] == ["Palacio de La Moneda"]
        assert loc["kind"] == "building"
        assert loc["character_id"] == "salvador-allende"
        assert loc["character_name"] == "Salvador Allende"
        assert loc["completeness_detail"]["score"] == 0.9

        assert len(body["facts"]) == 1
        fact = body["facts"][0]
        assert fact["kind"] == "appreciation"
        assert fact["reported_by"] == "el embajador"

        assert len(body["documents"]) == 1
        assert body["documents"][0]["title"] == "Informe La Moneda"

    async def test_detail_without_character_or_coords(self, client, temp_db):
        _seed(temp_db)
        loc = (await client.get("/api/locations/fabrica-temuco")).json()["location"]
        assert loc["character_id"] is None
        assert loc["character_name"] is None
        assert loc["latitude"] is None


class TestLocationsPage:
    async def test_page_is_served(self, client):
        r = await client.get("/locations")
        assert r.status_code == 200
        assert "Lugares" in r.text


class TestLocationGallery:
    async def test_detail_includes_images(self, client, temp_db):
        _seed(temp_db)
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "INSERT INTO location_images (location_id, filename, url, page_url, "
            "caption, source, created_at) VALUES ('la-moneda', 'moneda.jpg', "
            "'https://commons.wikimedia.org/wiki/Special:FilePath/Moneda.jpg?width=640', "
            "'https://commons.wikimedia.org/wiki/File:Moneda.jpg', NULL, 'wikidata', "
            "'2026-01-01')"
        )
        conn.commit()
        conn.close()

        body = (await client.get("/api/locations/la-moneda")).json()
        assert len(body["images"]) == 1
        image = body["images"][0]
        assert "Special:FilePath" in image["url"]
        assert image["source"] == "wikidata"

    async def test_detail_without_images_returns_empty_list(self, client, temp_db):
        _seed(temp_db)
        body = (await client.get("/api/locations/fabrica-temuco")).json()
        assert body["images"] == []
