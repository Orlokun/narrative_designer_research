"""Tests for the Cast Manager admin endpoints (/api/cast/*). Fully offline."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from admin.app import app
from lib.config import settings
from pipeline.cast_manager import ensure_cast_tables


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
        "INSERT INTO documents (doc_id, title, sha256) VALUES ('doc1', 'Discurso UNCTAD', 's1')"
    )
    conn.execute(
        """INSERT INTO characters
           (character_id, name, aliases, biography, completeness_score, mention_count,
            needs_research, first_seen_at, updated_at)
           VALUES ('salvador-allende', 'Salvador Allende', '["Allende"]',
                   'Presidente de Chile 1970-1973.', 0.8, 5, 0, ?, ?)""",
        (now, now),
    )
    conn.execute(
        """INSERT INTO characters
           (character_id, name, aliases, completeness_score, mention_count,
            needs_research, first_seen_at, updated_at)
           VALUES ('stafford-beer', 'Stafford Beer', '[]', 0.2, 1, 1, ?, ?)""",
        (now, now),
    )
    conn.execute(
        """INSERT INTO character_timeline
           (character_id, doc_id, date_iso, kind, description, created_at)
           VALUES ('salvador-allende', 'doc1', '1972-04', 'event',
                   'Discurso en la UNCTAD III', ?)""",
        (now,),
    )
    conn.execute(
        """INSERT INTO character_mentions (character_id, doc_id, mentioned_by, created_at)
           VALUES ('salvador-allende', 'doc1', 'Hernán Santa Cruz', ?)""",
        (now,),
    )
    conn.commit()
    conn.close()


class TestCharacterList:
    async def test_missing_db(self, client, temp_db):
        r = await client.get("/api/cast/characters")
        assert r.status_code == 200
        assert r.json() == {"available": False, "total": 0, "characters": []}

    async def test_lists_characters_with_scores(self, client, temp_db):
        _seed(temp_db)
        r = await client.get("/api/cast/characters")
        body = r.json()
        assert body["available"] is True
        assert body["total"] == 2
        first = body["characters"][0]
        assert first["character_id"] == "salvador-allende"  # most complete first
        assert first["completeness_score"] == pytest.approx(0.8)
        assert first["mention_count"] == 5
        assert body["characters"][1]["needs_research"] == 1

    async def test_search_filters_by_name_and_alias(self, client, temp_db):
        _seed(temp_db)
        r = await client.get("/api/cast/characters?search=beer")
        names = [c["name"] for c in r.json()["characters"]]
        assert names == ["Stafford Beer"]
        # alias match
        r = await client.get("/api/cast/characters?search=allende")
        assert [c["name"] for c in r.json()["characters"]] == ["Salvador Allende"]


class TestCharacterDetail:
    async def test_unknown_character(self, client, temp_db):
        _seed(temp_db)
        r = await client.get("/api/cast/characters/nadie")
        assert r.status_code == 404

    async def test_detail_includes_timeline_and_documents(self, client, temp_db):
        _seed(temp_db)
        r = await client.get("/api/cast/characters/salvador-allende")
        body = r.json()
        char = body["character"]
        assert char["name"] == "Salvador Allende"
        assert char["biography"].startswith("Presidente")
        assert char["aliases"] == ["Allende"]

        assert len(body["timeline"]) == 1
        fact = body["timeline"][0]
        assert fact["date_iso"] == "1972-04"
        assert fact["kind"] == "event"
        assert "UNCTAD" in fact["description"]

        assert len(body["documents"]) == 1
        doc = body["documents"][0]
        assert doc["doc_id"] == "doc1"
        assert doc["title"] == "Discurso UNCTAD"
        assert doc["mentioned_by"] == "Hernán Santa Cruz"


class TestCharacterTimelineChart:
    async def test_detail_includes_span_highlight_and_life_dates(self, client, temp_db):
        _seed(temp_db)
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE characters SET birth_date='1908', death_date='1973-09-11' "
            "WHERE character_id='salvador-allende'"
        )
        conn.commit()
        conn.close()

        body = (await client.get("/api/cast/characters/salvador-allende")).json()
        char = body["character"]
        assert char["birth_date"] == "1908"
        assert char["death_date"] == "1973-09-11"

        # Span runs from birth and always covers the 1970-1973 highlight band.
        assert body["span"]["start"] == 1908
        assert body["span"]["end"] >= 1974
        assert body["highlight"] == {"start": 1970, "end": 1974}

        # Each dated fact carries a fractional year for x-positioning.
        fact = body["timeline"][0]
        assert fact["year"] == pytest.approx(1972 + 3 / 12)  # 1972-04

    async def test_span_present_without_life_dates(self, client, temp_db):
        _seed(temp_db)
        body = (await client.get("/api/cast/characters/stafford-beer")).json()
        # No facts, no life dates → span is just the highlight period.
        assert body["span"] == {"start": 1970, "end": 1974}
        assert body["character"]["birth_date"] is None

    async def test_detail_exposes_wikidata_links(self, client, temp_db):
        _seed(temp_db)
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE characters SET wikidata_id='Q170581', "
            "wikipedia_url='https://es.wikipedia.org/wiki/Salvador_Allende', "
            "wikidata_desc='presidente de Chile' WHERE character_id='salvador-allende'"
        )
        conn.commit()
        conn.close()
        char = (await client.get("/api/cast/characters/salvador-allende")).json()["character"]
        assert char["wikidata_id"] == "Q170581"
        assert char["wikipedia_url"].endswith("Salvador_Allende")
        assert char["wikidata_desc"] == "presidente de Chile"

    async def test_wikidata_fields_default_none(self, client, temp_db):
        _seed(temp_db)
        char = (await client.get("/api/cast/characters/stafford-beer")).json()["character"]
        assert char["wikidata_id"] is None
        assert char["wikipedia_url"] is None

    async def test_detail_exposes_political_compass(self, client, temp_db):
        _seed(temp_db)
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE characters SET pol_economic=-0.8, pol_social=-0.3, "
            "pol_label='socialista' WHERE character_id='salvador-allende'"
        )
        conn.commit()
        conn.close()
        char = (await client.get("/api/cast/characters/salvador-allende")).json()["character"]
        assert char["pol_economic"] == -0.8
        assert char["pol_social"] == -0.3
        assert char["pol_label"] == "socialista"


from pipeline.cast_director import ensure_relation_tables  # noqa: E402


def _seed_relations(db_path):
    """Add a third character + two relations to the seeded DB."""
    conn = sqlite3.connect(str(db_path))
    ensure_relation_tables(conn)
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO characters (character_id, name, aliases, completeness_score, "
        "mention_count, needs_research, first_seen_at, updated_at) "
        "VALUES ('fernando-flores', 'Fernando Flores', '[]', 0.5, 2, 0, ?, ?)",
        (now, now),
    )
    conn.execute(
        "INSERT INTO character_relations (source_character_id, target_character_id, kind, "
        "description, confidence, mention_count, provenance, first_seen_at, updated_at) "
        "VALUES ('fernando-flores', 'salvador-allende', 'colega', 'Cybersyn', 0.67, 2, "
        '\'["doc1","doc2"]\', ?, ?)',
        (now, now),
    )
    conn.execute(
        "INSERT INTO character_relations (source_character_id, target_character_id, kind, "
        "description, confidence, mention_count, provenance, first_seen_at, updated_at) "
        "VALUES ('salvador-allende', 'stafford-beer', 'mentor', '', 0.33, 1, '[\"doc1\"]', ?, ?)",
        (now, now),
    )
    conn.commit()
    conn.close()


class TestCharacterRelations:
    async def test_no_relations_table_is_empty(self, client, temp_db):
        _seed(temp_db)
        body = (await client.get("/api/cast/characters/salvador-allende/relations")).json()
        assert body == {"relations": []}

    async def test_lists_relations_from_both_directions(self, client, temp_db):
        _seed(temp_db)
        _seed_relations(temp_db)
        body = (await client.get("/api/cast/characters/salvador-allende/relations")).json()
        rels = {r["other_id"]: r for r in body["relations"]}
        # Allende is target of the Flores colega edge and source of the Beer mentor edge.
        assert set(rels) == {"fernando-flores", "stafford-beer"}
        assert rels["fernando-flores"]["other_name"] == "Fernando Flores"
        assert rels["fernando-flores"]["kind"] == "colega"
        assert rels["fernando-flores"]["provenance"] == ["doc1", "doc2"]
        assert rels["fernando-flores"]["confidence"] == pytest.approx(0.67)
        # Direction is exposed for the directed mentor edge.
        assert rels["stafford-beer"]["kind"] == "mentor"
        assert rels["stafford-beer"]["direction"] == "outgoing"

    async def test_relations_sorted_by_confidence(self, client, temp_db):
        _seed(temp_db)
        _seed_relations(temp_db)
        body = (await client.get("/api/cast/characters/salvador-allende/relations")).json()
        confs = [r["confidence"] for r in body["relations"]]
        assert confs == sorted(confs, reverse=True)


class TestCastGraph:
    async def test_missing_db(self, client, temp_db):
        body = (await client.get("/api/cast/graph")).json()
        assert body == {"available": False, "nodes": [], "edges": []}

    async def test_graph_nodes_and_edges(self, client, temp_db):
        _seed(temp_db)
        _seed_relations(temp_db)
        body = (await client.get("/api/cast/graph")).json()
        assert body["available"] is True
        node_ids = {n["id"] for n in body["nodes"]}
        # Only characters that participate in a relation are nodes.
        assert node_ids == {"salvador-allende", "fernando-flores", "stafford-beer"}
        allende = next(n for n in body["nodes"] if n["id"] == "salvador-allende")
        assert allende["name"] == "Salvador Allende"
        assert allende["degree"] == 2  # two edges
        assert len(body["edges"]) == 2
        edge = next(e for e in body["edges"] if e["kind"] == "colega")
        assert {edge["source"], edge["target"]} == {"fernando-flores", "salvador-allende"}

    async def test_edges_carry_detail_and_directedness(self, client, temp_db):
        _seed(temp_db)
        _seed_relations(temp_db)
        body = (await client.get("/api/cast/graph")).json()
        colega = next(e for e in body["edges"] if e["kind"] == "colega")
        mentor = next(e for e in body["edges"] if e["kind"] == "mentor")
        # Detail for the edge panel.
        assert colega["description"] == "Cybersyn"
        assert colega["provenance"] == ["doc1", "doc2"]
        # Directedness drives the arrowhead: symmetric colega vs directed mentor.
        assert colega["directed"] is False
        assert mentor["directed"] is True


class TestTimelineNarrativeFields:
    async def test_timeline_exposes_statement_and_rumor_fields(self, client, temp_db):
        _seed(temp_db)
        import sqlite3

        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "INSERT INTO character_timeline (character_id, doc_id, date_iso, kind, "
            "description, speech_act, reported_by, confidence, created_at) "
            "VALUES ('salvador-allende', 'doc1', '1971-07', 'statement', "
            "'Anuncia la nacionalización', 'declarative', NULL, NULL, '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO character_timeline (character_id, doc_id, date_iso, kind, "
            "description, speech_act, reported_by, confidence, created_at) "
            "VALUES ('salvador-allende', NULL, NULL, 'rumor', "
            "'Se rumorea un plebiscito', NULL, 'la prensa', 0.2, '2026-01-01')"
        )
        conn.commit()
        conn.close()

        r = await client.get("/api/cast/characters/salvador-allende")
        by_kind = {f["kind"]: f for f in r.json()["timeline"]}
        assert by_kind["statement"]["speech_act"] == "declarative"
        assert by_kind["rumor"]["reported_by"] == "la prensa"
        assert by_kind["rumor"]["confidence"] == 0.2


class TestCastStats:
    async def test_stats_summarise_scores_and_caps(self, client, temp_db):
        _seed(temp_db)
        import json as _json
        import sqlite3

        conn = sqlite3.connect(str(temp_db))
        detail = _json.dumps({"caps": ["no_external_analysis", "no_research_mission"]})
        conn.execute(
            "UPDATE characters SET completeness_score=0.55, completeness_detail=? "
            "WHERE character_id='salvador-allende'",
            (detail,),
        )
        conn.execute(
            "UPDATE characters SET completeness_score=0.9 WHERE character_id='stafford-beer'"
        )
        conn.commit()
        conn.close()

        r = await client.get("/api/cast/stats")
        body = r.json()
        assert body["available"] is True
        assert body["total"] == 2
        assert body["complete"] == 1
        assert body["threshold"] == 0.85
        assert sum(body["histogram"]) == 2
        assert body["histogram"][5] == 1  # 0.55 falls in the [0.5, 0.6) bucket
        assert body["caps"]["no_external_analysis"] == 1
        assert body["caps"]["no_research_mission"] == 1

    async def test_stats_with_no_database(self, client, temp_db):
        # temp_db points settings at a path that was never created
        r = await client.get("/api/cast/stats")
        assert r.json()["available"] is False


class TestDetailBreakdown:
    async def test_detail_exposes_completeness_detail(self, client, temp_db):
        _seed(temp_db)
        import json as _json
        import sqlite3

        conn = sqlite3.connect(str(temp_db))
        detail = _json.dumps({"identity": 0.1, "caps": ["low_kind_diversity"], "score": 0.4})
        conn.execute(
            "UPDATE characters SET completeness_detail=? WHERE character_id='salvador-allende'",
            (detail,),
        )
        conn.commit()
        conn.close()

        r = await client.get("/api/cast/characters/salvador-allende")
        breakdown = r.json()["character"]["completeness_detail"]
        assert breakdown["identity"] == 0.1
        assert breakdown["caps"] == ["low_kind_diversity"]
