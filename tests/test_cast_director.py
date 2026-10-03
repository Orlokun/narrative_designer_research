"""Tests for the Cast Director agent (Ag-8) — the character relational layer.

Fully offline: pure-function and SQLite tests run without any LLM; the run_cycle
LLM path is exercised with a fake Ollama client. No live Ollama, no HTTP.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pytest

import pipeline.cast_director as cd
from pipeline.cast_director import (
    CastDirector,
    ensure_relation_tables,
    normalize_relation,
    normalize_relation_kind,
    parse_relations_response,
)
from pipeline.cast_manager import ensure_cast_tables

# ── Pure: normalize_relation_kind ──────────────────────────────────────────────


class TestNormalizeRelationKind:
    def test_symmetric_english_and_spanish(self):
        assert normalize_relation_kind("family") == ("familia", False)
        assert normalize_relation_kind("Aliado") == ("aliado", False)
        assert normalize_relation_kind("ENEMY") == ("enemigo", False)
        assert normalize_relation_kind("colleague") == ("colega", False)

    def test_directed_kinds(self):
        assert normalize_relation_kind("mentor") == ("mentor", False)
        assert normalize_relation_kind("superior") == ("superior", False)

    def test_inverse_kinds_flip_direction(self):
        assert normalize_relation_kind("subordinate") == ("superior", True)
        assert normalize_relation_kind("student") == ("mentor", True)

    def test_unknown_kind_is_other(self):
        assert normalize_relation_kind("frenemy-ish") == ("otro", False)
        assert normalize_relation_kind("") == ("otro", False)


# ── Pure: normalize_relation (pair + kind) ─────────────────────────────────────


class TestNormalizeRelation:
    def test_symmetric_pair_is_ordered(self):
        # A-B and B-A collapse to the same canonical row.
        assert normalize_relation("b", "a", "ally") == ("a", "b", "aliado")
        assert normalize_relation("a", "b", "ally") == ("a", "b", "aliado")

    def test_directed_pair_keeps_direction(self):
        assert normalize_relation("allende", "flores", "mentor") == (
            "allende", "flores", "mentor",
        )
        # reversed input is a different directed relation
        assert normalize_relation("flores", "allende", "mentor") == (
            "flores", "allende", "mentor",
        )

    def test_inverse_kind_flips_then_keeps_direction(self):
        # "flores subordinate-of allende" → allende superior-of flores
        assert normalize_relation("flores", "allende", "subordinate") == (
            "allende", "flores", "superior",
        )

    def test_self_relation_rejected(self):
        assert normalize_relation("allende", "allende", "ally") is None


# ── Pure: parse_relations_response ─────────────────────────────────────────────


class TestParseRelations:
    def test_parses_relations(self):
        raw = json.dumps({"relations": [
            {"source": "Salvador Allende", "target": "Fernando Flores",
             "kind": "colleague", "description": "worked together on Cybersyn"},
        ]})
        rels = parse_relations_response(raw)
        assert len(rels) == 1
        assert rels[0].source == "Salvador Allende"
        assert rels[0].target == "Fernando Flores"
        assert rels[0].kind == "colleague"
        assert "Cybersyn" in rels[0].description

    def test_missing_relations_key_returns_none(self):
        assert parse_relations_response('{"foo": 1}') is None

    def test_malformed_json_returns_none(self):
        assert parse_relations_response("not json") is None

    def test_skips_entries_without_both_endpoints(self):
        raw = json.dumps({"relations": [
            {"source": "A", "kind": "ally"},
            {"source": "Allende", "target": "Flores", "kind": "ally"},
            {"target": "B", "kind": "ally"},
        ]})
        rels = parse_relations_response(raw)
        assert len(rels) == 1
        assert rels[0].source == "Allende"


# ── SQLite: schema ─────────────────────────────────────────────────────────────


@pytest.fixture
def cast_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_cast_tables(conn)
    ensure_relation_tables(conn)
    now = datetime.now(UTC).isoformat()
    for cid, name in [
        ("salvador-allende", "Salvador Allende"),
        ("fernando-flores", "Fernando Flores"),
        ("augusto-pinochet", "Augusto Pinochet"),
    ]:
        conn.execute(
            "INSERT INTO characters (character_id, name, aliases, first_seen_at, updated_at) "
            "VALUES (?, ?, '[]', ?, ?)",
            (cid, name, now, now),
        )
    conn.commit()
    return conn


class TestSchema:
    def test_creates_relations_table(self, cast_db):
        cols = {r[1] for r in cast_db.execute("PRAGMA table_info(character_relations)")}
        assert {"source_character_id", "target_character_id", "kind",
                "confidence", "mention_count", "provenance"} <= cols

    def test_idempotent(self, cast_db):
        ensure_relation_tables(cast_db)  # must not raise

    def test_documents_gets_relations_extracted_column(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY)")
        ensure_relation_tables(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
        assert "relations_extracted_at" in cols


# ── SQLite: _upsert_relation accumulates evidence ──────────────────────────────


def _roster(conn):
    from pipeline.cast_manager import _load_character_roster
    return _load_character_roster(conn)


class TestUpsertRelation:
    def test_resolves_names_and_inserts(self, cast_db):
        status = cd._upsert_relation(
            cast_db, _roster(cast_db),
            "Salvador Allende", "Fernando Flores", "colleague",
            "Cybersyn collaborators", "doc1",
        )
        assert status == "new"
        row = cast_db.execute("SELECT * FROM character_relations").fetchone()
        assert row["source_character_id"] == "fernando-flores"  # symmetric → ordered
        assert row["target_character_id"] == "salvador-allende"
        assert row["kind"] == "colega"
        assert row["mention_count"] == 1
        assert json.loads(row["provenance"]) == ["doc1"]
        assert row["confidence"] == pytest.approx(1 / 3, abs=0.01)

    def test_corroboration_accumulates(self, cast_db):
        roster = _roster(cast_db)
        cd._upsert_relation(cast_db, roster, "Allende", "Flores", "ally", "", "doc1")
        cd._upsert_relation(cast_db, roster, "Flores", "Allende", "ally", "", "doc2")
        cd._upsert_relation(cast_db, roster, "Allende", "Flores", "ally", "", "doc2")  # dup doc
        row = cast_db.execute("SELECT * FROM character_relations").fetchone()
        assert cast_db.execute("SELECT COUNT(*) FROM character_relations").fetchone()[0] == 1
        assert row["mention_count"] == 2  # doc2 counted once
        assert set(json.loads(row["provenance"])) == {"doc1", "doc2"}
        assert row["confidence"] == pytest.approx(2 / 3, abs=0.01)

    def test_directed_relations_are_distinct_by_direction(self, cast_db):
        roster = _roster(cast_db)
        cd._upsert_relation(cast_db, roster, "Allende", "Flores", "mentor", "", "d1")
        cd._upsert_relation(cast_db, roster, "Flores", "Allende", "mentor", "", "d2")
        assert cast_db.execute("SELECT COUNT(*) FROM character_relations").fetchone()[0] == 2

    def test_unresolved_endpoint_is_skipped(self, cast_db):
        status = cd._upsert_relation(
            cast_db, _roster(cast_db),
            "Salvador Allende", "Richard Nixon", "enemy", "", "doc1",
        )
        assert status == "unresolved"
        assert cast_db.execute("SELECT COUNT(*) FROM character_relations").fetchone()[0] == 0

    def test_self_relation_skipped(self, cast_db):
        status = cd._upsert_relation(
            cast_db, _roster(cast_db),
            "Salvador Allende", "Allende", "ally", "", "doc1",
        )
        assert status == "self"
        assert cast_db.execute("SELECT COUNT(*) FROM character_relations").fetchone()[0] == 0


# ── Integration: run_cycle ─────────────────────────────────────────────────────


class _FakeOllama:
    def __init__(self, response: str) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeOllama:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def list_models(self) -> list[dict]:
        return [{"name": "gemma4:e4b"}]

    async def chat(self, **_kwargs: object) -> str:
        return self._response


def _seed_db(path):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    ensure_cast_tables(conn)
    conn.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY, title TEXT, text TEXT,
            verified_at TEXT, mapped_category_id INTEGER
        )
    """)
    now = datetime.now(UTC).isoformat()
    for cid, name in [("salvador-allende", "Salvador Allende"),
                      ("fernando-flores", "Fernando Flores")]:
        conn.execute(
            "INSERT INTO characters (character_id, name, aliases, first_seen_at, updated_at) "
            "VALUES (?, ?, '[]', ?, ?)",
            (cid, name, now, now),
        )
    conn.execute(
        "INSERT INTO documents VALUES ('doc1', 'Cybersyn', "
        "'Allende y Flores trabajaron juntos.', '2026-01-01', 1)"
    )
    conn.commit()
    conn.close()


class TestRunCycle:
    async def test_no_llm_stamps_docs_without_relations(self, tmp_path):
        db = tmp_path / "archivo.sqlite"
        _seed_db(db)
        result = await CastDirector(db_path=db, use_llm=False).run_cycle()
        assert result.processed == 1
        assert result.relations_new == 0

        conn = sqlite3.connect(str(db))
        stamped = conn.execute(
            "SELECT relations_extracted_at FROM documents WHERE doc_id='doc1'"
        ).fetchone()[0]
        conn.close()
        assert stamped is not None  # self-terminating

    async def test_llm_path_persists_relations(self, tmp_path, monkeypatch):
        db = tmp_path / "archivo.sqlite"
        _seed_db(db)
        response = json.dumps({"relations": [
            {"source": "Salvador Allende", "target": "Fernando Flores",
             "kind": "colleague", "description": "Cybersyn collaborators"},
        ]})
        monkeypatch.setattr(cd, "LLMClient", lambda: _FakeOllama(response))

        result = await CastDirector(db_path=db, use_llm=True).run_cycle()
        assert result.processed == 1
        assert result.relations_new == 1

        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM character_relations").fetchone()
        conn.close()
        assert row["kind"] == "colega"
        assert json.loads(row["provenance"]) == ["doc1"]

    async def test_second_run_is_idempotent(self, tmp_path, monkeypatch):
        db = tmp_path / "archivo.sqlite"
        _seed_db(db)
        monkeypatch.setattr(cd, "LLMClient", lambda: _FakeOllama('{"relations": []}'))
        await CastDirector(db_path=db, use_llm=True).run_cycle()
        result = await CastDirector(db_path=db, use_llm=True).run_cycle()
        assert result.processed == 0  # all docs already stamped


# ── Political compass: afiliacion between opposites → contraparte (roadmap #3) ──


class TestContraparteKind:
    def test_counterpart_maps_to_contraparte(self):
        assert normalize_relation_kind("counterpart") == ("contraparte", False)
        assert normalize_relation_kind("contraparte") == ("contraparte", False)

    def test_contraparte_is_symmetric(self):
        # A-B and B-A collapse to one ordered row.
        assert normalize_relation("b", "a", "counterpart") == ("a", "b", "contraparte")


class TestPoliticalRefinement:
    def _set_pos(self, conn, cid, econ, soc):
        conn.execute(
            "UPDATE characters SET pol_economic=?, pol_social=? WHERE character_id=?",
            (econ, soc, cid),
        )
        conn.commit()

    def test_load_positions_only_when_both_axes_set(self, cast_db):
        self._set_pos(cast_db, "salvador-allende", -0.8, -0.3)
        cast_db.execute(
            "UPDATE characters SET pol_economic=0.5 WHERE character_id='fernando-flores'"
        )  # only one axis → excluded
        cast_db.commit()
        positions = cd._load_political_positions(cast_db)
        assert positions == {"salvador-allende": (-0.8, -0.3)}

    def test_afiliacion_between_opposites_becomes_contraparte(self, cast_db):
        self._set_pos(cast_db, "salvador-allende", -0.8, -0.3)   # izquierda / libertario
        self._set_pos(cast_db, "augusto-pinochet", 0.7, 0.8)     # derecha / autoritario
        positions = cd._load_political_positions(cast_db)

        status = cd._upsert_relation(
            cast_db, _roster(cast_db), "Salvador Allende", "Augusto Pinochet",
            "political", "co-mentioned", "doc1", positions=positions,
        )
        assert status == "new"
        kind = cast_db.execute("SELECT kind FROM character_relations").fetchone()["kind"]
        assert kind == "contraparte"

    def test_afiliacion_kept_when_positions_unknown(self, cast_db):
        status = cd._upsert_relation(
            cast_db, _roster(cast_db), "Salvador Allende", "Augusto Pinochet",
            "political", "", "doc1", positions={},
        )
        assert status == "new"
        kind = cast_db.execute("SELECT kind FROM character_relations").fetchone()["kind"]
        assert kind == "afiliacion"
