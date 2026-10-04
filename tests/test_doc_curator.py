"""Tests for the Ag-9 Document Curator — utility-flag assignment over documents."""

from __future__ import annotations

import json
import sqlite3

import pytest

import pipeline.doc_curator as dc
from pipeline.doc_curator import (
    DocumentCurator,
    clear_flags,
    ensure_flags_table,
)


@pytest.fixture()
def db(tmp_path):
    """A temp archive DB with two verified+mapped documents."""
    path = tmp_path / "archivo.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            text TEXT,
            verified_at TEXT,
            mapped_category_id INTEGER,
            flags_curated_at TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO documents (doc_id, title, text, verified_at, mapped_category_id) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (
                "d1",
                "Memorándum militar",
                "El ejército y las fuerzas armadas preparan un golpe. El general dio la orden.",
                "2026-01-01T00:00:00Z",
                13,
            ),
            (
                "d2",
                "Nota económica",
                "Planificación de la producción y nacionalización de la industria (CORFO).",
                "2026-01-01T00:00:00Z",
                3,
            ),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _conn(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


class _FakeOllama:
    """Stand-in for OllamaClient: returns a canned chat response, model present."""

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


# ── Schema ────────────────────────────────────────────────────────────────────


def test_ensure_flags_table_creates_it():
    conn = sqlite3.connect(":memory:")
    ensure_flags_table(conn)
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "doc_flags" in tables


def test_ensure_flags_table_idempotent():
    conn = sqlite3.connect(":memory:")
    ensure_flags_table(conn)
    ensure_flags_table(conn)  # must not raise


# ── run_cycle: heuristic (--no-llm) path ──────────────────────────────────────


class TestRunCycleHeuristic:
    async def test_assigns_and_persists_flags(self, db):
        result = await DocumentCurator(db_path=db, use_llm=False).run_cycle()
        assert result.processed == 2
        conn = _conn(db)
        flags = {
            (r["doc_id"], r["flag"])
            for r in conn.execute("SELECT doc_id, flag FROM doc_flags").fetchall()
        }
        assert ("d1", "military-logic") in flags
        assert ("d2", "economic-planning") in flags
        # heuristic path stamps assigned_by='heuristic'
        by = conn.execute("SELECT DISTINCT assigned_by FROM doc_flags").fetchone()[0]
        assert by == "heuristic"

    async def test_stamps_and_skips_second_run(self, db):
        await DocumentCurator(db_path=db, use_llm=False).run_cycle()
        second = await DocumentCurator(db_path=db, use_llm=False).run_cycle()
        assert second.processed == 0  # all stamped flags_curated_at

    async def test_missing_db_returns_empty(self, tmp_path):
        result = await DocumentCurator(db_path=tmp_path / "nope.sqlite", use_llm=False).run_cycle()
        assert result.processed == 0


# ── run_cycle: LLM path ───────────────────────────────────────────────────────


class TestRunCycleLLM:
    async def test_persists_llm_flags(self, db, monkeypatch):
        response = json.dumps(
            {
                "flags": [
                    {"slug": "military-logic", "confidence": 0.9, "rationale": "coup planning"},
                    {"slug": "high-value", "confidence": 0.7, "rationale": "rich primary source"},
                ]
            }
        )
        monkeypatch.setattr(dc, "LLMClient", lambda: _FakeOllama(response))
        result = await DocumentCurator(db_path=db, use_llm=True).run_cycle()
        assert result.processed == 2
        conn = _conn(db)
        row = conn.execute(
            "SELECT flag, confidence, rationale, assigned_by FROM doc_flags "
            "WHERE doc_id='d1' AND flag='military-logic'"
        ).fetchone()
        assert row["confidence"] == 0.9
        assert row["rationale"] == "coup planning"
        assert row["assigned_by"] == "curator"

    async def test_unparseable_llm_writes_no_flags(self, db, monkeypatch):
        monkeypatch.setattr(dc, "LLMClient", lambda: _FakeOllama("totally not json"))
        result = await DocumentCurator(db_path=db, use_llm=True).run_cycle()
        assert result.processed == 2  # still stamped as curated
        assert result.flags_written == 0
        conn = _conn(db)
        assert conn.execute("SELECT COUNT(*) FROM doc_flags").fetchone()[0] == 0

    async def test_re_curate_replaces_flags(self, db, monkeypatch):
        first = json.dumps({"flags": [{"slug": "diplomacy", "confidence": 0.8}]})
        monkeypatch.setattr(dc, "LLMClient", lambda: _FakeOllama(first))
        await DocumentCurator(db_path=db, use_llm=True).run_cycle()

        second = json.dumps({"flags": [{"slug": "military-logic", "confidence": 0.6}]})
        monkeypatch.setattr(dc, "LLMClient", lambda: _FakeOllama(second))
        result = await DocumentCurator(db_path=db, use_llm=True).run_cycle(re_curate=True)
        assert result.processed == 2
        conn = _conn(db)
        slugs = {r["flag"] for r in conn.execute("SELECT flag FROM doc_flags").fetchall()}
        assert slugs == {"military-logic"}  # old diplomacy flags replaced


# ── status / clear ────────────────────────────────────────────────────────────


class TestStatusAndClear:
    async def test_status_counts(self, db):
        await DocumentCurator(db_path=db, use_llm=False).run_cycle()
        status = DocumentCurator(db_path=db).status()
        assert status["curated"] == 2
        assert status["flags"] >= 2

    async def test_clear_dry_run_then_apply(self, db):
        await DocumentCurator(db_path=db, use_llm=False).run_cycle()
        conn = _conn(db)
        n = clear_flags(conn, apply=False)
        assert n >= 2
        assert conn.execute("SELECT COUNT(*) FROM doc_flags").fetchone()[0] == n  # untouched
        clear_flags(conn, apply=True)
        assert conn.execute("SELECT COUNT(*) FROM doc_flags").fetchone()[0] == 0
        remaining = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE flags_curated_at IS NOT NULL"
        ).fetchone()[0]
        assert remaining == 0  # stamps reset so a re-run re-curates
