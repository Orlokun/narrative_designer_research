"""Tests for the Propositor's entity-driven missions (Cast Manager wiring).

Fully offline: pure-function tests plus SQLite integration on a temp database.
run_cycle is exercised with use_llm=False so no Ollama is touched.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime

import pytest

from lib.schemas import MissionKind, MissionStatus
from pipeline.cast_manager import ensure_cast_tables
from pipeline.location_manager import ensure_location_tables
from research_engine.propositor import (
    Propositor,
    _cleanup_duplicates,
    _ensure_missions_table,
    _read_queued_cells,
    _read_queued_characters,
    build_entity_missions,
    build_location_missions,
    entity_priority,
    entity_queries,
    location_queries,
    order_by_need,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def mem_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def _insert_character(
    conn: sqlite3.Connection,
    character_id: str,
    name: str,
    mention_count: int = 1,
    completeness_score: float = 0.0,
    needs_research: int = 1,
) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO characters
            (character_id, name, aliases, completeness_score, mention_count,
             needs_research, first_seen_at, updated_at)
        VALUES (?, ?, '[]', ?, ?, ?, ?, ?)
        """,
        (character_id, name, completeness_score, mention_count, needs_research, now, now),
    )
    conn.commit()


def _insert_mission_row(
    conn: sqlite3.Connection,
    mission_id: str,
    category_id: int = 1,
    month_iso: str = "1970-11",
    status: str = "pending",
    kind: str = "gap",
    character_id: str | None = None,
    genre_id: int | None = None,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> None:
    conn.execute(
        """
        INSERT INTO missions
            (mission_id, category, category_id, month_iso, priority,
             coverage_score_before, search_queries, target_sources, rationale,
             status, created_at, updated_at, llm_reformulated, kind, character_id,
             genre_id)
        VALUES (?, 'Política Nacional', ?, ?, 0.5, 0.0, '["q"]', '["openalex"]',
                'r', ?, ?, ?, 0, ?, ?, ?)
        """,
        (
            mission_id,
            category_id,
            month_iso,
            status,
            created_at,
            created_at,
            kind,
            character_id,
            genre_id,
        ),
    )
    conn.commit()


# ── Pure: entity_queries ───────────────────────────────────────────────────────


class TestEntityQueries:
    def test_includes_exact_name_query(self):
        queries = entity_queries("Carlos Prats")
        assert any('"Carlos Prats"' in q for q in queries)

    def test_all_queries_mention_the_person(self):
        queries = entity_queries("Stafford Beer")
        assert queries
        assert all("Stafford Beer" in q for q in queries)

    def test_returns_at_most_three_queries(self):
        assert 1 <= len(entity_queries("Salvador Allende")) <= 3

    def test_salient_fact_sharpens_a_query(self):
        queries = entity_queries("Donald Kendall", salient_fact="president of Pepsi-Cola")
        assert any("Pepsi-Cola" in q for q in queries)
        assert any('"Donald Kendall" president of Pepsi-Cola' in q for q in queries)

    def test_salient_fact_fragment_is_bounded(self):
        long_fact = "president of a very large multinational beverage corporation indeed"
        queries = entity_queries("X Y", salient_fact=long_fact)
        # Only the first handful of words survive into the query fragment.
        assert "indeed" not in " ".join(queries)

    def test_distinct_alias_broadens_recall(self):
        queries = entity_queries(
            "Salvador Allende", aliases=["Salvador Allende Gossens", "Salvador Allende"]
        )
        assert any("Gossens" in q for q in queries)

    def test_alias_equal_to_name_is_ignored(self):
        # An alias that is just a case/space variant must not add a redundant query.
        queries = entity_queries("Carlos Prats", aliases=["carlos  prats"])
        assert any("biografía" in q for q in queries)  # fell back to the bio query


# ── Pure: entity_priority ──────────────────────────────────────────────────────


class TestEntityPriority:
    def test_unknown_well_mentioned_character_is_top_priority(self):
        assert entity_priority(mention_count=5, completeness_score=0.0) == 1.0

    def test_more_mentions_means_higher_priority(self):
        low = entity_priority(mention_count=1, completeness_score=0.0)
        high = entity_priority(mention_count=4, completeness_score=0.0)
        assert high > low

    def test_more_completeness_means_lower_priority(self):
        unknown = entity_priority(mention_count=3, completeness_score=0.0)
        known = entity_priority(mention_count=3, completeness_score=0.8)
        assert known < unknown

    def test_bounded_zero_to_one(self):
        assert 0.0 <= entity_priority(mention_count=0, completeness_score=1.0) <= 1.0
        assert 0.0 <= entity_priority(mention_count=99, completeness_score=0.0) <= 1.0


# ── Pure: build_entity_missions ────────────────────────────────────────────────


def _char_row(character_id: str, name: str, mentions: int = 2, completeness: float = 0.1) -> dict:
    return {
        "character_id": character_id,
        "name": name,
        "mention_count": mentions,
        "completeness_score": completeness,
    }


class TestBuildEntityMissions:
    def test_builds_entity_mission_fields(self):
        missions = build_entity_missions([_char_row("carlos-prats", "Carlos Prats")], set(), 10)
        assert len(missions) == 1
        m = missions[0]
        assert m.kind == MissionKind.ENTITY
        assert m.character_id == "carlos-prats"
        assert m.status == MissionStatus.PENDING
        assert m.search_queries and m.target_sources
        assert m.mission_id.startswith("e-")

    def test_skips_characters_already_queued(self):
        rows = [_char_row("a-b", "A B"), _char_row("c-d", "C D")]
        missions = build_entity_missions(rows, {"a-b"}, 10)
        assert [m.character_id for m in missions] == ["c-d"]

    def test_respects_limit(self):
        rows = [_char_row(f"p-{i}", f"P {i}") for i in range(5)]
        missions = build_entity_missions(rows, set(), 2)
        assert len(missions) == 2

    def test_preserves_input_order(self):
        rows = [_char_row("first", "First"), _char_row("second", "Second")]
        missions = build_entity_missions(rows, set(), 10)
        assert [m.character_id for m in missions] == ["first", "second"]


# ── SQLite: missions schema migration ──────────────────────────────────────────


class TestMissionsMigration:
    def test_fresh_table_has_kind_and_character_id(self, mem_conn):
        _ensure_missions_table(mem_conn)
        cols = {r[1] for r in mem_conn.execute("PRAGMA table_info(missions)").fetchall()}
        assert "kind" in cols
        assert "character_id" in cols

    def test_legacy_table_is_migrated_with_gap_default(self, mem_conn):
        mem_conn.execute("""
            CREATE TABLE missions (
                mission_id TEXT PRIMARY KEY, category TEXT NOT NULL,
                category_id INTEGER NOT NULL, month_iso TEXT NOT NULL,
                priority REAL NOT NULL, coverage_score_before REAL NOT NULL,
                search_queries TEXT NOT NULL, target_sources TEXT NOT NULL,
                rationale TEXT NOT NULL, deadline TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                llm_reformulated INTEGER NOT NULL DEFAULT 0
            )
        """)
        mem_conn.execute("""
            INSERT INTO missions (mission_id, category, category_id, month_iso, priority,
                coverage_score_before, search_queries, target_sources, rationale,
                status, created_at, updated_at)
            VALUES ('m-old', 'Salud', 2, '1971-03', 0.5, 0.0, '["q"]', '["openalex"]',
                    'r', 'pending', '2026-01-01', '2026-01-01')
        """)
        _ensure_missions_table(mem_conn)
        row = mem_conn.execute(
            "SELECT kind, character_id FROM missions WHERE mission_id='m-old'"
        ).fetchone()
        assert row["kind"] == "gap"
        assert row["character_id"] is None


# ── SQLite: queue reads are kind-aware ─────────────────────────────────────────


class TestQueueReads:
    def test_entity_missions_do_not_block_gap_cells(self, mem_conn):
        _ensure_missions_table(mem_conn)
        _insert_mission_row(
            mem_conn,
            "e-1",
            category_id=1,
            month_iso="1970-11",
            kind="entity",
            character_id="carlos-prats",
        )
        assert not any(
            cell[0] == 1 and cell[2] == "1970-11" for cell in _read_queued_cells(mem_conn)
        )

    def test_gap_missions_block_their_3d_cell(self, mem_conn):
        _ensure_missions_table(mem_conn)
        _insert_mission_row(
            mem_conn, "m-1", category_id=3, month_iso="1972-10", kind="gap", genre_id=6
        )
        assert (3, 6, "1972-10") in _read_queued_cells(mem_conn)
        # Same theme+month, other genre: not blocked.
        assert (3, 1, "1972-10") not in _read_queued_cells(mem_conn)

    def test_legacy_gap_mission_without_genre_maps_to_genre_zero(self, mem_conn):
        _ensure_missions_table(mem_conn)
        _insert_mission_row(mem_conn, "m-old", category_id=3, month_iso="1972-10", kind="gap")
        assert (3, 0, "1972-10") in _read_queued_cells(mem_conn)

    def test_queued_characters_returned(self, mem_conn):
        _ensure_missions_table(mem_conn)
        _insert_mission_row(mem_conn, "e-1", kind="entity", character_id="carlos-prats")
        _insert_mission_row(mem_conn, "m-1", category_id=2, month_iso="1971-01", kind="gap")
        assert _read_queued_characters(mem_conn) == {"carlos-prats"}

    def test_only_in_flight_entity_missions_suppress_remint(self, mem_conn):
        # A done/failed mission must NOT count — the character can be researched again.
        _ensure_missions_table(mem_conn)
        _insert_mission_row(
            mem_conn, "e-done", kind="entity", character_id="done-char", status="done"
        )
        _insert_mission_row(
            mem_conn, "e-fail", kind="entity", character_id="fail-char", status="failed"
        )
        _insert_mission_row(
            mem_conn, "e-run", kind="entity", character_id="live-char", status="running"
        )
        assert _read_queued_characters(mem_conn) == {"live-char"}


# ── SQLite: duplicate cleanup is kind-aware ────────────────────────────────────


class TestCleanupDuplicatesEntityAware:
    def test_entity_missions_for_different_characters_both_survive(self, mem_conn):
        _ensure_missions_table(mem_conn)
        # Same (category_id, month_iso) — must NOT be collapsed across characters.
        _insert_mission_row(mem_conn, "e-1", kind="entity", character_id="carlos-prats")
        _insert_mission_row(mem_conn, "e-2", kind="entity", character_id="stafford-beer")
        deleted = _cleanup_duplicates(mem_conn)
        assert deleted == 0
        remaining = {r[0] for r in mem_conn.execute("SELECT mission_id FROM missions")}
        assert remaining == {"e-1", "e-2"}

    def test_newest_entity_mission_supersedes_older_for_same_character(self, mem_conn):
        # A fresh research attempt (newer created_at) must win over the character's
        # earlier done mission, so the research loop can iterate.
        _ensure_missions_table(mem_conn)
        _insert_mission_row(
            mem_conn,
            "e-1",
            kind="entity",
            character_id="carlos-prats",
            status="done",
            created_at="2026-01-01T00:00:00+00:00",
        )
        _insert_mission_row(
            mem_conn,
            "e-2",
            kind="entity",
            character_id="carlos-prats",
            status="pending",
            created_at="2026-02-01T00:00:00+00:00",
        )
        deleted = _cleanup_duplicates(mem_conn)
        assert deleted == 1
        remaining = {r[0] for r in mem_conn.execute("SELECT mission_id FROM missions")}
        assert remaining == {"e-2"}  # newest attempt wins

    def test_gap_missions_unaffected_by_entity_rows_on_same_cell(self, mem_conn):
        _ensure_missions_table(mem_conn)
        _insert_mission_row(mem_conn, "m-1", category_id=1, month_iso="1970-11", kind="gap")
        _insert_mission_row(
            mem_conn,
            "e-1",
            category_id=1,
            month_iso="1970-11",
            kind="entity",
            character_id="carlos-prats",
        )
        assert _cleanup_duplicates(mem_conn) == 0


# ── Integration: run_cycle mints entity missions ───────────────────────────────


class TestRunCycleEntityMissions:
    def _make_propositor(self, tmp_path):
        return Propositor(
            db_path=tmp_path / "archivo.sqlite",
            missions_json=tmp_path / "missions.json",
            use_llm=False,
        )

    def _seed_characters(self, db_path):
        conn = sqlite3.connect(str(db_path))
        ensure_cast_tables(conn)
        _insert_character(conn, "carlos-prats", "Carlos Prats", mention_count=4)
        _insert_character(conn, "stafford-beer", "Stafford Beer", mention_count=2)
        conn.close()

    def test_mints_entity_missions_and_clears_flags(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        self._seed_characters(propositor.db_path)

        missions = asyncio.run(propositor.run_cycle(max_new_missions=0))

        entity = [m for m in missions if m.kind == MissionKind.ENTITY]
        assert {m.character_id for m in entity} == {"carlos-prats", "stafford-beer"}

        conn = sqlite3.connect(str(propositor.db_path))
        conn.row_factory = sqlite3.Row
        flagged = conn.execute("SELECT COUNT(*) FROM characters WHERE needs_research=1").fetchone()[
            0
        ]
        assert flagged == 0
        db_kinds = {
            r["character_id"]: r["kind"]
            for r in conn.execute("SELECT character_id, kind FROM missions")
        }
        conn.close()
        assert db_kinds == {"carlos-prats": "entity", "stafford-beer": "entity"}

    def test_minting_records_a_research_attempt(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        self._seed_characters(propositor.db_path)

        asyncio.run(propositor.run_cycle(max_new_missions=0))

        conn = sqlite3.connect(str(propositor.db_path))
        conn.row_factory = sqlite3.Row
        rows = {
            r["character_id"]: (r["research_attempts"], r["last_research_at"])
            for r in conn.execute(
                "SELECT character_id, research_attempts, last_research_at FROM characters"
            )
        }
        conn.close()
        for cid in ("carlos-prats", "stafford-beer"):
            attempts, last = rows[cid]
            assert attempts == 1  # one mission minted = one attempt
            assert last is not None  # stamped so the cooldown can space attempts

    def test_second_cycle_does_not_duplicate_entity_missions(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        self._seed_characters(propositor.db_path)

        asyncio.run(propositor.run_cycle(max_new_missions=0))
        second = asyncio.run(propositor.run_cycle(max_new_missions=0))

        assert [m for m in second if m.kind == MissionKind.ENTITY] == []
        conn = sqlite3.connect(str(propositor.db_path))
        count = conn.execute("SELECT COUNT(*) FROM missions WHERE kind='entity'").fetchone()[0]
        conn.close()
        assert count == 2

    def test_entity_limit_leaves_overflow_flagged_for_next_cycle(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        self._seed_characters(propositor.db_path)

        missions = asyncio.run(propositor.run_cycle(max_new_missions=0, max_entity_missions=1))

        entity = [m for m in missions if m.kind == MissionKind.ENTITY]
        # Most-mentioned character first (queue is ordered by mention_count DESC).
        assert [m.character_id for m in entity] == ["carlos-prats"]

        conn = sqlite3.connect(str(propositor.db_path))
        conn.row_factory = sqlite3.Row
        flagged = [
            r["character_id"]
            for r in conn.execute("SELECT character_id FROM characters WHERE needs_research=1")
        ]
        conn.close()
        assert flagged == ["stafford-beer"]

    def test_entity_limit_zero_disables_entity_missions(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        self._seed_characters(propositor.db_path)

        missions = asyncio.run(propositor.run_cycle(max_new_missions=0, max_entity_missions=0))

        assert missions == []
        conn = sqlite3.connect(str(propositor.db_path))
        flagged = conn.execute("SELECT COUNT(*) FROM characters WHERE needs_research=1").fetchone()[
            0
        ]
        conn.close()
        assert flagged == 2  # untouched — queue intact for a later cycle

    def test_no_characters_table_is_harmless(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        missions = asyncio.run(propositor.run_cycle(max_new_missions=0))
        assert [m for m in missions if m.kind == MissionKind.ENTITY] == []

    def test_snapshot_includes_kind_and_character_id(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        self._seed_characters(propositor.db_path)

        asyncio.run(propositor.run_cycle(max_new_missions=0))

        snapshot = json.loads(propositor.missions_json.read_text(encoding="utf-8"))
        entity = [m for m in snapshot if m["kind"] == "entity"]
        assert {m["character_id"] for m in entity} == {"carlos-prats", "stafford-beer"}


# ── 3D gap missions (theme × genre × month) ────────────────────────────────────


class TestGenreAwareGapMissions:
    def test_cleanup_keeps_gap_missions_for_different_genres_on_same_cell(self, mem_conn):
        _ensure_missions_table(mem_conn)
        _insert_mission_row(
            mem_conn, "m-1", category_id=3, month_iso="1972-10", kind="gap", genre_id=6
        )
        _insert_mission_row(
            mem_conn, "m-2", category_id=3, month_iso="1972-10", kind="gap", genre_id=1
        )
        assert _cleanup_duplicates(mem_conn) == 0

    def test_cleanup_collapses_same_3d_cell(self, mem_conn):
        _ensure_missions_table(mem_conn)
        _insert_mission_row(
            mem_conn,
            "m-1",
            category_id=3,
            month_iso="1972-10",
            kind="gap",
            genre_id=6,
            status="done",
        )
        _insert_mission_row(
            mem_conn,
            "m-2",
            category_id=3,
            month_iso="1972-10",
            kind="gap",
            genre_id=6,
            status="pending",
        )
        assert _cleanup_duplicates(mem_conn) == 1
        remaining = {r[0] for r in mem_conn.execute("SELECT mission_id FROM missions")}
        assert remaining == {"m-1"}

    def test_migration_adds_genre_id(self, mem_conn):
        _ensure_missions_table(mem_conn)
        cols = {r[1] for r in mem_conn.execute("PRAGMA table_info(missions)").fetchall()}
        assert "genre_id" in cols

    def test_run_cycle_mints_genre_missions_with_plausible_cells_first(self, tmp_path):
        propositor = Propositor(
            db_path=tmp_path / "archivo.sqlite",
            missions_json=tmp_path / "missions.json",
            use_llm=False,
        )
        missions = asyncio.run(propositor.run_cycle(max_new_missions=5, max_entity_missions=0))
        assert len(missions) == 5
        for m in missions:
            assert m.kind == MissionKind.GAP
            assert m.genre_id is not None and 1 <= m.genre_id <= 13
            assert m.search_queries
            assert m.target_sources
        # On an empty DB every score is 0, so priorities reduce to
        # weight × plausibility; the top picks must be high-plausibility pairings.
        from lib.genres import plausibility as genre_plausibility

        for m in missions:
            assert genre_plausibility(m.category_id, m.genre_id) >= 0.9

    def test_run_cycle_routes_sources_genre_first(self, tmp_path):
        from lib.genres import genre_by_id

        propositor = Propositor(
            db_path=tmp_path / "archivo.sqlite",
            missions_json=tmp_path / "missions.json",
            use_llm=False,
        )
        missions = asyncio.run(propositor.run_cycle(max_new_missions=3, max_entity_missions=0))
        for m in missions:
            genre_sources = genre_by_id(m.genre_id).sources
            assert m.target_sources[: len(genre_sources)] == genre_sources

    def test_snapshot_includes_genre_id(self, tmp_path):
        import json as _json

        propositor = Propositor(
            db_path=tmp_path / "archivo.sqlite",
            missions_json=tmp_path / "missions.json",
            use_llm=False,
        )
        asyncio.run(propositor.run_cycle(max_new_missions=2, max_entity_missions=0))
        snapshot = _json.loads(propositor.missions_json.read_text(encoding="utf-8"))
        assert all("genre_id" in m for m in snapshot)
        assert all(m["genre_id"] is not None for m in snapshot if m["kind"] == "gap")

    def test_second_cycle_skips_queued_3d_cells(self, tmp_path):
        propositor = Propositor(
            db_path=tmp_path / "archivo.sqlite",
            missions_json=tmp_path / "missions.json",
            use_llm=False,
        )
        first = asyncio.run(propositor.run_cycle(max_new_missions=4, max_entity_missions=0))
        second = asyncio.run(propositor.run_cycle(max_new_missions=4, max_entity_missions=0))
        first_cells = {(m.category_id, m.genre_id, m.month_iso) for m in first}
        second_cells = {(m.category_id, m.genre_id, m.month_iso) for m in second}
        assert not first_cells & second_cells


# ── Focus modes: --focus cells | characters | mixed (F3) ────────────────────────


class TestOrderByNeed:
    def test_least_complete_first_ties_broken_by_mentions(self):
        rows = [
            {"completeness_score": 0.7, "mention_count": 9},
            {"completeness_score": 0.1, "mention_count": 2},
            {"completeness_score": 0.1, "mention_count": 5},
        ]
        ordered = order_by_need(rows)
        assert [r["completeness_score"] for r in ordered] == [0.1, 0.1, 0.7]
        assert ordered[0]["mention_count"] == 5  # tie → most-mentioned first

    def test_tolerates_null_fields(self):
        rows = [{"completeness_score": None, "mention_count": None}]
        assert order_by_need(rows) == rows


class TestRunCycleFocus:
    def _make_propositor(self, tmp_path):
        return Propositor(
            db_path=tmp_path / "archivo.sqlite",
            missions_json=tmp_path / "missions.json",
            use_llm=False,
        )

    def _seed(self, db_path):
        conn = sqlite3.connect(str(db_path))
        ensure_cast_tables(conn)
        _insert_character(
            conn, "carlos-prats", "Carlos Prats", mention_count=9, completeness_score=0.6
        )
        _insert_character(
            conn, "stafford-beer", "Stafford Beer", mention_count=2, completeness_score=0.1
        )
        conn.close()

    def test_focus_characters_skips_gap_missions(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        self._seed(propositor.db_path)

        missions = asyncio.run(propositor.run_cycle(max_new_missions=50, focus="characters"))

        assert missions  # entity missions were minted
        assert all(m.kind == MissionKind.ENTITY for m in missions)

    def test_focus_characters_chases_the_thinnest_first(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        self._seed(propositor.db_path)

        missions = asyncio.run(
            propositor.run_cycle(max_new_missions=0, max_entity_missions=1, focus="characters")
        )

        # Default queue order is mention_count DESC (prats); focus flips to
        # least-complete-first (beer at 0.1).
        assert [m.character_id for m in missions] == ["stafford-beer"]

    def test_focus_cells_leaves_the_research_queue_alone(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        self._seed(propositor.db_path)

        missions = asyncio.run(propositor.run_cycle(max_new_missions=2, focus="cells"))

        assert all(m.kind == MissionKind.GAP for m in missions)
        conn = sqlite3.connect(str(propositor.db_path))
        flagged = conn.execute("SELECT COUNT(*) FROM characters WHERE needs_research=1").fetchone()[
            0
        ]
        conn.close()
        assert flagged == 2  # untouched — still queued for a characters run


# ── Location missions (Location Manager wiring) ────────────────────────────────


def _insert_location(conn, location_id, name, kind="building", mentions=2, completeness=0.1):
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO locations (location_id, name, kind, mention_count, "
        "completeness_score, needs_research, first_seen_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
        (location_id, name, kind, mentions, completeness, now, now),
    )
    conn.commit()


class TestLocationQueries:
    def test_identity_and_kind_flavoured_queries(self):
        queries = location_queries("Palacio de La Moneda", kind="building")
        assert any(q == '"Palacio de La Moneda" Chile' for q in queries)
        assert any("arquitectura" in q and "planos" in q for q in queries)

    def test_factory_kind_uses_industrial_terms(self):
        queries = location_queries("Fábrica de harina de Temuco", kind="factory")
        assert any("industria" in q or "producción" in q for q in queries)

    def test_at_most_three_queries(self):
        assert 1 <= len(location_queries("La Moneda")) <= 3


class TestBuildLocationMissions:
    def test_builds_location_mission_fields(self):
        rows = [
            {
                "location_id": "la-moneda",
                "name": "La Moneda",
                "kind": "building",
                "mention_count": 3,
                "completeness_score": 0.2,
                "aliases": "[]",
            }
        ]
        missions = build_location_missions(rows, set(), 10)
        assert len(missions) == 1
        m = missions[0]
        assert m.kind == MissionKind.LOCATION
        assert m.location_id == "la-moneda"
        assert m.mission_id.startswith("l-")
        assert m.search_queries

    def test_skips_queued_and_respects_limit(self):
        rows = [
            {
                "location_id": f"l-{i}",
                "name": f"Lugar {i}",
                "kind": "other",
                "mention_count": 1,
                "completeness_score": 0.0,
                "aliases": "[]",
            }
            for i in range(5)
        ]
        missions = build_location_missions(rows, {"l-0"}, 2)
        assert [m.location_id for m in missions] == ["l-1", "l-2"]


class TestRunCycleLocationMissions:
    def _make_propositor(self, tmp_path):
        return Propositor(
            db_path=tmp_path / "archivo.sqlite",
            missions_json=tmp_path / "missions.json",
            use_llm=False,
        )

    def _seed(self, db_path):
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        ensure_location_tables(conn)
        _insert_location(conn, "la-moneda", "La Moneda", kind="building", mentions=3)
        conn.close()

    def test_mints_location_missions_and_clears_flags(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        self._seed(propositor.db_path)

        missions = asyncio.run(propositor.run_cycle(max_new_missions=0, max_entity_missions=0))

        located = [m for m in missions if m.kind == MissionKind.LOCATION]
        assert [m.location_id for m in located] == ["la-moneda"]

        conn = sqlite3.connect(str(propositor.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM locations").fetchone()
        assert row["needs_research"] == 0
        assert row["research_attempts"] == 1
        db_row = conn.execute("SELECT kind, location_id FROM missions").fetchone()
        conn.close()
        assert db_row["kind"] == "location"
        assert db_row["location_id"] == "la-moneda"

    def test_second_cycle_does_not_duplicate(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        self._seed(propositor.db_path)
        asyncio.run(propositor.run_cycle(max_new_missions=0, max_entity_missions=0))
        second = asyncio.run(propositor.run_cycle(max_new_missions=0, max_entity_missions=0))
        assert [m for m in second if m.kind == MissionKind.LOCATION] == []

    def test_characters_focus_stays_people_only(self, tmp_path):
        propositor = self._make_propositor(tmp_path)
        self._seed(propositor.db_path)
        missions = asyncio.run(
            propositor.run_cycle(max_new_missions=0, max_entity_missions=0, focus="characters")
        )
        assert [m for m in missions if m.kind == MissionKind.LOCATION] == []
        conn = sqlite3.connect(str(propositor.db_path))
        flagged = conn.execute("SELECT COUNT(*) FROM locations WHERE needs_research=1").fetchone()[
            0
        ]
        conn.close()
        assert flagged == 1  # queue untouched
