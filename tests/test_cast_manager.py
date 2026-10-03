"""Tests for the Cast Manager agent (Ag-7) — people extraction from documents.

Fully offline: pure-function and SQLite-helper tests run without any LLM, and the
run_cycle LLM path is exercised with a fake Ollama client. No live Ollama, no HTTP.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

import pipeline.cast_manager as cm
from pipeline.cast_manager import (
    CastManager,
    CompletenessInputs,
    build_wikidata_facts,
    character_key,
    compute_completeness,
    dedupe_timeline_facts,
    ensure_cast_tables,
    facts_are_duplicates,
    fetch_characters_needing_research,
    heuristic_extract_people,
    merge_duplicate_characters,
    parse_analysis_response,
    parse_people_response,
    parse_wikidata_claims,
    parse_wikidata_labels,
    parse_wikipedia_extract,
    refresh_research_flags,
    resolve_character_key,
    should_research,
    wikidata_claims_url,
    wikidata_identity_plausible,
    wikidata_labels_url,
    wikipedia_extract_url,
)

# ── Pure: character_key (dedup / coreference) ──────────────────────────────────


class TestCharacterKey:
    def test_basic_slug(self):
        assert character_key("Salvador Allende") == "salvador-allende"

    def test_strips_accents(self):
        assert character_key("Augusto Pinochet Ugarte") == "augusto-pinochet-ugarte"
        assert character_key("Aníbal Palma") == "anibal-palma"

    def test_case_and_whitespace_collapse(self):
        # Different surface forms must collapse to the same key.
        assert character_key("salvador  ALLENDE") == character_key("Salvador Allende")

    def test_punctuation_becomes_single_hyphen(self):
        assert character_key("Carlos Prats González.") == "carlos-prats-gonzalez"

    def test_empty_for_nonalphanumeric(self):
        assert character_key("—¿?—") == ""
        assert character_key("   ") == ""


# ── Pure: compute_completeness ─────────────────────────────────────────────────


def _full_inputs(**overrides) -> CompletenessInputs:
    """A character that satisfies every component and gate (score 1.0)."""
    base = dict(
        mention_count=8,
        fact_count=12,
        fact_kinds=frozenset({"role", "event", "statement", "rumor"}),
        dated_months=6,
        has_bio=True,
        has_vital_dates=True,
        wikidata_linked=True,
        wikidata_analyzed=True,
        wikipedia_linked=True,
        wikipedia_analyzed=True,
        entity_missions_done=1,
    )
    base.update(overrides)
    return CompletenessInputs(**base)


class TestCompleteness:
    def test_zero_when_nothing_known(self):
        result = compute_completeness(CompletenessInputs())
        assert result.score == 0.0
        # every gate is active on an empty character
        assert set(result.detail["caps"]) == {
            "no_external_analysis",
            "low_kind_diversity",
            "no_research_mission",
        }

    def test_full_character_reaches_one(self):
        result = compute_completeness(_full_inputs())
        assert result.score == 1.0
        assert result.detail["caps"] == []

    def test_legacy_rich_profile_no_longer_scores_high(self):
        # The old metric gave 1.0 to bio + 5 facts + 3 mentions. Now, without
        # external analysis, diversity or a research mission, it stays under 0.6.
        result = compute_completeness(
            CompletenessInputs(
                mention_count=3,
                fact_count=5,
                fact_kinds=frozenset({"role", "affiliation", "other"}),
                dated_months=2,
                has_bio=True,
            )
        )
        assert result.score < 0.6

    def test_gate_no_external_analysis_caps_at_060(self):
        # Linked but never analyzed: raw sum is high, the gate still bites.
        result = compute_completeness(
            _full_inputs(wikidata_analyzed=False, wikipedia_analyzed=False)
        )
        assert result.score == 0.6
        assert "no_external_analysis" in result.detail["caps"]

    def test_one_analyzed_source_lifts_the_external_gate(self):
        result = compute_completeness(_full_inputs(wikipedia_analyzed=False))
        assert "no_external_analysis" not in result.detail["caps"]
        assert result.score > 0.6

    def test_gate_low_kind_diversity_caps_at_075(self):
        # Only information-group kinds (role/affiliation/other collapse to one group).
        result = compute_completeness(
            _full_inputs(fact_kinds=frozenset({"role", "affiliation", "other"}))
        )
        assert result.score == 0.75
        assert "low_kind_diversity" in result.detail["caps"]

    def test_three_of_four_groups_passes_the_diversity_gate(self):
        result = compute_completeness(
            _full_inputs(fact_kinds=frozenset({"role", "event", "statement"}))
        )
        assert "low_kind_diversity" not in result.detail["caps"]

    def test_gate_no_research_mission_caps_at_080(self):
        result = compute_completeness(_full_inputs(entity_missions_done=0))
        assert result.score == 0.8
        assert "no_research_mission" in result.detail["caps"]

    def test_linked_only_earns_less_than_analyzed(self):
        linked = compute_completeness(
            CompletenessInputs(wikidata_linked=True, wikipedia_linked=True)
        )
        analyzed = compute_completeness(
            CompletenessInputs(
                wikidata_linked=True,
                wikidata_analyzed=True,
                wikipedia_linked=True,
                wikipedia_analyzed=True,
            )
        )
        assert 0.0 < linked.detail["external"] < analyzed.detail["external"]
        assert analyzed.detail["external"] == pytest.approx(0.25)

    def test_detail_carries_all_components(self):
        result = compute_completeness(_full_inputs())
        for key in ("identity", "documents", "external", "timeline", "pipeline", "raw", "score"):
            assert key in result.detail
        assert result.detail["score"] == result.score

    def test_saturates_not_exceeds_one(self):
        result = compute_completeness(
            _full_inputs(mention_count=99, fact_count=99, dated_months=48, entity_missions_done=7)
        )
        assert result.score == 1.0


# ── Pure: parse_people_response ────────────────────────────────────────────────


class TestParsePeople:
    def test_parses_people_with_facts(self):
        raw = json.dumps(
            {
                "people": [
                    {
                        "name": "Salvador Allende",
                        "mentioned_by": "Pablo Neruda",
                        "facts": [
                            {
                                "kind": "role",
                                "description": "elected President",
                                "date_iso": "1970-11",
                            }
                        ],
                    },
                ]
            }
        )
        people = parse_people_response(raw)
        assert len(people) == 1
        p = people[0]
        assert p.name == "Salvador Allende"
        assert p.mentioned_by == "Pablo Neruda"
        assert p.facts == [
            {
                "kind": "role",
                "description": "elected President",
                "date_iso": "1970-11",
                "speech_act": None,
                "reported_by": None,
                "confidence": None,
            }
        ]

    def test_empty_people_list_returns_empty(self):
        assert parse_people_response('{"people": []}') == []

    def test_missing_people_key_returns_none(self):
        assert parse_people_response('{"foo": 1}') is None

    def test_non_object_returns_none(self):
        assert parse_people_response("[1, 2, 3]") is None

    def test_malformed_json_returns_none(self):
        assert parse_people_response("not json") is None

    def test_skips_nameless_person_but_keeps_valid(self):
        raw = json.dumps(
            {
                "people": [
                    {"name": "", "facts": []},
                    {"name": "Orlando Letelier", "facts": []},
                ]
            }
        )
        people = parse_people_response(raw)
        assert [p.name for p in people] == ["Orlando Letelier"]

    def test_invalid_fact_kind_defaults_to_other(self):
        raw = json.dumps(
            {
                "people": [
                    {"name": "X Y", "facts": [{"kind": "nonsense", "description": "did a thing"}]},
                ]
            }
        )
        people = parse_people_response(raw)
        assert people[0].facts[0]["kind"] == "other"

    def test_fact_without_description_is_dropped(self):
        raw = json.dumps(
            {
                "people": [
                    {"name": "X Y", "facts": [{"kind": "role"}, {"description": "valid one"}]},
                ]
            }
        )
        people = parse_people_response(raw)
        assert len(people[0].facts) == 1
        assert people[0].facts[0]["description"] == "valid one"

    def test_blank_mentioned_by_becomes_none(self):
        raw = json.dumps({"people": [{"name": "X Y", "mentioned_by": "  ", "facts": []}]})
        assert parse_people_response(raw)[0].mentioned_by is None


# ── Pure: heuristic_extract_people ─────────────────────────────────────────────


class TestHeuristic:
    def test_extracts_capitalised_names(self):
        text = "El presidente Salvador Allende recibió a Pablo Neruda en Santiago."
        names = {p.name for p in heuristic_extract_people(text)}
        assert "Salvador Allende" in names
        assert "Pablo Neruda" in names

    def test_dedupes_by_key(self):
        text = "Salvador Allende habló. Salvador Allende insistió."
        people = heuristic_extract_people(text)
        assert sum(p.name == "Salvador Allende" for p in people) == 1

    def test_no_facts_or_mentions(self):
        people = heuristic_extract_people("Carlos Altamirano lideró el partido.")
        assert all(p.facts == [] and p.mentioned_by is None for p in people)


# ── DB fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path):
    """A temp archive DB with a documents table holding two verified+mapped docs."""
    path = tmp_path / "archivo.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            text TEXT,
            verified_at TEXT,
            mapped_category_id INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO documents (doc_id, title, text, verified_at, mapped_category_id) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (
                "d1",
                "Discurso",
                "El presidente Salvador Allende habló ante Pablo Neruda.",
                "2026-01-01T00:00:00Z",
                1,
            ),
            ("d2", "Memo", "Carlos Prats asumió el mando.", "2026-01-01T00:00:00Z", 13),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _conn(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


# ── ensure_cast_tables ─────────────────────────────────────────────────────────


def test_ensure_cast_tables_creates_all_three():
    conn = sqlite3.connect(":memory:")
    ensure_cast_tables(conn)
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"characters", "character_timeline", "character_mentions"} <= tables


def test_ensure_cast_tables_idempotent():
    conn = sqlite3.connect(":memory:")
    ensure_cast_tables(conn)
    ensure_cast_tables(conn)  # must not raise


# ── run_cycle: heuristic (--no-llm) path ───────────────────────────────────────


class TestRunCycleHeuristic:
    async def test_extracts_and_persists_characters(self, db):
        manager = CastManager(db_path=db, use_llm=False)
        result = await manager.run_cycle()
        assert result.processed == 2
        assert result.characters_new >= 3  # Allende, Neruda, Prats

        conn = _conn(db)
        keys = {r["character_id"] for r in conn.execute("SELECT character_id FROM characters")}
        assert "salvador-allende" in keys
        assert "carlos-prats" in keys

    async def test_stamps_documents_extracted(self, db):
        await CastManager(db_path=db, use_llm=False).run_cycle()
        conn = _conn(db)
        pending = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE cast_extracted_at IS NULL"
        ).fetchone()[0]
        assert pending == 0

    async def test_rerun_is_idempotent(self, db):
        await CastManager(db_path=db, use_llm=False).run_cycle()
        second = await CastManager(db_path=db, use_llm=False).run_cycle()
        # All docs already stamped → nothing to process.
        assert second.processed == 0
        assert second.characters_new == 0

    async def test_new_characters_flagged_needs_research(self, db):
        await CastManager(db_path=db, use_llm=False).run_cycle()
        conn = _conn(db)
        queue = fetch_characters_needing_research(conn)
        assert {r["character_id"] for r in queue} >= {"salvador-allende", "carlos-prats"}

    def test_research_queue_carries_aliases_and_salient_fact(self, tmp_path):
        # The Propositor seeds queries on the character's aliases + affiliation, so
        # the queue must expose them.
        path = tmp_path / "a.sqlite"
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        now = "2026-01-01T00:00:00Z"
        conn.execute(
            "INSERT INTO characters (character_id, name, aliases, needs_research, "
            "mention_count, first_seen_at, updated_at) VALUES "
            "('donald-kendall','Donald Kendall','[\"Don Kendall\"]',1,2,?,?)",
            (now, now),
        )
        conn.execute(
            "INSERT INTO character_timeline (character_id, doc_id, kind, description, created_at) "
            "VALUES ('donald-kendall','d1','affiliation','president of Pepsi-Cola',?)",
            (now,),
        )
        conn.commit()
        row = fetch_characters_needing_research(conn)[0]
        assert row["salient_fact"] == "president of Pepsi-Cola"
        assert "Don Kendall" in row["aliases"]

    async def test_only_verified_mapped_docs_are_mined(self, tmp_path):
        path = tmp_path / "archivo.sqlite"
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE documents (doc_id TEXT PRIMARY KEY, title TEXT, text TEXT, "
            "verified_at TEXT, mapped_category_id INTEGER)"
        )
        conn.executemany(
            "INSERT INTO documents VALUES (?,?,?,?,?)",
            [
                ("unverified", "t", "Pedro Vuskovic habló.", None, 3),  # not verified
                ("unmapped", "t", "Clodomiro Almeyda habló.", "2026-01-01", None),  # not mapped
                ("good", "t", "Jose Toha asumió.", "2026-01-01", 1),  # eligible
            ],
        )
        conn.commit()
        conn.close()

        await CastManager(db_path=path, use_llm=False).run_cycle()
        conn = _conn(path)
        keys = {r["character_id"] for r in conn.execute("SELECT character_id FROM characters")}
        assert "jose-toha" in keys
        assert "pedro-vuskovic" not in keys
        assert "clodomiro-almeyda" not in keys

    async def test_missing_db_returns_empty_result(self, tmp_path):
        result = await CastManager(db_path=tmp_path / "nope.sqlite", use_llm=False).run_cycle()
        assert result.processed == 0


# ── run_cycle: LLM path with a fake Ollama client ──────────────────────────────


class _FakeOllama:
    """Stand-in for OllamaClient: returns a canned chat response, reports the model present."""

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


class TestRunCycleLLM:
    async def test_persists_mentions_and_facts(self, db, monkeypatch):
        response = json.dumps(
            {
                "people": [
                    {
                        "name": "Salvador Allende",
                        "mentioned_by": "Pablo Neruda",
                        "facts": [
                            {
                                "kind": "role",
                                "description": "elected President",
                                "date_iso": "1970-11",
                            }
                        ],
                    },
                ]
            }
        )
        monkeypatch.setattr(cm, "LLMClient", lambda: _FakeOllama(response))

        result = await CastManager(db_path=db, use_llm=True).run_cycle()
        assert result.processed == 2
        assert result.mentions_new >= 1
        assert result.facts_new >= 1

        conn = _conn(db)
        mention = conn.execute(
            "SELECT mentioned_by FROM character_mentions WHERE character_id='salvador-allende' LIMIT 1"
        ).fetchone()
        assert mention["mentioned_by"] == "Pablo Neruda"
        fact = conn.execute(
            "SELECT description, kind, date_iso FROM character_timeline "
            "WHERE character_id='salvador-allende' LIMIT 1"
        ).fetchone()
        assert fact["description"] == "elected President"
        assert fact["kind"] == "role"
        assert fact["date_iso"] == "1970-11"

    async def test_completeness_score_reflects_evidence(self, db, monkeypatch):
        response = json.dumps(
            {
                "people": [
                    {
                        "name": "Salvador Allende",
                        "mentioned_by": None,
                        "facts": [
                            {
                                "kind": "role",
                                "description": "elected President",
                                "date_iso": "1970-11",
                            }
                        ],
                    },
                ]
            }
        )
        monkeypatch.setattr(cm, "LLMClient", lambda: _FakeOllama(response))
        await CastManager(db_path=db, use_llm=True).run_cycle()

        conn = _conn(db)
        score = conn.execute(
            "SELECT completeness_score FROM characters WHERE character_id='salvador-allende'"
        ).fetchone()[0]
        assert score > 0.0  # at least one mention + one fact

    async def test_unparseable_llm_falls_back_to_cued_people(self, db, monkeypatch):
        # When the model returns garbage, the cued backstop still recovers the
        # obvious people ("presidente Salvador Allende") and every doc is stamped —
        # only cued names, never hallucinations.
        monkeypatch.setattr(cm, "LLMClient", lambda: _FakeOllama("totally not json"))
        result = await CastManager(db_path=db, use_llm=True).run_cycle()
        assert result.processed == 2
        conn = _conn(db)
        ids = {r["character_id"] for r in conn.execute("SELECT character_id FROM characters")}
        pending = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE cast_extracted_at IS NULL"
        ).fetchone()[0]
        conn.close()
        assert "salvador-allende" in ids  # cued by the "presidente ..." title
        assert pending == 0


# ── Character research loop: should_research + refresh_research_flags ───────────


class TestShouldResearch:
    NOW = datetime(2026, 7, 9, tzinfo=UTC)

    def test_thin_and_untried_is_researched(self):
        assert should_research(0.2, 0, None, self.NOW) is True

    def test_complete_character_is_not_researched(self):
        assert should_research(0.9, 0, None, self.NOW) is False

    def test_below_complete_threshold_is_still_researched(self):
        # 0.8 passed the old 0.6 bar; the strict metric keeps chasing until 0.85.
        assert should_research(0.8, 0, None, self.NOW) is True

    def test_attempt_cap_stops_research(self):
        assert should_research(0.1, 5, None, self.NOW) is False
        assert should_research(0.1, 4, None, self.NOW) is True

    def test_cooldown_blocks_recent_attempt(self):
        recent = (self.NOW - timedelta(hours=12)).isoformat()
        assert should_research(0.1, 1, recent, self.NOW) is False

    def test_research_resumes_after_cooldown(self):
        old = (self.NOW - timedelta(days=5)).isoformat()
        assert should_research(0.1, 1, old, self.NOW) is True

    def test_malformed_timestamp_ignored(self):
        assert should_research(0.1, 0, "not-a-date", self.NOW) is True


class TestRefreshResearchFlags:
    def _db(self, tmp_path):
        path = tmp_path / "a.sqlite"
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        return conn, path

    def _add(self, conn, cid, completeness, needs, attempts=0, last=None):
        now = "2026-01-01T00:00:00Z"
        conn.execute(
            "INSERT INTO characters (character_id, name, completeness_score, "
            "needs_research, research_attempts, last_research_at, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, cid.title(), completeness, needs, attempts, last, now, now),
        )

    def test_clears_completed_and_flags_thin(self, tmp_path):
        conn, _ = self._db(tmp_path)
        self._add(conn, "complete", 0.9, needs=1)  # done → should clear
        self._add(conn, "thin", 0.2, needs=0)  # thin, untried → should flag
        conn.commit()
        changed = refresh_research_flags(conn, now=datetime(2026, 7, 9, tzinfo=UTC))
        assert changed == 2
        flags = dict(conn.execute("SELECT character_id, needs_research FROM characters").fetchall())
        assert flags["complete"] == 0
        assert flags["thin"] == 1

    def test_respects_attempt_cap_and_cooldown(self, tmp_path):
        conn, _ = self._db(tmp_path)
        now = datetime(2026, 7, 9, tzinfo=UTC)
        self._add(conn, "exhausted", 0.1, needs=1, attempts=5)  # capped → clear
        self._add(
            conn, "cooling", 0.1, needs=0, attempts=1, last=(now - timedelta(hours=6)).isoformat()
        )  # cooldown → stay 0
        conn.commit()
        refresh_research_flags(conn, now=now)
        flags = dict(conn.execute("SELECT character_id, needs_research FROM characters").fetchall())
        assert flags["exhausted"] == 0
        assert flags["cooling"] == 0


class TestRecomputeCompleteness:
    """_recompute_completeness gathers the full strict-metric inputs from SQLite."""

    def _db(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "a.sqlite"))
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        return conn

    def _add_character(self, conn, cid, **cols):
        now = "2026-01-01T00:00:00Z"
        base = {
            "biography": None,
            "birth_date": None,
            "death_date": None,
            "wikidata_id": None,
            "wikipedia_url": None,
            "wikidata_analyzed_at": None,
            "wikipedia_analyzed_at": None,
        }
        base.update(cols)
        conn.execute(
            "INSERT INTO characters (character_id, name, biography, birth_date, death_date, "
            "wikidata_id, wikipedia_url, wikidata_analyzed_at, wikipedia_analyzed_at, "
            "first_seen_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                cid,
                cid.title(),
                base["biography"],
                base["birth_date"],
                base["death_date"],
                base["wikidata_id"],
                base["wikipedia_url"],
                base["wikidata_analyzed_at"],
                base["wikipedia_analyzed_at"],
                now,
                now,
            ),
        )

    def _add_fact(self, conn, cid, kind, description, date_iso=None):
        conn.execute(
            "INSERT INTO character_timeline (character_id, doc_id, date_iso, kind, "
            "description, created_at) VALUES (?, 'd1', ?, ?, ?, '2026-01-01T00:00:00Z')",
            (cid, date_iso, kind, description),
        )

    def test_stores_score_and_detail_json(self, tmp_path):
        conn = self._db(tmp_path)
        self._add_character(
            conn,
            "allende",
            biography="Presidente de Chile.",
            birth_date="1908",
            wikidata_id="Q1",
            wikidata_analyzed_at="2026-01-01T00:00:00Z",
        )
        conn.execute(
            "INSERT INTO character_mentions (character_id, doc_id, created_at) "
            "VALUES ('allende', 'd1', '2026-01-01T00:00:00Z')"
        )
        self._add_fact(conn, "allende", "role", "Presidente", "1970-11")
        self._add_fact(conn, "allende", "event", "Discurso en la UNCTAD", "1972-04-03")
        self._add_fact(conn, "allende", "statement", "Anuncia la nacionalización", "1971-07")
        conn.commit()

        cm._recompute_completeness(conn, "allende")

        row = conn.execute(
            "SELECT completeness_score, completeness_detail, mention_count "
            "FROM characters WHERE character_id='allende'"
        ).fetchone()
        detail = json.loads(row["completeness_detail"])
        assert row["mention_count"] == 1
        assert 0.0 < row["completeness_score"] < 0.85
        assert detail["score"] == row["completeness_score"]
        assert set(detail["kind_groups"]) == {"information", "actions", "statements"}
        # wikidata analyzed → the external-analysis gate must not be active
        assert "no_external_analysis" not in detail["caps"]

    def test_counts_done_entity_missions(self, tmp_path):
        conn = self._db(tmp_path)
        self._add_character(conn, "flores")
        conn.execute(
            "CREATE TABLE missions (mission_id TEXT PRIMARY KEY, kind TEXT, "
            "character_id TEXT, status TEXT)"
        )
        conn.execute("INSERT INTO missions VALUES ('m1', 'entity', 'flores', 'done')")
        conn.commit()

        cm._recompute_completeness(conn, "flores")

        detail = json.loads(
            conn.execute(
                "SELECT completeness_detail FROM characters WHERE character_id='flores'"
            ).fetchone()[0]
        )
        assert "no_research_mission" not in detail["caps"]

    def test_missing_missions_table_is_tolerated(self, tmp_path):
        conn = self._db(tmp_path)
        self._add_character(conn, "prats")
        conn.commit()
        cm._recompute_completeness(conn, "prats")  # must not raise
        score = conn.execute(
            "SELECT completeness_score FROM characters WHERE character_id='prats'"
        ).fetchone()[0]
        assert score == 0.0


# ── Character research loop: force-attribution of entity-harvested docs ─────────


class TestSeedAttribution:
    def _seeded_db(self, tmp_path):
        """DB with a doc harvested by an entity mission for an existing seed."""
        path = tmp_path / "archivo.sqlite"
        conn = sqlite3.connect(str(path))
        conn.execute("""
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY, title TEXT, text TEXT,
                verified_at TEXT, mapped_category_id INTEGER,
                cast_extracted_at TEXT, seed_character_id TEXT
            )
        """)
        # Doc names nobody the LLM will return — the seed is only linked by the mission.
        conn.execute(
            "INSERT INTO documents (doc_id, title, text, verified_at, mapped_category_id, "
            "seed_character_id) VALUES ('d1','Perfil','Un texto sin nombres propios.',"
            "'2026-01-01T00:00:00Z', 1, 'agustin-edwards')"
        )
        ensure_cast_tables(conn)
        now = "2026-01-01T00:00:00Z"
        conn.execute(
            "INSERT INTO characters (character_id, name, first_seen_at, updated_at) "
            "VALUES ('agustin-edwards','Agustín Edwards', ?, ?)",
            (now, now),
        )
        conn.commit()
        conn.close()
        return path

    async def test_seed_gets_mention_even_when_llm_extracts_nobody(self, tmp_path, monkeypatch):
        db = self._seeded_db(tmp_path)
        monkeypatch.setattr(cm, "LLMClient", lambda: _FakeOllama(json.dumps({"people": []})))
        await CastManager(db_path=db, use_llm=True).run_cycle()

        conn = _conn(db)
        mention = conn.execute(
            "SELECT mentioned_by FROM character_mentions "
            "WHERE character_id='agustin-edwards' AND doc_id='d1'"
        ).fetchone()
        assert mention is not None  # loop closed: doc attributed to seed
        assert mention["mentioned_by"] == "entity-mission"

    async def test_no_attribution_when_seed_character_absent(self, tmp_path, monkeypatch):
        db = self._seeded_db(tmp_path)
        conn = sqlite3.connect(str(db))
        conn.execute("DELETE FROM characters")  # seed was merged/removed
        conn.commit()
        conn.close()
        monkeypatch.setattr(cm, "LLMClient", lambda: _FakeOllama(json.dumps({"people": []})))
        await CastManager(db_path=db, use_llm=True).run_cycle()

        conn = _conn(db)
        n = conn.execute("SELECT COUNT(*) FROM character_mentions").fetchone()[0]
        assert n == 0  # no orphan mention to a non-existent character


# ── status ─────────────────────────────────────────────────────────────────────


class TestStatus:
    def test_status_missing_db_empty(self, tmp_path):
        assert CastManager(db_path=tmp_path / "nope.sqlite").status() == {}

    async def test_status_after_run(self, db):
        await CastManager(db_path=db, use_llm=False).run_cycle()
        status = CastManager(db_path=db).status()
        assert status["characters"] >= 3
        assert status["pending_docs"] == 0
        assert status["needs_research"] == status["characters"]


# ── Coreference: one person, many surface forms ────────────────────────────────


def _roster(*entries):
    """Build a roster dict: {key: [name, *aliases]}."""
    out = {}
    for name, *aliases in entries:
        out[character_key(name)] = [name, *aliases]
    return out


class TestResolveCharacterKey:
    def test_surname_only_resolves_to_unique_full_name(self):
        roster = _roster(["Salvador Allende"], ["Henry Kissinger"])
        assert resolve_character_key("Allende", roster) == "salvador-allende"
        assert resolve_character_key("Kissinger", roster) == "henry-kissinger"

    def test_longer_form_resolves_to_existing_shorter(self):
        roster = _roster(["Salvador Allende"])
        assert resolve_character_key("Salvador Allende Gossens", roster) == "salvador-allende"

    def test_shorter_form_resolves_to_existing_longer(self):
        roster = _roster(["Salvador Allende Gossens"])
        assert resolve_character_key("Salvador Allende", roster) == "salvador-allende-gossens"
        assert resolve_character_key("Allende", roster) == "salvador-allende-gossens"

    def test_initials_match(self):
        roster = _roster(["Henry Kissinger"])
        assert resolve_character_key("Henry A. Kissinger", roster) == "henry-kissinger"

    def test_ambiguous_surname_stays_separate(self):
        # Both Salvador and Beatriz exist — "Allende" alone must NOT merge into either.
        roster = _roster(["Salvador Allende"], ["Beatriz Allende"])
        assert resolve_character_key("Allende", roster) is None

    def test_different_given_names_never_merge(self):
        roster = _roster(["Salvador Allende"])
        assert resolve_character_key("Isabel Allende", roster) is None

    def test_aliases_participate_in_matching(self):
        roster = {"salvador-allende": ["Salvador Allende", "Salvador Allende Gossens"]}
        assert resolve_character_key("Allende Gossens", roster) == "salvador-allende"

    def test_particles_do_not_count_as_surnames(self):
        roster = _roster(["Carlos de la Fuente"])
        # "de" / "la" alone are particles, never a surname match
        assert resolve_character_key("De", roster) is None

    def test_no_match_returns_none(self):
        roster = _roster(["Salvador Allende"])
        assert resolve_character_key("Fidel Castro", roster) is None

    def test_unrelated_single_token_returns_none(self):
        assert resolve_character_key("Allende", {}) is None

    def test_org_forms_never_anchor_a_match(self):
        """Regression (Banco Edwards): an organisation-shaped roster form must not
        pull person surface forms into that character — neither via the surname
        rule nor via subsumption."""
        roster = {"banco-edwards": ["Banco Edwards"]}
        assert resolve_character_key("Edwards", roster) is None
        assert resolve_character_key("Agustín Edwards", roster) is None

    def test_org_alias_on_a_character_does_not_bridge(self):
        # Even when an org form accrued as an alias, only person-shaped forms match.
        roster = {"banco-edwards": ["Banco Edwards", "Edwards"]}
        assert resolve_character_key("Agustín Edwards", roster) is None
        # A real person with the same surname still resolves to the person.
        roster["agustin-edwards"] = ["Agustín Edwards"]
        assert resolve_character_key("Edwards", roster) == "agustin-edwards"


# ── Merge pass: unify pre-existing duplicates ──────────────────────────────────


def _insert_char(conn, name, mentions=0, facts=0, needs_research=1):
    from datetime import UTC, datetime

    key = character_key(name)
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """INSERT INTO characters
           (character_id, name, aliases, completeness_score, mention_count,
            needs_research, first_seen_at, updated_at)
           VALUES (?, ?, '[]', 0.0, 0, ?, ?, ?)""",
        (key, name, needs_research, now, now),
    )
    for i in range(mentions):
        conn.execute(
            """INSERT OR IGNORE INTO character_mentions
               (character_id, doc_id, mentioned_by, created_at)
               VALUES (?, ?, NULL, ?)""",
            (key, f"doc-{character_key(name)}-{i}", now),
        )
    for i in range(facts):
        conn.execute(
            """INSERT OR IGNORE INTO character_timeline
               (character_id, doc_id, date_iso, kind, description, created_at)
               VALUES (?, ?, '1971-01', 'event', ?, ?)""",
            (key, f"doc-{character_key(name)}-{i}", f"fact {i} of {name}", now),
        )
    conn.commit()
    return key


class TestMergeDuplicateCharacters:
    def _db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        return conn

    def test_merges_variants_into_fullest_name(self):
        conn = self._db()
        _insert_char(conn, "Allende", mentions=1)
        _insert_char(conn, "Salvador Allende", mentions=2, facts=1)
        _insert_char(conn, "Salvador Allende Gossens", mentions=1, facts=2)

        merged = merge_duplicate_characters(conn)
        assert merged == 2  # two rows absorbed

        rows = conn.execute("SELECT * FROM characters").fetchall()
        assert len(rows) == 1
        survivor = rows[0]
        assert survivor["name"] == "Salvador Allende Gossens"
        aliases = set(json.loads(survivor["aliases"]))
        assert {"Allende", "Salvador Allende"} <= aliases

        # All mentions and facts re-pointed to the survivor
        keys = {r[0] for r in conn.execute("SELECT DISTINCT character_id FROM character_mentions")}
        assert keys == {survivor["character_id"]}
        assert survivor["mention_count"] == 4
        facts = conn.execute(
            "SELECT COUNT(*) FROM character_timeline WHERE character_id=?",
            (survivor["character_id"],),
        ).fetchone()[0]
        assert facts == 3

    def test_ambiguous_family_members_not_merged(self):
        conn = self._db()
        _insert_char(conn, "Salvador Allende")
        _insert_char(conn, "Beatriz Allende")
        _insert_char(conn, "Allende")  # could be either — must stay separate

        assert merge_duplicate_characters(conn) == 0
        assert conn.execute("SELECT COUNT(*) FROM characters").fetchone()[0] == 3

    def test_merge_is_idempotent(self):
        conn = self._db()
        _insert_char(conn, "Kissinger")
        _insert_char(conn, "Henry Kissinger")
        assert merge_duplicate_characters(conn) == 1
        assert merge_duplicate_characters(conn) == 0

    def test_needs_research_survives_if_any_flagged(self):
        conn = self._db()
        _insert_char(conn, "Henry Kissinger", needs_research=0)
        _insert_char(conn, "Kissinger", needs_research=1)
        merge_duplicate_characters(conn)
        row = conn.execute("SELECT needs_research FROM characters").fetchone()
        assert row[0] == 1


# ── In-run resolution: new mentions fold into the existing character ───────────


class TestProcessBatchResolution:
    def test_shorter_mention_folds_into_existing_character(self, tmp_path):
        db_path = tmp_path / "archivo.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        conn.execute("""
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY, title TEXT, text TEXT,
                verified_at TEXT, mapped_category_id INTEGER, cast_extracted_at TEXT
            )
        """)
        _insert_char(conn, "Salvador Allende Gossens")
        conn.execute(
            """INSERT INTO documents VALUES
               ('docX', 'Discurso', 'el presidente Salvador Allende habló ante el pleno del senado.',
                '2026-01-01', 1, NULL)"""
        )
        conn.commit()
        conn.close()

        manager = CastManager(db_path=db_path, use_llm=False)
        result = asyncio.run(manager.run_cycle(batch_size=10))

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT character_id, name, aliases FROM characters").fetchall()
        conn.close()
        # "Salvador Allende" must NOT create a second character
        allendes = [r for r in rows if "allende" in r["character_id"]]
        assert len(allendes) == 1
        assert allendes[0]["character_id"] == "salvador-allende-gossens"
        assert "Salvador Allende" in json.loads(allendes[0]["aliases"])
        assert result.characters_new == 0


# ── Timeline fact dedup: one fact, many restatements ───────────────────────────


class TestFactsAreDuplicates:
    def test_same_fact_different_casing_and_accents(self):
        assert facts_are_duplicates("Presidente de Chile", "presidente de chile.")

    def test_restatement_with_extra_detail_is_duplicate(self):
        # The repeated "Allende is president" problem: a more detailed restatement
        # of the same fact must collapse into one.
        assert facts_are_duplicates("President of Chile", "elected President of Chile")
        assert facts_are_duplicates("presidente", "presidente de Chile")

    def test_different_facts_are_not_duplicates(self):
        assert not facts_are_duplicates("Ministro de Defensa", "Ministro de Economía")
        assert not facts_are_duplicates("presidente de Chile", "senador por Valparaíso")

    def test_stopwords_do_not_create_matches(self):
        assert not facts_are_duplicates(
            "fue el responsable de la CORFO", "fue el embajador en la ONU"
        )

    def test_empty_descriptions_never_match(self):
        assert not facts_are_duplicates("", "presidente")


class TestDedupeTimelineFacts:
    def _db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        return conn

    def _add_fact(self, conn, key, doc_id, description, kind="role", date_iso=None):
        conn.execute(
            """INSERT OR IGNORE INTO character_timeline
               (character_id, doc_id, date_iso, kind, description, created_at)
               VALUES (?, ?, ?, ?, ?, '2026-01-01')""",
            (key, doc_id, date_iso, kind, description),
        )
        conn.commit()

    def test_collapses_restated_facts_keeping_richest(self):
        conn = self._db()
        _insert_char(conn, "Salvador Allende")
        key = "salvador-allende"
        self._add_fact(conn, key, "d1", "President of Chile")
        self._add_fact(conn, key, "d2", "elected President of Chile", date_iso="1970-11")
        self._add_fact(conn, key, "d3", "president of chile")
        self._add_fact(conn, key, "d4", "Senator", date_iso="1966-01")  # distinct fact

        removed = dedupe_timeline_facts(conn)
        assert removed == 2

        rows = conn.execute(
            "SELECT description, date_iso FROM character_timeline WHERE character_id=?",
            (key,),
        ).fetchall()
        descs = {r["description"] for r in rows}
        assert descs == {"elected President of Chile", "Senator"}
        # Survivor carries the earliest known date of the cluster
        president = next(r for r in rows if "President" in r["description"])
        assert president["date_iso"] == "1970-11"

    def test_different_kinds_are_not_merged(self):
        conn = self._db()
        _insert_char(conn, "Salvador Allende")
        key = "salvador-allende"
        self._add_fact(conn, key, "d1", "President of Chile", kind="role")
        self._add_fact(conn, key, "d2", "President of Chile", kind="event")
        assert dedupe_timeline_facts(conn) == 0

    def test_idempotent(self):
        conn = self._db()
        _insert_char(conn, "Salvador Allende")
        key = "salvador-allende"
        self._add_fact(conn, key, "d1", "presidente de Chile")
        self._add_fact(conn, key, "d2", "Presidente de Chile")
        assert dedupe_timeline_facts(conn) == 1
        assert dedupe_timeline_facts(conn) == 0


class TestAddTimelineFactsSkipsRestatements:
    def test_insert_time_dedup_across_documents(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        _insert_char(conn, "Salvador Allende")
        key = "salvador-allende"

        added = cm._add_timeline_facts(
            conn,
            key,
            "d1",
            [
                {"kind": "role", "description": "President of Chile", "date_iso": None},
            ],
        )
        assert added == 1
        # Same fact restated in another document — must NOT add a second row,
        # but its date should backfill the existing undated fact.
        added = cm._add_timeline_facts(
            conn,
            key,
            "d2",
            [
                {
                    "kind": "role",
                    "description": "elected President of Chile",
                    "date_iso": "1970-11",
                },
            ],
        )
        assert added == 0
        rows = conn.execute(
            "SELECT description, date_iso FROM character_timeline WHERE character_id=?",
            (key,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["date_iso"] == "1970-11"


class TestBilingualFactDedup:
    def test_english_and_spanish_restatements_collapse(self):
        assert facts_are_duplicates("President of Chile", "Presidente de la República")
        assert facts_are_duplicates("asumió la presidencia de Chile", "President of Chile")
        assert facts_are_duplicates("Presidente de ella", "President of Chile")

    def test_different_offices_still_distinct(self):
        assert not facts_are_duplicates("Ministro de Defensa", "Ministro de Economía")
        assert not facts_are_duplicates("President of Chile", "Senator for Valparaíso")

    def test_other_kind_acts_as_wildcard_in_dedupe(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        _insert_char(conn, "Salvador Allende")
        key = "salvador-allende"
        for doc, desc, kind in [
            ("d1", "President of Chile", "other"),
            ("d2", "Presidente de la Republica", "role"),
            ("d3", "asumió la presidencia de Chile", "role"),
        ]:
            conn.execute(
                """INSERT INTO character_timeline
                   (character_id, doc_id, date_iso, kind, description, created_at)
                   VALUES (?, ?, NULL, ?, ?, '2026-01-01')""",
                (key, doc, kind, desc),
            )
        conn.commit()
        assert dedupe_timeline_facts(conn) == 2
        rows = conn.execute(
            "SELECT description FROM character_timeline WHERE character_id=?", (key,)
        ).fetchall()
        assert len(rows) == 1

    def test_role_and_event_still_not_merged(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        _insert_char(conn, "Salvador Allende")
        key = "salvador-allende"
        for doc, desc, kind in [
            ("d1", "President of Chile", "role"),
            ("d2", "won the presidential election", "event"),
        ]:
            conn.execute(
                """INSERT INTO character_timeline
                   (character_id, doc_id, date_iso, kind, description, created_at)
                   VALUES (?, ?, NULL, ?, ?, '2026-01-01')""",
                (key, doc, kind, desc),
            )
        conn.commit()
        assert dedupe_timeline_facts(conn) == 0


# ── Life dates: birth / death for the timeline graphic ─────────────────────────


class TestParseLifeDates:
    def test_parses_birth_and_death(self):
        raw = json.dumps(
            {
                "people": [
                    {"name": "Salvador Allende", "birth_date": "1908", "death_date": "1973-09-11"},
                ]
            }
        )
        person = parse_people_response(raw)[0]
        assert person.birth_date == "1908"
        assert person.death_date == "1973-09-11"

    def test_missing_life_dates_default_none(self):
        person = parse_people_response('{"people": [{"name": "Carlos Prats"}]}')[0]
        assert person.birth_date is None
        assert person.death_date is None

    def test_blank_life_dates_become_none(self):
        raw = json.dumps(
            {
                "people": [
                    {"name": "X Y", "birth_date": "  ", "death_date": ""},
                ]
            }
        )
        person = parse_people_response(raw)[0]
        assert person.birth_date is None
        assert person.death_date is None


class TestUpsertLifeDates:
    def _db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        return conn

    def _life(self, conn, key):
        return conn.execute(
            "SELECT birth_date, death_date FROM characters WHERE character_id=?", (key,)
        ).fetchone()

    def test_set_on_insert(self):
        conn = self._db()
        cm._upsert_character(
            conn,
            "salvador-allende",
            cm.ExtractedPerson(name="Salvador Allende", birth_date="1908", death_date="1973"),
            None,
        )
        row = self._life(conn, "salvador-allende")
        assert row["birth_date"] == "1908"
        assert row["death_date"] == "1973"

    def test_backfills_when_absent(self):
        conn = self._db()
        cm._upsert_character(conn, "k", cm.ExtractedPerson(name="Henry Kissinger"), None)
        cm._upsert_character(
            conn,
            "k",
            cm.ExtractedPerson(name="Henry Kissinger", birth_date="1923"),
            None,
        )
        assert self._life(conn, "k")["birth_date"] == "1923"

    def test_upgrades_to_more_precise_date(self):
        conn = self._db()
        cm._upsert_character(
            conn, "a", cm.ExtractedPerson(name="Salvador Allende", death_date="1973"), None
        )
        cm._upsert_character(
            conn,
            "a",
            cm.ExtractedPerson(name="Salvador Allende", death_date="1973-09-11"),
            None,
        )
        assert self._life(conn, "a")["death_date"] == "1973-09-11"

    def test_does_not_overwrite_known_with_none(self):
        conn = self._db()
        cm._upsert_character(
            conn, "a", cm.ExtractedPerson(name="Salvador Allende", birth_date="1908"), None
        )
        cm._upsert_character(conn, "a", cm.ExtractedPerson(name="Salvador Allende"), None)
        assert self._life(conn, "a")["birth_date"] == "1908"


def test_merge_carries_over_life_dates():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_cast_tables(conn)
    # Survivor (fullest name) lacks a birth date; an absorbed variant has it.
    cm._upsert_character(
        conn,
        "salvador-allende-gossens",
        cm.ExtractedPerson(name="Salvador Allende Gossens", death_date="1973-09-11"),
        None,
    )
    cm._upsert_character(
        conn, "allende", cm.ExtractedPerson(name="Allende", birth_date="1908"), None
    )
    merge_duplicate_characters(conn)
    row = conn.execute(
        "SELECT birth_date, death_date FROM characters WHERE character_id='salvador-allende-gossens'"
    ).fetchone()
    assert row["birth_date"] == "1908"
    assert row["death_date"] == "1973-09-11"


# ── Extraction recall: cued backstop + re-extract (roadmap backlog #1) ──────────


class TestCuedPersonNames:
    def test_honorific_prefix(self):
        from pipeline.cast_manager import cued_person_names

        names = cued_person_names("A meeting with Mr. Donald Kendall took place.")
        assert "Donald Kendall" in names

    def test_role_after_comma(self):
        from pipeline.cast_manager import cued_person_names

        text = (
            "Agustín Edwards, publisher of El Mercurio, and "
            "Donald Kendall, president of Pepsi-Cola, met."
        )
        names = cued_person_names(text)
        assert "Agustín Edwards" in names
        assert "Donald Kendall" in names

    def test_single_surname_after_honorific(self):
        from pipeline.cast_manager import cued_person_names

        assert "Korry" in cued_person_names("Ambassador Korry cabled Washington.")

    def test_rejects_organisations(self):
        from pipeline.cast_manager import cued_person_names

        # "United States, president Nixon" must not yield the country as a person.
        names = cued_person_names("The United States, president of the bloc, acted.")
        assert "United States" not in names

    def test_ignores_boilerplate_without_a_cue(self):
        from pipeline.cast_manager import cued_person_names

        assert cued_person_names("Top Secret. National Security Council. Eyes Only.") == []


class TestBackstopAugment:
    def test_adds_missed_people_without_duplicating(self):
        from pipeline.cast_manager import ExtractedPerson, _augment_with_backstop

        people = [ExtractedPerson(name="Richard Helms")]
        text = (
            "Mr. Donald Kendall and Agustín Edwards, publisher of El Mercurio, met Richard Helms."
        )
        out = _augment_with_backstop(people, text)
        names = {p.name for p in out}
        assert "Donald Kendall" in names
        assert "Agustín Edwards" in names
        assert sum(p.name == "Richard Helms" for p in out) == 1  # not duplicated


def test_llm_prompt_demands_exhaustive_people_and_affiliations():
    from pipeline.cast_manager import _LLM_PROMPT

    lowered = _LLM_PROMPT.lower()
    assert "every" in lowered
    assert "affiliation" in lowered
    assert "businessm" in lowered  # foreign businessmen explicitly requested


class TestChunking:
    """Long documents are mined in overlapping chunks so no person is lost."""

    def test_chunk_text_short_stays_single(self):
        from pipeline.cast_manager import chunk_text

        assert chunk_text("hola", size=4000) == ["hola"]
        assert chunk_text("", size=4000) == []

    def test_chunk_text_covers_all_content(self):
        from pipeline.cast_manager import chunk_text

        text = "A" * 6000 + "B" * 6000
        chunks = chunk_text(text, size=4000, overlap=200)
        assert len(chunks) >= 3
        # Every character of the original appears in some chunk (no data lost).
        assert "".join(chunks).count("A") >= 6000
        assert "".join(chunks).count("B") >= 6000
        assert text[:10] in chunks[0]
        assert text[-10:] in chunks[-1]

    def test_chunk_overlap_between_consecutive(self):
        from pipeline.cast_manager import chunk_text

        chunks = chunk_text("Z" * 9000, size=4000, overlap=300)
        # Consecutive chunks share content (overlap), so total length exceeds original.
        assert sum(len(c) for c in chunks) > 9000

    def test_merge_unions_people_and_dedupes_facts(self):
        # Same person named identically in overlapping chunks (the common case);
        # cross-chunk surname↔full-name coreference happens later, at upsert.
        from pipeline.cast_manager import ExtractedPerson, merge_extracted_people

        g1 = [
            ExtractedPerson(
                name="Salvador Allende",
                facts=[{"kind": "role", "description": "Presidente", "date_iso": None}],
            )
        ]
        g2 = [
            ExtractedPerson(
                name="Salvador Allende",
                mentioned_by="Neruda",
                facts=[
                    {"kind": "role", "description": "Presidente", "date_iso": None},
                    {"kind": "event", "description": "Ganó la elección", "date_iso": "1970-09"},
                ],
            )
        ]
        out = merge_extracted_people([g1, g2])
        assert len(out) == 1
        assert out[0].mentioned_by == "Neruda"  # filled from the chunk that had it
        assert len(out[0].facts) == 2  # duplicate "Presidente" collapsed

    def test_merge_prefers_the_fuller_name_form(self):
        from pipeline.cast_manager import ExtractedPerson, merge_extracted_people

        # Both key to the same character; the longer surface form is kept.
        g1 = [ExtractedPerson(name="salvador  allende")]  # 17 chars, same key
        g2 = [ExtractedPerson(name="Allende")]  # different key — stays separate
        out = merge_extracted_people([g1, g2, [ExtractedPerson(name="Salvador Allende")]])
        names = {p.name for p in out}
        assert "salvador  allende" in names  # fuller of the two salvador-allende forms
        assert "Allende" in names  # surname-only stays until upsert coreference

    async def test_people_in_later_chunks_are_captured(self):
        """A person named only in the last chunk must survive to the merged result."""
        from pipeline.cast_manager import _LLM_CHUNK_SIZE, _llm_extract_people

        early = "El presidente Salvador Allende inauguró la obra."
        late = "El general Augusto Pinochet encabezó el golpe."
        text = early + ("\n relleno." * _LLM_CHUNK_SIZE) + late  # spans several chunks

        class _PerChunkOllama:
            async def chat(self, *, messages, **_kwargs):
                content = messages[-1]["content"]
                found = []
                if "Salvador Allende" in content:
                    found.append({"name": "Salvador Allende"})
                if "Augusto Pinochet" in content:
                    found.append({"name": "Augusto Pinochet"})
                return json.dumps({"people": found})

        people = await _llm_extract_people(text, "Memo", _PerChunkOllama())
        keys = {character_key(p.name) for p in people}
        assert "salvador-allende" in keys  # first chunk
        assert "augusto-pinochet" in keys  # last chunk — nothing lost

    async def test_all_chunks_unparseable_returns_none(self):
        """If every chunk fails to parse, return None so the backstop still runs."""
        from pipeline.cast_manager import _LLM_CHUNK_SIZE, _llm_extract_people

        text = "x" * (_LLM_CHUNK_SIZE * 2 + 500)

        class _JunkOllama:
            async def chat(self, **_kwargs):
                return "not json at all"

        assert await _llm_extract_people(text, "Memo", _JunkOllama()) is None


class TestReExtract:
    async def test_default_run_is_self_terminating(self, db):
        await CastManager(db_path=db, use_llm=False).run_cycle(all_docs=True)
        # Second default run finds nothing new (all stamped).
        result = await CastManager(db_path=db, use_llm=False).run_cycle(all_docs=True)
        assert result.processed == 0

    async def test_re_extract_reprocesses_stamped_docs(self, db):
        await CastManager(db_path=db, use_llm=False).run_cycle(all_docs=True)
        result = await CastManager(db_path=db, use_llm=False).run_cycle(re_extract=True)
        assert result.processed == 2  # both docs mined again

    async def test_llm_backstop_captures_people_the_model_missed(self, db, monkeypatch):
        conn = _conn(db)
        conn.execute(
            "UPDATE documents SET text=? WHERE doc_id='d1'",
            ("Mr. Donald Kendall met Agustín Edwards, publisher of El Mercurio.",),
        )
        conn.commit()
        conn.close()
        # The model returns only Helms; the backstop must recover the two missed names.
        monkeypatch.setattr(
            cm,
            "LLMClient",
            lambda: _FakeOllama(json.dumps({"people": [{"name": "Richard Helms"}]})),
        )
        await CastManager(db_path=db, use_llm=True).run_cycle()

        conn = _conn(db)
        ids = {r["character_id"] for r in conn.execute("SELECT character_id FROM characters")}
        conn.close()
        assert "donald-kendall" in ids
        assert "agustin-edwards" in ids


# ── Political assessment (roadmap #3, LLM) ─────────────────────────────────────


class TestAssessPolitics:
    def _make_db(self, tmp_path):
        path = tmp_path / "archivo.sqlite"
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        now = "2026-01-01T00:00:00Z"
        for cid, name in [
            ("salvador-allende", "Salvador Allende"),
            ("augusto-pinochet", "Augusto Pinochet"),
        ]:
            conn.execute(
                "INSERT INTO characters (character_id, name, aliases, wikidata_desc, "
                "first_seen_at, updated_at) VALUES (?, ?, '[]', 'político chileno', ?, ?)",
                (cid, name, now, now),
            )
        conn.commit()
        conn.close()
        return path

    async def test_places_and_stamps(self, tmp_path, monkeypatch):
        path = self._make_db(tmp_path)
        resp = json.dumps({"economic": -0.8, "social": -0.3, "label": "socialista"})
        monkeypatch.setattr(cm, "LLMClient", lambda: _FakeOllama(resp))
        result = await CastManager(db_path=path, use_llm=True).assess_politics()
        assert result["placed"] == 2

        conn = _conn(path)
        row = conn.execute(
            "SELECT pol_economic, pol_social, pol_label, pol_source, pol_checked_at "
            "FROM characters WHERE character_id='salvador-allende'"
        ).fetchone()
        conn.close()
        assert row["pol_economic"] == -0.8
        assert row["pol_social"] == -0.3
        assert row["pol_label"] == "socialista"
        assert row["pol_source"] == "llm"
        assert row["pol_checked_at"] is not None

    async def test_second_run_skips_checked(self, tmp_path, monkeypatch):
        path = self._make_db(tmp_path)
        monkeypatch.setattr(cm, "LLMClient", lambda: _FakeOllama('{"economic": 0, "social": 0}'))
        await CastManager(db_path=path, use_llm=True).assess_politics()
        result = await CastManager(db_path=path, use_llm=True).assess_politics()
        assert result == {"assessed": 0, "placed": 0}

    async def test_unparseable_stamps_without_placing(self, tmp_path, monkeypatch):
        path = self._make_db(tmp_path)
        monkeypatch.setattr(cm, "LLMClient", lambda: _FakeOllama("not json"))
        result = await CastManager(db_path=path, use_llm=True).assess_politics()
        assert result == {"assessed": 2, "placed": 0}
        conn = _conn(path)
        row = conn.execute(
            "SELECT pol_economic, pol_checked_at FROM characters "
            "WHERE character_id='salvador-allende'"
        ).fetchone()
        conn.close()
        assert row["pol_economic"] is None
        assert row["pol_checked_at"] is not None  # stamped → not re-queried


# ── Deterministic cast validity filter (roadmap #6) ────────────────────────────


class TestIsPlausiblePersonName:
    def test_accepts_real_people(self):
        from pipeline.cast_manager import is_plausible_person_name

        for name in [
            "Salvador Allende",
            "Henry A. Kissinger",
            "Agustín Edwards",
            "Lanusse",
            "Alessandri",
            "Caldera",
            "General Carlos Prats",
        ]:
            assert is_plausible_person_name(name), name

    def test_rejects_function_words_and_titles(self):
        from pipeline.cast_manager import is_plausible_person_name

        for name in [
            "El",
            "La",
            "Pero",
            "En",
            "Nos",
            "Presidente",
            "Senador",
            "Salió",
            "Classification",
        ]:
            assert not is_plausible_person_name(name), name

    def test_rejects_months(self):
        from pipeline.cast_manager import is_plausible_person_name

        assert not is_plausible_person_name("October")
        assert not is_plausible_person_name("Octubre")

    def test_rejects_organisations(self):
        from pipeline.cast_manager import is_plausible_person_name

        assert not is_plausible_person_name("United States")
        assert not is_plausible_person_name("Top Secret")

    def test_rejects_document_boilerplate_prefix(self):
        from pipeline.cast_manager import is_plausible_person_name

        assert not is_plausible_person_name("Memorandum From Arnold Nachmanoff")

    def test_rejects_org_designator_prefix(self):
        """Regression (Banco Edwards): a leading org designator means the name
        denotes an institution, however person-like the rest looks."""
        from pipeline.cast_manager import is_plausible_person_name

        for name in [
            "Banco Edwards",
            "Radio Magallanes",
            "Diario El Mercurio",
            "Compañía de Teléfonos",
            "Universidad de Chile",
        ]:
            assert not is_plausible_person_name(name), name
        # ...but the person behind the bank is still a person.
        assert is_plausible_person_name("Agustín Edwards")
        assert is_plausible_person_name("Edwards")


class TestCleanInvalidCharacters:
    def _db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        now = "2026-01-01T00:00:00Z"
        for cid, name in [
            ("salvador-allende", "Salvador Allende"),
            ("el", "El"),
            ("presidente", "Presidente"),
            ("october", "October"),
        ]:
            conn.execute(
                "INSERT INTO characters (character_id, name, aliases, first_seen_at, updated_at) "
                "VALUES (?, ?, '[]', ?, ?)",
                (cid, name, now, now),
            )
        # a mention + timeline row on a garbage character to test cascade
        conn.execute(
            "INSERT INTO character_mentions (character_id, doc_id, created_at) "
            "VALUES ('el', 'd1', ?)",
            (now,),
        )
        conn.execute(
            "INSERT INTO character_timeline (character_id, doc_id, kind, description, created_at) "
            "VALUES ('el', 'd1', 'other', 'x', ?)",
            (now,),
        )
        conn.commit()
        return conn

    def test_dry_run_lists_without_deleting(self):
        from pipeline.cast_manager import clean_invalid_characters

        conn = self._db()
        result = clean_invalid_characters(conn, apply=False)
        assert set(result["names"]) == {"El", "Presidente", "October"}
        assert result["applied"] is False
        assert conn.execute("SELECT COUNT(*) FROM characters").fetchone()[0] == 4  # unchanged

    def test_apply_deletes_and_cascades(self):
        from pipeline.cast_manager import clean_invalid_characters

        conn = self._db()
        result = clean_invalid_characters(conn, apply=True)
        assert result["applied"] is True
        remaining = {r["character_id"] for r in conn.execute("SELECT character_id FROM characters")}
        assert remaining == {"salvador-allende"}
        # cascaded
        assert conn.execute("SELECT COUNT(*) FROM character_mentions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM character_timeline").fetchone()[0] == 0


class TestProcessBatchSkipsGarbage:
    async def test_garbage_names_not_stored(self, tmp_path):
        db = tmp_path / "archivo.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE documents (doc_id TEXT PRIMARY KEY, title TEXT, text TEXT, "
            "verified_at TEXT, mapped_category_id INTEGER)"
        )
        conn.execute(
            "INSERT INTO documents VALUES ('d1', 'Memo', "
            "'El presidente Salvador Allende habló. Top Secret. October.', "
            "'2026-01-01', 1)"
        )
        conn.commit()
        conn.close()

        await CastManager(db_path=db, use_llm=False).run_cycle()
        conn = _conn(db)
        ids = {r["character_id"] for r in conn.execute("SELECT character_id FROM characters")}
        conn.close()
        # "Salvador Allende" kept; no "El"/"Top Secret"/"October" garbage.
        assert "salvador-allende" in ids
        assert not any(g in ids for g in ("el", "top-secret", "october", "presidente"))


# ── Analyze: Wikidata claims + Wikipedia article → profile/timeline (F1) ────────


_CLAIMS_BODY = json.dumps(
    {
        "entities": {
            "Q1": {
                "claims": {
                    "P569": [
                        {
                            "mainsnak": {
                                "datavalue": {
                                    "value": {"time": "+1908-06-26T00:00:00Z", "precision": 11}
                                }
                            }
                        }
                    ],
                    "P570": [
                        {
                            "mainsnak": {
                                "datavalue": {
                                    "value": {"time": "+1973-09-00T00:00:00Z", "precision": 10}
                                }
                            }
                        }
                    ],
                    "P39": [
                        {
                            "mainsnak": {"datavalue": {"value": {"id": "Q100"}}},
                            "qualifiers": {
                                "P580": [
                                    {
                                        "datavalue": {
                                            "value": {
                                                "time": "+1970-11-03T00:00:00Z",
                                                "precision": 11,
                                            }
                                        }
                                    }
                                ]
                            },
                        }
                    ],
                    "P102": [{"mainsnak": {"datavalue": {"value": {"id": "Q200"}}}}],
                    "P106": [{"mainsnak": {"datavalue": {"value": {"id": "Q300"}}}}],
                }
            }
        }
    }
)

_LABELS_BODY = json.dumps(
    {
        "entities": {
            "Q100": {"labels": {"es": {"value": "Presidente de Chile"}}},
            "Q200": {"labels": {"en": {"value": "Socialist Party of Chile"}}},
            "Q300": {"labels": {"es": {"value": "político"}}},
        }
    }
)


class TestWikidataClaims:
    def test_urls_target_the_wikidata_api(self):
        assert "wbgetentities" in wikidata_claims_url("Q1")
        assert "props=claims" in wikidata_claims_url("Q1")
        labels = wikidata_labels_url(["Q100", "Q200"])
        assert "Q100%7CQ200" in labels or "Q100|Q200" in labels
        assert "props=labels" in labels

    def test_parse_claims_extracts_vitals_and_positions(self):
        parsed = parse_wikidata_claims(_CLAIMS_BODY, "Q1")
        assert parsed is not None
        assert parsed["birth_date"] == "1908-06-26"
        assert parsed["death_date"] == "1973-09"  # month precision
        props = {c["prop"] for c in parsed["claims"]}
        assert props == {"P39", "P102", "P106"}
        p39 = next(c for c in parsed["claims"] if c["prop"] == "P39")
        assert p39["value_qid"] == "Q100"
        assert p39["start"] == "1970-11-03"

    def test_parse_claims_bad_body_returns_none(self):
        assert parse_wikidata_claims("not json", "Q1") is None
        assert parse_wikidata_claims(json.dumps({"entities": {}}), "Q1") is None

    def test_parse_labels_prefers_spanish(self):
        labels = parse_wikidata_labels(_LABELS_BODY)
        assert labels["Q100"] == "Presidente de Chile"
        assert labels["Q200"] == "Socialist Party of Chile"  # en fallback

    def test_build_facts_maps_props_to_kinds(self):
        parsed = parse_wikidata_claims(_CLAIMS_BODY, "Q1")
        facts = build_wikidata_facts(parsed["claims"], parse_wikidata_labels(_LABELS_BODY))
        by_desc = {f["description"]: f for f in facts}
        assert by_desc["Presidente de Chile"]["kind"] == "role"
        assert by_desc["Presidente de Chile"]["date_iso"] == "1970-11-03"
        assert by_desc["Socialist Party of Chile"]["kind"] == "affiliation"
        assert by_desc["político"]["kind"] == "role"

    def test_unlabelled_qids_are_skipped(self):
        parsed = parse_wikidata_claims(_CLAIMS_BODY, "Q1")
        facts = build_wikidata_facts(parsed["claims"], {})
        assert facts == []

    def test_identity_plausibility(self):
        assert wikidata_identity_plausible("1908-06-26") is True
        assert wikidata_identity_plausible(None) is True
        assert wikidata_identity_plausible("1990") is False  # born after the window


class TestWikipediaAnalysis:
    def test_extract_url_from_article_url(self):
        url = wikipedia_extract_url("https://es.wikipedia.org/wiki/Salvador_Allende")
        assert url is not None
        assert url.startswith("https://es.wikipedia.org/w/api.php")
        assert "titles=Salvador_Allende" in url
        assert "explaintext" in url

    def test_extract_url_rejects_garbage(self):
        assert wikipedia_extract_url("not a url") is None

    def test_parse_extract_returns_page_text(self):
        body = json.dumps(
            {"query": {"pages": {"123": {"extract": "Salvador Allende fue presidente."}}}}
        )
        assert parse_wikipedia_extract(body) == "Salvador Allende fue presidente."
        assert parse_wikipedia_extract("not json") is None
        assert parse_wikipedia_extract(json.dumps({"query": {"pages": {}}})) is None

    def test_parse_analysis_response(self):
        raw = json.dumps(
            {
                "biography": "Médico y político socialista, presidente de Chile 1970-1973.",
                "facts": [
                    {"kind": "event", "description": "Gana la elección", "date_iso": "1970-09-04"},
                    {"kind": "bogus", "description": "kind desconocido cae a other"},
                    "not a dict",
                ],
            }
        )
        parsed = parse_analysis_response(raw)
        assert parsed is not None
        assert parsed["biography"].startswith("Médico")
        assert len(parsed["facts"]) == 2
        assert parsed["facts"][0]["kind"] == "event"
        assert parsed["facts"][1]["kind"] == "other"

    def test_parse_analysis_response_bad_shapes(self):
        assert parse_analysis_response("nope") is None
        assert parse_analysis_response(json.dumps(["list"])) is None


class TestAnalyzeCycle:
    """CastManager.analyze() with a fake Gatekeeper fetch and a fake Ollama."""

    def _db(self, tmp_path):
        path = tmp_path / "archivo.sqlite"
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        cm.ensure_cast_tables(conn)
        now = "2026-01-01T00:00:00Z"
        conn.execute(
            "INSERT INTO characters (character_id, name, wikidata_id, wikipedia_url, "
            "wikidata_checked_at, first_seen_at, updated_at) VALUES "
            "('salvador-allende', 'Salvador Allende', 'Q1', "
            "'https://es.wikipedia.org/wiki/Salvador_Allende', ?, ?, ?)",
            (now, now, now),
        )
        conn.commit()
        conn.close()
        return path

    def _fetch(self, bodies: dict[str, str]):
        def fetch(url: str) -> str | None:
            for marker, body in bodies.items():
                if marker in url:
                    return body
            return None

        return fetch

    def _wikipedia_body(self):
        return json.dumps(
            {"query": {"pages": {"1": {"extract": "Allende fue presidente de Chile."}}}}
        )

    def _llm_body(self):
        return json.dumps(
            {
                "biography": "Presidente de Chile entre 1970 y 1973.",
                "facts": [
                    {"kind": "event", "description": "Nacionaliza el cobre", "date_iso": "1971-07"}
                ],
            }
        )

    def _mock_ollama(self, response: str):
        from unittest.mock import AsyncMock, MagicMock

        client = MagicMock()
        client.chat = AsyncMock(return_value=response)
        client.list_models = AsyncMock(return_value=[{"name": "gemma4:e4b"}])
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        return client

    def _run_analyze(self, path, bodies, llm_response=None, use_llm=True):
        from unittest.mock import patch

        manager = CastManager(db_path=path, use_llm=use_llm)
        mock = self._mock_ollama(llm_response or self._llm_body())
        with patch("pipeline.cast_manager.LLMClient", return_value=mock):
            return asyncio.run(manager.analyze(self._fetch(bodies), limit=10))

    def test_full_analysis_stamps_and_feeds_profile(self, tmp_path):
        path = self._db(tmp_path)
        result = self._run_analyze(
            path,
            {
                "props=claims": _CLAIMS_BODY,
                "props=labels": _LABELS_BODY,
                "prop=extracts": self._wikipedia_body(),
            },
        )
        assert result["wikidata_done"] == 1
        assert result["wikipedia_done"] == 1
        assert result["facts_new"] >= 3  # 3 wikidata claims + 1 article fact (dedup aside)

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM characters").fetchone()
        assert row["wikidata_analyzed_at"] is not None
        assert row["wikipedia_analyzed_at"] is not None
        assert row["birth_date"] == "1908-06-26"
        assert row["death_date"] == "1973-09"
        assert row["biography"].startswith("Presidente")
        detail = json.loads(row["completeness_detail"])
        assert "no_external_analysis" not in detail["caps"]
        kinds = {r["kind"] for r in conn.execute("SELECT kind FROM character_timeline").fetchall()}
        assert "role" in kinds and "affiliation" in kinds and "event" in kinds
        conn.close()

    def test_implausible_identity_is_delinked(self, tmp_path):
        path = self._db(tmp_path)
        claims = json.loads(_CLAIMS_BODY)
        claims["entities"]["Q1"]["claims"]["P569"][0]["mainsnak"]["datavalue"]["value"]["time"] = (
            "+1990-01-01T00:00:00Z"
        )
        result = self._run_analyze(
            path, {"props=claims": json.dumps(claims), "props=labels": _LABELS_BODY}
        )
        assert result["delinked"] == 1

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM characters").fetchone()
        assert row["wikidata_id"] is None
        assert row["wikidata_analyzed_at"] is None
        conn.close()

    def test_fetch_failure_leaves_no_stamp_for_retry(self, tmp_path):
        path = self._db(tmp_path)
        result = self._run_analyze(path, {})  # every fetch returns None
        assert result["wikidata_done"] == 0

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM characters").fetchone()
        assert row["wikidata_analyzed_at"] is None
        assert row["wikipedia_analyzed_at"] is None
        conn.close()

    def test_no_llm_analyzes_wikidata_only(self, tmp_path):
        path = self._db(tmp_path)
        result = self._run_analyze(
            path,
            {
                "props=claims": _CLAIMS_BODY,
                "props=labels": _LABELS_BODY,
                "prop=extracts": self._wikipedia_body(),
            },
            use_llm=False,
        )
        assert result["wikidata_done"] == 1
        assert result["wikipedia_done"] == 0

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM characters").fetchone()
        assert row["wikidata_analyzed_at"] is not None
        assert row["wikipedia_analyzed_at"] is None
        conn.close()

    def test_existing_biography_is_kept(self, tmp_path):
        path = self._db(tmp_path)
        conn = sqlite3.connect(str(path))
        conn.execute("UPDATE characters SET biography='Bio ya escrita.'")
        conn.commit()
        conn.close()
        self._run_analyze(
            path,
            {
                "props=claims": _CLAIMS_BODY,
                "props=labels": _LABELS_BODY,
                "prop=extracts": self._wikipedia_body(),
            },
        )
        conn = sqlite3.connect(str(path))
        bio = conn.execute("SELECT biography FROM characters").fetchone()[0]
        assert bio == "Bio ya escrita."
        conn.close()


# ── Statement / rumor extraction (F2) ───────────────────────────────────────────


class TestStatementRumorFacts:
    def test_statement_keeps_valid_speech_act(self):
        fact = cm._clean_fact(
            {
                "kind": "statement",
                "description": "Anuncia la nacionalización del cobre",
                "speech_act": "declarative",
            }
        )
        assert fact["kind"] == "statement"
        assert fact["speech_act"] == "declarative"

    def test_statement_invalid_speech_act_dropped_to_none(self):
        fact = cm._clean_fact(
            {"kind": "statement", "description": "Dice algo", "speech_act": "shouting"}
        )
        assert fact["kind"] == "statement"
        assert fact["speech_act"] is None

    def test_speech_act_ignored_outside_statements(self):
        fact = cm._clean_fact(
            {"kind": "role", "description": "Presidente", "speech_act": "assertive"}
        )
        assert fact["speech_act"] is None

    def test_rumor_without_reported_by_is_rejected(self):
        assert cm._clean_fact({"kind": "rumor", "description": "Se dice que conspira"}) is None

    def test_rumor_gets_default_low_confidence(self):
        fact = cm._clean_fact(
            {
                "kind": "rumor",
                "description": "Se rumorea un autogolpe",
                "reported_by": "prensa opositora",
            }
        )
        assert fact["reported_by"] == "prensa opositora"
        assert fact["confidence"] == cm._RUMOR_DEFAULT_CONFIDENCE

    def test_rumor_explicit_confidence_is_kept(self):
        fact = cm._clean_fact(
            {
                "kind": "rumor",
                "description": "Se rumorea contacto con la CIA",
                "reported_by": "un cable",
                "confidence": 0.15,
            }
        )
        assert fact["confidence"] == 0.15

    def test_out_of_range_confidence_dropped(self):
        fact = cm._clean_fact(
            {
                "kind": "event",
                "description": "Viaja a Moscú",
                "confidence": 7,
            }
        )
        assert fact["confidence"] is None

    def test_narrative_kinds_do_not_dedupe_against_other(self):
        # "other" is a wildcard only among information kinds; a statement or
        # rumor must never collapse into an information fact.
        assert cm._fact_kinds_compatible("role", "other") is True
        assert cm._fact_kinds_compatible("statement", "other") is False
        assert cm._fact_kinds_compatible("rumor", "other") is False
        assert cm._fact_kinds_compatible("statement", "statement") is True

    def test_add_timeline_facts_persists_new_columns(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "a.sqlite"))
        conn.row_factory = sqlite3.Row
        cm.ensure_cast_tables(conn)
        now = "2026-01-01T00:00:00Z"
        conn.execute(
            "INSERT INTO characters (character_id, name, first_seen_at, updated_at) "
            "VALUES ('allende', 'Salvador Allende', ?, ?)",
            (now, now),
        )
        added = cm._add_timeline_facts(
            conn,
            "allende",
            "doc1",
            [
                {
                    "kind": "statement",
                    "description": "«La historia es nuestra»",
                    "date_iso": "1973-09-11",
                    "speech_act": "assertive",
                    "reported_by": None,
                    "confidence": None,
                },
                {
                    "kind": "rumor",
                    "description": "Se rumorea que renunciará",
                    "date_iso": None,
                    "speech_act": None,
                    "reported_by": "El Mercurio",
                    "confidence": 0.2,
                },
            ],
        )
        assert added == 2
        rows = {
            r["kind"]: r
            for r in conn.execute(
                "SELECT kind, speech_act, reported_by, confidence FROM character_timeline"
            ).fetchall()
        }
        assert rows["statement"]["speech_act"] == "assertive"
        assert rows["rumor"]["reported_by"] == "El Mercurio"
        assert rows["rumor"]["confidence"] == 0.2
        conn.close()

    def test_parse_people_response_passes_statements_and_rumors(self):
        raw = json.dumps(
            {
                "people": [
                    {
                        "name": "Salvador Allende",
                        "mentioned_by": None,
                        "facts": [
                            {
                                "kind": "statement",
                                "description": "Anuncia el área social",
                                "speech_act": "commissive",
                            },
                            {
                                "kind": "rumor",
                                "description": "Se dice que negocia con la DC",
                                "reported_by": "el memo",
                            },
                        ],
                    }
                ]
            }
        )
        people = parse_people_response(raw)
        assert people is not None
        kinds = {f["kind"] for f in people[0].facts}
        assert kinds == {"statement", "rumor"}

    def test_extraction_prompt_teaches_the_new_kinds(self):
        assert "statement" in cm._LLM_PROMPT
        assert "rumor" in cm._LLM_PROMPT
        assert "reported_by" in cm._LLM_PROMPT
        assert "speech_act" in cm._LLM_PROMPT


# ── Bulk re-score under the current constants (F4) ──────────────────────────────


class TestRescoreCharacters:
    def test_stale_scores_are_recomputed_and_flags_refreshed(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "a.sqlite"))
        conn.row_factory = sqlite3.Row
        cm.ensure_cast_tables(conn)
        now = "2026-01-01T00:00:00Z"
        # A character the OLD metric called complete (1.0) and un-flagged:
        # bio + a few facts, no external analysis, no research mission.
        conn.execute(
            "INSERT INTO characters (character_id, name, biography, completeness_score, "
            "needs_research, first_seen_at, updated_at) "
            "VALUES ('allende', 'Salvador Allende', 'Presidente de Chile.', 1.0, 0, ?, ?)",
            (now, now),
        )
        for i in range(5):
            conn.execute(
                "INSERT INTO character_timeline (character_id, doc_id, kind, description, "
                "created_at) VALUES ('allende', ?, 'role', ?, ?)",
                (f"d{i}", f"cargo {i}", now),
            )
        conn.commit()

        result = cm.rescore_characters(conn)

        assert result["rescored"] == 1
        assert result["complete"] == 0
        row = conn.execute(
            "SELECT completeness_score, completeness_detail, needs_research FROM characters"
        ).fetchone()
        assert row["completeness_score"] <= 0.6  # deflated by the strict gates
        assert row["needs_research"] == 1  # re-queued for research
        detail = json.loads(row["completeness_detail"])
        assert "no_external_analysis" in detail["caps"]
        conn.close()

    def test_empty_database_is_a_noop(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "a.sqlite"))
        conn.row_factory = sqlite3.Row
        cm.ensure_cast_tables(conn)
        assert cm.rescore_characters(conn) == {"rescored": 0, "complete": 0, "requeued": 0}
        conn.close()
