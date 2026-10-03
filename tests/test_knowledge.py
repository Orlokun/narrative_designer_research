"""Tests for lib/knowledge.py — the Director's read-only consultation API."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lib.knowledge import (
    DirectorContext,
    DirectorQuery,
    KnowledgeBase,
    month_window,
    parse_theme_ids,
    score_document,
    snippet,
)

NOW = "2026-01-01T00:00:00+00:00"


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """A small archive with documents, cast, relations, places and flags."""
    db = tmp_path / "archivo.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY, title TEXT, text TEXT, lang TEXT, date_iso TEXT, authors TEXT, places TEXT,
            source_kind TEXT, source_id TEXT, provenance TEXT, rights TEXT, sha256 TEXT, ingested_at TEXT,
            quality_score REAL, mapped_month_iso TEXT, mapped_categories TEXT, mapped_genre_id INTEGER
        );
        CREATE TABLE doc_flags (id INTEGER PRIMARY KEY, doc_id TEXT, flag TEXT, confidence REAL, rationale TEXT, assigned_by TEXT, assigned_at TEXT);
        CREATE TABLE characters (
            character_id TEXT PRIMARY KEY, name TEXT, aliases TEXT, biography TEXT, completeness_score REAL, mention_count INTEGER,
            needs_research INTEGER, run_id TEXT, birth_date TEXT, death_date TEXT, first_seen_at TEXT, updated_at TEXT, pol_label TEXT
        );
        CREATE TABLE character_timeline (id INTEGER PRIMARY KEY, character_id TEXT, doc_id TEXT, date_iso TEXT, kind TEXT, description TEXT, created_at TEXT, speech_act TEXT, reported_by TEXT);
        CREATE TABLE character_relations (id INTEGER PRIMARY KEY, source_character_id TEXT, target_character_id TEXT, kind TEXT, description TEXT, confidence REAL, mention_count INTEGER, provenance TEXT, first_seen_at TEXT, updated_at TEXT);
        CREATE TABLE locations (location_id TEXT PRIMARY KEY, name TEXT, aliases TEXT, kind TEXT, description TEXT, latitude REAL, longitude REAL, mention_count INTEGER);
        CREATE TABLE location_facts (id INTEGER PRIMARY KEY, location_id TEXT, doc_id TEXT, date_iso TEXT, kind TEXT, detail TEXT, reported_by TEXT, created_at TEXT);
        """
    )

    def doc(doc_id, title, text, month, cats, genre=3, quality=0.9, source="archive.org"):
        conn.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                doc_id,
                title,
                text,
                "es",
                None,
                "[]",
                "[]",
                "archive",
                source,
                "{}",
                None,
                doc_id,
                NOW,
                quality,
                month,
                json.dumps(cats),
                genre,
            ),
        )

    doc(
        "d1",
        "Paro de camioneros: cable",
        "The truckers strike paralyses supply in October. " * 20,
        "1972-10",
        [11, 4, 1],
    )
    doc(
        "d2",
        "Informe económico",
        "Inflation report, escudo devaluation. " * 20,
        "1972-09",
        [3, 16, 11],
    )
    doc("d3", "Discurso cultural", "A speech about theatre. " * 20, "1971-08", [14, 1, 9], genre=1)
    doc("d4", "Low quality strike note", "truckers strike junk", "1972-10", [11], quality=0.2)
    doc(
        "d5",
        "Later strike analysis",
        "Analysis of the truckers strike written later. " * 20,
        "1973-02",
        [11, 3, 1],
        genre=9,
    )
    conn.executemany(
        "INSERT INTO doc_flags (doc_id, flag, confidence, assigned_by, assigned_at) VALUES (?,?,?,?,?)",
        [("d1", "military-logic", 0.9, "llm", NOW), ("d2", "economic-data", 0.9, "llm", NOW)],
    )
    conn.executemany(
        "INSERT INTO characters (character_id, name, aliases, biography, completeness_score, mention_count, needs_research, first_seen_at, updated_at, pol_label) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (
                "c1",
                "Salvador Allende",
                json.dumps(["Salvador Allende Gossens"]),
                "President of Chile 1970-73.",
                0.9,
                50,
                0,
                NOW,
                NOW,
                "socialista democrático",
            ),
            ("c2", "Fernando Flores", "[]", "Minister; Cybersyn lead.", 0.7, 20, 0, NOW, NOW, None),
            ("c3", "Henry Kissinger", "[]", None, 0.5, 10, 1, NOW, NOW, "realista conservador"),
        ],
    )
    conn.executemany(
        "INSERT INTO character_timeline (character_id, doc_id, date_iso, kind, description, created_at, speech_act, reported_by) VALUES (?,?,?,?,?,?,?,?)",
        [
            (
                "c1",
                "d1",
                "1972-10",
                "event",
                "Declares state of emergency during the strike",
                NOW,
                None,
                None,
            ),
            ("c1", "d3", "1970-11", "role", "Takes office as President", NOW, None, None),
            ("c1", "d3", None, "statement", "'We will not retreat'", NOW, "commit", None),
        ],
    )
    conn.executemany(
        "INSERT INTO character_relations (source_character_id, target_character_id, kind, description, confidence, mention_count, provenance, first_seen_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            ("c2", "c1", "colleague", "minister in his cabinet", 0.9, 3, "[]", NOW, NOW),
            ("c3", "c1", "contraparte", None, 0.8, 2, "[]", NOW, NOW),
        ],
    )
    conn.execute(
        "INSERT INTO locations VALUES (?,?,?,?,?,?,?,?)",
        (
            "l1",
            "La Moneda",
            json.dumps(["Palacio de La Moneda"]),
            "palace",
            "Presidential palace in Santiago.",
            -33.44,
            -70.65,
            12,
        ),
    )
    conn.execute(
        "INSERT INTO location_facts (location_id, doc_id, date_iso, kind, detail, created_at) VALUES (?,?,?,?,?,?)",
        ("l1", "d1", "1973-09", "event", "Bombed by the air force on 11 September 1973", NOW),
    )
    conn.commit()
    conn.close()
    return db


# ── Pure helpers ───────────────────────────────────────────────────────────────


def test_month_window_crosses_year_boundaries():
    assert month_window("1973-01", 1) == ["1972-12", "1973-01", "1973-02"]
    assert month_window("1972-10", 0) == ["1972-10"]


def test_parse_theme_ids_tolerates_bad_input():
    assert parse_theme_ids('[11, 4, "1"]') == [11, 4, 1]
    assert parse_theme_ids(None) == []
    assert parse_theme_ids("not json") == []


def test_score_document_prefers_primary_theme_and_exact_month():
    primary = score_document(
        theme_ids=[11, 4],
        wanted_themes=[11],
        month_iso="1972-10",
        wanted_months=["1972-09", "1972-10", "1972-11"],
        quality=0.9,
        flag_matches=0,
        keyword_matches=0,
    )
    secondary = score_document(
        theme_ids=[4, 11],
        wanted_themes=[11],
        month_iso="1972-09",
        wanted_months=["1972-09", "1972-10", "1972-11"],
        quality=0.9,
        flag_matches=0,
        keyword_matches=0,
    )
    assert primary > secondary > 0


def test_snippet_centres_on_keyword():
    text = "a" * 500 + " truckers strike " + "b" * 500
    out = snippet(text, ["strike"], chars=120)
    assert "strike" in out and out.startswith("…") and out.endswith("…")
    assert snippet("short text", []) == "short text"


# ── KnowledgeBase ──────────────────────────────────────────────────────────────


def test_missing_archive_yields_empty_context(tmp_path: Path):
    ctx = KnowledgeBase(tmp_path / "nope.sqlite").context(DirectorQuery(month_iso="1972-10"))
    assert ctx.documents == [] and ctx.characters == [] and ctx.locations == []
    assert ctx.project_slug == "cybersyn"


def test_documents_filtered_by_theme_month_and_quality(archive: Path):
    kb = KnowledgeBase(archive)
    ctx = kb.context(DirectorQuery(month_iso="1972-10", month_window=1, theme_ids=[11], limit=5))
    ids = [d.doc_id for d in ctx.documents]
    assert ids[0] == "d1"  # primary theme + exact month
    assert "d2" in ids  # window month, tertiary theme
    assert "d4" not in ids  # below quality threshold
    assert "d5" not in ids  # outside the window
    assert ctx.documents[0].flags == ["military-logic"]
    assert ctx.documents[0].theme_ids == [11, 4, 1]


def test_documents_filtered_by_flags_genre_and_keywords(archive: Path):
    kb = KnowledgeBase(archive)
    assert [d.doc_id for d in kb.context(DirectorQuery(flags=["economic-data"])).documents] == [
        "d2"
    ]
    assert [d.doc_id for d in kb.context(DirectorQuery(genre_ids=[9])).documents] == ["d5"]
    hits = kb.context(DirectorQuery(keywords=["truckers"], limit=10)).documents
    assert {d.doc_id for d in hits} == {"d1", "d5"}
    assert "truckers" in hits[0].snippet.lower()


def test_character_dossier_has_facts_and_relations(archive: Path):
    ctx = KnowledgeBase(archive).context(DirectorQuery(character_names=["allende", "Nobody Here"]))
    assert len(ctx.characters) == 1
    dossier = ctx.characters[0]
    assert dossier.name == "Salvador Allende"
    assert dossier.political_label == "socialista democrático"
    assert [f.kind for f in dossier.facts] == [
        "role",
        "event",
        "statement",
    ]  # dated first, chronological
    assert dossier.facts[2].speech_act == "commit"
    assert {(r.other_name, r.kind) for r in dossier.relations} == {
        ("Fernando Flores", "colleague"),
        ("Henry Kissinger", "contraparte"),
    }


def test_alias_lookup_and_dedup(archive: Path):
    ctx = KnowledgeBase(archive).context(
        DirectorQuery(character_names=["Gossens", "Salvador Allende"])
    )
    assert [c.character_id for c in ctx.characters] == ["c1"]


def test_location_dossier(archive: Path):
    ctx = KnowledgeBase(archive).context(DirectorQuery(location_names=["moneda"]))
    assert ctx.locations[0].name == "La Moneda"
    assert ctx.locations[0].facts == ["Bombed by the air force on 11 September 1973"]
    assert ctx.locations[0].latitude == pytest.approx(-33.44)


def test_find_characters(archive: Path):
    names = [c.name for c in KnowledgeBase(archive).find_characters("f")]
    assert "Fernando Flores" in names


def test_as_prompt_renders_sections_and_truncates(archive: Path):
    ctx = KnowledgeBase(archive).context(
        DirectorQuery(
            month_iso="1972-10",
            theme_ids=[11],
            character_names=["Allende"],
            location_names=["La Moneda"],
            limit=3,
        )
    )
    text = ctx.as_prompt()
    assert text.startswith("[Evidence from the cybersyn archive]")
    assert "## Salvador Allende — socialista democrático" in text
    assert "## Place: La Moneda (palace)" in text
    assert (
        "### Paro de camioneros: cable (archive.org, 1972-10; Transporte, Industria Nacional, Política Nacional)"
        in text
    )
    assert ctx.as_prompt(max_chars=80).endswith("…") and len(ctx.as_prompt(max_chars=80)) <= 80


def test_context_round_trips_through_json(archive: Path):
    ctx = KnowledgeBase(archive).context(DirectorQuery(month_iso="1972-10", theme_ids=[11]))
    assert DirectorContext.model_validate_json(ctx.model_dump_json()) == ctx
