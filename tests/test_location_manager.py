"""Tests for the Location Manager agent (Ag-4) — place extraction from documents.

Fully offline: pure-function and SQLite-helper tests run without any LLM, and the
run_cycle LLM path is exercised with a fake Ollama client. No live Ollama, no HTTP.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pipeline.location_manager as lm
from pipeline.location_manager import (
    LocationManager,
    compute_location_completeness,
    ensure_location_tables,
    fetch_locations_needing_research,
    location_key,
    parse_locations_response,
    refresh_location_research_flags,
    resolve_location_key,
)

NOW = "2026-01-01T00:00:00Z"


# ── Pure: location_key + resolve_location_key ──────────────────────────────────


class TestLocationKey:
    def test_basic_slug(self):
        assert location_key("La Moneda") == "la-moneda"
        assert location_key("Fábrica de harina de Temuco") == "fabrica-de-harina-de-temuco"

    def test_empty_for_garbage(self):
        assert location_key("—¿?—") == ""


class TestResolveLocationKey:
    def test_exact_and_alias_match(self):
        roster = {"la-moneda": ["La Moneda", "Palacio de La Moneda"]}
        assert resolve_location_key("palacio de la moneda", roster) == "la-moneda"

    def test_subsumed_form_matches(self):
        roster = {"la-moneda": ["Palacio de La Moneda"]}
        assert resolve_location_key("La Moneda", roster) == "la-moneda"

    def test_ambiguity_never_merges(self):
        roster = {
            "fabrica-temuco": ["Fábrica de harina de Temuco"],
            "fabrica-concepcion": ["Fábrica de harina de Concepción"],
        }
        assert resolve_location_key("Fábrica de harina", roster) is None

    def test_unknown_returns_none(self):
        assert resolve_location_key("Estadio Nacional", {}) is None


# ── Pure: parse_locations_response ─────────────────────────────────────────────


class TestParseLocations:
    def test_parses_locations_with_facts(self):
        raw = json.dumps(
            {
                "locations": [
                    {
                        "name": "La Moneda",
                        "kind": "building",
                        "associated_character": None,
                        "facts": [
                            {
                                "kind": "event",
                                "detail": "Reunión del comité económico",
                                "date_iso": "1972-10",
                            },
                            {
                                "kind": "appreciation",
                                "detail": "Un edificio frío y solemne",
                                "reported_by": "el embajador",
                            },
                        ],
                    }
                ]
            }
        )
        locations = parse_locations_response(raw)
        assert locations is not None
        loc = locations[0]
        assert loc.name == "La Moneda"
        assert loc.kind == "building"
        kinds = {f["kind"] for f in loc.facts}
        assert kinds == {"event", "appreciation"}

    def test_invalid_kind_falls_to_other(self):
        raw = json.dumps({"locations": [{"name": "Temuco", "kind": "galaxy", "facts": []}]})
        assert parse_locations_response(raw)[0].kind == "other"

    def test_fact_without_detail_is_dropped(self):
        raw = json.dumps(
            {"locations": [{"name": "Temuco", "kind": "city", "facts": [{"kind": "data"}]}]}
        )
        assert parse_locations_response(raw)[0].facts == []

    def test_bad_shapes_return_none(self):
        assert parse_locations_response("nope") is None
        assert parse_locations_response(json.dumps({"foo": 1})) is None

    def test_nameless_location_is_skipped(self):
        raw = json.dumps({"locations": [{"kind": "city", "facts": []}, "junk"]})
        assert parse_locations_response(raw) == []


# ── Pure: completeness ─────────────────────────────────────────────────────────


class TestLocationCompleteness:
    def test_empty_location_scores_zero(self):
        result = compute_location_completeness(
            mention_count=0,
            fact_count=0,
            wikidata_linked=False,
            wikidata_analyzed=False,
            wikipedia_linked=False,
            wikipedia_analyzed=False,
            has_coordinates=False,
        )
        assert result.score == 0.0
        assert "no_external_analysis" in result.detail["caps"]

    def test_full_location_reaches_one(self):
        result = compute_location_completeness(
            mention_count=4,
            fact_count=6,
            wikidata_linked=True,
            wikidata_analyzed=True,
            wikipedia_linked=True,
            wikipedia_analyzed=True,
            has_coordinates=True,
        )
        assert result.score == 1.0
        assert result.detail["caps"] == []

    def test_no_external_analysis_caps_at_060(self):
        result = compute_location_completeness(
            mention_count=99,
            fact_count=99,
            wikidata_linked=True,
            wikidata_analyzed=False,
            wikipedia_linked=True,
            wikipedia_analyzed=False,
            has_coordinates=True,
        )
        assert result.score == 0.6

    def test_coordinates_earn_their_component(self):
        without = compute_location_completeness(
            mention_count=0,
            fact_count=0,
            wikidata_linked=False,
            wikidata_analyzed=True,
            wikipedia_linked=False,
            wikipedia_analyzed=False,
            has_coordinates=False,
        )
        with_coords = compute_location_completeness(
            mention_count=0,
            fact_count=0,
            wikidata_linked=False,
            wikidata_analyzed=True,
            wikipedia_linked=False,
            wikipedia_analyzed=False,
            has_coordinates=True,
        )
        assert with_coords.score > without.score


# ── SQLite: extraction cycle with a fake Ollama ────────────────────────────────


def _seed_documents_db(path):
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY, title TEXT, text TEXT,
            verified_at TEXT, mapped_category_id INTEGER
        )"""
    )
    conn.execute(
        "INSERT INTO documents VALUES ('d1', 'Informe Temuco', "
        "'La fábrica de harina de Temuco aumentó su producción.', ?, 3)",
        (NOW,),
    )
    conn.commit()
    conn.close()


def _mock_ollama(response: str):
    client = MagicMock()
    client.chat = AsyncMock(return_value=response)
    client.list_models = AsyncMock(return_value=[{"name": "gemma4:e4b"}])
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


_LLM_RESPONSE = json.dumps(
    {
        "locations": [
            {
                "name": "Fábrica de harina de Temuco",
                "kind": "factory",
                "associated_character": None,
                "facts": [
                    {"kind": "data", "detail": "Produce 20 toneladas diarias"},
                    {
                        "kind": "appreciation",
                        "detail": "Orgullo industrial de la zona",
                        "reported_by": "el intendente",
                    },
                ],
            }
        ]
    }
)


class TestRunCycle:
    def test_extracts_locations_mentions_and_facts(self, tmp_path):
        path = tmp_path / "archivo.sqlite"
        _seed_documents_db(path)
        manager = LocationManager(db_path=path, use_llm=True)
        with patch(
            "pipeline.location_manager.LLMClient", return_value=_mock_ollama(_LLM_RESPONSE)
        ):
            result = asyncio.run(manager.run_cycle(batch_size=10))

        assert result.processed == 1
        assert result.locations_new == 1
        assert result.facts_new == 2

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        loc = conn.execute("SELECT * FROM locations").fetchone()
        assert loc["location_id"] == "fabrica-de-harina-de-temuco"
        assert loc["kind"] == "factory"
        assert loc["needs_research"] == 1
        assert conn.execute("SELECT COUNT(*) FROM location_mentions").fetchone()[0] == 1
        fact = conn.execute("SELECT * FROM location_facts WHERE kind='appreciation'").fetchone()
        assert fact["reported_by"] == "el intendente"
        # Document stamped — second run finds nothing.
        assert (
            conn.execute(
                "SELECT locations_extracted_at FROM documents WHERE doc_id='d1'"
            ).fetchone()[0]
            is not None
        )
        conn.close()

    def test_second_run_is_idempotent(self, tmp_path):
        path = tmp_path / "archivo.sqlite"
        _seed_documents_db(path)
        manager = LocationManager(db_path=path, use_llm=True)
        with patch(
            "pipeline.location_manager.LLMClient", return_value=_mock_ollama(_LLM_RESPONSE)
        ):
            asyncio.run(manager.run_cycle(batch_size=10))
            second = asyncio.run(manager.run_cycle(batch_size=10))
        assert second.processed == 0

    def test_seed_attribution_closes_the_loop(self, tmp_path):
        # A doc harvested by a location mission is force-attributed to its seed
        # even when the LLM does not re-extract the location's name.
        path = tmp_path / "archivo.sqlite"
        _seed_documents_db(path)
        conn = sqlite3.connect(str(path))
        ensure_location_tables(conn)
        conn.execute(
            "INSERT INTO locations (location_id, name, first_seen_at, updated_at) "
            "VALUES ('la-moneda', 'La Moneda', ?, ?)",
            (NOW, NOW),
        )
        conn.execute("ALTER TABLE documents ADD COLUMN seed_location_id TEXT")
        conn.execute("UPDATE documents SET seed_location_id='la-moneda' WHERE doc_id='d1'")
        conn.commit()
        conn.close()

        manager = LocationManager(db_path=path, use_llm=True)
        with patch(
            "pipeline.location_manager.LLMClient", return_value=_mock_ollama(_LLM_RESPONSE)
        ):
            asyncio.run(manager.run_cycle(batch_size=10))

        conn = sqlite3.connect(str(path))
        mentions = {
            r[0] for r in conn.execute("SELECT location_id FROM location_mentions").fetchall()
        }
        conn.close()
        assert "la-moneda" in mentions  # forced by the seed


# ── Enrich + analyze (Wikidata/Wikipedia) with a fake fetch ────────────────────


_WD_SEARCH = json.dumps(
    {"search": [{"id": "Q50", "label": "Palacio de La Moneda", "description": "sede presidencial"}]}
)
_WD_SITELINKS = json.dumps(
    {
        "entities": {
            "Q50": {
                "sitelinks": {
                    "eswiki": {"url": "https://es.wikipedia.org/wiki/Palacio_de_La_Moneda"}
                }
            }
        }
    }
)
_WD_CLAIMS = json.dumps(
    {
        "entities": {
            "Q50": {
                "claims": {
                    "P625": [
                        {
                            "mainsnak": {
                                "datavalue": {"value": {"latitude": -33.443, "longitude": -70.654}}
                            }
                        }
                    ],
                    "P571": [
                        {
                            "mainsnak": {
                                "datavalue": {
                                    "value": {"time": "+1805-00-00T00:00:00Z", "precision": 9}
                                }
                            }
                        }
                    ],
                    "P84": [{"mainsnak": {"datavalue": {"value": {"id": "Q900"}}}}],
                }
            }
        }
    }
)
_WD_LABELS = json.dumps({"entities": {"Q900": {"labels": {"es": {"value": "Joaquín Toesca"}}}}})
_WP_EXTRACT = json.dumps(
    {"query": {"pages": {"1": {"extract": "El Palacio de La Moneda es la sede presidencial."}}}}
)
_ANALYSIS = json.dumps(
    {
        "description": "Sede del gobierno de Chile en pleno centro de Santiago.",
        "facts": [
            {"kind": "data", "detail": "Fue una casa de acuñación de monedas"},
            {"kind": "event", "detail": "Bombardeada en el golpe", "date_iso": "1973-09-11"},
        ],
    }
)


def _fake_fetch(bodies: dict[str, str]):
    def fetch(url: str) -> str | None:
        for marker, body in bodies.items():
            if marker in url:
                return body
        return None

    return fetch


class TestEnrichAnalyze:
    def _db_with_location(self, tmp_path, **cols):
        path = tmp_path / "archivo.sqlite"
        conn = sqlite3.connect(str(path))
        ensure_location_tables(conn)
        base = {"wikidata_id": None, "wikipedia_url": None, "wikidata_checked_at": None}
        base.update(cols)
        conn.execute(
            "INSERT INTO locations (location_id, name, wikidata_id, wikipedia_url, "
            "wikidata_checked_at, first_seen_at, updated_at) VALUES "
            "('la-moneda', 'La Moneda', ?, ?, ?, ?, ?)",
            (base["wikidata_id"], base["wikipedia_url"], base["wikidata_checked_at"], NOW, NOW),
        )
        conn.commit()
        conn.close()
        return path

    def test_enrich_links_wikidata(self, tmp_path):
        path = self._db_with_location(tmp_path)
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        result = lm.enrich_locations(
            conn, _fake_fetch({"wbsearchentities": _WD_SEARCH, "sitelinks": _WD_SITELINKS})
        )
        assert result["resolved"] == 1
        row = conn.execute("SELECT * FROM locations").fetchone()
        assert row["wikidata_id"] == "Q50"
        assert row["wikipedia_url"].endswith("Palacio_de_La_Moneda")
        conn.close()

    def test_analyze_mines_coordinates_and_article(self, tmp_path):
        path = self._db_with_location(
            tmp_path,
            wikidata_id="Q50",
            wikipedia_url="https://es.wikipedia.org/wiki/Palacio_de_La_Moneda",
            wikidata_checked_at=NOW,
        )
        manager = LocationManager(db_path=path, use_llm=True)
        with patch("pipeline.location_manager.LLMClient", return_value=_mock_ollama(_ANALYSIS)):
            result = asyncio.run(
                manager.analyze(
                    _fake_fetch(
                        {
                            "props=claims": _WD_CLAIMS,
                            "props=labels": _WD_LABELS,
                            "prop=extracts": _WP_EXTRACT,
                        }
                    ),
                    limit=10,
                )
            )
        assert result["wikidata_done"] == 1
        assert result["wikipedia_done"] == 1

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM locations").fetchone()
        assert row["latitude"] == -33.443
        assert row["longitude"] == -70.654
        assert row["wikidata_analyzed_at"] is not None
        assert row["wikipedia_analyzed_at"] is not None
        assert row["description"].startswith("Sede del gobierno")
        details = {
            r["detail"] for r in conn.execute("SELECT detail FROM location_facts").fetchall()
        }
        assert any("Toesca" in d for d in details)  # architect from Wikidata
        assert any("acuñación" in d for d in details)  # trivia from the article
        detail_json = json.loads(row["completeness_detail"])
        assert "no_external_analysis" not in detail_json["caps"]
        conn.close()

    def test_analyze_fetch_failure_leaves_no_stamp(self, tmp_path):
        path = self._db_with_location(tmp_path, wikidata_id="Q50", wikidata_checked_at=NOW)
        manager = LocationManager(db_path=path, use_llm=False)
        result = asyncio.run(manager.analyze(_fake_fetch({}), limit=10))
        assert result["wikidata_done"] == 0
        conn = sqlite3.connect(str(path))
        assert conn.execute("SELECT wikidata_analyzed_at FROM locations").fetchone()[0] is None
        conn.close()


# ── Research queue ─────────────────────────────────────────────────────────────


class TestResearchQueue:
    def test_thin_locations_are_queued_and_complete_rest(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "a.sqlite"))
        conn.row_factory = sqlite3.Row
        ensure_location_tables(conn)
        for lid, score, needs in (("thin", 0.2, 0), ("done", 0.9, 1)):
            conn.execute(
                "INSERT INTO locations (location_id, name, completeness_score, "
                "needs_research, first_seen_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (lid, lid.title(), score, needs, NOW, NOW),
            )
        conn.commit()

        changed = refresh_location_research_flags(conn)
        assert changed == 2
        rows = fetch_locations_needing_research(conn)
        assert [r["location_id"] for r in rows] == ["thin"]
        conn.close()


# ── Image gallery: P18 + Wikipedia lead image → location_images (offline) ──────


class TestImageHelpers:
    def test_commons_image_url_encodes_filename(self):
        url = lm.commons_image_url("Palacio de La Moneda.jpg")
        assert url.startswith("https://commons.wikimedia.org/wiki/Special:FilePath/")
        assert "Palacio_de_La_Moneda.jpg" in url
        assert "width=" in url

    def test_image_filename_normalises_for_dedup(self):
        a = lm._image_filename("File:Palacio de La Moneda.jpg")
        b = lm._image_filename(
            "https://upload.wikimedia.org/wikipedia/commons/5/5a/Palacio_de_la_moneda.JPG"
        )
        assert a == b

    def test_parse_claims_collects_p18_images(self):
        body = json.dumps(
            {
                "entities": {
                    "Q50": {
                        "claims": {
                            "P18": [
                                {"mainsnak": {"datavalue": {"value": "La Moneda 1970.jpg"}}},
                                {"mainsnak": {"datavalue": {"value": {"not": "a-string"}}}},
                            ]
                        }
                    }
                }
            }
        )
        parsed = lm.parse_location_claims(body, "Q50")
        assert parsed["images"] == ["La Moneda 1970.jpg"]

    def test_wikipedia_url_with_images_extends_the_extract_url(self):
        url = lm.wikipedia_extract_images_url("https://es.wikipedia.org/wiki/Palacio_de_La_Moneda")
        assert "pageimages" in url
        assert "titles=Palacio_de_La_Moneda" in url

    def test_parse_wikipedia_page_image(self):
        body = json.dumps(
            {
                "query": {
                    "pages": {
                        "1": {
                            "extract": "texto",
                            "original": {
                                "source": "https://upload.wikimedia.org/wikipedia/commons/5/5a/Moneda.jpg"
                            },
                        }
                    }
                }
            }
        )
        assert lm.parse_wikipedia_page_image(body).endswith("Moneda.jpg")
        assert lm.parse_wikipedia_page_image(json.dumps({"query": {"pages": {}}})) is None


class TestAnalyzeHarvestsImages(TestEnrichAnalyze):
    def _claims_with_image(self):
        claims = json.loads(_WD_CLAIMS)
        claims["entities"]["Q50"]["claims"]["P18"] = [
            {"mainsnak": {"datavalue": {"value": "Palacio de La Moneda.jpg"}}}
        ]
        return json.dumps(claims)

    def _wikipedia_with_image(self):
        body = json.loads(_WP_EXTRACT)
        page = body["query"]["pages"]["1"]
        # Same file as P18 under a different URL — must dedupe by filename.
        page["original"] = {
            "source": "https://upload.wikimedia.org/wikipedia/commons/5/5a/Palacio_de_La_Moneda.jpg"
        }
        return json.dumps(body)

    def test_images_stored_and_deduped_across_sources(self, tmp_path):
        path = self._db_with_location(
            tmp_path,
            wikidata_id="Q50",
            wikipedia_url="https://es.wikipedia.org/wiki/Palacio_de_La_Moneda",
            wikidata_checked_at=NOW,
        )
        manager = LocationManager(db_path=path, use_llm=True)
        from unittest.mock import patch

        with patch("pipeline.location_manager.LLMClient", return_value=_mock_ollama(_ANALYSIS)):
            result = asyncio.run(
                manager.analyze(
                    _fake_fetch(
                        {
                            "props=claims": self._claims_with_image(),
                            "props=labels": _WD_LABELS,
                            "prop=extracts": self._wikipedia_with_image(),
                        }
                    ),
                    limit=10,
                )
            )
        assert result["images_new"] == 1  # P18 wins; the article's copy dedupes

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM location_images").fetchall()
        assert len(rows) == 1
        image = rows[0]
        assert image["source"] == "wikidata"
        assert "Special:FilePath" in image["url"]
        assert image["page_url"].startswith("https://commons.wikimedia.org/wiki/File:")
        conn.close()

    def test_distinct_article_image_is_added(self, tmp_path):
        path = self._db_with_location(
            tmp_path,
            wikidata_id="Q50",
            wikipedia_url="https://es.wikipedia.org/wiki/Palacio_de_La_Moneda",
            wikidata_checked_at=NOW,
        )
        body = json.loads(_WP_EXTRACT)
        body["query"]["pages"]["1"]["original"] = {
            "source": "https://upload.wikimedia.org/wikipedia/commons/9/99/Otra_vista.jpg"
        }
        manager = LocationManager(db_path=path, use_llm=True)
        from unittest.mock import patch

        with patch("pipeline.location_manager.LLMClient", return_value=_mock_ollama(_ANALYSIS)):
            result = asyncio.run(
                manager.analyze(
                    _fake_fetch(
                        {
                            "props=claims": self._claims_with_image(),
                            "props=labels": _WD_LABELS,
                            "prop=extracts": json.dumps(body),
                        }
                    ),
                    limit=10,
                )
            )
        assert result["images_new"] == 2  # P18 + a genuinely different article image
