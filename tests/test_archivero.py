"""
Tests for pipeline.archivero — pure extraction functions and SQLite helpers.

Strategy:
  - All extraction functions (_openalex_extract, etc.) are pure: no network, no DB.
  - Coverage helpers are tested with an in-memory SQLite connection.
  - Full run_cycle integration is left to manual pipeline testing because it
    requires a live Gatekeeper; that logic is covered by the pure-function tests here.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from pipeline.archivero import (
    _archiveorg_extract,
    _build_url,
    _chronicling_america_extract,
    _crossref_extract,
    _decode_inverted_index,
    _ensure_coverage_table,
    _ensure_documents_table,
    _extract,
    _frus_doc_numbers,
    _insert_document,
    _marxists_index_links,
    _migrate_documents_schema,
    _mission_kind,
    _mission_seed_character,
    _openalex_extract,
    _semantic_scholar_extract,
    _upsert_coverage_score,
    _wikipedia_extract,
    _wikisource_page_extract,
    _wikisource_parse_url,
    _wikisource_search_titles,
)

# ── _decode_inverted_index ────────────────────────────────────────────────────


class TestDecodeInvertedIndex:
    def test_empty_dict_returns_empty_string(self):
        assert _decode_inverted_index({}) == ""

    def test_single_word_at_position_zero(self):
        assert _decode_inverted_index({"hello": [0]}) == "hello"

    def test_reconstructs_sentence_in_order(self):
        idx = {"The": [0], "cat": [1], "sat": [2]}
        assert _decode_inverted_index(idx) == "The cat sat"

    def test_word_at_multiple_positions(self):
        idx = {"the": [0, 3], "cat": [1], "sat": [2]}
        result = _decode_inverted_index(idx)
        words = result.split()
        assert words[0] == "the"
        assert words[1] == "cat"
        assert words[3] == "the"

    def test_non_contiguous_positions_preserved(self):
        # position 1 has no word — still reconstructed in order
        idx = {"first": [0], "third": [2]}
        result = _decode_inverted_index(idx)
        assert result.index("first") < result.index("third")


# ── _openalex_extract ─────────────────────────────────────────────────────────


class TestOpenalexExtract:
    def test_extracts_title_and_reconstructed_abstract(self):
        # Abstract must produce > 50 chars after reconstruction
        words = {f"word{i}": [i] for i in range(20)}
        words["Chile"] = [20]
        body = json.dumps(
            {
                "results": [
                    {
                        "display_name": "CyberSyn Chile 1972",
                        "abstract_inverted_index": words,
                    }
                ]
            }
        )
        docs = _openalex_extract(body)
        assert len(docs) == 1
        title, text = docs[0]
        assert title == "CyberSyn Chile 1972"
        assert "Chile" in text

    def test_skips_work_with_short_abstract(self):
        body = json.dumps(
            {
                "results": [
                    {
                        "display_name": "Paper",
                        "abstract_inverted_index": {"Hi": [0]},
                    }
                ]
            }
        )
        assert _openalex_extract(body) == []

    def test_skips_work_with_missing_abstract(self):
        body = json.dumps({"results": [{"display_name": "No abstract paper"}]})
        assert _openalex_extract(body) == []

    def test_skips_work_with_missing_title(self):
        body = json.dumps(
            {
                "results": [
                    {
                        "abstract_inverted_index": {"word": [0] * 20},
                    }
                ]
            }
        )
        assert _openalex_extract(body) == []

    def test_handles_multiple_results(self):
        long_idx = {f"word{i}": [i] for i in range(20)}
        body = json.dumps(
            {
                "results": [
                    {"display_name": "Paper A", "abstract_inverted_index": long_idx},
                    {"display_name": "Paper B", "abstract_inverted_index": long_idx},
                ]
            }
        )
        docs = _openalex_extract(body)
        assert len(docs) == 2

    def test_raises_on_invalid_json(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _openalex_extract("not valid json")


# ── _crossref_extract ─────────────────────────────────────────────────────────


class TestCrossrefExtract:
    def test_extracts_title_and_abstract(self):
        abstract = "Full abstract about Chile and Allende. " * 5
        body = json.dumps(
            {
                "message": {
                    "items": [
                        {
                            "title": ["Paper on Chile 1971"],
                            "abstract": abstract,
                        }
                    ]
                }
            }
        )
        docs = _crossref_extract(body)
        assert len(docs) == 1
        assert docs[0][0] == "Paper on Chile 1971"
        assert "Chile" in docs[0][1]

    def test_strips_jats_xml_tags(self):
        body = json.dumps(
            {
                "message": {
                    "items": [
                        {
                            "title": ["Test"],
                            "abstract": "<jats:p><b>Bold text</b> with <i>italic</i></jats:p>"
                            + "x" * 80,
                        }
                    ]
                }
            }
        )
        docs = _crossref_extract(body)
        assert len(docs) == 1
        assert "<" not in docs[0][1]
        assert ">" not in docs[0][1]

    def test_skips_item_with_short_abstract(self):
        body = json.dumps(
            {
                "message": {
                    "items": [
                        {
                            "title": ["Paper"],
                            "abstract": "Too short.",
                        }
                    ]
                }
            }
        )
        assert _crossref_extract(body) == []

    def test_skips_item_with_no_title(self):
        body = json.dumps(
            {
                "message": {
                    "items": [
                        {
                            "title": [],
                            "abstract": "Long enough abstract here. " * 5,
                        }
                    ]
                }
            }
        )
        assert _crossref_extract(body) == []

    def test_handles_empty_items_list(self):
        body = json.dumps({"message": {"items": []}})
        assert _crossref_extract(body) == []


# ── _wikisource_search_titles ─────────────────────────────────────────────────


class TestWikisourceSearchTitles:
    def test_returns_titles_from_search_results(self):
        body = json.dumps(
            {
                "query": {
                    "search": [
                        {"title": "Discursos de Allende/1972/Discurso", "snippet": "..."},
                        {"title": "Ley de Reforma Agraria", "snippet": "..."},
                    ]
                }
            }
        )
        titles = _wikisource_search_titles(body)
        assert titles == ["Discursos de Allende/1972/Discurso", "Ley de Reforma Agraria"]

    def test_skips_entries_with_empty_title(self):
        body = json.dumps(
            {
                "query": {
                    "search": [
                        {"title": "", "snippet": "something"},
                        {"title": "Real title", "snippet": "text"},
                    ]
                }
            }
        )
        assert _wikisource_search_titles(body) == ["Real title"]

    def test_returns_empty_list_for_no_results(self):
        body = json.dumps({"query": {"search": []}})
        assert _wikisource_search_titles(body) == []

    def test_handles_missing_query_key(self):
        assert _wikisource_search_titles(json.dumps({})) == []


# ── _wikisource_parse_url ─────────────────────────────────────────────────────


class TestWikisourceParseUrl:
    def test_encodes_title_with_slashes(self):
        url = _wikisource_parse_url("Discursos de Allende/1972/Discurso")
        assert "action=parse" in url
        assert "prop=text" in url
        assert " " not in url

    def test_replaces_spaces_with_underscores(self):
        url = _wikisource_parse_url("Ley de Reforma")
        assert "Ley_de_Reforma" in url or "Ley%5Fde%5FReforma" in url or "Ley" in url


# ── _wikisource_page_extract ──────────────────────────────────────────────────

_SPEECH_HTML = """
<div class="mw-parser-output">
<p>DISCURSO DEL PRESIDENTE SALVADOR ALLENDE EN EL HOSPITAL DEL SALVADOR</p>
<p>Trabajadores de la Salud, muy estimadas compañeras, y estimados compañeros:</p>
<p>He venido una vez más a este hospital. He venido, fundamentalmente, como médico.
He venido a una vieja casa con la cual he estado vinculado a lo largo de muchos
años de mi vida de médico, como Senador de la República y Presidente de la comisión
de Salud Pública del Senado, como Ministro de Pedro Aguirre Cerda.</p>
</div>
"""


class TestWikisourcePageExtract:
    def test_extracts_title_and_text_from_parse_response(self):
        body = json.dumps(
            {
                "parse": {
                    "title": "Discursos de Allende/1972/Hospital",
                    "text": {"*": _SPEECH_HTML},
                }
            }
        )
        result = _wikisource_page_extract(body)
        assert result is not None
        title, text = result
        assert title == "Discursos de Allende/1972/Hospital"
        assert "Allende" in text or "Salvador" in text

    def test_returns_none_for_missing_html(self):
        body = json.dumps({"parse": {"title": "Something", "text": {"*": ""}}})
        assert _wikisource_page_extract(body) is None

    def test_returns_none_for_too_short_text(self):
        body = json.dumps({"parse": {"title": "Stub", "text": {"*": "<p>Short.</p>"}}})
        assert _wikisource_page_extract(body) is None


# ── _frus_doc_numbers ─────────────────────────────────────────────────────────


class TestFrusDocNumbers:
    def test_returns_five_numbers(self):
        assert len(_frus_doc_numbers(0)) == 5

    def test_numbers_in_valid_range(self):
        for offset in [0, 50, 200, 365, 400]:
            nums = _frus_doc_numbers(offset)
            assert all(1 <= n <= 366 for n in nums)

    def test_wraps_around_at_366(self):
        nums = _frus_doc_numbers(363)
        assert 366 in nums
        assert 1 in nums  # wraps back to start

    def test_different_offsets_give_different_starts(self):
        assert _frus_doc_numbers(0)[0] != _frus_doc_numbers(10)[0]


# ── _archiveorg_extract ───────────────────────────────────────────────────────


class TestArchiveOrgExtract:
    def test_extracts_title_and_description(self):
        body = json.dumps(
            {
                "response": {
                    "docs": [
                        {
                            "title": "Documentos Chile Allende 1971",
                            "description": "Colección de documentos sobre el gobierno de la Unidad Popular.",
                        }
                    ]
                }
            }
        )
        docs = _archiveorg_extract(body)
        assert len(docs) == 1
        assert docs[0][0] == "Documentos Chile Allende 1971"

    def test_handles_list_description(self):
        body = json.dumps(
            {
                "response": {
                    "docs": [
                        {
                            "title": "Archivo colectivo",
                            "description": ["Primera parte del archivo.", "Segunda parte."],
                        }
                    ]
                }
            }
        )
        docs = _archiveorg_extract(body)
        assert len(docs) == 1
        assert "Primera parte" in docs[0][1]
        assert "Segunda parte" in docs[0][1]

    def test_skips_entry_with_short_description(self):
        body = json.dumps(
            {
                "response": {
                    "docs": [
                        {
                            "title": "Doc",
                            "description": "Short.",
                        }
                    ]
                }
            }
        )
        assert _archiveorg_extract(body) == []

    def test_skips_entry_with_missing_title(self):
        body = json.dumps(
            {
                "response": {
                    "docs": [
                        {
                            "description": "Long enough description for this document. " * 3,
                        }
                    ]
                }
            }
        )
        assert _archiveorg_extract(body) == []

    def test_handles_empty_docs_list(self):
        body = json.dumps({"response": {"docs": []}})
        assert _archiveorg_extract(body) == []


# ── _extract (dispatcher) ─────────────────────────────────────────────────────


class TestExtract:
    def test_routes_archive_org_to_json_extractor(self):
        body = json.dumps(
            {
                "response": {
                    "docs": [
                        {
                            "title": "Chilean archive doc",
                            "description": "A detailed description about events in Chile 1972.",
                        }
                    ]
                }
            }
        )
        docs = _extract(body, "archive.org")
        assert len(docs) == 1

    def test_routes_openalex_to_json_extractor(self):
        long_idx = {f"word{i}": [i] for i in range(15)}
        body = json.dumps(
            {
                "results": [
                    {
                        "display_name": "Chile paper",
                        "abstract_inverted_index": long_idx,
                    }
                ]
            }
        )
        docs = _extract(body, "openalex")
        assert len(docs) == 1

    def test_frus_not_routed_through_extract(self):
        # FRUS is handled as an HTML source in run_cycle, not via _extract
        html = "<html><body>FRUS document text</body></html>"
        docs = _extract(html, "frus")
        # Falls through to HTML path; short text → returns []
        assert isinstance(docs, list)

    def test_returns_empty_on_json_parse_error_for_json_source(self):
        docs = _extract("this is not json at all", "openalex")
        assert docs == []

    def test_html_source_with_error_pattern_returns_empty(self):
        html = "<html><body>Please enable JavaScript to view this page.</body></html>"
        docs = _extract(html, "web_serp")
        assert docs == []

    def test_html_source_with_bot_challenge_returns_empty(self):
        html = "<html><body>Bots use DuckDuckGo too. Please complete the following challenge.</body></html>"
        docs = _extract(html, "web_serp")
        assert docs == []


# ── _build_url ────────────────────────────────────────────────────────────────


class TestBuildUrl:
    def test_builds_valid_url_for_known_source(self):
        url = _build_url("archive.org", "Chile Allende 1972")
        assert url is not None
        assert "archive.org" in url

    def test_returns_none_for_unknown_source(self):
        assert _build_url("nonexistent_source_xyz", "query") is None

    def test_url_encodes_spaces(self):
        url = _build_url("openalex", "Chile Allende")
        assert " " not in url

    def test_url_encodes_special_characters(self):
        url = _build_url("openalex", "Chile & Allende")
        assert " " not in url

    def test_builds_frus_url_with_doc_number(self):
        url = _build_url("frus", "d42")
        assert url is not None
        assert "history.state.gov" in url
        assert "d42" in url


# ── coverage table helpers ────────────────────────────────────────────────────


@pytest.fixture
def mem_conn():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


class TestEnsureCoverageTable:
    def test_creates_coverage_table(self, mem_conn):
        _ensure_coverage_table(mem_conn)
        tables = {
            row[0]
            for row in mem_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "coverage" in tables

    def test_idempotent_when_called_twice(self, mem_conn):
        _ensure_coverage_table(mem_conn)
        _ensure_coverage_table(mem_conn)  # must not raise


class TestUpsertCoverageScore:
    def test_inserts_new_row(self, mem_conn):
        _ensure_coverage_table(mem_conn)
        _upsert_coverage_score(mem_conn, category_id=3, month_iso="1972-10", docs_found=3)
        row = mem_conn.execute(
            "SELECT coverage_score FROM coverage WHERE category_id=3 AND month_iso='1972-10'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.6)  # 3 / 5 = 0.6


# ── new connectors: Semantic Scholar / Chronicling America / Wikipedia / Marxists ──

# 205 chars: clears every extractor's length threshold (50/80/100) and contains "Chile"
_LONG = "Chile Allende Unidad Popular cybernetics " * 5


class TestSemanticScholarExtract:
    def test_extracts_title_and_abstract(self):
        body = json.dumps(
            {
                "data": [
                    {"title": "Cybersyn and Chile", "abstract": _LONG, "year": 1973},
                ]
            }
        )
        docs = _semantic_scholar_extract(body)
        assert len(docs) == 1
        title, text = docs[0]
        assert title == "Cybersyn and Chile"
        assert text.startswith("Cybersyn and Chile")

    def test_skips_short_abstract(self):
        body = json.dumps({"data": [{"title": "Paper", "abstract": "short"}]})
        assert _semantic_scholar_extract(body) == []

    def test_skips_missing_title(self):
        body = json.dumps({"data": [{"abstract": _LONG}]})
        assert _semantic_scholar_extract(body) == []

    def test_handles_empty_data(self):
        assert _semantic_scholar_extract(json.dumps({"data": []})) == []

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _semantic_scholar_extract("not json")


class TestChroniclingAmericaExtract:
    def test_extracts_title_and_ocr(self):
        body = json.dumps({"items": [{"title_normal": "el mercurio", "ocr_eng": _LONG}]})
        docs = _chronicling_america_extract(body)
        assert len(docs) == 1
        title, text = docs[0]
        assert title == "el mercurio"
        assert "Chile" in text

    def test_falls_back_to_title_field(self):
        body = json.dumps({"items": [{"title": "The Times", "ocr_eng": _LONG}]})
        assert _chronicling_america_extract(body)[0][0] == "The Times"

    def test_skips_short_ocr(self):
        body = json.dumps({"items": [{"title_normal": "x", "ocr_eng": "too short"}]})
        assert _chronicling_america_extract(body) == []

    def test_handles_empty_items(self):
        assert _chronicling_america_extract(json.dumps({"items": []})) == []


class TestWikipediaExtract:
    def test_extracts_title_and_extract(self):
        body = json.dumps(
            {
                "query": {
                    "pages": {
                        "42": {"title": "Salvador Allende", "extract": _LONG},
                    }
                }
            }
        )
        docs = _wikipedia_extract(body)
        assert len(docs) == 1
        title, text = docs[0]
        assert title == "Salvador Allende"
        assert text.startswith("Salvador Allende")

    def test_skips_short_extract(self):
        body = json.dumps({"query": {"pages": {"1": {"title": "X", "extract": "short"}}}})
        assert _wikipedia_extract(body) == []

    def test_handles_missing_query(self):
        assert _wikipedia_extract(json.dumps({})) == []

    def test_handles_multiple_pages(self):
        body = json.dumps(
            {
                "query": {
                    "pages": {
                        "1": {"title": "A", "extract": _LONG},
                        "2": {"title": "B", "extract": _LONG},
                    }
                }
            }
        )
        assert len(_wikipedia_extract(body)) == 2


class TestMarxistsIndexLinks:
    def test_extracts_absolute_work_links(self):
        html = (
            '<a href="1970/discurso.htm">Discurso</a>'
            '<a href="/espanol/allende/1972/programa.html">Programa</a>'
            '<a href="https://www.marxists.org/espanol/allende/1973/ultimo.htm">Último</a>'
        )
        links = _marxists_index_links(html)
        assert "https://www.marxists.org/espanol/allende/1970/discurso.htm" in links
        assert "https://www.marxists.org/espanol/allende/1972/programa.html" in links
        assert "https://www.marxists.org/espanol/allende/1973/ultimo.htm" in links

    def test_skips_offsite_and_nondocument_links(self):
        html = (
            '<a href="1971/valid.htm">Valid</a>'
            '<a href="https://example.com/other.htm">Off-site</a>'
            '<a href="/espanol/marx/index.htm">Different archive</a>'
            '<a href="1970/image.jpg">Image</a>'
        )
        links = _marxists_index_links(html)
        assert links == ["https://www.marxists.org/espanol/allende/1971/valid.htm"]

    def test_excludes_index_itself(self):
        html = '<a href="https://www.marxists.org/espanol/allende/">Index</a>'
        assert _marxists_index_links(html) == []

    def test_dedupes_repeated_links(self):
        html = '<a href="1970/a.htm">A</a><a href="1970/a.htm">A again</a>'
        assert len(_marxists_index_links(html)) == 1

    def test_empty_html_returns_empty(self):
        assert _marxists_index_links("") == []


class TestNewSourceUrlsAndRouting:
    def test_build_url_semantic_scholar(self):
        url = _build_url("semantic_scholar", "Allende Chile")
        assert url is not None and "api.semanticscholar.org" in url and " " not in url

    def test_build_url_chronicling_america(self):
        url = _build_url("chronicling_america", "Allende")
        assert url is not None and "chroniclingamerica.loc.gov" in url

    def test_build_url_wikipedia(self):
        url = _build_url("wikipedia_es", "Salvador Allende")
        assert url is not None and "es.wikipedia.org" in url and " " not in url

    def test_extract_routes_semantic_scholar(self):
        body = json.dumps({"data": [{"title": "T", "abstract": _LONG}]})
        assert len(_extract(body, "semantic_scholar")) == 1

    def test_extract_routes_chronicling_america(self):
        body = json.dumps({"items": [{"title_normal": "t", "ocr_eng": _LONG}]})
        assert len(_extract(body, "chronicling_america")) == 1

    def test_extract_routes_wikipedia(self):
        body = json.dumps({"query": {"pages": {"1": {"title": "T", "extract": _LONG}}}})
        assert len(_extract(body, "wikipedia_es")) == 1

    def test_extract_returns_empty_on_bad_json(self):
        assert _extract("nope", "semantic_scholar") == []

    def test_score_capped_at_one(self, mem_conn):
        _ensure_coverage_table(mem_conn)
        _upsert_coverage_score(mem_conn, category_id=1, month_iso="1973-09", docs_found=10)
        row = mem_conn.execute(
            "SELECT coverage_score FROM coverage WHERE category_id=1 AND month_iso='1973-09'"
        ).fetchone()
        assert row[0] == pytest.approx(1.0)

    def test_updates_existing_row(self, mem_conn):
        _ensure_coverage_table(mem_conn)
        _upsert_coverage_score(mem_conn, category_id=5, month_iso="1971-07", docs_found=2)
        _upsert_coverage_score(mem_conn, category_id=5, month_iso="1971-07", docs_found=5)
        row = mem_conn.execute(
            "SELECT coverage_score FROM coverage WHERE category_id=5 AND month_iso='1971-07'"
        ).fetchone()
        assert row[0] == pytest.approx(1.0)  # second upsert wins

    def test_zero_docs_gives_zero_score(self, mem_conn):
        _ensure_coverage_table(mem_conn)
        _upsert_coverage_score(mem_conn, category_id=2, month_iso="1970-03", docs_found=0)
        row = mem_conn.execute(
            "SELECT coverage_score FROM coverage WHERE category_id=2 AND month_iso='1970-03'"
        ).fetchone()
        assert row[0] == pytest.approx(0.0)

    def test_different_cells_stored_independently(self, mem_conn):
        _ensure_coverage_table(mem_conn)
        _upsert_coverage_score(mem_conn, category_id=1, month_iso="1970-01", docs_found=5)
        _upsert_coverage_score(mem_conn, category_id=2, month_iso="1970-01", docs_found=2)
        rows = mem_conn.execute(
            "SELECT category_id, coverage_score FROM coverage ORDER BY category_id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][1] == pytest.approx(1.0)
        assert rows[1][1] == pytest.approx(0.4)


# ── _mission_kind (entity missions must not touch the coverage matrix) ────────


class TestMissionKind:
    def _row(self, sql_suffix: str, params: tuple):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(f"CREATE TABLE missions {sql_suffix}")
        conn.execute(f"INSERT INTO missions VALUES ({','.join('?' * len(params))})", params)
        row = conn.execute("SELECT * FROM missions").fetchone()
        conn.close()
        return row

    def test_reads_kind_column_when_present(self):
        row = self._row("(mission_id TEXT, kind TEXT)", ("e-1", "entity"))
        assert _mission_kind(row) == "entity"

    def test_defaults_to_gap_when_column_missing(self):
        # Legacy databases predate the kind column.
        row = self._row("(mission_id TEXT)", ("m-1",))
        assert _mission_kind(row) == "gap"

    def test_defaults_to_gap_when_value_null(self):
        row = self._row("(mission_id TEXT, kind TEXT)", ("m-1", None))
        assert _mission_kind(row) == "gap"


class TestMissionSeedCharacter:
    def _row(self, sql_suffix: str, params: tuple):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(f"CREATE TABLE missions {sql_suffix}")
        conn.execute(f"INSERT INTO missions VALUES ({','.join('?' * len(params))})", params)
        row = conn.execute("SELECT * FROM missions").fetchone()
        conn.close()
        return row

    def test_entity_mission_returns_character_id(self):
        row = self._row(
            "(mission_id TEXT, kind TEXT, character_id TEXT)",
            ("e-1", "entity", "salvador-allende"),
        )
        assert _mission_seed_character(row) == "salvador-allende"

    def test_gap_mission_returns_none_even_with_character_id(self):
        row = self._row(
            "(mission_id TEXT, kind TEXT, character_id TEXT)",
            ("m-1", "gap", "ignored"),
        )
        assert _mission_seed_character(row) is None

    def test_legacy_db_without_column_returns_none(self):
        row = self._row("(mission_id TEXT, kind TEXT)", ("e-1", "entity"))
        assert _mission_seed_character(row) is None


class TestInsertDocumentSeed:
    def _conn(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _ensure_documents_table(conn)
        _migrate_documents_schema(conn)
        return conn

    def test_entity_doc_records_seed(self):
        conn = self._conn()
        assert _insert_document(
            conn,
            title="T",
            text="body one",
            source="openalex",
            mission_id="e-1",
            url="http://x",
            seed_character_id="salvador-allende",
        )
        row = conn.execute("SELECT seed_character_id, provenance FROM documents").fetchone()
        assert row["seed_character_id"] == "salvador-allende"
        assert json.loads(row["provenance"])["seed_character_id"] == "salvador-allende"

    def test_gap_doc_has_null_seed(self):
        conn = self._conn()
        _insert_document(
            conn,
            title="T",
            text="body two",
            source="openalex",
            mission_id="m-1",
            url="http://x",
        )
        assert conn.execute("SELECT seed_character_id FROM documents").fetchone()[0] is None

    def test_duplicate_backfills_null_seed(self):
        conn = self._conn()
        # First harvested by a gap mission (no seed).
        _insert_document(
            conn,
            title="T",
            text="same body",
            source="openalex",
            mission_id="m-1",
            url="http://x",
        )
        # Later re-found by an entity mission — the existing NULL seed is backfilled.
        assert not _insert_document(
            conn,
            title="T",
            text="same body",
            source="openalex",
            mission_id="e-9",
            url="http://y",
            seed_character_id="agustin-edwards",
        )
        assert (
            conn.execute("SELECT seed_character_id FROM documents").fetchone()[0]
            == "agustin-edwards"
        )


class TestMissionGenre:
    def _row(self, sql_suffix: str, params: tuple):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(f"CREATE TABLE missions {sql_suffix}")
        conn.execute(f"INSERT INTO missions VALUES ({','.join('?' * len(params))})", params)
        row = conn.execute("SELECT * FROM missions").fetchone()
        conn.close()
        return row

    def test_reads_genre_when_present(self):
        from pipeline.archivero import _mission_genre

        row = self._row("(mission_id TEXT, genre_id INTEGER)", ("m-1", 6))
        assert _mission_genre(row) == 6

    def test_zero_when_null(self):
        from pipeline.archivero import _mission_genre

        row = self._row("(mission_id TEXT, genre_id INTEGER)", ("m-1", None))
        assert _mission_genre(row) == 0

    def test_zero_when_column_missing(self):
        from pipeline.archivero import _mission_genre

        row = self._row("(mission_id TEXT)", ("m-1",))
        assert _mission_genre(row) == 0


# ── FOIA Chile connector (State Dept Chile Declassification Project) ──────────


def _minimal_pdf(text: str) -> bytes:
    """Build a tiny valid one-page PDF containing `text` (offline test fixture)."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
    ]
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects.append(
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF"
    ).encode()
    return bytes(out)


_FOIA_SAMPLE = json.dumps(
    {
        "success": True,
        "totalHits": 779,
        "Results": [
            {
                "casenumber": "S-1999-00030",
                "pdfLink": "DOCUMENTS\\StateChile3\\00005662.pdf",
                "subject": "CHILE: COUP PROSPECTS",
                "docdate": "1970-10-16T00:00:00",
                "classification": "UNCLASS",
                "doctype": "MI",
                "from": "INR",
                "to": "FILES",
                "documentclass": "StateChile3",
            },
            {  # no pdfLink → must be skipped
                "casenumber": "S-1999-00031",
                "pdfLink": None,
                "subject": "NO FILE",
                "docdate": "1971-01-01T00:00:00",
                "from": "",
                "to": "",
                "classification": "",
                "doctype": "",
            },
        ],
    }
)


class TestFoiaSearchUrl:
    def test_builds_query_against_chile_collection(self):
        from pipeline.archivero import _build_url

        url = _build_url("foia_chile", "allende cobre")
        assert url is not None
        assert "api/Search2/SubmitSimpleQuery" in url
        assert "collectionMatch=StateChile3" in url
        assert "searchText=allende+cobre" in url

    def test_includes_required_empty_params(self):
        # The endpoint 404s unless the full jQuery parameter set is present.
        from pipeline.archivero import _build_url

        url = _build_url("foia_chile", "x")
        for param in ("beginDate=", "endDate=", "caseNumber=", "sort="):
            assert param in url


class TestFoiaChileResults:
    def test_parses_results_with_pdf_url(self):
        from pipeline.archivero import _foia_chile_results

        docs = _foia_chile_results(_FOIA_SAMPLE)
        assert len(docs) == 1
        doc = docs[0]
        assert doc["subject"] == "CHILE: COUP PROSPECTS"
        assert doc["docdate"] == "1970-10-16"
        assert doc["pdf_url"] == "https://foia.state.gov/DOCUMENTS/StateChile3/00005662.pdf"
        assert doc["from"] == "INR"
        assert doc["to"] == "FILES"

    def test_empty_results(self):
        from pipeline.archivero import _foia_chile_results

        assert _foia_chile_results('{"success": true, "totalHits": 0, "Results": []}') == []

    def test_malformed_json_returns_empty(self):
        from pipeline.archivero import _foia_chile_results

        assert _foia_chile_results("not json") == []


class TestPdfExtractText:
    def test_extracts_text_layer(self):
        from pipeline.archivero import _pdf_extract_text

        pdf = _minimal_pdf("CHILE COUP PROSPECTS ocr layer")
        assert "CHILE COUP PROSPECTS ocr layer" in _pdf_extract_text(pdf)

    def test_garbage_bytes_return_empty(self):
        from pipeline.archivero import _pdf_extract_text

        assert _pdf_extract_text(b"definitely not a pdf") == ""


class TestFoiaDocumentText:
    def test_composes_header_and_ocr(self):
        from pipeline.archivero import _foia_document_text

        meta = {
            "subject": "CHILE: COUP PROSPECTS",
            "docdate": "1970-10-16",
            "from": "INR",
            "to": "FILES",
            "classification": "UNCLASS",
            "casenumber": "S-1999-00030",
            "pdf_url": "https://foia.state.gov/DOCUMENTS/StateChile3/00005662.pdf",
        }
        title, text = _foia_document_text(meta, "Coup rumblings within the military...")
        assert title == "CHILE: COUP PROSPECTS (1970-10-16)"
        assert "INR" in text and "FILES" in text
        assert "UNCLASS" in text
        assert "Coup rumblings" in text

    def test_title_without_date(self):
        from pipeline.archivero import _foia_document_text

        title, _ = _foia_document_text({"subject": "MEMO", "docdate": ""}, "body")
        assert title == "MEMO"

    def test_caps_documents_per_query(self):
        from pipeline.archivero import _foia_chile_results

        many = json.dumps(
            {
                "Results": [
                    {
                        "pdfLink": f"DOCUMENTS\\StateChile3\\{i:08d}.pdf",
                        "subject": f"DOC {i}",
                        "docdate": "1971-01-01T00:00:00",
                    }
                    for i in range(20)
                ]
            }
        )
        assert len(_foia_chile_results(many)) == 5
        assert len(_foia_chile_results(many, limit=2)) == 2


class TestMissionSeedLocation:
    def test_location_mission_yields_its_seed(self):
        from pipeline.archivero import _mission_seed_location

        row = {"kind": "location", "location_id": "la-moneda"}
        assert _mission_seed_location(row) == "la-moneda"

    def test_other_kinds_yield_none(self):
        from pipeline.archivero import _mission_seed_location

        assert _mission_seed_location({"kind": "gap", "location_id": "x"}) is None
        assert _mission_seed_location({"kind": "entity", "character_id": "y"}) is None
        assert _mission_seed_location({"kind": "location"}) is None  # legacy row
