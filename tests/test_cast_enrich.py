"""Tests for the Cast Manager's Wikidata / Wikipedia enrichment (roadmap #2).

Fully offline: pure URL builders + JSON parsers, and the enrichment loop is
exercised with a fake fetch (no Gatekeeper, no HTTP).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from pipeline.cast_manager import (
    dedupe_timeline_facts,  # noqa: F401  (import sanity — module loads)
    enrich_characters,
    ensure_cast_tables,
    parse_wikidata_search,
    parse_wikipedia_url,
    resolve_wikidata,
    wikidata_search_url,
    wikidata_sitelinks_url,
)

_SEARCH = json.dumps({"search": [
    {"id": "Q170581", "label": "Salvador Allende",
     "description": "presidente de Chile (1908-1973)"},
    {"id": "Q9999", "label": "Otro", "description": "distinto"},
]})
_SITELINKS = json.dumps({"entities": {"Q170581": {"sitelinks": {
    "enwiki": {"url": "https://en.wikipedia.org/wiki/Salvador_Allende"},
    "eswiki": {"url": "https://es.wikipedia.org/wiki/Salvador_Allende"},
}}}})


class TestWikidataUrls:
    def test_search_url(self):
        url = wikidata_search_url("Salvador Allende")
        assert "wbsearchentities" in url
        assert "Salvador+Allende" in url or "Salvador%20Allende" in url

    def test_sitelinks_url(self):
        url = wikidata_sitelinks_url("Q170581")
        assert "wbgetentities" in url
        assert "Q170581" in url


class TestParseWikidataSearch:
    def test_takes_top_hit(self):
        assert parse_wikidata_search(_SEARCH) == (
            "Q170581", "Salvador Allende", "presidente de Chile (1908-1973)")

    def test_empty_results(self):
        assert parse_wikidata_search('{"search": []}') is None

    def test_malformed(self):
        assert parse_wikidata_search("not json") is None
        assert parse_wikidata_search('{"foo": 1}') is None


class TestParseWikipediaUrl:
    def test_prefers_spanish(self):
        assert parse_wikipedia_url(_SITELINKS, "Q170581") == \
            "https://es.wikipedia.org/wiki/Salvador_Allende"

    def test_falls_back_to_english(self):
        body = json.dumps({"entities": {"Q1": {"sitelinks": {
            "enwiki": {"url": "https://en.wikipedia.org/wiki/X"}}}}})
        assert parse_wikipedia_url(body, "Q1") == "https://en.wikipedia.org/wiki/X"

    def test_none_when_no_sitelinks(self):
        body = json.dumps({"entities": {"Q1": {"sitelinks": {}}}})
        assert parse_wikipedia_url(body, "Q1") is None

    def test_malformed(self):
        assert parse_wikipedia_url("not json", "Q1") is None


def _fake_fetch(url: str) -> str | None:
    if "wbsearchentities" in url:
        return _SEARCH if "Allende" in url else '{"search": []}'
    if "wbgetentities" in url:
        return _SITELINKS
    return None


class TestResolveWikidata:
    def test_full_resolution(self):
        got = resolve_wikidata("Salvador Allende", _fake_fetch)
        assert got["wikidata_id"] == "Q170581"
        assert got["wikidata_url"] == "https://www.wikidata.org/wiki/Q170581"
        assert got["wikipedia_url"] == "https://es.wikipedia.org/wiki/Salvador_Allende"
        assert "presidente" in got["description"]

    def test_no_match(self):
        assert resolve_wikidata("Nadie Que No Existe", _fake_fetch) is None


class TestEnrichCharacters:
    def _db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        ensure_cast_tables(conn)
        now = datetime.now(UTC).isoformat()
        for cid, name in [("salvador-allende", "Salvador Allende"),
                          ("nadie", "Nadie Que No Existe")]:
            conn.execute(
                "INSERT INTO characters (character_id, name, aliases, first_seen_at, updated_at) "
                "VALUES (?, ?, '[]', ?, ?)",
                (cid, name, now, now),
            )
        conn.commit()
        return conn

    def test_populates_and_stamps(self):
        conn = self._db()
        result = enrich_characters(conn, _fake_fetch)
        assert result == {"checked": 2, "resolved": 1}

        allende = conn.execute(
            "SELECT wikidata_id, wikipedia_url, wikidata_desc, wikidata_checked_at "
            "FROM characters WHERE character_id='salvador-allende'"
        ).fetchone()
        assert allende["wikidata_id"] == "Q170581"
        assert allende["wikipedia_url"].startswith("https://es.wikipedia.org/")
        assert "presidente" in allende["wikidata_desc"]
        assert allende["wikidata_checked_at"] is not None

        # Unmatched character is stamped (so it is not re-queried) but left empty.
        nadie = conn.execute(
            "SELECT wikidata_id, wikidata_checked_at FROM characters WHERE character_id='nadie'"
        ).fetchone()
        assert nadie["wikidata_id"] is None
        assert nadie["wikidata_checked_at"] is not None

    def test_second_run_skips_checked(self):
        conn = self._db()
        enrich_characters(conn, _fake_fetch)
        result = enrich_characters(conn, _fake_fetch)
        assert result == {"checked": 0, "resolved": 0}
