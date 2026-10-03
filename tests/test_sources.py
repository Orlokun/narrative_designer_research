"""Tests for lib/sources.py — the connector catalogue and project filtering."""

from __future__ import annotations

from lib.project import ResearchProject, load_project
from lib.sources import (
    CATALOGUE,
    applicable_connectors,
    connector_by_id,
    default_source_tiers,
    excluded_connectors,
)
from tests.test_project import minimal_spec


def test_catalogue_ids_are_unique_and_cover_the_archivero_sources():
    ids = [c.id for c in CATALOGUE]
    assert len(ids) == len(set(ids))
    assert {
        "archive.org",
        "openalex",
        "crossref",
        "wikipedia",
        "wikisource",
        "frus",
        "foia_chile",
    } <= set(ids)


def test_cybersyn_keeps_its_chile_specific_connectors():
    spec = load_project("cybersyn")
    ids = {a.source_id for a in applicable_connectors(spec)}
    assert {"frus", "foia_chile", "marxists", "wikipedia_es", "wikisource_es", "archive.org"} <= ids
    assert "chronicling_america" not in ids  # coverage ends 1963


def test_other_moment_drops_chile_connectors_and_localises_wikis():
    spec = ResearchProject.model_validate(minimal_spec(language="de"))
    applicable = {a.source_id for a in applicable_connectors(spec)}
    assert "wikipedia_de" in applicable and "wikisource_de" in applicable
    assert not {"frus", "foia_chile", "marxists", "bn_digital"} & applicable
    excluded = {a.connector.id: a.reasons for a in excluded_connectors(spec)}
    assert any("region" in r for r in excluded["frus"])
    assert any("language" in r for r in excluded["marxists"])


def test_english_1930s_moment_gets_chronicling_america():
    spec = ResearchProject.model_validate(minimal_spec(language="en"))
    assert "chronicling_america" in {a.source_id for a in applicable_connectors(spec)}


def test_connector_by_id_resolves_parametric_ids():
    assert connector_by_id("wikipedia_de").id == "wikipedia"
    assert connector_by_id("archive.org").id == "archive.org"
    assert connector_by_id("nope") is None


def test_default_tiers_have_all_four_and_web_search_only_in_critica():
    spec = ResearchProject.model_validate(minimal_spec(language="de"))
    tiers = default_source_tiers(spec)
    assert set(tiers) == {"critica", "alta", "media", "baja"}
    assert tiers["critica"][-1] == "web_serp"
    assert "web_serp" not in tiers["alta"] + tiers["media"] + tiers["baja"]
    assert tiers["critica"][0] == "archive.org"
