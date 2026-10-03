"""Tests for the Source Scout — probing connectors through a (mocked) Gatekeeper."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from lib.project import ResearchProject, dump_project, load_project
from research_engine.source_scout import (
    ProbeResult,
    SourceScout,
    SourceVerdict,
    apply_tiers,
    count_hits,
    main,
    probe_queries,
    recommend_tiers,
)
from tests.test_project import minimal_spec

GK = "http://127.0.0.1:8001"


def _spec() -> ResearchProject:
    raw = minimal_spec(language="de")
    raw["themes"][0]["search_terms"] = ["Reichstag Notverordnung"]
    raw["themes"][0]["weight"] = 1.0
    return ResearchProject.model_validate(raw)


def _verdict(source_id: str, kind: str, hits: list[int | None]) -> SourceVerdict:
    v = SourceVerdict(source_id, source_id, kind, "json")
    for h in hits:
        v.probes.append(
            ProbeResult(
                source_id, "q", h is not None, h or 0, 10.0, "" if h is not None else "HTTP 503"
            )
        )
    return v


# ── Pure functions ─────────────────────────────────────────────────────────────


def test_probe_queries_prefers_heavy_theme_terms_then_region_year():
    assert probe_queries(_spec(), probes_per_source=3) == [
        "Reichstag Notverordnung",
        "Culture Germany",
        "Germany 1933",
    ]
    assert probe_queries(_spec(), probes_per_source=1) == ["Reichstag Notverordnung"]


def test_count_hits_uses_archivero_extractors():
    inverted = {f"word{i}": [i] for i in range(20)}
    openalex = json.dumps(
        {
            "results": [
                {"display_name": "A", "abstract_inverted_index": inverted},
                {"display_name": "B", "abstract_inverted_index": {"x": [0]}},
            ]
        }
    )
    assert count_hits("openalex", openalex) == 1
    wikisource = json.dumps({"query": {"search": [{"title": "Rede"}, {"title": "Gesetz"}]}})
    assert count_hits("wikisource_de", wikisource) == 2
    assert count_hits("openalex", "<html>nope</html>") == 0


def test_recommend_tiers_ranks_by_yield_and_keeps_web_search_last():
    verdicts = [
        _verdict("openalex", "academic", [4, 2]),
        _verdict("archive.org", "archive", [1, 0]),
        _verdict("wikipedia_de", "reference", [5, 5]),
        _verdict("crossref", "academic", [0, 0]),  # reachable, empty
        _verdict("semantic_scholar", "academic", [None, None]),  # unreachable
        _verdict("web_serp", "other", [3, 3]),
    ]
    tiers = recommend_tiers(verdicts)
    assert tiers["critica"] == ["wikipedia_de", "openalex", "archive.org", "crossref", "web_serp"]
    assert tiers["alta"] == ["wikipedia_de", "openalex", "archive.org", "web_serp"]
    assert tiers["media"] == ["wikipedia_de", "openalex", "archive.org"]
    assert tiers["baja"] == ["wikipedia_de", "openalex", "archive.org"]
    assert "semantic_scholar" not in tiers["critica"]


def test_apply_tiers_updates_routing_and_entity_sources():
    spec = _spec()
    tiers = {
        "critica": ["a", "web_serp"],
        "alta": ["a", "b", "web_serp"],
        "media": ["a"],
        "baja": ["b"],
    }
    updated = apply_tiers(spec, tiers)
    assert updated.source_tiers == tiers
    assert updated.entity.sources == ["a", "b"]
    assert spec.source_tiers != tiers  # original untouched


# ── Scout against a mocked Gatekeeper ─────────────────────────────────────────


def _gatekeeper_side_effect(request: httpx.Request) -> httpx.Response:
    url = json.loads(request.content)["url"]
    if "wikipedia.org" in url:
        body = json.dumps({"query": {"pages": {"1": {"title": "Reichstag", "extract": "x" * 200}}}})
    elif "archive.org" in url:
        body = json.dumps(
            {"response": {"docs": [{"identifier": "id1", "title": "T", "description": "d" * 100}]}}
        )
    elif "openalex" in url:
        return httpx.Response(503, text="down")
    else:
        body = json.dumps({"nothing": []})
    return httpx.Response(
        200,
        json={
            "url": url,
            "status_code": 200,
            "body": body,
            "cached": False,
            "fetched_at": "2026-01-01T00:00:00Z",
        },
    )


@pytest.mark.asyncio
@respx.mock
async def test_run_probes_each_applicable_connector_and_recommends():
    route = respx.post(f"{GK}/fetch").mock(side_effect=_gatekeeper_side_effect)
    report = await SourceScout(GK, probes_per_source=1).run(_spec())
    by_id = {v.source_id: v for v in report.verdicts}
    assert by_id["wikipedia_de"].status == "productive"
    assert by_id["archive.org"].status == "productive"
    assert by_id["openalex"].status == "unreachable"
    assert by_id["crossref"].status == "empty"
    assert "frus" not in by_id and "frus" in {e.connector.id for e in report.excluded}
    assert report.source_tiers["alta"][:2] == ["archive.org", "wikipedia_de"] or set(
        report.source_tiers["alta"][:2]
    ) == {"archive.org", "wikipedia_de"}
    assert route.call_count == len(report.verdicts)


@pytest.mark.asyncio
@respx.mock
async def test_gatekeeper_down_marks_everything_unreachable():
    respx.post(f"{GK}/fetch").mock(side_effect=httpx.ConnectError("refused"))
    report = await SourceScout(GK, probes_per_source=1).run(_spec())
    assert all(v.status == "unreachable" for v in report.verdicts)
    assert report.source_tiers == {"critica": [], "alta": [], "media": [], "baja": []}


@respx.mock
def test_cli_apply_writes_tiers(tmp_path):
    respx.post(f"{GK}/fetch").mock(side_effect=_gatekeeper_side_effect)
    path = tmp_path / "project.yaml"
    dump_project(_spec(), path)
    assert (
        main(["run", "--project", str(path), "--probes", "1", "--apply", "--gatekeeper", GK]) == 0
    )
    reloaded = load_project(path)
    assert "wikipedia_de" in reloaded.source_tiers["alta"]
    assert "openalex" not in reloaded.source_tiers["critica"]


@respx.mock
def test_cli_exit_1_when_nothing_reachable(tmp_path):
    respx.post(f"{GK}/fetch").mock(side_effect=httpx.ConnectError("refused"))
    path = tmp_path / "project.yaml"
    dump_project(_spec(), path)
    assert main(["run", "--project", str(path), "--probes", "1", "--gatekeeper", GK]) == 1


def test_cli_catalogue(capsys):
    assert main(["catalogue", "--project", "cybersyn"]) == 0
    assert "foia_chile" in capsys.readouterr().out
