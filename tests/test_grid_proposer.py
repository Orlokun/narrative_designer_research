"""Tests for the Grid Proposer — moment -> validated project spec (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import research_engine.grid_proposer as gp
from lib.project import load_project
from research_engine.grid_proposer import (
    GridProposer,
    MomentBrief,
    build_project,
    extract_json_object,
    main,
    normalise_critical_months,
    normalise_themes,
    skeleton_proposal,
    slugify,
    tier_for_weight,
)

BRIEF = MomentBrief(
    moment="Weimar Berlin from the 1929 crash to Hitler's chancellorship",
    start_month="1929-10",
    end_month="1933-03",
    region="Germany",
    language="de",
    slug="weimar",
    n_themes=6,
)


def _llm_proposal() -> dict:
    return {
        "themes": [
            {
                "name": "Reichspolitik",
                "weight": 1.0,
                "description": "cabinet, Reichstag, decrees",
                "search_terms": ["Reichstag Notverordnung", "Brüning Kabinett"],
            },
            {
                "name": "Wirtschaftskrise",
                "weight": 0.95,
                "description": "unemployment, banks",
                "search_terms": ["Arbeitslosigkeit 1931"],
            },
            {"name": "Straßenkampf", "weight": 0.8, "description": "SA, RFB, political violence"},
            {"name": "Kultur", "weight": 0.4, "description": "theatre, film, cabaret"},
            {"name": "Presse", "weight": 0.5, "tier": "alta", "description": "newspapers"},
            {"name": "reichspolitik", "weight": 0.1},  # duplicate (case-insensitive) → dropped
            {"name": "Sport", "weight": "not a number"},
        ],
        "critical_months": [
            {"month_iso": "1930-09", "event": "Reichstag election: NSDAP surge", "label": "Wahl30"},
            {"month_iso": "1933-01", "event": "Hitler appointed Chancellor"},
            {"month_iso": "1945-05", "event": "outside the period"},
            {"month_iso": "1930-09", "event": "duplicate month"},
        ],
        "seeds": {
            "characters": [
                {"name": "Heinrich Brüning", "role": "Chancellor 1930-32", "aliases": ["Brüning"]},
                "Ernst Thälmann",
            ],
            "institutions": [{"name": "Reichstag"}],
            "locations": [{"name": "Karl-Liebknecht-Haus", "role": "KPD headquarters"}],
        },
        "relevance": {
            "relevant_topics": ["Presidential cabinets and emergency decrees"],
            "zero_score_rules": ["Anything after May 1945"],
        },
        "era_label": "Weimar Germany (1929-1933)",
        "era_label_local": "Weimarer Republik (1929-1933)",
        "reading_rule_examples": ["A Reichstag speech on unemployment = [1, 2, 3], NOT [2]"],
        "vocabulary_hints": ["Notverordnung", "Harzburger Front"],
        "query_suffix": "Deutschland 1929 1933",
        "context_suffix": "Weimarer Republik",
    }


# ── Pure helpers ───────────────────────────────────────────────────────────────


def test_slugify():
    assert slugify("Weimar Berlin, 1929–33!") == "weimar-berlin-1929-33"
    assert slugify("   ") == "project"


def test_tier_for_weight_thresholds():
    assert tier_for_weight(1.0) == "critica"
    assert tier_for_weight(0.8) == "alta"
    assert tier_for_weight(0.5) == "media"
    assert tier_for_weight(0.1) == "baja"


def test_extract_json_object_tolerates_fences_and_prose():
    raw = 'Sure! ```json\n{"themes": []}\n``` done'
    assert extract_json_object(raw) == {"themes": []}
    assert extract_json_object("no json here") is None
    assert extract_json_object("[1, 2]") is None


def test_normalise_themes_renumbers_dedups_and_derives_tiers():
    themes = normalise_themes(_llm_proposal()["themes"], n_themes=6)
    assert [t.id for t in themes] == [1, 2, 3, 4, 5, 6]
    assert [t.name for t in themes] == [
        "Reichspolitik",
        "Wirtschaftskrise",
        "Straßenkampf",
        "Kultur",
        "Presse",
        "Sport",
    ]
    assert themes[0].tier == "critica" and themes[2].tier == "alta" and themes[3].tier == "media"
    assert themes[4].tier == "alta"  # explicit tier wins
    assert themes[5].weight == 0.5  # unparsable weight → default


def test_normalise_critical_months_filters_and_sorts():
    period = gp.Period(start_month="1929-10", end_month="1933-03")
    months = normalise_critical_months(_llm_proposal()["critical_months"], period)
    assert [m.month_iso for m in months] == ["1930-09", "1933-01"]
    assert months[0].label == "Wahl30"
    assert months[1].label == "Hitler"


# ── build_project ──────────────────────────────────────────────────────────────


def test_build_project_completes_a_valid_spec():
    spec = build_project(BRIEF, _llm_proposal())
    assert spec.slug == "weimar" and spec.region == "Germany" and spec.language == "de"
    assert len(spec.themes) == 6
    assert len(spec.months()) == 42
    assert spec.entity.nominal_theme_id == 1  # heaviest theme
    assert spec.entity.nominal_month_iso == "1930-09"  # first critical month
    assert spec.entity.query_suffix == "Deutschland 1929 1933"
    assert "wikipedia_de" in spec.source_tiers["critica"]
    assert "frus" not in spec.source_tiers["critica"]
    assert spec.entity_prompts.era_label_local == "Weimarer Republik (1929-1933)"
    assert spec.entity_prompts.name_example == "Heinrich Brüning"
    assert {s.kind for s in spec.seeds} == {"character", "institution", "location"}
    assert "Notverordnung" in spec.reformulation.system
    assert spec.relevance.relevant_topics == ["Presidential cabinets and emergency decrees"]
    assert spec.classification.category_examples[0].startswith("A Reichstag speech")


def test_build_project_rejects_too_few_themes():
    with pytest.raises(ValueError, match="usable themes"):
        build_project(BRIEF, {"themes": [{"name": "Only one"}]})


def test_skeleton_proposal_builds_without_llm():
    spec = build_project(BRIEF, skeleton_proposal(BRIEF))
    assert len(spec.themes) == 6
    assert spec.critical_months == []
    assert spec.entity.nominal_month_iso == spec.period.middle_month()
    assert spec.relevance.zero_score_rules[-1] == "Any document not about Germany"


# ── GridProposer lifecycle ─────────────────────────────────────────────────────


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


@pytest.mark.asyncio
async def test_propose_uses_llm_reply(monkeypatch):
    fake = _FakeLLM("Here you go:\n" + json.dumps(_llm_proposal()))
    monkeypatch.setattr(gp, "LLMClient", lambda: fake)
    spec = await GridProposer(use_llm=True).propose(BRIEF)
    assert spec.themes[0].name == "Reichspolitik"
    assert "Weimar Berlin" in fake.calls[0]["messages"][1]["content"]
    assert fake.calls[0]["think"] is False


@pytest.mark.asyncio
async def test_propose_falls_back_to_skeleton_when_llm_not_json(monkeypatch):
    monkeypatch.setattr(gp, "LLMClient", lambda: _FakeLLM("I cannot help with that."))
    spec = await GridProposer(use_llm=True).propose(BRIEF)
    assert spec.themes[0].name == "Politics & Government"


@pytest.mark.asyncio
async def test_propose_falls_back_when_llm_errors(monkeypatch):
    class _Boom(_FakeLLM):
        async def chat(self, **kwargs):
            raise gp.LLMError("quota")

    monkeypatch.setattr(gp, "LLMClient", lambda: _Boom(""))
    spec = await GridProposer(use_llm=True).propose(BRIEF)
    assert len(spec.themes) == 6


def test_write_refuses_to_overwrite_without_force(tmp_path: Path):
    proposer = GridProposer(use_llm=False)
    spec = build_project(BRIEF, skeleton_proposal(BRIEF))
    out = tmp_path / "weimar" / "project.yaml"
    proposer.write(spec, out)
    with pytest.raises(FileExistsError):
        proposer.write(spec, out)
    proposer.write(spec, out, force=True)
    assert load_project(out).slug == "weimar"


def test_cli_propose_no_llm_and_show(tmp_path: Path, capsys):
    out = tmp_path / "p.yaml"
    code = main(
        [
            "propose",
            "--moment",
            "Test moment",
            "--start",
            "1900-01",
            "--end",
            "1900-12",
            "--region",
            "Ruritania",
            "--language",
            "en",
            "--slug",
            "ruritania",
            "--no-llm",
            "--out",
            str(out),
        ]
    )
    assert code == 0 and out.exists()
    assert main(["show", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Ruritania" in printed and "12 months" in printed
