"""Tests for lib/project.py — the research-project spec that makes the pipeline reusable."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lib.project import (
    DEFAULT_PROJECT_SLUG,
    PROJECTS_DIR,
    Period,
    ResearchProject,
    dump_project,
    load_project,
    project,
    resolve_project_path,
)

# ── Period ────────────────────────────────────────────────────────────────────


def test_period_months_spans_years_inclusive():
    period = Period(start_month="1969-10", end_month="1973-09")
    months = period.months()
    assert months[0] == "1969-10"
    assert months[-1] == "1973-09"
    assert len(months) == 48
    assert "1969-09" not in months and "1973-10" not in months


def test_period_single_month():
    assert Period(start_month="1929-10", end_month="1929-10").months() == ["1929-10"]


def test_period_rejects_reversed_range():
    with pytest.raises(ValueError):
        Period(start_month="1973-09", end_month="1969-10")


def test_period_fallback_defaults_to_middle():
    period = Period(start_month="1969-10", end_month="1973-09")
    assert period.middle_month() == "1971-10"
    explicit = Period(start_month="1969-10", end_month="1973-09", fallback_month="1971-06")
    assert explicit.middle_month() == "1971-06"


def test_period_fallback_must_be_inside():
    with pytest.raises(ValueError):
        Period(start_month="1969-10", end_month="1973-09", fallback_month="1980-01")


# ── Cybersyn spec (the shipped reference project) ─────────────────────────────


def test_default_spec_resolves_to_projects_dir():
    assert resolve_project_path(None) == PROJECTS_DIR / DEFAULT_PROJECT_SLUG / "project.yaml"
    assert resolve_project_path("cybersyn") == PROJECTS_DIR / "cybersyn" / "project.yaml"
    assert resolve_project_path("/x/y/other.yaml") == Path("/x/y/other.yaml")


def test_cybersyn_spec_loads_and_matches_the_matrix():
    spec = load_project("cybersyn")
    assert spec.slug == "cybersyn"
    assert spec.region == "Chile"
    assert len(spec.themes) == 16
    assert len(spec.months()) == 48
    assert spec.theme_by_id(11).name == "Transporte"
    assert spec.critical_month_events()["1972-10"].startswith("Paro de camioneros")
    assert spec.critical_month_labels()["1973-09"] == "11-S"
    assert set(spec.source_tiers) == {"critica", "alta", "media", "baja"}
    assert spec.entity.query_suffix == "Chile 1970 1973"
    assert spec.period_label() == "October 1969 – September 1973"
    assert spec.matrix_shape(13) == "16×13×48"


def test_cybersyn_theme_weights_in_id_order():
    weights = load_project("cybersyn").theme_weights()
    assert len(weights) == 16
    assert weights[2] == 1.0  # Economía Nacional (id 3) is critica


def test_cybersyn_category_prompt_block_is_numbered():
    block = load_project("cybersyn").category_prompt_block()
    assert block.startswith("1. Política Nacional — ")
    assert "16. Macroeconomía — " in block


def test_active_project_is_cached_default():
    assert project() is project()
    assert project().slug == "cybersyn"


# ── Validation of hand-written specs ──────────────────────────────────────────


def minimal_spec(**overrides) -> dict:
    """A small valid spec for another historical moment (shared with other tests)."""
    base = {
        "slug": "weimar",
        "title": "Weimar Berlin",
        "region": "Germany",
        "period": {"start_month": "1929-01", "end_month": "1933-03"},
        "themes": [
            {"id": 1, "name": "Politics", "weight": 1.0, "tier": "critica"},
            {"id": 2, "name": "Culture", "weight": 0.5, "tier": "media"},
        ],
        "critical_months": [{"month_iso": "1933-01", "event": "Hitler appointed Chancellor"}],
        "source_tiers": {
            "critica": ["archive.org"],
            "alta": ["archive.org"],
            "media": [],
            "baja": [],
        },
        "entity": {
            "nominal_theme_id": 1,
            "nominal_month_iso": "1930-01",
            "query_suffix": "Germany 1929 1933",
            "context_suffix": "Weimar Republic",
        },
        "relevance": {"domain": "Germany 1929-1933", "persona": "You are an archivist."},
        "classification": {"persona": "You classify documents."},
        "reformulation": {"system": "Rewrite queries in 1930s German vocabulary."},
        "entity_prompts": {
            "era_label": "Weimar Germany (1929-1933)",
            "era_label_local": "die Weimarer Republik (1929-1933)",
        },
    }
    base.update(overrides)
    return base


def test_minimal_spec_validates_and_derives_views():
    spec = ResearchProject.model_validate(minimal_spec())
    assert len(spec.months()) == 51
    assert spec.theme_names() == {1: "Politics", 2: "Culture"}
    assert spec.base_terms() == {1: [], 2: []}
    assert spec.critical_month_labels() == {"1933-01": "Hitler"}
    assert spec.period.end_year == 1933


def test_spec_rejects_duplicate_theme_ids():
    themes = [
        {"id": 1, "name": "A", "weight": 1, "tier": "alta"},
        {"id": 1, "name": "B", "weight": 1, "tier": "alta"},
    ]
    with pytest.raises(ValueError, match="unique"):
        ResearchProject.model_validate(minimal_spec(themes=themes))


def test_spec_rejects_missing_tier():
    with pytest.raises(ValueError, match="missing tier"):
        ResearchProject.model_validate(minimal_spec(source_tiers={"critica": [], "alta": []}))


def test_spec_rejects_critical_month_outside_period():
    bad = minimal_spec(critical_months=[{"month_iso": "1945-05", "event": "too late"}])
    with pytest.raises(ValueError, match="outside the period"):
        ResearchProject.model_validate(bad)


def test_spec_rejects_entity_theme_not_in_themes():
    bad = minimal_spec()
    bad["entity"]["nominal_theme_id"] = 99
    with pytest.raises(ValueError, match="nominal_theme_id"):
        ResearchProject.model_validate(bad)


def test_spec_rejects_unknown_keys():
    with pytest.raises(ValueError):
        ResearchProject.model_validate(minimal_spec(mystery="x"))


# ── Round trip ────────────────────────────────────────────────────────────────


def test_dump_and_reload_round_trip(tmp_path: Path):
    spec = ResearchProject.model_validate(minimal_spec())
    out = tmp_path / "weimar" / "project.yaml"
    dump_project(spec, out)
    raw = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert raw["slug"] == "weimar"
    assert load_project(out) == spec
