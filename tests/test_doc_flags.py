"""Tests for lib/doc_flags.py — the utility-flag registry and its pure helpers."""

from __future__ import annotations

import json

from lib.doc_flags import (
    FLAGS,
    VALID_SLUGS,
    AssignedFlag,
    build_flag_menu,
    flag_by_slug,
    heuristic_flags,
    is_valid_flag,
    parse_flags_response,
)

# ── Registry integrity ────────────────────────────────────────────────────────


def test_slugs_are_unique_and_kebab_case():
    slugs = [f.slug for f in FLAGS]
    assert len(slugs) == len(set(slugs)), "duplicate flag slug"
    for slug in slugs:
        assert slug == slug.lower()
        assert " " not in slug


def test_kinds_are_known():
    for flag in FLAGS:
        assert flag.kind in {"good-for", "quality"}


def test_valid_slugs_matches_registry():
    assert {f.slug for f in FLAGS} == VALID_SLUGS


def test_flag_by_slug_and_is_valid():
    assert flag_by_slug("military-logic").name == "Lógica militar"
    assert flag_by_slug("does-not-exist") is None
    assert is_valid_flag("high-value")
    assert not is_valid_flag("nonsense")


def test_menu_lists_every_flag():
    menu = build_flag_menu()
    for flag in FLAGS:
        assert flag.slug in menu
        assert flag.description in menu


# ── parse_flags_response ──────────────────────────────────────────────────────


def test_parse_wellformed():
    raw = json.dumps(
        {
            "flags": [
                {"slug": "military-logic", "confidence": 0.9, "rationale": "coup planning"},
                {"slug": "diplomacy", "confidence": 0.5, "rationale": "US cable"},
            ]
        }
    )
    out = parse_flags_response(raw)
    assert [a.slug for a in out] == ["military-logic", "diplomacy"]  # sorted by confidence
    assert out[0].rationale == "coup planning"


def test_parse_drops_unknown_slug():
    raw = json.dumps(
        {
            "flags": [
                {"slug": "military-logic", "confidence": 0.8},
                {"slug": "made-up", "confidence": 0.9},
            ]
        }
    )
    out = parse_flags_response(raw)
    assert [a.slug for a in out] == ["military-logic"]


def test_parse_collapses_duplicates_keeping_highest():
    raw = json.dumps(
        {
            "flags": [
                {"slug": "diplomacy", "confidence": 0.3, "rationale": "weak"},
                {"slug": "diplomacy", "confidence": 0.8, "rationale": "strong"},
            ]
        }
    )
    out = parse_flags_response(raw)
    assert len(out) == 1
    assert out[0].confidence == 0.8
    assert out[0].rationale == "strong"


def test_parse_clamps_confidence():
    raw = json.dumps(
        {
            "flags": [
                {"slug": "high-value", "confidence": 5},
                {"slug": "eyewitness", "confidence": "bad"},
            ]
        }
    )
    out = {a.slug: a.confidence for a in parse_flags_response(raw)}
    assert out["high-value"] == 1.0
    assert out["eyewitness"] == 0.6  # default for unparseable


def test_parse_caps_number_of_flags():
    raw = json.dumps({"flags": [{"slug": f.slug, "confidence": 0.9} for f in FLAGS]})
    out = parse_flags_response(raw)
    assert len(out) <= 5


def test_parse_empty_list_ok():
    assert parse_flags_response(json.dumps({"flags": []})) == []


def test_parse_malformed_returns_none():
    assert parse_flags_response("not json") is None
    assert parse_flags_response(json.dumps([1, 2, 3])) is None
    assert parse_flags_response(json.dumps({"nope": []})) is None


# ── heuristic_flags ───────────────────────────────────────────────────────────


def test_heuristic_detects_military():
    text = "El ejército y las fuerzas armadas planean un golpe. El general dio la orden militar."
    out = {a.slug for a in heuristic_flags(text, "Memorándum militar")}
    assert "military-logic" in out


def test_heuristic_requires_repetition_off_title():
    # A single incidental keyword, not in the title, should not fire.
    text = "Una breve nota sobre economía del país."  # 'economía' once, no title cue
    out = {a.slug for a in heuristic_flags(text, "Nota breve")}
    assert "economic-planning" not in out


def test_heuristic_fires_on_title_cue():
    out = {a.slug for a in heuristic_flags("Texto sin pistas.", "Informe económico")}
    assert "economic-planning" in out


def test_heuristic_skips_quality_flags():
    # high-value has no keywords — it can never be assigned heuristically.
    text = "high-value " * 10
    out = {a.slug for a in heuristic_flags(text, "")}
    assert "high-value" not in out


def test_heuristic_caps_and_sorts():
    text = (
        "ejército militar golpe fuerzas armadas general junta. "
        "economía planificación producción nacionalización corfo industria. "
        "diplomacia embajada cable telex kissinger nixon washington. "
        "socialismo revolución pueblo propaganda consigna clase. "
        "computador telex software ingeniero técnico modelo sistema. "
        "poblador barrio obrero campesino sindicato cordón."
    )
    out = heuristic_flags(text, "")
    assert len(out) <= 5
    confidences = [a.confidence for a in out]
    assert confidences == sorted(confidences, reverse=True)


def test_assigned_flag_is_frozen():
    a = AssignedFlag(slug="diplomacy", confidence=0.5, rationale="x")
    try:
        a.confidence = 0.9  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("AssignedFlag should be frozen")
