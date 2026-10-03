"""Tests for lib/politics.py — the character political compass + relation refinement."""

from __future__ import annotations

import pytest

from lib.politics import (
    FAR_THRESHOLD,
    clamp_axis,
    compass_quadrant,
    parse_political_response,
    political_distance,
    refine_relation_kind,
)


class TestClampAxis:
    def test_passes_through_in_range(self):
        assert clamp_axis(-0.8) == -0.8
        assert clamp_axis(0.0) == 0.0

    def test_clamps_out_of_range(self):
        assert clamp_axis(2.5) == 1.0
        assert clamp_axis(-9) == -1.0

    def test_none_for_non_number(self):
        assert clamp_axis("left") is None
        assert clamp_axis(None) is None


class TestPoliticalDistance:
    def test_euclidean(self):
        assert political_distance(0, 0, 0, 0) == 0.0
        assert political_distance(-1, -1, 1, 1) == pytest.approx((8) ** 0.5)

    def test_none_when_incomplete(self):
        assert political_distance(None, 0, 0, 0) is None
        assert political_distance(0, 0, 0, None) is None


class TestCompassQuadrant:
    def test_labels(self):
        assert compass_quadrant(-0.8, -0.3) == "izquierda · libertario"
        assert compass_quadrant(0.7, 0.6) == "derecha · autoritario"

    def test_dead_zone_is_centre(self):
        assert compass_quadrant(0.05, -0.05) == "centro · moderado"

    def test_empty_when_unknown(self):
        assert compass_quadrant(None, 0.2) == ""


class TestRefineRelationKind:
    # Kissinger (right/authoritarian) vs Allende (left/liberal) — far apart.
    KISS = (0.6, 0.5)
    ALLENDE = (-0.8, -0.3)

    def test_afiliacion_between_opposites_becomes_contraparte(self):
        assert refine_relation_kind("afiliacion", *self.KISS, *self.ALLENDE) == "contraparte"

    def test_afiliacion_between_allies_kept(self):
        # Two leftists close together stay affiliated.
        assert refine_relation_kind("afiliacion", -0.8, -0.2, -0.7, -0.1) == "afiliacion"

    def test_other_kinds_never_changed(self):
        # Only afiliacion is second-guessed; evidence-based kinds stand.
        assert refine_relation_kind("enemigo", *self.KISS, *self.ALLENDE) == "enemigo"
        assert refine_relation_kind("colega", *self.KISS, *self.ALLENDE) == "colega"

    def test_unchanged_when_position_unknown(self):
        assert refine_relation_kind("afiliacion", None, None, -0.8, -0.3) == "afiliacion"

    def test_threshold_is_exclusive(self):
        # Exactly at the threshold is not "far".
        d = FAR_THRESHOLD
        assert refine_relation_kind("afiliacion", 0.0, 0.0, d, 0.0) == "afiliacion"


class TestParsePoliticalResponse:
    def test_parses_and_clamps(self):
        raw = '{"economic": -0.8, "social": -0.3, "label": "socialista"}'
        assert parse_political_response(raw) == (-0.8, -0.3, "socialista")

    def test_clamps_out_of_range(self):
        raw = '{"economic": -5, "social": 9, "label": "x"}'
        assert parse_political_response(raw) == (-1.0, 1.0, "x")

    def test_label_optional(self):
        assert parse_political_response('{"economic": 0.1, "social": 0.2}') == (0.1, 0.2, "")

    def test_none_when_axes_missing(self):
        assert parse_political_response('{"label": "x"}') is None
        assert parse_political_response('{"economic": 0.1}') is None

    def test_none_for_malformed(self):
        assert parse_political_response("not json") is None
        assert parse_political_response('{"economic": "left", "social": 0}') is None
