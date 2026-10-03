"""Tests for lib/genres.py — the genre axis registry of the coverage matrix."""

from __future__ import annotations

from lib.genres import (
    GENRE_NAMES,
    GENRE_UNCLASSIFIED,
    GENRES,
    VALID_GENRE_IDS,
    genre_by_id,
    plausibility,
)

# Connectors actually wired through the Gatekeeper/Archivero today.
_KNOWN_CONNECTORS = {
    "archive.org", "openalex", "crossref", "wikisource_es", "frus",
    "semantic_scholar", "chronicling_america", "marxists", "wikipedia_es", "web_serp",
    "foia_chile",
}

_CATEGORY_IDS = range(1, 17)


class TestRegistryIntegrity:
    def test_thirteen_genres(self):
        assert len(GENRES) == 13

    def test_ids_are_one_to_thirteen_unique(self):
        assert [g.id for g in GENRES] == list(range(1, 14))
        assert set(range(1, 14)) == VALID_GENRE_IDS

    def test_unclassified_is_not_a_genre(self):
        assert GENRE_UNCLASSIFIED == 0
        assert GENRE_UNCLASSIFIED not in VALID_GENRE_IDS

    def test_names_and_slugs_unique_and_nonempty(self):
        names = [g.name for g in GENRES]
        slugs = [g.slug for g in GENRES]
        assert all(names) and len(set(names)) == len(names)
        assert all(slugs) and len(set(slugs)) == len(slugs)

    def test_every_genre_has_description_terms_and_sources(self):
        for genre in GENRES:
            assert genre.description.strip(), genre.slug
            assert genre.search_terms, genre.slug
            assert genre.sources, genre.slug

    def test_sources_are_known_connectors(self):
        for genre in GENRES:
            unknown = set(genre.sources) - _KNOWN_CONNECTORS
            assert not unknown, f"{genre.slug}: unknown connectors {unknown}"

    def test_genre_names_lookup_matches(self):
        assert GENRE_NAMES[1] == "Discurso"
        assert len(GENRE_NAMES) == 13


class TestPlausibility:
    def test_all_priors_in_unit_interval(self):
        for genre in GENRES:
            assert 0.0 <= genre.default_plausibility <= 1.0, genre.slug
            for category_id, weight in genre.plausibility_overrides.items():
                assert category_id in _CATEGORY_IDS, genre.slug
                assert 0.0 <= weight <= 1.0, f"{genre.slug}/{category_id}"

    def test_override_beats_default(self):
        # Cable diplomático: EEUU is the natural home, Deporte nearly empty.
        assert plausibility(6, 3) == 1.0
        assert plausibility(15, 3) == 0.1

    def test_default_used_when_no_override(self):
        genre = genre_by_id(9)  # Académico
        assert plausibility(7, 9) == genre.default_plausibility

    def test_unknown_genre_is_zero(self):
        assert plausibility(1, GENRE_UNCLASSIFIED) == 0.0
        assert plausibility(1, 99) == 0.0

    def test_every_theme_has_at_least_one_strong_genre(self):
        # No matrix row should be unreachable: each theme needs a genre with prior >= 0.7.
        for category_id in _CATEGORY_IDS:
            best = max(plausibility(category_id, g.id) for g in GENRES)
            assert best >= 0.7, f"category {category_id} has no strong genre (best={best})"


class TestGenreById:
    def test_finds_each_registered_genre(self):
        for genre in GENRES:
            assert genre_by_id(genre.id) is genre

    def test_none_for_unknown(self):
        assert genre_by_id(0) is None
        assert genre_by_id(14) is None
