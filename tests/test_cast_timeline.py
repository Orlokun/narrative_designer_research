"""Tests for the cast timeline span helpers (the character life-line chart)."""

from __future__ import annotations

import pytest

from admin.routers.cast import _timeline_span, _year_fraction

HIGHLIGHT_START = 1970
HIGHLIGHT_END = 1974


class TestYearFraction:
    def test_bare_year(self):
        assert _year_fraction("1908") == 1908.0

    def test_year_month(self):
        assert _year_fraction("1970-10") == pytest.approx(1970 + 9 / 12)

    def test_year_month_day(self):
        assert _year_fraction("1973-09-11") == pytest.approx(
            1973 + 8 / 12 + (11 - 1) / 31 / 12
        )

    def test_year_range_takes_first_year(self):
        # The corpus produces date_iso like "1970-1973"; treat as the first year.
        assert _year_fraction("1970-1973") == 1970.0

    def test_zero_month_is_year_start(self):
        assert _year_fraction("1970-00") == 1970.0

    def test_circa_prefix(self):
        assert _year_fraction("c. 1908") == 1908.0

    def test_none_and_undated(self):
        assert _year_fraction(None) is None
        assert _year_fraction("") is None
        assert _year_fraction("s/f") is None


class TestTimelineSpan:
    def test_always_includes_highlight_period(self):
        start, end = _timeline_span(None, None, [])
        assert start <= HIGHLIGHT_START
        assert end >= HIGHLIGHT_END

    def test_birth_and_death_bound_the_span(self):
        start, end = _timeline_span("1908", "1973-09-11", ["1970-10", "1972-06"])
        assert start == 1908
        assert end == HIGHLIGHT_END  # death 1973 < 1974, so the band fixes the end

    def test_death_after_the_period(self):
        start, end = _timeline_span("1923", "2023", [])
        assert start == 1923
        assert end == 2023

    def test_facts_extend_when_no_life_dates(self):
        start, end = _timeline_span(None, None, ["1969-10", "1973-08"])
        assert start == 1969
        assert end == HIGHLIGHT_END

    def test_returns_integers(self):
        start, end = _timeline_span("1908", "1973", ["1971-07"])
        assert isinstance(start, int) and isinstance(end, int)
        assert end > start

    def test_undated_facts_ignored(self):
        start, end = _timeline_span(None, None, [None, "s/f", "1971-01"])
        assert start == HIGHLIGHT_START
        assert end == HIGHLIGHT_END
