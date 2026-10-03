"""Tests for lib/coverage.py — the shared 3D (theme × genre × month) coverage table."""

from __future__ import annotations

import sqlite3

import pytest

from lib.coverage import ensure_coverage_table, upsert_coverage_score
from lib.genres import GENRE_UNCLASSIFIED


@pytest.fixture
def mem_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def _columns(conn) -> set[str]:
    return {r[1] for r in conn.execute("PRAGMA table_info(coverage)").fetchall()}


class TestEnsureCoverageTable:
    def test_creates_3d_table(self, mem_conn):
        ensure_coverage_table(mem_conn)
        assert {"category_id", "genre_id", "month_iso", "coverage_score"} <= _columns(mem_conn)

    def test_idempotent(self, mem_conn):
        ensure_coverage_table(mem_conn)
        ensure_coverage_table(mem_conn)  # must not raise

    def test_migrates_legacy_2d_table_to_genre_zero(self, mem_conn):
        mem_conn.execute("""
            CREATE TABLE coverage (
                category_id    INTEGER NOT NULL,
                month_iso      TEXT    NOT NULL,
                coverage_score REAL    NOT NULL DEFAULT 0.0,
                updated_at     TEXT    NOT NULL,
                PRIMARY KEY (category_id, month_iso)
            )
        """)
        mem_conn.execute(
            "INSERT INTO coverage VALUES (3, '1972-10', 0.6, '2026-01-01')"
        )
        ensure_coverage_table(mem_conn)
        row = mem_conn.execute(
            "SELECT genre_id, coverage_score FROM coverage "
            "WHERE category_id=3 AND month_iso='1972-10'"
        ).fetchone()
        assert row["genre_id"] == GENRE_UNCLASSIFIED
        assert row["coverage_score"] == pytest.approx(0.6)
        # Legacy scratch table must be gone.
        tables = {r[0] for r in mem_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert tables == {"coverage"}


class TestUpsertCoverageScore:
    def test_inserts_scored_cell(self, mem_conn):
        ensure_coverage_table(mem_conn)
        upsert_coverage_score(mem_conn, category_id=4, month_iso="1972-10",
                              docs_found=3, genre_id=8)
        row = mem_conn.execute(
            "SELECT coverage_score FROM coverage "
            "WHERE category_id=4 AND genre_id=8 AND month_iso='1972-10'"
        ).fetchone()
        assert row[0] == pytest.approx(0.6)  # 3/5

    def test_genre_defaults_to_unclassified(self, mem_conn):
        ensure_coverage_table(mem_conn)
        upsert_coverage_score(mem_conn, category_id=1, month_iso="1970-01", docs_found=5)
        row = mem_conn.execute("SELECT genre_id, coverage_score FROM coverage").fetchone()
        assert row["genre_id"] == GENRE_UNCLASSIFIED
        assert row["coverage_score"] == pytest.approx(1.0)

    def test_same_cell_updates_in_place(self, mem_conn):
        ensure_coverage_table(mem_conn)
        upsert_coverage_score(mem_conn, category_id=4, month_iso="1972-10",
                              docs_found=1, genre_id=8)
        upsert_coverage_score(mem_conn, category_id=4, month_iso="1972-10",
                              docs_found=4, genre_id=8)
        rows = mem_conn.execute("SELECT coverage_score FROM coverage").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == pytest.approx(0.8)

    def test_genres_are_independent_cells(self, mem_conn):
        ensure_coverage_table(mem_conn)
        upsert_coverage_score(mem_conn, category_id=4, month_iso="1972-10",
                              docs_found=5, genre_id=1)
        upsert_coverage_score(mem_conn, category_id=4, month_iso="1972-10",
                              docs_found=1, genre_id=6)
        rows = mem_conn.execute(
            "SELECT genre_id, coverage_score FROM coverage ORDER BY genre_id"
        ).fetchall()
        assert [(r[0], r[1]) for r in rows] == [(1, 1.0), (6, pytest.approx(0.2))]

    def test_caps_at_one(self, mem_conn):
        ensure_coverage_table(mem_conn)
        upsert_coverage_score(mem_conn, category_id=5, month_iso="1973-06",
                              docs_found=10, genre_id=2)
        assert mem_conn.execute("SELECT coverage_score FROM coverage").fetchone()[0] == 1.0
