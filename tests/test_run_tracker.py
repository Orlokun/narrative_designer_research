"""Tests for lib.run_tracker — pipeline run/stage tracking (offline, temp SQLite)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from lib.config import settings
from lib.run_tracker import (
    current_run_id,
    ensure_pipeline_tables,
    finish_run,
    record_stage,
    start_run,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point settings.archive_db at a temp file so run_tracker writes there."""
    path = tmp_path / "archivo.sqlite"
    monkeypatch.setattr(settings, "archive_db", path)
    return path


def _conn(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


class TestCurrentRunId:
    def test_returns_env_value(self, monkeypatch):
        monkeypatch.setenv("CYBERSYN_RUN_ID", "run-123")
        assert current_run_id() == "run-123"

    def test_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("CYBERSYN_RUN_ID", raising=False)
        assert current_run_id() is None

    def test_none_when_blank(self, monkeypatch):
        monkeypatch.setenv("CYBERSYN_RUN_ID", "   ")
        assert current_run_id() is None


class TestEnsurePipelineTables:
    def test_creates_both_tables(self):
        conn = sqlite3.connect(":memory:")
        ensure_pipeline_tables(conn)
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert {"pipeline_runs", "pipeline_run_stages"} <= tables

    def test_idempotent(self):
        conn = sqlite3.connect(":memory:")
        ensure_pipeline_tables(conn)
        ensure_pipeline_tables(conn)  # must not raise


class TestStartFinishRun:
    def test_start_inserts_running_row(self, db):
        start_run("run-1", trigger="pipeline")
        row = _conn(db).execute("SELECT * FROM pipeline_runs WHERE run_id='run-1'").fetchone()
        assert row["status"] == "running"
        assert row["trigger"] == "pipeline"
        assert row["started_at"]
        assert row["finished_at"] is None

    def test_finish_sets_finished_and_status(self, db):
        start_run("run-1", trigger="pipeline")
        finish_run("run-1", status="done")
        row = _conn(db).execute("SELECT * FROM pipeline_runs WHERE run_id='run-1'").fetchone()
        assert row["status"] == "done"
        assert row["finished_at"]

    def test_finish_without_start_creates_row(self, db):
        finish_run("orphan", status="failed")
        row = _conn(db).execute("SELECT * FROM pipeline_runs WHERE run_id='orphan'").fetchone()
        assert row is not None
        assert row["status"] == "failed"


class TestRecordStage:
    def test_inserts_stage_with_duration_and_counts(self, db):
        t0 = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        t1 = t0 + timedelta(seconds=4, milliseconds=500)
        record_stage("run-1", "archivero", t0, t1, {"documents_new": 7})
        row = _conn(db).execute("SELECT * FROM pipeline_run_stages WHERE run_id='run-1'").fetchone()
        assert row["stage"] == "archivero"
        assert row["duration_s"] == pytest.approx(4.5)
        assert json.loads(row["counts"]) == {"documents_new": 7}

    def test_multiple_stages_for_one_run(self, db):
        t0 = datetime.now(UTC)
        record_stage("run-1", "propositor", t0, t0, {})
        record_stage("run-1", "archivero", t0, t0, {"documents_new": 1})
        n = (
            _conn(db)
            .execute("SELECT COUNT(*) FROM pipeline_run_stages WHERE run_id='run-1'")
            .fetchone()[0]
        )
        assert n == 2

    def test_empty_counts_default(self, db):
        t0 = datetime.now(UTC)
        record_stage("run-1", "mapper", t0, t0, None)
        row = (
            _conn(db)
            .execute("SELECT counts FROM pipeline_run_stages WHERE run_id='run-1'")
            .fetchone()
        )
        assert json.loads(row["counts"]) == {}
