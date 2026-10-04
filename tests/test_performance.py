"""Tests for the Admin Performance router (admin.routers.performance).

Strategy:
  - httpx.AsyncClient + ASGITransport to talk to the FastAPI app in-process.
  - settings.archive_db monkeypatched to a temp SQLite file, populated through the
    real lib.run_tracker helpers so the test exercises the same schema the pipeline writes.
  - Fully offline; no Ollama, no live HTTP.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from admin.app import app
from lib.config import settings
from lib.run_tracker import finish_run, record_stage, start_run


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Point settings.archive_db at a temp file the router and run_tracker share."""
    path = tmp_path / "archivo.sqlite"
    monkeypatch.setattr(settings, "archive_db", path)
    return path


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _seed_run(run_id: str, *, trigger: str = "pipeline", started: datetime | None = None):
    """Create one finished run with two stages; returns nothing (writes to the temp DB)."""
    t0 = started or datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
    start_run(run_id, trigger=trigger)
    record_stage(run_id, "propositor", t0, t0 + timedelta(seconds=2), {"missions_new": 3})
    record_stage(
        run_id,
        "archivero",
        t0 + timedelta(seconds=2),
        t0 + timedelta(seconds=12),
        {"documents_new": 7},
    )
    finish_run(run_id, status="done")


# ── /performance (list) ────────────────────────────────────────────────────────


class TestPerformanceList:
    async def test_missing_db_returns_empty(self, client, temp_db):
        # temp_db points at a path that does not exist yet
        r = await client.get("/api/performance")
        assert r.status_code == 200
        body = r.json()
        assert body == {"available": False, "runs": []}

    async def test_lists_run_with_stage_breakdown(self, client, temp_db):
        _seed_run("run-1")
        r = await client.get("/api/performance")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert len(body["runs"]) == 1

        run = body["runs"][0]
        assert run["run_id"] == "run-1"
        assert run["trigger"] == "pipeline"
        assert run["status"] == "done"
        assert run["stage_count"] == 2
        assert run["stages_total_s"] == pytest.approx(12.0)
        assert [s["stage"] for s in run["stages"]] == ["propositor", "archivero"]
        assert run["stages"][1]["counts"] == {"documents_new": 7}

    async def test_runs_ordered_most_recent_first(self, client, temp_db):
        _seed_run("older", started=datetime(2026, 6, 1, 9, 0, 0, tzinfo=UTC))
        _seed_run("newer", started=datetime(2026, 6, 9, 9, 0, 0, tzinfo=UTC))
        r = await client.get("/api/performance")
        ids = [run["run_id"] for run in r.json()["runs"]]
        assert ids == ["newer", "older"]

    async def test_running_run_has_null_wall_clock(self, client, temp_db):
        start_run("live", trigger="pipeline")
        r = await client.get("/api/performance")
        run = r.json()["runs"][0]
        assert run["status"] == "running"
        assert run["duration_s"] is None
        assert run["stage_count"] == 0


# ── /performance/{run_id} (detail) ─────────────────────────────────────────────


class TestPerformanceDetail:
    async def test_missing_db_returns_empty(self, client, temp_db):
        r = await client.get("/api/performance/whatever")
        assert r.status_code == 200
        assert r.json() == {"available": False, "run": None, "stages": []}

    async def test_unknown_run_available_but_null(self, client, temp_db):
        _seed_run("run-1")
        r = await client.get("/api/performance/does-not-exist")
        body = r.json()
        assert body["available"] is True
        assert body["run"] is None
        assert body["stages"] == []

    async def test_returns_run_and_ordered_stages(self, client, temp_db):
        _seed_run("run-1")
        r = await client.get("/api/performance/run-1")
        body = r.json()
        assert body["available"] is True
        assert body["run"]["run_id"] == "run-1"
        assert body["run"]["duration_s"] is not None
        assert [s["stage"] for s in body["stages"]] == ["propositor", "archivero"]
        assert body["stages"][0]["duration_s"] == pytest.approx(2.0)
