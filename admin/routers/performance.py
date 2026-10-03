"""Per-run pipeline timing — the Admin Performance page backend.

Reads the `pipeline_runs` and `pipeline_run_stages` tables written by
`lib.run_tracker`. Each `make pipeline` run mints one `run_id`; every agent stamps
its stage timing + counts against it. This router exposes:

  GET /performance            → recent runs, each with a per-stage timing breakdown
  GET /performance/{run_id}   → a single run's stages in detail

All endpoints degrade gracefully: they return an empty, well-typed payload when the
database or the run-tracking tables do not yet exist.
"""

from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter

from lib.config import settings

router = APIRouter(tags=["performance"])

_RUN_LIMIT = 50  # most-recent runs surfaced by the list endpoint


def _stage_row(row: sqlite3.Row) -> dict:
    """Shape one pipeline_run_stages row for the API, parsing its JSON counts."""
    try:
        counts = json.loads(row["counts"]) if row["counts"] else {}
    except (ValueError, TypeError):
        counts = {}
    return {
        "stage": row["stage"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "duration_s": round(row["duration_s"], 3),
        "counts": counts,
    }


def _wall_clock_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    """Wall-clock duration of a run from its start/finish ISO timestamps, or None."""
    if not started_at or not finished_at:
        return None
    from datetime import datetime

    try:
        delta = datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)
    except ValueError:
        return None
    return round(delta.total_seconds(), 3)


@router.get("/performance")
async def performance_runs() -> dict:
    """
    Return the most-recent pipeline runs, each with its per-stage timing breakdown.

    Each run carries its wall-clock duration (finish minus start), the summed stage time,
    and the list of stages so the page can show how long each session took, broken
    down per agent/section.
    """
    db_path = settings.archive_db
    empty = {"available": False, "runs": []}

    if not db_path.exists():
        return empty

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except Exception:
        return empty

    try:
        try:
            run_rows = conn.execute(
                "SELECT run_id, trigger, started_at, finished_at, status "
                "FROM pipeline_runs ORDER BY started_at DESC LIMIT ?",
                (_RUN_LIMIT,),
            ).fetchall()
        except Exception:
            return empty

        stages_by_run: dict[str, list[dict]] = {}
        try:
            for s in conn.execute(
                "SELECT run_id, stage, started_at, finished_at, duration_s, counts "
                "FROM pipeline_run_stages ORDER BY started_at ASC"
            ).fetchall():
                stages_by_run.setdefault(s["run_id"], []).append(_stage_row(s))
        except Exception:
            stages_by_run = {}

        runs = []
        for r in run_rows:
            stages = stages_by_run.get(r["run_id"], [])
            runs.append(
                {
                    "run_id": r["run_id"],
                    "trigger": r["trigger"],
                    "started_at": r["started_at"],
                    "finished_at": r["finished_at"],
                    "status": r["status"],
                    "duration_s": _wall_clock_seconds(r["started_at"], r["finished_at"]),
                    "stages_total_s": round(sum(s["duration_s"] for s in stages), 3),
                    "stage_count": len(stages),
                    "stages": stages,
                }
            )
    finally:
        conn.close()

    return {"available": True, "runs": runs}


@router.get("/performance/{run_id}")
async def performance_run_detail(run_id: str) -> dict:
    """Return a single run's metadata plus its ordered per-stage timing rows."""
    db_path = settings.archive_db
    empty = {"available": False, "run": None, "stages": []}

    if not db_path.exists():
        return empty

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except Exception:
        return empty

    try:
        try:
            run = conn.execute(
                "SELECT run_id, trigger, started_at, finished_at, status "
                "FROM pipeline_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        except Exception:
            return empty

        if run is None:
            return {"available": True, "run": None, "stages": []}

        try:
            stage_rows = conn.execute(
                "SELECT stage, started_at, finished_at, duration_s, counts "
                "FROM pipeline_run_stages WHERE run_id = ? ORDER BY started_at ASC",
                (run_id,),
            ).fetchall()
        except Exception:
            stage_rows = []

        stages = [_stage_row(s) for s in stage_rows]
        run_payload = {
            "run_id": run["run_id"],
            "trigger": run["trigger"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "status": run["status"],
            "duration_s": _wall_clock_seconds(run["started_at"], run["finished_at"]),
            "stages_total_s": round(sum(s["duration_s"] for s in stages), 3),
            "stage_count": len(stages),
        }
    finally:
        conn.close()

    return {"available": True, "run": run_payload, "stages": stages}
