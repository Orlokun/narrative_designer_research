"""
Per-run tracking for full pipeline cycles.

A `make pipeline` run mints one `run_id` (exported as the CYBERSYN_RUN_ID env var) and
brackets the run with `pipeline-run start` / `pipeline-run finish`. Each agent reads the
env var and records its stage timing + counts. This is the backing data for the Admin
Performance page (per-stage timing) and Results page (what the latest run produced).

Schema (in archivo.sqlite):
    pipeline_runs        — one row per run (run_id, trigger, started/finished, status)
    pipeline_run_stages  — one row per agent/section per run (timing + JSON counts)

Stage recording is a no-op when CYBERSYN_RUN_ID is unset, so standalone single-agent
runs (e.g. `make archivero`) are unaffected.

CLI:
    uv run pipeline-run start  [--trigger pipeline]
    uv run pipeline-run finish [--status done]
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from lib.config import settings
from lib.logging_setup import get_logger

log = get_logger("run_tracker")

ENV_VAR: str = "CYBERSYN_RUN_ID"


# ── Env ──────────────────────────────────────────────────────────────────────

def current_run_id() -> str | None:
    """Return the active run id from the environment, or None if unset/blank."""
    rid = os.environ.get(ENV_VAR, "").strip()
    return rid or None


# ── Schema ───────────────────────────────────────────────────────────────────

def ensure_pipeline_tables(conn: sqlite3.Connection) -> None:
    """Create the pipeline_runs and pipeline_run_stages tables if absent (idempotent)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id      TEXT PRIMARY KEY,
            trigger     TEXT,
            started_at  TEXT NOT NULL,
            finished_at TEXT,
            status      TEXT NOT NULL DEFAULT 'running'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_run_stages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL,
            stage       TEXT NOT NULL,
            started_at  TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            duration_s  REAL NOT NULL,
            counts      TEXT NOT NULL DEFAULT '{}'
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_run_stages_run_id ON pipeline_run_stages(run_id)"
    )
    conn.commit()


def _connect() -> sqlite3.Connection:
    """Open the archive DB, creating its parent directory if needed."""
    db_path: Path = settings.archive_db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


# ── Run lifecycle ──────────────────────────────────────────────────────────────

def start_run(run_id: str, trigger: str = "manual") -> None:
    """Record the start of a pipeline run (upsert; resets status to 'running')."""
    now = datetime.now(UTC).isoformat()
    conn = _connect()
    try:
        ensure_pipeline_tables(conn)
        conn.execute(
            """
            INSERT INTO pipeline_runs (run_id, trigger, started_at, status)
            VALUES (?, ?, ?, 'running')
            ON CONFLICT(run_id) DO UPDATE
              SET trigger=excluded.trigger, started_at=excluded.started_at, status='running'
            """,
            (run_id, trigger, now),
        )
        conn.commit()
    finally:
        conn.close()
    log.info("run_tracker.run_started", run_id=run_id, trigger=trigger)


def finish_run(run_id: str, status: str = "done") -> None:
    """Mark a pipeline run finished (creates the row if start was missed)."""
    now = datetime.now(UTC).isoformat()
    conn = _connect()
    try:
        ensure_pipeline_tables(conn)
        updated = conn.execute(
            "UPDATE pipeline_runs SET finished_at=?, status=? WHERE run_id=?",
            (now, status, run_id),
        ).rowcount
        if not updated:
            conn.execute(
                """
                INSERT INTO pipeline_runs (run_id, trigger, started_at, finished_at, status)
                VALUES (?, 'manual', ?, ?, ?)
                """,
                (run_id, now, now, status),
            )
        conn.commit()
    finally:
        conn.close()
    log.info("run_tracker.run_finished", run_id=run_id, status=status)


def record_stage(
    run_id: str,
    stage: str,
    started_at: datetime,
    finished_at: datetime,
    counts: dict | None = None,
) -> None:
    """Append a per-stage timing row for a run. Safe to call from any agent's main()."""
    duration_s = (finished_at - started_at).total_seconds()
    conn = _connect()
    try:
        ensure_pipeline_tables(conn)
        conn.execute(
            """
            INSERT INTO pipeline_run_stages
                (run_id, stage, started_at, finished_at, duration_s, counts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                stage,
                started_at.isoformat(),
                finished_at.isoformat(),
                duration_s,
                json.dumps(counts or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    log.info(
        "run_tracker.stage_recorded",
        run_id=run_id,
        stage=stage,
        duration_s=round(duration_s, 2),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    from lib.logging_setup import configure_logging

    configure_logging()

    ap = argparse.ArgumentParser(
        prog="pipeline-run",
        description="Bracket a full pipeline run; reads the run id from $CYBERSYN_RUN_ID.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    start_p = sub.add_parser("start", help="Record the start of a pipeline run")
    start_p.add_argument("--trigger", default="pipeline", help="What launched the run")

    finish_p = sub.add_parser("finish", help="Mark the current pipeline run finished")
    finish_p.add_argument("--status", default="done", help="Final status (done|failed)")

    args = ap.parse_args()

    run_id = current_run_id()
    if not run_id:
        print(f"\n  ERROR: ${ENV_VAR} is not set; cannot {args.cmd} a run.\n")
        raise SystemExit(1)

    if args.cmd == "start":
        start_run(run_id, trigger=args.trigger)
        print(run_id)
    elif args.cmd == "finish":
        finish_run(run_id, status=args.status)
        print(run_id)


if __name__ == "__main__":
    main()
