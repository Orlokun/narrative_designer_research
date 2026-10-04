"""
Ag-9 Document Curator — assigns utility flags to verified, mapped documents.

Step 9 of the pipeline. The Mapper answers *what a document is about*; the Curator
answers *what it is especially good for*. For each verified + mapped document it asks
Gemma which utility flags from ``lib/doc_flags`` apply — "good-for" lenses (military
logic, economic planning, diplomacy, popular life, ideology, technology) and quality
judgements (high-value, eyewitness) — each with a confidence and a one-line rationale
(provenance). Results land in the ``doc_flags`` table and surface as filters in the
reader, feeding future curated visualisers.

Curation is idempotent per document: a document is re-flagged wholesale (its old flags
are replaced), and the ``flags_curated_at`` stamp makes ordinary runs skip already-
curated documents. ``--re-curate`` re-mines the existing corpus after a prompt change.

Two modes:
  --no-llm : deterministic keyword heuristic (``heuristic_flags``) — offline, fast,
             good-for flags only. Useful for testing or when Ollama is down.
  default  : calls Gemma. Requires Ollama running.

CLI:
    uv run doc-curator run [--batch 50] [--all] [--re-curate] [--no-llm]
    uv run doc-curator status
    uv run doc-curator clear        # wipe all flags (dry-run unless --apply)
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from lib.config import settings
from lib.doc_flags import (
    AssignedFlag,
    build_flag_menu,
    heuristic_flags,
    parse_flags_response,
)
from lib.llm import LLMClient, LLMError
from lib.logging_setup import get_logger
from lib.run_tracker import current_run_id, record_stage

log = get_logger("doc_curator")

# ── Constants ─────────────────────────────────────────────────────────────────

_LLM_TEMPERATURE: float = 0.2
_LLM_NUM_PREDICT: int = 320  # a short JSON list of flags
_TEXT_BUDGET: int = 4000  # characters of the document body fed to the model

_CURATOR_SYSTEM: str = (
    "You are a historical-archive curator for a Project Cybersyn (Chile, 1969-1973) "
    "research pipeline. You tag documents with the downstream uses they are strongest "
    "for. Answer only with JSON."
)

_CURATOR_PROMPT: str = """Assign utility flags to this 1969-1973 Chilean-history document.
Choose ONLY from this fixed vocabulary — never invent a slug:

{menu}

Rules:
- Assign a flag ONLY if the document is a genuinely strong source for it, not merely related.
- Most documents deserve 0-2 flags. Assign none if nothing fits well.
- "high-value" and "eyewitness" are judgements about the source itself; use them sparingly.
- confidence is 0..1; rationale is one short clause saying why.

Document title: {title}

Document text:
{text}

Respond with JSON only:
{{"flags": [{{"slug": "<slug>", "confidence": 0.0-1.0, "rationale": "<why>"}}]}}"""


# ── Schema ────────────────────────────────────────────────────────────────────


def ensure_flags_table(conn: sqlite3.Connection) -> None:
    """Create the ``doc_flags`` table (idempotent)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doc_flags (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id      TEXT NOT NULL,
            flag        TEXT NOT NULL,
            confidence  REAL NOT NULL DEFAULT 1.0,
            rationale   TEXT,
            assigned_by TEXT NOT NULL,
            assigned_at TEXT NOT NULL,
            UNIQUE(doc_id, flag)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_flags_flag ON doc_flags(flag)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_flags_doc ON doc_flags(doc_id)")
    conn.commit()


def _migrate_documents_schema(conn: sqlite3.Connection) -> None:
    """Add the ``flags_curated_at`` stamp column if not already present."""
    try:
        conn.execute("ALTER TABLE documents ADD COLUMN flags_curated_at TEXT")
    except sqlite3.OperationalError:
        pass  # column already present
    conn.commit()


def _fetch_uncurated_documents(
    conn: sqlite3.Connection, batch_size: int, before_ts: str | None = None
) -> list[sqlite3.Row]:
    """Fetch verified, mapped documents to curate.

    Default (``before_ts=None``): only documents never curated
    (``flags_curated_at IS NULL``). In re-curate mode, ``before_ts`` is the session
    start, so already-curated documents are re-processed once and the loop
    self-terminates when all have been re-stamped this session.
    """
    if before_ts is None:
        return conn.execute(
            """
            SELECT doc_id, title, text
            FROM documents
            WHERE verified_at IS NOT NULL
              AND mapped_category_id IS NOT NULL
              AND flags_curated_at IS NULL
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
    return conn.execute(
        """
        SELECT doc_id, title, text
        FROM documents
        WHERE verified_at IS NOT NULL
          AND mapped_category_id IS NOT NULL
          AND (flags_curated_at IS NULL OR flags_curated_at < ?)
        LIMIT ?
        """,
        (before_ts, batch_size),
    ).fetchall()


def _replace_flags(
    conn: sqlite3.Connection,
    doc_id: str,
    flags: list[AssignedFlag],
    assigned_by: str,
) -> int:
    """Replace a document's flags wholesale; stamp it curated. Returns flags written."""
    now = datetime.now(UTC).isoformat()
    conn.execute("DELETE FROM doc_flags WHERE doc_id = ?", (doc_id,))
    for flag in flags:
        conn.execute(
            """INSERT INTO doc_flags
               (doc_id, flag, confidence, rationale, assigned_by, assigned_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (doc_id, flag.slug, flag.confidence, flag.rationale, assigned_by, now),
        )
    conn.execute("UPDATE documents SET flags_curated_at = ? WHERE doc_id = ?", (now, doc_id))
    return len(flags)


async def _llm_assign_flags(
    text: str, title: str, client: LLMClient
) -> list[AssignedFlag] | None:
    """Ask Gemma which utility flags fit the document. None on error/parse failure."""
    prompt = _CURATOR_PROMPT.format(
        menu=build_flag_menu(),
        title=(title or "(untitled)")[:200],
        text=(text or "")[:_TEXT_BUDGET],
    )
    try:
        raw = await client.chat(
            model=settings.ollama_model_npc,
            messages=[
                {"role": "system", "content": _CURATOR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=_LLM_TEMPERATURE,
            num_predict=_LLM_NUM_PREDICT,
            think=False,
        )
        return parse_flags_response(raw)
    except LLMError as exc:
        log.warning("doc_curator.llm_error", reason=str(exc)[:120])
        return None
    except Exception as exc:
        log.warning("doc_curator.llm_unexpected_error", reason=str(exc)[:120])
        return None


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class CurationResult:
    """Tally of one curation cycle."""

    processed: int = 0
    flags_written: int = 0
    docs_flagged: int = 0  # documents that received at least one flag


# ── Document Curator ──────────────────────────────────────────────────────────


class DocumentCurator:
    """Ag-9: tags verified, mapped documents with utility flags.

    Reads documents with ``verified_at`` and ``mapped_category_id`` set but no
    ``flags_curated_at``, asks Gemma (or a keyword heuristic in ``--no-llm`` mode)
    which utility flags apply, and writes them to ``doc_flags``. Curation is
    idempotent per document.
    """

    def __init__(self, db_path: Path | None = None, use_llm: bool = True) -> None:
        self.db_path = db_path or settings.archive_db
        self.use_llm = use_llm

    async def run_cycle(
        self, batch_size: int = 50, all_docs: bool = False, re_curate: bool = False
    ) -> CurationResult:
        """Curate verified+mapped documents.

        Default: one batch of not-yet-curated documents. ``all_docs=True`` loops
        batch-by-batch until none remain (self-terminating via ``flags_curated_at``).
        ``re_curate=True`` re-mines already-curated documents once this session.
        """
        if not self.db_path.exists():
            log.info("doc_curator.db_not_found", path=str(self.db_path))
            return CurationResult()

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        result = CurationResult()
        looped = all_docs or re_curate
        session_start = datetime.now(UTC).isoformat() if re_curate else None

        try:
            ensure_flags_table(conn)
            _migrate_documents_schema(conn)

            client_ctx: LLMClient | None = None
            if self.use_llm:
                client_ctx = await self._open_ollama()
                if client_ctx is None:
                    return result

            try:
                batch_num = 0
                while True:
                    batch_num += 1
                    rows = _fetch_uncurated_documents(conn, batch_size, before_ts=session_start)
                    if not rows:
                        if batch_num == 1:
                            log.info("doc_curator.nothing_to_curate")
                        else:
                            log.info("doc_curator.session_done", batches=batch_num - 1)
                        break

                    log.info(
                        "doc_curator.cycle_start",
                        batch=len(rows),
                        batch_num=batch_num,
                        use_llm=self.use_llm,
                        re_curate=re_curate,
                    )
                    for row in rows:
                        flags = await self._assign(row, client_ctx)
                        assigned_by = "curator" if self.use_llm else "heuristic"
                        written = _replace_flags(conn, row["doc_id"], flags, assigned_by)
                        result.processed += 1
                        result.flags_written += written
                        if written:
                            result.docs_flagged += 1
                    conn.commit()

                    if not looped:
                        break
            finally:
                if client_ctx is not None:
                    await client_ctx.__aexit__(None, None, None)
        finally:
            conn.close()

        log.info(
            "doc_curator.cycle_done",
            processed=result.processed,
            flags_written=result.flags_written,
            docs_flagged=result.docs_flagged,
        )
        return result

    async def _assign(self, row: sqlite3.Row, client: LLMClient | None) -> list[AssignedFlag]:
        """Assign utility flags to one document (LLM, or the heuristic fallback)."""
        text = row["text"] or ""
        title = row["title"] or ""
        if self.use_llm and client is not None:
            return await _llm_assign_flags(text, title, client) or []
        return heuristic_flags(text, title)

    async def _open_ollama(self) -> LLMClient | None:
        """Open an Ollama client and verify the model is present, else None with guidance."""
        client = LLMClient()
        await client.__aenter__()
        try:
            models = await client.list_models()
            if not any(settings.ollama_model_npc in m.get("name", "") for m in models):
                log.error("doc_curator.model_not_found", model=settings.ollama_model_npc)
                print(
                    f"\n  ERROR: Model {settings.ollama_model_npc} not found in Ollama.\n"
                    "  Start Ollama with: ollama serve\n"
                    "  Or use --no-llm fallback.\n"
                )
                await client.__aexit__(None, None, None)
                return None
        except Exception as exc:
            log.error("doc_curator.ollama_unreachable", reason=str(exc)[:120])
            print(
                "\n  ERROR: Ollama is not running.\n"
                "  Start it with: ollama serve\n"
                "  Or use --no-llm fallback.\n"
            )
            await client.__aexit__(None, None, None)
            return None
        return client

    def status(self) -> dict[str, int]:
        """Return counts: curated documents, total flags, and flags per slug."""
        if not self.db_path.exists():
            return {"curated": 0, "flags": 0}
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            ensure_flags_table(conn)
            _migrate_documents_schema(conn)
            curated = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE flags_curated_at IS NOT NULL"
            ).fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM doc_flags").fetchone()[0]
            per_flag = {
                row["flag"]: row["n"]
                for row in conn.execute(
                    "SELECT flag, COUNT(*) AS n FROM doc_flags GROUP BY flag ORDER BY n DESC"
                ).fetchall()
            }
            return {"curated": curated, "flags": total, **per_flag}
        finally:
            conn.close()


def clear_flags(conn: sqlite3.Connection, apply: bool = False) -> int:
    """Count (dry-run) or delete every assigned flag; also clears the curated stamps.

    Returns the number of flag rows affected.
    """
    ensure_flags_table(conn)
    _migrate_documents_schema(conn)
    count = conn.execute("SELECT COUNT(*) FROM doc_flags").fetchone()[0]
    if apply:
        conn.execute("DELETE FROM doc_flags")
        conn.execute("UPDATE documents SET flags_curated_at = NULL")
        conn.commit()
    return count


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    import traceback

    from lib.logging_setup import configure_logging

    configure_logging()

    ap = argparse.ArgumentParser(
        prog="doc-curator",
        description="Ag-9 Document Curator — assign utility flags to verified, mapped documents",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Assign utility flags to verified+mapped documents")
    run_p.add_argument(
        "--batch",
        type=int,
        default=50,
        help="Documents per batch (default: 50). With --all the run loops.",
    )
    run_p.add_argument(
        "--all",
        action="store_true",
        help="Curate ALL uncurated documents in a self-terminating loop.",
    )
    run_p.add_argument(
        "--re-curate",
        action="store_true",
        help="Re-curate documents already processed (applies an improved prompt).",
    )
    run_p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Gemma — use the keyword heuristic (good-for flags only).",
    )

    sub.add_parser("status", help="Print curation counts (curated docs, flags per slug)")
    clear_p = sub.add_parser("clear", help="Remove all assigned flags")
    clear_p.add_argument(
        "--apply", action="store_true", help="Actually delete (default: dry-run — just counts them)"
    )

    args = ap.parse_args()

    if args.cmd == "run":
        use_llm = not args.no_llm
        log.info(
            "doc_curator.main_start",
            batch=args.batch,
            use_llm=use_llm,
            all_docs=args.all,
            re_curate=args.re_curate,
        )
        try:
            curator = DocumentCurator(use_llm=use_llm)
            t0 = datetime.now(UTC)
            result = asyncio.run(
                curator.run_cycle(
                    batch_size=args.batch, all_docs=args.all, re_curate=args.re_curate
                )
            )
            rid = current_run_id()
            if rid:
                record_stage(
                    rid,
                    "doc_curator",
                    t0,
                    datetime.now(UTC),
                    {
                        "processed": result.processed,
                        "flags_written": result.flags_written,
                        "docs_flagged": result.docs_flagged,
                    },
                )
            log.info(
                "doc_curator.main_done",
                processed=result.processed,
                flags_written=result.flags_written,
                docs_flagged=result.docs_flagged,
            )
            print(
                f"  Processed: {result.processed} docs\n"
                f"  Flagged:   {result.docs_flagged} docs\n"
                f"  Flags:     +{result.flags_written}"
            )
        except Exception as exc:
            log.error("doc_curator.main_error", error=str(exc)[:200])
            print(f"\n  ERROR: {exc}\n")
            traceback.print_exc()

    elif args.cmd == "status":
        counts = DocumentCurator().status()
        print(f"  Curated docs: {counts.pop('curated', 0)}")
        print(f"  Total flags:  {counts.pop('flags', 0)}")
        for flag, n in counts.items():
            print(f"    {flag:20s} {n}")

    elif args.cmd == "clear":
        conn = sqlite3.connect(str(settings.archive_db))
        conn.row_factory = sqlite3.Row
        try:
            n = clear_flags(conn, apply=args.apply)
            if args.apply:
                print(f"  Deleted {n} flags; curation stamps reset.")
            else:
                print(f"  {n} flags would be deleted (dry-run — pass --apply to delete).")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
