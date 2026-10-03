"""
Ag-2 Verificator — quality-scores documents in archivo.sqlite and marks approved ones.

Two-stage scoring pipeline:
  Stage 1 — Fast pre-filter (sync, no I/O):
      Hard-rejects obvious junk: texts that are too short, error pages, or bot-
      challenge pages. Everything else is passed to the LLM.

  Stage 2 — LLM scoring (Gemma4, async):
      Documents that pass the pre-filter are scored by the local Ollama model.
      The prompt is language-agnostic: English CIA cables, Spanish CORFO memos,
      and academic papers in any language all receive fair assessment.
      If the LLM is unavailable or returns an unparseable response, the document
      is LEFT UNSCORED (quality_score stays NULL) so it can be retried next run.
      Documents are NEVER given a passing score via fallback.

Behaviour by mode:
  --no-llm  : Only hard-rejects run (too short / error pages). Everything else
              stays NULL — the LLM is required to actually verify anything.
  default   : Pre-filter then LLM. Ollama must be running.

CLI:
    uv run verificator run [--min-score 0.6] [--batch 50] [--no-llm]
    uv run verificator status
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncIterator

from lib.config import settings
from lib.llm import LLMClient, LLMError
from lib.logging_setup import get_logger
from lib.project import project
from lib.run_tracker import current_run_id, record_stage

log = get_logger("verificator")

# ── Constants ──────────────────────────────────────────────────────────────────

MIN_DOCUMENT_LENGTH: int = 100
_DEFAULT_MIN_SCORE: float = 0.6
_LLM_TEMPERATURE: float = 0.1   # low temp → deterministic, consistent ratings
_LLM_NUM_PREDICT: int = 32      # thinking disabled; 32 tokens is ample for a decimal

# Documents shorter than this are excerpts/snippets (abstracts, WikiSource snippets).
# They pass verification but are flagged as incomplete so better sources can be sought.
MIN_COMPLETE_LENGTH: int = 1_000

# ── LLM prompt ────────────────────────────────────────────────────────────────

_PROJECT = project()

_LLM_SYSTEM = (
    _PROJECT.relevance.persona
    + " You evaluate documents for research relevance. "
    "Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation."
)

_LLM_PROMPT_TEMPLATE = """\
Rate the relevance of this document to this EXACT research domain:
  {domain}

Topics that ARE relevant (score high):
{relevant_topics}

CRITICAL TIME CONSTRAINT — the following score 0.0 regardless of content:
{zero_score_rules}

The document may be in any language. English diplomatic cables score as high as Spanish sources.

Title: {title}
Text:
{text}

Respond with ONLY a decimal number between 0.0 and 1.0:
{score_scale}"""


def _render_prompt_template() -> str:
    """Fill the project-specific parts of the relevance prompt ({title}/{text} stay)."""
    relevance = _PROJECT.relevance
    return (
        _LLM_PROMPT_TEMPLATE.replace("{domain}", relevance.domain)
        .replace("{relevant_topics}", "\n".join(f"- {t}" for t in relevance.relevant_topics))
        .replace("{zero_score_rules}", "\n".join(f"- {r}" for r in relevance.zero_score_rules))
        .replace("{score_scale}", "\n".join(relevance.score_scale))
    )


_LLM_PROMPT = _render_prompt_template()

# ── Fast pre-filter patterns ──────────────────────────────────────────────────

_REJECT_PATTERNS: tuple[str, ...] = (
    "javascript is required",
    "please enable javascript",
    "access denied",
    "403 forbidden",
    "404 not found",
    "bot challenge",
    "are you a human",
    "captcha",
    "enable cookies",
    "bots use duckduckgo",
    "please complete the following challenge",
    "unfortunately, bots",
    "verify you are human",
)


# ── Stage 1 — Fast pre-filter ─────────────────────────────────────────────────

def score_document(text: str, title: str) -> float:  # noqa: ARG001
    """
    Fast pre-filter gate. Returns 0.0 for hard rejects (too short, error pages).
    Returns 0.5 as a neutral sentinel for anything that needs LLM judgment.

    This score is NEVER used as a final quality signal — it is only used to
    decide whether the LLM should be called at all.
    """
    if len(text) < MIN_DOCUMENT_LENGTH:
        return 0.0
    if any(pattern in text.lower() for pattern in _REJECT_PATTERNS):
        return 0.0
    return 0.5  # sentinel: "pass to LLM"


# ── Stage 2 — LLM scoring ─────────────────────────────────────────────────────

async def _llm_score(text: str, title: str, client: LLMClient) -> float | None:
    """
    Ask Gemma4 to rate document relevance on a 0.0-1.0 scale.

    Returns a float score on success, or None if the call fails or the response
    cannot be parsed. A None result means the document is left unscored so it
    can be retried next run — documents are NEVER silently verified by fallback.
    """
    prompt = _LLM_PROMPT.format(
        title=title[:200] or "(no title)",
        text=text[:800],
    )
    try:
        # think=False: Gemma4's thinking phase uses ~400 tokens before outputting
        # the score. Disabling it makes scoring ~10x faster and fixes empty responses.
        raw = await client.chat(
            model=settings.ollama_model_npc,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=_LLM_TEMPERATURE,
            num_predict=_LLM_NUM_PREDICT,
            think=False,
        )
        log.debug("verificator.llm_raw_response", raw=raw.strip()[:80])

        # Match floats (0.8, 0.85) and plain integers (0, 1) in the response.
        # \b ensures we don't match "1" from "10" or "0" from "0.8".
        match = re.search(r"\b([01](?:\.\d+)?)\b", raw.strip())
        if match:
            score = float(match.group())
            return round(min(1.0, max(0.0, score)), 4)

        log.warning(
            "verificator.llm_unparseable",
            raw=raw.strip()[:120],
            doc_title=title[:60],
        )
        return None

    except LLMError as exc:
        log.warning("verificator.llm_error", reason=str(exc)[:120])
        return None
    except Exception as exc:
        log.warning("verificator.llm_unexpected_error", reason=str(exc)[:120])
        return None


# ── Ollama pre-flight ──────────────────────────────────────────────────────────

async def _check_ollama(client: LLMClient) -> bool:
    """Return True if Ollama is reachable and the required model is available."""
    try:
        models = await client.list_models()
        names  = [m.get("name", "") for m in models]
        if not any(settings.ollama_model_npc in n for n in names):
            log.warning(
                "verificator.model_not_found",
                model=settings.ollama_model_npc,
                available=names,
            )
        return True
    except Exception as exc:
        log.error("verificator.ollama_unreachable", reason=str(exc)[:120])
        return False


# ── Null async context (for use_llm=False path) ───────────────────────────────

@asynccontextmanager
async def _no_client() -> AsyncIterator[None]:
    yield None


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    processed: int = 0   # hard-rejects + LLM-scored docs
    verified:  int = 0
    rejected:  int = 0
    skipped:   int = 0   # docs left unscored (no LLM, or LLM failed)


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def _migrate_documents_schema(conn: sqlite3.Connection) -> None:
    """Add quality_score, verified_at, is_complete and run_id columns if not present."""
    for column_def in (
        "quality_score REAL",
        "verified_at TEXT",
        "is_complete INTEGER DEFAULT 1",
        "run_id TEXT",
    ):
        try:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {column_def}")
        except sqlite3.OperationalError:
            pass  # column already present
    conn.commit()


def _fetch_unscored(conn: sqlite3.Connection, batch_size: int) -> list[sqlite3.Row]:
    """Return documents that have not yet been scored (quality_score IS NULL)."""
    return conn.execute(
        "SELECT doc_id, title, text FROM documents WHERE quality_score IS NULL LIMIT ?",
        (batch_size,),
    ).fetchall()


def _save_score(
    conn: sqlite3.Connection,
    doc_id: str,
    quality_score: float,
    verified: bool,
    is_complete: bool = True,
) -> None:
    # run_id is stamped only when the document is verified — it records the pipeline
    # run during which the doc passed (powers the Admin Results page). NULL otherwise.
    verified_at = datetime.now(UTC).isoformat() if verified else None
    run_id = current_run_id() if verified else None
    conn.execute(
        "UPDATE documents SET quality_score=?, verified_at=?, is_complete=?, run_id=? WHERE doc_id=?",
        (quality_score, verified_at, int(is_complete), run_id, doc_id),
    )


def _clean_rejected(conn: sqlite3.Connection) -> int:
    """Remove documents that were scored but failed verification.

    Only touches rows where quality_score is set but verified_at is NULL.
    Leaves unscored documents (quality_score IS NULL) untouched.
    Returns the number of rows deleted.
    """
    cursor = conn.execute(
        "DELETE FROM documents WHERE quality_score IS NOT NULL AND verified_at IS NULL"
    )
    conn.commit()
    return cursor.rowcount


# ── Verificator ────────────────────────────────────────────────────────────────

class Verificator:
    """
    Ag-2 Verificator — scores and gates documents produced by the Archivero.

    With use_llm=True (default): pre-filters junk, then asks Gemma4 to judge
    every substantive document. Only documents with an explicit LLM score
    >= min_score are marked verified. If Ollama is down the run is aborted
    with an error — documents are never silently verified by fallback.

    With use_llm=False: only hard-rejects run (error pages, too short).
    Everything else stays NULL — the LLM is required for real verification.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        min_score: float = _DEFAULT_MIN_SCORE,
        use_llm: bool = True,
    ) -> None:
        self.db_path   = db_path or settings.archive_db
        self.min_score = min_score
        self.use_llm   = use_llm

    async def run_cycle(self, batch_size: int = 50) -> VerificationResult:
        """
        Score up to batch_size unverified documents.

        Hard rejects (error pages, too short) are always written to the DB.
        LLM scores are written only when the model returns a parseable response.
        Documents that can't be scored (no LLM, LLM error) stay NULL and are
        retried on the next call.
        """
        if not self.db_path.exists():
            log.info("verificator.db_not_found", path=str(self.db_path))
            return VerificationResult()

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        result = VerificationResult()

        try:
            _migrate_documents_schema(conn)
            rows = _fetch_unscored(conn, batch_size)

            if not rows:
                log.info("verificator.nothing_to_score")
                return result

            log.info("verificator.cycle_start", batch=len(rows), use_llm=self.use_llm)

            ctx = LLMClient() if self.use_llm else _no_client()
            async with ctx as client:

                # Pre-flight: confirm Ollama is reachable before processing
                if client is not None and not await _check_ollama(client):
                    log.error(
                        "verificator.aborting",
                        reason="Ollama unreachable — run `ollama serve` or use --no-llm",
                    )
                    print(
                        "\n  ERROR: Ollama is not running.\n"
                        "  Start it with: ollama serve\n"
                        "  Or skip LLM scoring: uv run verificator run --no-llm\n"
                    )
                    return result

                for row in rows:
                    text  = row["text"]
                    title = row["title"]

                    # Stage 1: fast pre-filter — hard-rejects need no LLM
                    if score_document(text, title) == 0.0:
                        _save_score(conn, row["doc_id"], 0.0, verified=False)
                        result.rejected += 1
                        result.processed += 1
                        log.debug("verificator.hard_rejected", doc_id=row["doc_id"])
                        continue

                    # Stage 2: LLM judgment for all substantive content
                    if client is None:
                        # No LLM mode: leave unscored, report as skipped
                        result.skipped += 1
                        log.debug("verificator.skipped_no_llm", doc_id=row["doc_id"])
                        continue

                    q_score = await _llm_score(text, title, client)

                    if q_score is None:
                        # LLM failed for this doc — leave unscored, retry next run
                        result.skipped += 1
                        log.warning(
                            "verificator.doc_skipped_llm_failed",
                            doc_id=row["doc_id"],
                            title=title[:60],
                        )
                        continue

                    verified    = q_score >= self.min_score
                    is_complete = len(text) >= MIN_COMPLETE_LENGTH
                    _save_score(conn, row["doc_id"], q_score, verified=verified, is_complete=is_complete)
                    result.processed += 1

                    if verified and not is_complete:
                        result.verified += 1
                        log.info(
                            "verificator.doc_verified_incomplete",
                            doc_id=row["doc_id"],
                            score=q_score,
                            length=len(text),
                            title=title[:60],
                        )
                    elif verified:
                        result.verified += 1
                        log.info(
                            "verificator.doc_verified",
                            doc_id=row["doc_id"],
                            score=q_score,
                            title=title[:60],
                        )
                    else:
                        result.rejected += 1
                        log.info(
                            "verificator.doc_rejected",
                            doc_id=row["doc_id"],
                            score=q_score,
                            title=title[:60],
                        )

            conn.commit()

        finally:
            conn.close()

        log.info(
            "verificator.cycle_done",
            processed=result.processed,
            verified=result.verified,
            rejected=result.rejected,
            skipped=result.skipped,
        )
        return result

    def status(self) -> dict[str, int]:
        """
        Return counts of documents by verification state.

        Keys:
          "verified" — verified_at IS NOT NULL
          "pending"  — quality_score IS NULL (not yet scored by LLM)

        Returns empty dict if the database doesn't exist.
        """
        if not self.db_path.exists():
            return {}
        conn = sqlite3.connect(str(self.db_path))
        try:
            _migrate_documents_schema(conn)
            verified = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE verified_at IS NOT NULL"
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE quality_score IS NULL"
            ).fetchone()[0]
            return {"verified": verified, "pending": pending}
        except Exception:
            return {}
        finally:
            conn.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import traceback

    from lib.logging_setup import configure_logging, current_log_file

    configure_logging()

    ap = argparse.ArgumentParser(
        prog="verificator",
        description="Ag-2 Verificator — score and gate documents in archivo.sqlite",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Score a batch of unverified documents")
    run_p.add_argument(
        "--min-score", type=float, default=_DEFAULT_MIN_SCORE,
        help=f"Quality threshold for acceptance (default: {_DEFAULT_MIN_SCORE})",
    )
    run_p.add_argument(
        "--batch", type=int, default=50,
        help="Max documents to score per run (default: 50)",
    )
    run_p.add_argument(
        "--no-llm", action="store_true",
        help="Skip Gemma4 — only hard-rejects run, other docs stay pending",
    )

    sub.add_parser("status", help="Print verification counts")

    clean_p = sub.add_parser("clean", help="Remove rejected documents from the database")
    clean_p.add_argument(
        "--dry-run", action="store_true",
        help="Count rejected docs without deleting",
    )

    args = ap.parse_args()

    if args.cmd == "run":
        use_llm = not args.no_llm
        log.info("verificator.main_start", min_score=args.min_score, batch=args.batch, use_llm=use_llm)
        try:
            verificator = Verificator(min_score=args.min_score, use_llm=use_llm)
            t0 = datetime.now(UTC)
            result      = asyncio.run(verificator.run_cycle(batch_size=args.batch))
            rid = current_run_id()
            if rid:
                record_stage(rid, "verificator", t0, datetime.now(UTC), {
                    "processed": result.processed,
                    "verified": result.verified,
                    "rejected": result.rejected,
                    "skipped": result.skipped,
                })
            log.info(
                "verificator.main_done",
                processed=result.processed,
                verified=result.verified,
                rejected=result.rejected,
                skipped=result.skipped,
            )
        except Exception as exc:
            log.error("verificator.main_crash", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            log_file = current_log_file()
            if log_file:
                print(f"\n  Log: {log_file}")

        print("\nVerificación completada:")
        print(f"  Procesados (LLM)  : {result.processed}")
        print(f"  Aprobados         : {result.verified}")
        print(f"  Rechazados        : {result.rejected}")
        if result.skipped:
            print(f"  Pendientes (LLM)  : {result.skipped}  ← re-run to score these")

    elif args.cmd == "status":
        verificator = Verificator()
        counts = verificator.status()
        if not counts:
            print("Sin documentos en la base de datos.")
        else:
            print(f"  Verificados : {counts.get('verified', 0)}")
            print(f"  Pendientes  : {counts.get('pending', 0)}")

    elif args.cmd == "clean":
        conn = sqlite3.connect(str(settings.archive_db))
        try:
            rejected_count = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE quality_score IS NOT NULL AND verified_at IS NULL"
            ).fetchone()[0]
            if args.dry_run:
                print(f"Would delete {rejected_count} rejected documents (--dry-run)")
            else:
                deleted = _clean_rejected(conn)
                log.info("verificator.clean_done", deleted=deleted)
                print(f"Deleted {deleted} rejected documents from the database.")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
