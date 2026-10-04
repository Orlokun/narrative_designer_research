"""
Tests for pipeline.verificator — Ag-2 document quality scoring.

Key invariants verified:
  - Documents are NEVER silently verified by fallback (0.5 bug).
  - A failed/unparseable LLM response leaves quality_score NULL (skipped).
  - --no-llm only removes hard junk; everything else stays NULL.
  - The LLM regex parses both "0.8" (decimal) and "0" / "1" (plain integers).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.verificator import (
    MIN_DOCUMENT_LENGTH,
    VerificationResult,
    Verificator,
    _migrate_documents_schema,
    _save_score,
    score_document,
)

# ── score_document (fast pre-filter) ─────────────────────────────────────────


class TestScoreDocument:
    """score_document is a gate only — it rejects obvious junk, nothing more."""

    def test_empty_text_scores_zero(self):
        assert score_document("", "") == 0.0

    def test_text_below_min_length_scores_zero(self):
        assert score_document("a" * (MIN_DOCUMENT_LENGTH - 1), "") == 0.0

    def test_text_at_min_length_returns_sentinel(self):
        # 0.5 is the "pass to LLM" sentinel — not a verified score
        assert score_document("a" * MIN_DOCUMENT_LENGTH, "") == 0.5

    def test_rejects_javascript_required_page(self):
        assert score_document("Please enable JavaScript to view this page. " * 5, "") == 0.0

    def test_rejects_403_forbidden_page(self):
        assert score_document("403 Forbidden — access denied to this resource. " * 5, "") == 0.0

    def test_rejects_bot_challenge_page(self):
        assert score_document("Bot challenge detected. Verify you are human. " * 5, "") == 0.0

    def test_english_text_passes_prefilter(self):
        # English content must reach the LLM — pre-filter is language-agnostic
        text = "The Nixon administration authorized covert operations against Chile. " * 5
        assert score_document(text, "") == 0.5

    def test_unrelated_content_passes_prefilter(self):
        # Pre-filter is NOT a relevance judge — irrelevant content passes to LLM
        text = "Isotopic characteristics of the extreme precipitation event of 2015. " * 5
        assert score_document(text, "") == 0.5


# ── VerificationResult dataclass ─────────────────────────────────────────────


class TestVerificationResult:
    def test_fields_exist(self):
        r = VerificationResult(processed=10, verified=7, rejected=2, skipped=1)
        assert r.processed == 10
        assert r.verified == 7
        assert r.rejected == 2
        assert r.skipped == 1

    def test_default_zero(self):
        r = VerificationResult()
        assert r.processed == 0 == r.verified == r.rejected == r.skipped


# ── Test fixtures ─────────────────────────────────────────────────────────────


def _seed_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL,
            source_kind TEXT NOT NULL DEFAULT 'academic',
            source_id   TEXT NOT NULL DEFAULT 'archivero/openalex',
            provenance  TEXT NOT NULL DEFAULT '{}',
            sha256 TEXT NOT NULL, ingested_at TEXT NOT NULL,
            is_complete INTEGER DEFAULT 1
        )
    """)
    now = datetime.now(UTC).isoformat()
    rows = [
        # d001: relevant Chilean history → LLM should rate high
        ("d001", "Chile Allende 1972", "La CORFO y la Unidad Popular en Chile. " * 8, "sha001"),
        # d002: climate science 2015 — irrelevant → LLM should rate low
        (
            "d002",
            "Isotopic characteristics 2015 northern Chile",
            "Isotopic characteristics and paleoclimate implications of the extreme "
            "precipitation event of March 2015 in northern Chile. " * 8,
            "sha002",
        ),
        # d003: too short → hard rejected by pre-filter, never reaches LLM
        ("d003", "Short", "Brief.", "sha003"),
        # d004: English diplomatic source → should pass LLM
        (
            "d004",
            "Nixon Chile CIA 1971",
            "The Nixon administration authorized covert action against Allende. " * 8,
            "sha004",
        ),
    ]
    conn.executemany(
        "INSERT INTO documents (doc_id, title, text, sha256, ingested_at) VALUES (?,?,?,?,?)",
        [(r[0], r[1], r[2], r[3], now) for r in rows],
    )
    conn.commit()
    conn.close()


def _mock_client(score: float = 0.8) -> MagicMock:
    """Mock OllamaClient that returns the given score string from chat()."""
    client = MagicMock()
    client.chat = AsyncMock(return_value=str(score))
    client.list_models = AsyncMock(return_value=[{"name": "gemma4:e4b"}])
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


# ── Verificator with LLM ──────────────────────────────────────────────────────


class TestVerificatorWithLLM:
    async def test_llm_called_for_substantive_content(self, tmp_path: Path):
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        mock = _mock_client(0.8)
        with patch("pipeline.verificator.LLMClient", return_value=mock):
            await Verificator(db_path=db_path, min_score=0.3, use_llm=True).run_cycle()
        # d001, d002, d004 pass pre-filter; d003 hard-rejected before LLM
        assert mock.chat.call_count == 3

    async def test_hard_reject_never_reaches_llm(self, tmp_path: Path):
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        mock = _mock_client(0.9)
        with patch("pipeline.verificator.LLMClient", return_value=mock):
            await Verificator(db_path=db_path, use_llm=True).run_cycle()
        assert mock.chat.call_count == 3  # not 4

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT quality_score, verified_at FROM documents WHERE doc_id='d003'"
        ).fetchone()
        conn.close()
        assert row[0] == 0.0
        assert row[1] is None

    async def test_high_score_marks_document_verified(self, tmp_path: Path):
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        mock = _mock_client(0.9)
        with patch("pipeline.verificator.LLMClient", return_value=mock):
            await Verificator(db_path=db_path, min_score=0.3, use_llm=True).run_cycle()
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT quality_score, verified_at FROM documents WHERE doc_id='d001'"
        ).fetchone()
        conn.close()
        assert row[0] == 0.9
        assert row[1] is not None

    async def test_low_score_marks_document_rejected(self, tmp_path: Path):
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        mock = _mock_client(0.1)  # LLM says irrelevant
        with patch("pipeline.verificator.LLMClient", return_value=mock):
            await Verificator(db_path=db_path, min_score=0.3, use_llm=True).run_cycle()
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT quality_score, verified_at FROM documents WHERE doc_id='d002'"
        ).fetchone()
        conn.close()
        assert row[0] == 0.1
        assert row[1] is None  # rejected — not verified

    async def test_llm_failure_leaves_document_unscored(self, tmp_path: Path):
        """When LLM returns an error the document stays NULL — no silent 0.5 fallback."""
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        mock = _mock_client()
        mock.chat = AsyncMock(side_effect=OllamaErrorForTest("timeout"))
        with patch("pipeline.verificator.LLMClient", return_value=mock):
            result = await Verificator(db_path=db_path, use_llm=True).run_cycle()

        assert result.skipped == 3  # d001, d002, d004 skipped (LLM failed)
        conn = sqlite3.connect(str(db_path))
        # None of the substantive docs should have a quality_score
        rows = conn.execute(
            "SELECT quality_score FROM documents WHERE doc_id IN ('d001','d002','d004')"
        ).fetchall()
        conn.close()
        assert all(r[0] is None for r in rows), "LLM failure must leave docs unscored"

    async def test_unparseable_llm_response_leaves_document_unscored(self, tmp_path: Path):
        """When LLM returns text that can't be parsed, doc stays NULL — not 0.5."""
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        mock = _mock_client()
        mock.chat = AsyncMock(return_value="I think this document is quite relevant!")
        with patch("pipeline.verificator.LLMClient", return_value=mock):
            result = await Verificator(db_path=db_path, use_llm=True).run_cycle()

        assert result.skipped == 3
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT quality_score FROM documents WHERE doc_id IN ('d001','d002','d004')"
        ).fetchall()
        conn.close()
        assert all(r[0] is None for r in rows), "Unparseable response must leave docs unscored"

    async def test_regex_parses_plain_integer_zero(self, tmp_path: Path):
        """Gemma4 sometimes returns '0' or '1' without decimals — must parse correctly."""
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        mock = _mock_client()
        mock.chat = AsyncMock(return_value="0")
        with patch("pipeline.verificator.LLMClient", return_value=mock):
            await Verificator(db_path=db_path, min_score=0.3, use_llm=True).run_cycle()
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT quality_score FROM documents WHERE doc_id='d001'").fetchone()
        conn.close()
        assert row[0] == 0.0  # parsed correctly, not 0.5 fallback

    async def test_regex_parses_plain_integer_one(self, tmp_path: Path):
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        mock = _mock_client()
        mock.chat = AsyncMock(return_value="1")
        with patch("pipeline.verificator.LLMClient", return_value=mock):
            await Verificator(db_path=db_path, min_score=0.3, use_llm=True).run_cycle()
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT quality_score FROM documents WHERE doc_id='d001'").fetchone()
        conn.close()
        assert row[0] == 1.0

    async def test_ollama_unreachable_aborts_run(self, tmp_path: Path):
        """When Ollama is unreachable, run_cycle returns early with zero processed."""
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        mock = _mock_client()
        mock.list_models = AsyncMock(side_effect=Exception("connection refused"))
        with patch("pipeline.verificator.LLMClient", return_value=mock):
            result = await Verificator(db_path=db_path, use_llm=True).run_cycle()
        assert result.processed == 0
        assert result.verified == 0


# ── Verificator without LLM ───────────────────────────────────────────────────


class TestVerificatorNoLLM:
    async def test_no_llm_never_calls_ollama(self, tmp_path: Path):
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        with patch("pipeline.verificator.LLMClient") as mock_cls:
            await Verificator(db_path=db_path, use_llm=False).run_cycle()
            mock_cls.assert_not_called()

    async def test_no_llm_only_rejects_hard_junk(self, tmp_path: Path):
        """With --no-llm, only d003 (too short) is hard-rejected. Others stay NULL."""
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        result = await Verificator(db_path=db_path, use_llm=False).run_cycle()

        assert result.rejected == 1  # only d003
        assert result.verified == 0  # nothing verified without LLM
        assert result.skipped == 3  # d001, d002, d004 stay pending

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT doc_id, quality_score FROM documents WHERE doc_id != 'd003'"
        ).fetchall()
        conn.close()
        assert all(r[1] is None for r in rows), "Substantive docs must stay NULL without LLM"

    async def test_no_llm_does_not_falsely_verify_irrelevant_content(self, tmp_path: Path):
        """The climate paper must NOT be verified without LLM judgment."""
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        await Verificator(db_path=db_path, use_llm=False).run_cycle()
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT verified_at FROM documents WHERE doc_id='d002'").fetchone()
        conn.close()
        assert row[0] is None, "Irrelevant doc must not be verified without LLM"

    async def test_no_llm_is_idempotent_for_hard_rejects(self, tmp_path: Path):
        """d003 is scored 0.0 on first run. It must not be re-processed on the second run."""
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        v = Verificator(db_path=db_path, use_llm=False)
        r1 = await v.run_cycle()
        r2 = await v.run_cycle()
        # d003 scored in run 1; d001/d002/d004 stay NULL (skipped) in both runs
        assert r1.rejected == 1
        assert r2.rejected == 0  # d003 already scored, not re-processed
        assert r2.skipped == 3  # d001/d002/d004 still pending LLM

    async def test_status_counts_correctly(self, tmp_path: Path):
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        v = Verificator(db_path=db_path, use_llm=False)
        assert v.status().get("pending", 0) == 4  # all unscored before run
        await v.run_cycle()
        status = v.status()
        assert status["verified"] == 0  # nothing verified without LLM
        assert status["pending"] == 3  # d001, d002, d004 still pending

    async def test_handles_missing_database(self, tmp_path: Path):
        result = await Verificator(db_path=tmp_path / "none.sqlite", use_llm=False).run_cycle()
        assert result.processed == 0

    def test_status_handles_missing_database(self, tmp_path: Path):
        assert Verificator(db_path=tmp_path / "none.sqlite").status() == {}

    async def test_short_doc_verified_but_marked_incomplete(self, tmp_path: Path):
        """Short texts (abstracts/snippets) pass LLM but get is_complete=0."""
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        mock = _mock_client(score=0.8)
        with patch("pipeline.verificator.LLMClient", return_value=mock):
            await Verificator(db_path=db_path, min_score=0.3, use_llm=True).run_cycle()
        conn = sqlite3.connect(str(db_path))
        # d001 text is ~400 chars — under MIN_COMPLETE_LENGTH
        row = conn.execute(
            "SELECT verified_at, is_complete FROM documents WHERE doc_id='d001'"
        ).fetchone()
        conn.close()
        assert row[0] is not None, "short but relevant doc should still be verified"
        assert row[1] == 0, "short doc must be marked incomplete (is_complete=0)"

    async def test_long_doc_marked_complete(self, tmp_path: Path):
        """Documents >= MIN_COMPLETE_LENGTH chars are marked complete."""
        db_path = tmp_path / "archivo.sqlite"
        _seed_db(db_path)
        conn2 = sqlite3.connect(str(db_path))
        now = datetime.now(UTC).isoformat()
        # Text is > 1000 chars → complete
        long_text = "Chile bajo Allende y la CORFO implementaron la UP. " * 25
        conn2.execute(
            "INSERT INTO documents (doc_id, title, text, sha256, ingested_at) VALUES (?,?,?,?,?)",
            ("d_long", "Fuente larga", long_text, "sha_long", now),
        )
        conn2.commit()
        conn2.close()

        mock = _mock_client(score=0.85)
        with patch("pipeline.verificator.LLMClient", return_value=mock):
            await Verificator(db_path=db_path, min_score=0.3, use_llm=True).run_cycle()
        conn3 = sqlite3.connect(str(db_path))
        row = conn3.execute("SELECT is_complete FROM documents WHERE doc_id='d_long'").fetchone()
        conn3.close()
        assert row[0] == 1, "long doc should be marked complete"


# ── Helper: fake LLMError for tests ───────────────────────────────────────


class OllamaErrorForTest(Exception):
    pass


# ── run_id stamping (Admin Results foundation) ───────────────────────────────


class TestSaveScoreRunId:
    """_save_score stamps the current run_id only when a document is verified."""

    @staticmethod
    def _doc_conn() -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, title TEXT, text TEXT)")
        conn.executemany(
            "INSERT INTO documents (doc_id, title, text) VALUES (?, ?, ?)",
            [("ok", "t", "x"), ("bad", "t", "x")],
        )
        _migrate_documents_schema(conn)  # adds quality_score/verified_at/is_complete/run_id
        return conn

    def test_verified_doc_gets_run_id(self, monkeypatch):
        monkeypatch.setenv("CYBERSYN_RUN_ID", "run-xyz")
        conn = self._doc_conn()
        _save_score(conn, "ok", 0.9, verified=True)
        run_id = conn.execute("SELECT run_id FROM documents WHERE doc_id='ok'").fetchone()[0]
        assert run_id == "run-xyz"

    def test_rejected_doc_has_null_run_id(self, monkeypatch):
        monkeypatch.setenv("CYBERSYN_RUN_ID", "run-xyz")
        conn = self._doc_conn()
        _save_score(conn, "bad", 0.1, verified=False)
        run_id = conn.execute("SELECT run_id FROM documents WHERE doc_id='bad'").fetchone()[0]
        assert run_id is None

    def test_verified_doc_null_when_no_run(self, monkeypatch):
        monkeypatch.delenv("CYBERSYN_RUN_ID", raising=False)
        conn = self._doc_conn()
        _save_score(conn, "ok", 0.9, verified=True)
        run_id = conn.execute("SELECT run_id FROM documents WHERE doc_id='ok'").fetchone()[0]
        assert run_id is None


# ── Default threshold (decided: 0.6 — docs below are dropped by the Cleaner) ──


def test_default_min_score_is_06():
    from pipeline.verificator import _DEFAULT_MIN_SCORE

    assert _DEFAULT_MIN_SCORE == 0.6
    assert Verificator(db_path=Path(":memory:")).min_score == 0.6
