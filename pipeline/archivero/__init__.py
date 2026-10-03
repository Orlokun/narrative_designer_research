"""
Ag-1 Archivero — fetches documents from historical sources via the Gatekeeper.

Cycle:
1. Reset any missions stuck in 'running' (crashed process recovery)
2. Pick pending missions from SQLite, ordered by priority (highest first)
3. Mark each as 'running'
4. For each (source, query) pair: construct URL → POST to Gatekeeper → extract text
5. Deduplicate by SHA-256; store new documents in archivo.sqlite
6. Mark mission 'done' if any content was found (new or duplicate), 'failed' otherwise

JSON API sources (structured extractors):
    openalex            — OpenAlex REST API (no key required)
    crossref            — CrossRef REST API (no key required)
    archive.org         — Internet Archive advancedsearch JSON API
    semantic_scholar    — Semantic Scholar graph search (no key required)
    chronicling_america — Library of Congress historic US newspapers (OCR)
    wikipedia_es        — Spanish Wikipedia generator=search + extracts

Multi-step sources (handled in run_cycle):
    wikisource_es  — Wikisource ES: search titles → parse rendered HTML
    frus           — US State Dept FRUS Chile vol. XXI, sequential doc pages
    marxists       — Marxists.org Allende archive: crawl index → fetch work pages
    foia_chile     — State Dept FOIA Chile Declassification Project:
                     JSON search → PDF fetch (binary via Gatekeeper) → pypdf OCR text

HTML sources (trafilatura extraction):
    bn_digital  — Biblioteca Nacional Digital
    web_serp    — DuckDuckGo HTML search

CLI:
    uv run archivero run [--batch 10] [--missions 50]
    uv run archivero status
    uv run archivero export
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import re
import sqlite3
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import trafilatura
from pypdf import PdfReader

from lib.config import settings
from lib.coverage import ensure_coverage_table, upsert_coverage_score
from lib.logging_setup import get_logger
from lib.project import project
from lib.run_tracker import current_run_id, record_stage
from lib.schemas import MissionStatus

log = get_logger("archivero")

# ── Constants ──────────────────────────────────────────────────────────────────

_DOC_ID_LENGTH: int = 16
_STALE_RUNNING_TIMEOUT: timedelta = timedelta(minutes=30)


# ── Source URL templates ───────────────────────────────────────────────────────
# {q} is replaced with urllib.parse.quote_plus(query)

_SOURCE_URLS: dict[str, str] = {
    # ── HTML sources (trafilatura extraction) ──────────────────────────────
    "bn_digital": "https://www.bibliotecanacionaldigital.gob.cl/?s={q}",
    "web_serp":   "https://html.duckduckgo.com/html/?q={q}+{region}+historia",

    # ── JSON API sources (structured extractors) ───────────────────────────
    # archive.org advancedsearch: returns JSON with title + description
    "archive.org": (
        "https://archive.org/advancedsearch.php"
        "?q={q}&fl[]=identifier&fl[]=title&fl[]=description"
        "&rows=5&output=json"
    ),
    # FRUS: State Dept FRUS Chile vol. XXI (1969-76) — full HTML pages, trafilatura-extracted.
    # {q} is a document number (d1 … d366); see _frus_doc_url() for URL construction.
    # The source is handled in run_cycle with sequential doc selection, not a search query.
    "frus": "https://history.state.gov/historicaldocuments/frus1969-76v21/{q}",
    # mailto= puts us in OpenAlex's polite pool (higher rate limits)
    "openalex": (
        "https://api.openalex.org/works"
        "?search={q}&per-page=5"
        "&mailto=oguerrerofarias@gmail.com"
    ),
    "crossref": (
        "https://api.crossref.org/works"
        "?query={q}&rows=5"
    ),
    # WikiSource step-1: search for article titles.
    # Step-2 (parse rendered HTML) is handled in run_cycle via _wikisource_parse_url().
    "wikisource_es": (
        "https://es.wikisource.org/w/api.php"
        "?action=query&list=search&srsearch={q}&format=json&srlimit=5"
    ),
    # Semantic Scholar graph search — JSON, no key (unauthenticated shared pool).
    "semantic_scholar": (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        "?query={q}&limit=5&fields=title,abstract,year"
    ),
    # Library of Congress Chronicling America — historic US newspapers (OCR text).
    # NB: digitised coverage is strongest pre-1963, so 1969-73 hits may be sparse.
    "chronicling_america": (
        "https://chroniclingamerica.loc.gov/search/pages/results/"
        "?andtext={q}&format=json&rows=5"
    ),
    # State Dept FOIA Virtual Reading Room — Chile Declassification Project.
    # IDOL search API; the endpoint 404s unless every parameter is present (empty ok).
    "foia_chile": (
        "https://foia.state.gov/api/Search2/SubmitSimpleQuery"
        "?searchText={q}&collectionMatch=StateChile3&page=1&start=0&limit=5"
        "&beginDate=&endDate=&postedBeginDate=&postedEndDate=&caseNumber="
        "&docFrom=&docTo=&email=&telegram=&misc=&me=&gc=&cc=&md=&pr=&sc="
        "&rp=&tn=&dd=&cd=&mf=&exclude=&sort="
    ),
    # Spanish Wikipedia — one-shot generator=search + plain-text extracts.
    "wikipedia_es": (
        "https://es.wikipedia.org/w/api.php"
        "?action=query&generator=search&gsrsearch={q}&gsrlimit=5"
        "&prop=extracts&explaintext=1&format=json"
    ),
}

_SOURCE_KINDS: dict[str, str] = {
    "bn_digital":          "archive",
    "prensa_epoca":        "press",
    "web_serp":            "other",
    "archive.org":         "archive",
    "frus":                "government",   # US State Dept declassified diplomatic cables
    "openalex":            "academic",
    "crossref":            "academic",
    "wikisource_es":       "archive",
    "semantic_scholar":    "academic",
    "chronicling_america": "press",
    "wikipedia_es":        "reference",
    "marxists":            "archive",      # Allende / Unidad Popular primary texts
    "foia_chile":          "government",   # Chile Declassification Project (State/CIA/DOD/FBI)
}

# Sources whose responses are JSON (not HTML) — routed to structured extractors.
# wikisource_es, frus, marxists and foia_chile are handled with multi-step fetches in run_cycle.
_JSON_SOURCES = frozenset({
    "archive.org", "openalex", "crossref",
    "semantic_scholar", "chronicling_america", "wikipedia_es",
})

# Substrings that indicate an error or bot-challenge page rather than real content
_ERROR_PATTERNS = (
    "la página solicitada no está disponible",
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


# ── JSON extractors ────────────────────────────────────────────────────────────

def _decode_inverted_index(inverted_index: dict) -> str:
    """Reconstruct plain text from OpenAlex abstract_inverted_index format."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, position_list in inverted_index.items():
        for pos in position_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))


def _openalex_extract(body: str) -> list[tuple[str, str]]:
    data = json.loads(body)
    results = []
    for work in data.get("results", []):
        title    = work.get("display_name") or ""
        abstract = _decode_inverted_index(work.get("abstract_inverted_index") or {})
        if title and len(abstract) > 50:
            results.append((title, f"{title}\n\n{abstract}"))
    return results


def _crossref_extract(body: str) -> list[tuple[str, str]]:
    data  = json.loads(body)
    items = data.get("message", {}).get("items", [])
    results = []
    for item in items:
        title_list = item.get("title") or []
        title      = title_list[0] if title_list else ""
        abstract   = item.get("abstract") or ""
        abstract   = re.sub(r"<[^>]+>", " ", abstract).strip()  # strip JATS XML tags
        if title and len(abstract) > 50:
            results.append((title, f"{title}\n\n{abstract}"))
    return results


def _wikisource_search_titles(body: str) -> list[str]:
    """Extract article titles from a WikiSource list=search response (step 1)."""
    data = json.loads(body)
    return [
        p.get("title", "")
        for p in data.get("query", {}).get("search", [])
        if p.get("title")
    ]


def _wikisource_parse_url(title: str, lang: str = "es") -> str:
    """Build the action=parse URL that returns fully rendered HTML for a title (step 2)."""
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    return (
        f"https://{lang}.wikisource.org/w/api.php"
        f"?action=parse&page={encoded}&prop=text&format=json"
    )


def _wikisource_page_extract(body: str) -> tuple[str, str] | None:
    """Extract (title, text) from a WikiSource action=parse response.

    The rendered HTML includes DjVu-transcluded text (full speeches, decrees, etc.)
    that is absent from the raw wikitext. Returns None if the page has no content.
    """
    data  = json.loads(body)
    parse = data.get("parse", {})
    title = parse.get("title", "")
    html  = parse.get("text", {}).get("*", "")
    if not html:
        return None
    text = trafilatura.extract(html, include_tables=False)
    if not text or len(text) < 100:
        return None
    return (title, f"{title}\n\n{text}")


def _archiveorg_extract(body: str) -> list[tuple[str, str]]:
    data     = json.loads(body)
    raw_docs = data.get("response", {}).get("docs", [])
    results  = []
    for doc in raw_docs:
        title = doc.get("title") or ""
        desc  = doc.get("description") or ""
        if isinstance(desc, list):
            desc = " ".join(str(d) for d in desc)
        if title and len(desc) > 30:
            results.append((title, f"{title}\n\n{desc}"))
    return results


def _semantic_scholar_extract(body: str) -> list[tuple[str, str]]:
    """Extract (title, text) pairs from a Semantic Scholar graph search response."""
    data = json.loads(body)
    results = []
    for paper in data.get("data", []):
        title    = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        if title and len(abstract) > 50:
            results.append((title, f"{title}\n\n{abstract}"))
    return results


def _chronicling_america_extract(body: str) -> list[tuple[str, str]]:
    """Extract (title, OCR text) pairs from a Chronicling America JSON search response."""
    data = json.loads(body)
    results = []
    for item in data.get("items", []):
        title = item.get("title_normal") or item.get("title") or ""
        ocr   = item.get("ocr_eng") or ""
        if title and len(ocr) > 80:
            results.append((title, f"{title}\n\n{ocr}"))
    return results


def _wikipedia_extract(body: str) -> list[tuple[str, str]]:
    """Extract (title, plain-text extract) pairs from a Wikipedia generator=search response."""
    data  = json.loads(body)
    pages = data.get("query", {}).get("pages", {})
    results = []
    for page in pages.values():
        title   = page.get("title") or ""
        extract = page.get("extract") or ""
        if title and len(extract) > 100:
            results.append((title, f"{title}\n\n{extract}"))
    return results


_JSON_EXTRACTORS: dict[str, callable] = {
    "archive.org":         _archiveorg_extract,
    "openalex":            _openalex_extract,
    "crossref":            _crossref_extract,
    "semantic_scholar":    _semantic_scholar_extract,
    "chronicling_america": _chronicling_america_extract,
    "wikipedia_es":        _wikipedia_extract,
    # frus, wikisource_es and marxists use multi-step fetches handled in run_cycle
}

# FRUS Chile vol. XXI has 366 documents (d1–d366).
# Each mission gets a batch of 5 consecutive docs, offset by mission position
# within the 366-doc range so repeated runs cover new documents.
_FRUS_TOTAL_DOCS: int = 366
_FRUS_BATCH_SIZE: int = 5


def _frus_doc_numbers(offset: int) -> list[int]:
    """Return 5 FRUS document numbers starting at offset (wraps around 366)."""
    start = (offset % _FRUS_TOTAL_DOCS) + 1
    return [((start - 1 + i) % _FRUS_TOTAL_DOCS) + 1 for i in range(_FRUS_BATCH_SIZE)]


# Marxists.org Allende archive: no search API, so crawl the index for work pages.
# The index URL is the only uncertain constant — verified during the live smoke test;
# if wrong, the connector simply yields nothing (graceful failure).
_MARXISTS_INDEX_URL: str = "https://www.marxists.org/espanol/allende/"
_MARXISTS_BATCH_SIZE: int = 5


def _marxists_index_links(html: str, base_url: str = _MARXISTS_INDEX_URL) -> list[str]:
    """Extract work-page URLs from the Marxists.org Allende archive index.

    Returns absolute .htm/.html URLs under the archive path, de-duplicated and
    excluding the index itself and non-document links. Pure (no network) for testing.
    """
    hrefs = re.findall(r'href="([^"#]+)"', html, flags=re.IGNORECASE)
    seen: set[str] = set()
    links: list[str] = []
    for href in hrefs:
        if href.lower().startswith(("mailto:", "javascript:")):
            continue
        absolute = urllib.parse.urljoin(base_url, href)
        if "/espanol/allende/" not in absolute:
            continue
        if not absolute.lower().endswith((".htm", ".html")):
            continue
        if absolute.rstrip("/") == base_url.rstrip("/"):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)
    return links


# ── Text extraction dispatcher ─────────────────────────────────────────────────

def _extract(body: str, source: str) -> list[tuple[str, str]]:
    """
    Return a list of (title, text) pairs from a Gatekeeper response body.
    JSON API sources return multiple results; HTML sources return at most one.
    """
    if source in _JSON_SOURCES or _is_wikipedia(source):
        try:
            extractor = _JSON_EXTRACTORS.get(source, _wikipedia_extract)
            return extractor(body)
        except Exception as exc:
            log.debug("archivero.json_extract_failed", source=source, error=str(exc)[:80])
            return []

    # HTML path — trafilatura
    raw = trafilatura.extract(
        body,
        output_format="json",
        with_metadata=True,
        include_tables=False,
    )
    if not raw:
        return []
    data  = json.loads(raw)
    text  = data.get("text") or ""
    title = data.get("title") or ""
    if len(text) < 80:
        return []
    if any(pattern in text.lower() for pattern in _ERROR_PATTERNS):
        log.debug("archivero.error_page_rejected", title=title[:60])
        return []
    return [(title, text)]


# ── FOIA Chile (State Dept Chile Declassification Project) ────────────────────

_FOIA_BASE_URL: str = "https://foia.state.gov"


_FOIA_MAX_DOCS_PER_QUERY: int = 5  # the API ignores its limit param and returns 20


def _foia_chile_results(body: str, limit: int = _FOIA_MAX_DOCS_PER_QUERY) -> list[dict]:
    """Parse the SubmitSimpleQuery JSON into per-document metadata dicts.

    Entries without a pdfLink are skipped (nothing to fetch). pdfLink arrives
    with backslashes ("DOCUMENTS\\StateChile3\\00005662.pdf") and is normalised
    into an absolute https URL. At most `limit` documents are returned per
    query — each one costs a PDF download through the Gatekeeper.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return []
    results = []
    for entry in data.get("Results") or []:
        if len(results) >= limit:
            break
        pdf_link = entry.get("pdfLink")
        if not pdf_link:
            continue
        pdf_path = pdf_link.replace("\\", "/").lstrip("/")
        docdate = (entry.get("docdate") or "")[:10]
        results.append({
            "subject":        entry.get("subject") or "",
            "docdate":        docdate,
            "pdf_url":        f"{_FOIA_BASE_URL}/{pdf_path}",
            "from":           entry.get("from") or "",
            "to":             entry.get("to") or "",
            "classification": entry.get("classification") or "",
            "doctype":        entry.get("doctype") or "",
            "casenumber":     entry.get("casenumber") or "",
        })
    return results


def _pdf_extract_text(data: bytes) -> str:
    """Extract the OCR text layer from a PDF. Returns "" on any failure."""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception:
        return ""


def _foia_document_text(meta: dict, ocr_text: str) -> tuple[str, str]:
    """Compose (title, text) for one FOIA document.

    The header preserves the declassification metadata (sender, recipient,
    classification, case) that the OCR layer usually garbles; the body is the
    extracted text.
    """
    subject = meta.get("subject") or "FOIA Chile document"
    docdate = meta.get("docdate") or ""
    title = f"{subject} ({docdate})" if docdate else subject
    header_fields = [
        ("Subject", subject),
        ("Date", docdate),
        ("From", meta.get("from", "")),
        ("To", meta.get("to", "")),
        ("Classification", meta.get("classification", "")),
        ("Case", meta.get("casenumber", "")),
        ("Source", meta.get("pdf_url", "")),
    ]
    header = "\n".join(f"{k}: {v}" for k, v in header_fields if v)
    return title, f"{header}\n\n{ocr_text}".strip()


# Language-parametric MediaWiki connectors: ``wikipedia_<lang>`` / ``wikisource_<lang>``.
# ``wikipedia_es`` and ``wikisource_es`` keep their explicit entries above; any other
# language code resolves through these templates so a project in German or English
# gets its own Wikipedia / Wikisource without new code.
_WIKIPEDIA_URL_TEMPLATE: str = (
    "https://{lang}.wikipedia.org/w/api.php"
    "?action=query&generator=search&gsrsearch={q}&gsrlimit=5"
    "&prop=extracts&explaintext=1&format=json"
)
_WIKISOURCE_URL_TEMPLATE: str = (
    "https://{lang}.wikisource.org/w/api.php"
    "?action=query&list=search&srsearch={q}&format=json&srlimit=5"
)
_LANG_CONNECTOR_RE = re.compile(r"^(wikipedia|wikisource)_([a-z]{2,3})$")


def connector_language(source: str) -> str | None:
    """Return the language code of a ``wikipedia_xx`` / ``wikisource_xx`` connector, else None."""
    match = _LANG_CONNECTOR_RE.match(source)
    return match.group(2) if match else None


def _is_wikipedia(source: str) -> bool:
    return source.startswith("wikipedia_") and connector_language(source) is not None


def _is_wikisource(source: str) -> bool:
    return source.startswith("wikisource_") and connector_language(source) is not None


def _build_url(source: str, query: str) -> str | None:
    template = _SOURCE_URLS.get(source)
    if not template:
        lang = connector_language(source)
        if lang and source.startswith("wikipedia_"):
            template = _WIKIPEDIA_URL_TEMPLATE.replace("{lang}", lang)
        elif lang and source.startswith("wikisource_"):
            template = _WIKISOURCE_URL_TEMPLATE.replace("{lang}", lang)
        else:
            return None
    return template.format(
        q=urllib.parse.quote_plus(query), region=urllib.parse.quote_plus(project().region)
    )


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def _ensure_documents_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id      TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            text        TEXT NOT NULL,
            lang        TEXT NOT NULL DEFAULT 'es',
            date_iso    TEXT,
            authors     TEXT NOT NULL DEFAULT '[]',
            places      TEXT NOT NULL DEFAULT '[]',
            source_kind TEXT NOT NULL,
            source_id   TEXT NOT NULL,
            provenance  TEXT NOT NULL DEFAULT '{}',
            rights      TEXT,
            sha256      TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_sha256 ON documents(sha256)"
    )
    conn.commit()


def _insert_document(
    conn: sqlite3.Connection,
    *,
    title: str,
    text: str,
    source: str,
    mission_id: str,
    url: str,
    seed_character_id: str | None = None,
    seed_location_id: str | None = None,
) -> bool:
    """Insert document; returns True if new, False if duplicate (sha256 collision).

    ``seed_character_id`` / ``seed_location_id`` record the character or location
    whose research mission harvested this document (NULL for ordinary gap
    missions). They close the research loops: the Cast Manager / Location Manager
    force-attribute such a document to its seed even if the LLM does not
    re-extract the seed's exact name. On a duplicate, an existing NULL seed is
    backfilled — a doc already harvested by a gap mission is still evidence
    about the seed.
    """
    sha    = hashlib.sha256(text.encode()).hexdigest()
    doc_id = sha[:_DOC_ID_LENGTH]
    prov   = json.dumps(
        {
            "url": url,
            "mission_id": mission_id,
            "seed_character_id": seed_character_id,
            "seed_location_id": seed_location_id,
        },
        ensure_ascii=False,
    )
    now    = datetime.now(UTC).isoformat()

    try:
        conn.execute(
            """INSERT INTO documents
               (doc_id, title, text, source_kind, source_id, provenance, sha256,
                ingested_at, seed_character_id, seed_location_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                title or url,
                text,
                _SOURCE_KINDS.get(source, "reference" if _is_wikipedia(source) else "archive" if _is_wikisource(source) else "other"),
                f"archivero/{source}",
                prov,
                sha,
                now,
                seed_character_id,
                seed_location_id,
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        if seed_character_id:
            conn.execute(
                "UPDATE documents SET seed_character_id = ? "
                "WHERE sha256 = ? AND seed_character_id IS NULL",
                (seed_character_id, sha),
            )
        if seed_location_id:
            conn.execute(
                "UPDATE documents SET seed_location_id = ? "
                "WHERE sha256 = ? AND seed_location_id IS NULL",
                (seed_location_id, sha),
            )
        if seed_character_id or seed_location_id:
            conn.commit()
        return False  # duplicate sha256


def _migrate_documents_schema(conn: sqlite3.Connection) -> None:
    """Add verificator columns to documents table if not already present."""
    for column_def in (
        "quality_score REAL",
        "verified_at TEXT",
        "is_complete INTEGER DEFAULT 1",
        "seed_character_id TEXT",  # entity mission that harvested this doc (research loop)
        "seed_location_id TEXT",  # location mission that harvested this doc
    ):
        try:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {column_def}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()


# Coverage lives in the shared 3D (theme × genre × month) table — see lib/coverage.py.
# Re-exported under the historical names for existing callers and tests.
_ensure_coverage_table = ensure_coverage_table
_upsert_coverage_score = upsert_coverage_score


def _mission_kind(row: sqlite3.Row) -> str:
    """Return a mission row's kind ('gap' | 'entity' | 'location'), defaulting to 'gap'.

    Tolerates legacy databases that predate the kind column. Entity and location
    missions must never update the coverage matrix — their (category, month) is
    nominal.
    """
    try:
        return row["kind"] or "gap"
    except (IndexError, KeyError):
        return "gap"


def _mission_genre(row: sqlite3.Row) -> int:
    """Return a mission row's genre_id for coverage writes; 0 when absent/legacy."""
    try:
        return int(row["genre_id"]) if row["genre_id"] is not None else 0
    except (IndexError, KeyError):
        return 0


def _mission_seed_character(row: sqlite3.Row) -> str | None:
    """Return the character a mission researches, or None for gap missions/legacy DBs.

    Only entity missions attribute their harvest to a seed character; a gap
    mission's ``character_id`` (if any) is ignored.
    """
    if _mission_kind(row) != "entity":
        return None
    try:
        return row["character_id"] or None
    except (IndexError, KeyError):
        return None


def _mission_seed_location(row: sqlite3.Row) -> str | None:
    """Return the location a mission researches, or None for other kinds/legacy DBs."""
    if _mission_kind(row) != "location":
        return None
    try:
        return row["location_id"] or None
    except (IndexError, KeyError):
        return None


def _set_mission_status(
    conn: sqlite3.Connection,
    mission_id: str,
    status: MissionStatus,
) -> None:
    conn.execute(
        "UPDATE missions SET status = ?, updated_at = ? WHERE mission_id = ?",
        (status.value, datetime.now(UTC).isoformat(), mission_id),
    )
    conn.commit()


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class CycleResult:
    missions_processed:  int = 0
    missions_done:       int = 0
    missions_failed:     int = 0
    documents_new:       int = 0
    documents_duplicate: int = 0
    fetch_errors:        int = 0


# ── Archivero ──────────────────────────────────────────────────────────────────

class Archivero:
    def __init__(
        self,
        db_path: Path | None = None,
        gatekeeper_url: str | None = None,
        batch_size: int = 10,
    ) -> None:
        self.db_path = db_path or settings.archive_db
        self.gatekeeper_url = gatekeeper_url or (
            f"http://{settings.gatekeeper_host}:{settings.gatekeeper_port}"
        )
        self.batch_size = batch_size

    async def _gatekeeper_fetch(
        self,
        client: httpx.AsyncClient,
        url: str,
        mission_id: str,
    ) -> str | None:
        """POST to Gatekeeper /fetch; return body text or None on any failure."""
        try:
            resp = await client.post(
                f"{self.gatekeeper_url}/fetch",
                json={"url": url, "mission_id": mission_id, "timeout_s": 30.0},
                timeout=40.0,
            )
            if resp.status_code != 200:
                log.warning("archivero.gatekeeper_error", url=url, status=resp.status_code)
                return None
            data = resp.json()
            if data.get("binary"):
                return None  # skip PDFs and images
            return data.get("body")
        except Exception as exc:
            log.warning("archivero.fetch_failed", url=url, error=str(exc)[:120])
            return None

    async def _gatekeeper_fetch_binary(
        self,
        client: httpx.AsyncClient,
        url: str,
        mission_id: str,
    ) -> bytes | None:
        """POST to Gatekeeper /fetch expecting binary content (PDFs).

        Returns the decoded bytes, or None on failure or when the response is
        not binary (e.g. an HTML error page where the PDF should be).
        """
        try:
            resp = await client.post(
                f"{self.gatekeeper_url}/fetch",
                json={"url": url, "mission_id": mission_id, "timeout_s": 30.0},
                timeout=60.0,
            )
            if resp.status_code != 200:
                log.warning("archivero.gatekeeper_error", url=url, status=resp.status_code)
                return None
            data = resp.json()
            if not data.get("binary"):
                return None
            return base64.b64decode(data.get("body") or "")
        except Exception as exc:
            log.warning("archivero.fetch_failed", url=url, error=str(exc)[:120])
            return None

    async def run_cycle(self, max_missions: int = 50) -> CycleResult:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        _ensure_documents_table(conn)
        _migrate_documents_schema(conn)
        _ensure_coverage_table(conn)

        try:
            # Reset missions stuck in 'running' for longer than the timeout (crashed process)
            stale_cutoff = (datetime.now(UTC) - _STALE_RUNNING_TIMEOUT).isoformat()
            stale_reset_count = conn.execute(
                "UPDATE missions SET status='pending', updated_at=? "
                "WHERE status='running' AND updated_at < ?",
                (datetime.now(UTC).isoformat(), stale_cutoff),
            ).rowcount
            if stale_reset_count:
                conn.commit()
                log.warning("archivero.stale_missions_reset", count=stale_reset_count)

            rows = conn.execute(
                "SELECT * FROM missions WHERE status = ? ORDER BY priority DESC LIMIT ?",
                (MissionStatus.PENDING.value, min(self.batch_size, max_missions)),
            ).fetchall()

            if not rows:
                log.info("archivero.no_pending_missions")
                return CycleResult()

            result = CycleResult(missions_processed=len(rows))
            log.info("archivero.cycle_start", missions=len(rows))

            async with httpx.AsyncClient(timeout=40.0) as client:
                for _mission_idx, row in enumerate(rows):
                    mission_id = row["mission_id"]
                    category   = row["category"]
                    month_iso  = row["month_iso"]
                    queries    = json.loads(row["search_queries"])
                    sources    = json.loads(row["target_sources"])
                    seed_char  = _mission_seed_character(row)
                    seed_loc   = _mission_seed_location(row)

                    _set_mission_status(conn, mission_id, MissionStatus.RUNNING)
                    log.info(
                        "archivero.mission_start",
                        mission_id=mission_id,
                        category=category,
                        month=month_iso,
                    )

                    docs_new_this_mission   = 0
                    docs_found_this_mission = 0  # new + duplicate; drives mission status
                    extractions_attempted   = 0  # fetches that returned a body (not None)

                    for source in sources:
                        # ── WikiSource: 2-step (search titles → parse rendered HTML) ──
                        if _is_wikisource(source):
                            for query in queries:
                                search_url = _build_url(source, query)
                                if search_url is None:
                                    continue
                                search_body = await self._gatekeeper_fetch(
                                    client, search_url, mission_id
                                )
                                if search_body is None:
                                    result.fetch_errors += 1
                                    continue
                                extractions_attempted += 1
                                titles = _wikisource_search_titles(search_body)
                                for title in titles:
                                    parse_url  = _wikisource_parse_url(title, connector_language(source) or "es")
                                    parse_body = await self._gatekeeper_fetch(
                                        client, parse_url, mission_id
                                    )
                                    if parse_body is None:
                                        continue
                                    page_result = _wikisource_page_extract(parse_body)
                                    if page_result is None:
                                        continue
                                    ptitle, ptext = page_result
                                    is_new = _insert_document(
                                        conn,
                                        title=ptitle,
                                        text=ptext,
                                        source=source,
                                        mission_id=mission_id,
                                        url=parse_url,
                                        seed_character_id=seed_char, seed_location_id=seed_loc,
                                    )
                                    docs_found_this_mission += 1
                                    if is_new:
                                        result.documents_new += 1
                                        docs_new_this_mission += 1
                                        log.info(
                                            "archivero.doc_saved",
                                            title=ptitle[:60],
                                            source=source,
                                        )
                                    else:
                                        result.documents_duplicate += 1
                            continue  # next source

                        # ── FRUS: sequential doc fetch from history.state.gov ──────
                        if source == "frus":
                            # Distribute doc numbers across missions using category_id
                            # and mission index so each run covers different documents.
                            offset = (row["category_id"] * 23 + _mission_idx * 5)
                            doc_nums = _frus_doc_numbers(offset)
                            for doc_num in doc_nums:
                                frus_url  = _SOURCE_URLS["frus"].format(q=f"d{doc_num}")
                                frus_body = await self._gatekeeper_fetch(
                                    client, frus_url, mission_id
                                )
                                if frus_body is None:
                                    result.fetch_errors += 1
                                    continue
                                extractions_attempted += 1
                                text = trafilatura.extract(frus_body, include_tables=False)
                                if not text or len(text) < 80:
                                    continue
                                if any(p in text.lower() for p in _ERROR_PATTERNS):
                                    continue
                                # Title: first non-blank line of the extracted text
                                title = next(
                                    (ln.strip() for ln in text.splitlines() if ln.strip()),
                                    f"FRUS Chile d{doc_num}",
                                )
                                is_new = _insert_document(
                                    conn,
                                    title=title[:200],
                                    text=text,
                                    source="frus",
                                    mission_id=mission_id,
                                    url=frus_url,
                                    seed_character_id=seed_char, seed_location_id=seed_loc,
                                )
                                docs_found_this_mission += 1
                                if is_new:
                                    result.documents_new += 1
                                    docs_new_this_mission += 1
                                    log.info(
                                        "archivero.doc_saved",
                                        title=title[:60],
                                        source="frus",
                                    )
                                else:
                                    result.documents_duplicate += 1
                            continue  # next source

                        # ── Marxists.org: crawl the Allende archive index, then pages ──
                        # ── FOIA Chile: JSON search → PDF fetch → OCR text ─────────
                        if source == "foia_chile":
                            for query in queries:
                                search_url = _build_url("foia_chile", query)
                                search_body = await self._gatekeeper_fetch(
                                    client, search_url, mission_id
                                )
                                if search_body is None:
                                    result.fetch_errors += 1
                                    continue
                                extractions_attempted += 1
                                for meta in _foia_chile_results(search_body):
                                    pdf_bytes = await self._gatekeeper_fetch_binary(
                                        client, meta["pdf_url"], mission_id
                                    )
                                    if pdf_bytes is None:
                                        result.fetch_errors += 1
                                        continue
                                    ocr_text = _pdf_extract_text(pdf_bytes)
                                    if len(ocr_text) < 80:
                                        continue  # no usable OCR layer
                                    title, text = _foia_document_text(meta, ocr_text)
                                    is_new = _insert_document(
                                        conn,
                                        title=title[:200],
                                        text=text,
                                        source="foia_chile",
                                        mission_id=mission_id,
                                        url=meta["pdf_url"],
                                        seed_character_id=seed_char, seed_location_id=seed_loc,
                                    )
                                    docs_found_this_mission += 1
                                    if is_new:
                                        result.documents_new += 1
                                        docs_new_this_mission += 1
                                        log.info(
                                            "archivero.doc_saved",
                                            title=title[:60],
                                            source="foia_chile",
                                        )
                                    else:
                                        result.documents_duplicate += 1
                            continue  # next source

                        if source == "marxists":
                            index_body = await self._gatekeeper_fetch(
                                client, _MARXISTS_INDEX_URL, mission_id
                            )
                            if index_body is None:
                                result.fetch_errors += 1
                                continue
                            extractions_attempted += 1
                            links = _marxists_index_links(index_body)
                            if not links:
                                continue
                            # Distribute pages across missions so repeated runs cover new ones
                            offset = (row["category_id"] * 23 + _mission_idx * 5) % len(links)
                            batch = list(dict.fromkeys(
                                links[(offset + i) % len(links)]
                                for i in range(_MARXISTS_BATCH_SIZE)
                            ))
                            for page_url in batch:
                                page_body = await self._gatekeeper_fetch(
                                    client, page_url, mission_id
                                )
                                if page_body is None:
                                    continue
                                text = trafilatura.extract(page_body, include_tables=False)
                                if not text or len(text) < 80:
                                    continue
                                if any(p in text.lower() for p in _ERROR_PATTERNS):
                                    continue
                                title = next(
                                    (ln.strip() for ln in text.splitlines() if ln.strip()),
                                    "Marxists.org — Allende",
                                )
                                is_new = _insert_document(
                                    conn,
                                    title=title[:200],
                                    text=text,
                                    source="marxists",
                                    mission_id=mission_id,
                                    url=page_url,
                                    seed_character_id=seed_char, seed_location_id=seed_loc,
                                )
                                docs_found_this_mission += 1
                                if is_new:
                                    result.documents_new += 1
                                    docs_new_this_mission += 1
                                    log.info(
                                        "archivero.doc_saved",
                                        title=title[:60],
                                        source="marxists",
                                    )
                                else:
                                    result.documents_duplicate += 1
                            continue  # next source

                        # ── Generic path: JSON APIs and HTML sources ────────────────
                        for query in queries:
                            url = _build_url(source, query)
                            if url is None:
                                continue

                            body = await self._gatekeeper_fetch(client, url, mission_id)
                            if body is None:
                                result.fetch_errors += 1
                                continue

                            extractions_attempted += 1
                            extracted = _extract(body, source)
                            if not extracted:
                                log.debug("archivero.no_text", url=url, source=source)
                                continue

                            for title, text in extracted:
                                is_new = _insert_document(
                                    conn,
                                    title=title,
                                    text=text,
                                    source=source,
                                    mission_id=mission_id,
                                    url=url,
                                    seed_character_id=seed_char, seed_location_id=seed_loc,
                                )
                                docs_found_this_mission += 1
                                if is_new:
                                    result.documents_new += 1
                                    docs_new_this_mission += 1
                                    log.info(
                                        "archivero.doc_saved",
                                        title=title[:60],
                                        source=source,
                                    )
                                else:
                                    result.documents_duplicate += 1

                    # Three outcomes:
                    # DONE    — at least one document found (new or duplicate)
                    # PENDING — every fetch failed at network level; retry next cycle
                    # FAILED  — sources were reachable but genuinely had no content
                    if docs_found_this_mission > 0:
                        final = MissionStatus.DONE
                    elif extractions_attempted == 0:
                        final = MissionStatus.PENDING  # all network errors → retry
                    else:
                        final = MissionStatus.FAILED   # reached sources, no content

                    _set_mission_status(conn, mission_id, final)

                    if final == MissionStatus.DONE:
                        result.missions_done += 1
                        # Entity/location missions carry only a nominal cell —
                        # bumping it would fake coverage the matrix never earned.
                        if _mission_kind(row) == "gap":
                            _upsert_coverage_score(
                                conn,
                                category_id=row["category_id"],
                                genre_id=_mission_genre(row),
                                month_iso=month_iso,
                                docs_found=docs_found_this_mission,
                            )
                            conn.commit()
                        log.info(
                            "archivero.mission_done",
                            mission_id=mission_id,
                            docs_new=docs_new_this_mission,
                            docs_found=docs_found_this_mission,
                        )
                    elif final == MissionStatus.PENDING:
                        log.warning(
                            "archivero.mission_network_failure_reset",
                            mission_id=mission_id,
                            category=category,
                            month=month_iso,
                        )
                    else:
                        result.missions_failed += 1
                        log.warning(
                            "archivero.mission_failed",
                            mission_id=mission_id,
                            category=category,
                            month=month_iso,
                        )

        finally:
            conn.close()

        log.info(
            "archivero.cycle_done",
            processed=result.missions_processed,
            done=result.missions_done,
            failed=result.missions_failed,
            new_docs=result.documents_new,
            duplicates=result.documents_duplicate,
            errors=result.fetch_errors,
        )
        return result

    def mission_summary(self) -> dict[str, int]:
        if not self.db_path.exists():
            return {}
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM missions GROUP BY status"
            ).fetchall()
            return {r[0]: r[1] for r in rows}
        except Exception:
            return {}
        finally:
            conn.close()

    def documents_count(self) -> int:
        if not self.db_path.exists():
            return 0
        conn = sqlite3.connect(str(self.db_path))
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE source_id LIKE 'archivero/%'"
            ).fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
        finally:
            conn.close()

    def export_documents(self, out_path: Path | None = None) -> Path:
        """Export verified documents as a human-readable JSON organised by category × month.

        Only documents stamped by the Verificator (verified_at IS NOT NULL) are
        included. Run `uv run verificator run` before exporting to score new docs.
        """
        out = out_path or (self.db_path.parent.parent / "documents.json")
        if not self.db_path.exists():
            out.write_text("[]", encoding="utf-8")
            return out

        # Canonical 16 categories — id → name (matches admin/heatmap)
        category_names: dict[int, str] = {
            1: "Política Nacional", 2: "Salud", 3: "Economía Nacional",
            4: "Industria Nacional", 5: "Industria Privada", 6: "Estados Unidos",
            7: "Unión Soviética", 8: "Política Internacional", 9: "Educación",
            10: "Ciencia y Tecnología", 11: "Transporte",
            12: "Urbanismo y Megaproyectos", 13: "Política Militar",
            14: "Artes y Cultura", 15: "Deporte", 16: "Macroeconomía",
        }

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("""
                SELECT d.doc_id, d.title, d.text, d.source_id, d.source_kind,
                       d.ingested_at, d.provenance, d.quality_score,
                       d.mapped_category_id, d.mapped_categories, d.mapped_month_iso,
                       d.doc_tag,
                       m.category AS mission_category,
                       m.category_id AS mission_category_id,
                       m.month_iso AS mission_month, m.priority
                FROM documents d
                LEFT JOIN missions m
                       ON m.mission_id = json_extract(d.provenance, '$.mission_id')
                WHERE d.verified_at IS NOT NULL
                ORDER BY COALESCE(d.mapped_category_id, m.category_id) ASC,
                         COALESCE(d.mapped_month_iso, m.month_iso)     ASC,
                         d.quality_score DESC
            """).fetchall()
        finally:
            conn.close()

        # Group by primary category → month, but include the full ranked list per doc
        matrix: dict = {}
        for row in rows:
            # Decode mapped_categories JSON → list[int]
            ids: list[int] = []
            if row["mapped_categories"]:
                try:
                    decoded = json.loads(row["mapped_categories"])
                    if isinstance(decoded, list):
                        ids = [int(x) for x in decoded if isinstance(x, int)]
                except (json.JSONDecodeError, TypeError, ValueError):
                    ids = []
            if not ids and row["mapped_category_id"] is not None:
                ids = [int(row["mapped_category_id"])]

            categories = [category_names[i] for i in ids if i in category_names]
            primary_category = (
                categories[0]
                if categories
                else row["mission_category"] or "Desconocido"
            )
            month = row["mapped_month_iso"] or row["mission_month"] or "?"

            matrix.setdefault(primary_category, {}).setdefault(month, []).append({
                "doc_id":        row["doc_id"],
                "title":         row["title"],
                "quality_score": row["quality_score"],
                "categories":    categories or [primary_category],
                "month_iso":     month,
                "doc_tag":       row["doc_tag"],
                "source":        row["source_id"],
                "ingested_at":   row["ingested_at"],
                "text":          row["text"],
            })

        export = {
            "exported_at":     datetime.now(UTC).isoformat(),
            "total_documents": len(rows),
            "matrix":          matrix,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(export, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import traceback

    from lib.logging_setup import configure_logging, current_log_file

    configure_logging()

    ap = argparse.ArgumentParser(
        prog="archivero",
        description="Ag-1 Archivero — fetch documents for pending missions via Gatekeeper",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Execute one fetch cycle")
    run_p.add_argument(
        "--batch", type=int, default=10,
        help="Max missions per cycle (default: 10)",
    )
    run_p.add_argument(
        "--missions", type=int, default=50,
        help="Total missions limit for this run (default: 50)",
    )

    sub.add_parser("status", help="Print mission queue summary")
    sub.add_parser(
        "export",
        help="Export all documents to data/documents.json (readable, ordered by category × month)",
    )

    args = ap.parse_args()

    if args.cmd == "run":
        log.info("archivero.main_start", cmd="run", batch=args.batch, missions=args.missions)
        try:
            archivero = Archivero(batch_size=args.batch)
            t0 = datetime.now(UTC)
            result    = asyncio.run(archivero.run_cycle(max_missions=args.missions))
            rid = current_run_id()
            if rid:
                record_stage(rid, "archivero", t0, datetime.now(UTC), {
                    "missions_done": result.missions_done,
                    "documents_new": result.documents_new,
                    "documents_duplicate": result.documents_duplicate,
                    "fetch_errors": result.fetch_errors,
                })
            log.info(
                "archivero.main_done",
                processed=result.missions_processed,
                done=result.missions_done,
                failed=result.missions_failed,
                new_docs=result.documents_new,
                duplicates=result.documents_duplicate,
                fetch_errors=result.fetch_errors,
            )
        except Exception as exc:
            log.error("archivero.main_crash", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            log_file = current_log_file()
            if log_file:
                print(f"\n  Log: {log_file}")

        print("\nCiclo completado:")
        print(f"  Misiones procesadas : {result.missions_processed}")
        print(f"  Misiones exitosas   : {result.missions_done}")
        print(f"  Misiones fallidas   : {result.missions_failed}")
        print(f"  Documentos nuevos   : {result.documents_new}")
        print(f"  Duplicados omitidos : {result.documents_duplicate}")
        print(f"  Errores de fetch    : {result.fetch_errors}")

    elif args.cmd == "status":
        archivero = Archivero()
        summary   = archivero.mission_summary()
        docs      = archivero.documents_count()
        if not summary:
            print("Sin misiones en la base de datos.")
        else:
            total = sum(summary.values())
            for status, count in sorted(summary.items()):
                print(f"  {status:<12} {count:>4}  ({count / total * 100:.0f}%)")
            print(f"  {'TOTAL':<12} {total:>4}")
        print(f"\n  Documentos en archivo (Archivero): {docs}")

    elif args.cmd == "export":
        archivero = Archivero()
        out       = archivero.export_documents()
        docs      = archivero.documents_count()
        print(f"Exportado: {out}  ({docs} documentos)")


if __name__ == "__main__":
    main()
