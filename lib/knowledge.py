"""
Knowledge API — how the game Director consults the research archive.

The pipeline fills one SQLite database with documents (verified, mapped to
theme × genre × month), characters with timelines and relations, and places
with facts. At play time the Director needs the opposite movement: *given what
the player is doing right now* — a month in the simulation, the themes in play,
the people in the room, the place — return the period evidence an LLM director
can react with.

``KnowledgeBase.context(...)`` answers that question with a ``DirectorContext``:
a compact, serialisable bundle of documents, character dossiers, relations and
locations, ranked by relevance to the query. It is read-only, project-agnostic
(the theme ids come from the active spec) and tolerant of a partially built
archive (missing tables yield empty sections).

Usage:
    kb = KnowledgeBase()                      # settings.archive_db
    ctx = kb.context(DirectorQuery(month_iso="1972-10", theme_ids=[11, 4],
                                   character_names=["Allende"], limit=8))
    prompt_block = ctx.as_prompt()            # ready to paste into the Director prompt
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from lib.config import settings
from lib.project import project

_SNIPPET_CHARS: int = 600
_MAX_FACTS: int = 8
_MAX_RELATIONS: int = 8
_MIN_QUALITY: float = 0.6


class DirectorQuery(BaseModel):
    """What the Director knows about the current game situation."""

    model_config = ConfigDict(extra="forbid")

    month_iso: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}$")
    month_window: int = Field(
        default=1, ge=0, le=12, description="± months around month_iso to include."
    )
    theme_ids: list[int] = Field(default_factory=list)
    genre_ids: list[int] = Field(default_factory=list)
    character_names: list[str] = Field(default_factory=list)
    location_names: list[str] = Field(default_factory=list)
    flags: list[str] = Field(
        default_factory=list, description="doc_flags lenses, e.g. 'military-logic'."
    )
    keywords: list[str] = Field(
        default_factory=list, description="Free-text terms matched against title/text."
    )
    limit: int = Field(default=8, ge=1, le=50)


class DocumentHit(BaseModel):
    doc_id: str
    title: str
    source_id: str
    month_iso: str | None = None
    theme_ids: list[int] = Field(default_factory=list)
    genre_id: int | None = None
    quality_score: float | None = None
    flags: list[str] = Field(default_factory=list)
    snippet: str
    score: float = Field(description="Relevance to the query (higher is better).")


class CharacterFactHit(BaseModel):
    kind: str
    description: str
    date_iso: str | None = None
    speech_act: str | None = None
    reported_by: str | None = None


class RelationHit(BaseModel):
    other_name: str
    kind: str
    description: str | None = None
    confidence: float = 0.0


class CharacterDossier(BaseModel):
    character_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    biography: str | None = None
    political_label: str | None = None
    facts: list[CharacterFactHit] = Field(default_factory=list)
    relations: list[RelationHit] = Field(default_factory=list)


class LocationDossier(BaseModel):
    location_id: str
    name: str
    kind: str
    description: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    facts: list[str] = Field(default_factory=list)


class DirectorContext(BaseModel):
    """Everything the Director gets back for one query."""

    query: DirectorQuery
    project_slug: str
    documents: list[DocumentHit] = Field(default_factory=list)
    characters: list[CharacterDossier] = Field(default_factory=list)
    locations: list[LocationDossier] = Field(default_factory=list)

    def as_prompt(self, max_chars: int = 6000) -> str:
        """Render the context as a plain-text evidence block for an LLM director."""
        names = project().theme_names()
        lines: list[str] = [f"[Evidence from the {self.project_slug} archive]"]
        for dossier in self.characters:
            lines.append(
                f"\n## {dossier.name}"
                + (f" — {dossier.political_label}" if dossier.political_label else "")
            )
            if dossier.biography:
                lines.append(dossier.biography.strip())
            for fact in dossier.facts:
                when = f"{fact.date_iso} · " if fact.date_iso else ""
                lines.append(f"- {when}{fact.kind}: {fact.description}")
            for rel in dossier.relations:
                lines.append(
                    f"- relation · {rel.kind} → {rel.other_name}"
                    + (f" ({rel.description})" if rel.description else "")
                )
        for loc in self.locations:
            lines.append(f"\n## Place: {loc.name} ({loc.kind})")
            if loc.description:
                lines.append(loc.description.strip())
            lines.extend(f"- {fact}" for fact in loc.facts)
        if self.documents:
            lines.append("\n## Documents")
        for doc in self.documents:
            themes = ", ".join(names.get(t, str(t)) for t in doc.theme_ids)
            lines.append(
                f"\n### {doc.title} ({doc.source_id}, {doc.month_iso or 'undated'}; {themes})"
            )
            lines.append(doc.snippet.strip())
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


# ── Pure helpers ───────────────────────────────────────────────────────────────


def month_window(month_iso: str, radius: int) -> list[str]:
    """ISO months from month_iso - radius to month_iso + radius, inclusive."""
    year, month = (int(p) for p in month_iso.split("-"))
    index = year * 12 + (month - 1)
    out: list[str] = []
    for i in range(index - radius, index + radius + 1):
        out.append(f"{i // 12:04d}-{i % 12 + 1:02d}")
    return out


def parse_theme_ids(raw: object) -> list[int]:
    """documents.mapped_categories is a JSON array of ints (or NULL)."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return []
    return [int(x) for x in parsed if str(x).isdigit()] if isinstance(parsed, list) else []


def score_document(
    *,
    theme_ids: list[int],
    wanted_themes: list[int],
    month_iso: str | None,
    wanted_months: list[str],
    quality: float | None,
    flag_matches: int,
    keyword_matches: int,
) -> float:
    """Relevance of one document to a query. Pure, deterministic.

    Primary-theme match counts double; an exact month beats a window month;
    quality and lens/keyword matches add smaller bonuses.
    """
    score = 0.0
    for rank, theme in enumerate(theme_ids[:3]):
        if theme in wanted_themes:
            score += 2.0 if rank == 0 else 1.0
    if month_iso and wanted_months:
        if month_iso == wanted_months[len(wanted_months) // 2]:
            score += 1.5
        elif month_iso in wanted_months:
            score += 0.75
    score += (quality or 0.0) * 0.5
    score += 0.5 * flag_matches
    score += 0.75 * keyword_matches
    return round(score, 3)


def snippet(text: str, keywords: list[str], chars: int = _SNIPPET_CHARS) -> str:
    """The first keyword's neighbourhood, else the document head."""
    lowered = text.lower()
    for keyword in keywords:
        pos = lowered.find(keyword.lower())
        if pos >= 0:
            start = max(0, pos - chars // 3)
            return (
                ("…" if start else "")
                + text[start : start + chars].strip()
                + ("…" if start + chars < len(text) else "")
            )
    return text[:chars].strip() + ("…" if len(text) > chars else "")


# ── Knowledge base ─────────────────────────────────────────────────────────────


class KnowledgeBase:
    """Read-only queries over the research archive for the Director."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or settings.archive_db)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _has_table(conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    # ── Public API ────────────────────────────────────────────────────────────

    def context(self, query: DirectorQuery) -> DirectorContext:
        """Assemble the evidence bundle for a game situation."""
        if not self.db_path.exists():
            return DirectorContext(query=query, project_slug=project().slug)
        with self._connect() as conn:
            characters = self._characters(conn, query.character_names)
            locations = self._locations(conn, query.location_names)
            documents = self._documents(conn, query)
        return DirectorContext(
            query=query,
            project_slug=project().slug,
            documents=documents,
            characters=characters,
            locations=locations,
        )

    def find_characters(self, name_fragment: str, limit: int = 10) -> list[CharacterDossier]:
        """Name/alias search, no facts — for autocompletion in a director UI."""
        if not self.db_path.exists():
            return []
        with self._connect() as conn:
            if not self._has_table(conn, "characters"):
                return []
            rows = conn.execute(
                "SELECT character_id, name, aliases, biography, pol_label FROM characters "
                "WHERE lower(name) LIKE ? OR lower(aliases) LIKE ? ORDER BY mention_count DESC LIMIT ?",
                (f"%{name_fragment.lower()}%", f"%{name_fragment.lower()}%", limit),
            ).fetchall()
        return [self._dossier_from_row(row) for row in rows]

    # ── Sections ──────────────────────────────────────────────────────────────

    @staticmethod
    def _dossier_from_row(row: sqlite3.Row) -> CharacterDossier:
        keys = row.keys()
        try:
            aliases = json.loads(row["aliases"] or "[]")
        except (json.JSONDecodeError, TypeError):
            aliases = []
        return CharacterDossier(
            character_id=row["character_id"],
            name=row["name"],
            aliases=[str(a) for a in aliases],
            biography=row["biography"],
            political_label=row["pol_label"] if "pol_label" in keys else None,
        )

    def _characters(self, conn: sqlite3.Connection, names: list[str]) -> list[CharacterDossier]:
        if not names or not self._has_table(conn, "characters"):
            return []
        has_pol = "pol_label" in self._columns(conn, "characters")
        pol = ", pol_label" if has_pol else ""
        out: list[CharacterDossier] = []
        seen: set[str] = set()
        for name in names:
            row = conn.execute(
                f"SELECT character_id, name, aliases, biography{pol} FROM characters "
                "WHERE lower(name) LIKE ? OR lower(aliases) LIKE ? ORDER BY mention_count DESC LIMIT 1",
                (f"%{name.lower()}%", f"%{name.lower()}%"),
            ).fetchone()
            if row is None or row["character_id"] in seen:
                continue
            seen.add(row["character_id"])
            dossier = self._dossier_from_row(row)
            dossier.facts = self._facts(conn, dossier.character_id)
            dossier.relations = self._relations(conn, dossier.character_id)
            out.append(dossier)
        return out

    def _facts(self, conn: sqlite3.Connection, character_id: str) -> list[CharacterFactHit]:
        if not self._has_table(conn, "character_timeline"):
            return []
        cols = self._columns(conn, "character_timeline")
        extra = ", speech_act, reported_by" if {"speech_act", "reported_by"} <= cols else ""
        rows = conn.execute(
            f"SELECT kind, description, date_iso{extra} FROM character_timeline WHERE character_id = ? "
            "ORDER BY date_iso IS NULL, date_iso LIMIT ?",
            (character_id, _MAX_FACTS),
        ).fetchall()
        return [
            CharacterFactHit(
                kind=r["kind"],
                description=r["description"],
                date_iso=r["date_iso"],
                speech_act=r["speech_act"] if extra else None,
                reported_by=r["reported_by"] if extra else None,
            )
            for r in rows
        ]

    def _relations(self, conn: sqlite3.Connection, character_id: str) -> list[RelationHit]:
        if not self._has_table(conn, "character_relations"):
            return []
        rows = conn.execute(
            """
            SELECT r.kind, r.description, r.confidence,
                   CASE WHEN r.source_character_id = ? THEN t.name ELSE s.name END AS other_name
            FROM character_relations r
            JOIN characters s ON s.character_id = r.source_character_id
            JOIN characters t ON t.character_id = r.target_character_id
            WHERE r.source_character_id = ? OR r.target_character_id = ?
            ORDER BY r.confidence DESC LIMIT ?
            """,
            (character_id, character_id, character_id, _MAX_RELATIONS),
        ).fetchall()
        return [
            RelationHit(
                other_name=r["other_name"],
                kind=r["kind"],
                description=r["description"],
                confidence=r["confidence"] or 0.0,
            )
            for r in rows
        ]

    def _locations(self, conn: sqlite3.Connection, names: list[str]) -> list[LocationDossier]:
        if not names or not self._has_table(conn, "locations"):
            return []
        has_facts = self._has_table(conn, "location_facts")
        out: list[LocationDossier] = []
        for name in names:
            row = conn.execute(
                "SELECT location_id, name, kind, description, latitude, longitude FROM locations "
                "WHERE lower(name) LIKE ? OR lower(aliases) LIKE ? ORDER BY mention_count DESC LIMIT 1",
                (f"%{name.lower()}%", f"%{name.lower()}%"),
            ).fetchone()
            if row is None:
                continue
            facts: list[str] = []
            if has_facts:
                facts = [
                    r["detail"]
                    for r in conn.execute(
                        "SELECT detail FROM location_facts WHERE location_id = ? ORDER BY date_iso IS NULL, date_iso LIMIT ?",
                        (row["location_id"], _MAX_FACTS),
                    ).fetchall()
                ]
            out.append(
                LocationDossier(
                    location_id=row["location_id"],
                    name=row["name"],
                    kind=row["kind"],
                    description=row["description"],
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                    facts=facts,
                )
            )
        return out

    def _documents(self, conn: sqlite3.Connection, query: DirectorQuery) -> list[DocumentHit]:
        if not self._has_table(conn, "documents"):
            return []
        cols = self._columns(conn, "documents")
        if not {"mapped_categories", "mapped_month_iso"} <= cols:
            return []
        months = month_window(query.month_iso, query.month_window) if query.month_iso else []
        where: list[str] = ["mapped_categories IS NOT NULL"]
        params: list[object] = []
        if "quality_score" in cols:
            where.append("(quality_score IS NULL OR quality_score >= ?)")
            params.append(_MIN_QUALITY)
        if months:
            where.append(f"mapped_month_iso IN ({','.join('?' * len(months))})")
            params.extend(months)
        if query.genre_ids and "mapped_genre_id" in cols:
            where.append(f"mapped_genre_id IN ({','.join('?' * len(query.genre_ids))})")
            params.extend(query.genre_ids)
        if query.keywords:
            where.append(
                "("
                + " OR ".join("lower(title) LIKE ? OR lower(text) LIKE ?" for _ in query.keywords)
                + ")"
            )
            for keyword in query.keywords:
                params.extend([f"%{keyword.lower()}%", f"%{keyword.lower()}%"])
        genre_col = "mapped_genre_id" if "mapped_genre_id" in cols else "NULL"
        quality_col = "quality_score" if "quality_score" in cols else "NULL"
        rows = conn.execute(
            f"SELECT doc_id, title, text, source_id, mapped_month_iso, mapped_categories, "
            f"{genre_col} AS genre_id, {quality_col} AS quality FROM documents WHERE {' AND '.join(where)} LIMIT 2000",
            params,
        ).fetchall()
        flags_by_doc = self._flags(conn, [r["doc_id"] for r in rows]) if rows else {}

        hits: list[DocumentHit] = []
        for row in rows:
            theme_ids = parse_theme_ids(row["mapped_categories"])
            if query.theme_ids and not set(theme_ids) & set(query.theme_ids):
                continue
            flags = flags_by_doc.get(row["doc_id"], [])
            if query.flags and not set(flags) & set(query.flags):
                continue
            text = row["text"] or ""
            lowered = text.lower()
            keyword_matches = sum(
                1
                for k in query.keywords
                if k.lower() in lowered or k.lower() in row["title"].lower()
            )
            score = score_document(
                theme_ids=theme_ids,
                wanted_themes=query.theme_ids,
                month_iso=row["mapped_month_iso"],
                wanted_months=months,
                quality=row["quality"],
                flag_matches=len(set(flags) & set(query.flags)),
                keyword_matches=keyword_matches,
            )
            hits.append(
                DocumentHit(
                    doc_id=row["doc_id"],
                    title=row["title"],
                    source_id=row["source_id"],
                    month_iso=row["mapped_month_iso"],
                    theme_ids=theme_ids,
                    genre_id=row["genre_id"],
                    quality_score=row["quality"],
                    flags=flags,
                    snippet=snippet(text, query.keywords),
                    score=score,
                )
            )
        hits.sort(key=lambda h: (-h.score, h.month_iso or "", h.title))
        return hits[: query.limit]

    def _flags(self, conn: sqlite3.Connection, doc_ids: list[str]) -> dict[str, list[str]]:
        if not doc_ids or not self._has_table(conn, "doc_flags"):
            return {}
        out: dict[str, list[str]] = {}
        for i in range(0, len(doc_ids), 500):
            chunk = doc_ids[i : i + 500]
            rows = conn.execute(
                f"SELECT doc_id, flag FROM doc_flags WHERE doc_id IN ({','.join('?' * len(chunk))})",
                chunk,
            ).fetchall()
            for r in rows:
                out.setdefault(r["doc_id"], []).append(r["flag"])
        return out
