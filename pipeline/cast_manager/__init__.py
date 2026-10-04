"""
Ag-7 Cast Manager — extracts the people mentioned in verified documents.

Step 7 of the canonical pipeline. Runs after the Mapper: only verified, mapped
documents are mined. For each document it extracts the people named in it, records
who mentions them (when the text says so), and accumulates a per-character timeline
of dated facts. New characters are flagged ``needs_research`` so the Propositor can
later mint entity-driven missions (research that person). The Cast Director (Step 8)
builds the relational layer on top of these hard facts.

De-duplication / coreference is name-key based: ``character_key()`` normalises a name
(accents stripped, lower-cased, slugified) into the ``character_id`` primary key, so
"Salvador Allende" and "salvador  allende" collapse to one character. Alternate surface
forms are kept in ``aliases``. Fuzzy / LLM coreference is left as future work.

Two modes:
  --no-llm  : Heuristic name extraction (capitalised multi-word runs). Deterministic,
              offline, no facts — useful for fast testing or when Ollama is down.
  default   : Calls Gemma to extract people, mentions and timeline facts as JSON.
              Requires Ollama running.

CLI:
    uv run cast-manager run [--batch 50] [--all] [--no-llm]
    uv run cast-manager merge   # unify duplicate characters (also runs after every cycle)
    uv run cast-manager status
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import unicodedata
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lib.config import settings
from lib.llm import LLMClient, LLMError
from lib.logging_setup import get_logger
from lib.politics import parse_political_response
from lib.project import project
from lib.run_tracker import current_run_id, record_stage
from lib.schemas import SpeechAct

log = get_logger("cast_manager")

# ── Constants ──────────────────────────────────────────────────────────────────

_LLM_TEMPERATURE: float = 0.2
_LLM_NUM_PREDICT: int = 512  # JSON list of people with facts
# Long documents are split into overlapping chunks and mined piece by piece, then
# the people are unioned — nothing is dropped. A single arbitrary text[:N] cut hid
# everyone named later in the document (the 15 kB "89. Memorandum for the Record"
# names attendees far past any fixed cut). Chunks are sized so Gemma can enumerate
# each reliably; the overlap carries a name that straddles a boundary; the cap
# bounds LLM calls for pathological inputs (a warning fires if it bites).
_LLM_CHUNK_SIZE: int = 4000  # characters of body per LLM call
_LLM_CHUNK_OVERLAP: int = 250  # carried between consecutive chunks
_LLM_MAX_CHUNKS: int = 20  # ~75k chars — covers every realistic document

_MAX_PEOPLE_PER_DOC: int = 25
_MAX_FACTS_PER_PERSON: int = 10
_FACT_KINDS: frozenset[str] = frozenset(
    {"role", "event", "affiliation", "other", "statement", "rumor"}
)

# The four narrative groups a timeline can cover — acciones, enunciados,
# información, rumores. Diversity across groups (not raw kinds) feeds the
# completeness metric: role/affiliation/other all describe *information about*
# the character, while event/statement/rumor are distinct narrative material.
_KIND_GROUPS: dict[str, str] = {
    "event": "actions",
    "statement": "statements",
    "rumor": "rumors",
    "role": "information",
    "affiliation": "information",
    "other": "information",
}

# Completeness weighting (see compute_completeness). A component saturates at its
# target; the weights sum to 1.0 and the caps below bound the total regardless.
_FACTS_TARGET: int = 12
_MENTIONS_TARGET: int = 8
_SPREAD_TARGET_MONTHS: int = 6  # distinct dated months in-window for full spread credit
_PROJECT = project()
_SPREAD_WINDOW: tuple[str, str] = (_PROJECT.period.start_month, _PROJECT.period.end_month)  # the matrix window

_WEIGHT_IDENTITY: float = 0.15
_WEIGHT_DOCUMENTS: float = 0.20
_WEIGHT_EXTERNAL: float = 0.25
_WEIGHT_TIMELINE: float = 0.30
_WEIGHT_PIPELINE: float = 0.10

_TIMELINE_VOLUME_SHARE: float = 0.50
_TIMELINE_DIVERSITY_SHARE: float = 0.35
_TIMELINE_SPREAD_SHARE: float = 0.15

_EXTERNAL_LINKED_CREDIT: float = 0.06  # per source: linked but not yet analyzed
_EXTERNAL_ANALYZED_CREDIT: float = 0.125  # per source: fetched and mined into the profile

# Hard gates — a score can never exceed these caps while the condition holds.
_CAP_NO_EXTERNAL_ANALYSIS: float = 0.60  # neither Wikidata nor Wikipedia analyzed
_CAP_LOW_KIND_DIVERSITY: float = 0.75  # fewer than _MIN_KIND_GROUPS groups covered
_CAP_NO_RESEARCH_MISSION: float = 0.80  # never the subject of a completed entity mission
_MIN_KIND_GROUPS: int = 3

_COMPLETE_THRESHOLD: float = 0.85  # at/above this a character counts as complete


# ── Pure functions (offline-testable) ──────────────────────────────────────────


def character_key(name: str) -> str:
    """Normalise a person's name into a stable slug used as the dedup key.

    Strips accents, lower-cases, and collapses any run of non-alphanumerics into a
    single hyphen. Returns "" for a name with no usable characters (callers skip it).
    """
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in decomposed if not unicodedata.combining(c))
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_name).strip("-").lower()
    return slug


# Name particles — never usable as a surname for matching ("de", "la", …).
_NAME_PARTICLES: frozenset[str] = frozenset(
    {"de", "del", "la", "las", "los", "y", "e", "da", "dos", "van", "von", "der"}
)


def _name_tokens(name: str) -> list[str]:
    """Normalised tokens of a name (accents stripped, lower-cased)."""
    return [token for token in character_key(name).split("-") if token]


def _tokens_subsume(short: list[str], long: list[str]) -> bool:
    """True when `short` appears within `long` as an ordered subsequence.

    Single-letter tokens act as initials: "a" matches "augusto". Used to decide
    that "Salvador Allende" and "Salvador Allende Gossens" are one person, or
    "Henry A. Kissinger" and "Henry Kissinger".
    """
    remaining = iter(long)
    for short_token in short:
        for long_token in remaining:
            if (
                short_token == long_token
                or (len(short_token) == 1 and long_token.startswith(short_token))
                or (len(long_token) == 1 and short_token.startswith(long_token))
            ):
                break
        else:
            return False
    return True


def resolve_character_key(name: str, roster: dict[str, list[str]]) -> str | None:
    """Resolve a surface form to an existing character, or None if it is new.

    ``roster`` maps character_id → known surface forms (canonical name + aliases).
    Two rules, both safe-by-default (ambiguity never merges):

    - Multi-token names match a character when one form subsumes the other as an
      ordered subsequence ("Salvador Allende" ↔ "Salvador Allende Gossens",
      initials included). Exactly one matching character is required.
    - Single-token names ("Allende", "Kissinger") match only when exactly ONE
      character has that token as a surname (a non-first, non-particle token of
      any known form) — so "Allende" stays separate while both Salvador and
      Beatriz Allende exist.

    Only person-shaped roster forms anchor a match: an organisation form
    ("Banco Edwards") never pulls person surface forms into its character.
    """
    new_tokens = _name_tokens(name)
    if not new_tokens:
        return None

    candidates: set[str] = set()
    if len(new_tokens) == 1:
        token = new_tokens[0]
        if token in _NAME_PARTICLES or len(token) < 3:
            return None
        for key, forms in roster.items():
            for form in forms:
                if not _looks_like_person(form):
                    continue
                form_tokens = _name_tokens(form)
                if len(form_tokens) > 1 and token in form_tokens[1:]:
                    candidates.add(key)
                    break
    else:
        for key, forms in roster.items():
            for form in forms:
                if not _looks_like_person(form):
                    continue
                form_tokens = _name_tokens(form)
                if len(form_tokens) < 2:
                    continue
                short, long = sorted((new_tokens, form_tokens), key=len)
                if _tokens_subsume(short, long):
                    candidates.add(key)
                    break

    if len(candidates) == 1:
        return candidates.pop()
    return None  # no match, or ambiguous — never guess


# Stopwords (es/en) excluded from a fact's content signature.
_FACT_STOPWORDS: frozenset[str] = frozenset(
    {
        "de",
        "del",
        "la",
        "las",
        "los",
        "el",
        "en",
        "y",
        "a",
        "al",
        "un",
        "una",
        "su",
        "sus",
        "por",
        "con",
        "como",
        "fue",
        "es",
        "era",
        "ser",
        "the",
        "of",
        "in",
        "as",
        "was",
        "is",
        "were",
        "to",
        "for",
        "by",
        "at",
        "and",
        "an",
        "he",
        "she",
        "his",
        "her",
        "ella",
        "ellos",
        "lo",
        "le",
        "republica",
        "republic",
    }
)

# A new fact whose content tokens are >= this fraction contained in (or
# containing) an existing fact's tokens counts as a restatement of it.
_FACT_DUP_CONTAINMENT: float = 0.8


# Bilingual / derivational equivalences — the corpus mixes Spanish and English
# ("President of Chile" / "Presidente de la República" / "asumió la presidencia").
_FACT_TOKEN_CANON: dict[str, str] = {
    "president": "presidente",
    "presidenta": "presidente",
    "presidencia": "presidente",
    "presidential": "presidente",
    "presidencial": "presidente",
    "minister": "ministro",
    "ministra": "ministro",
    "ministerio": "ministro",
    "senator": "senador",
    "senadora": "senador",
    "ambassador": "embajador",
    "embajadora": "embajador",
    "deputy": "diputado",
    "diputada": "diputado",
    "commander": "comandante",
    "election": "eleccion",
    "elections": "eleccion",
    "elecciones": "eleccion",
    "electoral": "eleccion",
    "elected": "electo",
    "elegido": "electo",
    "government": "gobierno",
    "socialist": "socialista",
    "chilean": "chile",
    "chileno": "chile",
    "chilena": "chile",
}


def _fact_tokens(description: str) -> frozenset[str]:
    """Content tokens of a fact description (normalised, canonicalised es↔en,
    stopwords removed)."""
    return frozenset(
        _FACT_TOKEN_CANON.get(token, token)
        for token in _name_tokens(description)
        if token not in _FACT_STOPWORDS
    )


# statement/rumor are narrative material, not restatements of profile info: they
# never dedupe against the information kinds (the "other" wildcard excludes them).
_NARRATIVE_KINDS: frozenset[str] = frozenset({"statement", "rumor"})


def _fact_kinds_compatible(a: str, b: str) -> bool:
    """Kinds match for dedup purposes; "other" is a wildcard (the LLM labels the
    same fact inconsistently as role/other across documents) — but only among the
    information kinds: a statement or rumor never collapses into an info fact."""
    if a == b:
        return True
    if a in _NARRATIVE_KINDS or b in _NARRATIVE_KINDS:
        return False
    return a == "other" or b == "other"


def facts_are_duplicates(a: str, b: str) -> bool:
    """True when two descriptions state the same fact in different words.

    Containment test on content tokens: the smaller token set must be almost
    fully inside the larger one ("President of Chile" ⊂ "elected President of
    Chile"). Empty signatures never match.
    """
    tokens_a, tokens_b = _fact_tokens(a), _fact_tokens(b)
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b)
    return overlap / min(len(tokens_a), len(tokens_b)) >= _FACT_DUP_CONTAINMENT


@dataclass(frozen=True)
class CompletenessInputs:
    """Everything the strict completeness metric needs to know about a character."""

    mention_count: int = 0
    fact_count: int = 0
    fact_kinds: frozenset[str] = frozenset()
    dated_months: int = 0  # distinct in-window months with a dated timeline fact
    has_bio: bool = False
    has_vital_dates: bool = False  # birth_date or death_date known
    wikidata_linked: bool = False
    wikidata_analyzed: bool = False
    wikipedia_linked: bool = False
    wikipedia_analyzed: bool = False
    entity_missions_done: int = 0  # completed 'entity' research missions


@dataclass(frozen=True)
class CompletenessResult:
    """The 0-1 score plus its per-component breakdown (persisted for the admin)."""

    score: float
    detail: dict


def compute_completeness(inputs: CompletenessInputs) -> CompletenessResult:
    """Return the strict 0-1 completeness score for a character, with breakdown.

    Weighted components: identity (bio + a role + vital dates), documentary base
    (verified-doc mentions), external linking (Wikidata/Wikipedia — analyzed earns
    full credit, linked-only a fraction), timeline richness (volume x kind-group
    diversity x temporal spread) and pipeline passage (extracted + researched).

    Hard gates then cap the total: no external source analyzed → ≤0.60; fewer
    than three of the four kind groups (actions, statements, information, rumors)
    → ≤0.75; no completed entity mission → ≤0.80. A character therefore cannot
    look complete (≥ ``_COMPLETE_THRESHOLD``) without having gone through the
    whole research pipeline. Pure; all values rounded to 3 decimals.
    """
    identity = (
        (1.0 if inputs.has_bio else 0.0)
        + (1.0 if "role" in inputs.fact_kinds else 0.0)
        + (1.0 if inputs.has_vital_dates else 0.0)
    ) / 3.0

    documents = min(1.0, inputs.mention_count / _MENTIONS_TARGET)

    def _source_credit(linked: bool, analyzed: bool) -> float:
        if analyzed:
            return _EXTERNAL_ANALYZED_CREDIT
        if linked:
            return _EXTERNAL_LINKED_CREDIT
        return 0.0

    external = _source_credit(inputs.wikidata_linked, inputs.wikidata_analyzed) + _source_credit(
        inputs.wikipedia_linked, inputs.wikipedia_analyzed
    )

    groups = {_KIND_GROUPS[k] for k in inputs.fact_kinds if k in _KIND_GROUPS}
    timeline = (
        _TIMELINE_VOLUME_SHARE * min(1.0, inputs.fact_count / _FACTS_TARGET)
        + _TIMELINE_DIVERSITY_SHARE * (len(groups) / len(set(_KIND_GROUPS.values())))
        + _TIMELINE_SPREAD_SHARE * min(1.0, inputs.dated_months / _SPREAD_TARGET_MONTHS)
    )

    pipeline_part = 0.5 * (1.0 if inputs.mention_count > 0 else 0.0) + 0.5 * (
        1.0 if inputs.entity_missions_done > 0 else 0.0
    )

    # external's credits are absolute (they already sum to _WEIGHT_EXTERNAL at max)
    raw = (
        _WEIGHT_IDENTITY * identity
        + _WEIGHT_DOCUMENTS * documents
        + external
        + _WEIGHT_TIMELINE * timeline
        + _WEIGHT_PIPELINE * pipeline_part
    )

    caps: list[str] = []
    score = min(1.0, raw)
    if not (inputs.wikidata_analyzed or inputs.wikipedia_analyzed):
        caps.append("no_external_analysis")
        score = min(score, _CAP_NO_EXTERNAL_ANALYSIS)
    if len(groups) < _MIN_KIND_GROUPS:
        caps.append("low_kind_diversity")
        score = min(score, _CAP_LOW_KIND_DIVERSITY)
    if inputs.entity_missions_done == 0:
        caps.append("no_research_mission")
        score = min(score, _CAP_NO_RESEARCH_MISSION)
    score = round(score, 3)

    detail = {
        "identity": round(_WEIGHT_IDENTITY * identity, 3),
        "documents": round(_WEIGHT_DOCUMENTS * documents, 3),
        "external": round(external, 3),
        "timeline": round(_WEIGHT_TIMELINE * timeline, 3),
        "pipeline": round(_WEIGHT_PIPELINE * pipeline_part, 3),
        "kind_groups": sorted(groups),
        "raw": round(raw, 3),
        "caps": caps,
        "score": score,
    }
    return CompletenessResult(score=score, detail=detail)


# Character research loop — how thin a character must be, and how hard we chase it.
# The threshold equals _COMPLETE_THRESHOLD: a character is researched until it is
# actually complete under the strict metric, not merely half-documented.
_RESEARCH_THRESHOLD: float = _COMPLETE_THRESHOLD
_RESEARCH_MAX_ATTEMPTS: int = 5  # give up after this many entity missions
_RESEARCH_COOLDOWN_DAYS: float = 3.0  # wait this long between attempts on one character


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp, tolerating None / malformed values."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def should_research(
    completeness: float,
    attempts: int,
    last_research_at: str | None,
    now: datetime,
    *,
    threshold: float = _RESEARCH_THRESHOLD,
    max_attempts: int = _RESEARCH_MAX_ATTEMPTS,
    cooldown_days: float = _RESEARCH_COOLDOWN_DAYS,
) -> bool:
    """Whether a character still warrants an entity research mission.

    True when the record is thin (``completeness < threshold``), we have not
    already tried ``max_attempts`` times, and the cooldown since the last attempt
    has elapsed (so we do not re-mint a mission whose harvest is still in flight).
    Pure and offline — the whole research lifecycle is decided here.
    """
    if completeness >= threshold:
        return False
    if attempts >= max_attempts:
        return False
    last = _parse_iso(last_research_at)
    if last is not None and (now - last) < timedelta(days=cooldown_days):
        return False
    return True


@dataclass
class ExtractedPerson:
    """One person extracted from a single document (LLM or heuristic)."""

    name: str
    mentioned_by: str | None = None
    facts: list[dict] = field(default_factory=list)  # {kind, description, date_iso}
    birth_date: str | None = None  # ISO year/date of birth, when the text states it
    death_date: str | None = None  # ISO year/date of death


_SPEECH_ACT_VALUES: frozenset[str] = frozenset(act.value for act in SpeechAct)
_RUMOR_DEFAULT_CONFIDENCE: float = 0.3  # a rumor is low-confidence by definition


def _clean_fact(raw: object) -> dict | None:
    """Validate and normalise one fact dict; return None if unusable.

    Narrative kinds carry extra context: a ``statement`` may name its Searle/Flores
    ``speech_act``; a ``rumor`` MUST say who asserts it (``reported_by``) — an
    unattributed rumor is noise and is rejected — and defaults to a low
    ``confidence`` when the extractor gives none.
    """
    if not isinstance(raw, dict):
        return None
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        return None
    kind = raw.get("kind")
    kind = kind if isinstance(kind, str) and kind in _FACT_KINDS else "other"
    date_iso = raw.get("date_iso")
    date_iso = date_iso if isinstance(date_iso, str) and date_iso.strip() else None

    speech_act = raw.get("speech_act")
    speech_act = (
        speech_act
        if kind == "statement" and isinstance(speech_act, str) and speech_act in _SPEECH_ACT_VALUES
        else None
    )
    reported_by = raw.get("reported_by")
    reported_by = (
        reported_by.strip() if isinstance(reported_by, str) and reported_by.strip() else None
    )
    confidence = raw.get("confidence")
    confidence = (
        float(confidence)
        if isinstance(confidence, int | float) and 0.0 <= float(confidence) <= 1.0
        else None
    )
    if kind == "rumor":
        if reported_by is None:
            return None
        if confidence is None:
            confidence = _RUMOR_DEFAULT_CONFIDENCE

    return {
        "kind": kind,
        "description": description.strip(),
        "date_iso": date_iso,
        "speech_act": speech_act,
        "reported_by": reported_by,
        "confidence": confidence,
    }


def parse_people_response(response: str) -> list[ExtractedPerson] | None:
    """Parse the LLM's JSON into a list of ExtractedPerson.

    Expected shape: ``{"people": [{"name": str, "mentioned_by": str|null,
    "facts": [{"kind": str, "description": str, "date_iso": str|null}]}]}``.

    Lenient per-person: malformed people or facts are skipped rather than failing
    the whole document. Returns None only when the top-level shape is wrong (not an
    object, or no "people" list). An empty but well-formed list returns [].
    """
    try:
        data = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None
    raw_people = data.get("people")
    if not isinstance(raw_people, list):
        return None

    people: list[ExtractedPerson] = []
    for raw in raw_people[:_MAX_PEOPLE_PER_DOC]:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        mentioned_by = raw.get("mentioned_by")
        mentioned_by = (
            mentioned_by.strip() if isinstance(mentioned_by, str) and mentioned_by.strip() else None
        )
        raw_facts = raw.get("facts")
        facts: list[dict] = []
        if isinstance(raw_facts, list):
            for raw_fact in raw_facts[:_MAX_FACTS_PER_PERSON]:
                cleaned = _clean_fact(raw_fact)
                if cleaned is not None:
                    facts.append(cleaned)
        people.append(
            ExtractedPerson(
                name=name.strip(),
                mentioned_by=mentioned_by,
                facts=facts,
                birth_date=_clean_life_date(raw.get("birth_date")),
                death_date=_clean_life_date(raw.get("death_date")),
            )
        )

    return people


def _clean_life_date(raw: object) -> str | None:
    """Normalise a birth/death date string; None when blank or not a string."""
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _better_life_date(current: str | None, candidate: str | None) -> str | None:
    """Pick the more useful life date: keep a known value, prefer more precision.

    Never overwrites a known date with None; when both are present the longer
    (more granular) string wins ("1973" → "1973-09-11").
    """
    candidate = _clean_life_date(candidate)
    if candidate is None:
        return current
    if not current:
        return candidate
    return candidate if len(candidate) > len(current) else current


# Two+ consecutive capitalised tokens (Spanish letters), allowing lowercase
# connectors (de, del, la, los) — a rough proper-name heuristic for --no-llm mode.
_NAME_RE = re.compile(
    r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+(?:de|del|la|los|las|y)\s+|\s+)"
    r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*"
)


def heuristic_extract_people(text: str) -> list[ExtractedPerson]:
    """Offline fallback: extract candidate names as runs of capitalised words.

    Rough and recall-oriented — it records names only (no mentions, no facts) and
    de-duplicates by character key. Used by ``--no-llm`` mode and for offline tests.
    """
    seen: set[str] = set()
    people: list[ExtractedPerson] = []
    for match in _NAME_RE.finditer(text):
        name = match.group(0).strip()
        key = character_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        people.append(ExtractedPerson(name=name))
        if len(people) >= _MAX_PEOPLE_PER_DOC:
            break
    return people


# ── Cued backstop: recover named people the LLM misses (recall aid) ─────────────
# A capitalised-name run that is anchored by a PERSON cue — preceded by an
# honorific/title, or followed by ", <role>" — is very likely a real person, so we
# can add it deterministically without the noise of a bare capitalised-run scan.

_NAME_CORE = (
    r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+"
    r"(?:\s+(?:de|del|la|los|las|y|von|van)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+"
    r"|\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*"
)
_HONORIFICS = (
    "mr|mrs|ms|dr|doctor|president|vice[- ]?president|ambassador|secretary|"
    "under[- ]?secretary|minister|general|admiral|colonel|captain|senator|"
    "congressman|representative|se[ñn]or|sra?|don|do[ñn]a|presidente|ministro|"
    "embajador|senador|coronel|comandante|capit[aá]n|almirante"
)
_HONORIFIC_RE = re.compile(rf"(?i:\b(?:{_HONORIFICS}))\.?\s+({_NAME_CORE})")
_NAME_ROLE_RE = re.compile(rf"({_NAME_CORE}),\s+(?:the\s+|el\s+|la\s+)?([A-Za-zÁÉÍÓÚÑáéíóúñ]+)")
_ROLE_WORDS: frozenset[str] = frozenset(
    {
        "president",
        "chairman",
        "director",
        "publisher",
        "owner",
        "head",
        "chief",
        "editor",
        "manager",
        "founder",
        "leader",
        "presidente",
        "dueño",
        "jefe",
        "gerente",
        "fundador",
        "lider",
        "propietario",
        "secretario",
        "embajador",
        "ministro",
    }
)
# Organisation / place / demonym tokens (accent-stripped) that must never form a
# "person" — used to reject cued matches like "United States, president of …".
_ORG_PLACE_STOPWORDS: frozenset[str] = frozenset(
    {
        "united",
        "states",
        "state",
        "national",
        "security",
        "council",
        "central",
        "intelligence",
        "agency",
        "department",
        "house",
        "embassy",
        "committee",
        "board",
        "corporation",
        "company",
        "university",
        "party",
        "government",
        "congress",
        "senate",
        "chamber",
        "ministry",
        "court",
        "supreme",
        "bank",
        "foundation",
        "institute",
        "association",
        "federation",
        "union",
        "republic",
        "commission",
        "bureau",
        "office",
        "command",
        "force",
        "forces",
        "army",
        "navy",
        "air",
        "group",
        "movement",
        "front",
        "society",
        "enterprise",
        "america",
        "american",
        "americans",
        "chile",
        "chilean",
        "santiago",
        "washington",
        "moscow",
        "cuba",
        "cuban",
        "soviet",
        "york",
        "angeles",
        "aires",
        "buenos",
        "estados",
        "unidos",
        "gobierno",
        "ministerio",
        "partido",
        "embajada",
        "universidad",
        "congreso",
        "senado",
        "camara",
        "republica",
        "ejercito",
        "armada",
        "fuerza",
        "fuerzas",
        "junta",
        "unidad",
        "popular",
        "sociedad",
        "empresa",
        "corporacion",
        "compania",
        "comando",
        "comision",
        "oficina",
        "nacional",
        "banco",
        "fundacion",
        "instituto",
        "asociacion",
        "movimiento",
        "frente",
    }
)


def _looks_like_person(name: str) -> bool:
    """True when a candidate looks like a personal name.

    Rejects any form carrying an org/place token, or opening with an institution
    designator ("Banco Edwards", "Radio Magallanes"). Used to filter cued
    backstop candidates and to keep non-person forms from anchoring coreference.
    """
    tokens = _name_tokens(name)
    if tokens and tokens[0] in _ORG_DESIGNATOR_STARTS:
        return False
    significant = [t for t in tokens if t not in _NAME_PARTICLES]
    if not significant:
        return False
    return not any(token in _ORG_PLACE_STOPWORDS for token in significant)


# Tokens (accent-stripped) that are never part of a real person name — months,
# bare titles, and Spanish function words / pronouns / common verbs. Combined
# with _ORG_PLACE_STOPWORDS to reject garbage characters.
_NON_PERSON_WORDS: frozenset[str] = frozenset(
    {
        # months (es + en)
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "setiembre",
        "octubre",
        "noviembre",
        "diciembre",
        # bare titles / honorifics
        "presidente",
        "president",
        "senador",
        "senator",
        "ministro",
        "minister",
        "general",
        "embajador",
        "ambassador",
        "coronel",
        "colonel",
        "comandante",
        "commander",
        "doctor",
        "secretary",
        "secretario",
        "diputado",
        "gobernador",
        "governor",
        "mr",
        "mrs",
        "ms",
        "dr",
        "don",
        "sr",
        "sra",
        # document boilerplate
        "memorandum",
        "memo",
        "classification",
        "secret",
        "confidential",
        "unclassified",
        "top",
        "eyes",
        "only",
        "subject",
        "telegram",
        "cable",
        "airgram",
        "record",
        "conversation",
        "note",
        "annex",
        "enclosure",
        "attachment",
        "page",
        "ref",
        "from",
        "reference",
        "dispatch",
        "despatch",
        "letter",
        # Spanish function words / pronouns / common verbs seen as OCR garbage
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "y",
        "o",
        "u",
        "pero",
        "en",
        "con",
        "por",
        "para",
        "su",
        "sus",
        "este",
        "esta",
        "esto",
        "estos",
        "estas",
        "ese",
        "esa",
        "eso",
        "esos",
        "esas",
        "aquel",
        "nos",
        "les",
        "se",
        "lo",
        "le",
        "que",
        "quien",
        "quienes",
        "como",
        "cuando",
        "donde",
        "cual",
        "cuales",
        "salio",
        "dijo",
        "fue",
        "era",
        "son",
        "han",
        "hay",
        "habia",
        "segun",
        "tambien",
        "entonces",
        "asi",
        "aqui",
        "alli",
        "ademas",
        "sobre",
        "entre",
        "hacia",
        "desde",
        "hasta",
        "muy",
        "mas",
    }
)

# A name that STARTS with one of these is a document fragment, not a person
# ("Memorandum From Arnold Nachmanoff") — even though it contains real tokens.
_DOC_BOILERPLATE_STARTS: frozenset[str] = frozenset(
    {
        "memorandum",
        "memo",
        "classification",
        "subject",
        "telegram",
        "cable",
        "airgram",
        "dispatch",
        "despatch",
        "reference",
        "ref",
        "enclosure",
        "annex",
        "attachment",
        "note",
        "letter",
    }
)

# A leading institution designator means the name denotes an organisation, not a
# person, no matter how person-like the rest looks ("Banco Edwards", "Radio
# Magallanes"). Kept narrow — only unambiguous designators, never words that
# could plausibly open a real name.
_ORG_DESIGNATOR_STARTS: frozenset[str] = frozenset(
    {
        "banco",
        "banca",
        "radio",
        "diario",
        "revista",
        "editorial",
        "club",
        "hotel",
        "teatro",
        "cine",
        "compania",
        "company",
        "corporacion",
        "corporation",
        "sociedad",
        "empresa",
        "fundacion",
        "foundation",
        "instituto",
        "institute",
        "universidad",
        "university",
        "ministerio",
        "ministry",
        "fabrica",
    }
)


def is_plausible_person_name(name: str) -> bool:
    """Deterministic validity check: does this name plausibly denote a person?

    Rejects boilerplate, bare titles, months, organisations/places and function
    words — while keeping real people, including surname-only ("Lanusse") and
    title-prefixed ("General Carlos Prats") forms (only *entirely* non-person
    names, org-designator-prefixed names ("Banco Edwards"), or document-fragment
    prefixes, are rejected). Used to gate extraction and to clean the cast.
    """
    tokens = _name_tokens(name)
    if not tokens:
        return False
    if tokens[0] in _DOC_BOILERPLATE_STARTS or tokens[0] in _ORG_DESIGNATOR_STARTS:
        return False
    bad = _ORG_PLACE_STOPWORDS | _NON_PERSON_WORDS
    return not all(token in bad for token in tokens)


def clean_invalid_characters(conn: sqlite3.Connection, apply: bool = False) -> dict:
    """Find (and, with ``apply=True``, delete) implausible characters.

    Deletes cascade to mentions, timeline facts and relations. Dry-run by
    default — returns ``{"names": [...], "applied": bool}`` so callers can review
    before removing anything.
    """
    ensure_cast_tables(conn)
    rows = conn.execute("SELECT character_id, name FROM characters").fetchall()
    bad = [(r["character_id"], r["name"]) for r in rows if not is_plausible_person_name(r["name"])]

    if apply and bad:
        ids = [cid for cid, _ in bad]
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM character_mentions WHERE character_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM character_timeline WHERE character_id IN ({placeholders})", ids)
        for column in ("source_character_id", "target_character_id"):
            try:
                conn.execute(
                    f"DELETE FROM character_relations WHERE {column} IN ({placeholders})", ids
                )
            except sqlite3.OperationalError:
                pass  # relations table not created yet
        conn.execute(f"DELETE FROM characters WHERE character_id IN ({placeholders})", ids)
        conn.commit()
        for _cid, name in bad:
            log.info("cast_manager.character_removed", name=name[:60])

    return {"names": [name for _, name in bad], "applied": apply}


def cued_person_names(text: str) -> list[str]:
    """Names anchored by a person cue (honorific before, or ", role" after).

    High precision by design — it only fires on a cue, so document boilerplate
    ("Top Secret", "National Security Council") is not matched, and organisation
    names that carry a cue are filtered out by ``_looks_like_person``.
    """
    seen: set[str] = set()
    out: list[str] = []

    def add(name: str) -> None:
        name = name.strip()
        key = character_key(name)
        if not key or key in seen or not _looks_like_person(name):
            return
        seen.add(key)
        out.append(name)

    for match in _HONORIFIC_RE.finditer(text):
        add(match.group(1))
    for match in _NAME_ROLE_RE.finditer(text):
        if match.group(2).lower() in _ROLE_WORDS:
            add(match.group(1))
    return out[:_MAX_PEOPLE_PER_DOC]


def chunk_text(
    text: str, size: int = _LLM_CHUNK_SIZE, overlap: int = _LLM_CHUNK_OVERLAP
) -> list[str]:
    """Split ``text`` into overlapping chunks that together cover all of it.

    A document longer than ``size`` is mined piece by piece instead of being
    truncated, so no named person is lost. Each cut prefers a nearby paragraph or
    sentence boundary (so a name is not split mid-token); consecutive chunks share
    ``overlap`` characters so a name straddling a boundary appears whole in one of
    them. Returns [] for empty text, [text] when it already fits.
    """
    text = text or ""
    if len(text) <= size:
        return [text] if text else []
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            boundary = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if boundary > size * 0.6:  # only honour a boundary that isn't too early
                end = start + boundary + 1
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def merge_extracted_people(
    groups: list[list[ExtractedPerson]],
) -> list[ExtractedPerson]:
    """Union people extracted from several chunks of one document, keyed by name.

    Facts are concatenated and de-duplicated (``facts_are_duplicates``);
    ``mentioned_by`` and life dates are filled from whichever chunk supplied them;
    the fullest name form wins. First-seen order is preserved.
    """
    merged: dict[str, ExtractedPerson] = {}
    order: list[str] = []
    for group in groups:
        for person in group:
            key = character_key(person.name)
            if not key:
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = ExtractedPerson(
                    name=person.name,
                    mentioned_by=person.mentioned_by,
                    facts=list(person.facts),
                    birth_date=person.birth_date,
                    death_date=person.death_date,
                )
                order.append(key)
                continue
            if len(person.name) > len(existing.name):
                existing.name = person.name
            if not existing.mentioned_by and person.mentioned_by:
                existing.mentioned_by = person.mentioned_by
            existing.birth_date = _better_life_date(existing.birth_date, person.birth_date)
            existing.death_date = _better_life_date(existing.death_date, person.death_date)
            for fact in person.facts:
                if not any(
                    facts_are_duplicates(fact["description"], f["description"])
                    for f in existing.facts
                ):
                    existing.facts.append(fact)
            existing.facts = existing.facts[:_MAX_FACTS_PER_PERSON]
    return [merged[key] for key in order][:_MAX_PEOPLE_PER_DOC]


def _augment_with_backstop(people: list[ExtractedPerson], text: str) -> list[ExtractedPerson]:
    """Add cued person names the LLM missed, without duplicating what it found."""
    known = {character_key(person.name) for person in people}
    for name in cued_person_names(text):
        key = character_key(name)
        if key and key not in known:
            people.append(ExtractedPerson(name=name))
            known.add(key)
            if len(people) >= _MAX_PEOPLE_PER_DOC:
                break
    return people


# ── LLM prompt ──────────────────────────────────────────────────────────────

_LLM_SYSTEM = (
    "You are a prosopographer for a Digital Humanities project at Cambridge University, "
    f"building a cast of people from documents about {_PROJECT.entity_prompts.era_label}. "
    "You respond ONLY with a JSON object. No explanation, no markdown."
)

_LLM_PROMPT = """\
Extract EVERY identifiable PERSON named in this document — be exhaustive, do not
stop at the obvious figures. Include foreign businessmen, executives, diplomats,
military officers and officials, not only Chilean political figures. For each
person return:
  - "name": their full name as written (canonical form, not a pronoun or title alone)
  - "mentioned_by": the speaker or source who refers to them, if the text makes it
    explicit; otherwise null
  - "birth_date": their year or date of birth ("YYYY" / "YYYY-MM-DD") if the text
    states it, else null
  - "death_date": their year or date of death if the text states it, else null
  - "facts": 0-{max_facts} short dated facts the document states about them, each:
      - "kind": one of "role", "event", "affiliation", "statement", "rumor", "other"
      - "description": one concise clause (e.g. "appointed Minister of Economy")
      - "date_iso": "YYYY-MM" or "YYYY-MM-DD" if the text gives a date, else null
      - "speech_act": ONLY for kind "statement" — one of "assertive", "directive",
        "commissive", "expressive", "declarative"; else null
      - "reported_by": ONLY for kind "rumor" — who asserts or spreads it (a person,
        a newspaper, "US embassy cable"); REQUIRED for rumors
      - "confidence": ONLY for kind "rumor" — 0.0-1.0, how credible the document
        itself treats it (rumors are low, e.g. 0.2-0.4)

RULES:
- Only real, named people. Skip organisations, places, and unnamed roles.
- Be exhaustive: a foreign businessman mentioned once still counts (e.g. Donald
  Kendall, or Agustín Edwards).
- If the document states a person's role, office or affiliation, record it as a
  fact (kind "role" or "affiliation"), e.g. "president of Pepsi-Cola" or
  "publisher of El Mercurio".
- If the document quotes or reports something the person SAID or announced, record
  it as kind "statement" with its "speech_act" (e.g. "announced the nationalization
  of copper" → declarative; "promised to respect the constitution" → commissive).
- If the document passes on hearsay, an unverified claim or an attributed suspicion
  ABOUT the person ("it is said that…", "rumors that…", "X claims that…"), record it
  as kind "rumor" with "reported_by" (who says it) and a low "confidence". Never
  present a rumor as a plain fact.
- Use the fullest form of the name that appears (e.g. "Salvador Allende", not "Allende").
- Do not invent facts. If the document states nothing specific about a person beyond
  their being mentioned, return them with an empty "facts" list.

Respond with EXACTLY this shape:
{{"people": [{{"name": "...", "mentioned_by": null, "birth_date": null, "death_date": null, "facts": [{{"kind": "role", "description": "...", "date_iso": null, "speech_act": null, "reported_by": null, "confidence": null}}]}}]}}

TITLE: {title}

DOCUMENT:
{text}
"""


async def _llm_extract_chunk(
    text: str,
    title: str,
    client: LLMClient,
) -> list[ExtractedPerson] | None:
    """Call Gemma on one chunk of a document. Returns None on error/parse failure."""
    prompt = _LLM_PROMPT.format(
        title=title[:200] or "(no title)",
        text=text,
        max_facts=_MAX_FACTS_PER_PERSON,
    )
    try:
        raw = await client.chat(
            model=settings.ollama_model_npc,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=_LLM_TEMPERATURE,
            num_predict=_LLM_NUM_PREDICT,
            think=False,
        )
        log.debug("cast_manager.llm_raw_response", raw=raw.strip()[:120])
        people = parse_people_response(raw)
        if people is None:
            log.warning("cast_manager.llm_unparseable", raw=raw.strip()[:150], title=title[:60])
        return people
    except LLMError as exc:
        log.warning("cast_manager.llm_error", reason=str(exc)[:120])
        return None
    except Exception as exc:
        log.warning("cast_manager.llm_unexpected_error", reason=str(exc)[:120])
        return None


async def _llm_extract_people(
    text: str,
    title: str,
    client: LLMClient,
) -> list[ExtractedPerson] | None:
    """Extract people from a whole document, chunking long ones so nothing is lost.

    The body is split into overlapping chunks (``chunk_text``); each is mined and
    the people are unioned (``merge_extracted_people``). Returns None only when
    *every* chunk fails to parse — so the caller still falls back to the cued
    backstop — and [] when the document is empty.
    """
    chunks = chunk_text(text or "")
    if not chunks:
        return []
    if len(chunks) > _LLM_MAX_CHUNKS:
        log.warning(
            "cast_manager.doc_over_chunk_cap",
            chunks=len(chunks),
            cap=_LLM_MAX_CHUNKS,
            title=title[:60],
        )
        chunks = chunks[:_LLM_MAX_CHUNKS]

    groups: list[list[ExtractedPerson]] = []
    any_parsed = False
    for chunk in chunks:
        people = await _llm_extract_chunk(chunk, title, client)
        if people is not None:
            any_parsed = True
            groups.append(people)
    if not any_parsed:
        return None
    return merge_extracted_people(groups)


# ── Political assessment prompt (roadmap #3) ────────────────────────────────────

_POLITICS_SYSTEM = (
    f"You are a historian of {_PROJECT.entity_prompts.era_label} placing public figures "
    "on a two-axis political compass. You respond ONLY with a JSON object."
)

_POLITICS_PROMPT = """\
Place this person on a two-axis political compass as of 1969-1973:
  - "economic": -1.0 (far left / socialist) … 0 (centre) … +1.0 (far right / free-market)
  - "social":   -1.0 (libertarian / liberal-democratic) … +1.0 (authoritarian)
  - "label": a 2-4 word Spanish description (e.g. "socialista democrático",
    "conservador autoritario")

Judge by their known politics; if genuinely unknown, still give your best estimate.
Respond with EXACTLY: {{"economic": <num>, "social": <num>, "label": "..."}}

PERSON: {name}
CONTEXT: {context}"""


async def _llm_assess_politics(
    name: str, context: str, client: LLMClient
) -> tuple[float, float, str] | None:
    """Ask Gemma to place a person on the compass. None on error/parse failure."""
    prompt = _POLITICS_PROMPT.format(name=name[:120], context=context[:600] or "(none)")
    try:
        raw = await client.chat(
            model=settings.ollama_model_npc,
            messages=[
                {"role": "system", "content": _POLITICS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=_LLM_TEMPERATURE,
            num_predict=96,
            think=False,
        )
        return parse_political_response(raw)
    except LLMError as exc:
        log.warning("cast_manager.politics_llm_error", reason=str(exc)[:120])
        return None
    except Exception as exc:
        log.warning("cast_manager.politics_unexpected_error", reason=str(exc)[:120])
        return None


# ── Schema ──────────────────────────────────────────────────────────────────


def ensure_cast_tables(conn: sqlite3.Connection) -> None:
    """Create the characters, character_timeline and character_mentions tables (idempotent)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS characters (
            character_id       TEXT PRIMARY KEY,
            name               TEXT NOT NULL,
            aliases            TEXT NOT NULL DEFAULT '[]',
            biography          TEXT,
            completeness_score REAL NOT NULL DEFAULT 0.0,
            mention_count      INTEGER NOT NULL DEFAULT 0,
            needs_research     INTEGER NOT NULL DEFAULT 1,
            run_id             TEXT,
            birth_date         TEXT,
            death_date         TEXT,
            first_seen_at      TEXT NOT NULL,
            updated_at         TEXT NOT NULL
        )
    """)
    for column in (
        "birth_date TEXT",
        "death_date TEXT",
        "wikidata_id TEXT",
        "wikipedia_url TEXT",
        "wikidata_desc TEXT",
        "wikidata_checked_at TEXT",
        "pol_economic REAL",
        "pol_social REAL",
        "pol_label TEXT",
        "pol_source TEXT",
        "pol_checked_at TEXT",
        "research_attempts INTEGER NOT NULL DEFAULT 0",  # entity missions minted so far
        "last_research_at TEXT",  # when the last attempt was minted
        "completeness_detail TEXT",  # JSON breakdown of the strict metric (admin)
        "wikidata_analyzed_at TEXT",  # entity claims mined into the profile (not just linked)
        "wikipedia_analyzed_at TEXT",  # article fetched and mined into the profile
    ):
        try:
            conn.execute(f"ALTER TABLE characters ADD COLUMN {column}")
        except sqlite3.OperationalError:
            pass  # column already present
    conn.execute("""
        CREATE TABLE IF NOT EXISTS character_timeline (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            character_id TEXT NOT NULL,
            doc_id       TEXT,
            date_iso     TEXT,
            kind         TEXT NOT NULL DEFAULT 'other',
            description  TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            UNIQUE(character_id, doc_id, description)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS character_mentions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            character_id TEXT NOT NULL,
            doc_id       TEXT NOT NULL,
            mentioned_by TEXT,
            created_at   TEXT NOT NULL,
            UNIQUE(character_id, doc_id)
        )
    """)
    for column in (
        "speech_act TEXT",  # Searle/Flores act for kind='statement' (lib.schemas.SpeechAct)
        "reported_by TEXT",  # who asserts/propagates it — required context for kind='rumor'
        "confidence REAL",  # extractor confidence; rumors are low by definition
    ):
        try:
            conn.execute(f"ALTER TABLE character_timeline ADD COLUMN {column}")
        except sqlite3.OperationalError:
            pass  # column already present
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_timeline_character ON character_timeline(character_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mentions_character ON character_mentions(character_id)"
    )
    conn.commit()


def _migrate_documents_schema(conn: sqlite3.Connection) -> None:
    """Add columns the Cast Manager reads to documents if absent (idempotent).

    ``seed_character_id`` is normally written by the Archivero, but ensure it here
    too so the Cast Manager can force-attribute entity-harvested docs even when it
    runs against a database the current Archivero has not yet migrated.
    """
    for column_def in ("cast_extracted_at TEXT", "seed_character_id TEXT"):
        try:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {column_def}")
        except sqlite3.OperationalError:
            pass  # Column already present
    conn.commit()


# ── DB helpers ────────────────────────────────────────────────────────────────


def _fetch_uncast_documents(
    conn: sqlite3.Connection, batch_size: int, before_ts: str | None = None
) -> list[sqlite3.Row]:
    """Fetch verified, mapped documents to mine for cast info.

    Default (``before_ts=None``): only documents never mined
    (``cast_extracted_at IS NULL``). In re-extract mode, ``before_ts`` is the
    session start, so already-mined documents are re-processed once (their stamp
    is older than the session) and the loop self-terminates when all have been
    re-stamped this session.
    """
    if before_ts is None:
        return conn.execute(
            """
            SELECT doc_id, title, text, seed_character_id
            FROM documents
            WHERE verified_at IS NOT NULL
              AND mapped_category_id IS NOT NULL
              AND cast_extracted_at IS NULL
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
    return conn.execute(
        """
        SELECT doc_id, title, text, seed_character_id
        FROM documents
        WHERE verified_at IS NOT NULL
          AND mapped_category_id IS NOT NULL
          AND (cast_extracted_at IS NULL OR cast_extracted_at < ?)
        LIMIT ?
        """,
        (before_ts, batch_size),
    ).fetchall()


def _upsert_character(
    conn: sqlite3.Connection,
    key: str,
    person: ExtractedPerson,
    run_id: str | None,
) -> bool:
    """Insert a new character or fold an alias into an existing one.

    Returns True if a new character row was created, False if it already existed.
    New characters are stamped with ``run_id`` and ``needs_research=1``.
    """
    now = datetime.now(UTC).isoformat()
    existing = conn.execute(
        "SELECT name, aliases FROM characters WHERE character_id=?", (key,)
    ).fetchone()

    if existing is None:
        conn.execute(
            """
            INSERT INTO characters
                (character_id, name, aliases, completeness_score, mention_count,
                 needs_research, run_id, birth_date, death_date, first_seen_at, updated_at)
            VALUES (?, ?, '[]', 0.0, 0, 1, ?, ?, ?, ?, ?)
            """,
            (key, person.name, run_id, person.birth_date, person.death_date, now, now),
        )
        return True

    # Existing: fold the surface form in. The fullest form becomes the canonical
    # name; every other form accrues in aliases.
    if person.name and person.name != existing["name"]:
        aliases = set(json.loads(existing["aliases"] or "[]"))
        canonical = existing["name"]
        if len(_name_tokens(person.name)) > len(_name_tokens(canonical)):
            aliases.add(canonical)
            aliases.discard(person.name)
            conn.execute(
                "UPDATE characters SET name=?, aliases=?, updated_at=? WHERE character_id=?",
                (person.name, json.dumps(sorted(aliases), ensure_ascii=False), now, key),
            )
        elif person.name not in aliases:
            aliases.add(person.name)
            conn.execute(
                "UPDATE characters SET aliases=?, updated_at=? WHERE character_id=?",
                (json.dumps(sorted(aliases), ensure_ascii=False), now, key),
            )

    # Backfill / upgrade life dates when a better value arrives.
    life = conn.execute(
        "SELECT birth_date, death_date FROM characters WHERE character_id=?", (key,)
    ).fetchone()
    new_birth = _better_life_date(life["birth_date"], person.birth_date)
    new_death = _better_life_date(life["death_date"], person.death_date)
    if new_birth != life["birth_date"] or new_death != life["death_date"]:
        conn.execute(
            "UPDATE characters SET birth_date=?, death_date=?, updated_at=? WHERE character_id=?",
            (new_birth, new_death, now, key),
        )
    return False


def _add_mention(
    conn: sqlite3.Connection,
    key: str,
    doc_id: str,
    mentioned_by: str | None,
) -> bool:
    """Record that ``key`` is mentioned in ``doc_id``. Returns True if newly added."""
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO character_mentions
            (character_id, doc_id, mentioned_by, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (key, doc_id, mentioned_by, now),
    )
    return cur.rowcount > 0


def _row_seed_character(row: sqlite3.Row) -> str | None:
    """The seed character an entity mission harvested this doc for, or None."""
    try:
        return row["seed_character_id"] or None
    except (IndexError, KeyError):
        return None


def _force_seed_mention(conn: sqlite3.Connection, seed: str, doc_id: str) -> bool:
    """Attribute an entity-harvested doc to its seed character.

    Closes the character research loop: a document fetched *because it is about*
    the seed enriches the seed even when the LLM does not re-extract the seed's
    exact name from it. No-op if the seed character no longer exists (e.g. it was
    absorbed by a merge). Returns True if a new mention was recorded.
    """
    if not conn.execute("SELECT 1 FROM characters WHERE character_id = ?", (seed,)).fetchone():
        return False
    return _add_mention(conn, seed, doc_id, "entity-mission")


def _add_timeline_facts(
    conn: sqlite3.Connection,
    key: str,
    doc_id: str,
    facts: list[dict],
) -> int:
    """Append timeline facts for a character. Returns the count newly inserted.

    Two dedup layers: the UNIQUE(character, doc, description) constraint catches
    exact re-runs, and ``facts_are_duplicates`` catches the same fact restated
    from another document ("President of Chile" vs "elected President of Chile")
    — the restatement is skipped, but its date backfills an undated existing
    fact of the same kind.
    """
    now = datetime.now(UTC).isoformat()
    existing = conn.execute(
        "SELECT id, kind, description, date_iso FROM character_timeline WHERE character_id = ?",
        (key,),
    ).fetchall()

    added = 0
    for fact in facts:
        kind = fact.get("kind", "other")
        description = fact["description"]
        date_iso = fact.get("date_iso")

        restated = next(
            (
                row
                for row in existing
                if _fact_kinds_compatible(row["kind"], kind)
                and facts_are_duplicates(row["description"], description)
            ),
            None,
        )
        if restated is not None:
            if date_iso and not restated["date_iso"]:
                conn.execute(
                    "UPDATE character_timeline SET date_iso=? WHERE id=?",
                    (date_iso, restated["id"]),
                )
            continue

        cur = conn.execute(
            """
            INSERT OR IGNORE INTO character_timeline
                (character_id, doc_id, date_iso, kind, description,
                 speech_act, reported_by, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                doc_id,
                date_iso,
                kind,
                description,
                fact.get("speech_act"),
                fact.get("reported_by"),
                fact.get("confidence"),
                now,
            ),
        )
        if cur.rowcount > 0:
            added += 1
            existing.append(
                {
                    "id": cur.lastrowid,
                    "kind": kind,
                    "description": description,
                    "date_iso": date_iso,
                }
            )
    return added


def _count_done_entity_missions(conn: sqlite3.Connection, key: str) -> int:
    """Completed 'entity' research missions for a character (0 if no missions table)."""
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM missions "
            "WHERE kind='entity' AND character_id=? AND status='done'",
            (key,),
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return 0  # missions table is the Propositor's; absent in cast-only databases


def _recompute_completeness(conn: sqlite3.Connection, key: str) -> None:
    """Regather the strict-metric inputs for a character and persist score + detail."""
    mentions = conn.execute(
        "SELECT COUNT(*) FROM character_mentions WHERE character_id=?", (key,)
    ).fetchone()[0]
    fact_rows = conn.execute(
        "SELECT kind, date_iso FROM character_timeline WHERE character_id=?", (key,)
    ).fetchall()
    profile = conn.execute(
        "SELECT biography, birth_date, death_date, wikidata_id, wikipedia_url, "
        "wikidata_analyzed_at, wikipedia_analyzed_at FROM characters WHERE character_id=?",
        (key,),
    ).fetchone()
    if profile is None:
        return

    months = {
        str(row["date_iso"])[:7]
        for row in fact_rows
        if row["date_iso"] and _SPREAD_WINDOW[0] <= str(row["date_iso"])[:7] <= _SPREAD_WINDOW[1]
    }
    bio = profile["biography"]
    result = compute_completeness(
        CompletenessInputs(
            mention_count=mentions,
            fact_count=len(fact_rows),
            fact_kinds=frozenset(str(row["kind"]) for row in fact_rows),
            dated_months=len(months),
            has_bio=bool(bio and bio.strip()),
            has_vital_dates=bool(profile["birth_date"] or profile["death_date"]),
            wikidata_linked=bool(profile["wikidata_id"]),
            wikidata_analyzed=bool(profile["wikidata_analyzed_at"]),
            wikipedia_linked=bool(profile["wikipedia_url"]),
            wikipedia_analyzed=bool(profile["wikipedia_analyzed_at"]),
            entity_missions_done=_count_done_entity_missions(conn, key),
        )
    )
    conn.execute(
        "UPDATE characters SET mention_count=?, completeness_score=?, "
        "completeness_detail=?, updated_at=? WHERE character_id=?",
        (
            mentions,
            result.score,
            json.dumps(result.detail, ensure_ascii=False),
            datetime.now(UTC).isoformat(),
            key,
        ),
    )


def rescore_characters(conn: sqlite3.Connection) -> dict[str, int]:
    """Recompute every character's completeness under the current constants.

    Cached scores go stale whenever the metric changes (weights, targets, gates)
    — this recomputes all of them, persists the fresh breakdown, and then
    re-evaluates ``needs_research`` so deflated characters re-enter the research
    queue. Returns ``{"rescored", "complete", "requeued"}``.
    """
    ensure_cast_tables(conn)
    keys = [row[0] for row in conn.execute("SELECT character_id FROM characters").fetchall()]
    for key in keys:
        _recompute_completeness(conn, key)
    conn.commit()
    requeued = refresh_research_flags(conn)
    complete = conn.execute(
        "SELECT COUNT(*) FROM characters WHERE completeness_score >= ?",
        (_COMPLETE_THRESHOLD,),
    ).fetchone()[0]
    return {"rescored": len(keys), "complete": complete, "requeued": requeued}


def _stamp_cast_extracted(conn: sqlite3.Connection, doc_id: str) -> None:
    """Mark a document as mined for cast info so it is not reprocessed."""
    conn.execute(
        "UPDATE documents SET cast_extracted_at=? WHERE doc_id=?",
        (datetime.now(UTC).isoformat(), doc_id),
    )


def fetch_characters_needing_research(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return characters flagged needs_research — the Propositor's entity-mission queue.

    Exposed for the (separate) entity-driven-missions work: the Propositor polls this
    to mint "research that person" missions. Ordered most-mentioned first.
    """
    try:
        return conn.execute(
            """
            SELECT c.character_id, c.name, c.mention_count, c.completeness_score,
                   c.aliases,
                   (SELECT t.description FROM character_timeline t
                    WHERE t.character_id = c.character_id
                      AND t.kind IN ('affiliation', 'role')
                    ORDER BY CASE t.kind WHEN 'affiliation' THEN 0 ELSE 1 END, t.id
                    LIMIT 1) AS salient_fact
            FROM characters c
            WHERE c.needs_research = 1
            ORDER BY c.mention_count DESC, c.character_id ASC
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def refresh_research_flags(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    """Re-evaluate every character's ``needs_research`` flag from its completeness.

    The research lifecycle is owned here: a character is (re-)queued when
    ``should_research`` says it is still thin, under the attempt cap and past the
    cooldown; otherwise it is cleared. Runs at the end of every cast cycle, so new
    evidence (facts/mentions just harvested) immediately decides whether to chase
    the character again or let it rest. Returns the number of rows whose flag
    changed.
    """
    now = now or datetime.now(UTC)
    try:
        rows = conn.execute(
            "SELECT character_id, needs_research, completeness_score, "
            "research_attempts, last_research_at FROM characters"
        ).fetchall()
    except sqlite3.OperationalError:
        return 0

    changed = 0
    stamp = now.isoformat()
    for row in rows:
        want = should_research(
            float(row["completeness_score"] or 0.0),
            int(row["research_attempts"] or 0),
            row["last_research_at"],
            now,
        )
        if int(row["needs_research"] or 0) != int(want):
            conn.execute(
                "UPDATE characters SET needs_research = ?, updated_at = ? WHERE character_id = ?",
                (int(want), stamp, row["character_id"]),
            )
            changed += 1
    if changed:
        conn.commit()
    return changed


def _load_character_roster(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """All known surface forms per character: canonical name + aliases."""
    roster: dict[str, list[str]] = {}
    for row in conn.execute("SELECT character_id, name, aliases FROM characters"):
        try:
            aliases = json.loads(row["aliases"] or "[]")
        except json.JSONDecodeError:
            aliases = []
        roster[row["character_id"]] = [row["name"], *aliases]
    return roster


def merge_duplicate_characters(conn: sqlite3.Connection) -> int:
    """Unify characters that are the same person under different surface forms.

    Clusters rows with ``resolve_character_key`` semantics (safe-by-default:
    ambiguous surnames like "Allende" with several Allendes never merge). The
    survivor of each cluster is the fullest name (most tokens, then most
    mentions); the others' mentions, timeline facts and entity missions are
    re-pointed to it, their names accrue as aliases, and the rows are deleted.
    Idempotent. Returns the number of absorbed rows.
    """
    roster = _load_character_roster(conn)
    if len(roster) < 2:
        return 0

    # Resolve each character against the rest of the roster; union the matches.
    parent: dict[str, str] = {key: key for key in roster}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Resolve to a fixed point: multi-token forms unify first; once clusters
    # exist, single-token forms ("Allende") are matched against CLUSTERS, so two
    # rows that are really one person never count as an ambiguity.
    for _ in range(3):  # roster sizes are small; 2 passes reach the fixed point
        grouped: dict[str, list[str]] = {}
        for key, forms in roster.items():
            grouped.setdefault(find(key), []).extend(forms)

        changed = False
        for root, forms in grouped.items():
            rest = {other: f for other, f in grouped.items() if other != root}
            for form in forms:
                match = resolve_character_key(form, rest)
                if match is not None and find(match) != find(root):
                    union(root, match)
                    changed = True
        if not changed:
            break

    clusters: dict[str, list[str]] = {}
    for key in roster:
        clusters.setdefault(find(key), []).append(key)

    now = datetime.now(UTC).isoformat()
    merged = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        rows = {
            row["character_id"]: row
            for row in conn.execute(
                f"SELECT * FROM characters WHERE character_id IN ({','.join('?' * len(members))})",
                members,
            )
        }
        survivor_id = max(
            members,
            key=lambda k: (len(_name_tokens(rows[k]["name"])), rows[k]["mention_count"], k),
        )
        survivor = rows[survivor_id]
        absorbed = [k for k in members if k != survivor_id]

        aliases = set(json.loads(survivor["aliases"] or "[]"))
        needs_research = int(survivor["needs_research"])
        first_seen = survivor["first_seen_at"]
        birth_date = survivor["birth_date"]
        death_date = survivor["death_date"]
        for key in absorbed:
            row = rows[key]
            aliases.add(row["name"])
            aliases.update(json.loads(row["aliases"] or "[]"))
            needs_research = max(needs_research, int(row["needs_research"]))
            first_seen = min(first_seen, row["first_seen_at"])
            birth_date = _better_life_date(birth_date, row["birth_date"])
            death_date = _better_life_date(death_date, row["death_date"])

            # Re-point mentions/facts; OR IGNORE + DELETE handles UNIQUE collisions.
            conn.execute(
                "UPDATE OR IGNORE character_mentions SET character_id=? WHERE character_id=?",
                (survivor_id, key),
            )
            conn.execute("DELETE FROM character_mentions WHERE character_id=?", (key,))
            conn.execute(
                "UPDATE OR IGNORE character_timeline SET character_id=? WHERE character_id=?",
                (survivor_id, key),
            )
            conn.execute("DELETE FROM character_timeline WHERE character_id=?", (key,))
            try:
                conn.execute(
                    "UPDATE missions SET character_id=? WHERE character_id=?",
                    (survivor_id, key),
                )
            except sqlite3.OperationalError:
                pass  # missions table absent (cast-only databases, tests)
            conn.execute("DELETE FROM characters WHERE character_id=?", (key,))
            merged += 1
            log.info(
                "cast_manager.character_merged",
                absorbed=key,
                into=survivor_id,
                name=survivor["name"],
            )

        aliases.discard(survivor["name"])
        conn.execute(
            "UPDATE characters SET aliases=?, needs_research=?, first_seen_at=?, "
            "birth_date=?, death_date=?, updated_at=? WHERE character_id=?",
            (
                json.dumps(sorted(aliases), ensure_ascii=False),
                needs_research,
                first_seen,
                birth_date,
                death_date,
                now,
                survivor_id,
            ),
        )
        _recompute_completeness(conn, survivor_id)

    conn.commit()
    return merged


def dedupe_timeline_facts(conn: sqlite3.Connection) -> int:
    """Collapse restated timeline facts per (character, kind). Returns rows removed.

    Greedy clustering with ``facts_are_duplicates``: within each cluster the
    survivor is the richest description (longest), it inherits the earliest
    known date of the cluster, and the other rows are deleted. Idempotent;
    runs after character merging (which can bring restatements together).
    """
    removed = 0
    characters = [
        row[0] for row in conn.execute("SELECT DISTINCT character_id FROM character_timeline")
    ]
    for key in characters:
        rows = conn.execute(
            "SELECT id, kind, description, date_iso FROM character_timeline "
            "WHERE character_id = ? ORDER BY id",
            (key,),
        ).fetchall()

        clusters: list[list[sqlite3.Row]] = []
        for row in rows:
            for cluster in clusters:
                if _fact_kinds_compatible(cluster[0]["kind"], row["kind"]) and any(
                    facts_are_duplicates(member["description"], row["description"])
                    for member in cluster
                ):
                    cluster.append(row)
                    break
            else:
                clusters.append([row])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            survivor = max(cluster, key=lambda r: len(r["description"]))
            dates = sorted(r["date_iso"] for r in cluster if r["date_iso"])
            if dates and dates[0] != survivor["date_iso"]:
                conn.execute(
                    "UPDATE character_timeline SET date_iso=? WHERE id=?",
                    (dates[0], survivor["id"]),
                )
            for row in cluster:
                if row["id"] != survivor["id"]:
                    conn.execute("DELETE FROM character_timeline WHERE id=?", (row["id"],))
                    removed += 1
        if any(len(cluster) > 1 for cluster in clusters):
            _recompute_completeness(conn, key)

    conn.commit()
    return removed


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class CastResult:
    """Result of one Cast Manager run cycle."""

    processed: int = 0  # documents mined
    characters_new: int = 0  # new character rows created
    characters_updated: int = 0  # existing characters touched (alias/mention/fact)
    mentions_new: int = 0  # new (character, doc) mention rows
    facts_new: int = 0  # new timeline facts
    characters_merged: int = 0  # duplicate characters absorbed by the merge pass
    facts_deduped: int = 0  # restated timeline facts collapsed by the dedupe pass
    research_requeued: int = 0  # characters whose needs_research flag flipped this run


# ── Wikidata / Wikipedia enrichment (roadmap #2) ────────────────────────────────
# Each character is linked to its Wikidata entity and Wikipedia page for grounding
# and disambiguation. Two keyless Wikidata API calls per character (search →
# sitelinks), routed through the Gatekeeper like every other outbound request.

_WIKIDATA_API: str = "https://www.wikidata.org/w/api.php"
_WIKIPEDIA_PREFERRED_WIKIS: tuple[str, ...] = ("eswiki", "enwiki")


def wikidata_search_url(name: str) -> str:
    """Wikidata ``wbsearchentities`` URL for a person name (Spanish-first)."""
    query = urllib.parse.quote_plus(name)
    return (
        f"{_WIKIDATA_API}?action=wbsearchentities&search={query}"
        "&language=es&uselang=es&format=json&type=item&limit=5"
    )


def wikidata_sitelinks_url(qid: str) -> str:
    """Wikidata ``wbgetentities`` URL returning an entity's Wikipedia sitelinks."""
    return (
        f"{_WIKIDATA_API}?action=wbgetentities&ids={urllib.parse.quote_plus(qid)}"
        "&props=sitelinks/urls&format=json"
    )


def parse_wikidata_search(body: str) -> tuple[str, str, str] | None:
    """Return (qid, label, description) of the top search hit, or None."""
    try:
        results = json.loads(body).get("search") or []
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    if not results:
        return None
    top = results[0]
    qid = top.get("id")
    if not isinstance(qid, str) or not qid:
        return None
    return qid, top.get("label") or "", top.get("description") or ""


def parse_wikipedia_url(body: str, qid: str) -> str | None:
    """Extract the preferred (Spanish, then English) Wikipedia URL for an entity."""
    try:
        sitelinks = json.loads(body)["entities"][qid].get("sitelinks") or {}
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        return None
    for wiki in _WIKIPEDIA_PREFERRED_WIKIS:
        link = sitelinks.get(wiki)
        if isinstance(link, dict) and link.get("url"):
            return link["url"]
    return None


def resolve_wikidata(name: str, fetch: Callable[[str], str | None]) -> dict | None:
    """Resolve a character name to its Wikidata entity + Wikipedia URL.

    ``fetch(url)`` returns the response body (or None). Returns a dict with
    ``wikidata_id`` / ``wikidata_url`` / ``wikipedia_url`` / ``description``, or
    None when there is no match. Injecting ``fetch`` keeps this offline-testable.
    """
    search_body = fetch(wikidata_search_url(name))
    if search_body is None:
        return None
    parsed = parse_wikidata_search(search_body)
    if parsed is None:
        return None
    qid, _label, description = parsed
    wikipedia_url = None
    sitelinks_body = fetch(wikidata_sitelinks_url(qid))
    if sitelinks_body is not None:
        wikipedia_url = parse_wikipedia_url(sitelinks_body, qid)
    return {
        "wikidata_id": qid,
        "wikidata_url": f"https://www.wikidata.org/wiki/{qid}",
        "wikipedia_url": wikipedia_url,
        "description": description,
    }


def enrich_characters(
    conn: sqlite3.Connection,
    fetch: Callable[[str], str | None],
    limit: int = 50,
) -> dict[str, int]:
    """Resolve Wikidata/Wikipedia links for characters not yet checked.

    Stamps ``wikidata_checked_at`` on every character it processes (matched or
    not) so runs are self-terminating and don't re-query. Returns
    ``{"checked": n, "resolved": m}``.
    """
    ensure_cast_tables(conn)
    rows = conn.execute(
        "SELECT character_id, name FROM characters "
        "WHERE wikidata_checked_at IS NULL "
        "ORDER BY completeness_score DESC, mention_count DESC LIMIT ?",
        (limit,),
    ).fetchall()

    checked = resolved = 0
    for row in rows:
        now = datetime.now(UTC).isoformat()
        info = resolve_wikidata(row["name"], fetch)
        if info is not None:
            conn.execute(
                "UPDATE characters SET wikidata_id=?, wikipedia_url=?, wikidata_desc=?, "
                "wikidata_checked_at=?, updated_at=? WHERE character_id=?",
                (
                    info["wikidata_id"],
                    info["wikipedia_url"],
                    info["description"],
                    now,
                    now,
                    row["character_id"],
                ),
            )
            resolved += 1
            # Linking earns partial external credit in the strict metric.
            _recompute_completeness(conn, row["character_id"])
            log.info(
                "cast_manager.character_enriched",
                character=row["character_id"],
                wikidata=info["wikidata_id"],
            )
        else:
            conn.execute(
                "UPDATE characters SET wikidata_checked_at=?, updated_at=? WHERE character_id=?",
                (now, now, row["character_id"]),
            )
        # Commit per character: the loop is network-bound and slow, so an
        # interrupted run keeps its progress (each row is stamped, never re-queried).
        conn.commit()
        checked += 1
    return {"checked": checked, "resolved": resolved}


# ── Analyze: mine the linked Wikidata entity + Wikipedia article (F1) ───────────
# `cast-manager enrich` only LINKS a character (QID + article URL). This step
# ANALYZES those links — Wikidata claims and the article text become profile
# fields and timeline facts — and stamps wikidata_analyzed_at /
# wikipedia_analyzed_at, the gates the strict completeness metric requires.

_WIKIDATA_FACT_PROPS: dict[str, str] = {
    "P39": "role",  # position held
    "P106": "role",  # occupation
    "P102": "affiliation",  # political party
}
_WIKIDATA_BIRTH_PROP: str = "P569"
_WIKIDATA_DEATH_PROP: str = "P570"
_WIKIDATA_START_QUALIFIER: str = "P580"

# A person born after the research window cannot appear in 1969-73 documents:
# the link is a homonym (e.g. a modern footballer) and must be undone. A death
# before the window is NOT implausible — historical figures are legitimately
# mentioned (Marx, O'Higgins), so only the birth bound de-links.
_IDENTITY_MAX_BIRTH_YEAR: int = _PROJECT.period.end_year

_ANALYZE_ARTICLE_MAX_CHARS: int = 6000
_ANALYZE_BIO_MAX_CHARS: int = 1200
_ANALYZE_MAX_FACTS: int = 15


def wikidata_claims_url(qid: str) -> str:
    """Wikidata ``wbgetentities`` URL returning an entity's claims."""
    return (
        f"{_WIKIDATA_API}?action=wbgetentities&ids={urllib.parse.quote_plus(qid)}"
        "&props=claims&format=json"
    )


def wikidata_labels_url(qids: list[str]) -> str:
    """Wikidata ``wbgetentities`` URL returning es/en labels for value entities."""
    ids = urllib.parse.quote_plus("|".join(qids))
    return (
        f"{_WIKIDATA_API}?action=wbgetentities&ids={ids}&props=labels&languages=es|en&format=json"
    )


def _parse_wikidata_time(value: object) -> str | None:
    """Convert a Wikidata time value to ISO, honouring its precision.

    ``{"time": "+1908-06-26T00:00:00Z", "precision": 11}`` → ``1908-06-26``;
    precision 10 → ``YYYY-MM``; precision 9 (or lower) → ``YYYY``.
    """
    if not isinstance(value, dict):
        return None
    time_str = value.get("time")
    if not isinstance(time_str, str) or not time_str.startswith("+"):
        return None
    date_part = time_str[1:].split("T")[0]  # 1908-06-26
    precision = value.get("precision")
    if precision is None or precision >= 11:
        return date_part
    if precision == 10:
        return date_part[:7]
    return date_part[:4]


def _claim_time(statement: dict, path: str = "mainsnak") -> str | None:
    """The ISO time inside a claim's mainsnak, or None."""
    snak = statement.get(path) or {}
    datavalue = snak.get("datavalue") or {}
    return _parse_wikidata_time(datavalue.get("value"))


def parse_wikidata_claims(body: str, qid: str) -> dict | None:
    """Parse a ``wbgetentities&props=claims`` body into vitals + fact claims.

    Returns ``{"birth_date", "death_date", "claims": [{prop, value_qid, start}]}``
    or None when the body is malformed or the entity is absent.
    """
    try:
        claims = json.loads(body)["entities"][qid]["claims"]
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        return None
    if not isinstance(claims, dict):
        return None

    def _first_time(prop: str) -> str | None:
        for statement in claims.get(prop) or []:
            parsed = _claim_time(statement)
            if parsed:
                return parsed
        return None

    fact_claims: list[dict] = []
    for prop in _WIKIDATA_FACT_PROPS:
        for statement in claims.get(prop) or []:
            snak = statement.get("mainsnak") or {}
            value = (snak.get("datavalue") or {}).get("value") or {}
            value_qid = value.get("id") if isinstance(value, dict) else None
            if not isinstance(value_qid, str) or not value_qid:
                continue
            start = None
            qualifiers = statement.get("qualifiers") or {}
            for qualifier in qualifiers.get(_WIKIDATA_START_QUALIFIER) or []:
                start = _parse_wikidata_time((qualifier.get("datavalue") or {}).get("value"))
                if start:
                    break
            fact_claims.append({"prop": prop, "value_qid": value_qid, "start": start})

    return {
        "birth_date": _first_time(_WIKIDATA_BIRTH_PROP),
        "death_date": _first_time(_WIKIDATA_DEATH_PROP),
        "claims": fact_claims,
    }


def parse_wikidata_labels(body: str) -> dict[str, str]:
    """Map entity QIDs to their Spanish (fallback English) labels."""
    try:
        entities = json.loads(body)["entities"]
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        return {}
    labels: dict[str, str] = {}
    for qid, entity in entities.items():
        entity_labels = entity.get("labels") or {} if isinstance(entity, dict) else {}
        for lang in ("es", "en"):
            value = (entity_labels.get(lang) or {}).get("value")
            if isinstance(value, str) and value.strip():
                labels[qid] = value.strip()
                break
    return labels


def build_wikidata_facts(claims: list[dict], labels: dict[str, str]) -> list[dict]:
    """Turn parsed claims + labels into timeline-fact dicts. Unlabelled QIDs skip."""
    facts: list[dict] = []
    for claim in claims:
        label = labels.get(claim["value_qid"])
        if not label:
            continue
        facts.append(
            {
                "kind": _WIKIDATA_FACT_PROPS[claim["prop"]],
                "description": label,
                "date_iso": claim.get("start"),
            }
        )
    return facts


def wikidata_identity_plausible(birth_date: str | None) -> bool:
    """False when the linked entity cannot be the person in our documents."""
    if not birth_date:
        return True
    try:
        return int(birth_date[:4]) <= _IDENTITY_MAX_BIRTH_YEAR
    except ValueError:
        return True


def wikipedia_extract_url(article_url: str) -> str | None:
    """MediaWiki plain-text-extract API URL for a Wikipedia article URL."""
    parsed = urllib.parse.urlparse(article_url)
    if not parsed.netloc.endswith("wikipedia.org") or "/wiki/" not in parsed.path:
        return None
    title = parsed.path.split("/wiki/", 1)[1]
    if not title:
        return None
    return (
        f"https://{parsed.netloc}/w/api.php?action=query&prop=extracts"
        f"&explaintext=1&redirects=1&format=json&titles={title}"
    )


def parse_wikipedia_extract(body: str) -> str | None:
    """The plain-text extract of the first page in a MediaWiki extracts response."""
    try:
        pages = json.loads(body)["query"]["pages"]
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        return None
    for page in pages.values():
        extract = page.get("extract") if isinstance(page, dict) else None
        if isinstance(extract, str) and extract.strip():
            return extract.strip()
    return None


_ANALYZE_SYSTEM: str = (
    f"Eres un historiador especializado en {_PROJECT.entity_prompts.era_label_local}. "
    "Respondes SOLO con JSON válido, sin texto adicional."
)

_ANALYZE_PROMPT: str = """Lee este artículo de Wikipedia sobre {name} y devuelve JSON:

{{"biography": "un solo párrafo (máx. 3 frases) sobre quién es y su papel en el
periodo 1969-1973 chileno",
 "facts": [{{"kind": "role|event|affiliation|other", "description": "hecho concreto",
 "date_iso": "YYYY-MM o YYYY-MM-DD o null"}}]}}

Reglas: máximo {max_facts} hechos, prioriza el periodo 1969-1973 y fechas exactas;
no inventes nada que el artículo no diga.

ARTÍCULO:
{article}"""


def parse_analysis_response(raw: str) -> dict | None:
    """Parse the article-analysis JSON: a biography + cleaned timeline facts."""
    try:
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    if not isinstance(data, dict):
        return None
    biography = data.get("biography")
    biography = (
        biography.strip()[:_ANALYZE_BIO_MAX_CHARS]
        if isinstance(biography, str) and biography.strip()
        else None
    )
    facts = [
        cleaned
        for raw_fact in (data.get("facts") or [])[:_ANALYZE_MAX_FACTS]
        if (cleaned := _clean_fact(raw_fact)) is not None
    ]
    if biography is None and not facts:
        return None
    return {"biography": biography, "facts": facts}


async def _llm_analyze_article(name: str, article: str, client: LLMClient) -> dict | None:
    """Ask Gemma to mine a Wikipedia article for a bio + dated facts."""
    prompt = _ANALYZE_PROMPT.format(
        name=name[:120],
        max_facts=_ANALYZE_MAX_FACTS,
        article=article[:_ANALYZE_ARTICLE_MAX_CHARS],
    )
    try:
        raw = await client.chat(
            model=settings.ollama_model_npc,
            messages=[
                {"role": "system", "content": _ANALYZE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=_LLM_TEMPERATURE,
            num_predict=700,
            think=False,
        )
        return parse_analysis_response(raw)
    except LLMError as exc:
        log.warning("cast_manager.analyze_llm_error", reason=str(exc)[:120])
        return None
    except Exception as exc:
        log.warning("cast_manager.analyze_unexpected_error", reason=str(exc)[:120])
        return None


# ── Cast Manager agent ──────────────────────────────────────────────────────────


class CastManager:
    """Ag-7: Extracts people from verified, mapped documents into the cast tables.

    Reads documents with ``verified_at`` and ``mapped_category_id`` set but no
    ``cast_extracted_at``, extracts the people in each (via Gemma, or a heuristic in
    ``--no-llm`` mode), and upserts characters, mentions and timeline facts. New
    characters are flagged ``needs_research`` and stamped with the active run id.
    """

    def __init__(self, db_path: Path | None = None, use_llm: bool = True) -> None:
        self.db_path = db_path or settings.archive_db
        self.use_llm = use_llm

    async def run_cycle(
        self, batch_size: int = 50, all_docs: bool = False, re_extract: bool = False
    ) -> CastResult:
        """Mine verified+mapped documents for cast information.

        Default: process one batch of not-yet-extracted documents. With
        ``all_docs=True`` the run loops batch-by-batch until none remain — naturally
        self-terminating, since each processed document is stamped ``cast_extracted_at``.
        With ``re_extract=True`` it re-mines documents that were *already* processed
        (applying prompt/backstop improvements to the existing corpus); a session
        timestamp keeps it self-terminating.

        With ``use_llm=True`` (default) each document goes to Gemma; if Ollama is
        unavailable the run aborts cleanly with guidance. With ``use_llm=False`` a
        name heuristic is used instead (no facts, no mentioned_by).
        """
        if not self.db_path.exists():
            log.info("cast_manager.db_not_found", path=str(self.db_path))
            return CastResult()

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        result = CastResult()
        run_id = current_run_id()
        looped = all_docs or re_extract
        session_start = datetime.now(UTC).isoformat() if re_extract else None

        try:
            ensure_cast_tables(conn)
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
                    rows = _fetch_uncast_documents(conn, batch_size, before_ts=session_start)
                    if not rows:
                        if batch_num == 1:
                            log.info("cast_manager.nothing_to_extract")
                        else:
                            log.info("cast_manager.session_done", batches=batch_num - 1)
                        break

                    log.info(
                        "cast_manager.cycle_start",
                        batch=len(rows),
                        batch_num=batch_num,
                        use_llm=self.use_llm,
                        re_extract=re_extract,
                    )
                    await self._process_batch(rows, conn, client_ctx, run_id, result)
                    conn.commit()

                    if not looped:
                        break
            finally:
                if client_ctx is not None:
                    await client_ctx.__aexit__(None, None, None)

            result.characters_merged = merge_duplicate_characters(conn)
            result.facts_deduped = dedupe_timeline_facts(conn)
            # Re-evaluate the research queue against the freshly harvested evidence:
            # thin characters are (re-)flagged, complete ones rest (character loop).
            result.research_requeued = refresh_research_flags(conn)
        finally:
            conn.close()

        log.info(
            "cast_manager.cycle_done",
            processed=result.processed,
            characters_new=result.characters_new,
            mentions_new=result.mentions_new,
            facts_new=result.facts_new,
            characters_merged=result.characters_merged,
            facts_deduped=result.facts_deduped,
        )
        return result

    async def _open_ollama(self) -> LLMClient | None:
        """Open an Ollama client and verify the model is present, else None with guidance."""
        client = LLMClient()
        await client.__aenter__()
        try:
            models = await client.list_models()
            if not any(settings.ollama_model_npc in m.get("name", "") for m in models):
                log.error("cast_manager.model_not_found", model=settings.ollama_model_npc)
                print(
                    f"\n  ERROR: Model {settings.ollama_model_npc} not found in Ollama.\n"
                    "  Start Ollama with: ollama serve\n"
                    "  Or use --no-llm fallback.\n"
                )
                await client.__aexit__(None, None, None)
                return None
        except Exception as exc:
            log.error("cast_manager.ollama_unreachable", reason=str(exc)[:120])
            print(
                "\n  ERROR: Ollama is not running.\n"
                "  Start it with: ollama serve\n"
                "  Or use --no-llm fallback.\n"
            )
            await client.__aexit__(None, None, None)
            return None
        return client

    async def _extract(
        self,
        row: sqlite3.Row,
        client: LLMClient | None,
    ) -> list[ExtractedPerson]:
        """Extract people from one document via the LLM, or the heuristic fallback."""
        text = row["text"] or ""
        if self.use_llm and client is not None:
            people = await _llm_extract_people(text, row["title"] or "", client) or []
            # Recover named people the model missed (cued backstop — recall aid).
            return _augment_with_backstop(people, text)
        return heuristic_extract_people(text)

    async def _process_batch(
        self,
        rows: list[sqlite3.Row],
        conn: sqlite3.Connection,
        client: LLMClient | None,
        run_id: str | None,
        result: CastResult,
    ) -> None:
        """Extract and persist people for a batch of documents.

        Each surface form is resolved against the known roster first, so
        "Allende" / "Salvador Allende" / "Salvador Allende Gossens" land on one
        character instead of three (see ``resolve_character_key``).
        """
        roster = _load_character_roster(conn)
        for row in rows:
            doc_id = row["doc_id"]
            people = await self._extract(row, client)
            for person in people:
                key = character_key(person.name)
                if not key:
                    continue
                # Prevention: never store implausible names (boilerplate, titles,
                # months, function words) — from the LLM or the heuristic alike.
                if not is_plausible_person_name(person.name):
                    continue
                if key not in roster:
                    resolved = resolve_character_key(person.name, roster)
                    if resolved is not None:
                        key = resolved
                if _upsert_character(conn, key, person, run_id):
                    result.characters_new += 1
                else:
                    result.characters_updated += 1
                forms = roster.setdefault(key, [])
                if person.name not in forms:
                    forms.append(person.name)
                if _add_mention(conn, key, doc_id, person.mentioned_by):
                    result.mentions_new += 1
                result.facts_new += _add_timeline_facts(conn, key, doc_id, person.facts)
                _recompute_completeness(conn, key)
            # Close the research loop: a doc harvested by a character's entity
            # mission is attributed to that seed even if the LLM missed the name.
            seed = _row_seed_character(row)
            if seed and _force_seed_mention(conn, seed, doc_id):
                result.mentions_new += 1
                _recompute_completeness(conn, seed)
            _stamp_cast_extracted(conn, doc_id)
            result.processed += 1

    async def assess_politics(self, limit: int = 200) -> dict[str, int]:
        """Place characters on the political compass via Gemma (roadmap #3).

        Processes characters not yet assessed (``pol_checked_at IS NULL``), using
        the character's name plus its Wikidata description and a few timeline
        facts as context. Stamps every processed character so runs are
        idempotent and self-terminating. Returns ``{"assessed": n, "placed": m}``.
        """
        if not self.db_path.exists():
            return {"assessed": 0, "placed": 0}

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        assessed = placed = 0
        try:
            ensure_cast_tables(conn)
            rows = conn.execute(
                "SELECT character_id, name, wikidata_desc FROM characters "
                "WHERE pol_checked_at IS NULL "
                "ORDER BY completeness_score DESC, mention_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
            if not rows:
                return {"assessed": 0, "placed": 0}

            client = await self._open_ollama()
            if client is None:
                return {"assessed": 0, "placed": 0}
            try:
                for row in rows:
                    context = self._political_context(conn, row)
                    result = await _llm_assess_politics(row["name"], context, client)
                    now = datetime.now(UTC).isoformat()
                    if result is not None:
                        econ, soc, label = result
                        conn.execute(
                            "UPDATE characters SET pol_economic=?, pol_social=?, pol_label=?, "
                            "pol_source='llm', pol_checked_at=?, updated_at=? WHERE character_id=?",
                            (econ, soc, label, now, now, row["character_id"]),
                        )
                        placed += 1
                    else:
                        conn.execute(
                            "UPDATE characters SET pol_checked_at=?, updated_at=? "
                            "WHERE character_id=?",
                            (now, now, row["character_id"]),
                        )
                    conn.commit()
                    assessed += 1
            finally:
                await client.__aexit__(None, None, None)
        finally:
            conn.close()

        log.info("cast_manager.politics_done", assessed=assessed, placed=placed)
        return {"assessed": assessed, "placed": placed}

    async def analyze(self, fetch: Callable[[str], str | None], limit: int = 50) -> dict[str, int]:
        """Mine each linked character's Wikidata claims + Wikipedia article (F1).

        Wikidata (no LLM): positions/party/occupation become timeline facts,
        vital dates backfill birth/death, and a birth after 1973 de-links the
        homonym. Wikipedia (Gemma): the article extract becomes a one-paragraph
        biography (only if none exists) plus dated facts. Each side stamps its
        ``*_analyzed_at`` — the external-analysis gates of the strict metric.
        Fetch/LLM failures leave no stamp, so the character is retried next run.
        """
        zeros = {
            "analyzed": 0,
            "wikidata_done": 0,
            "wikipedia_done": 0,
            "facts_new": 0,
            "bios_set": 0,
            "delinked": 0,
        }
        if not self.db_path.exists():
            return zeros

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        counts = dict(zeros)
        try:
            ensure_cast_tables(conn)
            rows = conn.execute(
                "SELECT character_id, name, biography, birth_date, death_date, "
                "       wikidata_id, wikipedia_url, wikidata_analyzed_at, wikipedia_analyzed_at "
                "FROM characters "
                "WHERE wikidata_id IS NOT NULL "
                "  AND (wikidata_analyzed_at IS NULL "
                "       OR (wikipedia_url IS NOT NULL AND wikipedia_analyzed_at IS NULL)) "
                "ORDER BY completeness_score DESC, mention_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
            if not rows:
                return counts

            client = await self._open_ollama() if self.use_llm else None
            try:
                for row in rows:
                    if row["wikidata_analyzed_at"] is None:
                        outcome = self._analyze_wikidata(conn, row, fetch, counts)
                        if outcome != "analyzed":
                            counts["analyzed"] += 1
                            continue  # de-linked or fetch failed — skip Wikipedia too
                    if (
                        client is not None
                        and row["wikipedia_url"]
                        and row["wikipedia_analyzed_at"] is None
                    ):
                        await self._analyze_wikipedia(conn, row, fetch, client, counts)
                    counts["analyzed"] += 1
                    conn.commit()
            finally:
                if client is not None:
                    await client.__aexit__(None, None, None)
        finally:
            conn.close()

        log.info("cast_manager.analyze_done", **counts)
        return counts

    @staticmethod
    def _analyze_wikidata(
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        fetch: Callable[[str], str | None],
        counts: dict[str, int],
    ) -> str:
        """One character's Wikidata side. Returns 'analyzed', 'delinked' or 'skipped'."""
        key = row["character_id"]
        body = fetch(wikidata_claims_url(row["wikidata_id"]))
        if body is None:
            return "skipped"  # network/Gatekeeper failure — retry next run
        parsed = parse_wikidata_claims(body, row["wikidata_id"])
        if parsed is None:
            return "skipped"
        now = datetime.now(UTC).isoformat()

        if not wikidata_identity_plausible(parsed["birth_date"]):
            # Homonym (born after the window): undo the link, keep
            # wikidata_checked_at so enrich does not immediately re-link it.
            conn.execute(
                "UPDATE characters SET wikidata_id=NULL, wikipedia_url=NULL, "
                "wikidata_desc=NULL, updated_at=? WHERE character_id=?",
                (now, key),
            )
            counts["delinked"] += 1
            _recompute_completeness(conn, key)
            conn.commit()
            log.info(
                "cast_manager.wikidata_delinked",
                character=key,
                birth_date=parsed["birth_date"],
            )
            return "delinked"

        labels: dict[str, str] = {}
        value_qids = sorted({claim["value_qid"] for claim in parsed["claims"]})
        if value_qids:
            labels_body = fetch(wikidata_labels_url(value_qids))
            if labels_body is not None:
                labels = parse_wikidata_labels(labels_body)
        counts["facts_new"] += _add_timeline_facts(
            conn, key, None, build_wikidata_facts(parsed["claims"], labels)
        )

        # Vital dates: fill when missing, upgrade when Wikidata is more precise.
        for column in ("birth_date", "death_date"):
            value = parsed[column]
            if value and (not row[column] or len(value) > len(str(row[column]))):
                conn.execute(
                    f"UPDATE characters SET {column}=?, updated_at=? WHERE character_id=?",
                    (value, now, key),
                )

        conn.execute(
            "UPDATE characters SET wikidata_analyzed_at=?, updated_at=? WHERE character_id=?",
            (now, now, key),
        )
        counts["wikidata_done"] += 1
        _recompute_completeness(conn, key)
        conn.commit()
        return "analyzed"

    @staticmethod
    async def _analyze_wikipedia(
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        fetch: Callable[[str], str | None],
        client: LLMClient,
        counts: dict[str, int],
    ) -> None:
        """One character's Wikipedia side: article → bio + facts (LLM)."""
        key = row["character_id"]
        api_url = wikipedia_extract_url(row["wikipedia_url"])
        if api_url is None:
            return
        body = fetch(api_url)
        article = parse_wikipedia_extract(body) if body is not None else None
        if not article:
            return  # unreachable/empty — no stamp, retried next run
        analysis = await _llm_analyze_article(row["name"], article, client)
        if analysis is None:
            return
        now = datetime.now(UTC).isoformat()
        existing_bio = row["biography"]
        if analysis["biography"] and not (existing_bio and existing_bio.strip()):
            conn.execute(
                "UPDATE characters SET biography=?, updated_at=? WHERE character_id=?",
                (analysis["biography"], now, key),
            )
            counts["bios_set"] += 1
        counts["facts_new"] += _add_timeline_facts(conn, key, None, analysis["facts"])
        conn.execute(
            "UPDATE characters SET wikipedia_analyzed_at=?, updated_at=? WHERE character_id=?",
            (now, now, key),
        )
        counts["wikipedia_done"] += 1
        _recompute_completeness(conn, key)
        conn.commit()

    @staticmethod
    def _political_context(conn: sqlite3.Connection, row: sqlite3.Row) -> str:
        """Grounding context for a compass placement: Wikidata desc + a few facts."""
        parts: list[str] = []
        if row["wikidata_desc"]:
            parts.append(row["wikidata_desc"])
        facts = conn.execute(
            "SELECT description FROM character_timeline WHERE character_id=? "
            "AND kind IN ('role', 'affiliation') LIMIT 4",
            (row["character_id"],),
        ).fetchall()
        parts.extend(f["description"] for f in facts)
        return "; ".join(parts)

    def status(self) -> dict[str, int]:
        """Return cast counts: characters, mentions, timeline facts, needs_research, pending docs."""
        if not self.db_path.exists():
            return {}
        conn = sqlite3.connect(str(self.db_path))
        try:
            ensure_cast_tables(conn)
            _migrate_documents_schema(conn)

            def _count(sql: str) -> int:
                try:
                    return conn.execute(sql).fetchone()[0]
                except sqlite3.OperationalError:
                    return 0

            return {
                "characters": _count("SELECT COUNT(*) FROM characters"),
                "mentions": _count("SELECT COUNT(*) FROM character_mentions"),
                "timeline_facts": _count("SELECT COUNT(*) FROM character_timeline"),
                "needs_research": _count("SELECT COUNT(*) FROM characters WHERE needs_research=1"),
                "pending_docs": _count(
                    "SELECT COUNT(*) FROM documents WHERE verified_at IS NOT NULL "
                    "AND mapped_category_id IS NOT NULL AND cast_extracted_at IS NULL"
                ),
            }
        finally:
            conn.close()


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_gatekeeper_fetch() -> Callable[[str], str | None]:
    """A fetch(url) callable that routes through the Gatekeeper's POST /fetch."""
    import httpx

    gatekeeper_url = f"http://{settings.gatekeeper_host}:{settings.gatekeeper_port}"

    def _gatekeeper_json(url: str) -> str | None:
        try:
            resp = httpx.post(
                f"{gatekeeper_url}/fetch",
                json={"url": url, "timeout_s": 20.0},
                timeout=30.0,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return None if data.get("binary") else data.get("body")
        except Exception as exc:
            log.warning("cast_manager.gatekeeper_fetch_failed", url=url[:80], error=str(exc)[:120])
            return None

    return _gatekeeper_json


def main() -> None:
    import argparse
    import traceback

    from lib.logging_setup import configure_logging

    configure_logging()

    ap = argparse.ArgumentParser(
        prog="cast-manager",
        description="Ag-7 Cast Manager — extract people from verified, mapped documents",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Extract people from verified+mapped documents")
    run_p.add_argument(
        "--batch",
        type=int,
        default=50,
        help="Documents per batch (default: 50). With --all the run loops internally "
        "until every uncast document is processed.",
    )
    run_p.add_argument(
        "--all",
        action="store_true",
        help="Process ALL uncast documents in a self-terminating loop, not just one batch.",
    )
    run_p.add_argument(
        "--re-extract",
        action="store_true",
        help="Re-mine documents already processed (applies improved extraction to the "
        "existing corpus). Self-terminating.",
    )
    run_p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Gemma — use the capitalised-name heuristic (no facts, no mentions).",
    )

    sub.add_parser("merge", help="Unify duplicate characters (same person, different name forms)")
    clean_p = sub.add_parser(
        "clean", help="Remove implausible characters (boilerplate, titles, months, orgs)"
    )
    clean_p.add_argument(
        "--apply", action="store_true", help="Actually delete (default: dry-run — just lists them)"
    )
    enrich_p = sub.add_parser(
        "enrich", help="Link characters to Wikidata + Wikipedia (via the Gatekeeper)"
    )
    enrich_p.add_argument(
        "--limit", type=int, default=200, help="Max characters to resolve this run (default: 200)"
    )
    analyze_p = sub.add_parser(
        "analyze",
        help="Mine linked Wikidata claims + Wikipedia articles into profiles/timelines",
    )
    analyze_p.add_argument(
        "--limit", type=int, default=50, help="Max characters to analyze this run (default: 50)"
    )
    analyze_p.add_argument(
        "--no-llm",
        action="store_true",
        help="Wikidata claims only — skip the Wikipedia article analysis (needs Gemma).",
    )
    politics_p = sub.add_parser(
        "politics", help="Place characters on the political compass (Gemma)"
    )
    politics_p.add_argument(
        "--limit", type=int, default=200, help="Max characters to assess this run (default: 200)"
    )
    sub.add_parser(
        "score",
        help="Recompute every character's completeness under the current constants",
    )
    sub.add_parser("status", help="Print cast counts (characters, mentions, facts, pending)")

    args = ap.parse_args()

    if args.cmd == "run":
        use_llm = not args.no_llm
        log.info(
            "cast_manager.main_start",
            batch=args.batch,
            use_llm=use_llm,
            all_docs=args.all,
            re_extract=args.re_extract,
        )
        try:
            manager = CastManager(use_llm=use_llm)
            t0 = datetime.now(UTC)
            result = asyncio.run(
                manager.run_cycle(
                    batch_size=args.batch, all_docs=args.all, re_extract=args.re_extract
                )
            )
            rid = current_run_id()
            if rid:
                record_stage(
                    rid,
                    "cast_manager",
                    t0,
                    datetime.now(UTC),
                    {
                        "processed": result.processed,
                        "characters_new": result.characters_new,
                        "mentions_new": result.mentions_new,
                        "facts_new": result.facts_new,
                        "characters_merged": result.characters_merged,
                    },
                )
            log.info(
                "cast_manager.main_done",
                processed=result.processed,
                characters_new=result.characters_new,
                characters_updated=result.characters_updated,
                mentions_new=result.mentions_new,
                facts_new=result.facts_new,
            )
            print(
                f"  Processed: {result.processed} docs\n"
                f"  Characters: +{result.characters_new} new, {result.characters_updated} updated\n"
                f"  Mentions:  +{result.mentions_new}\n"
                f"  Facts:     +{result.facts_new}\n"
                f"  Merged:    {result.characters_merged} duplicados unificados"
            )
        except Exception as exc:
            log.error("cast_manager.main_error", error=str(exc)[:200])
            print(f"\n  ERROR: {exc}\n")
            traceback.print_exc()

    elif args.cmd == "merge":
        conn = sqlite3.connect(str(settings.archive_db))
        conn.row_factory = sqlite3.Row
        try:
            ensure_cast_tables(conn)
            merged = merge_duplicate_characters(conn)
            deduped = dedupe_timeline_facts(conn)
            log.info("cast_manager.merge_done", merged=merged, facts_deduped=deduped)
            print(f"  {merged} personajes duplicados unificados.")
            print(f"  {deduped} hechos repetidos de la línea de tiempo eliminados.")
        finally:
            conn.close()

    elif args.cmd == "enrich":
        conn = sqlite3.connect(str(settings.archive_db))
        conn.row_factory = sqlite3.Row
        try:
            result = enrich_characters(conn, _build_gatekeeper_fetch(), limit=args.limit)
            log.info("cast_manager.enrich_done", **result)
            print(f"  Revisados: {result['checked']} personajes")
            print(f"  Vinculados a Wikidata: {result['resolved']}")
            if result["checked"] == 0:
                print("  (nada pendiente — todos ya revisados)")
        finally:
            conn.close()

    elif args.cmd == "analyze":
        use_llm = not args.no_llm
        log.info("cast_manager.analyze_start", limit=args.limit, use_llm=use_llm)
        manager = CastManager(use_llm=use_llm)
        t0 = datetime.now(UTC)
        result = asyncio.run(manager.analyze(_build_gatekeeper_fetch(), limit=args.limit))
        rid = current_run_id()
        if rid:
            record_stage(rid, "cast_analyze", t0, datetime.now(UTC), result)
        print(f"  Analizados:  {result['analyzed']} personajes")
        print(
            f"  Wikidata:    {result['wikidata_done']} minados ({result['delinked']} deslinkeados)"
        )
        print(f"  Wikipedia:   {result['wikipedia_done']} artículos → {result['bios_set']} bios")
        print(f"  Hechos:      +{result['facts_new']} en líneas de tiempo")
        if result["analyzed"] == 0:
            print("  (nada pendiente — todos los linkeados ya analizados)")

    elif args.cmd == "clean":
        conn = sqlite3.connect(str(settings.archive_db))
        conn.row_factory = sqlite3.Row
        try:
            ensure_cast_tables(conn)
            result = clean_invalid_characters(conn, apply=args.apply)
            names = result["names"]
            if args.apply:
                log.info("cast_manager.clean_done", removed=len(names))
                print(f"  Eliminados {len(names)} personajes implausibles.")
            else:
                print(
                    f"  {len(names)} personajes implausibles (dry-run — usa --apply para eliminar):"
                )
            for name in names[:40]:
                print(f"    - {name}")
            if len(names) > 40:
                print(f"    … y {len(names) - 40} más.")
        finally:
            conn.close()

    elif args.cmd == "score":
        conn = sqlite3.connect(str(settings.archive_db))
        conn.row_factory = sqlite3.Row
        try:
            result = rescore_characters(conn)
            log.info("cast_manager.score_done", **result)
            print(f"  Re-evaluados: {result['rescored']} personajes")
            print(f"  Completos (≥ {_COMPLETE_THRESHOLD}): {result['complete']}")
            print(f"  Re-encolados para investigación: {result['requeued']}")
        finally:
            conn.close()

    elif args.cmd == "politics":
        log.info("cast_manager.politics_start", limit=args.limit)
        result = asyncio.run(CastManager(use_llm=True).assess_politics(limit=args.limit))
        print(f"  Evaluados: {result['assessed']} personajes")
        print(f"  Ubicados en la brújula: {result['placed']}")
        if result["assessed"] == 0:
            print("  (nada pendiente — todos ya evaluados)")

    elif args.cmd == "status":
        try:
            status = CastManager().status()
            if status:
                print(f"  Characters:     {status['characters']}")
                print(f"  Mentions:       {status['mentions']}")
                print(f"  Timeline facts: {status['timeline_facts']}")
                print(f"  Needs research: {status['needs_research']}")
                print(f"  Pending docs:   {status['pending_docs']}")
            else:
                print("Database not found or empty.")
        except Exception as exc:
            log.error("cast_manager.status_error", error=str(exc)[:200])
            print(f"\n  ERROR: {exc}\n")


if __name__ == "__main__":
    main()
