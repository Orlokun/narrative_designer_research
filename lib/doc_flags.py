"""
Document utility-flag registry — the controlled vocabulary of "good-for" tags.

The Mapper answers *what a document is about* (theme × genre × month). This module
answers a different question: *what is a document especially good for?* A single
source can be a strong input for one downstream visualiser and useless for another —
e.g. "188. Memorandum for the Record" is a strong source for **military logic**.

Utility flags are cross-cutting, curated tags a document carries so a downstream view
can pull a curated set (e.g. every ``military-logic`` source, or every ``high-value``
primary account). They are assigned by the Ag-9 Document Curator (``pipeline/doc_curator``),
stored in the ``doc_flags`` table, and surfaced as filters in the reader.

This module is the single source of truth for the vocabulary — every consumer (the
Curator, the Admin reader, future visualisers) imports ``FLAGS`` from here rather than
hard-coding slugs. Two kinds of flag:

  - ``good-for``: this document is a strong input for a particular lens / visualiser
    (military logic, economic planning, diplomacy, popular life, ideology, technology).
  - ``quality``: a cross-cutting judgement about the source itself (high-value,
    eyewitness).

Pure and offline — no I/O, no LLM. The Curator's LLM path uses ``build_flag_menu`` for
its prompt and ``parse_flags_response`` for its output; ``heuristic_flags`` is the
deterministic ``--no-llm`` fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# ── Registry ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UtilityFlag:
    """One curated utility tag a document can carry."""

    slug: str
    name: str
    kind: str  # 'good-for' | 'quality'
    description: str  # used verbatim in the Curator's LLM prompt + the UI tooltip
    keywords: list[str] = field(default_factory=list)  # heuristic (--no-llm) cues


# Ordered — the reader renders chips in this order. Slugs are stable identifiers;
# renaming one is a data migration, so treat them as an API.
FLAGS: list[UtilityFlag] = [
    UtilityFlag(
        slug="military-logic",
        name="Lógica militar",
        kind="good-for",
        description=(
            "Strong source for military reasoning: armed-forces strategy, coup planning, "
            "the logic of intervention or repression, chain of command, use of force."
        ),
        keywords=[
            "militar",
            "military",
            "ejército",
            "ejercito",
            "army",
            "golpe",
            "coup",
            "fuerzas armadas",
            "armed forces",
            "general",
            "junta",
            "insurrección",
            "insurreccion",
            "cuartel",
            "regimiento",
            "sublevación",
            "sublevacion",
            "intervención militar",
            "intervencion militar",
            "war",
            "guerra",
        ],
    ),
    UtilityFlag(
        slug="economic-planning",
        name="Planificación económica",
        kind="good-for",
        description=(
            "Strong source for economic and cybernetic planning: central planning, "
            "production targets, nationalisation, supply, prices, the Cybersyn system "
            "as an economic instrument."
        ),
        keywords=[
            "planificación",
            "planificacion",
            "planning",
            "econom",
            "económ",
            "producción",
            "produccion",
            "nacionalización",
            "nacionalizacion",
            "cybersyn",
            "cybernet",
            "cibernética",
            "cibernetica",
            "abastecimiento",
            "precios",
            "escasez",
            "corfo",
            "empresa",
            "industria",
            "gdp",
            "pib",
        ],
    ),
    UtilityFlag(
        slug="diplomacy",
        name="Diplomacia",
        kind="good-for",
        description=(
            "Strong source for foreign relations: diplomacy, cables and telexes between "
            "states, US or Soviet posture toward Chile, covert action, foreign policy."
        ),
        keywords=[
            "diploma",
            "embajada",
            "embassy",
            "ambassador",
            "embajador",
            "state department",
            "departamento de estado",
            "cia",
            "kissinger",
            "nixon",
            "foreign",
            "exterior",
            "cable",
            "telex",
            "relaciones exteriores",
            "política exterior",
            "politica exterior",
            "intervención",
            "intervencion",
            "washington",
            "moscú",
            "moscu",
        ],
    ),
    UtilityFlag(
        slug="popular-life",
        name="Vida popular",
        kind="good-for",
        description=(
            "Strong source for everyday life and popular mood: how people lived, worked "
            "and felt — testimony, street-level detail, culture, the texture of daily life "
            "under the Unidad Popular."
        ),
        keywords=[
            "pobla",
            "barrio",
            "obrero",
            "campesino",
            "vecino",
            "cotidian",
            "trabajador",
            "sindicato",
            "cordón",
            "cordon",
            "mujer",
            "familia",
            "vida diaria",
            "everyday",
            "testimonio",
            "testimony",
            "olla común",
            "olla comun",
        ],
    ),
    UtilityFlag(
        slug="ideology-rhetoric",
        name="Ideología y retórica",
        kind="good-for",
        description=(
            "Strong source for political persuasion: ideological framing, rhetoric, "
            "propaganda, the language used to mobilise or attack — the how of political "
            "argument rather than the facts of an event."
        ),
        keywords=[
            "socialismo",
            "socialism",
            "revolución",
            "revolucion",
            "imperialismo",
            "imperialism",
            "pueblo",
            "compañero",
            "companero",
            "burgues",
            "clase",
            "propaganda",
            "consigna",
            "manifiesto",
            "vía chilena",
            "via chilena",
            "reaccionari",
            "camarada",
            "lucha",
        ],
    ),
    UtilityFlag(
        slug="science-technology",
        name="Ciencia y tecnología",
        kind="good-for",
        description=(
            "Strong source for technical and technological detail: computing, telex "
            "networks, the Cybersyn operations room, engineering, scientific method — "
            "the how of the technical systems."
        ),
        keywords=[
            "computador",
            "computer",
            "software",
            "telex",
            "red",
            "network",
            "sala de operaciones",
            "operations room",
            "opsroom",
            "ingenier",
            "engineer",
            "técnic",
            "tecnic",
            "cybernet",
            "cibernét",
            "cibernet",
            "algoritmo",
            "modelo",
            "sistema",
            "beer",
            "stafford",
        ],
    ),
    UtilityFlag(
        slug="high-value",
        name="Alto valor",
        kind="quality",
        description=(
            "An especially rich, information-dense or historically important source — "
            "one a curator would single out regardless of theme."
        ),
        keywords=[],  # judgement, not keyword-detectable — LLM only
    ),
    UtilityFlag(
        slug="eyewitness",
        name="Testimonio directo",
        kind="quality",
        description=(
            "A first-hand, eyewitness or primary account — someone recording what they "
            "themselves saw, said or did, rather than a later summary."
        ),
        keywords=[
            "yo vi",
            "presencié",
            "presencie",
            "recuerdo",
            "estuve",
            "declaro",
            "i saw",
            "i witnessed",
            "eyewitness",
            "en persona",
            "de mi puño",
            "de mi puno",
            "primera persona",
        ],
    ),
]

_BY_SLUG: dict[str, UtilityFlag] = {f.slug: f for f in FLAGS}
VALID_SLUGS: frozenset[str] = frozenset(_BY_SLUG)

_MAX_FLAGS_PER_DOC: int = 5  # a document that is "good for everything" is good for nothing


def flag_by_slug(slug: str) -> UtilityFlag | None:
    """Return the registered flag for ``slug``, or None if unknown."""
    return _BY_SLUG.get(slug)


def is_valid_flag(slug: str) -> bool:
    """True if ``slug`` names a registered utility flag."""
    return slug in _BY_SLUG


def build_flag_menu() -> str:
    """Render the flag vocabulary as a numbered menu for the Curator's LLM prompt."""
    lines = [f'  - "{f.slug}" ({f.kind}): {f.description}' for f in FLAGS]
    return "\n".join(lines)


# ── LLM output parsing ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AssignedFlag:
    """One flag the Curator assigns to a document, with confidence and rationale."""

    slug: str
    confidence: float
    rationale: str


def _clamp_confidence(raw: object) -> float:
    """Coerce a model-supplied confidence to a float in [0, 1]; default 0.6."""
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.6
    return max(0.0, min(1.0, value))


def parse_flags_response(response: str) -> list[AssignedFlag] | None:
    """Parse the Curator LLM's JSON into a list of AssignedFlag.

    Expected shape: ``{"flags": [{"slug": str, "confidence": 0..1,
    "rationale": str}]}``. Lenient per-flag: entries with an unknown or missing
    slug are dropped rather than failing the whole document; duplicates collapse
    to the highest-confidence occurrence. Returns None only when the top-level
    shape is wrong (not an object, or no "flags" list). A well-formed empty list
    returns [].
    """
    try:
        data = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    if not isinstance(data, dict):
        return None
    raw_flags = data.get("flags")
    if not isinstance(raw_flags, list):
        return None

    best: dict[str, AssignedFlag] = {}
    for raw in raw_flags:
        if not isinstance(raw, dict):
            continue
        slug = raw.get("slug")
        if not isinstance(slug, str) or slug not in _BY_SLUG:
            continue
        confidence = _clamp_confidence(raw.get("confidence"))
        rationale = raw.get("rationale")
        rationale = rationale.strip() if isinstance(rationale, str) else ""
        existing = best.get(slug)
        if existing is None or confidence > existing.confidence:
            best[slug] = AssignedFlag(slug=slug, confidence=confidence, rationale=rationale)

    # Keep the strongest few, registry order broken by confidence.
    ordered = sorted(best.values(), key=lambda a: (-a.confidence, a.slug))
    return ordered[:_MAX_FLAGS_PER_DOC]


# ── Deterministic fallback (--no-llm) ─────────────────────────────────────────


def heuristic_flags(text: str, title: str = "") -> list[AssignedFlag]:
    """Assign utility flags by keyword frequency — the deterministic --no-llm path.

    Only ``good-for`` flags with keyword cues can fire (quality flags are judgement
    calls the LLM makes). A flag is assigned when its keywords appear at least twice
    (or once in the title); confidence scales with hit count. Returns the strongest
    few, most-confident first.
    """
    haystack = f"{title}\n{text}".lower()
    title_l = title.lower()
    assigned: list[AssignedFlag] = []
    for flag in FLAGS:
        if not flag.keywords:
            continue
        hits = sum(haystack.count(kw) for kw in flag.keywords)
        in_title = any(kw in title_l for kw in flag.keywords)
        if hits < 2 and not in_title:
            continue
        confidence = min(0.85, 0.4 + 0.1 * hits)
        matched = sorted({kw for kw in flag.keywords if kw in haystack})[:4]
        rationale = "keyword cues: " + ", ".join(matched) if matched else "keyword cues"
        assigned.append(AssignedFlag(slug=flag.slug, confidence=confidence, rationale=rationale))

    assigned.sort(key=lambda a: (-a.confidence, a.slug))
    return assigned[:_MAX_FLAGS_PER_DOC]
