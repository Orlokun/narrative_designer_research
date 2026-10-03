"""
Genre / document-type registry — the X axis of the coverage matrix.

The research matrix is 16 themes × 13 genres × 48 months. This module is the
single source of truth for the genre dimension: every agent (Mapper, Propositor,
Archivero, Admin) imports from here rather than keeping its own copy.

Each genre carries:
  - a description used verbatim in the Mapper's LLM classification prompt;
  - Spanish search-term fragments the Propositor combines with theme terms;
  - an ordered source-affinity list (which connectors naturally hold this genre);
  - per-theme plausibility priors (0-1): how likely it is that documents of this
    genre about that theme exist in 1969-73 archives. The Propositor multiplies
    cell priority by this prior so it never burns missions on cells history
    barely filled (e.g. Deporte × Cable diplomático), without hard-excluding them.

GENRE_UNCLASSIFIED (0) is the bucket for documents mapped before the genre axis
existed (or that the classifier could not type); it is never a mission target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GENRE_UNCLASSIFIED: int = 0


@dataclass(frozen=True)
class Genre:
    """One column of the matrix's genre axis."""

    id: int
    name: str
    slug: str
    description: str
    search_terms: list[str]
    sources: list[str]
    default_plausibility: float
    plausibility_overrides: dict[int, float] = field(default_factory=dict)

    def plausibility(self, category_id: int) -> float:
        """Prior (0-1) that this genre exists for the given theme in 1969-73 archives."""
        return self.plausibility_overrides.get(category_id, self.default_plausibility)


# Theme ids for reference (see lib docs / wiki coverage-matrix page):
#  1 Política Nacional   2 Salud              3 Economía Nacional  4 Industria Nacional
#  5 Industria Privada   6 Estados Unidos     7 Unión Soviética    8 Política Internacional
#  9 Educación          10 Ciencia y Tecnol. 11 Transporte        12 Urbanismo
# 13 Política Militar   14 Artes y Cultura   15 Deporte           16 Macroeconomía

GENRES: list[Genre] = [
    Genre(
        id=1,
        name="Discurso",
        slug="discurso",
        description=(
            "Speech, address, rally oratory or radio talk delivered by a person "
            "(Allende, ministers, union leaders). First-person public oratory."
        ),
        search_terms=["discurso", "alocución presidencial"],
        sources=["marxists", "wikisource_es", "archive.org"],
        default_plausibility=0.6,
        plausibility_overrides={1: 1.0, 3: 0.9, 4: 0.9, 6: 0.7, 9: 0.7, 13: 0.7, 16: 0.7, 14: 0.5, 15: 0.3},
    ),
    Genre(
        id=2,
        name="Carta y telex",
        slug="carta-telex",
        description=(
            "Letter, telex, telegram or memorandum addressed to a specific person or "
            "office (e.g. the Beer-Flores telexes). Direct interpersonal register."
        ),
        search_terms=["carta", "telex correspondencia"],
        sources=["archive.org", "wikisource_es", "marxists"],
        default_plausibility=0.5,
        plausibility_overrides={1: 0.8, 10: 0.9, 4: 0.7, 6: 0.7, 15: 0.2},
    ),
    Genre(
        id=3,
        name="Cable diplomático e inteligencia",
        slug="cable-diplomatico",
        description=(
            "Declassified diplomatic cable, embassy telegram or intelligence assessment "
            "(FRUS, State Department, CIA). Clinical third-person institutional register."
        ),
        search_terms=["cable diplomático", "embassy Santiago telegram"],
        sources=["foia_chile", "frus", "archive.org"],
        default_plausibility=0.3,
        plausibility_overrides={6: 1.0, 8: 0.9, 1: 0.8, 13: 0.8, 3: 0.7, 7: 0.7, 4: 0.6, 16: 0.6,
                                2: 0.2, 9: 0.2, 12: 0.2, 14: 0.15, 15: 0.1},
    ),
    Genre(
        id=4,
        name="Documento oficial",
        slug="documento-oficial",
        description=(
            "Decree, law, official act, ministerial resolution or government record "
            "(Diario Oficial, ministries). Performative legal language."
        ),
        search_terms=["decreto ley", "acta oficial gobierno"],
        sources=["wikisource_es", "archive.org"],
        default_plausibility=0.7,
        plausibility_overrides={1: 1.0, 3: 0.9, 4: 0.9, 2: 0.8, 9: 0.8, 11: 0.8, 16: 0.8,
                                12: 0.7, 13: 0.7, 14: 0.5, 15: 0.4},
    ),
    Genre(
        id=5,
        name="Transcripción parlamentaria y radial",
        slug="transcripcion",
        description=(
            "Transcript of a parliamentary session (Senate, Chamber of Deputies) or a "
            "radio/TV broadcast. Formal debate with named speakers and interjections."
        ),
        search_terms=["sesión cámara diputados", "transcripción debate senado"],
        sources=["wikisource_es", "archive.org"],
        default_plausibility=0.5,
        plausibility_overrides={1: 1.0, 3: 0.8, 4: 0.7, 13: 0.7, 9: 0.6, 14: 0.4, 15: 0.2},
    ),
    Genre(
        id=6,
        name="Prensa informativa",
        slug="prensa-informativa",
        description=(
            "News report from a daily or weekly paper (El Mercurio, La Nación, Clarín, "
            "foreign press). Journalistic third-person reporting of events."
        ),
        search_terms=["noticia prensa", "diario crónica"],
        sources=["chronicling_america", "archive.org", "wikipedia_es"],
        default_plausibility=0.8,
        plausibility_overrides={1: 1.0, 11: 0.9, 14: 0.7, 15: 0.7},
    ),
    Genre(
        id=7,
        name="Prensa de opinión",
        slug="prensa-opinion",
        description=(
            "Editorial, opinion column or satirical piece (Topaze). Persuasive, polemic "
            "first-person-plural register arguing a position."
        ),
        search_terms=["editorial opinión", "columna prensa política"],
        sources=["chronicling_america", "archive.org", "marxists"],
        default_plausibility=0.6,
        plausibility_overrides={1: 1.0, 3: 0.9, 5: 0.8, 6: 0.8, 13: 0.7, 2: 0.5, 15: 0.3},
    ),
    Genre(
        id=8,
        name="Informe técnico",
        slug="informe-tecnico",
        description=(
            "Technical or policy report from an institution (CORFO, INTEC, ODEPLAN, "
            "Banco Central, Cybersyn project documentation). Technocratic register with "
            "data, tables and recommendations."
        ),
        search_terms=["informe técnico", "memoria institucional informe"],
        sources=["archive.org", "openalex", "semantic_scholar"],
        default_plausibility=0.5,
        plausibility_overrides={10: 1.0, 3: 0.9, 4: 0.9, 16: 0.9, 11: 0.8, 12: 0.8, 2: 0.7,
                                9: 0.6, 13: 0.5, 6: 0.4, 14: 0.2, 15: 0.15},
    ),
    Genre(
        id=9,
        name="Artículo académico",
        slug="academico",
        description=(
            "Scholarly article, book chapter or thesis — contemporary with the period or "
            "retrospective historiography. Analytic register with citations."
        ),
        search_terms=["análisis estudio", "investigación académica"],
        sources=["openalex", "crossref", "semantic_scholar"],
        default_plausibility=0.8,
        plausibility_overrides={10: 0.9, 3: 0.9, 1: 0.9, 15: 0.5},
    ),
    Genre(
        id=10,
        name="Testimonio y memoria",
        slug="testimonio",
        description=(
            "Memoir, oral-history interview or first-person testimony, usually "
            "retrospective. Personal reminiscence register."
        ),
        search_terms=["testimonio memoria", "entrevista historia oral"],
        sources=["archive.org", "openalex", "wikipedia_es"],
        default_plausibility=0.5,
        plausibility_overrides={1: 0.9, 13: 0.9, 10: 0.7, 14: 0.6, 15: 0.3},
    ),
    Genre(
        id=11,
        name="Poesía y canción",
        slug="poesia-cancion",
        description=(
            "Poem or song lyric (Neruda, Víctor Jara, Nueva Canción Chilena). Lyric "
            "register, verse form."
        ),
        search_terms=["poema canción", "nueva canción chilena"],
        sources=["wikisource_es", "archive.org", "wikipedia_es"],
        default_plausibility=0.2,
        plausibility_overrides={14: 1.0, 1: 0.7, 4: 0.4, 2: 0.15, 15: 0.15, 16: 0.1},
    ),
    Genre(
        id=12,
        name="Narrativa y teatro",
        slug="narrativa-teatro",
        description=(
            "Novel, short story or play written in or about the period. Fictional "
            "narrative or dramatic dialogue."
        ),
        search_terms=["novela cuento", "obra teatro chileno"],
        sources=["archive.org", "wikisource_es", "wikipedia_es"],
        default_plausibility=0.2,
        plausibility_overrides={14: 1.0, 1: 0.6, 13: 0.4, 9: 0.3},
    ),
    Genre(
        id=13,
        name="Propaganda y panfleto",
        slug="propaganda",
        description=(
            "Political manifesto, party programme, pamphlet, campaign material or "
            "poster/mural slogan text. Imperative mass-mobilisation language."
        ),
        search_terms=["manifiesto panfleto", "programa político propaganda"],
        sources=["marxists", "archive.org", "wikisource_es"],
        default_plausibility=0.4,
        plausibility_overrides={1: 1.0, 4: 0.8, 3: 0.7, 6: 0.7, 5: 0.6, 13: 0.6, 9: 0.5,
                                14: 0.5, 2: 0.4, 15: 0.2},
    ),
]

GENRE_NAMES: dict[int, str] = {genre.id: genre.name for genre in GENRES}
VALID_GENRE_IDS: set[int] = set(GENRE_NAMES)


def genre_by_id(genre_id: int) -> Genre | None:
    """Return the Genre with the given id, or None if unknown."""
    for genre in GENRES:
        if genre.id == genre_id:
            return genre
    return None


def plausibility(category_id: int, genre_id: int) -> float:
    """Prior (0-1) that (theme, genre) cells exist in period archives.

    Unknown genre ids (including GENRE_UNCLASSIFIED) return 0.0 — they are never
    mission targets.
    """
    genre = genre_by_id(genre_id)
    if genre is None:
        return 0.0
    return genre.plausibility(category_id)
