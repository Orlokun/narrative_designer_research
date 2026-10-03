"""
Connector catalogue — what the engine *can* harvest from, and for which moments.

The Archivero knows how to call each connector (URL template + extractor); the
Gatekeeper knows how politely to call its domain. This module adds the third,
project-facing half: for a given historical moment, which connectors are even
worth probing? A connector declares the languages, regions and years it covers,
so the Source Scout can skip the Chile Declassification Project when researching
Weimar Berlin, and propose ``wikipedia_de`` instead of ``wikipedia_es``.

``applicable_connectors(project)`` is a pure function over this catalogue; the
Source Scout then probes the survivors against the live APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lib.project import ResearchProject

ANY = "*"


@dataclass(frozen=True)
class Connector:
    """One harvestable source as the project layer sees it."""

    id: str
    label: str
    domain: str
    kind: str  # archive | academic | government | press | reference | other
    access: str  # json | html | multistep
    languages: tuple[str, ...] = (ANY,)
    regions: tuple[str, ...] = (ANY,)
    coverage_years: tuple[int | None, int | None] = (None, None)
    notes: str = ""
    language_parametric: bool = False  # id pattern ``<name>_<lang>``

    def covers_language(self, language: str) -> bool:
        return ANY in self.languages or language in self.languages

    def covers_region(self, region: str) -> bool:
        return ANY in self.regions or region.lower() in {r.lower() for r in self.regions}

    def covers_years(self, start_year: int, end_year: int) -> bool:
        lo, hi = self.coverage_years
        if lo is not None and end_year < lo:
            return False
        return not (hi is not None and start_year > hi)

    def for_language(self, language: str) -> str:
        """The connector id to use for a language (``wikipedia_de`` for ``de``)."""
        if self.language_parametric:
            return f"{self.id}_{language}"
        return self.id


CATALOGUE: list[Connector] = [
    Connector(
        id="archive.org",
        label="Internet Archive",
        domain="archive.org",
        kind="archive",
        access="json",
        notes="advancedsearch JSON; broad, multilingual, all periods.",
    ),
    Connector(
        id="openalex",
        label="OpenAlex",
        domain="api.openalex.org",
        kind="academic",
        access="json",
        notes="Scholarly works; secondary literature about any period.",
    ),
    Connector(
        id="crossref",
        label="CrossRef",
        domain="api.crossref.org",
        kind="academic",
        access="json",
        notes="DOI metadata; secondary literature.",
    ),
    Connector(
        id="semantic_scholar",
        label="Semantic Scholar",
        domain="api.semanticscholar.org",
        kind="academic",
        access="json",
        notes="Shared unauthenticated pool is strict; a free key helps.",
    ),
    Connector(
        id="wikipedia",
        label="Wikipedia",
        domain="{lang}.wikipedia.org",
        kind="reference",
        access="json",
        language_parametric=True,
        notes="generator=search + extracts in the project language.",
    ),
    Connector(
        id="wikisource",
        label="Wikisource",
        domain="{lang}.wikisource.org",
        kind="archive",
        access="multistep",
        language_parametric=True,
        notes="Primary texts (speeches, laws, letters) in the project language.",
    ),
    Connector(
        id="chronicling_america",
        label="Chronicling America (LoC)",
        domain="chroniclingamerica.loc.gov",
        kind="press",
        access="json",
        languages=("en",),
        coverage_years=(1756, 1963),
        notes="US newspapers OCR; strongest before 1963.",
    ),
    Connector(
        id="frus",
        label="FRUS — Chile vol. XXI",
        domain="history.state.gov",
        kind="government",
        access="multistep",
        languages=("en", "es"),
        regions=("Chile",),
        coverage_years=(1969, 1976),
        notes="Sequential document crawl of one FRUS volume; Cybersyn-specific.",
    ),
    Connector(
        id="foia_chile",
        label="State Dept FOIA — Chile Declassification Project",
        domain="foia.state.gov",
        kind="government",
        access="multistep",
        languages=("en", "es"),
        regions=("Chile",),
        coverage_years=(1968, 1991),
        notes="JSON search → PDF → OCR; Cybersyn-specific collection.",
    ),
    Connector(
        id="marxists",
        label="Marxists.org — Allende archive",
        domain="www.marxists.org",
        kind="archive",
        access="multistep",
        languages=("es",),
        regions=("Chile",),
        coverage_years=(1969, 1973),
        notes="Index crawl of one author archive; Cybersyn-specific.",
    ),
    Connector(
        id="bn_digital",
        label="Biblioteca Nacional Digital (Chile)",
        domain="bibliotecanacionaldigital.gob.cl",
        kind="archive",
        access="html",
        languages=("es",),
        regions=("Chile",),
        notes="Returns homepage HTML, not results — kept for completeness.",
    ),
    Connector(
        id="web_serp",
        label="DuckDuckGo HTML search",
        domain="html.duckduckgo.com",
        kind="other",
        access="html",
        notes="Last resort; blocks bots most of the time.",
    ),
]

_BY_ID: dict[str, Connector] = {c.id: c for c in CATALOGUE}


def connector_by_id(connector_id: str) -> Connector | None:
    """Look a connector up by id, resolving ``wikipedia_de`` to the parametric entry."""
    if connector_id in _BY_ID:
        return _BY_ID[connector_id]
    base, _, _lang = connector_id.rpartition("_")
    connector = _BY_ID.get(base)
    if connector and connector.language_parametric:
        return connector
    return None


@dataclass
class Applicable:
    """A connector that fits a project, with the concrete id to route missions to."""

    connector: Connector
    source_id: str
    reasons: list[str] = field(default_factory=list)


def applicable_connectors(spec: ResearchProject) -> list[Applicable]:
    """Filter the catalogue down to connectors that fit the project's moment.

    A connector survives when it covers the project's language (or any), region
    (or any) and overlaps its years. Language-parametric connectors are resolved
    to ``<name>_<lang>``.
    """
    out: list[Applicable] = []
    start_year, end_year = spec.period.start_year, spec.period.end_year
    for connector in CATALOGUE:
        reasons: list[str] = []
        if not connector.covers_language(spec.language):
            reasons.append(f"language {spec.language} not covered")
        if not connector.covers_region(spec.region):
            reasons.append(f"region {spec.region} not covered")
        if not connector.covers_years(start_year, end_year):
            reasons.append(f"years {start_year}-{end_year} outside coverage")
        if reasons:
            continue
        out.append(Applicable(connector, connector.for_language(spec.language), ["fits"]))
    return out


def excluded_connectors(spec: ResearchProject) -> list[Applicable]:
    """The complement of ``applicable_connectors`` with the reasons each was dropped."""
    kept = {a.connector.id for a in applicable_connectors(spec)}
    out: list[Applicable] = []
    start_year, end_year = spec.period.start_year, spec.period.end_year
    for connector in CATALOGUE:
        if connector.id in kept:
            continue
        reasons: list[str] = []
        if not connector.covers_language(spec.language):
            reasons.append(f"language {spec.language} not covered")
        if not connector.covers_region(spec.region):
            reasons.append(f"region {spec.region} not covered")
        if not connector.covers_years(start_year, end_year):
            reasons.append(f"years {start_year}-{end_year} outside coverage")
        out.append(Applicable(connector, connector.for_language(spec.language), reasons))
    return out


def default_source_tiers(spec: ResearchProject) -> dict[str, list[str]]:
    """A sensible generic routing before the Source Scout has probed anything.

    Archives and the language Wikisource lead the critical tier; academic
    connectors follow; reference (Wikipedia) fills the lower tiers; the web
    search stays last and only for the critical tier.
    """
    ids = [a.source_id for a in applicable_connectors(spec)]
    kinds = {a.source_id: a.connector.kind for a in applicable_connectors(spec)}

    def pick(*wanted: str) -> list[str]:
        return [i for i in ids if kinds[i] in wanted and i != "web_serp"]

    archives = pick("archive", "government", "press")
    academic = pick("academic")
    reference = pick("reference")
    critica = archives + academic + reference + (["web_serp"] if "web_serp" in ids else [])
    alta = archives + academic + reference
    media = archives[:2] + academic[:1] + reference
    baja = academic[:1] + archives[:2] + reference
    return {"critica": critica, "alta": alta, "media": media, "baja": baja}
