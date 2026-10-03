"""
Research project specification — the layer that makes the pipeline reusable.

Everything the agents need to know about *which* historical moment they are
researching (period, themes, critical months, source tiers, search vocabulary,
relevance rules, entity hints) lives in one declarative YAML file instead of in
the agents' code. The Cybersyn project is the first such spec
(``projects/cybersyn/project.yaml``); a new historical moment is a new spec,
drafted by the Grid Proposer and refined by the Source Scout.

Agents import ``project()`` and read from it at module import time, so the
module-level constants they used to hardcode (``_CATEGORIES``, ``_MONTHS`` …)
are still there — now derived from the active spec.

The active spec is chosen with ``settings.research_project`` (env
``RESEARCH_PROJECT``), which may be a path to a YAML file or a project slug
under ``projects/<slug>/project.yaml``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Tier = Literal["critica", "alta", "media", "baja"]

TIER_ORDER: tuple[str, ...] = ("critica", "alta", "media", "baja")

_MONTH_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"


class Theme(BaseModel):
    """One row of the coverage matrix: a research theme (category)."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1)
    name: str
    weight: float = Field(
        ge=0.0, le=1.0, description="Thematic relevance 0-1 in the Propositor's priority formula."
    )
    tier: Tier = Field(description="Source-routing tier; selects a source list from source_tiers.")
    description: str = Field(
        default="", description="One line used verbatim in the Mapper's classification prompt."
    )
    search_terms: list[str] = Field(
        default_factory=list, description="Period-appropriate query fragments for the Propositor."
    )


class CriticalMonth(BaseModel):
    """A month that gets a priority boost and an axis label in the heatmap."""

    model_config = ConfigDict(extra="forbid")

    month_iso: str = Field(pattern=_MONTH_PATTERN)
    event: str
    label: str = Field(
        default="", description="Short heatmap tick label (6 chars or fewer recommended)."
    )


class Period(BaseModel):
    """The time axis of the matrix: an inclusive range of months."""

    model_config = ConfigDict(extra="forbid")

    start_month: str = Field(pattern=_MONTH_PATTERN)
    end_month: str = Field(pattern=_MONTH_PATTERN)
    fallback_month: str | None = Field(
        default=None,
        pattern=_MONTH_PATTERN,
        description="Month the Mapper uses for undatable documents (defaults to the middle of the range).",
    )

    @model_validator(mode="after")
    def _check_order(self) -> Period:
        if self.end_month < self.start_month:
            raise ValueError("period.end_month must not precede period.start_month")
        if self.fallback_month and not (self.start_month <= self.fallback_month <= self.end_month):
            raise ValueError("period.fallback_month must fall inside the period")
        return self

    def months(self) -> list[str]:
        """All ISO months in the range, inclusive, in order."""
        year, month = (int(part) for part in self.start_month.split("-"))
        out: list[str] = []
        while True:
            iso = f"{year:04d}-{month:02d}"
            out.append(iso)
            if iso == self.end_month:
                return out
            month += 1
            if month > 12:
                month, year = 1, year + 1

    @property
    def start_year(self) -> int:
        return int(self.start_month[:4])

    @property
    def end_year(self) -> int:
        return int(self.end_month[:4])

    def middle_month(self) -> str:
        """The fallback month: explicit if set, else the middle of the range."""
        if self.fallback_month:
            return self.fallback_month
        all_months = self.months()
        return all_months[len(all_months) // 2]


class EntityDefaults(BaseModel):
    """Nominal cell + sources for entity (person / place) research missions."""

    model_config = ConfigDict(extra="forbid")

    nominal_theme_id: int = Field(
        ge=1, description="Nominal theme for entity missions (never written to coverage)."
    )
    nominal_month_iso: str = Field(pattern=_MONTH_PATTERN)
    sources: list[str] = Field(default_factory=list)
    query_suffix: str = Field(description="Appended to quoted names, e.g. 'Chile 1970 1973'.")
    context_suffix: str = Field(
        description="Alternative suffix for the second query, e.g. 'Chile Unidad Popular'."
    )
    biography_suffix: str = Field(
        default="biografía política", description="Suffix for the biography fallback query."
    )


class RelevancePrompt(BaseModel):
    """Domain-specific pieces of the Verificator's relevance prompt."""

    model_config = ConfigDict(extra="forbid")

    domain: str = Field(
        description="One line: 'Chile under Salvador Allende, October 1969 - September 1973.'"
    )
    persona: str = Field(description="Who the LLM is, e.g. the research archivist line.")
    relevant_topics: list[str] = Field(default_factory=list)
    zero_score_rules: list[str] = Field(
        default_factory=list, description="Content that scores 0.0 regardless (wrong era / place)."
    )
    score_scale: list[str] = Field(
        default_factory=list, description="Lines explaining 0.0 … 1.0 for this domain."
    )


class ClassificationGuidance(BaseModel):
    """Domain-specific pieces of the Mapper's classification prompt."""

    model_config = ConfigDict(extra="forbid")

    persona: str
    reading_rule: str = Field(
        default="", description="Worked explanation of 'what the document IS vs mentions'."
    )
    category_examples: list[str] = Field(default_factory=list)
    json_examples: list[str] = Field(
        default_factory=list, description="Example JSON lines appended to the prompt."
    )


class ReformulationPrompt(BaseModel):
    """System prompt for the Propositor's period-vocabulary query rewriting."""

    model_config = ConfigDict(extra="forbid")

    system: str


class EntityPrompts(BaseModel):
    """Historian personas for the Cast / Location managers."""

    model_config = ConfigDict(extra="forbid")

    era_label: str = Field(
        description="e.g. 'Allende-era Chile (1969-1973)' — used in English prompts."
    )
    era_label_local: str = Field(
        description="e.g. 'el Chile de la Unidad Popular (1969-1973)' — used in local-language prompts."
    )
    name_example: str = Field(
        default="", description="A full name used as a prompt example, e.g. 'Salvador Allende'."
    )
    short_name_example: str = Field(default="", description="Its short form, e.g. 'Allende'.")


class SeedEntity(BaseModel):
    """A character, institution or place the Grid Proposer suggests researching first."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["character", "institution", "location"]
    role: str = ""
    aliases: list[str] = Field(default_factory=list)


class ResearchProject(BaseModel):
    """The whole research grid for one historical moment."""

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str
    description: str = ""
    language: str = Field(default="es", description="Primary language of the sources (ISO 639-1).")
    region: str = Field(description="Country / region keyword used in queries, e.g. 'Chile'.")
    period: Period
    themes: list[Theme]
    critical_months: list[CriticalMonth] = Field(default_factory=list)
    source_tiers: dict[str, list[str]] = Field(description="tier -> ordered connector list.")
    entity: EntityDefaults
    relevance: RelevancePrompt
    classification: ClassificationGuidance
    reformulation: ReformulationPrompt
    entity_prompts: EntityPrompts
    seeds: list[SeedEntity] = Field(default_factory=list)

    @field_validator("themes")
    @classmethod
    def _unique_theme_ids(cls, themes: list[Theme]) -> list[Theme]:
        ids = [t.id for t in themes]
        if len(ids) != len(set(ids)):
            raise ValueError("theme ids must be unique")
        if not themes:
            raise ValueError("a project needs at least one theme")
        return themes

    @model_validator(mode="after")
    def _cross_checks(self) -> ResearchProject:
        for tier in TIER_ORDER:
            if tier not in self.source_tiers:
                raise ValueError(f"source_tiers is missing tier '{tier}'")
        theme_ids = {t.id for t in self.themes}
        if self.entity.nominal_theme_id not in theme_ids:
            raise ValueError("entity.nominal_theme_id must be one of the themes")
        valid = set(self.period.months())
        for cm in self.critical_months:
            if cm.month_iso not in valid:
                raise ValueError(f"critical month {cm.month_iso} is outside the period")
        if self.entity.nominal_month_iso not in valid:
            raise ValueError("entity.nominal_month_iso is outside the period")
        return self

    # ── Derived views used by the agents ──────────────────────────────────────

    def months(self) -> list[str]:
        return self.period.months()

    def theme_by_id(self, theme_id: int) -> Theme | None:
        return next((t for t in self.themes if t.id == theme_id), None)

    def theme_names(self) -> dict[int, str]:
        return {t.id: t.name for t in self.themes}

    def theme_weights(self) -> list[float]:
        return [t.weight for t in sorted(self.themes, key=lambda t: t.id)]

    def critical_month_events(self) -> dict[str, str]:
        return {cm.month_iso: cm.event for cm in self.critical_months}

    def critical_month_labels(self) -> dict[str, str]:
        return {cm.month_iso: (cm.label or cm.event[:6]) for cm in self.critical_months}

    def base_terms(self) -> dict[int, list[str]]:
        return {t.id: list(t.search_terms) for t in self.themes}

    def category_prompt_block(self) -> str:
        """Numbered theme list for the Mapper prompt."""
        return "\n".join(f"{t.id}. {t.name} — {t.description}" for t in self.themes)

    def period_label(self) -> str:
        """Human label like 'October 1969 – September 1973'."""
        return f"{_month_name(self.period.start_month)} – {_month_name(self.period.end_month)}"

    def matrix_shape(self, n_genres: int) -> str:
        return f"{len(self.themes)}×{n_genres}×{len(self.months())}"


_MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _month_name(month_iso: str) -> str:
    year, month = month_iso.split("-")
    return f"{_MONTH_NAMES[int(month) - 1]} {year}"


# ── Loading ───────────────────────────────────────────────────────────────────

PROJECTS_DIR = Path(__file__).resolve().parent.parent / "projects"
DEFAULT_PROJECT_SLUG = "cybersyn"


def resolve_project_path(spec: str | Path | None) -> Path:
    """Turn a slug or a path into the YAML file path.

    ``None`` -> the default Cybersyn spec. A bare slug -> ``projects/<slug>/project.yaml``.
    """
    if spec is None or str(spec).strip() == "":
        return PROJECTS_DIR / DEFAULT_PROJECT_SLUG / "project.yaml"
    candidate = Path(spec)
    if candidate.suffix in {".yaml", ".yml"} or candidate.exists():
        return candidate
    return PROJECTS_DIR / str(spec) / "project.yaml"


def load_project(spec: str | Path | None = None) -> ResearchProject:
    """Read and validate a project spec from disk."""
    path = resolve_project_path(spec)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return ResearchProject.model_validate(raw)


def dump_project(project_spec: ResearchProject, path: Path) -> None:
    """Write a project spec to YAML (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = project_spec.model_dump(mode="json", exclude_none=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False, width=100)


@lru_cache(maxsize=1)
def project() -> ResearchProject:
    """The active project (cached). Chosen by ``settings.research_project``."""
    from lib.config import settings  # local import: config must not depend on this module

    return load_project(settings.research_project)
