"""
RE-0 Grid Proposer — turns a historical moment into a research grid.

Given a moment ("Weimar Berlin between the 1929 crash and Hitler's chancellorship"),
a period and a region, the Grid Proposer drafts the whole project spec the
pipeline needs (see lib/project.py): the themes of the coverage matrix with
weights, tiers and period vocabulary; the critical months; the relevance and
classification rules; and the seed cast — institutions, characters and places
to research first. The LLM proposes, pure code validates and fills the
boilerplate, and the result is written to ``projects/<slug>/project.yaml`` for
the historian to edit before the Source Scout and the pipeline take over.

Operation:
1. Build the moment brief from the CLI inputs.
2. Ask the LLM (``LLMClient``) for a JSON proposal; ``--no-llm`` yields a generic
   skeleton grid instead so the spec can be hand-written from a template.
3. Parse, validate and complete the proposal into a ``ResearchProject``.
4. Dump it to YAML. The Cybersyn spec is never overwritten by this agent.

CLI:
    uv run grid-proposer propose --moment "..." --start 1929-01 --end 1933-03 \
        --region Germany --language de --slug weimar [--themes 12] [--no-llm] [--out PATH]
    uv run grid-proposer show weimar
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from lib.config import settings
from lib.llm import LLMClient, LLMError
from lib.logging_setup import configure_logging, get_logger
from lib.project import (
    PROJECTS_DIR,
    ClassificationGuidance,
    CriticalMonth,
    EntityDefaults,
    EntityPrompts,
    Period,
    ReformulationPrompt,
    RelevancePrompt,
    ResearchProject,
    SeedEntity,
    Theme,
    dump_project,
    load_project,
    resolve_project_path,
)
from lib.sources import default_source_tiers

log = get_logger("grid_proposer")

_LLM_TEMPERATURE: float = 0.4
_LLM_NUM_PREDICT: int = 4096
_DEFAULT_THEMES: int = 12
_MIN_THEMES: int = 4
_MAX_THEMES: int = 24
_TIERS_BY_WEIGHT: tuple[tuple[float, str], ...] = (
    (0.95, "critica"),
    (0.7, "alta"),
    (0.4, "media"),
    (0.0, "baja"),
)


@dataclass(frozen=True)
class MomentBrief:
    """What the user tells us about the historical moment."""

    moment: str
    start_month: str
    end_month: str
    region: str
    language: str = "en"
    slug: str = ""
    n_themes: int = _DEFAULT_THEMES
    title: str = ""

    def resolved_slug(self) -> str:
        return self.slug or slugify(self.title or self.moment)

    def resolved_title(self) -> str:
        return self.title or self.moment.strip().rstrip(".")

    def years(self) -> tuple[int, int]:
        return int(self.start_month[:4]), int(self.end_month[:4])


# ── Pure helpers ───────────────────────────────────────────────────────────────


def slugify(text: str) -> str:
    """Lower-case ASCII slug: 'Weimar Berlin, 1929–33' -> 'weimar-berlin-1929-33'."""
    ascii_text = re.sub(r"[^a-z0-9]+", "-", text.lower()).encode("ascii", "ignore").decode()
    slug = ascii_text.strip("-")[:48].strip("-")
    return slug or "project"


def tier_for_weight(weight: float) -> str:
    """Map a 0-1 thematic weight onto the Propositor's four source tiers."""
    for threshold, tier in _TIERS_BY_WEIGHT:
        if weight >= threshold:
            return tier
    return "baja"


def extract_json_object(raw: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply (tolerates prose and fences)."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clamp(value: object, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _str_list(value: object, limit: int = 24) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:limit]


def normalise_themes(raw_themes: object, n_themes: int) -> list[Theme]:
    """Validate the LLM's theme list: re-number 1..n, clamp weights, derive tiers."""
    if not isinstance(raw_themes, list):
        return []
    themes: list[Theme] = []
    seen: set[str] = set()
    for item in raw_themes:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        weight = round(_clamp(item.get("weight"), 0.0, 1.0, 0.5), 2)
        tier = str(item.get("tier", "")).strip().lower()
        if tier not in {"critica", "alta", "media", "baja"}:
            tier = tier_for_weight(weight)
        themes.append(
            Theme(
                id=len(themes) + 1,
                name=name,
                weight=weight,
                tier=tier,  # type: ignore[arg-type]
                description=str(item.get("description", "")).strip(),
                search_terms=_str_list(item.get("search_terms")),
            )
        )
        if len(themes) >= n_themes:
            break
    return themes


def normalise_critical_months(raw: object, period: Period) -> list[CriticalMonth]:
    """Keep only well-formed months inside the period; derive short labels."""
    valid = set(period.months())
    out: list[CriticalMonth] = []
    seen: set[str] = set()
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        month = str(item.get("month_iso", "")).strip()[:7]
        event = str(item.get("event", "")).strip()
        if month not in valid or not event or month in seen:
            continue
        seen.add(month)
        label = str(item.get("label", "")).strip()[:6] or event.split()[0][:6]
        out.append(CriticalMonth(month_iso=month, event=event, label=label))
    return sorted(out, key=lambda cm: cm.month_iso)


def normalise_seeds(raw: object) -> list[SeedEntity]:
    """Characters, institutions and locations the pipeline should research first."""
    out: list[SeedEntity] = []
    if not isinstance(raw, dict):
        return out
    for kind in ("characters", "institutions", "locations"):
        for item in raw.get(kind, []) or []:
            if isinstance(item, str):
                item = {"name": item}
            if not isinstance(item, dict) or not str(item.get("name", "")).strip():
                continue
            out.append(
                SeedEntity(
                    name=str(item["name"]).strip(),
                    kind=kind[:-1],  # type: ignore[arg-type]
                    role=str(item.get("role", "")).strip(),
                    aliases=_str_list(item.get("aliases"), limit=6),
                )
            )
    return out


def skeleton_proposal(brief: MomentBrief) -> dict:
    """A generic, editable grid for ``--no-llm`` runs (no historical claims)."""
    generic = [
        (
            "Politics & Government",
            1.0,
            "state institutions, parties, elections, legislation, unrest",
        ),
        ("Economy", 1.0, "prices, employment, trade, fiscal and monetary policy"),
        ("Society & Daily Life", 0.8, "work, housing, family, consumption, public mood"),
        ("Military & Security", 0.8, "armed forces, police, violence, defence policy"),
        ("International Relations", 0.8, "foreign powers, diplomacy, treaties, blockades"),
        ("Industry & Labour", 0.8, "production, unions, strikes, enterprises"),
        ("Science & Technology", 0.5, "research, engineering, infrastructure, innovation"),
        ("Press & Media", 0.5, "newspapers, radio, propaganda, public debate"),
        ("Arts & Culture", 0.5, "literature, theatre, music, cinema, cultural policy"),
        ("Education", 0.5, "schools, universities, reform"),
        ("Health", 0.4, "public health, medicine, welfare"),
        ("Urbanism & Territory", 0.4, "cities, construction, transport, land"),
    ]
    start_year, end_year = brief.years()
    themes = [
        {
            "name": name,
            "weight": weight,
            "description": description,
            "search_terms": [
                f"{name.split(' &')[0].lower()} {brief.region} {start_year}",
                f"{brief.region} {end_year}",
            ],
        }
        for name, weight, description in generic[
            : max(_MIN_THEMES, min(brief.n_themes, len(generic)))
        ]
    ]
    return {
        "themes": themes,
        "critical_months": [],
        "seeds": {"characters": [], "institutions": [], "locations": []},
        "relevance": {
            "relevant_topics": [f"{brief.region} {start_year}-{end_year}: {brief.moment}"],
            "zero_score_rules": [
                f"Any document primarily about {brief.region} outside {start_year}-{end_year}",
                f"Any document not about {brief.region}",
            ],
        },
        "era_label": f"{brief.region} ({start_year}-{end_year})",
        "era_label_local": f"{brief.region} ({start_year}-{end_year})",
        "reading_rule_examples": [],
        "query_suffix": f"{brief.region} {start_year} {end_year}",
        "context_suffix": f"{brief.region} {brief.moment.split(',')[0][:40]}",
        "vocabulary_hints": [],
    }


def build_project(brief: MomentBrief, proposal: dict) -> ResearchProject:
    """Complete a (parsed) proposal into a valid ResearchProject.

    Everything the LLM did not or could not supply is filled deterministically:
    source tiers from the connector catalogue, entity defaults, prompt personas.
    Raises ValueError when the themes are unusable.
    """
    period = Period(start_month=brief.start_month, end_month=brief.end_month)
    themes = normalise_themes(proposal.get("themes"), brief.n_themes)
    if len(themes) < _MIN_THEMES:
        raise ValueError(f"proposal has {len(themes)} usable themes; need at least {_MIN_THEMES}")
    critical = normalise_critical_months(proposal.get("critical_months"), period)
    start_year, end_year = brief.years()
    relevance_raw = proposal.get("relevance") if isinstance(proposal.get("relevance"), dict) else {}
    era_label = str(
        proposal.get("era_label") or f"{brief.region} ({start_year}-{end_year})"
    ).strip()
    era_label_local = str(proposal.get("era_label_local") or era_label).strip()
    nominal_month = critical[0].month_iso if critical else period.middle_month()
    title = brief.resolved_title()
    region = brief.region
    examples = _str_list(proposal.get("reading_rule_examples"), limit=4)
    vocabulary = _str_list(proposal.get("vocabulary_hints"), limit=12)
    seeds = normalise_seeds(proposal.get("seeds"))

    # Provisional source routing; the Source Scout replaces it after probing.
    draft = ResearchProject.model_construct(
        language=brief.language, region=region, period=period, themes=themes
    )
    source_tiers = default_source_tiers(draft)
    nominal_theme = max(themes, key=lambda t: t.weight).id

    return ResearchProject(
        slug=brief.resolved_slug(),
        title=title,
        description=brief.moment.strip(),
        language=brief.language,
        region=region,
        period=period,
        themes=themes,
        critical_months=critical,
        source_tiers=source_tiers,
        entity=EntityDefaults(
            nominal_theme_id=nominal_theme,
            nominal_month_iso=nominal_month,
            sources=[s for s in source_tiers["alta"] if s != "web_serp"][:5],
            query_suffix=str(
                proposal.get("query_suffix") or f"{region} {start_year} {end_year}"
            ).strip(),
            context_suffix=str(proposal.get("context_suffix") or f"{region} {title}").strip(),
            biography_suffix=f"biography {region}",
        ),
        relevance=RelevancePrompt(
            domain=f"{title}: {region}, {period_label(period)}.",
            persona=(
                "You are a research archivist for a historical narrative-design project "
                f"studying {era_label}."
            ),
            relevant_topics=_str_list(relevance_raw.get("relevant_topics"), limit=10)
            or [t.name for t in themes[:6]],
            zero_score_rules=_str_list(relevance_raw.get("zero_score_rules"), limit=8)
            or [
                f"Any document primarily about {region} outside {start_year}-{end_year}",
                f"Any document not about {region}",
            ],
            score_scale=[
                "0.0 = wrong era, wrong country, or completely irrelevant",
                f"0.2 = mentions {region} but wrong time period or tangentially related",
                f"0.5 = {region} history broadly around {start_year}-{end_year}, adjacent topics",
                f"0.7 = directly about {era_label}",
                "0.9 = core topic of the research grid",
                "1.0 = primary source from the period",
            ],
        ),
        classification=ClassificationGuidance(
            persona=(
                "You are a research archivist for a historical narrative-design project. "
                f"Your task is to classify historical documents into a coverage matrix for {era_label}."
            ),
            reading_rule=(
                "Identify what the document IS, not what it MENTIONS. The event a document "
                "describes is its SETTING; the kind of act the document performs (a speech, a "
                "memo, a report) and the subject it is about decide the primary category."
            ),
            category_examples=examples,
            json_examples=[
                '{"category_ids": [1, 2, 3], "genre_id": 1, "month_iso": "'
                + nominal_month
                + '", "confidence": 0.9}  // a public speech on the main theme',
            ],
        ),
        reformulation=ReformulationPrompt(
            system=(
                f"You are a historian specialised in {era_label}. Rewrite search queries for "
                f"historical archives in {brief.language.upper()} using the vocabulary of the period"
                + (": " + ", ".join(vocabulary) if vocabulary else "")
                + ". Reply ONLY with a JSON array of strings, no explanation, no markdown."
            )
        ),
        entity_prompts=EntityPrompts(
            era_label=era_label,
            era_label_local=era_label_local,
            name_example=next((s.name for s in seeds if s.kind == "character"), ""),
            short_name_example=next(
                (s.name.split()[-1] for s in seeds if s.kind == "character"), ""
            ),
        ),
        seeds=seeds,
    )


def period_label(period: Period) -> str:
    """'January 1929 – March 1933' for prompts."""
    return ResearchProject.model_construct(period=period).period_label()


# ── LLM prompt ─────────────────────────────────────────────────────────────────

_LLM_SYSTEM = (
    "You are a senior historian and narrative designer. You design research grids: the "
    "themes, turning points and cast a team must document to simulate a historical moment "
    "faithfully in a game. You answer ONLY with one JSON object, no markdown, no prose."
)

_LLM_PROMPT = """\
Design the research grid for this historical moment.

MOMENT: {moment}
REGION: {region}
PERIOD: {start_month} to {end_month} (inclusive, ISO months)
SOURCE LANGUAGE: {language}
NUMBER OF THEMES: {n_themes}

Return a JSON object with exactly these keys:
{{
  "themes": [  // {n_themes} items, ordered by importance; these become the rows of a coverage matrix
    {{"name": "short theme name in {language}", "weight": 0.0-1.0,
      "description": "one line, in English, of what documents belong here",
      "search_terms": ["8-16 period-appropriate search queries in {language}, naming real institutions, laws, events, actors"]}}
  ],
  "critical_months": [  // 4-8 turning points inside the period
    {{"month_iso": "YYYY-MM", "event": "what happened", "label": "<=6 chars"}}
  ],
  "seeds": {{
    "characters": [{{"name": "full name", "role": "one line", "aliases": ["..."]}}],      // 8-15 key people
    "institutions": [{{"name": "...", "role": "one line", "aliases": ["acronym"]}}],       // 6-12 institutions
    "locations": [{{"name": "...", "role": "one line"}}]                                   // 6-12 places where events happen
  }},
  "relevance": {{
    "relevant_topics": ["6-8 lines naming the core topics, actors and events that make a document relevant"],
    "zero_score_rules": ["4-6 lines of content that must score 0.0: other periods, other regions, look-alike topics"]
  }},
  "era_label": "English label like 'Weimar Germany (1929-1933)'",
  "era_label_local": "the same label in {language}",
  "reading_rule_examples": ["2-3 worked examples: 'X document = [primary, secondary, tertiary theme ids], NOT [..]'"],
  "vocabulary_hints": ["8-12 period words, acronyms and proper nouns a search engine of the time would need"],
  "query_suffix": "words to append to a quoted person name when searching, e.g. '{region} {start_year} {end_year}'",
  "context_suffix": "an alternative suffix naming the regime or movement, e.g. '{region} Weimar Republic'"
}}

Weights: 1.0 for themes central to the moment, 0.3 for peripheral ones. Use real names
and real dates; never invent institutions. Keep every string short."""


class GridProposer:
    """Lifecycle of one proposal: brief -> LLM (or skeleton) -> validated spec -> YAML."""

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm

    async def propose(self, brief: MomentBrief) -> ResearchProject:
        """Return the completed project spec for the brief (does not write to disk)."""
        if not self.use_llm:
            log.info("grid_proposer.skeleton", slug=brief.resolved_slug())
            return build_project(brief, skeleton_proposal(brief))
        proposal = await self._ask_llm(brief)
        if proposal is None:
            log.warning("grid_proposer.llm_unusable_falling_back_to_skeleton")
            return build_project(brief, skeleton_proposal(brief))
        return build_project(brief, proposal)

    async def _ask_llm(self, brief: MomentBrief) -> dict | None:
        start_year, end_year = brief.years()
        prompt = _LLM_PROMPT.format(
            moment=brief.moment,
            region=brief.region,
            start_month=brief.start_month,
            end_month=brief.end_month,
            language=brief.language,
            n_themes=brief.n_themes,
            start_year=start_year,
            end_year=end_year,
        )
        try:
            async with LLMClient() as client:
                raw = await client.chat(
                    model=settings.ollama_model_director,
                    messages=[
                        {"role": "system", "content": _LLM_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=_LLM_TEMPERATURE,
                    num_predict=_LLM_NUM_PREDICT,
                    think=False,
                )
        except LLMError as exc:
            log.error("grid_proposer.llm_error", reason=str(exc)[:160])
            return None
        parsed = extract_json_object(raw)
        if parsed is None:
            log.error("grid_proposer.llm_not_json", head=raw[:120])
        return parsed

    def write(self, spec: ResearchProject, out: Path | None = None, *, force: bool = False) -> Path:
        """Dump the spec to ``projects/<slug>/project.yaml`` (or ``out``). Refuses to overwrite unless forced."""
        path = out or (PROJECTS_DIR / spec.slug / "project.yaml")
        if path.exists() and not force:
            raise FileExistsError(f"{path} exists — pass --force to overwrite")
        dump_project(spec, path)
        log.info(
            "grid_proposer.written",
            path=str(path),
            themes=len(spec.themes),
            months=len(spec.months()),
        )
        return path


# ── CLI ────────────────────────────────────────────────────────────────────────


def _print_summary(spec: ResearchProject) -> None:
    print(f"\n  {spec.title}  [{spec.slug}]")
    print(
        f"  {spec.region} · {spec.language} · {spec.period_label()} · {len(spec.months())} months"
    )
    print(f"  Matrix: {len(spec.themes)} themes × genres × {len(spec.months())} months\n")
    for theme in spec.themes:
        print(
            f"  {theme.id:>2}. {theme.name:<32} w={theme.weight:.2f} {theme.tier:<8} {len(theme.search_terms)} terms"
        )
    if spec.critical_months:
        print("\n  Critical months:")
        for cm in spec.critical_months:
            print(f"    {cm.month_iso}  {cm.event}")
    if spec.seeds:
        by_kind: dict[str, list[str]] = {}
        for seed in spec.seeds:
            by_kind.setdefault(seed.kind, []).append(seed.name)
        print("\n  Seeds:")
        for kind, names in by_kind.items():
            print(f"    {kind:<12} {', '.join(names[:8])}{' …' if len(names) > 8 else ''}")
    print("\n  Sources (critica):", ", ".join(spec.source_tiers["critica"]), "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="grid-proposer", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    propose = sub.add_parser("propose", help="Draft a project spec for a historical moment")
    propose.add_argument("--moment", required=True, help="One-paragraph description of the moment")
    propose.add_argument("--start", required=True, help="First month, YYYY-MM")
    propose.add_argument("--end", required=True, help="Last month, YYYY-MM")
    propose.add_argument("--region", required=True, help="Country / region keyword for queries")
    propose.add_argument("--language", default="en", help="ISO 639-1 language of the sources")
    propose.add_argument("--slug", default="", help="Project slug (default: from the title)")
    propose.add_argument("--title", default="", help="Project title (default: the moment)")
    propose.add_argument(
        "--themes", type=int, default=_DEFAULT_THEMES, help="Number of matrix themes"
    )
    propose.add_argument(
        "--no-llm", action="store_true", help="Write a generic skeleton grid instead"
    )
    propose.add_argument("--out", type=Path, default=None, help="Output YAML path")
    propose.add_argument("--force", action="store_true", help="Overwrite an existing spec")

    show = sub.add_parser("show", help="Print a project spec summary")
    show.add_argument("project", help="Slug or YAML path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(argv)
    if args.command == "show":
        _print_summary(load_project(resolve_project_path(args.project)))
        return 0

    n_themes = max(_MIN_THEMES, min(_MAX_THEMES, args.themes))
    brief = MomentBrief(
        moment=args.moment,
        start_month=args.start,
        end_month=args.end,
        region=args.region,
        language=args.language,
        slug=args.slug,
        n_themes=n_themes,
        title=args.title,
    )
    proposer = GridProposer(use_llm=not args.no_llm)
    try:
        spec = asyncio.run(proposer.propose(brief))
        path = proposer.write(spec, args.out, force=args.force)
    except (ValueError, FileExistsError) as exc:
        print(f"\n  ERROR: {exc}\n", file=sys.stderr)
        return 1
    _print_summary(spec)
    print(f"  Written to {path}")
    print(f"  Next: uv run source-scout run --project {spec.slug} --apply\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
