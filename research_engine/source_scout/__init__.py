"""
RE-0b Source Scout — finds which APIs actually have material for a project.

The connector catalogue (lib/sources.py) says which sources *could* fit a
historical moment. The Scout checks which ones *do*: for every applicable
connector it sends a few probe queries drawn from the project's own themes
through the Gatekeeper, runs the Archivero's extractors over the responses and
counts real hits. The report ranks connectors by yield and proposes the four
source tiers the Propositor routes missions with; ``--apply`` writes them into
the project YAML.

Operation:
1. Load the project spec; filter the catalogue by language / region / years.
2. Build probe queries per theme (first search terms + region + a critical year).
3. Fetch each probe via the Gatekeeper (``POST /fetch``); extract; count hits.
4. Rank, recommend tiers, print the report; optionally apply to the YAML.

Multi-step connectors (FRUS, FOIA, Marxists, Wikisource) are probed on their
first step only (search / index), which is enough to tell "reachable with hits"
from "dead". Runs fully offline in tests (respx on the Gatekeeper).

CLI:
    uv run source-scout run --project weimar [--probes 2] [--apply] [--gatekeeper URL]
    uv run source-scout catalogue [--project weimar]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from lib.config import settings
from lib.logging_setup import configure_logging, get_logger
from lib.project import ResearchProject, dump_project, load_project, resolve_project_path
from lib.sources import CATALOGUE, Applicable, applicable_connectors, excluded_connectors
from pipeline.archivero import _build_url, _extract, _wikisource_search_titles

log = get_logger("source_scout")

_FETCH_TIMEOUT_S: float = 45.0
_DEFAULT_PROBES: int = 2
_MAX_PROBE_THEMES: int = 4
_MIN_HITS_CRITICA: int = 3
_MIN_HITS_ALTA: int = 1


# ── Report model ───────────────────────────────────────────────────────────────


@dataclass
class ProbeResult:
    """One query against one connector."""

    source_id: str
    query: str
    ok: bool
    hits: int
    latency_ms: float
    error: str = ""


@dataclass
class SourceVerdict:
    """Aggregate for one connector across its probes."""

    source_id: str
    label: str
    kind: str
    access: str
    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def reachable(self) -> bool:
        return any(p.ok for p in self.probes)

    @property
    def hits(self) -> int:
        return sum(p.hits for p in self.probes)

    @property
    def hit_rate(self) -> float:
        return (
            (sum(1 for p in self.probes if p.hits > 0) / len(self.probes)) if self.probes else 0.0
        )

    @property
    def status(self) -> str:
        if not self.probes:
            return "skipped"
        if not self.reachable:
            return "unreachable"
        return "productive" if self.hits >= _MIN_HITS_ALTA else "empty"


@dataclass
class ScoutReport:
    """The Scout's output: verdicts, exclusions and the recommended tiers."""

    project_slug: str
    verdicts: list[SourceVerdict]
    excluded: list[Applicable]
    source_tiers: dict[str, list[str]]


# ── Pure functions ─────────────────────────────────────────────────────────────


def probe_queries(spec: ResearchProject, probes_per_source: int = _DEFAULT_PROBES) -> list[str]:
    """Pick a few representative queries from the heaviest themes.

    One query per theme (its first search term, else the theme name + region),
    plus one anchored on the first critical month's year; truncated to the
    requested number. Deterministic.
    """
    themes = sorted(spec.themes, key=lambda t: (-t.weight, t.id))[:_MAX_PROBE_THEMES]
    queries: list[str] = []
    for theme in themes:
        queries.append(
            theme.search_terms[0] if theme.search_terms else f"{theme.name} {spec.region}"
        )
    year = (
        spec.critical_months[0].month_iso[:4]
        if spec.critical_months
        else spec.period.start_month[:4]
    )
    queries.append(f"{spec.region} {year}")
    deduped: list[str] = []
    for query in queries:
        if query not in deduped:
            deduped.append(query)
    return deduped[: max(1, probes_per_source)]


def count_hits(source_id: str, body: str) -> int:
    """How many usable results a response body holds for a connector.

    JSON / HTML connectors go through the Archivero extractors; Wikisource's
    first step yields titles, so those are counted instead.
    """
    if source_id.startswith("wikisource_"):
        return len(_wikisource_search_titles(body))
    return len(_extract(body, source_id))


def recommend_tiers(verdicts: list[SourceVerdict]) -> dict[str, list[str]]:
    """Turn verdicts into the Propositor's four tiers.

    critica: every reachable connector, most productive first (web search last).
    alta:    productive connectors only.
    media:   the top-four productive, archives/reference preferred.
    baja:    the top-three cheapest (reference + academic) productive ones.
    """
    reachable = [v for v in verdicts if v.reachable]
    productive = sorted(
        (v for v in reachable if v.hits >= _MIN_HITS_ALTA),
        key=lambda v: (-v.hits, -v.hit_rate, v.source_id),
    )
    strong = [v for v in productive if v.hits >= _MIN_HITS_CRITICA]
    weak = [v for v in productive if v.hits < _MIN_HITS_CRITICA]
    others = [v for v in reachable if v not in productive]

    def ids(items: list[SourceVerdict]) -> list[str]:
        out = [v.source_id for v in items if v.source_id != "web_serp"]
        if any(v.source_id == "web_serp" for v in items):
            out.append("web_serp")
        return out

    critica = ids(strong + weak + others)
    alta = ids(productive)
    media = [s for s in ids(productive) if s != "web_serp"][:4]
    cheap_kinds = {"reference", "academic", "archive"}
    baja = [v.source_id for v in productive if v.kind in cheap_kinds and v.source_id != "web_serp"][
        :3
    ]
    return {"critica": critica, "alta": alta, "media": media, "baja": baja}


def apply_tiers(spec: ResearchProject, tiers: dict[str, list[str]]) -> ResearchProject:
    """Return a copy of the spec with the recommended routing (entity sources follow alta)."""
    entity = spec.entity.model_copy(
        update={"sources": [s for s in tiers["alta"] if s != "web_serp"][:5]}
    )
    return spec.model_copy(update={"source_tiers": tiers, "entity": entity})


# ── Scout ──────────────────────────────────────────────────────────────────────


class SourceScout:
    """Probe the applicable connectors for a project through the Gatekeeper."""

    def __init__(
        self, gatekeeper_url: str | None = None, probes_per_source: int = _DEFAULT_PROBES
    ) -> None:
        self.gatekeeper_url = (
            gatekeeper_url or f"http://{settings.gatekeeper_host}:{settings.gatekeeper_port}"
        )
        self.probes_per_source = probes_per_source

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> tuple[str | None, str]:
        """POST to the Gatekeeper; return (body, error)."""
        try:
            response = await client.post(
                f"{self.gatekeeper_url}/fetch",
                json={"url": url, "mission_id": "source-scout"},
                timeout=_FETCH_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            return None, f"gatekeeper unreachable: {exc.__class__.__name__}"
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"
        data = response.json()
        body = data.get("body")
        if not body:
            return None, data.get("error") or "empty body"
        return body, ""

    async def probe(
        self, applicable: Applicable, queries: list[str], client: httpx.AsyncClient
    ) -> SourceVerdict:
        """Run every query against one connector and aggregate."""
        connector = applicable.connector
        verdict = SourceVerdict(
            applicable.source_id, connector.label, connector.kind, connector.access
        )
        for query in queries:
            url = _build_url(applicable.source_id, query)
            if url is None:
                verdict.probes.append(
                    ProbeResult(applicable.source_id, query, False, 0, 0.0, "no URL builder")
                )
                continue
            started = time.monotonic()
            body, error = await self._fetch(client, url)
            latency = (time.monotonic() - started) * 1000
            if body is None:
                verdict.probes.append(
                    ProbeResult(applicable.source_id, query, False, 0, latency, error)
                )
                continue
            hits = count_hits(applicable.source_id, body)
            verdict.probes.append(ProbeResult(applicable.source_id, query, True, hits, latency))
        log.info(
            "source_scout.probed",
            source=applicable.source_id,
            status=verdict.status,
            hits=verdict.hits,
        )
        return verdict

    async def run(self, spec: ResearchProject) -> ScoutReport:
        """Probe every applicable connector and build the report."""
        queries = probe_queries(spec, self.probes_per_source)
        applicable = applicable_connectors(spec)
        log.info(
            "source_scout.start", project=spec.slug, connectors=len(applicable), probes=len(queries)
        )
        async with httpx.AsyncClient() as client:
            verdicts = [await self.probe(item, queries, client) for item in applicable]
        tiers = recommend_tiers(verdicts)
        return ScoutReport(spec.slug, verdicts, excluded_connectors(spec), tiers)


# ── CLI ────────────────────────────────────────────────────────────────────────


def _print_report(report: ScoutReport) -> None:
    print(f"\n  Source Scout — {report.project_slug}\n")
    print(f"  {'source':<22} {'kind':<11} {'status':<12} {'hits':>5}  {'rate':>5}  note")
    for v in sorted(report.verdicts, key=lambda v: (-v.hits, v.source_id)):
        note = next((p.error for p in v.probes if p.error), "")
        print(
            f"  {v.source_id:<22} {v.kind:<11} {v.status:<12} {v.hits:>5}  {v.hit_rate:>5.2f}  {note}"
        )
    if report.excluded:
        print("\n  Not applicable to this moment:")
        for item in report.excluded:
            print(f"    {item.connector.id:<22} {'; '.join(item.reasons)}")
    print("\n  Recommended tiers:")
    for tier, sources in report.source_tiers.items():
        print(f"    {tier:<8} {', '.join(sources) or '—'}")
    print()


def _print_catalogue(spec: ResearchProject | None) -> None:
    print(
        f"\n  {'connector':<22} {'kind':<11} {'access':<10} {'languages':<10} {'regions':<10} years"
    )
    for c in CATALOGUE:
        years = (
            f"{c.coverage_years[0] or ''}-{c.coverage_years[1] or ''}"
            if any(c.coverage_years)
            else "any"
        )
        print(
            f"  {c.id:<22} {c.kind:<11} {c.access:<10} {','.join(c.languages):<10} {','.join(c.regions):<10} {years}"
        )
    if spec is not None:
        ids = [a.source_id for a in applicable_connectors(spec)]
        print(f"\n  Applicable to {spec.slug}: {', '.join(ids)}\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="source-scout", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Probe the applicable connectors and recommend tiers")
    run.add_argument("--project", default=settings.research_project, help="Slug or YAML path")
    run.add_argument("--probes", type=int, default=_DEFAULT_PROBES, help="Queries per connector")
    run.add_argument(
        "--apply", action="store_true", help="Write the recommended tiers into the YAML"
    )
    run.add_argument("--gatekeeper", default=None, help="Gatekeeper base URL")
    cat = sub.add_parser("catalogue", help="List every connector the engine knows")
    cat.add_argument("--project", default=None, help="Also show which apply to this project")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(argv)
    if args.command == "catalogue":
        spec = load_project(resolve_project_path(args.project)) if args.project else None
        _print_catalogue(spec)
        return 0

    path: Path = resolve_project_path(args.project)
    spec = load_project(path)
    report = asyncio.run(SourceScout(args.gatekeeper, args.probes).run(spec))
    _print_report(report)
    if not any(v.reachable for v in report.verdicts):
        print(
            "  No connector reachable — is the Gatekeeper running? (make gatekeeper)\n",
            file=sys.stderr,
        )
        return 1
    if args.apply:
        dump_project(apply_tiers(spec, report.source_tiers), path)
        print(f"  Applied tiers to {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
