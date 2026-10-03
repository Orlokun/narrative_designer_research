"""
CyberSyn Research Gatekeeper — FastAPI microservice (port 8001).

Responsibilities:
  - Rate limiting per domain (minimum interval between requests)
  - Disk cache with configurable TTL (avoids re-downloading documents)
  - Circuit breaker (consecutive 5xx/network errors → cooldown)
  - 429 handling (source rate-limited us → immediate domain pause)
  - Structured fetch log (data/logs/fetch_log.jsonl)

All external HTTP traffic from the Research Engine passes through
POST /fetch. The Archivero never calls external URLs directly.

Run:
    uv run uvicorn research_engine.gatekeeper.app:app --port 8001 --reload
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import diskcache
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from lib.config import settings
from lib.logging_setup import get_logger
from lib.schemas import FetchRequest, FetchResponse
from research_engine.gatekeeper.domain_state import DomainRegistry

log = get_logger("gatekeeper")

# ---------------------------------------------------------------------------
# Module-level singletons (initialised in lifespan, available to endpoints)
# ---------------------------------------------------------------------------

_registry: DomainRegistry | None = None
_cache: diskcache.Cache | None = None


def _get_registry() -> DomainRegistry:
    assert _registry is not None, "Gatekeeper not started"
    return _registry


def _get_cache() -> diskcache.Cache:
    assert _cache is not None, "Gatekeeper not started"
    return _cache


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global _registry, _cache

    settings.ensure_dirs()
    settings.gatekeeper_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.gatekeeper_fetch_log.parent.mkdir(parents=True, exist_ok=True)

    rate_limits_path = settings.gatekeeper_rate_limits_path
    _registry = DomainRegistry.from_yaml(rate_limits_path)
    _cache = diskcache.Cache(str(settings.gatekeeper_cache_dir))

    log.info(
        "gatekeeper.started",
        port=settings.gatekeeper_port,
        rate_limits=str(rate_limits_path),
        cache_dir=str(settings.gatekeeper_cache_dir),
        cache_ttl_days=settings.gatekeeper_cache_ttl_days,
        known_domains=len(_registry),
    )
    yield

    _cache.close()
    log.info("gatekeeper.stopped")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CyberSyn Research Gatekeeper",
    description="Rate-limited HTTP proxy for the CyberSyn Research Engine.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    return urlparse(url).netloc


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header (seconds form). Returns None when absent/unparseable."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form — rare; fall back to escalating backoff


def _api_key_headers(domain: str) -> dict[str, str]:
    """Optional per-source API keys, read from settings (.env) — never hardcoded.

    Semantic Scholar grants a free key with a dedicated 1 rps pool; without it,
    requests share a global unauthenticated pool that 429s under load no matter
    how polite our own rate is.
    """
    if domain == "api.semanticscholar.org" and settings.semantic_scholar_api_key:
        return {"x-api-key": settings.semantic_scholar_api_key}
    return {}


def _is_binary(content_type: str) -> bool:
    ct = content_type.lower()
    return not (
        ct.startswith("text/")
        or "json" in ct
        or "xml" in ct
        or "javascript" in ct
    )


def _append_log_sync(path: Path, entry: dict) -> None:
    line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


async def _write_log(entry: dict) -> None:
    await asyncio.to_thread(_append_log_sync, settings.gatekeeper_fetch_log, entry)


def _cache_get(url: str) -> FetchResponse | None:
    raw = _get_cache().get(url)
    if raw is None:
        return None
    return FetchResponse.model_validate_json(raw)


def _cache_set(url: str, response: FetchResponse) -> None:
    ttl = settings.gatekeeper_cache_ttl_days * 86_400
    _get_cache().set(url, response.model_dump_json(), expire=ttl)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/fetch", response_model=FetchResponse, summary="Fetch a URL via the Gatekeeper")
async def fetch(req: FetchRequest) -> FetchResponse:
    """
    Fetch `url` with rate limiting and caching.

    - Returns cached response immediately if available (no rate-limit cost).
    - Enforces per-domain minimum interval between live requests.
    - Returns 503 if the domain is in circuit-breaker cooldown.
    - Returns 429 if the upstream source returned 429 (domain is then paused).
    """
    domain = _extract_domain(req.url)
    registry = _get_registry()

    # --- Cache check (no lock needed) ---
    cached = await asyncio.to_thread(_cache_get, req.url)
    if cached is not None:
        log.debug("gatekeeper.cache_hit", url=req.url, domain=domain)
        return cached.model_copy(update={"cached": True})

    # --- Acquire per-domain lock (serialises requests to same domain) ---
    state = await registry.get_state(domain)
    async with state.lock:

        # Double-check cache: another coroutine may have fetched while we waited
        cached = await asyncio.to_thread(_cache_get, req.url)
        if cached is not None:
            return cached.model_copy(update={"cached": True})

        # --- Circuit breaker ---
        if state.is_in_cooldown():
            remaining = state.cooldown_remaining()
            log.warning("gatekeeper.cooldown", domain=domain, remaining_s=round(remaining, 1))
            raise HTTPException(
                status_code=503,
                detail=f"Domain '{domain}' is in cooldown for {remaining:.0f}s more.",
            )

        # --- Rate limit delay ---
        wait = state.wait_seconds()
        if wait > 0:
            log.debug("gatekeeper.rate_limit_wait", domain=domain, wait_s=round(wait, 2))
            await asyncio.sleep(wait)

        # --- Live fetch ---
        t0 = time.monotonic()
        request_headers = dict(req.headers or {})
        request_headers.update(_api_key_headers(domain))
        try:
            async with httpx.AsyncClient(
                timeout=req.timeout_s,
                follow_redirects=True,
                headers={"User-Agent": "CyberSynResearchBot/0.1 (academic; oguerrerofarias@gmail.com)"},
            ) as client:
                resp = await client.get(req.url, headers=request_headers)
        except Exception as exc:
            tripped = state.record_failure(
                registry.circuit_breaker_threshold,
                registry.cooldown_seconds,
            )
            await _write_log({
                "timestamp": datetime.now(UTC).isoformat(),
                "url": req.url,
                "domain": domain,
                "mission_id": req.mission_id,
                "error": str(exc),
                "circuit_breaker_tripped": tripped,
            })
            log.error("gatekeeper.fetch_error", url=req.url, domain=domain, error=str(exc), tripped=tripped)
            raise HTTPException(status_code=502, detail=f"Network error fetching '{req.url}': {exc}") from exc

        duration_ms = (time.monotonic() - t0) * 1000

        # --- Handle upstream 429 ---
        if resp.status_code == 429:
            paused_s = state.record_rate_limited(_parse_retry_after(resp.headers.get("retry-after")))
            await _write_log({
                "timestamp": datetime.now(UTC).isoformat(),
                "url": req.url,
                "domain": domain,
                "mission_id": req.mission_id,
                "status_code": 429,
                "action": f"domain_paused_{int(paused_s)}s",
            })
            log.warning(
                "gatekeeper.source_rate_limited",
                domain=domain,
                paused_s=int(paused_s),
                consecutive=state.consecutive_rate_limits,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Source '{domain}' returned 429; domain paused for {int(paused_s)}s.",
            )

        # --- Handle upstream 5xx ---
        if resp.status_code >= 500:
            tripped = state.record_failure(
                registry.circuit_breaker_threshold,
                registry.cooldown_seconds,
            )
            await _write_log({
                "timestamp": datetime.now(UTC).isoformat(),
                "url": req.url,
                "domain": domain,
                "mission_id": req.mission_id,
                "status_code": resp.status_code,
                "circuit_breaker_tripped": tripped,
                "duration_ms": round(duration_ms, 1),
            })
            log.error("gatekeeper.upstream_error", domain=domain, status=resp.status_code, tripped=tripped)
            raise HTTPException(
                status_code=502,
                detail=f"Source '{domain}' returned {resp.status_code}.",
            )

        # --- Success ---
        state.record_success()

        content_type = resp.headers.get("content-type", "")
        binary = _is_binary(content_type)
        body = base64.b64encode(resp.content).decode() if binary else resp.text

        response = FetchResponse(
            url=req.url,
            status_code=resp.status_code,
            content_type=content_type,
            body=body,
            binary=binary,
            cached=False,
            fetched_at=datetime.now(UTC),
            mission_id=req.mission_id,
        )

        await asyncio.to_thread(_cache_set, req.url, response)

        await _write_log({
            "timestamp": response.fetched_at.isoformat(),
            "url": req.url,
            "domain": domain,
            "mission_id": req.mission_id,
            "status_code": resp.status_code,
            "content_type": content_type,
            "binary": binary,
            "cached": False,
            "duration_ms": round(duration_ms, 1),
        })

        log.info(
            "gatekeeper.fetched",
            url=req.url,
            domain=domain,
            status=resp.status_code,
            duration_ms=round(duration_ms, 1),
            binary=binary,
        )

        return response


@app.get("/status", summary="Circuit-breaker and rate-limit state per domain")
async def status(domain: str | None = Query(default=None, description="Filter by domain substring")) -> JSONResponse:
    """Returns current state for all domains seen since startup."""
    statuses = _get_registry().all_statuses(domain_filter=domain)
    return JSONResponse(content={"domains": statuses, "count": len(statuses)})


@app.get("/log", summary="Recent fetch log entries")
async def fetch_log(n: int = Query(default=50, ge=1, le=1000)) -> JSONResponse:
    """Returns the last `n` lines from fetch_log.jsonl."""
    log_path = settings.gatekeeper_fetch_log
    if not log_path.exists():
        return JSONResponse(content={"entries": [], "total": 0})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    recent = lines[-n:]
    entries = []
    for line in recent:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return JSONResponse(content={"entries": entries, "total": len(lines)})


@app.delete("/cache", summary="Clear disk cache (all or one domain)")
async def clear_cache(domain: str | None = Query(default=None, description="If set, only evict entries for this domain")) -> JSONResponse:
    """
    Clears cached responses. Use `?domain=memoriachilena.gob.cl` to be selective.
    Without a domain filter, clears everything.
    """
    cache = _get_cache()
    if domain is None:
        count = len(cache)
        await asyncio.to_thread(cache.clear)
        log.info("gatekeeper.cache_cleared", entries=count)
        return JSONResponse(content={"cleared": count})

    # Filter eviction: diskcache doesn't support prefix queries, so we iterate keys
    def _evict_domain(dc: diskcache.Cache, d: str) -> int:
        removed = 0
        for key in list(dc.iterkeys()):
            if isinstance(key, str) and d in key:
                dc.delete(key)
                removed += 1
        return removed

    removed = await asyncio.to_thread(_evict_domain, cache, domain)
    log.info("gatekeeper.cache_cleared_domain", domain=domain, entries=removed)
    return JSONResponse(content={"domain": domain, "cleared": removed})


@app.get("/health", summary="Liveness check")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok", "service": "gatekeeper"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    from lib.logging_setup import configure_logging
    configure_logging()
    uvicorn.run(
        "research_engine.gatekeeper.app:app",
        host=settings.gatekeeper_host,
        port=settings.gatekeeper_port,
        reload=False,
        log_config=None,  # let structlog handle it
    )


if __name__ == "__main__":
    main()
