"""
Tests for research_engine.gatekeeper.app

Strategy:
  - Use httpx.AsyncClient with ASGITransport to talk to the FastAPI app in-process.
  - Use respx to mock the outbound HTTP calls the Gatekeeper makes.
  - Each test gets a fresh DomainRegistry and a temp-dir diskcache via fixtures,
    so tests are fully isolated.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import diskcache
import pytest
import respx
from httpx import ASGITransport, AsyncClient, Response

import research_engine.gatekeeper.app as gk_app
from research_engine.gatekeeper.app import app
from research_engine.gatekeeper.domain_state import DomainRegistry, GatekeeperYAMLConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> diskcache.Cache:
    cache = diskcache.Cache(str(tmp_path / "cache"))
    yield cache
    cache.close()


@pytest.fixture()
def fresh_registry() -> DomainRegistry:
    """A registry with generous rate limits so tests don't sleep."""
    config = GatekeeperYAMLConfig.model_validate(
        {
            "default": {"rps": 100.0, "rpm": 6000},
            "circuit_breaker_threshold": 3,
            "cooldown_minutes": 0.1,  # 6-second cooldown for fast tests
        }
    )
    return DomainRegistry(config)


@pytest.fixture()
def tmp_log(tmp_path: Path) -> Path:
    return tmp_path / "fetch_log.jsonl"


@pytest.fixture()
async def client(fresh_registry, tmp_cache, tmp_log, tmp_path):
    """AsyncClient wired to the app with isolated state."""
    with (
        patch.object(gk_app, "_registry", fresh_registry),
        patch.object(gk_app, "_cache", tmp_cache),
        patch("lib.config.settings.gatekeeper_fetch_log", tmp_log),
        patch("lib.config.settings.gatekeeper_cache_ttl_days", 1),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


async def test_health(client: AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /fetch — success path
# ---------------------------------------------------------------------------


@respx.mock
async def test_fetch_html_success(client: AsyncClient, tmp_log: Path):
    url = "https://memoriachilena.gob.cl/test/doc"
    respx.get(url).mock(
        return_value=Response(
            200,
            text="<html><body>Memoria Chilena</body></html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )
    )

    r = await client.post("/fetch", json={"url": url, "mission_id": "m-001"})

    assert r.status_code == 200
    data = r.json()
    assert data["status_code"] == 200
    assert data["binary"] is False
    assert data["cached"] is False
    assert "Memoria Chilena" in data["body"]
    assert data["mission_id"] == "m-001"

    # Verify log entry was written
    assert tmp_log.exists()
    entry = json.loads(tmp_log.read_text().strip())
    assert entry["url"] == url
    assert entry["status_code"] == 200
    assert entry["mission_id"] == "m-001"


@respx.mock
async def test_fetch_pdf_is_binary(client: AsyncClient):
    url = "https://memoriachilena.gob.cl/doc.pdf"
    pdf_bytes = b"%PDF-1.4 fake pdf content"
    respx.get(url).mock(
        return_value=Response(
            200,
            content=pdf_bytes,
            headers={"content-type": "application/pdf"},
        )
    )

    r = await client.post("/fetch", json={"url": url})

    assert r.status_code == 200
    data = r.json()
    assert data["binary"] is True

    import base64

    assert base64.b64decode(data["body"]) == pdf_bytes


# ---------------------------------------------------------------------------
# /fetch — cache
# ---------------------------------------------------------------------------


@respx.mock
async def test_fetch_cache_hit_on_second_call(client: AsyncClient):
    url = "https://bndch.cl/revista/1972/01"
    respx.get(url).mock(
        return_value=Response(
            200,
            text="Article content",
            headers={"content-type": "text/html"},
        )
    )

    r1 = await client.post("/fetch", json={"url": url})
    assert r1.json()["cached"] is False

    # Second call — external mock should NOT be hit again
    r2 = await client.post("/fetch", json={"url": url})
    assert r2.status_code == 200
    assert r2.json()["cached"] is True
    assert r2.json()["body"] == r1.json()["body"]

    # respx should have been called exactly once
    assert respx.calls.call_count == 1


# ---------------------------------------------------------------------------
# /fetch — circuit breaker
# ---------------------------------------------------------------------------


@respx.mock
async def test_circuit_breaker_opens_after_threshold(
    client: AsyncClient, fresh_registry: DomainRegistry
):
    url = "https://jstor.org/stable/broken"
    # Simulate 3 consecutive 500s
    respx.get(url).mock(return_value=Response(500))

    for _ in range(3):
        r = await client.post("/fetch", json={"url": url})
        assert r.status_code == 502  # Gatekeeper wraps 5xx as 502

    # 4th call: domain is now in cooldown → 503
    r = await client.post("/fetch", json={"url": url})
    assert r.status_code == 503
    assert "cooldown" in r.json()["detail"].lower()


@respx.mock
async def test_circuit_breaker_opens_on_network_error(client: AsyncClient):
    url = "https://unreachable.example.com/page"
    respx.get(url).mock(side_effect=Exception("connection refused"))

    for _ in range(3):
        r = await client.post("/fetch", json={"url": url})
        assert r.status_code == 502

    r = await client.post("/fetch", json={"url": url})
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# /fetch — upstream 429
# ---------------------------------------------------------------------------


@respx.mock
async def test_upstream_429_pauses_domain(client: AsyncClient, fresh_registry: DomainRegistry):
    url = "https://jstor.org/page"
    respx.get(url).mock(return_value=Response(429))

    r = await client.post("/fetch", json={"url": url})
    assert r.status_code == 429

    # Domain is now paused — next request hits cooldown
    r2 = await client.post("/fetch", json={"url": url})
    assert r2.status_code == 503


# ---------------------------------------------------------------------------
# /fetch — 4xx passthrough (should NOT trigger circuit breaker)
# ---------------------------------------------------------------------------


@respx.mock
async def test_404_does_not_trip_circuit_breaker(client: AsyncClient):
    url = "https://memoriachilena.gob.cl/nonexistent"
    respx.get(url).mock(
        return_value=Response(
            404,
            text="Not found",
            headers={"content-type": "text/html"},
        )
    )

    # 404 is a valid response — should be passed through, not treated as failure
    for _ in range(5):
        r = await client.post("/fetch", json={"url": url})
        # Note: 404 is not 5xx so no circuit breaker; but Gatekeeper returns it as-is
        # The Gatekeeper currently only gates on 5xx and network errors.
        assert r.status_code == 200  # Gatekeeper returns 200 with the upstream status in body
        assert r.json()["status_code"] == 404

    # Domain should NOT be in cooldown
    status_r = await client.get("/status")
    domain_info = next(
        (d for d in status_r.json()["domains"] if "memoriachilena" in d["domain"]),
        None,
    )
    if domain_info:
        assert not domain_info["in_cooldown"]


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


@respx.mock
async def test_status_after_fetch(client: AsyncClient):
    url = "https://archive.org/details/doc1"
    respx.get(url).mock(
        return_value=Response(200, text="ok", headers={"content-type": "text/plain"})
    )

    await client.post("/fetch", json={"url": url})

    r = await client.get("/status")
    assert r.status_code == 200
    domains = r.json()["domains"]
    archive_status = next((d for d in domains if "archive.org" in d["domain"]), None)
    assert archive_status is not None
    assert archive_status["total_requests"] == 1
    assert not archive_status["in_cooldown"]


# ---------------------------------------------------------------------------
# /log
# ---------------------------------------------------------------------------


@respx.mock
async def test_log_endpoint(client: AsyncClient):
    url = "https://bndch.cl/article/42"
    respx.get(url).mock(
        return_value=Response(200, text="content", headers={"content-type": "text/html"})
    )

    await client.post("/fetch", json={"url": url, "mission_id": "m-007"})

    r = await client.get("/log?n=10")
    assert r.status_code == 200
    entries = r.json()["entries"]
    assert len(entries) >= 1
    assert entries[-1]["url"] == url
    assert entries[-1]["mission_id"] == "m-007"


# ---------------------------------------------------------------------------
# /cache DELETE
# ---------------------------------------------------------------------------


@respx.mock
async def test_cache_clear_all(client: AsyncClient):
    url = "https://memoriachilena.gob.cl/doc/clear"
    respx.get(url).mock(
        return_value=Response(200, text="data", headers={"content-type": "text/html"})
    )

    await client.post("/fetch", json={"url": url})

    r = await client.delete("/cache")
    assert r.status_code == 200
    assert r.json()["cleared"] >= 1

    # After clearing, next fetch should hit network again
    r2 = await client.post("/fetch", json={"url": url})
    assert r2.json()["cached"] is False
    assert respx.calls.call_count == 2


# ---------------------------------------------------------------------------
# 429 handling — Retry-After + escalating backoff (Semantic Scholar etiquette)
# ---------------------------------------------------------------------------


@respx.mock
async def test_429_honors_retry_after_header(client: AsyncClient, fresh_registry: DomainRegistry):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?query=x"
    respx.get(url).mock(return_value=Response(429, headers={"Retry-After": "120"}))

    r = await client.post("/fetch", json={"url": url})
    assert r.status_code == 429

    state = await fresh_registry.get_state("api.semanticscholar.org")
    # The pause must follow the source's own instruction (120s), not the 60s default.
    assert 110 < state.cooldown_remaining() <= 120


@respx.mock
async def test_429_backoff_escalates_on_consecutive_hits(
    client: AsyncClient, fresh_registry: DomainRegistry
):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?query=y"
    respx.get(url).mock(return_value=Response(429))

    await client.post("/fetch", json={"url": url})
    state = await fresh_registry.get_state("api.semanticscholar.org")
    first = state.cooldown_remaining()
    assert 50 < first <= 60  # base backoff

    state.cooldown_until = 0.0  # simulate the pause elapsing
    await client.post("/fetch", json={"url": url})
    second = state.cooldown_remaining()
    assert second > first  # 2nd consecutive 429 → longer pause (120s)
    assert 110 < second <= 120


def test_rate_limit_backoff_caps_and_resets():
    from research_engine.gatekeeper.domain_state import DomainState

    state = DomainState(min_interval=0.01)
    for _ in range(10):
        state.cooldown_until = 0.0
        state.record_rate_limited()
    # Doubling stops at the 30-minute cap.
    assert state.cooldown_remaining() <= 1800.0
    assert state.cooldown_remaining() > 1700.0

    state.record_success()
    state.cooldown_until = 0.0
    state.record_rate_limited()
    assert state.cooldown_remaining() <= 60.0  # success resets the escalation


def test_rate_limit_escalation_decays_with_time_served(monkeypatch):
    """A domain that waits out its penalties must not ratchet to the cap forever.

    Regression for the Semantic Scholar 'permanent 30-min cooldown': the only
    non-time reset was a successful fetch, which the cooldown guard prevents.
    """
    from research_engine.gatekeeper import domain_state as ds
    from research_engine.gatekeeper.domain_state import DomainState

    state = DomainState(min_interval=0.01)
    # Climb to the cap.
    for _ in range(8):
        state.cooldown_until = 0.0
        state.record_rate_limited()
    assert state.cooldown_remaining() > 1700.0  # pinned at the 30-min cap

    # Simulate the domain having served a long penalty: >1 base period elapsed
    # since the last 429. Advance the clock instead of rewinding the stamp —
    # on a freshly-booted machine time.monotonic() is small, and a rewound
    # stamp would go negative and read as "never rate limited".
    frozen_now = time.monotonic() + 20 * ds._RATE_LIMIT_BASE_BACKOFF_S
    monkeypatch.setattr(ds.time, "monotonic", lambda: frozen_now)
    state.cooldown_until = 0.0
    pause = state.record_rate_limited()
    assert pause <= 60.0  # decayed back to the base pause


def test_rate_limit_prefers_short_retry_after_over_escalation():
    """A polite short Retry-After must be honored even mid-escalation, not floored."""
    from research_engine.gatekeeper.domain_state import DomainState

    state = DomainState(min_interval=0.01)
    for _ in range(5):  # escalate well past 30s
        state.cooldown_until = 0.0
        state.record_rate_limited()
    assert state.cooldown_remaining() > 200.0

    state.cooldown_until = 0.0
    pause = state.record_rate_limited(retry_after_s=30.0)
    assert pause == 30.0  # source's shorter hint wins over the escalated floor


# ---------------------------------------------------------------------------
# Semantic Scholar API key injection (free key → dedicated pool, no 429s)
# ---------------------------------------------------------------------------


@respx.mock
async def test_semantic_scholar_key_injected_when_configured(client: AsyncClient):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?query=z"
    route = respx.get(url).mock(
        return_value=Response(
            200,
            text='{"data": []}',
            headers={"content-type": "application/json"},
        )
    )
    with patch("lib.config.settings.semantic_scholar_api_key", "test-key-123"):
        r = await client.post("/fetch", json={"url": url})
    assert r.status_code == 200
    assert route.calls.last.request.headers.get("x-api-key") == "test-key-123"


@respx.mock
async def test_no_key_header_without_configuration(client: AsyncClient):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?query=w"
    route = respx.get(url).mock(
        return_value=Response(
            200,
            text='{"data": []}',
            headers={"content-type": "application/json"},
        )
    )
    with patch("lib.config.settings.semantic_scholar_api_key", None):
        r = await client.post("/fetch", json={"url": url})
    assert r.status_code == 200
    assert "x-api-key" not in route.calls.last.request.headers
