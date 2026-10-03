"""
Per-domain rate-limit config and runtime state for the Gatekeeper.

DomainRegistry is the single source of truth for all domain states.
Each domain gets an asyncio.Lock so concurrent /fetch requests to the
same domain are serialized and the rate limit is respected.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config (parsed from rate_limits.yaml)
# ---------------------------------------------------------------------------


class DomainRateConfig(BaseModel):
    """Rate limit config for one domain."""

    rps: float = Field(default=0.2, gt=0.0, description="Max requests per second")
    rpm: int = Field(default=10, gt=0, description="Informational; rps is enforced")


class GatekeeperYAMLConfig(BaseModel):
    """Top-level structure of rate_limits.yaml."""

    domains: dict[str, DomainRateConfig] = Field(default_factory=dict)
    default: DomainRateConfig = Field(default_factory=DomainRateConfig)
    circuit_breaker_threshold: int = Field(default=3, gt=0)
    cooldown_minutes: float = Field(default=15.0, gt=0.0)


# ---------------------------------------------------------------------------
# Runtime state (one per domain, lives in memory)
# ---------------------------------------------------------------------------

# 429 backoff: base pause, doubled per consecutive rate-limit, capped.
_RATE_LIMIT_BASE_BACKOFF_S: float = 60.0
_RATE_LIMIT_MAX_BACKOFF_S: float = 1800.0  # 30 min


@dataclass
class DomainState:
    min_interval: float  # seconds between requests = 1 / rps
    last_request_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_rate_limits: int = 0
    last_rate_limited_at: float = 0.0
    cooldown_until: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def is_in_cooldown(self) -> bool:
        return time.monotonic() < self.cooldown_until

    def cooldown_remaining(self) -> float:
        return max(0.0, self.cooldown_until - time.monotonic())

    def wait_seconds(self) -> float:
        """Seconds to sleep before the next request is allowed."""
        elapsed = time.monotonic() - self.last_request_time
        return max(0.0, self.min_interval - elapsed)

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.consecutive_rate_limits = 0
        self.last_rate_limited_at = 0.0
        self.total_requests += 1
        self.last_request_time = time.monotonic()

    def record_failure(self, threshold: int, cooldown_s: float) -> bool:
        """Increments failure counter. Returns True if circuit breaker just opened."""
        self.consecutive_failures += 1
        self.total_errors += 1
        self.last_request_time = time.monotonic()
        if self.consecutive_failures >= threshold:
            self.cooldown_until = time.monotonic() + cooldown_s
            self.consecutive_failures = 0
            return True
        return False

    def record_rate_limited(self, retry_after_s: float | None = None) -> float:
        """Source returned 429 — pause this domain immediately.

        Honors the source's own ``Retry-After`` instruction when given; otherwise
        applies an escalating backoff (60s doubled per consecutive 429, capped at
        30 min) so we never hammer a saturated shared pool. Returns the pause
        actually applied, in seconds.

        The escalation **decays with time already served**: each full base-backoff
        period elapsed since the previous 429 relaxes the counter by one step.
        Without this, a domain on a strict shared pool (e.g. unauthenticated
        Semantic Scholar) ratchets permanently to the 30-min cap — because the
        only other thing that resets escalation is a successful fetch, and the
        cooldown guard rejects every request *before* it can fetch. With the decay,
        the escalation converges to the base pause once the domain has waited out
        its penalties, and a single success still resets it entirely.
        """
        now = time.monotonic()
        if self.last_rate_limited_at > 0.0:
            decay_steps = int((now - self.last_rate_limited_at) // _RATE_LIMIT_BASE_BACKOFF_S)
            if decay_steps > 0:
                self.consecutive_rate_limits = max(0, self.consecutive_rate_limits - decay_steps)
        self.consecutive_rate_limits += 1
        escalated = min(
            _RATE_LIMIT_BASE_BACKOFF_S * (2 ** (self.consecutive_rate_limits - 1)),
            _RATE_LIMIT_MAX_BACKOFF_S,
        )
        # Honor the source's own Retry-After when it sends one (documented
        # etiquette); otherwise fall back to our escalating floor. Always capped.
        if retry_after_s and retry_after_s > 0:
            backoff_s = min(retry_after_s, _RATE_LIMIT_MAX_BACKOFF_S)
        else:
            backoff_s = escalated
        self.cooldown_until = now + backoff_s
        self.last_request_time = now
        self.last_rate_limited_at = now
        return backoff_s

    def to_dict(self, domain: str) -> dict[str, Any]:
        return {
            "domain": domain,
            "in_cooldown": self.is_in_cooldown(),
            "cooldown_remaining_s": round(self.cooldown_remaining(), 1),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_rate_limits": self.consecutive_rate_limits,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "min_interval_s": round(self.min_interval, 3),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class DomainRegistry:
    """
    Thread-safe manager of per-domain state objects.

    Use `await registry.get_state(domain)` to get the DomainState for a
    domain, creating it on first access. All actual request serialization
    happens through DomainState.lock, which callers must hold while fetching.
    """

    def __init__(self, config: GatekeeperYAMLConfig) -> None:
        self._config = config
        self._states: dict[str, DomainState] = {}
        self._registry_lock = asyncio.Lock()

    @classmethod
    def from_yaml(cls, path: Path) -> DomainRegistry:
        """Load from a rate_limits.yaml file. Returns a default config if file is absent."""
        if not path.exists():
            return cls(GatekeeperYAMLConfig())
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config = GatekeeperYAMLConfig.model_validate(raw)
        return cls(config)

    async def get_state(self, domain: str) -> DomainState:
        """Get or lazily create the DomainState for `domain`."""
        async with self._registry_lock:
            if domain not in self._states:
                cfg = self._config.domains.get(domain, self._config.default)
                self._states[domain] = DomainState(
                    min_interval=1.0 / cfg.rps,
                )
            return self._states[domain]

    def all_statuses(self, domain_filter: str | None = None) -> list[dict[str, Any]]:
        """Snapshot of all known domain states. Optionally filter by domain substring."""
        items = self._states.items()
        if domain_filter:
            items = ((d, s) for d, s in items if domain_filter in d)  # type: ignore[assignment]
        return [state.to_dict(domain) for domain, state in items]

    @property
    def circuit_breaker_threshold(self) -> int:
        return self._config.circuit_breaker_threshold

    @property
    def cooldown_seconds(self) -> float:
        return self._config.cooldown_minutes * 60.0

    def __len__(self) -> int:
        return len(self._config.domains)
