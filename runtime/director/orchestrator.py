"""
Custom orchestrator for the CyberSyn runtime.

Design tenets:
  1. One mutable state object (WorldState) flows through every node.
  2. Nodes are async callables registered by name; each mutates state and
     sets state.current_node to route to the next one.
  3. Coordination logic lives in explicit node code (especially Director),
     never inside a framework's implicit behavior.
  4. Everything observable (transitions, errors, checkpoints) goes through
     hooks so you can plug structlog, SQLite, or nothing at all.

Deliberately NOT using LangGraph/CrewAI/AutoGen. The loop below is the whole
orchestrator — read it top to bottom and you're done.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runtime.director.world_state import WorldState

# A node is an async function that mutates state in place.
Node = Callable[[WorldState], Awaitable[None]]

# Hook signatures
TransitionHook = Callable[["TransitionEvent"], Awaitable[None]]
ErrorHook = Callable[["ErrorEvent"], Awaitable[None]]
CheckpointHook = Callable[[WorldState], Awaitable[None]]


@dataclass
class TransitionEvent:
    step: int
    from_node: str
    to_node: str
    turn: int
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ErrorEvent:
    step: int
    node: str
    attempt: int
    exception: BaseException
    traceback_str: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class OrchestratorError(RuntimeError):
    """Raised when the orchestrator cannot recover (max retries, max steps)."""


class Orchestrator:
    """
    A minimal node-based orchestrator.

    Example:
        orch = Orchestrator(initial_state=ws, checkpoint_path=Path("session.json"))

        @orch.node("director")
        async def director(state): ...

        @orch.node("npc")
        async def npc(state): ...

        await orch.run()
    """

    def __init__(
        self,
        initial_state: WorldState,
        *,
        checkpoint_path: Path | None = None,
        max_steps: int = 1000,
        max_retries_per_node: int = 2,
        retry_backoff_s: float = 1.0,
    ) -> None:
        self.state = initial_state
        self.checkpoint_path = checkpoint_path
        self.max_steps = max_steps
        self.max_retries_per_node = max_retries_per_node
        self.retry_backoff_s = retry_backoff_s

        self._nodes: dict[str, Node] = {}
        self._transition_hooks: list[TransitionHook] = []
        self._error_hooks: list[ErrorHook] = []
        self._checkpoint_hooks: list[CheckpointHook] = []

    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------
    def node(self, name: str) -> Callable[[Node], Node]:
        """Decorator to register a node function."""
        def decorator(fn: Node) -> Node:
            if name in self._nodes:
                raise ValueError(f"Node '{name}' already registered")
            self._nodes[name] = fn
            return fn
        return decorator

    def register_node(self, name: str, fn: Node) -> None:
        """Non-decorator form; useful when nodes are created dynamically."""
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already registered")
        self._nodes[name] = fn

    def on_transition(self, hook: TransitionHook) -> TransitionHook:
        self._transition_hooks.append(hook)
        return hook

    def on_error(self, hook: ErrorHook) -> ErrorHook:
        self._error_hooks.append(hook)
        return hook

    def on_checkpoint(self, hook: CheckpointHook) -> CheckpointHook:
        self._checkpoint_hooks.append(hook)
        return hook

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------
    def registered_nodes(self) -> list[str]:
        return sorted(self._nodes.keys())

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    async def run(self) -> WorldState:
        """Run the session until state.finished is True or max_steps is hit."""
        loop = asyncio.get_running_loop()
        step = 0

        while not self.state.finished:
            if step >= self.max_steps:
                raise OrchestratorError(
                    f"Exceeded max_steps={self.max_steps} at node='{self.state.current_node}'"
                )

            node_name = self.state.current_node
            fn = self._nodes.get(node_name)
            if fn is None:
                raise OrchestratorError(
                    f"Unknown node '{node_name}'. Registered: {self.registered_nodes()}"
                )

            before = node_name
            t0 = loop.time()

            # Execute with retry
            attempt = 0
            while True:
                try:
                    await fn(self.state)
                    break
                except Exception as exc:  # noqa: BLE001 — broad is intentional at boundary
                    attempt += 1
                    err = ErrorEvent(
                        step=step,
                        node=node_name,
                        attempt=attempt,
                        exception=exc,
                        traceback_str=traceback.format_exc(),
                    )
                    await self._fire_error(err)
                    if attempt > self.max_retries_per_node:
                        raise OrchestratorError(
                            f"Node '{node_name}' failed after {attempt} attempts"
                        ) from exc
                    await asyncio.sleep(self.retry_backoff_s * attempt)

            dt_ms = (loop.time() - t0) * 1000.0
            step += 1

            await self._fire_transition(TransitionEvent(
                step=step,
                from_node=before,
                to_node=self.state.current_node,
                turn=self.state.turn,
                duration_ms=dt_ms,
            ))

            await self._checkpoint()

        return self.state

    # -----------------------------------------------------------------
    # Hook firing & checkpointing
    # -----------------------------------------------------------------
    async def _fire_transition(self, event: TransitionEvent) -> None:
        for hook in self._transition_hooks:
            await hook(event)

    async def _fire_error(self, event: ErrorEvent) -> None:
        for hook in self._error_hooks:
            await hook(event)

    async def _checkpoint(self) -> None:
        if self.checkpoint_path is not None:
            payload = self.state.model_dump_json(indent=2)
            # file IO is cheap here relative to LLM calls; sync is fine
            self.checkpoint_path.write_text(payload, encoding="utf-8")
        for hook in self._checkpoint_hooks:
            await hook(self.state)


# -----------------------------------------------------------------
# A couple of batteries-included hooks you'll probably want
# -----------------------------------------------------------------

async def print_transition(event: TransitionEvent) -> None:
    """Default transition hook: print a one-line trace. Replace with structlog."""
    print(
        f"step={event.step:04d}  {event.from_node:>20s} -> {event.to_node:<20s}"
        f"  turn={event.turn:03d}  {event.duration_ms:6.1f}ms"
    )


async def print_error(event: ErrorEvent) -> None:
    """Default error hook: print a compact failure line."""
    exc = event.exception
    print(
        f"ERROR step={event.step} node={event.node} attempt={event.attempt} "
        f"{type(exc).__name__}: {exc}"
    )


def jsonl_logger(path: Path) -> TransitionHook:
    """Factory: a transition hook that appends JSONL records to `path`."""
    async def _hook(event: TransitionEvent) -> None:
        record: dict[str, Any] = {
            "ts": event.timestamp.isoformat(),
            "step": event.step,
            "from": event.from_node,
            "to": event.to_node,
            "turn": event.turn,
            "duration_ms": round(event.duration_ms, 2),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    return _hook
