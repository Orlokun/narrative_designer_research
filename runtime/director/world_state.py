"""
WorldState: the single mutable object that flows through the runtime orchestrator.

Agents read from it and write to it. Nothing else crosses node boundaries.
Serializes cleanly to JSON via Pydantic, which is what enables checkpointing
and post-hoc replay of sessions for the experimental analysis.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from lib.schemas import Commitment, Indicator, Utterance


class WorldState(BaseModel):
    """Mutable state of a single game session."""
    model_config = ConfigDict(extra="forbid")

    # ---- Identity ----
    session_id: str
    scenario_id: str
    narrative_node_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    experiment_condition: str = Field(
        default="baseline",
        description="'baseline' or 'floresian' — which model config the NPCs use",
    )

    # ---- Time & turn ----
    turn: int = 0
    tick: int = 0
    game_time_minutes: float = 0.0

    # ---- Orchestrator routing ----
    current_node: str = "director"
    active_speaker: str | None = None
    pending_event_id: str | None = None
    finished: bool = False
    finished_reason: str | None = None

    # ---- World indicators (Cybersyn-style) ----
    indicators: dict[str, Indicator] = Field(default_factory=dict)
    tension: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Narrative pressure; Event Generator uses this",
    )

    # ---- Conversation ----
    log: list[Utterance] = Field(default_factory=list)
    commitments: dict[str, Commitment] = Field(default_factory=dict)

    # ---- NPCs present ----
    present_npcs: list[str] = Field(default_factory=list)

    # ---- Freeform scratch for Director reasoning ----
    director_notes: list[str] = Field(default_factory=list)

    # -----------------------------------------------------------------
    # Convenience accessors (read-only semantics)
    # -----------------------------------------------------------------
    def last_utterance(self) -> Utterance | None:
        return self.log[-1] if self.log else None

    def open_commitments(self) -> list[Commitment]:
        from lib.schemas import CommitmentStage
        closed = {CommitmentStage.CLOSED_SATISFIED, CommitmentStage.CLOSED_BROKEN}
        return [c for c in self.commitments.values() if c.stage not in closed]

    def indicator(self, key: str) -> float | None:
        ind = self.indicators.get(key)
        return ind.value if ind else None
