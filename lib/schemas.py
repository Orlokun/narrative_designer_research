"""
Shared Pydantic schemas for the CyberSyn agent system.

Every agent speaks these types. If a type isn't here, it probably shouldn't
cross an agent boundary. Keep this file small and stable; breaking changes
here ripple everywhere.

Theoretical grounding:
    - SpeechAct follows Searle's taxonomy as adopted (and reinterpreted)
      by Flores 1982 and Flores & Winograd 1986.
    - CommitmentStage follows the Flores–Winograd ActionWorkflow loop
      (Preparation → Negotiation → Performance → Acceptance).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Floresian / Searle speech-act schema
# ---------------------------------------------------------------------------


class SpeechAct(str, Enum):
    """Five illocutionary act types, per Searle 1969, adopted by Flores 1982."""

    ASSERTIVE = "assertive"  # claims about the world (assertions, reports)
    DIRECTIVE = "directive"  # requests, orders, questions
    COMMISSIVE = "commissive"  # promises, offers, threats
    EXPRESSIVE = "expressive"  # thanks, apologies, congratulations
    DECLARATIVE = "declarative"  # pronouncements that change institutional reality


class CommitmentAction(str, Enum):
    """Moves inside the ActionWorkflow loop (Flores–Winograd 1986)."""

    REQUEST = "request"
    OFFER = "offer"
    PROMISE = "promise"
    COUNTER_OFFER = "counter_offer"
    DECLINE = "decline"
    CANCEL = "cancel"
    DECLARE_COMPLETE = "declare_complete"
    DECLARE_SATISFIED = "declare_satisfied"
    WITHDRAW = "withdraw"


class CommitmentStage(str, Enum):
    """Phases of a conversation-for-action (Flores–Winograd 1986, ch. 5)."""

    PREPARATION = "preparation"
    NEGOTIATION = "negotiation"
    PERFORMANCE = "performance"
    ACCEPTANCE = "acceptance"
    CLOSED_SATISFIED = "closed_satisfied"
    CLOSED_BROKEN = "closed_broken"


# ---------------------------------------------------------------------------
# Archive: canonical documents produced by the Canonicalizador (agent 2)
# ---------------------------------------------------------------------------


class SourceKind(str, Enum):
    PRESS = "press"  # newspaper article, editorial
    GOVERNMENT = "government"  # CORFO, ODEPLAN memos, decrees
    ACADEMIC = "academic"  # Flores, Medina, Beer papers/books
    ARCHIVE = "archive"  # Stafford Beer Collection items
    INTERVIEW = "interview"  # oral history
    CORRESPONDENCE = "correspondence"  # letters, telex
    OTHER = "other"


class Document(BaseModel):
    """A canonical entry in the historical archive."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    doc_id: str = Field(description="Stable content-addressed ID (sha256 prefix)")
    title: str
    text: str
    lang: str = Field(default="es", description="ISO 639-1")
    date_iso: str | None = Field(default=None, description="YYYY-MM-DD if known")
    authors: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    source_kind: SourceKind = SourceKind.OTHER
    source_id: str = Field(description="Which scraper/module produced this")
    provenance: dict[str, str] = Field(
        default_factory=dict,
        description="url, file_path, collection, folio — free-form but typed as strings",
    )
    rights: str | None = None
    sha256: str
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Cast: people extracted from documents (output of Cast Manager, agent 7)
# ---------------------------------------------------------------------------


class CharacterFact(BaseModel):
    """One dated fact about a character, drawn from a single document."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        default="other",
        description="role | event | affiliation | other | statement | rumor",
    )
    description: str
    date_iso: str | None = Field(default=None, description="YYYY-MM[-DD] if the text gives one")
    doc_id: str | None = Field(default=None, description="Source document the fact came from")
    speech_act: SpeechAct | None = Field(
        default=None, description="Searle/Flores act, for kind='statement'"
    )
    reported_by: str | None = Field(
        default=None, description="Who asserts/propagates it — required context for kind='rumor'"
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Extractor confidence; rumors are low"
    )


class CharacterMention(BaseModel):
    """A record that a character is mentioned in a document, and by whom."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    doc_id: str
    mentioned_by: str | None = Field(
        default=None, description="Speaker/source who mentions them, if the text says so"
    )


class Character(BaseModel):
    """A person extracted from the archive, with a timeline of facts.

    Produced by the Cast Manager (agent 7) and consumed by the Cast Director
    (agent 8) and the admin dashboard. ``character_id`` is a normalised slug of
    the canonical name, used as the de-duplication / coreference key.
    """

    model_config = ConfigDict(extra="forbid")

    character_id: str = Field(description="Normalised slug of the canonical name (dedup key)")
    name: str = Field(description="Canonical display name")
    aliases: list[str] = Field(
        default_factory=list, description="Alternate surface forms seen across documents"
    )
    biography: str | None = Field(
        default=None, description="<=1 paragraph summary; filled as evidence accrues"
    )
    completeness_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0-1: how much is known about this character"
    )
    completeness_detail: dict | None = Field(
        default=None, description="Per-component breakdown of the strict metric, incl. caps"
    )
    mention_count: int = Field(default=0, ge=0)
    timeline: list[CharacterFact] = Field(default_factory=list)
    needs_research: bool = Field(
        default=True, description="Signals the Propositor to mint an entity-driven mission"
    )
    run_id: str | None = Field(
        default=None, description="The pipeline run that first added this character"
    )


# ---------------------------------------------------------------------------
# Locations: places extracted from documents (output of Location Manager, Ag-4)
# ---------------------------------------------------------------------------


class LocationFact(BaseModel):
    """One fact about a place, drawn from a document or an external source."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(default="data", description="description | appreciation | event | data")
    detail: str
    date_iso: str | None = Field(default=None, description="YYYY[-MM[-DD]] if dated")
    doc_id: str | None = Field(default=None, description="Source document, if any")
    reported_by: str | None = Field(
        default=None, description="Who voices an appreciation, when stated"
    )


class Location(BaseModel):
    """A place extracted from the archive, with facts and external grounding.

    Produced by the Location Manager (Ag-4); ``location_id`` is a normalised
    slug of the canonical name (dedup key). Mirrors the ``locations`` table.
    """

    model_config = ConfigDict(extra="forbid")

    location_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    kind: str = Field(
        default="other",
        description="building | factory | residence | city | region | street | office | other",
    )
    description: str | None = Field(default=None, description="<=1 paragraph")
    latitude: float | None = None
    longitude: float | None = None
    character_id: str | None = Field(
        default=None, description="Character the place belongs to / is tied to"
    )
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_detail: dict | None = None
    mention_count: int = Field(default=0, ge=0)
    facts: list[LocationFact] = Field(default_factory=list)
    needs_research: bool = Field(
        default=True, description="Signals the Propositor to mint a location mission"
    )
    run_id: str | None = None


# ---------------------------------------------------------------------------
# Floresian dataset (output of Curador, agent 4; input to QLoRA)
# ---------------------------------------------------------------------------


class DialogueExample(BaseModel):
    """One annotated example for SFT / QLoRA fine-tuning."""

    example_id: str
    context: list[str] = Field(description="Preceding turns, most recent last")
    utterance: str = Field(description="The target utterance to learn")
    act: SpeechAct
    commitment_action: CommitmentAction | None = None
    commitment_stage: CommitmentStage | None = None
    source_doc_id: str | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# NPC personas (consumed by the NPC pool, agent 10)
# ---------------------------------------------------------------------------


class NPCPersona(BaseModel):
    """Prompt-level specification of a character."""

    npc_id: str
    display_name: str
    historical: bool = True
    role: str = Field(description="e.g., 'cybernetician', 'president', 'union leader'")
    biography: str = Field(description="Short prose bio used in system prompt")
    speech_style: str = Field(description="Idiolect notes; phrases to lean on")
    political_position: str | None = None
    relationships: dict[str, str] = Field(
        default_factory=dict,
        description="Map of other npc_id → relation descriptor",
    )
    rag_anchors: list[str] = Field(
        default_factory=list,
        description="doc_id list used to bias RAG for this character",
    )
    portrait_asset: str | None = Field(
        default=None,
        description="Unity asset path under Resources/Portraits",
    )


# ---------------------------------------------------------------------------
# Narrative graph (output of Diseñador Narrativo, agent 5)
# ---------------------------------------------------------------------------


class NarrativeBranch(BaseModel):
    target_node_id: str
    condition: str = Field(description="Natural-language condition; also see predicate")
    predicate: str | None = Field(
        default=None,
        description="Optional small expression language; evaluated over WorldState",
    )
    historical: bool = Field(
        default=True,
        description="False if this branch is counterfactual",
    )


class NarrativeNode(BaseModel):
    node_id: str
    title: str
    date_range: tuple[str, str] | None = None
    synopsis: str
    state_preconditions: dict[str, str] = Field(default_factory=dict)
    decision_points: list[str] = Field(default_factory=list)
    branches: list[NarrativeBranch] = Field(default_factory=list)
    sources: list[str] = Field(
        default_factory=list,
        description="doc_id citations backing this node",
    )


# ---------------------------------------------------------------------------
# Scenarios (output of Level Designer, agent 6)
# ---------------------------------------------------------------------------


class Indicator(BaseModel):
    """A Cybersyn-style indicator shown in the Operations Room."""

    key: str
    display_name: str
    value: float = Field(ge=0.0, le=1.0)
    trend: float = 0.0
    alert_threshold: float = 0.3


class Scenario(BaseModel):
    scenario_id: str
    narrative_node_id: str
    title: str
    intro_text: str
    initial_indicators: dict[str, Indicator]
    present_npcs: list[str] = Field(description="npc_id list")
    decision_points: list[str] = Field(default_factory=list)
    victory_condition: str = Field(
        description="Floresian framing, e.g., 'sustain coordination under the paro'",
    )
    defeat_condition: str
    time_budget_minutes: int = 30


# ---------------------------------------------------------------------------
# Game rules (output of Game Designer, agent 7)
# ---------------------------------------------------------------------------


class MetricSpec(BaseModel):
    key: str
    description: str
    unit: str = ""
    higher_is_better: bool | None = None


class GameRules(BaseModel):
    tick_seconds: int = 60
    indicator_decay_per_tick: float = 0.01
    metrics: list[MetricSpec] = Field(
        default_factory=lambda: [
            MetricSpec(
                key="agreement_rate",
                description="Fraction of directives followed by a commissive",
                higher_is_better=True,
            ),
            MetricSpec(
                key="time_to_resolution",
                description="Seconds from request opening to commitment closure",
                unit="s",
                higher_is_better=False,
            ),
            MetricSpec(
                key="commitment_fulfillment_ratio",
                description="Closed-satisfied / all opened",
                higher_is_better=True,
            ),
            MetricSpec(
                key="speech_act_distribution",
                description="Counts per SpeechAct value",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Conversation log primitives
# ---------------------------------------------------------------------------


class Utterance(BaseModel):
    turn: int
    session_id: str
    speaker: str = Field(description="'player' or npc_id")
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    act: SpeechAct | None = None
    commitment_action: CommitmentAction | None = None
    opens_commitment: str | None = None
    closes_commitment: str | None = None
    annotator: str | None = Field(
        default=None,
        description="'registrador' (live) or 'evaluador' (post-hoc)",
    )


class Commitment(BaseModel):
    commitment_id: str
    session_id: str
    speaker: str
    addressee: str
    content: str
    stage: CommitmentStage
    opened_at_turn: int
    opened_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    closed_at_turn: int | None = None
    closed_at: datetime | None = None
    opening_utterance: int = Field(description="Turn number of the opening utterance")
    related_utterances: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Event (Event Generator, agent 11)
# ---------------------------------------------------------------------------


class EventTemplate(BaseModel):
    event_id: str
    speaker_npc: str | None = Field(
        default=None,
        description="If set, event is delivered as an utterance by this NPC",
    )
    text: str
    triggers: list[str] = Field(
        default_factory=list,
        description="Human-readable trigger descriptions",
    )
    indicator_deltas: dict[str, float] = Field(default_factory=dict)
    tension_delta: float = 0.0


# ---------------------------------------------------------------------------
# Research Engine — Propositor (RE-1): missions
# ---------------------------------------------------------------------------


class MissionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class MissionKind(str, Enum):
    """What drives a mission: a coverage-matrix gap, a character, or a location."""

    GAP = "gap"
    ENTITY = "entity"
    LOCATION = "location"


class Mission(BaseModel):
    """A search mission generated by the Propositor and consumed by the Archivero."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    mission_id: str
    category: str
    category_id: int = Field(ge=1, le=16)
    month_iso: str = Field(description="YYYY-MM, within Oct 1969 – Sep 1973")
    priority: float = Field(ge=0.0, le=1.0)
    coverage_score_before: float = Field(ge=0.0, le=1.0)
    search_queries: list[str] = Field(min_length=1)
    target_sources: list[str] = Field(min_length=1)
    rationale: str
    deadline: str | None = None
    status: MissionStatus = MissionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    llm_reformulated: bool = False
    kind: MissionKind = MissionKind.GAP
    character_id: str | None = Field(
        default=None, description="Set on entity missions: the character being researched"
    )
    location_id: str | None = Field(
        default=None, description="Set on location missions: the place being researched"
    )
    genre_id: int | None = Field(
        default=None,
        ge=1,
        le=13,
        description="Genre cell targeted by gap missions (lib/genres.py); None on entity/legacy missions",
    )


# ---------------------------------------------------------------------------
# Research Engine — Gatekeeper (RE-GK)
# ---------------------------------------------------------------------------


class FetchRequest(BaseModel):
    """What the Archivero (or any caller) sends to POST /fetch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(description="Absolute URL to fetch")
    mission_id: str | None = Field(
        default=None, description="Propositor mission that triggered this fetch"
    )
    priority: float = Field(default=0.5, ge=0.0, le=1.0, description="0 = low, 1 = high")
    headers: dict[str, str] = Field(
        default_factory=dict, description="Extra HTTP headers for the request"
    )
    timeout_s: float = Field(default=30.0, gt=0.0)


class FetchResponse(BaseModel):
    """What the Gatekeeper returns from POST /fetch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    status_code: int
    content_type: str = ""
    body: str = Field(description="Text content, or base64-encoded bytes when binary=True")
    binary: bool = Field(default=False, description="True for PDFs, images, etc.")
    cached: bool = Field(description="True if served from disk cache")
    fetched_at: datetime
    mission_id: str | None = None


class DomainStatus(BaseModel):
    """Circuit-breaker + rate-limit state for one domain. Returned by GET /status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str
    in_cooldown: bool
    cooldown_remaining_s: float
    consecutive_failures: int
    total_requests: int
    total_errors: int
    min_interval_s: float
