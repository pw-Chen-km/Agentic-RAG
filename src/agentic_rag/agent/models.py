"""Stable contracts for query-time agent decisions, state, and trajectories."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class AgentModel(BaseModel):
    """Strict base model used by all persisted agent records."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class AssessmentStatus(StrEnum):
    INSUFFICIENT = "INSUFFICIENT"
    SUFFICIENT = "SUFFICIENT"
    UNCERTAIN = "UNCERTAIN"


class ContextMode(StrEnum):
    COMPACT_EVIDENCE = "compact_evidence"
    APPEND_ONLY = "append_only"


class SearchMethod(StrEnum):
    LEXICAL = "LEXICAL"
    BM25 = "BM25"
    DENSE = "DENSE"


class SearchTarget(StrEnum):
    ENTITY = "ENTITY"
    SENTENCE = "SENTENCE"
    CHUNK = "CHUNK"


VALID_SEARCH_PAIRS: frozenset[tuple[SearchMethod, SearchTarget]] = frozenset(
    {
        (SearchMethod.LEXICAL, SearchTarget.ENTITY),
        (SearchMethod.BM25, SearchTarget.SENTENCE),
        (SearchMethod.BM25, SearchTarget.CHUNK),
        (SearchMethod.DENSE, SearchTarget.ENTITY),
        (SearchMethod.DENSE, SearchTarget.SENTENCE),
        (SearchMethod.DENSE, SearchTarget.CHUNK),
    }
)


class ExpansionKind(StrEnum):
    ENTITY_MENTIONED_IN_SENTENCE = "ENTITY_MENTIONED_IN_SENTENCE"
    SENTENCE_MENTIONS_ENTITY = "SENTENCE_MENTIONS_ENTITY"
    ENTITY_CO_OCCURS_ENTITY_SENTENCE = "ENTITY_CO_OCCURS_ENTITY_SENTENCE"
    CHUNK_ADJACENT_CHUNK = "CHUNK_ADJACENT_CHUNK"
    ENTITY_MENTIONED_IN_CHUNK = "ENTITY_MENTIONED_IN_CHUNK"
    ENTITY_CO_OCCURS_ENTITY_CHUNK = "ENTITY_CO_OCCURS_ENTITY_CHUNK"
    CHUNK_CONTAINS_SENTENCE = "CHUNK_CONTAINS_SENTENCE"
    CHUNK_MENTIONS_ENTITY = "CHUNK_MENTIONS_ENTITY"


DEFAULT_ENABLED_EXPANSIONS: tuple[ExpansionKind, ...] = (
    ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
    ExpansionKind.SENTENCE_MENTIONS_ENTITY,
    ExpansionKind.ENTITY_CO_OCCURS_ENTITY_SENTENCE,
    ExpansionKind.CHUNK_ADJACENT_CHUNK,
)


class ExpansionDirection(StrEnum):
    PREV = "PREV"
    NEXT = "NEXT"
    BOTH = "BOTH"


class SentenceRef(AgentModel):
    unit: Literal["SENTENCE"] = "SENTENCE"
    id: str = Field(min_length=1)


class ChunkRef(AgentModel):
    unit: Literal["CHUNK"] = "CHUNK"
    id: str = Field(min_length=1)


EvidenceRef = Annotated[SentenceRef | ChunkRef, Field(discriminator="unit")]


class EvidenceAssessment(AgentModel):
    status: AssessmentStatus
    supported_facts: list[str] = Field(default_factory=list, max_length=5)
    missing_information: list[str] = Field(default_factory=list, max_length=3)
    selected_evidence_refs: list[EvidenceRef] = Field(
        default_factory=list, max_length=20
    )

    @model_validator(mode="after")
    def selected_evidence_refs_must_be_unique(self) -> Self:
        keys = [(ref.unit, ref.id) for ref in self.selected_evidence_refs]
        if len(keys) != len(set(keys)):
            raise ValueError("selected_evidence_refs must not contain duplicates")
        return self


class SearchAction(AgentModel):
    type: Literal["SEARCH"] = "SEARCH"
    query: str = Field(min_length=1)
    method: SearchMethod
    target: SearchTarget
    top_k: Literal[5] = 5

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def validate_method_target(self) -> Self:
        if (self.method, self.target) not in VALID_SEARCH_PAIRS:
            raise ValueError(
                f"unsupported retrieval pair: {self.method.value} → "
                f"{self.target.value}"
            )
        return self


class ExpandAction(AgentModel):
    type: Literal["EXPAND"] = "EXPAND"
    kind: ExpansionKind
    source_id: str = Field(min_length=1)
    direction: ExpansionDirection | None = None
    query: str | None = None
    top_k: Literal[5] = 5

    @field_validator("source_id")
    @classmethod
    def source_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_id must contain non-whitespace characters")
        return value

    @field_validator("query")
    @classmethod
    def optional_query_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("query must be omitted rather than blank")
        return value

    @model_validator(mode="after")
    def validate_direction(self) -> Self:
        is_adjacent = (
            self.kind.value == ExpansionKind.CHUNK_ADJACENT_CHUNK.value
        )
        if is_adjacent and self.direction is None:
            raise ValueError("CHUNK_ADJACENT_CHUNK requires direction")
        if not is_adjacent and self.direction is not None:
            raise ValueError("direction is only valid for CHUNK_ADJACENT_CHUNK")
        return self


class ReadAction(AgentModel):
    type: Literal["READ"] = "READ"
    chunk_id: str = Field(min_length=1)

    @field_validator("chunk_id")
    @classmethod
    def chunk_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk_id must contain non-whitespace characters")
        return value


class FinishAction(AgentModel):
    type: Literal["FINISH"] = "FINISH"
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def evidence_refs_must_be_unique(self) -> Self:
        keys = [(ref.unit, ref.id) for ref in self.evidence_refs]
        if len(keys) != len(set(keys)):
            raise ValueError("evidence_refs must not contain duplicates")
        return self


AgentAction = Annotated[
    SearchAction | ExpandAction | ReadAction | FinishAction,
    Field(discriminator="type"),
]


class PolicyDecision(AgentModel):
    assessment: EvidenceAssessment
    action: AgentAction


class ObservationStatus(StrEnum):
    OK = "ok"
    INVALID_ACTION = "invalid_action"
    DUPLICATE_ACTION = "duplicate_action"
    ERROR = "error"


class Observation(AgentModel):
    action_id: str | None = None
    status: ObservationStatus
    action: AgentAction | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_tokens: int = Field(default=0, ge=0)
    novel_node_ids: list[str] = Field(default_factory=list)
    already_seen_node_ids: list[str] = Field(default_factory=list)
    error_code: str | None = None
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(AgentModel):
    policy_calls: int = Field(default=0, ge=0)
    answer_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    retrieved_tokens: int = Field(default=0, ge=0)
    policy_input_tokens: int = Field(default=0, ge=0)
    policy_output_tokens: int = Field(default=0, ge=0)
    policy_reasoning_tokens: int = Field(default=0, ge=0)
    answer_input_tokens: int = Field(default=0, ge=0)
    answer_output_tokens: int = Field(default=0, ge=0)
    answer_reasoning_tokens: int = Field(default=0, ge=0)

    def __add__(self, other: object) -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            policy_calls=self.policy_calls + other.policy_calls,
            answer_calls=self.answer_calls + other.answer_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            retrieved_tokens=self.retrieved_tokens + other.retrieved_tokens,
            policy_input_tokens=(
                self.policy_input_tokens + other.policy_input_tokens
            ),
            policy_output_tokens=(
                self.policy_output_tokens + other.policy_output_tokens
            ),
            policy_reasoning_tokens=(
                self.policy_reasoning_tokens
                + other.policy_reasoning_tokens
            ),
            answer_input_tokens=(
                self.answer_input_tokens + other.answer_input_tokens
            ),
            answer_output_tokens=(
                self.answer_output_tokens + other.answer_output_tokens
            ),
            answer_reasoning_tokens=(
                self.answer_reasoning_tokens
                + other.answer_reasoning_tokens
            ),
        )


class HandleSentencePreview(AgentModel):
    sentence_id: str = Field(min_length=1)
    text: str


class EntityHandle(AgentModel):
    node_type: Literal["ENTITY"] = "ENTITY"
    id: str = Field(min_length=1)
    label: str
    entity_type: str | None = None


class SentenceHandle(AgentModel):
    node_type: Literal["SENTENCE"] = "SENTENCE"
    id: str = Field(min_length=1)
    text: str
    parent_chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str | None = None
    can_use_as_evidence: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "can_use_as_evidence",
            "eligible",
        ),
    )

    @property
    def eligible(self) -> bool:
        """Backward-compatible accessor for pre-annotation callers."""

        return self.can_use_as_evidence


class ChunkHandle(AgentModel):
    node_type: Literal["CHUNK"] = "CHUNK"
    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str | None = None
    read: bool = False
    can_use_as_evidence: bool = False
    previews: list[HandleSentencePreview] = Field(
        default_factory=list, max_length=2
    )


NodeHandle = Annotated[
    EntityHandle | SentenceHandle | ChunkHandle,
    Field(discriminator="node_type"),
]


class ControllerState(AgentModel):
    step: int = Field(default=0, ge=0)
    visible_entity_ids: set[str] = Field(default_factory=set)
    visible_sentence_ids: set[str] = Field(default_factory=set)
    visible_chunk_ids: set[str] = Field(default_factory=set)
    eligible_sentence_ids: set[str] = Field(default_factory=set)
    read_chunk_ids: set[str] = Field(default_factory=set)
    action_signatures: set[str] = Field(default_factory=set)
    remaining_step_budget: int = Field(ge=0)
    remaining_retrieved_token_budget: int = Field(ge=0)
    last_assessment: EvidenceAssessment | None = None
    newest_observation: Observation | None = None
    selected_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    node_handles: dict[str, NodeHandle] = Field(default_factory=dict)

    @classmethod
    def initial(
        cls, *, max_steps: int = 10, max_retrieved_tokens: int = 12_000
    ) -> "ControllerState":
        return cls(
            remaining_step_budget=max_steps,
            remaining_retrieved_token_budget=max_retrieved_tokens,
        )

    @field_serializer(
        "visible_entity_ids",
        "visible_sentence_ids",
        "visible_chunk_ids",
        "eligible_sentence_ids",
        "read_chunk_ids",
        "action_signatures",
    )
    def serialize_sets(self, values: set[str]) -> list[str]:
        return sorted(values)


class ValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    POLICY_ERROR = "policy_error"


class ResolvedEvidence(AgentModel):
    ref: EvidenceRef
    text: str
    document_id: str
    title: str | None = None
    parent_chunk_id: str | None = None
    contained_sentence_ids: list[str] = Field(default_factory=list)


class LastValidAssessmentView(AgentModel):
    status: AssessmentStatus
    missing_information: list[str] = Field(default_factory=list, max_length=3)


class AttemptedActionView(AgentModel):
    step: int = Field(ge=1)
    action: AgentAction | None = None
    validation_status: ValidationStatus
    observation_status: ObservationStatus | None = None
    error_code: str | None = None
    message: str | None = None
    new_node_count: int = Field(default=0, ge=0)
    retrieved_tokens: int = Field(default=0, ge=0)


class PolicyBudgetView(AgentModel):
    remaining_steps: int = Field(ge=0)
    remaining_retrieved_tokens: int = Field(ge=0)


class PolicyStateView(AgentModel):
    step: int = Field(ge=0)
    last_valid_assessment: LastValidAssessmentView | None = None
    selected_evidence: list[ResolvedEvidence] = Field(default_factory=list)
    latest_observation: dict[str, Any] | None = None
    actionable_handles: list[NodeHandle] = Field(default_factory=list)
    attempted_actions: list[AttemptedActionView] = Field(default_factory=list)
    budget: PolicyBudgetView


class PolicyView(AgentModel):
    context_mode: ContextMode
    instruction: Literal["Produce the next PolicyDecision."] = (
        "Produce the next PolicyDecision."
    )
    policy_state: PolicyStateView


class StepRecord(AgentModel):
    step: int = Field(ge=1)
    decision: PolicyDecision | None = None
    validation_status: ValidationStatus
    validation_error: str | None = None
    observation: Observation | None = None
    state_before: ControllerState
    state_after: ControllerState
    usage: Usage = Field(default_factory=Usage)
    policy_view: PolicyView | None = None


class TerminationReason(StrEnum):
    FINISH = "finish"
    BUDGET_EXHAUSTED = "budget_exhausted"
    POLICY_ERROR = "policy_error"
    ANSWER_GENERATION_ERROR = "answer_generation_error"
    RUNTIME_ERROR = "runtime_error"


class EpisodeResult(AgentModel):
    episode_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    scope_id: str = Field(min_length=1)
    termination_reason: TerminationReason
    answer: str | None = None
    selected_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    resolved_evidence: list[ResolvedEvidence] = Field(default_factory=list)
    trajectory: list[StepRecord] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    artifact_dir: str | None = None
    final_state: ControllerState | None = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def question(self) -> str:
        return self.query

    @property
    def selected_refs(self) -> list[EvidenceRef]:
        return self.selected_evidence_refs


class Message(AgentModel):
    role: Literal["system", "user", "assistant"]
    content: str

    def as_openai_input(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def action_signature(action: AgentAction) -> str:
    """Return a deterministic signature used for duplicate-action detection."""

    payload = action.model_dump(mode="json", exclude_none=True)
    query = payload.get("query")
    if isinstance(query, str):
        payload["query"] = " ".join(query.casefold().split())
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Compatibility alias for callers that prefer an explicit name.
AgentUsage = Usage
