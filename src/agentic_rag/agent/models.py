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


class ActionType(StrEnum):
    """High-level action family selected by the V2-2 root policy stage."""

    SEARCH = "SEARCH"
    EXPAND = "EXPAND"
    READ = "READ"
    FINISH = "FINISH"


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


class ActionSelection(AgentModel):
    """High-level V2-2 choice made before an action skill is disclosed.

    Action-specific fields deliberately do not belong in this contract.  The
    intent carries the information gap and rationale into the second policy
    stage, while the evidence references remain pending until the resulting
    action has passed validation.
    """

    action_type: ActionType
    action_intent: str = Field(min_length=1)
    selected_evidence_refs: list[EvidenceRef] = Field(max_length=20)

    @field_validator("action_intent")
    @classmethod
    def action_intent_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "action_intent must contain non-whitespace characters"
            )
        return value

    @model_validator(mode="after")
    def selected_evidence_refs_must_be_unique(self) -> Self:
        keys = [(ref.unit, ref.id) for ref in self.selected_evidence_refs]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "selected_evidence_refs must not contain duplicates"
            )
        return self


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
    # In single-agent V2 the retrieval Policy also supplies the grounded final
    # answer, avoiding a lossy hand-off to a separate Answer call. Legacy
    # scripted decisions may omit it and continue to use AnswerGenerator.
    answer: str | None = None

    @field_validator("answer")
    @classmethod
    def optional_answer_must_not_be_blank(
        cls, value: str | None
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("answer must be omitted rather than blank")
        return value

    @model_validator(mode="after")
    def evidence_refs_must_be_unique(self) -> Self:
        keys = [(ref.unit, ref.id) for ref in self.evidence_refs]
        if len(keys) != len(set(keys)):
            raise ValueError("evidence_refs must not contain duplicates")
        return self


class SearchParameters(AgentModel):
    """V2-2 SEARCH fields emitted after the search skill is disclosed."""

    query: str = Field(min_length=1)
    method: SearchMethod
    target: SearchTarget
    top_k: Literal[5]

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
                f"unsupported retrieval pair: {self.method.value} -> "
                f"{self.target.value}"
            )
        return self


class ExpandParameters(AgentModel):
    """V2-2 EXPAND fields emitted after the expand skill is disclosed."""

    kind: ExpansionKind
    source_id: str = Field(min_length=1)
    direction: ExpansionDirection | None
    query: str | None
    top_k: Literal[5]

    @field_validator("source_id")
    @classmethod
    def source_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "source_id must contain non-whitespace characters"
            )
        return value

    @field_validator("query")
    @classmethod
    def optional_query_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("query must be null rather than blank")
        return value

    @model_validator(mode="after")
    def validate_direction(self) -> Self:
        is_adjacent = self.kind.value == ExpansionKind.CHUNK_ADJACENT_CHUNK.value
        if is_adjacent and self.direction is None:
            raise ValueError("CHUNK_ADJACENT_CHUNK requires direction")
        if not is_adjacent and self.direction is not None:
            raise ValueError("direction is only valid for CHUNK_ADJACENT_CHUNK")
        return self


class ReadParameters(AgentModel):
    """V2-2 READ fields emitted after the read skill is disclosed."""

    chunk_id: str = Field(min_length=1)

    @field_validator("chunk_id")
    @classmethod
    def chunk_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk_id must contain non-whitespace characters")
        return value


class FinishParameters(AgentModel):
    """V2-2 FINISH fields emitted after the finish skill is disclosed."""

    answer: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=20)

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def evidence_refs_must_be_unique(self) -> Self:
        keys = [(ref.unit, ref.id) for ref in self.evidence_refs]
        if len(keys) != len(set(keys)):
            raise ValueError("evidence_refs must not contain duplicates")
        return self


ActionParameters = (
    SearchParameters | ExpandParameters | ReadParameters | FinishParameters
)


AgentAction = Annotated[
    SearchAction | ExpandAction | ReadAction | FinishAction,
    Field(discriminator="type"),
]


class PolicyDecision(AgentModel):
    assessment: EvidenceAssessment
    action: AgentAction


def assemble_policy_decision(
    selection: ActionSelection,
    parameters: ActionParameters,
) -> PolicyDecision:
    """Combine the two V2-2 stages into the existing stable decision model.

    Evidence eligibility and the FINISH-reference subset rule are intentionally
    runtime-validator concerns because they depend on the frozen controller
    state.  This helper only enforces that the parameter contract matches the
    action family chosen in stage one.
    """

    parameter_types: dict[ActionType, type[AgentModel]] = {
        ActionType.SEARCH: SearchParameters,
        ActionType.EXPAND: ExpandParameters,
        ActionType.READ: ReadParameters,
        ActionType.FINISH: FinishParameters,
    }
    expected_type = parameter_types[selection.action_type]
    if not isinstance(parameters, expected_type):
        raise ValueError(
            f"{selection.action_type.value} selection requires "
            f"{expected_type.__name__}"
        )

    payload = parameters.model_dump(mode="python")
    action_constructors: dict[ActionType, type[AgentModel]] = {
        ActionType.SEARCH: SearchAction,
        ActionType.EXPAND: ExpandAction,
        ActionType.READ: ReadAction,
        ActionType.FINISH: FinishAction,
    }
    action = action_constructors[selection.action_type](
        type=selection.action_type.value,
        **payload,
    )
    is_finish = selection.action_type is ActionType.FINISH
    assessment = EvidenceAssessment(
        status=(
            AssessmentStatus.SUFFICIENT
            if is_finish
            else AssessmentStatus.INSUFFICIENT
        ),
        supported_facts=[],
        missing_information=[] if is_finish else [selection.action_intent],
        selected_evidence_refs=selection.selected_evidence_refs,
    )
    return PolicyDecision(assessment=assessment, action=action)


class V3EvidenceAssessment(AgentModel):
    """Assessment-only reasoning for semantic-memory Policy calls."""

    status: AssessmentStatus
    supported_facts: list[str] = Field(default_factory=list, max_length=5)
    missing_information: list[str] = Field(default_factory=list, max_length=3)


class V3ExpandAction(AgentModel):
    type: Literal["EXPAND"] = "EXPAND"
    kind: ExpansionKind
    source_context_index: int = Field(ge=1)
    direction: ExpansionDirection | None = None
    query: str | None = None
    top_k: Literal[5] = 5

    @field_validator("query")
    @classmethod
    def optional_query_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("query must be omitted rather than blank")
        return value

    @model_validator(mode="after")
    def validate_direction(self) -> Self:
        is_adjacent = self.kind is ExpansionKind.CHUNK_ADJACENT_CHUNK
        if is_adjacent and self.direction is None:
            raise ValueError("CHUNK_ADJACENT_CHUNK requires direction")
        if not is_adjacent and self.direction is not None:
            raise ValueError("direction is only valid for CHUNK_ADJACENT_CHUNK")
        return self


class V3ReadAction(AgentModel):
    type: Literal["READ"] = "READ"
    chunk_context_index: int = Field(ge=1)


class V3FinishAction(AgentModel):
    type: Literal["FINISH"] = "FINISH"
    answer: str = Field(min_length=1)
    citations: list[int] = Field(min_length=1, max_length=20)

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def citations_must_be_unique(self) -> Self:
        if any(index < 1 for index in self.citations):
            raise ValueError("citations must contain positive context indices")
        if len(self.citations) != len(set(self.citations)):
            raise ValueError("citations must not contain duplicates")
        return self


V3AgentAction = Annotated[
    SearchAction | V3ExpandAction | V3ReadAction | V3FinishAction,
    Field(discriminator="type"),
]


class V3PolicyDecision(AgentModel):
    assessment: V3EvidenceAssessment
    action: V3AgentAction


_TYPED_NODE_REF_PATTERN = r"^[ESC][1-9][0-9]*$"
TypedNodeRef = Annotated[
    str,
    Field(min_length=2, pattern=_TYPED_NODE_REF_PATTERN),
]


def _normalize_typed_node_ref(value: object) -> object:
    """Normalize harmless formatting without guessing a reference target."""

    if isinstance(value, str):
        return value.strip().upper()
    return value


class V31ExpandAction(AgentModel):
    type: Literal["EXPAND"] = "EXPAND"
    kind: ExpansionKind
    source_ref: TypedNodeRef
    direction: ExpansionDirection | None = None
    query: str | None = None
    top_k: Literal[5] = 5

    @field_validator("source_ref", mode="before")
    @classmethod
    def normalize_source_ref(cls, value: object) -> object:
        return _normalize_typed_node_ref(value)

    @field_validator("query")
    @classmethod
    def optional_query_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("query must be omitted rather than blank")
        return value

    @model_validator(mode="after")
    def validate_direction(self) -> Self:
        is_adjacent = self.kind is ExpansionKind.CHUNK_ADJACENT_CHUNK
        if is_adjacent and self.direction is None:
            raise ValueError("CHUNK_ADJACENT_CHUNK requires direction")
        if not is_adjacent and self.direction is not None:
            raise ValueError("direction is only valid for CHUNK_ADJACENT_CHUNK")
        return self


class V31ReadAction(AgentModel):
    type: Literal["READ"] = "READ"
    chunk_ref: TypedNodeRef

    @field_validator("chunk_ref", mode="before")
    @classmethod
    def normalize_chunk_ref(cls, value: object) -> object:
        return _normalize_typed_node_ref(value)


class V31FinishAction(AgentModel):
    type: Literal["FINISH"] = "FINISH"
    answer: str = Field(min_length=1)
    evidence_refs: list[TypedNodeRef] = Field(min_length=1, max_length=20)

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must contain non-whitespace characters")
        return value

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def normalize_evidence_refs(cls, value: object) -> object:
        if isinstance(value, list):
            return [_normalize_typed_node_ref(item) for item in value]
        return value

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_refs must not contain duplicates")
        return value


V31AgentAction = Annotated[
    SearchAction | V31ExpandAction | V31ReadAction | V31FinishAction,
    Field(discriminator="type"),
]


class V31PolicyDecision(AgentModel):
    assessment: V3EvidenceAssessment
    action: V31AgentAction


PolicyDecisionOutput = PolicyDecision | V3PolicyDecision | V31PolicyDecision


class ObservationStatus(StrEnum):
    OK = "ok"
    INVALID_ACTION = "invalid_action"
    DUPLICATE_ACTION = "duplicate_action"
    ERROR = "error"


class Observation(AgentModel):
    """Complete internal environment record used for state updates and audit."""

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


class ObservationOutcome(StrEnum):
    """Small outcome vocabulary exposed to the Policy LLM."""

    SUCCESS = "success"
    EMPTY = "empty"
    INVALID_ACTION = "invalid_action"
    DUPLICATE_ACTION = "duplicate_action"
    BUDGET_REJECTED = "budget_rejected"
    TOOL_ERROR = "tool_error"


class PolicyObservationUsage(AgentModel):
    retrieved_tokens: int = Field(default=0, ge=0)


class PolicyObservationError(AgentModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class PolicyObservation(AgentModel):
    """Minimal handle-safe observation shown to the Policy LLM.

    Controller-only novelty, visibility, stable IDs, and budget transitions stay
    on :class:`Observation` and :class:`ControllerState` rather than leaking
    into this provider-facing contract.
    """

    action_id: str | None = None
    action: AgentAction | None = None
    outcome: ObservationOutcome
    results: list[dict[str, Any]] = Field(default_factory=list)
    usage: PolicyObservationUsage = Field(
        default_factory=PolicyObservationUsage
    )
    error: PolicyObservationError | None = None


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
    has_been_read: bool = Field(
        default=False,
        validation_alias=AliasChoices("has_been_read", "read"),
    )
    can_use_as_evidence: bool = False
    previews: list[HandleSentencePreview] = Field(
        default_factory=list, max_length=2
    )

    @property
    def read(self) -> bool:
        """Backward-compatible accessor for the previous field name."""

        return self.has_been_read


NodeHandle = Annotated[
    EntityHandle | SentenceHandle | ChunkHandle,
    Field(discriminator="node_type"),
]


class PolicyEntityHandle(AgentModel):
    node_type: Literal["ENTITY"] = "ENTITY"
    id: str = Field(min_length=1)
    label: str
    entity_type: str | None = None
    can_expand: bool


class PolicySentenceHandle(AgentModel):
    node_type: Literal["SENTENCE"] = "SENTENCE"
    id: str = Field(min_length=1)
    text: str
    parent_chunk_id: str = Field(min_length=1)
    title: str | None = None
    can_use_as_evidence: bool
    can_expand: bool


class PolicyChunkHandle(AgentModel):
    node_type: Literal["CHUNK"] = "CHUNK"
    id: str = Field(min_length=1)
    title: str | None = None
    has_been_read: bool
    can_read: bool
    can_expand: bool
    can_use_as_evidence: bool
    previews: list[HandleSentencePreview] = Field(
        default_factory=list, max_length=2
    )


PolicyNodeHandle = Annotated[
    PolicyEntityHandle | PolicySentenceHandle | PolicyChunkHandle,
    Field(discriminator="node_type"),
]


_HANDLE_PREFIX_BY_NODE_TYPE: dict[str, str] = {
    "ENTITY": "E",
    "SENTENCE": "S",
    "CHUNK": "C",
}
_NODE_TYPE_BY_HANDLE_PREFIX: dict[str, str] = {
    prefix: node_type
    for node_type, prefix in _HANDLE_PREFIX_BY_NODE_TYPE.items()
}


class NodeHandleRegistry(AgentModel):
    """Episode-local, persisted bijection between handles and substrate IDs.

    Stable substrate IDs remain the source of truth for runtime control state.
    Policy-facing records use only the compact handles allocated here.
    """

    handle_to_stable_id: dict[str, str] = Field(default_factory=dict)
    stable_id_to_handle: dict[str, str] = Field(default_factory=dict)
    next_entity_index: int = Field(default=1, ge=1)
    next_sentence_index: int = Field(default=1, ge=1)
    next_chunk_index: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def mappings_must_be_a_bijection(self) -> Self:
        inverse = {
            stable_id: handle
            for handle, stable_id in self.handle_to_stable_id.items()
        }
        if inverse != self.stable_id_to_handle:
            raise ValueError(
                "handle_to_stable_id and stable_id_to_handle must be inverse "
                "mappings"
            )
        for handle in self.handle_to_stable_id:
            if self.node_type_for_handle(handle) is None:
                raise ValueError(f"invalid node handle: {handle}")
        return self

    def register(self, stable_id: str, node_type: str) -> str:
        """Return the existing handle or allocate the next typed handle."""

        normalized_type = _normalize_node_type(node_type)
        existing = self.stable_id_to_handle.get(stable_id)
        if existing is not None:
            if self.node_type_for_handle(existing) != normalized_type:
                raise ValueError(
                    f"stable ID is already registered as another node type: "
                    f"{stable_id}"
                )
            return existing

        prefix = _HANDLE_PREFIX_BY_NODE_TYPE[normalized_type]
        counter_field = {
            "ENTITY": "next_entity_index",
            "SENTENCE": "next_sentence_index",
            "CHUNK": "next_chunk_index",
        }[normalized_type]
        index = getattr(self, counter_field)
        handle = f"{prefix}{index}"
        while handle in self.handle_to_stable_id:
            index += 1
            handle = f"{prefix}{index}"
        setattr(self, counter_field, index + 1)
        self.handle_to_stable_id[handle] = stable_id
        self.stable_id_to_handle[stable_id] = handle
        return handle

    def stable_id_for(
        self,
        handle: str,
        expected_type: str | None = None,
    ) -> str | None:
        """Resolve a handle, returning ``None`` for unknown/type mismatch."""

        if expected_type is not None:
            normalized_type = _normalize_node_type(expected_type)
            if self.node_type_for_handle(handle) != normalized_type:
                return None
        return self.handle_to_stable_id.get(handle)

    def handle_for(
        self,
        stable_id: str,
        expected_type: str | None = None,
    ) -> str | None:
        """Resolve a stable ID, returning ``None`` for unknown/type mismatch."""

        handle = self.stable_id_to_handle.get(stable_id)
        if handle is None:
            return None
        if expected_type is not None:
            normalized_type = _normalize_node_type(expected_type)
            if self.node_type_for_handle(handle) != normalized_type:
                return None
        return handle

    @staticmethod
    def node_type_for_handle(handle: str) -> str | None:
        if (
            len(handle) < 2
            or not handle[1:].isdigit()
            or int(handle[1:]) < 1
        ):
            return None
        return _NODE_TYPE_BY_HANDLE_PREFIX.get(handle[0])


def _normalize_node_type(node_type: object) -> str:
    raw = getattr(node_type, "value", node_type)
    normalized = str(raw).upper()
    if normalized not in _HANDLE_PREFIX_BY_NODE_TYPE:
        raise ValueError(f"unsupported node type: {node_type}")
    return normalized


class ControllerState(AgentModel):
    step: int = Field(default=0, ge=0)
    policy_attempts: int = Field(default=0, ge=0)
    visible_entity_ids: set[str] = Field(default_factory=set)
    visible_sentence_ids: set[str] = Field(default_factory=set)
    visible_chunk_ids: set[str] = Field(default_factory=set)
    eligible_sentence_ids: set[str] = Field(default_factory=set)
    read_chunk_ids: set[str] = Field(default_factory=set)
    action_signatures: set[str] = Field(default_factory=set)
    remaining_step_budget: int = Field(ge=0)
    remaining_policy_attempt_budget: int = Field(default=0, ge=0)
    remaining_retrieved_token_budget: int = Field(ge=0)
    last_assessment: EvidenceAssessment | None = None
    newest_observation: Observation | None = None
    selected_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    # First-seen order for the V3 semantic-memory projection. Stable IDs stay
    # internal and are never exposed in the V3 Policy context.
    semantic_memory_node_ids: list[str] = Field(default_factory=list)
    node_handles: dict[str, NodeHandle] = Field(default_factory=dict)
    handle_registry: NodeHandleRegistry = Field(
        default_factory=NodeHandleRegistry
    )

    @classmethod
    def initial(
        cls,
        *,
        max_steps: int = 10,
        max_retrieved_tokens: int = 12_000,
        max_policy_attempts: int | None = None,
    ) -> "ControllerState":
        attempt_budget = (
            max_steps + 2
            if max_policy_attempts is None
            else max_policy_attempts
        )
        return cls(
            remaining_step_budget=max_steps,
            remaining_policy_attempt_budget=attempt_budget,
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


class PolicyEvidenceView(AgentModel):
    """Handle-safe evidence projection for the Policy LLM only."""

    ref: EvidenceRef
    text: str
    title: str | None = None
    parent_chunk_id: str | None = None
    contained_sentence_ids: list[str] = Field(default_factory=list)


class LastValidAssessmentView(AgentModel):
    status: AssessmentStatus
    missing_information: list[str] = Field(default_factory=list, max_length=3)


class AttemptedActionView(AgentModel):
    step: int = Field(ge=1)
    policy_attempt: int = Field(default=1, ge=1)
    action: AgentAction | None = None
    repaired_action: AgentAction | None = None
    repair_code: str | None = None
    validation_status: ValidationStatus
    observation_status: ObservationStatus | None = None
    error_code: str | None = None
    message: str | None = None
    new_node_count: int = Field(default=0, ge=0)
    retrieved_tokens: int = Field(default=0, ge=0)


class PolicyBudgetView(AgentModel):
    remaining_steps: int = Field(ge=0)
    remaining_policy_attempts: int = Field(default=0, ge=0)
    remaining_retrieved_tokens: int = Field(ge=0)


class PolicyStateView(AgentModel):
    step: int = Field(ge=0)
    policy_attempts: int = Field(default=0, ge=0)
    last_valid_assessment: LastValidAssessmentView | None = None
    selected_evidence: list[PolicyEvidenceView] = Field(default_factory=list)
    latest_observation: dict[str, Any] | None = None
    actionable_handles: list[PolicyNodeHandle] = Field(default_factory=list)
    allowed_expansions: dict[ExpansionKind, list[str]] = Field(
        default_factory=dict
    )
    attempted_actions: list[AttemptedActionView] = Field(default_factory=list)
    budget: PolicyBudgetView


class PolicyView(AgentModel):
    context_mode: ContextMode
    instruction: Literal["Produce the next PolicyDecision."] = (
        "Produce the next PolicyDecision."
    )
    policy_state: PolicyStateView


class V3EntityMemoryItem(AgentModel):
    context_index: int = Field(ge=1)
    node_type: Literal["ENTITY"] = "ENTITY"
    canonical_name: str
    entity_type: str | None = None


class V3SentenceMemoryItem(AgentModel):
    context_index: int = Field(ge=1)
    node_type: Literal["SENTENCE"] = "SENTENCE"
    citation_index: int = Field(ge=1)
    title: str | None = None
    text: str
    parent_chunk_context_index: int = Field(ge=1)


class V3ChunkMemoryItem(AgentModel):
    context_index: int = Field(ge=1)
    node_type: Literal["CHUNK"] = "CHUNK"
    citation_index: int | None = Field(default=None, ge=1)
    title: str | None = None
    chunk_position: int = Field(ge=0)
    has_been_read: bool
    text: str | None = None
    previews: list[str] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def read_content_and_citation_are_consistent(self) -> Self:
        if self.has_been_read and (self.text is None or self.citation_index is None):
            raise ValueError("a read Chunk requires text and a citation index")
        if not self.has_been_read and (
            self.text is not None or self.citation_index is not None
        ):
            raise ValueError("an unread Chunk cannot expose text or a citation index")
        return self


V3SemanticMemoryItem = Annotated[
    V3EntityMemoryItem | V3SentenceMemoryItem | V3ChunkMemoryItem,
    Field(discriminator="node_type"),
]


class ContextNodeReference(AgentModel):
    """One audit-only mapping from a context index to a stable node ID."""

    node_type: Literal["ENTITY", "SENTENCE", "CHUNK"]
    stable_id: str = Field(min_length=1)
    can_read: bool = False


class ContextReferenceMap(AgentModel):
    """Frozen reference namespaces associated with exactly one Policy call."""

    memory: dict[int, ContextNodeReference] = Field(default_factory=dict)
    citations: dict[int, EvidenceRef] = Field(default_factory=dict)


class TypedContextNodeReference(AgentModel):
    """One V3.1 typed ref resolved only within its frozen prompt snapshot."""

    node_type: Literal["ENTITY", "SENTENCE", "CHUNK"]
    stable_id: str = Field(min_length=1)
    can_read: bool = False
    can_use_as_evidence: bool = False


class TypedContextReferenceMap(AgentModel):
    """Single E#/S#/C# namespace associated with exactly one Policy call."""

    typed_refs: dict[str, TypedContextNodeReference] = Field(
        default_factory=dict
    )


class V3PolicyStateView(AgentModel):
    step: int = Field(ge=0)
    policy_attempts: int = Field(default=0, ge=0)
    last_assessment: V3EvidenceAssessment | None = None
    semantic_memory: list[V3SemanticMemoryItem] = Field(default_factory=list)
    latest_event: dict[str, Any] | None = None
    attempted_actions: list[dict[str, Any]] = Field(default_factory=list)
    budget: PolicyBudgetView


class V3PolicyView(AgentModel):
    context_mode: Literal["semantic_memory_v3"] = "semantic_memory_v3"
    instruction: Literal["Produce the next V3PolicyDecision."] = (
        "Produce the next V3PolicyDecision."
    )
    policy_state: V3PolicyStateView


class V31EntityMemoryItem(AgentModel):
    ref: str = Field(min_length=2, pattern=_TYPED_NODE_REF_PATTERN)
    node_type: Literal["ENTITY"] = "ENTITY"
    canonical_name: str
    entity_type: str | None = None


class V31SentenceMemoryItem(AgentModel):
    ref: str = Field(min_length=2, pattern=_TYPED_NODE_REF_PATTERN)
    node_type: Literal["SENTENCE"] = "SENTENCE"
    title: str | None = None
    text: str
    parent_chunk_ref: str = Field(
        min_length=2,
        pattern=r"^C[1-9][0-9]*$",
    )


class V31ChunkMemoryItem(AgentModel):
    ref: str = Field(min_length=2, pattern=_TYPED_NODE_REF_PATTERN)
    node_type: Literal["CHUNK"] = "CHUNK"
    title: str | None = None
    chunk_position: int = Field(ge=0)
    has_been_read: bool
    text: str | None = None
    previews: list[str] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def read_content_is_consistent(self) -> Self:
        if self.has_been_read and self.text is None:
            raise ValueError("a read Chunk requires full text")
        if not self.has_been_read and self.text is not None:
            raise ValueError("an unread Chunk cannot expose full text")
        return self


V31SemanticMemoryItem = Annotated[
    V31EntityMemoryItem | V31SentenceMemoryItem | V31ChunkMemoryItem,
    Field(discriminator="node_type"),
]


class V31PolicyStateView(AgentModel):
    step: int = Field(ge=0)
    policy_attempts: int = Field(default=0, ge=0)
    last_assessment: V3EvidenceAssessment | None = None
    semantic_memory: list[V31SemanticMemoryItem] = Field(default_factory=list)
    latest_event: dict[str, Any] | None = None
    attempted_actions: list[dict[str, Any]] = Field(default_factory=list)
    budget: PolicyBudgetView | str


class V31PolicyView(AgentModel):
    context_mode: Literal[
        "semantic_memory_v3_typed_refs",
        "semantic_memory_v3_2",
    ] = (
        "semantic_memory_v3_typed_refs"
    )
    instruction: Literal["Produce the next V31PolicyDecision."] = (
        "Produce the next V31PolicyDecision."
    )
    policy_state: V31PolicyStateView


PolicyViewOutput = PolicyView | V3PolicyView | V31PolicyView


class Message(AgentModel):
    role: Literal["system", "user", "assistant"]
    content: str

    def as_openai_input(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class PolicyStagePhase(StrEnum):
    """Auditable phases of one progressive V2-2 policy cycle."""

    ACTION_SELECTION = "action_selection"
    ACTION_DRAFT = "action_draft"
    ACTION_REPAIR = "action_repair"


class PolicyStageRecord(AgentModel):
    """Provider-call audit record for one V2-2 policy stage."""

    phase: PolicyStagePhase
    messages: list[Message] = Field(default_factory=list)
    output: dict[str, Any] | None = None
    error: str | None = None
    action_type: ActionType | None = None
    disclosed_skill_paths: list[str] = Field(default_factory=list)
    disclosed_skill_hashes: dict[str, str] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)


class StepRecord(AgentModel):
    step: int = Field(ge=1)
    policy_attempt: int = Field(default=1, ge=1)
    # The raw Policy LLM decision uses episode-local S#/C#/E# handles.
    decision: PolicyDecisionOutput | None = None
    # A narrowly repaired handle-form decision, when the source handle makes
    # one inverse expansion direction unambiguous. The raw decision above is
    # always retained for audit and SkillOpt diagnosis.
    repaired_decision: PolicyDecision | None = None
    repair_code: str | None = None
    # The Controller-owned audit form resolves those handles back to stable
    # substrate IDs before validation and execution.
    resolved_decision: PolicyDecision | None = None
    validation_status: ValidationStatus
    validation_error: str | None = None
    # The raw environment Observation remains available for replay/audit.
    observation: Observation | None = None
    # This is the handle-safe semantic feedback that the Policy LLM can see on
    # the following turn and that SkillOpt should learn from.
    agent_visible_observation: dict[str, Any] | None = None
    state_before: ControllerState
    state_after: ControllerState
    usage: Usage = Field(default_factory=Usage)
    policy_view: PolicyViewOutput | None = None
    context_reference_map: (
        ContextReferenceMap | TypedContextReferenceMap | None
    ) = None
    policy_stages: list[PolicyStageRecord] = Field(default_factory=list)


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


def action_signature(action: AgentAction) -> str:
    """Return a deterministic signature used for duplicate-action detection."""

    payload = action.model_dump(mode="json", exclude_none=True)
    # FINISH identity is its evidence selection. Rewording an answer must not
    # make the same terminal action appear novel.
    if isinstance(action, FinishAction):
        payload.pop("answer", None)
    query = payload.get("query")
    if isinstance(query, str):
        payload["query"] = " ".join(query.casefold().split())
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Compatibility alias for callers that prefer an explicit name.
AgentUsage = Usage
