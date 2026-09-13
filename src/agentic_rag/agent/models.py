"""Canonical contracts for the single Agentic RAG workflow."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator


class AgentModel(BaseModel):
    """Strict base model used by provider messages and persisted artifacts."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


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


class Assessment(AgentModel):
    """The Policy's semantic judgment, deliberately free of references."""

    supported_facts: list[str] = Field(default_factory=list, max_length=5)
    missing_information: list[str] = Field(default_factory=list, max_length=3)


_TYPED_REF_PATTERN = r"^[ESC][1-9][0-9]*$"
TypedReference = Annotated[str, Field(min_length=2, pattern=_TYPED_REF_PATTERN)]


def _normalize_typed_ref(value: object) -> object:
    if isinstance(value, str):
        return value.strip().upper()
    return value


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
    def method_target_pair_must_be_legal(self) -> Self:
        if (self.method, self.target) not in VALID_SEARCH_PAIRS:
            raise ValueError(
                f"unsupported SEARCH pair: {self.method.value}->{self.target.value}"
            )
        return self


class ExpandAction(AgentModel):
    type: Literal["EXPAND"] = "EXPAND"
    kind: ExpansionKind
    source_ref: TypedReference
    direction: ExpansionDirection | None = None
    query: str | None = None
    top_k: Literal[5] = 5

    @field_validator("source_ref", mode="before")
    @classmethod
    def normalize_source_ref(cls, value: object) -> object:
        return _normalize_typed_ref(value)

    @field_validator("query")
    @classmethod
    def optional_query_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("query must be omitted rather than blank")
        return value

    @model_validator(mode="after")
    def direction_must_match_kind(self) -> Self:
        adjacent = self.kind is ExpansionKind.CHUNK_ADJACENT_CHUNK
        if adjacent and self.direction is None:
            raise ValueError("CHUNK_ADJACENT_CHUNK requires direction")
        if not adjacent and self.direction is not None:
            raise ValueError("direction is only valid for CHUNK_ADJACENT_CHUNK")
        return self


class ReadAction(AgentModel):
    type: Literal["READ"] = "READ"
    chunk_ref: TypedReference

    @field_validator("chunk_ref", mode="before")
    @classmethod
    def normalize_chunk_ref(cls, value: object) -> object:
        return _normalize_typed_ref(value)


class FinishAction(AgentModel):
    type: Literal["FINISH"] = "FINISH"
    answer: str = Field(min_length=1)
    evidence_refs: list[TypedReference] = Field(min_length=1, max_length=20)

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
            return [_normalize_typed_ref(item) for item in value]
        return value

    @field_validator("evidence_refs")
    @classmethod
    def evidence_refs_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_refs must not contain duplicates")
        return value


AgentAction = Annotated[
    SearchAction | ExpandAction | ReadAction | FinishAction,
    Field(discriminator="type"),
]


class PolicyDecision(AgentModel):
    assessment: Assessment
    action: AgentAction


class SentenceRef(AgentModel):
    unit: Literal["SENTENCE"] = "SENTENCE"
    id: str = Field(min_length=1)


class ChunkRef(AgentModel):
    unit: Literal["CHUNK"] = "CHUNK"
    id: str = Field(min_length=1)


EvidenceRef = Annotated[SentenceRef | ChunkRef, Field(discriminator="unit")]


class ResolvedExpandAction(AgentModel):
    type: Literal["EXPAND"] = "EXPAND"
    kind: ExpansionKind
    source_id: str = Field(min_length=1)
    direction: ExpansionDirection | None = None
    query: str | None = None
    top_k: Literal[5] = 5


class ResolvedReadAction(AgentModel):
    type: Literal["READ"] = "READ"
    chunk_id: str = Field(min_length=1)


class ResolvedFinishAction(AgentModel):
    type: Literal["FINISH"] = "FINISH"
    answer: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=20)


ResolvedAction = Annotated[
    SearchAction | ResolvedExpandAction | ResolvedReadAction | ResolvedFinishAction,
    Field(discriminator="type"),
]


class ResolvedDecision(AgentModel):
    assessment: Assessment
    action: ResolvedAction


class ObservationStatus(StrEnum):
    OK = "ok"
    INVALID_ACTION = "invalid_action"
    DUPLICATE_ACTION = "duplicate_action"
    ERROR = "error"


class ObservationOutcome(StrEnum):
    SUCCESS = "success"
    EMPTY = "empty"
    INVALID_ACTION = "invalid_action"
    DUPLICATE_ACTION = "duplicate_action"
    BUDGET_REJECTED = "budget_rejected"
    TOOL_ERROR = "tool_error"


class Observation(AgentModel):
    """Complete environment fact retained by State Management and audit."""

    action_id: str | None = None
    status: ObservationStatus
    action: ResolvedAction | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_tokens: int = Field(default=0, ge=0)
    novel_node_ids: list[str] = Field(default_factory=list)
    already_seen_node_ids: list[str] = Field(default_factory=list)
    error_code: str | None = None
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(AgentModel):
    policy_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    retrieved_tokens: int = Field(default=0, ge=0)

    def __add__(self, other: object) -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            policy_calls=self.policy_calls + other.policy_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            retrieved_tokens=self.retrieved_tokens + other.retrieved_tokens,
        )


class SentencePreview(AgentModel):
    sentence_ref: str = Field(min_length=2, pattern=r"^S[1-9][0-9]*$")
    text: str


_PREFIX_BY_NODE_TYPE = {"ENTITY": "E", "SENTENCE": "S", "CHUNK": "C"}
_NODE_TYPE_BY_PREFIX = {value: key for key, value in _PREFIX_BY_NODE_TYPE.items()}


class ReferenceRegistry(AgentModel):
    """Episode-local stable bijection between typed refs and substrate IDs."""

    ref_to_stable_id: dict[str, str] = Field(default_factory=dict)
    stable_id_to_ref: dict[str, str] = Field(default_factory=dict)
    next_entity_index: int = Field(default=1, ge=1)
    next_sentence_index: int = Field(default=1, ge=1)
    next_chunk_index: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def mappings_must_be_a_bijection(self) -> Self:
        inverse = {stable_id: ref for ref, stable_id in self.ref_to_stable_id.items()}
        if inverse != self.stable_id_to_ref:
            raise ValueError("reference registry mappings must be inverse")
        if any(self.node_type_for_ref(ref) is None for ref in self.ref_to_stable_id):
            raise ValueError("reference registry contains an invalid typed ref")
        return self

    def register(self, stable_id: str, node_type: str) -> str:
        normalized_type = str(getattr(node_type, "value", node_type)).upper()
        if normalized_type not in _PREFIX_BY_NODE_TYPE:
            raise ValueError(f"unsupported node type: {node_type}")
        existing = self.stable_id_to_ref.get(stable_id)
        if existing is not None:
            if self.node_type_for_ref(existing) != normalized_type:
                raise ValueError("stable ID is already registered as another node type")
            return existing
        field_name = {
            "ENTITY": "next_entity_index",
            "SENTENCE": "next_sentence_index",
            "CHUNK": "next_chunk_index",
        }[normalized_type]
        index = getattr(self, field_name)
        prefix = _PREFIX_BY_NODE_TYPE[normalized_type]
        ref = f"{prefix}{index}"
        while ref in self.ref_to_stable_id:
            index += 1
            ref = f"{prefix}{index}"
        setattr(self, field_name, index + 1)
        self.ref_to_stable_id[ref] = stable_id
        self.stable_id_to_ref[stable_id] = ref
        return ref

    def ref_for(self, stable_id: str, expected_type: str | None = None) -> str | None:
        ref = self.stable_id_to_ref.get(stable_id)
        if ref is None:
            return None
        if expected_type is not None and self.node_type_for_ref(ref) != expected_type.upper():
            return None
        return ref

    def stable_id_for(self, ref: str, expected_type: str | None = None) -> str | None:
        normalized = str(_normalize_typed_ref(ref))
        if expected_type is not None and self.node_type_for_ref(normalized) != expected_type.upper():
            return None
        return self.ref_to_stable_id.get(normalized)

    @staticmethod
    def node_type_for_ref(ref: str) -> str | None:
        if len(ref) < 2 or not ref[1:].isdigit() or int(ref[1:]) < 1:
            return None
        return _NODE_TYPE_BY_PREFIX.get(ref[0].upper())


class EpisodeState(AgentModel):
    step: int = Field(default=0, ge=0)
    policy_attempts: int = Field(default=0, ge=0)
    visible_entity_ids: set[str] = Field(default_factory=set)
    visible_sentence_ids: set[str] = Field(default_factory=set)
    visible_chunk_ids: set[str] = Field(default_factory=set)
    eligible_sentence_ids: set[str] = Field(default_factory=set)
    read_chunk_ids: set[str] = Field(default_factory=set)
    action_signatures: set[str] = Field(default_factory=set)
    remaining_step_budget: int = Field(ge=0)
    remaining_policy_attempt_budget: int = Field(ge=0)
    remaining_retrieved_token_budget: int = Field(ge=0)
    last_assessment: Assessment | None = None
    newest_observation: Observation | None = None
    semantic_memory_node_ids: list[str] = Field(default_factory=list)
    chunk_previews: dict[str, list[SentencePreview]] = Field(default_factory=dict)
    reference_registry: ReferenceRegistry = Field(default_factory=ReferenceRegistry)

    @classmethod
    def initial(
        cls,
        *,
        max_steps: int = 10,
        max_policy_attempts: int = 12,
        max_retrieved_tokens: int = 12_000,
    ) -> "EpisodeState":
        return cls(
            remaining_step_budget=max_steps,
            remaining_policy_attempt_budget=max_policy_attempts,
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


class ContextNodeReference(AgentModel):
    node_type: Literal["ENTITY", "SENTENCE", "CHUNK"]
    stable_id: str = Field(min_length=1)
    can_read: bool = False
    can_use_as_evidence: bool = False


class ContextReferenceMap(AgentModel):
    """Frozen typed-ref map bound to one exact Policy prompt."""

    typed_refs: dict[str, ContextNodeReference] = Field(default_factory=dict)


class ActionSpaceMode(StrEnum):
    NORMAL = "NORMAL"
    BUDGET_FINALIZE = "BUDGET_FINALIZE"


class SearchActionOption(AgentModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    method: SearchMethod
    target: SearchTarget


class ExpandActionOption(AgentModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    kind: ExpansionKind
    source_refs: tuple[TypedReference, ...]
    directions: tuple[ExpansionDirection, ...] = ()


class AvailableActionSpace(AgentModel):
    """Immutable structural action affordances for one exact Policy turn."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    mode: ActionSpaceMode = ActionSpaceMode.NORMAL
    search_options: tuple[SearchActionOption, ...] = ()
    expand_options: tuple[ExpandActionOption, ...] = ()
    read_refs: tuple[TypedReference, ...] = ()
    finish_evidence_refs: tuple[TypedReference, ...] = ()

    @property
    def has_actions(self) -> bool:
        return bool(
            self.search_options
            or self.expand_options
            or self.read_refs
            or self.finish_evidence_refs
        )


class EntityMemoryItem(AgentModel):
    ref: str = Field(min_length=2, pattern=r"^E[1-9][0-9]*$")
    node_type: Literal["ENTITY"] = "ENTITY"
    canonical_name: str
    entity_type: str | None = None


class SentenceMemoryItem(AgentModel):
    ref: str = Field(min_length=2, pattern=r"^S[1-9][0-9]*$")
    node_type: Literal["SENTENCE"] = "SENTENCE"
    title: str | None = None
    text: str
    parent_chunk_ref: str = Field(min_length=2, pattern=r"^C[1-9][0-9]*$")


class ChunkMemoryItem(AgentModel):
    ref: str = Field(min_length=2, pattern=r"^C[1-9][0-9]*$")
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


SemanticMemoryItem = Annotated[
    EntityMemoryItem | SentenceMemoryItem | ChunkMemoryItem,
    Field(discriminator="node_type"),
]


class PolicyStateView(AgentModel):
    step: int = Field(ge=0)
    policy_attempts: int = Field(ge=0)
    last_assessment: Assessment | None = None
    semantic_memory: list[SemanticMemoryItem] = Field(default_factory=list)
    latest_attempt: dict[str, Any] | None = None
    attempted_actions: list[dict[str, Any]] = Field(default_factory=list)
    budget: str


class PolicyView(AgentModel):
    instruction: Literal["Produce the next PolicyDecision."] = "Produce the next PolicyDecision."
    policy_state: PolicyStateView


class Message(AgentModel):
    role: Literal["system", "user", "assistant"]
    content: str

    def as_openai_input(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class StepRecord(AgentModel):
    step: int = Field(ge=1)
    policy_attempt: int = Field(ge=1)
    decision: PolicyDecision | None = None
    resolved_decision: ResolvedDecision | None = None
    validation_status: ValidationStatus
    validation_error: str | None = None
    observation: Observation | None = None
    agent_visible_observation: dict[str, Any] | None = None
    state_before: EpisodeState
    state_after: EpisodeState
    usage: Usage = Field(default_factory=Usage)
    policy_view: PolicyView | None = None
    context_reference_map: ContextReferenceMap | None = None
    available_action_space: AvailableActionSpace | None = None
    decision_schema_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    messages: list[Message] = Field(default_factory=list)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    visible_source_spans: list[dict[str, Any]] = Field(default_factory=list)


class TerminationReason(StrEnum):
    FINISH = "finish"
    BUDGET_EXHAUSTED = "budget_exhausted"
    POLICY_ERROR = "policy_error"
    RUNTIME_ERROR = "runtime_error"


class EpisodeResult(AgentModel):
    episode_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    scope_id: str = Field(min_length=1)
    termination_reason: TerminationReason
    answer: str | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    resolved_evidence: list[ResolvedEvidence] = Field(default_factory=list)
    trajectory: list[StepRecord] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    artifact_dir: str | None = None
    final_state: EpisodeState | None = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def question(self) -> str:
        return self.query


def action_signature(action: ResolvedAction) -> str:
    """Return the semantic signature used for duplicate suppression."""

    payload = action.model_dump(mode="json", exclude_none=True)
    if isinstance(action, ResolvedFinishAction):
        payload.pop("answer", None)
    query = payload.get("query")
    if isinstance(query, str):
        payload["query"] = " ".join(query.casefold().split())
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
