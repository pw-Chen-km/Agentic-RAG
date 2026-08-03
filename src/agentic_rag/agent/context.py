"""Policy-context construction with compact evidence memory and a baseline."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    AgentAction,
    AttemptedActionView,
    ChunkRef,
    ContextMode,
    ControllerState,
    ExpansionKind,
    EntityHandle,
    ChunkHandle,
    LastValidAssessmentView,
    Message,
    PolicyDecision,
    PolicyBudgetView,
    PolicyEvidenceView,
    PolicyChunkHandle,
    PolicyEntityHandle,
    PolicyNodeHandle,
    PolicySentenceHandle,
    PolicyStateView,
    PolicyView,
    SentenceRef,
    SentenceHandle,
    StepRecord,
)
from agentic_rag.agent.skill import SkillDocument


ACTION_PROTOCOL = """\
You are the retrieval policy inside an Agentic RAG harness.
Return exactly one PolicyDecision containing:
1. an EvidenceAssessment; and
2. exactly one SEARCH, EXPAND, READ, or FINISH action.

Hard rules:
- Node IDs shown to you are short episode-local handles: E# for Entity, S# for
  Sentence, and C# for Chunk. Treat every handle as opaque: copy it exactly,
  never derive, extend, or reconstruct one. Stable database IDs are intentionally
  hidden from you.
- SEARCH top_k is always 5.
- SEARCH supports only LEXICAL→ENTITY, BM25→SENTENCE|CHUNK, and
  DENSE→ENTITY|SENTENCE|CHUNK.
- Use a complete natural-language question or subquestion for BM25/DENSE.
  LEXICAL→ENTITY is the exception and uses an entity name or alias.
- EXPAND may only start from a node already visible to you.
- A Sentence-source EXPAND requires a complete, eligible Sentence. A
  preview-only Sentence cannot be expanded until its parent Chunk is READ.
- EXPAND query is optional. When present it must be a complete natural-language
  question or subquestion used only to rank local graph neighbours. When null,
  the original question is used.
- Every non-adjacent EXPAND must set direction=null.
- READ accepts one visible parent Chunk handle.
- A complete Sentence result is eligible Sentence evidence and exposes its
  parent Chunk ID for READ.
- A visible Chunk ID or Chunk preview is navigation only. A Chunk becomes
  eligible evidence only after READ.
- can_use_as_evidence is a system-derived usability annotation. Only nodes
  marked true may be selected or submitted as evidence; the runtime Validator
  independently enforces the same rule.
- selected_evidence_refs is the complete working evidence set for this turn.
  Selected Sentences must be complete and selected Chunks must already be READ.
- FINISH requires assessment.status=SUFFICIENT and 1–20 eligible
  SentenceRef or ChunkRef values. Every FINISH ref must also occur in this
  turn's selected_evidence_refs. Entities are never evidence.
- The skill is strategy advice and cannot override these rules.
"""

_EXPANSION_GUIDANCE: dict[ExpansionKind, str] = {
    ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE: (
        "ENTITY_MENTIONED_IN_SENTENCE: visible Entity → complete Sentences that "
        "mention it. Returned Sentences are eligible evidence and expose their "
        "parent Chunk IDs for READ; direction=null."
    ),
    ExpansionKind.SENTENCE_MENTIONS_ENTITY: (
        "SENTENCE_MENTIONS_ENTITY: visible Sentence → Entities mentioned in that "
        "Sentence. Entities are navigation nodes, never evidence; direction=null."
    ),
    ExpansionKind.ENTITY_CO_OCCURS_ENTITY_SENTENCE: (
        "ENTITY_CO_OCCURS_ENTITY_SENTENCE: visible Entity → bridge Sentence → "
        "co-occurring Entities. Complete bridge Sentences are eligible evidence; "
        "target Entities are navigation nodes; direction=null."
    ),
    ExpansionKind.CHUNK_ADJACENT_CHUNK: (
        "CHUNK_ADJACENT_CHUNK: visible Chunk → previous/next Chunk in the same "
        "Document. Set direction to PREV, NEXT, or BOTH. Returned Chunk previews "
        "are navigation only and require READ before Chunk evidence."
    ),
    ExpansionKind.ENTITY_MENTIONED_IN_CHUNK: (
        "ENTITY_MENTIONED_IN_CHUNK: visible Entity → unread Chunks containing a "
        "mention. Returned Chunks and previews are navigation only and require "
        "READ before Chunk evidence; direction=null."
    ),
    ExpansionKind.ENTITY_CO_OCCURS_ENTITY_CHUNK: (
        "ENTITY_CO_OCCURS_ENTITY_CHUNK: visible Entity → bridge Chunk → "
        "co-occurring Entities. Target Entities are navigation nodes; bridge "
        "previews are navigation only and require READ; direction=null."
    ),
    ExpansionKind.CHUNK_CONTAINS_SENTENCE: (
        "CHUNK_CONTAINS_SENTENCE: visible Chunk → contained Sentence previews. "
        "These previews are navigation only and cannot become evidence until the "
        "Chunk is READ; direction=null."
    ),
    ExpansionKind.CHUNK_MENTIONS_ENTITY: (
        "CHUNK_MENTIONS_ENTITY: visible Chunk → Entities mentioned across that "
        "Chunk. Entities are navigation nodes, never evidence; direction=null."
    ),
}

_LATEST_METADATA_KEYS = frozenset(
    {
        "candidate_count_before_truncation",
        "finish",
        "retrieved_budget_exhausted",
        "source_degree",
        "truncated_by_retrieved_token_budget",
    }
)
_NODE_ID_KEYS = frozenset(
    {
        "id",
        "source_id",
        "entity_id",
        "target_entity_id",
        "sentence_id",
        "bridge_sentence_id",
        "chunk_id",
        "parent_chunk_id",
        "bridge_chunk_id",
    }
)
_DOCUMENT_ID_KEYS = frozenset({"document_id", "doc_id"})
_ACTION_ADAPTER = TypeAdapter(AgentAction)
_ENTITY_EXPANSION_KINDS = frozenset(
    {
        ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
        ExpansionKind.ENTITY_CO_OCCURS_ENTITY_SENTENCE,
        ExpansionKind.ENTITY_MENTIONED_IN_CHUNK,
        ExpansionKind.ENTITY_CO_OCCURS_ENTITY_CHUNK,
    }
)
_SENTENCE_EXPANSION_KINDS = frozenset(
    {ExpansionKind.SENTENCE_MENTIONS_ENTITY}
)
_CHUNK_EXPANSION_KINDS = frozenset(
    {
        ExpansionKind.CHUNK_ADJACENT_CHUNK,
        ExpansionKind.CHUNK_CONTAINS_SENTENCE,
        ExpansionKind.CHUNK_MENTIONS_ENTITY,
    }
)


@dataclass(frozen=True, slots=True)
class BuiltPolicyContext:
    messages: list[Message]
    policy_view: PolicyView

    def __iter__(self):
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, index):
        return self.messages[index]


class PolicyContextBuilder:
    """Build either the compact online context or the append-only baseline."""

    def __init__(
        self,
        enabled_expansions: Sequence[
            ExpansionKind
        ] = DEFAULT_ENABLED_EXPANSIONS,
        *,
        context_mode: ContextMode | str = ContextMode.COMPACT_EVIDENCE,
        evidence_resolver: EvidenceResolver | None = None,
    ) -> None:
        self.enabled_expansions = tuple(enabled_expansions)
        self.context_mode = ContextMode(context_mode)
        self.evidence_resolver = evidence_resolver

    def build(
        self,
        query: str,
        skill: SkillDocument | str,
        state: ControllerState,
        trajectory: Sequence[StepRecord],
        *,
        scope_id: str | None = None,
    ) -> BuiltPolicyContext:
        skill_text = skill.content if isinstance(skill, SkillDocument) else skill
        policy_view = self.build_policy_view(
            state, trajectory, scope_id=scope_id
        )
        messages = self._base_messages(query, skill_text)
        if self.context_mode is ContextMode.APPEND_ONLY:
            messages.extend(self._append_only_history(trajectory, state))
            messages.append(
                Message(
                    role="user",
                    content=self._json(
                        {
                            "instruction": policy_view.instruction,
                            "current_state": self._append_only_state(
                                state, policy_view
                            ),
                        }
                    ),
                )
            )
        else:
            payload = policy_view.model_dump(
                mode="json", exclude={"context_mode"}
            )
            messages.append(
                Message(role="user", content=self._json(payload))
            )
        return BuiltPolicyContext(messages=messages, policy_view=policy_view)

    def build_policy_view(
        self,
        state: ControllerState,
        trajectory: Sequence[StepRecord],
        *,
        scope_id: str | None,
    ) -> PolicyView:
        selected = self._selected_evidence(state, scope_id)
        latest = self.project_observation(
            state.newest_observation, state
        )
        represented_ids = {
            item.ref.id for item in selected
        }
        for item in selected:
            if item.parent_chunk_id:
                represented_ids.add(item.parent_chunk_id)
        represented_ids.update(_node_ids(latest))
        handles = [
            self._policy_handle(handle)
            for _, handle in sorted(
                state.node_handles.items(),
                key=lambda item: (item[1].node_type, item[0]),
            )
            if handle.id not in represented_ids
        ]
        last_assessment = (
            LastValidAssessmentView(
                status=state.last_assessment.status,
                missing_information=list(
                    state.last_assessment.missing_information
                ),
            )
            if state.last_assessment is not None
            else None
        )
        attempted = [
            AttemptedActionView(
                step=record.step,
                action=(
                    _project_action(record.decision.action, state)
                    if record.decision is not None
                    else None
                ),
                validation_status=record.validation_status,
                observation_status=(
                    record.observation.status
                    if record.observation is not None
                    else None
                ),
                error_code=(
                    record.observation.error_code
                    if record.observation is not None
                    else None
                ),
                message=(
                    _handle_safe_value(record.observation.message, state)
                    if record.observation is not None
                    else _handle_safe_value(record.validation_error, state)
                ),
                new_node_count=(
                    len(record.observation.novel_node_ids)
                    if record.observation is not None
                    else 0
                ),
                retrieved_tokens=(
                    record.observation.retrieved_tokens
                    if record.observation is not None
                    else 0
                ),
            )
            for record in trajectory
        ]
        return PolicyView(
            context_mode=self.context_mode,
            policy_state=PolicyStateView(
                step=state.step,
                last_valid_assessment=last_assessment,
                selected_evidence=selected,
                latest_observation=latest,
                actionable_handles=handles,
                attempted_actions=attempted,
                budget=PolicyBudgetView(
                    remaining_steps=state.remaining_step_budget,
                    remaining_retrieved_tokens=(
                        state.remaining_retrieved_token_budget
                    ),
                ),
            ),
        )

    def _policy_handle(self, handle: Any) -> PolicyNodeHandle:
        enabled = frozenset(self.enabled_expansions)
        if isinstance(handle, EntityHandle):
            return PolicyEntityHandle(
                id=handle.id,
                label=handle.label,
                entity_type=handle.entity_type,
                can_expand=bool(enabled & _ENTITY_EXPANSION_KINDS),
            )
        if isinstance(handle, SentenceHandle):
            return PolicySentenceHandle(
                id=handle.id,
                text=handle.text,
                parent_chunk_id=handle.parent_chunk_id,
                title=handle.title,
                can_use_as_evidence=handle.can_use_as_evidence,
                can_expand=(
                    handle.can_use_as_evidence
                    and bool(enabled & _SENTENCE_EXPANSION_KINDS)
                ),
            )
        if isinstance(handle, ChunkHandle):
            return PolicyChunkHandle(
                id=handle.id,
                title=handle.title,
                has_been_read=handle.has_been_read,
                can_read=not handle.has_been_read,
                can_expand=bool(enabled & _CHUNK_EXPANSION_KINDS),
                can_use_as_evidence=handle.can_use_as_evidence,
                previews=list(handle.previews),
            )
        raise TypeError(f"unsupported node handle: {type(handle).__name__}")

    def _selected_evidence(
        self,
        state: ControllerState,
        scope_id: str | None,
    ) -> list[PolicyEvidenceView]:
        if not state.selected_evidence_refs:
            return []
        if self.evidence_resolver is None or scope_id is None:
            raise ValueError(
                "selected evidence projection requires a resolver and scope_id"
            )
        resolved: list[PolicyEvidenceView] = []
        for ref in sorted(
            state.selected_evidence_refs,
            key=lambda item: (item.unit, item.id),
        ):
            for item in self.evidence_resolver.resolve(
                [ref], state, scope_id
            ):
                if isinstance(item.ref, SentenceRef):
                    handle = _require_handle(
                        state, item.ref.id, "SENTENCE"
                    )
                    projected_ref = SentenceRef(id=handle)
                else:
                    handle = _require_handle(
                        state, item.ref.id, "CHUNK"
                    )
                    projected_ref = ChunkRef(id=handle)
                parent_chunk_id = (
                    _require_handle(
                        state, item.parent_chunk_id, "CHUNK"
                    )
                    if item.parent_chunk_id is not None
                    else None
                )
                resolved.append(
                    PolicyEvidenceView(
                        ref=projected_ref,
                        text=item.text,
                        title=item.title,
                        parent_chunk_id=parent_chunk_id,
                        contained_sentence_ids=[
                            _require_handle(state, sentence_id, "SENTENCE")
                            for sentence_id in item.contained_sentence_ids
                        ],
                    )
                )
        return resolved

    def _base_messages(
        self, query: str, skill_text: str
    ) -> list[Message]:
        return [
            Message(
                role="system",
                content=(
                    f"{ACTION_PROTOCOL}\nEnabled EXPAND kinds for this run:\n"
                    f"{self._expansion_guidance()}"
                ),
            ),
            Message(
                role="system",
                content=(
                    "Current trainable retrieval skill (plain Markdown):\n\n"
                    f"{skill_text}"
                ),
            ),
            Message(role="user", content=f"Original question:\n{query}"),
        ]

    def project_observation(
        self,
        observation: Any,
        state: ControllerState,
    ) -> dict[str, Any] | None:
        """Project one raw observation using this run's action config."""

        return project_observation_for_policy(
            observation,
            state,
            enabled_expansions=self.enabled_expansions,
        )

    def _append_only_history(
        self,
        trajectory: Sequence[StepRecord],
        state: ControllerState,
    ) -> list[Message]:
        messages: list[Message] = []
        for record in trajectory:
            decision_payload = (
                _project_decision(record.decision, state)
                if record.decision is not None
                else {
                    "policy_error": _handle_safe_value(
                        record.validation_error, state
                    )
                    or "Policy did not produce a valid decision."
                }
            )
            messages.append(
                Message(
                    role="assistant",
                    content=self._json(
                        {
                            "step": record.step,
                            "policy_decision": decision_payload,
                        }
                    ),
                )
            )
            messages.append(
                Message(
                    role="user",
                    content=self._json(
                        {
                            "step": record.step,
                            "validation_status": record.validation_status,
                            "validation_error": _handle_safe_value(
                                record.validation_error, state
                            ),
                            "environment_observation": (
                                self.project_observation(
                                    record.observation, state
                                )
                                if record.observation is not None
                                else None
                            ),
                        }
                    ),
                )
            )
        return messages

    @staticmethod
    def _append_only_state(
        state: ControllerState,
        policy_view: PolicyView,
    ) -> dict[str, Any]:
        return {
            "step": state.step,
            "selected_evidence": [
                item.model_dump(mode="json")
                for item in policy_view.policy_state.selected_evidence
            ],
            "actionable_handles": [
                item.model_dump(mode="json")
                for item in policy_view.policy_state.actionable_handles
            ],
            "remaining_step_budget": state.remaining_step_budget,
            "remaining_retrieved_token_budget": (
                state.remaining_retrieved_token_budget
            ),
        }

    def _expansion_guidance(self) -> str:
        if not self.enabled_expansions:
            return "- (none; do not choose EXPAND)"
        return "\n".join(
            f"- {_EXPANSION_GUIDANCE[item]}"
            for item in self.enabled_expansions
        )

    @staticmethod
    def to_openai_input(
        messages: Sequence[Message],
    ) -> list[dict[str, str]]:
        return [message.as_openai_input() for message in messages]

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )


class AppendOnlyContextBuilder(PolicyContextBuilder):
    """Explicit baseline that preserves the original replay behavior."""

    def __init__(
        self,
        enabled_expansions: Sequence[
            ExpansionKind
        ] = DEFAULT_ENABLED_EXPANSIONS,
        *,
        evidence_resolver: EvidenceResolver | None = None,
    ) -> None:
        super().__init__(
            enabled_expansions,
            context_mode=ContextMode.APPEND_ONLY,
            evidence_resolver=evidence_resolver,
        )


def project_observation_for_policy(
    observation: Any,
    state: ControllerState,
    *,
    enabled_expansions: Sequence[
        ExpansionKind
    ] = DEFAULT_ENABLED_EXPANSIONS,
) -> dict[str, Any] | None:
    """Return the exact handle-safe observation projection shown to Policy."""

    if observation is None:
        return None
    action = (
        _project_action(observation.action, state).model_dump(mode="json")
        if observation.action is not None
        else None
    )
    results = [
        _project_result(
            item,
            state=state,
            is_read=bool(action and action["type"] == "READ"),
            enabled_expansions=enabled_expansions,
        )
        for item in observation.results
    ]
    metadata = {
        key: observation.metadata[key]
        for key in sorted(_LATEST_METADATA_KEYS)
        if key in observation.metadata
    }
    return {
        "action": action,
        "status": observation.status,
        "results": results,
        "retrieved_tokens": observation.retrieved_tokens,
        "error_code": observation.error_code,
        "message": _handle_safe_value(observation.message, state),
        "metadata": metadata,
    }


def _project_result(
    result: Mapping[str, Any],
    *,
    state: ControllerState,
    is_read: bool,
    enabled_expansions: Sequence[ExpansionKind],
) -> dict[str, Any]:
    projected = {
        key: _project_result_value(
            value,
            state=state,
            enabled_expansions=enabled_expansions,
        )
        for key, value in result.items()
        if key
        not in {
            "content_read",
            "evidence_eligible",
            *_DOCUMENT_ID_KEYS,
        }
    }
    if is_read and "sentences" in projected:
        projected.pop("text", None)
    _annotate_evidence_usability(
        projected, state, enabled_expansions
    )
    return projected


def _project_result_value(
    value: Any,
    *,
    state: ControllerState,
    enabled_expansions: Sequence[ExpansionKind],
) -> Any:
    if isinstance(value, list):
        return [
            _project_result_value(
                item,
                state=state,
                enabled_expansions=enabled_expansions,
            )
            for item in value
        ]
    if not isinstance(value, Mapping):
        return _handle_safe_value(value, state)
    projected = {
        key: _project_result_value(
            child,
            state=state,
            enabled_expansions=enabled_expansions,
        )
        for key, child in value.items()
        if key
        not in {
            "content_read",
            "evidence_eligible",
            *_DOCUMENT_ID_KEYS,
        }
    }
    _annotate_evidence_usability(
        projected, state, enabled_expansions
    )
    return projected


def _annotate_evidence_usability(
    result: dict[str, Any],
    state: ControllerState,
    enabled_expansions: Sequence[ExpansionKind],
) -> None:
    enabled = frozenset(enabled_expansions)
    entity_id = result.get(
        "entity_id",
        result.get("target_entity_id"),
    )
    if isinstance(entity_id, str):
        result["can_expand"] = bool(enabled & _ENTITY_EXPANSION_KINDS)
        return

    sentence_id = result.get(
        "sentence_id",
        result.get("bridge_sentence_id"),
    )
    if isinstance(sentence_id, str):
        stable_sentence_id = (
            state.handle_registry.stable_id_for(
                sentence_id, "SENTENCE"
            )
            or sentence_id
        )
        result["can_use_as_evidence"] = (
            stable_sentence_id in state.eligible_sentence_ids
        )
        result["can_expand"] = (
            stable_sentence_id in state.eligible_sentence_ids
            and bool(enabled & _SENTENCE_EXPANSION_KINDS)
        )
        return

    chunk_id = result.get("chunk_id")
    is_chunk_result = isinstance(chunk_id, str) and (
        result.get("target") == "CHUNK"
        or "sentences" in result
        or "previews" in result
    )
    if is_chunk_result:
        stable_chunk_id = (
            state.handle_registry.stable_id_for(chunk_id, "CHUNK")
            or chunk_id
        )
        read = stable_chunk_id in state.read_chunk_ids
        result["has_been_read"] = read
        result["can_read"] = not read
        result["can_expand"] = bool(enabled & _CHUNK_EXPANSION_KINDS)
        result["can_use_as_evidence"] = read


def _project_action(
    action: AgentAction,
    state: ControllerState,
) -> AgentAction:
    payload = _handle_safe_value(action.model_dump(mode="json"), state)
    return _ACTION_ADAPTER.validate_python(payload)


def _project_decision(
    decision: PolicyDecision,
    state: ControllerState,
) -> dict[str, Any]:
    return _handle_safe_value(decision.model_dump(mode="json"), state)


def _handle_safe_value(value: Any, state: ControllerState) -> Any:
    """Recursively replace registered stable node IDs with short handles."""

    if isinstance(value, Mapping):
        return {
            key: _handle_safe_value(child, state)
            for key, child in value.items()
            if key not in _DOCUMENT_ID_KEYS
        }
    if isinstance(value, list):
        return [_handle_safe_value(item, state) for item in value]
    if isinstance(value, tuple):
        return [_handle_safe_value(item, state) for item in value]
    if not isinstance(value, str):
        return value
    exact = state.handle_registry.handle_for(value)
    if exact is not None:
        return exact
    projected = value
    for stable_id, handle in sorted(
        state.handle_registry.stable_id_to_handle.items(),
        key=lambda item: (-len(item[0]), item[0]),
    ):
        projected = projected.replace(stable_id, handle)
    return projected


def _require_handle(
    state: ControllerState,
    stable_id: str,
    expected_type: str,
) -> str:
    handle = state.handle_registry.handle_for(stable_id, expected_type)
    if handle is None:
        raise ValueError(
            f"missing {expected_type} handle for visible stable node"
        )
    return handle


def _node_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, list):
        for item in value:
            found.update(_node_ids(item))
        return found
    if not isinstance(value, Mapping):
        return found
    for key, child in value.items():
        if key in _NODE_ID_KEYS and isinstance(child, str):
            found.add(child)
        elif key in {
            "node_ids",
            "bridge_sentence_ids",
            "bridge_chunk_ids",
        } and isinstance(child, list):
            found.update(item for item in child if isinstance(item, str))
        else:
            found.update(_node_ids(child))
    return found
