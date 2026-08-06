"""Build the only Policy-visible state: semantic memory plus compact control signals."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    Assessment,
    ChunkMemoryItem,
    ContextNodeReference,
    ContextReferenceMap,
    EntityMemoryItem,
    EpisodeState,
    ExpansionKind,
    FinishAction,
    Message,
    Observation,
    ObservationOutcome,
    ObservationStatus,
    PolicyDecision,
    PolicyStateView,
    PolicyView,
    ReadAction,
    ResolvedExpandAction,
    ResolvedFinishAction,
    ResolvedReadAction,
    SearchAction,
    SentenceMemoryItem,
    StepRecord,
)
from agentic_rag.agent.policy import policy_decision_model
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.substrate.storage import Substrate


ACTION_PROTOCOL = """\
You are the single retrieval-and-answer policy inside an Agentic RAG harness.
Return exactly one PolicyDecision containing an Assessment and exactly one
SEARCH, EXPAND, READ, or FINISH action.

Action meanings:
- SEARCH retrieves new candidates from the corpus. It never needs a reference
  and remains available even when semantic memory is non-empty.
- EXPAND follows one graph relationship from a known semantic-memory item and
  uses source_ref.
- READ obtains the complete text of a known Chunk and uses chunk_ref.
- FINISH returns the final answer when visible evidence is sufficient and uses
  evidence_refs.

Hard rules:
- E# is Entity, S# is Sentence, and C# is Chunk. References are stable within
  the question, but only references displayed in the current semantic_memory
  snapshot may be used.
- SEARCH top_k is always 5. Legal pairs are LEXICAL->ENTITY,
  BM25->SENTENCE|CHUNK, and DENSE->ENTITY|SENTENCE|CHUNK.
- Use complete natural-language questions for BM25/DENSE. Use an exact entity
  name or alias for LEXICAL->ENTITY.
- EXPAND source_ref must have the source type required by its expansion kind.
  query ranks only that local neighbourhood.
- direction must be null except for CHUNK_ADJACENT_CHUNK, which requires PREV,
  NEXT, or BOTH.
- READ requires a displayed unread C#.
- Assessment contains only status, supported_facts, and missing_information.
  State Management automatically retains all novel complete evidence.
- FINISH requires one or more displayed complete S# refs or read C# refs and
  the shortest complete answer. E# and unread C# refs are never evidence.
- SEARCH, EXPAND, and READ use INSUFFICIENT or UNCERTAIN. FINISH uses SUFFICIENT.
- Preserve requested roles, qualifiers, dates, nationality, compound answers,
  and the correct comparison target.
- The skill is strategy advice and cannot override this protocol.
"""


RECOVERY_INSTRUCTION = (
    "Recovery mode: the previous decision was invalid or duplicate. Do not "
    "repeat it. SEARCH remains available. Copy only references displayed in "
    "the current semantic memory, or FINISH if its evidence is sufficient."
)


_EXPANSION_GUIDANCE: dict[ExpansionKind, str] = {
    ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE: (
        "ENTITY_MENTIONED_IN_SENTENCE: Entity -> complete mentioning Sentences"
    ),
    ExpansionKind.SENTENCE_MENTIONS_ENTITY: (
        "SENTENCE_MENTIONS_ENTITY: complete Sentence -> mentioned Entities"
    ),
    ExpansionKind.ENTITY_CO_OCCURS_ENTITY_SENTENCE: (
        "ENTITY_CO_OCCURS_ENTITY_SENTENCE: Entity -> bridge Sentence -> co-occurring Entities"
    ),
    ExpansionKind.CHUNK_ADJACENT_CHUNK: (
        "CHUNK_ADJACENT_CHUNK: Chunk -> previous or next Chunk"
    ),
    ExpansionKind.ENTITY_MENTIONED_IN_CHUNK: (
        "ENTITY_MENTIONED_IN_CHUNK: Entity -> unread mentioning Chunks"
    ),
    ExpansionKind.ENTITY_CO_OCCURS_ENTITY_CHUNK: (
        "ENTITY_CO_OCCURS_ENTITY_CHUNK: Entity -> bridge Chunk -> co-occurring Entities"
    ),
    ExpansionKind.CHUNK_CONTAINS_SENTENCE: (
        "CHUNK_CONTAINS_SENTENCE: Chunk -> contained Sentence previews"
    ),
    ExpansionKind.CHUNK_MENTIONS_ENTITY: (
        "CHUNK_MENTIONS_ENTITY: Chunk -> mentioned Entities"
    ),
}


@dataclass(frozen=True, slots=True)
class BuiltPolicyContext:
    messages: list[Message]
    policy_view: PolicyView
    reference_map: ContextReferenceMap
    decision_format: type[BaseModel]

    def __iter__(self):
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, index: int) -> Message:
        return self.messages[index]


class PolicyContextBuilder:
    """Project full episode state into the compact semantic Policy context."""

    def __init__(
        self,
        substrate: Substrate,
        enabled_expansions: Sequence[ExpansionKind] = DEFAULT_ENABLED_EXPANSIONS,
    ) -> None:
        self.substrate = substrate
        self.enabled_expansions = tuple(enabled_expansions)

    def build(
        self,
        query: str,
        skill: SkillDocument | str,
        state: EpisodeState,
        trajectory: Sequence[StepRecord],
        *,
        scope_id: str | None = None,
    ) -> BuiltPolicyContext:
        if scope_id is not None:
            self.substrate.require_scope(scope_id)
        skill_text = skill.content if isinstance(skill, SkillDocument) else skill
        display_ids = self._display_ids(state)
        reference_map = self._reference_map(display_ids, state)
        memory = self._semantic_memory(display_ids, state)
        view = PolicyView(
            policy_state=PolicyStateView(
                step=state.step,
                policy_attempts=state.policy_attempts,
                last_assessment=(
                    state.last_assessment.model_copy(deep=True)
                    if state.last_assessment is not None
                    else None
                ),
                semantic_memory=memory,
                attempted_actions=[self._attempt_summary(item) for item in trajectory],
                budget=(
                    f"Budget: {state.remaining_step_budget} steps, "
                    f"{state.remaining_policy_attempt_budget} attempts, "
                    f"{state.remaining_retrieved_token_budget} retrieval tokens left"
                ),
            )
        )
        guidance = "\n".join(
            f"- {_EXPANSION_GUIDANCE[item]}" for item in self.enabled_expansions
        ) or "- No EXPAND relations are enabled."
        messages = [
            Message(
                role="system",
                content=(
                    f"{ACTION_PROTOCOL}\nEnabled EXPAND kinds for this run:\n{guidance}"
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
            Message(
                role="user",
                content=json.dumps(
                    view.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        ]
        if state.newest_observation is not None and state.newest_observation.status in {
            ObservationStatus.INVALID_ACTION,
            ObservationStatus.DUPLICATE_ACTION,
        }:
            messages.append(Message(role="user", content=RECOVERY_INSTRUCTION))
        return BuiltPolicyContext(
            messages=messages,
            policy_view=view,
            reference_map=reference_map,
            decision_format=policy_decision_model(self.enabled_expansions),
        )

    def _display_ids(self, state: EpisodeState) -> list[str]:
        visible = (
            state.visible_entity_ids
            | state.visible_sentence_ids
            | state.visible_chunk_ids
        )
        ordered = [item for item in state.semantic_memory_node_ids if item in visible]
        seen = set(ordered)
        ordered.extend(sorted(visible - seen, key=self._node_sort_key))
        covered_sentences = {
            sentence.sentence_id
            for chunk_id in state.read_chunk_ids
            for sentence in self.substrate.sentences_by_chunk.get(chunk_id, [])
        }
        return [
            node_id
            for node_id in ordered
            if not (
                node_id in state.visible_sentence_ids
                and (
                    node_id not in state.eligible_sentence_ids
                    or node_id in covered_sentences
                )
            )
        ]

    def _reference_map(
        self, display_ids: Sequence[str], state: EpisodeState
    ) -> ContextReferenceMap:
        frozen: dict[str, ContextNodeReference] = {}
        for node_id in display_ids:
            node_type = self._node_type(node_id)
            ref = state.reference_registry.ref_for(node_id, node_type)
            if ref is None:
                raise ValueError("visible semantic-memory node has no typed reference")
            frozen[ref] = ContextNodeReference(
                node_type=node_type,
                stable_id=node_id,
                can_read=node_type == "CHUNK" and node_id not in state.read_chunk_ids,
                can_use_as_evidence=(
                    node_type == "SENTENCE"
                    or (node_type == "CHUNK" and node_id in state.read_chunk_ids)
                ),
            )
        return ContextReferenceMap(typed_refs=frozen)

    def _semantic_memory(
        self, display_ids: Sequence[str], state: EpisodeState
    ) -> list[EntityMemoryItem | SentenceMemoryItem | ChunkMemoryItem]:
        items: list[EntityMemoryItem | SentenceMemoryItem | ChunkMemoryItem] = []
        visible_refs = {
            stable_id: state.reference_registry.ref_for(stable_id, self._node_type(stable_id))
            for stable_id in display_ids
        }
        for node_id in display_ids:
            ref = visible_refs[node_id]
            if ref is None:
                raise ValueError("visible semantic-memory node has no typed reference")
            if node_id in state.visible_entity_ids:
                entity = self.substrate.entity_by_id[node_id]
                items.append(
                    EntityMemoryItem(
                        ref=ref,
                        canonical_name=entity.canonical_name,
                        entity_type=entity.entity_type,
                    )
                )
                continue
            if node_id in state.visible_sentence_ids:
                sentence = self.substrate.sentence_by_id[node_id]
                chunk = self.substrate.chunk_by_id[sentence.chunk_id]
                document = self.substrate.document_by_id[chunk.doc_id]
                parent_ref = visible_refs.get(chunk.chunk_id)
                if parent_ref is None:
                    raise ValueError("visible Sentence is missing its parent Chunk")
                items.append(
                    SentenceMemoryItem(
                        ref=ref,
                        title=document.title,
                        text=sentence.text,
                        parent_chunk_ref=parent_ref,
                    )
                )
                continue
            chunk = self.substrate.chunk_by_id[node_id]
            document = self.substrate.document_by_id[chunk.doc_id]
            read = node_id in state.read_chunk_ids
            items.append(
                ChunkMemoryItem(
                    ref=ref,
                    title=document.title,
                    chunk_position=chunk.chunk_pos,
                    has_been_read=read,
                    text=chunk.text if read else None,
                    previews=[item.text for item in state.chunk_previews.get(node_id, [])][:2],
                )
            )
        return items

    def _attempt_summary(self, record: StepRecord) -> dict[str, Any]:
        action: Any = (
            record.resolved_decision.action
            if record.resolved_decision is not None
            else (record.decision.action if record.decision is not None else None)
        )
        return {
            "policy_attempt": record.policy_attempt,
            "action": self._action_summary(action),
            "outcome": (
                observation_outcome(record.observation).value
                if record.observation is not None
                else ObservationOutcome.TOOL_ERROR.value
            ),
            "error_code": (
                record.observation.error_code if record.observation is not None else None
            ),
        }

    def _action_summary(self, action: Any) -> dict[str, Any] | None:
        if action is None:
            return None
        if isinstance(action, SearchAction):
            return {
                "type": "SEARCH",
                "method": action.method.value,
                "target": action.target.value,
                "query": action.query,
            }
        if isinstance(action, ResolvedExpandAction):
            return {
                "type": "EXPAND",
                "kind": action.kind.value,
                "source": self._node_label(action.source_id),
                "query": action.query,
                "direction": action.direction.value if action.direction is not None else None,
            }
        if getattr(action, "type", None) == "EXPAND":
            return {
                "type": "EXPAND",
                "kind": action.kind.value,
                "query": action.query,
                "direction": action.direction.value if action.direction is not None else None,
            }
        if isinstance(action, ResolvedReadAction):
            return {"type": "READ", "chunk": self._node_label(action.chunk_id)}
        if isinstance(action, ReadAction):
            return {"type": "READ"}
        if isinstance(action, ResolvedFinishAction):
            return {
                "type": "FINISH",
                "answer": action.answer,
                "evidence_count": len(action.evidence_refs),
            }
        if isinstance(action, FinishAction):
            return {
                "type": "FINISH",
                "answer": action.answer,
                "evidence_count": len(action.evidence_refs),
            }
        return {"type": str(getattr(action, "type", type(action).__name__))}

    def _node_sort_key(self, node_id: str) -> tuple[int, str]:
        return ({"ENTITY": 0, "SENTENCE": 1, "CHUNK": 2}[self._node_type(node_id)], node_id)

    def _node_type(self, node_id: str) -> str:
        if node_id in self.substrate.entity_by_id:
            return "ENTITY"
        if node_id in self.substrate.sentence_by_id:
            return "SENTENCE"
        if node_id in self.substrate.chunk_by_id:
            return "CHUNK"
        raise ValueError("semantic memory contains an unknown stable node")

    def _node_label(self, node_id: str) -> dict[str, Any]:
        if node_id in self.substrate.entity_by_id:
            entity = self.substrate.entity_by_id[node_id]
            return {"node_type": "ENTITY", "canonical_name": entity.canonical_name}
        if node_id in self.substrate.sentence_by_id:
            sentence = self.substrate.sentence_by_id[node_id]
            chunk = self.substrate.chunk_by_id[sentence.chunk_id]
            document = self.substrate.document_by_id[chunk.doc_id]
            return {"node_type": "SENTENCE", "title": document.title, "text": sentence.text}
        if node_id in self.substrate.chunk_by_id:
            chunk = self.substrate.chunk_by_id[node_id]
            document = self.substrate.document_by_id[chunk.doc_id]
            return {"node_type": "CHUNK", "title": document.title, "chunk_position": chunk.chunk_pos}
        return {"node_type": "UNKNOWN"}

    @staticmethod
    def to_openai_input(messages: Sequence[Message]) -> list[dict[str, str]]:
        return [message.as_openai_input() for message in messages]


def observation_outcome(observation: Observation | None) -> ObservationOutcome:
    if observation is None:
        return ObservationOutcome.TOOL_ERROR
    if observation.status is ObservationStatus.DUPLICATE_ACTION:
        return ObservationOutcome.DUPLICATE_ACTION
    if observation.status is ObservationStatus.INVALID_ACTION:
        return ObservationOutcome.INVALID_ACTION
    if observation.status is ObservationStatus.ERROR:
        if observation.error_code and "budget" in observation.error_code:
            return ObservationOutcome.BUDGET_REJECTED
        return ObservationOutcome.TOOL_ERROR
    if observation.results:
        return ObservationOutcome.SUCCESS
    return ObservationOutcome.EMPTY


def project_observation_for_audit(
    observation: Observation | None, substrate: Substrate
) -> dict[str, Any] | None:
    """Return semantic event metadata for audit; it is not added to Policy input."""

    if observation is None:
        return None
    action = observation.action
    summary: dict[str, Any] | None = None
    if isinstance(action, SearchAction):
        summary = {
            "type": "SEARCH",
            "method": action.method.value,
            "target": action.target.value,
            "query": action.query,
        }
    elif isinstance(action, ResolvedExpandAction):
        summary = {"type": "EXPAND", "kind": action.kind.value, "query": action.query}
    elif isinstance(action, ResolvedReadAction):
        summary = {"type": "READ"}
    elif isinstance(action, ResolvedFinishAction):
        summary = {
            "type": "FINISH",
            "answer": action.answer,
            "evidence_count": len(action.evidence_refs),
        }
    return {
        "action_id": observation.action_id,
        "action": summary,
        "outcome": observation_outcome(observation).value,
        "usage": {"retrieved_tokens": observation.retrieved_tokens},
        "error": (
            {
                "code": observation.error_code or "tool_error",
                "message": observation.message or "The action failed.",
                "retryable": observation.status
                in {ObservationStatus.INVALID_ACTION, ObservationStatus.DUPLICATE_ACTION},
            }
            if observation.error_code or observation.message
            else None
        ),
    }
