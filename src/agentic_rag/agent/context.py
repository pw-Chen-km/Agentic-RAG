"""Build the only Policy-visible state: semantic memory plus compact control signals."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agentic_rag.agent.action_schema import (
    ActionSchemaBuilder,
    InterfaceDecisionSchemaBuilder,
    decision_schema_sha256,
)
from agentic_rag.agent.action_space import AvailableActionSpaceBuilder
from agentic_rag.agent.interface import InterfaceContract
from agentic_rag.agent.interface_action_catalog import render_action_guide
from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    ActionSpaceMode,
    AvailableActionSpace,
    ChunkMemoryItem,
    ContextNodeReference,
    ContextReferenceMap,
    EntityMemoryItem,
    EpisodeState,
    ExpandAction,
    ExpansionKind,
    FinishAction,
    Message,
    Observation,
    ObservationOutcome,
    ObservationStatus,
    PolicyStateView,
    PolicyView,
    ReadAction,
    ResolvedExpandAction,
    ResolvedFinishAction,
    ResolvedReadAction,
    SearchAction,
    SentenceMemoryItem,
    StepRecord,
    ToolCall,
)
from agentic_rag.agent.policy import policy_decision_model
from agentic_rag.agent.protocol import (
    render_action_protocol,
    render_available_action_options,
)
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.tool_calling import build_tool_definitions, tool_schema_sha256
from agentic_rag.agent.context_rendering import render_context
from agentic_rag.substrate.storage import Substrate


@dataclass(frozen=True, slots=True)
class BuiltPolicyContext:
    messages: list[Message]
    policy_view: PolicyView
    reference_map: ContextReferenceMap
    available_action_space: AvailableActionSpace
    decision_format: type[BaseModel]
    decision_schema_sha256: str
    decision_schema: dict[str, Any]
    tool_definitions: list[dict[str, Any]]
    tool_schema_sha256: str
    provider_tools: list[dict[str, Any]] | None = None
    visible_source_spans: list[dict[str, Any]] = field(default_factory=list)

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
        *,
        show_available_action_options: bool = True,
        use_state_conditioned_schema: bool = True,
        interface_contract: InterfaceContract | None = None,
        require_evidence_assessment: bool = True,
    ) -> None:
        if isinstance(enabled_expansions, InterfaceContract) and interface_contract is None:
            interface_contract = enabled_expansions
            enabled_expansions = interface_contract.enabled_expansions
        self.substrate = substrate
        self.enabled_expansions = tuple(enabled_expansions)
        self.show_available_action_options = show_available_action_options
        self.use_state_conditioned_schema = use_state_conditioned_schema
        self.interface_contract = interface_contract
        self.require_evidence_assessment = require_evidence_assessment
        self.action_space_builder = AvailableActionSpaceBuilder(
            self.enabled_expansions,
            interface_contract,
        )
        self.action_schema_builder = ActionSchemaBuilder()
        self.interface_decision_schema_builder = InterfaceDecisionSchemaBuilder()

    def build(
        self,
        query: str,
        skill: SkillDocument | str,
        state: EpisodeState,
        trajectory: Sequence[StepRecord],
        *,
        scope_id: str | None = None,
        action_space_mode: ActionSpaceMode = ActionSpaceMode.NORMAL,
    ) -> BuiltPolicyContext:
        if scope_id is not None:
            self.substrate.require_scope(scope_id)
        skill_text = skill.content if isinstance(skill, SkillDocument) else skill
        display_ids = self._display_ids(state)
        reference_map = self._reference_map(display_ids, state)
        if self.interface_contract is not None:
            # Keep previously displayed sentence labels usable after their text
            # is subsumed by a complete passage. No new labels are invented.
            for record in trajectory:
                if record.context_reference_map is None:
                    continue
                for ref, node in record.context_reference_map.typed_refs.items():
                    if node.node_type == "SENTENCE" and node.stable_id in state.eligible_sentence_ids:
                        reference_map.typed_refs.setdefault(ref, node.model_copy(deep=True))
        available_action_space = self.action_space_builder.build(
            state,
            reference_map,
            mode=action_space_mode,
        )
        if self.interface_contract is not None:
            return self._build_native(query, skill_text, state, trajectory,
                                      display_ids, reference_map, available_action_space)
        tool_definitions = build_tool_definitions(available_action_space)
        decision_format = (
            self.action_schema_builder.build(available_action_space)
            if self.use_state_conditioned_schema
            else policy_decision_model(self.enabled_expansions)
        )
        memory = self._semantic_memory(display_ids, state)
        attempted_actions = [self._attempt_summary(item) for item in trajectory]
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
                latest_attempt=(
                    self._latest_attempt_summary(attempted_actions[-1])
                    if attempted_actions
                    else None
                ),
                attempted_actions=attempted_actions,
                budget=(
                    f"Budget: {state.remaining_step_budget} steps, "
                    f"{state.remaining_policy_attempt_budget} attempts, "
                    f"{state.remaining_retrieved_token_budget} retrieval tokens left"
                ),
            )
        )
        protocol = (
            self.interface_contract.protocol
            if self.interface_contract is not None
            else render_action_protocol(self.enabled_expansions)
        )
        if self.show_available_action_options:
            protocol = (
                f"{protocol}\n\n"
                f"{render_available_action_options(available_action_space)}"
            )
        messages = [
            Message(
                role="system",
                content=(
                    f"{protocol}\n\n"
                    "Current retrieval skill (plain Markdown):\n\n"
                    f"{skill_text}"
                ),
            ),
            Message(role="user", content=f"Original question:\n{query}"),
        ]
        # Preserve the provider-native tool protocol across turns.  The
        # semantic-memory view above remains the authoritative compact state;
        # these messages only give the model the required assistant/tool
        # pairing for its previous calls.
        for record in trajectory:
            raw_calls = record.provider_metadata.get("raw_tool_calls", [])
            if not isinstance(raw_calls, list):
                continue
            for raw_call in raw_calls:
                try:
                    call = ToolCall.model_validate(raw_call)
                except Exception:
                    continue
                messages.append(
                    Message(role="assistant", content="", tool_calls=[call])
                )
                result_payload = project_observation_for_audit(record.observation, self.substrate)
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(
                            result_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                )
        messages.append(
            Message(
                role="user",
                content=json.dumps(
                    view.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
        return BuiltPolicyContext(
            messages=messages,
            policy_view=view,
            reference_map=reference_map,
            available_action_space=available_action_space,
            decision_format=decision_format,
            decision_schema_sha256=tool_schema_sha256(tool_definitions),
            decision_schema=decision_format.model_json_schema(),
            tool_definitions=tool_definitions,
            tool_schema_sha256=tool_schema_sha256(tool_definitions),
            provider_tools=tool_definitions,
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
        return [
            node_id
            for node_id in ordered
            if not (
                node_id in state.visible_sentence_ids
                and node_id not in state.eligible_sentence_ids
            )
            and not (
                self.interface_contract is not None
                and node_id in state.visible_chunk_ids
                and node_id not in state.visible_passage_ids
                and node_id not in state.read_chunk_ids
            )
            and not (
                self.interface_contract is not None
                and node_id in self.substrate.sentence_by_id
                and self.substrate.sentence_by_id[node_id].chunk_id
                in (state.visible_passage_ids | state.read_chunk_ids)
            )
        ]

    def _build_native(self, query, skill_text, state, trajectory, display_ids, references, space):
        """One cumulative observation, no raw backend payload or repeated source text.

        Earlier calls are summarized in attempted_actions.  Only the latest
        native call/result pair is retained, with the current cumulative visible
        state as its result.  This is the same memory policy for every condition.
        """
        # Retain a plain capability registry for audit only. The provider receives
        # the single decision schema below, not native tool definitions.
        tools = build_tool_definitions(space, require_evidence_assessment=False)
        memory, spans, entity_cards = [], [], []
        names = {}
        for node_id in display_ids:
            if node_id in state.visible_entity_ids and self.interface_contract.entity_annotation:
                name = self.substrate.entity_by_id[node_id].canonical_name
                names.setdefault(name.casefold(), []).append(node_id)
        for node_id in display_ids:
            ref = state.reference_registry.ref_for(node_id)
            if ref is None:
                raise ValueError("visible source has no episode-local reference")
            if node_id in state.visible_entity_ids:
                if not self.interface_contract.entity_annotation:
                    continue
                name = self.substrate.entity_by_id[node_id].canonical_name
                label = f"{ref} — {name}"
                if len(names[name.casefold()]) > 1:
                    locations = []
                    for mention in self.substrate.mentions:
                        if mention.entity_id == node_id and mention.sentence_id in state.eligible_sentence_ids:
                            sentence = self.substrate.sentence_by_id[mention.sentence_id]
                            parent = state.reference_registry.ref_for(sentence.chunk_id)
                            locations.append(f"Passage {parent}, Sentence {sentence.sentence_pos + 1}")
                    label += " (" + "; ".join(sorted(set(locations))) + ")"
                entity_cards.append(label)
                continue
            if node_id in self.substrate.chunk_by_id:
                chunk = self.substrate.chunk_by_id[node_id]
                title = self.substrate.document_by_id[chunk.doc_id].title
                text = f"Passage {ref}" + (f" — {title}" if title else "") + f"\n{chunk.text}"
                sentences = self.substrate.sentences_by_chunk.get(node_id, [])
                aliases = [f"{label}: sentence {self.substrate.sentence_by_id[node.stable_id].sentence_pos + 1}"
                           for label, node in references.typed_refs.items()
                           if node.node_type == "SENTENCE"
                           and self.substrate.sentence_by_id[node.stable_id].chunk_id == node_id]
                if aliases:
                    text += "\nSentence labels within this passage: " + "; ".join(aliases)
            else:
                sentence = self.substrate.sentence_by_id[node_id]
                chunk = self.substrate.chunk_by_id[sentence.chunk_id]
                title = self.substrate.document_by_id[chunk.doc_id].title
                text = f"Sentence {ref}" + (f" — {title}" if title else "") + f"\n{sentence.text}"
                sentences = [sentence]
            memory.append({"ref": ref, "text": text, "sentence_ids": [s.sentence_id for s in sentences]})
            for sentence in sentences:
                spans.append({"span_type": "sentence", "sentence_id": sentence.sentence_id,
                              "chunk_id": sentence.chunk_id, "source_ref": ref,
                              "start": 0, "end": len(sentence.text), "text": sentence.text,
                              "visible": True, "complete": True, "seen_by_policy": True})
        latest_decision = next(
            (
                record.decision or record.resolved_decision
                for record in reversed(trajectory)
                if (record.decision or record.resolved_decision) is not None
                and (record.decision or record.resolved_decision).assessment is not None
            ),
            None,
        )
        content, audit, attempted = render_context(
            memory, spans, entity_cards, trajectory, state,
            require_assessment=self.require_evidence_assessment,
        )
        guide = render_action_guide(
            space,
            entity_annotation=self.interface_contract.entity_annotation,
            entity_navigation_possible=bool(self.interface_contract.enabled_expansions),
            require_evidence_assessment=self.require_evidence_assessment,
        )
        messages = [Message(role="system", content=skill_text.strip() + "\n\n" + self.interface_contract.protocol + "\n\n" + guide),
                    Message(role="user", content=f"Original question:\n{query}")]
        # The study protocol uses one schema-constrained decision per turn.
        # Prior actions are rendered in the cumulative observation rather than
        # replayed as provider-native assistant/tool messages.
        messages.append(Message(role="user", content=content))
        view = PolicyView(context_audit=audit, policy_state=PolicyStateView(
            step=state.step, policy_attempts=state.policy_attempts,
            last_assessment=(
                latest_decision.assessment.model_copy(deep=True)
                if latest_decision is not None and self.require_evidence_assessment
                else None
            ),
            semantic_memory=self._semantic_memory(display_ids, state),
            attempted_actions=attempted,
            budget=f"{state.remaining_step_budget} decisions; {state.remaining_retrieved_token_budget} retrieval tokens"))
        decision_format = self.interface_decision_schema_builder.build(
            space,
            require_evidence_assessment=self.require_evidence_assessment,
        )
        return BuiltPolicyContext(messages=messages, policy_view=view, reference_map=references,
                                  available_action_space=space,
                                  decision_format=decision_format,
                                  decision_schema_sha256=decision_schema_sha256(decision_format),
                                  decision_schema=decision_format.model_json_schema(),
                                  tool_definitions=tools, tool_schema_sha256=tool_schema_sha256(tools),
                                  provider_tools=None, visible_source_spans=spans)

    def _reference_map(
        self, display_ids: Sequence[str], state: EpisodeState
    ) -> ContextReferenceMap:
        frozen: dict[str, ContextNodeReference] = {}
        for node_id in display_ids:
            node_type = self._node_type(node_id)
            if node_type == "ENTITY" and self.interface_contract is not None and not self.interface_contract.entity_annotation:
                continue
            ref = state.reference_registry.ref_for(node_id, node_type)
            if ref is None:
                raise ValueError("visible semantic-memory node has no typed reference")
            frozen[ref] = ContextNodeReference(
                node_type=node_type,
                stable_id=node_id,
                can_read=node_type == "CHUNK" and node_id not in state.read_chunk_ids,
                can_use_as_evidence=(
                    node_type == "SENTENCE"
                    or (
                        node_type == "CHUNK"
                        and (
                            (
                                self.interface_contract is None
                                and node_id in state.read_chunk_ids
                            )
                            or (
                                self.interface_contract is not None
                                and (
                                    node_id in state.visible_passage_ids
                                    or node_id in state.read_chunk_ids
                                )
                            )
                        )
                    )
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
                parent_ref = (
                    state.reference_registry.ref_for(chunk.chunk_id, "CHUNK")
                    if self.interface_contract is not None
                    else visible_refs.get(chunk.chunk_id)
                )
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
            items.append(
                ChunkMemoryItem(
                    ref=ref,
                    title=document.title,
                    chunk_position=chunk.chunk_pos,
                    has_been_read=True,
                    text=chunk.text,
                    previews=[],
                )
            )
        return items

    def _attempt_summary(self, record: StepRecord) -> dict[str, Any]:
        action: Any = (
            record.decision.action
            if record.decision is not None
            else (
                record.resolved_decision.action
                if record.resolved_decision is not None
                else None
            )
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
            "message": (
                record.observation.message if record.observation is not None else None
            ),
        }

    @staticmethod
    def _latest_attempt_summary(attempt: dict[str, Any]) -> dict[str, Any]:
        return {
            "policy_attempt": attempt["policy_attempt"],
            "submitted_action": attempt["action"],
            "outcome": attempt["outcome"],
            "error_code": attempt["error_code"],
            "message": attempt["message"],
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
        if isinstance(action, ExpandAction):
            return {
                "type": "EXPAND",
                "kind": action.kind.value,
                "source_ref": action.source_ref,
                "query": action.query,
                "direction": action.direction.value if action.direction is not None else None,
            }
        if isinstance(action, ResolvedExpandAction):
            return {
                "type": "EXPAND",
                "kind": action.kind.value,
                "source": self._node_label(action.source_id),
                "query": action.query,
                "direction": action.direction.value if action.direction is not None else None,
            }
        if isinstance(action, ResolvedReadAction):
            return {"type": "READ", "chunk": self._node_label(action.chunk_id)}
        if isinstance(action, ReadAction):
            return {"type": "READ", "chunk_ref": action.chunk_ref}
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
                "evidence_refs": list(action.evidence_refs),
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
