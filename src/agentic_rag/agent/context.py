"""Build the only Policy-visible state: semantic memory plus compact control signals."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agentic_rag.agent.action_schema import (
    ActionSchemaBuilder,
    decision_schema_sha256,
    native_action_tools,
)
from agentic_rag.agent.action_space import AvailableActionSpaceBuilder
from agentic_rag.agent.evidence_cards import EvidenceCard, EvidenceCardView, EvidenceRegistry
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
    Usage,
)
from agentic_rag.agent.policy import policy_decision_model
from agentic_rag.agent.protocol import (
    render_action_protocol,
    render_available_action_options,
)
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.reader import EvidenceReader, reader_items_as_dicts
from agentic_rag.substrate.storage import Substrate


@dataclass(frozen=True, slots=True)
class BuiltPolicyContext:
    messages: list[Message]
    policy_view: PolicyView
    reference_map: ContextReferenceMap
    available_action_space: AvailableActionSpace
    decision_format: type[BaseModel]
    decision_schema_sha256: str
    native_tools: list[dict[str, Any]]
    native_tool_models: dict[str, type[BaseModel]]
    rendered_context: dict[str, Any] | None = None
    reader_usage: Usage = field(default_factory=Usage)

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
        observation_mode: str = "raw",
        reader: EvidenceReader | None = None,
    ) -> None:
        self.substrate = substrate
        self.enabled_expansions = tuple(enabled_expansions)
        self.show_available_action_options = show_available_action_options
        self.use_state_conditioned_schema = use_state_conditioned_schema
        if observation_mode not in {"raw", "program", "reader", "reader_assessed"}:
            raise ValueError("unsupported observation_mode")
        self.observation_mode = observation_mode
        self.reader = reader
        self.evidence_registry = EvidenceRegistry(substrate)
        self.action_space_builder = AvailableActionSpaceBuilder(
            self.enabled_expansions
        )
        self.action_schema_builder = ActionSchemaBuilder()

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
        available_action_space = self.action_space_builder.build(
            state,
            reference_map,
            mode=action_space_mode,
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
        protocol = render_action_protocol(self.enabled_expansions)
        rendered_context: dict[str, Any] | None = None
        reader_usage = Usage()
        decision_format = (
            self.action_schema_builder.build(available_action_space)
            if self.use_state_conditioned_schema
            else policy_decision_model(self.enabled_expansions)
        )
        native_tools, native_tool_models = native_action_tools(available_action_space)
        if self.observation_mode == "raw":
            if self.show_available_action_options:
                protocol = (
                    f"{protocol}\n\n"
                    f"{render_available_action_options(available_action_space)}"
                )
        else:
            card_view, reader_usage = self._build_evidence_view(
                query=query,
                state=state,
                trajectory=trajectory,
                action_space=available_action_space,
            )
            # Reader-organized modes expose only the sources selected by the
            # Reader.  Keep the action schema in lockstep with that filtered
            # view so the model cannot select a hidden READ/EXPAND/FINISH ref.
            if self.observation_mode in {"reader", "reader_assessed"}:
                available_action_space = self._filter_action_space_for_cards(
                    available_action_space, card_view
                )
            decision_format = (
                self.action_schema_builder.build(available_action_space)
                if self.use_state_conditioned_schema
                else policy_decision_model(self.enabled_expansions)
            )
            native_tools, native_tool_models = native_action_tools(available_action_space)
            rendered_context = self._render_observation_context(
                state=state,
                trajectory=trajectory,
                cards=card_view,
                action_space=available_action_space,
            )
            if self.observation_mode == "reader_assessed":
                sufficiency = rendered_context.get("reader_sufficiency")
                if sufficiency is not None:
                    rendered_context["reader_sufficiency"] = sufficiency
            context_text = self._render_context_text(rendered_context)
            messages = [
                Message(role="system", content=protocol),
                Message(
                    role="system",
                    content=(
                        "Current retrieval skill (plain Markdown):\n\n"
                        f"{skill_text}"
                    ),
                ),
                Message(role="user", content=f"Original question:\n{query}"),
                Message(role="user", content=context_text),
            ]
            return BuiltPolicyContext(
                messages=messages,
                policy_view=view,
                reference_map=reference_map,
                available_action_space=available_action_space,
                decision_format=decision_format,
                decision_schema_sha256=decision_schema_sha256(decision_format),
                native_tools=native_tools,
                native_tool_models=native_tool_models,
                rendered_context=rendered_context,
                reader_usage=reader_usage,
            )
        messages = [
            Message(
                role="system",
                content=protocol,
            ),
            Message(
                role="system",
                content=(
                    "Current retrieval skill (plain Markdown):\n\n"
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
        return BuiltPolicyContext(
            messages=messages,
            policy_view=view,
            reference_map=reference_map,
            available_action_space=available_action_space,
            decision_format=decision_format,
            decision_schema_sha256=decision_schema_sha256(decision_format),
            native_tools=native_tools,
            native_tool_models=native_tool_models,
            rendered_context=rendered_context,
            reader_usage=reader_usage,
        )

    def _build_evidence_view(
        self,
        *,
        query: str,
        state: EpisodeState,
        trajectory: Sequence[StepRecord],
        action_space: AvailableActionSpace,
    ) -> tuple[EvidenceCardView, Usage]:
        reader_items: tuple[dict[str, Any], ...] = ()
        reader_error: str | None = None
        reader_usage = Usage()
        observation = state.newest_observation
        if (
            self.observation_mode in {"reader", "reader_assessed"}
            and self.reader is not None
            and observation is not None
            and observation.results
        ):
            try:
                previous_view = self.evidence_registry.cards(
                    state,
                    action_space,
                    tuple(trajectory),
                    mode="program",
                )
                previous = (
                    self._render_previous_context(state, trajectory)
                    + "\n"
                    + self.evidence_registry.render(previous_view)
                )
                result = self.reader.read(
                    question=query,
                    previous_context=previous,
                    observation=observation,
                    reader_assessment=self.observation_mode == "reader_assessed",
                )
                reader_items = reader_items_as_dicts(result)
                reader_usage = getattr(self.reader, "last_usage", Usage())
                reader_view = self.evidence_registry.cards(
                    state,
                    action_space,
                    tuple(trajectory),
                    reader_items=self._reader_history(trajectory) + reader_items,
                    mode=self.observation_mode,
                    reader_only=True,
                )
                if result.sufficiency is not None:
                    reader_view = EvidenceCardView(
                        cards=reader_view.cards,
                        mode=reader_view.mode,
                        reader_error=reader_view.reader_error,
                        reader_sufficiency=result.sufficiency.model_dump(mode="json"),
                    )
                return reader_view, reader_usage
            except Exception as exc:
                reader_error = str(exc)[:240]
                reader_usage = getattr(self.reader, "last_usage", Usage())
        return (
            self.evidence_registry.cards(
                state,
                action_space,
                tuple(trajectory),
                reader_items=self._reader_history(trajectory),
                mode=self.observation_mode,
                reader_error=reader_error,
                reader_only=self.observation_mode in {"reader", "reader_assessed"},
            ),
            reader_usage,
        )

    def _render_observation_context(
        self,
        *,
        state: EpisodeState,
        trajectory: Sequence[StepRecord],
        cards: EvidenceCardView,
        action_space: AvailableActionSpace,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "observation_mode": self.observation_mode,
            "previous_assessment": (
                state.last_assessment.model_dump(mode="json")
                if state.last_assessment is not None else None
            ),
            "previous_action": (
                self._compact_previous_action(trajectory[-1]) if trajectory else None
            ),
            "evidence_cards": cards.to_dict(),
            "global_search": "SEARCH is available; choose a legal method/target pair and write a query.",
            "remaining_budget": {
                "steps": state.remaining_step_budget,
                "policy_attempts": state.remaining_policy_attempt_budget,
                "retrieval_tokens": state.remaining_retrieved_token_budget,
            },
        }
        if cards.reader_sufficiency is not None:
            value["reader_sufficiency"] = cards.reader_sufficiency
        return value

    @staticmethod
    def _reader_history(
        trajectory: Sequence[StepRecord],
    ) -> tuple[dict[str, Any], ...]:
        """Recover prior Reader selections without exposing their raw text.

        Reader summaries live in the per-step rendered context for audit and
        are carried into the next turn as compact claim/quote records.  The
        original text/preview fields are deliberately ignored here.
        """

        history: dict[str, dict[str, Any]] = {}
        for record in trajectory:
            context = record.rendered_context or {}
            cards = context.get("evidence_cards", {})
            if not isinstance(cards, dict):
                continue
            for card in cards.get("cards", []) or []:
                if not isinstance(card, dict):
                    continue
                ref = card.get("ref")
                claim = card.get("reader_claim")
                quote = card.get("reader_quote")
                if not isinstance(ref, str) or not claim or not quote:
                    continue
                history[ref] = {
                    "source_ref": ref,
                    "role": card.get("reader_role") or "context",
                    "claim": str(claim),
                    "quote": str(quote),
                    "confidence": card.get("reader_confidence", "medium"),
                }
        return tuple(history.values())

    @staticmethod
    def _filter_action_space_for_cards(
        action_space: AvailableActionSpace,
        cards: EvidenceCardView,
    ) -> AvailableActionSpace:
        """Restrict source-specific actions to Reader-selected cards."""

        visible_refs = {card.ref for card in cards.cards}
        expand_options = tuple(
            option.model_copy(
                update={
                    "source_refs": tuple(
                        ref for ref in option.source_refs if ref in visible_refs
                    )
                }
            )
            for option in action_space.expand_options
            if any(ref in visible_refs for ref in option.source_refs)
        )
        return action_space.model_copy(
            update={
                "expand_options": expand_options,
                "read_refs": tuple(
                    ref for ref in action_space.read_refs if ref in visible_refs
                ),
                "finish_evidence_refs": tuple(
                    ref
                    for ref in action_space.finish_evidence_refs
                    if ref in visible_refs
                ),
            }
        )

    def _render_context_text(self, context: dict[str, Any]) -> str:
        previous = context.get("previous_assessment")
        action = context.get("previous_action")
        lines = [
            "Previous assessment — model judgment, not verified fact:",
            json.dumps(previous, ensure_ascii=False) if previous else "None",
            "\nPrevious action and outcome:",
            json.dumps(action, ensure_ascii=False) if action else "None",
            "\n" + self.evidence_registry.render(
                EvidenceCardView(
                    cards=tuple(
                        EvidenceCard(**item)
                        for item in context["evidence_cards"].get("cards", [])
                    ),
                    mode=context["evidence_cards"].get("mode", "program"),
                    reader_error=context["evidence_cards"].get("reader_error"),
                    reader_sufficiency=context["evidence_cards"].get("reader_sufficiency"),
                )
            ),
            "\nGlobal SEARCH instruction:",
            str(context["global_search"]),
        ]
        if context.get("reader_sufficiency") is not None:
            lines.extend([
                "\nReader assessment — advisory only:",
                json.dumps(context["reader_sufficiency"], ensure_ascii=False),
            ])
        budget = context["remaining_budget"]
        lines.extend([
            "\nRemaining budget:",
            f"- steps: {budget['steps']}",
            f"- policy attempts: {budget['policy_attempts']}",
            f"- retrieval tokens: {budget['retrieval_tokens']}",
        ])
        return "\n".join(lines)

    def _render_previous_context(
        self, state: EpisodeState, trajectory: Sequence[StepRecord]
    ) -> str:
        return json.dumps(
            {
                "previous_assessment": (
                    state.last_assessment.model_dump(mode="json")
                    if state.last_assessment else None
                ),
                "previous_action": (
                    self._compact_previous_action(trajectory[-1]) if trajectory else None
                ),
            },
            ensure_ascii=False,
        )

    def _compact_previous_action(self, record: StepRecord) -> dict[str, Any]:
        attempt = self._attempt_summary(record)
        compact = {
            "action": attempt.get("action"),
            "outcome": attempt.get("outcome"),
            "error_code": attempt.get("error_code"),
        }
        message = attempt.get("message")
        if message and attempt.get("error_code"):
            compact["message"] = str(message)[:240]
        return compact

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
