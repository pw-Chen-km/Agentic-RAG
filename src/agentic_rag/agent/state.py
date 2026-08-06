"""Deterministic controller-state updates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentic_rag.agent.models import (
    ChunkHandle,
    ControllerState,
    EvidenceAssessment,
    EntityHandle,
    HandleSentencePreview,
    Observation,
    SentenceHandle,
)
from agentic_rag.storage import Substrate


_VISIBILITY_FIELDS = (
    "visible_entity_ids",
    "visible_sentence_ids",
    "visible_chunk_ids",
    "eligible_sentence_ids",
    "read_chunk_ids",
)

_SUMMARY_LIMIT = 160


class StateUpdater:
    def __init__(
        self,
        substrate: Substrate | None = None,
        *,
        accumulate_selected_evidence: bool = False,
    ) -> None:
        self.substrate = substrate
        self.accumulate_selected_evidence = accumulate_selected_evidence

    def apply(
        self,
        state: ControllerState,
        *,
        assessment: EvidenceAssessment | None,
        observation: Observation,
        action_signature: str | None,
        commit_assessment: bool = True,
        consume_step: bool = True,
    ) -> ControllerState:
        updated = state.model_copy(deep=True)
        updated.policy_attempts += 1
        updated.remaining_policy_attempt_budget = max(
            0, updated.remaining_policy_attempt_budget - 1
        )
        if consume_step:
            updated.step += 1
            updated.remaining_step_budget = max(
                0, updated.remaining_step_budget - 1
            )
        updated.remaining_retrieved_token_budget = max(
            0,
            updated.remaining_retrieved_token_budget
            - observation.retrieved_tokens,
        )
        if observation.metadata.get("retrieved_budget_exhausted") is True:
            updated.remaining_retrieved_token_budget = 0

        delta = observation.metadata.get("visibility_delta", {})
        if isinstance(delta, dict):
            for field_name in _VISIBILITY_FIELDS:
                raw_ids = delta.get(field_name, [])
                if isinstance(raw_ids, (list, tuple, set, frozenset)):
                    getattr(updated, field_name).update(
                        str(node_id) for node_id in raw_ids
                    )
            ordered_candidates = [
                *observation.novel_node_ids,
                *(
                    str(node_id)
                    for field_name in (
                        "visible_entity_ids",
                        "visible_sentence_ids",
                        "visible_chunk_ids",
                    )
                    for node_id in delta.get(field_name, [])
                ),
            ]
            known_memory_ids = set(updated.semantic_memory_node_ids)
            for node_id in ordered_candidates:
                if node_id in known_memory_ids:
                    continue
                known_memory_ids.add(node_id)
                updated.semantic_memory_node_ids.append(node_id)

        if action_signature is not None:
            updated.action_signatures.add(action_signature)
        if commit_assessment and assessment is not None:
            updated.last_assessment = assessment
            if self.accumulate_selected_evidence:
                updated.selected_evidence_refs = _merge_evidence_refs(
                    updated.selected_evidence_refs,
                    assessment.selected_evidence_refs,
                )
            else:
                updated.selected_evidence_refs = list(
                    assessment.selected_evidence_refs
                )
        updated.newest_observation = observation
        self._update_node_handles(updated, observation)
        return updated
    def _update_node_handles(
        self,
        state: ControllerState,
        observation: Observation,
    ) -> None:
        if self.substrate is None:
            return

        registry = state.handle_registry
        # Allocate handles by type and stable-ID order. Counters are independent
        # per type, so the same episode and observations always produce the same
        # S#/C#/E# mapping without changing the substrate IDs used by control
        # state.
        for entity_id in sorted(state.visible_entity_ids):
            registry.register(entity_id, "ENTITY")
        for chunk_id in sorted(state.visible_chunk_ids):
            registry.register(chunk_id, "CHUNK")
        for sentence_id in sorted(state.visible_sentence_ids):
            registry.register(sentence_id, "SENTENCE")

        raw_previews = _preview_text_by_chunk(observation.results)
        previews = {
            chunk_id: [
                HandleSentencePreview(
                    sentence_id=registry.register(
                        sentence_id, "SENTENCE"
                    ),
                    text=text,
                )
                for sentence_id, text in items
            ]
            for chunk_id, items in raw_previews.items()
        }
        for entity_id in sorted(state.visible_entity_ids):
            entity = self.substrate.entity_by_id.get(entity_id)
            if entity is None:
                continue
            state.node_handles[entity_id] = EntityHandle(
                id=registry.register(entity_id, "ENTITY"),
                label=entity.canonical_name,
                entity_type=entity.entity_type,
            )

        for sentence_id in sorted(state.visible_sentence_ids):
            sentence = self.substrate.sentence_by_id.get(sentence_id)
            if sentence is None:
                continue
            chunk = self.substrate.chunk_by_id[sentence.chunk_id]
            document = self.substrate.document_by_id[chunk.doc_id]
            state.node_handles[sentence_id] = SentenceHandle(
                id=registry.register(sentence_id, "SENTENCE"),
                text=_summary(sentence.text),
                parent_chunk_id=registry.register(
                    chunk.chunk_id, "CHUNK"
                ),
                document_id=document.doc_id,
                title=document.title,
                can_use_as_evidence=(
                    sentence_id in state.eligible_sentence_ids
                ),
            )

        for chunk_id in sorted(state.visible_chunk_ids):
            chunk = self.substrate.chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            document = self.substrate.document_by_id[chunk.doc_id]
            previous = state.node_handles.get(chunk_id)
            previous_previews = (
                list(previous.previews)
                if isinstance(previous, ChunkHandle)
                else []
            )
            current_previews = previews.get(chunk_id, [])
            state.node_handles[chunk_id] = ChunkHandle(
                id=registry.register(chunk_id, "CHUNK"),
                document_id=document.doc_id,
                title=document.title,
                has_been_read=chunk_id in state.read_chunk_ids,
                can_use_as_evidence=chunk_id in state.read_chunk_ids,
                previews=(
                    current_previews
                    if current_previews
                    else previous_previews
                )[:2],
            )


def _merge_evidence_refs(first: list, second: list) -> list:
    """Preserve first-seen evidence order while removing typed duplicates."""

    merged = []
    seen: set[tuple[object, str]] = set()
    for ref in [*first, *second]:
        key = (ref.unit, ref.id)
        if key in seen:
            continue
        seen.add(key)
        merged.append(ref)
    return merged


def _summary(text: str) -> str:
    if len(text) <= _SUMMARY_LIMIT:
        return text
    return text[: _SUMMARY_LIMIT - 1].rstrip() + "…"


def _preview_text_by_chunk(
    results: list[dict[str, Any]],
) -> dict[str, list[tuple[str, str]]]:
    previews: dict[str, list[tuple[str, str]]] = {}

    def visit(
        value: Any,
        *,
        preview_context: bool = False,
        inherited_chunk_id: str | None = None,
    ) -> None:
        if isinstance(value, list):
            for item in value:
                visit(
                    item,
                    preview_context=preview_context,
                    inherited_chunk_id=inherited_chunk_id,
                )
            return
        if not isinstance(value, Mapping):
            return

        navigation = preview_context or bool(
            value.get("navigation_only", False)
        )
        sentence_id = value.get("sentence_id")
        text = value.get("text")
        local_chunk_id = value.get(
            "parent_chunk_id",
            value.get("chunk_id", value.get("bridge_chunk_id")),
        )
        chunk_id = (
            local_chunk_id
            if isinstance(local_chunk_id, str)
            else inherited_chunk_id
        )
        if (
            navigation
            and isinstance(sentence_id, str)
            and isinstance(text, str)
            and isinstance(chunk_id, str)
        ):
            bucket = previews.setdefault(chunk_id, [])
            if sentence_id not in {item[0] for item in bucket}:
                bucket.append((sentence_id, _summary(text)))

        for key, child in value.items():
            visit(
                child,
                preview_context=(
                    navigation
                    or key
                    in {
                        "previews",
                        "sentence_previews",
                        "trigger_sentences",
                        "bridge_previews",
                    }
                ),
                inherited_chunk_id=chunk_id,
            )

    visit(results)
    return previews
