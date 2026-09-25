"""Deterministic episode-state transitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentic_rag.agent.models import Assessment, EpisodeState, Observation, SentencePreview
from agentic_rag.agent.entity_visibility import EntityVisibilityPolicy
from agentic_rag.agent.interface import InterfaceContract
from agentic_rag.substrate.storage import Substrate

_VISIBILITY_FIELDS = (
    "visible_entity_ids",
    "visible_sentence_ids",
    "visible_chunk_ids",
    "visible_passage_ids",
    "eligible_sentence_ids",
    "read_chunk_ids",
)
_SUMMARY_LIMIT = 160


class StateUpdater:
    """Pure transition logic used only by the episode State Manager."""

    def __init__(
        self, substrate: Substrate, interface_contract: InterfaceContract | None = None
    ) -> None:
        self.substrate = substrate
        self.interface_contract = interface_contract
        self.entity_visibility_policy = EntityVisibilityPolicy(substrate)

    def apply(
        self,
        state: EpisodeState,
        *,
        assessment: Assessment | None,
        observation: Observation,
        action_signature: str | None,
        scope_id: str | None = None,
        commit_assessment: bool = True,
        consume_step: bool = True,
        consume_policy_attempt: bool = True,
    ) -> EpisodeState:
        updated = state.model_copy(deep=True)
        if consume_policy_attempt:
            updated.policy_attempts += 1
            updated.remaining_policy_attempt_budget = max(
                0, updated.remaining_policy_attempt_budget - 1
            )
        if consume_step:
            updated.step += 1
            updated.remaining_step_budget = max(0, updated.remaining_step_budget - 1)
        updated.remaining_retrieved_token_budget = max(
            0,
            updated.remaining_retrieved_token_budget - observation.retrieved_tokens,
        )
        if observation.metadata.get("retrieved_budget_exhausted") is True:
            updated.remaining_retrieved_token_budget = 0

        delta = observation.metadata.get("visibility_delta", {})
        if isinstance(delta, dict):
            for field_name in _VISIBILITY_FIELDS:
                values = delta.get(field_name, [])
                if isinstance(values, (list, tuple, set, frozenset)):
                    getattr(updated, field_name).update(str(item) for item in values)
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
            known = set(updated.semantic_memory_node_ids)
            for node_id in ordered_candidates:
                if node_id not in known:
                    known.add(node_id)
                    updated.semantic_memory_node_ids.append(node_id)

        if action_signature is not None:
            updated.action_signatures.add(action_signature)
        if commit_assessment and assessment is not None:
            updated.last_assessment = assessment.model_copy(deep=True)
        updated.newest_observation = observation
        self._update_references_and_previews(updated, observation, scope_id=scope_id)
        return updated

    def _update_references_and_previews(
        self, state: EpisodeState, observation: Observation, *, scope_id: str | None
    ) -> None:
        registry = state.reference_registry
        entity_ids = state.visible_entity_ids
        interface_contract = getattr(self, "interface_contract", None)
        if interface_contract is not None:
            if not interface_contract.entity_annotation:
                entity_ids = set()
            elif scope_id is not None:
                landing = interface_contract.entity_continuation.value
                entity_ids, _ = self.entity_visibility_policy.evaluate(
                    state, scope_id,
                    landing=landing if landing in {"chunk", "sentence"} else None,
                )
        for entity_id in sorted(entity_ids):
            registry.register(entity_id, "ENTITY")
        for chunk_id in sorted(state.visible_chunk_ids):
            registry.register(chunk_id, "CHUNK")
        for sentence_id in sorted(state.visible_sentence_ids):
            registry.register(sentence_id, "SENTENCE")

        for chunk_id, previews in _preview_text_by_chunk(observation.results).items():
            projected = [
                SentencePreview(
                    sentence_ref=registry.register(sentence_id, "SENTENCE"),
                    text=text,
                )
                for sentence_id, text in previews
            ]
            if projected:
                state.chunk_previews[chunk_id] = projected[:2]


def _summary(text: str) -> str:
    return text if len(text) <= _SUMMARY_LIMIT else text[: _SUMMARY_LIMIT - 1].rstrip() + "…"


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
        navigation = preview_context or bool(value.get("navigation_only", False))
        sentence_id = value.get("sentence_id")
        text = value.get("text")
        local_chunk_id = value.get(
            "parent_chunk_id", value.get("chunk_id", value.get("bridge_chunk_id"))
        )
        chunk_id = local_chunk_id if isinstance(local_chunk_id, str) else inherited_chunk_id
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
                    or key in {"previews", "sentence_previews", "trigger_sentences", "bridge_previews"}
                ),
                inherited_chunk_id=chunk_id,
            )

    visit(results)
    return previews
