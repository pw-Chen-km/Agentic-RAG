"""Evidence cards used to render compact, source-grounded policy context.

The registry is deliberately deterministic.  It never decides which source is
"important" and it never writes a recommendation for the Agent.  It only
joins stable references with the text and operations that are already legal
for the current state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from agentic_rag.agent.models import (
    AvailableActionSpace,
    EpisodeState,
    StepRecord,
)
from agentic_rag.substrate.storage import Substrate


@dataclass(frozen=True, slots=True)
class EvidenceCard:
    ref: str
    node_type: str
    source: str | None = None
    text: str | None = None
    preview: str | None = None
    name: str | None = None
    parent_ref: str | None = None
    new_this_turn: bool = False
    can_do: tuple[str, ...] = ()
    reader_claim: str | None = None
    reader_role: str | None = None
    reader_quote: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "ref": self.ref,
            "node_type": self.node_type,
            "new_this_turn": self.new_this_turn,
            "can_do": list(self.can_do),
        }
        for key in (
            "source", "text", "preview", "name", "parent_ref",
            "reader_claim", "reader_role", "reader_quote",
        ):
            item = getattr(self, key)
            if item is not None:
                value[key] = item
        return value


@dataclass(frozen=True, slots=True)
class EvidenceCardView:
    """Rendered card set and audit metadata for one policy turn."""

    cards: tuple[EvidenceCard, ...] = ()
    mode: str = "program"
    reader_error: str | None = None
    reader_sufficiency: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "cards": [card.to_dict() for card in self.cards],
            "reader_error": self.reader_error,
            "reader_sufficiency": self.reader_sufficiency,
        }


class EvidenceRegistry:
    """Build one deduplicated source view from the current State snapshot."""

    def __init__(self, substrate: Substrate) -> None:
        self.substrate = substrate

    def cards(
        self,
        state: EpisodeState,
        action_space: AvailableActionSpace,
        trajectory: tuple[StepRecord, ...] | list[StepRecord] = (),
        *,
        reader_items: tuple[dict[str, Any], ...] = (),
        mode: str = "program",
        reader_error: str | None = None,
    ) -> EvidenceCardView:
        display_ids = self._display_ids(state)
        new_ids = self._new_ids(state)
        operations = self._operations(action_space)
        reader_by_ref = {
            str(item.get("source_ref")): item
            for item in reader_items
            if isinstance(item, dict) and item.get("source_ref")
        }
        result: list[EvidenceCard] = []
        for stable_id in display_ids:
            ref = state.reference_registry.ref_for(stable_id)
            if ref is None:
                continue
            card = self._card(stable_id, ref, state, new_ids, operations)
            item = reader_by_ref.get(ref)
            if item is not None:
                card = replace(
                    card,
                    reader_claim=str(item.get("claim")) if item.get("claim") else None,
                    reader_role=str(item.get("role")) if item.get("role") else None,
                    reader_quote=str(item.get("quote")) if item.get("quote") else None,
                )
            result.append(card)
        return EvidenceCardView(tuple(result), mode=mode, reader_error=reader_error)

    def render(self, view: EvidenceCardView) -> str:
        lines = ["Evidence cards:"]
        if not view.cards:
            lines.append("- None yet. Use SEARCH to obtain the first source.")
        for card in view.cards:
            marker = " | new" if card.new_this_turn else ""
            lines.append(f"\n{card.ref} [{card.node_type.title()}{marker}]")
            if card.source:
                lines.append(f"source: {card.source}")
            if card.name:
                lines.append(f"name: {card.name}")
            if card.parent_ref:
                lines.append(f"found in: {card.parent_ref}")
            if card.text:
                lines.append(f"text: {card.text}")
            elif card.preview:
                lines.append(f"preview: {card.preview}")
            if card.reader_claim:
                lines.append(f"reader summary: {card.reader_claim}")
            if card.reader_quote and card.reader_quote != card.text:
                lines.append(f"reader quote: {card.reader_quote}")
            if card.can_do:
                lines.append("can do: " + "; ".join(card.can_do))
        if view.reader_error:
            lines.append(f"\nReader status: unavailable ({view.reader_error})")
        return "\n".join(lines)

    def _card(
        self,
        stable_id: str,
        ref: str,
        state: EpisodeState,
        new_ids: set[str],
        operations: dict[str, tuple[str, ...]],
    ) -> EvidenceCard:
        if stable_id in self.substrate.entity_by_id:
            entity = self.substrate.entity_by_id[stable_id]
            parent_ref = self._entity_parent_ref(stable_id, state)
            return EvidenceCard(
                ref=ref,
                node_type="ENTITY",
                name=entity.canonical_name,
                parent_ref=parent_ref,
                new_this_turn=stable_id in new_ids,
                can_do=operations.get(ref, ()),
            )
        if stable_id in self.substrate.sentence_by_id:
            sentence = self.substrate.sentence_by_id[stable_id]
            chunk = self.substrate.chunk_by_id[sentence.chunk_id]
            document = self.substrate.document_by_id[chunk.doc_id]
            return EvidenceCard(
                ref=ref,
                node_type="SENTENCE",
                source=document.title,
                text=sentence.text,
                parent_ref=state.reference_registry.ref_for(sentence.chunk_id),
                new_this_turn=stable_id in new_ids,
                can_do=operations.get(ref, ()),
            )
        chunk = self.substrate.chunk_by_id[stable_id]
        document = self.substrate.document_by_id[chunk.doc_id]
        previews = state.chunk_previews.get(stable_id, [])
        return EvidenceCard(
            ref=ref,
            node_type="CHUNK",
            source=document.title,
            text=chunk.text if stable_id in state.read_chunk_ids else None,
            preview=" ".join(item.text for item in previews[:2]) or None,
            new_this_turn=stable_id in new_ids,
            can_do=operations.get(ref, ()),
        )

    def _operations(self, action_space: AvailableActionSpace) -> dict[str, tuple[str, ...]]:
        operations: dict[str, list[str]] = {}
        for ref in action_space.read_refs:
            operations.setdefault(ref, []).append(f"READ {ref}")
        for option in action_space.expand_options:
            for ref in option.source_refs:
                suffix = (
                    " (" + ", ".join(direction.value for direction in option.directions) + ")"
                    if option.directions else ""
                )
                operations.setdefault(ref, []).append(
                    f"EXPAND {option.kind.value}{suffix}"
                )
        for ref in action_space.finish_evidence_refs:
            operations.setdefault(ref, []).append(f"FINISH with {ref}")
        return {ref: tuple(values) for ref, values in operations.items()}

    @staticmethod
    def _new_ids(state: EpisodeState) -> set[str]:
        observation = state.newest_observation
        if observation is None:
            return set()
        values = set(observation.novel_node_ids)
        delta = observation.metadata.get("visibility_delta", {})
        if isinstance(delta, dict):
            for key in ("visible_entity_ids", "visible_sentence_ids", "visible_chunk_ids"):
                values.update(str(item) for item in delta.get(key, []) or [])
        return values

    @staticmethod
    def _display_ids(state: EpisodeState) -> list[str]:
        visible = (
            state.visible_entity_ids
            | state.visible_sentence_ids
            | state.visible_chunk_ids
        )
        ordered = [item for item in state.semantic_memory_node_ids if item in visible]
        seen = set(ordered)
        ordered.extend(sorted(visible - seen))
        return [
            item for item in ordered
            if item not in state.visible_sentence_ids
            or item in state.eligible_sentence_ids
        ]

    def _entity_parent_ref(self, entity_id: str, state: EpisodeState) -> str | None:
        for mention in self.substrate.mentions:
            if mention.entity_id != entity_id:
                continue
            if mention.sentence_id in state.visible_sentence_ids:
                return state.reference_registry.ref_for(mention.sentence_id)
        return None


def cards_json(view: EvidenceCardView) -> str:
    """Stable audit serialization."""

    return json.dumps(view.to_dict(), ensure_ascii=False, sort_keys=True)
