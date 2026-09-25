"""Deterministic Policy-visible entity filtering for interface-study v6.2.

The substrate keeps every NER mention for provenance.  This module only
decides which already-visible entities are useful navigation references in the
current episode.  It never uses gold evidence or the answer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from agentic_rag.agent.models import EpisodeState
from agentic_rag.substrate.storage import Substrate


EXCLUDED_NER_TYPES = frozenset(
    {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "QUANTITY", "MONEY"}
)


class EntityVisibilityPolicy:
    """Compute visible and navigable entity IDs without changing the substrate."""

    version = "entity-navigation-filter-v1"

    def __init__(self, substrate: Substrate) -> None:
        self.substrate = substrate
        self._sentences_by_entity: dict[str, set[str]] = defaultdict(set)
        for mention in substrate.mentions:
            self._sentences_by_entity[mention.entity_id].add(mention.sentence_id)

    def evaluate(
        self,
        state: EpisodeState,
        scope_id: str,
        *,
        landing: str | None,
    ) -> tuple[set[str], list[dict[str, Any]]]:
        """Return Policy-visible entity IDs and an audit row for each visible entity.

        ``landing`` is ``"chunk"`` for passage navigation, ``"sentence"``
        for sentence navigation, and ``None`` for annotation-only.  The
        annotation-only mode still hides entities with no other unseen linked
        unit, because there is no useful structural annotation to expose.
        """

        self.substrate.require_scope(scope_id)
        scope_sentences = self.substrate.sentence_ids_by_scope[scope_id]
        scope_chunks = self.substrate.chunk_ids_by_scope[scope_id]
        visible_sentences = set(state.visible_sentence_ids)
        visible_passages = set(state.visible_passage_ids)
        navigable: set[str] = set()
        audit: list[dict[str, Any]] = []

        for entity_id in sorted(state.visible_entity_ids):
            entity = self.substrate.entity_by_id.get(entity_id)
            if entity is None:
                audit.append({
                    "entity_id": entity_id,
                    "filter_reason": "unknown_entity",
                    "visible": False,
                })
                continue
            entity_type = entity.entity_type.upper() if entity.entity_type else None
            linked_sentences = self._sentences_by_entity.get(entity_id, set()) & scope_sentences
            linked_passages = {
                self.substrate.sentence_by_id[sentence_id].chunk_id
                for sentence_id in linked_sentences
            } & scope_chunks
            unseen_sentences = linked_sentences - visible_sentences
            unseen_passages = linked_passages - visible_passages

            row: dict[str, Any] = {
                "entity_id": entity_id,
                "canonical_name": entity.canonical_name,
                "entity_type": entity.entity_type,
                "linked_sentence_count": len(linked_sentences),
                "linked_passage_count": len(linked_passages),
                "unseen_sentence_count": len(unseen_sentences),
                "unseen_passage_count": len(unseen_passages),
                "visible": True,
            }
            if entity_type in EXCLUDED_NER_TYPES:
                row["filter_reason"] = "excluded_ner_type"
                audit.append(row)
                continue

            if landing == "chunk":
                has_unseen_target = bool(unseen_passages)
                target_kind = "passage"
            elif landing == "sentence":
                has_unseen_target = bool(unseen_sentences)
                target_kind = "sentence"
            else:
                has_unseen_target = bool(unseen_passages or unseen_sentences)
                target_kind = "linked_unit"

            if not has_unseen_target:
                row["filter_reason"] = f"no_unseen_linked_{target_kind}"
                audit.append(row)
                continue
            row["filter_reason"] = None
            navigable.add(entity_id)
            audit.append(row)

        return navigable, audit
