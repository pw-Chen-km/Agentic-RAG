"""Policy observation projection with explicit visibility boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentic_rag.substrate.storage import Substrate


class ObservationProjector:
    """Project backend results after display truncation and before references.

    Stable substrate IDs are retained in the audit payload.  Only IDs whose
    complete mention span falls inside the displayed text are marked visible.
    """

    def __init__(self, substrate: Substrate, *, expose_entities: bool = False) -> None:
        self.substrate = substrate
        self.expose_entities = expose_entities

    def project(
        self,
        results: list[dict[str, Any]],
        *,
        action_type: str,
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
        projected: list[dict[str, Any]] = []
        audit_spans: list[dict[str, Any]] = []
        visible_entities: set[str] = set()
        visible_sentences: set[str] = set()
        visible_chunks: set[str] = set()
        eligible_sentences: set[str] = set()
        for original in results:
            item = _copy_mapping(original)
            self._project_value(
                item,
                action_type=action_type,
                navigation_context=False,
                visible_entities=visible_entities,
                visible_sentences=visible_sentences,
                visible_chunks=visible_chunks,
                eligible_sentences=eligible_sentences,
                audit_spans=audit_spans,
            )
            projected.append(item)
        delta = {
            "visible_entity_ids": sorted(visible_entities),
            "visible_sentence_ids": sorted(visible_sentences),
            "visible_chunk_ids": sorted(visible_chunks),
            "eligible_sentence_ids": sorted(eligible_sentences),
            "read_chunk_ids": [],
        }
        return projected, delta, {"visible_source_spans": audit_spans}

    def _project_value(
        self,
        value: Any,
        *,
        action_type: str,
        navigation_context: bool,
        visible_entities: set[str],
        visible_sentences: set[str],
        visible_chunks: set[str],
        eligible_sentences: set[str],
        audit_spans: list[dict[str, Any]],
    ) -> None:
        if isinstance(value, list):
            for child in value:
                self._project_value(
                    child,
                    action_type=action_type,
                    navigation_context=navigation_context,
                    visible_entities=visible_entities,
                    visible_sentences=visible_sentences,
                    visible_chunks=visible_chunks,
                    eligible_sentences=eligible_sentences,
                    audit_spans=audit_spans,
                )
            return
        if not isinstance(value, Mapping):
            return
        sentence_id = value.get("sentence_id")
        chunk_id = value.get("parent_chunk_id", value.get("chunk_id"))
        if isinstance(sentence_id, str) and sentence_id in self.substrate.sentence_by_id:
            visible_sentences.add(sentence_id)
            if (
                value.get("evidence_eligible", True) is not False
                and not value.get("navigation_only", False)
                and not navigation_context
            ):
                eligible_sentences.add(sentence_id)
            text = value.get("text")
            if isinstance(text, str):
                if action_type in {"SEARCH", "EXPAND"} and (
                    value.get("navigation_only", False) or navigation_context
                ):
                    text = text[:160]
                    value["text"] = text
                self._attach_mentions(
                    value,
                    sentence_id,
                    text,
                    visible_entities,
                    audit_spans,
                )
        if isinstance(chunk_id, str) and chunk_id in self.substrate.chunk_by_id:
            visible_chunks.add(chunk_id)
        for key, child in list(value.items()):
            self._project_value(
                child,
                action_type=action_type,
                navigation_context=navigation_context or key in {"previews", "bridge_previews", "sentence_previews"},
                visible_entities=visible_entities,
                visible_sentences=visible_sentences,
                visible_chunks=visible_chunks,
                eligible_sentences=eligible_sentences,
                audit_spans=audit_spans,
            )

    def _attach_mentions(
        self,
        value: dict[str, Any],
        sentence_id: str,
        displayed_text: str,
        visible_entities: set[str],
        audit_spans: list[dict[str, Any]],
    ) -> None:
        annotations: list[dict[str, Any]] = []
        for mention in self.substrate.mentions:
            if mention.sentence_id != sentence_id:
                continue
            complete = 0 <= mention.mention_start < mention.mention_end <= len(displayed_text)
            audit_spans.append(
                {
                    "sentence_id": sentence_id,
                    "entity_id": mention.entity_id,
                    "start": mention.mention_start,
                    "end": mention.mention_end,
                    "surface_form": mention.surface_form,
                    "visible": complete,
                }
            )
            if complete and self.expose_entities:
                visible_entities.add(mention.entity_id)
                annotations.append(
                    {
                        "entity_id": mention.entity_id,
                        "surface_form": mention.surface_form,
                        "start": mention.mention_start,
                        "end": mention.mention_end,
                    }
                )
        if self.expose_entities and annotations:
            value["visible_entity_mentions"] = annotations


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, child in value.items():
        if isinstance(child, Mapping):
            output[str(key)] = _copy_mapping(child)
        elif isinstance(child, list):
            output[str(key)] = [_copy_mapping(item) if isinstance(item, Mapping) else item for item in child]
        else:
            output[str(key)] = child
    return output
