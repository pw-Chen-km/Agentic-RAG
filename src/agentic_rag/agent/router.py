"""Execute validated query-time actions against the Phase 1–2 substrate."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from agentic_rag.agent.expansion import ExpansionEngine
from agentic_rag.agent.models import (
    EpisodeState,
    Observation,
    ObservationStatus,
    ResolvedAction,
    ResolvedExpandAction,
    ResolvedReadAction,
    SearchAction,
)
from agentic_rag.errors import NodeNotFoundError
from agentic_rag.substrate.retrieval import Retriever
from agentic_rag.substrate.storage import Substrate

_TOKEN_RE = re.compile(r"(?u)\b\w+\b|[^\w\s]")
_TEXT_KEYS = frozenset(
    {
        "text",
        "title",
        "canonical_name",
        "target_canonical_name",
        "matched_alias",
        "bridge_text",
    }
)


class ActionRouter:
    def __init__(
        self,
        substrate: Substrate,
        retriever: Retriever,
        expansion_engine: ExpansionEngine,
    ) -> None:
        self.substrate = substrate
        self.retriever = retriever
        self.expansion_engine = expansion_engine

    def execute(
        self,
        action: ResolvedAction,
        state: EpisodeState,
        *,
        question: str,
        scope_id: str,
        action_id: str,
    ) -> Observation:
        if isinstance(action, SearchAction):
            hits = self.retriever.search(
                query=action.query,
                method=action.method.value,
                target=action.target.value,
                scope_id=scope_id,
                top_k=action.top_k,
            )
            raw_results = [
                hit.model_dump(mode="json") for hit in hits
            ]
            metadata: dict[str, Any] = {
                "method": action.method.value,
                "target": action.target.value,
                "query_used": action.query,
            }
        elif isinstance(action, ResolvedExpandAction):
            expanded = self.expansion_engine.expand(
                action, question, scope_id
            )
            raw_results = _json_results(expanded.results)
            metadata = {
                "kind": action.kind.value,
                "query_used": expanded.query_used,
                "source_degree": expanded.source_degree,
                "candidate_count_before_truncation": (
                    expanded.candidate_count_before_truncation
                ),
                "internal_request": _json_value(expanded.internal_request),
            }
        elif isinstance(action, ResolvedReadAction):
            self.substrate.require_scope(scope_id)
            if action.chunk_id not in self.substrate.chunk_ids_by_scope[scope_id]:
                raise NodeNotFoundError(
                    f"Chunk {action.chunk_id} is not present in scope {scope_id}"
                )
            read = self.substrate.read_chunk(action.chunk_id)
            raw_result = read.model_dump(mode="json")
            result_tokens = _count_result_tokens(raw_result)
            if result_tokens > state.remaining_retrieved_token_budget:
                return Observation(
                    action_id=action_id,
                    status=ObservationStatus.ERROR,
                    action=action,
                    error_code="retrieved_token_budget_exceeded",
                    message=(
                        "The complete Chunk does not fit in the remaining "
                        "retrieved-token budget"
                    ),
                    metadata={"retrieved_budget_exhausted": True},
                )
            delta = {
                "visible_chunk_ids": [read.chunk_id],
                "read_chunk_ids": [read.chunk_id],
                "visible_sentence_ids": [
                    item.sentence_id for item in read.sentences
                ],
                "eligible_sentence_ids": [
                    item.sentence_id for item in read.sentences
                ],
            }
            novel, already = _novelty(delta, state)
            return Observation(
                action_id=action_id,
                status=ObservationStatus.OK,
                action=action,
                results=[raw_result],
                retrieved_tokens=result_tokens,
                novel_node_ids=novel,
                already_seen_node_ids=already,
                metadata={"visibility_delta": delta},
            )
        else:
            raise TypeError(
                f"ActionRouter cannot execute {type(action).__name__}"
            )

        kept, retrieved_tokens, truncated = _fit_results(
            raw_results, state.remaining_retrieved_token_budget
        )
        if raw_results and not kept:
            return Observation(
                action_id=action_id,
                status=ObservationStatus.ERROR,
                action=action,
                error_code="retrieved_token_budget_exceeded",
                message=(
                    "No complete result fits in the remaining "
                    "retrieved-token budget"
                ),
                metadata={
                    **metadata,
                    "retrieved_budget_exhausted": True,
                },
            )

        delta = _visibility_delta(kept)
        novel, already = _novelty(delta, state)
        metadata["visibility_delta"] = delta
        metadata["truncated_by_retrieved_token_budget"] = truncated
        return Observation(
            action_id=action_id,
            status=ObservationStatus.OK,
            action=action,
            results=kept,
            retrieved_tokens=retrieved_tokens,
            novel_node_ids=novel,
            already_seen_node_ids=already,
            metadata=metadata,
        )


def _json_value(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _json_results(values: Iterable[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for value in values:
        converted = _json_value(value)
        if not isinstance(converted, dict):
            raise TypeError("Expansion results must serialize to JSON objects")
        results.append(converted)
    return results


def _fit_results(
    results: list[dict[str, Any]],
    token_budget: int,
) -> tuple[list[dict[str, Any]], int, bool]:
    kept: list[dict[str, Any]] = []
    used = 0
    for result in results:
        tokens = _count_result_tokens(result)
        if used + tokens > token_budget:
            break
        kept.append(result)
        used += tokens
    return kept, used, len(kept) != len(results)


def _count_result_tokens(value: Any, *, key: str | None = None) -> int:
    if isinstance(value, str):
        return len(_TOKEN_RE.findall(value)) if key in _TEXT_KEYS else 0
    if isinstance(value, Mapping):
        return sum(
            _count_result_tokens(child, key=str(child_key))
            for child_key, child in value.items()
        )
    if isinstance(value, list):
        return sum(_count_result_tokens(child, key=key) for child in value)
    return 0


def _visibility_delta(
    results: list[dict[str, Any]],
) -> dict[str, list[str]]:
    visible_entities: set[str] = set()
    visible_sentences: set[str] = set()
    visible_chunks: set[str] = set()
    eligible_sentences: set[str] = set()

    for result in results:
        _collect_visibility(
            result,
            visible_entities=visible_entities,
            visible_sentences=visible_sentences,
            visible_chunks=visible_chunks,
            eligible_sentences=eligible_sentences,
            navigation_only=bool(result.get("navigation_only", False)),
        )
    return {
        "visible_entity_ids": sorted(visible_entities),
        "visible_sentence_ids": sorted(visible_sentences),
        "visible_chunk_ids": sorted(visible_chunks),
        "eligible_sentence_ids": sorted(eligible_sentences),
        "read_chunk_ids": [],
    }


def _collect_visibility(
    value: Any,
    *,
    visible_entities: set[str],
    visible_sentences: set[str],
    visible_chunks: set[str],
    eligible_sentences: set[str],
    navigation_only: bool,
    key: str | None = None,
) -> None:
    if isinstance(value, list):
        child_navigation = navigation_only or key in {
            "previews",
            "sentence_previews",
            "trigger_sentences",
        }
        for child in value:
            _collect_visibility(
                child,
                visible_entities=visible_entities,
                visible_sentences=visible_sentences,
                visible_chunks=visible_chunks,
                eligible_sentences=eligible_sentences,
                navigation_only=child_navigation,
                key=key,
            )
        return
    if not isinstance(value, Mapping):
        return

    local_navigation = navigation_only or bool(
        value.get("navigation_only", False)
    )
    entity_id = value.get("entity_id", value.get("target_entity_id"))
    if isinstance(entity_id, str):
        visible_entities.add(entity_id)

    chunk_id = value.get(
        "parent_chunk_id",
        value.get("chunk_id", value.get("bridge_chunk_id")),
    )
    if isinstance(chunk_id, str):
        visible_chunks.add(chunk_id)

    sentence_id = value.get(
        "sentence_id", value.get("bridge_sentence_id")
    )
    if isinstance(sentence_id, str):
        visible_sentences.add(sentence_id)
        has_complete_text = any(
            isinstance(value.get(text_key), str)
            for text_key in ("text", "bridge_text")
        )
        if has_complete_text and not local_navigation:
            eligible_sentences.add(sentence_id)

    for child_key, child in value.items():
        if child_key in {
            "entity_id",
            "target_entity_id",
            "chunk_id",
            "parent_chunk_id",
            "bridge_chunk_id",
            "sentence_id",
            "bridge_sentence_id",
        }:
            continue
        _collect_visibility(
            child,
            visible_entities=visible_entities,
            visible_sentences=visible_sentences,
            visible_chunks=visible_chunks,
            eligible_sentences=eligible_sentences,
            navigation_only=local_navigation,
            key=str(child_key),
        )


def _novelty(
    delta: Mapping[str, list[str]],
    state: EpisodeState,
) -> tuple[list[str], list[str]]:
    previously_visible = (
        state.visible_entity_ids
        | state.visible_sentence_ids
        | state.visible_chunk_ids
    )
    now_visible = set(delta.get("visible_entity_ids", []))
    now_visible.update(delta.get("visible_sentence_ids", []))
    now_visible.update(delta.get("visible_chunk_ids", []))
    return (
        sorted(now_visible - previously_visible),
        sorted(now_visible & previously_visible),
    )
