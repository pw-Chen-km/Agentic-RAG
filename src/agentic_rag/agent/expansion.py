"""Scope-safe local graph expansion with query-conditioned ranking.

The engine deliberately owns graph traversal, not controller policy.  It accepts
any action object (or mapping) exposing ``kind``, ``source_id``, ``query``,
``direction`` and ``top_k`` so that it remains decoupled from the policy models.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.substrate.embedding import (
    EmbeddingBackend,
    create_embedding_backend,
    normalize_embeddings,
)
from agentic_rag.errors import AgenticRAGError, NodeNotFoundError
from agentic_rag.substrate.storage import Substrate
from agentic_rag.substrate.ranking import RankingService


ENTITY_MENTIONED_IN_SENTENCE = "ENTITY_MENTIONED_IN_SENTENCE"
SENTENCE_MENTIONS_ENTITY = "SENTENCE_MENTIONS_ENTITY"
ENTITY_CO_OCCURS_ENTITY_SENTENCE = "ENTITY_CO_OCCURS_ENTITY_SENTENCE"
CHUNK_ADJACENT_CHUNK = "CHUNK_ADJACENT_CHUNK"
ENTITY_MENTIONED_IN_CHUNK = "ENTITY_MENTIONED_IN_CHUNK"
ENTITY_CO_OCCURS_ENTITY_CHUNK = "ENTITY_CO_OCCURS_ENTITY_CHUNK"
CHUNK_CONTAINS_SENTENCE = "CHUNK_CONTAINS_SENTENCE"
CHUNK_MENTIONS_ENTITY = "CHUNK_MENTIONS_ENTITY"

SUPPORTED_EXPANSION_KINDS: frozenset[str] = frozenset(
    {
        ENTITY_MENTIONED_IN_SENTENCE,
        SENTENCE_MENTIONS_ENTITY,
        ENTITY_CO_OCCURS_ENTITY_SENTENCE,
        CHUNK_ADJACENT_CHUNK,
        ENTITY_MENTIONED_IN_CHUNK,
        ENTITY_CO_OCCURS_ENTITY_CHUNK,
        CHUNK_CONTAINS_SENTENCE,
        CHUNK_MENTIONS_ENTITY,
    }
)


class ExpansionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InternalExpansionRequest(ExpansionRecord):
    source_type: Literal["ENTITY", "SENTENCE", "CHUNK"]
    relation: Literal["MENTIONED_IN", "MENTIONS", "CO_OCCURS", "CONTAINS", "ADJACENT"]
    target: Literal["ENTITY", "SENTENCE", "CHUNK"]
    scope: Literal["SENTENCE", "CHUNK"] | None = None
    direction: Literal["PREV", "NEXT", "BOTH"] | None = None


class ExpansionPath(ExpansionRecord):
    """One concrete substrate path supporting a returned candidate."""

    relation: str
    node_ids: list[str]
    bridge_sentence_ids: list[str] = Field(default_factory=list)
    bridge_chunk_ids: list[str] = Field(default_factory=list)


class SentenceResult(ExpansionRecord):
    """A complete Sentence observation that may be submitted as evidence."""

    target: Literal["SENTENCE"] = "SENTENCE"
    sentence_id: str
    score: float
    text: str
    parent_chunk_id: str
    document_id: str
    title: str | None
    navigation_only: Literal[False] = False
    evidence_eligible: Literal[True] = True
    paths: list[ExpansionPath] = Field(default_factory=list)


class NavigationSentencePreview(ExpansionRecord):
    """A sentence-shaped navigation hint that is never evidence eligible."""

    target: Literal["SENTENCE_PREVIEW"] = "SENTENCE_PREVIEW"
    sentence_id: str
    score: float
    text: str
    parent_chunk_id: str
    document_id: str
    title: str | None
    navigation_only: Literal[True] = True
    evidence_eligible: Literal[False] = False
    paths: list[ExpansionPath] = Field(default_factory=list)


class EntityResult(ExpansionRecord):
    target: Literal["ENTITY"] = "ENTITY"
    entity_id: str
    score: float
    canonical_name: str
    entity_type: str | None
    mention_count: int
    paths: list[ExpansionPath] = Field(default_factory=list)
    bridge_sentence_ids: list[str] = Field(default_factory=list)
    bridge_chunk_ids: list[str] = Field(default_factory=list)
    bridge_sentences: list[SentenceResult] = Field(default_factory=list)
    bridge_previews: list[NavigationSentencePreview] = Field(default_factory=list)


class ChunkResult(ExpansionRecord):
    """An unread Chunk handle plus at most two navigation-only previews."""

    target: Literal["CHUNK"] = "CHUNK"
    chunk_id: str
    score: float
    document_id: str
    title: str | None
    navigation_only: Literal[True] = True
    evidence_eligible: Literal[False] = False
    content_read: Literal[False] = False
    previews: list[NavigationSentencePreview] = Field(default_factory=list)
    paths: list[ExpansionPath] = Field(default_factory=list)


ExpansionResult = (
    SentenceResult | NavigationSentencePreview | EntityResult | ChunkResult
)


class ExpansionObservation(ExpansionRecord):
    status: Literal["ok"] = "ok"
    action_type: Literal["EXPAND"] = "EXPAND"
    kind: str
    source_id: str
    query_used: str
    internal_request: InternalExpansionRequest
    source_degree: int
    candidate_count: int
    results: list[ExpansionResult] = Field(default_factory=list)

    @property
    def candidate_count_before_truncation(self) -> int:
        """Compatibility name used by controller/router trajectory metadata."""

        return self.candidate_count


_SPECS: dict[str, InternalExpansionRequest] = {
    ENTITY_MENTIONED_IN_SENTENCE: InternalExpansionRequest(
        source_type="ENTITY", relation="MENTIONED_IN", target="SENTENCE"
    ),
    SENTENCE_MENTIONS_ENTITY: InternalExpansionRequest(
        source_type="SENTENCE", relation="MENTIONS", target="ENTITY"
    ),
    ENTITY_CO_OCCURS_ENTITY_SENTENCE: InternalExpansionRequest(
        source_type="ENTITY",
        relation="CO_OCCURS",
        target="ENTITY",
        scope="SENTENCE",
    ),
    CHUNK_ADJACENT_CHUNK: InternalExpansionRequest(
        source_type="CHUNK", relation="ADJACENT", target="CHUNK"
    ),
    ENTITY_MENTIONED_IN_CHUNK: InternalExpansionRequest(
        source_type="ENTITY", relation="MENTIONED_IN", target="CHUNK"
    ),
    ENTITY_CO_OCCURS_ENTITY_CHUNK: InternalExpansionRequest(
        source_type="ENTITY",
        relation="CO_OCCURS",
        target="ENTITY",
        scope="CHUNK",
    ),
    CHUNK_CONTAINS_SENTENCE: InternalExpansionRequest(
        source_type="CHUNK", relation="CONTAINS", target="SENTENCE"
    ),
    CHUNK_MENTIONS_ENTITY: InternalExpansionRequest(
        source_type="CHUNK", relation="MENTIONS", target="ENTITY"
    ),
}

KeyT = TypeVar("KeyT", bound=Hashable)


class ExpansionEngine:
    """Enumerate structurally local candidates, then rank only that candidate set."""

    def __init__(
        self,
        substrate: Substrate | str | Path,
        *,
        embedding_backend: EmbeddingBackend | None = None,
        ranking_service: RankingService | None = None,
    ) -> None:
        self.substrate = (
            substrate if isinstance(substrate, Substrate) else Substrate.open(substrate)
        )
        self._embedding_backend = embedding_backend
        self.ranking_service = ranking_service or RankingService(
            self.substrate, embedding_backend=embedding_backend
        )

        entity_to_sentences: dict[str, set[str]] = defaultdict(set)
        sentence_to_entities: dict[str, set[str]] = defaultdict(set)
        for mention in self.substrate.mentions:
            entity_to_sentences[mention.entity_id].add(mention.sentence_id)
            sentence_to_entities[mention.sentence_id].add(mention.entity_id)
        self._entity_to_sentences = {
            key: tuple(sorted(values)) for key, values in entity_to_sentences.items()
        }
        self._sentence_to_entities = {
            key: tuple(sorted(values)) for key, values in sentence_to_entities.items()
        }

        aliases: dict[str, set[str]] = defaultdict(set)
        for alias in self.substrate.entity_aliases:
            aliases[alias.entity_id].add(alias.alias)
        self._aliases_by_entity = {
            key: tuple(sorted(values)) for key, values in aliases.items()
        }

        chunks_by_doc: dict[str, list[Any]] = defaultdict(list)
        for chunk in self.substrate.chunks:
            chunks_by_doc[chunk.doc_id].append(chunk)
        self._chunks_by_doc = {}
        self._chunk_index_in_doc: dict[str, int] = {}
        for doc_id, chunks in chunks_by_doc.items():
            ordered = tuple(
                sorted(chunks, key=lambda item: (item.chunk_pos, item.chunk_id))
            )
            self._chunks_by_doc[doc_id] = ordered
            for index, chunk in enumerate(ordered):
                self._chunk_index_in_doc[chunk.chunk_id] = index

        self._mention_counts_by_scope: dict[str, Counter[str]] = {
            scope_id: Counter() for scope_id in self.substrate.doc_ids_by_scope
        }
        for mention in self.substrate.mentions:
            for scope_id in self.substrate.scope_ids_by_sentence.get(
                mention.sentence_id, set()
            ):
                self._mention_counts_by_scope[scope_id][mention.entity_id] += 1

    def expand(
        self,
        action: Any,
        question: str,
        scope_id: str,
    ) -> ExpansionObservation:
        """Execute one expansion action within ``scope_id``.

        ``top_k`` is intentionally fixed at five.  The controller decides which
        expansion kinds are policy-enabled; this engine supports all eight kinds.
        """

        self.substrate.require_scope(scope_id)
        kind = self._enum_value(self._action_value(action, "kind"))
        if kind not in SUPPORTED_EXPANSION_KINDS:
            raise AgenticRAGError(f"Unsupported expansion kind: {kind}")

        source_id = str(self._action_value(action, "source_id"))
        top_k = int(self._action_value(action, "top_k", 5))
        if top_k != 5:
            raise AgenticRAGError("EXPAND top_k is fixed at 5")

        raw_direction = self._action_value(action, "direction", None)
        direction = (
            self._enum_value(raw_direction) if raw_direction is not None else None
        )
        if kind == CHUNK_ADJACENT_CHUNK:
            if direction not in {"PREV", "NEXT", "BOTH"}:
                raise AgenticRAGError(
                    "CHUNK_ADJACENT_CHUNK requires PREV, NEXT, or BOTH direction"
                )
        elif direction is not None:
            raise AgenticRAGError(
                f"direction is only valid for {CHUNK_ADJACENT_CHUNK}"
            )

        specification = _SPECS[kind].model_copy(
            update={"direction": direction if kind == CHUNK_ADJACENT_CHUNK else None}
        )
        self._require_source_in_scope(
            source_id, specification.source_type, scope_id
        )

        action_query = self._action_value(action, "query", None)
        query_used = (
            str(action_query)
            if action_query is not None and str(action_query).strip()
            else question
        )

        dispatch = {
            ENTITY_MENTIONED_IN_SENTENCE: self._entity_mentioned_in_sentence,
            SENTENCE_MENTIONS_ENTITY: self._sentence_mentions_entity,
            ENTITY_CO_OCCURS_ENTITY_SENTENCE: self._entity_co_occurs_sentence,
            CHUNK_ADJACENT_CHUNK: self._chunk_adjacent_chunk,
            ENTITY_MENTIONED_IN_CHUNK: self._entity_mentioned_in_chunk,
            ENTITY_CO_OCCURS_ENTITY_CHUNK: self._entity_co_occurs_chunk,
            CHUNK_CONTAINS_SENTENCE: self._chunk_contains_sentence,
            CHUNK_MENTIONS_ENTITY: self._chunk_mentions_entity,
        }
        source_degree, candidate_count, results = dispatch[kind](
            source_id, query_used, scope_id, direction
        )
        return ExpansionObservation(
            kind=kind,
            source_id=source_id,
            query_used=query_used,
            internal_request=specification,
            source_degree=source_degree,
            candidate_count=candidate_count,
            results=results,
        )

    def _entity_mentioned_in_sentence(
        self,
        source_id: str,
        query: str,
        scope_id: str,
        _direction: str | None,
    ) -> tuple[int, int, list[ExpansionResult]]:
        sentence_ids = [
            sentence_id
            for sentence_id in self._entity_to_sentences.get(source_id, ())
            if sentence_id in self.substrate.sentence_ids_by_scope[scope_id]
        ]
        ranked = self._rank_candidates(
            {
                sentence_id: self._sentence_ranking_text(sentence_id)
                for sentence_id in sentence_ids
            },
            query,
        )
        results: list[ExpansionResult] = []
        for sentence_id, score in ranked:
            results.append(
                self._sentence_result(
                    sentence_id,
                    score,
                    [
                        ExpansionPath(
                            relation="MENTIONED_IN",
                            node_ids=[source_id, sentence_id],
                            bridge_sentence_ids=[sentence_id],
                        )
                    ],
                )
            )
        return len(sentence_ids), len(sentence_ids), results

    def _sentence_mentions_entity(
        self,
        source_id: str,
        query: str,
        scope_id: str,
        _direction: str | None,
    ) -> tuple[int, int, list[ExpansionResult]]:
        entity_ids = list(self._sentence_to_entities.get(source_id, ()))
        sentence_text = self._sentence_ranking_text(source_id)
        ranked = self._rank_candidates(
            {
                entity_id: f"{self._entity_ranking_text(entity_id)}\n{sentence_text}"
                for entity_id in entity_ids
            },
            query,
        )
        results: list[ExpansionResult] = []
        for entity_id, score in ranked:
            results.append(
                self._entity_result(
                    entity_id,
                    score,
                    scope_id,
                    paths=[
                        ExpansionPath(
                            relation="MENTIONS",
                            node_ids=[source_id, entity_id],
                        )
                    ],
                )
            )
        return len(entity_ids), len(entity_ids), results

    def _entity_co_occurs_sentence(
        self,
        source_id: str,
        query: str,
        scope_id: str,
        _direction: str | None,
    ) -> tuple[int, int, list[ExpansionResult]]:
        bridge_sentence_ids = [
            sentence_id
            for sentence_id in self._entity_to_sentences.get(source_id, ())
            if sentence_id in self.substrate.sentence_ids_by_scope[scope_id]
        ]
        supports: dict[str, set[str]] = defaultdict(set)
        for sentence_id in bridge_sentence_ids:
            for target_entity_id in self._sentence_to_entities.get(sentence_id, ()):
                if target_entity_id != source_id:
                    supports[target_entity_id].add(sentence_id)

        ranked = self._rank_grouped_entities(supports, query, support_type="SENTENCE")
        results: list[ExpansionResult] = []
        for entity_id, score, support_scores in ranked:
            support_ids = sorted(supports[entity_id])
            paths = [
                ExpansionPath(
                    relation="CO_OCCURS",
                    node_ids=[source_id, sentence_id, entity_id],
                    bridge_sentence_ids=[sentence_id],
                )
                for sentence_id in support_ids
            ]
            bridge_sentences = [
                self._sentence_result(
                    sentence_id,
                    support_scores[sentence_id],
                    [
                        ExpansionPath(
                            relation="CO_OCCURS",
                            node_ids=[source_id, sentence_id, entity_id],
                            bridge_sentence_ids=[sentence_id],
                        )
                    ],
                )
                for sentence_id in support_ids
            ]
            results.append(
                self._entity_result(
                    entity_id,
                    score,
                    scope_id,
                    paths=paths,
                    bridge_sentence_ids=support_ids,
                    bridge_sentences=bridge_sentences,
                )
            )
        return len(bridge_sentence_ids), len(supports), results

    def _chunk_adjacent_chunk(
        self,
        source_id: str,
        query: str,
        _scope_id: str,
        direction: str | None,
    ) -> tuple[int, int, list[ExpansionResult]]:
        source = self.substrate.chunk_by_id[source_id]
        chunks = self._chunks_by_doc[source.doc_id]
        index = self._chunk_index_in_doc[source_id]
        candidate_ids: list[str] = []
        if direction in {"PREV", "BOTH"} and index > 0:
            candidate_ids.append(chunks[index - 1].chunk_id)
        if direction in {"NEXT", "BOTH"} and index + 1 < len(chunks):
            candidate_ids.append(chunks[index + 1].chunk_id)
        ranked = self._rank_candidates(
            {
                chunk_id: self._chunk_ranking_text(chunk_id)
                for chunk_id in candidate_ids
            },
            query,
        )
        results: list[ExpansionResult] = []
        for chunk_id, score in ranked:
            results.append(
                self._chunk_result(
                    chunk_id,
                    score,
                    query,
                    [
                        ExpansionPath(
                            relation="ADJACENT",
                            node_ids=[source_id, chunk_id],
                            bridge_chunk_ids=[],
                        )
                    ],
                )
            )
        return len(candidate_ids), len(candidate_ids), results

    def _entity_mentioned_in_chunk(
        self,
        source_id: str,
        query: str,
        scope_id: str,
        _direction: str | None,
    ) -> tuple[int, int, list[ExpansionResult]]:
        sentence_ids = [
            sentence_id
            for sentence_id in self._entity_to_sentences.get(source_id, ())
            if sentence_id in self.substrate.sentence_ids_by_scope[scope_id]
        ]
        support_sentences: dict[str, list[str]] = defaultdict(list)
        for sentence_id in sentence_ids:
            chunk_id = self.substrate.sentence_by_id[sentence_id].chunk_id
            support_sentences[chunk_id].append(sentence_id)
        ranked = self._rank_candidates(
            {
                chunk_id: self._chunk_ranking_text(chunk_id)
                for chunk_id in support_sentences
            },
            query,
        )
        results: list[ExpansionResult] = []
        for chunk_id, score in ranked:
            paths = [
                ExpansionPath(
                    relation="MENTIONED_IN",
                    node_ids=[source_id, sentence_id, chunk_id],
                    bridge_sentence_ids=[sentence_id],
                    bridge_chunk_ids=[chunk_id],
                )
                for sentence_id in sorted(support_sentences[chunk_id])
            ]
            results.append(self._chunk_result(chunk_id, score, query, paths))
        return len(support_sentences), len(support_sentences), results

    def _entity_co_occurs_chunk(
        self,
        source_id: str,
        query: str,
        scope_id: str,
        _direction: str | None,
    ) -> tuple[int, int, list[ExpansionResult]]:
        source_sentence_ids = [
            sentence_id
            for sentence_id in self._entity_to_sentences.get(source_id, ())
            if sentence_id in self.substrate.sentence_ids_by_scope[scope_id]
        ]
        bridge_chunk_ids = sorted(
            {
                self.substrate.sentence_by_id[sentence_id].chunk_id
                for sentence_id in source_sentence_ids
            }
        )
        supports: dict[str, set[str]] = defaultdict(set)
        supporting_sentences: dict[tuple[str, str], set[str]] = defaultdict(set)
        for chunk_id in bridge_chunk_ids:
            for sentence in self.substrate.sentences_by_chunk.get(chunk_id, []):
                for target_entity_id in self._sentence_to_entities.get(
                    sentence.sentence_id, ()
                ):
                    if target_entity_id == source_id:
                        continue
                    supports[target_entity_id].add(chunk_id)
                    supporting_sentences[(target_entity_id, chunk_id)].add(
                        sentence.sentence_id
                    )

        ranked = self._rank_grouped_entities(supports, query, support_type="CHUNK")
        results: list[ExpansionResult] = []
        for entity_id, score, support_scores in ranked:
            chunk_ids = sorted(supports[entity_id])
            paths: list[ExpansionPath] = []
            previews: list[NavigationSentencePreview] = []
            for chunk_id in chunk_ids:
                sentence_ids = sorted(supporting_sentences[(entity_id, chunk_id)])
                for sentence_id in sentence_ids:
                    path = ExpansionPath(
                        relation="CO_OCCURS",
                        node_ids=[source_id, chunk_id, sentence_id, entity_id],
                        bridge_sentence_ids=[sentence_id],
                        bridge_chunk_ids=[chunk_id],
                    )
                    paths.append(path)
                    previews.append(
                        self._sentence_preview(
                            sentence_id,
                            support_scores[chunk_id],
                            [path],
                        )
                    )
            results.append(
                self._entity_result(
                    entity_id,
                    score,
                    scope_id,
                    paths=paths,
                    bridge_sentence_ids=sorted(
                        {
                            sentence_id
                            for path in paths
                            for sentence_id in path.bridge_sentence_ids
                        }
                    ),
                    bridge_chunk_ids=chunk_ids,
                    bridge_previews=previews,
                )
            )
        return len(bridge_chunk_ids), len(supports), results

    def _chunk_contains_sentence(
        self,
        source_id: str,
        query: str,
        _scope_id: str,
        _direction: str | None,
    ) -> tuple[int, int, list[ExpansionResult]]:
        sentence_ids = [
            item.sentence_id
            for item in self.substrate.sentences_by_chunk.get(source_id, [])
        ]
        ranked = self._rank_candidates(
            {
                sentence_id: self._sentence_ranking_text(sentence_id)
                for sentence_id in sentence_ids
            },
            query,
        )
        results: list[ExpansionResult] = []
        for sentence_id, score in ranked:
            results.append(
                self._sentence_preview(
                    sentence_id,
                    score,
                    [
                        ExpansionPath(
                            relation="CONTAINS",
                            node_ids=[source_id, sentence_id],
                        )
                    ],
                )
            )
        return len(sentence_ids), len(sentence_ids), results

    def _chunk_mentions_entity(
        self,
        source_id: str,
        query: str,
        scope_id: str,
        _direction: str | None,
    ) -> tuple[int, int, list[ExpansionResult]]:
        supports: dict[str, set[str]] = defaultdict(set)
        for sentence in self.substrate.sentences_by_chunk.get(source_id, []):
            for entity_id in self._sentence_to_entities.get(sentence.sentence_id, ()):
                supports[entity_id].add(sentence.sentence_id)
        ranked = self._rank_grouped_entities(supports, query, support_type="SENTENCE")
        results: list[ExpansionResult] = []
        for entity_id, score, support_scores in ranked:
            sentence_ids = sorted(supports[entity_id])
            paths = [
                ExpansionPath(
                    relation="MENTIONS",
                    node_ids=[source_id, sentence_id, entity_id],
                    bridge_sentence_ids=[sentence_id],
                    bridge_chunk_ids=[source_id],
                )
                for sentence_id in sentence_ids
            ]
            previews = [
                self._sentence_preview(
                    sentence_id,
                    support_scores[sentence_id],
                    [path],
                )
                for sentence_id, path in zip(sentence_ids, paths, strict=True)
            ]
            results.append(
                self._entity_result(
                    entity_id,
                    score,
                    scope_id,
                    paths=paths,
                    bridge_sentence_ids=sentence_ids,
                    bridge_chunk_ids=[source_id],
                    bridge_previews=previews,
                )
            )
        return len(supports), len(supports), results

    def _rank_grouped_entities(
        self,
        supports: Mapping[str, set[str]],
        query: str,
        *,
        support_type: Literal["SENTENCE", "CHUNK"],
    ) -> list[tuple[str, float, dict[str, float]]]:
        texts: dict[tuple[str, str], str] = {}
        for entity_id, support_ids in supports.items():
            entity_text = self._entity_ranking_text(entity_id)
            for support_id in support_ids:
                support_text = (
                    self._sentence_ranking_text(support_id)
                    if support_type == "SENTENCE"
                    else self._chunk_ranking_text(support_id)
                )
                texts[(entity_id, support_id)] = (
                    f"{entity_text}\nSupporting context: {support_text}"
                )
        edge_scores = self._score_candidates(texts, query)
        grouped_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for (entity_id, support_id), score in edge_scores.items():
            grouped_scores[entity_id][support_id] = score
        ranked = [
            (entity_id, max(scores.values()), scores)
            for entity_id, scores in grouped_scores.items()
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:5]

    def _rank_candidates(
        self, candidates: Mapping[KeyT, str], query: str
    ) -> list[tuple[KeyT, float]]:
        # Most continuation candidates are already indexed source units. Use
        # the same cached vectors as global retrieval and encode the query only
        # once. Composite entity-edge rankings retain the deterministic
        # fallback below because they are not standalone substrate units.
        if candidates:
            keys = list(candidates)
            target = None
            if all(str(key) in self.substrate.sentence_by_id for key in keys):
                target = "sentence"
            elif all(str(key) in self.substrate.chunk_by_id for key in keys):
                target = "chunk"
            elif all(str(key) in self.substrate.entity_by_id for key in keys):
                target = "entity"
            expected_text = {
                "sentence": self._sentence_ranking_text,
                "chunk": self._chunk_ranking_text,
                "entity": self._entity_ranking_text,
            }.get(target)
            if target is not None and expected_text is not None and all(
                candidates[key] == expected_text(str(key)) for key in keys
            ):
                ranked = self.ranking_service.rank(
                    query, target=target, candidate_ids=(str(key) for key in keys), top_k=5
                )
                return [(next(key for key in keys if str(key) == node_id), score) for node_id, score in ranked]
        scores = self._score_candidates(candidates, query)
        ranked = list(scores.items())
        ranked.sort(key=lambda item: (-item[1], str(item[0])))
        return ranked[:5]

    def _score_candidates(
        self, candidates: Mapping[KeyT, str], query: str
    ) -> dict[KeyT, float]:
        if not candidates:
            return {}
        ordered = sorted(candidates.items(), key=lambda item: str(item[0]))
        matrix = normalize_embeddings(
            self._get_embedding_backend().encode(
                [query, *(text for _, text in ordered)]
            )
        )
        if matrix.shape[0] != len(ordered) + 1:
            raise AgenticRAGError(
                "Embedding backend returned the wrong number of expansion rows"
            )
        scores = np.asarray(matrix[1:] @ matrix[0], dtype=np.float32)
        result: dict[KeyT, float] = {}
        for (candidate_id, _), raw_score in zip(ordered, scores, strict=True):
            score = float(raw_score)
            # Expansion artifacts are persisted as strict JSON; never leak a
            # backend NaN/Infinity into the trajectory.
            result[candidate_id] = score if np.isfinite(score) else -1.0
        return result

    def _sentence_result(
        self,
        sentence_id: str,
        score: float,
        paths: Sequence[ExpansionPath],
    ) -> SentenceResult:
        sentence = self.substrate.sentence_by_id[sentence_id]
        chunk = self.substrate.chunk_by_id[sentence.chunk_id]
        document = self.substrate.document_by_id[chunk.doc_id]
        return SentenceResult(
            sentence_id=sentence.sentence_id,
            score=score,
            text=sentence.text,
            parent_chunk_id=chunk.chunk_id,
            document_id=document.doc_id,
            title=document.title,
            paths=list(paths),
        )

    def _sentence_preview(
        self,
        sentence_id: str,
        score: float,
        paths: Sequence[ExpansionPath],
    ) -> NavigationSentencePreview:
        sentence = self.substrate.sentence_by_id[sentence_id]
        chunk = self.substrate.chunk_by_id[sentence.chunk_id]
        document = self.substrate.document_by_id[chunk.doc_id]
        return NavigationSentencePreview(
            sentence_id=sentence.sentence_id,
            score=score,
            text=sentence.text,
            parent_chunk_id=chunk.chunk_id,
            document_id=document.doc_id,
            title=document.title,
            paths=list(paths),
        )

    def _chunk_result(
        self,
        chunk_id: str,
        score: float,
        query: str,
        paths: Sequence[ExpansionPath],
    ) -> ChunkResult:
        chunk = self.substrate.chunk_by_id[chunk_id]
        document = self.substrate.document_by_id[chunk.doc_id]
        sentence_ids = [
            sentence.sentence_id
            for sentence in self.substrate.sentences_by_chunk.get(chunk_id, [])
        ]
        ranked_previews = self._rank_candidates(
            {
                sentence_id: self._sentence_ranking_text(sentence_id)
                for sentence_id in sentence_ids
            },
            query,
        )[:2]
        return ChunkResult(
            chunk_id=chunk.chunk_id,
            score=score,
            document_id=document.doc_id,
            title=document.title,
            previews=[
                self._sentence_preview(
                    sentence_id,
                    sentence_score,
                    [
                        ExpansionPath(
                            relation="CONTAINS",
                            node_ids=[chunk_id, sentence_id],
                        )
                    ],
                )
                for sentence_id, sentence_score in ranked_previews
            ],
            paths=list(paths),
        )

    def _entity_result(
        self,
        entity_id: str,
        score: float,
        scope_id: str,
        *,
        paths: Sequence[ExpansionPath],
        bridge_sentence_ids: Sequence[str] = (),
        bridge_chunk_ids: Sequence[str] = (),
        bridge_sentences: Sequence[SentenceResult] = (),
        bridge_previews: Sequence[NavigationSentencePreview] = (),
    ) -> EntityResult:
        entity = self.substrate.entity_by_id[entity_id]
        return EntityResult(
            entity_id=entity.entity_id,
            score=score,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            mention_count=self._mention_counts_by_scope[scope_id][entity_id],
            paths=list(paths),
            bridge_sentence_ids=list(bridge_sentence_ids),
            bridge_chunk_ids=list(bridge_chunk_ids),
            bridge_sentences=list(bridge_sentences),
            bridge_previews=list(bridge_previews),
        )

    def _sentence_ranking_text(self, sentence_id: str) -> str:
        sentence = self.substrate.sentence_by_id[sentence_id]
        chunk = self.substrate.chunk_by_id[sentence.chunk_id]
        document = self.substrate.document_by_id[chunk.doc_id]
        return (
            f"{document.title}\n{sentence.text}"
            if document.title
            else sentence.text
        )

    def _chunk_ranking_text(self, chunk_id: str) -> str:
        chunk = self.substrate.chunk_by_id[chunk_id]
        document = self.substrate.document_by_id[chunk.doc_id]
        return f"{document.title}\n{chunk.text}" if document.title else chunk.text

    def _entity_ranking_text(self, entity_id: str) -> str:
        entity = self.substrate.entity_by_id[entity_id]
        values = [entity.canonical_name]
        if entity.entity_type:
            values.append(entity.entity_type)
        values.extend(self._aliases_by_entity.get(entity_id, ()))
        return "\n".join(values)

    def _require_source_in_scope(
        self,
        source_id: str,
        source_type: Literal["ENTITY", "SENTENCE", "CHUNK"],
        scope_id: str,
    ) -> None:
        if source_type == "ENTITY":
            exists = source_id in self.substrate.entity_by_id
            allowed = source_id in self.substrate.entity_ids_by_scope[scope_id]
        elif source_type == "SENTENCE":
            exists = source_id in self.substrate.sentence_by_id
            allowed = source_id in self.substrate.sentence_ids_by_scope[scope_id]
        else:
            exists = source_id in self.substrate.chunk_by_id
            allowed = source_id in self.substrate.chunk_ids_by_scope[scope_id]
        if not exists:
            raise NodeNotFoundError(f"Unknown {source_type.title()} ID: {source_id}")
        if not allowed:
            raise NodeNotFoundError(
                f"{source_type.title()} {source_id} is not present in scope {scope_id}"
            )

    def _get_embedding_backend(self) -> EmbeddingBackend:
        if self._embedding_backend is None:
            self._embedding_backend = create_embedding_backend(
                self.substrate.manifest.embedding_model.name,
                backend=self.substrate.manifest.embedding_backend,
            )
        return self._embedding_backend

    @staticmethod
    def _action_value(action: Any, name: str, default: Any = ...) -> Any:
        if isinstance(action, Mapping):
            if name in action:
                return action[name]
        elif hasattr(action, name):
            return getattr(action, name)
        if default is not ...:
            return default
        raise AgenticRAGError(f"EXPAND action is missing required field: {name}")

    @staticmethod
    def _enum_value(value: Any) -> str:
        if isinstance(value, Enum):
            return str(value.value)
        return str(value)
