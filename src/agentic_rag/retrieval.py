"""Scope-safe lexical, BM25, and exact dense retrieval."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal, Sequence, cast

import numpy as np

from agentic_rag.bm25_index import BM25Index
from agentic_rag.embedding import (
    EmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
    load_dense_index,
    normalize_embeddings,
)
from agentic_rag.errors import (
    AgenticRAGError,
    InvalidRetrievalPairError,
    NodeNotFoundError,
)
from agentic_rag.models import (
    ChunkHit,
    EntityHit,
    SearchHit,
    SentenceHit,
    SentencePreview,
)
from agentic_rag.storage import Substrate
from agentic_rag.text import normalize_entity_name

Method = Literal["LEXICAL", "BM25", "DENSE"]
Target = Literal["ENTITY", "SENTENCE", "CHUNK"]

VALID_PAIRS: set[tuple[str, str]] = {
    ("LEXICAL", "ENTITY"),
    ("BM25", "SENTENCE"),
    ("BM25", "CHUNK"),
    ("DENSE", "ENTITY"),
    ("DENSE", "SENTENCE"),
    ("DENSE", "CHUNK"),
}


class Retriever:
    def __init__(
        self,
        substrate: Substrate | str | Path,
        *,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self.substrate = (
            substrate if isinstance(substrate, Substrate) else Substrate.open(substrate)
        )
        self._embedding_backend = embedding_backend
        self._bm25: dict[str, BM25Index] = {}
        self._dense: dict[str, tuple[list[str], np.ndarray]] = {}
        lookup_path = (
            self.substrate.root / "indexes" / "entity_alias" / "lookup.json"
        )
        if not lookup_path.exists():
            raise NodeNotFoundError(f"Entity alias index is missing: {lookup_path}")
        with lookup_path.open("r", encoding="utf-8") as handle:
            self.alias_lookup: dict[str, list[str]] = json.load(handle)

        self.mention_counts_by_scope: dict[str, Counter[str]] = {}
        for scope_id in self.substrate.doc_ids_by_scope:
            self.mention_counts_by_scope[scope_id] = Counter()
        for mention in self.substrate.mentions:
            for scope_id in self.substrate.scope_ids_by_sentence.get(
                mention.sentence_id, set()
            ):
                self.mention_counts_by_scope[scope_id][mention.entity_id] += 1

    def search(
        self,
        query: str,
        method: str,
        target: str,
        scope_id: str,
        top_k: int = 5,
    ) -> list[SearchHit]:
        normalized_method = method.upper()
        normalized_target = target.upper()
        if (normalized_method, normalized_target) not in VALID_PAIRS:
            raise InvalidRetrievalPairError(
                f"Unsupported retrieval pair: {normalized_method} → {normalized_target}"
            )
        if top_k < 1:
            raise AgenticRAGError("top_k must be at least 1")
        self.substrate.require_scope(scope_id)

        if normalized_method == "LEXICAL":
            return self._search_lexical_entity(query, scope_id, top_k)
        if normalized_method == "BM25":
            return self._search_bm25(
                query, cast(Target, normalized_target), scope_id, top_k
            )
        return self._search_dense(
            query, cast(Target, normalized_target), scope_id, top_k
        )

    def _search_lexical_entity(
        self, query: str, scope_id: str, top_k: int
    ) -> list[SearchHit]:
        normalized_query = normalize_entity_name(query)
        allowed = self.substrate.entity_ids_by_scope[scope_id]
        entity_ids = [
            entity_id
            for entity_id in self.alias_lookup.get(normalized_query, [])
            if entity_id in allowed
        ][:top_k]
        return [
            self._entity_hit(entity_id, 1.0, scope_id) for entity_id in entity_ids
        ]

    def _search_bm25(
        self, query: str, target: Target, scope_id: str, top_k: int
    ) -> list[SearchHit]:
        if target == "ENTITY":
            raise InvalidRetrievalPairError("BM25 → ENTITY is not supported")
        key = target.lower()
        index = self._get_bm25(key)
        scores = index.score(query)
        allowed = (
            self.substrate.sentence_ids_by_scope[scope_id]
            if target == "SENTENCE"
            else self.substrate.chunk_ids_by_scope[scope_id]
        )
        ranked = self._rank(index.ids, scores, allowed, top_k)
        if target == "SENTENCE":
            return [
                self._sentence_hit(node_id, score) for node_id, score in ranked
            ]
        sentence_index = self._get_bm25("sentence")
        sentence_scores = dict(
            zip(sentence_index.ids, sentence_index.score(query), strict=True)
        )
        return [
            self._chunk_hit(node_id, score, sentence_scores)
            for node_id, score in ranked
        ]

    def _search_dense(
        self, query: str, target: Target, scope_id: str, top_k: int
    ) -> list[SearchHit]:
        backend = self._get_embedding_backend()
        query_embedding = normalize_embeddings(backend.encode([query]))[0]
        ids, matrix = self._get_dense(target.lower())
        if matrix.shape[1] != query_embedding.shape[0]:
            raise AgenticRAGError(
                f"Query embedding dimension {query_embedding.shape[0]} does not "
                f"match index dimension {matrix.shape[1]}"
            )
        scores = np.asarray(matrix @ query_embedding, dtype=np.float32)
        if target == "ENTITY":
            allowed = self.substrate.entity_ids_by_scope[scope_id]
        elif target == "SENTENCE":
            allowed = self.substrate.sentence_ids_by_scope[scope_id]
        else:
            allowed = self.substrate.chunk_ids_by_scope[scope_id]
        ranked = self._rank(ids, scores, allowed, top_k)
        if target == "ENTITY":
            return [
                self._entity_hit(node_id, score, scope_id)
                for node_id, score in ranked
            ]
        if target == "SENTENCE":
            return [
                self._sentence_hit(node_id, score) for node_id, score in ranked
            ]
        sentence_ids, sentence_matrix = self._get_dense("sentence")
        sentence_scores_array = np.asarray(
            sentence_matrix @ query_embedding, dtype=np.float32
        )
        sentence_scores = dict(
            zip(sentence_ids, sentence_scores_array, strict=True)
        )
        return [
            self._chunk_hit(node_id, score, sentence_scores)
            for node_id, score in ranked
        ]

    @staticmethod
    def _rank(
        ids: Sequence[str],
        scores: Sequence[float],
        allowed: set[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        candidates = []
        for node_id, raw_score in zip(ids, scores, strict=True):
            if node_id not in allowed:
                continue
            score = float(raw_score)
            if np.isnan(score):
                score = float("-inf")
            candidates.append((node_id, score))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:top_k]

    def _get_bm25(self, target: str) -> BM25Index:
        if target not in self._bm25:
            self._bm25[target] = BM25Index(
                self.substrate.root / "indexes" / f"bm25_{target}"
            )
        return self._bm25[target]

    def _get_dense(self, target: str) -> tuple[list[str], np.ndarray]:
        if target not in self._dense:
            self._dense[target] = load_dense_index(
                self.substrate.root / "indexes" / f"dense_{target}"
            )
        return self._dense[target]

    def _get_embedding_backend(self) -> EmbeddingBackend:
        if self._embedding_backend is None:
            self._embedding_backend = SentenceTransformerEmbeddingBackend(
                self.substrate.manifest.embedding_model.name
            )
        return self._embedding_backend

    def _sentence_hit(self, sentence_id: str, score: float) -> SentenceHit:
        sentence = self.substrate.sentence_by_id[sentence_id]
        chunk = self.substrate.chunk_by_id[sentence.chunk_id]
        document = self.substrate.document_by_id[chunk.doc_id]
        return SentenceHit(
            sentence_id=sentence.sentence_id,
            score=score,
            text=sentence.text,
            parent_chunk_id=chunk.chunk_id,
            document_id=document.doc_id,
            title=document.title,
        )

    def _chunk_hit(
        self, chunk_id: str, score: float, sentence_scores: dict[str, float]
    ) -> ChunkHit:
        chunk = self.substrate.chunk_by_id[chunk_id]
        document = self.substrate.document_by_id[chunk.doc_id]
        contained = self.substrate.sentences_by_chunk.get(chunk_id, [])
        ranked_sentences = sorted(
            contained,
            key=lambda item: (
                -float(sentence_scores.get(item.sentence_id, 0.0)),
                item.sentence_id,
            ),
        )[:2]
        return ChunkHit(
            chunk_id=chunk_id,
            score=score,
            doc_id=document.doc_id,
            title=document.title,
            previews=[
                SentencePreview(sentence_id=item.sentence_id, text=item.text)
                for item in ranked_sentences
            ],
        )

    def _entity_hit(
        self, entity_id: str, score: float, scope_id: str
    ) -> EntityHit:
        entity = self.substrate.entity_by_id[entity_id]
        return EntityHit(
            entity_id=entity.entity_id,
            score=score,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            mention_count=self.mention_counts_by_scope[scope_id][entity_id],
        )
