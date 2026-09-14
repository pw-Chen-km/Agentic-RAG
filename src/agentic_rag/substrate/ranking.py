"""Shared deterministic dense ranking for retrieval and entity continuation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np

from agentic_rag.errors import AgenticRAGError
from agentic_rag.substrate.embedding import (
    EmbeddingBackend,
    create_embedding_backend,
    load_dense_index,
    normalize_embeddings,
)
from agentic_rag.substrate.storage import Substrate


class RankingService:
    """Use precomputed substrate vectors with one query encoding per request."""

    def __init__(
        self,
        substrate: Substrate | str | Path,
        *,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self.substrate = substrate if isinstance(substrate, Substrate) else Substrate.open(substrate)
        self.embedding_backend = embedding_backend
        self._dense: dict[str, tuple[list[str], np.ndarray]] = {}
        self._query_cache: dict[str, np.ndarray] = {}
        self.query_encodes = 0
        self.candidate_scores = 0
        self.last_wall_time_ms = 0.0

    def rank(
        self,
        query: str,
        *,
        target: str,
        candidate_ids: Iterable[str] | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        started = time.perf_counter()
        key = target.casefold()
        ids, matrix = self._get_dense(key)
        query_vector = self._query_vector(query)
        if matrix.ndim != 2 or matrix.shape[1] != query_vector.shape[0]:
            raise AgenticRAGError("ranking vector dimensions do not match")
        allowed = set(candidate_ids) if candidate_ids is not None else None
        values: list[tuple[str, float]] = []
        for node_id, raw_score in zip(ids, matrix @ query_vector, strict=True):
            if allowed is not None and node_id not in allowed:
                continue
            score = float(raw_score)
            values.append((node_id, score if np.isfinite(score) else float("-inf")))
        values.sort(key=lambda item: (-item[1], item[0]))
        self.candidate_scores += len(values)
        self.last_wall_time_ms = (time.perf_counter() - started) * 1000
        return values[:top_k]

    def score(self, query: str, *, target: str, candidate_ids: Iterable[str]) -> dict[str, float]:
        return dict(self.rank(query, target=target, candidate_ids=candidate_ids, top_k=10**9))

    def _query_vector(self, query: str) -> np.ndarray:
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached
        backend = self.embedding_backend or self._default_backend()
        vector = normalize_embeddings(backend.encode([query]))[0]
        self._query_cache[query] = vector
        self.query_encodes += 1
        return vector

    def _get_dense(self, target: str) -> tuple[list[str], np.ndarray]:
        if target not in self._dense:
            self._dense[target] = load_dense_index(
                self.substrate.root / "indexes" / f"dense_{target}"
            )
        return self._dense[target]

    def _default_backend(self) -> EmbeddingBackend:
        if self.embedding_backend is None:
            self.embedding_backend = create_embedding_backend(
                self.substrate.manifest.embedding_model.name,
                backend=self.substrate.manifest.embedding_backend,
            )
        return self.embedding_backend
