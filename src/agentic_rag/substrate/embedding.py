"""Pluggable embedding backends and exact dense index persistence."""

from __future__ import annotations

import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from agentic_rag.errors import BuildError, IndexNotFoundError


class EmbeddingBackend(Protocol):
    name: str
    version: str | None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbeddingBackend:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise BuildError(
                "sentence-transformers is required for the default dense backend"
            ) from exc
        self.name = model_name
        try:
            self.version = importlib.metadata.version("sentence-transformers")
        except importlib.metadata.PackageNotFoundError:
            self.version = None
        self.batch_size = batch_size
        self._model = _load_sentence_transformer(model_name, device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dimension = int(self._model.get_sentence_embedding_dimension())
            return np.empty((0, dimension), dtype=np.float32)
        try:
            result = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise BuildError(
                f"Embedding model {self.name!r} failed to encode text: {exc}"
            ) from exc
        return np.asarray(result, dtype=np.float32)


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str | None):
    """Load once per process and prefer an already-cached offline snapshot."""

    from sentence_transformers import SentenceTransformer

    offline_error: Exception | None = None
    try:
        return SentenceTransformer(
            model_name,
            device=device,
            local_files_only=True,
        )
    except Exception as exc:
        offline_error = exc
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        raise BuildError(
            f"Could not load embedding model {model_name!r}: {exc}; "
            f"offline cache attempt: {offline_error}"
        ) from exc


def normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise BuildError(f"Embedding backend returned shape {matrix.shape}, expected 2D")
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(matrix / norms, dtype=np.float32)


def save_dense_index(
    directory: Path,
    ids: Sequence[str],
    texts: Sequence[str],
    backend: EmbeddingBackend,
) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    embeddings = normalize_embeddings(backend.encode(texts))
    if embeddings.shape[0] != len(ids):
        raise BuildError(
            f"Embedding row count {embeddings.shape[0]} does not match IDs {len(ids)}"
        )
    np.save(directory / "embeddings.npy", embeddings, allow_pickle=False)
    with (directory / "ids.json").open("w", encoding="utf-8") as handle:
        json.dump(list(ids), handle, ensure_ascii=False)
    with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": backend.name,
                "backend_version": backend.version,
                "dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                "normalized": True,
                "dtype": "float32",
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    return int(embeddings.shape[1]) if embeddings.ndim == 2 else 0


def load_dense_index(directory: Path) -> tuple[list[str], np.ndarray]:
    ids_path = directory / "ids.json"
    embeddings_path = directory / "embeddings.npy"
    if not ids_path.exists() or not embeddings_path.exists():
        raise IndexNotFoundError(f"Dense index is incomplete: {directory}")
    with ids_path.open("r", encoding="utf-8") as handle:
        ids = [str(value) for value in json.load(handle)]
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.shape[0] != len(ids):
        raise IndexNotFoundError(
            f"Dense index row mismatch in {directory}: {embeddings.shape[0]} vs {len(ids)}"
        )
    return ids, embeddings
