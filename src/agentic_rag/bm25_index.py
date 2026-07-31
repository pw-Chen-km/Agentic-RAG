"""BM25S persistence with a deterministic Unicode tokenizer."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Sequence

import bm25s
import numpy as np

from agentic_rag.errors import BuildError, IndexNotFoundError

_TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def tokenize(text: str, stopwords: str | None = None) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = _TOKEN_PATTERN.findall(normalized)
    if stopwords in ("english", "en"):
        excluded = set(bm25s.stopwords.STOPWORDS_EN)
        tokens = [token for token in tokens if token not in excluded]
    elif stopwords is not None:
        raise BuildError(f"Unsupported BM25 stopword setting: {stopwords}")
    return tokens


def build_bm25_index(
    directory: Path,
    ids: Sequence[str],
    texts: Sequence[str],
    *,
    stopwords: str | None = None,
) -> None:
    if len(ids) != len(texts):
        raise BuildError(
            f"BM25 ID/text count mismatch: {len(ids)} IDs vs {len(texts)} texts"
        )
    if not ids:
        raise BuildError("Cannot build a BM25 index over an empty corpus")
    directory.mkdir(parents=True, exist_ok=True)
    corpus_tokens = [tokenize(text, stopwords) for text in texts]
    retriever = bm25s.BM25(method="lucene", backend="numpy", csc_backend="scipy")
    retriever.index(corpus_tokens, show_progress=False)
    retriever.save(directory, show_progress=False)
    with (directory / "ids.json").open("w", encoding="utf-8") as handle:
        json.dump(list(ids), handle, ensure_ascii=False)
    with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "lucene",
                "k1": 1.5,
                "b": 0.75,
                "tokenizer": "unicode-nfkc-casefold-regex-v1",
                "stopwords": stopwords,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


class BM25Index:
    def __init__(self, directory: Path) -> None:
        ids_path = directory / "ids.json"
        metadata_path = directory / "metadata.json"
        if not ids_path.exists() or not metadata_path.exists():
            raise IndexNotFoundError(f"BM25 index is incomplete: {directory}")
        with ids_path.open("r", encoding="utf-8") as handle:
            self.ids = [str(value) for value in json.load(handle)]
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.stopwords = metadata.get("stopwords")
        self.retriever = bm25s.BM25.load(
            directory,
            mmap=True,
            load_corpus=False,
            show_progress=False,
        )
        num_docs = int(self.retriever.scores["num_docs"])
        if num_docs != len(self.ids):
            raise IndexNotFoundError(
                f"BM25 row mismatch in {directory}: {num_docs} vs {len(self.ids)}"
            )

    def score(self, query: str) -> np.ndarray:
        query_tokens = tokenize(query, self.stopwords)
        if not query_tokens:
            return np.zeros(len(self.ids), dtype=np.float32)
        return np.asarray(self.retriever.get_scores(query_tokens), dtype=np.float32)
