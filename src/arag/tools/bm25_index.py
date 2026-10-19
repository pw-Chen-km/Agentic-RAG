"""Small deterministic BM25 index used by the EntityHop variants.

The implementation is intentionally self contained.  EntityHop is an
experimental retrieval action and should not change the dependency or ranking
behaviour of the original A-RAG tools.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_text(value: str) -> str:
    """Normalize only representation, never add corpus or gold information."""

    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(value.split())


def tokenize(value: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(value))


class BM25Index:
    """A deterministic BM25Okapi-style index over short text documents."""

    VERSION = "entityhop-bm25-v1"

    def __init__(
        self,
        documents: Mapping[str, str] | Sequence[Mapping[str, str]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if isinstance(documents, Mapping):
            pairs = [(str(key), str(value)) for key, value in documents.items()]
        else:
            pairs = [(str(row["id"]), str(row.get("text", row.get("search_text", "")))) for row in documents]
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_ids = [doc_id for doc_id, _ in pairs]
        self.texts = {doc_id: text for doc_id, text in pairs}
        self.tokens = {doc_id: tokenize(text) for doc_id, text in pairs}
        self.lengths = {doc_id: len(tokens) for doc_id, tokens in self.tokens.items()}
        self.avgdl = sum(self.lengths.values()) / len(self.doc_ids) if self.doc_ids else 0.0
        self.df: dict[str, int] = {}
        for tokens in self.tokens.values():
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self.df.get(term, 0)
        if not n or not df:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        allowed_ids: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        allowed = {str(value) for value in allowed_ids} if allowed_ids is not None else None
        scored: list[tuple[str, float]] = []
        for doc_id in self.doc_ids:
            if allowed is not None and doc_id not in allowed:
                continue
            doc_tokens = self.tokens[doc_id]
            if not doc_tokens:
                continue
            frequencies: dict[str, int] = {}
            for term in doc_tokens:
                frequencies[term] = frequencies.get(term, 0) + 1
            score = 0.0
            for term in terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                norm = self.k1 * (1.0 - self.b + self.b * self.lengths[doc_id] / (self.avgdl or 1.0))
                score += self._idf(term) * (tf * (self.k1 + 1.0)) / (tf + norm)
            if score > 0.0:
                scored.append((doc_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: max(0, int(top_k))]

    def to_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "k1": self.k1,
            "b": self.b,
            "documents": [{"id": doc_id, "text": self.texts[doc_id]} for doc_id in self.doc_ids],
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != cls.VERSION:
            raise ValueError(f"Unsupported BM25 index version: {data.get('version')!r}")
        return cls(data.get("documents", []), k1=data.get("k1", 1.2), b=data.get("b", 0.75))
