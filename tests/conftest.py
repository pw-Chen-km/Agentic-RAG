from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from agentic_rag.builder import SubstrateBuilder
from agentic_rag.config import BuildConfig
from agentic_rag.models import (
    AbbreviationLink,
    EntityMentionDraft,
    ProcessedDocument,
    ProcessedSentence,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hotpotqa_mini.json"


class FakeDocumentProcessor:
    segmenter_name = "fake-sentence-segmenter"
    segmenter_version = "1"
    ner_name = "fake-ner"
    ner_version = "1"
    abbreviation_name = "fake-abbreviation-detector"
    abbreviation_version = "1"
    chunk_tokenizer_name = "fake-whitespace-v1"

    entities = {
        "Retrieval-Augmented Generation": "CONCEPT",
        "Marie Curie": "PERSON",
        "Warsaw": "GPE",
        "Poland": "GPE",
        "RAG": "CONCEPT",
        "Paris": "GPE",
        "France": "GPE",
        "Europe": "LOC",
    }

    def process(self, text: str) -> ProcessedDocument:
        sentence_texts = re.split(r"(?<=[.!?])\s+", text)
        sentences = []
        abbreviations = []
        for sentence_index, sentence_text in enumerate(sentence_texts):
            mentions = []
            occupied: list[tuple[int, int]] = []
            for surface in sorted(self.entities, key=lambda value: (-len(value), value)):
                for match in re.finditer(re.escape(surface), sentence_text):
                    span = (match.start(), match.end())
                    if any(span[0] < end and start < span[1] for start, end in occupied):
                        continue
                    occupied.append(span)
                    mentions.append(
                        EntityMentionDraft(
                            sentence_index=sentence_index,
                            surface_form=surface,
                            start=span[0],
                            end=span[1],
                            entity_type=self.entities[surface],
                        )
                    )
            mentions.sort(key=lambda item: (item.start, item.end))
            if "Retrieval-Augmented Generation (RAG)" in sentence_text:
                abbreviations.append(
                    AbbreviationLink(
                        short_form="RAG",
                        long_form="Retrieval-Augmented Generation",
                        source_sentence_index=sentence_index,
                    )
                )
            sentences.append(
                ProcessedSentence(
                    text=sentence_text,
                    token_count=len(re.findall(r"\w+", sentence_text)),
                    mentions=tuple(mentions),
                )
            )
        return ProcessedDocument(
            sentences=tuple(sentences),
            abbreviations=tuple(abbreviations),
        )


class FakeEmbeddingBackend:
    name = "fake-hash-embedding"
    version = "1"
    dimension = 32

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"(?u)\b\w+\b", text.casefold()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                matrix[row, index] += 1.0
        return matrix


@pytest.fixture
def fake_processor() -> FakeDocumentProcessor:
    return FakeDocumentProcessor()


@pytest.fixture
def fake_embedder() -> FakeEmbeddingBackend:
    return FakeEmbeddingBackend()


@pytest.fixture
def built_substrate(
    tmp_path: Path,
    fake_processor: FakeDocumentProcessor,
    fake_embedder: FakeEmbeddingBackend,
) -> Path:
    output = tmp_path / "substrate"
    builder = SubstrateBuilder(
        BuildConfig(corpus_id="fixture", split="dev", max_chunk_tokens=20),
        processor=fake_processor,
        embedding_backend=fake_embedder,
    )
    builder.build(FIXTURE_PATH, output)
    return output
