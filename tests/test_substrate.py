from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.agent.expansion import ENTITY_MENTIONED_IN_SENTENCE, ExpansionEngine
from agentic_rag.errors import InvalidRetrievalPairError
from agentic_rag.substrate.models import ChunkHit, SentenceHit
from agentic_rag.substrate.retrieval import Retriever
from agentic_rag.substrate.storage import Substrate
from agentic_rag.substrate.validation import validate_substrate
from conftest import FakeEmbeddingBackend


def test_built_substrate_validates_and_retrieves_with_scope(
    built_substrate: Path, fake_embedder: FakeEmbeddingBackend
) -> None:
    report = validate_substrate(built_substrate)
    assert report.valid

    retriever = Retriever(built_substrate, embedding_backend=fake_embedder)
    hits = retriever.search(
        "Where was Marie Curie born?", "BM25", "SENTENCE", "q1", top_k=5
    )
    assert hits and all(isinstance(hit, SentenceHit) for hit in hits)
    assert hits[0].text == "Marie Curie was born in Warsaw."
    assert all(":q1:" in hit.document_id for hit in hits)

    chunk_hits = retriever.search(
        "capital Poland", "DENSE", "CHUNK", "q1", top_k=5
    )
    assert chunk_hits and all(isinstance(hit, ChunkHit) for hit in chunk_hits)
    read = Substrate.open(built_substrate).read_chunk(chunk_hits[0].chunk_id)
    assert read.text and read.sentences


def test_invalid_search_pair_is_rejected(
    built_substrate: Path, fake_embedder: FakeEmbeddingBackend
) -> None:
    with pytest.raises(InvalidRetrievalPairError):
        Retriever(built_substrate, embedding_backend=fake_embedder).search(
            "Warsaw", "BM25", "ENTITY", "q1", top_k=5
        )


def test_expansion_returns_complete_sentence_provenance(
    built_substrate: Path, fake_embedder: FakeEmbeddingBackend
) -> None:
    substrate = Substrate.open(built_substrate)
    marie = next(
        entity.entity_id
        for entity in substrate.entities
        if entity.canonical_name == "Marie Curie"
    )
    action = type(
        "Action",
        (),
        {
            "kind": ENTITY_MENTIONED_IN_SENTENCE,
            "source_id": marie,
            "query": "Where was Marie Curie born?",
            "direction": None,
            "top_k": 5,
        },
    )()
    result = ExpansionEngine(substrate, embedding_backend=fake_embedder).expand(
        action, "Where was Marie Curie born?", "q1"
    )
    assert result.results
    sentence = result.results[0]
    assert sentence.text == "Marie Curie was born in Warsaw."
    assert sentence.parent_chunk_id
    assert sentence.document_id
    assert sentence.evidence_eligible
