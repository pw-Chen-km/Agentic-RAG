from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.bridge import SubstrateBridge
from agentic_rag.errors import InvalidRetrievalPairError, ScopeNotFoundError
from agentic_rag.models import ChunkHit, SentenceHit, SentenceResult
from agentic_rag.retrieval import Retriever
from agentic_rag.storage import Substrate
from conftest import FakeEmbeddingBackend


def test_scope_safe_lexical_and_bm25_retrieval(
    built_substrate: Path, fake_embedder: FakeEmbeddingBackend
) -> None:
    retriever = Retriever(built_substrate, embedding_backend=fake_embedder)
    q1_warsaw = retriever.search("Warsaw", "LEXICAL", "ENTITY", "q1")
    assert len(q1_warsaw) == 1
    assert retriever.search("Warsaw", "LEXICAL", "ENTITY", "q2") == []

    sentence_hits = retriever.search(
        "Where was Marie Curie born?", "BM25", "SENTENCE", "q1", top_k=3
    )
    assert sentence_hits
    assert all(isinstance(hit, SentenceHit) for hit in sentence_hits)
    assert all(":q1:" in hit.doc_id for hit in sentence_hits)
    assert sentence_hits[0].text == "Marie Curie was born in Warsaw."

    with pytest.raises(ScopeNotFoundError):
        retriever.search("Warsaw", "LEXICAL", "ENTITY", "missing")


def test_sentence_result_accepts_legacy_input_and_writes_new_provenance_names() -> None:
    legacy = {
        "sentence_id": "sentence:S12",
        "text": "Marie Curie was born in Warsaw.",
        "chunk_id": "chunk:C4",
        "doc_id": "document:D1",
        "title": "Marie Curie",
    }
    result = SentenceResult.model_validate(legacy)
    assert result.parent_chunk_id == "chunk:C4"
    assert result.document_id == "document:D1"
    assert result.chunk_id == "chunk:C4"
    assert result.doc_id == "document:D1"

    serialized = result.model_dump(mode="json")
    assert serialized["parent_chunk_id"] == "chunk:C4"
    assert serialized["document_id"] == "document:D1"
    assert "chunk_id" not in serialized
    assert "doc_id" not in serialized
    assert SentenceResult.model_validate(serialized) == result

    legacy_hit = SentenceHit.model_validate(
        {
            **legacy,
            "target": "SENTENCE",
            "score": 1.0,
        }
    )
    hit_json = legacy_hit.model_dump(mode="json")
    assert hit_json["parent_chunk_id"] == "chunk:C4"
    assert hit_json["document_id"] == "document:D1"


def test_dense_search_and_chunk_preview_contract(
    built_substrate: Path, fake_embedder: FakeEmbeddingBackend
) -> None:
    retriever = Retriever(built_substrate, embedding_backend=fake_embedder)
    hits = retriever.search(
        "What is the capital of Poland?", "DENSE", "CHUNK", "q1", top_k=2
    )
    assert hits
    assert all(isinstance(hit, ChunkHit) for hit in hits)
    assert all(len(hit.previews) <= 2 for hit in hits)
    assert all(":q1:" in hit.doc_id for hit in hits)
    assert "text" not in hits[0].model_fields_set

    substrate = Substrate.open(built_substrate)
    read = substrate.read_chunk(hits[0].chunk_id)
    assert read.text
    assert read.sentences
    assert {item.sentence_id for item in read.sentences} == {
        item.sentence_id
        for item in substrate.sentences_by_chunk[hits[0].chunk_id]
    }


def test_invalid_method_target_pair_is_rejected(
    built_substrate: Path, fake_embedder: FakeEmbeddingBackend
) -> None:
    retriever = Retriever(built_substrate, embedding_backend=fake_embedder)
    with pytest.raises(InvalidRetrievalPairError):
        retriever.search("Warsaw", "BM25", "ENTITY", "q1")


def test_entity_sentence_entity_bridge_has_provenance(
    built_substrate: Path, fake_embedder: FakeEmbeddingBackend
) -> None:
    retriever = Retriever(built_substrate, embedding_backend=fake_embedder)
    warsaw = retriever.search("Warsaw", "LEXICAL", "ENTITY", "q1")[0]
    bridge = SubstrateBridge(built_substrate)
    sentence_ids = bridge.sentences_for_entity(warsaw.entity_id, "q1")
    assert sentence_ids
    hits = bridge.entity_sentence_entity(warsaw.entity_id, "q1")
    assert any(hit.target_canonical_name == "Poland" for hit in hits)
    assert all(hit.bridge_sentence_id in sentence_ids for hit in hits)
    assert all(":q1:" in hit.doc_id for hit in hits)
