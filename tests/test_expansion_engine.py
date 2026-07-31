from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pytest

from agentic_rag.agent.expansion import (
    CHUNK_ADJACENT_CHUNK,
    CHUNK_CONTAINS_SENTENCE,
    CHUNK_MENTIONS_ENTITY,
    ENTITY_CO_OCCURS_ENTITY_CHUNK,
    ENTITY_CO_OCCURS_ENTITY_SENTENCE,
    ENTITY_MENTIONED_IN_CHUNK,
    ENTITY_MENTIONED_IN_SENTENCE,
    SENTENCE_MENTIONS_ENTITY,
    ChunkResult,
    EntityResult,
    ExpansionEngine,
    NavigationSentencePreview,
    SentenceResult,
)
from agentic_rag.builder import SubstrateBuilder
from agentic_rag.config import BuildConfig
from agentic_rag.errors import AgenticRAGError, NodeNotFoundError
from agentic_rag.storage import Substrate
from conftest import (
    FIXTURE_PATH,
    FakeDocumentProcessor,
    FakeEmbeddingBackend,
)


def _action(
    kind: str,
    source_id: str,
    *,
    query: str | None = None,
    direction: str | None = None,
    top_k: int = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        kind=kind,
        source_id=source_id,
        query=query,
        direction=direction,
        top_k=top_k,
    )


def _entity_id(substrate: Substrate, canonical_name: str) -> str:
    return next(
        entity.entity_id
        for entity in substrate.entities
        if entity.canonical_name == canonical_name
    )


def _sentence_id(substrate: Substrate, text: str) -> str:
    return next(
        sentence.sentence_id
        for sentence in substrate.sentences
        if sentence.text == text
    )


@pytest.fixture
def multi_chunk_substrate(
    tmp_path: Path,
    fake_processor: FakeDocumentProcessor,
    fake_embedder: FakeEmbeddingBackend,
) -> Path:
    output = tmp_path / "multi-chunk-substrate"
    SubstrateBuilder(
        BuildConfig(corpus_id="fixture", split="dev", max_chunk_tokens=6),
        processor=fake_processor,
        embedding_backend=fake_embedder,
    ).build(FIXTURE_PATH, output)
    return output


def test_default_four_expansions_and_complete_sentence_provenance(
    multi_chunk_substrate: Path,
    fake_embedder: FakeEmbeddingBackend,
) -> None:
    substrate = Substrate.open(multi_chunk_substrate)
    engine = ExpansionEngine(substrate, embedding_backend=fake_embedder)
    marie = _entity_id(substrate, "Marie Curie")
    born_sentence = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )

    mentioned = engine.expand(
        _action(ENTITY_MENTIONED_IN_SENTENCE, marie),
        "Where was Marie Curie born?",
        "q1",
    )
    assert mentioned.internal_request.model_dump() == {
        "source_type": "ENTITY",
        "relation": "MENTIONED_IN",
        "target": "SENTENCE",
        "scope": None,
        "direction": None,
    }
    assert mentioned.query_used == "Where was Marie Curie born?"
    assert mentioned.source_degree == mentioned.candidate_count == 1
    sentence_result = mentioned.results[0]
    assert isinstance(sentence_result, SentenceResult)
    assert sentence_result.sentence_id == born_sentence
    assert sentence_result.text == "Marie Curie was born in Warsaw."
    assert sentence_result.parent_chunk_id == substrate.chunk_for_sentence(
        born_sentence
    ).chunk_id
    assert sentence_result.document_id.startswith("hotpotqa:dev:q1:")
    assert sentence_result.title == "Marie Curie"
    assert sentence_result.evidence_eligible
    assert not sentence_result.navigation_only
    assert sentence_result.paths[0].node_ids == [marie, born_sentence]

    entities = engine.expand(
        _action(SENTENCE_MENTIONS_ENTITY, born_sentence),
        "Which location is mentioned?",
        "q1",
    )
    assert entities.internal_request.source_type == "SENTENCE"
    assert entities.internal_request.relation == "MENTIONS"
    assert {
        result.canonical_name
        for result in entities.results
        if isinstance(result, EntityResult)
    } == {"Marie Curie", "Warsaw"}

    co_occurs = engine.expand(
        _action(ENTITY_CO_OCCURS_ENTITY_SENTENCE, marie),
        "Where was Marie Curie born?",
        "q1",
    )
    assert co_occurs.internal_request.scope == "SENTENCE"
    warsaw_result = next(
        result
        for result in co_occurs.results
        if isinstance(result, EntityResult)
        and result.canonical_name == "Warsaw"
    )
    assert warsaw_result.bridge_sentence_ids == [born_sentence]
    assert warsaw_result.paths[0].node_ids == [
        marie,
        born_sentence,
        warsaw_result.entity_id,
    ]
    assert len(warsaw_result.bridge_sentences) == 1
    assert warsaw_result.bridge_sentences[0].evidence_eligible
    assert (
        warsaw_result.bridge_sentences[0].parent_chunk_id
        == sentence_result.parent_chunk_id
    )

    source_chunk = substrate.chunk_for_sentence(born_sentence)
    adjacent = engine.expand(
        _action(
            CHUNK_ADJACENT_CHUNK,
            source_chunk.chunk_id,
            direction="NEXT",
        ),
        "In which country was Marie Curie born?",
        "q1",
    )
    assert adjacent.internal_request.direction == "NEXT"
    assert adjacent.source_degree == adjacent.candidate_count == 1
    adjacent_chunk = adjacent.results[0]
    assert isinstance(adjacent_chunk, ChunkResult)
    assert adjacent_chunk.chunk_id != source_chunk.chunk_id
    assert adjacent_chunk.content_read is False
    assert adjacent_chunk.evidence_eligible is False
    assert adjacent_chunk.navigation_only is True
    assert all(preview.navigation_only for preview in adjacent_chunk.previews)
    assert all(
        preview.evidence_eligible is False for preview in adjacent_chunk.previews
    )


def test_four_internal_expansions_are_navigation_safe(
    built_substrate: Path,
    fake_embedder: FakeEmbeddingBackend,
) -> None:
    substrate = Substrate.open(built_substrate)
    engine = ExpansionEngine(substrate, embedding_backend=fake_embedder)
    marie = _entity_id(substrate, "Marie Curie")
    marie_chunk = substrate.chunk_for_sentence(
        _sentence_id(substrate, "Marie Curie was born in Warsaw.")
    )

    mentioned_chunks = engine.expand(
        _action(ENTITY_MENTIONED_IN_CHUNK, marie),
        "Read more about Marie Curie's birthplace.",
        "q1",
    )
    assert mentioned_chunks.source_degree == mentioned_chunks.candidate_count == 1
    chunk_result = mentioned_chunks.results[0]
    assert isinstance(chunk_result, ChunkResult)
    assert chunk_result.chunk_id == marie_chunk.chunk_id
    assert 1 <= len(chunk_result.previews) <= 2
    assert all(
        isinstance(preview, NavigationSentencePreview)
        and preview.navigation_only
        and not preview.evidence_eligible
        for preview in chunk_result.previews
    )
    assert chunk_result.paths[0].bridge_sentence_ids

    chunk_co_occurs = engine.expand(
        _action(ENTITY_CO_OCCURS_ENTITY_CHUNK, marie),
        "In which country was Marie Curie born?",
        "q1",
    )
    assert chunk_co_occurs.internal_request.scope == "CHUNK"
    names = {
        result.canonical_name
        for result in chunk_co_occurs.results
        if isinstance(result, EntityResult)
    }
    assert {"Warsaw", "Poland"} <= names
    poland = next(
        result
        for result in chunk_co_occurs.results
        if isinstance(result, EntityResult)
        and result.canonical_name == "Poland"
    )
    assert poland.bridge_chunk_ids == [marie_chunk.chunk_id]
    assert poland.bridge_previews
    assert not poland.bridge_sentences
    assert all(preview.navigation_only for preview in poland.bridge_previews)
    assert all(not preview.evidence_eligible for preview in poland.bridge_previews)

    contained = engine.expand(
        _action(CHUNK_CONTAINS_SENTENCE, marie_chunk.chunk_id),
        "In which country was Marie Curie born?",
        "q1",
    )
    assert contained.candidate_count == 2
    assert all(
        isinstance(result, NavigationSentencePreview)
        and result.navigation_only
        and not result.evidence_eligible
        for result in contained.results
    )

    mentions = engine.expand(
        _action(CHUNK_MENTIONS_ENTITY, marie_chunk.chunk_id),
        "In which country was Marie Curie born?",
        "q1",
    )
    assert {
        result.canonical_name
        for result in mentions.results
        if isinstance(result, EntityResult)
    } == {"Marie Curie", "Warsaw", "Poland"}
    assert all(
        preview.navigation_only and not preview.evidence_eligible
        for result in mentions.results
        if isinstance(result, EntityResult)
        for preview in result.bridge_previews
    )


def test_scope_and_source_validation(
    built_substrate: Path,
    fake_embedder: FakeEmbeddingBackend,
) -> None:
    substrate = Substrate.open(built_substrate)
    engine = ExpansionEngine(substrate, embedding_backend=fake_embedder)
    warsaw = _entity_id(substrate, "Warsaw")
    q1_sentence = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )

    with pytest.raises(NodeNotFoundError, match="not present in scope q2"):
        engine.expand(
            _action(ENTITY_MENTIONED_IN_SENTENCE, warsaw),
            "Where is Warsaw?",
            "q2",
        )
    with pytest.raises(NodeNotFoundError, match="not present in scope q2"):
        engine.expand(
            _action(SENTENCE_MENTIONS_ENTITY, q1_sentence),
            "What entities occur?",
            "q2",
        )
    with pytest.raises(NodeNotFoundError, match="Unknown Entity ID"):
        engine.expand(
            _action(ENTITY_MENTIONED_IN_SENTENCE, "entity:missing"),
            "Where?",
            "q1",
        )


class RecordingPreferenceEmbedding:
    name = "recording-preference"
    version = "1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        matrix = np.zeros((len(texts), 2), dtype=np.float32)
        for index, text in enumerate(texts):
            first_line = text.splitlines()[0]
            if text == "prefer Poland" or first_line == "Poland":
                matrix[index, 0] = 1.0
            else:
                matrix[index, 1] = 1.0
        return matrix


def test_query_fallback_and_ranking_are_applied_after_local_enumeration(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    backend = RecordingPreferenceEmbedding()
    engine = ExpansionEngine(substrate, embedding_backend=backend)
    chunk_id = substrate.chunk_for_sentence(
        _sentence_id(substrate, "Marie Curie was born in Warsaw.")
    ).chunk_id

    observation = engine.expand(
        _action(CHUNK_MENTIONS_ENTITY, chunk_id),
        "prefer Poland",
        "q1",
    )
    assert observation.query_used == "prefer Poland"
    assert isinstance(observation.results[0], EntityResult)
    assert observation.results[0].canonical_name == "Poland"
    assert backend.calls[0][0] == "prefer Poland"
    encoded_candidate_text = "\n".join(backend.calls[0][1:])
    assert "Paris" not in encoded_candidate_text
    assert "France" not in encoded_candidate_text

    backend.calls.clear()
    explicit = engine.expand(
        _action(
            CHUNK_MENTIONS_ENTITY,
            chunk_id,
            query="prefer Poland",
        ),
        "This original question must not replace the subquestion.",
        "q1",
    )
    assert explicit.query_used == "prefer Poland"
    assert backend.calls[0][0] == "prefer Poland"


class ZeroEmbeddingBackend:
    name = "zero"
    version = "1"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(texts), 2), dtype=np.float32)


def test_top_five_and_stable_id_tie_break(
    tmp_path: Path,
    fake_processor: FakeDocumentProcessor,
) -> None:
    source = tmp_path / "many.json"
    source.write_text(
        json.dumps(
            [
                {
                    "_id": "many",
                    "question": "Where is Paris mentioned?",
                    "answer": "France",
                    "supporting_facts": [],
                    "context": [
                        [
                            "Repeated",
                            [
                                f"Paris appears in example {index}."
                                for index in range(7)
                            ],
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "many-substrate"
    SubstrateBuilder(
        BuildConfig(corpus_id="many", split="dev", max_chunk_tokens=256),
        processor=fake_processor,
        embedding_backend=ZeroEmbeddingBackend(),
    ).build(source, output)
    substrate = Substrate.open(output)
    engine = ExpansionEngine(substrate, embedding_backend=ZeroEmbeddingBackend())
    paris = _entity_id(substrate, "Paris")

    observation = engine.expand(
        _action(ENTITY_MENTIONED_IN_SENTENCE, paris),
        "Where is Paris mentioned?",
        "many",
    )
    result_ids = [
        result.sentence_id
        for result in observation.results
        if isinstance(result, SentenceResult)
    ]
    assert observation.source_degree == observation.candidate_count == 7
    assert len(result_ids) == 5
    assert result_ids == sorted(result_ids)


def test_action_shape_invariants_are_enforced(
    built_substrate: Path,
    fake_embedder: FakeEmbeddingBackend,
) -> None:
    substrate = Substrate.open(built_substrate)
    engine = ExpansionEngine(substrate, embedding_backend=fake_embedder)
    sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    chunk_id = substrate.chunk_for_sentence(sentence_id).chunk_id

    with pytest.raises(AgenticRAGError, match="requires"):
        engine.expand(
            _action(CHUNK_ADJACENT_CHUNK, chunk_id),
            "What follows?",
            "q1",
        )
    with pytest.raises(AgenticRAGError, match="only valid"):
        engine.expand(
            _action(SENTENCE_MENTIONS_ENTITY, sentence_id, direction="NEXT"),
            "What entities occur?",
            "q1",
        )
    with pytest.raises(AgenticRAGError, match="fixed at 5"):
        engine.expand(
            _action(SENTENCE_MENTIONS_ENTITY, sentence_id, top_k=3),
            "What entities occur?",
            "q1",
        )
    with pytest.raises(AgenticRAGError, match="Unsupported expansion kind"):
        engine.expand(
            _action("SENTENCE_PART_OF_CHUNK", sentence_id),
            "Read context.",
            "q1",
        )
