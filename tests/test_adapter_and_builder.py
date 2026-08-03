from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from agentic_rag.adapters import HotpotQAAdapter, HotpotQABenchmarkExactAdapter
from agentic_rag.builder import SubstrateBuilder
from agentic_rag.config import BuildConfig
from agentic_rag.storage import Substrate
from agentic_rag.validation import validate_substrate
from conftest import FIXTURE_PATH, FakeDocumentProcessor, FakeEmbeddingBackend

BENCHMARK_EXACT_PATH = Path(__file__).parent / "fixtures" / "benchmark_exact"


def test_hotpotqa_adapter_produces_stable_scoped_ids() -> None:
    output = HotpotQAAdapter().load(FIXTURE_PATH, "dev")
    assert output.documents[0].doc_id == "hotpotqa:dev:q1:ctx:0"
    assert {item.scope_id for item in output.scopes} == {"q1", "q2"}
    assert output.gold_support[0].sentence_id is None


def test_benchmark_exact_adapter_uses_one_global_scope_and_source_chunks() -> None:
    output = HotpotQABenchmarkExactAdapter().load(BENCHMARK_EXACT_PATH, "dev")

    assert [item.scope_id for item in output.scopes] == [
        "hotpotqa:benchmark_exact:dev"
    ]
    assert len(output.documents) == 1
    assert [item.chunk_pos for item in output.source_chunks] == [0, 1]
    assert output.source_chunks[0].text == (
        "Marie Curie was born in Warsaw. Warsaw is in Poland."
    )
    assert output.source_chunks[1].text.startswith("##land")
    assert {item.scope_id for item in output.benchmark_questions} == {
        "hotpotqa:benchmark_exact:dev"
    }
    assert not output.gold_support
    assert {item.role for item in output.source_artifacts} == {
        "chunks",
        "questions",
    }


def test_benchmark_exact_builder_preserves_chunks_and_hides_questions(
    tmp_path: Path,
    fake_processor: FakeDocumentProcessor,
    fake_embedder: FakeEmbeddingBackend,
) -> None:
    output_path = tmp_path / "benchmark-substrate"
    config = BuildConfig(
        corpus_id="benchmark_fixture",
        split="dev",
        source_format="hotpotqa_benchmark_exact",
        max_chunk_tokens=1,
    )
    SubstrateBuilder(
        config,
        processor=fake_processor,
        embedding_backend=fake_embedder,
    ).build(BENCHMARK_EXACT_PATH, output_path)

    substrate = Substrate.open(output_path)
    assert substrate.manifest.source_format == "hotpotqa_benchmark_exact"
    assert substrate.manifest.scope_mode == "global"
    assert substrate.manifest.preserved_source_chunks is True
    assert substrate.manifest.overlapping_chunks is True
    assert substrate.manifest.record_counts["documents"] == 1
    assert substrate.manifest.record_counts["chunks"] == 2
    assert substrate.manifest.record_counts["benchmark_questions"] == 2
    assert len(substrate.manifest.source_artifacts) == 2

    scope_id = "hotpotqa:benchmark_exact:dev"
    assert substrate.chunk_ids_by_scope[scope_id] == {
        item.chunk_id for item in substrate.chunks
    }
    assert [item.text for item in substrate.chunks] == [
        "Marie Curie was born in Warsaw. Warsaw is in Poland.",
        "##land is a country in Europe. Marie Curie became a physicist.",
    ]
    assert not hasattr(substrate, "benchmark_questions")
    questions = pq.read_table(
        output_path / "evaluation" / "benchmark_questions.parquet"
    ).to_pylist()
    assert [item["question_id"] for item in questions] == [
        "benchmark-q1",
        "benchmark-q2",
    ]
    assert {item["scope_id"] for item in questions} == {scope_id}


def test_builder_writes_valid_typed_substrate(built_substrate: Path) -> None:
    report = validate_substrate(built_substrate)
    assert report.valid, report.errors
    substrate = Substrate.open(built_substrate)
    assert substrate.manifest.overlapping_chunks is False
    assert substrate.manifest.record_counts["documents"] == 5
    assert not hasattr(substrate, "gold_support")
    gold_path = built_substrate / "evaluation" / "gold_support.parquet"
    gold = pq.read_table(gold_path).to_pylist()
    assert gold
    assert all(item["sentence_id"] is not None for item in gold)


def test_abbreviation_alias_is_provenanced(built_substrate: Path) -> None:
    substrate = Substrate.open(built_substrate)
    abbreviation_aliases = [
        item
        for item in substrate.entity_aliases
        if item.alias == "RAG" and item.alias_type == "ABBREVIATION"
    ]
    assert len(abbreviation_aliases) == 1
    alias = abbreviation_aliases[0]
    assert alias.source_sentence_id is not None
    entity = substrate.entity_by_id[alias.entity_id]
    assert entity.canonical_name == "Retrieval-Augmented Generation"


def test_rebuild_keeps_ids_stable(
    tmp_path: Path,
    fake_processor: FakeDocumentProcessor,
    fake_embedder: FakeEmbeddingBackend,
) -> None:
    paths = [tmp_path / "first", tmp_path / "second"]
    for path in paths:
        SubstrateBuilder(
            BuildConfig(corpus_id="fixture", split="dev", max_chunk_tokens=20),
            processor=fake_processor,
            embedding_backend=fake_embedder,
        ).build(FIXTURE_PATH, path)
    first = Substrate.open(paths[0])
    second = Substrate.open(paths[1])
    assert [item.sentence_id for item in first.sentences] == [
        item.sentence_id for item in second.sentences
    ]
    assert [item.entity_id for item in first.entities] == [
        item.entity_id for item in second.entities
    ]
    assert [item.alias_id for item in first.entity_aliases] == [
        item.alias_id for item in second.entity_aliases
    ]
