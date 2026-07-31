from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from agentic_rag.adapters import HotpotQAAdapter
from agentic_rag.builder import SubstrateBuilder
from agentic_rag.config import BuildConfig
from agentic_rag.storage import Substrate
from agentic_rag.validation import validate_substrate
from conftest import FIXTURE_PATH, FakeDocumentProcessor, FakeEmbeddingBackend


def test_hotpotqa_adapter_produces_stable_scoped_ids() -> None:
    output = HotpotQAAdapter().load(FIXTURE_PATH, "dev")
    assert output.documents[0].doc_id == "hotpotqa:dev:q1:ctx:0"
    assert {item.scope_id for item in output.scopes} == {"q1", "q2"}
    assert output.gold_support[0].sentence_id is None


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
