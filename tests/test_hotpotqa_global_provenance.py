from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentic_rag.config import BuildConfig
from agentic_rag.errors import InputFormatError
from agentic_rag.substrate.adapters import HotpotQAGlobalProvenanceAdapter
from agentic_rag.substrate.builder import SubstrateBuilder
from agentic_rag.substrate.storage import EvaluationSidecars, Substrate
from agentic_rag.substrate.validation import validate_substrate
from conftest import FakeDocumentProcessor, FakeEmbeddingBackend


def _rows() -> list[dict]:
    shared = [
        "Marie Curie lived in Warsaw. She studied science.",
        "",
    ]
    return [
        {
            "_id": "hotpot-q-1",
            "question": "Where did Marie Curie live and what country is it in?",
            "answer": "Warsaw, Poland",
            "type": "bridge",
            "level": "hard",
            "supporting_facts": [["Shared Article", 0], ["Poland", 0]],
            "context": [
                ["Shared Article", shared],
                ["Poland", ["Warsaw is the capital of Poland."]],
            ],
        },
        {
            "_id": "hotpot-q-2",
            "question": "Are Warsaw and Paris both in Europe?",
            "answer": "yes",
            "type": "comparison",
            "level": "hard",
            "supporting_facts": [["Shared Article", 0], ["France", 0]],
            "context": [
                ["France", ["Paris is in France.", "France is in Europe."]],
                ["Shared Article", shared],
            ],
        },
    ]


def _write_source(path: Path, rows: list[dict] | None = None) -> Path:
    path.write_text(
        json.dumps(rows if rows is not None else _rows(), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_global_provenance_build_preserves_exact_hotpot_lineage(
    tmp_path: Path,
    fake_processor: FakeDocumentProcessor,
    fake_embedder: FakeEmbeddingBackend,
) -> None:
    source = _write_source(tmp_path / "hotpotqa.json")
    output = tmp_path / "substrate"
    manifest = SubstrateBuilder(
        BuildConfig(
            corpus_id="hotpot-global-fixture",
            dataset="hotpotqa",
            source_format="hotpotqa_global_provenance",
            benchmark_scope_id="hotpotqa:benchmark_exact:dev",
            max_chunk_tokens=50,
        ),
        processor=fake_processor,
        embedding_backend=fake_embedder,
    ).build(source, output)

    assert manifest.schema_version == "2.1"
    assert manifest.source_format == "hotpotqa_global_provenance"
    assert manifest.scope_mode == "global"
    assert manifest.record_counts["documents"] == 3
    assert manifest.record_counts["source_sentence_provenance"] == 5
    assert manifest.source_artifacts[0].role == "hotpotqa_source"
    assert manifest.source_artifacts[0].sha256 == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    substrate = Substrate.open(output)
    assert set(substrate.doc_ids_by_scope) == {
        "hotpotqa:benchmark_exact:dev"
    }
    assert len(substrate.doc_ids_by_scope["hotpotqa:benchmark_exact:dev"]) == 3
    assert not hasattr(substrate, "gold_support")
    assert not hasattr(substrate, "source_sentence_provenance")

    evaluation = EvaluationSidecars.open(output)
    assert len(evaluation.benchmark_questions) == 2
    assert len(evaluation.gold_support) == 4
    assert len(evaluation.source_sentence_provenance) == 5
    blank = [
        item
        for item in evaluation.source_sentence_provenance
        if item.original_title == "Shared Article"
        and item.original_sentence_id == 1
    ][0]
    assert blank.original_sentence_text == ""
    assert blank.sentence_id is None

    shared = [
        item
        for item in evaluation.source_sentence_provenance
        if item.original_title == "Shared Article"
        and item.original_sentence_id == 0
    ][0]
    assert shared.sentence_id is not None
    # The fake NLP processor splits this raw source sentence twice, but the
    # provenance build deliberately keeps exactly one substrate Sentence.
    assert substrate.sentence_by_id[shared.sentence_id].text == (
        "Marie Curie lived in Warsaw. She studied science."
    )

    facts = {
        (item.question_id, item.fact_id): item
        for item in evaluation.gold_support
    }
    assert set(facts) == {
        ("hotpot-q-1", "sf:0000"),
        ("hotpot-q-1", "sf:0001"),
        ("hotpot-q-2", "sf:0000"),
        ("hotpot-q-2", "sf:0001"),
    }
    assert facts[("hotpot-q-1", "sf:0000")].sentence_id == shared.sentence_id
    assert validate_substrate(output).valid


def test_global_provenance_rejects_inconsistent_repeated_title(
    tmp_path: Path,
) -> None:
    rows = _rows()
    rows[1]["context"][1][1] = ["Conflicting source text.", ""]
    source = _write_source(tmp_path / "hotpotqa.json", rows)

    with pytest.raises(InputFormatError, match="inconsistent sentence text"):
        HotpotQAGlobalProvenanceAdapter().load(source, "dev")


def test_global_provenance_rejects_invalid_gold_sentence_id(
    tmp_path: Path,
) -> None:
    rows = _rows()
    rows[0]["supporting_facts"][0][1] = 99
    source = _write_source(tmp_path / "hotpotqa.json", rows)

    with pytest.raises(InputFormatError, match="sentence index 99 is invalid"):
        HotpotQAGlobalProvenanceAdapter().load(source, "dev")


def test_checked_in_global_provenance_config_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = BuildConfig.from_yaml(
        root / "configs" / "datasets" / "hotpotqa_global_provenance.yaml"
    )

    assert config.source_format == "hotpotqa_global_provenance"
    assert config.benchmark_scope_id == "hotpotqa:benchmark_exact:dev"
    assert config.validate_benchmark_profile is True
