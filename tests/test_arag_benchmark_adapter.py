from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from agentic_rag.adapters import (
    ARAGBenchmarkAdapter,
    HotpotQABenchmarkExactAdapter,
)
from agentic_rag.benchmark_profiles import (
    ARAG_DATASET_PROFILES,
    AnswerMode,
    DuplicateQuestionIdPolicy,
    EvaluationMetric,
    get_arag_dataset_profile,
)
from agentic_rag.builder import SubstrateBuilder
from agentic_rag.config import BuildConfig
from agentic_rag.errors import InputFormatError
from agentic_rag.storage import Substrate
from agentic_rag.validation import validate_substrate
from conftest import FakeDocumentProcessor, FakeEmbeddingBackend


PROFILE_CASES = (
    (
        "musique",
        1_000,
        1_354,
        1_000,
        "musique_2hop__fixture",
        "",
        "2_hop",
        "",
        (EvaluationMetric.LLM_ACC, EvaluationMetric.CONTAIN_ACC),
        AnswerMode.SHORT,
    ),
    (
        "hotpotqa",
        1_000,
        1_311,
        1_000,
        "hotpot-fixture",
        "bridge",
        "bridge",
        [["Fixture title", ["Fixture evidence."]]],
        (EvaluationMetric.LLM_ACC, EvaluationMetric.CONTAIN_ACC),
        AnswerMode.SHORT,
    ),
    (
        "2wikimultihop",
        1_000,
        658,
        1_000,
        "2wiki-fixture",
        "bridge-comparison",
        "bridge_comparison",
        [["Fixture title", "Fixture evidence."]],
        (EvaluationMetric.LLM_ACC, EvaluationMetric.CONTAIN_ACC),
        AnswerMode.SHORT,
    ),
    (
        "medical",
        2_062,
        225,
        2_062,
        "medical-fixture",
        "Fact Retrieval",
        "fact_retrieval",
        [{"passage": "Fixture evidence."}],
        (EvaluationMetric.LLM_ACC,),
        AnswerMode.LONG,
    ),
    (
        "novel",
        2_010,
        1_117,
        2_009,
        "novel-fixture",
        "Creative Generation",
        "creative_generation",
        "Fixture evidence.",
        (EvaluationMetric.LLM_ACC,),
        AnswerMode.LONG,
    ),
)


def _question(
    dataset: str,
    question_id: str,
    question_type: str,
    *,
    question: str | None = None,
    answer: str | None = None,
    evidence: object = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": question_id,
        "source": dataset,
        "question": question or f"Question for {question_id}?",
        "answer": answer or f"Answer for {question_id}.",
        "question_type": question_type,
        "evidence": evidence,
    }
    if dataset == "2wikimultihop":
        row["evidence_relations"] = [["first", "relates_to", "second"]]
    return row


def _write_benchmark(
    root: Path,
    dataset: str,
    questions: list[dict[str, Any]],
    *,
    chunks: list[str] | None = None,
    nested: bool = True,
) -> Path:
    data_dir = root / dataset if nested else root
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "chunks.json").write_text(
        json.dumps(
            chunks
            or [
                "0:Marie Curie lived in Warsaw: a city in Poland.",
                "1:##tail Poland is in Europe.",
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (data_dir / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False),
        encoding="utf-8",
    )
    return root if nested else data_dir


@pytest.mark.parametrize(
    (
        "dataset",
        "question_count",
        "chunk_count",
        "unique_question_ids",
        "source_question_id",
        "raw_question_type",
        "normalized_question_type",
        "evidence",
        "metrics",
        "answer_mode",
    ),
    PROFILE_CASES,
)
def test_all_profiles_load_nested_fixture_with_namespaced_global_records(
    tmp_path: Path,
    dataset: str,
    question_count: int,
    chunk_count: int,
    unique_question_ids: int,
    source_question_id: str,
    raw_question_type: str,
    normalized_question_type: str,
    evidence: object,
    metrics: tuple[EvaluationMetric, ...],
    answer_mode: AnswerMode,
) -> None:
    profile = get_arag_dataset_profile(dataset)
    source = _write_benchmark(
        tmp_path,
        dataset,
        [
            _question(
                dataset,
                source_question_id,
                raw_question_type,
                evidence=evidence,
            )
        ],
    )

    output = ARAGBenchmarkAdapter(profile).load(source, "dev")

    assert set(ARAG_DATASET_PROFILES) == {
        "musique",
        "hotpotqa",
        "2wikimultihop",
        "medical",
        "novel",
    }
    assert profile.reference_question_count == question_count
    assert profile.reference_chunk_count == chunk_count
    assert profile.reference_unique_question_ids == unique_question_ids
    assert profile.reported_metrics == metrics
    assert profile.answer_mode is answer_mode

    scope_id = f"{dataset}:benchmark_exact:dev"
    document_id = f"{dataset}:dev:benchmark_exact:corpus"
    assert output.scope_mode == "global"
    assert output.overlapping_chunks is True
    assert output.gold_support == ()
    assert output.documents[0].doc_id == document_id
    assert output.documents[0].text == ""
    assert output.scopes[0].scope_id == scope_id
    assert output.scopes[0].doc_id == document_id
    assert [chunk.chunk_id for chunk in output.source_chunks] == [
        f"{dataset}:dev:benchmark_exact:c:000000",
        f"{dataset}:dev:benchmark_exact:c:000001",
    ]
    assert [chunk.chunk_pos for chunk in output.source_chunks] == [0, 1]
    assert output.source_chunks[0].text == (
        "Marie Curie lived in Warsaw: a city in Poland."
    )

    benchmark_question = output.benchmark_questions[0]
    assert benchmark_question.question_id == (
        f"{dataset}:benchmark_exact:q:000000"
    )
    assert benchmark_question.source_question_id == source_question_id
    assert benchmark_question.source_row_index == 0
    assert benchmark_question.scope_id == scope_id
    assert benchmark_question.source == dataset
    assert benchmark_question.question_type == normalized_question_type
    assert {artifact.role for artifact in output.source_artifacts} == {
        "chunks",
        "questions",
    }
    assert all(
        f"/{dataset}/" in artifact.path
        for artifact in output.source_artifacts
    )


@pytest.mark.parametrize(
    ("question_id", "expected"),
    [
        ("musique_2hop__one_two", "2_hop"),
        ("musique_3hop1__one_two_three", "3_hop"),
        ("musique_4hop2__one_two_three_four", "4_hop"),
    ],
)
def test_musique_infers_task_type_from_source_question_id(
    tmp_path: Path, question_id: str, expected: str
) -> None:
    source = _write_benchmark(
        tmp_path,
        "musique",
        [_question("musique", question_id, "")],
    )

    output = ARAGBenchmarkAdapter("musique").load(source, "dev")

    assert output.benchmark_questions[0].question_type == expected


def test_novel_duplicate_source_ids_keep_distinct_stable_row_identities(
    tmp_path: Path,
) -> None:
    duplicate_id = "duplicate-source-id"
    source = _write_benchmark(
        tmp_path,
        "novel",
        [
            _question(
                "novel",
                duplicate_id,
                "Fact Retrieval",
                question="What happened first?",
                answer="The first event.",
            ),
            _question(
                "novel",
                duplicate_id,
                "Complex Reasoning",
                question="Why did it happen?",
                answer="Because of the second event.",
            ),
        ],
    )
    adapter = ARAGBenchmarkAdapter("novel")

    first = adapter.load(source, "dev").benchmark_questions
    second = adapter.load(source, "dev").benchmark_questions

    assert [item.question_id for item in first] == [
        "novel:benchmark_exact:q:000000",
        "novel:benchmark_exact:q:000001",
    ]
    assert [item.question_id for item in first] == [
        item.question_id for item in second
    ]
    assert [item.source_question_id for item in first] == [
        duplicate_id,
        duplicate_id,
    ]
    assert [item.source_row_index for item in first] == [0, 1]
    assert len({item.question_id for item in first}) == 2


@pytest.mark.parametrize(
    ("dataset", "question_id", "question_type"),
    [
        ("musique", "musique_2hop__duplicate", ""),
        ("hotpotqa", "duplicate", "bridge"),
        ("2wikimultihop", "duplicate", "inference"),
        ("medical", "duplicate", "Fact Retrieval"),
    ],
)
def test_non_novel_profiles_reject_duplicate_source_question_ids(
    tmp_path: Path,
    dataset: str,
    question_id: str,
    question_type: str,
) -> None:
    source = _write_benchmark(
        tmp_path,
        dataset,
        [
            _question(dataset, question_id, question_type),
            _question(dataset, question_id, question_type),
        ],
    )

    with pytest.raises(
        InputFormatError,
        match=rf"Duplicate A-RAG {dataset} question IDs",
    ):
        ARAGBenchmarkAdapter(dataset).load(source, "dev")


def test_reference_count_validation_is_strict_only_when_requested(
    tmp_path: Path,
) -> None:
    source = _write_benchmark(
        tmp_path,
        "musique",
        [_question("musique", "musique_2hop__fixture", "")],
    )

    permissive = ARAGBenchmarkAdapter("musique").load(source, "dev")
    assert len(permissive.source_chunks) == 2
    assert len(permissive.benchmark_questions) == 1

    with pytest.raises(InputFormatError) as captured:
        ARAGBenchmarkAdapter(
            "musique", validate_reference_counts=True
        ).load(source, "dev")

    message = str(captured.value)
    assert "A-RAG musique reference profile mismatch" in message
    assert "Chunks expected 1354, found 2" in message
    assert "questions expected 1000, found 1" in message
    assert "unique question IDs expected 1000, found 1" in message
    assert "task type counts expected" in message


def test_generic_builder_preserves_chunks_and_isolates_gold_sidecar(
    tmp_path: Path,
    fake_processor: FakeDocumentProcessor,
    fake_embedder: FakeEmbeddingBackend,
) -> None:
    question_marker = "QUESTION_ONLY_8B71E5"
    answer_marker = "ANSWER_ONLY_4C92AF"
    evidence_marker = "EVIDENCE_ONLY_6D03BC"
    source = _write_benchmark(
        tmp_path / "source",
        "musique",
        [
            _question(
                "musique",
                "musique_2hop__builder_fixture",
                "",
                question=f"Where is {question_marker}?",
                answer=answer_marker,
                evidence=evidence_marker,
            )
        ],
        chunks=[
            "0:Marie Curie worked in Warsaw.",
            "1:Warsaw is in Poland: a country in Europe.",
        ],
    )
    output_path = tmp_path / "substrate"

    manifest = SubstrateBuilder(
        BuildConfig(
            corpus_id="arag-musique-fixture",
            split="dev",
            dataset="musique",
            source_format="arag_benchmark_exact",
            max_chunk_tokens=1,
        ),
        processor=fake_processor,
        embedding_backend=fake_embedder,
    ).build(source, output_path)

    assert manifest.dataset == "musique"
    assert manifest.source_format == "arag_benchmark_exact"
    assert manifest.scope_mode == "global"
    assert manifest.preserved_source_chunks is True
    assert manifest.overlapping_chunks is True
    assert manifest.record_counts["documents"] == 1
    assert manifest.record_counts["chunks"] == 2
    assert manifest.record_counts["benchmark_questions"] == 1
    assert {artifact.role for artifact in manifest.source_artifacts} == {
        "chunks",
        "questions",
    }

    substrate = Substrate.open(output_path)
    report = validate_substrate(output_path)
    assert report.valid, report.errors
    assert substrate.chunk_ids_by_scope["musique:benchmark_exact:dev"] == {
        "musique:dev:benchmark_exact:c:000000",
        "musique:dev:benchmark_exact:c:000001",
    }
    assert [chunk.text for chunk in substrate.chunks] == [
        "Marie Curie worked in Warsaw.",
        "Warsaw is in Poland: a country in Europe.",
    ]
    assert not hasattr(substrate, "benchmark_questions")
    assert not hasattr(substrate, "gold_support")

    runtime_values = [
        *(document.title or "" for document in substrate.documents),
        *(chunk.text for chunk in substrate.chunks),
        *(sentence.text for sentence in substrate.sentences),
        *(entity.canonical_name for entity in substrate.entities),
        *(alias.alias for alias in substrate.entity_aliases),
    ]
    runtime_text = "\n".join(runtime_values)
    assert question_marker not in runtime_text
    assert answer_marker not in runtime_text
    assert evidence_marker not in runtime_text

    question_rows = pq.read_table(
        output_path / "evaluation" / "benchmark_questions.parquet"
    ).to_pylist()
    assert question_rows == [
        {
            "question_id": "musique:benchmark_exact:q:000000",
            "scope_id": "musique:benchmark_exact:dev",
            "source": "musique",
            "question": f"Where is {question_marker}?",
            "answer": answer_marker,
            "question_type": "2_hop",
            "source_question_id": "musique_2hop__builder_fixture",
            "source_row_index": 0,
        }
    ]
    assert pq.read_table(
        output_path / "evaluation" / "gold_support.parquet"
    ).num_rows == 0


def test_legacy_hotpotqa_adapter_keeps_existing_public_contract() -> None:
    fixture = Path(__file__).parent / "fixtures" / "benchmark_exact"

    legacy = HotpotQABenchmarkExactAdapter().load(fixture, "dev")
    generic = ARAGBenchmarkAdapter("hotpotqa").load(fixture, "dev")

    assert legacy.documents[0].doc_id == "hotpotqa:dev:benchmark_exact:corpus"
    assert legacy.documents[0].title == "HotpotQA reduced benchmark corpus"
    assert [item.scope_id for item in legacy.scopes] == [
        "hotpotqa:benchmark_exact:dev"
    ]
    assert [item.chunk_id for item in legacy.source_chunks] == [
        "hotpotqa:dev:benchmark_exact:c:000000",
        "hotpotqa:dev:benchmark_exact:c:000001",
    ]
    assert [item.question_id for item in legacy.benchmark_questions] == [
        "benchmark-q1",
        "benchmark-q2",
    ]
    # The compatibility adapter keeps the legacy sidecar shape semantically:
    # its public question ID is still the source ID, while newly introduced
    # generic-only lineage columns remain null.
    assert [item.source_question_id for item in legacy.benchmark_questions] == [
        None,
        None,
    ]
    assert [item.source_row_index for item in legacy.benchmark_questions] == [
        None,
        None,
    ]
    assert legacy.scope_mode == "global"
    assert legacy.overlapping_chunks is True
    assert legacy.gold_support == ()

    assert generic.source_chunks == legacy.source_chunks
    assert generic.scopes == legacy.scopes
    assert [item.question for item in generic.benchmark_questions] == [
        item.question for item in legacy.benchmark_questions
    ]
    assert [item.answer for item in generic.benchmark_questions] == [
        item.answer for item in legacy.benchmark_questions
    ]
    assert [item.question_id for item in generic.benchmark_questions] == [
        "hotpotqa:benchmark_exact:q:000000",
        "hotpotqa:benchmark_exact:q:000001",
    ]
    assert get_arag_dataset_profile("hotpot").key == "hotpotqa"
    assert (
        get_arag_dataset_profile("2WikiMultiHopQA").key
        == "2wikimultihop"
    )
    assert (
        get_arag_dataset_profile("novel").duplicate_question_id_policy
        is DuplicateQuestionIdPolicy.DISAMBIGUATE_WITH_ROW
    )
