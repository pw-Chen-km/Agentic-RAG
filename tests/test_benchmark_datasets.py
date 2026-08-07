from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_rag.cli import app
from agentic_rag.config import BuildConfig
from agentic_rag.evaluation import EpisodeEvaluator, JudgeResponse, ScriptedJudge
from agentic_rag.evaluation.profiles import (
    DATASET_PROFILES,
    AnswerMode,
    get_dataset_profile,
)
from agentic_rag.skillopt.benchmark import (
    prepare_benchmark_splits,
    split_manifest_profile,
)
from agentic_rag.substrate.adapters import BenchmarkExactAdapter
from agentic_rag.substrate.builder import SubstrateBuilder
from agentic_rag.substrate.storage import Substrate
from agentic_rag.substrate.validation import validate_substrate
from conftest import FakeDocumentProcessor, FakeEmbeddingBackend


DATASETS = (
    "2wikimultihop",
    "hotpotqa",
    "medical",
    "musique",
    "novel",
)


def _question_rows(dataset: str, count: int = 24) -> list[dict[str, str]]:
    profile = get_dataset_profile(dataset)
    task_types = profile.allowed_task_types
    return [
        {
            "id": f"{dataset}-q-{index:03d}",
            "source": dataset,
            "question": f"Question {index} for {dataset}?",
            "answer": f"Answer {index}",
            "question_type": task_types[index % len(task_types)],
        }
        for index in range(count)
    ]


def _write_dataset(root: Path, dataset: str, *, count: int = 24) -> Path:
    subset = root / dataset
    subset.mkdir(parents=True)
    (subset / "chunks.json").write_text(
        json.dumps(
            [
                "0:Marie Curie lived in Warsaw.",
                "1:Warsaw is the capital of Poland.",
            ]
        ),
        encoding="utf-8",
    )
    (subset / "questions.json").write_text(
        json.dumps(_question_rows(dataset, count), ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def test_all_five_profiles_are_canonical() -> None:
    assert set(DATASET_PROFILES) == set(DATASETS)
    assert get_dataset_profile("2WikiMultiHopQA").key == "2wikimultihop"
    assert get_dataset_profile("hotpot").key == "hotpotqa"
    assert get_dataset_profile("medical").answer_mode is AnswerMode.LONG
    assert get_dataset_profile("novel").answer_mode is AnswerMode.LONG


@pytest.mark.parametrize("dataset", DATASETS)
def test_benchmark_adapter_loads_every_dataset(
    tmp_path: Path, dataset: str
) -> None:
    source = _write_dataset(tmp_path / "source", dataset)
    output = BenchmarkExactAdapter(dataset).load(source, "dev")

    assert output.scope_mode == "global"
    assert output.scopes[0].scope_id == f"{dataset}:benchmark_exact:dev"
    assert len(output.source_chunks) == 2
    assert len(output.benchmark_questions) == 24
    assert all(item.source == dataset for item in output.benchmark_questions)
    assert all(
        item.question_type in get_dataset_profile(dataset).allowed_task_types
        for item in output.benchmark_questions
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_profile_driven_splits_are_disjoint_and_reloadable(
    tmp_path: Path, dataset: str
) -> None:
    source = _write_dataset(tmp_path / "source", dataset, count=48)
    split_dir = tmp_path / "splits"

    manifest = prepare_benchmark_splits(
        source,
        split_dir,
        dataset=dataset,
        seed=42,
        split_size=6,
        validate_reference_counts=False,
    )

    assert manifest["schema_version"] == "2.0"
    assert manifest["dataset"]["subset"] == dataset
    assert split_manifest_profile(split_dir).key == dataset
    ids = [
        item["id"]
        for split in ("train", "validation", "test")
        for item in manifest["splits"][split]["items"]
    ]
    assert len(ids) == 18
    assert len(ids) == len(set(ids))


def test_generic_builder_keeps_gold_out_of_runtime(
    tmp_path: Path,
    fake_processor: FakeDocumentProcessor,
    fake_embedder: FakeEmbeddingBackend,
) -> None:
    source = _write_dataset(tmp_path / "source", "medical")
    output = tmp_path / "substrate"
    manifest = SubstrateBuilder(
        BuildConfig(
            corpus_id="medical-fixture",
            dataset="medical",
            source_format="benchmark_exact",
        ),
        processor=fake_processor,
        embedding_backend=fake_embedder,
    ).build(source, output)

    assert manifest.dataset == "medical"
    assert manifest.source_format == "benchmark_exact"
    assert manifest.record_counts["benchmark_questions"] == 24
    assert validate_substrate(output).valid
    substrate = Substrate.open(output)
    runtime_text = "\n".join(chunk.text for chunk in substrate.chunks)
    assert "Question 0 for medical" not in runtime_text
    assert "Answer 0" not in runtime_text


def test_long_answer_profile_uses_llm_accuracy_for_both_skillopt_metrics() -> None:
    evaluator = EpisodeEvaluator(
        ScriptedJudge([JudgeResponse(correct=True, raw_output={"correct": True})]),
        profile="medical",
    )
    result = evaluator.evaluate(
        question="Summarize the mechanism.",
        predicted_answer="A supported long answer.",
        gold_answer="A reference long answer.",
    )
    assert result.hard == 1
    assert result.soft == 1


def test_cli_prepares_a_non_hotpot_benchmark_split(tmp_path: Path) -> None:
    source = _write_dataset(tmp_path / "source", "medical", count=48)
    split_dir = tmp_path / "splits"

    result = CliRunner().invoke(
        app,
        [
            "skillopt-prepare",
            "--dataset-dir",
            str(source),
            "--split-dir",
            str(split_dir),
            "--dataset",
            "medical",
            "--split-size",
            "6",
            "--allow-subset",
        ],
    )

    assert result.exit_code == 0, result.output
    assert split_manifest_profile(split_dir).key == "medical"
    assert (split_dir / "test.jsonl").is_file()


@pytest.mark.parametrize("dataset", DATASETS)
def test_checked_in_dataset_build_configs_load(dataset: str) -> None:
    root = Path(__file__).resolve().parents[1]
    config = BuildConfig.from_yaml(root / "configs" / "datasets" / f"{dataset}.yaml")

    assert config.dataset == dataset
    assert config.source_format == "benchmark_exact"
    assert config.validate_benchmark_profile is True
