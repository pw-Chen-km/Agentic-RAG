from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_rag.errors import InputFormatError
from agentic_rag.models import SourceArtifact
from agentic_rag.skillopt.data import (
    ARAG_DATASET_REPO_ID,
    ARAG_DATASET_REVISION,
    HOTPOTQA_BENCHMARK_SCOPE_ID,
    SELECTION_ALGORITHM,
    load_smoke_split,
    prepare_hotpotqa_smoke_splits,
    validate_hotpotqa_smoke_lineage,
)


def _question(question_id: str, question_type: str) -> dict[str, Any]:
    return {
        "id": question_id,
        "source": "hotpotqa",
        "question": f"Question for {question_id}?",
        "answer": f"answer-{question_id}",
        "question_type": question_type,
        "evidence": [["title", [f"evidence for {question_id}"]]],
    }


def _questions() -> list[dict[str, Any]]:
    return [
        *[_question(f"bridge-{index:02d}", "bridge") for index in range(15)],
        *[
            _question(f"comparison-{index:02d}", "comparison")
            for index in range(9)
        ],
    ]


def _write_dataset(
    root: Path,
    questions: list[dict[str, Any]],
    *,
    chunks: list[Any] | None = None,
    nested: bool = True,
) -> Path:
    dataset_dir = root / "dataset"
    data_dir = dataset_dir / "hotpotqa" if nested else dataset_dir
    data_dir.mkdir(parents=True)
    (data_dir / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False),
        encoding="utf-8",
    )
    (data_dir / "chunks.json").write_text(
        json.dumps(chunks or ["0:first Chunk", "1:second Chunk"]),
        encoding="utf-8",
    )
    return dataset_dir


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_type_ids(
    questions: list[dict[str, Any]], question_type: str
) -> list[str]:
    ids = [item["id"] for item in questions if item["question_type"] == question_type]
    return sorted(
        ids,
        key=lambda question_id: (
            hashlib.sha256(f"42\0{question_id}".encode()).hexdigest(),
            question_id,
        ),
    )


def _assert_no_key(value: Any, forbidden_key: str) -> None:
    if isinstance(value, dict):
        assert forbidden_key not in value
        for child in value.values():
            _assert_no_key(child, forbidden_key)
    elif isinstance(value, list):
        for child in value:
            _assert_no_key(child, forbidden_key)


def _substrate_manifest(
    split_manifest: dict[str, Any],
    *,
    question_count: int,
    chunk_count: int,
) -> SimpleNamespace:
    source_files = split_manifest["dataset"]["source_files"]
    return SimpleNamespace(
        record_counts={
            "chunks": chunk_count,
            "benchmark_questions": question_count,
        },
        source_artifacts=[
            SourceArtifact(
                role=role,
                path=source_files[role]["path"],
                sha256=source_files[role]["sha256"],
                size_bytes=source_files[role]["size_bytes"],
            )
            for role in ("chunks", "questions")
        ],
    )


def _rewrite_manifest(split_dir: Path, manifest: dict[str, Any]) -> None:
    (split_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _rewrite_split(
    split_dir: Path,
    manifest: dict[str, Any],
    split_name: str,
    rows: list[dict[str, Any]],
) -> None:
    path = split_dir / f"{split_name}.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    metadata = manifest["splits"][split_name]
    metadata["count"] = len(rows)
    metadata["question_type_counts"] = dict(
        sorted(Counter(row["question_type"] for row in rows).items())
    )
    metadata["items"] = [
        {"id": row["id"], "question_type": row["question_type"]}
        for row in rows
    ]
    metadata["file"] = {
        "path": path.name,
        "sha256": _hash(path),
        "size_bytes": path.stat().st_size,
    }


def _prepared_lineage(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], SimpleNamespace]:
    questions = _questions()
    dataset_dir = _write_dataset(tmp_path, questions)
    split_dir = tmp_path / "splits"
    manifest = prepare_hotpotqa_smoke_splits(
        dataset_dir,
        split_dir,
        expected_question_count=len(questions),
        expected_chunk_count=2,
    )
    return (
        split_dir,
        manifest,
        _substrate_manifest(
            manifest,
            question_count=len(questions),
            chunk_count=2,
        ),
    )


def test_prepare_writes_balanced_disjoint_deterministic_splits(
    tmp_path: Path,
) -> None:
    questions = list(reversed(_questions()))
    dataset_dir = _write_dataset(tmp_path, questions)
    split_dir = tmp_path / "splits"

    manifest = prepare_hotpotqa_smoke_splits(
        dataset_dir,
        split_dir,
        expected_question_count=len(questions),
        expected_chunk_count=2,
    )

    split_items = {
        split: load_smoke_split(split_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    all_ids: list[str] = []
    for items in split_items.values():
        assert len(items) == 6
        assert Counter(item.question_type for item in items) == {
            "bridge": 4,
            "comparison": 2,
        }
        assert {item.scope_id for item in items} == {
            HOTPOTQA_BENCHMARK_SCOPE_ID
        }
        all_ids.extend(item.id for item in items)
    assert len(all_ids) == len(set(all_ids)) == 18

    expected_bridge = _expected_type_ids(questions, "bridge")
    expected_comparison = _expected_type_ids(questions, "comparison")
    assert {
        item.id for item in split_items["train"]
    } == set(expected_bridge[:4] + expected_comparison[:2])
    assert {
        item.id for item in split_items["validation"]
    } == set(expected_bridge[4:8] + expected_comparison[2:4])
    assert {
        item.id for item in split_items["test"]
    } == set(expected_bridge[8:12] + expected_comparison[4:6])

    assert manifest["purpose"] == "workflow_smoke"
    assert manifest["dataset"]["repo_id"] == ARAG_DATASET_REPO_ID
    assert manifest["dataset"]["revision"] == ARAG_DATASET_REVISION
    assert manifest["dataset"]["scope_id"] == HOTPOTQA_BENCHMARK_SCOPE_ID
    assert manifest["selection"]["algorithm"] == SELECTION_ALGORITHM
    assert manifest["selection"]["seed"] == 42
    assert manifest["dataset"]["source_files"]["questions"]["sha256"] == _hash(
        dataset_dir / "hotpotqa" / "questions.json"
    )
    assert manifest["dataset"]["source_files"]["chunks"]["sha256"] == _hash(
        dataset_dir / "hotpotqa" / "chunks.json"
    )
    _assert_no_key(manifest, "answer")
    _assert_no_key(manifest, "evidence")

    on_disk_manifest = json.loads(
        (split_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert on_disk_manifest == manifest
    for split_name in split_items:
        split_file = manifest["splits"][split_name]["file"]
        assert split_file["sha256"] == _hash(split_dir / split_file["path"])


def test_lineage_validation_binds_splits_to_substrate_sources(
    tmp_path: Path,
) -> None:
    split_dir, manifest, substrate_manifest = _prepared_lineage(tmp_path)

    report = validate_hotpotqa_smoke_lineage(
        split_dir,
        substrate_manifest,
        expected_question_count=len(_questions()),
        expected_chunk_count=2,
    )

    assert report["valid"] is True
    assert report["total_question_count"] == 18
    assert set(report["splits"]) == {"train", "validation", "test"}
    assert report["source_artifacts"]["chunks"]["sha256"] == (
        manifest["dataset"]["source_files"]["chunks"]["sha256"]
    )


def test_lineage_validation_rejects_modified_split_bytes(
    tmp_path: Path,
) -> None:
    split_dir, _, substrate_manifest = _prepared_lineage(tmp_path)
    with (split_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(InputFormatError, match="SHA-256"):
        validate_hotpotqa_smoke_lineage(
            split_dir,
            substrate_manifest,
            expected_question_count=len(_questions()),
            expected_chunk_count=2,
        )


def test_lineage_validation_rejects_manifest_id_projection_mismatch(
    tmp_path: Path,
) -> None:
    split_dir, manifest, substrate_manifest = _prepared_lineage(tmp_path)
    manifest["splits"]["train"]["items"][0]["id"] = "substituted-id"
    _rewrite_manifest(split_dir, manifest)

    with pytest.raises(InputFormatError, match="IDs/types"):
        validate_hotpotqa_smoke_lineage(
            split_dir,
            substrate_manifest,
            expected_question_count=len(_questions()),
            expected_chunk_count=2,
        )


def test_lineage_validation_rejects_self_consistent_wrong_type_ratio(
    tmp_path: Path,
) -> None:
    split_dir, manifest, substrate_manifest = _prepared_lineage(tmp_path)
    path = split_dir / "validation.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    comparison = next(
        row for row in rows if row["question_type"] == "comparison"
    )
    comparison["question_type"] = "bridge"
    _rewrite_split(split_dir, manifest, "validation", rows)
    _rewrite_manifest(split_dir, manifest)

    with pytest.raises(InputFormatError, match="4 bridge and 2 comparison"):
        validate_hotpotqa_smoke_lineage(
            split_dir,
            substrate_manifest,
            expected_question_count=len(_questions()),
            expected_chunk_count=2,
        )


def test_lineage_validation_rejects_cross_split_duplicate(
    tmp_path: Path,
) -> None:
    split_dir, manifest, substrate_manifest = _prepared_lineage(tmp_path)
    train_rows = [
        json.loads(line)
        for line in (split_dir / "train.jsonl").read_text("utf-8").splitlines()
    ]
    validation_path = split_dir / "validation.jsonl"
    validation_rows = [
        json.loads(line)
        for line in validation_path.read_text("utf-8").splitlines()
    ]
    replacement = train_rows[0]
    replace_index = next(
        index
        for index, row in enumerate(validation_rows)
        if row["question_type"] == replacement["question_type"]
    )
    validation_rows[replace_index] = replacement
    _rewrite_split(split_dir, manifest, "validation", validation_rows)
    _rewrite_manifest(split_dir, manifest)

    with pytest.raises(InputFormatError, match="must be disjoint"):
        validate_hotpotqa_smoke_lineage(
            split_dir,
            substrate_manifest,
            expected_question_count=len(_questions()),
            expected_chunk_count=2,
        )


def test_lineage_validation_rejects_different_substrate_source(
    tmp_path: Path,
) -> None:
    split_dir, _, substrate_manifest = _prepared_lineage(tmp_path)
    substrate_manifest.source_artifacts[1] = SourceArtifact(
        role="questions",
        path="hotpotqa/questions.json",
        sha256="0" * 64,
        size_bytes=substrate_manifest.source_artifacts[1].size_bytes,
    )

    with pytest.raises(InputFormatError, match="substrate source artifact"):
        validate_hotpotqa_smoke_lineage(
            split_dir,
            substrate_manifest,
            expected_question_count=len(_questions()),
            expected_chunk_count=2,
        )


def test_selection_does_not_depend_on_question_file_order(tmp_path: Path) -> None:
    questions = _questions()
    first_dataset = _write_dataset(tmp_path / "first", questions)
    second_dataset = _write_dataset(tmp_path / "second", list(reversed(questions)))
    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"

    for dataset_dir, output_dir in (
        (first_dataset, first_output),
        (second_dataset, second_output),
    ):
        prepare_hotpotqa_smoke_splits(
            dataset_dir,
            output_dir,
            expected_question_count=len(questions),
            expected_chunk_count=2,
        )

    for split_name in ("train", "validation", "test"):
        assert (first_output / f"{split_name}.jsonl").read_bytes() == (
            second_output / f"{split_name}.jsonl"
        ).read_bytes()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda rows: rows[0].pop("evidence"),
            "missing evidence",
        ),
        (
            lambda rows: rows[0].update(question_type="single-hop"),
            "unsupported question_type",
        ),
        (
            lambda rows: rows[0].update(source="other"),
            "expected 'hotpotqa'",
        ),
        (
            lambda rows: rows[1].update(id=rows[0]["id"]),
            "Duplicate A-RAG HotpotQA question ID",
        ),
    ],
)
def test_prepare_rejects_invalid_question_schema(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    questions = _questions()
    mutate(questions)
    dataset_dir = _write_dataset(tmp_path, questions, nested=False)

    with pytest.raises(InputFormatError, match=message):
        prepare_hotpotqa_smoke_splits(
            dataset_dir,
            tmp_path / "splits",
            expected_question_count=len(questions),
            expected_chunk_count=2,
        )


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        (["1:first Chunk", "0:second Chunk"], "ordered and contiguous"),
        (["0:first Chunk", "0:second Chunk"], "Duplicate"),
        (["0:first Chunk", 2], "must be an 'id:text' string"),
        (["0:   ", "1:second Chunk"], "empty text"),
    ],
)
def test_prepare_rejects_invalid_chunk_schema(
    tmp_path: Path,
    chunks: list[Any],
    message: str,
) -> None:
    questions = _questions()
    dataset_dir = _write_dataset(
        tmp_path,
        questions,
        chunks=chunks,
        nested=False,
    )

    with pytest.raises(InputFormatError, match=message):
        prepare_hotpotqa_smoke_splits(
            dataset_dir,
            tmp_path / "splits",
            expected_question_count=len(questions),
            expected_chunk_count=2,
        )


def test_prepare_enforces_pinned_dataset_counts_by_default(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(tmp_path, _questions())

    with pytest.raises(InputFormatError, match="expected 1311"):
        prepare_hotpotqa_smoke_splits(dataset_dir, tmp_path / "splits")


def test_load_smoke_split_rejects_wrong_scope_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    row = {
        **_question("q1", "bridge"),
        "scope_id": "wrong-scope",
    }
    row.pop("evidence")
    path = tmp_path / "split.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(InputFormatError, match="expected"):
        load_smoke_split(path)

    row["scope_id"] = HOTPOTQA_BENCHMARK_SCOPE_ID
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(InputFormatError, match="Duplicate"):
        load_smoke_split(path)
