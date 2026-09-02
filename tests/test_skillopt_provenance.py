from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentic_rag.errors import InputFormatError
from agentic_rag.skillopt.provenance import (
    PROVENANCE_SPLIT_SCHEMA_VERSION,
    prepare_hotpotqa_provenance_splits,
    validate_hotpotqa_provenance_lineage,
)


def test_provenance_split_preserves_ids_and_adds_support_text(
    tmp_path: Path,
) -> None:
    source, base, substrate = _fixture(tmp_path)
    output = tmp_path / "enriched"
    manifest = prepare_hotpotqa_provenance_splits(
        source_path=source,
        base_split_dir=base,
        substrate_path=substrate,
        output_dir=output,
        expected_source_sha256=_sha256(source),
    )

    assert manifest["schema_version"] == PROVENANCE_SPLIT_SCHEMA_VERSION
    assert manifest["dataset"]["raw_hotpotqa_source"]["sha256"] == _sha256(
        source
    )
    train = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    assert train["id"] == "train-id"
    assert train["supporting_facts"] == [
        {"title": "Alpha", "sentence_id": 1, "text": "Gold train."}
    ]
    report = validate_hotpotqa_provenance_lineage(output, substrate)
    assert report["valid"] is True
    assert report["total_question_count"] == 3


def test_provenance_lineage_rejects_mutated_split(tmp_path: Path) -> None:
    source, base, substrate = _fixture(tmp_path)
    output = tmp_path / "enriched"
    prepare_hotpotqa_provenance_splits(
        source_path=source,
        base_split_dir=base,
        substrate_path=substrate,
        output_dir=output,
        expected_source_sha256=_sha256(source),
    )
    with (output / "validation.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(InputFormatError, match="SHA-256"):
        validate_hotpotqa_provenance_lineage(output, substrate)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "hotpotqa.json"
    source_rows = [
        {
            "_id": f"{split}-id",
            "question": f"{split} question?",
            "answer": f"{split} answer",
            "type": "bridge",
            "supporting_facts": [["Alpha", 1]],
            "context": [["Alpha", ["Distractor.", f"Gold {split}."]]],
        }
        for split in ("train", "validation", "test")
    ]
    source.write_text(json.dumps(source_rows), encoding="utf-8")

    base = tmp_path / "base"
    base.mkdir()
    split_metadata = {}
    for split, row in zip(("train", "validation", "test"), source_rows, strict=True):
        item = {
            "id": row["_id"],
            "question": row["question"],
            "answer": row["answer"],
            "question_type": row["type"],
            "scope_id": "hotpotqa:benchmark_exact:dev",
            "source": "hotpotqa",
        }
        text = json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
        (base / f"{split}.jsonl").write_text(text, encoding="utf-8")
        split_metadata[split] = {"count": 1}
    (base / "split_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "purpose": "workflow_smoke",
                "dataset": {
                    "subset": "hotpotqa",
                    "scope_id": "hotpotqa:benchmark_exact:dev",
                },
                "selection": {"seed": 42},
                "splits": split_metadata,
            }
        ),
        encoding="utf-8",
    )

    substrate = tmp_path / "substrate"
    substrate.mkdir()
    (substrate / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "dataset": "hotpotqa",
                "corpus_id": "fixture",
                "source_format": "hotpotqa_scoped",
                "split": "dev",
                "record_counts": {"chunks": 3},
                "source_artifacts": [
                    {
                        "role": "source",
                        "path": source.as_posix(),
                        "sha256": _sha256(source),
                        "size_bytes": source.stat().st_size,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source, base, substrate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
