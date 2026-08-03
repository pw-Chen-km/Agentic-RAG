"""Deterministic A-RAG HotpotQA smoke-split preparation.

This module owns evaluation data, including gold answers.  Target-agent code
must consume only the ``id``, ``question``, and ``scope_id`` fields projected
by the SkillOpt rollout adapter.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from agentic_rag.benchmark_profiles import (
    ARAG_DATASET_REPO_ID,
    ARAG_DATASET_REVISION,
    get_arag_dataset_profile,
)
from agentic_rag.errors import InputFormatError
from agentic_rag.skillopt.lineage import (
    atomic_write_text as _atomic_write_text,
    load_json as _load_json,
    object_field as _object_field,
    require_mapping as _require_mapping,
    sha256_file as _sha256,
    source_file_manifest as _source_file_manifest,
    validate_source_lineage as _validate_source_lineage,
)

ARAG_HOTPOTQA_SUBSET: Final = "hotpotqa"
_HOTPOTQA_PROFILE: Final = get_arag_dataset_profile("hotpotqa")
HOTPOTQA_BENCHMARK_SCOPE_ID: Final = _HOTPOTQA_PROFILE.scope_id("dev")
ARAG_HOTPOTQA_QUESTION_COUNT: Final = (
    _HOTPOTQA_PROFILE.reference_question_count
)
ARAG_HOTPOTQA_CHUNK_COUNT: Final = _HOTPOTQA_PROFILE.reference_chunk_count
ARAG_HOTPOTQA_QUESTIONS_SHA256: Final = (
    "ecc641d532a4d2518f1ceb57627f2e41044e0c4fd07012bf0aaa02327dc770a9"
)
ARAG_HOTPOTQA_CHUNKS_SHA256: Final = (
    "cb76f6fdb54e7b2853d51d400bacdba01c814baf43b74207bc79c2a06474d231"
)
SMOKE_SPLIT_SEED: Final = 42
SMOKE_SPLIT_ORDER: Final = ("train", "validation", "test")
SMOKE_BRIDGE_PER_SPLIT: Final = 4
SMOKE_COMPARISON_PER_SPLIT: Final = 2
SELECTION_ALGORITHM: Final = "sha256(f'{seed}\\0{question_id}')-v1"

QuestionType = Literal["bridge", "comparison"]


@dataclass(frozen=True, slots=True)
class SmokeBenchmarkItem:
    """One scored benchmark item used by a SkillOpt rollout.

    ``answer`` is evaluation-only.  It must never be included in the target
    Policy or Answer Generator input.
    """

    id: str
    question: str
    answer: str
    question_type: QuestionType
    scope_id: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "scope_id": self.scope_id,
            "source": self.source,
        }


def prepare_hotpotqa_smoke_splits(
    dataset_dir: Path,
    split_dir: Path,
    *,
    seed: int = SMOKE_SPLIT_SEED,
    expected_question_count: int | None = ARAG_HOTPOTQA_QUESTION_COUNT,
    expected_chunk_count: int | None = ARAG_HOTPOTQA_CHUNK_COUNT,
) -> dict[str, Any]:
    """Validate a local pinned A-RAG dataset and write 6/6/6 smoke splits.

    The default count checks identify the exact reduced HotpotQA benchmark.
    Tests and deliberately reduced local fixtures may override the two
    expected counts without changing the selection algorithm.
    """

    if not isinstance(seed, int):
        raise InputFormatError("Smoke split seed must be an integer")
    _validate_expected_count("question", expected_question_count)
    _validate_expected_count("Chunk", expected_chunk_count)

    chunks_path, questions_path = _resolve_hotpotqa_files(dataset_dir)
    chunks = _load_json(chunks_path)
    questions = _load_json(questions_path)
    chunk_count = _validate_chunks(chunks, expected_chunk_count)
    items = _validate_questions(questions, expected_question_count)
    if (
        expected_question_count == ARAG_HOTPOTQA_QUESTION_COUNT
        and expected_chunk_count == ARAG_HOTPOTQA_CHUNK_COUNT
    ):
        _require_sha256(
            questions_path,
            expected=ARAG_HOTPOTQA_QUESTIONS_SHA256,
            role="questions.json",
        )
        _require_sha256(
            chunks_path,
            expected=ARAG_HOTPOTQA_CHUNKS_SHA256,
            role="chunks.json",
        )

    selected = _select_splits(items, seed=seed)
    split_dir.mkdir(parents=True, exist_ok=True)

    output_files: dict[str, dict[str, Any]] = {}
    for split_name in SMOKE_SPLIT_ORDER:
        output_path = split_dir / f"{split_name}.jsonl"
        text = "".join(
            json.dumps(
                item.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in selected[split_name]
        )
        _atomic_write_text(output_path, text)
        output_files[split_name] = {
            "path": output_path.name,
            "sha256": _sha256(output_path),
            "size_bytes": output_path.stat().st_size,
        }

    type_counts = Counter(item.question_type for item in items)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "purpose": "workflow_smoke",
        "dataset": {
            "repo_id": ARAG_DATASET_REPO_ID,
            "revision": ARAG_DATASET_REVISION,
            "subset": ARAG_HOTPOTQA_SUBSET,
            "scope_id": HOTPOTQA_BENCHMARK_SCOPE_ID,
            "question_count": len(items),
            "chunk_count": chunk_count,
            "question_type_counts": {
                key: type_counts[key] for key in sorted(type_counts)
            },
            "source_files": {
                "chunks": _source_file_manifest(chunks_path, dataset_dir),
                "questions": _source_file_manifest(
                    questions_path, dataset_dir
                ),
            },
        },
        "selection": {
            "seed": seed,
            "algorithm": SELECTION_ALGORITHM,
            "split_order": list(SMOKE_SPLIT_ORDER),
            "per_split": {
                "bridge": SMOKE_BRIDGE_PER_SPLIT,
                "comparison": SMOKE_COMPARISON_PER_SPLIT,
                "total": (
                    SMOKE_BRIDGE_PER_SPLIT
                    + SMOKE_COMPARISON_PER_SPLIT
                ),
            },
        },
        "splits": {
            split_name: {
                "count": len(selected[split_name]),
                "question_type_counts": dict(
                    sorted(
                        Counter(
                            item.question_type
                            for item in selected[split_name]
                        ).items()
                    )
                ),
                "items": [
                    {"id": item.id, "question_type": item.question_type}
                    for item in selected[split_name]
                ],
                "file": output_files[split_name],
            }
            for split_name in SMOKE_SPLIT_ORDER
        },
    }
    _atomic_write_text(
        split_dir / "split_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    return manifest


def load_smoke_split(path: Path) -> tuple[SmokeBenchmarkItem, ...]:
    """Load and validate a JSONL split produced by this module."""

    if not path.is_file():
        raise InputFormatError(f"Smoke split file does not exist: {path}")
    items: list[SmokeBenchmarkItem] = []
    seen_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise InputFormatError(
                        f"Smoke split line {line_number} is not an object"
                    )
                item = _smoke_item_from_row(
                    row,
                    location=f"Smoke split line {line_number}",
                    require_global_scope=True,
                )
                if item.id in seen_ids:
                    raise InputFormatError(
                        f"Duplicate smoke split question ID: {item.id}"
                    )
                seen_ids.add(item.id)
                items.append(item)
    except json.JSONDecodeError as exc:
        raise InputFormatError(f"Invalid JSON in {path}: {exc}") from exc
    if not items:
        raise InputFormatError(f"Smoke split file is empty: {path}")
    return tuple(items)


def validate_hotpotqa_smoke_lineage(
    split_dir: Path,
    substrate_manifest: object,
    *,
    expected_question_count: int = ARAG_HOTPOTQA_QUESTION_COUNT,
    expected_chunk_count: int = ARAG_HOTPOTQA_CHUNK_COUNT,
) -> dict[str, Any]:
    """Verify prepared splits and their source identity before training.

    This check deliberately uses both sides of the provenance chain:

    * each JSONL file must match the digest and item projection recorded by
      ``split_manifest.json``;
    * the manifest's original ``questions.json`` and ``chunks.json`` digests
      must match the source artifacts recorded when the substrate was built.

    The second check prevents a valid split from being paired with an index
    built from a different A-RAG corpus checkout.
    """

    _validate_expected_count("question", expected_question_count)
    _validate_expected_count("Chunk", expected_chunk_count)
    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.is_file():
        raise InputFormatError(
            f"Missing prepared split manifest: {manifest_path}"
        )
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise InputFormatError("split_manifest.json must contain an object")
    if manifest.get("schema_version") != "1.0":
        raise InputFormatError(
            "split manifest schema_version must be '1.0'"
        )
    if manifest.get("purpose") != "workflow_smoke":
        raise InputFormatError(
            "split manifest purpose must be 'workflow_smoke'"
        )

    dataset = _require_mapping(manifest, "dataset", "split manifest")
    expected_dataset_values = {
        "repo_id": ARAG_DATASET_REPO_ID,
        "revision": ARAG_DATASET_REVISION,
        "subset": ARAG_HOTPOTQA_SUBSET,
        "scope_id": HOTPOTQA_BENCHMARK_SCOPE_ID,
        "question_count": expected_question_count,
        "chunk_count": expected_chunk_count,
    }
    for key, expected in expected_dataset_values.items():
        if dataset.get(key) != expected:
            raise InputFormatError(
                f"split manifest dataset.{key} is {dataset.get(key)!r}; "
                f"expected {expected!r}"
            )
    dataset_type_counts = dataset.get("question_type_counts")
    if (
        not isinstance(dataset_type_counts, Mapping)
        or set(dataset_type_counts) != {"bridge", "comparison"}
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in dataset_type_counts.values()
        )
        or sum(dataset_type_counts.values()) != expected_question_count
    ):
        raise InputFormatError(
            "split manifest dataset.question_type_counts are invalid"
        )

    selection = _require_mapping(manifest, "selection", "split manifest")
    expected_selection = {
        "seed": SMOKE_SPLIT_SEED,
        "algorithm": SELECTION_ALGORITHM,
        "split_order": list(SMOKE_SPLIT_ORDER),
        "per_split": {
            "bridge": SMOKE_BRIDGE_PER_SPLIT,
            "comparison": SMOKE_COMPARISON_PER_SPLIT,
            "total": SMOKE_BRIDGE_PER_SPLIT + SMOKE_COMPARISON_PER_SPLIT,
        },
    }
    if dict(selection) != expected_selection:
        raise InputFormatError(
            "split manifest selection does not match the workflow-smoke "
            "seed, algorithm, and 4/2 split contract"
        )

    record_counts = _object_field(substrate_manifest, "record_counts")
    if not isinstance(record_counts, Mapping):
        raise InputFormatError(
            "substrate manifest has no record_counts mapping"
        )
    expected_records = {
        "chunks": expected_chunk_count,
        "benchmark_questions": expected_question_count,
    }
    for key, expected in expected_records.items():
        if record_counts.get(key) != expected:
            raise InputFormatError(
                f"substrate manifest record_counts.{key} is "
                f"{record_counts.get(key)!r}; expected {expected}"
            )

    source_report = _validate_source_lineage(dataset, substrate_manifest)
    if (
        expected_question_count == ARAG_HOTPOTQA_QUESTION_COUNT
        and expected_chunk_count == ARAG_HOTPOTQA_CHUNK_COUNT
    ):
        pinned_hashes = {
            "questions": ARAG_HOTPOTQA_QUESTIONS_SHA256,
            "chunks": ARAG_HOTPOTQA_CHUNKS_SHA256,
        }
        for role, expected_hash in pinned_hashes.items():
            if source_report[role]["sha256"] != expected_hash:
                raise InputFormatError(
                    f"split/substrate {role} source SHA-256 does not match "
                    f"pinned A-RAG revision {ARAG_DATASET_REVISION}"
                )
    split_metadata = _require_mapping(manifest, "splits", "split manifest")
    if set(split_metadata) != set(SMOKE_SPLIT_ORDER):
        raise InputFormatError(
            "split manifest must contain exactly train, validation, and test"
        )

    split_report: dict[str, dict[str, Any]] = {}
    all_ids: list[str] = []
    for split_name in SMOKE_SPLIT_ORDER:
        metadata = _require_mapping(
            split_metadata,
            split_name,
            "split manifest splits",
        )
        expected_filename = f"{split_name}.jsonl"
        file_metadata = _require_mapping(
            metadata,
            "file",
            f"split manifest {split_name}",
        )
        if file_metadata.get("path") != expected_filename:
            raise InputFormatError(
                f"split manifest {split_name} file.path must be "
                f"{expected_filename!r}"
            )
        split_path = split_dir / expected_filename
        if not split_path.is_file():
            raise InputFormatError(f"Missing prepared split file: {split_path}")
        actual_sha256 = _sha256(split_path)
        if file_metadata.get("sha256") != actual_sha256:
            raise InputFormatError(
                f"{expected_filename} SHA-256 does not match split manifest"
            )
        actual_size = split_path.stat().st_size
        if file_metadata.get("size_bytes") != actual_size:
            raise InputFormatError(
                f"{expected_filename} size does not match split manifest"
            )

        items = load_smoke_split(split_path)
        if any(item.source != ARAG_HOTPOTQA_SUBSET for item in items):
            raise InputFormatError(
                f"{expected_filename} contains a non-HotpotQA source"
            )
        actual_counts = Counter(item.question_type for item in items)
        expected_counts = {
            "bridge": SMOKE_BRIDGE_PER_SPLIT,
            "comparison": SMOKE_COMPARISON_PER_SPLIT,
        }
        if len(items) != sum(expected_counts.values()):
            raise InputFormatError(
                f"workflow smoke {split_name} split must contain 6 items"
            )
        if dict(actual_counts) != expected_counts:
            raise InputFormatError(
                f"workflow smoke {split_name} split must contain exactly "
                "4 bridge and 2 comparison questions"
            )
        if metadata.get("count") != len(items):
            raise InputFormatError(
                f"split manifest {split_name} count does not match JSONL"
            )
        if metadata.get("question_type_counts") != expected_counts:
            raise InputFormatError(
                f"split manifest {split_name} question_type_counts do not "
                "match JSONL"
            )
        actual_projection = [
            {"id": item.id, "question_type": item.question_type}
            for item in items
        ]
        if metadata.get("items") != actual_projection:
            raise InputFormatError(
                f"split manifest {split_name} IDs/types do not match JSONL"
            )
        all_ids.extend(item.id for item in items)
        split_report[split_name] = {
            "count": len(items),
            "question_type_counts": expected_counts,
            "sha256": actual_sha256,
        }

    if len(all_ids) != len(set(all_ids)):
        raise InputFormatError(
            "SkillOpt train/validation/test question IDs must be disjoint"
        )
    return {
        "valid": True,
        "split_manifest_sha256": _sha256(manifest_path),
        "total_question_count": len(all_ids),
        "source_artifacts": source_report,
        "splits": split_report,
    }


def _resolve_hotpotqa_files(dataset_dir: Path) -> tuple[Path, Path]:
    if not dataset_dir.is_dir():
        raise InputFormatError(
            f"A-RAG dataset directory does not exist: {dataset_dir}"
        )
    direct = dataset_dir
    nested = dataset_dir / ARAG_HOTPOTQA_SUBSET
    for root in (direct, nested):
        chunks_path = root / "chunks.json"
        questions_path = root / "questions.json"
        if chunks_path.is_file() and questions_path.is_file():
            return chunks_path, questions_path
    raise InputFormatError(
        "A-RAG dataset directory must contain chunks.json and questions.json "
        "directly or under hotpotqa/"
    )


def _validate_chunks(raw: Any, expected_count: int | None) -> int:
    if not isinstance(raw, list):
        raise InputFormatError(
            "A-RAG HotpotQA chunks.json must be a JSON list"
        )
    seen_ids: set[int] = set()
    positions: list[int] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, str):
            raise InputFormatError(
                f"A-RAG HotpotQA Chunk {index} must be an 'id:text' string"
            )
        source_id, separator, text = entry.partition(":")
        if not separator or not source_id.isdigit():
            raise InputFormatError(
                f"A-RAG HotpotQA Chunk {index} must use a numeric 'id:text' prefix"
            )
        numeric_id = int(source_id)
        if numeric_id in seen_ids:
            raise InputFormatError(
                f"Duplicate A-RAG HotpotQA Chunk ID: {numeric_id}"
            )
        if not text.strip():
            raise InputFormatError(
                f"A-RAG HotpotQA Chunk {numeric_id} has empty text"
            )
        seen_ids.add(numeric_id)
        positions.append(numeric_id)
    if positions != list(range(len(raw))):
        raise InputFormatError(
            "A-RAG HotpotQA Chunk IDs must be ordered and contiguous from 0"
        )
    if expected_count is not None and len(raw) != expected_count:
        raise InputFormatError(
            "A-RAG HotpotQA chunks.json contains "
            f"{len(raw)} Chunks; expected {expected_count}"
        )
    return len(raw)


def _validate_questions(
    raw: Any, expected_count: int | None
) -> tuple[SmokeBenchmarkItem, ...]:
    if not isinstance(raw, list):
        raise InputFormatError(
            "A-RAG HotpotQA questions.json must be a JSON list"
        )
    items: list[SmokeBenchmarkItem] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise InputFormatError(
                f"A-RAG HotpotQA question {index} must be an object"
            )
        if "evidence" not in row:
            raise InputFormatError(
                f"A-RAG HotpotQA question {index} is missing evidence"
            )
        item = _smoke_item_from_row(
            {
                **row,
                "scope_id": HOTPOTQA_BENCHMARK_SCOPE_ID,
            },
            location=f"A-RAG HotpotQA question {index}",
            require_global_scope=True,
        )
        if item.source != ARAG_HOTPOTQA_SUBSET:
            raise InputFormatError(
                f"A-RAG HotpotQA question {item.id} has source "
                f"{item.source!r}; expected 'hotpotqa'"
            )
        if item.id in seen_ids:
            raise InputFormatError(
                f"Duplicate A-RAG HotpotQA question ID: {item.id}"
            )
        seen_ids.add(item.id)
        items.append(item)
    if expected_count is not None and len(items) != expected_count:
        raise InputFormatError(
            "A-RAG HotpotQA questions.json contains "
            f"{len(items)} questions; expected {expected_count}"
        )
    return tuple(items)


def _smoke_item_from_row(
    row: dict[str, Any],
    *,
    location: str,
    require_global_scope: bool,
) -> SmokeBenchmarkItem:
    string_fields: dict[str, str] = {}
    for field_name in ("id", "question", "answer", "source", "scope_id"):
        value = row.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise InputFormatError(
                f"{location} has no non-empty string {field_name}"
            )
        string_fields[field_name] = value
    raw_question_type = row.get("question_type")
    if raw_question_type not in ("bridge", "comparison"):
        raise InputFormatError(
            f"{location} has unsupported question_type "
            f"{raw_question_type!r}; expected 'bridge' or 'comparison'"
        )
    scope_id = string_fields["scope_id"]
    if require_global_scope and scope_id != HOTPOTQA_BENCHMARK_SCOPE_ID:
        raise InputFormatError(
            f"{location} has scope_id {scope_id!r}; expected "
            f"{HOTPOTQA_BENCHMARK_SCOPE_ID!r}"
        )
    return SmokeBenchmarkItem(
        id=string_fields["id"],
        question=string_fields["question"],
        answer=string_fields["answer"],
        question_type=raw_question_type,
        scope_id=scope_id,
        source=string_fields["source"],
    )


def _select_splits(
    items: tuple[SmokeBenchmarkItem, ...], *, seed: int
) -> dict[str, tuple[SmokeBenchmarkItem, ...]]:
    by_type: dict[QuestionType, list[SmokeBenchmarkItem]] = {
        "bridge": [],
        "comparison": [],
    }
    for item in items:
        by_type[item.question_type].append(item)
    for question_type, required_per_split in (
        ("bridge", SMOKE_BRIDGE_PER_SPLIT),
        ("comparison", SMOKE_COMPARISON_PER_SPLIT),
    ):
        required = required_per_split * len(SMOKE_SPLIT_ORDER)
        if len(by_type[question_type]) < required:
            raise InputFormatError(
                f"A-RAG HotpotQA smoke split needs at least {required} "
                f"{question_type} questions; found {len(by_type[question_type])}"
            )
        by_type[question_type].sort(key=lambda item: _rank(item.id, seed))

    result: dict[str, tuple[SmokeBenchmarkItem, ...]] = {}
    bridge_offset = 0
    comparison_offset = 0
    for split_name in SMOKE_SPLIT_ORDER:
        selected = (
            by_type["bridge"][
                bridge_offset : bridge_offset + SMOKE_BRIDGE_PER_SPLIT
            ]
            + by_type["comparison"][
                comparison_offset : (
                    comparison_offset + SMOKE_COMPARISON_PER_SPLIT
                )
            ]
        )
        result[split_name] = tuple(
            sorted(selected, key=lambda item: _rank(item.id, seed))
        )
        bridge_offset += SMOKE_BRIDGE_PER_SPLIT
        comparison_offset += SMOKE_COMPARISON_PER_SPLIT

    selected_ids = [item.id for split in result.values() for item in split]
    if len(selected_ids) != len(set(selected_ids)):
        raise InputFormatError(
            "Deterministic smoke split unexpectedly selected duplicate IDs"
        )
    return result


def _rank(question_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{seed}\0{question_id}".encode("utf-8")
    ).hexdigest()
    return digest, question_id


def _validate_expected_count(name: str, value: int | None) -> None:
    if value is not None and (not isinstance(value, int) or value < 1):
        raise InputFormatError(
            f"Expected {name} count must be a positive integer or null"
        )


def _require_sha256(path: Path, *, expected: str, role: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise InputFormatError(
            f"A-RAG HotpotQA {role} SHA-256 {actual} does not match "
            f"pinned revision {ARAG_DATASET_REVISION} ({expected})"
        )
