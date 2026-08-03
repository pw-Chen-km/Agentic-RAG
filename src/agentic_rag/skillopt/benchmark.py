"""Profile-driven A-RAG benchmark splits and substrate lineage checks."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from agentic_rag.adapters import ARAGBenchmarkAdapter
from agentic_rag.benchmark_profiles import (
    AragDatasetProfile,
    get_arag_dataset_profile,
)
from agentic_rag.errors import InputFormatError
from agentic_rag.skillopt.lineage import (
    atomic_write_text,
    load_json,
    object_field,
    require_mapping,
    sha256_file,
    source_file_manifest,
    validate_source_lineage,
)

SPLIT_ORDER: Final = ("train", "validation", "test")
GENERIC_SPLIT_SCHEMA_VERSION: Final = "2.0"
GENERIC_SELECTION_ALGORITHM: Final = (
    "sha256(f'{seed}\\0{runtime_question_id}')-v2"
)


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """One evaluation-owned item; gold never crosses into the Harness call."""

    id: str
    question: str
    answer: str
    question_type: str
    scope_id: str
    source: str
    source_question_id: str
    source_row_index: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "scope_id": self.scope_id,
            "source": self.source,
            "source_question_id": self.source_question_id,
            "source_row_index": self.source_row_index,
        }


def smoke_type_quotas(
    dataset: str | AragDatasetProfile,
    *,
    split_size: int = 6,
) -> dict[str, int]:
    """Allocate a small stratified split with deterministic largest remainders.

    When the split can cover every task type, each type receives one item first.
    Remaining slots follow the reference task distribution. This keeps a smoke
    run broad without turning its tiny sample into a paper-parity claim.
    """

    profile = _profile(dataset)
    if not isinstance(split_size, int) or isinstance(split_size, bool):
        raise InputFormatError("split_size must be an integer")
    if split_size < 1:
        raise InputFormatError("split_size must be positive")

    counts = dict(profile.reference_task_type_counts)
    ordered_types = sorted(counts)
    quotas = {task_type: 0 for task_type in ordered_types}
    if split_size >= len(ordered_types):
        for task_type in ordered_types:
            quotas[task_type] = 1
        remaining = split_size - len(ordered_types)
    else:
        for task_type in sorted(
            ordered_types,
            key=lambda value: (-counts[value], value),
        )[:split_size]:
            quotas[task_type] = 1
        remaining = 0

    if remaining:
        total = sum(counts.values())
        exact = {
            task_type: remaining * counts[task_type] / total
            for task_type in ordered_types
        }
        floors = {task_type: int(exact[task_type]) for task_type in ordered_types}
        for task_type, count in floors.items():
            quotas[task_type] += count
        leftover = remaining - sum(floors.values())
        for task_type in sorted(
            ordered_types,
            key=lambda value: (
                -(exact[value] - floors[value]),
                -counts[value],
                value,
            ),
        )[:leftover]:
            quotas[task_type] += 1
    return {key: quotas[key] for key in ordered_types if quotas[key]}


def prepare_arag_smoke_splits(
    dataset_dir: Path,
    split_dir: Path,
    *,
    dataset: str | AragDatasetProfile,
    seed: int = 42,
    split_size: int = 6,
    benchmark_split: str = "dev",
    validate_reference_counts: bool = True,
) -> dict[str, Any]:
    """Create deterministic train/validation/test splits for any A-RAG set."""

    profile = _profile(dataset)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise InputFormatError("Smoke split seed must be an integer")
    if not benchmark_split.strip():
        raise InputFormatError("benchmark_split must not be blank")
    dataset_dir = dataset_dir.resolve()
    split_dir = split_dir.resolve()
    quotas = smoke_type_quotas(profile, split_size=split_size)
    adapter_output = ARAGBenchmarkAdapter(
        profile,
        validate_reference_counts=validate_reference_counts,
    ).load(dataset_dir, benchmark_split)

    items = tuple(
        BenchmarkItem(
            id=question.question_id,
            question=question.question,
            answer=question.answer,
            question_type=_required_text(
                question.question_type,
                f"A-RAG {profile.key} question {question.question_id} type",
            ),
            scope_id=question.scope_id,
            source=profile.key,
            source_question_id=_required_text(
                question.source_question_id,
                f"A-RAG {profile.key} question {question.question_id} source ID",
            ),
            source_row_index=_required_row_index(
                question.source_row_index,
                f"A-RAG {profile.key} question {question.question_id}",
            ),
        )
        for question in adapter_output.benchmark_questions
    )
    selected = _select_splits(items, quotas=quotas, seed=seed)
    split_dir.mkdir(parents=True, exist_ok=True)

    output_files: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_ORDER:
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
        atomic_write_text(output_path, text)
        output_files[split_name] = {
            "path": output_path.name,
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
        }

    source_paths = {
        artifact.role: Path(artifact.path)
        for artifact in adapter_output.source_artifacts
    }
    type_counts = Counter(item.question_type for item in items)
    metric_contract = {
        "reported": [metric.value for metric in profile.reported_metrics],
        "hard": profile.skillopt_hard_metric.value,
        "soft": profile.skillopt_soft_metric.value,
    }
    manifest: dict[str, Any] = {
        "schema_version": GENERIC_SPLIT_SCHEMA_VERSION,
        "purpose": "workflow_smoke",
        "paper_parity": False,
        "dataset": {
            "repo_id": profile.repo_id,
            "revision": profile.revision,
            "subset": profile.key,
            "benchmark_split": benchmark_split,
            "scope_id": profile.scope_id(benchmark_split),
            "question_count": len(items),
            "unique_source_question_ids": len(
                {item.source_question_id for item in items}
            ),
            "chunk_count": len(adapter_output.source_chunks),
            "question_type_counts": {
                key: type_counts[key] for key in sorted(type_counts)
            },
            "answer_mode": profile.answer_mode.value,
            "metric_contract": metric_contract,
            "reference_validation": (
                "profile_counts" if validate_reference_counts else "schema_only"
            ),
            "source_files": {
                role: source_file_manifest(path, dataset_dir)
                for role, path in sorted(source_paths.items())
            },
        },
        "selection": {
            "seed": seed,
            "algorithm": GENERIC_SELECTION_ALGORITHM,
            "split_order": list(SPLIT_ORDER),
            "split_size": split_size,
            "per_split_by_task_type": quotas,
        },
        "splits": {
            split_name: {
                "count": len(selected[split_name]),
                "question_type_counts": dict(
                    sorted(
                        Counter(
                            item.question_type for item in selected[split_name]
                        ).items()
                    )
                ),
                "items": [_item_projection(item) for item in selected[split_name]],
                "file": output_files[split_name],
            }
            for split_name in SPLIT_ORDER
        },
    }
    atomic_write_text(
        split_dir / "split_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def load_arag_split(
    path: Path,
    *,
    dataset: str | AragDatasetProfile,
    scope_id: str | None = None,
) -> tuple[BenchmarkItem, ...]:
    """Load a generic split while enforcing its profile-level contract."""

    profile = _profile(dataset)
    expected_scope = scope_id or profile.scope_id("dev")
    if not path.is_file():
        raise InputFormatError(f"Smoke split file does not exist: {path}")
    items: list[BenchmarkItem] = []
    seen_ids: set[str] = set()
    seen_rows: set[int] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise InputFormatError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                item = _item_from_row(
                    raw,
                    profile=profile,
                    scope_id=expected_scope,
                    location=f"{path}:{line_number}",
                )
                if item.id in seen_ids:
                    raise InputFormatError(
                        f"Duplicate smoke split question ID: {item.id}"
                    )
                if item.source_row_index in seen_rows:
                    raise InputFormatError(
                        "Duplicate smoke split source_row_index: "
                        f"{item.source_row_index}"
                    )
                seen_ids.add(item.id)
                seen_rows.add(item.source_row_index)
                items.append(item)
    except json.JSONDecodeError as exc:
        raise InputFormatError(f"Invalid JSON in {path}: {exc}") from exc
    if not items:
        raise InputFormatError(f"Smoke split file is empty: {path}")
    return tuple(items)


def split_manifest_profile(
    split_dir: str | Path,
    *,
    expected_dataset: str | None = None,
) -> AragDatasetProfile:
    """Resolve the dataset profile from either generic or legacy manifests."""

    manifest_path = Path(split_dir) / "split_manifest.json"
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise InputFormatError("split_manifest.json must contain an object")
    dataset = manifest.get("dataset")
    subset = dataset.get("subset") if isinstance(dataset, Mapping) else None
    profile = get_arag_dataset_profile(
        str(subset or expected_dataset or "hotpotqa")
    )
    if expected_dataset is not None:
        expected = get_arag_dataset_profile(expected_dataset)
        if profile.key != expected.key:
            raise InputFormatError(
                f"split manifest dataset is {profile.key!r}; expected "
                f"{expected.key!r}"
            )
    return profile


def validate_arag_smoke_lineage(
    split_dir: Path,
    substrate_manifest: object,
    *,
    dataset: str | AragDatasetProfile | None = None,
) -> dict[str, Any]:
    """Bind a generic split to the exact substrate source artifacts."""

    manifest_path = split_dir / "split_manifest.json"
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise InputFormatError("split_manifest.json must contain an object")
    if manifest.get("schema_version") != GENERIC_SPLIT_SCHEMA_VERSION:
        raise InputFormatError(
            "generic split manifest schema_version must be "
            f"{GENERIC_SPLIT_SCHEMA_VERSION!r}"
        )
    if manifest.get("purpose") != "workflow_smoke":
        raise InputFormatError("split manifest purpose must be 'workflow_smoke'")

    dataset_metadata = require_mapping(manifest, "dataset", "split manifest")
    profile = split_manifest_profile(
        split_dir,
        expected_dataset=(
            _profile(dataset).key if dataset is not None else None
        ),
    )
    benchmark_split = dataset_metadata.get("benchmark_split")
    if not isinstance(benchmark_split, str) or not benchmark_split.strip():
        raise InputFormatError(
            "split manifest dataset.benchmark_split must be non-empty"
        )
    metric_contract = {
        "reported": [metric.value for metric in profile.reported_metrics],
        "hard": profile.skillopt_hard_metric.value,
        "soft": profile.skillopt_soft_metric.value,
    }
    expected_dataset_values = {
        "repo_id": profile.repo_id,
        "revision": profile.revision,
        "subset": profile.key,
        "scope_id": profile.scope_id(benchmark_split),
        "answer_mode": profile.answer_mode.value,
        "metric_contract": metric_contract,
    }
    for key, expected in expected_dataset_values.items():
        if dataset_metadata.get(key) != expected:
            raise InputFormatError(
                f"split manifest dataset.{key} is "
                f"{dataset_metadata.get(key)!r}; expected {expected!r}"
            )

    question_count = _positive_int(
        dataset_metadata.get("question_count"),
        "split manifest dataset.question_count",
    )
    chunk_count = _positive_int(
        dataset_metadata.get("chunk_count"),
        "split manifest dataset.chunk_count",
    )
    unique_source_ids = _positive_int(
        dataset_metadata.get("unique_source_question_ids"),
        "split manifest dataset.unique_source_question_ids",
    )
    reference_validation = dataset_metadata.get("reference_validation")
    if reference_validation not in {"profile_counts", "schema_only"}:
        raise InputFormatError(
            "split manifest dataset.reference_validation is invalid"
        )
    if reference_validation == "profile_counts":
        expected_counts = {
            "question_count": profile.reference_question_count,
            "chunk_count": profile.reference_chunk_count,
            "unique_source_question_ids": profile.reference_unique_question_ids,
        }
        actual_counts = {
            "question_count": question_count,
            "chunk_count": chunk_count,
            "unique_source_question_ids": unique_source_ids,
        }
        if actual_counts != expected_counts:
            raise InputFormatError(
                f"A-RAG {profile.key} reference counts do not match profile"
            )

    raw_type_counts = dataset_metadata.get("question_type_counts")
    if not isinstance(raw_type_counts, Mapping):
        raise InputFormatError(
            "split manifest dataset.question_type_counts must be an object"
        )
    type_counts = {
        str(key): _positive_int(
            value,
            f"split manifest dataset.question_type_counts.{key}",
        )
        for key, value in raw_type_counts.items()
    }
    if set(type_counts) != set(profile.allowed_task_types):
        raise InputFormatError(
            "split manifest dataset question types do not match profile"
        )
    if sum(type_counts.values()) != question_count:
        raise InputFormatError(
            "split manifest dataset question type counts do not sum to total"
        )

    manifest_dataset = object_field(substrate_manifest, "dataset")
    if manifest_dataset != profile.key:
        raise InputFormatError(
            f"substrate dataset is {manifest_dataset!r}; expected {profile.key!r}"
        )
    source_format = object_field(substrate_manifest, "source_format")
    allowed_source_formats = {"arag_benchmark_exact"}
    if profile.key == "hotpotqa":
        allowed_source_formats.add("hotpotqa_benchmark_exact")
    if source_format not in allowed_source_formats:
        raise InputFormatError(
            "generic SkillOpt workflow requires arag_benchmark_exact substrate"
        )
    substrate_split = object_field(substrate_manifest, "split")
    if substrate_split != benchmark_split:
        raise InputFormatError(
            f"substrate split is {substrate_split!r}; expected {benchmark_split!r}"
        )
    record_counts = object_field(substrate_manifest, "record_counts")
    if not isinstance(record_counts, Mapping):
        raise InputFormatError("substrate manifest has no record_counts mapping")
    for key, expected in {
        "chunks": chunk_count,
        "benchmark_questions": question_count,
    }.items():
        if record_counts.get(key) != expected:
            raise InputFormatError(
                f"substrate manifest record_counts.{key} is "
                f"{record_counts.get(key)!r}; expected {expected}"
            )
    source_report = validate_source_lineage(dataset_metadata, substrate_manifest)

    selection = require_mapping(manifest, "selection", "split manifest")
    seed = selection.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise InputFormatError("split manifest selection.seed must be an integer")
    split_size = _positive_int(
        selection.get("split_size"), "split manifest selection.split_size"
    )
    quotas = smoke_type_quotas(profile, split_size=split_size)
    expected_selection = {
        "seed": seed,
        "algorithm": GENERIC_SELECTION_ALGORITHM,
        "split_order": list(SPLIT_ORDER),
        "split_size": split_size,
        "per_split_by_task_type": quotas,
    }
    if dict(selection) != expected_selection:
        raise InputFormatError(
            "split manifest selection does not match the profile-driven contract"
        )

    split_metadata = require_mapping(manifest, "splits", "split manifest")
    if set(split_metadata) != set(SPLIT_ORDER):
        raise InputFormatError(
            "split manifest must contain exactly train, validation, and test"
        )
    all_ids: list[str] = []
    all_rows: list[int] = []
    split_report: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_ORDER:
        metadata = require_mapping(
            split_metadata, split_name, "split manifest splits"
        )
        filename = f"{split_name}.jsonl"
        file_metadata = require_mapping(
            metadata, "file", f"split manifest {split_name}"
        )
        split_path = split_dir / filename
        if file_metadata.get("path") != filename or not split_path.is_file():
            raise InputFormatError(
                f"split manifest {split_name} does not reference {filename}"
            )
        actual_sha = sha256_file(split_path)
        if file_metadata.get("sha256") != actual_sha:
            raise InputFormatError(
                f"{filename} SHA-256 does not match split manifest"
            )
        if file_metadata.get("size_bytes") != split_path.stat().st_size:
            raise InputFormatError(f"{filename} size does not match manifest")
        items = load_arag_split(
            split_path,
            dataset=profile,
            scope_id=profile.scope_id(benchmark_split),
        )
        actual_type_counts = dict(
            sorted(Counter(item.question_type for item in items).items())
        )
        if len(items) != split_size or actual_type_counts != quotas:
            raise InputFormatError(
                f"workflow smoke {split_name} does not match task quotas"
            )
        if metadata.get("count") != len(items):
            raise InputFormatError(
                f"split manifest {split_name} count does not match JSONL"
            )
        if metadata.get("question_type_counts") != quotas:
            raise InputFormatError(
                f"split manifest {split_name} type counts do not match JSONL"
            )
        projection = [_item_projection(item) for item in items]
        if metadata.get("items") != projection:
            raise InputFormatError(
                f"split manifest {split_name} item projection does not match JSONL"
            )
        all_ids.extend(item.id for item in items)
        all_rows.extend(item.source_row_index for item in items)
        split_report[split_name] = {
            "count": len(items),
            "question_type_counts": actual_type_counts,
            "sha256": actual_sha,
        }
    if len(all_ids) != len(set(all_ids)) or len(all_rows) != len(set(all_rows)):
        raise InputFormatError(
            "SkillOpt train/validation/test items must be disjoint"
        )
    return {
        "valid": True,
        "dataset": profile.key,
        "scope_id": profile.scope_id(benchmark_split),
        "metric_contract": metric_contract,
        "split_manifest_sha256": sha256_file(manifest_path),
        "total_question_count": len(all_ids),
        "source_artifacts": source_report,
        "splits": split_report,
    }


def _select_splits(
    items: tuple[BenchmarkItem, ...],
    *,
    quotas: Mapping[str, int],
    seed: int,
) -> dict[str, tuple[BenchmarkItem, ...]]:
    by_type: dict[str, list[BenchmarkItem]] = {
        task_type: [] for task_type in quotas
    }
    for item in items:
        if item.question_type in by_type:
            by_type[item.question_type].append(item)
    for task_type, per_split in quotas.items():
        required = per_split * len(SPLIT_ORDER)
        if len(by_type[task_type]) < required:
            raise InputFormatError(
                f"A-RAG smoke split needs at least {required} {task_type} "
                f"questions; found {len(by_type[task_type])}"
            )
        by_type[task_type].sort(key=lambda item: _rank(item.id, seed))

    offsets = {task_type: 0 for task_type in quotas}
    result: dict[str, tuple[BenchmarkItem, ...]] = {}
    for split_name in SPLIT_ORDER:
        selected: list[BenchmarkItem] = []
        for task_type, per_split in quotas.items():
            offset = offsets[task_type]
            selected.extend(by_type[task_type][offset : offset + per_split])
            offsets[task_type] += per_split
        result[split_name] = tuple(
            sorted(selected, key=lambda item: _rank(item.id, seed))
        )
    return result


def _item_from_row(
    row: Mapping[str, Any],
    *,
    profile: AragDatasetProfile,
    scope_id: str,
    location: str,
) -> BenchmarkItem:
    strings = {
        field: _required_text(row.get(field), f"{location} {field}")
        for field in (
            "id",
            "question",
            "answer",
            "question_type",
            "scope_id",
            "source",
            "source_question_id",
        )
    }
    if strings["scope_id"] != scope_id:
        raise InputFormatError(
            f"{location} has scope_id {strings['scope_id']!r}; expected "
            f"{scope_id!r}"
        )
    if strings["source"] != profile.key:
        raise InputFormatError(
            f"{location} has source {strings['source']!r}; expected {profile.key!r}"
        )
    if strings["question_type"] not in profile.allowed_task_types:
        raise InputFormatError(
            f"{location} has unsupported question_type "
            f"{strings['question_type']!r}"
        )
    item_id = strings["id"]
    if not _safe_path_component(item_id):
        raise InputFormatError(f"{location} id must be one safe path component")
    row_index = _required_row_index(row.get("source_row_index"), location)
    expected_id = f"{profile.key}:benchmark_exact:q:{row_index:06d}"
    if item_id != expected_id:
        raise InputFormatError(
            f"{location} id is {item_id!r}; expected {expected_id!r}"
        )
    return BenchmarkItem(
        id=item_id,
        question=strings["question"],
        answer=strings["answer"],
        question_type=strings["question_type"],
        scope_id=strings["scope_id"],
        source=strings["source"],
        source_question_id=strings["source_question_id"],
        source_row_index=row_index,
    )


def _item_projection(item: BenchmarkItem) -> dict[str, str | int]:
    return {
        "id": item.id,
        "source_question_id": item.source_question_id,
        "source_row_index": item.source_row_index,
        "question_type": item.question_type,
    }


def _rank(question_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{seed}\0{question_id}".encode("utf-8")
    ).hexdigest()
    return digest, question_id


def _profile(value: str | AragDatasetProfile) -> AragDatasetProfile:
    return (
        value
        if isinstance(value, AragDatasetProfile)
        else get_arag_dataset_profile(value)
    )


def _required_text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputFormatError(f"{location} must be a non-empty string")
    return value


def _required_row_index(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputFormatError(
            f"{location} source_row_index must be a non-negative integer"
        )
    return value


def _positive_int(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise InputFormatError(f"{location} must be a positive integer")
    return value


def _safe_path_component(value: str) -> bool:
    return not (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    )


__all__ = [
    "BenchmarkItem",
    "GENERIC_SELECTION_ALGORITHM",
    "GENERIC_SPLIT_SCHEMA_VERSION",
    "SPLIT_ORDER",
    "load_arag_split",
    "prepare_arag_smoke_splits",
    "smoke_type_quotas",
    "split_manifest_profile",
    "validate_arag_smoke_lineage",
]
