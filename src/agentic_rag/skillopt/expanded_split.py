"""Deterministically expand a frozen SkillOpt split without contaminating held-out data."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentic_rag.errors import InputFormatError


EXPANDED_SPLIT_SCHEMA_VERSION = "1.0"
EXPANDED_SELECTION_ALGORITHM = (
    "preserve-base-then-sha256(f'{seed}\\0{question_id}')-stratified-v1"
)
_SPLITS = ("train", "validation", "test")
_QUESTION_TYPES = ("bridge", "comparison")


def prepare_expanded_hotpotqa_splits(
    *,
    questions_path: str | Path,
    base_split_dir: str | Path,
    excluded_questions_path: str | Path,
    output_dir: str | Path,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Preserve an existing split and deterministically extend it to new sizes.

    The external held-out question file is a required exclusion source.  This
    makes data separation explicit and auditable instead of relying on a later
    overlap check.
    """

    targets = {
        "train": _type_counts(train_size),
        "validation": _type_counts(validation_size),
        "test": _type_counts(test_size),
    }
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise InputFormatError("seed must be an integer")

    questions_file = Path(questions_path).resolve()
    base = Path(base_split_dir).resolve()
    excluded_file = Path(excluded_questions_path).resolve()
    destination = Path(output_dir).resolve()

    source_rows = _load_json_list(questions_file, "HotpotQA questions")
    source_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(source_rows):
        row = _project_question(raw, location=f"HotpotQA question {index}")
        question_id = str(row["id"])
        if question_id in source_by_id:
            raise InputFormatError(f"duplicate HotpotQA question ID: {question_id}")
        source_by_id[question_id] = row

    excluded_rows = _load_json_list(excluded_file, "held-out questions")
    excluded_ids = {
        str(row.get("id") or row.get("_id") or "").strip()
        for row in excluded_rows
        if isinstance(row, Mapping)
    }
    if "" in excluded_ids or len(excluded_ids) != len(excluded_rows):
        raise InputFormatError("held-out questions contain missing or duplicate IDs")
    unknown_excluded = excluded_ids.difference(source_by_id)
    if unknown_excluded:
        raise InputFormatError(
            f"held-out IDs are absent from HotpotQA source: {sorted(unknown_excluded)[:3]}"
        )

    base_manifest_path = base / "split_manifest.json"
    base_manifest = _load_json_object(base_manifest_path, "base split manifest")
    base_rows: dict[str, list[dict[str, Any]]] = {}
    used_ids: set[str] = set()
    for split in _SPLITS:
        rows = [
            _project_question(row, location=f"base {split}")
            for row in _load_jsonl(base / f"{split}.jsonl")
        ]
        for row in rows:
            question_id = str(row["id"])
            source = source_by_id.get(question_id)
            if source is None or any(
                row[field] != source[field]
                for field in (
                    "question",
                    "answer",
                    "question_type",
                    "scope_id",
                    "source",
                )
            ):
                raise InputFormatError(
                    f"base split item {question_id!r} does not match source"
                )
            if question_id in used_ids:
                raise InputFormatError(
                    f"duplicate base split question ID: {question_id}"
                )
            if question_id in excluded_ids:
                raise InputFormatError(
                    f"base split overlaps held-out question: {question_id}"
                )
            used_ids.add(question_id)
        base_rows[split] = rows

    for split in _SPLITS:
        current = Counter(str(row["question_type"]) for row in base_rows[split])
        for question_type in _QUESTION_TYPES:
            if current[question_type] > targets[split][question_type]:
                raise InputFormatError(
                    f"requested {split} {question_type} target is smaller than frozen base"
                )

    candidates: dict[str, list[dict[str, Any]]] = {kind: [] for kind in _QUESTION_TYPES}
    for question_id, row in source_by_id.items():
        if question_id not in used_ids and question_id not in excluded_ids:
            candidates[str(row["question_type"])].append(row)
    for rows in candidates.values():
        rows.sort(key=lambda row: _rank(str(row["id"]), seed))

    cursors = {kind: 0 for kind in _QUESTION_TYPES}
    rendered_rows: dict[str, list[dict[str, Any]]] = {}
    for split in _SPLITS:
        rows = list(base_rows[split])
        current = Counter(str(row["question_type"]) for row in rows)
        for question_type in _QUESTION_TYPES:
            needed = targets[split][question_type] - current[question_type]
            start = cursors[question_type]
            stop = start + needed
            additions = candidates[question_type][start:stop]
            if len(additions) != needed:
                raise InputFormatError(
                    f"not enough {question_type} questions to expand {split}"
                )
            cursors[question_type] = stop
            rows.extend(additions)
            used_ids.update(str(row["id"]) for row in additions)
        rows.sort(key=lambda row: _rank(str(row["id"]), seed))
        rendered_rows[split] = rows

    if used_ids.intersection(excluded_ids):
        raise InputFormatError("expanded split overlaps held-out questions")
    expected_total = train_size + validation_size + test_size
    if len(used_ids) != expected_total:
        raise InputFormatError(
            f"expanded split selected {len(used_ids)} unique IDs; expected {expected_total}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    split_metadata: dict[str, dict[str, Any]] = {}
    for split in _SPLITS:
        text = "".join(_canonical_json(row) + "\n" for row in rendered_rows[split])
        path = destination / f"{split}.jsonl"
        _atomic_write(path, text)
        split_metadata[split] = {
            "count": len(rendered_rows[split]),
            "question_type_counts": dict(
                sorted(
                    Counter(
                        str(row["question_type"]) for row in rendered_rows[split]
                    ).items()
                )
            ),
            "items": [
                {"id": str(row["id"]), "question_type": str(row["question_type"])}
                for row in rendered_rows[split]
            ],
            "file": {
                "path": path.name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            },
        }

    manifest: dict[str, Any] = {
        "schema_version": EXPANDED_SPLIT_SCHEMA_VERSION,
        "purpose": "skillopt_expanded_trajectory_ablation",
        "dataset": base_manifest.get("dataset"),
        "selection": {
            "algorithm": EXPANDED_SELECTION_ALGORITHM,
            "seed": seed,
            "split_order": list(_SPLITS),
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
            "per_split_by_question_type": targets,
            "preserves_base_split": True,
            "base_split_manifest": {
                "path": base_manifest_path.as_posix(),
                "sha256": _sha256_file(base_manifest_path),
            },
            "excluded_questions": {
                "path": excluded_file.as_posix(),
                "sha256": _sha256_file(excluded_file),
                "count": len(excluded_ids),
            },
        },
        "splits": split_metadata,
    }
    _atomic_write(
        destination / "split_manifest.json",
        json.dumps(
            manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
    )
    return manifest


def _type_counts(size: int) -> dict[str, int]:
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise InputFormatError("split sizes must be positive integers")
    # Pinned HotpotQA-1k distribution: 811 bridge / 189 comparison.
    comparison = round(size * 189 / 1000)
    comparison = max(1, min(size - 1, comparison)) if size > 1 else 0
    return {"bridge": size - comparison, "comparison": comparison}


def _project_question(raw: object, *, location: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise InputFormatError(f"{location} must be an object")
    required = ("id", "question", "answer", "question_type", "scope_id", "source")
    values = {field: raw.get(field) for field in required}
    values["scope_id"] = values["scope_id"] or "hotpotqa:benchmark_exact:dev"
    values["source"] = values["source"] or "hotpotqa"
    if any(
        not isinstance(value, str) or not value.strip() for value in values.values()
    ):
        raise InputFormatError(f"{location} has missing required string fields")
    if values["question_type"] not in _QUESTION_TYPES:
        raise InputFormatError(f"{location} has invalid question_type")
    return {field: str(values[field]) for field in required}


def _rank(question_id: str, seed: int) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{seed}\0{question_id}".encode("utf-8")).digest()
    return digest, question_id


def _load_json_list(path: Path, role: str) -> list[Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise InputFormatError(f"{role} must be a non-empty JSON list")
    return value


def _load_json_object(path: Path, role: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InputFormatError(f"{role} must be a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise InputFormatError(f"{path}:{line_number} must be an object")
            rows.append(value)
    if not rows:
        raise InputFormatError(f"split file is empty: {path}")
    return rows


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
