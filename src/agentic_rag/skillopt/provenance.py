"""HotpotQA provenance splits for trajectory-representation experiments.

The ordinary benchmark split deliberately contains only the answer required by
the episode evaluator.  Reflection ablations additionally need the original
HotpotQA supporting facts, while still keeping every label outside the target
Agent boundary.  This module enriches an existing deterministic split without
changing its IDs or order and binds the result to both the raw source file and
the exact substrate manifest used by the experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentic_rag.errors import InputFormatError


PROVENANCE_SPLIT_SCHEMA_VERSION = "1.2"
PROVENANCE_SPLIT_PURPOSE = "trajectory_representation_ablation"
HOTPOTQA_PROVENANCE_SHA256 = (
    "3ad9c0bcbf93f41d7004ca6007049c904d2605046b314a2f7ecfb379c64cba6d"
)
_SPLITS = ("train", "validation", "test")


def prepare_hotpotqa_provenance_splits(
    *,
    source_path: str | Path,
    base_split_dir: str | Path,
    substrate_path: str | Path,
    output_dir: str | Path,
    expected_source_sha256: str = HOTPOTQA_PROVENANCE_SHA256,
) -> dict[str, Any]:
    """Add supporting-fact text to a fixed split and write its lineage.

    ``base_split_dir`` remains authoritative for question IDs, order, answers,
    and split membership.  ``source_path`` is used only to verify those fields
    and attach the original HotpotQA supporting facts.  No new sampling occurs.
    """

    source = Path(source_path).resolve()
    base = Path(base_split_dir).resolve()
    substrate = Path(substrate_path).resolve()
    destination = Path(output_dir).resolve()
    _require_sha256(source, expected_source_sha256, "HotpotQA provenance source")

    source_rows = _load_json(source, "HotpotQA provenance source")
    if not isinstance(source_rows, list) or not source_rows:
        raise InputFormatError("HotpotQA provenance source must be a non-empty JSON list")
    source_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(source_rows):
        if not isinstance(raw, Mapping):
            raise InputFormatError(f"HotpotQA source row {index} must be an object")
        question_id = str(raw.get("_id") or raw.get("id") or "").strip()
        if not question_id or question_id in source_by_id:
            raise InputFormatError(
                f"HotpotQA source row {index} has a missing or duplicate ID"
            )
        source_by_id[question_id] = raw

    base_manifest_path = base / "split_manifest.json"
    base_manifest = _load_json(base_manifest_path, "base split manifest")
    if not isinstance(base_manifest, Mapping):
        raise InputFormatError("base split manifest must contain an object")
    dataset = base_manifest.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("subset") != "hotpotqa":
        raise InputFormatError("base split manifest must declare HotpotQA")

    substrate_manifest_path = substrate / "manifest.json"
    substrate_manifest = _load_json(substrate_manifest_path, "substrate manifest")
    if not isinstance(substrate_manifest, Mapping):
        raise InputFormatError("substrate manifest must contain an object")
    if substrate_manifest.get("dataset") != "hotpotqa":
        raise InputFormatError("substrate manifest must declare HotpotQA")

    rendered: dict[str, str] = {}
    split_metadata: dict[str, dict[str, Any]] = {}
    all_ids: list[str] = []
    for split in _SPLITS:
        base_rows = _load_jsonl(base / f"{split}.jsonl")
        enriched: list[dict[str, Any]] = []
        for row in base_rows:
            question_id = str(row.get("id") or "")
            source_row = source_by_id.get(question_id)
            if source_row is None:
                raise InputFormatError(
                    f"base split question {question_id!r} is absent from raw HotpotQA"
                )
            _require_equal(row, source_row, "question", question_id)
            _require_equal(row, source_row, "answer", question_id)
            source_type = source_row.get("type")
            if row.get("question_type") != source_type:
                raise InputFormatError(
                    f"question type mismatch for HotpotQA item {question_id}"
                )
            item = dict(row)
            item["supporting_facts"] = _supporting_fact_records(
                source_row, question_id=question_id
            )
            enriched.append(item)
            all_ids.append(question_id)
        text = "".join(_canonical_json(row) + "\n" for row in enriched)
        rendered[split] = text
        split_metadata[split] = {
            "count": len(enriched),
            "question_type_counts": dict(
                sorted(Counter(str(row["question_type"]) for row in enriched).items())
            ),
            "items": [
                {"id": str(row["id"]), "question_type": str(row["question_type"])}
                for row in enriched
            ],
            "file": {
                "path": f"{split}.jsonl",
                "sha256": _sha256_bytes(text.encode("utf-8")),
                "size_bytes": len(text.encode("utf-8")),
            },
        }
    if len(all_ids) != len(set(all_ids)):
        raise InputFormatError("provenance split IDs must be disjoint")

    raw_source = {
        "path": source.as_posix(),
        "sha256": _sha256_file(source),
        "size_bytes": source.stat().st_size,
        "record_count": len(source_rows),
    }
    substrate_lineage = _substrate_lineage(
        substrate_manifest_path, substrate_manifest
    )
    if not any(
        isinstance(artifact, Mapping)
        and artifact.get("sha256") == raw_source["sha256"]
        for artifact in substrate_lineage["source_artifacts"]
    ):
        raise InputFormatError(
            "substrate source artifacts are not bound to the pinned raw "
            "HotpotQA source"
        )
    raw_type_counts = Counter(str(row.get("type") or "") for row in source_rows)
    manifest: dict[str, Any] = {
        "schema_version": PROVENANCE_SPLIT_SCHEMA_VERSION,
        "purpose": PROVENANCE_SPLIT_PURPOSE,
        "dataset": {
            "subset": "hotpotqa",
            "scope_id": dataset.get("scope_id"),
            "question_count": len(source_rows),
            "question_type_counts": dict(sorted(raw_type_counts.items())),
            "chunk_count": substrate_manifest.get("record_counts", {}).get(
                "chunks"
            ),
            "raw_hotpotqa_source": raw_source,
            "substrate": substrate_lineage,
        },
        "selection": base_manifest.get("selection"),
        "base_split": {
            "path": base.as_posix(),
            "manifest_sha256": _sha256_file(base_manifest_path),
        },
        "splits": split_metadata,
    }
    manifest_text = json.dumps(
        manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    destination.mkdir(parents=True, exist_ok=True)
    for split, text in rendered.items():
        _atomic_write(destination / f"{split}.jsonl", text)
    _atomic_write(destination / "split_manifest.json", manifest_text)
    return manifest


def validate_hotpotqa_provenance_lineage(
    split_dir: str | Path,
    substrate_path: str | Path,
) -> dict[str, Any]:
    """Verify every external and generated artifact used by the ablation."""

    split_root = Path(split_dir).resolve()
    substrate_root = Path(substrate_path).resolve()
    manifest_path = split_root / "split_manifest.json"
    manifest = _load_json(manifest_path, "provenance split manifest")
    if not isinstance(manifest, Mapping):
        raise InputFormatError("provenance split manifest must be an object")
    if manifest.get("schema_version") != PROVENANCE_SPLIT_SCHEMA_VERSION:
        raise InputFormatError(
            "provenance split schema_version must be "
            f"{PROVENANCE_SPLIT_SCHEMA_VERSION!r}"
        )
    if manifest.get("purpose") != PROVENANCE_SPLIT_PURPOSE:
        raise InputFormatError("invalid provenance split purpose")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("subset") != "hotpotqa":
        raise InputFormatError("provenance split must declare HotpotQA")

    raw_source = dataset.get("raw_hotpotqa_source")
    if not isinstance(raw_source, Mapping):
        raise InputFormatError("provenance split has no raw source lineage")
    raw_path = Path(str(raw_source.get("path") or ""))
    if not raw_path.is_file():
        raise InputFormatError(f"raw HotpotQA source is missing: {raw_path}")
    _require_file_record(raw_path, raw_source, "raw HotpotQA source")

    substrate_record = dataset.get("substrate")
    substrate_manifest_path = substrate_root / "manifest.json"
    if not isinstance(substrate_record, Mapping):
        raise InputFormatError("provenance split has no substrate lineage")
    _require_file_record(
        substrate_manifest_path,
        {
            "sha256": substrate_record.get("manifest_sha256"),
            "size_bytes": substrate_record.get("manifest_size_bytes"),
        },
        "substrate manifest",
    )
    current_substrate = _load_json(substrate_manifest_path, "substrate manifest")
    if not isinstance(current_substrate, Mapping):
        raise InputFormatError("substrate manifest must be an object")
    for field in ("dataset", "corpus_id", "schema_version", "source_format", "split"):
        if substrate_record.get(field) != current_substrate.get(field):
            raise InputFormatError(f"substrate lineage field {field} does not match")
    if substrate_record.get("source_artifacts") != current_substrate.get(
        "source_artifacts", []
    ):
        raise InputFormatError("substrate source artifact lineage does not match")
    if not any(
        isinstance(artifact, Mapping)
        and artifact.get("sha256") == raw_source.get("sha256")
        for artifact in current_substrate.get("source_artifacts", [])
    ):
        raise InputFormatError(
            "substrate is not sourced from the pinned raw HotpotQA file"
        )

    split_records = manifest.get("splits")
    if not isinstance(split_records, Mapping) or set(split_records) != set(_SPLITS):
        raise InputFormatError("provenance manifest must contain three fixed splits")
    all_ids: list[str] = []
    split_report: dict[str, Any] = {}
    for split in _SPLITS:
        record = split_records.get(split)
        if not isinstance(record, Mapping):
            raise InputFormatError(f"missing provenance metadata for {split}")
        file_record = record.get("file")
        path = split_root / f"{split}.jsonl"
        if not isinstance(file_record, Mapping):
            raise InputFormatError(f"missing file metadata for {split}")
        if file_record.get("path") != path.name:
            raise InputFormatError(f"invalid file path metadata for {split}")
        _require_file_record(path, file_record, f"{split} split")
        rows = _load_jsonl(path)
        if record.get("count") != len(rows):
            raise InputFormatError(f"{split} split count does not match")
        for row in rows:
            supporting = row.get("supporting_facts")
            if not isinstance(supporting, list) or not supporting:
                raise InputFormatError(
                    f"{split} item {row.get('id')} has no supporting facts"
                )
            all_ids.append(str(row.get("id") or ""))
        split_report[split] = {
            "count": len(rows),
            "sha256": _sha256_file(path),
        }
    if not all(all_ids) or len(all_ids) != len(set(all_ids)):
        raise InputFormatError("provenance split IDs must be non-empty and disjoint")
    return {
        "valid": True,
        "schema_version": PROVENANCE_SPLIT_SCHEMA_VERSION,
        "split_manifest_sha256": _sha256_file(manifest_path),
        "raw_source_sha256": _sha256_file(raw_path),
        "substrate_manifest_sha256": _sha256_file(substrate_manifest_path),
        "total_question_count": len(all_ids),
        "splits": split_report,
    }


def _supporting_fact_records(
    row: Mapping[str, Any], *, question_id: str
) -> list[dict[str, Any]]:
    raw_context = row.get("context")
    raw_supports = row.get("supporting_facts")
    if not isinstance(raw_context, Sequence) or isinstance(raw_context, (str, bytes)):
        raise InputFormatError(f"HotpotQA item {question_id} has invalid context")
    if not isinstance(raw_supports, Sequence) or isinstance(raw_supports, (str, bytes)):
        raise InputFormatError(
            f"HotpotQA item {question_id} has invalid supporting_facts"
        )
    contexts: dict[str, list[str]] = {}
    for entry in raw_context:
        if (
            isinstance(entry, Sequence)
            and not isinstance(entry, (str, bytes))
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], Sequence)
            and not isinstance(entry[1], (str, bytes))
        ):
            contexts.setdefault(str(entry[0]), [str(text) for text in entry[1]])
    records: list[dict[str, Any]] = []
    for support in raw_supports:
        if (
            not isinstance(support, Sequence)
            or isinstance(support, (str, bytes))
            or len(support) != 2
            or not isinstance(support[0], str)
            or not isinstance(support[1], int)
            or isinstance(support[1], bool)
        ):
            raise InputFormatError(
                f"HotpotQA item {question_id} has an invalid supporting fact"
            )
        title = str(support[0])
        sentence_id = int(support[1])
        sentences = contexts.get(title)
        if sentences is None or sentence_id < 0 or sentence_id >= len(sentences):
            raise InputFormatError(
                f"HotpotQA item {question_id} cannot resolve {title}[{sentence_id}]"
            )
        records.append(
            {
                "title": title,
                "sentence_id": sentence_id,
                "text": sentences[sentence_id],
            }
        )
    if not records:
        raise InputFormatError(f"HotpotQA item {question_id} has no supporting facts")
    return records


def _substrate_lineage(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "dataset": manifest.get("dataset"),
        "corpus_id": manifest.get("corpus_id"),
        "schema_version": manifest.get("schema_version"),
        "source_format": manifest.get("source_format"),
        "split": manifest.get("split"),
        "record_counts": manifest.get("record_counts"),
        "source_artifacts": manifest.get("source_artifacts", []),
    }


def _require_equal(
    split_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    field: str,
    question_id: str,
) -> None:
    if split_row.get(field) != source_row.get(field):
        raise InputFormatError(
            f"{field} mismatch between split and HotpotQA source for {question_id}"
        )


def _load_json(path: Path, role: str) -> Any:
    if not path.is_file():
        raise InputFormatError(f"{role} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputFormatError(f"{role} contains invalid JSON: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise InputFormatError(f"split file is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise InputFormatError(f"{path}:{line_number} must be an object")
                rows.append(row)
    except json.JSONDecodeError as exc:
        raise InputFormatError(f"{path} contains invalid JSON: {exc}") from exc
    if not rows:
        raise InputFormatError(f"split file is empty: {path}")
    return rows


def _require_sha256(path: Path, expected: str, role: str) -> None:
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise InputFormatError(f"{role} expected SHA-256 is invalid")
    actual = _sha256_file(path)
    if actual != expected:
        raise InputFormatError(
            f"{role} SHA-256 is {actual}; expected {expected}"
        )


def _require_file_record(
    path: Path, record: Mapping[str, Any], role: str
) -> None:
    if not path.is_file():
        raise InputFormatError(f"{role} is missing: {path}")
    if record.get("sha256") != _sha256_file(path):
        raise InputFormatError(f"{role} SHA-256 does not match lineage")
    if record.get("size_bytes") != path.stat().st_size:
        raise InputFormatError(f"{role} size does not match lineage")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write(path: Path, text: str) -> None:
    payload = text.encode("utf-8")
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"provenance artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "HOTPOTQA_PROVENANCE_SHA256",
    "PROVENANCE_SPLIT_PURPOSE",
    "PROVENANCE_SPLIT_SCHEMA_VERSION",
    "prepare_hotpotqa_provenance_splits",
    "validate_hotpotqa_provenance_lineage",
]
