"""Shared immutable-file and substrate-lineage helpers for SkillOpt data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentic_rag.errors import InputFormatError


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise InputFormatError(f"Invalid JSON in {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def source_file_manifest(path: Path, dataset_dir: Path) -> dict[str, Any]:
    try:
        relative_path = path.relative_to(dataset_dir)
    except ValueError:
        relative_path = Path(path.name)
    return {
        "path": relative_path.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def require_mapping(
    parent: Mapping[str, Any], key: str, location: str
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise InputFormatError(f"{location}.{key} must be an object")
    return value


def object_field(value: object, field_name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(field_name)
    return getattr(value, field_name, None)


def validate_source_lineage(
    dataset: Mapping[str, Any], substrate_manifest: object
) -> dict[str, dict[str, Any]]:
    declared_sources = require_mapping(
        dataset,
        "source_files",
        "split manifest dataset",
    )
    raw_artifacts = object_field(substrate_manifest, "source_artifacts")
    if (
        not isinstance(raw_artifacts, Sequence)
        or isinstance(raw_artifacts, (str, bytes))
    ):
        raise InputFormatError(
            "substrate manifest has no source_artifacts sequence"
        )
    artifacts_by_role: dict[str, object] = {}
    for artifact in raw_artifacts:
        role = object_field(artifact, "role")
        if not isinstance(role, str) or not role:
            raise InputFormatError(
                "substrate manifest contains an invalid source artifact role"
            )
        if role in artifacts_by_role:
            raise InputFormatError(
                f"substrate manifest has duplicate source artifact role {role!r}"
            )
        artifacts_by_role[role] = artifact

    report: dict[str, dict[str, Any]] = {}
    for role in ("chunks", "questions"):
        declared = require_mapping(
            declared_sources,
            role,
            "split manifest dataset.source_files",
        )
        artifact = artifacts_by_role.get(role)
        if artifact is None:
            raise InputFormatError(
                f"substrate manifest is missing {role!r} source artifact"
            )
        declared_sha = declared.get("sha256")
        artifact_sha = object_field(artifact, "sha256")
        if not is_sha256(declared_sha):
            raise InputFormatError(
                f"split manifest {role} source SHA-256 is invalid"
            )
        if artifact_sha != declared_sha:
            raise InputFormatError(
                f"split manifest {role} source SHA-256 does not match "
                "the substrate source artifact"
            )
        declared_size = declared.get("size_bytes")
        artifact_size = object_field(artifact, "size_bytes")
        if (
            not isinstance(declared_size, int)
            or isinstance(declared_size, bool)
            or declared_size < 1
        ):
            raise InputFormatError(
                f"split manifest {role} source size is invalid"
            )
        if artifact_size != declared_size:
            raise InputFormatError(
                f"split manifest {role} source size does not match "
                "the substrate source artifact"
            )
        report[role] = {
            "sha256": declared_sha,
            "size_bytes": declared_size,
        }
    return report


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "atomic_write_text",
    "load_json",
    "object_field",
    "require_mapping",
    "sha256_file",
    "source_file_manifest",
    "validate_source_lineage",
]
