"""Validation helpers for evaluation-only gold evidence sidecars."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agentic_rag.substrate.storage import EvaluationSidecars


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_gold_sidecars(substrate: str | Path, dataset: str) -> dict[str, Any]:
    """Validate the versioned gold evidence sidecar before a formal run.

    The sidecar is deliberately separate from runtime substrate tables.  This
    check only confirms lineage and presence; unresolved evidence is retained
    with an explicit ``not_evaluable`` status rather than converted to zero.
    """

    root = Path(substrate)
    evaluation = root / "evaluation"
    payload_path = evaluation / "gold_evidence.json"
    manifest_path = evaluation / "gold_evidence_manifest.json"
    if not payload_path.exists() or not manifest_path.exists():
        raise ValueError(
            f"{dataset}: gold evidence sidecar is missing; run repair_gold_sidecars.py"
        )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset or manifest.get("dataset") != dataset:
        raise ValueError(f"{dataset}: gold evidence sidecar dataset mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{dataset}: gold evidence sidecar contains no records")
    question_ids = [str(item.get("question_id") or "") for item in records]
    if any(not item for item in question_ids) or len(question_ids) != len(set(question_ids)):
        raise ValueError(f"{dataset}: gold evidence question IDs are missing or duplicated")
    expected_hash = manifest.get("gold_evidence_sha256")
    actual_hash = sha256(payload_path)
    if expected_hash != actual_hash:
        raise ValueError(f"{dataset}: gold evidence hash mismatch")
    expected_substrate = manifest.get("substrate_manifest_sha256")
    actual_substrate = sha256(root / "manifest.json")
    if expected_substrate != actual_substrate:
        raise ValueError(f"{dataset}: gold evidence was built for another substrate")

    sidecars = EvaluationSidecars.open(root)
    benchmark_ids = {item.question_id for item in sidecars.benchmark_questions}
    if set(question_ids) != benchmark_ids:
        raise ValueError(
            f"{dataset}: gold evidence IDs do not match benchmark question IDs"
        )
    if dataset == "hotpotqa" and not sidecars.gold_support:
        raise ValueError("hotpotqa: gold_support.parquet is empty")

    raw_evidence = sum(
        len(item.get("evidence") or [])
        + (1 if item.get("evidence_triple") else 0)
        + len(item.get("evidence_relations") or [])
        for item in records
    )
    if dataset in {"novel", "medical"} and raw_evidence == 0:
        raise ValueError(f"{dataset}: official evidence is absent from the sidecar")
    return {
        "dataset": dataset,
        "question_count": len(question_ids),
        "gold_support_count": len(sidecars.gold_support),
        "raw_evidence_count": raw_evidence,
        "gold_evidence_sha256": actual_hash,
        "substrate_manifest_sha256": actual_substrate,
        "status": "ok",
    }
