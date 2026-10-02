"""Audit an existing multi-dataset run without changing Policy inputs or artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from agentic_rag.evaluation.retrieval_coverage import CoverageAuditor, summarize
from agentic_rag.substrate.storage import Substrate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_manifest(dataset: str, run_manifest_path: Path, substrate_manifest_path: Path) -> tuple[dict, str]:
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest_hash = digest(substrate_manifest_path)
    if run_manifest.get("dataset") != dataset:
        raise ValueError(f"run dataset mismatch: {dataset}")
    if run_manifest.get("substrate_manifest_sha256") != manifest_hash:
        raise ValueError(f"substrate manifest mismatch: {dataset}")
    return run_manifest, manifest_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--substrate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    rows, lineage = [], {}
    for dataset in ("hotpotqa", "novel", "medical"):
        run = args.run_root / dataset
        substrate_path = args.substrate_root / dataset
        manifest_path = substrate_path / "manifest.json"
        run_manifest_path = run / "run_manifest.json"
        _, manifest_hash = validate_manifest(dataset, run_manifest_path, manifest_path)
        substrate = Substrate.open(substrate_path)
        auditor = CoverageAuditor(substrate)
        episode_paths = sorted((run / "episodes").glob("*/episode.json"))
        if not episode_paths:
            raise ValueError(f"no episode artifacts: {dataset}")
        for path in episode_paths:
            episode = json.loads(path.read_text(encoding="utf-8"))
            rows.append(auditor.audit_episode(episode, dataset))
        lineage[dataset] = {"run_manifest_sha256": digest(run_manifest_path),
                            "substrate_manifest_sha256": manifest_hash,
                            "ner_model": auditor.ner_model,
                            "query_ner_available": auditor._nlp is not None,
                            "episodes": len(episode_paths)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8")
    write_json(args.output / "summary.json", summarize(rows))
    write_json(args.output / "audit_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_run_root": str(args.run_root.resolve()),
        "dataset_lineage": lineage,
        "audit_code_sha256": digest(Path(__file__)),
        "policy_input_changed": False,
        "gold_used": False,
    })


if __name__ == "__main__":
    main()
