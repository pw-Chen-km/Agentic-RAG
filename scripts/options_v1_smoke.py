#!/usr/bin/env python3
"""Prepare and validate the fixed 20-per-dataset Options v1 smoke sample.

This is an offline data-preparation utility. It only reads existing substrate
evaluation sidecars and manifests; it does not rebuild indexes or call a model.
The selected question JSONL keeps answers for evaluation, while the generated
runtime contract explicitly requires that only ``question`` is sent to Agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/options_v1_smoke.json"
DATASETS = ("hotpotqa", "novel", "medical")


class PreparationError(ValueError):
    """The sample cannot be prepared without risking a bad or overwritten run."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _write_immutable(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise PreparationError(f"Refusing to overwrite a different existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _rank(seed: int, question_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{question_id}".encode("utf-8")).hexdigest()


def _read_source(dataset: str, substrate_path: Path, expected_scope: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not substrate_path.is_dir():
        raise PreparationError(f"{dataset}: substrate directory does not exist: {substrate_path}")
    manifest_path = substrate_path / "manifest.json"
    questions_path = substrate_path / "evaluation/benchmark_questions.parquet"
    if not manifest_path.is_file() or not questions_path.is_file():
        raise PreparationError(
            f"{dataset}: expected manifest.json and evaluation/benchmark_questions.parquet under {substrate_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != dataset:
        raise PreparationError(f"{dataset}: substrate manifest names dataset {manifest.get('dataset')!r}")
    if manifest.get("split") != "dev":
        raise PreparationError(f"{dataset}: expected dev substrate; got split={manifest.get('split')!r}")
    actual_scope = manifest.get("benchmark_scope_id") or expected_scope
    if actual_scope != expected_scope:
        raise PreparationError(f"{dataset}: expected scope {expected_scope!r}; got {actual_scope!r}")
    rows = pq.read_table(questions_path).to_pylist()
    if not rows:
        raise PreparationError(f"{dataset}: benchmark question table is empty")
    seen: set[str] = set()
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id or question_id in seen:
            raise PreparationError(f"{dataset}: missing or duplicate canonical question_id {question_id!r}")
        seen.add(question_id)
        if row.get("scope_id") != expected_scope:
            raise PreparationError(f"{dataset}: question {question_id} has wrong scope {row.get('scope_id')!r}")
        if row.get("source") != dataset:
            raise PreparationError(f"{dataset}: question {question_id} has wrong source {row.get('source')!r}")
        if not str(row.get("question") or "").strip() or not str(row.get("answer") or "").strip():
            raise PreparationError(f"{dataset}: question {question_id} lacks question or gold answer")
    return rows, {
        "dataset": dataset,
        "substrate": substrate_path.resolve().as_posix(),
        "substrate_manifest_sha256": sha256_file(manifest_path),
        "benchmark_questions_sha256": sha256_file(questions_path),
        "question_count": len(rows),
        "scope_id": expected_scope,
    }


def prepare(config_path: Path, *, output_override: Path | None = None,
            substrate_overrides: dict[str, Path] | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "agentic-rag-options-v1-smoke-1":
        raise PreparationError("Unsupported Options v1 smoke config schema_version")
    seed = int(config["seed"])
    quota = int(config["questions_per_dataset"])
    if quota != 20:
        raise PreparationError("This first smoke protocol is fixed at exactly 20 questions per dataset")
    if set(config.get("datasets", {})) != set(DATASETS):
        raise PreparationError(f"Config must name exactly these datasets: {', '.join(DATASETS)}")

    output_root = output_override or Path(config["paths"]["output"])
    question_dir = output_root / config["paths"].get("questions_dir", "questions")
    samples: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        entry = config["datasets"][dataset]
        substrate_path = (substrate_overrides or {}).get(dataset, Path(entry["substrate"]))
        rows, source_record = _read_source(dataset, substrate_path, entry["scope_id"])
        if len(rows) < quota:
            raise PreparationError(f"{dataset}: only {len(rows)} questions; need {quota}")
        rows.sort(key=lambda row: (_rank(seed, str(row["question_id"])), str(row["question_id"])))
        selected = rows[:quota]
        samples[dataset] = [
            {
                "question_id": str(row["question_id"]),
                "scope_id": str(row["scope_id"]),
                "source": dataset,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "question_type": row.get("question_type"),
            }
            for row in selected
        ]
        sources[dataset] = source_record

    all_ids = [row["question_id"] for rows in samples.values() for row in rows]
    if len(all_ids) != 60 or len(set(all_ids)) != 60:
        raise PreparationError("Expected 60 unique canonical IDs across the three selected samples")

    output_root.mkdir(parents=True, exist_ok=True)
    sample_records: dict[str, Any] = {}
    for dataset in DATASETS:
        path = question_dir / f"{dataset}.jsonl"
        payload = "".join(_canonical_json(row) + "\n" for row in samples[dataset])
        _write_immutable(path, payload)
        sample_records[dataset] = {
            "path": path.resolve().as_posix(),
            "sha256": sha256_file(path),
            "count": len(samples[dataset]),
            "question_ids": [row["question_id"] for row in samples[dataset]],
            "question_type_counts": _counts(samples[dataset], "question_type"),
        }

    contract = {
        "schema_version": "agentic-rag-options-v1-smoke-run-1",
        "seed": seed,
        "options_v1": config["options_v1"],
        "model": config["model"],
        "execution": config["execution"],
        "datasets": {
            dataset: {
                "substrate": sources[dataset]["substrate"],
                "scope_id": sources[dataset]["scope_id"],
                "questions": sample_records[dataset]["path"],
                "expected_count": quota,
                "output": (output_root / config["paths"].get("results_dir", "results") / dataset).as_posix(),
            }
            for dataset in DATASETS
        },
        "calls_model": False,
    }
    contract_path = output_root / "runtime_config.json"
    _write_immutable(contract_path, json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n")

    manifest = {
        "schema_version": "agentic-rag-options-v1-smoke-sample-1",
        "seed": seed,
        "sampling": "sort each dataset by sha256(f'{seed}\\0{canonical_question_id}'), take first 20",
        "questions_per_dataset": quota,
        "total_questions": len(all_ids),
        "source_substrates": sources,
        "samples": sample_records,
        "runtime_config": {
            "path": contract_path.resolve().as_posix(),
            "sha256": sha256_file(contract_path),
        },
        "model_calls": 0,
        "gold_usage": "answers are stored only for evaluation; runtime must send question text alone to the target agent",
    }
    manifest_path = output_root / "sample_manifest.json"
    _write_immutable(manifest_path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    validate(output_root)
    return {
        "output": output_root.resolve().as_posix(),
        "datasets": {dataset: {"count": sample_records[dataset]["count"],
                               "question_type_counts": sample_records[dataset]["question_type_counts"]}
                     for dataset in DATASETS},
        "manifest_sha256": sha256_file(manifest_path),
        "model_calls": 0,
    }


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "unspecified")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def validate(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "sample_manifest.json"
    contract_path = output_root / "runtime_config.json"
    if not manifest_path.is_file() or not contract_path.is_file():
        raise PreparationError("sample_manifest.json and runtime_config.json must exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    ids: list[str] = []
    for dataset in DATASETS:
        source = manifest["source_substrates"][dataset]
        substrate = Path(source["substrate"])
        source_manifest = substrate / "manifest.json"
        source_questions = substrate / "evaluation/benchmark_questions.parquet"
        if (not source_manifest.is_file() or not source_questions.is_file()
                or sha256_file(source_manifest) != source["substrate_manifest_sha256"]
                or sha256_file(source_questions) != source["benchmark_questions_sha256"]):
            raise PreparationError(f"{dataset}: source substrate missing or changed since sampling")
        record = manifest["samples"][dataset]
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise PreparationError(f"{dataset}: sample file missing or hash mismatch")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 20 or len({row.get("question_id") for row in rows}) != 20:
            raise PreparationError(f"{dataset}: sample must have exactly 20 unique question IDs")
        if any(row.get("source") != dataset or row.get("scope_id") != manifest["source_substrates"][dataset]["scope_id"]
               for row in rows):
            raise PreparationError(f"{dataset}: source/scope mismatch in sample")
        if len(rows) != record["count"] or [row["question_id"] for row in rows] != record["question_ids"]:
            raise PreparationError(f"{dataset}: manifest count/order does not match the sample")
        ids.extend(row["question_id"] for row in rows)
        entry = contract["datasets"][dataset]
        if Path(entry["questions"]).resolve() != path.resolve() or entry["expected_count"] != 20:
            raise PreparationError(f"{dataset}: runtime contract does not point at fixed sample")
    if len(ids) != 60 or len(set(ids)) != 60:
        raise PreparationError("Expected 60 unique question IDs total")
    if manifest["runtime_config"]["sha256"] != sha256_file(contract_path):
        raise PreparationError("runtime_config.json hash differs from sample manifest")
    return {"valid": True, "dataset_counts": {dataset: 20 for dataset in DATASETS}, "total": len(ids)}


def dry_run(config_path: Path, *, substrate_overrides: dict[str, Path] | None = None) -> dict[str, Any]:
    """Check every source and report deterministic sample sizes without writes."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "agentic-rag-options-v1-smoke-1":
        raise PreparationError("Unsupported Options v1 smoke config schema_version")
    if int(config.get("questions_per_dataset", 0)) != 20:
        raise PreparationError("This first smoke protocol is fixed at exactly 20 questions per dataset")
    reports: dict[str, Any] = {}
    for dataset in DATASETS:
        entry = config["datasets"][dataset]
        source_path = (substrate_overrides or {}).get(dataset, Path(entry["substrate"]))
        rows, source_record = _read_source(dataset, source_path, entry["scope_id"])
        if len(rows) < 20:
            raise PreparationError(f"{dataset}: only {len(rows)} questions; need 20")
        selected = sorted(rows, key=lambda row: (_rank(int(config["seed"]), str(row["question_id"])),
                                                  str(row["question_id"])))[:20]
        reports[dataset] = {
            "available_questions": len(rows),
            "selected_questions": 20,
            "sample_question_ids": [str(row["question_id"]) for row in selected],
            "question_type_counts": _counts(selected, "question_type"),
            "source": source_record,
        }
    return {"valid": True, "writes": 0, "model_calls": 0, "datasets": reports,
            "planned_output": config["paths"]["output"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="Override config output path")
    parser.add_argument("--substrate", action="append", default=[], metavar="DATASET=PATH",
                        help="Override one substrate path; repeat for hotpotqa, novel, and medical")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="Validate an already-prepared output")
    mode.add_argument("--dry-run", action="store_true", help="Read and validate sources; no files or model calls")
    args = parser.parse_args(argv)
    overrides: dict[str, Path] = {}
    for item in args.substrate:
        name, sep, value = item.partition("=")
        if not sep or name not in DATASETS or name in overrides:
            parser.error("--substrate must be unique DATASET=PATH for hotpotqa, novel, or medical")
        overrides[name] = Path(value)
    output = args.output or Path(json.loads(args.config.read_text(encoding="utf-8"))["paths"]["output"])
    if args.validate_only:
        result = validate(output)
    elif args.dry_run:
        result = dry_run(args.config, substrate_overrides=overrides)
    else:
        result = prepare(args.config, output_override=args.output, substrate_overrides=overrides)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
