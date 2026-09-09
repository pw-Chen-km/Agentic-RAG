#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/rag_test/hotpotqa/questions.json"
SKILLOPT = ROOT / "data/skillopt/hotpotqa_smoke"
HELDOUT = ROOT / "data/evaluations/hotpotqa_resample20_seed20260805/questions.json"
OUTPUT = ROOT / "data/evaluations/hotpotqa_sample100_seed42"
SEED = 42
QUOTAS = {"bridge": 81, "comparison": 19}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(question_id: str) -> str:
    return hashlib.sha256(f"{SEED}\0{question_id}".encode()).hexdigest()


def main() -> None:
    rows = json.loads(RAW.read_text(encoding="utf-8"))
    excluded: set[str] = set()
    exclusion_sources: dict[str, list[str]] = {}
    for name in ("train", "validation", "test"):
        path = SKILLOPT / f"{name}.jsonl"
        ids = [str(row["id"]) for row in read_jsonl(path)]
        exclusion_sources[f"skillopt_{name}"] = ids
        excluded.update(ids)
    heldout_ids = [str(row["id"]) for row in json.loads(HELDOUT.read_text(encoding="utf-8"))]
    exclusion_sources["heldout20"] = heldout_ids
    excluded.update(heldout_ids)

    selected: list[dict] = []
    for question_type, quota in QUOTAS.items():
        candidates = [
            row for row in rows
            if str(row["id"]) not in excluded and row["question_type"] == question_type
        ]
        candidates.sort(key=lambda row: rank(str(row["id"])))
        if len(candidates) < quota:
            raise RuntimeError(f"not enough {question_type} candidates")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda row: rank(str(row["id"])))

    records = [
        {
            "answer": row["answer"],
            "id": str(row["id"]),
            "question": row["question"],
            "question_type": row["question_type"],
            "scope_id": "hotpotqa:benchmark_exact:dev",
            "source": "hotpotqa",
        }
        for row in selected
    ]
    if len(records) != 100 or len({row["id"] for row in records}) != 100:
        raise RuntimeError("sample must contain 100 unique IDs")
    if any(row["id"] in excluded for row in records):
        raise RuntimeError("sample overlaps excluded IDs")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT / "questions.json"
    jsonl_path = OUTPUT / "questions.jsonl"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0",
        "seed": SEED,
        "algorithm": "per-type sha256(f'{seed}\\0{id}') rank, then global hash order",
        "quotas": QUOTAS,
        "count": len(records),
        "source": {"path": str(RAW.relative_to(ROOT)), "sha256": digest(RAW)},
        "excluded_unique_count": len(excluded),
        "exclusion_sources": exclusion_sources,
        "selected_ids": [row["id"] for row in records],
        "outputs": {
            "questions.json": {"sha256": digest(json_path)},
            "questions.jsonl": {"sha256": digest(jsonl_path)},
        },
    }
    (OUTPUT / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"count": 100, "quotas": QUOTAS, "excluded": len(excluded), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
