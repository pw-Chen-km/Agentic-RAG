"""Pin HotpotQA provenance and create the deterministic interface-study split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


CANONICAL_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
MIRROR_URL = "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json?download=true"
LICENSE = "CC BY-SA 4.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON array of objects: {path}")
    return value


def prepare(
    official_path: Path,
    local_questions_path: Path,
    output_dir: Path,
    *,
    seed: int = 20260805,
    bridge_count: int = 24,
    comparison_count: int = 6,
) -> dict[str, Any]:
    official = load_rows(official_path)
    local = load_rows(local_questions_path)
    official_by_id = {str(row.get("_id")): row for row in official}
    if len(official_by_id) != len(official):
        raise ValueError("official HotpotQA source contains duplicate _id values")
    selected: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for index, row in enumerate(local):
        question_id = str(row.get("id") or row.get("_id") or "")
        source = official_by_id.get(question_id)
        if source is None:
            mismatches.append(f"missing:{question_id}")
            continue
        if row.get("question") != source.get("question"):
            mismatches.append(f"question:{question_id}")
        if row.get("answer") != source.get("answer"):
            mismatches.append(f"answer:{question_id}")
        selected.append(source)
    if mismatches or len(selected) != 1000:
        raise ValueError(
            "local HotpotQA reference does not exactly match the official source: "
            + ", ".join(mismatches[:20])
        )

    by_type: dict[str, list[dict[str, Any]]] = {"bridge": [], "comparison": []}
    for row in selected:
        question_type = str(row.get("type") or "").casefold()
        if question_type not in by_type:
            raise ValueError(f"unsupported HotpotQA type: {question_type!r}")
        by_type[question_type].append(row)
    rng = random.Random(seed)
    pilot = [
        *rng.sample(sorted(by_type["bridge"], key=lambda row: row["_id"]), bridge_count),
        *rng.sample(sorted(by_type["comparison"], key=lambda row: row["_id"]), comparison_count),
    ]
    pilot.sort(key=lambda row: (str(row.get("type")), str(row.get("_id"))))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hotpot_dev_distractor_v1.selected.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pilot_questions.json").write_text(
        json.dumps(pilot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pilot_ids = [str(row["_id"]) for row in pilot]
    manifest = {
        "manifest_version": "interface-study-source-v1",
        "dataset": "hotpotqa",
        "split": "dev_distractor",
        "dataset_version": "hotpot_dev_distractor_v1",
        "canonical_url": CANONICAL_URL,
        "retrieved_url": MIRROR_URL,
        "sha256": sha256(official_path),
        "size_bytes": official_path.stat().st_size,
        "license": LICENSE,
        "official_record_count": len(official),
        "reference_record_count": len(selected),
        "reference_question_ids_sha256": hashlib.sha256(
            "\n".join(str(row["_id"]) for row in selected).encode()
        ).hexdigest(),
        "local_reference_path": local_questions_path.resolve().as_posix(),
        "local_reference_sha256": sha256(local_questions_path),
        "local_reference_match": True,
        "sampling": {
            "seed": seed,
            "strata": {"bridge": bridge_count, "comparison": comparison_count},
            "pilot_count": len(pilot),
            "pilot_ids": pilot_ids,
            "pilot_type_counts": dict(Counter(str(row.get("type")) for row in pilot)),
        },
        "source_file": official_path.resolve().as_posix(),
        "selected_file": (output_dir / "hotpot_dev_distractor_v1.selected.json").resolve().as_posix(),
        "pilot_file": (output_dir / "pilot_questions.json").resolve().as_posix(),
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--local-questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    manifest = prepare(
        args.official,
        args.local_questions,
        args.output,
        seed=args.seed,
    )
    print(json.dumps({"sha256": manifest["sha256"], "pilot_count": manifest["sampling"]["pilot_count"]}))


if __name__ == "__main__":
    main()
