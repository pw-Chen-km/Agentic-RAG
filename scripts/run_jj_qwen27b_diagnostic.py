"""Small, stratified JJ diagnostic for the current interface and Qwen 27B.

One question is selected per official question type, then run under all seven
conditions. This checks workflow and natural tool uptake, not research effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml


SEED = 20260805
CONDITIONS = ("C0", "C1", "C2", "C3", "C5", "C4", "A1")
TYPES = {
    "hotpotqa": ("bridge", "comparison"),
    "novel": ("Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"),
    "medical": ("Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"),
}


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rows(path: Path, dataset: str) -> list[tuple[int, dict]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    field = "type" if dataset == "hotpotqa" else "question_type"
    rng = random.Random(SEED)
    selected = []
    for kind in TYPES[dataset]:
        candidates = [(index, row) for index, row in enumerate(rows) if row.get(field) == kind]
        if not candidates:
            raise ValueError(f"{dataset}: no questions of type {kind!r}")
        selected.append(rng.choice(candidates))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    host = "http://127.0.0.1:11440"
    with urllib.request.urlopen(host + "/api/tags", timeout=30) as response:
        tags = json.load(response)
    model = "qwen3.8:27b-q4_K_M"
    embedding = "qwen3-embedding:4b"
    model_by_name = {entry["name"]: entry for entry in tags.get("models", [])}
    for name in (model, embedding):
        if name not in model_by_name:
            raise ValueError(f"required Ollama model not installed: {name}")
    config = yaml.safe_load((repo / "configs/interface_study_v2_qwen38_vllm.yaml").read_text(encoding="utf-8"))
    config["agent"]["require_evidence_assessment"] = True
    config["policy"] = {
        "provider": "ollama", "model": model, "host": host,
        "temperature": 0, "think": False, "num_ctx": 32768,
        "max_output_tokens": 2048, "timeout_seconds": 600, "max_retries": 2,
    }
    config_path = output / "target_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "purpose": "stratified engineering diagnostic; one fixed-seed question per type, seven conditions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "model": model, "model_digest": model_by_name[model].get("digest"),
        "embedding_model": embedding, "embedding_digest": model_by_name[embedding].get("digest"),
        "conditions": CONDITIONS, "expected_episodes": 70, "datasets": {},
        "skill_sha256": digest(repo / "skills/interface_study.md"),
        "runner_sha256": digest(repo / "scripts/run_interface_study.py"),
    }
    status = {"status": "running", "datasets": {}}
    save(output / "status.json", status)
    try:
        for dataset in TYPES:
            substrate = root / "substrates" / dataset
            if dataset == "hotpotqa":
                source = root / "sources/hotpotqa/hotpot_dev_distractor_v1.selected.json"
                source_manifest = output / "hotpotqa_source_inventory.json"
                save(source_manifest, {"path": str(source), "sha256": digest(source),
                                       "substrate_manifest_sha256": digest(substrate / "manifest.json")})
            else:
                source = root / "sources/graphrag_benchmark" / dataset / "questions.json"
                source_manifest = root / "sources/graphrag_benchmark" / dataset / "source_manifest.json"
            selected = select_rows(source, dataset)
            selected_indices = [index for index, _ in selected]
            selected_path = output / "selected_questions" / f"{dataset}.json"
            save(selected_path, [row for _, row in selected])
            manifest["datasets"][dataset] = {
                "source_sha256": digest(source),
                "substrate_manifest_sha256": digest(substrate / "manifest.json"),
                "selected": [{"source_row_index": index,
                              "id": row.get("_id") or row.get("id"),
                              "type": row.get("type") or row.get("question_type"),
                              "question": row["question"]} for index, row in selected],
            }
            save(output / "diagnostic_manifest.json", manifest)
            status["active_dataset"] = dataset
            save(output / "status.json", status)
            with (output / f"{dataset}.log").open("w", encoding="utf-8") as log:
                for script, arguments in (
                    ("validate_v2_substrate.py", ["--substrate", str(substrate),
                                                   "--output", str(output / f"{dataset}_substrate_validation.json")]),
                    ("calibrate_interface.py", ["--substrate", str(substrate),
                                                "--output", str(output / f"{dataset}_static_calibration.json")]),
                    ("run_interface_study.py", ["--dataset", dataset, "--substrate", str(substrate),
                                                "--questions", str(source),
                                                "--source-manifest", str(source_manifest),
                                                "--config", str(config_path),
                                                "--skill", str(repo / "skills/interface_study.md"),
                                                "--output", str(output / dataset),
                                                "--seed", str(SEED),
                                                "--question-indices", *(str(index) for index in selected_indices),
                                                "--conditions", *CONDITIONS]),
                ):
                    command = [sys.executable, "-X", "faulthandler", "-u",
                               str(repo / "scripts" / script), *arguments]
                    save(output / f"{dataset}_{script}_command.json", command)
                    subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
            results = json.loads((output / dataset / "summary.json").read_text(encoding="utf-8"))["results"]
            status["datasets"][dataset] = {
                "status": "completed", "episodes": len(results),
                "terminal_reasons": [row.get("termination_reason") for row in results],
            }
            save(output / "status.json", status)
    except Exception:
        status["status"] = "error"
        status["error"] = traceback.format_exc()
        save(output / "status.json", status)
        raise
    status.pop("active_dataset", None)
    status["status"] = "finished_requires_artifact_audit"
    save(output / "status.json", status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
