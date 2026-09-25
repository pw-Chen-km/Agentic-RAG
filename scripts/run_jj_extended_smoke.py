"""Run a fixed, stratified Novel/Medical action-selection smoke.

This is an engineering calibration, not a statistical experiment.  It selects
five multi-evidence Complex Reasoning questions and five Contextual Summarize
questions per dataset, then runs every question under all seven conditions.
Gold evidence is used only to define the test-set strata; it is never passed to
the policy.
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
from typing import Any

import yaml


SEED = 20260805
CONDITIONS = ("C0", "C1", "C2", "C3", "C5", "C4", "A1")
DATASETS = ("novel", "medical")
QUESTION_TYPES = ("Complex Reasoning", "Contextual Summarize")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_count(row: dict[str, Any]) -> int:
    value = row.get("evidence")
    if isinstance(value, list):
        return len([item for item in value if str(item).strip()])
    text = str(value or "")
    units = [item.strip() for item in text.split(";") if item.strip()]
    if len(units) > 1:
        return len(units)
    triple = str(row.get("evidence_triple") or row.get("evidence_relations") or "")
    return max(1, len([item for item in triple.split(";") if item.strip()])) if (text or triple) else 0


def stable_key(dataset: str, row: dict[str, Any]) -> str:
    identity = str(row.get("id") or row.get("_id") or row.get("question"))
    return hashlib.sha256(f"{SEED}:{dataset}:{identity}".encode("utf-8")).hexdigest()


def select_type(dataset: str, rows: list[dict[str, Any]], question_type: str, count: int = 5) -> list[tuple[int, dict[str, Any]]]:
    candidates = [
        (index, row) for index, row in enumerate(rows)
        if row.get("question_type") == question_type and evidence_count(row) >= 2
    ]
    if len(candidates) < count:
        raise ValueError(f"{dataset}/{question_type}: only {len(candidates)} multi-evidence candidates")
    candidates.sort(key=lambda item: stable_key(dataset, item[1]))

    # Keep the smoke balanced across evidence lengths instead of selecting only
    # the easiest two-hop or the longest contexts.
    bands = ((2, 3), (4, 5), (6, 10**9))
    chosen: list[tuple[int, dict[str, Any]]] = []
    used_sources: set[str] = set()
    for low, high in bands:
        band = [item for item in candidates if low <= evidence_count(item[1]) <= high]
        for item in band:
            source = str(item[1].get("source") or "")
            if dataset == "novel" and source in used_sources:
                continue
            chosen.append(item)
            used_sources.add(source)
            break
        if len(chosen) >= count:
            break
    for item in candidates:
        if len(chosen) >= count:
            break
        if item in chosen:
            continue
        source = str(item[1].get("source") or "")
        if dataset == "novel" and source in used_sources and len(used_sources) < count:
            continue
        chosen.append(item)
        used_sources.add(source)
    if len(chosen) < count:
        for item in candidates:
            if item not in chosen:
                chosen.append(item)
            if len(chosen) >= count:
                break
    return chosen[:count]


def select_rows(path: Path, dataset: str) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array")
    selected: list[tuple[int, dict[str, Any]]] = []
    report: dict[str, Any] = {"source_question_count": len(rows), "types": {}}
    for question_type in QUESTION_TYPES:
        candidates = [row for row in rows if row.get("question_type") == question_type and evidence_count(row) >= 2]
        picked = select_type(dataset, rows, question_type)
        selected.extend(picked)
        report["types"][question_type] = {
            "candidate_count_multi_evidence": len(candidates),
            "selected_count": len(picked),
            "selected_evidence_counts": [evidence_count(row) for _, row in picked],
        }
    if len({index for index, _ in selected}) != len(selected):
        raise ValueError(f"{dataset}: duplicated selected source row")
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=5)
    parser.add_argument("--require-assessment", action="store_true",
                        help="include the optional evidence assessment in tool arguments")
    args = parser.parse_args()
    if args.per_type != 5:
        raise ValueError("this smoke is intentionally fixed at five questions per type")
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
    config["agent"]["require_evidence_assessment"] = bool(args.require_assessment)
    config["policy"] = {
        "provider": "ollama", "model": model, "host": host, "temperature": 0,
        "think": False, "num_ctx": 32768, "max_output_tokens": 2048,
        "timeout_seconds": 600, "max_retries": 2,
    }
    config_path = output / "target_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest: dict[str, Any] = {
        "purpose": "expanded natural action-selection smoke; 5 multi-evidence reasoning + 5 contextual summarization per dataset",
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "model": model, "model_digest": model_by_name[model].get("digest"),
        "embedding_model": embedding, "embedding_digest": model_by_name[embedding].get("digest"),
        "conditions": list(CONDITIONS), "expected_episodes": len(DATASETS) * 10 * len(CONDITIONS),
        "question_selection": {"per_dataset": 10, "per_type": 5, "types": list(QUESTION_TYPES), "min_evidence_units": 2},
        "require_evidence_assessment": bool(args.require_assessment),
        "datasets": {}, "skill_sha256": digest(repo / "skills/interface_study.md"),
        "runner_sha256": digest(repo / "scripts/run_interface_study.py"),
    }
    status: dict[str, Any] = {"status": "running", "datasets": {}}
    save(output / "status.json", status)
    try:
        for dataset in DATASETS:
            substrate = root / "substrates" / dataset
            source = root / "sources/graphrag_benchmark" / dataset / "questions.json"
            source_manifest = root / "sources/graphrag_benchmark" / dataset / "source_manifest.json"
            selected, selection_report = select_rows(source, dataset)
            selected_indices = [index for index, _ in selected]
            selected_path = output / "selected_questions" / f"{dataset}.json"
            save(selected_path, [row for _, row in selected])
            manifest["datasets"][dataset] = {
                "source_sha256": digest(source), "substrate_manifest_sha256": digest(substrate / "manifest.json"),
                "selection_report": selection_report,
                "selected": [{"source_row_index": index, "id": row.get("_id") or row.get("id"),
                              "type": row.get("question_type"), "evidence_units": evidence_count(row),
                              "source": row.get("source"), "question": row["question"]} for index, row in selected],
            }
            save(output / "question_selection_manifest.json", manifest)
            status["active_dataset"] = dataset
            save(output / "status.json", status)
            log_path = output / f"{dataset}.log"
            with log_path.open("w", encoding="utf-8") as log:
                commands = (
                    ("validate_v2_substrate.py", ["--substrate", str(substrate), "--output", str(output / f"{dataset}_substrate_validation.json")]),
                    ("calibrate_interface.py", ["--substrate", str(substrate), "--output", str(output / f"{dataset}_static_calibration.json")]),
                    ("run_interface_study.py", ["--dataset", dataset, "--substrate", str(substrate), "--questions", str(source),
                        "--source-manifest", str(source_manifest), "--config", str(config_path),
                        "--skill", str(repo / "skills/interface_study.md"), "--output", str(output / dataset),
                        "--seed", str(SEED), "--question-indices", *(str(index) for index in selected_indices),
                        "--conditions", *CONDITIONS]),
                )
                for script, arguments in commands:
                    command = [sys.executable, "-X", "faulthandler", "-u", str(repo / "scripts" / script), *arguments]
                    save(output / f"{dataset}_{script}_command.json", command)
                    subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
            results = json.loads((output / dataset / "summary.json").read_text(encoding="utf-8"))["results"]
            status["datasets"][dataset] = {"status": "completed", "episodes": len(results),
                                             "terminal_reasons": [row.get("termination_reason") for row in results]}
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
