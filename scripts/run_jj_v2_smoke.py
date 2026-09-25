"""Run the existing v2 runner on one question per dataset, in a fresh directory.

This is an engineering smoke test, not a statistical calibration certificate.
Uses the installed JJ Ollama model; never starts a full experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assessment", choices=("on", "off"), default="on")
    a = p.parse_args()
    root = a.data_root.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo / "configs/interface_study_v62_qwen27b_ollama.yaml").read_text())
    config["agent"]["require_evidence_assessment"] = a.assessment == "on"
    config["policy"] = dict(provider="ollama", model="qwen3.8:27b-q4_K_M",
        host="http://127.0.0.1:11440", temperature=0, think=False,
        num_ctx=32768, max_output_tokens=2048, timeout_seconds=600, max_retries=2)
    config_path = out / "target_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with urllib.request.urlopen("http://127.0.0.1:11440/api/tags", timeout=30) as resp:
        tags = json.load(resp)
    files = [*sorted((repo / "src").rglob("*.py")), *sorted((repo / "scripts").glob("*.py")), repo / "skills/interface_study.md"]
    save(out / "smoke_manifest.json", dict(
        created_at=datetime.now(timezone.utc).isoformat(), expected_episodes=21,
        seed=20260805, model_tags=tags, code_files={str(f.relative_to(repo)):digest(f) for f in files},
        scope="one first source question per dataset, seven conditions; engineering test only",
        semantic_judge="not run by this target smoke; separate evaluator checks required"))
    status = {"status":"running", "datasets":{}}
    save(out / "status.json", status)
    for ds in ("hotpotqa", "novel", "medical"):
        status["active_dataset"] = ds
        save(out / "status.json", status)
        try:
            substrate = root / "substrates" / ds
            if ds == "hotpotqa":
                questions = root / "sources/hotpotqa/hotpot_dev_distractor_v1.selected.json"
                source_manifest = out / "hotpotqa_source_inventory.json"
                save(source_manifest, {"path":str(questions), "sha256":digest(questions),
                    "lineage_status":"existing JJ source snapshot; upstream equality not reverified by smoke",
                    "substrate_manifest_sha256":digest(substrate / "manifest.json")})
            else:
                questions = root / "sources/graphrag_benchmark" / ds / "questions.json"
                source_manifest = root / "sources/graphrag_benchmark" / ds / "source_manifest.json"
            with (out / f"{ds}.log").open("w", encoding="utf-8") as log:
                # Static registry validation is explicitly separate from live outcomes.
                subprocess.run([sys.executable, str(repo / "scripts/calibrate_interface.py"),
                    "--substrate", str(substrate), "--output", str(out / f"{ds}_static_calibration.json")],
                    cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
                cmd = [sys.executable, "-u", str(repo / "scripts/run_interface_study.py"),
                    "--dataset", ds, "--substrate", str(substrate), "--questions", str(questions),
                    "--source-manifest", str(source_manifest), "--config", str(config_path),
                    "--skill", str(repo / "skills/interface_study.md"), "--output", str(out / ds),
                    "--seed", "20260805", "--limit", "1", "--conditions", "C0", "C1", "C2", "C3", "C5", "C4", "A1"]
                save(out / f"{ds}_command.json", cmd)
                subprocess.run(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((out / ds / "summary.json").read_text())
            status["datasets"][ds] = {"status":"completed", "episodes":len(result["results"]),
                "terminal_reasons":[r.get("termination_reason") for r in result["results"]]}
        except Exception:
            status["datasets"][ds] = {"status":"error", "traceback":traceback.format_exc()}
        save(out / "status.json", status)
    status["status"] = "finished_requires_artifact_audit"
    status.pop("active_dataset", None)
    save(out / "status.json", status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
