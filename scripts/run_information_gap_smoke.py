"""Gate the information-gap contract, then run the paired smoke workflow.

The first phase deliberately stays small and is run for all three datasets and
all seven conditions.  It combines provider-format probes, deterministic
backend branch coverage, and one natural episode per condition.  The extended
140-episode smoke is started only when the first phase meets the protocol and
execution gates.

This script never starts Ollama or a remote service.  It invokes the existing
calibration, branch-coverage, and study runners against the configured host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONDITIONS = ("C0", "A0", "C2", "C3", "C1", "A1", "C5", "C4")
DATASETS = ("hotpotqa", "novel", "medical")
MODEL = "qwen3.8:27b-q4_K_M"
HOST = "http://127.0.0.1:11440"
SEED = 20260805


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _question_paths(root: Path, dataset: str, output: Path) -> tuple[Path, Path]:
    if dataset == "hotpotqa":
        questions = root / "sources/hotpotqa/hotpot_dev_distractor_v1.selected.json"
        inventory = output / "hotpotqa_source_inventory.json"
        save(
            inventory,
            {
                "path": questions.resolve().as_posix(),
                "sha256": digest(questions),
                "substrate_manifest_sha256": digest(root / "substrates/hotpotqa/manifest.json"),
                "lineage_status": "existing JJ source snapshot",
            },
        )
        return questions, inventory
    questions = root / f"sources/graphrag_benchmark/{dataset}/questions.json"
    return questions, root / f"sources/graphrag_benchmark/{dataset}/source_manifest.json"


def run_command(command: list[str], *, cwd: Path, log: Path) -> None:
    save(log.with_suffix(log.suffix + ".command.json"), command)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}; see {log}"
        )


def _episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((dataset_root / "episodes").glob("*/episode.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def inspect_natural_run(root: Path) -> dict[str, Any]:
    """Compute protocol and execution rates from immutable episode artifacts."""
    episodes: list[dict[str, Any]] = []
    protocol_total = protocol_valid = 0
    execution_total = execution_success = 0
    assessment_total = assessment_valid = 0
    failures: dict[str, int] = {}
    for dataset in DATASETS:
        rows = _episode_rows(root / dataset)
        if len(rows) != len(CONDITIONS):
            failures[f"{dataset}:episode_count"] = len(rows)
        for episode in rows:
            steps = episode.get("trajectory", [])
            terminal = episode.get("termination_reason")
            terminal_ok = terminal in {"finish", "budget_exhausted", "policy_error", "runtime_error"}
            episode_protocol_total = episode_protocol_valid = 0
            episode_execution_total = episode_execution_success = 0
            episode_assessment_total = episode_assessment_valid = 0
            if not steps:
                # A provider failure before a StepRecord is still one failed
                # protocol/assessment probe; do not let an empty artifact
                # disappear from the denominator.
                protocol_total += 1
                assessment_total += 1
                failures["no_policy_step"] = failures.get("no_policy_step", 0) + 1
            if terminal in {"policy_error", "runtime_error"}:
                # A terminal runtime/provider error is an unsuccessful live
                # calibration outcome even when earlier turns were valid.
                protocol_total += 1
                failures[f"terminal_{terminal}"] = failures.get(
                    f"terminal_{terminal}", 0
                ) + 1
            for step in steps:
                provider = step.get("provider_metadata") or {}
                validation = step.get("validation_status")
                raw_calls = provider.get("raw_tool_calls") or []
                selected = provider.get("selected_action")
                is_protocol_valid = (
                    validation == "valid"
                    and provider.get("native_tool_calling") is True
                    and provider.get("tool_call_count") == 1
                    and len(raw_calls) == 1
                    and selected is not None
                )
                protocol_total += 1
                episode_protocol_total += 1
                if is_protocol_valid:
                    protocol_valid += 1
                    episode_protocol_valid += 1
                elif validation == "invalid":
                    key = str(provider.get("failure_category") or "protocol_invalid")
                    failures[key] = failures.get(key, 0) + 1

                requested = selected not in {None, "finish", "FINISH"}
                if requested:
                    execution_total += 1
                    episode_execution_total += 1
                    observation = step.get("observation") or {}
                    if validation == "valid" and observation.get("status") == "ok":
                        execution_success += 1
                        episode_execution_success += 1

                assessment_requested = bool(
                    (step.get("context_audit") or {}).get("assessment_requested", True)
                )
                if assessment_requested:
                    assessment_total += 1
                    episode_assessment_total += 1
                    decision = step.get("decision") or step.get("raw_policy_decision") or {}
                    # StepRecord artifacts serialize this under decision; older
                    # artifacts use raw_policy_decision in IO traces.
                    assessment = decision.get("assessment") if isinstance(decision, dict) else None
                    if validation == "valid" and isinstance(assessment, dict):
                        missing = assessment.get("missing_information")
                        if isinstance(missing, list) and len(missing) <= 3 and all(
                            isinstance(item, str) and item.strip() for item in missing
                        ):
                            assessment_valid += 1
                            episode_assessment_valid += 1

            episodes.append(
                {
                    "dataset": dataset,
                    "episode_id": episode.get("episode_id"),
                    "termination_reason": terminal,
                    "terminal_artifact": terminal_ok,
                    "protocol_calls": episode_protocol_total,
                    "protocol_valid_calls": episode_protocol_valid,
                    "execution_calls": episode_execution_total,
                    "execution_successes": episode_execution_success,
                    "assessment_requested": episode_assessment_total,
                    "assessment_valid": episode_assessment_valid,
                }
            )

    protocol_rate = protocol_valid / protocol_total if protocol_total else 0.0
    execution_rate = execution_success / execution_total if execution_total else 0.0
    assessment_rate = assessment_valid / assessment_total if assessment_total else 0.0
    return {
        "expected_episodes": len(DATASETS) * len(CONDITIONS),
        "observed_episodes": len(episodes),
        "all_terminal_artifacts": len(episodes) == len(DATASETS) * len(CONDITIONS)
        and all(item["terminal_artifact"] for item in episodes),
        "protocol_total": protocol_total,
        "protocol_valid": protocol_valid,
        "protocol_valid_rate": protocol_rate,
        "execution_total": execution_total,
        "execution_success": execution_success,
        "execution_success_rate": execution_rate,
        "assessment_total": assessment_total,
        "assessment_valid": assessment_valid,
        "assessment_parse_rate": assessment_rate,
        "failures": failures,
        "episodes": episodes,
        "gate_passed": (
            protocol_rate >= 0.99
            and execution_rate >= 0.95
            and assessment_rate >= 0.99
            and len(episodes) == len(DATASETS) * len(CONDITIONS)
            and all(item["terminal_artifact"] for item in episodes)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extended-output",
        type=Path,
        required=True,
        help="new directory for the 140-episode smoke; must not already exist",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        required=True,
        help="manifest from the paired 140-episode selection to reuse exactly",
    )
    parser.add_argument(
        "--require-assessment",
        action="store_true",
        help="required gate: run the information-gap schema rather than the legacy contract",
    )
    args = parser.parse_args()
    if not args.require_assessment:
        raise ValueError("this runner requires --require-assessment")
    if args.output.exists() or args.extended_output.exists():
        raise ValueError("both output directories must be new")
    repo = Path(__file__).resolve().parents[1]
    root = args.data_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "model": MODEL,
        "host": HOST,
        "conditions": list(CONDITIONS),
        "datasets": {},
        "assessment_schema_version": "information-gap-v2-resolved-gaps",
    }
    save(output / "status.json", status)
    config = repo / "configs/interface_study_v62_qwen27b_ollama.yaml"
    try:
        for dataset in DATASETS:
            substrate = root / f"substrates/{dataset}"
            questions, source_manifest = _question_paths(root, dataset, output)
            dataset_out = output / dataset
            dataset_out.mkdir(parents=True, exist_ok=True)
            run_command(
                [sys.executable, "scripts/validate_v2_substrate.py", "--substrate", str(substrate)],
                cwd=repo,
                log=output / f"{dataset}.substrate.log",
            )
            run_command(
                [
                    sys.executable,
                    "scripts/calibrate_interface.py",
                    "--substrate",
                    str(substrate),
                    "--conditions",
                    *CONDITIONS,
                    "--live",
                    "--model",
                    MODEL,
                    "--host",
                    HOST,
                    "--seed",
                    str(SEED),
                    "--output",
                    str(output / f"{dataset}.live_calibration.json"),
                ],
                cwd=repo,
                log=output / f"{dataset}.calibration.log",
            )
            calibration_report = json.loads(
                (output / f"{dataset}.live_calibration.json").read_text(encoding="utf-8")
            )
            if calibration_report.get("gate") != "live_protocol_and_assessment_passed":
                raise RuntimeError(
                    f"{dataset}: live protocol/assessment calibration failed; "
                    f"see {output / f'{dataset}.live_calibration.json'}"
                )
            run_command(
                [
                    sys.executable,
                    "scripts/probe_interface_backend_branches.py",
                    "--substrate",
                    str(substrate),
                    "--config",
                    str(config),
                    "--skill",
                    str(repo / "skills/interface_study.md"),
                    "--output",
                    str(output / f"{dataset}.backend_branches"),
                ],
                cwd=repo,
                log=output / f"{dataset}.backend.log",
            )
            status["datasets"][dataset] = {"status": "calibration_completed"}
            save(output / "status.json", status)

        # Natural calibration: one question for every condition, with the
        # required information-gap assessment turned on by run_jj_v2_smoke.
        run_command(
            [
                sys.executable,
                "scripts/run_jj_v2_smoke.py",
                "--data-root",
                str(root),
                "--output",
                str(output / "natural_21"),
                "--assessment",
                "on",
            ],
            cwd=repo,
            log=output / "natural_21.log",
        )
        natural_report = inspect_natural_run(output / "natural_21")
        save(output / "natural_21_gate.json", natural_report)
        status["natural_gate"] = natural_report
        if not natural_report["gate_passed"]:
            status["status"] = "gate_failed_extended_smoke_not_started"
            save(output / "status.json", status)
            raise RuntimeError("information-gap calibration gate failed; extended smoke was not started")

        run_command(
            [
                sys.executable,
                "scripts/run_jj_extended_smoke.py",
                "--data-root",
                str(root),
                "--output",
                str(args.extended_output.resolve()),
                "--require-assessment",
                "--selection-manifest",
                str(args.selection_manifest.resolve()),
            ],
            cwd=repo,
            log=output / "extended_140.log",
        )
        status["status"] = "finished_requires_artifact_audit"
        status["extended_output"] = args.extended_output.resolve().as_posix()
        save(output / "status.json", status)
    except Exception:
        status["status"] = "error"
        status["error"] = traceback.format_exc()
        save(output / "status.json", status)
        raise
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
