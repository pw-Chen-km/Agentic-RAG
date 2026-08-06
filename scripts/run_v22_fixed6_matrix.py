"""Run V2-2 without SkillOpt on five datasets, three splits, six items each."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness


WORKSPACE = Path(__file__).resolve().parents[1]
DATASETS = {
    "2wikimultihop": (
        WORKSPACE / "artifacts" / "2wikimultihop_benchmark_exact",
        WORKSPACE / "data" / "skillopt" / "2wikimultihop_smoke",
    ),
    "hotpotqa": (
        WORKSPACE / "artifacts" / "hotpotqa_benchmark_exact",
        WORKSPACE / "data" / "skillopt" / "hotpotqa_smoke",
    ),
    "medical": (
        WORKSPACE / "artifacts" / "medical_benchmark_exact",
        WORKSPACE / "data" / "skillopt" / "medical_smoke",
    ),
    "musique": (
        WORKSPACE / "artifacts" / "musique_benchmark_exact",
        WORKSPACE / "data" / "skillopt" / "musique_smoke",
    ),
    "novel": (
        WORKSPACE / "artifacts" / "novel_benchmark_exact",
        WORKSPACE / "data" / "skillopt" / "novel_smoke",
    ),
}
SPLITS = ("train", "validation", "test")
ITEMS_PER_SPLIT = 6


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(DATASETS),
        help="Run only this dataset; repeat to select several.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=SPLITS,
        help="Run only this split; repeat to select several.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job_contract(
    *,
    dataset: str,
    split: str,
    substrate: Path,
    split_file: Path,
    config_file: Path,
    skill_file: Path,
    items: list[dict[str, Any]],
    config: AgentConfig,
) -> dict[str, Any]:
    return {
        "workflow_mode": config.workflow_mode,
        "policy_provider": config.policy.provider,
        "policy_model": config.policy.model,
        "policy_think": getattr(config.policy, "think", None),
        "answer_provider": config.answer.provider,
        "answer_model": config.answer.model,
        "dataset": dataset,
        "split": split,
        "substrate": substrate.as_posix(),
        "source_split": split_file.as_posix(),
        "source_split_sha256": _sha256(split_file),
        "selection": "first-6-in-fixed-jsonl-order-v1",
        "selected_ids": [str(item["id"]) for item in items],
        "config": config_file.as_posix(),
        "config_sha256": _sha256(config_file),
        "skill": skill_file.as_posix(),
        "skill_root_sha256": _sha256(skill_file),
        "skillopt_used": False,
        "llm_judge_used": False,
        "accuracy_is_completion_gate": False,
    }


def _row(result: Any, trace: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    action_counts = {
        action_type: sum(
            step.resolved_decision is not None
            and step.resolved_decision.action.type == action_type
            for step in result.trajectory
        )
        for action_type in ("SEARCH", "EXPAND", "READ", "FINISH")
    }
    return {
        "id": str(item["id"]),
        "question": str(item["question"]),
        "gold_answer": str(item.get("answer") or ""),
        "predicted_answer": result.answer or "",
        "termination_reason": result.termination_reason.value,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "environment_steps": result.final_state.step,
        "action_cycles": result.final_state.policy_attempts,
        "policy_calls": result.usage.policy_calls,
        "invalid_cycles": sum(
            step.validation_status.value == "invalid"
            for step in result.trajectory
        ),
        "action_counts": action_counts,
        "v2_2_metrics": trace.get("v2_2_metrics", {}),
        "usage": result.usage.model_dump(mode="json"),
        "artifact_dir": result.artifact_dir,
    }


def _job_summary(
    contract: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    usage: dict[str, int] = {}
    for row in rows:
        for key, value in row["usage"].items():
            usage[key] = usage.get(key, 0) + int(value)
    return {
        "run_contract": contract,
        "question_count": len(rows),
        "finish_count": sum(
            row["termination_reason"] == "finish" for row in rows
        ),
        "error_count": sum(row["error_code"] is not None for row in rows),
        "total_invalid_cycles": sum(row["invalid_cycles"] for row in rows),
        "usage": usage,
        "results": rows,
    }


def main() -> None:
    args = _arguments()
    config_file = args.config.resolve()
    skill_file = args.skill.resolve()
    output_root = args.output.resolve()
    config = AgentConfig.from_yaml(config_file)
    if config.workflow_mode != "single_agent_v2_2":
        raise ValueError("matrix runner requires single_agent_v2_2")
    if config.policy.provider != "ollama":
        raise ValueError("this local fixed-6 run requires the Ollama policy")
    if getattr(config.policy, "think", None) is not False:
        raise ValueError("this fixed-6 run requires policy think: false")
    if skill_file.name != "SKILL.md":
        raise ValueError("--skill must point to the V2-2 bundle root SKILL.md")
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"output already exists: {output_root}")

    # These exact substrates were constructed with MiniLM and the model is
    # already cached locally. Prevent hub availability from affecting a local
    # Ollama workflow when DENSE retrieval or semantic expansion is selected.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    selected_datasets = tuple(args.dataset or DATASETS)
    selected_splits = tuple(args.split or SPLITS)
    output_root.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []

    for dataset in selected_datasets:
        substrate, split_root = DATASETS[dataset]
        dataset_output = output_root / dataset
        harness = AgentHarness.from_config(
            substrate,
            config,
            skill_file,
            dataset_output / "predictions",
        )
        for split in selected_splits:
            split_file = split_root / f"{split}.jsonl"
            available = _read_jsonl(split_file)
            if len(available) < ITEMS_PER_SPLIT:
                raise ValueError(
                    f"{dataset}/{split} has {len(available)} items; "
                    f"expected at least {ITEMS_PER_SPLIT}"
                )
            items = available[:ITEMS_PER_SPLIT]
            contract = _job_contract(
                dataset=dataset,
                split=split,
                substrate=substrate,
                split_file=split_file,
                config_file=config_file,
                skill_file=skill_file,
                items=items,
                config=config,
            )
            progress_path = dataset_output / f"{split}.progress.json"
            summary_path = dataset_output / f"{split}.summary.json"
            rows: list[dict[str, Any]] = []
            if args.resume and progress_path.is_file():
                progress = json.loads(
                    progress_path.read_text(encoding="utf-8")
                )
                if progress.get("run_contract") != contract:
                    raise ValueError(
                        f"resume contract mismatch for {dataset}/{split}"
                    )
                rows = list(progress.get("results") or [])
            completed_ids = {str(row["id"]) for row in rows}

            for index, item in enumerate(items, start=1):
                task_id = str(item["id"])
                if task_id in completed_ids:
                    print(
                        f"SKIP {dataset}/{split} {index}/6 {task_id}",
                        flush=True,
                    )
                    continue
                print(
                    f"START {dataset}/{split} {index}/6 {task_id} "
                    f"{item['question']}",
                    flush=True,
                )
                episode_id = f"{dataset}-{split}-{task_id}"
                result = harness.run(
                    str(item["question"]),
                    str(item["scope_id"]),
                    episode_id=episode_id,
                )
                trace = harness.build_io_trace(result)
                artifact_dir = Path(result.artifact_dir or "")
                _write_json(artifact_dir / "io_trace.json", trace)
                row = _row(result, trace, item)
                rows.append(row)
                _write_json(
                    progress_path,
                    {"run_contract": contract, "results": rows},
                )
                print(
                    f"DONE {dataset}/{split} {index}/6 "
                    f"termination={row['termination_reason']} "
                    f"steps={row['environment_steps']} "
                    f"cycles={row['action_cycles']} "
                    f"calls={row['policy_calls']} "
                    f"invalid={row['invalid_cycles']}",
                    flush=True,
                )

            summary = _job_summary(contract, rows)
            _write_json(summary_path, summary)
            all_summaries.append(summary)

    matrix_summary = {
        "schema_version": "1.0",
        "expected_questions_per_job": ITEMS_PER_SPLIT,
        "selected_datasets": list(selected_datasets),
        "selected_splits": list(selected_splits),
        "job_count": len(all_summaries),
        "question_count": sum(
            summary["question_count"] for summary in all_summaries
        ),
        "finish_count": sum(
            summary["finish_count"] for summary in all_summaries
        ),
        "error_count": sum(
            summary["error_count"] for summary in all_summaries
        ),
        "skillopt_used": False,
        "llm_judge_used": False,
        "accuracy_is_completion_gate": False,
        "jobs": all_summaries,
    }
    _write_json(output_root / "matrix_summary.json", matrix_summary)
    print(
        "MATRIX DONE "
        f"jobs={matrix_summary['job_count']} "
        f"questions={matrix_summary['question_count']} "
        f"finish={matrix_summary['finish_count']} "
        f"errors={matrix_summary['error_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
