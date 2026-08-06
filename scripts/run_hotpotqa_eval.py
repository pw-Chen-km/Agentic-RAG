"""Run a resumable HotpotQA evaluation split and persist full traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.evaluation import contain_accuracy, normalize_answer


_REFERENCE_ERROR_CODES = {
    "chunk_not_readable",
    "expansion_not_valid_for_node",
    "reference_not_available",
    "reference_type_mismatch",
    "reference_not_evidence",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(payload + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(args: argparse.Namespace, config: AgentConfig) -> dict[str, Any]:
    return {
        "architecture": "semantic_memory_typed_refs_compact",
        "policy_provider": config.policy.provider,
        "policy_model": config.policy.model,
        "openai_used": config.policy.provider == "openai",
        "split": args.split.as_posix(),
        "split_sha256": _sha256(args.split),
        "skill": args.skill.as_posix(),
        "skill_sha256": _sha256(args.skill),
        "config": args.config.as_posix(),
        "config_sha256": _sha256(args.config),
        "substrate": args.substrate.as_posix(),
        "expected_count": args.expected_count,
    }


def _usage_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        for key, value in row["usage"].items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def _summary(
    contract: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "run_contract": contract,
        "question_count": len(rows),
        "normalized_exact_correct": sum(row["normalized_exact"] for row in rows),
        "contain_correct": sum(row["contain_acc"] for row in rows),
        "total_invalid_attempts": sum(row["invalid_attempts"] for row in rows),
        "action_counts": {
            action_type: sum(row["action_counts"][action_type] for row in rows)
            for action_type in ("SEARCH", "EXPAND", "READ", "FINISH")
        },
        "search_after_observation": sum(
            row["search_after_observation"] for row in rows
        ),
        "search_after_no_progress": sum(
            row["search_after_no_progress"] for row in rows
        ),
        "reference_errors": sum(row["reference_errors"] for row in rows),
        "usage": _usage_totals(rows),
        "results": rows,
    }


def main() -> None:
    args = _arguments()
    config = AgentConfig.from_yaml(args.config)
    items = _load_jsonl(args.split)
    if len(items) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} items, received {len(items)}"
        )
    contract = _contract(args, config)
    progress_path = args.output / "progress.json"
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output already exists: {args.output}")

    rows: list[dict[str, Any]] = []
    if args.resume:
        if not progress_path.is_file():
            raise FileNotFoundError(f"resume progress is missing: {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("run_contract") != contract:
            raise ValueError("resume contract does not match current inputs")
        rows = list(progress.get("results") or [])
    else:
        args.output.mkdir(parents=True)

    completed_ids = {str(row["id"]) for row in rows}
    harness = AgentHarness.from_config(
        args.substrate,
        config,
        args.skill,
        args.output / "predictions",
    )

    for index, item in enumerate(items, start=1):
        task_id = str(item["id"])
        if task_id in completed_ids:
            print(f"SKIP {index}/{len(items)} {task_id}", flush=True)
            continue
        print(
            f"START {index}/{len(items)} {task_id} {item['question']}",
            flush=True,
        )
        result = harness.run(
            str(item["question"]),
            str(item["scope_id"]),
            episode_id=task_id,
        )
        task_dir = Path(result.artifact_dir or "")
        _write_json(task_dir / "io_trace.json", harness.build_io_trace(result))

        prediction = result.answer or ""
        gold = str(item["answer"])
        action_counts = {
            action_type: sum(
                step.decision is not None
                and step.decision.action.type == action_type
                for step in result.trajectory
            )
            for action_type in ("SEARCH", "EXPAND", "READ", "FINISH")
        }
        search_after_observation = sum(
            step.decision is not None
            and step.decision.action.type == "SEARCH"
            and any(
                prior.observation is not None and bool(prior.observation.results)
                for prior in result.trajectory[:position]
            )
            for position, step in enumerate(result.trajectory)
        )
        search_after_no_progress = sum(
            position > 0
            and step.decision is not None
            and step.decision.action.type == "SEARCH"
            and result.trajectory[position - 1].observation is not None
            and not result.trajectory[position - 1].observation.novel_node_ids
            for position, step in enumerate(result.trajectory)
        )
        reference_errors = sum(
            step.observation is not None
            and step.observation.error_code in _REFERENCE_ERROR_CODES
            for step in result.trajectory
        )
        row = {
            "id": task_id,
            "question": str(item["question"]),
            "question_type": str(item.get("question_type") or ""),
            "gold_answer": gold,
            "predicted_answer": prediction,
            "normalized_exact": int(
                normalize_answer(prediction) == normalize_answer(gold)
            ),
            "contain_acc": contain_accuracy(prediction, gold),
            "termination_reason": result.termination_reason.value,
            "error_code": result.error_code,
            "environment_steps": result.final_state.step if result.final_state else 0,
            "policy_attempts": (
                result.final_state.policy_attempts if result.final_state else 0
            ),
            "trajectory_records": len(result.trajectory),
            "invalid_attempts": sum(
                step.validation_status.value == "invalid"
                for step in result.trajectory
            ),
            "action_counts": action_counts,
            "search_after_observation": search_after_observation,
            "search_after_no_progress": search_after_no_progress,
            "reference_errors": reference_errors,
            "usage": result.usage.model_dump(mode="json"),
            "artifact_dir": result.artifact_dir,
        }
        rows.append(row)
        _write_json(
            progress_path,
            {"run_contract": contract, "results": rows},
        )
        print(
            f"DONE {index}/{len(items)} answer={prediction!r} "
            f"contain={row['contain_acc']} exact={row['normalized_exact']} "
            f"termination={row['termination_reason']} "
            f"steps={row['environment_steps']} "
            f"attempts={row['policy_attempts']} "
            f"invalid={row['invalid_attempts']}",
            flush=True,
        )

    summary = _summary(contract, rows)
    _write_json(args.output / "summary.json", summary)
    print(
        f"SUMMARY contain={summary['contain_correct']}/{len(items)} "
        f"exact={summary['normalized_exact_correct']}/{len(items)} "
        f"invalid={summary['total_invalid_attempts']} "
        f"policy_calls={summary['usage'].get('policy_calls', 0)} "
        f"total_tokens={summary['usage'].get('total_tokens', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
