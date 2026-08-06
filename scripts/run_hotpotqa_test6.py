"""Run the fixed HotpotQA Test-6 split with the configured provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.evaluation import contain_accuracy, normalize_answer


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        choices=range(1, 7),
        metavar="1..6",
        help="Run only the first N items for a bounded provider smoke test.",
    )
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


def main() -> None:
    args = _arguments()
    config = AgentConfig.from_yaml(args.config)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    all_items = _load_jsonl(args.split)
    if len(all_items) != 6:
        raise ValueError(
            f"expected exactly 6 test items, received {len(all_items)}"
        )
    items = all_items[: args.limit] if args.limit is not None else all_items
    question_count = len(items)
    args.output.mkdir(parents=True)
    harness = AgentHarness.from_config(
        args.substrate,
        config,
        args.skill,
        args.output / "predictions",
    )

    rows: list[dict[str, Any]] = []
    usage_totals: dict[str, int] = {}
    for index, item in enumerate(items, start=1):
        task_id = str(item["id"])
        print(
            f"START {index}/{question_count} {task_id} {item['question']}",
            flush=True,
        )
        result = harness.run(
            str(item["question"]),
            str(item["scope_id"]),
            episode_id=task_id,
        )
        task_dir = Path(result.artifact_dir or "")
        _write_json(task_dir / "io_trace.json", harness.build_io_trace(result))

        usage = result.usage.model_dump(mode="json")
        for key, value in usage.items():
            usage_totals[key] = usage_totals.get(key, 0) + int(value)
        prediction = result.answer or ""
        gold = str(item["answer"])
        row = {
            "id": task_id,
            "question": str(item["question"]),
            "gold_answer": gold,
            "predicted_answer": prediction,
            "normalized_exact": int(
                normalize_answer(prediction) == normalize_answer(gold)
            ),
            "contain_acc": contain_accuracy(prediction, gold),
            "termination_reason": result.termination_reason.value,
            "error_code": result.error_code,
            "environment_steps": result.final_state.step,
            "policy_attempts": result.final_state.policy_attempts,
            "trajectory_records": len(result.trajectory),
            "invalid_attempts": sum(
                step.validation_status.value == "invalid"
                for step in result.trajectory
            ),
            "repairs": [
                step.repair_code
                for step in result.trajectory
                if step.repair_code is not None
            ],
            "usage": usage,
            "artifact_dir": result.artifact_dir,
        }
        rows.append(row)
        print(
            "DONE "
            f"{index}/{question_count} answer={prediction!r} "
            f"contain={row['contain_acc']} exact={row['normalized_exact']} "
            f"termination={row['termination_reason']} "
            f"steps={row['environment_steps']} "
            f"attempts={row['policy_attempts']} "
            f"repairs={len(row['repairs'])}",
            flush=True,
        )

    providers = {config.policy.provider, config.answer.provider}
    summary = {
        "run_contract": {
            "workflow_mode": config.workflow_mode,
            "policy_provider": config.policy.provider,
            "policy_model": config.policy.model,
            "answer_provider": config.answer.provider,
            "answer_model": config.answer.model,
            "openai_used": "openai" in providers,
            "split": args.split.as_posix(),
            "skill": args.skill.as_posix(),
            "config": args.config.as_posix(),
            "limit": args.limit,
        },
        "question_count": question_count,
        "normalized_exact_correct": sum(
            row["normalized_exact"] for row in rows
        ),
        "contain_correct": sum(row["contain_acc"] for row in rows),
        "total_invalid_attempts": sum(
            row["invalid_attempts"] for row in rows
        ),
        "total_repairs": sum(len(row["repairs"]) for row in rows),
        "usage": usage_totals,
        "results": rows,
    }
    _write_json(args.output / "summary.json", summary)
    print(
        "SUMMARY "
        f"contain={summary['contain_correct']}/{question_count} "
        f"exact={summary['normalized_exact_correct']}/{question_count} "
        f"policy_calls={usage_totals.get('policy_calls', 0)} "
        f"total_tokens={usage_totals.get('total_tokens', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
