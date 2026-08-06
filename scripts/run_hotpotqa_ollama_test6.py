"""Run the fixed HotpotQA Test-6 split with Ollama and persist full traces."""

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
    providers = (config.policy, config.answer)
    if any(provider.provider != "ollama" for provider in providers):
        raise ValueError("This runner requires Ollama for policy and answer")
    if any("gpt" in provider.model.casefold() for provider in providers):
        raise ValueError("GPT-named models are forbidden for this run")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    items = _load_jsonl(args.split)
    if len(items) != 6:
        raise ValueError(f"expected exactly 6 test items, received {len(items)}")
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
            f"START {index}/6 {task_id} {item['question']}",
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
                prior.observation is not None
                and bool(prior.observation.results)
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
        index_resolution_errors = sum(
            step.observation is not None
            and step.observation.error_code
            in {
                "memory_index_out_of_range",
                "citation_index_out_of_range",
                "memory_node_type_mismatch",
                "chunk_not_readable",
                "expansion_not_valid_for_node",
                "reference_not_available",
                "reference_type_mismatch",
                "reference_not_evidence",
            }
            for step in result.trajectory
        )
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
            "action_counts": action_counts,
            "search_after_observation": search_after_observation,
            "search_after_no_progress": search_after_no_progress,
            "context_index_errors": index_resolution_errors,
            "usage": usage,
            "artifact_dir": result.artifact_dir,
        }
        rows.append(row)
        print(
            "DONE "
            f"{index}/6 answer={prediction!r} "
            f"contain={row['contain_acc']} exact={row['normalized_exact']} "
            f"termination={row['termination_reason']} "
            f"steps={row['environment_steps']} "
            f"attempts={row['policy_attempts']} "
            f"repairs={len(row['repairs'])}",
            flush=True,
        )

    summary = {
        "run_contract": {
            "policy_provider": config.policy.provider,
            "policy_model": config.policy.model,
            "answer_provider": config.answer.provider,
            "answer_model": config.answer.model,
            "gpt_or_openai_used": False,
            "split": args.split.as_posix(),
            "skill": args.skill.as_posix(),
            "config": args.config.as_posix(),
        },
        "question_count": len(rows),
        "normalized_exact_correct": sum(
            row["normalized_exact"] for row in rows
        ),
        "contain_correct": sum(row["contain_acc"] for row in rows),
        "total_invalid_attempts": sum(
            row["invalid_attempts"] for row in rows
        ),
        "total_repairs": sum(len(row["repairs"]) for row in rows),
        "action_counts": {
            action_type: sum(
                row["action_counts"][action_type] for row in rows
            )
            for action_type in ("SEARCH", "EXPAND", "READ", "FINISH")
        },
        "search_after_observation": sum(
            row["search_after_observation"] for row in rows
        ),
        "search_after_no_progress": sum(
            row["search_after_no_progress"] for row in rows
        ),
        "context_index_errors": sum(
            row["context_index_errors"] for row in rows
        ),
        "usage": usage_totals,
        "results": rows,
    }
    _write_json(args.output / "summary.json", summary)
    print(
        "SUMMARY "
        f"contain={summary['contain_correct']}/6 "
        f"exact={summary['normalized_exact_correct']}/6 "
        f"policy_calls={usage_totals.get('policy_calls', 0)} "
        f"total_tokens={usage_totals.get('total_tokens', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
