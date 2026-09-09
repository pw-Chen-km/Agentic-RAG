"""Run a resumable five-dataset benchmark split and persist full traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ollama import Client

from agentic_rag.agent.config import AgentConfig, OllamaPolicyConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.evaluation import (
    EpisodeEvaluator,
    JudgeMessage,
    JudgeResponse,
    JudgeResponseError,
    JudgeUsage,
    normalize_answer,
)
from agentic_rag.evaluation.profiles import get_dataset_profile
from agentic_rag.skillopt.benchmark import validate_benchmark_lineage
from agentic_rag.skillopt.data import validate_hotpotqa_smoke_lineage
from agentic_rag.substrate.storage import Substrate


_REFERENCE_ERROR_CODES = {
    "chunk_not_readable",
    "expansion_not_valid_for_node",
    "reference_not_available",
    "reference_type_mismatch",
    "reference_not_evidence",
}


class LocalQwenJudge:
    """Structured semantic judge on the same fixed Ollama endpoint."""

    def __init__(self, config: OllamaPolicyConfig) -> None:
        self.model = config.model
        self.host = config.host
        self.think = config.think
        self.num_ctx = config.num_ctx
        self.keep_alive = config.keep_alive
        self.client = Client(host=config.host, timeout=config.timeout_seconds)

    def judge(self, messages: Sequence[JudgeMessage]) -> JudgeResponse:
        schema = {
            "type": "object",
            "properties": {"correct": {"type": "boolean"}},
            "required": ["correct"],
            "additionalProperties": False,
        }
        response = self.client.chat(
            model=self.model,
            messages=[message.as_provider_input() for message in messages],
            stream=False,
            format=schema,
            think=self.think,
            options={"temperature": 0, "num_ctx": self.num_ctx},
            keep_alive=self.keep_alive,
        )
        try:
            payload = json.loads(response.message.content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise JudgeResponseError("local Qwen judge returned invalid JSON") from exc
        correct = payload.get("correct")
        if type(correct) is not bool:
            raise JudgeResponseError("local Qwen judge omitted boolean correct")
        input_tokens = max(int(response.prompt_eval_count or 0), 0)
        output_tokens = max(int(response.eval_count or 0), 0)
        return JudgeResponse(
            correct=correct,
            raw_output=payload,
            model=self.model,
            usage=JudgeUsage(
                calls=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            request_options={
                "provider": "ollama",
                "host": self.host,
                "think": self.think,
                "temperature": 0,
                "num_ctx": self.num_ctx,
            },
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--dataset", default="hotpotqa")
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
    profile = get_dataset_profile(args.dataset)
    return {
        "architecture": "semantic_memory_typed_refs_compact",
        "policy_provider": config.policy.provider,
        "policy_model": config.policy.model,
        "judge_model": config.policy.model,
        "judge_host": getattr(config.policy, "host", None),
        "judge_thinking": getattr(config.policy, "think", None),
        "openai_used": config.policy.provider == "openai",
        "split": args.split.as_posix(),
        "split_sha256": _sha256(args.split),
        "skill": args.skill.as_posix(),
        "skill_sha256": _sha256(args.skill),
        "config": args.config.as_posix(),
        "config_sha256": _sha256(args.config),
        "substrate": args.substrate.as_posix(),
        "substrate_manifest_sha256": _sha256(args.substrate / "manifest.json"),
        "expected_count": args.expected_count,
        "dataset": profile.key,
        "scope_id": profile.scope_id("dev"),
        "reported_metrics": [
            metric.value for metric in profile.reported_metrics
        ],
    }


def _usage_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        for key, value in row["usage"].items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def _judge_usage_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        for key, value in (row.get("judge_usage") or {}).items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def _summary(
    contract: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "run_contract": contract,
        "question_count": len(rows),
        "llm_acc_correct": sum(row["llm_acc"] for row in rows),
        "hard_correct": sum(row["hard"] for row in rows),
        "soft_total": sum(float(row["soft"]) for row in rows),
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
        "judge_usage": _judge_usage_totals(rows),
        "results": rows,
    }


def main() -> None:
    args = _arguments()
    config = AgentConfig.from_yaml(args.config)
    profile = get_dataset_profile(args.dataset)
    if not isinstance(config.policy, OllamaPolicyConfig):
        raise ValueError("benchmark LLM judge requires an Ollama policy config")
    evaluator = EpisodeEvaluator(LocalQwenJudge(config.policy), profile=profile)
    substrate = Substrate.open(args.substrate)
    if substrate.manifest.dataset != profile.key:
        raise ValueError(
            f"substrate dataset is {substrate.manifest.dataset!r}; expected "
            f"{profile.key!r}"
        )
    substrate.require_scope(profile.scope_id("dev"))
    split_manifest_path = args.split.parent / "split_manifest.json"
    if split_manifest_path.is_file():
        split_manifest = json.loads(
            split_manifest_path.read_text(encoding="utf-8")
        )
        if split_manifest.get("schema_version") in {"1.0", "1.1"}:
            if profile.key != "hotpotqa":
                raise ValueError("schema 1.x splits support only HotpotQA")
            validate_hotpotqa_smoke_lineage(
                args.split.parent, substrate.manifest
            )
        else:
            validate_benchmark_lineage(
                args.split.parent,
                substrate.manifest,
                dataset=profile,
            )
    items = _load_jsonl(args.split)
    if len(items) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} items, received {len(items)}"
        )
    expected_scope = profile.scope_id("dev")
    for item in items:
        if item.get("scope_id") != expected_scope:
            raise ValueError(
                f"item {item.get('id')} scope does not match {expected_scope}"
            )
        if item.get("source") != profile.key:
            raise ValueError(
                f"item {item.get('id')} source does not match {profile.key}"
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
        evaluation = evaluator.evaluate(
            question=str(item["question"]),
            predicted_answer=prediction,
            gold_answer=gold,
        )
        _write_json(
            task_dir / "evaluation.json",
            evaluation.model_dump(mode="json"),
        )
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
            "llm_acc": evaluation.llm_acc,
            "hard": evaluation.hard,
            "soft": evaluation.soft,
            "metric_contract": {
                "hard": evaluation.hard_metric.value,
                "soft": evaluation.soft_metric.value,
            },
            "normalized_exact": int(
                normalize_answer(prediction) == normalize_answer(gold)
            ),
            "contain_acc": evaluation.contain_acc,
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
            "judge_usage": evaluation.judge_usage.model_dump(mode="json"),
            "artifact_dir": result.artifact_dir,
        }
        rows.append(row)
        _write_json(
            progress_path,
            {"run_contract": contract, "results": rows},
        )
        print(
            f"DONE {index}/{len(items)} answer={prediction!r} "
            f"hard={row['hard']} contain={row['contain_acc']} "
            f"exact={row['normalized_exact']} "
            f"termination={row['termination_reason']} "
            f"steps={row['environment_steps']} "
            f"attempts={row['policy_attempts']} "
            f"invalid={row['invalid_attempts']}",
            flush=True,
        )

    summary = _summary(contract, rows)
    _write_json(args.output / "summary.json", summary)
    print(
        f"SUMMARY hard={summary['hard_correct']}/{len(items)} "
        f"contain={summary['contain_correct']}/{len(items)} "
        f"exact={summary['normalized_exact_correct']}/{len(items)} "
        f"invalid={summary['total_invalid_attempts']} "
        f"policy_calls={summary['usage'].get('policy_calls', 0)} "
        f"total_tokens={summary['usage'].get('total_tokens', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
