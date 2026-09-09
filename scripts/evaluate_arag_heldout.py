#!/usr/bin/env python3
"""Evaluate an existing A-RAG prediction file with Agentic-RAG's judge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from agentic_rag.agent.config import AgentConfig, OllamaPolicyConfig
from agentic_rag.evaluation import EpisodeEvaluator, normalize_answer
from agentic_rag.evaluation.profiles import get_dataset_profile

from run_benchmark_eval import LocalQwenJudge


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elapsed-seconds-file", type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = arguments()
    predictions = load_jsonl(args.predictions)
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    if not isinstance(sample, list) or len(sample) != 100:
        raise ValueError("sample must contain exactly 100 questions")
    if len(predictions) != len(sample):
        raise ValueError(
            f"prediction count {len(predictions)} does not match sample {len(sample)}"
        )
    sample_ids = [str(row["id"]) for row in sample]
    prediction_ids = [str(row.get("qid") or row.get("id") or "") for row in predictions]
    if prediction_ids != sample_ids:
        raise ValueError("A-RAG prediction IDs do not match the ordered Sample-100")

    config = AgentConfig.from_yaml(args.config)
    if not isinstance(config.policy, OllamaPolicyConfig):
        raise ValueError("the common Qwen judge requires an Ollama policy config")
    profile = get_dataset_profile("hotpotqa")
    evaluator = EpisodeEvaluator(LocalQwenJudge(config.policy), profile=profile)
    rows: list[dict[str, Any]] = []
    for index, (source, prediction) in enumerate(zip(sample, predictions, strict=True), 1):
        task_id = str(source["id"])
        print(f"JUDGE {index}/100 {task_id}", flush=True)
        predicted_answer = str(prediction.get("pred_answer") or "")
        gold_answer = str(source["answer"])
        evaluation = evaluator.evaluate(
            question=str(source["question"]),
            predicted_answer=predicted_answer,
            gold_answer=gold_answer,
        )
        rows.append(
            {
                "id": task_id,
                "question": str(source["question"]),
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "llm_acc": evaluation.llm_acc,
                "hard": evaluation.hard,
                "soft": evaluation.soft,
                "normalized_exact": int(
                    normalize_answer(predicted_answer)
                    == normalize_answer(gold_answer)
                ),
                "contain_acc": evaluation.contain_acc,
                "error_code": prediction.get("error"),
                "termination_reason": (
                    "error" if prediction.get("error") else "completed"
                ),
                "loops": int(prediction.get("loops") or 0),
                "chunks_read_count": int(prediction.get("chunks_read_count") or 0),
                "usage": {
                    "policy_calls": int(prediction.get("llm_calls") or 0),
                    "input_tokens": int(prediction.get("llm_input_tokens") or 0),
                    "output_tokens": int(prediction.get("llm_output_tokens") or 0),
                    "total_tokens": int(prediction.get("llm_total_tokens") or 0),
                    "retrieved_tokens": int(
                        prediction.get("total_retrieved_tokens") or 0
                    ),
                },
                "judge_usage": evaluation.judge_usage.model_dump(mode="json"),
            }
        )

    elapsed_seconds = None
    if args.elapsed_seconds_file and args.elapsed_seconds_file.is_file():
        elapsed_seconds = float(
            args.elapsed_seconds_file.read_text(encoding="utf-8").strip()
        )
    count = len(rows)
    payload = {
        "schema_version": "1.0",
        "run_contract": {
            "architecture": "A-RAG",
            "policy_model": config.policy.model,
            "judge_model": config.policy.model,
            "judge_host": config.policy.host,
            "judge_thinking": config.policy.think,
            "sample_sha256": sha256(args.sample),
            "predictions_sha256": sha256(args.predictions),
            "question_ids": sample_ids,
            "question_count": count,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "workers": 1,
        },
        "question_count": count,
        "llm_acc_correct": sum(int(row["llm_acc"]) for row in rows),
        "hard_correct": sum(int(row["hard"]) for row in rows),
        "soft_total": sum(float(row["soft"]) for row in rows),
        "normalized_exact_correct": sum(
            int(row["normalized_exact"]) for row in rows
        ),
        "contain_correct": sum(int(row["contain_acc"]) for row in rows),
        "blank_answers": sum(
            not str(row["predicted_answer"]).strip() for row in rows
        ),
        "errors": sum(bool(row["error_code"]) for row in rows),
        "elapsed_seconds": elapsed_seconds,
        "results": rows,
    }
    write_json(args.output, payload)
    print(
        "SUMMARY "
        f"hard={payload['llm_acc_correct']}/{count} "
        f"contain={payload['contain_correct']}/{count} "
        f"exact={payload['normalized_exact_correct']}/{count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
