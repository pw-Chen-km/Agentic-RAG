#!/usr/bin/env python3
"""Run the fixed Options v1 smoke sample on an existing substrate.

The preparation script is deliberately separate: it creates the immutable
20/20/20 sample without model calls, while this command performs resumable
target inference only.  Gold answers remain in the result-side question file
and are never put into the Options harness input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_rag.agent.config import AgentConfig
from agentic_rag.evaluation import normalize_answer
from agentic_rag.options.harness import OptionsAgentHarness


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config(raw: dict[str, Any]) -> AgentConfig:
    options = raw["options_v1"]
    model = raw["model"]
    return AgentConfig.model_validate(
        {
            "max_steps": options.get("max_steps", 10),
            "max_policy_attempts": options.get("max_policy_attempts", 12),
            "max_retrieved_tokens": options.get("max_retrieved_tokens", 12_000),
            "show_available_action_options": True,
            "use_state_conditioned_schema": True,
            "policy": {"provider": "ollama", **model},
        }
    )


def run_dataset(runtime: dict[str, Any], dataset: str, *, resume: bool) -> dict[str, Any]:
    entry = runtime["datasets"][dataset]
    questions_path = Path(entry["questions"])
    rows = _read_jsonl(questions_path)
    if len(rows) != int(entry["expected_count"]):
        raise ValueError(f"{dataset}: expected {entry['expected_count']} questions, got {len(rows)}")
    output = Path(entry["output"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.jsonl"
    existing = _read_jsonl(result_path) if resume and result_path.is_file() else []
    done = {str(row["question_id"]) for row in existing}
    harness = OptionsAgentHarness.from_config(
        entry["substrate"],
        _config(runtime),
        None,
        runtime["options_v1"]["option_store"],
        output / "episodes",
    )
    all_rows = list(existing)
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in done:
            continue
        result = harness.run(
            str(row["question"]), str(row["scope_id"]),
            episode_id=f"options-v1-{dataset}-{question_id}",
        )
        all_rows.append({
            "question_id": question_id,
            "source": dataset,
            "question": row["question"],
            "gold_answer": row.get("answer"),
            "question_type": row.get("question_type"),
            "answer": result.answer,
            "normalized_exact_match": int(
                result.answer is not None
                and normalize_answer(result.answer) == normalize_answer(row.get("answer"))
            ),
            "termination_reason": result.termination_reason.value,
            "usage": result.usage.model_dump(mode="json"),
            "option_trace": result.option_trace,
            "artifact_dir": result.artifact_dir,
            "error_code": result.error_code,
            "error_message": result.error_message,
        })
        _write_jsonl(result_path, all_rows)
    summary = {
        "dataset": dataset,
        "question_count": len(all_rows),
        "completed": len(all_rows) == len(rows),
        "model": runtime["model"],
        "options_v1": runtime["options_v1"],
        "results": result_path.as_posix(),
        "normalized_exact_match": (
            sum(int(row.get("normalized_exact_match", 0)) for row in all_rows) / len(all_rows)
            if all_rows else 0.0
        ),
        "empty_answers": sum(not str(row.get("answer") or "").strip() for row in all_rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hotpotqa", "novel", "medical"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    datasets = [args.dataset] if args.dataset else ["hotpotqa", "novel", "medical"]
    summaries = [run_dataset(runtime, dataset, resume=args.resume) for dataset in datasets]
    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
