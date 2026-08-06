"""Convert A-RAG predictions.jsonl into the shared HotpotQA summary schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agentic_rag.evaluation import contain_accuracy, normalize_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object in {path}")
            rows.append(value)
    return rows


def main() -> None:
    args = parse_args()
    source_rows = load_jsonl(args.predictions)
    results: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    for row in source_rows:
        prediction = str(row.get("pred_answer") or "")
        gold = str(row.get("gold_answer") or "")
        trajectory = row.get("trajectory") or []
        for event in trajectory:
            if isinstance(event, dict) and event.get("tool_name"):
                action_counts[str(event["tool_name"])] += 1
        results.append(
            {
                "id": str(row.get("qid") or row.get("id") or ""),
                "question": str(row.get("question") or ""),
                "gold_answer": gold,
                "predicted_answer": prediction,
                "normalized_exact": int(
                    bool(prediction.strip())
                    and normalize_answer(prediction) == normalize_answer(gold)
                ),
                "contain_acc": contain_accuracy(prediction, gold),
                "termination_reason": (
                    "error"
                    if row.get("error")
                    else "blank"
                    if not prediction.strip()
                    else "answered"
                ),
                "loops": int(row.get("loops") or 0),
                "llm_calls": int(row.get("llm_calls") or 0),
                "llm_input_tokens": int(row.get("llm_input_tokens") or 0),
                "llm_output_tokens": int(row.get("llm_output_tokens") or 0),
                "llm_total_tokens": int(row.get("llm_total_tokens") or 0),
                "retrieved_tokens": int(row.get("total_retrieved_tokens") or 0),
                "chunks_read_count": int(row.get("chunks_read_count") or 0),
                "trajectory_records": len(trajectory),
                "error": row.get("error"),
            }
        )

    summary = {
        "run_contract": {
            "system": "A-RAG",
            "model": args.model,
            "source_predictions": args.predictions.as_posix(),
            "max_loops": 10,
            "max_token_budget": 128000,
        },
        "question_count": len(results),
        "normalized_exact_correct": sum(r["normalized_exact"] for r in results),
        "contain_correct": sum(r["contain_acc"] for r in results),
        "blank_answers": sum(not r["predicted_answer"].strip() for r in results),
        "error_answers": sum(bool(r["error"]) for r in results),
        "action_counts": dict(sorted(action_counts.items())),
        "usage": {
            "policy_calls": sum(r["llm_calls"] for r in results),
            "input_tokens": sum(r["llm_input_tokens"] for r in results),
            "output_tokens": sum(r["llm_output_tokens"] for r in results),
            "total_tokens": sum(r["llm_total_tokens"] for r in results),
            "retrieved_tokens": sum(r["retrieved_tokens"] for r in results),
            "loops": sum(r["loops"] for r in results),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"SUMMARY questions={len(results)} contain={summary['contain_correct']} "
        f"exact={summary['normalized_exact_correct']} blanks={summary['blank_answers']} "
        f"calls={summary['usage']['policy_calls']} "
        f"tokens={summary['usage']['total_tokens']}"
    )


if __name__ == "__main__":
    main()
