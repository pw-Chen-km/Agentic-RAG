#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from agentic_rag.evaluation import contain_accuracy, normalize_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentic-summary", type=Path, required=True)
    parser.add_argument("--arag-predictions", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--agentic-seconds", type=Path, required=True)
    parser.add_argument("--arag-seconds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def number(path: Path) -> float:
    return float(path.read_text(encoding="utf-8").strip().splitlines()[-1])


def aggregate(label: str, rows: list[dict], elapsed: float) -> dict:
    n = len(rows)
    def avg(key: str) -> float:
        values = [float(row.get(key) or 0) for row in rows]
        return mean(values) if values else 0.0
    def total(key: str) -> int:
        return int(sum(float(row.get(key) or 0) for row in rows))
    termination_reasons = Counter(str(row.get("termination_reason") or "unknown") for row in rows)
    return {
        "label": label,
        "question_count": n,
        "normalized_exact_correct": sum(int(row["normalized_exact"]) for row in rows),
        "normalized_exact_accuracy": sum(int(row["normalized_exact"]) for row in rows) / n,
        "contain_correct": sum(int(row["contain_acc"]) for row in rows),
        "contain_accuracy": sum(int(row["contain_acc"]) for row in rows) / n,
        "blank_answers": sum(not str(row.get("predicted_answer") or "").strip() for row in rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "termination_reasons": dict(sorted(termination_reasons.items())),
        "invalid_attempts_total": None if label == "A-RAG" else total("invalid_attempts"),
        "llm_calls_total": total("llm_calls"),
        "llm_calls_average": avg("llm_calls"),
        "input_tokens_total": total("input_tokens"),
        "output_tokens_total": total("output_tokens"),
        "total_tokens": total("total_tokens"),
        "total_tokens_average": avg("total_tokens"),
        "retrieved_tokens_total": total("retrieved_tokens"),
        "retrieved_tokens_average": avg("retrieved_tokens"),
        "steps_or_loops_total": total("steps_or_loops"),
        "steps_or_loops_average": avg("steps_or_loops"),
        "read_actions_or_chunks_total": total("read_actions_or_chunks"),
        "read_actions_or_chunks_average": avg("read_actions_or_chunks"),
        "elapsed_seconds": elapsed,
        "average_seconds_per_question": elapsed / n,
    }


def main() -> None:
    args = parse_args()
    sample = read_json(args.sample)
    sample_ids = [str(row["id"]) for row in sample]

    agentic_source = read_json(args.agentic_summary)
    agentic_rows = []
    for row in agentic_source["results"]:
        usage = row.get("usage") or {}
        agentic_rows.append({
            "id": str(row["id"]),
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "predicted_answer": row.get("predicted_answer") or "",
            "normalized_exact": int(row.get("normalized_exact") or 0),
            "contain_acc": int(row.get("contain_acc") or 0),
            "error": row.get("error_code"),
            "termination_reason": row.get("termination_reason"),
            "invalid_attempts": row.get("invalid_attempts") or 0,
            "llm_calls": usage.get("policy_calls") or 0,
            "input_tokens": usage.get("input_tokens") or 0,
            "output_tokens": usage.get("output_tokens") or 0,
            "total_tokens": usage.get("total_tokens") or 0,
            "retrieved_tokens": usage.get("retrieved_tokens") or 0,
            "steps_or_loops": row.get("environment_steps") or 0,
            "read_actions_or_chunks": (row.get("action_counts") or {}).get("READ", 0),
        })

    arag_raw = read_jsonl(args.arag_predictions)
    arag_rows = []
    for row in arag_raw:
        prediction = str(row.get("pred_answer") or "")
        gold = str(row.get("gold_answer") or "")
        arag_rows.append({
            "id": str(row.get("qid") or row.get("id")),
            "question": row["question"],
            "gold_answer": gold,
            "predicted_answer": prediction,
            "normalized_exact": int(normalize_answer(prediction) == normalize_answer(gold)),
            "contain_acc": contain_accuracy(prediction, gold),
            "error": row.get("error"),
            "termination_reason": "error" if row.get("error") else "completed",
            "llm_calls": row.get("llm_calls") or 0,
            "input_tokens": row.get("llm_input_tokens") or 0,
            "output_tokens": row.get("llm_output_tokens") or 0,
            "total_tokens": row.get("llm_total_tokens") or 0,
            "retrieved_tokens": row.get("total_retrieved_tokens") or 0,
            "steps_or_loops": row.get("loops") or 0,
            "read_actions_or_chunks": row.get("chunks_read_count") or 0,
        })

    agentic_ids = [row["id"] for row in agentic_rows]
    arag_ids = [row["id"] for row in arag_rows]
    if len(sample_ids) != 100 or len(set(sample_ids)) != 100:
        raise RuntimeError("sample does not contain 100 unique IDs")
    if set(agentic_ids) != set(sample_ids) or set(arag_ids) != set(sample_ids):
        raise RuntimeError("result IDs do not match the common sample")
    if len(agentic_ids) != 100 or len(arag_ids) != 100:
        raise RuntimeError("both systems must have exactly 100 results")

    agentic_by_id = {row["id"]: row for row in agentic_rows}
    arag_by_id = {row["id"]: row for row in arag_rows}
    ordered_agentic = [agentic_by_id[qid] for qid in sample_ids]
    ordered_arag = [arag_by_id[qid] for qid in sample_ids]
    systems = [
        aggregate("Agentic-RAG", ordered_agentic, number(args.agentic_seconds)),
        aggregate("A-RAG", ordered_arag, number(args.arag_seconds)),
    ]
    payload = {
        "contract": {
            "dataset": "hotpotqa",
            "question_count": 100,
            "question_types": {"bridge": 81, "comparison": 19},
            "model": "qwen3.6:35b-a3b-bf16",
            "ollama_base_url": "http://127.0.0.1:11435",
            "thinking": "disabled",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "workers": 1,
        },
        "systems": systems,
        "results": {"agentic_rag": ordered_agentic, "arag": ordered_arag},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Qwen 3.6 HotpotQA Sample-100 Comparison",
        "",
        "Both systems used the same 100 questions, Qwen 3.6 model, no-thinking mode, and MiniLM embeddings.",
        "",
        "| Metric | Agentic-RAG | A-RAG |",
        "|---|---:|---:|",
    ]
    metrics = [
        ("Normalized exact", "normalized_exact_accuracy", ".1%"),
        ("Contain accuracy", "contain_accuracy", ".1%"),
        ("Blank answers", "blank_answers", "d"),
        ("Errors", "errors", "d"),
        ("Invalid attempts", "invalid_attempts_total", "d"),
        ("LLM calls", "llm_calls_total", "d"),
        ("LLM calls/question", "llm_calls_average", ".2f"),
        ("Input tokens", "input_tokens_total", "d"),
        ("Output tokens", "output_tokens_total", "d"),
        ("Total tokens", "total_tokens", "d"),
        ("Tokens/question", "total_tokens_average", ".2f"),
        ("Retrieved tokens", "retrieved_tokens_total", "d"),
        ("Retrieved tokens/question", "retrieved_tokens_average", ".2f"),
        ("Steps/loops", "steps_or_loops_total", "d"),
        ("Steps/loops per question", "steps_or_loops_average", ".2f"),
        ("Read actions/chunks", "read_actions_or_chunks_total", "d"),
        ("Read actions/chunks per question", "read_actions_or_chunks_average", ".2f"),
        ("Elapsed seconds", "elapsed_seconds", ".1f"),
        ("Seconds/question", "average_seconds_per_question", ".2f"),
    ]
    for title, key, fmt in metrics:
        values = ["N/A" if system[key] is None else format(system[key], fmt) for system in systems]
        lines.append(f"| {title} | {values[0]} | {values[1]} |")
    lines.extend([
        "",
        "## Termination reasons",
        "",
        f"- Agentic-RAG: `{json.dumps(systems[0]['termination_reasons'], sort_keys=True)}`",
        f"- A-RAG: `{json.dumps(systems[1]['termination_reasons'], sort_keys=True)}`",
        "",
        "See `summary.json` for per-question outputs and detailed token/retrieval metrics.",
        "",
    ])
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "systems": systems}, ensure_ascii=False))


if __name__ == "__main__":
    main()
