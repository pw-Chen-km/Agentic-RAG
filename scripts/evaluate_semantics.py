"""Backfill GraphRAG-Benchmark semantic metrics for completed episodes."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from collections import defaultdict
from pathlib import Path

import yaml

from agentic_rag.evaluation.graphrag_bench import GraphRAGSemanticEvaluator, OllamaSemanticJudge
from agentic_rag.evaluation.question_identity import question_lookup
from agentic_rag.substrate.storage import EvaluationSidecars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True, help="runs/.../episodes")
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--judge-config", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.judge_config:
        raw = yaml.safe_load(args.judge_config.read_text(encoding="utf-8")) or {}
        args.model = raw.get("model", args.model); args.host = raw.get("host", args.host); args.embedding_model = raw.get("embedding_model", args.embedding_model)
    sidecars = EvaluationSidecars.open(args.substrate)
    questions = question_lookup(sidecars.benchmark_questions)
    judge = GraphRAGSemanticEvaluator(OllamaSemanticJudge(model=args.model, host=args.host, embedding_model=args.embedding_model))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.episodes.glob("*/episode.json")):
        if args.limit is not None and len(rows) >= args.limit:
            break
        episode = json.loads(path.read_text(encoding="utf-8"))
        episode_id = str(episode.get("episode_id") or path.parent.name)
        qid = episode_id.split("--", 1)[-1]
        question = questions.get(qid)
        if question is None:
            raise ValueError(f"unresolved or ambiguous episode question ID: {qid}")
        visible = [span for step in episode.get("trajectory", []) for span in step.get("visible_source_spans", []) if span.get("visible") is True]
        payload = judge.evaluate(question=asdict(question), answer=episode.get("answer"), contexts=[str(s.get("text") or s.get("source_text") or "") for s in visible], evidence=question.evidence)
        payload.update({"episode_id": episode_id, "question_id": qid, "condition": episode_id.split("--", 1)[0]})
        (args.output / f"{episode_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(payload)
    aggregate = defaultdict(list)
    for row in rows:
        aggregate[row["condition"]].append(row)
    def mean(name, values):
        nums = [float(item["metrics"][name]) for item in values if item.get("metrics", {}).get(name) is not None]
        return sum(nums) / len(nums) if nums else None
    summary = {condition: {**{metric: mean(metric, values) for metric in ("answer_correctness", "rouge_l", "coverage", "faithfulness", "context_relevancy", "evidence_recall")}, "judge_calls": sum(int(item.get("judge_usage", {}).get("calls") or 0) for item in values), "judge_input_tokens": sum(int(item.get("judge_usage", {}).get("input_tokens") or 0) for item in values), "judge_output_tokens": sum(int(item.get("judge_usage", {}).get("output_tokens") or 0) for item in values)} for condition, values in aggregate.items()}
    (args.output / "summary.json").write_text(json.dumps({"episodes": len(rows), "conditions": summary}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
