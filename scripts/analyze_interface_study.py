"""Recompute configuration summaries from immutable pilot progress."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from agentic_rag.evaluation.interface_study import aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/interface-study-v1"))
    args = parser.parse_args()
    progress = args.run / "progress.jsonl"
    rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("condition", "unknown")].append(row)
    summary = {
        "run_manifest": json.loads((args.run / "run_manifest.json").read_text(encoding="utf-8")),
        "conditions": {name: aggregate(values) for name, values in sorted(groups.items())},
        "episode_count": len(rows),
    }
    by_question = defaultdict(dict)
    for row in rows:
        by_question[row.get("question_id")][row.get("condition")] = row
    cases = {"benefited": [], "harmed": [], "no_improvement": []}
    for question_id, values in by_question.items():
        baseline = values.get("C0")
        if baseline is None:
            continue
        for condition, row in values.items():
            if condition == "C0":
                continue
            baseline_score = (bool(baseline.get("answer_correct_exact")), float(baseline.get("support_recall") or 0))
            score = (bool(row.get("answer_correct_exact")), float(row.get("support_recall") or 0))
            bucket = "benefited" if score > baseline_score else "harmed" if score < baseline_score else "no_improvement"
            cases[bucket].append({"question_id": question_id, "condition": condition})
    summary["trajectory_cases"] = cases
    semantic_dir = args.run / "semantic_evaluations"
    semantic_rows = []
    if semantic_dir.exists():
        for path in semantic_dir.glob("*.json"):
            if path.name == "summary.json":
                continue
            semantic_rows.append(json.loads(path.read_text(encoding="utf-8")))
    semantic_groups = defaultdict(list)
    for row in semantic_rows:
        semantic_groups[row.get("condition", "unknown")].append(row)
    metrics = ("answer_correctness", "rouge_l", "coverage", "faithfulness", "context_relevancy", "evidence_recall")
    summary["semantic_evaluation"] = {
        condition: {
            metric: (sum(float(item.get("metrics", {}).get(metric)) for item in values if item.get("metrics", {}).get(metric) is not None) / len([item for item in values if item.get("metrics", {}).get(metric) is not None]) if any(item.get("metrics", {}).get(metric) is not None for item in values) else None)
            for metric in metrics
        }
        for condition, values in sorted(semantic_groups.items())
    }
    (args.run / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["conditions"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
