"""Summarize the stratified Qwen 27B smoke without pooling datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root: Path) -> dict:
    selection = read_json(root / "diagnostic_manifest.json")
    rows = []
    expected = selection["expected_episodes"]
    for dataset, dataset_info in selection["datasets"].items():
        kinds = {str(item["id"]): item["type"] for item in dataset_info["selected"]}
        progress_path = root / dataset / "progress.jsonl"
        progress = {}
        if progress_path.exists():
            for line in progress_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    progress[item["episode_id"]] = item
        for episode_path in sorted((root / dataset / "episodes").glob("*/episode.json")):
            episode = read_json(episode_path)
            episode_id = episode["episode_id"]
            condition, question_id = episode_id.split("--", 1)
            outcome = progress.get(episode_id, {})
            source_id = str(outcome.get("source_question_id") or question_id)
            kind = kinds.get(source_id)
            if kind is None:
                matches = [value for key, value in kinds.items() if question_id.startswith(key)]
                kind = matches[0] if len(matches) == 1 else "unmatched"
            attempted, executed, offered = Counter(), Counter(), Counter()
            failures = Counter()
            first_visible_spans = set()
            for turn in episode.get("trajectory", []):
                available = [entry["function"]["name"] for entry in turn.get("tool_definitions", [])]
                offered.update(available)
                provider = turn.get("provider_metadata") or {}
                selected = provider.get("selected_action")
                if selected:
                    attempted[selected] += 1
                observation = turn.get("observation") or {}
                if turn.get("validation_status") == "valid" and observation.get("status") == "ok" and selected:
                    executed[selected] += 1
                if turn.get("validation_status") == "invalid":
                    failures[provider.get("failure_category") or "state_invalid"] += 1
                if observation.get("error_code") == "duplicate_action":
                    failures["duplicate_action"] += 1
                for span in turn.get("visible_source_spans", []):
                    if span.get("seen_by_policy") and span.get("complete"):
                        first_visible_spans.add((span.get("sentence_id"), span.get("start"), span.get("end")))
            usage = episode.get("usage") or {}
            rows.append({
                "dataset": dataset, "question_type": kind, "question_id": question_id,
                "condition": condition, "episode_id": episode_id,
                "termination_reason": episode.get("termination_reason"),
                "policy_calls": len(episode.get("trajectory", [])),
                "attempted_actions": dict(attempted), "executed_actions": dict(executed),
                "offered_actions": dict(offered), "failures": dict(failures),
                "unique_visible_source_spans": len(first_visible_spans),
                "target_input_tokens": usage.get("input_tokens"),
                "target_output_tokens": usage.get("output_tokens"),
                "target_total_tokens": usage.get("total_tokens"),
                "exact": outcome.get("answer_correct_exact"),
                "contain": outcome.get("answer_correct_contain"),
                "support_recall": outcome.get("support_recall"),
                "complete_support": outcome.get("complete_support"),
            })
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["dataset"], row["question_type"])].append(row)
    aggregates = []
    for (dataset, kind), group in sorted(by_group.items()):
        attempted, executed, failures = Counter(), Counter(), Counter()
        for row in group:
            attempted.update(row["attempted_actions"])
            executed.update(row["executed_actions"])
            failures.update(row["failures"])
        aggregates.append({
            "dataset": dataset, "question_type": kind, "episodes": len(group),
            "conditions": sorted(row["condition"] for row in group),
            "attempted_actions": dict(attempted), "executed_actions": dict(executed),
            "failures": dict(failures),
            "policy_calls": sum(row["policy_calls"] for row in group),
            "target_total_tokens": sum(row["target_total_tokens"] or 0 for row in group),
        })
    return {
        "scope": "engineering diagnostic; one fixed-seed question per type, not an effect estimate",
        "model": selection["model"], "model_digest": selection["model_digest"],
        "expected_episodes": expected, "observed_episodes": len(rows),
        "status": "complete" if len(rows) == expected and all(
            row["termination_reason"] == "finish" for row in rows) else "incomplete_or_failed",
        "aggregates": aggregates, "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "observed_episodes", "expected_episodes", "aggregates")}, ensure_ascii=False))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
