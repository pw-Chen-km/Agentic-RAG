"""Descriptive paired comparison of the v5 and v6 21-episode smoke runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DATASETS = ("hotpotqa", "novel", "medical")
MATCH_FIELDS = ("dataset", "seed", "question_sha256", "source_manifest_sha256",
                "substrate_manifest_sha256", "embedding_model_identity", "conditions",
                "provider_protocol", "budget", "config_sha256")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_rows(root: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for dataset in DATASETS:
        progress = {row["episode_id"]: row for line in
                    (root / dataset / "progress.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip() for row in [json.loads(line)]}
        for path in sorted((root / dataset / "episodes").glob("*/episode.json")):
            episode = load(path)
            episode_id = episode["episode_id"]
            if episode_id not in progress:
                raise ValueError(f"episode lacks progress row: {dataset}/{episode_id}")
            tools = Counter()
            duplicate = 0
            new_spans = 0
            for step in episode["trajectory"]:
                raw = (step.get("provider_metadata") or {}).get("raw_structured_decision") or {}
                name = (raw.get("action") or {}).get("name")
                if name:
                    tools[name] += 1
                duplicate += (step.get("observation") or {}).get("error_code") == "duplicate_action"
                new_spans += len((step.get("context_audit") or {}).get("output_delivery", {}).get(
                    "newly_projected_source_spans", []))
            result = progress[episode_id]
            rows[dataset, episode_id] = {
                "dataset": dataset, "condition": result["condition"],
                "episode_id": episode_id, "question_id": result["question_id"],
                "termination_reason": episode["termination_reason"],
                "duplicate_rejections": duplicate, "tool_use": dict(tools),
                "new_source_sentence_spans": new_spans,
                "policy_calls": len(episode["trajectory"]),
                "input_tokens": episode["usage"].get("input_tokens"),
                "output_tokens": episode["usage"].get("output_tokens"),
                "total_tokens": episode["usage"].get("total_tokens"),
                "complete_support": result.get("complete_support"),
                "answer_correct_exact": result.get("answer_correct_exact"),
                "answer_correct_contain": result.get("answer_correct_contain"),
            }
        if len([key for key in rows if key[0] == dataset]) != 7:
            raise ValueError(f"expected seven episodes for {dataset}")
    return rows


def compare(before: Path, after: Path) -> dict:
    manifest_pairs = {}
    for dataset in DATASETS:
        old = load(before / dataset / "run_manifest.json")
        new = load(after / dataset / "run_manifest.json")
        changed = [field for field in MATCH_FIELDS if old.get(field) != new.get(field)]
        if changed:
            raise ValueError(f"unpaired manifests for {dataset}: {changed}")
        if old.get("renderer_sha256") == new.get("renderer_sha256"):
            raise ValueError(f"renderer did not change for {dataset}")
        manifest_pairs[dataset] = {"before_renderer": old["renderer_version"],
                                   "after_renderer": new["renderer_version"]}
    old_rows, new_rows = stage_rows(before), stage_rows(after)
    if old_rows.keys() != new_rows.keys() or len(old_rows) != 21:
        raise ValueError("smoke episodes are not paired one-to-one")
    paired = []
    totals = {stage: defaultdict(Counter) for stage in ("before", "after")}
    for key in sorted(old_rows):
        old, new = old_rows[key], new_rows[key]
        if old["question_id"] != new["question_id"] or old["condition"] != new["condition"]:
            raise ValueError(f"question or condition changed: {key}")
        paired.append({"dataset": key[0], "episode_id": key[1],
                       "question_id": old["question_id"], "condition": old["condition"],
                       "before": old, "after": new})
        for stage, row in (("before", old), ("after", new)):
            bucket = totals[stage][key[0]]
            bucket["episodes"] += 1
            for field in ("duplicate_rejections", "new_source_sentence_spans", "policy_calls",
                          "input_tokens", "output_tokens", "total_tokens"):
                if row[field] is not None:
                    bucket[field] += row[field]
            for tool, count in row["tool_use"].items():
                bucket["tool:" + tool] += count
            bucket["complete_support"] += row["complete_support"] is True
            bucket["answer_correct_exact"] += row["answer_correct_exact"] is True
            bucket["answer_correct_contain"] += row["answer_correct_contain"] is True
    return {"scope": "engineering smoke only; one question per dataset, no effect inference",
            "manifest_pairs": manifest_pairs,
            "totals": {stage: {ds: dict(counts) for ds, counts in groups.items()}
                       for stage, groups in totals.items()},
            "paired": paired}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    result = compare(args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
