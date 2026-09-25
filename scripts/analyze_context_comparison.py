"""Audit the two-stage smoke and report descriptive paired differences."""
import argparse
import json
from collections import Counter
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(root):
    stages, violations = {}, []
    for stage in ("stage1", "stage2"):
        rows = {}
        for dataset in ("hotpotqa", "novel", "medical"):
            progress = {r["episode_id"]: r for r in
                        (json.loads(line) for line in (root / stage / dataset / "progress.jsonl").read_text().splitlines())}
            for path in sorted((root / stage / dataset / "episodes").glob("*/episode.json")):
                e = load(path)
                condition = e["episode_id"].split("--", 1)[0]
                calls, executed, duplicates, longest, streak = Counter(), Counter(), 0, 0, 0
                queries, assessments, seen, old_refs = [], [], set(), set()
                failures = Counter()
                usage = {k: [] for k in ("input", "output", "total")}
                signatures = set()
                repeated = 0
                finalize = 0
                for index, step in enumerate(e["trajectory"]):
                    label = f"{stage}/{dataset}/{condition}/{index + 1}"
                    provider = step.get("provider_metadata", {})
                    observation = step.get("observation") or {}
                    category = provider.get("failure_category") or observation.get("metadata", {}).get("failure_category")
                    if category:
                        failures[category] += 1
                    is_duplicate = observation.get("error_code") == "duplicate_action"
                    duplicates += is_duplicate
                    streak = streak + 1 if is_duplicate else 0
                    longest = max(longest, streak)
                    finalize += bool(observation.get("metadata", {}).get("budget_finalize"))
                    available = {t["function"]["name"]: t["function"] for t in step.get("tool_definitions", [])}
                    allowed = {"find_passages", "finish"}
                    if condition in {"C1", "C4", "C5", "A1"}:
                        allowed.add("find_sentences")
                    if condition in {"C2", "C5"}:
                        allowed.add("follow_entity_to_passages")
                    if condition in {"C3", "C4"}:
                        allowed.add("follow_entity_to_sentences")
                    if set(available) - allowed:
                        violations.append(label + ":capability_leak")
                    if observation.get("metadata", {}).get("budget_finalize") and set(available) != {"finish"}:
                        violations.append(label + ":finalize_has_retrieval")
                    provider_calls = provider.get("raw_tool_calls") or []
                    if not provider_calls and isinstance(provider.get("raw_structured_decision"), dict):
                        structured_action = provider["raw_structured_decision"].get("action") or {}
                        provider_calls = [{"function": {"name": structured_action.get("name"),
                                                        "arguments": {k: v for k, v in structured_action.items() if k != "name"}}}]
                    for call in provider_calls:
                        fn = call.get("function", {})
                        name = fn.get("name")
                        calls[name] += 1
                        if step.get("validation_status") == "valid" and observation.get("status") == "ok":
                            executed[name] += 1
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except ValueError:
                                continue
                        if not isinstance(args, dict):
                            continue
                        args = {k: v for k, v in args.items() if k != "assessment"}
                        if name not in available and step.get("validation_status") == "valid":
                            violations.append(label + ":unavailable_tool_executed")
                        if args.get("entity_ref") and step.get("validation_status") == "valid":
                            enum = available.get(name, {}).get("parameters", {}).get("properties", {}).get("entity_ref", {}).get("enum", [])
                            if args["entity_ref"] not in enum:
                                violations.append(label + ":hidden_entity_executed")
                        signature = json.dumps([name, args], sort_keys=True)
                        repeated += signature in signatures
                        signatures.add(signature)
                        if "query" in args:
                            queries.append({"tool": name, "query": args["query"], "entity_ref": args.get("entity_ref")})
                    audit = step.get("context_audit", {})
                    spans = step.get("visible_source_spans", [])
                    keys = {(s["sentence_id"], s["start"], s["end"]) for s in spans}
                    new = {(s["sentence_id"], s["start"], s["end"]) for s in audit.get("newly_visible_source_spans", [])}
                    content = "\n".join(m.get("content") or "" for m in step["messages"])
                    if not seen.issubset(keys):
                        violations.append(label + ":lost_source_spans")
                    if new != keys - seen:
                        violations.append(label + ":incorrect_new_spans")
                    if any(s["text"] not in content for s in spans):
                        violations.append(label + ":span_missing_from_input")
                    newrefs, oldrefs = audit.get("new_section_references", []), audit.get("old_section_references", [])
                    if set(newrefs) & set(oldrefs) or len(newrefs + oldrefs) != len(set(newrefs + oldrefs)):
                        violations.append(label + ":duplicate_source_block")
                    refs = set((step.get("context_reference_map") or {}).get("typed_refs", {}))
                    if not old_refs.issubset(refs):
                        violations.append(label + ":lost_reference")
                    old_refs, seen = refs, keys
                    requested = stage == "stage2"
                    if audit.get("assessment_requested") != requested:
                        violations.append(label + ":assessment_mode_mismatch")
                    assessment = (step.get("decision") or {}).get("assessment")
                    if requested and step.get("validation_status") == "valid" and assessment is None:
                        violations.append(label + ":missing_assessment")
                    if not requested and assessment is not None:
                        violations.append(label + ":unexpected_assessment")
                    assessments.append({"turn": index + 1, "status": step.get("assessment_status"),
                                        "assessment": assessment, "action": (step.get("decision") or {}).get("action")})
                    for k in usage:
                        usage[k].append(provider.get(f"provider_{k}_tokens"))
                if finalize > 1 or len(e["trajectory"]) - finalize > 15:
                    violations.append(f"{stage}/{dataset}/{condition}:budget")
                key = f"{dataset}/{e['episode_id']}"
                row = dict(dataset=dataset, condition=condition, episode_id=e["episode_id"],
                           terminal=e["termination_reason"], policy_calls=len(e["trajectory"]),
                           duplicate_rejections=duplicates, repeated_tool_arguments=repeated,
                           longest_duplicate_streak=longest, failure_categories=dict(failures),
                           tool_uptake=dict(calls), executed_tools=dict(executed), queries=queries, unique_source_sentences=len(seen),
                           query_changes=sum(a["query"] != b["query"] for a,b in zip(queries, queries[1:])),
                           assessments=assessments, wall_time_seconds=progress[e["episode_id"]].get("wall_time_seconds"))
                for k, values in usage.items():
                    row[k + "_tokens"] = sum(values) if values and all(v is not None for v in values) else None
                rows[key] = row
            actual = {r['condition'] for r in rows.values() if r['dataset'] == dataset}
            if actual != {"C0", "C1", "C2", "C3", "C4", "C5", "A1"}:
                violations.append(f"{stage}/{dataset}:missing_conditions")
        stages[stage] = rows
    # Compare complete YAML configs with the sole intended flag removed.
    import yaml
    configs = [yaml.safe_load((root / s / "target_config.yaml").read_text()) for s in ("stage1", "stage2")]
    for c in configs:
        c["agent"].pop("require_evidence_assessment")
    if configs[0] != configs[1]:
        violations.append("target_configs_differ_beyond_assessment")
    snapshots = [load(root / s / "smoke_manifest.json") for s in ("stage1", "stage2")]
    if snapshots[0]["code_files"] != snapshots[1]["code_files"]:
        violations.append("code_snapshot_changed_between_stages")
    identities = [sorted((m["name"], m["digest"]) for m in snap["model_tags"]["models"]) for snap in snapshots]
    if identities[0] != identities[1]:
        violations.append("model_digests_changed_between_stages")
    for dataset in ("hotpotqa", "novel", "medical"):
        ma, mb = [load(root / s / dataset / "run_manifest.json") for s in ("stage1", "stage2")]
        for k in ("question_sha256", "substrate_manifest_sha256", "source_manifest_sha256", "seed", "conditions",
                  "skill_sha256", "renderer_sha256", "provider_protocol_sha256", "target_prompt_digest", "embedding_model_identity"):
            if ma[k] != mb[k]:
                violations.append(f"{dataset}:mismatched_{k}")
        schedules = [[json.loads(line)["episode_id"] for line in
                      (root/s/dataset/"progress.jsonl").read_text().splitlines()] for s in ("stage1", "stage2")]
        if schedules[0] != schedules[1]:
            violations.append(f"{dataset}:schedule_mismatch")
    pairs = []
    metrics = ("policy_calls", "duplicate_rejections", "repeated_tool_arguments", "longest_duplicate_streak",
               "query_changes", "unique_source_sentences", "input_tokens", "output_tokens", "total_tokens", "wall_time_seconds")
    for key in sorted(set(stages["stage1"]) | set(stages["stage2"])):
        a, b = stages["stage1"].get(key), stages["stage2"].get(key)
        if a is None or b is None:
            violations.append(key + ":missing_pair")
            continue
        pairs.append({"pair": key, "stage1": a, "stage2": b,
                      "delta_stage2_minus_stage1": {k: b[k]-a[k] if a[k] is not None and b[k] is not None else None for k in metrics}})
    aggregates = []
    for stage, rows in stages.items():
        for dataset in ("hotpotqa", "novel", "medical"):
            selected = [r for r in rows.values() if r["dataset"] == dataset]
            errors, attempted_tools, executed_tools = Counter(), Counter(), Counter()
            for row in selected:
                errors.update(row["failure_categories"])
                attempted_tools.update(row["tool_uptake"])
                executed_tools.update(row["executed_tools"])
            aggregate = {"stage": stage, "dataset": dataset, "episodes": len(selected),
                         "failure_categories": dict(errors), "attempted_tools": dict(attempted_tools),
                         "executed_tools": dict(executed_tools)}
            for key in ("policy_calls", "duplicate_rejections", "input_tokens", "output_tokens", "total_tokens", "wall_time_seconds"):
                values = [r[key] for r in selected]
                aggregate[key] = sum(values) if values and all(v is not None for v in values) else None
            aggregates.append(aggregate)
    return {"episodes": sum(map(len, stages.values())), "paired_count": len(pairs), "violations": violations,
            "status": "passed" if len(pairs) == 21 and not violations else "failed_or_incomplete",
            "dataset_aggregates": aggregates, "pairs": pairs,
            "metric_notes": {"tool_uptake": "raw attempted calls, including rejected calls; one decision can incorrectly contain multiple calls",
                             "executed_tools": "valid calls with an ok result; includes finish",
                             "query_changes": "literal changes across attempted queries, including local ranking queries and invalid calls",
                             "unique_source_sentences": "unique sentence/span keys actually present in policy inputs, not relevance or gold support"},
            "manual_assessment_review": "assistant qualitative reading in docs/context_assessment_v3.md; not independent human annotation or a semantic judge score",
            "limitations": ["One question per dataset; descriptive only", "Stage order and model warmup can affect time"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.run)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k,v in result.items() if k != "pairs"}, ensure_ascii=False))
