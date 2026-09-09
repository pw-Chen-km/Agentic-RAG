"""Check saved training episodes in all four views without generating answers.

With --tokenize, count fully assembled native analyst prompts using the already
loaded, pinned Ollama model. This tool does not load models or start training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path

from agentic_rag.agent.models import EpisodeResult
from agentic_rag.skillopt.adapter import _native_reflection_conversation
from agentic_rag.skillopt.reflection_budget import capture_native_prompt, prompt_messages, sha
from agentic_rag.skillopt.reflection_format import VERSION, compact_reflection_input, expand_reflection_input
from agentic_rag.skillopt.reflection_tokenizer import SshOllamaTokenCounter
from agentic_rag.skillopt.trainer import load_skillopt_config
from agentic_rag.skillopt.trajectory import (
    REFLECTION_SCHEMA_VERSION, TRAJECTORY_REPRESENTATIONS, build_reflection_input, build_training_reference_text,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--only-known-blockers", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.run_root / "configurations/execution_manifest.json").read_text())
    entries = {e["task_id"]: e for e in manifest["experiments"]}
    native = importlib.import_module("skillopt.gradient.reflect")
    code = Path(__file__).resolve().parents[1] / "src/agentic_rag/skillopt"
    config = load_skillopt_config(entries["train:medical:raw"]["lane_variants"][0]["skillopt_config"]["path"])
    counter = SshOllamaTokenCounter(config) if args.tokenize else None
    report = {"format": VERSION, "reflection_schema": REFLECTION_SCHEMA_VERSION,
              "generation_calls": 0, "source_run": str(args.run_root), "input_token_limit": 244736,
              "scope": "two known blocked episodes" if args.only_known_blockers else "all 80 saved Medical training episodes",
              "code_sha256": {p.name: digest(p) for p in [code / "trajectory.py", code / "adapter.py", code / "reflection_format.py"]},
              "core_prompt_sha256": {p.name: digest(p) for p in sorted((code / "prompts").glob("*.md"))},
              "rows": []}
    source_hashes = {}
    for source_arm, known in (("raw", "medical:benchmark_exact:q:000207"), ("organized", "medical:benchmark_exact:q:001172")):
        entry = entries[f"train:medical:{source_arm}"]
        items = {str(x["id"]): x for x in map(json.loads, (Path(entry["split_dir"]) / "train.jsonl").read_text().splitlines())}
        paths = sorted((Path(entry["training_output"]) / "steps/step_0001/rollout").rglob("rollout_result.json"))
        if len(paths) != 40:
            raise ValueError("Expected exactly forty saved training episodes per source arm")
        for source_path in paths:
            row = json.loads(source_path.read_text())
            if args.only_known_blockers and row["id"] != known:
                continue
            folder = source_path.parent
            old = json.loads((folder / "reflection_conversation.json").read_text())
            episode = EpisodeResult.model_validate_json((folder / "episode.json").read_text())
            skill = (folder / "skill.md").read_text()
            item = items[row["id"]]
            effective = json.loads((folder / "effective_config.json").read_text())
            for f in folder.iterdir():
                if f.is_file():
                    source_hashes[str(f)] = digest(f)
            restored_contexts, views, calls = [], {}, {}
            for arm in TRAJECTORY_REPRESENTATIONS:
                rendered, audit_manifest = build_reflection_input(
                    episode=episode, item=item, skill_content=skill, rollout_phase="train", rollout_split="train",
                    trajectory_representation=arm, evaluation=old["episode_outcome"]["evaluation"],
                    target_system_prompt=(folder / "target_system_prompt.txt").read_text(), effective_config=effective,
                )
                compact = compact_reflection_input(rendered)
                assert expand_reflection_input(compact) == rendered
                assert rendered["hidden_reference"] == old["hidden_reference"]
                if arm == source_arm:
                    expected = copy.deepcopy(old)
                    expected["schema_version"] = REFLECTION_SCHEMA_VERSION
                    assert expected == rendered, "Expanded source-arm audit changed beyond version"
                views[arm] = rendered
                trajectory = rendered["trajectory"]
                steps = next(trajectory[k] for k in ("raw_steps", "action_ledger", "abstract_steps") if k in trajectory)
                restored_contexts.append([s["decision_context"] for s in steps])
                target = args.output / source_arm / folder.name / arm
                native_dir = target / "predictions" / row["id"]
                write(native_dir / "conversation.json", _native_reflection_conversation(rendered))
                for prompt in ("target_system_prompt.txt", "target_user_prompt.txt"):
                    (native_dir / prompt).write_bytes((folder / prompt).read_bytes())
                enriched = {**row, "reference_text": build_training_reference_text(item)}
                kind = "succ" if row["hard"] else "fail"
                kwargs = {"edit_budget": 1, "system_prompt": (code / "prompts" / ("analyst_success.md" if kind == "succ" else "analyst_error.md")).read_text(),
                          "skill_aware_reflection": False, "update_mode": "patch"}
                call = capture_native_prompt(native, kind, skill, [enriched], str(target / "predictions"), kwargs)
                assert call["max_completion_tokens"] == 16384
                calls[arm] = call
                check = {"source_arm": source_arm, "episode_id": row["id"], "arm": arm,
                         "source_episode_sha256": digest(folder / "episode.json"),
                         "roundtrip_equal": True, "optimizer_prompt_sha256": sha(prompt_messages(call)),
                         "expanded_chars": len(json.dumps(rendered, ensure_ascii=False)),
                         "compact_chars": len(json.dumps(compact, ensure_ascii=False))}
                if counter:
                    check["tokenization"] = counter(prompt_messages(call))
                    check["input_tokens"] = check["tokenization"]["input_tokens"]
                    check["fits"] = check["input_tokens"] <= report["input_token_limit"]
                write(target / "reflection_manifest.json", audit_manifest)
                # Known failures retain complete before/after calls for review.
                if row["id"] == known:
                    write(target / "expanded_audit.json", rendered)
                    write(target / "checked_native_call.json", call)
                    if arm == source_arm:
                        legacy_dir = target / "legacy" / row["id"]
                        write(legacy_dir / "conversation.json", _native_reflection_conversation(old))
                        for prompt in ("target_system_prompt.txt", "target_user_prompt.txt"):
                            (legacy_dir / prompt).write_bytes((folder / prompt).read_bytes())
                        old_call = capture_native_prompt(native, kind, skill, [enriched], str(target / "legacy"), kwargs)
                        write(target / "legacy_native_call.json", old_call)
                        if counter:
                            check["old_tokenization"] = counter(prompt_messages(old_call))
                            check["old_input_tokens"] = check["old_tokenization"]["input_tokens"]
                report["rows"].append(check)
                write(args.output / "report.json", report)
                print(json.dumps({k: check.get(k) for k in ("source_arm", "episode_id", "arm", "input_tokens", "fits", "old_input_tokens")}), flush=True)
            assert all(c == restored_contexts[0] for c in restored_contexts)
            labels = views["organized_support_labels"]["trajectory"]["step_diagnostics"]
            abstract = views["progress_abstracted"]["trajectory"]["abstract_steps"]
            for a, b in zip(labels, abstract, strict=True):
                for key in ("information_progress", "supporting_fact_progress", "reference_progress"):
                    if key in a:
                        assert a[key] == b[key]
            # All four calls keep the same question, actual skill and hidden reference.
            for call in calls.values():
                assert skill in call["user"]
                assert item["question"] in call["user"]
                assert build_training_reference_text(item) in call["user"]
    assert all(digest(Path(path)) == before for path, before in source_hashes.items())
    report.update(source_files_unchanged=True, source_sha256=source_hashes,
                  shared_context_equal=True, third_fourth_progress_equal=True,
                  tested_inputs=len(report["rows"]), complete=True,
                  all_single_inputs_fit=all(r.get("fits", False) for r in report["rows"]) if counter else None)
    write(args.output / "report.json", report)


if __name__ == "__main__":
    main()
