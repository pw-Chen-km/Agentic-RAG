"""Run Initial/final Skill on held-out test AFTER the workflow completes."""

import argparse
import json
from pathlib import Path

import requests
import yaml
from run_phase1 import ROOT, file_hash

from arag_skillopt.checkpoint import load_checkpoint
from arag_skillopt.final_test import FinalTestRunner
from arag_skillopt.phase1_runtime import (
    AnswerJudge,
    ARAGRollout,
    load_questions,
    make_tools,
)
from arag_skillopt.phase1 import Phase1Config
from arag_skillopt.skill_rules import load_rules, skill_sha256
from arag_skillopt.support import digest
from arag_skillopt.training_profile import (
    PROFILES,
    agent_prompt,
    validate_profile,
    validate_runtime_guidance,
    validate_split_sources,
)


def training_data_path(cfg, key):
    """Resolve a sealed data hash using the same key-to-path map as run_phase1."""
    data_keys = {
        "train", "validation", "test", "questions", "chunks_file",
        "support_map", "source_questions",
    }
    if key in data_keys:
        path = cfg.get("data", {}).get(key)
    elif key == "index":
        index_dir = cfg.get("data", {}).get("index_dir")
        path = Path(index_dir) / "sentence_index.pkl" if index_dir else None
    elif key == "preparation_manifest":
        path = cfg.get("experiment", {}).get("preparation_manifest")
    elif key == "ablation_reference_contract":
        reference = cfg.get("experiment", {}).get("ablation_reference")
        path = Path(reference) / "contract.json" if reference else None
    elif key == "reviewed_test_manifest":
        path = cfg.get("experiment", {}).get("reviewed_test_manifest")
    elif key in {"entity_hop_index_file", "entity_hop_entity_index_file"}:
        hop_key = key.removeprefix("entity_hop_")
        path = cfg.get("entity_hop", {}).get(hop_key)
    elif key == "linear_hop_graph_index_file":
        path = cfg.get("linear_hop", {}).get("graph_index_file")
    else:
        raise ValueError("unknown training data hash key: " + key)
    if not path:
        raise ValueError("missing training data path for: " + key)
    return Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    training = Path(args.training_output).resolve()
    output = Path(args.output).resolve()
    if output == training or training in output.parents:
        raise ValueError("test must use a separate output outside training")
    summary = load_checkpoint(training / "summary.json")
    checkpoint = load_checkpoint(training / "checkpoint.json")
    contract = load_checkpoint(training / "contract.json")
    if (
        not summary
        or summary.get("status") != "complete"
        or not checkpoint
        or checkpoint.get("stage") != "complete"
        or checkpoint.get("status") != "complete"
    ):
        raise ValueError("training must complete before test inference")
    if cfg != contract["runtime"]["settings"]:
        raise ValueError("test must use the exact training runtime settings")
    validate_profile(cfg, Phase1Config(**cfg["phase1"]))
    for key, expected in contract["runtime"]["data_hashes"].items():
        path = training_data_path(cfg, key)
        if file_hash(path) != expected:
            raise ValueError("training/test data changed: " + key)
    for name, expected in contract["runtime"]["source_hashes"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("training/test source changed: " + name)
    initial = load_rules(ROOT / cfg["skill_path"])
    final = load_rules(training / "final_skill.json")
    original = agent_prompt(ROOT, cfg)
    if digest(original) != contract["runtime"]["agent_prompt_hash"]:
        raise ValueError("training/test Agent prompt changed")
    tools = None
    if cfg.get("optimizer_prompt_version") in PROFILES:
        validate_split_sources(cfg)
        tools = make_tools(cfg)
        schemas = tools.get_all_schemas()
        validate_runtime_guidance(cfg, schemas, original, initial)
        if digest(schemas) != contract["runtime"]["tool_schemas_hash"]:
            raise ValueError("training/test Agent tool schemas changed")
    if (
        skill_sha256(initial) != contract["initial_skill_hash"]
        or skill_sha256(final) != summary["final_skill_hash"]
        or skill_sha256(final) != checkpoint["active_skill_hash"]
    ):
        raise ValueError("sealed Skill hash mismatch")
    train, validation, test = [
        load_questions(cfg["data"][key]) for key in ("train", "validation", "test")
    ]
    ids = [{q["id"] for q in rows} for rows in (train, validation, test)]
    if any(len(i) != len(r) for i, r in zip(ids, (train, validation, test))) or any(
        ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("split IDs overlap or repeat")
    if contract["train_hash"] != digest(train) or contract["validation_hash"] != digest(
        validation
    ):
        raise ValueError("training split hash mismatch")
    print(
        json.dumps(
            {
                "preflight": "passed",
                "test_questions": len(test),
                "arms": ["initial", "optimized"],
                "execute": args.execute,
            },
            indent=2,
        )
    )
    if not args.execute:
        return
    for name in ("llm", "judge"):
        for endpoint, expected in contract["runtime"]["model_digests"][name].items():
            models = requests.get(endpoint + "/api/tags", timeout=15).json()["models"]
            matches = [m for m in models if m["name"] == cfg[name]["model"]]
            if len(matches) != 1 or matches[0]["digest"] != expected:
                raise ValueError("training/test model digest changed")
    tools = tools if tools is not None else make_tools(cfg)
    runtime = {
        "training_contract_hash": digest(contract),
        "model_digests": contract["runtime"]["model_digests"],
        "data_hashes": contract["runtime"]["data_hashes"],
        "source_hashes": contract["runtime"]["source_hashes"],
        "settings": cfg,
    }
    runner = FinalTestRunner(
        output,
        ARAGRollout(cfg, tools, original),
        AnswerJudge(cfg["judge"]),
        runtime,
        workers=cfg["phase1"]["episode_workers"],
        support_map=json.loads(Path(cfg["data"]["support_map"]).read_text()),
    )
    print(json.dumps(runner.run(test, initial, final, ids[0], ids[1]), indent=2))


if __name__ == "__main__":
    main()
