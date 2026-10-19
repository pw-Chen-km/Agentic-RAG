"""Shared CLI preflight/runtime wiring. Standalone entry executes only Phase 1."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arag_skillopt.meta_stage import MetaConfig
from arag_skillopt.phase1 import Phase1, Phase1Config
from arag_skillopt.phase1_runtime import (
    JUDGE_PROMPT,
    AnswerJudge,
    ARAGRollout,
    client,
    load_questions,
    make_tools,
)
from arag_skillopt.skill_rules import load_rules
from arag_skillopt.support import digest
from arag_skillopt.workflow_training import WorkflowTrainer
from arag_skillopt.training_profile import (
    PROFILES, agent_prompt, prompt_bundle, validate_profile,
    validate_runtime_guidance, validate_split_sources,
)

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(workflow=False):
    parser = argparse.ArgumentParser()
    default_config = "arag_workflow_meta.yaml" if workflow else "arag_phase1.yaml"
    parser.add_argument("--config", default=str(ROOT / "configs" / default_config))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--smoke-train",
        type=int,
        help="Use first N TRAIN rows in a separate smoke output; never test rows",
    )
    parser.add_argument("--smoke-validation", type=int, default=2)
    parser.add_argument("--output")
    parser.add_argument("--preflight-report", help="Save offline contract, prompts and tool schemas; no model inference")
    parser.add_argument(
        "--support-map",
        help="Evaluator-only mapping path override for preflight/review",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.support_map:
        cfg["data"]["support_map"] = args.support_map
    phase_cfg = Phase1Config(**cfg["phase1"])
    validate_profile(cfg, phase_cfg)
    split_audit = validate_split_sources(cfg)
    meta_cfg = MetaConfig(**cfg.get("meta", {})) if workflow else None
    train, val = (load_questions(cfg["data"][key]) for key in ("train", "validation"))
    test = load_questions(
        cfg["data"]["test"]
    )  # ID-only disjointness check; no rollout/evaluation.
    all_ids = [{q["id"] for q in rows} for rows in (train, val, test)]
    if any(
        len(ids) != len(rows) for ids, rows in zip(all_ids, (train, val, test))
    ) or any(all_ids[i] & all_ids[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("split IDs are duplicated or overlapping")
    arag_questions = {q["id"]: q for q in load_questions(cfg["data"]["questions"])}
    if any(arag_questions.get(q["id"]) != q for q in train + val):
        raise ValueError(
            "original A-RAG and selected train/validation questions do not agree"
        )
    if args.smoke_train is not None:
        if args.smoke_train <= 0 or args.smoke_validation <= 0 or not args.output:
            raise ValueError(
                "smoke requires positive sizes and an explicit separate --output"
            )
        train, val = train[: args.smoke_train], val[: args.smoke_validation]
    index = Path(cfg["data"]["index_dir"]) / "sentence_index.pkl"
    mapping = json.loads(Path(cfg["data"]["support_map"]).read_text())
    if mapping["chunks_sha256"] != digest(
        json.loads(Path(cfg["data"]["chunks_file"]).read_text())
    ):
        raise ValueError("support map belongs to a different chunks file")
    skill = load_rules(ROOT / cfg["skill_path"])
    prompts = prompt_bundle(ROOT, cfg, workflow)
    original = agent_prompt(ROOT, cfg)
    data_hashes = {
        k: file_hash(cfg["data"][k])
        for k in (
            "train",
            "validation",
            "test",
            "questions",
            "chunks_file",
            "support_map",
        )
    }
    data_hashes["index"] = file_hash(index)
    if cfg["data"].get("source_questions"):
        data_hashes["source_questions"] = file_hash(cfg["data"]["source_questions"])
    if cfg.get("experiment", {}).get("preparation_manifest"):
        data_hashes["preparation_manifest"] = file_hash(cfg["experiment"]["preparation_manifest"])
    if cfg.get("experiment", {}).get("ablation_reference"):
        data_hashes["ablation_reference_contract"] = file_hash(Path(cfg["experiment"]["ablation_reference"]) / "contract.json")
    for key in ("index_file", "entity_index_file"):
        if cfg.get("entity_hop", {}).get(key):
            data_hashes["entity_hop_" + key] = file_hash(cfg["entity_hop"][key])
    if cfg.get("experiment", {}).get("reviewed_test_manifest"):
        data_hashes["reviewed_test_manifest"] = file_hash(cfg["experiment"]["reviewed_test_manifest"])
    source_hashes = {
        str(p.relative_to(ROOT)): file_hash(p)
        for folder in ("src", "scripts", "prompts", "skills")
        for p in sorted((ROOT / folder).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("._")
    }
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    contract = {
        "settings": cfg,
        "data_hashes": data_hashes,
        "source_hashes": source_hashes,
        "git_head": git_head.stdout.strip() if git_head.returncode == 0 else None,
        "judge_prompt_hash": digest(JUDGE_PROMPT),
        "smoke": args.smoke_train is not None,
        "budget_tokenizer": "o200k_base",
        "agent_budget_tokenizer": "loaded_qwen_gguf_actual_ollama_template" if cfg["llm"].get("transport") == "ollama_exact" else "o200k_base_estimate",
        "agent_context_policy": {key: cfg["llm"].get(key) for key in (
            "transport", "context_length", "max_tokens", "input_token_limit",
            "context_safety_tokens", "require_complete_output")},
        "test_used_for_optimization": False,
        "analysis_protocol": phase_cfg.analysis_protocol,
        "agent_prompt_hash": digest(original),
        "optimizer_prompt_version": cfg.get("optimizer_prompt_version", "legacy"),
        "optimizer_prompts_hash": digest(prompts),
        "split_audit": split_audit,
        "config_file_sha256": file_hash(args.config),
    }
    tools = None
    if cfg.get("optimizer_prompt_version") in PROFILES or args.preflight_report:
        tools = make_tools(cfg)  # load existing indices; do not generate or embed a query
        schemas = tools.get_all_schemas()
        validate_runtime_guidance(cfg, schemas, original, skill)
        contract["tool_schemas_hash"] = digest(schemas)
        if args.preflight_report:
            from arag_skillopt.skill_rules import render_formatted_prompt
            path = Path(args.preflight_report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "preflight": "passed", "model_inference_calls": 0,
                "execute_requested": args.execute, "contract": contract,
                "agent_system_prompt": render_formatted_prompt(original, skill),
                "current_skill": skill, "tool_schemas": schemas,
                "optimizer_prompts": prompts,
                "answer_guidance": original[original.index("## When Answering"):],
            }, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "preflight": "passed",
                "train": len(train),
                "validation": len(val),
                "meta": bool(workflow and meta_cfg.enabled),
                "candidate_train_replay": bool(workflow and meta_cfg.enabled),
                "test_rollouts": 0,
                "execute": args.execute,
                "split_audit": split_audit,
                "optimizer_prompt_version": cfg.get("optimizer_prompt_version", "legacy"),
            },
            indent=2,
        )
    )
    if not args.execute:
        return
    # Each endpoint's model digest is pinned in the resume contract.
    digests = {}
    for name in ("llm", "optimizer", "judge"):
        spec = cfg[name]
        endpoints = (
            spec.get("urls")
            or spec.get("base_urls")
            or [spec.get("url") or spec["base_url"]]
        )
        verified = {}
        for endpoint in endpoints:
            endpoint = endpoint.removesuffix("/v1").rstrip("/")
            response = requests.get(endpoint + "/api/tags", timeout=15)
            response.raise_for_status()
            matches = [
                m for m in response.json()["models"] if m["name"] == spec["model"]
            ]
            if len(matches) != 1:
                raise ValueError(f"{name} configured model unavailable at {endpoint}")
            verified[endpoint] = matches[0]["digest"]
        if len(set(verified.values())) != 1 or (
            spec.get("expected_digest")
            and spec["expected_digest"] != next(iter(verified.values()))
        ):
            raise ValueError("replicas must use the exact configured model digest")
        digests[name] = verified
    contract["model_digests"] = digests
    tools = tools if tools is not None else make_tools(cfg)
    guidance = json.dumps(tools.get_all_schemas(), ensure_ascii=False)
    # Original fixed answer rules also appear once, not in every case.
    guidance += "\n" + original[original.index("## When Answering") :]
    runner = WorkflowTrainer if workflow else Phase1
    extras = {"meta_config": meta_cfg} if workflow else {}
    engine = runner(
        args.output or cfg["output"],
        skill,
        client(cfg["optimizer"]),
        ARAGRollout(cfg, tools, original),
        AnswerJudge(cfg["judge"]),
        prompts,
        guidance,
        phase_cfg,
        mapping,
        contract,
        **extras,
    )
    print(json.dumps(engine.run(train, val), indent=2))


if __name__ == "__main__":
    main()
