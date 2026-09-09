"""Offline configuration bundles and deliberately gated final-test execution.

Configuration generation, validation, dry runs, and completion sealing do not
instantiate a model client.  Actual test inference needs ``execute=True`` and
four successfully completed, identity-checked training runs for that dataset.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import yaml

from agentic_rag.skillopt.multidataset_prepare import validate_prepared_split


CONFIG_BUNDLE_VERSION = "skillopt-multidataset-execution-v1"
CROSS_DATASET_BUNDLE_VERSION = "skillopt-single-source-transfer-execution-v1"
CROSS_DATASET_DESIGN = "single_source_cross_dataset_v1"
LEGACY_DESIGN = "independent_five_dataset_v1"
COMPLETION_VERSION = "skillopt-multidataset-completion-v1"
TRAINING_DATASETS = ("hotpotqa", "medical", "novel")
DATASETS = ("hotpotqa", "2wikimultihop", "musique", "medical", "novel")
REPRESENTATIONS = ("raw", "organized", "organized_support_labels", "progress_abstracted")
MODEL = "qwen3.6:35b-a3b-bf16"
MODEL_DIGEST = "94061ddd23a7de9c902f7bc468455aff03564d26e929b7e2e9ce62c5e6a3a492"
_REFLECTION_TOKEN_BUDGET = {
    "enabled": True, "context_tokens": 262144, "output_reserve_tokens": 16384,
    "safety_margin_tokens": 1024, "input_token_limit": 244736,
    "max_minibatch_size": 5, "oversized_single_trajectory": "stop_without_truncation",
}
_ROOT = Path(__file__).resolve().parents[3]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _hash(path), "size_bytes": path.stat().st_size}


def _check(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file() or _hash(path) != record.get("sha256") or path.stat().st_size != record.get("size_bytes"):
        raise ValueError(f"Pinned file changed or is missing: {path}")
    return path


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace a different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode())


def _write_yaml(path: Path, value: Any) -> None:
    _write_bytes(path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False).encode())


def _source_files() -> list[dict[str, Any]]:
    package = _ROOT / "src" / "agentic_rag"
    paths = list(package.rglob("*.py"))
    paths += [_ROOT / "scripts" / name for name in (
        "run_local_qwen_skillopt.py", "write_multidataset_skillopt_configs.py", "run_prepared_skillopt_test.py",
        "run_multidataset_experiments.py", "run_skillopt_capacity.py")]
    return [_record(path) for path in sorted(paths) if path.is_file()]


def _git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _counts(manifest: Mapping[str, Any] | None) -> dict[str, int | None]:
    if manifest is None:
        return {split: None for split in ("train", "validation", "test")}
    counts = {}
    for split in ("train", "validation", "test"):
        value = manifest.get("splits", {}).get(split, {}).get("count")
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid actual split count: {split}")
        counts[split] = value
    return counts


def _targets(report: Mapping[str, Any]) -> dict[str, int | None]:
    count = report.get("eligible_count")
    if type(count) is not int:
        return {split: None for split in ("train", "validation", "test")}
    size = count // 5
    return {"train": size, "validation": size, "test": count - 2 * size}


def write_experiment_configs(
    prepared_root: str | Path,
    initial_skill: str | Path,
    optimizer_config: str | Path,
    agent_config: str | Path,
    output: str | Path,
    *,
    design: str = LEGACY_DESIGN,
    lanes: Sequence[Mapping[str, Any]] | None = None,
    model_digest: str = MODEL_DIGEST,
    reflection_minibatch_size: int = 8,
    reflection_token_budget_enabled: bool = False,
) -> dict[str, Any]:
    """Write pinned configs, including clearly non-executable blockers.

    The explicit reflection override applies only to the cross-dataset design;
    legacy generation retains its original fixed settings.
    """
    if type(reflection_minibatch_size) is not int or not 1 <= reflection_minibatch_size <= 40:
        raise ValueError("reflection_minibatch_size must be an integer from 1 to the rollout batch size 40")
    if type(reflection_token_budget_enabled) is not bool:
        raise ValueError("reflection_token_budget_enabled must be an explicit boolean")
    if reflection_token_budget_enabled and (design != CROSS_DATASET_DESIGN or reflection_minibatch_size != 5):
        raise ValueError("Token-budget reflection requires the cross-dataset design with reflection_minibatch_size=5")
    if design == CROSS_DATASET_DESIGN:
        return _write_cross_dataset_configs(
            prepared_root, initial_skill, optimizer_config, agent_config, output,
            lanes=lanes, model_digest=model_digest, reflection_minibatch_size=reflection_minibatch_size,
            reflection_token_budget_enabled=reflection_token_budget_enabled,
        )
    if design != LEGACY_DESIGN:
        raise ValueError(f"Unsupported experiment design: {design}")
    if lanes is not None:
        raise ValueError("Lane variants are only supported by the explicit cross-dataset design")
    if reflection_minibatch_size != 8:
        raise ValueError("The reflection_minibatch_size override requires the cross-dataset design")
    prepared, destination = Path(prepared_root).resolve(), Path(output).resolve()
    skill, optimizer_source, agent_source = map(lambda p: Path(p).resolve(), (initial_skill, optimizer_config, agent_config))
    if not skill.read_text(encoding="utf-8").strip():
        raise ValueError("Initial Skill must be non-empty")
    optimizer = yaml.safe_load(optimizer_source.read_text(encoding="utf-8"))
    agent = yaml.safe_load(agent_source.read_text(encoding="utf-8"))
    if not isinstance(optimizer, dict) or not isinstance(agent, dict):
        raise ValueError("Both input configs must contain YAML mappings")
    policy = agent.get("policy", {})
    if policy.get("provider") != "ollama" or policy.get("model") != MODEL or policy.get("think") is not False or policy.get("temperature") != 0:
        raise ValueError("Target config must retain Ollama Qwen 3.6, think=false, temperature=0")
    model = optimizer.get("model", {})
    if model.get("optimizer") != MODEL or model.get("target") != MODEL:
        raise ValueError("Optimizer and target must use the pinned Qwen 3.6 model")
    if model.get("optimizer_qwen_chat_enable_thinking", model.get("qwen_chat_enable_thinking")) is not True:
        raise ValueError("Retain the existing explicitly thinking-enabled Optimizer")
    model_host = str(model.get("optimizer_qwen_chat_base_url", model.get("qwen_chat_base_url", ""))).rstrip("/")
    if model_host != str(policy.get("host", "")).rstrip("/") + "/v1":
        raise ValueError("Optimizer and Target must use the same endpoint")
    shared_agent = copy.deepcopy(agent)
    shared_agent.setdefault("agent", {}).update(max_steps=10, max_policy_attempts=12, max_retrieved_tokens=12000)
    shared_agent["agent"].setdefault("show_available_action_options", True)
    shared_agent["agent"]["use_state_conditioned_schema"] = True
    archived_skill = destination / "initial_skill.md"
    shared_agent_path = destination / "agent.yaml"
    _write_bytes(archived_skill, skill.read_bytes())
    _write_yaml(shared_agent_path, shared_agent)
    manifest_path = destination / "execution_manifest.json"
    python = str(_ROOT / ".venv" / "bin" / "python")
    trainer_script = str(_ROOT / "scripts" / "run_local_qwen_skillopt.py")
    test_script = str(_ROOT / "scripts" / "run_prepared_skillopt_test.py")
    entries = []
    dataset_reports = {}
    for dataset in DATASETS:
        split_dir = prepared / dataset
        readiness_path = split_dir / "readiness.json"
        readiness = _load(readiness_path) if readiness_path.is_file() else {
            "dataset": dataset, "split_status": "blocked", "all_four_arms_ready": False,
            "error": {"code": "missing_preparation", "message": "Dataset preparation has not produced a readiness report"}}
        split_manifest_path = split_dir / "split_manifest.json"
        split_manifest = _load(split_manifest_path) if split_manifest_path.is_file() else None
        counts = _counts(split_manifest)
        substrate = (split_manifest or {}).get("dataset", {}).get("substrate", {}).get("path")
        ready = bool(readiness.get("split_status") == "ready" and readiness.get("all_four_arms_ready") and split_manifest and substrate)
        if ready:
            validate_prepared_split(split_dir, substrate, require_reference_progress=True)
        dataset_reports[dataset] = {
            "ready": ready, "actual_counts": counts, "target_counts": _targets(readiness),
            "readiness": _record(readiness_path) if readiness_path.is_file() else None,
            "split_manifest": _record(split_manifest_path) if split_manifest_path.is_file() else None,
            "reason": None if ready else readiness.get("split_error") or readiness.get("error") or "reference_progress_unavailable",
        }
        for representation in REPRESENTATIONS:
            config = copy.deepcopy(optimizer)
            run_root = destination / "training_runs" / dataset / representation
            config["run_name"] = f"{dataset}_full_pool_{representation}"
            config.setdefault("train", {}).update(train_size=counts["train"], batch_size=20, accumulation=1, num_epochs=1, seed=42)
            config.setdefault("gradient", {})["minibatch_size"] = 5
            config.setdefault("evaluation", {}).update(sel_env_num=counts["validation"], test_env_num=counts["test"], eval_test=False)
            config.setdefault("env", {}).update(name=f"agentic_rag_{dataset}", workflow_label="full_pool_trajectory_representation_ablation")
            config_path = destination / dataset / f"{representation}.yaml"
            _write_yaml(config_path, config)
            argv = [python, trainer_script, "--substrate", str(substrate or "UNAVAILABLE_SUBSTRATE"), "--split-dir", str(split_dir), "--agent-config", str(shared_agent_path), "--skillopt-config", str(config_path), "--skill", str(archived_skill), "--output", str(run_root), "--trajectory-representation", representation]
            entries.append({
                "dataset": dataset, "representation": representation, "ready": ready,
                "actual_counts": counts, "target_counts": _targets(readiness),
                "skillopt_config": _record(config_path), "substrate": substrate,
                "split_dir": str(split_dir), "training_output": str(run_root),
                "completion_path": str(run_root / "training_completion.json"),
                "train_argv": argv, "train_command": shlex.join(argv),
                "train_command_executable": ready,
                "maximum_skill_updates": math.ceil(counts["train"] / 20) if counts["train"] is not None else None,
                "candidate_validation_episode_upper_bound": math.ceil(counts["train"] / 20) * counts["validation"] if counts["train"] is not None else None,
                "seal_completion_argv": [python, test_script, "--manifest", str(manifest_path), "--dataset", dataset, "--arm", representation, "--seal-completion"],
            })
    final_commands = []
    for dataset in DATASETS:
        for arm in ("initial", *REPRESENTATIONS):
            argv = [python, test_script, "--manifest", str(manifest_path), "--dataset", dataset, "--arm", arm]
            final_commands.append({"dataset": dataset, "arm": arm, "dry_run_argv": argv,
                                   "execute_argv_after_all_four_complete": argv + ["--execute"],
                                   "output": str(destination / "final_test" / dataset / arm)})
    core_prompts = [_record(path) for path in sorted((Path(__file__).parent / "prompts").glob("*.md"))]
    result = {
        "version": CONFIG_BUNDLE_VERSION, "prepared_root": str(prepared),
        "output_root": str(destination), "initial_skill": _record(archived_skill),
        "agent_config": _record(shared_agent_path), "source_initial_skill": _record(skill),
        "source_optimizer_config": _record(optimizer_source), "source_agent_config": _record(agent_source),
        "git_revision": _git_revision(), "code_files": _source_files(), "core_prompt_files": core_prompts,
        "target_model": MODEL, "optimizer_model": MODEL, "target_thinking": False,
        "optimizer_thinking": True, "train_batch_size": 20, "reflection_minibatch_size": 5,
        "num_epochs": 1, "eval_test_during_training": False,
        "datasets": dataset_reports, "experiments": entries, "final_test_commands": final_commands,
        "model_calls": 0, "final_test_gate": "all_four_training_runs_completed_and_sealed",
        "completion_instructions": "After a runner exits successfully and writes workflow_summary.json, run that arm's seal_completion_argv. Final test --execute requires all four sealed completions. Default test invocation only validates/dry-runs and never opens a model client.",
    }
    _write_json(manifest_path, result)
    return result


def _cross_bundle(manifest_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(manifest_path).resolve()
    bundle = _load(path)
    if bundle.get("version") != CROSS_DATASET_BUNDLE_VERSION or bundle.get("design") != CROSS_DATASET_DESIGN:
        raise ValueError("Invalid single-source cross-dataset bundle")
    if bundle.get("eval_test_during_training") is not False:
        raise ValueError("Execution bundle permits test leakage during training")
    minibatch = bundle.get("reflection_minibatch_size")
    if type(minibatch) is not int or not 1 <= minibatch <= 40:
        raise ValueError("Invalid frozen reflection_minibatch_size")
    _validate_reflection_token_budget(bundle)
    for name in ("initial_skill", "source_optimizer_config", "source_agent_config"):
        _check(bundle[name])
    for record in bundle["code_files"] + bundle["core_prompt_files"]:
        _check(record)
    for lane in bundle["lanes"]:
        _check(lane["agent_config"])
    return path, bundle


def _cross_entry(bundle: Mapping[str, Any], dataset: str, representation: str) -> dict[str, Any]:
    if dataset not in TRAINING_DATASETS or representation not in REPRESENTATIONS:
        raise ValueError("Unknown source training dataset or representation")
    found = [entry for entry in bundle["experiments"]
             if entry["source_dataset"] == dataset and entry["representation"] == representation]
    if len(found) != 1 or not found[0].get("ready"):
        raise ValueError("Source training task is missing or not ready")
    entry = found[0]
    report = bundle["datasets"][dataset]
    _check(report["readiness"])
    _check(report["split_manifest"])
    for variant in entry["lane_variants"]:
        _check(variant["skillopt_config"])
        _check(variant["agent_config"])
        config = yaml.safe_load(Path(variant["skillopt_config"]["path"]).read_text(encoding="utf-8"))
        if config.get("gradient", {}).get("minibatch_size") != bundle["reflection_minibatch_size"]:
            raise ValueError("Pinned reflection minibatch differs from the execution manifest")
        if _validate_reflection_token_budget(bundle):
            expected = _reflection_budget_env(variant)
            if any(config.get("env", {}).get(key) != value for key, value in expected.items()):
                raise ValueError("Pinned reflection token budget or tokenizer lane differs from the execution manifest")
            if any(config.get("model", {}).get(key) != 16384 for key in
                   ("qwen_chat_max_tokens", "optimizer_qwen_chat_max_tokens")):
                raise ValueError("Optimizer output limit differs from its reserved reflection budget")
        elif config.get("env", {}).get("reflection_token_budget_enabled"):
            raise ValueError("Pinned config enables adaptive reflection without manifest opt-in")
    return entry


def _cross_start_expected(
    bundle: Mapping[str, Any], entry: Mapping[str, Any], variant: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "dataset": entry["source_dataset"], "trajectory_representation": entry["representation"],
        "split_manifest_sha256": bundle["datasets"][entry["source_dataset"]]["split_manifest"]["sha256"],
        "skillopt_config_sha256": variant["skillopt_config"]["sha256"],
        "agent_config_sha256": variant["agent_config"]["sha256"],
        "initial_skill_sha256": bundle["initial_skill"]["sha256"],
    }


def _check_completed_training_artifacts(run_root: Path, dataset: str, question_count: int) -> tuple[Path, Path, Path]:
    summary_path, integration_path = run_root / "workflow_summary.json", run_root / "skillopt_integration.json"
    summary = _load(summary_path)
    if summary.get("dataset") != dataset or not isinstance(summary.get("trainer_summary"), dict):
        raise ValueError("Missing successful post-training workflow summary")
    if summary.get("unique_question_count") != question_count:
        raise ValueError("Completed source run used a different question pool")
    integration = _load(integration_path)
    if not integration.get("code_sha256") or not integration.get("prompt_sha256"):
        raise ValueError("Training integration contract is incomplete")
    for name, digest in integration["code_sha256"].items():
        current = Path(__file__).parent / name
        if not current.is_file() or _hash(current) != digest:
            raise ValueError(f"Completed training code differs: {name}")
    for name, digest in integration["prompt_sha256"].items():
        current = Path(__file__).parent / "prompts" / name
        if not current.is_file() or _hash(current) != digest:
            raise ValueError(f"Completed training prompt differs: {name}")
    best_path = run_root / "best_skill.md"
    if not best_path.read_text(encoding="utf-8").strip():
        raise ValueError("Completed source Best Skill is empty")
    return summary_path, integration_path, best_path


def _seal_cross_training_completion(manifest_path: str | Path, dataset: str, representation: str) -> dict[str, Any]:
    path, bundle = _cross_bundle(manifest_path)
    entry = _cross_entry(bundle, dataset, representation)
    validate_prepared_split(entry["split_dir"], entry["substrate"], require_reference_progress=True)
    run_root = Path(entry["training_output"])
    start_path = run_root / "prepared_training_contract.json"
    start = _load(start_path)
    variants = [variant for variant in entry["lane_variants"]
                if all(start.get(key) == value for key, value in _cross_start_expected(bundle, entry, variant).items())]
    if len(variants) != 1:
        raise ValueError("Training startup contract does not identify exactly one pinned lane variant")
    variant = variants[0]
    summary_path, integration_path, best = _check_completed_training_artifacts(
        run_root, dataset, sum(entry["actual_counts"].values()),
    )
    result = {
        "version": COMPLETION_VERSION, "design": CROSS_DATASET_DESIGN, "status": "complete",
        **_cross_start_expected(bundle, entry, variant),
        "source_dataset": dataset, "lane_id": variant["lane_id"], "endpoint": variant["host"],
        "expected_model_digest": bundle["expected_model_digest"],
        "execution_manifest": _record(path), "startup_contract": _record(start_path),
        "workflow_summary": _record(summary_path), "integration_contract": _record(integration_path),
        "best_skill": _record(best), "test_metrics_used_for_sealing": False,
    }
    _write_json(Path(entry["completion_path"]), result)
    return result


def _cross_completions(
    path: Path, bundle: Mapping[str, Any], *, require_completed: bool,
) -> dict[str, dict[str, Any]]:
    completions = {}
    for source in TRAINING_DATASETS:
        for representation in REPRESENTATIONS:
            entry = _cross_entry(bundle, source, representation)
            completion_path = Path(entry["completion_path"])
            if not completion_path.exists():
                if require_completed:
                    raise ValueError(f"All twelve training runs must complete before ANY final test: missing {source}/{representation}")
                continue
            completion = _load(completion_path)
            if (completion.get("version") != COMPLETION_VERSION or completion.get("design") != CROSS_DATASET_DESIGN
                    or completion.get("status") != "complete" or completion.get("source_dataset") != source
                    or completion.get("trajectory_representation") != representation):
                raise ValueError("Invalid source training completion seal")
            variants = [variant for variant in entry["lane_variants"] if variant["lane_id"] == completion.get("lane_id")]
            if len(variants) != 1:
                raise ValueError("Completion uses an unknown lane")
            expected = _cross_start_expected(bundle, entry, variants[0])
            if any(completion.get(key) != value for key, value in expected.items()):
                raise ValueError("Completion does not match its source Skill, split, or lane configuration")
            if (completion.get("execution_manifest", {}).get("sha256") != _hash(path)
                    or completion.get("expected_model_digest") != bundle["expected_model_digest"]):
                raise ValueError("Completion belongs to another bundle or model digest contract")
            for name in ("execution_manifest", "startup_contract", "workflow_summary", "integration_contract", "best_skill"):
                _check(completion[name])
            if Path(completion["best_skill"]["path"]).resolve() != Path(entry["training_output"]) / "best_skill.md":
                raise ValueError("Completion points to another source run's Best Skill")
            completions[f"{source}/{representation}"] = completion
    return completions


def _validate_cross_final_test_request(
    manifest_path: str | Path, target_dataset: str, arm: str, *, require_completed: bool,
    source_dataset: str | None, lane_id: str | None,
) -> dict[str, Any]:
    path, bundle = _cross_bundle(manifest_path)
    if target_dataset not in DATASETS or arm not in ("initial", *REPRESENTATIONS):
        raise ValueError("Unknown target dataset or final-test arm")
    if arm == "initial" and source_dataset is not None:
        raise ValueError("Initial baseline has no source training dataset and runs only once per target")
    if arm != "initial" and source_dataset not in TRAINING_DATASETS:
        raise ValueError("A trained Skill requires an explicit source training dataset")
    target = bundle["datasets"][target_dataset]
    if not target.get("inference_ready"):
        raise ValueError("Target dataset is not ready for answer-only inference")
    _check(target["readiness"])
    _check(target["split_manifest"])
    split_validation = validate_prepared_split(target["split_dir"], target["substrate"], require_reference_progress=False)
    completions = _cross_completions(path, bundle, require_completed=require_completed)
    if lane_id is None and require_completed:
        raise ValueError("Final inference requires an explicit scheduler-approved --lane-id")
    candidates = [lane for lane in bundle["lanes"] if lane_id is None or lane["lane_id"] == lane_id]
    if not candidates:
        raise ValueError("Unknown final-test lane")
    lane = candidates[0]
    selected_completion = completions.get(f"{source_dataset}/{arm}") if source_dataset else None
    selected_skill = bundle["initial_skill"] if arm == "initial" else (
        selected_completion["best_skill"] if selected_completion else None)
    tasks = [task for task in bundle["final_test_commands"] if task["target_dataset"] == target_dataset
             and task["source_dataset"] == source_dataset and task["arm"] == arm]
    if len(tasks) != 1:
        raise ValueError("Missing or duplicate final-test matrix cell")
    task = tasks[0]
    return {
        "version": CROSS_DATASET_BUNDLE_VERSION, "design": CROSS_DATASET_DESIGN,
        "task_id": task["task_id"], "dataset": target_dataset, "target_dataset": target_dataset,
        "source_dataset": source_dataset, "arm": arm, "execution_manifest": _record(path),
        "selected_skill": selected_skill,
        "source_training_completion": selected_completion,
        "target_split_manifest": target["split_manifest"],
        "agent_config": lane["agent_config"], "lane_id": lane["lane_id"], "endpoint": lane["host"],
        "expected_model_digest": bundle["expected_model_digest"],
        "substrate": target["substrate"], "split_dir": target["split_dir"],
        "split_validation": split_validation, "output": task["output"], "test_count": task["test_count"],
        "completed_training_runs": sorted(completions), "all_twelve_completed": len(completions) == 12,
        "reference_progress_required": False, "model_calls": 0, "dry_run": not require_completed,
    }


def _bundle(manifest_path: str | Path, dataset: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(manifest_path).resolve()
    manifest = _load(path)
    if manifest.get("version") != CONFIG_BUNDLE_VERSION or dataset not in DATASETS:
        raise ValueError("Invalid execution bundle or dataset")
    if manifest.get("eval_test_during_training") is not False:
        raise ValueError("Execution bundle permits test leakage during training")
    for name in ("initial_skill", "agent_config", "source_optimizer_config", "source_agent_config"):
        _check(manifest[name])
    for record in manifest["code_files"] + manifest["core_prompt_files"]:
        _check(record)
    report = manifest["datasets"][dataset]
    if not report.get("ready"):
        raise ValueError(f"Dataset is not ready for a four-arm comparison: {dataset}: {report.get('reason')}")
    _check(report["readiness"])
    _check(report["split_manifest"])
    return path, manifest, report


def _entry(manifest: Mapping[str, Any], dataset: str, representation: str) -> dict[str, Any]:
    found = [entry for entry in manifest["experiments"] if entry["dataset"] == dataset and entry["representation"] == representation]
    if len(found) != 1 or not found[0].get("ready"):
        raise ValueError("Missing or blocked experiment configuration")
    entry = found[0]
    _check(entry["skillopt_config"])
    return entry


def seal_training_completion(manifest_path: str | Path, dataset: str, representation: str) -> dict[str, Any]:
    """Offline sealing of a successful run; no inference or test metrics read."""
    if _load(Path(manifest_path)).get("version") == CROSS_DATASET_BUNDLE_VERSION:
        return _seal_cross_training_completion(manifest_path, dataset, representation)
    if representation not in REPRESENTATIONS:
        raise ValueError("Only a trained representation can have a completion seal")
    path, bundle, report = _bundle(manifest_path, dataset)
    entry = _entry(bundle, dataset, representation)
    validate_prepared_split(entry["split_dir"], entry["substrate"], require_reference_progress=True)
    run_root = Path(entry["training_output"])
    start_path = run_root / "prepared_training_contract.json"
    start = _load(start_path)
    expected = {
        "dataset": dataset, "trajectory_representation": representation,
        "split_manifest_sha256": report["split_manifest"]["sha256"],
        "skillopt_config_sha256": entry["skillopt_config"]["sha256"],
        "agent_config_sha256": bundle["agent_config"]["sha256"],
        "initial_skill_sha256": bundle["initial_skill"]["sha256"],
    }
    if any(start.get(key) != value for key, value in expected.items()):
        raise ValueError("Training startup contract does not match the prepared experiment")
    summary_path = run_root / "workflow_summary.json"
    summary = _load(summary_path)
    if summary.get("dataset") != dataset or not isinstance(summary.get("trainer_summary"), dict):
        raise ValueError("Missing successful post-training workflow summary")
    if summary.get("unique_question_count") != sum(entry["actual_counts"].values()):
        raise ValueError("Completed run used a different question pool")
    # The wrapper writes this summary only after trainer.train() returns. The
    # integration contract binds its core code/prompts, not test performance.
    integration_path = run_root / "skillopt_integration.json"
    integration = _load(integration_path)
    for name, digest in integration.get("code_sha256", {}).items():
        current = Path(__file__).parent / name
        if not current.is_file() or _hash(current) != digest:
            raise ValueError(f"Completed training code differs: {name}")
    if not integration.get("code_sha256") or not integration.get("prompt_sha256"):
        raise ValueError("Training integration contract is incomplete")
    for name, digest in integration["prompt_sha256"].items():
        current = Path(__file__).parent / "prompts" / name
        if not current.is_file() or _hash(current) != digest:
            raise ValueError(f"Completed training prompt differs: {name}")
    best = run_root / "best_skill.md"
    if not best.read_text(encoding="utf-8").strip():
        raise ValueError("Completed training best Skill is empty")
    result = {
        "version": COMPLETION_VERSION, "status": "complete", **expected,
        "execution_manifest": _record(path), "startup_contract": _record(start_path),
        "workflow_summary": _record(summary_path), "integration_contract": _record(integration_path),
        "best_skill": _record(best), "test_metrics_used_for_sealing": False,
    }
    _write_json(Path(entry["completion_path"]), result)
    return result


def validate_final_test_request(
    manifest_path: str | Path, dataset: str, arm: str, *, require_completed: bool = True,
    source_dataset: str | None = None, lane_id: str | None = None,
) -> dict[str, Any]:
    if _load(Path(manifest_path)).get("version") == CROSS_DATASET_BUNDLE_VERSION:
        return _validate_cross_final_test_request(
            manifest_path, dataset, arm, require_completed=require_completed,
            source_dataset=source_dataset, lane_id=lane_id,
        )
    if source_dataset is not None or lane_id is not None:
        raise ValueError("Source dataset and lane selection require the cross-dataset design")
    if arm not in ("initial", *REPRESENTATIONS):
        raise ValueError("Unknown final-test arm")
    path, bundle, report = _bundle(manifest_path, dataset)
    representative = _entry(bundle, dataset, "raw" if arm == "initial" else arm)
    split_validation = validate_prepared_split(representative["split_dir"], representative["substrate"], require_reference_progress=True)
    completions = {}
    for representation in REPRESENTATIONS:
        entry = _entry(bundle, dataset, representation)
        completion_path = Path(entry["completion_path"])
        if not completion_path.exists():
            if require_completed:
                raise ValueError(f"All four runs must complete before ANY test, including Initial: missing {representation}")
            continue
        completion = _load(completion_path)
        if completion.get("version") != COMPLETION_VERSION or completion.get("status") != "complete" or completion.get("dataset") != dataset or completion.get("trajectory_representation") != representation:
            raise ValueError("Invalid training completion seal")
        if completion.get("execution_manifest", {}).get("sha256") != _hash(path):
            raise ValueError("Completion belongs to another execution bundle")
        if completion.get("skillopt_config_sha256") != entry["skillopt_config"]["sha256"] or completion.get("split_manifest_sha256") != report["split_manifest"]["sha256"] or completion.get("initial_skill_sha256") != bundle["initial_skill"]["sha256"] or completion.get("agent_config_sha256") != bundle["agent_config"]["sha256"]:
            raise ValueError("Completion does not match configuration, Initial Skill or split")
        for name in ("execution_manifest", "startup_contract", "workflow_summary", "integration_contract", "best_skill"):
            _check(completion[name])
        if Path(completion["best_skill"]["path"]).resolve() != Path(entry["training_output"]) / "best_skill.md":
            raise ValueError("Completion points to another run's best Skill")
        completions[representation] = completion
    selected = bundle["initial_skill"] if arm == "initial" else completions.get(arm, {}).get("best_skill")
    return {
        "version": CONFIG_BUNDLE_VERSION, "dataset": dataset, "arm": arm,
        "execution_manifest": _record(path), "agent_config": bundle["agent_config"],
        "selected_skill": selected, "substrate": representative["substrate"],
        "split_dir": representative["split_dir"], "split_validation": split_validation,
        "output": str(Path(bundle["output_root"]) / "final_test" / dataset / arm),
        "completed_arms": sorted(completions), "all_four_completed": len(completions) == 4,
        "test_count": representative["actual_counts"]["test"], "model_calls": 0,
        "dry_run": not require_completed,
    }


def run_prepared_final_test(
    manifest_path: str | Path, dataset: str, arm: str, *, execute: bool = False,
    source_dataset: str | None = None, lane_id: str | None = None,
) -> dict[str, Any]:
    """Only this explicit execute boundary can instantiate an Agent or Judge."""
    contract = validate_final_test_request(
        manifest_path, dataset, arm, require_completed=execute,
        source_dataset=source_dataset, lane_id=lane_id,
    )
    if not execute:
        return contract
    # Lazy imports guarantee --dry-run and offline preparation use no model.
    from agentic_rag.agent.config import AgentConfig
    from agentic_rag.agent.harness import AgentHarness
    from agentic_rag.evaluation import EpisodeEvaluator
    from agentic_rag.evaluation.profiles import get_dataset_profile
    from agentic_rag.skillopt.rollout import RolloutBatch, run_rollout_batch

    spec = importlib.util.spec_from_file_location("prepared_local_qwen_judge", _ROOT / "scripts" / "run_local_qwen_skillopt.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the existing local Qwen Judge")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = AgentConfig.from_yaml(contract["agent_config"]["path"])
    skill_content = Path(contract["selected_skill"]["path"]).read_text(encoding="utf-8")
    test_path = Path(contract["split_dir"]) / "test.jsonl"
    items = tuple(json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(items) != contract["test_count"]:
        raise ValueError("Final test must use the full frozen test set")
    output = Path(contract["output"])
    _write_json(output / "final_test_contract.json", contract)

    def factory(*, skill_content: str, output_root: Path) -> AgentHarness:
        return AgentHarness.from_skill_content(substrate_path=contract["substrate"], config=config, skill_content=skill_content,
                                               skill_source_path=contract["selected_skill"]["path"], output_root=output_root)

    started_ns, started = time.time_ns(), time.perf_counter()
    completed = False
    try:
        rows = run_rollout_batch(batch=RolloutBatch(items, phase="eval", split="test"), out_root=output,
                                 skill_content=skill_content, harness_factory=factory,
                                 evaluator=EpisodeEvaluator(runner.LocalQwenJudge(config.policy), profile=get_dataset_profile(dataset)),
                                 workers=1, resume=True, trajectory_representation="raw")
        completed = True
    finally:
        # A separate immutable record preserves completed invocation time across
        # resume. Abrupt process death can prevent this record; never infer time
        # from filesystem timestamps or pretend it measures only model latency.
        _write_json(output / "execution_timing" / f"{started_ns}.json", {
            "started_unix_ns": started_ns, "elapsed_seconds": time.perf_counter() - started,
            "batch_returned": completed, "includes": "target, retrieval, judge, persistence, and resume checks",
        })
    summary = {"dataset": dataset, "arm": arm, "count": len(rows), "complete": len(rows) == len(items),
               "artifact_root": str(output), "test_used_for_skill_selection": False,
               "metric_means": {metric: sum(float(row["metrics"][metric]) for row in rows) / len(rows)
                                for metric in rows[0].get("metrics", {})} if rows else {}}
    if contract.get("design") == CROSS_DATASET_DESIGN:
        summary.update(source_dataset=source_dataset, target_dataset=dataset,
                       lane_id=contract["lane_id"], task_id=contract["task_id"])
    _write_json(output / "summary.json", summary)
    return summary


def _normalized_lanes(lanes: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if not lanes:
        raise ValueError("Cross-dataset design requires an explicit ordered lane list")
    result, seen_ids, seen_hosts = [], set(), set()
    for raw in lanes:
        lane_id, host = str(raw.get("lane_id") or ""), str(raw.get("host") or "").rstrip("/")
        parsed = urlparse(host)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", lane_id) or lane_id in seen_ids:
            raise ValueError("Lane IDs must be unique nonempty names")
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path or
                parsed.username or parsed.password or parsed.query or parsed.fragment or host in seen_hosts):
            raise ValueError("Each lane needs a distinct credential-free Ollama host URL")
        gpu_id, amd_port = raw.get("gpu_id"), raw.get("amd_port")
        if type(gpu_id) is not int or gpu_id < 0 or type(amd_port) is not int or not 1 <= amd_port <= 65535:
            raise ValueError("Lane GPU ID and AMD port must be explicit valid integers")
        seen_ids.add(lane_id)
        seen_hosts.add(host)
        result.append({"lane_id": lane_id, "host": host, "amd_port": amd_port, "gpu_id": gpu_id})
    return result


def _write_cross_dataset_configs(
    prepared_root: str | Path, initial_skill: str | Path,
    optimizer_config: str | Path, agent_config: str | Path, output: str | Path,
    *, lanes: Sequence[Mapping[str, Any]] | None, model_digest: str, reflection_minibatch_size: int,
    reflection_token_budget_enabled: bool,
) -> dict[str, Any]:
    """Twelve single-source training tasks and sixty-five paired test tasks."""
    lane_records = _normalized_lanes(lanes)
    if not re.fullmatch(r"[0-9a-f]{64}", model_digest):
        raise ValueError("Expected Ollama model digest must be a full SHA-256")
    prepared, destination = Path(prepared_root).resolve(), Path(output).resolve()
    skill, optimizer_source, agent_source = [Path(path).resolve() for path in
                                            (initial_skill, optimizer_config, agent_config)]
    optimizer = yaml.safe_load(optimizer_source.read_text(encoding="utf-8"))
    agent = yaml.safe_load(agent_source.read_text(encoding="utf-8"))
    if not isinstance(optimizer, dict) or not isinstance(agent, dict) or not skill.read_text(encoding="utf-8").strip():
        raise ValueError("Configs must be mappings and Initial Skill must be nonempty")
    policy, model = agent.get("policy", {}), optimizer.get("model", {})
    if (policy.get("provider") != "ollama" or policy.get("model") != MODEL or
            policy.get("think") is not False or policy.get("temperature") != 0):
        raise ValueError("Target config must retain Ollama Qwen 3.6, think=false, temperature=0")
    if model.get("optimizer") != MODEL or model.get("target") != MODEL:
        raise ValueError("Optimizer and target must use the pinned Qwen 3.6 model")
    if model.get("optimizer_qwen_chat_enable_thinking", model.get("qwen_chat_enable_thinking")) is not True:
        raise ValueError("Optimizer thinking must remain explicitly enabled")
    if optimizer.get("env", {}).get("reflection_token_budget_enabled") and not reflection_token_budget_enabled:
        raise ValueError("Adaptive reflection in the source config requires explicit generation opt-in")
    archived_skill = destination / "initial_skill.md"
    _write_bytes(archived_skill, skill.read_bytes())
    for lane in lane_records:
        lane_agent = copy.deepcopy(agent)
        lane_agent.setdefault("agent", {}).update(max_steps=10, max_policy_attempts=12, max_retrieved_tokens=12000)
        lane_agent["agent"]["show_available_action_options"] = True
        lane_agent["agent"]["use_state_conditioned_schema"] = True
        lane_agent["policy"]["host"] = lane["host"]
        lane_agent["policy"]["num_ctx"] = 262144
        lane_path = destination / "lanes" / lane["lane_id"] / "agent.yaml"
        _write_yaml(lane_path, lane_agent)
        lane["agent_config"] = _record(lane_path)
    dataset_reports = {}
    for dataset in DATASETS:
        directory = prepared / dataset
        readiness_path, split_path = directory / "readiness.json", directory / "split_manifest.json"
        readiness = _load(readiness_path) if readiness_path.is_file() else {}
        split = _load(split_path) if split_path.is_file() else None
        counts = _counts(split)
        substrate = (split or {}).get("dataset", {}).get("substrate", {}).get("path")
        inference_ready = bool(split and substrate and readiness.get("split_status") == "ready"
                               and readiness.get("inference_ready", True))
        reference_ready = bool(readiness.get("reference_progress_ready", readiness.get("all_four_arms_ready", False)))
        training_ready = bool(dataset in TRAINING_DATASETS and inference_ready and reference_ready
                              and readiness.get("training_ready", True))
        if inference_ready:
            validate_prepared_split(directory, substrate, require_reference_progress=False)
        if training_ready:
            validate_prepared_split(directory, substrate, require_reference_progress=True)
        dataset_reports[dataset] = {
            "dataset_role": "training_source" if dataset in TRAINING_DATASETS else "transfer_only",
            "inference_ready": inference_ready, "training_ready": training_ready,
            "reference_progress_ready": reference_ready,
            "actual_counts": counts, "target_counts": _targets(readiness),
            "substrate": substrate, "split_dir": str(directory),
            "readiness": _record(readiness_path) if readiness_path.is_file() else None,
            "split_manifest": _record(split_path) if split_path.is_file() else None,
            "reason": readiness.get("split_error") or readiness.get("error") or
                      (None if inference_ready else "split_unavailable"),
        }
    manifest_path = destination / "execution_manifest.json"
    python = str(_ROOT / ".venv" / "bin" / "python")
    trainer_script = str(_ROOT / "scripts" / "run_local_qwen_skillopt.py")
    test_script = str(_ROOT / "scripts" / "run_prepared_skillopt_test.py")
    entries = []
    for dataset in TRAINING_DATASETS:
        report = dataset_reports[dataset]
        counts = report["actual_counts"]
        for representation in REPRESENTATIONS:
            run_root = destination / "training_runs" / dataset / representation
            variants = []
            for lane in lane_records:
                config = copy.deepcopy(optimizer)
                config["run_name"] = f"{dataset}_single_source_{representation}"
                config.setdefault("train", {}).update(train_size=counts["train"], batch_size=40,
                                                        accumulation=1, num_epochs=1, seed=42)
                config.setdefault("gradient", {}).update(minibatch_size=reflection_minibatch_size, analyst_workers=1)
                config.setdefault("evaluation", {}).update(sel_env_num=counts["validation"],
                                                            test_env_num=counts["test"], eval_test=False)
                config.setdefault("env", {}).update(name=f"agentic_rag_{dataset}", workers=1,
                                                    workflow_label=CROSS_DATASET_DESIGN,
                                                    durable_checkpoint_enabled=True)
                if reflection_token_budget_enabled:
                    config["env"].update(_reflection_budget_env(lane))
                    config["model"]["qwen_chat_max_tokens"] = 16384
                    config["model"]["optimizer_qwen_chat_max_tokens"] = 16384
                for prefix in ("", "optimizer_", "target_"):
                    config["model"][prefix + "qwen_chat_base_url"] = lane["host"] + "/v1"
                config["model"]["optimizer_qwen_chat_enable_thinking"] = True
                config["model"]["target_qwen_chat_enable_thinking"] = False
                config_path = destination / dataset / representation / f"{lane['lane_id']}.yaml"
                _write_yaml(config_path, config)
                argv = [python, trainer_script, "--substrate", str(report["substrate"] or "UNAVAILABLE_SUBSTRATE"),
                        "--split-dir", report["split_dir"], "--agent-config", lane["agent_config"]["path"],
                        "--skillopt-config", str(config_path), "--skill", str(archived_skill),
                        "--output", str(run_root), "--trajectory-representation", representation]
                variants.append({**lane, "skillopt_config": _record(config_path),
                                 "train_argv": argv, "train_command": shlex.join(argv)})
            entries.append({
                "task_id": f"train:{dataset}:{representation}", "dataset": dataset,
                "source_dataset": dataset, "representation": representation,
                "ready": report["training_ready"], "actual_counts": counts,
                "split_dir": report["split_dir"], "substrate": report["substrate"],
                "training_output": str(run_root), "completion_path": str(run_root / "training_completion.json"),
                "lane_variants": variants, "train_command_executable": report["training_ready"],
                "maximum_skill_updates": math.ceil(counts["train"] / 40) if counts["train"] is not None else None,
                "candidate_validation_episode_upper_bound": math.ceil(counts["train"] / 40) * counts["validation"]
                    if counts["train"] is not None else None,
                "seal_completion_argv": [python, test_script, "--manifest", str(manifest_path),
                                         "--dataset", dataset, "--arm", representation, "--seal-completion"],
            })
    final_tasks = []
    skill_sources = [(None, "initial")] + [(dataset, arm) for dataset in TRAINING_DATASETS for arm in REPRESENTATIONS]
    for target in DATASETS:
        target_report = dataset_reports[target]
        for source, arm in skill_sources:
            label = "initial" if source is None else f"{source}/{arm}"
            output_root = destination / "final_test" / target / label
            variants = []
            for lane in lane_records:
                argv = [python, test_script, "--manifest", str(manifest_path), "--dataset", target,
                        "--arm", arm, "--lane-id", lane["lane_id"]]
                if source is not None:
                    argv.extend(["--source-dataset", source])
                variants.append({**lane, "dry_run_argv": argv, "execute_argv": argv + ["--execute"]})
            final_tasks.append({
                "task_id": f"test:{target}:{source or 'initial'}:{arm}",
                "dataset": target, "target_dataset": target, "source_dataset": source, "arm": arm,
                "ready": target_report["inference_ready"], "test_count": target_report["actual_counts"]["test"],
                "output": str(output_root), "lane_variants": variants,
            })
    result = {
        "version": CROSS_DATASET_BUNDLE_VERSION, "design": CROSS_DATASET_DESIGN,
        "prepared_root": str(prepared), "output_root": str(destination),
        "initial_skill": _record(archived_skill), "source_initial_skill": _record(skill),
        "source_optimizer_config": _record(optimizer_source), "source_agent_config": _record(agent_source),
        "lanes": lane_records, "training_datasets": list(TRAINING_DATASETS), "test_datasets": list(DATASETS),
        "git_revision": _git_revision(), "code_files": _source_files(),
        "core_prompt_files": [_record(path) for path in sorted((Path(__file__).parent / "prompts").glob("*.md"))],
        "target_model": MODEL, "optimizer_model": MODEL, "expected_model_digest": model_digest,
        "model_digest_verification": "Runtime scheduler must verify each assigned endpoint before execution; config generation makes no API calls.",
        "target_thinking": False, "optimizer_thinking": True, "train_batch_size": 40,
        "reflection_minibatch_size": reflection_minibatch_size, "target_workers_per_task": 1, "analyst_workers_per_task": 1,
        "num_epochs": 1, "eval_test_during_training": False,
        "datasets": dataset_reports, "experiments": entries, "final_test_commands": final_tasks,
        "final_test_gate": "all_twelve_training_runs_completed_and_sealed",
        "training_task_count": len(entries), "final_test_task_count": len(final_tasks),
        "final_inference_episode_count": sum(task["test_count"] for task in final_tasks)
            if all(task["test_count"] is not None for task in final_tasks) else None,
        "lane_policy": "Runtime scheduler chooses an approved free lane; every task must resume its original lane variant.",
        "aggregation_argv": [python, test_script, "--manifest", str(manifest_path), "--aggregate"],
        "aggregation_summary_path": str(destination / "comparisons" / "summary.json"),
        "model_calls": 0,
    }
    if reflection_token_budget_enabled:
        result["reflection_grouping"] = "adaptive"
        result["reflection_token_budget"] = dict(_REFLECTION_TOKEN_BUDGET)
    _write_json(manifest_path, result)
    return result


def _reflection_budget_env(lane: Mapping[str, Any]) -> dict[str, Any]:
    """Native SkillOpt passes unknown env fields through its flat config."""
    return {
        "reflection_token_budget_enabled": True,
        "reflection_context_tokens": _REFLECTION_TOKEN_BUDGET["context_tokens"],
        "reflection_output_reserve_tokens": _REFLECTION_TOKEN_BUDGET["output_reserve_tokens"],
        "reflection_safety_margin_tokens": _REFLECTION_TOKEN_BUDGET["safety_margin_tokens"],
        "reflection_tokenizer_amd_port": lane["amd_port"],
        "reflection_tokenizer_ssh_host": "root@140.116.240.181",
        "reflection_tokenizer_ssh_port": 45026,
        "reflection_tokenizer_ssh_key": "/home/jj/.ssh/amd_root_key",
    }


def _validate_reflection_token_budget(bundle: Mapping[str, Any]) -> bool:
    if "reflection_grouping" not in bundle and "reflection_token_budget" not in bundle:
        return False
    if (bundle.get("reflection_grouping") != "adaptive"
            or bundle.get("reflection_token_budget") != _REFLECTION_TOKEN_BUDGET
            or bundle.get("reflection_minibatch_size") != 5):
        raise ValueError("Invalid frozen adaptive reflection token-budget policy")
    return True


def _paired_bootstrap(
    questions: Sequence[str], differences: Sequence[int], *, seed: int, samples: int,
) -> tuple[list[float], Any]:
    """Resample paired question groups, keeping duplicate questions together."""
    import numpy as np
    from agentic_rag.skillopt.multidataset_prepare import normalized_text

    groups: dict[str, list[int]] = {}
    for question, difference in zip(questions, differences, strict=True):
        groups.setdefault(normalized_text(question), []).append(difference)
    sums = np.asarray([sum(values) for values in groups.values()], dtype=float)
    sizes = np.asarray([len(values) for values in groups.values()], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, samples, 250):
        indices = rng.integers(0, len(groups), size=(min(250, samples - start), len(groups)))
        estimates.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
    array = np.asarray(estimates)
    return [float(value) for value in np.quantile(array, [0.025, 0.975])], array


def _read_completed_final_cell(
    bundle: Mapping[str, Any], manifest_path: Path, task: Mapping[str, Any], items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from agentic_rag.paths import portable_path_component

    output = Path(task["output"])
    summary, contract = _load(output / "summary.json"), _load(output / "final_test_contract.json")
    if (summary.get("complete") is not True or summary.get("count") != len(items)
            or summary.get("task_id") != task["task_id"]):
        raise ValueError(f"Final task is not completely saved: {task['task_id']}")
    lane = next((row for row in bundle["lanes"] if row["lane_id"] == contract.get("lane_id")), None)
    if lane is None:
        raise ValueError("Final task used an unrecognized lane")
    source = task["source_dataset"]
    selected = (bundle["initial_skill"] if source is None else
                _load(Path(bundle["output_root"]) / "training_runs" / source / task["arm"] /
                      "training_completion.json")["best_skill"])
    if (contract.get("task_id") != task["task_id"] or contract.get("target_dataset") != task["target_dataset"]
            or contract.get("source_dataset") != source or contract.get("arm") != task["arm"]
            or contract.get("selected_skill") != selected or contract.get("agent_config") != lane["agent_config"]
            or contract.get("execution_manifest") != _record(manifest_path)
            or contract.get("expected_model_digest") != bundle["expected_model_digest"]
            or contract.get("target_split_manifest") != bundle["datasets"][task["target_dataset"]]["split_manifest"]):
        raise ValueError("Saved final task identity differs from its pinned matrix cell")
    result_paths = list((output / "predictions").glob("*/rollout_result.json"))
    expected_paths = [output / "predictions" / portable_path_component(str(item["id"])) / "rollout_result.json"
                      for item in items]
    if set(result_paths) != set(expected_paths):
        raise ValueError(f"Final task has missing, duplicate, or unexpected question artifacts: {task['task_id']}")
    rows = []
    for item, path in zip(items, expected_paths, strict=True):
        row, evaluation = _load(path), _load(path.parent / "evaluation.json")
        score = row.get("metrics", {}).get("llm_acc")
        if (type(score) is not int or score not in (0, 1) or row.get("hard") != score
                or row.get("metric_contract", {}).get("hard") != "llm_acc"):
            raise ValueError("Final comparison requires the saved binary Qwen llm_acc, not a substituted metric")
        if (row.get("id") != item["id"] or row.get("question") != item["question"]
                or row.get("dataset") != task["target_dataset"]
                or evaluation.get("question") != item["question"] or evaluation.get("gold_answer") != item["answer"]
                or evaluation.get("predicted_answer") != row.get("predicted_answer") or evaluation.get("llm_acc") != score):
            raise ValueError("Saved final answer/evaluation does not match its frozen question or gold answer")
        rows.append(row)
    count = len(rows)
    usage_fields = ("policy_calls", "input_tokens", "output_tokens", "total_tokens", "retrieved_tokens")
    usage = {field: sum(int(row.get("usage", {}).get(field, 0)) for row in rows) for field in usage_fields}
    judge_tokens = sum(int(row.get("judge_usage", {}).get("total_tokens", 0)) for row in rows)
    judge_calls = sum(int(row.get("judge_usage", {}).get("calls", 0)) for row in rows)
    timings = [_load(path) for path in sorted((output / "execution_timing").glob("*.json"))]
    elapsed = sum(float(record["elapsed_seconds"]) for record in timings) if timings else None
    termination: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("termination_reason", "unknown"))
        termination[reason] = termination.get(reason, 0) + 1
    stats = {
        "task_id": task["task_id"], "target_dataset": task["target_dataset"], "source_dataset": source,
        "arm": task["arm"], "count": count, "lane_id": lane["lane_id"],
        "correct": sum(row["metrics"]["llm_acc"] for row in rows),
        "accuracy": sum(row["metrics"]["llm_acc"] for row in rows) / count,
        "empty_answers": sum(not row.get("predicted_answer", "").strip() for row in rows),
        "execution_errors": sum(termination.get(reason, 0) for reason in ("policy_error", "runtime_error")),
        "termination_reasons": termination,
        "average_steps": sum(int(row["n_turns"]) for row in rows) / count,
        "usage_totals": usage, "usage_per_question": {key: value / count for key, value in usage.items()},
        "judge_calls": judge_calls, "judge_tokens": judge_tokens,
        "recorded_runner_elapsed_seconds": elapsed,
        "recorded_runner_seconds_per_question": elapsed / count if elapsed is not None else None,
        "final_contract": _record(output / "final_test_contract.json"), "summary": _record(output / "summary.json"),
    }
    stats["empty_answer_rate"] = stats["empty_answers"] / count
    stats["execution_error_rate"] = stats["execution_errors"] / count
    return rows, stats


def aggregate_final_test_results(
    manifest_path: str | Path, *, bootstrap_samples: int = 5000, seed: int = 42,
) -> dict[str, Any]:
    """Offline paired comparison, deliberately unavailable for a partial matrix."""
    import numpy as np

    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    path, bundle = _cross_bundle(manifest_path)
    _cross_completions(path, bundle, require_completed=True)
    if len(bundle["final_test_commands"]) != 65:
        raise ValueError("The cross-dataset comparison requires all 65 frozen matrix cells")
    cells, rows_by_task, questions_by_target = [], {}, {}
    # Complete and validate the whole matrix before writing any comparison.
    for target in DATASETS:
        report = bundle["datasets"][target]
        validate_prepared_split(report["split_dir"], report["substrate"], require_reference_progress=False)
        items = [json.loads(line) for line in (Path(report["split_dir"]) / "test.jsonl").read_text().splitlines()
                 if line.strip()]
        if len(items) != report["actual_counts"]["test"] or len({item["id"] for item in items}) != len(items):
            raise ValueError("Test count or canonical question IDs are invalid")
        questions_by_target[target] = [item["question"] for item in items]
        tasks = [task for task in bundle["final_test_commands"] if task["target_dataset"] == target]
        if len(tasks) != 13 or sum(task["source_dataset"] is None for task in tasks) != 1:
            raise ValueError("Each target must contain twelve learned Skills and exactly one Initial")
        for task in tasks:
            rows, stats = _read_completed_final_cell(bundle, path, task, items)
            rows_by_task[task["task_id"]] = rows
            cells.append(stats)
    comparisons, paired_rows, macro_samples = [], [], {}
    for target in DATASETS:
        initial = next(cell for cell in cells if cell["target_dataset"] == target and cell["source_dataset"] is None)
        initial_rows = rows_by_task[initial["task_id"]]
        for cell in [cell for cell in cells if cell["target_dataset"] == target and cell["source_dataset"] is not None]:
            rows = rows_by_task[cell["task_id"]]
            differences = [row["metrics"]["llm_acc"] - base["metrics"]["llm_acc"]
                           for row, base in zip(rows, initial_rows, strict=True)]
            local_seed = int(hashlib.sha256(f"{seed}\0{cell['task_id']}".encode()).hexdigest()[:16], 16)
            ci, samples = _paired_bootstrap(questions_by_target[target], differences, seed=local_seed,
                                             samples=bootstrap_samples)
            key = (cell["source_dataset"], cell["arm"])
            macro_samples.setdefault(key, []).append(samples)
            comparison = {
                "target_dataset": target, "source_dataset": cell["source_dataset"], "arm": cell["arm"],
                "count": cell["count"], "initial_accuracy": initial["accuracy"], "learned_accuracy": cell["accuracy"],
                "accuracy_delta": cell["accuracy"] - initial["accuracy"], "paired_bootstrap_95_ci": ci,
                "both_correct": sum(a["metrics"]["llm_acc"] == b["metrics"]["llm_acc"] == 1
                                    for a, b in zip(rows, initial_rows, strict=True)),
                "learned_only": differences.count(1), "initial_only": differences.count(-1),
                "both_wrong": sum(a["metrics"]["llm_acc"] == b["metrics"]["llm_acc"] == 0
                                  for a, b in zip(rows, initial_rows, strict=True)),
            }
            comparisons.append(comparison)
            for row, base, delta in zip(rows, initial_rows, differences, strict=True):
                paired_rows.append({"target_dataset": target, "source_dataset": cell["source_dataset"], "arm": cell["arm"],
                                    "id": row["id"], "question": row["question"], "initial_correct": base["metrics"]["llm_acc"],
                                    "learned_correct": row["metrics"]["llm_acc"], "delta": delta,
                                    "initial_answer": base["predicted_answer"], "learned_answer": row["predicted_answer"]})
    macro = []
    for (source, arm), samples in macro_samples.items():
        subset = [row for row in comparisons if row["source_dataset"] == source and row["arm"] == arm]
        estimates = np.mean(np.stack(samples), axis=0)
        macro.append({"source_dataset": source, "arm": arm, "datasets": len(subset),
                      "initial_accuracy": sum(row["initial_accuracy"] for row in subset) / len(subset),
                      "learned_accuracy": sum(row["learned_accuracy"] for row in subset) / len(subset),
                      "accuracy_delta": sum(row["accuracy_delta"] for row in subset) / len(subset),
                      "paired_bootstrap_95_ci": [float(v) for v in np.quantile(estimates, [0.025, 0.975])]})
    result = {
        "version": "skillopt-cross-dataset-comparison-v1", "execution_manifest": _record(path),
        "primary_metric": "saved_binary_qwen_llm_acc", "additional_judge_calls": 0,
        "final_task_count": len(cells), "inference_episode_count": sum(cell["count"] for cell in cells),
        "bootstrap": {"samples": bootstrap_samples, "seed": seed, "method": "paired normalized-question-group resampling within each target",
                      "macro": "equal weight to each target; independently resampled targets"},
        "limitations": ["Intervals are descriptive and unadjusted for the 60 per-target comparisons.",
                        "Recorded runner time includes retrieval, target generation, judging, persistence and resume checks; it excludes downtime and may omit a killed invocation.",
                        "Execution errors count policy_error/runtime_error; budget exhaustion is reported separately, not conflated with incorrect answers.",
                        "One saved inference and one existing Qwen binary evaluation per answer; no new judging or causal attribution is performed."],
        "cells": cells, "comparisons_vs_initial": comparisons, "macro_average": macro,
    }
    destination = Path(bundle["output_root"]) / "comparisons"
    lines = ["# Cross-dataset Skill comparison", "", "Primary metric: saved Qwen binary correctness. No additional Judge calls.", "",
             "| Target | Skill trained on | Input representation | Initial | Learned | Difference | Paired 95% CI |", "|---|---|---|---:|---:|---:|---|" ]
    for row in comparisons:
        low, high = row["paired_bootstrap_95_ci"]
        lines.append(f"| {row['target_dataset']} | {row['source_dataset']} | {row['arm']} | {row['initial_accuracy']:.3f} | {row['learned_accuracy']:.3f} | {row['accuracy_delta']:+.3f} | [{low:+.3f}, {high:+.3f}] |")
    lines += ["", "## Equal-weight average across the five datasets", "", "| Skill trained on | Representation | Initial | Learned | Difference | Paired 95% CI |", "|---|---|---:|---:|---:|---|"]
    for row in macro:
        low, high = row["paired_bootstrap_95_ci"]
        lines.append(f"| {row['source_dataset']} | {row['arm']} | {row['initial_accuracy']:.3f} | {row['learned_accuracy']:.3f} | {row['accuracy_delta']:+.3f} | [{low:+.3f}, {high:+.3f}] |")
    lines += ["", "## Cost and failure counts", "", "| Target | Source | Representation | Empty | Execution error | Mean steps | Target tokens | Retrieval tokens | Recorded seconds |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in cells:
        elapsed = row["recorded_runner_elapsed_seconds"]
        duration = f"{elapsed:.1f}" if elapsed is not None else "not recorded"
        lines.append(f"| {row['target_dataset']} | {row['source_dataset'] or 'Initial'} | {row['arm']} | {row['empty_answers']} | {row['execution_errors']} | {row['average_steps']:.2f} | {row['usage_totals']['total_tokens']} | {row['usage_totals']['retrieved_tokens']} | {duration} |")
    lines += ["", "## Interpretation limits", ""] + ["- " + item for item in result["limitations"]]
    _write_bytes(destination / "per_question.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in paired_rows).encode())
    _write_bytes(destination / "report.md", ("\n".join(lines) + "\n").encode())
    _write_json(destination / "summary.json", result)
    return result
