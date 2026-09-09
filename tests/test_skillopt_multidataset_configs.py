from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

import agentic_rag.skillopt.multidataset_configs as configs


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def bundle_inputs(tmp_path, monkeypatch):
    calls = []

    def validate(split_dir, substrate, require_reference_progress=False):
        calls.append((str(split_dir), str(substrate), require_reference_progress))
        return {"valid": True, "all_four_arms_ready": True}

    monkeypatch.setattr(configs, "validate_prepared_split", validate)
    monkeypatch.setattr(configs, "_source_files", lambda: [])
    monkeypatch.setattr(configs, "_git_revision", lambda: "frozen-commit")
    prepared = tmp_path / "prepared"
    for dataset in configs.DATASETS:
        substrate = tmp_path / "indexes" / dataset
        _json(substrate / "manifest.json", {"dataset": dataset})
        _json(prepared / dataset / "readiness.json", {
            "dataset": dataset, "eligible_count": 1000, "split_status": "ready", "all_four_arms_ready": True})
        _json(prepared / dataset / "split_manifest.json", {
            "dataset": {"subset": dataset, "substrate": {"path": str(substrate)}},
            "splits": {role: {"count": count} for role, count in {"train": 200, "validation": 200, "test": 600}.items()}})
    initial = tmp_path / "source_initial.md"
    initial.write_text("## Trainable action and answer workflow\nSearch and finish.\n", encoding="utf-8")
    agent = tmp_path / "agent_source.yaml"
    agent.write_text(yaml.safe_dump({"agent": {"enabled_expansions": ["CHUNK_ADJACENT_CHUNK"], "show_available_action_options": False},
                                    "policy": {"provider": "ollama", "model": configs.MODEL,
                                               "host": "http://127.0.0.1:11435", "temperature": 0, "think": False}}), encoding="utf-8")
    optimizer = tmp_path / "optimizer_source.yaml"
    optimizer.write_text(yaml.safe_dump({
        "model": {"optimizer": configs.MODEL, "target": configs.MODEL,
                  "optimizer_qwen_chat_base_url": "http://127.0.0.1:11435/v1", "optimizer_qwen_chat_enable_thinking": True},
        "train": {"train_size": 100, "batch_size": 1, "seed": 9},
        "gradient": {"minibatch_size": 1, "merge_batch_size": 3},
        "evaluation": {"sel_env_num": 50, "test_env_num": 50, "eval_test": True, "use_gate": True}}), encoding="utf-8")
    return {"prepared_root": prepared, "initial_skill": initial, "optimizer_config": optimizer,
            "agent_config": agent, "output": tmp_path / "bundle", "calls": calls}


def _generate(inputs):
    return configs.write_experiment_configs(**{key: value for key, value in inputs.items() if key != "calls"})


def _manifest_path(inputs):
    return inputs["output"] / "execution_manifest.json"


def _make_completed(inputs, bundle, dataset, representation):
    entry = next(e for e in bundle["experiments"] if e["dataset"] == dataset and e["representation"] == representation)
    run = Path(entry["training_output"])
    start = {
        "dataset": dataset, "trajectory_representation": representation,
        "split_manifest_sha256": bundle["datasets"][dataset]["split_manifest"]["sha256"],
        "skillopt_config_sha256": entry["skillopt_config"]["sha256"],
        "agent_config_sha256": bundle["agent_config"]["sha256"],
        "initial_skill_sha256": bundle["initial_skill"]["sha256"],
    }
    _json(run / "prepared_training_contract.json", start)
    _json(run / "workflow_summary.json", {"dataset": dataset, "trainer_summary": {"best_step": 1},
                                           "unique_question_count": sum(entry["actual_counts"].values())})
    module = Path(configs.__file__)
    prompt = module.parent / "prompts" / "ranking.md"
    _json(run / "skillopt_integration.json", {
        "code_sha256": {module.name: hashlib.sha256(module.read_bytes()).hexdigest()},
        "prompt_sha256": {prompt.name: hashlib.sha256(prompt.read_bytes()).hexdigest()}})
    (run / "best_skill.md").write_text(f"Learned {representation} Skill\n", encoding="utf-8")
    return run


def test_all_twenty_configs_use_actual_counts_and_shared_controls(bundle_inputs):
    bundle = _generate(bundle_inputs)
    assert len(bundle["experiments"]) == 20
    assert len(bundle["final_test_commands"]) == 25
    assert set(e["representation"] for e in bundle["experiments"]) == set(configs.REPRESENTATIONS)
    assert len(bundle_inputs["calls"]) == 5
    assert all(call[2] for call in bundle_inputs["calls"])
    for entry in bundle["experiments"]:
        config = yaml.safe_load(Path(entry["skillopt_config"]["path"]).read_text())
        assert config["train"] == {"train_size": 200, "batch_size": 20, "seed": 42, "accumulation": 1, "num_epochs": 1}
        assert config["gradient"]["minibatch_size"] == 5
        assert config["gradient"]["merge_batch_size"] == 3
        assert config["evaluation"] == {"sel_env_num": 200, "test_env_num": 600, "eval_test": False, "use_gate": True}
        assert entry["maximum_skill_updates"] == 10
        assert entry["candidate_validation_episode_upper_bound"] == 2000
        assert entry["train_command_executable"] is True
        assert "--execute" not in entry["train_argv"]
    agent = yaml.safe_load(Path(bundle["agent_config"]["path"]).read_text())
    assert agent["agent"]["max_steps"] == 10
    assert agent["agent"]["max_policy_attempts"] == 12
    assert agent["agent"]["max_retrieved_tokens"] == 12000
    assert agent["agent"]["show_available_action_options"] is False
    assert agent["agent"]["use_state_conditioned_schema"] is True
    assert bundle["model_calls"] == 0
    assert Path(bundle["initial_skill"]["path"]).read_bytes() == bundle_inputs["initial_skill"].read_bytes()


def test_blocked_split_has_no_fabricated_actual_counts(bundle_inputs):
    directory = bundle_inputs["prepared_root"] / "hotpotqa"
    (directory / "split_manifest.json").unlink()
    _json(directory / "readiness.json", {"split_status": "blocked", "all_four_arms_ready": False,
                                        "eligible_count": 1000, "split_error": {"code": "history_conflict"}})
    bundle = _generate(bundle_inputs)
    blocked = [e for e in bundle["experiments"] if e["dataset"] == "hotpotqa"]
    assert len(blocked) == 4
    for entry in blocked:
        assert not entry["ready"] and not entry["train_command_executable"]
        assert entry["actual_counts"] == {"train": None, "validation": None, "test": None}
        assert entry["target_counts"] == {"train": 200, "validation": 200, "test": 600}
        assert yaml.safe_load(Path(entry["skillopt_config"]["path"]).read_text())["train"]["train_size"] is None
    with pytest.raises(ValueError, match="not ready"):
        configs.run_prepared_final_test(_manifest_path(bundle_inputs), "hotpotqa", "initial")


def test_unmapped_reference_blocks_all_arms_but_keeps_actual_counts(bundle_inputs):
    _json(bundle_inputs["prepared_root"] / "musique" / "readiness.json", {
        "eligible_count": 1000, "split_status": "ready", "all_four_arms_ready": False, "mapping_issue_count": 1000})
    bundle = _generate(bundle_inputs)
    entries = [e for e in bundle["experiments"] if e["dataset"] == "musique"]
    assert all(not e["ready"] and e["actual_counts"]["test"] == 600 for e in entries)


def test_bundle_is_deterministic_and_never_overwrites_new_skill(bundle_inputs):
    first = _generate(bundle_inputs)
    assert first == _generate(bundle_inputs)
    bundle_inputs["initial_skill"].write_text("Different initial skill")
    with pytest.raises(FileExistsError, match="different artifact"):
        _generate(bundle_inputs)


def test_rejects_a_different_model_before_creating_output(bundle_inputs):
    agent = yaml.safe_load(bundle_inputs["agent_config"].read_text())
    agent["policy"]["model"] = "different-model"
    bundle_inputs["agent_config"].write_text(yaml.safe_dump(agent))
    with pytest.raises(ValueError, match="Target config"):
        _generate(bundle_inputs)
    assert not bundle_inputs["output"].exists()


def test_final_test_default_is_offline_dry_run_without_output(bundle_inputs):
    _generate(bundle_inputs)
    result = configs.run_prepared_final_test(_manifest_path(bundle_inputs), "medical", "initial")
    assert result["dry_run"] is True
    assert result["all_four_completed"] is False
    assert result["test_count"] == 600
    assert result["model_calls"] == 0
    assert not (bundle_inputs["output"] / "final_test").exists()


def test_best_dry_run_shows_pending_skill_without_guessing(bundle_inputs):
    _generate(bundle_inputs)
    result = configs.run_prepared_final_test(_manifest_path(bundle_inputs), "novel", "organized")
    assert result["selected_skill"] is None
    assert result["completed_arms"] == []


def test_initial_test_execute_is_blocked_until_every_arm_finishes(bundle_inputs):
    bundle = _generate(bundle_inputs)
    _make_completed(bundle_inputs, bundle, "medical", "raw")
    configs.seal_training_completion(_manifest_path(bundle_inputs), "medical", "raw")
    with pytest.raises(ValueError, match="All four runs must complete"):
        configs.run_prepared_final_test(_manifest_path(bundle_inputs), "medical", "initial", execute=True)
    assert not (bundle_inputs["output"] / "final_test").exists()


def test_seals_all_four_and_selects_correct_frozen_best(bundle_inputs):
    bundle = _generate(bundle_inputs)
    for arm in configs.REPRESENTATIONS:
        _make_completed(bundle_inputs, bundle, "novel", arm)
        seal = configs.seal_training_completion(_manifest_path(bundle_inputs), "novel", arm)
        assert seal["status"] == "complete"
        assert seal["test_metrics_used_for_sealing"] is False
    result = configs.validate_final_test_request(_manifest_path(bundle_inputs), "novel", "progress_abstracted")
    assert result["all_four_completed"] is True
    assert "progress_abstracted" in result["selected_skill"]["path"]
    initial = configs.validate_final_test_request(_manifest_path(bundle_inputs), "novel", "initial")
    assert initial["selected_skill"] == bundle["initial_skill"]
    assert not (bundle_inputs["output"] / "final_test").exists()


def test_rejects_tampered_best_after_sealing(bundle_inputs):
    bundle = _generate(bundle_inputs)
    for arm in configs.REPRESENTATIONS:
        _make_completed(bundle_inputs, bundle, "medical", arm)
        configs.seal_training_completion(_manifest_path(bundle_inputs), "medical", arm)
    Path(bundle["experiments"][12]["training_output"]).joinpath("best_skill.md").write_text("Changed after seal")
    with pytest.raises(ValueError, match="Pinned file changed"):
        configs.validate_final_test_request(_manifest_path(bundle_inputs), "medical", "initial")


def test_rejects_unfinished_run_without_workflow_summary(bundle_inputs):
    bundle = _generate(bundle_inputs)
    run = _make_completed(bundle_inputs, bundle, "hotpotqa", "raw")
    (run / "workflow_summary.json").unlink()
    with pytest.raises(FileNotFoundError):
        configs.seal_training_completion(_manifest_path(bundle_inputs), "hotpotqa", "raw")


def test_rejects_startup_contract_with_other_split(bundle_inputs):
    bundle = _generate(bundle_inputs)
    run = _make_completed(bundle_inputs, bundle, "hotpotqa", "raw")
    path = run / "prepared_training_contract.json"
    contract = json.loads(path.read_text())
    contract["split_manifest_sha256"] = "0" * 64
    _json(path, contract)
    with pytest.raises(ValueError, match="startup contract"):
        configs.seal_training_completion(_manifest_path(bundle_inputs), "hotpotqa", "raw")


def test_rejects_tampered_runtime_config_in_dry_run(bundle_inputs):
    bundle = _generate(bundle_inputs)
    path = Path(bundle["agent_config"]["path"])
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="Pinned file changed"):
        configs.run_prepared_final_test(_manifest_path(bundle_inputs), "medical", "initial")


def test_final_commands_never_execute_by_default(bundle_inputs):
    bundle = _generate(bundle_inputs)
    for command in bundle["final_test_commands"]:
        assert "--execute" not in command["dry_run_argv"]
        assert command["execute_argv_after_all_four_complete"] == command["dry_run_argv"] + ["--execute"]


def test_nondivisible_train_batch_is_not_dropped(bundle_inputs):
    path = bundle_inputs["prepared_root"] / "medical" / "split_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"] = {role: {"count": count} for role, count in {"train": 412, "validation": 412, "test": 1238}.items()}
    _json(path, manifest)
    bundle = _generate(bundle_inputs)
    medical = [entry for entry in bundle["experiments"] if entry["dataset"] == "medical"]
    assert all(entry["maximum_skill_updates"] == 21 for entry in medical)
    assert all(entry["actual_counts"]["train"] == 412 for entry in medical)
