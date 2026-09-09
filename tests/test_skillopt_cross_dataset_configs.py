from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

import agentic_rag.skillopt.multidataset_configs as configs
from test_skillopt_multidataset_configs import _json, _manifest_path, bundle_inputs


LANES = [
    {"lane_id": "gpu1_a", "host": "http://127.0.0.1:11437", "amd_port": 11437, "gpu_id": 1},
    {"lane_id": "gpu1_b", "host": "http://127.0.0.1:11438", "amd_port": 11438, "gpu_id": 1},
    {"lane_id": "gpu0_a", "host": "http://127.0.0.1:11436", "amd_port": 11436, "gpu_id": 0},
    {"lane_id": "gpu0_b", "host": "http://127.0.0.1:11439", "amd_port": 11434, "gpu_id": 0},
]


@pytest.fixture
def cross_inputs(bundle_inputs):
    totals = {"hotpotqa": 1000, "medical": 2062, "novel": 2010, "musique": 1000, "2wikimultihop": 991}
    for dataset, total in totals.items():
        folder = bundle_inputs["prepared_root"] / dataset
        split = json.loads((folder / "split_manifest.json").read_text())
        size = total // 5
        split["splits"] = {role: {"count": count} for role, count in {
            "train": size, "validation": size, "test": total - 2 * size}.items()}
        _json(folder / "split_manifest.json", split)
        training = dataset in configs.TRAINING_DATASETS
        _json(folder / "readiness.json", {
            "dataset": dataset, "eligible_count": total, "split_status": "ready",
            "all_four_arms_ready": training, "reference_progress_ready": training,
            "training_ready": training, "inference_ready": True,
            "mapping_issue_count": 0 if training else 1000,
        })
    return bundle_inputs


def generate(inputs, **kwargs):
    return configs.write_experiment_configs(
        **{key: value for key, value in inputs.items() if key != "calls"},
        design=configs.CROSS_DATASET_DESIGN, lanes=LANES, **kwargs,
    )


def complete(inputs, bundle, source, representation, *, lane_id="gpu1_a"):
    entry = next(row for row in bundle["experiments"]
                 if row["source_dataset"] == source and row["representation"] == representation)
    lane = next(row for row in entry["lane_variants"] if row["lane_id"] == lane_id)
    run = Path(entry["training_output"])
    _json(run / "prepared_training_contract.json", {
        "dataset": source, "trajectory_representation": representation,
        "split_manifest_sha256": bundle["datasets"][source]["split_manifest"]["sha256"],
        "skillopt_config_sha256": lane["skillopt_config"]["sha256"],
        "agent_config_sha256": lane["agent_config"]["sha256"],
        "initial_skill_sha256": bundle["initial_skill"]["sha256"],
    })
    _json(run / "workflow_summary.json", {"dataset": source, "trainer_summary": {"best_step": 1},
                                         "unique_question_count": sum(entry["actual_counts"].values())})
    module = Path(configs.__file__)
    prompt = module.parent / "prompts" / "ranking.md"
    _json(run / "skillopt_integration.json", {
        "code_sha256": {module.name: hashlib.sha256(module.read_bytes()).hexdigest()},
        "prompt_sha256": {prompt.name: hashlib.sha256(prompt.read_bytes()).hexdigest()},
    })
    (run / "best_skill.md").write_text(f"Learned only from {source}; {representation}.")
    return configs.seal_training_completion(_manifest_path(inputs), source, representation)


def complete_all(inputs, bundle):
    for source in configs.TRAINING_DATASETS:
        for arm in configs.REPRESENTATIONS:
            complete(inputs, bundle, source, arm)


def test_cross_bundle_has_exact_twelve_sources_and_sixty_five_matrix_cells(cross_inputs):
    bundle = generate(cross_inputs)
    assert bundle["version"] == configs.CROSS_DATASET_BUNDLE_VERSION
    assert bundle["training_task_count"] == len(bundle["experiments"]) == 12
    assert {entry["source_dataset"] for entry in bundle["experiments"]} == {"hotpotqa", "medical", "novel"}
    assert bundle["final_test_task_count"] == len(bundle["final_test_commands"]) == 65
    assert len({row["task_id"] for row in bundle["final_test_commands"]}) == 65
    assert len({row["output"] for row in bundle["final_test_commands"]}) == 65
    assert bundle["final_inference_episode_count"] == 55107
    for target in configs.DATASETS:
        cells = [row for row in bundle["final_test_commands"] if row["target_dataset"] == target]
        assert len(cells) == 13
        assert sum(row["arm"] == "initial" for row in cells) == 1
        assert all(row["source_dataset"] is None for row in cells if row["arm"] == "initial")
    assert bundle["expected_model_digest"] == configs.MODEL_DIGEST
    assert bundle["model_calls"] == 0


def test_cross_training_variants_pin_lane_and_40_8_single_worker_controls(cross_inputs):
    bundle = generate(cross_inputs)
    assert bundle["reflection_minibatch_size"] == 8
    expected_updates = {"hotpotqa": 5, "medical": 11, "novel": 11}
    for entry in bundle["experiments"]:
        assert entry["maximum_skill_updates"] == expected_updates[entry["dataset"]]
        assert len(entry["lane_variants"]) == 4
        assert len({row["skillopt_config"]["sha256"] for row in entry["lane_variants"]}) == 4
        for variant in entry["lane_variants"]:
            optimizer = yaml.safe_load(Path(variant["skillopt_config"]["path"]).read_text())
            agent = yaml.safe_load(Path(variant["agent_config"]["path"]).read_text())
            assert optimizer["train"]["batch_size"] == 40
            assert optimizer["train"]["accumulation"] == optimizer["train"]["num_epochs"] == 1
            assert optimizer["gradient"]["minibatch_size"] == 8
            assert optimizer["gradient"]["analyst_workers"] == optimizer["env"]["workers"] == 1
            assert optimizer["env"]["durable_checkpoint_enabled"] is True
            assert optimizer["evaluation"]["eval_test"] is False
            assert optimizer["gradient"]["merge_batch_size"] == 3
            assert agent["policy"]["host"] == variant["host"]
            assert agent["policy"]["think"] is False and agent["policy"]["temperature"] == 0
            assert agent["policy"]["num_ctx"] == 262144
            assert agent["agent"]["show_available_action_options"] is True
            assert agent["agent"]["use_state_conditioned_schema"] is True
            for prefix in ("", "optimizer_", "target_"):
                assert optimizer["model"][prefix + "qwen_chat_base_url"] == variant["host"] + "/v1"
            assert str(Path(entry["training_output"])) in variant["train_argv"]
    assert bundle["lanes"][-1]["host"].endswith(":11439")
    assert bundle["lanes"][-1]["amd_port"] == 11434


def test_explicit_reflection_five_is_pinned_in_every_lane_without_changing_rollout(cross_inputs):
    bundle = generate(cross_inputs, reflection_minibatch_size=5)
    assert bundle["reflection_minibatch_size"] == 5
    assert bundle["train_batch_size"] == 40
    assert bundle["training_task_count"] == 12 and bundle["final_test_task_count"] == 65
    assert bundle["final_inference_episode_count"] == 55107
    assert "reflection_grouping" not in bundle and "reflection_token_budget" not in bundle
    assert generate(cross_inputs, reflection_minibatch_size=5) == bundle
    for entry in bundle["experiments"]:
        for variant in entry["lane_variants"]:
            cfg = yaml.safe_load(Path(variant["skillopt_config"]["path"]).read_text())
            assert cfg["gradient"]["minibatch_size"] == 5
            assert cfg["train"]["batch_size"] == 40
            assert cfg["gradient"]["analyst_workers"] == cfg["env"]["workers"] == 1
            assert cfg["train"]["accumulation"] == cfg["train"]["num_epochs"] == 1
            assert "reflection_token_budget_enabled" not in cfg["env"]
    # This is a new frozen experiment, never an in-place conversion of 8 to 5.
    before = _manifest_path(cross_inputs).read_bytes()
    with pytest.raises(FileExistsError):
        generate(cross_inputs, reflection_minibatch_size=8)
    assert _manifest_path(cross_inputs).read_bytes() == before


def test_explicit_token_budget_pins_identical_limits_and_lane_tokenizers(cross_inputs):
    bundle = generate(cross_inputs, reflection_minibatch_size=5, reflection_token_budget_enabled=True)
    expected = {"enabled": True, "context_tokens": 262144, "output_reserve_tokens": 16384,
                "safety_margin_tokens": 1024, "input_token_limit": 244736,
                "max_minibatch_size": 5, "oversized_single_trajectory": "stop_without_truncation"}
    assert bundle["reflection_grouping"] == "adaptive" and bundle["reflection_token_budget"] == expected
    assert bundle["train_batch_size"] == 40 and bundle["reflection_minibatch_size"] == 5
    assert bundle["training_task_count"] == 12 and bundle["final_test_task_count"] == 65
    for entry in bundle["experiments"]:
        configs._cross_entry(bundle, entry["dataset"], entry["representation"])
        for variant in entry["lane_variants"]:
            cfg = yaml.safe_load(Path(variant["skillopt_config"]["path"]).read_text())
            assert cfg["train"]["batch_size"] == 40 and cfg["gradient"]["minibatch_size"] == 5
            assert cfg["env"]["reflection_token_budget_enabled"] is True
            assert cfg["env"]["reflection_context_tokens"] == 262144
            assert cfg["env"]["reflection_output_reserve_tokens"] == 16384
            assert cfg["env"]["reflection_safety_margin_tokens"] == 1024
            assert cfg["env"]["reflection_tokenizer_amd_port"] == variant["amd_port"]
            assert cfg["env"]["reflection_tokenizer_ssh_host"] == "root@140.116.240.181"
            assert cfg["env"]["reflection_tokenizer_ssh_port"] == 45026
            assert cfg["env"]["reflection_tokenizer_ssh_key"] == "/home/jj/.ssh/amd_root_key"
            assert cfg["model"]["qwen_chat_max_tokens"] == cfg["model"]["optimizer_qwen_chat_max_tokens"] == 16384
    assert generate(cross_inputs, reflection_minibatch_size=5, reflection_token_budget_enabled=True) == bundle


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_token_budget_requires_boolean_opt_in(cross_inputs, value):
    with pytest.raises(ValueError, match="must be an explicit boolean"):
        generate(cross_inputs, reflection_minibatch_size=5, reflection_token_budget_enabled=value)
    assert not cross_inputs["output"].exists()


def test_token_budget_requires_explicit_max_five_cross_design(cross_inputs):
    with pytest.raises(ValueError, match="requires the cross-dataset design.*=5"):
        generate(cross_inputs, reflection_token_budget_enabled=True)
    with pytest.raises(ValueError, match="requires the cross-dataset design.*=5"):
        configs.write_experiment_configs(
            **{key: value for key, value in cross_inputs.items() if key != "calls"},
            reflection_minibatch_size=5, reflection_token_budget_enabled=True)
    assert not cross_inputs["output"].exists()


def test_enabling_token_budget_cannot_overwrite_existing_fixed_bundle(cross_inputs):
    bundle = generate(cross_inputs, reflection_minibatch_size=5)
    manifest = _manifest_path(cross_inputs).read_bytes()
    first_config = Path(bundle["experiments"][0]["lane_variants"][0]["skillopt_config"]["path"])
    original = first_config.read_bytes()
    with pytest.raises(FileExistsError):
        generate(cross_inputs, reflection_minibatch_size=5, reflection_token_budget_enabled=True)
    assert _manifest_path(cross_inputs).read_bytes() == manifest
    assert first_config.read_bytes() == original


def test_source_cannot_silently_enable_token_budget_without_explicit_opt_in(cross_inputs):
    path = cross_inputs["optimizer_config"]
    source = yaml.safe_load(path.read_text())
    source.setdefault("env", {})["reflection_token_budget_enabled"] = True
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="requires explicit generation opt-in"):
        generate(cross_inputs, reflection_minibatch_size=5)
    assert not cross_inputs["output"].exists()


def test_manifest_token_budget_limits_and_lane_mapping_are_validated(cross_inputs):
    bundle = generate(cross_inputs, reflection_minibatch_size=5, reflection_token_budget_enabled=True)
    bundle["reflection_token_budget"]["safety_margin_tokens"] = 0
    with pytest.raises(ValueError, match="Invalid frozen adaptive"):
        configs._cross_entry(bundle, "hotpotqa", "raw")
    bundle["reflection_token_budget"]["safety_margin_tokens"] = 1024
    bundle["experiments"][0]["lane_variants"][0]["amd_port"] = 1
    with pytest.raises(ValueError, match="tokenizer lane differs"):
        configs._cross_entry(bundle, "hotpotqa", "raw")


@pytest.mark.parametrize("value", [0, -1, 41, True, 5.0, "5", None])
def test_invalid_reflection_minibatch_rejected_before_writing(cross_inputs, value):
    with pytest.raises(ValueError, match="reflection_minibatch_size must be an integer"):
        generate(cross_inputs, reflection_minibatch_size=value)
    assert not cross_inputs["output"].exists()


def test_reflection_override_cannot_silently_change_legacy_design(cross_inputs):
    with pytest.raises(ValueError, match="override requires the cross-dataset design"):
        configs.write_experiment_configs(
            **{key: value for key, value in cross_inputs.items() if key != "calls"},
            reflection_minibatch_size=5,
        )
    assert not cross_inputs["output"].exists()


def test_frozen_reflection_metadata_must_match_lane_config(cross_inputs):
    bundle = generate(cross_inputs, reflection_minibatch_size=5)
    bundle["reflection_minibatch_size"] = 8
    with pytest.raises(ValueError, match="differs from the execution manifest"):
        configs._cross_entry(bundle, "hotpotqa", "raw")


def test_config_cli_accepts_explicit_reflection_minibatch_five(cross_inputs, capsys):
    import importlib.util

    script = Path(configs.__file__).resolve().parents[3] / "scripts" / "write_multidataset_skillopt_configs.py"
    spec = importlib.util.spec_from_file_location("test_write_cross_configs_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lanes = cross_inputs["output"].parent / "lanes.json"
    lanes.write_text(json.dumps(LANES))
    args = []
    for key in ("prepared_root", "initial_skill", "optimizer_config", "agent_config", "output"):
        args.extend(["--" + key.replace("_", "-"), str(cross_inputs[key])])
    module.main(args + ["--design", configs.CROSS_DATASET_DESIGN, "--lanes", str(lanes),
                        "--reflection-minibatch-size", "5"])
    result = json.loads(capsys.readouterr().out)
    assert result["reflection_minibatch_size"] == 5 and result["model_calls"] == 0
    assert json.loads(_manifest_path(cross_inputs).read_text())["reflection_minibatch_size"] == 5


def test_config_cli_accepts_explicit_token_budget_flag(cross_inputs, capsys):
    import importlib.util

    script = Path(configs.__file__).resolve().parents[3] / "scripts" / "write_multidataset_skillopt_configs.py"
    spec = importlib.util.spec_from_file_location("test_write_adaptive_configs_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lanes = cross_inputs["output"].parent / "lanes.json"
    lanes.write_text(json.dumps(LANES))
    args = []
    for key in ("prepared_root", "initial_skill", "optimizer_config", "agent_config", "output"):
        args.extend(["--" + key.replace("_", "-"), str(cross_inputs[key])])
    module.main(args + ["--design", configs.CROSS_DATASET_DESIGN, "--lanes", str(lanes),
                        "--reflection-minibatch-size", "5", "--reflection-token-budget-enabled"])
    result = json.loads(capsys.readouterr().out)
    assert result["reflection_grouping"] == "adaptive" and result["model_calls"] == 0
    assert json.loads(_manifest_path(cross_inputs).read_text())["reflection_token_budget"]["max_minibatch_size"] == 5


def test_transfer_targets_need_no_reference_progress_or_own_training(cross_inputs):
    bundle = generate(cross_inputs)
    for dataset in ("musique", "2wikimultihop"):
        report = bundle["datasets"][dataset]
        assert report["inference_ready"] and not report["training_ready"]
        assert report["reference_progress_ready"] is False
        calls = [call for call in cross_inputs["calls"] if str(Path(call[0]).name) == dataset]
        assert calls and all(call[2] is False for call in calls)
    contract = configs.run_prepared_final_test(
        _manifest_path(cross_inputs), "musique", "raw", source_dataset="hotpotqa", lane_id="gpu1_b")
    assert contract["reference_progress_required"] is False
    assert contract["selected_skill"] is None
    assert contract["test_count"] == 600
    assert contract["model_calls"] == 0 and contract["dry_run"] is True
    assert not (cross_inputs["output"] / "final_test").exists()


def test_every_initial_and_transfer_execute_waits_for_all_twelve_completions(cross_inputs):
    bundle = generate(cross_inputs)
    for arm in configs.REPRESENTATIONS:
        complete(cross_inputs, bundle, "hotpotqa", arm)
    with pytest.raises(ValueError, match="All twelve training runs"):
        configs.validate_final_test_request(_manifest_path(cross_inputs), "musique", "initial", lane_id="gpu1_a")
    with pytest.raises(ValueError, match="All twelve training runs"):
        configs.validate_final_test_request(_manifest_path(cross_inputs), "musique", "raw", source_dataset="hotpotqa", lane_id="gpu1_a")
    assert not (cross_inputs["output"] / "final_test").exists()


def test_cross_test_uses_source_best_but_target_corpus_and_profile(cross_inputs):
    bundle = generate(cross_inputs)
    complete_all(cross_inputs, bundle)
    contract = configs.validate_final_test_request(
        _manifest_path(cross_inputs), "2wikimultihop", "organized",
        source_dataset="medical", lane_id="gpu1_b")
    assert contract["all_twelve_completed"] is True
    assert contract["dataset"] == contract["target_dataset"] == "2wikimultihop"
    assert contract["source_dataset"] == "medical"
    assert "training_runs/medical/organized/best_skill.md" in contract["selected_skill"]["path"]
    assert Path(contract["substrate"]).name == "2wikimultihop"
    assert Path(contract["split_dir"]).name == "2wikimultihop"
    assert contract["test_count"] == 595
    assert contract["lane_id"] == "gpu1_b" and contract["endpoint"].endswith(":11438")
    assert contract["source_training_completion"]["lane_id"] == "gpu1_a"
    assert contract["expected_model_digest"] == configs.MODEL_DIGEST


def test_sealer_resolves_chosen_lane_from_startup_hashes(cross_inputs):
    bundle = generate(cross_inputs)
    seal = complete(cross_inputs, bundle, "novel", "raw", lane_id="gpu0_b")
    assert seal["lane_id"] == "gpu0_b" and seal["endpoint"].endswith(":11439")
    assert seal["source_dataset"] == "novel"
    assert seal["test_metrics_used_for_sealing"] is False


def test_all_twelve_seals_and_best_hashes_remain_checked_for_initial(cross_inputs):
    bundle = generate(cross_inputs)
    complete_all(cross_inputs, bundle)
    best = Path(bundle["experiments"][-1]["training_output"]) / "best_skill.md"
    best.write_text("changed after sealing")
    with pytest.raises(ValueError, match="Pinned file changed"):
        configs.validate_final_test_request(_manifest_path(cross_inputs), "musique", "initial", lane_id="gpu1_a")


def test_initial_cannot_be_duplicated_per_source(cross_inputs):
    generate(cross_inputs)
    with pytest.raises(ValueError, match="Initial baseline has no source"):
        configs.run_prepared_final_test(_manifest_path(cross_inputs), "novel", "initial", source_dataset="medical")


def test_trained_cross_skill_requires_explicit_source_dataset(cross_inputs):
    generate(cross_inputs)
    with pytest.raises(ValueError, match="explicit source training dataset"):
        configs.run_prepared_final_test(_manifest_path(cross_inputs), "novel", "raw")


def test_execute_requires_explicit_lane_even_after_all_training_completed(cross_inputs):
    bundle = generate(cross_inputs)
    complete_all(cross_inputs, bundle)
    with pytest.raises(ValueError, match="explicit scheduler-approved --lane-id"):
        configs.validate_final_test_request(_manifest_path(cross_inputs), "novel", "initial")


def test_cross_configs_are_idempotent_but_changed_lane_is_not_overwritten(cross_inputs):
    first = generate(cross_inputs)
    assert generate(cross_inputs) == first
    lanes = [dict(row) for row in LANES]
    lanes[0]["host"] = "http://127.0.0.1:12000"
    with pytest.raises(FileExistsError):
        configs.write_experiment_configs(
            **{key: value for key, value in cross_inputs.items() if key != "calls"},
            design=configs.CROSS_DATASET_DESIGN, lanes=lanes)


@pytest.mark.parametrize("lanes", [None, [], [LANES[0], LANES[0]],
                                  [{**LANES[0], "host": "http://secret:password@localhost:11437"}]])
def test_invalid_or_implicit_lane_configuration_rejected_before_output(cross_inputs, lanes):
    with pytest.raises(ValueError):
        configs.write_experiment_configs(
            **{key: value for key, value in cross_inputs.items() if key != "calls"},
            design=configs.CROSS_DATASET_DESIGN, lanes=lanes)
    assert not cross_inputs["output"].exists()


def test_model_digest_contract_requires_full_hash(cross_inputs):
    with pytest.raises(ValueError, match="full SHA-256"):
        configs.write_experiment_configs(
            **{key: value for key, value in cross_inputs.items() if key != "calls"},
            design=configs.CROSS_DATASET_DESIGN, lanes=LANES, model_digest="short")
    assert not cross_inputs["output"].exists()


def test_matrix_command_variants_are_offline_until_execute(cross_inputs):
    bundle = generate(cross_inputs)
    for task in bundle["final_test_commands"]:
        for variant in task["lane_variants"]:
            assert "--execute" not in variant["dry_run_argv"]
            assert variant["execute_argv"] == variant["dry_run_argv"] + ["--execute"]
            assert "--lane-id" in variant["dry_run_argv"]
            assert ("--source-dataset" in variant["dry_run_argv"]) == (task["source_dataset"] is not None)


@pytest.fixture
def saved_small_matrix(cross_inputs):
    """Only synthetic saved outputs; never instantiate a Target or Judge."""
    from agentic_rag.paths import portable_path_component

    for dataset in configs.DATASETS:
        folder = cross_inputs["prepared_root"] / dataset
        split = json.loads((folder / "split_manifest.json").read_text())
        split["splits"] = {role: {"count": 3} for role in ("train", "validation", "test")}
        _json(folder / "split_manifest.json", split)
        items = [{"id": f"{dataset}_{i}", "question": f"question {i}", "answer": f"gold {i}", "scope_id": dataset}
                 for i in range(3)]
        (folder / "test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in items))
    bundle = generate(cross_inputs)
    complete_all(cross_inputs, bundle)
    for task in bundle["final_test_commands"]:
        output = Path(task["output"])
        contract = configs.validate_final_test_request(
            _manifest_path(cross_inputs), task["target_dataset"], task["arm"],
            source_dataset=task["source_dataset"], lane_id="gpu1_a")
        _json(output / "final_test_contract.json", contract)
        _json(output / "summary.json", {"complete": True, "count": 3, "task_id": task["task_id"]})
        _json(output / "execution_timing" / "1.json", {"elapsed_seconds": 6, "batch_returned": True})
        for i in range(3):
            score = int(i == 0 or (task["source_dataset"] is not None and i == 1))
            identifier, answer = f"{task['target_dataset']}_{i}", f"answer {i}"
            episode = output / "predictions" / portable_path_component(identifier)
            _json(episode / "rollout_result.json", {
                "id": identifier, "question": f"question {i}", "dataset": task["target_dataset"],
                "hard": score, "metrics": {"llm_acc": score, "contain_acc": 0},
                "metric_contract": {"hard": "llm_acc"}, "predicted_answer": answer,
                "termination_reason": "finish", "n_turns": 2,
                "usage": {"policy_calls": 2, "total_tokens": 100, "retrieved_tokens": 40},
                "judge_usage": {"calls": 1, "total_tokens": 10},
            })
            _json(episode / "evaluation.json", {"question": f"question {i}", "predicted_answer": answer,
                                               "gold_answer": f"gold {i}", "llm_acc": score})
    return cross_inputs, bundle


def test_saved_matrix_aggregates_actual_binary_judge_and_paired_difference(saved_small_matrix):
    inputs, bundle = saved_small_matrix
    result = configs.aggregate_final_test_results(_manifest_path(inputs), bootstrap_samples=100)
    assert result["additional_judge_calls"] == 0
    assert result["primary_metric"] == "saved_binary_qwen_llm_acc"
    assert result["final_task_count"] == 65 and result["inference_episode_count"] == 195
    assert len(result["comparisons_vs_initial"]) == 60 and len(result["macro_average"]) == 12
    comparison = result["comparisons_vs_initial"][0]
    assert comparison["initial_accuracy"] == pytest.approx(1 / 3)
    assert comparison["learned_accuracy"] == pytest.approx(2 / 3)
    assert comparison["learned_only"] == 1 and comparison["initial_only"] == 0
    assert comparison["both_correct"] == comparison["both_wrong"] == 1
    stats = result["cells"][0]
    assert stats["usage_totals"]["total_tokens"] == 300
    assert stats["judge_tokens"] == 30 and stats["recorded_runner_seconds_per_question"] == 2
    assert Path(bundle["aggregation_summary_path"]).is_file()
    assert bundle["aggregation_argv"][-1] == "--aggregate"
    assert configs.aggregate_final_test_results(_manifest_path(inputs), bootstrap_samples=100) == result


def test_incomplete_final_matrix_is_not_reported_as_complete(saved_small_matrix):
    inputs, bundle = saved_small_matrix
    summary = Path(bundle["final_test_commands"][-1]["output"]) / "summary.json"
    _json(summary, {"complete": False, "count": 2})
    with pytest.raises(ValueError, match="not completely saved"):
        configs.aggregate_final_test_results(_manifest_path(inputs), bootstrap_samples=100)
    assert not (inputs["output"] / "comparisons").exists()


def test_aggregator_rejects_wrong_gold_or_nonbinary_metric(saved_small_matrix):
    inputs, bundle = saved_small_matrix
    result_path = next((Path(bundle["final_test_commands"][-1]["output"]) / "predictions").glob("*/rollout_result.json"))
    value = json.loads(result_path.read_text())
    value["metrics"]["llm_acc"] = 0.5
    _json(result_path, value)
    with pytest.raises(ValueError, match="saved binary Qwen"):
        configs.aggregate_final_test_results(_manifest_path(inputs), bootstrap_samples=100)
    value["metrics"]["llm_acc"] = value["hard"]
    _json(result_path, value)
    evaluation = result_path.parent / "evaluation.json"
    value = json.loads(evaluation.read_text())
    value["gold_answer"] = "a different gold"
    _json(evaluation, value)
    with pytest.raises(ValueError, match="frozen question or gold answer"):
        configs.aggregate_final_test_results(_manifest_path(inputs), bootstrap_samples=100)


def test_paired_bootstrap_keeps_normalized_duplicate_questions_together():
    ci, samples = configs._paired_bootstrap([" Q  one", "q one ", "other"], [1, -1, 0], seed=42, samples=100)
    assert ci == [0.0, 0.0] and set(samples) == {0.0}
