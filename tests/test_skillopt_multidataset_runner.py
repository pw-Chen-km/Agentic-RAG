from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rag.agent.config import AgentConfig, OllamaPolicyConfig
from agentic_rag.skillopt.dataloader import AgenticRAGSkillOptDataLoader
from agentic_rag.skillopt.multidataset_prepare import PREPARATION_SPLIT_SCHEMA_VERSION


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_local_qwen_skillopt.py"
    spec = importlib.util.spec_from_file_location("tested_multidataset_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configured_runner(runner, tmp_path, monkeypatch):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    manifest = {
        "schema_version": PREPARATION_SPLIT_SCHEMA_VERSION,
        "dataset": {"subset": "medical", "scope_id": "medical:benchmark_exact:dev"},
        "splits": {name: {"count": count} for name, count in (
            ("train", 412), ("validation", 412), ("test", 1238))},
    }
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest))
    args = SimpleNamespace(
        substrate=tmp_path / "substrate", split_dir=split_dir,
        agent_config=tmp_path / "agent.yaml", skillopt_config=tmp_path / "optimizer.yaml",
        skill=tmp_path / "skill.md", output=tmp_path / "run",
        trajectory_representation="progress_abstracted", dry_run=True,
    )
    args.agent_config.write_text("test agent config")
    args.skillopt_config.write_text("test SkillOpt config")
    args.skill.write_text("test initial skill")
    calls = []
    monkeypatch.setattr(runner, "arguments", lambda: args)
    monkeypatch.setattr(runner.Substrate, "open", lambda path: SimpleNamespace(
        manifest={"dataset": "medical"}, require_scope=lambda scope: calls.append(("scope", scope))))
    monkeypatch.setattr(runner, "validate_prepared_split", lambda *a, **kw: (
        calls.append(("prepared", kw)), {"scope_id": "medical:benchmark_exact:dev", "valid": True})[1])
    policy = OllamaPolicyConfig(model="qwen3.6:35b-a3b-bf16", host="http://127.0.0.1:11435", think=False)
    monkeypatch.setattr(runner.AgentConfig, "from_yaml", lambda path: SimpleNamespace(policy=policy))
    monkeypatch.setattr(runner.SkillDocument, "load", lambda path: SimpleNamespace(fixed_answer_contract="Do not invent evidence."))
    config = {
        "optimizer_model": policy.model, "target_model": policy.model,
        "optimizer_qwen_chat_enable_thinking": True, "train_size": 100,
        "sel_env_num": 50, "test_env_num": 50, "eval_test": True,
        "batch_size": 20, "minibatch_size": 5, "accumulation": 1,
        "num_epochs": 1, "seed": 42, "workers": 1, "analyst_workers": 1,
        "edit_budget": 1,
    }
    monkeypatch.setattr(runner, "load_skillopt_config", lambda path: dict(config))

    def forbidden(*args, **kwargs):
        raise AssertionError("A dry run created a client/trainer or accessed sentence-only sidecars")

    monkeypatch.setattr(runner.EvaluationSidecars, "open", forbidden)
    monkeypatch.setattr(runner, "LocalQwenJudge", forbidden)
    monkeypatch.setattr(runner, "AgenticRAGSkillOptAdapter", forbidden)
    monkeypatch.setattr(runner, "run_skillopt_training", forbidden)
    monkeypatch.setattr(runner, "Client", forbidden)
    return runner, args, config, calls, manifest


def test_dry_run_uses_all_manifest_counts_and_disables_test(configured_runner, capsys):
    runner, args, config, calls, manifest = configured_runner
    runner.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "validated"
    assert result["split_counts"] == {"train": 412, "validation": 412, "test": 1238}
    assert result["effective_training"]["train_size"] == 412
    assert result["effective_training"]["sel_env_num"] == 412
    assert result["effective_training"]["test_env_num"] == 1238
    assert result["effective_training"]["eval_test"] is False
    assert result["effective_training"]["batch_size"] == 20
    assert result["effective_training"]["minibatch_size"] == 5
    assert result["model_clients_created"] is False and result["training_started"] is False
    assert ("prepared", {"require_reference_progress": True}) in calls
    assert not args.output.exists()
    assert result["prepared_training_contract"]["dataset"] == "medical"
    assert len(result["prepared_training_contract"]["initial_skill_sha256"]) == 64


@pytest.mark.parametrize("representation,required", [
    ("raw", False), ("organized", False), ("organized_support_labels", True), ("progress_abstracted", True),
])
def test_new_schema_requires_reference_progress_only_for_labeled_arms(configured_runner, representation, required):
    runner, args, config, calls, manifest = configured_runner
    args.trajectory_representation = representation
    runner.main()
    assert ("prepared", {"require_reference_progress": required}) in calls


def test_new_schema_unready_reference_stops_before_clients(configured_runner, monkeypatch):
    runner, args, config, calls, manifest = configured_runner

    def unavailable(*args, **kwargs):
        assert kwargs["require_reference_progress"] is True
        raise ValueError("reference progress unavailable")

    monkeypatch.setattr(runner, "validate_prepared_split", unavailable)
    with pytest.raises(ValueError, match="reference progress unavailable"):
        runner.main()
    assert not args.output.exists()


def test_actual_new_training_config_still_disables_test(configured_runner, monkeypatch, capsys):
    runner, args, config, calls, manifest = configured_runner
    args.dry_run = False
    monkeypatch.setattr(runner, "LocalQwenJudge", lambda policy: object())
    monkeypatch.setattr(runner, "EpisodeEvaluator", lambda *args, **kwargs: object())
    captured = {}
    monkeypatch.setattr(runner, "AgenticRAGSkillOptAdapter", lambda **kwargs: captured.setdefault("adapter", kwargs))
    monkeypatch.setattr(runner, "run_skillopt_training", lambda cfg, adapter: (captured.update(config=cfg), {"done": True})[1])
    runner.main()
    assert captured["config"]["eval_test"] is False
    assert captured["config"]["train_size"] == 412
    assert captured["adapter"]["sentence_provenance"] is None
    contract = json.loads((args.output / "prepared_training_contract.json").read_text())
    assert contract["trajectory_representation"] == "progress_abstracted"
    assert not (args.output / "workflow_summary.json").exists()
    assert json.loads(capsys.readouterr().out) == {"done": True}


@pytest.mark.parametrize("schema", ["1.0", "1.1", "1.2", "2.0"])
def test_legacy_schema_keeps_existing_lineage_branches(configured_runner, monkeypatch, schema, capsys):
    runner, args, config, calls, manifest = configured_runner
    manifest["schema_version"] = schema
    manifest["dataset"] = {"subset": "hotpotqa", "scope_id": "hotpotqa:benchmark_exact:dev"}
    (args.split_dir / "split_manifest.json").write_text(json.dumps(manifest))
    args.trajectory_representation = "raw"
    monkeypatch.setattr(runner.EvaluationSidecars, "open", lambda path: SimpleNamespace(source_sentence_provenance=[]))
    monkeypatch.setattr(runner, "validate_hotpotqa_smoke_lineage", lambda *args: calls.append(("legacy_smoke",)))
    monkeypatch.setattr(runner, "validate_hotpotqa_provenance_lineage", lambda *args: calls.append(("legacy_provenance",)))
    monkeypatch.setattr(runner, "validate_benchmark_lineage", lambda *args, **kwargs: (
        calls.append(("legacy_generic",)), {"scope_id": "hotpotqa:benchmark_exact:dev"})[1])
    runner.main()
    result = json.loads(capsys.readouterr().out)
    assert result["effective_training"]["eval_test"] is True
    assert not any(call[0] == "prepared" for call in calls)
    expected = "legacy_smoke" if schema in {"1.0", "1.1"} else "legacy_provenance" if schema == "1.2" else "legacy_generic"
    assert (expected,) in calls


def test_legacy_labels_still_require_sentence_provenance(configured_runner, monkeypatch):
    runner, args, config, calls, manifest = configured_runner
    manifest["schema_version"] = "2.0"
    (args.split_dir / "split_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runner.EvaluationSidecars, "open", lambda path: SimpleNamespace(source_sentence_provenance=[]))
    with pytest.raises(ValueError, match="requires deterministic source sentence provenance"):
        runner.main()


@pytest.mark.parametrize("bad_count", [0, -1, True, "412", None])
def test_dry_run_rejects_invalid_manifest_counts(configured_runner, bad_count):
    runner, args, config, calls, manifest = configured_runner
    manifest["splits"]["train"]["count"] = bad_count
    (args.split_dir / "split_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="count must be positive"):
        runner.main()


def test_dry_run_flag_parses_without_constructing_client(runner, monkeypatch, tmp_path):
    argv = ["run_local_qwen_skillopt.py"]
    for flag in ("substrate", "split-dir", "agent-config", "skillopt-config", "skill", "output"):
        argv.extend([f"--{flag}", str(tmp_path / flag)])
    argv.extend(["--trajectory-representation", "raw", "--dry-run"])
    monkeypatch.setattr(sys, "argv", argv)
    assert runner.arguments().dry_run is True


def test_contract_mismatch_rejects_resume_even_in_dry_run(configured_runner):
    runner, args, config, calls, manifest = configured_runner
    args.output.mkdir()
    (args.output / "prepared_training_contract.json").write_text(json.dumps({"different": True}))
    with pytest.raises(ValueError, match="contract changed"):
        runner.main()


def test_old_output_not_upgraded_with_contract(configured_runner):
    runner, args, config, calls, manifest = configured_runner
    args.output.mkdir()
    (args.output / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="has no prepared training contract"):
        runner.main()
    assert not (args.output / "prepared_training_contract.json").exists()


def test_transfer_only_manifest_cannot_train(configured_runner):
    runner, args, config, calls, manifest = configured_runner
    manifest["dataset_role"] = "transfer_only"
    (args.split_dir / "split_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Transfer-only"):
        runner.main()
    assert not args.output.exists()


def test_batch40_reflection8_reaches_actual_adapter(configured_runner, monkeypatch):
    runner, args, config, calls, manifest = configured_runner
    config.update(batch_size=40, minibatch_size=8)
    args.dry_run = False
    monkeypatch.setattr(runner, "LocalQwenJudge", lambda policy: object())
    monkeypatch.setattr(runner, "EpisodeEvaluator", lambda *a, **kw: object())
    captured = {}
    monkeypatch.setattr(runner, "AgenticRAGSkillOptAdapter", lambda **kw: captured.setdefault("adapter", kw))
    monkeypatch.setattr(runner, "run_skillopt_training", lambda cfg, adapter: (captured.update(config=cfg), {})[1])
    runner.main()
    assert captured["config"]["batch_size"] == 40
    assert captured["adapter"]["minibatch_size"] == 8
    assert captured["adapter"]["workers"] == captured["adapter"]["analyst_workers"] == 1


@pytest.mark.parametrize("count,sizes", [(200, [40]*5), (412, [40]*10+[12]), (402, [40]*10+[2])])
def test_batch40_plans_one_pass_without_dropping_tail(tmp_path, count, sizes):
    loader = AgenticRAGSkillOptDataLoader(tmp_path, dataset="medical")
    loader._splits = {"train": [{"id": str(i)} for i in range(count)]}
    plan = loader.plan_train_epoch(epoch=0, steps_per_epoch=len(sizes), accumulation=1, batch_size=40, seed=42)
    assert [b.batch_size for b in plan] == sizes
    ids = [r["id"] for b in plan for r in b.payload]
    assert len(ids) == len(set(ids)) == count


@pytest.mark.parametrize("train_count,last_batch_size", [(412, 12), (402, 2), (198, 18)])
def test_actual_dataloader_preserves_partial_rollout_batch_without_repeating_questions(
    tmp_path, train_count, last_batch_size,
):
    loader = AgenticRAGSkillOptDataLoader(tmp_path, dataset="medical")
    original = [{"id": f"q:{index:04d}"} for index in range(train_count)]
    loader._splits = {"train": list(original)}
    planned_steps = (train_count + 19) // 20
    batches = loader.plan_train_epoch(
        epoch=0, steps_per_epoch=planned_steps, accumulation=1,
        batch_size=20, seed=42,
    )
    assert len(batches) == planned_steps
    assert all(batch.batch_size == 20 for batch in batches[:-1])
    assert batches[-1].batch_size == last_batch_size
    ids = [row["id"] for batch in batches for row in batch.payload]
    assert len(ids) == train_count
    assert len(set(ids)) == train_count
    assert set(ids) == {row["id"] for row in original}
    assert loader._splits["train"] == original
    repeated = loader.plan_train_epoch(
        epoch=0, steps_per_epoch=planned_steps, accumulation=1,
        batch_size=20, seed=42,
    )
    assert [batch.payload for batch in batches] == [batch.payload for batch in repeated]
