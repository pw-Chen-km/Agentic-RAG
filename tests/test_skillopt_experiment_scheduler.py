"""Offline process/endpoint fakes: these tests never run an Agent or an LLM."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from agentic_rag.skillopt import experiment_scheduler as module
from agentic_rag.skillopt.experiment_scheduler import (
    CAPACITY_VERSION, RESUME_VERSION, ExperimentScheduler, ProcessBackend,
    command_sha256, exclusive_scheduler_lock, run_worker,
)


DIGEST = "a" * 64


def dump(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def record(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class FakeBackend(ProcessBackend):
    def __init__(self):
        self.specs = []
        self.live_ids = set()
        self.preflights = []
        self.gates = []
        self.bad_digest = False
        self.raise_spawn = False
        self.forbid_gate = False

    def spawn(self, spec_path):
        if self.raise_spawn:
            raise OSError("pretend process start result was lost")
        spec = json.loads(spec_path.read_text())
        identity = {"pid": 1000 + len(self.specs), "start": str(len(self.specs)), "host": "test"}
        self.specs.append((spec_path, spec, identity))
        self.live_ids.add(identity["pid"])
        return identity

    def live(self, identity):
        if not identity:
            return "unknown"
        return "same" if identity["pid"] in self.live_ids else "dead"

    def preflight(self, lane, model, expected_digest):
        self.preflights.append(lane["lane_id"])
        return {"host": lane["host"], "lane_id": lane["lane_id"], "model": model,
                "actual_model_digest": "b" * 64 if self.bad_digest else expected_digest}

    def final_gate(self, manifest, entry, lane_id):
        self.gates.append((entry["task_id"], lane_id))
        if self.forbid_gate:
            raise ValueError("completion seal changed")

    def validate_capacity(self, manifest, report, concurrency):
        # Shared capacity tests cover resource sampling; this suite mocks it.
        return json.loads(report.read_text())

    def receipt(self, index, code=None):
        path, spec, identity = self.specs[index]
        value = {"version": module.SCHEDULER_VERSION, "token": spec["token"], "task_id": spec["task_id"],
                 "stage": spec["stage"], "command_sha256": spec["command_sha256"], "worker": identity,
                 "state": "running" if code is None else "exited"}
        if code is not None:
            value.update(returncode=code, finished_at="synthetic-time")
            self.live_ids.discard(identity["pid"])
        dump(path.parent / "receipt.json", value)


@pytest.fixture
def setup(tmp_path):
    skill = tmp_path / "initial.md"
    skill.write_text("Do evidence-based retrieval.")
    base = tmp_path / "source.yaml"
    base.write_text("model: {}\n")
    lanes = [{"lane_id": f"lane{i}", "host": f"http://127.0.0.1:{11435 + i}", "gpu_id": 1 if i < 2 else 0,
              "amd_port": 11500 + i} for i in range(4)]
    for lane in lanes:
        path = tmp_path / lane["lane_id"] / "agent.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"policy": {"host": lane["host"]}}))
        lane["agent_config"] = record(path)
    training = []
    for i in range(12):
        variants = []
        for lane in lanes:
            path = tmp_path / "configs" / f"train{i}_{lane['lane_id']}.yaml"
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump({"train": {"batch_size": 40}, "gradient": {"minibatch_size": 8, "analyst_workers": 1},
                                            "env": {"workers": 1}, "model": {"qwen_chat_base_url": lane["host"] + "/v1"}}))
            variants.append({**lane, "skillopt_config": record(path), "train_argv": ["synthetic-python", "train", str(i), lane["lane_id"]]})
        output = tmp_path / "runs" / f"train{i}"
        training.append({"task_id": f"train:{i}", "ready": True, "dataset": "synthetic", "representation": "raw",
                         "training_output": str(output), "completion_path": str(output / "training_completion.json"),
                         "seal_completion_argv": ["synthetic-python", "seal", str(i)], "lane_variants": variants})
    tests = [{"task_id": f"test:{i}", "ready": True, "dataset": "synthetic", "arm": "initial", "source_dataset": None,
              "test_count": 5, "output": str(tmp_path / "runs" / f"test{i}"),
              "lane_variants": [{**lane, "execute_argv": ["synthetic-python", "test", str(i), lane["lane_id"]]} for lane in lanes]}
             for i in range(65)]
    manifest = {"design": "single_source_cross_dataset_v1", "experiments": training, "final_test_commands": tests,
                "lanes": lanes, "expected_model_digest": DIGEST, "target_model": "synthetic-model",
                "eval_test_during_training": False, "target_workers_per_task": 1, "analyst_workers_per_task": 1,
                "train_batch_size": 40, "reflection_minibatch_size": 8,
                "initial_skill": record(skill), "source_optimizer_config": record(base), "source_agent_config": record(base),
                "code_files": [], "core_prompt_files": []}
    path = dump(tmp_path / "manifest.json", manifest)
    capacity = {"version": CAPACITY_VERSION, "measured": True, "execution_manifest_sha256": record(path)["sha256"],
                "workload_sha256": "c" * 64, "model_digest": DIGEST, "context_length": 262144,
                "approved_lane_ids": [lane["lane_id"] for lane in lanes], "approved_concurrency": 4,
                "headroom_fraction_required": .1,
                "stages": [{"concurrency": n, "lane_ids": [lane["lane_id"] for lane in lanes[:n]], "status": "passed", "errors": []}
                           for n in (1, 2, 4)]}
    cap = dump(tmp_path / "capacity.json", capacity)
    backend = FakeBackend()

    def scheduler(n=2, **kwargs):
        return ExperimentScheduler(path, tmp_path / "progress", max_lanes=n, capacity_report=cap, backend=backend, **kwargs)

    return path, manifest, cap, backend, scheduler


def test_plan_is_model_free_and_does_not_create_output(setup):
    path, _, _, backend, _ = setup
    scheduler = ExperimentScheduler(path, path.parent / "unused", backend=backend)
    assert scheduler.plan()["training_tasks"] == 12
    assert scheduler.plan()["final_test_tasks"] == 65
    assert scheduler.plan()["max_lanes"] == 1
    assert not (path.parent / "unused").exists()
    assert not backend.specs and not backend.preflights
    with pytest.raises(ValueError, match="measured capacity"):
        scheduler.tick()
    assert not backend.specs


def test_first_free_dynamic_lanes_fixed_queue_and_no_duplicate_on_restart(setup):
    _, _, _, backend, make = setup
    first = make().tick()
    assert len(backend.specs) == 2
    assert [(spec["task_id"], spec["lane_id"]) for _, spec, _ in backend.specs] == [("train:0", "lane0"), ("train:1", "lane1")]
    assert first["counts"]["starting"] == 2
    backend.receipt(0)
    backend.receipt(1)
    resumed = make().tick()
    assert resumed["counts"]["running"] == 2
    assert len(backend.specs) == 2
    assert backend.preflights == ["lane0", "lane1"]


def test_completion_receipt_resumes_sealing_not_training(setup):
    _, manifest, _, backend, make = setup
    make().tick()
    backend.receipt(0, 0)
    make().tick()
    assert [spec["stage"] for _, spec, _ in backend.specs] == ["train", "train", "seal"]
    backend.receipt(2, 0)
    dump(Path(manifest["experiments"][0]["completion_path"]), {"status": "complete"})
    result = make().tick()
    assert result["tasks"]["train:0"]["state"] == "succeeded"
    assert backend.specs[-1][1]["task_id"] == "train:2"
    assert backend.specs[-1][1]["lane_id"] == "lane0"


def test_failed_training_pauses_new_dispatch_and_is_never_replayed(setup):
    _, _, _, backend, make = setup
    make().tick()
    backend.receipt(0, 1)
    result = make().tick()
    assert result["state"] == "paused"
    assert result["tasks"]["train:0"]["pause"]["code"] == "command_failed_no_automatic_replay"
    make().tick()
    assert len(backend.specs) == 2


@pytest.mark.parametrize("reason", ["dead", "missing_identity", "pid_reused"])
def test_missing_receipt_never_means_command_can_be_replayed(setup, reason):
    path, _, _, backend, make = setup
    scheduler = make()
    scheduler.tick()
    backend.live_ids.clear()
    if reason == "missing_identity":
        status = json.loads(scheduler.status_path.read_text())
        status["tasks"]["train:0"]["attempts"][-1].pop("worker")
        dump(scheduler.status_path, status)
    if reason == "pid_reused":
        backend.live = lambda identity: "reused"
    result = make().tick()
    assert result["tasks"]["train:0"]["state"] == "paused"
    assert len(backend.specs) == 2


def test_dead_wrapper_with_live_child_does_not_free_lane(setup):
    _, _, _, backend, make = setup
    make().tick()
    backend.receipt(0)
    path = backend.specs[0][0].parent / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["child"] = {"pid": 7000, "start": "child"}
    dump(path, receipt)
    backend.live_ids.discard(1000)
    backend.live_ids.add(7000)
    result = make().tick()
    assert result["tasks"]["train:0"]["pause"]["code"] == "worker_disappeared_child_may_still_run"
    assert len(backend.specs) == 2


def test_mismatched_receipt_and_corrupt_receipt_are_not_adopted(setup):
    _, _, _, backend, make = setup
    make().tick()
    backend.receipt(0, 0)
    path = backend.specs[0][0].parent / "receipt.json"
    value = json.loads(path.read_text())
    value["token"] = "different invocation"
    dump(path, value)
    assert make().tick()["tasks"]["train:0"]["pause"]["code"] == "invalid_exit_receipt"
    path.write_text("not json")
    assert make().tick()["tasks"]["train:0"]["pause"]["code"] == "invalid_exit_receipt"
    assert len(backend.specs) == 2


def test_successful_exit_without_completion_artifact_pauses(setup):
    _, _, _, backend, make = setup
    make().tick()
    backend.receipt(0, 0)
    make().tick()
    backend.receipt(2, 0)
    result = make().tick()
    assert result["tasks"]["train:0"]["pause"]["code"] == "success_without_completion_artifact"


def test_preflight_digest_mismatch_stops_without_process_start(setup):
    _, _, _, backend, make = setup
    backend.bad_digest = True
    status = make().tick()
    assert status["state"] == "paused"
    assert status["tasks"]["train:0"]["pause"]["code"] == "preflight_or_gate_failed"
    assert not backend.specs


def test_changed_pinned_config_stops_dispatch(setup):
    _, manifest, _, backend, make = setup
    path = Path(manifest["experiments"][0]["lane_variants"][0]["skillopt_config"]["path"])
    path.write_text(path.read_text() + "changed: true\n")
    assert make().tick()["state"] == "paused"
    assert not backend.specs


def test_changed_manifest_cannot_resume_old_queue(setup):
    path, manifest, cap, _, make = setup
    make().tick()
    manifest["new_value"] = True
    dump(path, manifest)
    value = json.loads(cap.read_text())
    value["execution_manifest_sha256"] = record(path)["sha256"]
    dump(cap, value)
    with pytest.raises(ValueError, match="different manifest"):
        make().tick()


def test_existing_unowned_output_is_not_adopted(setup):
    _, manifest, _, backend, make = setup
    output = Path(manifest["experiments"][0]["training_output"])
    dump(output / "runtime_state.json", {"last_completed_step": 0})
    with pytest.raises(ValueError, match="without this scheduler"):
        make().tick()
    assert not backend.specs


@pytest.mark.parametrize("field,value", [("measured", False), ("approved_concurrency", 1), ("context_length", 32000),
                                          ("model_digest", "b" * 64), ("headroom_fraction_required", .01)])
def test_capacity_requires_measured_matching_scope(setup, field, value):
    _, _, cap, backend, make = setup
    report = json.loads(cap.read_text())
    report[field] = value
    dump(cap, report)
    with pytest.raises(ValueError, match="Capacity report"):
        make()
    assert not backend.specs


def test_four_lane_limit_requires_passed_four_lane_stage(setup):
    _, _, cap, backend, make = setup
    report = json.loads(cap.read_text())
    report["stages"][-1]["status"] = "failed"
    dump(cap, report)
    with pytest.raises(ValueError, match="passed measured"):
        make(4)
    report["stages"][-1]["status"] = "passed"
    dump(cap, report)
    make(4).tick()
    assert len(backend.specs) == 4


def test_manifest_lane_or_output_collision_rejected(setup):
    path, manifest, _, _, _ = setup
    manifest["experiments"][1]["training_output"] = manifest["experiments"][0]["training_output"]
    dump(path, manifest)
    with pytest.raises(ValueError, match="distinct output"):
        ExperimentScheduler(path, path.parent / "progress")


def test_sealed_all_twelve_are_required_before_first_final_dispatch(setup):
    _, manifest, _, backend, make = setup
    scheduler = make()
    scheduler.tick()
    processed = set()
    for _ in range(30):
        for index, (_, spec, _) in enumerate(list(backend.specs)):
            if index in processed:
                continue
            assert spec["stage"] != "test"
            processed.add(index)
            backend.receipt(index, 0)
            if spec["stage"] == "seal":
                entry = next(row for row in manifest["experiments"] if row["task_id"] == spec["task_id"])
                dump(Path(entry["completion_path"]), {"status": "complete"})
        result = make().tick()
        if backend.gates:
            break
    assert all(result["tasks"][f"train:{i}"]["state"] == "succeeded" for i in range(12))
    assert len([spec for _, spec, _ in backend.specs if spec["stage"] == "train"]) == 12
    assert len([spec for _, spec, _ in backend.specs if spec["stage"] == "seal"]) == 12
    assert len(backend.gates) == 2
    # The final gate is a separate required validation, not inferred from status.
    assert [spec["task_id"] for _, spec, _ in backend.specs[-2:]] == ["test:0", "test:1"]


def test_audited_resume_is_explicit_same_lane_and_pinned_checkpoint(setup):
    path, _, _, backend, make = setup
    scheduler = make()
    scheduler.tick()
    backend.receipt(0, 1)
    scheduler.tick()
    checkpoint = dump(path.parent / "checkpoint.json", {"completed_step": 1})
    spec = backend.specs[0][1]
    audit = dump(path.parent / "resume.json", {"version": RESUME_VERSION, "status": "verified_safe_boundary",
                "task_id": "train:0", "execution_manifest_sha256": record(path)["sha256"], "lane_id": "lane0",
                "train_command_sha256": spec["command_sha256"], "safe_to_resume_without_repeating_update": True,
                "checkpoint_files": [record(checkpoint)]})
    scheduler.authorize_training_resume("train:0", audit)
    make().tick()
    assert len(backend.specs) == 3
    assert backend.specs[-1][1]["task_id"] == "train:0"
    assert backend.specs[-1][1]["lane_id"] == "lane0"
    assert backend.specs[-1][1]["command_sha256"] == spec["command_sha256"]
    backend.receipt(2, 1)
    scheduler.tick()
    checkpoint.write_text("changed checkpoint")
    with pytest.raises(ValueError, match="Pinned file"):
        scheduler.authorize_training_resume("train:0", audit)


def test_resume_rejected_when_old_child_alive_or_exit_uncertain(setup):
    path, _, _, backend, make = setup
    scheduler = make()
    scheduler.tick()
    backend.receipt(0, 1)
    scheduler.tick()
    receipt_path = backend.specs[0][0].parent / "receipt.json"
    value = json.loads(receipt_path.read_text())
    value["child"] = {"pid": 7777, "start": "still alive"}
    dump(receipt_path, value)
    backend.live_ids.add(7777)
    with pytest.raises(ValueError, match="still be alive"):
        scheduler.authorize_training_resume("train:0", path.parent / "no-audit.json")
    receipt_path.unlink()
    with pytest.raises(ValueError, match="Missing exit receipt"):
        scheduler.authorize_training_resume("train:0", path.parent / "no-audit.json")


def test_kernel_lock_blocks_second_owner_without_force_or_removal(tmp_path):
    with exclusive_scheduler_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another scheduler"):
            with exclusive_scheduler_lock(tmp_path):
                pass
    # A leftover lock file is harmless once its owner has released the kernel lock.
    with exclusive_scheduler_lock(tmp_path):
        assert (tmp_path / "scheduler.lock").is_file()


def test_worker_claim_and_receipt_prevent_second_execution(tmp_path, monkeypatch):
    calls = []

    class FakeProcess:
        pid = 900

        def __init__(self, argv, **kwargs):
            calls.append(argv)

        def wait(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(module, "process_identity", lambda pid: {"pid": pid, "start": "synthetic"})
    argv = [sys.executable, "synthetic-only"]
    spec = dump(tmp_path / "spec.json", {"token": "one", "task_id": "train:0", "stage": "train", "argv": argv,
                "command_sha256": command_sha256(argv), "cwd": str(tmp_path)})
    assert run_worker(spec) == 0
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["state"] == "exited" and receipt["returncode"] == 0
    assert receipt["child"]["pid"] == 900
    with pytest.raises(FileExistsError):
        run_worker(spec)
    assert len(calls) == 1


def test_spawn_uncertainty_preserves_dispatch_intent_and_pauses(setup):
    _, _, _, backend, make = setup
    backend.raise_spawn = True
    status = make().tick()
    task = status["tasks"]["train:0"]
    assert task["state"] == "paused"
    assert len(task["attempts"]) == 1
    assert Path(task["attempts"][0]["spec_path"]).is_file()
    backend.raise_spawn = False
    make().tick()
    assert not backend.specs


def test_active_lanes_follow_measured_stage_not_manifest_prefix(setup):
    _, _, cap, backend, make = setup
    report = json.loads(cap.read_text())
    report["stages"][1]["lane_ids"] = ["lane2", "lane3"]
    dump(cap, report)
    make().tick()
    assert [spec["lane_id"] for _, spec, _ in backend.specs] == ["lane2", "lane3"]
    with pytest.raises(ValueError, match="remove an assigned lane"):
        make(1).tick()


def test_shared_capacity_validator_is_not_bypassed_on_execute(setup):
    _, _, _, backend, make = setup
    scheduler = make()

    def fail(*args):
        raise ValueError("resource samples do not support the claimed approval")

    backend.validate_capacity = fail
    with pytest.raises(ValueError, match="resource samples"):
        scheduler.tick()
    assert not backend.specs


def test_all_cells_then_single_durable_offline_aggregation(setup):
    path, manifest, cap, backend, make = setup
    summary_path = path.parent / "comparison" / "summary.json"
    manifest.update(aggregation_argv=["synthetic-python", "aggregate"], aggregation_summary_path=str(summary_path))
    dump(path, manifest)
    report = json.loads(cap.read_text())
    report["execution_manifest_sha256"] = record(path)["sha256"]
    dump(cap, report)
    make().tick()
    processed = set()
    for _ in range(100):
        for index, (_, spec, _) in enumerate(list(backend.specs)):
            if index in processed:
                continue
            processed.add(index)
            backend.receipt(index, 0)
            if spec["stage"] == "seal":
                entry = next(row for row in manifest["experiments"] if row["task_id"] == spec["task_id"])
                dump(Path(entry["completion_path"]), {"status": "complete"})
            elif spec["stage"] == "test":
                entry = next(row for row in manifest["final_test_commands"] if row["task_id"] == spec["task_id"])
                dump(Path(entry["output"]) / "summary.json", {"complete": True})
            elif spec["stage"] == "aggregate":
                assert len([row for _, row, _ in backend.specs if row["stage"] == "test"]) == 65
                dump(summary_path, {"final_task_count": 65, "additional_judge_calls": 0,
                                    "execution_manifest": {"sha256": record(path)["sha256"]}})
        status = make().tick()
        if status["state"] == "complete":
            break
    assert status["state"] == "complete"
    assert len([spec for _, spec, _ in backend.specs if spec["stage"] == "aggregate"]) == 1
    assert len(backend.preflights) == 12 + 65  # Sealing/aggregation never contact a model endpoint.
    before = len(backend.specs)
    assert make().tick()["state"] == "complete"
    assert len(backend.specs) == before


def test_final_gate_failure_is_not_turned_into_a_success(setup):
    _, _, _, backend, make = setup
    scheduler = make()
    scheduler.tick()
    status = json.loads(scheduler.status_path.read_text())
    # Model the already-sealed earlier phase; the independent gate still must run.
    for key in scheduler.train_ids:
        status["tasks"][key]["state"] = "succeeded"
    dump(scheduler.status_path, status)
    backend.forbid_gate = True
    result = make().tick()
    assert result["state"] == "paused"
    assert result["tasks"]["test:0"]["state"] == "paused"
    assert not any(spec["stage"] == "test" for _, spec, _ in backend.specs)


def test_worker_launch_error_records_failure_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "process_identity", lambda pid: {"pid": pid, "start": "synthetic"})

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("Synthetic command missing")

    monkeypatch.setattr(module.subprocess, "Popen", unavailable)
    argv = ["synthetic-missing-command"]
    spec = dump(tmp_path / "spec.json", {"token": "two", "task_id": "train:0", "stage": "train", "argv": argv,
                "command_sha256": command_sha256(argv), "cwd": str(tmp_path)})
    assert run_worker(spec) == 125
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["state"] == "worker_error"
    assert receipt["error_type"] == "FileNotFoundError"
    assert "Synthetic command missing" not in json.dumps(receipt)


@pytest.mark.parametrize("loaded,match", [
    ([{"name": "gemma", "digest": "b" * 64}], "Another model"),
    ([{"name": "qwen", "digest": DIGEST, "context_length": 32000, "size": 100, "size_vram": 100}], "context"),
    ([{"name": "qwen", "digest": DIGEST, "context_length": 262144, "size": 100, "size_vram": 99}], "CPU offload"),
    ([{"name": "qwen", "digest": DIGEST, "context_length": 262144, "size": 100}], "CPU offload"),
])
def test_runtime_preflight_protects_loaded_models_and_gpu(monkeypatch, loaded, match):
    calls = []

    class Response:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.value).encode()

    def get(request, timeout):
        calls.append((request.full_url, request.method))
        return Response({"models": [{"name": "qwen", "digest": DIGEST}]} if request.full_url.endswith("/api/tags") else {"models": loaded})

    monkeypatch.setattr(module.urllib.request, "urlopen", get)
    with pytest.raises(ValueError, match=match):
        ProcessBackend().preflight({"host": "http://localhost:1234", "lane_id": "lane0"}, "qwen", DIGEST)
    assert calls == [("http://localhost:1234/api/tags", "GET"), ("http://localhost:1234/api/ps", "GET")]


@pytest.mark.parametrize("loaded", [[], [{"name": "qwen", "digest": DIGEST, "context_length": 262144, "size": 100, "size_vram": 100}]])
def test_runtime_preflight_allows_empty_or_verified_correct_model(monkeypatch, loaded):
    import io

    def get(request, timeout):
        value = {"models": [{"name": "qwen", "digest": DIGEST}]} if request.full_url.endswith("/api/tags") else {"models": loaded}
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", get)
    audit = ProcessBackend().preflight({"host": "http://localhost:1234", "lane_id": "lane0"}, "qwen", DIGEST)
    assert audit["actual_model_digest"] == DIGEST
    assert audit["gpu_residency"] == ("verified_full" if loaded else "not_loaded_yet")


def test_empty_unmeasured_capacity_report_denies_even_single_lane(setup, monkeypatch):
    path, _, cap, _, _ = setup
    dump(cap, {"version": CAPACITY_VERSION, "measured": False, "approved_lane_ids": [], "approved_concurrency": 0})

    def no_network(*args, **kwargs):
        raise AssertionError("An unmeasured approval must fail before any network request")

    monkeypatch.setattr(module.urllib.request, "urlopen", no_network)
    with pytest.raises(ValueError, match="Capacity approval"):
        ExperimentScheduler(path, path.parent / "progress", max_lanes=1, capacity_report=cap)
