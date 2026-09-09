"""Capacity checks use fake endpoints only: no network, model, or GPU access."""

import copy
import hashlib
import json
import time

import pytest

from agentic_rag.skillopt.amd_capacity import (
    CONTEXT_LENGTH, MODEL, _AMD_SNAPSHOT_SCRIPT, _sha, capacity_gate,
    run_capacity_test, validate_capacity_report, validate_capacity_workload,
)


DIGEST = "a" * 64
LANES = [
    {"lane_id": "gpu0a", "gpu_id": 0, "amd_port": 11434, "host": "http://lane0a"},
    {"lane_id": "gpu0b", "gpu_id": 0, "amd_port": 11436, "host": "http://lane0b"},
    {"lane_id": "gpu1a", "gpu_id": 1, "amd_port": 11437, "host": "http://lane1a"},
    {"lane_id": "gpu1b", "gpu_id": 1, "amd_port": 11438, "host": "http://lane1b"},
]


def snapshot(*, gemma=False):
    services = {}
    for lane in LANES:
        services[str(lane["amd_port"])] = {
            "env": {"OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1",
                    "OLLAMA_CONTEXT_LENGTH": str(CONTEXT_LENGTH), "ROCR_VISIBLE_DEVICES": str(lane["gpu_id"])},
            "active_connections": 0,
            "models": [{"name": MODEL, "digest": DIGEST, "size": 70, "size_vram": 70,
                        "context_length": CONTEXT_LENGTH}],
        }
    if gemma:
        services["11434"]["models"][0].update(name="gemma", digest="b" * 64)
    return {"services": services, "gpus": {str(i): {"total_bytes": 192, "used_bytes": 100} for i in (0, 1)},
            "memory": {"limit_bytes": 400, "used_bytes": 200}}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return path


@pytest.fixture
def prepared(tmp_path):
    requests = []
    for kind in ("target", "reflection"):
        payload = {"model": MODEL, "stream": False,
                   "messages": [{"role": "user", "content": f"Actual saved {kind} fixture"}]}
        if kind == "target":
            payload.update(think=False, format={"type": "object"}, options={"num_ctx": CONTEXT_LENGTH, "temperature": 0, "num_predict": 100})
        else:
            payload.update(temperature=0, max_tokens=16384, chat_template_kwargs={"enable_thinking": True})
        requests.append({"kind": kind, "path": "/api/chat" if kind == "target" else "/v1/chat/completions",
                         "payload": payload, "prompt_sha256": _sha(payload["messages"]),
                         "metadata": {"split": "train", "reflection_minibatch_size": 8,
                                      "episode_ids": [f"id{i}" for i in range(8)]}})
    evidence = write_json(tmp_path / "tokenization.json", [{"prompt_sha256": row["prompt_sha256"], "input_tokens": 100} for row in requests])
    records = {row["prompt_sha256"]: {"prompt_sha256": row["prompt_sha256"], "input_tokens": 100,
        "method": "llama_server_apply_template_and_tokenize", "model_digest": DIGEST,
        "evidence_path": str(evidence), "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()} for row in requests}
    workload = {"version": "amd-capacity-workload-v1", "requests": requests, "tokenization": records}
    manifest = {"expected_model_digest": DIGEST, "lanes": LANES}
    return write_json(tmp_path / "manifest.json", manifest), write_json(tmp_path / "workload.json", workload), workload


def no_call(*args, **kwargs):
    raise AssertionError("Must not call an endpoint")


def fake_post(url, payload):
    time.sleep(0.005)
    if url.endswith("/api/chat"):
        return {"message": {"content": "{}"}, "prompt_eval_count": 100, "eval_count": 20, "done_reason": "stop"}
    return {"choices": [{"message": {"content": '{"edits": []}', "reasoning": "reason"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def run_fake(prepared, **kwargs):
    manifest, workload, _ = prepared
    return run_capacity_test(manifest, workload, snapshot_provider=lambda: snapshot(),
        get_provider=lambda _: {"models": [{"name": MODEL, "digest": DIGEST}]},
        post_provider=fake_post, execute=True, sample_interval=0.01, **kwargs)


def test_default_dry_run_has_zero_calls_and_only_gpu1(prepared):
    manifest, workload, _ = prepared
    report = run_capacity_test(manifest, workload, snapshot_provider=no_call, get_provider=no_call, post_provider=no_call)
    assert report["model_calls"] == 0
    assert report["measured"] is False
    assert report["planned_lane_groups"] == [["gpu1a"], ["gpu1a", "gpu1b"]]


def test_raw_eight_prompt_over_context_blocks_even_dry_run(prepared):
    manifest, workload, data = prepared
    key = data["requests"][1]["prompt_sha256"]
    data["tokenization"][key]["input_tokens"] = 322982
    # The overlong reflection must be diagnosed even if a target has no output limit.
    del data["requests"][0]["payload"]["options"]["num_predict"]
    write_json(workload, data)
    for execute in (False, True):
        report = run_capacity_test(manifest, workload, execute=execute,
            snapshot_provider=no_call, get_provider=no_call, post_provider=no_call)
        assert report["measured"] is False
        assert report["approved_concurrency"] == report["model_calls"] == 0
        assert "322982" in report["preflight_errors"][0]


def test_input_plus_output_context_overflow_blocked(prepared):
    manifest, workload, data = prepared
    data["tokenization"][data["requests"][1]["prompt_sha256"]]["input_tokens"] = CONTEXT_LENGTH - 100
    write_json(workload, data)
    report = run_capacity_test(manifest, workload, execute=True, snapshot_provider=no_call)
    assert "plus output budget exceeds context" in report["preflight_errors"][0]


@pytest.mark.parametrize("change", ["missing", "hash", "digest", "method"])
def test_invalid_token_evidence_blocks_without_calls(prepared, change):
    manifest, workload, data = prepared
    if change == "missing":
        data.pop("tokenization")
    else:
        record = next(iter(data["tokenization"].values()))
        record[{"hash": "evidence_sha256", "digest": "model_digest", "method": "method"}[change]] = "invalid"
    write_json(workload, data)
    report = run_capacity_test(manifest, workload, execute=True, snapshot_provider=no_call)
    assert report["preflight_errors"] and report["model_calls"] == 0


def test_actual_ollama_token_count_certificate_accepted(prepared):
    manifest, workload, data = prepared
    next(iter(data["tokenization"].values()))["method"] = "ollama_prompt_eval_count"
    write_json(workload, data)
    assert not run_capacity_test(manifest, workload).get("preflight_errors")


@pytest.mark.parametrize("path,value", [("split", "test"), ("reflection_minibatch_size", 5), ("episode_ids", ["same"] * 8)])
def test_reflection_must_be_eight_distinct_train_trajectories(prepared, path, value):
    data = prepared[2]
    data["requests"][1]["metadata"][path] = value
    with pytest.raises(ValueError):
        validate_capacity_workload(data)


def test_saved_prompt_hash_and_dynamic_schema_are_required(prepared):
    data = prepared[2]
    data["requests"][0]["payload"]["messages"][0]["content"] += "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_capacity_workload(data)
    data["requests"][0]["prompt_sha256"] = _sha(data["requests"][0]["payload"]["messages"])
    data["requests"][0]["payload"].pop("format")
    with pytest.raises(ValueError, match="actual dynamic schema"):
        validate_capacity_workload(data)


def test_gpu1_allowed_but_gpu0_foreign_model_protected():
    assert not capacity_gate(LANES[2:], snapshot(gemma=True), DIGEST, require_loaded=True, require_idle=True)
    errors = capacity_gate(LANES, snapshot(gemma=True), DIGEST)
    assert "foreign_model_on_shared_gpu_port_11434" in errors


@pytest.mark.parametrize("area,field,value,expected", [
    ("memory", "used_bytes", 361, "cgroup_memory_headroom_below_10_percent"),
    ("memory", "limit_bytes", None, "cgroup_memory_measurement_unavailable"),
    ("gpu", "used_bytes", 173, "gpu_1_headroom_below_10_percent"),
    ("gpu", "used_bytes", -1, "gpu_1_memory_measurement_unavailable"),
    ("env", "OLLAMA_CONTEXT_LENGTH", "8192", "service_environment_mismatch_11437"),
    ("env", "OLLAMA_NUM_PARALLEL", "4", "service_environment_mismatch_11437"),
    ("env", "ROCR_VISIBLE_DEVICES", "0", "service_environment_mismatch_11437"),
    ("model", "size_vram", 69, "cpu_offload_or_unknown_vram_11437"),
    ("model", "context_length", 8192, "wrong_context_11437"),
    ("service", "active_connections", 1, "lane_not_confirmed_idle_11437"),
])
def test_live_gate_checks_resources_and_exact_runtime(area, field, value, expected):
    data = snapshot()
    target = {"memory": data["memory"], "gpu": data["gpus"]["1"],
              "env": data["services"]["11437"]["env"], "model": data["services"]["11437"]["models"][0],
              "service": data["services"]["11437"]}[area]
    target[field] = value
    assert expected in capacity_gate([LANES[2]], data, DIGEST, require_loaded=True, require_idle=True)


def test_snapshot_script_parses_null_delimited_environment():
    assert ".split('\\0')" in _AMD_SNAPSHOT_SCRIPT
    assert ".split('\\\\0')" not in _AMD_SNAPSHOT_SCRIPT
    compile(_AMD_SNAPSHOT_SCRIPT, "<snapshot>", "exec")


def test_measured_stages_have_equal_work_and_exclude_warmup(prepared, tmp_path):
    report = run_fake(prepared)
    assert report["approved_concurrency"] == 2
    assert report["approved_lane_ids"] == ["gpu1a", "gpu1b"]
    assert report["model_calls"] == 22
    for stage in report["stages"]:
        assert stage["status"] == "passed"
        assert len([row for row in stage["requests"] if not row["warmup"]]) == 8
        assert len([row for row in stage["requests"] if row["warmup"]]) == 2 * stage["concurrency"]
        assert stage["sampled_minimum_headroom"]["cgroup_memory_free_fraction"] == 0.5
    path = write_json(tmp_path / "report.json", report)
    approved = validate_capacity_report(prepared[0], path, 2)
    assert [row["lane_id"] for row in approved["approved_lanes"]] == ["gpu1a", "gpu1b"]


def test_four_lane_stage_does_not_touch_active_gemma(prepared):
    touched = []
    def post(url, payload):
        touched.append(url)
        return fake_post(url, payload)
    report = run_capacity_test(prepared[0], prepared[1], execute=True, allowed_gpu_ids=[0, 1],
        snapshot_provider=lambda: snapshot(gemma=True), post_provider=post,
        get_provider=lambda _: {"models": [{"name": MODEL, "digest": DIGEST}]})
    assert report["approved_concurrency"] == 2
    assert report["stages"][-1]["status"] == "blocked"
    assert not any("lane0" in url for url in touched)


def test_actual_token_count_mismatch_never_approves(prepared):
    def post(url, payload):
        response = fake_post(url, payload)
        if "prompt_eval_count" in response:
            response["prompt_eval_count"] = 97
        return response
    report = run_capacity_test(prepared[0], prepared[1], execute=True, snapshot_provider=lambda: snapshot(),
        post_provider=post, get_provider=lambda _: {"models": [{"name": MODEL, "digest": DIGEST}]})
    assert report["approved_concurrency"] == 0
    assert "truncation or template mismatch" in report["stages"][0]["errors"][0]


@pytest.mark.parametrize("change", ["manifest", "workload", "sample", "count", "speed", "unmeasured"])
def test_scheduler_revalidates_saved_capacity_evidence(prepared, tmp_path, change):
    report = run_fake(prepared)
    if change == "manifest":
        data = json.loads(prepared[0].read_text())
        data["changed"] = True
        write_json(prepared[0], data)
    elif change == "workload":
        data = json.loads(prepared[1].read_text())
        data["changed"] = True
        write_json(prepared[1], data)
    elif change == "sample":
        report["stages"][1]["samples"][-1]["memory"]["used_bytes"] = 399
    elif change == "count":
        report["stages"][1]["requests"].pop()
    elif change == "speed":
        report["stages"][1]["measured_seconds"] = report["stages"][0]["measured_seconds"]
    else:
        report["measured"] = False
    path = write_json(tmp_path / "report.json", report)
    with pytest.raises(ValueError):
        validate_capacity_report(prepared[0], path, 2)


def set_reflection_size(prepared, manifest_size, workload_size):
    manifest_path, workload_path, data = prepared
    manifest = json.loads(manifest_path.read_text())
    manifest["reflection_minibatch_size"] = manifest_size
    request = data["requests"][1]
    request["metadata"]["reflection_minibatch_size"] = workload_size
    request["metadata"]["episode_ids"] = [f"train-{i}" for i in range(workload_size)]
    write_json(manifest_path, manifest)
    write_json(workload_path, data)


def test_five_reflections_match_manifest_and_are_recorded(prepared, tmp_path):
    set_reflection_size(prepared, 5, 5)
    assert len(validate_capacity_workload(prepared[2], expected_reflection_minibatch_size=5)) == 2
    report = run_fake(prepared)
    assert report["reflection_minibatch_size"] == 5
    assert report["approved_concurrency"] == 2
    path = write_json(tmp_path / "capacity5.json", report)
    assert validate_capacity_report(prepared[0], path, 2)["reflection_minibatch_size"] == 5


@pytest.mark.parametrize("manifest_size,workload_size", [(5, 8), (8, 5), (5, 4), (5, 6)])
def test_declared_batch_does_not_override_manifest(prepared, manifest_size, workload_size):
    set_reflection_size(prepared, manifest_size, workload_size)
    with pytest.raises(ValueError, match="matching the execution manifest"):
        run_capacity_test(prepared[0], prepared[1], execute=True,
            snapshot_provider=no_call, get_provider=no_call, post_provider=no_call)


@pytest.mark.parametrize("ids", [["a", "b", "c", "d"], ["a", "b", "c", "d", "d"], ["a", "b", "c", "d", ""]])
def test_declared_five_needs_five_distinct_nonempty_ids(prepared, ids):
    set_reflection_size(prepared, 5, 5)
    prepared[2]["requests"][1]["metadata"]["episode_ids"] = ids
    with pytest.raises(ValueError, match="5 distinct"):
        validate_capacity_workload(prepared[2], expected_reflection_minibatch_size=5)


@pytest.mark.parametrize("size", [None, 0, -5, True, "5", 5.0])
def test_manifest_reflection_size_is_not_coerced(prepared, size):
    manifest = json.loads(prepared[0].read_text())
    manifest["reflection_minibatch_size"] = size
    write_json(prepared[0], manifest)
    with pytest.raises(ValueError, match="positive reflection minibatch"):
        run_capacity_test(prepared[0], prepared[1])


def test_legacy_standalone_validation_keeps_eight_default(prepared):
    assert validate_capacity_workload(prepared[2])
    set_reflection_size(prepared, 5, 5)
    with pytest.raises(ValueError, match="8 distinct"):
        validate_capacity_workload(prepared[2])


def test_five_report_cannot_claim_old_batch_eight(prepared, tmp_path):
    set_reflection_size(prepared, 5, 5)
    report = run_fake(prepared)
    report["reflection_minibatch_size"] = 8
    path = write_json(tmp_path / "capacity5_wrong_summary.json", report)
    with pytest.raises(ValueError, match="pinned execution requirements"):
        validate_capacity_report(prepared[0], path, 1)


def install_actual_openai_certificate(prepared, tmp_path, *, tokens=103):
    request = prepared[2]["requests"][1]
    response = {"choices": [{"message": {"content": '{"edits": []}', "reasoning": "why"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": tokens, "completion_tokens": 20}}
    receipt = {"request_sha256": _sha(request["payload"]), "response": response}
    path = write_json(tmp_path / "actual_openai_warmup.json", receipt)
    record = prepared[2]["tokenization"][request["prompt_sha256"]]
    record.update(method="openai_usage_prompt_tokens", request_sha256=_sha(request["payload"]),
        evidence_path=str(path), evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), input_tokens=tokens)
    write_json(prepared[1], prepared[2])
    return request, record, path, receipt


def test_actual_openai_count_uses_receipt_not_guessed_template_offset(prepared, tmp_path):
    set_reflection_size(prepared, 5, 5)
    install_actual_openai_certificate(prepared, tmp_path, tokens=103)
    report = run_capacity_test(prepared[0], prepared[1], snapshot_provider=no_call, get_provider=no_call)
    assert not report.get("preflight_errors")
    observed = {row["prompt_sha256"]: row for row in report["tokenization_evidence"]}
    assert observed[prepared[2]["requests"][1]["prompt_sha256"]]["input_tokens"] == 103
    assert report["model_calls"] == 0


@pytest.mark.parametrize("change", ["request", "receipt_request", "count", "truncated", "empty", "missing_usage"])
def test_actual_openai_certificate_rejects_nonmatching_receipt(prepared, tmp_path, change):
    request, record, path, receipt = install_actual_openai_certificate(prepared, tmp_path)
    if change == "request":
        request["payload"]["max_tokens"] -= 1
    elif change == "receipt_request":
        receipt["request_sha256"] = "b" * 64
    elif change == "count":
        record["input_tokens"] += 3
    elif change == "truncated":
        receipt["response"]["choices"][0]["finish_reason"] = "length"
    elif change == "empty":
        receipt["response"]["choices"][0]["message"]["content"] = ""
    else:
        receipt["response"].pop("usage")
    write_json(path, receipt)
    record["evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(prepared[1], prepared[2])
    report = run_capacity_test(prepared[0], prepared[1], execute=True, snapshot_provider=no_call, get_provider=no_call)
    assert report["preflight_errors"]
    assert report["model_calls"] == report["approved_concurrency"] == 0


def set_adaptive_workload(prepared, group_sizes=(2, 3)):
    set_reflection_size(prepared, 5, 5)
    manifest_path, workload_path, data = prepared
    manifest = json.loads(manifest_path.read_text())
    manifest.update(reflection_grouping="adaptive", reflection_token_budget={
        "enabled": True, "context_tokens": CONTEXT_LENGTH, "output_reserve_tokens": 16384,
        "safety_margin_tokens": 1024, "input_token_limit": 244736, "max_minibatch_size": 5,
        "oversized_single_trajectory": "stop_without_truncation"})
    original = copy.deepcopy(data["requests"][1])
    original_ids = original["metadata"]["episode_ids"]
    data["reflection_partition"] = {"mode": "adaptive", "max_minibatch_size": 5,
                                    "original_episode_ids": original_ids}
    requests = [data["requests"][0]]
    start = 0
    for size in group_sizes:
        request = copy.deepcopy(original)
        ids = original_ids[start:start + size]
        request["metadata"]["episode_ids"] = ids
        request["payload"]["messages"] = [{"role": "user", "content": "Complete fixture records " + json.dumps(ids)}]
        request["prompt_sha256"] = _sha(request["payload"]["messages"])
        requests.append(request)
        start += size
    data["requests"] = requests
    evidence_path = write_json(workload_path.parent / "adaptive_tokens.json", [
        {"prompt_sha256": request["prompt_sha256"], "input_tokens": 100} for request in requests])
    data["tokenization"] = {request["prompt_sha256"]: {
        "prompt_sha256": request["prompt_sha256"], "input_tokens": 100,
        "method": "llama_server_apply_template_and_tokenize", "model_digest": DIGEST,
        "evidence_path": str(evidence_path), "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
    } for request in requests}
    write_json(manifest_path, manifest)
    write_json(workload_path, data)
    return data, manifest


@pytest.mark.parametrize("sizes", [(2, 3), (3, 2), (1, 2, 2), (1, 1, 1, 1, 1), (5,)])
def test_adaptive_groups_keep_all_original_five(prepared, sizes):
    data, _ = set_adaptive_workload(prepared, sizes)
    assert validate_capacity_workload(data, expected_reflection_minibatch_size=5, expected_reflection_grouping="adaptive")
    report = run_capacity_test(prepared[0], prepared[1], snapshot_provider=no_call, get_provider=no_call)
    assert not report.get("preflight_errors")
    assert report["reflection_grouping"] == "adaptive"
    assert report["reflection_partition"]["original_episode_ids"] == [f"train-{i}" for i in range(5)]
    assert report["model_calls"] == 0


@pytest.mark.parametrize("change", ["missing_id", "repeated_id", "unknown_id", "reordered_id", "original_missing", "wrong_max", "empty_group", "oversized_group"])
def test_adaptive_groups_cannot_drop_duplicate_or_reorder_records(prepared, change):
    data, _ = set_adaptive_workload(prepared)
    if change == "missing_id":
        data["requests"][2]["metadata"]["episode_ids"].pop()
    elif change == "repeated_id":
        data["requests"][2]["metadata"]["episode_ids"][0] = "train-0"
    elif change == "unknown_id":
        data["requests"][2]["metadata"]["episode_ids"][0] = "not-original"
    elif change == "reordered_id":
        data["requests"][2]["metadata"]["episode_ids"].reverse()
    elif change == "original_missing":
        data["reflection_partition"]["original_episode_ids"].pop()
    elif change == "wrong_max":
        data["reflection_partition"]["max_minibatch_size"] = 8
    elif change == "empty_group":
        data["requests"][1]["metadata"]["episode_ids"] = []
    else:
        data["requests"][1]["metadata"]["episode_ids"] = [f"train-{i}" for i in range(6)]
    write_json(prepared[1], data)
    with pytest.raises(ValueError):
        run_capacity_test(prepared[0], prepared[1], execute=True, snapshot_provider=no_call, get_provider=no_call)


def test_adaptive_partition_requires_explicit_manifest_opt_in(prepared):
    _, manifest = set_adaptive_workload(prepared)
    manifest.pop("reflection_grouping")
    manifest.pop("reflection_token_budget")
    write_json(prepared[0], manifest)
    with pytest.raises(ValueError, match="Fixed execution manifests"):
        run_capacity_test(prepared[0], prepared[1])


def test_adaptive_manifest_requires_complete_partition(prepared):
    data, _ = set_adaptive_workload(prepared)
    data.pop("reflection_partition")
    write_json(prepared[1], data)
    with pytest.raises(ValueError, match="original reflection partition"):
        run_capacity_test(prepared[0], prepared[1])


@pytest.mark.parametrize("field,value", [("enabled", False), ("context_tokens", 8192), ("output_reserve_tokens", 8192),
    ("safety_margin_tokens", 0), ("input_token_limit", 262144), ("max_minibatch_size", 8),
    ("oversized_single_trajectory", "truncate")])
def test_adaptive_budget_is_fully_pinned(prepared, field, value):
    _, manifest = set_adaptive_workload(prepared)
    manifest["reflection_token_budget"][field] = value
    write_json(prepared[0], manifest)
    with pytest.raises(ValueError, match="must pin context"):
        run_capacity_test(prepared[0], prepared[1])


def test_adaptive_preflight_reserves_output_plus_safety_margin(prepared):
    data, _ = set_adaptive_workload(prepared)
    certificate = data["tokenization"][data["requests"][1]["prompt_sha256"]]
    certificate["input_tokens"] = 244737
    write_json(prepared[1], data)
    report = run_capacity_test(prepared[0], prepared[1], execute=True, snapshot_provider=no_call)
    assert "safety margin exceeds context" in report["preflight_errors"][0]
    assert report["model_calls"] == 0


def test_adaptive_output_cannot_be_shortened_for_capacity_pass(prepared):
    data, _ = set_adaptive_workload(prepared)
    data["requests"][1]["payload"]["max_tokens"] = 1
    write_json(prepared[1], data)
    report = run_capacity_test(prepared[0], prepared[1], execute=True, snapshot_provider=no_call)
    assert "output budget does not match" in report["preflight_errors"][0]
    assert report["model_calls"] == 0


def test_adaptive_stages_keep_equal_total_work_and_bind_report(prepared, tmp_path):
    set_adaptive_workload(prepared)
    report = run_fake(prepared)
    assert report["approved_concurrency"] == 2
    assert report["model_calls"] == 33
    for stage in report["stages"]:
        assert len([row for row in stage["requests"] if row["warmup"] is False]) == 12
        assert len([row for row in stage["requests"] if row["warmup"] is True]) == 3 * stage["concurrency"]
    path = write_json(tmp_path / "adaptive_report.json", report)
    assert validate_capacity_report(prepared[0], path, 2)["reflection_grouping"] == "adaptive"
    report["reflection_partition"]["original_episode_ids"].reverse()
    write_json(path, report)
    with pytest.raises(ValueError, match="recorded reflection partition"):
        validate_capacity_report(prepared[0], path, 2)


def test_adaptive_report_cannot_lower_safety_margin(prepared, tmp_path):
    set_adaptive_workload(prepared)
    report = run_fake(prepared)
    report["reflection_safety_margin_tokens"] = 0
    path = write_json(tmp_path / "unsafe_report.json", report)
    with pytest.raises(ValueError, match="pinned execution requirements"):
        validate_capacity_report(prepared[0], path, 1)
