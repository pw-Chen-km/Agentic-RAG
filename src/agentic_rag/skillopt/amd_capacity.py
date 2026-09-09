"""Measured capacity gates for existing AMD Ollama lanes; no service changes.

Only ``execute=True`` sends generation requests. The workload is a pinned replay
of real training requests, not synthetic short prompts. Every timed stage runs
four identical workload copies, excluding model warmup. Observed resource minima
are sampled checks, not a claim that an unsampled instantaneous peak is known.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CAPACITY_VERSION = "amd-capacity-v1"
WORKLOAD_VERSION = "amd-capacity-workload-v1"
CONTEXT_LENGTH = 262144
MIN_HEADROOM = 0.10
MIN_SPEEDUP = 1.10
MODEL = "qwen3.6:35b-a3b-bf16"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _load(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    source = Path(path).resolve()
    raw = source.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {source}")
    return source, value, hashlib.sha256(raw).hexdigest()


def _reflection_minibatch_size(manifest: Mapping[str, Any]) -> int:
    """Old capacity manifests predate this field and represented batches of 8."""
    size = manifest.get("reflection_minibatch_size", 8)
    if type(size) is not int or size <= 0:
        raise ValueError("Execution manifest must pin a positive reflection minibatch size")
    return size


def _reflection_grouping(manifest: Mapping[str, Any]) -> str:
    mode = manifest.get("reflection_grouping", "fixed")
    if mode not in {"fixed", "adaptive"}:
        raise ValueError("Execution manifest has unsupported reflection grouping")
    return mode


def _adaptive_budget(manifest: Mapping[str, Any]) -> tuple[int, int]:
    """New adaptive runs must pin all admission limits; legacy runs need none."""
    if _reflection_grouping(manifest) == "fixed":
        if isinstance(manifest.get("reflection_token_budget"), dict) and manifest["reflection_token_budget"].get("enabled"):
            raise ValueError("Fixed grouping cannot silently enable adaptive reflection")
        return 0, 0
    budget = manifest.get("reflection_token_budget")
    expected = {"context_tokens": CONTEXT_LENGTH, "output_reserve_tokens": 16384,
                "safety_margin_tokens": 1024, "input_token_limit": 244736,
                "max_minibatch_size": _reflection_minibatch_size(manifest)}
    if (not isinstance(budget, dict) or budget.get("enabled") is not True
            or budget.get("oversized_single_trajectory") != "stop_without_truncation"
            or any(type(budget.get(key)) is not int or budget[key] != value for key, value in expected.items())):
        raise ValueError("Adaptive reflection must pin context 262144, output reserve 16384, and safety margin 1024")
    return expected["output_reserve_tokens"], expected["safety_margin_tokens"]


def validate_capacity_workload(
    workload: Mapping[str, Any], *, expected_reflection_minibatch_size: int = 8,
    expected_reflection_grouping: str = "fixed",
) -> list[dict[str, Any]]:
    """Match actual workload size to its execution manifest, not caller metadata.

    The default preserves the original standalone validation API. Runtime and
    saved-report validation always pass the execution manifest's pinned value.
    """
    if type(expected_reflection_minibatch_size) is not int or expected_reflection_minibatch_size <= 0:
        raise ValueError("Expected reflection minibatch size must be a positive integer")
    if expected_reflection_grouping not in {"fixed", "adaptive"}:
        raise ValueError("Expected reflection grouping must be fixed or adaptive")
    partition = workload.get("reflection_partition")
    if expected_reflection_grouping == "adaptive":
        if (not isinstance(partition, dict) or partition.get("mode") != "adaptive"
                or type(partition.get("max_minibatch_size")) is not int
                or partition["max_minibatch_size"] != expected_reflection_minibatch_size):
            raise ValueError("Adaptive capacity needs the original reflection partition matching the manifest")
        original_ids = partition.get("original_episode_ids")
        if (not isinstance(original_ids, list)
                or any(not isinstance(identifier, str) or not identifier for identifier in original_ids)
                or len(original_ids) != expected_reflection_minibatch_size
                or len(set(original_ids)) != expected_reflection_minibatch_size):
            raise ValueError("Adaptive capacity must retain every unique ID from the original full minibatch")
    elif partition is not None:
        raise ValueError("Fixed execution manifests cannot approve an adaptive workload")
    if workload.get("version") != WORKLOAD_VERSION:
        raise ValueError("Unsupported capacity workload version")
    requests = workload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Capacity needs recorded target and full-minibatch reflection requests")
    kinds = set()
    reflected_ids = []
    for request in requests:
        kind, path, payload = request.get("kind"), request.get("path"), request.get("payload")
        if kind not in {"target", "reflection"} or not isinstance(payload, dict):
            raise ValueError("Unsupported capacity request")
        kinds.add(kind)
        if payload.get("model") != MODEL or payload.get("stream") is not False:
            raise ValueError("Capacity replay must explicitly use pinned Qwen and non-streaming responses")
        if not isinstance(payload.get("messages"), list) or not payload["messages"]:
            raise ValueError("Capacity replay is missing actual recorded messages")
        if request.get("prompt_sha256") != _sha(payload["messages"]):
            raise ValueError("Recorded capacity prompt hash mismatch")
        metadata = request.get("metadata", {})
        if metadata.get("split") != "train":
            raise ValueError("Capacity workload must not use validation/test examples")
        if kind == "target":
            if (path != "/api/chat" or payload.get("think") is not False
                    or not isinstance(payload.get("format"), dict)
                    or payload.get("options", {}).get("num_ctx") != CONTEXT_LENGTH
                    or payload.get("options", {}).get("temperature") != 0):
                raise ValueError("Target probe needs actual dynamic schema, no thinking, and full context")
        else:
            effort = payload.get("reasoning_effort")
            enabled = payload.get("chat_template_kwargs", {}).get("enable_thinking")
            if enabled is None:
                enabled = payload.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking")
            if path != "/v1/chat/completions" or payload.get("temperature") != 0 or not (effort in {"low", "medium", "high"} or enabled is True):
                raise ValueError("Reflection probe must preserve native thinking-enabled OpenAI request")
            ids = metadata.get("episode_ids")
            if (type(metadata.get("reflection_minibatch_size")) is not int
                    or metadata["reflection_minibatch_size"] != expected_reflection_minibatch_size
                    or not isinstance(ids, list)
                    or any(not isinstance(identifier, str) or not identifier for identifier in ids)
                    or not 1 <= len(ids) <= expected_reflection_minibatch_size
                    or len(set(ids)) != len(ids)
                    or (expected_reflection_grouping == "fixed" and len(ids) != expected_reflection_minibatch_size)):
                raise ValueError(f"Reflection probe must contain {expected_reflection_minibatch_size} distinct actual training trajectories, matching the execution manifest")
            reflected_ids.extend(ids)
    if kinds != {"target", "reflection"}:
        raise ValueError("Both target and reflection workloads are required")
    if expected_reflection_grouping == "adaptive" and reflected_ids != original_ids:
        raise ValueError("Adaptive reflection requests must cover all original IDs exactly once and preserve their order")
    return requests


def _http_json(url: str, payload: Mapping[str, Any] | None = None, timeout: float = 600) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Connection": "close"}
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("Model endpoint did not return a JSON object")
    return result


_AMD_SNAPSHOT_SCRIPT = r'''
import json,pathlib,subprocess,urllib.request
keys={'OLLAMA_NUM_PARALLEL','OLLAMA_MAX_LOADED_MODELS','OLLAMA_CONTEXT_LENGTH','OLLAMA_HOST','ROCR_VISIBLE_DEVICES'}
services={}
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if (p/'comm').read_text().strip()!='ollama':continue
  env=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
  host=env.get('OLLAMA_HOST','')
  if not host:continue
  port=int(host.rsplit(':',1)[1])
  with urllib.request.urlopen('http://127.0.0.1:%d/api/ps'%port,timeout=5) as r: models=json.load(r).get('models',[])
  connections=subprocess.run(['ss','-tnH','state','established','( sport = :%d )'%port],capture_output=True,text=True,check=True).stdout.splitlines()
  services[str(port)]={'pid':int(p.name),'env':{k:env[k] for k in keys if k in env},'models':models,'active_connections':len(connections)}
 except (OSError,ValueError,subprocess.SubprocessError):pass
raw=subprocess.run(['rocm-smi','--showmeminfo','vram','--showuse','--json'],capture_output=True,text=True,check=True)
gpu_data=json.loads(raw.stdout)
gpus={name.removeprefix('card'):{'total_bytes':int(row['VRAM Total Memory (B)']),'used_bytes':int(row['VRAM Total Used Memory (B)']),'utilization_percent':float(row['GPU use (%)'])} for name,row in gpu_data.items() if name.startswith('card')}
memory={'limit_bytes':int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text()),'used_bytes':int(pathlib.Path('/sys/fs/cgroup/memory.current').read_text())}
print(json.dumps({'gpus':gpus,'memory':memory,'services':services}))
'''


def ssh_snapshot(ssh_prefix: Sequence[str]) -> dict[str, Any]:
    """Read existing AMD container resources; caller supplies an authorized SSH prefix."""
    import shlex
    command = list(ssh_prefix) + ["python3 -c " + shlex.quote(_AMD_SNAPSHOT_SCRIPT)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    snapshot = json.loads(result.stdout)
    snapshot["observed_at"] = datetime.now(timezone.utc).isoformat()
    return snapshot


def capacity_gate(
    lanes: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any], model_digest: str,
    *, require_loaded: bool = False, require_idle: bool = False,
) -> list[str]:
    """Return concrete blockers. A missing measurement is never treated as zero."""
    errors = []
    memory = snapshot.get("memory", {})
    limit, used = memory.get("limit_bytes"), memory.get("used_bytes")
    if type(limit) is not int or type(used) is not int or limit <= 0 or used < 0:
        errors.append("cgroup_memory_measurement_unavailable")
    elif (limit - used) / limit < MIN_HEADROOM:
        errors.append("cgroup_memory_headroom_below_10_percent")
    selected_gpus = {str(lane["gpu_id"]) for lane in lanes}
    for gpu in selected_gpus:
        info = snapshot.get("gpus", {}).get(gpu, {})
        total, allocated = info.get("total_bytes"), info.get("used_bytes")
        if type(total) is not int or type(allocated) is not int or total <= 0 or allocated < 0:
            errors.append(f"gpu_{gpu}_memory_measurement_unavailable")
        elif (total - allocated) / total < MIN_HEADROOM:
            errors.append(f"gpu_{gpu}_headroom_below_10_percent")
    # Protect other existing endpoints sharing a selected GPU, including Gemma.
    for port, service in snapshot.get("services", {}).items():
        if str(service.get("env", {}).get("ROCR_VISIBLE_DEVICES")) in selected_gpus:
            if any(model.get("digest") != model_digest for model in service.get("models", [])):
                errors.append(f"foreign_model_on_shared_gpu_port_{port}")
    for lane in lanes:
        port = str(lane["amd_port"])
        service = snapshot.get("services", {}).get(port)
        if not isinstance(service, dict):
            errors.append(f"missing_ollama_service_{port}")
            continue
        env = service.get("env", {})
        expected = {"OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1",
                    "OLLAMA_CONTEXT_LENGTH": str(CONTEXT_LENGTH), "ROCR_VISIBLE_DEVICES": str(lane["gpu_id"])}
        if any(env.get(key) != value for key, value in expected.items()):
            errors.append(f"service_environment_mismatch_{port}")
        if require_idle and service.get("active_connections") != 0:
            errors.append(f"lane_not_confirmed_idle_{port}")
        models = service.get("models", [])
        if require_loaded and len(models) != 1:
            errors.append(f"expected_single_loaded_model_{port}")
        for model in models:
            if model.get("digest") != model_digest or model.get("name", model.get("model")) != MODEL:
                errors.append(f"wrong_model_{port}")
            if model.get("context_length") != CONTEXT_LENGTH:
                errors.append(f"wrong_context_{port}")
            size, size_vram = model.get("size"), model.get("size_vram")
            if type(size) is not int or type(size_vram) is not int or size <= 0 or size_vram < size:
                errors.append(f"cpu_offload_or_unknown_vram_{port}")
    return sorted(set(errors))


def _response_record(request: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    if response.get("error"):
        raise ValueError(f"Model endpoint error: {response['error']}")
    if request["kind"] == "target":
        message = response.get("message", {})
        content = message.get("content")
        json.loads(content)
        if response.get("done_reason") == "length":
            raise ValueError("Target response was truncated")
        if message.get("thinking"):
            raise ValueError("Target probe unexpectedly emitted thinking")
        return {"input_tokens": response.get("prompt_eval_count"), "output_tokens": response.get("eval_count"),
                "thinking_chars": 0, "response_json_valid": True,
                "load_duration_ns": response.get("load_duration"), "done_reason": response.get("done_reason")}
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Reflection returned no choices")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Reflection response was truncated")
    message = choice.get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Reflection returned empty content")
    usage = response.get("usage", {})
    thinking = message.get("reasoning", message.get("reasoning_content", "")) or ""
    reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    return {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
            "thinking_chars": len(thinking), "reasoning_tokens": reasoning_tokens,
            "thinking_observed": bool(thinking or reasoning_tokens),
            "response_nonempty": True, "finish_reason": choice.get("finish_reason")}


def _tokenization_checks(
    workload: Mapping[str, Any], digest: str, base_path: Path, *,
    reflection_output_reserve_tokens: int = 0, reflection_safety_margin_tokens: int = 0,
) -> dict[str, dict[str, Any]]:
    """Require saved full-template token counts, never character estimates."""
    records = workload.get("tokenization", [])
    if isinstance(records, dict):
        if any(not isinstance(record, dict) or record.get("prompt_sha256") != key for key, record in records.items()):
            raise ValueError("Tokenization key does not match its recorded prompt hash")
        records = list(records.values())
    if not isinstance(records, list):
        raise ValueError("Invalid exact tokenization evidence")
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid exact tokenization record")
        prompt_hash = record.get("prompt_sha256")
        if prompt_hash in result:
            raise ValueError("Duplicate tokenization evidence")
        if (record.get("method") not in {"llama_server_apply_template_and_tokenize", "ollama_prompt_eval_count", "openai_usage_prompt_tokens"}
                or record.get("model_digest") != digest
                or type(record.get("input_tokens")) is not int or record["input_tokens"] <= 0):
            raise ValueError("Exact full-template token count for pinned model is required")
        path = Path(str(record.get("evidence_path", "")))
        if not path.is_absolute():
            path = base_path / path
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("evidence_sha256"):
            raise ValueError("Tokenization evidence file missing or hash mismatch")
        result[prompt_hash] = dict(record, evidence_path=str(path.resolve()))
    for request in workload["requests"]:
        record = result.get(request["prompt_sha256"])
        if record is None:
            raise ValueError("Exact full-template token count missing; no generation was started")
        if record.get("request_sha256") is not None and record["request_sha256"] != _sha(request["payload"]):
            raise ValueError("Token certificate was measured with a different request payload")
        if record["method"] == "openai_usage_prompt_tokens":
            if request["kind"] != "reflection" or record.get("request_sha256") != _sha(request["payload"]):
                raise ValueError("Actual OpenAI token certificate must pin the exact reflection request")
            evidence = json.loads(Path(record["evidence_path"]).read_text(encoding="utf-8"))
            if not isinstance(evidence, dict):
                raise ValueError("Actual OpenAI token evidence must contain a saved response")
            if evidence.get("request_sha256") is not None and evidence["request_sha256"] != record["request_sha256"]:
                raise ValueError("Saved warmup receipt request hash mismatch")
            response = evidence.get("response", evidence)
            if not isinstance(response, dict):
                raise ValueError("Actual OpenAI token evidence must contain a saved response")
            measured = _response_record(request, response)
            if type(measured["input_tokens"]) is not int or measured["input_tokens"] != record["input_tokens"]:
                raise ValueError("Token certificate count differs from actual OpenAI usage")
        if record["input_tokens"] > CONTEXT_LENGTH:
            raise ValueError(f"Full recorded input ({record['input_tokens']} tokens) exceeds {CONTEXT_LENGTH}; truncation is forbidden")
    for request in workload["requests"]:
        record = result[request["prompt_sha256"]]
        payload = request["payload"]
        reserved = payload.get("max_tokens") if request["kind"] == "reflection" else payload.get("options", {}).get("num_predict")
        if type(reserved) is not int or reserved <= 0:
            raise ValueError("Probe must pin a positive output-token budget before generation")
        margin = reflection_safety_margin_tokens if request["kind"] == "reflection" else 0
        if request["kind"] == "reflection" and reflection_output_reserve_tokens and reserved != reflection_output_reserve_tokens:
            raise ValueError("Adaptive reflection output budget does not match the execution manifest")
        if record["input_tokens"] + reserved + margin > CONTEXT_LENGTH:
            if margin:
                raise ValueError("Full recorded input plus output budget and safety margin exceeds context; truncation is forbidden")
            raise ValueError("Full recorded input plus output budget exceeds context; truncation is forbidden")
    return result


def _check_response_tokens(item: Mapping[str, Any], tokenization: Mapping[str, Any]) -> None:
    if type(item.get("input_tokens")) is not int or type(item.get("output_tokens")) is not int:
        raise ValueError("Endpoint omitted measured input/output token usage")
    if item["input_tokens"] != tokenization["input_tokens"]:
        raise ValueError("Actual prompt token count differs from full-template count; truncation or template mismatch")


def _resource_minima(samples: Sequence[Mapping[str, Any]], lanes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    memory = [(sample["memory"]["limit_bytes"] - sample["memory"]["used_bytes"]) / sample["memory"]["limit_bytes"] for sample in samples]
    gpus = {}
    for gpu in {str(lane["gpu_id"]) for lane in lanes}:
        values = [(sample["gpus"][gpu]["total_bytes"] - sample["gpus"][gpu]["used_bytes"]) / sample["gpus"][gpu]["total_bytes"] for sample in samples]
        gpus[gpu] = min(values)
    return {"cgroup_memory_free_fraction": min(memory), "gpu_free_fraction": gpus}


def run_capacity_test(
    execution_manifest: str | Path, workload_path: str | Path, *,
    snapshot_provider: Callable[[], dict[str, Any]] | None = None,
    post_provider: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
    get_provider: Callable[[str], dict[str, Any]] | None = None,
    allowed_gpu_ids: Sequence[int] = (1,), execute: bool = False,
    sample_interval: float = 2.0, timeout_seconds: float = 600,
) -> dict[str, Any]:
    manifest_path, bundle, manifest_hash = _load(execution_manifest)
    workload_file, workload, workload_hash = _load(workload_path)
    reflection_minibatch_size = _reflection_minibatch_size(bundle)
    reflection_grouping = _reflection_grouping(bundle)
    output_reserve, safety_margin = _adaptive_budget(bundle)
    requests = validate_capacity_workload(workload, expected_reflection_minibatch_size=reflection_minibatch_size,
                                          expected_reflection_grouping=reflection_grouping)
    digest = bundle.get("expected_model_digest", bundle.get("model_digest"))
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Execution manifest must pin the exact Ollama model digest")
    if not math.isfinite(sample_interval) or sample_interval <= 0 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Capacity sampling interval and timeout must be positive finite values")
    allowed = set(allowed_gpu_ids)
    lanes = [lane for lane in bundle.get("lanes", []) if lane["gpu_id"] in allowed]
    if not lanes or len({lane["lane_id"] for lane in lanes}) != len(lanes):
        raise ValueError("No unique authorized lanes available")
    # Prefer the currently unused GPU1 pair; never include GPU0 by default.
    lanes.sort(key=lambda lane: (lane["gpu_id"] != 1, lane["gpu_id"], lane["amd_port"]))
    first_gpu_lanes = [lane for lane in lanes if lane["gpu_id"] == lanes[0]["gpu_id"]]
    stage_lanes = [first_gpu_lanes[:1]]
    if len(first_gpu_lanes) >= 2:
        stage_lanes.append(first_gpu_lanes[:2])
    if len(lanes) >= 4 and len({lane["gpu_id"] for lane in lanes[:4]}) == 2:
        stage_lanes.append(lanes[:4])
    report = {
        "version": CAPACITY_VERSION, "measured": False, "execution_manifest_sha256": manifest_hash,
        "execution_manifest_path": str(manifest_path), "workload_sha256": workload_hash,
        "workload_path": str(workload_file), "model_digest": digest,
        "context_length": CONTEXT_LENGTH, "allowed_gpu_ids": sorted(allowed),
        "reflection_minibatch_size": reflection_minibatch_size,
        "reflection_grouping": reflection_grouping,
        "reflection_partition": copy.deepcopy(workload.get("reflection_partition")),
        "reflection_output_reserve_tokens": output_reserve,
        "reflection_safety_margin_tokens": safety_margin,
        "approved_lane_ids": [], "approved_concurrency": 0,
        "planned_lane_groups": [[lane["lane_id"] for lane in group] for group in stage_lanes],
        "headroom_fraction_required": MIN_HEADROOM, "total_workload_copies_per_timed_stage": 4,
        "minimum_incremental_throughput_speedup": MIN_SPEEDUP,
        "sample_interval_seconds": sample_interval, "stages": [], "model_calls": 0,
        "limitations": ["Observed snapshots cannot prove the absence of unsampled instantaneous resource peaks.",
                        "Approval covers this recorded workload and context, not every future prompt length."],
    }
    try:
        tokenization = _tokenization_checks(workload, digest, workload_file.parent,
            reflection_output_reserve_tokens=output_reserve, reflection_safety_margin_tokens=safety_margin)
    except ValueError as exc:
        report["preflight_errors"] = [str(exc)]
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        return report
    report["tokenization_evidence"] = list(tokenization.values())
    if not execute:
        return report
    if snapshot_provider is None:
        raise ValueError("Executing capacity requires a real AMD resource snapshot provider")
    post = post_provider or (lambda url, payload: _http_json(url, payload, timeout_seconds))
    get = get_provider or (lambda url: _http_json(url, timeout=15))
    calls_lock = threading.Lock()
    for selected in stage_lanes:
        stage = {"concurrency": len(selected), "lane_ids": [lane["lane_id"] for lane in selected],
                 "status": "blocked", "samples": [], "requests": [], "errors": [],
                 "total_workload_copies": 4}
        report["stages"].append(stage)
        try:
            before = snapshot_provider()
            stage["samples"].append(before)
            errors = capacity_gate(selected, before, digest, require_idle=True)
            for lane in selected:
                tags = get(lane["host"] + "/api/tags")
                if not any(item.get("name") == MODEL and item.get("digest") == digest for item in tags.get("models", [])):
                    errors.append(f"pinned_model_missing_{lane['lane_id']}")
            if errors:
                stage["errors"] = errors
                break

            def replay(lane: Mapping[str, Any], *, warmup: bool) -> None:
                for request in requests:
                    started = time.monotonic()
                    with calls_lock:
                        report["model_calls"] += 1
                    response = post(lane["host"] + request["path"], copy.deepcopy(request["payload"]))
                    item = {"lane_id": lane["lane_id"], "kind": request["kind"],
                            "prompt_sha256": request["prompt_sha256"], "request_sha256": _sha(request["payload"]),
                            "warmup": warmup, "seconds": time.monotonic() - started,
                            **_response_record(request, response)}
                    _check_response_tokens(item, tokenization[request["prompt_sha256"]])
                    with calls_lock:
                        stage["requests"].append(item)

            started = time.monotonic()
            for lane in selected:
                replay(lane, warmup=True)
            stage["warmup_seconds"] = time.monotonic() - started
            after_warmup = snapshot_provider()
            stage["samples"].append(after_warmup)
            errors = capacity_gate(selected, after_warmup, digest, require_loaded=True)
            if errors:
                stage["status"], stage["errors"] = "failed", errors
                break
            stop = threading.Event()
            sampler_errors: list[str] = []

            def sample() -> None:
                while not stop.wait(sample_interval):
                    try:
                        snapshot = snapshot_provider()
                        stage["samples"].append(snapshot)
                        sampler_errors.extend(capacity_gate(selected, snapshot, digest, require_loaded=True))
                    except Exception as exc:
                        sampler_errors.append(f"resource_sampler_failed:{type(exc).__name__}:{exc}")

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            started = time.monotonic()
            try:
                def work(lane):
                    for _ in range(4 // len(selected)):
                        replay(lane, warmup=False)
                with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                    list(pool.map(work, selected))
            finally:
                stage["measured_seconds"] = time.monotonic() - started
                stop.set()
                sampler.join(timeout=65)
                if sampler.is_alive():
                    sampler_errors.append("resource_sampler_did_not_finish")
            after = snapshot_provider()
            stage["samples"].append(after)
            errors = sampler_errors + capacity_gate(selected, after, digest, require_loaded=True)
            stage["errors"] = sorted(set(errors))
            stage["status"] = "failed" if errors else "passed"
            report["measured"] = True
            if errors:
                break
            stage["sampled_minimum_headroom"] = _resource_minima(stage["samples"], selected)
            previous = [row for row in report["stages"][:-1] if row["status"] == "passed"]
            if previous:
                stage["throughput_speedup_vs_previous"] = previous[-1]["measured_seconds"] / stage["measured_seconds"]
                if stage["throughput_speedup_vs_previous"] < MIN_SPEEDUP:
                    stage["status"] = "failed"
                    stage["errors"] = ["measured_throughput_gain_below_10_percent"]
                    break
            report["approved_lane_ids"] = list(stage["lane_ids"])
            report["approved_concurrency"] = len(selected)
        except Exception as exc:
            stage["status"] = "failed"
            stage["errors"].append(f"{type(exc).__name__}:{exc}")
            break
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


def validate_capacity_report(
    execution_manifest: str | Path, report_path: str | Path, requested_concurrency: int,
) -> dict[str, Any]:
    """Recheck saved measurements before a scheduler may select any lane.

    This does not replace live idle/model checks. Returned lane order comes from
    the passed measurement, not manifest order, which could select a busy GPU0.
    """
    _, bundle, manifest_hash = _load(execution_manifest)
    _, report, _ = _load(report_path)
    digest = bundle.get("expected_model_digest", bundle.get("model_digest"))
    reflection_minibatch_size = _reflection_minibatch_size(bundle)
    reflection_grouping = _reflection_grouping(bundle)
    output_reserve, safety_margin = _adaptive_budget(bundle)
    if requested_concurrency not in {1, 2, 4}:
        raise ValueError("Only measured concurrency 1, 2, or 4 is supported")
    if (report.get("version") != CAPACITY_VERSION or report.get("measured") is not True
            or report.get("execution_manifest_sha256") != manifest_hash
            or report.get("model_digest") != digest or report.get("context_length") != CONTEXT_LENGTH
            or report.get("reflection_minibatch_size", 8) != reflection_minibatch_size
            or report.get("reflection_grouping", "fixed") != reflection_grouping
            or report.get("reflection_output_reserve_tokens", 0) != output_reserve
            or report.get("reflection_safety_margin_tokens", 0) != safety_margin
            or report.get("headroom_fraction_required") != MIN_HEADROOM
            or report.get("minimum_incremental_throughput_speedup") != MIN_SPEEDUP
            or report.get("approved_concurrency", 0) < requested_concurrency):
        raise ValueError("Capacity approval does not match pinned execution requirements")
    workload_path, workload, workload_hash = _load(report["workload_path"])
    if workload_hash != report.get("workload_sha256"):
        raise ValueError("Capacity workload changed after measurement")
    requests = validate_capacity_workload(workload, expected_reflection_minibatch_size=reflection_minibatch_size,
                                          expected_reflection_grouping=reflection_grouping)
    if report.get("reflection_partition") != workload.get("reflection_partition"):
        raise ValueError("Capacity report changed the recorded reflection partition")
    tokenization = _tokenization_checks(workload, digest, workload_path.parent,
        reflection_output_reserve_tokens=output_reserve, reflection_safety_margin_tokens=safety_margin)
    lane_by_id = {lane["lane_id"]: lane for lane in bundle["lanes"]}
    previous = None
    selected_stage = None
    for concurrency in (1, 2, 4):
        if concurrency > requested_concurrency:
            break
        matches = [stage for stage in report.get("stages", []) if stage.get("concurrency") == concurrency and stage.get("status") == "passed"]
        if len(matches) != 1:
            raise ValueError("Missing unique passed measured capacity stage")
        stage = matches[0]
        ids = stage.get("lane_ids", [])
        if (stage.get("errors") or len(ids) != concurrency or len(set(ids)) != concurrency
                or any(lane_id not in lane_by_id for lane_id in ids)
                or not set(ids).issubset(set(report.get("approved_lane_ids", [])))):
            raise ValueError("Capacity stage has errors or unapproved lanes")
        lanes = [lane_by_id[lane_id] for lane_id in ids]
        if previous and not set(previous["lane_ids"]).issubset(set(ids)):
            raise ValueError("Measured capacity stages changed the baseline lane set")
        samples = stage.get("samples", [])
        if len(samples) < 3 or capacity_gate(lanes, samples[0], digest, require_idle=True):
            raise ValueError("Capacity lacks valid idle preflight evidence")
        if any(capacity_gate(lanes, sample, digest, require_loaded=True) for sample in samples[1:]):
            raise ValueError("Capacity resource or full-GPU model checks failed")
        measured = [row for row in stage.get("requests", []) if row.get("warmup") is False]
        warmup = [row for row in stage.get("requests", []) if row.get("warmup") is True]
        if len(measured) != 4 * len(requests) or len(warmup) != concurrency * len(requests):
            raise ValueError("Capacity compared unequal or incomplete workloads")
        for lane_id in ids:
            for request in requests:
                for rows, count in ((measured, 4 // concurrency), (warmup, 1)):
                    matching = [row for row in rows if row.get("lane_id") == lane_id and row.get("request_sha256") == _sha(request["payload"])]
                    if len(matching) != count:
                        raise ValueError("Capacity did not replay the same requests on every lane")
                    for item in matching:
                        if item.get("prompt_sha256") != request["prompt_sha256"]:
                            raise ValueError("Capacity prompt hash mismatch")
                        _check_response_tokens(item, tokenization[request["prompt_sha256"]])
        seconds = stage.get("measured_seconds")
        if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Missing measured wall time")
        if previous and previous["measured_seconds"] / seconds < MIN_SPEEDUP:
            raise ValueError("Measured additional lanes did not improve throughput by 10 percent")
        previous = selected_stage = stage
    return dict(report, requested_stage=selected_stage,
                approved_lanes=[lane_by_id[lane_id] for lane_id in selected_stage["lane_ids"]])
