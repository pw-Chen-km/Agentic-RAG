"""Exact tokenizer ownership and transport tests; no SSH/model calls execute."""

from __future__ import annotations

import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from agentic_rag.skillopt import reflection_tokenizer as module
from agentic_rag.skillopt.reflection_tokenizer import ReflectionTokenizationError, SshOllamaTokenCounter


CONFIG = {"reflection_tokenizer_ssh_host": "root@140.116.240.181", "reflection_tokenizer_ssh_port": 45026,
          "reflection_tokenizer_ssh_key": "/home/jj/.ssh/amd_root_key", "reflection_tokenizer_amd_port": 11501}
MESSAGES = [{"role": "system", "content": "Synthetic system"}, {"role": "user", "content": "PROMPT_ONLY_ON_STDIN 雪"}]


def payload():
    return {"messages": MESSAGES, "prompt_sha256": module._sha(MESSAGES), "model": module.MODEL,
            "model_digest": module.MODEL_DIGEST, "model_blob_digest": module.MODEL_BLOB_DIGEST,
            "amd_port": CONFIG["reflection_tokenizer_amd_port"], "context_length": module.CONTEXT_LENGTH}


@pytest.fixture
def remote():
    namespace = {"__name__": "offline_tokenizer_test"}
    exec(compile(module._REMOTE_PROGRAM, "synthetic_remote_helper", "exec"), namespace)
    service = {"pid": 100, "ppid": 1, "start": "123", "args": ["/usr/bin/ollama", "serve"]}
    backend = {"pid": 102, "ppid": 101, "start": "125", "args": ["/usr/lib/ollama/llama-server",
               "--model", "/root/.ollama/models/blobs/sha256-" + module.MODEL_BLOB_DIGEST,
               "--port", "39001", "--chat-template", "qwen"]}
    # An identically named model under a different service must never be chosen.
    other = copy.deepcopy(backend)
    other.update(pid=202, ppid=200)
    other["args"][other["args"].index("--port") + 1] = "39002"
    processes = {100: service, 101: {"pid": 101, "ppid": 100, "start": "124", "args": ["wrapper"]},
                 102: backend, 200: {"pid": 200, "ppid": 1, "start": "223", "args": ["ollama", "serve"]}, 202: other}
    owners = {11501: {100}, 39001: {102}, 39002: {202}}
    loaded = {"name": module.MODEL, "digest": module.MODEL_DIGEST, "context_length": module.CONTEXT_LENGTH,
              "size": 1000, "size_vram": 1000}
    ps = {"models": [loaded]}
    calls = []
    responses = {"/apply-template": {"prompt": "<chat>FULL SYNTHETIC PROMPT<think>"}, "/tokenize": {"tokens": [1, 2, 3, 4]}}

    def http(port, path, body, deadline):
        calls.append((port, path, body))
        return copy.deepcopy(ps if path == "/api/ps" else responses[path])

    def run(**kwargs):
        return namespace["count"](payload(), http=kwargs.get("http", http),
                                  snapshot=kwargs.get("snapshot", lambda: copy.deepcopy(processes)),
                                  listeners=kwargs.get("listeners", lambda port: set(owners[port])))

    return SimpleNamespace(run=run, ns=namespace, processes=processes, owners=owners,
                           ps=ps, responses=responses, calls=calls)


def test_remote_counts_complete_template_from_correct_service_descendant(remote):
    record = remote.run()
    assert record["input_tokens"] == 4
    assert record["backend_pid"] == 102 and record["backend_port"] == 39001
    assert record["service_pid"] == 100
    assert record["method"] == module.METHOD and record["generation_calls"] == 0
    assert record["verified_before_and_after"] is True
    assert [(port, path) for port, path, _ in remote.calls] == [(11501, "/api/ps"), (39001, "/apply-template"),
                                                              (39001, "/tokenize"), (11501, "/api/ps")]
    assert remote.calls[1][2] == {"messages": MESSAGES, "enable_thinking": True, "add_generation_prompt": True}
    assert remote.calls[2][2] == {"content": "<chat>FULL SYNTHETIC PROMPT<think>", "add_special": False}
    assert "PROMPT_ONLY_ON_STDIN" not in json.dumps(record)
    assert "FULL SYNTHETIC PROMPT" not in json.dumps(record)


@pytest.mark.parametrize("models,reason", [([], "expected_one_loaded"),
    ([{"name": "gemma", "digest": "b" * 64}], "loaded_model_mismatch"),
    ([{}, {}], "expected_one_loaded")])
def test_absent_or_foreign_loaded_model_never_auto_loads_or_replaces(remote, models, reason):
    remote.ps["models"] = models
    with pytest.raises(remote.ns["VerificationError"], match=reason):
        remote.run()
    assert [path for _, path, _ in remote.calls] == ["/api/ps"]


@pytest.mark.parametrize("field,value,reason", [("context_length", 8192, "context_mismatch"),
    ("size_vram", 999, "cpu_offload"), ("size_vram", None, "unknown_gpu"), ("size", 0, "cpu_offload")])
def test_loaded_context_and_gpu_residency_are_required(remote, field, value, reason):
    remote.ps["models"][0][field] = value
    with pytest.raises(remote.ns["VerificationError"], match=reason):
        remote.run()
    assert len(remote.calls) == 1


def test_matching_backend_elsewhere_cannot_replace_missing_service_child(remote):
    del remote.processes[102]
    with pytest.raises(remote.ns["VerificationError"], match="matching_backend"):
        remote.run()
    assert len(remote.calls) == 1


def test_two_matching_children_are_ambiguous(remote):
    second = copy.deepcopy(remote.processes[102])
    second["pid"] = 103
    remote.processes[103] = second
    with pytest.raises(remote.ns["VerificationError"], match="ambiguous_matching_backend"):
        remote.run()


def test_wrong_model_blob_is_rejected_even_when_model_name_matches(remote):
    args = remote.processes[102]["args"]
    args[args.index("--model") + 1] = "/models/sha256-" + "b" * 64
    with pytest.raises(remote.ns["VerificationError"], match="matching_backend"):
        remote.run()


@pytest.mark.parametrize("owners", [set(), {999}, {100, 200}])
def test_service_requires_unambiguous_actual_listener_ownership(remote, owners):
    remote.owners[11501] = owners
    with pytest.raises(remote.ns["VerificationError"]):
        remote.run()
    # With a different owner, no matching descendant may be borrowed from100.
    assert not any(path == "/apply-template" for _, path, _ in remote.calls)


def test_service_is_selected_by_actual_configured_port_not_first_process(remote):
    remote.owners[11501] = {200}
    result = remote.run()
    assert result["service_pid"] == 200 and result["backend_pid"] == 202
    assert result["backend_port"] == 39002


def test_backend_port_cannot_point_to_another_process(remote):
    remote.owners[39001] = {202}
    with pytest.raises(remote.ns["VerificationError"], match="listener_ownership"):
        remote.run()


@pytest.mark.parametrize("stage,reply,reason", [("/apply-template", {}, "complete_chat_template"),
    ("/apply-template", {"prompt": ""}, "complete_chat_template"),
    ("/tokenize", {"tokens": []}, "exact_tokens"), ("/tokenize", {"tokens": [1, True]}, "exact_tokens"),
    ("/tokenize", {"tokens": ["one"]}, "exact_tokens")])
def test_missing_or_approximate_token_outputs_are_not_counts(remote, stage, reply, reason):
    remote.responses[stage] = reply
    with pytest.raises(remote.ns["VerificationError"], match=reason):
        remote.run()


def test_process_restart_during_token_count_is_detected(remote):
    snapshots = 0

    def snapshot():
        nonlocal snapshots
        snapshots += 1
        values = copy.deepcopy(remote.processes)
        if snapshots > 1:
            values[102]["start"] = "new-process-same-pid"
        return values

    with pytest.raises(remote.ns["VerificationError"], match="changed_during_count"):
        remote.run(snapshot=snapshot)


def test_remote_prompt_hash_must_match_exact_messages(remote):
    value = payload()
    value["messages"] = [{"role": "user", "content": "changed"}]
    with pytest.raises(remote.ns["VerificationError"], match="prompt_hash_mismatch"):
        remote.ns["count"](value, http=lambda *args: pytest.fail("Must stop before HTTP"))


def test_transport_uses_stdin_not_command_arguments_and_saves_no_files(remote, monkeypatch, tmp_path):
    calls = []
    record = remote.run()
    record["unexpected_full_text"] = "FULL SYNTHETIC PROMPT"

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "result": record}), stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.chdir(tmp_path)
    counter = SshOllamaTokenCounter(CONFIG)
    result = counter(MESSAGES)
    assert result["input_tokens"] == 4 and "unexpected_full_text" not in result
    assert not list(tmp_path.iterdir())
    argv, kwargs = calls[0]
    assert argv[0] == "ssh" and CONFIG["reflection_tokenizer_ssh_host"] in argv
    assert "PROMPT_ONLY_ON_STDIN" not in " ".join(argv)
    assert json.loads(kwargs["input"])["messages"] == MESSAGES
    assert kwargs["timeout"] == 120 and kwargs["check"] is False
    assert "StrictHostKeyChecking=yes" in argv and "BatchMode=yes" in argv


@pytest.mark.parametrize("field,value", [("reflection_tokenizer_ssh_host", "root@host; touch injected"),
    ("reflection_tokenizer_ssh_host", "-oProxyCommand=bad"), ("reflection_tokenizer_ssh_port", 0),
    ("reflection_tokenizer_amd_port", True), ("reflection_tokenizer_ssh_key", "relative/key"),
    ("expected_model_digest", "short")])
def test_invalid_config_fails_before_ssh(monkeypatch, field, value):
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("No SSH"))
    with pytest.raises(ValueError):
        SshOllamaTokenCounter({**CONFIG, field: value})


@pytest.mark.parametrize("messages", [[], "text only", [{"content": "missing role"}], [{"role": "user", "content": ["image"]}]])
def test_only_complete_text_chat_messages_are_accepted(monkeypatch, messages):
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("No SSH"))
    with pytest.raises(ValueError):
        SshOllamaTokenCounter(CONFIG)(messages)


@pytest.mark.parametrize("kind", ["timeout", "unavailable", "bad_json", "remote_error"])
def test_transport_failure_never_returns_estimate_or_leaks_input(monkeypatch, kind):
    def run(argv, **kwargs):
        if kind == "timeout":
            raise subprocess.TimeoutExpired(argv, 120, output="PROMPT_ONLY_ON_STDIN")
        if kind == "unavailable":
            raise OSError("PROMPT_ONLY_ON_STDIN")
        if kind == "bad_json":
            return SimpleNamespace(returncode=0, stdout="PROMPT_ONLY_ON_STDIN", stderr="")
        return SimpleNamespace(returncode=2, stdout=json.dumps({"ok": False, "error_code": "foreign_model"}), stderr="PROMPT_ONLY_ON_STDIN")

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(ReflectionTokenizationError) as error:
        SshOllamaTokenCounter(CONFIG)(MESSAGES)
    assert "PROMPT_ONLY_ON_STDIN" not in str(error.value)


@pytest.mark.parametrize("field,value", [("input_tokens", 0), ("input_tokens", True),
    ("model_digest", "b" * 64), ("prompt_sha256", "b" * 64), ("backend_pid", 0),
    ("backend_port", 99999), ("verified_before_and_after", False), ("rendered_prompt_sha256", "unknown")])
def test_returned_count_requires_matching_identity_and_exact_evidence(remote, monkeypatch, field, value):
    record = remote.run()
    record[field] = value
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps({"ok": True, "result": record}), stderr=""))
    with pytest.raises(ReflectionTokenizationError):
        SshOllamaTokenCounter(CONFIG)(MESSAGES)
