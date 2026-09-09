"""Exercise installed SkillOpt prompt assembly; all token/model calls are fake.

These checks intentionally skip when native SkillOpt is not installed. They
must also run in JJ's actual environment before claiming native compatibility.
"""
from __future__ import annotations

import copy
import importlib
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rag.skillopt.reflection_budget import (
    build_budgeted_dispatcher,
    capture_native_prompt,
    prompt_messages,
    sha,
    use_token_budget_reflection,
)


SKILL = "FROZEN-SKILL: search, expand, read, then answer from evidence."
SETTINGS = {
    "reflection_token_budget_enabled": True, "minibatch_size": 5,
    "reflection_context_tokens": 262144, "reflection_output_reserve_tokens": 16384,
    "reflection_safety_margin_tokens": 1024,
}
SYSTEM = "NATIVE-ANALYST-SYSTEM: preserve all evidence and propose supported edits."
NOOP = {"patch": {"reasoning": "No reliable shared change.", "edits": []}}


@pytest.fixture
def native():
    pytest.importorskip("skillopt")
    return importlib.import_module("skillopt.gradient.reflect")


def write_rows(tmp_path: Path, count: int, failures: int, *, large: bool = False):
    prediction = tmp_path / "predictions"
    rows, sentinels = [], {}
    for index in range(count):
        identifier = f"episode-{index:02d}"
        text = f"TEXT-START-{identifier}\n" + ("entire original observation, not a summary. " * (1000 if large else 1)) + f"\nTEXT-END-{identifier}"
        sentinels[identifier] = text
        directory = prediction / identifier
        directory.mkdir(parents=True)
        (directory / "conversation.json").write_text(json.dumps([
            {"type": "tool_call", "cmd": f"SEARCH query for {identifier}", "obs": text},
            {"role": "system", "content": f"FINAL-STATE-{identifier}"},
        ]))
        (directory / "target_system_prompt.txt").write_text(f"TARGET-PROTOCOL-{identifier}")
        (directory / "target_user_prompt.txt").write_text(f"ORIGINAL-QUESTION-{identifier}")
        rows.append({"id": identifier, "hard": int(index >= failures), "task_description": f"QUESTION-{identifier}",
                     "reference_text": f"HIDDEN-REFERENCE-{identifier}", "n_turns": 2,
                     "fail_reason": "wrong answer" if index < failures else ""})
    return prediction, rows, sentinels


def ids_in_call(call):
    return re.findall(r"^### Trajectory \d+ \(id=([^\)]+)\)", call["user"], flags=re.MULTILINE)


def fake_counter(messages):
    identifiers = ids_in_call({"user": messages[1]["content"]})
    # A full group of five is too long; three complete records fit.
    return {"input_tokens": 70000 * len(identifiers), "prompt_sha256": sha(messages), "method": "test-only"}


def dispatch_kwargs(prediction, rows, tmp_path):
    return {"results": rows, "skill_content": SKILL, "prediction_dir": str(prediction),
            "patches_dir": str(tmp_path / "patches"), "workers": 1, "failure_only": False,
            "minibatch_size": 5, "edit_budget": 1, "random_seed": 42,
            "error_system": SYSTEM, "success_system": SYSTEM, "skill_aware_reflection": False}


def fake_transport(native, monkeypatch, *, behavior=None):
    calls = []

    def post(payload, timeout, config):
        call = {"system": payload["messages"][0]["content"], "user": payload["messages"][1]["content"]}
        calls.append(copy.deepcopy(call))
        response = {"choices": [{"message": {"content": json.dumps(NOOP), "reasoning": "Test-only reasoning"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 70000 * len(ids_in_call(call)) + 3, "completion_tokens": 50}}
        return behavior(response, call, len(calls)) if behavior is not None else response

    backend = SimpleNamespace(_post_chat_completion=post)

    def chat(**call):
        response = backend._post_chat_completion({"messages": prompt_messages(call), "max_tokens": call["max_completion_tokens"]}, 1, {})
        return response["choices"][0]["message"]["content"], response["usage"]

    monkeypatch.setattr(native, "chat_optimizer", chat)
    return backend, calls, chat, post


@pytest.mark.parametrize("kind", ["fail", "succ"])
def test_capture_preserves_exact_native_prompt_and_all_original_text(native, tmp_path, monkeypatch, kind):
    prediction, rows, texts = write_rows(tmp_path, 2, 2 if kind == "fail" else 0, large=True)
    direct = []

    def chat(**kwargs):
        direct.append(copy.deepcopy(kwargs))
        return json.dumps(NOOP), {}

    monkeypatch.setattr(native, "chat_optimizer", chat)
    kwargs = {"edit_budget": 1, "system_prompt": SYSTEM, "step_buffer_context": "PREVIOUS-STEP-CONTEXT",
              "trajectory_memory_context": "MEMORY-CONTEXT", "meta_skill_context": "", "update_mode": "patch",
              "skill_aware_reflection": False}
    function = native.run_error_analyst_minibatch if kind == "fail" else native.run_success_analyst_minibatch
    function(SKILL, rows, str(prediction), **kwargs)
    assert len(direct) == 1
    captured = capture_native_prompt(native, kind, SKILL, rows, str(prediction), kwargs)
    assert captured == direct[0]
    assert native.chat_optimizer is chat
    assert len(direct) == 1  # Capturing made no optimizer call.
    assert captured["system"] == SYSTEM
    assert SKILL in captured["user"]
    assert "PREVIOUS-STEP-CONTEXT" in captured["user"]
    for row in rows:
        identifier = row["id"]
        assert texts[identifier] in captured["user"]
        assert f"HIDDEN-REFERENCE-{identifier}" in captured["user"]
        assert f"TARGET-PROTOCOL-{identifier}" in captured["user"]
        assert f"ORIGINAL-QUESTION-{identifier}" in captured["user"]


@pytest.mark.parametrize("failures", [0, 1, 20, 39, 40])
def test_one_forty_rollout_batch_keeps_frozen_skill_and_complete_records(native, tmp_path, monkeypatch, failures):
    prediction, rows, texts = write_rows(tmp_path, 40, failures)
    backend, calls, chat, post = fake_transport(native, monkeypatch)
    original_dispatch = native.run_minibatch_reflect
    with use_token_budget_reflection(SETTINGS, native=native, counter=fake_counter, response_backend=backend):
        patches = native.run_minibatch_reflect(**dispatch_kwargs(prediction, rows, tmp_path))
    assert native.run_minibatch_reflect is original_dispatch
    assert native.chat_optimizer is chat
    assert backend._post_chat_completion is post
    observed_ids = [identifier for call in calls for identifier in ids_in_call(call)]
    assert Counter(observed_ids) == Counter(row["id"] for row in rows)
    assert all(1 <= len(ids_in_call(call)) <= 3 for call in calls)
    assert len(patches) == len(calls)
    assert all(patch["patch"]["edits"] == [] for patch in patches)
    for call in calls:
        assert SKILL in call["user"]
        assert call["system"] == SYSTEM
        for identifier in ids_in_call(call):
            assert texts[identifier] in call["user"]
    # No merge/update is invoked inside the dispatcher. All calls used the same
    # incoming Skill; it returns the whole batch's proposals to the outer loop.
    plan = json.loads((tmp_path / "patches/reflection_groups.json").read_text())
    assert sorted(identifier for group in plan["groups"] for identifier in group["episode_ids"]) == sorted(observed_ids)
    assert all(group["input_tokens"] + 16384 + 1024 <= 262144 for group in plan["groups"])


def test_completed_empty_edits_are_not_regenerated_on_resume(native, tmp_path, monkeypatch):
    prediction, rows, _ = write_rows(tmp_path, 5, 5)
    backend, calls, _, _ = fake_transport(native, monkeypatch)
    kwargs = dispatch_kwargs(prediction, rows, tmp_path)
    dispatch = build_budgeted_dispatcher(native, SETTINGS, fake_counter, response_backend=backend)
    original = dispatch(**kwargs)
    assert len(calls) == 2
    receipts = sorted((tmp_path / "patches/completed_groups").glob("*.json"))
    assert len(receipts) == 2
    hashes_before = [path.read_bytes() for path in receipts]

    def no_count(messages):
        raise AssertionError("Resume should reuse the exact saved grouping")

    resumed = build_budgeted_dispatcher(native, SETTINGS, no_count, response_backend=backend)(**kwargs)
    assert resumed == original
    assert len(calls) == 2
    assert [path.read_bytes() for path in receipts] == hashes_before


def test_later_group_error_resumes_only_unfinished_group(native, tmp_path, monkeypatch):
    prediction, rows, _ = write_rows(tmp_path, 5, 5)
    def fail_second(response, call, index):
        if index == 2:
            raise RuntimeError("Injected service failure")
        return response
    backend, calls, chat, post = fake_transport(native, monkeypatch, behavior=fail_second)
    kwargs = dispatch_kwargs(prediction, rows, tmp_path)
    with pytest.raises(RuntimeError, match="no Skill update"):
        build_budgeted_dispatcher(native, SETTINGS, fake_counter, response_backend=backend)(**kwargs)
    assert len(list((tmp_path / "patches/completed_groups").glob("*.json"))) == 1
    assert native.chat_optimizer is chat and backend._post_chat_completion is post
    # The same transport succeeds on call3; only the unfinished group is sent.
    recovered = build_budgeted_dispatcher(native, SETTINGS, fake_counter, response_backend=backend)(**kwargs)
    assert len(recovered) == 2 and len(calls) == 3


@pytest.mark.parametrize("fault", ["truncated", "below_input", "above_margin", "missing_usage", "invalid_json"])
def test_native_error_swallowing_cannot_mark_bad_response_complete(native, tmp_path, monkeypatch, fault):
    prediction, rows, _ = write_rows(tmp_path, 1, 1)
    def corrupt(response, call, index):
        if fault == "truncated":
            response["choices"][0]["finish_reason"] = "length"
        elif fault == "below_input":
            response["usage"]["prompt_tokens"] = 69999
        elif fault == "above_margin":
            response["usage"]["prompt_tokens"] = 71025
        elif fault == "missing_usage":
            response["usage"].pop("prompt_tokens")
        else:
            response["choices"][0]["message"]["content"] = "not valid patch JSON"
        return response
    backend, _, chat, post = fake_transport(native, monkeypatch, behavior=corrupt)
    with pytest.raises((RuntimeError, ValueError)):
        build_budgeted_dispatcher(native, SETTINGS, fake_counter, response_backend=backend)(**dispatch_kwargs(prediction, rows, tmp_path))
    assert not list((tmp_path / "patches/completed_groups").glob("*.json"))
    assert native.chat_optimizer is chat and backend._post_chat_completion is post


def test_completion_receipt_patch_tampering_is_rejected(native, tmp_path, monkeypatch):
    prediction, rows, _ = write_rows(tmp_path, 1, 1)
    backend, calls, _, _ = fake_transport(native, monkeypatch)
    dispatch = build_budgeted_dispatcher(native, SETTINGS, fake_counter, response_backend=backend)
    kwargs = dispatch_kwargs(prediction, rows, tmp_path)
    dispatch(**kwargs)
    receipt_path = next((tmp_path / "patches/completed_groups").glob("*.json"))
    receipt = json.loads(receipt_path.read_text())
    receipt["patch"]["patch"]["reasoning"] = "Modified after completion"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt"):
        dispatch(**kwargs)
    assert len(calls) == 1


def test_missing_backend_observation_cannot_bypass_response_guard(native, tmp_path, monkeypatch):
    prediction, rows, _ = write_rows(tmp_path, 1, 1)
    backend, calls, _, post = fake_transport(native, monkeypatch)
    def bypass(**kwargs):
        return json.dumps(NOOP), {}
    monkeypatch.setattr(native, "chat_optimizer", bypass)
    with pytest.raises(RuntimeError):
        build_budgeted_dispatcher(native, SETTINGS, fake_counter, response_backend=backend)(**dispatch_kwargs(prediction, rows, tmp_path))
    assert not calls
    assert native.chat_optimizer is bypass and backend._post_chat_completion is post


def test_context_and_capture_restore_native_functions_on_exception(native, tmp_path, monkeypatch):
    prediction, rows, _ = write_rows(tmp_path, 1, 1)
    backend, _, chat, post = fake_transport(native, monkeypatch)
    original_dispatch = native.run_minibatch_reflect
    with pytest.raises(RuntimeError, match="outer failure"):
        with use_token_budget_reflection(SETTINGS, native=native, counter=fake_counter, response_backend=backend):
            raise RuntimeError("outer failure")
    assert native.run_minibatch_reflect is original_dispatch
    assert native.chat_optimizer is chat and backend._post_chat_completion is post
    with pytest.raises(ValueError, match="exactly one complete prompt"):
        capture_native_prompt(native, "fail", SKILL, rows, str(tmp_path / "missing"), {"system_prompt": SYSTEM})
    assert native.chat_optimizer is chat
