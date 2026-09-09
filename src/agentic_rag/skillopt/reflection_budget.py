"""Project-scoped, lossless token-budget grouping of native SkillOpt analysts.

Native analysts still assemble and parse their prompts. Only grouping and the
durable per-group receipt are replaced; merge/ranking and update gates are not.
Every group is checked before the first analyst generation in the rollout batch.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping


VERSION = "skillopt-reflection-token-budget-v1"


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _write_once(path: Path, value: Any) -> None:
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Reflection checkpoint changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".reflection-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def budget_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "version": VERSION, "mode": "adaptive", "max_minibatch_size": config.get("minibatch_size", 5),
        "context_tokens": config.get("reflection_context_tokens", 262144),
        "output_reserve_tokens": config.get("reflection_output_reserve_tokens", 16384),
        "safety_margin_tokens": config.get("reflection_safety_margin_tokens", 1024),
    }
    if any(type(values[key]) is not int for key in values if key.endswith("tokens") or key == "max_minibatch_size"):
        raise ValueError("Reflection token settings must be integers")
    if (not 1 <= values["max_minibatch_size"] <= 5 or values["context_tokens"] <= 0
            or values["output_reserve_tokens"] <= 0 or values["safety_margin_tokens"] < 0
            or values["output_reserve_tokens"] + values["safety_margin_tokens"] >= values["context_tokens"]):
        raise ValueError("Invalid reflection token budget")
    return values


class _CapturedPrompt(BaseException):
    """Bypass the native analyst's broad Exception handler without generation."""


def capture_native_prompt(native: Any, kind: str, skill: str, items: list[dict],
                          prediction_dir: str, analyst_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    captured = []
    original = native.chat_optimizer

    def capture(**kwargs):
        captured.append(copy.deepcopy(kwargs))
        raise _CapturedPrompt()

    native.chat_optimizer = capture
    try:
        try:
            getattr(native, "run_error_analyst_minibatch" if kind == "fail" else "run_success_analyst_minibatch")(
                skill, items, prediction_dir, **dict(analyst_kwargs))
        except _CapturedPrompt:
            pass
    finally:
        native.chat_optimizer = original
    if len(captured) != 1:
        raise ValueError("Native analyst did not assemble exactly one complete prompt")
    call = captured[0]
    if call.get("stage") != "analyst" or not isinstance(call.get("system"), str) or not isinstance(call.get("user"), str):
        raise ValueError("Unexpected native analyst call format")
    return call


def prompt_messages(call: Mapping[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": call["system"]}, {"role": "user", "content": call["user"]}]


def partition_group(items: list[dict], build_prompt, counter, settings: Mapping[str, Any], *,
                    authorized_overlong_ids=()) -> tuple[list[dict], list[dict]]:
    """Largest fitting ordered prefix, only within one original <=5 group."""
    groups, attempts = [], []
    remaining = list(items)
    while remaining:
        for size in range(min(len(remaining), settings["max_minibatch_size"]), 0, -1):
            subset = remaining[:size]
            call = build_prompt(subset)
            if call.get("max_completion_tokens") != settings["output_reserve_tokens"]:
                raise ValueError("Native analyst output limit differs from reserved capacity")
            messages = prompt_messages(call)
            counted = dict(counter(messages))
            count = counted.get("input_tokens")
            if type(count) is not int or count <= 0 or counted.get("prompt_sha256") != sha(messages):
                raise ValueError("Missing or mismatched full-prompt tokenizer evidence")
            fits = count + settings["output_reserve_tokens"] + settings["safety_margin_tokens"] <= settings["context_tokens"]
            record = {"episode_ids": [str(row["id"]) for row in subset], "prompt_sha256": sha(messages),
                      "input_tokens": count, "fits": fits, "tokenization": counted}
            attempts.append(record)
            if fits:
                groups.append({**record, "call": call})
                remaining = remaining[size:]
                break
        else:
            identifier = str(remaining[0]['id'])
            if identifier not in authorized_overlong_ids:
                raise ValueError(f"Single complete trajectory cannot fit reflection context: {identifier}")
            attempts[-1].update(skipped=True, reason="user_authorized_single_input_over_context")
            remaining = remaining[1:]
    skipped = {row['episode_ids'][0] for row in attempts if row.get('skipped')}
    assert [identifier for group in groups for identifier in group["episode_ids"]] == [str(row["id"]) for row in items if str(row['id']) not in skipped]
    return groups, attempts


def build_budgeted_dispatcher(native: Any, config: Mapping[str, Any], counter, *, response_backend=None):
    signature = inspect.signature(native.run_minibatch_reflect)
    settings = budget_settings(config)

    def dispatch(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments
        if params["workers"] != 1 or params["minibatch_size"] != settings["max_minibatch_size"]:
            raise ValueError("Budgeted reflection requires one analyst worker and its pinned group limit")
        if params.get("update_mode", "patch") != "patch":
            raise ValueError("Token-budget reflection is currently pinned to patch mode")
        results = params["results"]
        ids = [str(row["id"]) for row in results]
        if len(set(ids)) != len(ids) or any(row.get("hard") not in (0, 1, False, True) for row in results):
            raise ValueError("Expected unique episode IDs and binary outcomes")
        prediction = Path(params["prediction_dir"])
        # Native formatting silently skips missing/empty conversations; do not
        # turn that behavior into a lossless-splitting claim.
        for identifier in ids:
            path = prediction / identifier / "conversation.json"
            if not path.is_file() or not json.loads(path.read_text()):
                raise ValueError(f"Missing complete reflection conversation: {identifier}")
        folder = Path(params["patches_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        authorized = dict(config.get("reflection_authorized_overlong_conversations", {}))
        for identifier in set(ids) & authorized.keys():
            digest = hashlib.sha256((prediction / identifier / "conversation.json").read_bytes()).hexdigest()
            if digest != authorized[identifier]:
                raise ValueError(f"Authorized reflection skip belongs to a different conversation: {identifier}")
        contract = {"settings": settings, "skill_sha256": hashlib.sha256(params["skill_content"].encode()).hexdigest(),
                    "authorized_overlong_conversations": authorized,
                    "results_sha256": sha(results), "kwargs_sha256": sha({key: value for key, value in params.items() if key != "results"}),
                    "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for identifier in ids for path in sorted((prediction / identifier).glob("*")) if path.is_file()}}
        plan_path = folder / "reflection_groups.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan.get("contract") != contract or plan.get("plan_sha256") != sha(plan.get("groups")):
                raise ValueError("Reflection grouping checkpoint differs from this exact batch")
        else:
            if any(folder.glob("minibatch_*.json")):
                raise ValueError("Cannot adopt native patch files without the adaptive grouping contract")
            groups, attempts = [], []
            skill_aware = params.get("skill_aware_reflection")
            if skill_aware is None:
                skill_aware = native.is_skill_aware_enabled()
            appendix = params.get("skill_aware_appendix_source") or native.get_skill_aware_appendix_source()
            for kind in ("fail", "succ"):
                chosen = [row for row in results if bool(row["hard"]) == (kind == "succ")]
                if kind == "succ" and params["failure_only"]:
                    chosen = []
                seed = params["random_seed"]
                ordered = native._shuffle_for_minibatch(chosen, seed if kind == "fail" or seed is None else seed + 1)
                analyst_kwargs = {key: params[key] for key in (
                    "edit_budget", "step_buffer_context", "trajectory_memory_context", "meta_skill_context", "update_mode")}
                analyst_kwargs.update(system_prompt=params["error_system" if kind == "fail" else "success_system"],
                                      skill_aware_reflection=skill_aware)
                if kind == "fail":
                    analyst_kwargs["rejection_context"] = params["rejection_context"]
                else:
                    analyst_kwargs["emit_appendix_notes"] = appendix != "failure_only"
                kind_index = 0
                for original_index, original_group in enumerate(native._split_minibatches(ordered, params["minibatch_size"])):
                    def render(rows):
                        return capture_native_prompt(native, kind, params["skill_content"], rows, str(prediction), analyst_kwargs)
                    divided, checked = partition_group(original_group, render, counter, settings,
                                                       authorized_overlong_ids=authorized)
                    attempts.extend({**row, "kind": kind, "original_group": original_index} for row in checked)
                    for group in divided:
                        groups.append({**group, "kind": kind, "tag": f"minibatch_{kind}_{kind_index:03d}",
                                       "original_group": original_index, "analyst_kwargs": analyst_kwargs})
                        kind_index += 1
            plan = {"contract": contract, "groups": groups, "plan_sha256": sha(groups), "checked_candidates": attempts}
            _write_once(plan_path, plan)
        skipped = [row for row in plan['checked_candidates'] if row.get('skipped')]
        _write_once(folder / 'reflection_skips.json', {
            'authorized_conversations': authorized, 'skipped': skipped,
            'rollout_episode_count': len(results),
            'note': 'Reflection only; rollout answers and evaluation results remain unchanged.'})
        print(f"    [REFLECT token budget] {len(results)} episodes -> {len(plan['groups'])} groups (max={settings['max_minibatch_size']}, no truncation)", flush=True)
        by_id = {str(row["id"]): row for row in results}
        patches = []
        for group in plan["groups"]:
            receipt_path = folder / "completed_groups" / (group["tag"] + ".json")
            identity = {"version": VERSION, "plan_sha256": plan["plan_sha256"], "group_sha256": sha(group)}
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if (receipt.get("identity") != identity or receipt.get("response_sha256") != sha(receipt.get("response"))
                        or receipt.get("patch_sha256") != sha(receipt.get("patch"))):
                    raise ValueError("Corrupt or mismatched reflection completion receipt")
                patch = receipt["patch"]
            else:
                original_chat = native.chat_optimizer
                errors, responses, returned = [], [], []
                original_post = response_backend._post_chat_completion if response_backend is not None else None

                def guarded_post(payload, timeout, backend_config):
                    if payload.get("messages") != prompt_messages(group["call"]):
                        raise RuntimeError("Actual backend messages differ from checked reflection input")
                    if payload.get("max_tokens") != settings["output_reserve_tokens"]:
                        raise RuntimeError("Actual backend output limit differs from checked reserve")
                    response = original_post(payload, timeout, backend_config)
                    responses.append(response)
                    choices = response.get("choices") or []
                    tokens = (response.get("usage") or {}).get("prompt_tokens")
                    if (not choices or choices[0].get("finish_reason") == "length" or type(tokens) is not int
                            or not group["input_tokens"] <= tokens <= group["input_tokens"] + settings["safety_margin_tokens"]
                            or tokens + settings["output_reserve_tokens"] > settings["context_tokens"]):
                        # Persist the response; stop outside native broad retry
                        # handlers instead of accepting a truncated analysis.
                        errors.append("Actual reflection response was truncated or token/template verification failed")
                    return response

                def guarded_chat(**chat_kwargs):
                    if chat_kwargs != group["call"]:
                        errors.append("Native reflection prompt changed after token checks")
                        raise RuntimeError(errors[-1])
                    try:
                        answer = original_chat(**chat_kwargs)
                        returned.append(answer)
                        return answer
                    except Exception as exc:
                        errors.append(f"Reflection model call failed: {type(exc).__name__}: {exc}")
                        raise

                native.chat_optimizer = guarded_chat
                if response_backend is not None:
                    response_backend._post_chat_completion = guarded_post
                try:
                    patch = getattr(native, "run_error_analyst_minibatch" if group["kind"] == "fail" else "run_success_analyst_minibatch")(
                        params["skill_content"], [by_id[i] for i in group["episode_ids"]], str(prediction), **group["analyst_kwargs"])
                finally:
                    native.chat_optimizer = original_chat
                    if response_backend is not None:
                        response_backend._post_chat_completion = original_post
                response_record = {"raw_responses": responses, "chat_returns": returned}
                if response_backend is not None and not responses:
                    errors.append("No actual Qwen response was observed by the token guard")
                if len(returned) == 1:
                    try:
                        parsed = native.extract_json(returned[0][0])
                    except Exception:
                        parsed = None
                    if (not isinstance(parsed, dict) or not isinstance(parsed.get("patch"), dict)
                            or not isinstance(parsed["patch"].get("edits"), list) or patch is None):
                        errors.append("Analyst output did not contain its required patch object; not a valid no-op")
                if errors or len(returned) != 1:
                    _write_once(folder / "group_errors" / (group["tag"] + ".json"),
                                {"identity": identity, "errors": errors, "response": response_record})
                    raise RuntimeError("Reflection group failed; no Skill update is allowed: " + "; ".join(errors))
                receipt = {"identity": identity, "patch": patch, "response": response_record,
                           "response_sha256": sha(response_record), "patch_sha256": sha(patch)}
                _write_once(receipt_path, receipt)
            # Write even None: a completed no-op must not be re-generated on resume.
            _write_once(folder / (group["tag"] + ".json"), patch)
            if patch is not None:
                patches.append(patch)
        return patches

    return dispatch


@contextmanager
def use_token_budget_reflection(config: Mapping[str, Any], *, native=None, counter=None, response_backend=None):
    if not config.get("reflection_token_budget_enabled"):
        yield
        return
    if native is None:
        import skillopt.gradient.reflect as native
    if response_backend is None:
        import skillopt.model.qwen_backend as response_backend
    if counter is None:
        from agentic_rag.skillopt.reflection_tokenizer import SshOllamaTokenCounter
        counter = SshOllamaTokenCounter(config)
    original = native.run_minibatch_reflect
    native.run_minibatch_reflect = build_budgeted_dispatcher(native, config, counter, response_backend=response_backend)
    try:
        yield
    finally:
        native.run_minibatch_reflect = original
