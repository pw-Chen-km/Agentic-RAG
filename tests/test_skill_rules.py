from __future__ import annotations

import json
from pathlib import Path

import pytest

from arag_skillopt.skill_rules import RuleStore, compile_initial_skill, load_rules
from arag_skillopt.optimizer_tools import parse_tool_call, tool_schemas
from arag_skillopt.candidate_replay import replay_rejected_candidate
from arag_skillopt.checkpoint import load_checkpoint, save_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def test_initial_skill_is_a_faithful_small_schema() -> None:
    payload = load_rules(ROOT / "skills/arag_initial_rules.json")
    assert [rule["rule_id"] for rule in payload["retrieval_policy"]] == ["R01", "R02"]
    assert all(set(rule) == {"rule_id", "title", "when", "action_sequence"} for rule in payload["retrieval_policy"])
    rendered = compile_initial_skill(payload)
    assert "stop_or_recovery" not in rendered
    assert "after_observation" not in rendered


def test_question_entities_initial_skill_exposes_all_enabled_retrieval_choices() -> None:
    payload = load_rules(ROOT / "skills/arag_entityhop_question_entities_initial_rules.json")
    rendered = compile_initial_skill(payload)
    assert [rule["rule_id"] for rule in payload["retrieval_policy"]] == ["R01", "R02", "R03"]
    for action in ("semantic_search", "entity_hop", "linear_hop", "read_chunk", "stop_and_answer"):
        assert action in rendered
    assert "do not call both automatically" in rendered


def test_rule_store_supports_add_replace_delete() -> None:
    payload = load_rules(ROOT / "skills/arag_initial_rules.json")
    store = RuleStore(payload)
    new = {
        "rule_id": "R08",
        "title": "Read candidates",
        "when": "A search returns a promising chunk",
        "action_sequence": ["Read the chunk before another search"],
    }
    store.apply([{"operation": "add", "rule_id": "R08", "rule": new}])
    assert any(rule["rule_id"] == "R08" for rule in store.rules)
    new["title"] = "Read useful candidates"
    store.apply([{"operation": "replace", "rule_id": "R08", "rule": new}])
    store.apply([{"operation": "delete", "rule_id": "R08"}])
    assert not any(rule["rule_id"] == "R08" for rule in store.rules)


def test_forbidden_fields_are_rejected() -> None:
    payload = load_rules(ROOT / "skills/arag_initial_rules.json")
    bad = json.loads(json.dumps(payload))
    bad["retrieval_policy"][0]["stop_or_recovery"] = "never"
    path = ROOT / "tests/_bad_rules.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            load_rules(path)
    finally:
        path.unlink(missing_ok=True)


def test_native_tool_uses_outer_rule_id_and_normalizes_it() -> None:
    call = {
        "function": {
            "name": "add_rule",
            "arguments": {
                "section": "retrieval_policy",
                "rule_id": "R08",
                "rule": {
                    "title": "Read candidates",
                    "when": "A search returns a promising chunk",
                    "action_sequence": ["Read the chunk before searching again"],
                },
                "reason": "Repeated across two questions",
                "supporting_case_ids": ["q1", "q2"],
            },
        }
    }
    normalized = parse_tool_call(call)
    assert normalized["operation"] == "add"
    assert normalized["rule"]["rule_id"] == "R08"


def test_native_tool_rejects_conflicting_nested_id() -> None:
    call = {
        "function": {
            "name": "replace_rule",
            "arguments": {
                "rule_id": "R01",
                "rule": {"rule_id": "R02", "title": "x", "when": "y", "action_sequence": ["z"]},
                "reason": "reason",
                "supporting_case_ids": [],
            },
        }
    }
    with pytest.raises(ValueError, match="outer and inner"):
        parse_tool_call(call)


def test_add_schema_keeps_id_at_outer_level() -> None:
    add = next(item for item in tool_schemas(["R01"])
               if item["function"]["name"] == "add_rule")
    params = add["function"]["parameters"]
    assert "rule_id" in params["properties"]
    assert "rule_id" not in params["properties"]["rule"]["properties"]


def test_optimizer_prompts_are_exclusive_to_arag_actions() -> None:
    root = ROOT / "prompts"
    text = (root / "retrieval.md").read_text(encoding="utf-8") + (root / "meta.md").read_text(encoding="utf-8")
    assert "EXPAND" not in text
    assert "E/S/C" not in text


def test_replay_is_train_only_and_deduplicated() -> None:
    seen = set()
    calls = []
    def rollout(question_id: str, skill_hash: str):
        calls.append((question_id, skill_hash))
        return {"question_id": question_id}
    first = replay_rejected_candidate(candidate_skill_hash="c", parent_skill_hash="p",
                                      train_question_ids=["q1", "q2"], rollout=rollout, seen_keys=seen)
    second = replay_rejected_candidate(candidate_skill_hash="c", parent_skill_hash="p",
                                       train_question_ids=["q1", "q2"], rollout=rollout, seen_keys=seen)
    assert len(first) == 2 and second == []
    assert calls == [("q1", "c"), ("q2", "c")]


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, {"stage": "retrieval", "batch": 1})
    assert load_checkpoint(path) == {"stage": "retrieval", "batch": 1}
