import json
from pathlib import Path

import pytest

from agentic_rag.skillopt.workflow.config import WorkflowConfig
from agentic_rag.skillopt.workflow.demo import DemoBackend
from agentic_rag.skillopt.workflow.rule_store import RuleStore
from agentic_rag.skillopt.workflow.runner import WorkflowRunner


ROOT = Path(__file__).parents[1]


def store():
    return RuleStore.from_json(ROOT / "skills" / "workflow_rules_v2.json")


def test_json_is_source_for_stable_compiled_markdown():
    rules = store()
    markdown = rules.render_markdown()
    assert markdown == (ROOT / "skills" / "workflow_seed_v2.md").read_text()
    assert RuleStore.from_markdown(markdown).json_hash() == rules.json_hash()
    assert [r.rule_id for r in rules.all_rules()] == ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "A01"]


def test_stage_scope_and_edit_limits():
    rules = store()
    replacement = {
        "operation": "replace", "rule_id": "R05", "section": "recovery_policy",
        "rule": {"title": "short", "when": "no progress", "action_sequence": ["READ"],
                  "stop_or_recovery": "change path", "exceptions": []},
    }
    candidate, audit = rules.apply_edits([replacement], stage="retrieval")
    assert candidate.rule_index()["R05"].title == "short"
    assert audit["changed_rule_ids"] == ["R05"]
    with pytest.raises(ValueError, match="forbidden"):
        rules.apply_edits([{**replacement, "rule_id": "A01"}], stage="retrieval")
    with pytest.raises(ValueError, match="too many"):
        rules.apply_edits([replacement, {**replacement, "rule_id": "R06"}], stage="retrieval", max_edits=1)
    with pytest.raises(ValueError, match="target not found"):
        rules.apply_edits([{**replacement, "rule_id": "R99"}], stage="retrieval")


def test_add_duplicate_and_semantic_duplicate_are_rejected():
    rules = store()
    duplicate_id = {"operation": "add", "rule_id": "R05", "section": "recovery_policy",
                    "rule": {"title": "x", "when": "x", "action_sequence": ["x"],
                              "stop_or_recovery": "x", "exceptions": []}}
    with pytest.raises(ValueError, match="existing"):
        rules.apply_edits([duplicate_id], stage="retrieval")
    duplicate_content = {"operation": "add", "rule_id": "R08", "section": "recovery_policy",
                         "rule": rules.rule_index()["R05"].to_mapping()}
    with pytest.raises(ValueError, match="duplicate semantic"):
        rules.apply_edits([duplicate_content], stage="retrieval")


def test_v2_runner_uses_small_edits_and_meta_hashes(tmp_path):
    rules = store()
    seed = rules.render_markdown()
    train = [{"id": f"train-{i}", "question": f"question {i}", "answer": "a",
              "scope_id": "s", "split": "train"} for i in range(6)]
    validation = [{"id": f"val-{i}", "question": f"validation {i}", "answer": "a",
                  "scope_id": "s", "split": "validation"} for i in range(2)]
    config = WorkflowConfig(rollout_batch_size=4, reflection_minibatch_size=2,
                            use_validation_gate=False, enable_answer_updates=False)
    result = WorkflowRunner(backend=DemoBackend(), output=tmp_path, config=config,
        contract={"fake": True}, train=train, validation=validation, skill=seed, rules=rules).run()
    assert result["status"] == "complete"
    assert result["test_executed"] is False
    receipts = sorted(tmp_path.glob("retrieval/batch_*/completed.json"))
    assert len(receipts) == 2
    for path in receipts:
        receipt = json.loads(path.read_text())
        assert len(receipt["edits"]) <= 2
        assert "parent_rules" in receipt and "candidate_rules" in receipt
    meta_requests = list(tmp_path.glob("meta/batch_*/reflect_*.json"))
    assert meta_requests
    request = json.loads(meta_requests[0].read_text())["request"]
    assert "rule_catalog" in request
    assert all("parent_skill_hash" in case and "candidate_skill_hash" in case
               for case in request["cases"])
    assert not list(tmp_path.glob("answer/batch_*/completed.json"))


def test_config_exposes_v2_limits():
    config = WorkflowConfig.from_mapping({"max_optimizer_input_tokens": 16000,
                                          "max_edits_per_reflection": 2,
                                          "max_edits_per_batch": 2,
                                          "max_edit_tokens": 250,
                                          "max_trainable_skill_tokens": 1500})
    assert (config.max_optimizer_input_tokens, config.max_edits_per_reflection,
            config.max_edits_per_batch, config.max_edit_tokens,
            config.max_trainable_skill_tokens) == (16000, 2, 2, 250, 1500)
