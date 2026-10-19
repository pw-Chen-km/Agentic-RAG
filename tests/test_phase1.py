import copy
import json
from pathlib import Path

import pytest
from phase1_fixtures import raw_example

from arag_skillopt.optimizer_tools import parse_tool_call
from arag_skillopt.phase1 import Phase1, Phase1Config
from arag_skillopt.phase1_gate import decide
from arag_skillopt.skill_rules import RuleStore, load_rules, skill_sha256

ROOT = Path(__file__).parents[1]


def rules():
    return load_rules(ROOT / "skills/arag_initial_rules.json")


def prompts():
    return {
        k: (ROOT / "prompts" / f"phase1_{k}.md").read_text()
        for k in ("common", "success", "failure", "merge")
    }


def edits_call(rid="R01", ids=("t00", "t02"), operation="replace_rule"):
    args = {
        "rule_id": rid,
        "reason": "Observed repeated pattern; test an unverified policy hypothesis. Consider added calls.",
        "supporting_case_ids": list(ids),
        "rule": {
            "title": "Candidate check " + rid,
            "when": "A relevant candidate " + rid + " exists",
            "action_sequence": [
                "Read a promising candidate before searching for already available information."
            ],
        },
    }
    if operation == "add_rule":
        args["section"] = "retrieval_policy"
    return {"function": {"name": operation, "arguments": args}}


def diagnostic_call(ids, meta=False):
    return {
        "function": {
            "name": "report_diagnoses",
            "arguments": {
                "reason": "Save observed decisions for cross-group review, not adoption.",
                "diagnoses": [
                    {
                        "pattern": "Inspect a relevant retrieved candidate",
                        "failure_layer": "policy",
                        "basis": "policy_hypothesis",
                        "policy_limitation": "Candidate information may be unused.",
                        "hypothesis": "Test inspecting a promising candidate before another search.",
                        "cost_tradeoff": "May add a read; savings are not demonstrated.",
                        "target_rule_ids": ["R01"],
                        "observations": [
                            {
                                "case_id": qid,
                                "decisions": ["A1", "B1"] if meta else ["1"],
                                "behavior": "Selected search before reading.",
                                "outcome": "Recorded answer and cost.",
                                "interpretation": "support",
                            }
                            for qid in ids
                        ],
                    }
                ]
                if ids
                else [],
            },
        }
    }


class Optimizer:
    def __init__(self, events):
        self.events = events
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        text = kwargs["messages"][0]["content"]
        user = kwargs["messages"][1]["content"]
        if "Batch diagnoses (all groups):\n" in user:
            self.events.append("merge")
            calls = [
                edits_call(),
                edits_call("R02"),
                edits_call("R03", operation="add_rule"),
            ]
        else:
            ids = [
                line.split(": ", 1)[1]
                for line in user.splitlines()
                if line.startswith("Case ID:")
            ]
            kind = "success" if "marked correct" in text else "failure"
            self.events.append((kind, ids))
            calls = [diagnostic_call(ids)]
        return {
            "message": {"tool_calls": calls},
            "input_tokens": 100,
            "output_tokens": 50,
        }


def make_engine(tmp_path, events, accept=True):
    initial = rules()

    def rollout(q, skill):
        assert set(q) == {"id", "question"}
        events.append(("rollout", q["id"], skill_sha256(skill)))
        raw = raw_example(q["id"])
        raw["question"] = q["question"]
        changed = skill != initial
        raw["answer"] = (
            "good"
            if (changed and accept)
            or (q["id"].startswith("t") and int(q["id"][1:]) % 2 == 0)
            else "bad"
        )
        raw["usage"]["total_tokens"] = 500
        return raw

    def evaluator(q, raw):
        return {"correct": raw["answer"] == "good", "status": "ok"}

    engine = Phase1(
        tmp_path,
        initial,
        Optimizer(events),
        rollout,
        evaluator,
        prompts(),
        "Fixed tools",
    )
    return engine


def questions(prefix, n):
    return [
        {
            "id": f"{prefix}{i:02}",
            "question": f"Question {prefix}{i}",
            "answer": "reference",
        }
        for i in range(n)
    ]


def test_full40_batch_then_one_candidate_three_edits_gate_and_remainder_resume(
    tmp_path,
):
    events = []
    engine = make_engine(tmp_path, events)
    summary = engine.run(questions("t", 42), questions("v", 2))
    first_reflect = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, tuple) and e[0] in ("success", "failure")
    )
    assert first_reflect == 40
    assert len({e[2] for e in events[:40]}) == 1
    reflections = [
        e for e in events if isinstance(e, tuple) and e[0] in ("success", "failure")
    ]
    assert all(len(e[1]) <= 5 for e in reflections)
    assert len(reflections) == 9
    assert summary["completed_batches"] == 2 and summary["accepted_updates"] == 1
    receipt = json.loads((tmp_path / "batches/001/completed.json").read_text())["value"]
    assert len(receipt["edits"]) == 3
    assert receipt["gate"]["accepted"]
    assert len([e for e in events if e[0] == "rollout" and e[1].startswith("v")]) == 4
    assert not summary["meta_executed"] and not summary["test_executed"]
    request = engine.client.requests[0]["messages"][1]["content"]
    assert "source_prompt_sha256" not in request and "skill_hash" not in request
    calls = len(events)
    again = make_engine(tmp_path, events).run(questions("t", 42), questions("v", 2))
    assert again == summary and len(events) == calls


def test_gate_reject_keeps_parent(tmp_path):
    engine = make_engine(tmp_path, [], accept=False)
    summary = engine.run(questions("t", 4), questions("v", 2))
    assert summary["accepted_updates"] == 0
    assert summary["final_skill_hash"] == skill_sha256(rules())


def test_optimizer_three_reflection_edits_rejected_and_three_merge_accepted(tmp_path):
    engine = make_engine(tmp_path, [])

    class Bad:
        def chat(self, **kwargs):
            return {
                "message": {
                    "tool_calls": [
                        edits_call(),
                        edits_call("R02"),
                        edits_call("R03", operation="add_rule"),
                    ]
                }
            }

    engine.client = Bad()
    proposal = engine.propose(
        "success", rules(), "cases", ["t00", "t02"], tmp_path / "reflect.json", 2
    )
    assert proposal["status"] == "no_change_after_retry" and len(proposal["audit"]) == 3
    assert proposal["edits"] == []
    proposal = engine.propose(
        "merge", rules(), "cases", ["t00", "t02"], tmp_path / "merge.json", 3
    )
    assert len(proposal["edits"]) == 3


def test_token_budget_splits_then_skips_singleton_without_truncation(tmp_path):
    engine = make_engine(tmp_path, [])
    engine.cfg = Phase1Config(max_input_tokens=50)
    episodes = [
        {
            "question_id": f"t{i}",
            "view": {
                "question": "x" * 100,
                "question_id": f"t{i}",
                "skill_version": "batch parent",
                "decisions": [],
                "episode_evaluation": {},
            },
        }
        for i in range(5)
    ]
    proposals = engine.reflect_group(
        "success", rules(), episodes, tmp_path / "reflections", "g"
    )
    assert len(proposals) == 5 and all(
        p["status"] == "input_too_long" for p in proposals
    )
    assert not engine.client.requests


def test_no_change_or_missing_evaluation_never_updates_or_validates(tmp_path):
    engine = make_engine(tmp_path, [])
    engine.evaluator = lambda q, raw: {"correct": None, "status": "service_error"}
    summary = engine.run(questions("t", 4), questions("v", 2))
    assert summary["accepted_updates"] == 0 and not engine.client.requests
    assert not (tmp_path / "episodes/validation").exists()
    groups = json.loads((tmp_path / "batches/001/groups.json").read_text())
    assert len(groups["unavailable"]) == 4


def test_resume_after_interrupted_rollout_no_duplicate_completed_episode(tmp_path):
    events = []
    engine = make_engine(tmp_path, events)
    original = engine.rollout

    def failing(q, skill):
        if q["id"] == "t02":
            raise RuntimeError("simulated interruption")
        return original(q, skill)

    engine.rollout = failing
    with pytest.raises(RuntimeError):
        engine.run(questions("t", 4), questions("v", 2))
    resumed = make_engine(tmp_path, events)
    resumed.run(questions("t", 4), questions("v", 2))
    assert len([e for e in events if e[0] == "rollout" and e[1] == "t00"]) == 1


def test_contract_changes_and_split_overlap_rejected(tmp_path):
    engine = make_engine(tmp_path, [])
    with pytest.raises(ValueError, match="disjoint"):
        engine.run(questions("t", 2), questions("t", 2))
    engine.run(questions("t", 2), questions("v", 2))
    with pytest.raises(ValueError, match="contract changed"):
        engine.run(questions("t", 4), questions("v", 2))


def test_per_rule_not_total_edit_tokens_and_no_audit_cost_count():
    edit1 = parse_tool_call(edits_call())
    edit2 = parse_tool_call(edits_call("R02"))
    edit1["reason"] = "a" * 10000
    store = RuleStore(rules())
    assert store.apply([edit1, edit2])
    edit1["rule"]["action_sequence"] = ["different words " * 400]
    with pytest.raises(ValueError, match="300"):
        RuleStore(rules()).apply([edit1])
    with pytest.raises(ValueError, match="same rule"):
        RuleStore(rules()).apply([edit2, edit2])


def test_gate_tolerance_and_token_cost_not_retrieved_cost():
    def row(acc, tokens, calls):
        return [
            {
                "question_id": f"q{i}",
                "evaluation": {"correct": i < acc, "status": "ok"},
                "raw": {
                    "usage": {"complete": True, "total_tokens": tokens},
                    "policy_calls": calls,
                },
            }
            for i in range(100)
        ]

    assert decide(row(80, 100, 5), row(81, 104, 5))["accepted"]
    assert not decide(row(80, 100, 5), row(90, 160, 4))["accepted"]
    assert decide(row(80, 100, 5), row(79, 90, 5))["accepted"]
    assert not decide(row(80, 100, 5), row(70, 90, 5))["accepted"]
