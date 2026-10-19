"""Synthetic tests of the real runtime/archive/optimizer scheduling boundaries."""

import copy
import json
from pathlib import Path

import pytest
from phase1_fixtures import (
    alternative_example,
    chunks_example,
    raw_example,
    source_example,
)
from test_phase1 import diagnostic_call, edits_call, questions, rules

from arag_skillopt.checkpoint import load_checkpoint, save_checkpoint
from arag_skillopt.meta_comparison import (
    comparison_views,
    load_branch,
    make_comparison,
    render_comparisons,
    rule_differences,
)
from arag_skillopt.meta_stage import MetaConfig, MetaStage
from arag_skillopt.diagnostics import diagnostic_tools
from arag_skillopt.phase1 import Phase1Config
from arag_skillopt.skill_rules import RuleStore, skill_sha256
from arag_skillopt.support import build_support_map
from arag_skillopt.tokens import count
from arag_skillopt.workflow_training import WorkflowTrainer

ROOT = Path(__file__).parents[1]


def prompts():
    result = {
        k: (ROOT / "prompts" / f"phase1_{k}.md").read_text()
        for k in ("common", "success", "failure", "merge")
    }
    result.update(
        {k: (ROOT / "prompts" / f"{k}.md").read_text() for k in ("meta", "meta_merge")}
    )
    return result


def native_edit(title, ids, rid="R01", operation="replace_rule"):
    call = edits_call(rid, tuple(ids), operation)
    call["function"]["arguments"]["rule"] = {
        "title": title,
        "when": "A candidate source concerns the unresolved relation " + rid,
        "action_sequence": [
            "Inspect the candidate, then target the remaining relation with the available tools."
        ],
    }
    return call


class Optimizer:
    def __init__(self, events):
        self.events, self.requests = events, []
        self.interrupt_ids = None

    def chat(self, **request):
        system = request["messages"][0]["content"]
        user = request["messages"][1]["content"]
        current = json.loads(
            user.split("Current Skill:\n", 1)[1].split("\nFixed tool guidance:", 1)[0]
        )
        merge = "Batch diagnoses (all groups):\n" in user
        meta = (
            "offline Meta retrieval-policy" in system
            or "Merge all Meta reflections" in system
        )
        if merge:
            proposals = json.loads(user.split("Batch diagnoses (all groups):\n", 1)[1])
            ids = sorted({i for p in proposals for i in p["case_ids"]})
        else:
            ids = [
                line.split(": ", 1)[1]
                for line in user.splitlines()
                if line.startswith("Case ID:")
            ]
        self.requests.append(copy.deepcopy(request))
        self.events.append(
            (
                "meta_merge"
                if meta and merge
                else "meta_reflect"
                if meta
                else "phase1_merge"
                if merge
                else "phase1_reflect",
                ids,
                skill_sha256(current),
            )
        )
        if meta and not merge and self.interrupt_ids and ids[0] == self.interrupt_ids:
            self.interrupt_ids = None
            raise KeyboardInterrupt("simulated process interruption")
        number = int(ids[0][1:]) // 40 + 1 if ids else 1
        title = ("Meta " if meta else "Retrieval ") + str(number)
        calls = (
            [diagnostic_call(ids, meta)]
            if not merge
            else (
                [native_edit(title, ids[:2])]
                if len(ids) >= 2
                else [
                    {
                        "function": {
                            "name": "no_change",
                            "arguments": {"reason": "Only one independent question"},
                        }
                    }
                ]
            )
        )
        return {
            "message": {"tool_calls": calls},
            "input_tokens": 100,
            "output_tokens": 30,
        }


def engine(root, events, meta_config=None):
    initial = rules()

    def rollout(q, skill):
        assert set(q) == {"id", "question"}  # gold never enters runtime
        title = skill["retrieval_policy"][0]["title"]
        events.append(("rollout", q["id"], title))
        raw = raw_example(q["id"])
        raw["question"] = q["question"]
        raw["usage"]["total_tokens"] = (
            700
            if title.startswith("Retrieval")
            else 450
            if title.startswith("Meta")
            else 500
        )
        # Retrieval improves validation accuracy but loses the cost gate.
        # Meta 1 improves at lower cost; Meta 2 worsens and must be rejected.
        raw["answer"] = (
            "good"
            if title.startswith("Retrieval")
            or title == "Meta 1"
            or (title != "Meta 2" and int(q["id"][1:]) % 2 == 0)
            else "bad"
        )
        return raw

    return WorkflowTrainer(
        root,
        initial,
        Optimizer(events),
        rollout,
        lambda q, raw: {"correct": raw["answer"] == "good", "status": "ok"},
        prompts(),
        "Fixed tool guidance",
        meta_config=meta_config,
    )


def bank(trainer, batches=1):
    return [
        load_checkpoint(
            trainer.phase1.root / "batches" / f"{i:03d}" / "completed.json"
        )["value"]
        for i in range(1, batches + 1)
    ]


def test_full40_pairs_frozen_meta_then_merge_gate_latest_skill_resume(tmp_path):
    events = []
    trainer = engine(tmp_path, events)
    summary = trainer.run(questions("t", 80), questions("v", 4))
    assert summary["phase1"]["accepted_updates"] == 0
    assert summary["meta"]["comparison_count"] == 80
    assert summary["meta"]["accepted_updates"] == 1
    assert summary["meta"]["reflection_count"] == 16
    assert not summary["test_executed"]
    first_meta = next(i for i, e in enumerate(events) if e[0] == "meta_reflect")
    prior_train = [
        e for e in events[:first_meta] if e[0] == "rollout" and e[1].startswith("t")
    ]
    assert len(prior_train) == 160  # 80 parent + 80 actual candidate replays
    assert len([e for e in events[:first_meta] if e[0] == "phase1_merge"]) == 2
    meta = [e for e in events if e[0] == "meta_reflect"]
    assert all(len(e[1]) == 5 for e in meta)
    assert len({e[2] for e in meta[:8]}) == 1
    assert len({e[2] for e in meta[8:]}) == 1 and meta[0][2] != meta[8][2]
    batches = [
        load_checkpoint(tmp_path / "meta/batches" / f"{i:03d}" / "completed.json")[
            "value"
        ]
        for i in (1, 2)
    ]
    assert batches[0]["gate"]["accepted"] and not batches[1]["gate"]["accepted"]
    assert batches[1]["proposal_parent_skill_hash"] == batches[0]["active_skill_hash"]
    assert (
        batches[1]["evidence_parent_skill_hash"]
        != batches[1]["proposal_parent_skill_hash"]
    )
    for r in bank(trainer, 2):
        assert not r["gate"]["accepted"] and len(r["meta_replay"]["pairs"]) == 40
        assert r["meta_replay"]["purpose"] == "meta_analysis_only"
        assert all(
            p["candidate"]["split"] == p["parent"]["split"] == "train"
            for p in r["meta_replay"]["pairs"]
        )
    requests = [
        r
        for r in trainer.phase1.client.requests
        if "offline Meta retrieval-policy" in r["messages"][0]["content"]
    ]
    assert len(requests) == 16
    assert all("skill_hash" not in r["messages"][1]["content"] for r in requests)
    assert all("\nCase ID: v" not in r["messages"][1]["content"] for r in requests)
    assert all(
        "Historical Skill A -> B rule differences" in r["messages"][1]["content"]
        for r in requests
    )
    before = len(events)
    assert (
        engine(tmp_path, events).run(questions("t", 80), questions("v", 4)) == summary
    )
    assert len(events) == before


def test_interrupted_candidate_replay_resumes_without_duplicate_answers(tmp_path):
    events = []
    trainer = engine(tmp_path, events)
    original = trainer.phase1.rollout

    def interrupted(q, skill):
        if q["id"] == "t02" and skill["retrieval_policy"][0]["title"].startswith(
            "Retrieval"
        ):
            raise KeyboardInterrupt("candidate interruption")
        return original(q, skill)

    trainer.phase1.rollout = interrupted
    with pytest.raises(KeyboardInterrupt):
        trainer.run(questions("t", 4), questions("v", 2))
    assert not (tmp_path / "phase1/batches/001/completed.json").exists()
    engine(tmp_path, events).run(questions("t", 4), questions("v", 2))
    assert (
        len([e for e in events if e[0] == "rollout" and e[1] == "t00"]) == 2
    )  # parent and candidate, once each


def test_meta_interruption_after_six_groups_reuses_completed_reflections(tmp_path):
    events = []
    trainer = engine(tmp_path, events)
    trainer.phase1.client.interrupt_ids = "t30"
    with pytest.raises(KeyboardInterrupt):
        trainer.run(questions("t", 40), questions("v", 2))
    prior = list((tmp_path / "meta/batches/001/reflections").glob("meta-*.json"))
    assert len([p for p in prior if ".attempt" not in p.name]) == 6
    train_calls = len([e for e in events if e[0] == "rollout" and e[1].startswith("t")])
    engine(tmp_path, events).run(questions("t", 40), questions("v", 2))
    assert (
        len([e for e in events if e[0] == "rollout" and e[1].startswith("t")])
        == train_calls
    )
    for start in ("t00", "t05", "t10", "t15", "t20", "t25"):
        assert (
            len([e for e in events if e[0] == "meta_reflect" and e[1][0] == start]) == 1
        )


def test_branch_id_hash_split_and_content_checked_and_order_not_assumed(tmp_path):
    trainer = engine(tmp_path, [])
    train, val = questions("t", 4), questions("v", 2)
    trainer.phase1.run(train, val)
    receipt = bank(trainer)[0]
    ref = receipt["meta_replay"]["pairs"][0]["parent"]
    for field, value in (
        ("split", "validation"),
        ("skill_hash", "wrong"),
        ("question_id", "t99"),
        ("episode_directory", "../../validation"),
    ):
        bad = {**ref, field: value}
        with pytest.raises(ValueError):
            load_branch(
                trainer.phase1, bad, receipt["parent_skill_hash"], "t00", train[0]
            )
    stage = MetaStage(trainer.phase1, tmp_path / "isolated_meta", rules(), MetaConfig())
    reordered = copy.deepcopy(receipt)
    reordered["meta_replay"]["pairs"].reverse()
    comps, _ = stage.comparisons(1, train, reordered, tmp_path / "comparison_test")
    assert [c["question_id"] for c in comps] == [q["id"] for q in train]
    raw_path = trainer.phase1.root / ref["episode_directory"] / "raw.json"
    corrupted = load_checkpoint(raw_path)
    corrupted["value"]["answer"] = "changed after archive"
    save_checkpoint(raw_path, corrupted)
    with pytest.raises(ValueError, match="content changed"):
        load_branch(trainer.phase1, ref, receipt["parent_skill_hash"], "t00", train[0])


def test_cross_branch_dedup_preserves_visibility_novelty_and_recall():
    source = source_example()
    entry = build_support_map(source, chunks_example())["questions"]["q1"]
    branch = {"raw": raw_example(), "evaluation": {"correct": True, "status": "ok"}}
    views = comparison_views(branch, copy.deepcopy(branch), entry)
    for a, b in zip(views["A"]["decisions"], views["B"]["decisions"]):
        assert a["retrieval_statistics"] == b["retrieval_statistics"]
        assert (
            a["supporting_fact_progress_evaluator_only"]
            == b["supporting_fact_progress_evaluator_only"]
        )
    content = views["B"]["decisions"][0]["results"][0]["items"][0]["content"]
    assert content[0]["same_as"].startswith("Branch A / Decision 1")
    candidate = RuleStore(rules()).apply(
        [
            {
                "operation": "replace",
                "rule_id": "R01",
                "rule": {
                    **rules()["retrieval_policy"][0],
                    "title": "Different wording",
                },
            }
        ]
    )
    q = {"id": "q1", "question": branch["raw"]["question"]}
    comparison = make_comparison(
        q, rules(), candidate, branch, copy.deepcopy(branch), 1
    )
    assert comparison["skip_reason"] == "identical_behavior_and_cost"
    assert rule_differences(
        rules(), candidate
    )  # wording is a recorded edit, not evidence of action change
    comparison["views"] = views
    rendered = render_comparisons([comparison])
    assert rendered.count("The River was written by Alice.") == 1
    assert (
        "parent_skill_hash" not in rendered and "candidate_skill_hash" not in rendered
    )


def test_overlong_meta_splits_and_skips_single_case_without_model_request(tmp_path):
    trainer = engine(tmp_path, [])
    train, val = questions("t", 5), questions("v", 2)
    trainer.phase1.run(train, val)
    stage = MetaStage(trainer.phase1, tmp_path / "meta_budget", rules(), MetaConfig())
    comps, _ = stage.comparisons(1, train, bank(trainer)[0], tmp_path / "budget_cases")
    stage.cfg = Phase1Config(max_input_tokens=50)
    before = len(trainer.phase1.client.requests)
    proposals = stage.reflect_comparisons(
        rules(), comps, tmp_path / "budget_reflect", "g"
    )
    assert len(proposals) == 5 and all(
        p["status"] == "input_too_long" for p in proposals
    )
    assert len(trainer.phase1.client.requests) == before
    assert list((tmp_path / "budget_reflect").glob("*.split.json"))


def test_budget_split_sends_bounded_two_and_three_case_native_requests(tmp_path):
    trainer = engine(tmp_path, [])
    train, val = questions("t", 5), questions("v", 2)
    trainer.phase1.run(train, val)
    stage = MetaStage(trainer.phase1, tmp_path / "meta_budget", rules(), MetaConfig())
    comps, _ = stage.comparisons(1, train, bank(trainer)[0], tmp_path / "split_cases")
    tools = diagnostic_tools()

    def size(cases):
        return count(
            {
                "messages": stage.messages("meta", rules(), render_comparisons(cases)),
                "tools": tools,
            }
        )

    budget = max(size(comps[:2]), size(comps[2:]))
    assert size(comps) > budget
    stage.cfg = Phase1Config(max_input_tokens=budget)
    before = len(stage.client.requests)
    proposals = stage.reflect_comparisons(
        rules(), comps, tmp_path / "split_reflect", "g"
    )
    assert [len(p["case_ids"]) for p in proposals] == [2, 3]
    assert all(p["status"] == "ok" for p in proposals)
    sent = stage.client.requests[before:]
    assert len(sent) == 2 and all(
        count({"messages": r["messages"], "tools": r["tools"]}) <= budget for r in sent
    )


def test_accepted_candidate_also_has_train_branches_and_latest_skill_for_meta(tmp_path):
    trainer = engine(tmp_path, [])
    rollout = trainer.phase1.rollout

    def cheaper_candidate(q, skill):
        raw = rollout(q, skill)
        if skill["retrieval_policy"][0]["title"].startswith("Retrieval"):
            raw["usage"]["total_tokens"] = 500
        return raw

    trainer.phase1.rollout = cheaper_candidate
    summary = trainer.run(questions("t", 4), questions("v", 2))
    receipt = bank(trainer)[0]
    assert summary["phase1"]["accepted_updates"] == 1
    assert receipt["gate"]["accepted"] and len(receipt["meta_replay"]["pairs"]) == 4
    assert receipt["meta_replay"]["candidate_gate_accepted"]
    meta_receipt = load_checkpoint(tmp_path / "meta/batches/001/completed.json")[
        "value"
    ]
    assert meta_receipt["proposal_parent_skill_hash"] == receipt["active_skill_hash"]
    assert meta_receipt["proposal_parent_skill_hash"] != receipt["parent_skill_hash"]


def test_actual_tool_difference_does_not_claim_identical_hidden_state():
    a, b = raw_example(), alternative_example()
    parent = rules()
    candidate = RuleStore(parent).apply(
        [
            {
                "operation": "replace",
                "rule_id": "R01",
                "rule": {**parent["retrieval_policy"][0], "title": "Refined guidance"},
            }
        ]
    )
    comparison = make_comparison(
        {"id": "q1", "question": a["question"]},
        parent,
        candidate,
        {"raw": a, "evaluation": {"status": "ok", "correct": True}},
        {"raw": b, "evaluation": {"status": "ok", "correct": True}},
        1,
    )
    assert comparison["eligible_for_reflection"]
    assert comparison["first_difference"] == {
        "decision": 1,
        "earlier_actions_and_results_equal": True,
        "confirmed_visible_results_equal": True,
        "exact_state_match": False,
    }
    assert comparison["costs"]["A"]["tool_calls"] == 3
    assert comparison["costs"]["B"]["tool_calls"] == 2
    assert (
        comparison["costs"]["A"]["model_calls"]
        == comparison["costs"]["B"]["model_calls"]
        == 3
    )
    assert not comparison["causal_claim"]


def test_same_skill_no_replay_meta_disabled_and_missing_evaluation(tmp_path):
    trainer = engine(tmp_path / "disabled", [], MetaConfig(enabled=False))
    summary = trainer.run(questions("t", 4), questions("v", 2))
    assert summary["meta"] is None and not summary["meta_executed"]
    assert bank(trainer)[0]["meta_replay"] is None
    trainer = engine(tmp_path / "unknown", [])
    trainer.phase1.evaluator = lambda q, raw: {"correct": None, "status": "unavailable"}
    summary = trainer.run(questions("t", 4), questions("v", 2))
    assert not trainer.phase1.client.requests
    assert bank(trainer)[0]["meta_replay"]["status"] == "no_changed_candidate"
    assert summary["meta"]["comparison_count"] == 0


def test_native_meta_three_edits_retry_and_batch_three_accepted(tmp_path):
    trainer = engine(tmp_path, [])
    stage = MetaStage(trainer.phase1, tmp_path / "meta_calls", rules(), MetaConfig())

    class Three:
        def chat(self, **kwargs):
            return {
                "message": {
                    "tool_calls": [
                        native_edit("One", ["t00", "t01"]),
                        native_edit("Two", ["t00", "t01"], "R02"),
                        native_edit("Three", ["t00", "t01"], "R03", "add_rule"),
                    ]
                }
            }

    stage.client = Three()
    reflected = stage.propose(
        "meta", rules(), "Comparison", ["t00", "t01"], tmp_path / "reflect.json", 2
    )
    assert (
        reflected["status"] == "no_change_after_retry" and len(reflected["audit"]) == 3
    )
    merged = stage.propose(
        "meta_merge", rules(), "Proposals", ["t00", "t01"], tmp_path / "merge.json", 3
    )
    assert len(merged["edits"]) == 3


def test_meta_prompt_and_total_skill_token_limits():
    text = prompts()["meta"]
    assert "CURRENT Skill" in text and "behavior did not occur" in text
    assert "both-wrong" in text and "hidden state" in text
    assert "shorter Skill or answer" in text and "hypothesis" in text
    assert count(text) < 1200
    with pytest.raises(ValueError):
        MetaConfig(reflection_minibatch_size=6)
    with pytest.raises(ValueError):
        MetaConfig(comparison_batch_size=20)
