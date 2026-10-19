import copy

import pytest
from phase1_fixtures import raw_example
from test_phase1 import rules

from arag_skillopt.checkpoint import load_checkpoint
from arag_skillopt.final_test import FinalTestRunner


def runner(root, calls, runtime=None, unknown=False):
    def rollout(q, skill):
        assert set(q) == {"id", "question"}  # test gold never reaches Agent
        calls.append((q["id"], skill["retrieval_policy"][0]["title"]))
        raw = raw_example(q["id"])
        raw["question"] = q["question"]
        return raw

    return FinalTestRunner(
        root,
        rollout,
        lambda q, raw: (
            {"correct": None, "status": "unavailable"}
            if unknown
            else {"correct": True, "status": "ok"}
        ),
        runtime or {"snapshot": "v1"},
    )


def questions():
    return [
        {"id": "t1", "question": "Where?", "answer": "York"},
        {"id": "t2", "question": "Who?", "answer": "Alice"},
    ]


def test_test_only_two_arms_resume_and_contract(tmp_path):
    calls = []
    initial = rules()
    final = copy.deepcopy(initial)
    final["retrieval_policy"][0]["title"] = "Optimized"
    engine = runner(tmp_path, calls)
    result = engine.run(questions(), initial, final, {"train1"}, {"val1"})
    assert len(calls) == 4 and result["paired"]["both_correct"] == 2
    assert result["initial"]["accuracy"] == result["optimized"]["accuracy"] == 1
    assert result["accuracy_delta"] == 0 and not result["test_feedback_to_optimizer"]
    assert not (tmp_path / "batches").exists()
    engine.run(questions(), initial, final, {"train1"}, {"val1"})
    assert len(calls) == 4
    assert load_checkpoint(tmp_path / "checkpoint.json")["status"] == "complete"
    with pytest.raises(ValueError, match="contract changed"):
        runner(tmp_path, calls, {"snapshot": "v2"}).run(
            questions(), initial, final, {"train1"}, {"val1"}
        )


def test_identical_sealed_skills_reuse_not_fake_independent_run(tmp_path):
    calls = []
    initial = rules()
    result = runner(tmp_path, calls).run(questions(), initial, initial, set(), set())
    assert len(calls) == 2 and result["identical_skills_reused"]


def test_unknown_is_not_wrong_and_split_leakage_rejected(tmp_path):
    initial = rules()
    with pytest.raises(ValueError, match="disjoint"):
        runner(tmp_path, []).run(questions(), initial, initial, {"t1"}, set())
    result = runner(tmp_path, [], unknown=True).run(
        questions(), initial, initial, set(), set()
    )
    assert result["paired"]["unknown"] == 2
    assert result["accuracy_delta"] is None
    assert result["initial"]["accuracy"] is None and result["initial"]["correct"] == 0
    assert result["initial"]["unknown"] == 2
