from arag_skillopt.paired_summary import (
    action_sequence,
    is_informative_pair,
    summarize_pair,
)


def episode(actions, evidence=(), tokens=100, calls=2):
    return {
        "question_id": "q1",
        "trajectory": [
            {
                "resolved_decision": {"action": action, "substrate_id": f"S{i}"},
                "novel_node_ids": [node],
            }
            for i, (action, node) in enumerate(zip(actions, evidence), 1)
        ],
        "policy_calls": calls,
        "usage": {"total_tokens": tokens, "complete": True},
    }


def evaluation(value):
    return {"status": "ok", "hard": value}


def test_identical_actions_have_no_first_divergence():
    parent = episode(["SEARCH", "READ"], ["n1", "n2"])
    candidate = episode(["SEARCH", "READ"], ["n1", "n2"])
    summary = summarize_pair("q1", parent, candidate, evaluation(True), evaluation(True))
    assert summary["first_divergence"] is None
    assert summary["status"] == "identical"
    assert not is_informative_pair(summary)


def test_first_divergence_is_one_based_and_keeps_stable_action():
    parent = episode(["SEARCH", "READ", "FINISH"], ["n1", "n2", "n3"])
    candidate = episode(["SEARCH", "EXPAND", "FINISH"], ["n1", "n2", "n3"])
    summary = summarize_pair("q1", parent, candidate, evaluation(True), evaluation(True))
    assert summary["first_divergence"] == {
        "step": 2,
        "parent_action": "READ:S2",
        "candidate_action": "EXPAND:S2",
    }


def test_new_evidence_delta_is_counted():
    parent = episode(["SEARCH"], ["n1"])
    candidate = episode(["SEARCH"], ["n1", "n2"])
    # The candidate's second evidence record is represented by a second step.
    candidate["trajectory"].append(
        {"resolved_decision": {"action": "READ", "substrate_id": "S2"},
         "novel_node_ids": ["n2"]}
    )
    summary = summarize_pair("q1", parent, candidate, evaluation(True), evaluation(True))
    assert summary["parent_new_evidence_count"] == 1
    assert summary["candidate_new_evidence_count"] == 2
    assert summary["new_evidence_delta"] == 1


def test_unknown_is_not_false():
    parent = episode(["SEARCH"], ["n1"])
    candidate = episode(["SEARCH"], ["n1"], tokens=120)
    summary = summarize_pair("q1", parent, candidate, evaluation(None), evaluation(False))
    assert summary["parent_answer_correct"] is None
    assert summary["candidate_answer_correct"] is False
    assert summary["status"] == "unknown"
    assert summary["informative"]


def test_action_difference_is_informative_even_when_answer_and_cost_match():
    parent = episode(["SEARCH"], ["n1"])
    candidate = episode(["READ"], ["n1"])
    summary = summarize_pair("q1", parent, candidate, evaluation(True), evaluation(True))
    assert summary["first_divergence"] is not None
    assert is_informative_pair(summary)
