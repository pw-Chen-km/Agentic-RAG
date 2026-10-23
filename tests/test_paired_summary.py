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
        "parent_action": ["READ:S2"],
        "candidate_action": ["EXPAND:S2"],
    }


def _parallel_episode(turns):
    trajectory = []
    for loop, calls in enumerate(turns, 1):
        for action, target in calls:
            trajectory.append({
                "loop": loop,
                "tool_name": action,
                "arguments": {"target": target},
                "tool_result": "",
            })
    return {"trajectory": trajectory, "policy_calls": len(turns)}


def test_parallel_call_order_is_not_a_policy_turn_divergence():
    parent = _parallel_episode([
        [("semantic_search", "Alice"), ("entity_hop", "Alice")],
        [("read_chunk", "17")],
    ])
    candidate = _parallel_episode([
        [("entity_hop", "Alice"), ("semantic_search", "Alice")],
        [("read_chunk", "17")],
    ])
    summary = summarize_pair("q1", parent, candidate, evaluation(True), evaluation(True))
    assert summary["first_divergence"] is None
    assert summary["parent_action_bundles"] == summary["candidate_action_bundles"] or {
        tuple(sorted(bundle)) for bundle in summary["parent_action_bundles"]
    } == {
        tuple(sorted(bundle)) for bundle in summary["candidate_action_bundles"]
    }


def test_parallel_calls_report_the_policy_turn_not_flattened_call_index():
    parent = _parallel_episode([
        [("semantic_search", "Alice"), ("entity_hop", "Alice")],
        [("read_chunk", "17"), ("read_chunk", "31")],
    ])
    candidate = _parallel_episode([
        [("semantic_search", "Alice"), ("entity_hop", "Alice")],
        [("linear_hop", "Alice"), ("read_chunk", "31")],
    ])
    summary = summarize_pair("q1", parent, candidate, evaluation(True), evaluation(True))
    assert summary["first_divergence"]["step"] == 2


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


def test_snippet_visibility_and_full_read_are_distinct():
    snippet = "Chunk ID: 15 (Similarity: 1.0)\nMatched: ... Alice was born in York ..."
    full = "[Chunk 15]\n" + "-" * 80 + "\nAlice was born in York.\n" + "=" * 80
    parent = {
        "trajectory": [{
            "loop": 1,
            "tool_name": "semantic_search",
            "arguments": {"query": "Alice"},
            "tool_result": snippet,
        }],
        "policy_calls": 1,
        "usage": {"total_tokens": 100},
    }
    candidate = {
        "trajectory": [
            {
                "loop": 1,
                "tool_name": "semantic_search",
                "arguments": {"query": "Alice"},
                "tool_result": snippet,
            },
            {
                "loop": 2,
                "tool_name": "read_chunk",
                "arguments": {"chunk_ids": ["15"]},
                "tool_result": full,
            },
        ],
        "policy_calls": 2,
        "usage": {"total_tokens": 120},
    }
    summary = summarize_pair("q1", parent, candidate, evaluation(False), evaluation(True))
    assert summary["new_evidence_delta"] == 0
    assert summary["parent_full_read_ids"] == []
    assert summary["candidate_full_read_ids"] == ["15"]
    assert summary["candidate_snippet_to_full_ids"] == ["15"]
    assert summary["full_read_delta"] == 1


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
    assert summary["action_changed"] is True
    assert is_informative_pair(summary)


def test_cost_summary_keeps_model_retrieval_and_tool_costs_separate():
    parent = episode(["SEARCH"], ["n1"], tokens=100, calls=1)
    parent.update(tool_calls=1, total_retrieved_tokens=20)
    candidate = episode(["SEARCH", "READ"], ["n1", "n2"], tokens=160, calls=2)
    candidate.update(tool_calls=2, total_retrieved_tokens=50)
    summary = summarize_pair("q1", parent, candidate, evaluation(True), evaluation(True))
    assert summary["tool_call_delta"] == 1
    assert summary["retrieved_token_delta"] == 30
    assert summary["model_token_delta"] == 60
    assert summary["token_delta"] == 60
    assert summary["informative"] is True
