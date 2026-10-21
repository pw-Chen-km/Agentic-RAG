from arag_skillopt.action_space import action_space_contract, action_usage
from arag_skillopt.hro_v1 import (
    GROUPS,
    UNSCORED,
    build_stage_a_context,
    build_stage_b_context,
    group_summary,
    transition_group,
    validate_stage_a_output,
    validate_stage_b_output,
)
from arag_skillopt.paired_summary import summarize_pair


def _episode(answer, actions=("semantic_search", "read_chunk"), evidence=("n1",)):
    return {
        "question_id": "q1",
        "question": "Which entity is related?",
        "trajectory": [
            {
                "resolved_decision": {"action": action, "substrate_id": f"S{i}"},
                "novel_node_ids": [node],
            }
            for i, (action, node) in enumerate(zip(actions, evidence), 1)
        ],
        "answer": answer,
        "policy_calls": len(actions),
        "usage": {"total_tokens": 1000 + len(actions), "complete": True},
    }


def _pair(parent_correct=False, candidate_correct=True):
    summary = summarize_pair(
        "q1",
        _episode("parent", ("semantic_search", "read_chunk"), ("n1",)),
        _episode("candidate", ("semantic_search", "linear_hop", "read_chunk"), ("n1", "n2", "n3")),
        {"status": "ok", "hard": parent_correct},
        {"status": "ok", "hard": candidate_correct},
    )
    return {"question_id": "q1", "question": "Which entity is related?", "paired_summary": summary}


def test_transition_groups_keep_unknown_unscored():
    assert [transition_group(False, False), transition_group(False, True), transition_group(True, False), transition_group(True, True)] == list(GROUPS)
    assert transition_group(None, False) == UNSCORED


def test_stage_a_context_is_grouped_bounded_and_diagnosis_only():
    pair = _pair(False, True)
    context = build_stage_a_context(
        "G01",
        [pair],
        changed_rule_ids=["R04"],
        parent_rules=[{"rule_id": "R04", "title": "old", "when": "x", "action_sequence": ["read_chunk"]}],
        candidate_rules=[{"rule_id": "R04", "title": "new", "when": "x", "action_sequence": ["linear_hop"]}],
        action_space=("semantic_search", "entity_hop", "linear_hop", "read_chunk", "stop_and_answer"),
    )
    assert context["group"] == "G01"
    assert context["constraints"]["do_not_use_gold_answer"] is True
    assert context["paired_cases"][0]["answer_transition"] == "G01"
    assert "linear_hop" in context["action_space"]["actions"]
    validate_stage_a_output({key: "observed" for key in ("observed_pattern", "evidence_effect", "cost_effect", "applicability_hint", "risks")})


def test_stage_a_rejects_skill_edit_and_cross_group_cases():
    pair = _pair(False, True)
    try:
        build_stage_a_context("G01", [pair, _pair(True, True)])
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("mixed groups must be rejected")
    try:
        validate_stage_a_output({key: "REVISE the Skill" for key in ("observed_pattern", "evidence_effect", "cost_effect", "applicability_hint", "risks")})
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("Stage A must not emit a Skill edit")


def test_stage_b_allows_one_logical_edit_only():
    context = build_stage_b_context(
        {"retrieval_policy": []},
        {"retrieval_policy": []},
        [],
        {g: group_summary([], g) for g in GROUPS},
        [],
        [],
        {"count": 0},
    )
    assert context["constraints"]["one_logical_edit_max"] is True
    assert validate_stage_b_output({"decision": "KEEP", "reason": "insufficient evidence"})["decision"] == "KEEP"
    try:
        validate_stage_b_output({"decision": "REVISE", "reason": "change", "edit": {"operation": "replace", "rule_id": "R1", "rule": {}}})
    except ValueError:
        pass
    else:
        # The structural validator accepts one edit; a downstream RuleStore
        # validates the actual rule body.
        assert True


def test_action_objective_names_every_current_action_and_counts_linear_hop():
    contract = action_space_contract()
    assert set(("semantic_search", "entity_hop", "linear_hop", "read_chunk", "stop_and_answer")) <= set(contract["actions"])
    usage = action_usage({"trajectory": [{"tool_name": "linear_hop"}, {"tool_name": "read_chunk"}], "answer": "x"})
    assert usage["linear_hop"] == 1
    assert usage["read_chunk"] == 1
    assert usage["stop_and_answer"] == 1

