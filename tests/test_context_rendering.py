from arag_skillopt.context_rendering import (
    CONTEXT_RENDERER_VERSION,
    render_phase1_merge_context,
    render_phase1_reflection_context,
    render_stage_a_context,
    render_stage_b_context,
)


def _action_space():
    return {
        "actions": ["semantic_search", "entity_hop", "linear_hop", "read_chunk", "stop_and_answer"],
        "terminal_action": "stop_and_answer",
    }


def _stage_a():
    return {
        "meta_task": {"objective": "diagnose the supplied group"},
        "group": "G01",
        "action_space": _action_space(),
        "changed_rule_ids": ["R04"],
        "parent_rules": [{"rule_id": "R04", "title": "Read first", "when": "a result may answer", "action_sequence": ["read_chunk"]}],
        "candidate_rules": [{"rule_id": "R04", "title": "Hop first", "when": "an entity is visible", "action_sequence": ["linear_hop"]}],
        "group_summary": {
            "count": 2,
            "action_changed_count": 1,
            "action_changed_rate": 0.5,
            "mean_first_divergence_turn": 3.0,
            "mean_new_evidence_count": 2.0,
            "mean_new_evidence_delta": 1.0,
            "mean_policy_call_delta": 1.0,
            "mean_retrieved_token_delta": 820,
            "informative_count": 2,
        },
        "paired_cases": [{
            "question_id": "q1",
            "question": "Which entity is related?",
            "parent": {"action_bundles": [["SEARCH"], ["FINISH"]], "answer_correct": False},
            "candidate": {"action_bundles": [["SEARCH"], ["HOP"], ["FINISH"]], "answer_correct": True},
            "answer_transition": "G01",
            "action_changed": True,
            "first_divergence_turn": 2,
            "evidence_diff": {"added_ids": ["sentence-42"], "removed_ids": [], "new_evidence_count": 1},
            "cost_diff": {"policy_call_delta": 1, "tool_call_delta": 1, "retrieved_token_delta": 820},
        }],
    }


def test_stage_a_renderer_explains_group_and_metrics():
    text = render_stage_a_context(_stage_a())
    assert CONTEXT_RENDERER_VERSION in text
    assert "G01: parent incorrect -> candidate correct" in text
    assert "15/18" not in text
    assert "50%" in text
    assert "Mean retrieved-token delta" in text
    assert "sentence-42" in text
    assert "Do not choose KEEP, REVISE, or REJECT" in text


def test_stage_b_renderer_keeps_one_edit_contract_readable():
    context = {
        "parent_skill": {"retrieval_policy": _stage_a()["parent_rules"]},
        "candidate_skill": {"retrieval_policy": _stage_a()["candidate_rules"]},
        "changed_rule_ids": ["R04"],
        "group_summaries": {"G01": _stage_a()["group_summary"]},
        "group_reflections": [{"proposal_id": "g01-000", "status": "ok"}],
        "representative_pairs": [{"question_id": "q1"}],
        "train_statistics": {"paired_count": 2},
        "action_space": _action_space(),
    }
    text = render_stage_b_context(context)
    assert "[GROUP FINDINGS]" in text
    assert "G01 (parent incorrect -> candidate correct)" in text
    assert "At most 3 logical edits" in text
    assert "q1" in text


def test_stage_b_renderer_explains_reflections_pairs_and_statistics_without_json_dump():
    context = {
        "parent_skill": {"retrieval_policy": _stage_a()["parent_rules"]},
        "candidate_skill": {"retrieval_policy": _stage_a()["candidate_rules"]},
        "changed_rule_ids": ["R04"],
        "group_summaries": {"G01": _stage_a()["group_summary"]},
        "group_reflections": [{
            "proposal_id": "g01-000",
            "group": "G01",
            "status": "ok",
            "case_ids": ["q1"],
            "diagnoses": [{
                "observations": [{"case_id": "q1", "behavior": "candidate read the hop result"}],
                "analysis": "The changed rule encouraged full reads after a useful hop.",
                "possible_skill_improvement": "Use the hop when it adds a bridge.",
                "limitations": "The group is small.",
            }],
        }],
        "representative_pairs": [{
            "question_id": "q1",
            "question": "Which entity is related?",
            "transition_group": "G01",
            "paired_summary": {
                "parent_answer_correct": False,
                "candidate_answer_correct": True,
                "action_changed": True,
                "first_divergence": {"step": 2, "parent_action": ["READ_CHUNK"], "candidate_action": ["LINEAR_HOP"]},
                "parent_action_bundles": [["SEARCH"], ["READ_CHUNK"]],
                "candidate_action_bundles": [["SEARCH"], ["LINEAR_HOP"], ["READ_CHUNK"]],
                "parent_full_read_ids": [],
                "candidate_full_read_ids": ["c1"],
                "full_read_delta": 1,
                "snippet_to_full_delta": 1,
                "policy_call_delta": 1,
                "tool_call_delta": 1,
                "retrieved_token_delta": 820,
                "model_token_delta": 1200,
            },
        }],
        "train_statistics": {
            "paired_count": 2,
            "informative_pair_count": 1,
            "unscored_count": 0,
            "outcomes": {"candidate_only": 1},
            "skipped": {"not_informative_pair": 1},
        },
        "action_space": _action_space(),
    }
    text = render_stage_b_context(context)
    assert "The changed rule encouraged full reads" in text
    assert "First action difference: turn 2" in text
    assert "Training comparison coverage:" in text
    assert "candidate only=1" in text
    assert '"proposal_id"' not in text
    assert '"paired_summary"' not in text


def test_phase1_reflection_uses_same_renderer_and_keeps_case_marker():
    text = render_phase1_reflection_context(
        {"retrieval_policy": _stage_a()["parent_rules"]},
        [{
            "question_id": "q1",
            "view": {
                "question_id": "q1",
                "question": "Which entity is related?",
                "skill_version": "phase1 parent",
                "decisions": [],
                "episode_evaluation": {},
            },
        }],
        "failure",
        action_space=_action_space(),
    )
    assert CONTEXT_RENDERER_VERSION in text
    assert "[TASK / PHASE 1 REFLECTION]" in text
    assert text.count("Case ID: q1") == 1
    assert "Do not edit the Skill" in text


def test_phase1_merge_uses_the_unified_renderer_version():
    proposals = [{
        "proposal_id": "failure-000",
        "analysis_kind": "failure",
        "case_ids": ["q1", "q2"],
        "status": "ok",
        "diagnoses": [{"pattern": "read before repeating search"}],
        "reason": "observed in both cases",
    }]
    merge = render_phase1_merge_context(
        {"retrieval_policy": _stage_a()["parent_rules"]},
        proposals,
        ["q1", "q2"],
        action_space=_action_space(),
    )
    assert CONTEXT_RENDERER_VERSION in merge
    assert "[DIAGNOSIS REPORTS]" in merge
    assert "Batch diagnoses (all groups):" in merge

