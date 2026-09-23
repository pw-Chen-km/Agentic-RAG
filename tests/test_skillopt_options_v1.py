from pathlib import Path

import pytest

from agentic_rag.skillopt.options import (
    OptionStore,
    OptionSkillOptCoordinator,
    OptionTrajectory,
    apply_edits,
    build_optimizer_trajectory,
    option_policy_schema,
    optimizer_edit_schema,
    option_selector_schema,
    route_episode,
    validate_store,
)


ROOT = Path(__file__).parents[1]


def seed() -> OptionStore:
    return OptionStore.from_json(ROOT / "skills" / "options_v1.json")


def test_seed_has_five_options_and_fallback_and_valid_policy_boundary():
    store = seed()
    validate_store(store)
    assert store.selectable_ids() == (
        "O1_START_SEARCH", "O2_RESOLVE_FACT", "O3_RESOLVE_BRIDGE",
        "O4_RECOVER", "O5_ANSWER", "FALLBACK",
    )
    assert "FINISH" not in store.get("O3_RESOLVE_BRIDGE").primitive_actions
    assert store.get("O5_ANSWER").primitive_actions == ("FINISH",)


def test_markdown_is_derived_and_selected_option_is_progressively_disclosed():
    store = seed()
    selector = store.markdown()
    selected = store.markdown(selected_option="O3_RESOLVE_BRIDGE")
    assert "Choose one option" in selector
    assert "Observation-driven policy" in selected
    assert store.json_hash() == OptionStore.from_mapping(store.to_mapping()).json_hash()


def test_refine_policy_is_small_and_does_not_mutate_parent():
    parent = seed()
    candidate, receipt = apply_edits(parent, [{
        "operation": "refine",
        "option_id": "O3_RESOLVE_BRIDGE",
        "field": "policy.no_progress",
        "new_value": "Report BLOCKED and let the selector choose an unexplored legal path.",
        "reason": "Repeated bridge cases need an explicit hand-back.",
        "supporting_case_ids": ["q1", "q2"],
    }], stage="policy")
    assert receipt.no_change is False
    assert parent.get("O3_RESOLVE_BRIDGE").policy["no_progress"] != candidate.get("O3_RESOLVE_BRIDGE").policy["no_progress"]
    assert parent.json_hash() != candidate.json_hash()


def test_forbidden_reference_and_fixed_or_wrong_stage_edits_are_rejected():
    store = seed()
    with pytest.raises(ValueError, match="episode-specific reference"):
        apply_edits(store, [{"operation": "refine", "option_id": "O3_RESOLVE_BRIDGE", "field": "goal", "new_value": "Use C4", "reason": "case pattern", "supporting_case_ids": ["q1", "q2"]}], stage="selection")
    with pytest.raises(ValueError, match="cannot edit field"):
        apply_edits(store, [{"operation": "refine", "option_id": "O3_RESOLVE_BRIDGE", "field": "termination.blocked", "new_value": "Stop", "reason": "case pattern", "supporting_case_ids": ["q1", "q2"]}], stage="policy")
    with pytest.raises(ValueError, match="protected option"):
        apply_edits(store, [{"operation": "delete_option", "option_id": "O5_ANSWER", "reason": "case pattern", "supporting_case_ids": ["q1", "q2"]}], stage="selection")


def test_no_change_and_edit_limit():
    store = seed()
    unchanged, receipt = apply_edits(store, [{"operation": "no_change"}], stage="policy")
    assert receipt.no_change and unchanged.json_hash() == store.json_hash()
    edits = [{"operation": "refine", "option_id": "O3_RESOLVE_BRIDGE", "field": "goal", "new_value": "a"},
             {"operation": "refine", "option_id": "O4_RECOVER", "field": "goal", "new_value": "b", "reason": "r", "supporting_case_ids": ["q1", "q2"]},
             {"operation": "refine", "option_id": "O2_RESOLVE_FACT", "field": "goal", "new_value": "c", "reason": "r", "supporting_case_ids": ["q1", "q2"]}]
    edits[0]["reason"] = "r"; edits[0]["supporting_case_ids"] = ["q1", "q2"]
    with pytest.raises(ValueError, match="too many edits"):
        apply_edits(store, edits, stage="selection")
    with pytest.raises(ValueError, match="two distinct"):
        apply_edits(store, [{"operation": "refine", "option_id": "O3_RESOLVE_BRIDGE", "field": "goal", "new_value": "Follow a useful bridge.", "reason": "r", "supporting_case_ids": ["q1"]}], stage="selection")


def test_json_schemas_limit_option_and_primitive_choices():
    store = seed()
    selector = option_selector_schema(store.selectable_ids())
    assert selector["properties"]["option_id"]["enum"][-1] == "FALLBACK"
    policy = option_policy_schema(store.get("O3_RESOLVE_BRIDGE"))
    branches = policy["oneOf"]
    actions = branches[0]["properties"]["action"]["oneOf"]
    assert {item["properties"]["type"]["const"] for item in actions} == {"SEARCH", "EXPAND", "READ"}
    edit_schema = optimizer_edit_schema("meta")
    assert edit_schema["properties"]["edits"]["maxItems"] == 2
    assert edit_schema["properties"]["edits"]["items"]["properties"]["operation"]["enum"][0] == "refine"


def test_trajectory_and_renderer_do_not_repeat_evidence_text():
    trajectory = OptionTrajectory()
    trajectory.record_start(0, "O1_START_SEARCH")
    trajectory.record_action(0, "O1_START_SEARCH", {"type": "SEARCH"}, {"new": 1})
    trajectory.record_end(1, "O1_START_SEARCH", "COMPLETE")
    assert [item["event"] for item in trajectory.to_list()] == ["start", "continue", "complete"]
    view = build_optimizer_trajectory([
        {"step": 0, "new_evidence": [{"stable_id": "S1", "text": "first"}]},
        {"step": 1, "new_evidence": [{"stable_id": "S1", "text": "first"}, {"stable_id": "S2", "text": "second"}]},
    ])
    assert view[0]["new_evidence"][0]["text"] == "first"
    assert "text" not in view[1]["new_evidence"][0]
    assert view[1]["new_evidence"][1]["text"] == "second"


def test_router_separates_selection_policy_and_termination():
    cases = route_episode({"id": "q1", "steps": [
        {"event": "start", "option_id": "O1_START_SEARCH"},
        {"event": "continue", "option_id": "O1_START_SEARCH", "primitive_action": {"type": "SEARCH"}},
        {"event": "complete", "option_id": "O1_START_SEARCH", "option_status": "COMPLETE"},
    ]})
    assert len(cases.selection) == 1
    assert len(cases.policy) == 1
    assert len(cases.termination) == 1


def test_coordinator_caps_reflection_cases_and_pairs_meta_by_question():
    coordinator = OptionSkillOptCoordinator(seed(), max_cases_per_reflection=5)
    text = coordinator.build_input("policy", [{"question_id": str(i)} for i in range(8)])
    assert '"case_count": 5' in text
    pairs = coordinator.same_question_meta_cases(
        [{"question_id": "q1", "skill_sha256": "p", "split": "train"}, {"question_id": "q2", "skill_sha256": "p", "split": "train"}],
        [{"question_id": "q1", "skill_sha256": "c", "split": "train"}, {"question_id": "q3", "skill_sha256": "c", "split": "train"}],
    )
    assert len(pairs) == 1 and pairs[0]["question_id"] == "q1"
    result = coordinator.apply_response("policy", {"no_change": True}, cases_used=5)
    assert result.receipt.no_change and result.cases_used == 5
