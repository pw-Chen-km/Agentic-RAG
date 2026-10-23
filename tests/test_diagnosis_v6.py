"""Offline behavioral checks for plain-language state/action diagnoses.

The replies and trajectories in this file are synthetic, not model results.
"""

import copy
import json
import re
from pathlib import Path

import pytest
from phase1_fixtures import alternative_example, raw_example
from test_phase1 import diagnostic_call, edits_call, make_engine, questions, rules

from arag_skillopt.diagnostics import (
    STATE_ACTION_PROTOCOL,
    citation_advisories,
    diagnostic_tools,
    parse_diagnoses,
    support_ids,
)
from arag_skillopt.phase1 import Phase1Config
from arag_skillopt.checkpoint import load_checkpoint
from arag_skillopt.meta_comparison import comparison_views, make_comparison, render_comparisons_with_audit
from arag_skillopt.meta_stage import MetaConfig, MetaStage
from arag_skillopt.skill_rules import compile_initial_skill, skill_sha256
from arag_skillopt.trajectory import render_view_with_audit
from arag_skillopt.training_profile import NEW_PROFILE, PROTOCOL, prompt_bundle
from arag_skillopt.workflow_training import WorkflowTrainer


ROOT = Path(__file__).parents[1]


def state_action_call(case_ids=("q1",)):
    return {
        "function": {
            "name": "report_diagnoses",
            "arguments": {
                "diagnoses": [
                    {
                        "observations": [
                            {
                                "case_id": qid,
                                "state": "The question still needed a birthplace, and an unread biography was already available.",
                                "action": "The agent searched for the same author again instead of inspecting the biography.",
                                "result": "The search returned already seen chunks and did not provide the missing birthplace.",
                            }
                            for qid in case_ids
                        ],
                        "analysis": "The agent did not use its known information gap to choose the next action.",
                        "possible_skill_improvement": "When relevant information has already been retrieved but not read, inspect it before repeating the search.",
                        "limitations": "The record does not show whether that alternative would have produced a correct answer.",
                    }
                ]
                if case_ids
                else [],
                "reason": "Describe the observed behavior and a possible improvement for later testing.",
            },
        }
    }


class SyntheticReply:
    def __init__(self, calls):
        self.calls = calls
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return {"message": {"tool_calls": copy.deepcopy(self.calls)}}


def v6_engine(tmp_path, calls):
    engine = make_engine(tmp_path, [])
    engine.cfg = Phase1Config(
        min_support_cases=1,
        analysis_protocol=STATE_ACTION_PROTOCOL,
        max_retries=1,
    )
    engine.prompts = prompt_bundle(
        ROOT, {"optimizer_prompt_version": NEW_PROFILE}, workflow=True
    )
    engine.client = SyntheticReply(calls)
    return engine


def report(qid="q1"):
    parsed = parse_diagnoses(
        state_action_call([qid]), [qid], ["R01", "R02"],
        protocol=STATE_ACTION_PROTOCOL,
    )
    return {
        "proposal_id": "observed-" + qid,
        "analysis_kind": "failure",
        "case_ids": [qid],
        "status": "ok",
        **parsed,
    }


def test_v6_schema_does_not_require_a_rule_or_decision_assignment():
    schema = diagnostic_tools(protocol=STATE_ACTION_PROTOCOL)[0]["function"]["parameters"]
    diagnosis = schema["properties"]["diagnoses"]["items"]
    assert set(diagnosis["required"]) == {
        "observations", "analysis", "possible_skill_improvement", "limitations"
    }
    assert set(diagnosis["properties"]) == set(diagnosis["required"])
    observation = diagnosis["properties"]["observations"]["items"]
    assert set(observation["required"]) == {"case_id", "state", "action", "result"}
    assert set(observation["properties"]) == set(observation["required"])
    parsed = parse_diagnoses(
        state_action_call(), ["q1"], [], {"q1": []},
        protocol=STATE_ACTION_PROTOCOL,
    )
    assert len(parsed["diagnoses"]) == 1


@pytest.mark.parametrize("mutation", [
    "fabricated_case", "duplicate_case", "legacy_diagnosis_field",
    "legacy_observation_field", "blank_state", "blank_action", "blank_result",
    "blank_analysis", "missing_improvement", "too_many_diagnoses", "too_many_cases",
])
def test_v6_parser_rejects_invalid_grounding_or_old_output_shape(mutation):
    call = state_action_call()
    args = call["function"]["arguments"]
    diagnosis = args["diagnoses"][0]
    observation = diagnosis["observations"][0]
    allowed = ["q1", "q2", "q3", "q4", "q5", "q6"]
    if mutation == "fabricated_case":
        observation["case_id"] = "not_supplied"
    elif mutation == "duplicate_case":
        diagnosis["observations"].append(copy.deepcopy(observation))
    elif mutation == "legacy_diagnosis_field":
        diagnosis["target_rule_ids"] = ["R01"]
    elif mutation == "legacy_observation_field":
        observation["decisions"] = ["1"]
    elif mutation.startswith("blank_"):
        field = mutation.removeprefix("blank_")
        (diagnosis if field == "analysis" else observation)[field] = "  "
    elif mutation == "missing_improvement":
        del diagnosis["possible_skill_improvement"]
    elif mutation == "too_many_diagnoses":
        args["diagnoses"] *= 3
    else:
        diagnosis["observations"] = [
            {**copy.deepcopy(observation), "case_id": qid} for qid in allowed
        ]
    with pytest.raises(ValueError):
        parse_diagnoses(call, allowed, ["R01"], protocol=STATE_ACTION_PROTOCOL)


def test_v6_allows_empty_report_and_does_not_force_an_improvement_or_limitation():
    empty = parse_diagnoses(
        state_action_call([]), ["q1"], [], protocol=STATE_ACTION_PROTOCOL
    )
    assert empty["diagnoses"] == []
    call = state_action_call()
    call["function"]["arguments"]["diagnoses"][0].update(
        possible_skill_improvement="", limitations=""
    )
    parsed = parse_diagnoses(call, ["q1"], [], protocol=STATE_ACTION_PROTOCOL)
    assert parsed["diagnoses"][0]["possible_skill_improvement"] == ""


def test_v6_case_citations_are_factual_provenance_not_a_classification_vote():
    observed = report()
    assert support_ids([observed]) == {"q1"}
    assert citation_advisories([observed]) == []
    legacy = {
        "proposal_id": "legacy",
        "diagnoses": diagnostic_call(["q2", "q3"])["function"]["arguments"]["diagnoses"],
    }
    legacy["diagnoses"][0]["observations"][0]["interpretation"] = "neutral"
    legacy["diagnoses"][0]["observations"][1]["interpretation"] = "counterexample"
    assert support_ids([observed, legacy]) == {"q1"}


def test_legacy_parser_and_schema_are_still_explicitly_distinct():
    assert "target_rule_ids" in diagnostic_tools()[0]["function"]["parameters"]["properties"]["diagnoses"]["items"]["properties"]
    assert parse_diagnoses(diagnostic_call(["q1"]), ["q1"], ["R01"])["diagnoses"]
    with pytest.raises(ValueError):
        parse_diagnoses(state_action_call(), ["q1"], ["R01"])
    with pytest.raises(ValueError):
        parse_diagnoses(diagnostic_call(["q1"]), ["q1"], ["R01"], protocol=STATE_ACTION_PROTOCOL)


def test_v6_reflection_has_only_diagnosis_tool_and_no_decision_reference_requirement(tmp_path):
    engine = v6_engine(tmp_path, [state_action_call()])
    result = engine.diagnose(
        "failure", rules(), "Synthetic TRAIN trajectory", ["q1"],
        tmp_path / "reflection.json", {"q1": ["1", "2"]},
    )
    assert result["status"] == "ok" and result["edits"] == []
    request = engine.client.requests[0]
    assert [tool["function"]["name"] for tool in request["tools"]] == ["report_diagnoses"]
    assert "Allowed exact Decision references" not in request["messages"][1]["content"]
    assert "target_rule_ids" not in json.dumps(request["tools"])


def test_v6_reflection_cannot_emit_a_formal_edit(tmp_path):
    engine = v6_engine(tmp_path, [edits_call(ids=["q1"])])
    result = engine.diagnose(
        "failure", rules(), "Synthetic TRAIN trajectory", ["q1"],
        tmp_path / "reflection.json", {"q1": ["1"]},
    )
    assert result["status"] == "diagnosis_failed_after_retry"
    assert result["edits"] == []
    assert "report_diagnoses" in result["audit"][0]["validation_error"]


def test_one_v6_observation_can_add_a_rule_without_targeting_an_existing_rule(tmp_path):
    engine = v6_engine(tmp_path, [edits_call("R03", ids=["q1"], operation="add_rule")])
    original = rules()
    result = engine.merge(original, [report()], tmp_path)
    assert result["status"] == "ok" and len(result["edits"]) == 1
    candidate = engine.store(original, 3).apply(result["edits"])
    assert candidate["retrieval_policy"][:2] == original["retrieval_policy"]
    assert candidate["retrieval_policy"][-1]["rule_id"] == "R03"
    assert "supporting_case_ids" not in compile_initial_skill(candidate)
    assert "target_rule_ids" not in json.dumps(engine.client.requests[0]["messages"])


def test_v6_merge_can_create_independent_improvements_from_different_diagnoses(tmp_path):
    stop = edits_call("R01", ids=["q1"])
    stop["function"]["arguments"]["rule"] = {
        "title": "Answer when supported",
        "when": "The read evidence supports all required parts of the answer",
        "action_sequence": ["Answer from the evidence already available."],
    }
    bundle = edits_call("R03", ids=["q2"], operation="add_rule")
    bundle["function"]["arguments"]["rule"] = {
        "title": "Choose useful searches together",
        "when": "Several independent information needs remain unresolved",
        "action_sequence": ["Choose searches together only when each can be selected from the information already visible."],
    }
    reports = [report("q1"), report("q2")]
    reports[0]["diagnoses"][0]["analysis"] = "The agent continued after evidence was sufficient."
    reports[1]["diagnoses"][0]["analysis"] = "Independent searches were unnecessarily spread over several calls."
    engine = v6_engine(tmp_path, [stop, bundle])
    result = engine.merge(rules(), reports, tmp_path)
    assert result["status"] == "ok" and len(result["edits"]) == 2
    assert engine.store(rules(), 3).apply(result["edits"])["retrieval_policy"][-1]["rule_id"] == "R03"


@pytest.mark.parametrize("case_ids", [["invented"], []])
def test_v6_merge_still_requires_a_supplied_observation(tmp_path, case_ids):
    engine = v6_engine(tmp_path, [edits_call(ids=case_ids)])
    result = engine.merge(rules(), [report()], tmp_path)
    assert result["status"] == "no_change_after_retry" and result["edits"] == []


def test_v6_merge_cannot_create_a_rule_without_any_diagnosis(tmp_path):
    engine = v6_engine(tmp_path, [edits_call(ids=["q1"])])
    empty = {**report(), "diagnoses": []}
    result = engine.merge(rules(), [empty], tmp_path)
    assert result["status"] == "no_documented_support"
    assert result["edits"] == [] and not engine.client.requests


def test_v6_cannot_reintroduce_disabled_keyword_search(tmp_path):
    call = edits_call(ids=["q1"])
    call["function"]["arguments"]["rule"]["action_sequence"] = ["Use keyword_search for the entity."]
    engine = v6_engine(tmp_path, [call])
    result = engine.merge(rules(), [report()], tmp_path)
    assert result["status"] == "no_change_after_retry" and result["edits"] == []
    assert "keyword_search" in result["audit"][0]["validation_error"]


def test_v6_contract_uses_selected_schema_and_does_not_reuse_v4_prompt_hash(tmp_path):
    legacy = make_engine(tmp_path / "legacy", [])
    cfg = Phase1Config(min_support_cases=1, analysis_protocol=STATE_ACTION_PROTOCOL)
    revised = type(legacy)(
        tmp_path / "revised", rules(), legacy.client, legacy.rollout,
        legacy.evaluator, prompt_bundle(ROOT, {"optimizer_prompt_version": NEW_PROFILE}, workflow=True),
        legacy.tool_guidance, cfg,
    )
    assert revised.contract["analysis_protocol"] == STATE_ACTION_PROTOCOL
    assert revised.contract["diagnostic_schema_hash"] != legacy.contract["diagnostic_schema_hash"]
    assert revised.contract["prompts_hash"] != legacy.contract["prompts_hash"]
    assert PROTOCOL != STATE_ACTION_PROTOCOL


def historical_comparison():
    earlier = rules()
    earlier["retrieval_policy"][1]["title"] = "Earlier guidance left unchanged by the proposal"
    proposed = copy.deepcopy(earlier)
    proposed["retrieval_policy"][0]["title"] = "Earlier proposed retrieval change"
    current = rules()
    current["retrieval_policy"][1]["title"] = "Current guidance that may now be improved"
    branch_a = {"raw": raw_example(), "evaluation": {"correct": True, "status": "ok"}}
    branch_b = {"raw": alternative_example(), "evaluation": {"correct": True, "status": "ok"}}
    question = {"id": "q1", "question": branch_a["raw"]["question"]}
    comparison = make_comparison(question, earlier, proposed, branch_a, branch_b, 1)
    comparison["views"] = comparison_views(branch_a, branch_b, None)
    return earlier, proposed, current, comparison


def test_v6_meta_reflection_sees_full_historical_guidance_and_distinct_current_skill(tmp_path):
    earlier, proposed, current, comparison = historical_comparison()
    engine = v6_engine(tmp_path / "phase1", [state_action_call()])
    stage = MetaStage(engine, tmp_path / "meta", current, MetaConfig())
    reports = stage.reflect_comparisons(
        current, [comparison], tmp_path / "meta" / "reflections", "g",
        historical_skill_a=earlier, historical_skill_b=proposed,
    )
    assert reports[0]["status"] == "ok" and reports[0]["edits"] == []
    request = engine.client.requests[0]
    assert [tool["function"]["name"] for tool in request["tools"]] == ["report_diagnoses"]
    user = request["messages"][1]["content"]
    # Historical R02 was not in the edit diff. Without complete A it was missing
    # whenever the Current Skill differed, making the observed runs ambiguous.
    assert earlier["retrieval_policy"][1]["title"] in user
    assert proposed["retrieval_policy"][0]["title"] in user
    assert current["retrieval_policy"][1]["title"] in user
    editable = json.loads(user.split("Current Skill:\n", 1)[1].split("\nFixed tool guidance:", 1)[0])
    assert editable == {key: current[key] for key in ("schema_version", "retrieval_policy")}
    assert "exact_state_match" not in user
    assert "Allowed exact Decision references" not in user
    assert "Branch A" in user or "Run A" in user or "run A" in user
    assert "Branch B" in user or "Run B" in user or "run B" in user


def test_v6_meta_renderer_cannot_guess_missing_historical_rule_context():
    earlier, proposed, _, comparison = historical_comparison()
    with pytest.raises(ValueError):
        render_comparisons_with_audit([comparison], plain_language=True)
    text, _ = render_comparisons_with_audit(
        [comparison], plain_language=True,
        historical_skill_a=earlier, historical_skill_b=proposed,
    )
    assert earlier["retrieval_policy"][1]["title"] in text
    wrong_proposed = copy.deepcopy(proposed)
    wrong_proposed["retrieval_policy"][1]["title"] = "Unrecorded historical change"
    with pytest.raises(ValueError):
        render_comparisons_with_audit(
            [comparison], plain_language=True,
            historical_skill_a=earlier, historical_skill_b=wrong_proposed,
        )


def test_v6_meta_merge_modifies_current_skill_and_cannot_cite_a_case_only_in_limitations(tmp_path):
    earlier, proposed, current, _ = historical_comparison()
    engine = v6_engine(tmp_path / "phase1", [edits_call("R03", ids=["q1"], operation="add_rule")])
    stage = MetaStage(engine, tmp_path / "meta", current, MetaConfig())
    observed = report()
    observed["analysis_kind"] = "meta"
    observed["diagnoses"][0]["limitations"] = "Question q2 showed the opposite effect and does not support this finding."
    observed["case_ids"] = ["q1", "q2"]
    result = stage.merge(current, [observed], tmp_path / "meta", stage="meta_merge")
    assert result["status"] == "ok"
    revised = stage.store(current, 3).apply(result["edits"])
    assert revised["retrieval_policy"][:2] == current["retrieval_policy"]
    assert revised["retrieval_policy"][:2] != earlier["retrieval_policy"]
    assert revised["retrieval_policy"][:2] != proposed["retrieval_policy"]
    allowed = engine.client.requests[0]["tools"][0]["function"]["parameters"]["properties"]["supporting_case_ids"]["items"]["enum"]
    assert allowed == ["q1"]
    rejected = v6_engine(tmp_path / "reject", [edits_call(ids=["q2"])])
    rejected_result = rejected.merge(current, [observed], tmp_path / "reject", stage="meta_merge")
    assert rejected_result["status"] == "no_change_after_retry" and rejected_result["edits"] == []


def test_v6_diagnostic_retry_requests_state_action_explanation_without_old_decision_keys(tmp_path):
    engine = v6_engine(tmp_path, [])
    engine.cfg = Phase1Config(
        min_support_cases=1, analysis_protocol=STATE_ACTION_PROTOCOL, max_retries=2
    )

    class CorrectedReply:
        def __init__(self):
            self.requests = []

        def chat(self, **kwargs):
            self.requests.append(copy.deepcopy(kwargs))
            # A stale cached model habit returns the previous schema first.
            call = diagnostic_call(["q1"]) if len(self.requests) == 1 else state_action_call()
            return {"message": {"tool_calls": [call]}}

    engine.client = CorrectedReply()
    result = engine.diagnose(
        "failure", rules(), "Synthetic TRAIN trajectory", ["q1"],
        tmp_path / "reflection.json", {"q1": ["1", "2"]},
    )
    assert result["status"] == "ok" and len(result["audit"]) == 2
    correction = engine.client.requests[1]["messages"][-1]["content"]
    assert all(word in correction for word in ("state", "action", "result", "possible_skill_improvement"))
    assert "Copy only these exact Decision references" not in correction
    assert "combine that case's decisions" not in correction


def test_v6_merge_rejects_legacy_diagnoses_before_sending_an_optimizer_request(tmp_path):
    engine = v6_engine(tmp_path, [edits_call(ids=["q1"])])
    legacy = {
        "proposal_id": "old-report", "analysis_kind": "failure", "case_ids": ["q1"],
        "status": "ok", **diagnostic_call(["q1"])["function"]["arguments"],
    }
    with pytest.raises(ValueError, match="state-action diagnosis fields"):
        engine.merge(rules(), [legacy], tmp_path)
    assert engine.client.requests == []


def test_phase1_v6_prints_decision_labels_and_preserves_returned_text(tmp_path):
    engine = v6_engine(tmp_path, [state_action_call()])
    question = {"id": "q1", "question": raw_example()["question"], "answer": "York"}

    def repeated_search(question, skill):
        raw = raw_example(question["id"])
        # Two identical searches were actually selected together. Keep the
        # response, flat tool record, and later visible messages consistent.
        raw["trajectory"][1] = copy.deepcopy(raw["trajectory"][0])
        function = raw["decision_audit"][0]["response"]["message"]["tool_calls"][0]["function"]
        for decision in raw["decision_audit"]:
            for message in decision["request"]["messages"] + [decision["response"]["message"]]:
                for call in message.get("tool_calls", []):
                    if call["id"] == "call1-1":
                        call["function"] = copy.deepcopy(function)
                if message.get("tool_call_id") == "call1-1":
                    message["content"] = raw["trajectory"][0]["tool_result"]
        return raw

    engine.rollout = repeated_search
    episode = engine.episode(question, rules(), "train", "Initial")
    legacy_text, legacy_audit = render_view_with_audit(episode["view"])
    plain_text, plain_audit = render_view_with_audit(episode["view"], plain_language=True)
    assert "-> D2" in legacy_text and "-> Decision 2" in plain_text
    assert "identical to D1.R1" in legacy_text
    assert "identical to Decision 1, result 1" in plain_text
    assert not re.search(r"\bD\d+\b", plain_text)
    for text in ("The River was written by Alice.", "Alice was born in York."):
        assert text in legacy_text and text in plain_text
    # Display wording changes, but the original-body references are unchanged.
    assert plain_audit["text_references"] == legacy_audit["text_references"]
    saved = list((tmp_path / "episodes" / "train").glob("*/*/optimizer_view.txt"))
    assert len(saved) == 1 and saved[0].read_text() == plain_text
    engine.reflect_group("failure", rules(), [episode], tmp_path / "reflections", "g")
    user = engine.client.requests[0]["messages"][1]["content"]
    assert "-> Decision 2" in user and not re.search(r"\bD\d+\b", user)


def test_v6_complete_workflow_replays_candidate_train_then_uses_actual_pairs_for_meta(tmp_path):
    events = []
    initial = rules()

    class WorkflowReply:
        def __init__(self):
            self.requests = []
            self.stages = []

        def chat(self, **request):
            self.requests.append(copy.deepcopy(request))
            user = request["messages"][1]["content"]
            if request["tools"][0]["function"]["name"] == "report_diagnoses":
                ids = [line.removeprefix("Case ID: ") for line in user.splitlines() if line.startswith("Case ID: ")]
                stage = "meta_reflect" if "Complete rules used by Run A" in user else "phase1_reflect"
                call = state_action_call(ids)
            else:
                reports = json.loads(user.split("Batch diagnoses (all groups):\n", 1)[1])
                ids = sorted({o["case_id"] for report in reports for d in report["diagnoses"] for o in d["observations"]})
                meta = any(report["analysis_kind"] == "meta" for report in reports)
                stage = "meta_merge" if meta else "phase1_merge"
                if meta:
                    call = edits_call("R02", ids=ids[:1])
                    call["function"]["arguments"]["rule"] = {
                        "title": "Use observed progress to choose the next action",
                        "when": "Read passages leave part of the question unresolved",
                        "action_sequence": ["Identify the missing information before choosing another search."],
                    }
                else:
                    call = edits_call("R03", ids=ids[:1], operation="add_rule")
                    call["function"]["arguments"]["rule"] = {
                        "title": "Read promising material before repeating a search",
                        "when": "An already retrieved passage may answer the unresolved part",
                        "action_sequence": ["Read the promising passage before searching again for the same information."],
                    }
            self.stages.append(stage)
            events.append((stage, tuple(ids)))
            return {"message": {"tool_calls": [call]}}

    def rollout(question, skill):
        assert set(question) == {"id", "question"}
        revised = len(skill["retrieval_policy"]) == 3
        meta = skill["retrieval_policy"][1]["title"] == "Use observed progress to choose the next action"
        raw = alternative_example(question["id"]) if revised else raw_example(question["id"])
        raw["question"] = question["question"]
        raw["answer"] = "good" if revised or int(question["id"][1:]) % 2 == 0 else "bad"
        if meta:
            raw["usage"].update(input_tokens=270, total_tokens=350)
            raw["decision_audit"][-1]["response"]["input_tokens"] = 110
        events.append(("rollout", question["id"], skill_sha256(skill)))
        return raw

    client = WorkflowReply()
    config = Phase1Config(
        min_support_cases=1, analysis_protocol=STATE_ACTION_PROTOCOL,
        max_token_cost_increase=0.10, max_call_cost_increase=0.05,
        min_validation_pair_coverage=0.95,
    )
    trainer = WorkflowTrainer(
        tmp_path, initial, client, rollout,
        lambda question, raw: {"correct": raw["answer"] == "good", "status": "ok"},
        prompt_bundle(ROOT, {"optimizer_prompt_version": NEW_PROFILE}, workflow=True),
        "Synthetic tool descriptions", config,
    )
    summary = trainer.run(questions("t", 4), questions("v", 4))
    assert summary["status"] == "complete" and summary["meta_executed"]
    assert summary["phase1"]["accepted_updates"] == 1
    assert summary["meta"]["accepted_updates"] == 1
    assert summary["meta"]["comparison_count"] == 4 and not summary["test_executed"]
    phase_receipt = load_checkpoint(tmp_path / "phase1/batches/001/completed.json")["value"]
    meta_receipt = load_checkpoint(tmp_path / "meta/batches/001/completed.json")["value"]
    assert len(phase_receipt["meta_replay"]["pairs"]) == 4
    assert phase_receipt["meta_replay"]["purpose"] == "meta_analysis_only"
    assert all(pair[arm]["split"] == "train" for pair in phase_receipt["meta_replay"]["pairs"] for arm in ("parent", "candidate"))
    assert phase_receipt["gate"]["paired_count"] == meta_receipt["gate"]["paired_count"] == 4
    first_meta = next(i for i, event in enumerate(events) if event[0] == "meta_reflect")
    assert len([event for event in events[:first_meta] if event[0] == "rollout" and event[1].startswith("t")]) == 8
    assert len([event for event in events if event[0] == "rollout" and event[1].startswith("t")]) == 8
    # The accepted Phase 1 rule is the editable Meta input and survives the
    # independent Meta change; historical A was the original two-rule Skill.
    assert meta_receipt["proposal_parent_skill_hash"] == phase_receipt["phase1_candidate_skill_hash"]
    final = load_checkpoint(tmp_path / "final_skill.json")
    assert final["retrieval_policy"][-1] == phase_receipt["active_skill"]["retrieval_policy"][-1]
    assert final["retrieval_policy"][1]["title"] == "Use observed progress to choose the next action"
    reflected = []
    for stage, request in zip(client.stages, client.requests):
        if stage.endswith("reflect"):
            observed_schema = request["tools"][0]["function"]["parameters"]["properties"]["diagnoses"]["items"]["properties"]["observations"]["items"]
            assert set(observed_schema["properties"]) == {"case_id", "state", "action", "result"}
            assert "\nCase ID: v" not in request["messages"][1]["content"]
        if stage == "meta_reflect":
            reflected.append(request)
    assert len(reflected) == 1
    meta_user = reflected[0]["messages"][1]["content"]
    assert "Complete rules used by Run A" in meta_user and "Run B rule order" in meta_user
    assert initial["retrieval_policy"][1]["title"] in meta_user
    assert phase_receipt["active_skill"]["retrieval_policy"][-1]["title"] in meta_user
    before = len(events)
    assert trainer.run(questions("t", 4), questions("v", 4)) == summary
    assert len(events) == before
