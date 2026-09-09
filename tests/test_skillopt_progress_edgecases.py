"""Independent integration review using synthetic episodes only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rag.agent.models import ObservationStatus, TerminationReason, ValidationStatus
from agentic_rag.evaluation import EvaluationResult
from agentic_rag.skillopt.adapter import _native_reflection_conversation
from agentic_rag.skillopt.offline_examples import _cases
from agentic_rag.skillopt.rollout import RolloutBatch, run_rollout_batch
from agentic_rag.skillopt.trajectory import (
    TRAJECTORY_REPRESENTATIONS, build_reflection_input, build_training_reference_text,
)


def _case(identifier="expand_then_read"):
    case = next(item for item in _cases() if item["case_id"] == identifier)
    episode = case["episode"]
    item = {"id": episode.episode_id, "question": episode.query, "answer": episode.answer,
            "scope_id": episode.scope_id, "source": "synthetic", "reference_evidence": case["reference"]}
    return episode, item


def _render(episode, item, arm, *, phase="train"):
    return build_reflection_input(
        episode=episode, item=item, skill_content="same synthetic skill", target_system_prompt="same protocol",
        rollout_phase=phase, rollout_split="train" if phase == "train" else "test",
        trajectory_representation=arm,
    )[0]


def _rows(output):
    trajectory = output["trajectory"]
    return trajectory.get("step_diagnostics", trajectory.get("abstract_steps", []))


def test_raw_organized_no_derived_labels_but_identical_training_reference():
    episode, item = _case()
    outputs = [_render(episode, item, arm) for arm in TRAJECTORY_REPRESENTATIONS]
    assert all(output["hidden_reference"] == outputs[0]["hidden_reference"] for output in outputs)
    for output in outputs[:2]:
        assert "reference_progress" not in json.dumps(output["trajectory"])
        assert "supporting_fact_progress" not in json.dumps(output["trajectory"])
    assert [step["reference_progress"] for step in _rows(outputs[2])] == [
        step["reference_progress"] for step in _rows(outputs[3])]


@pytest.mark.parametrize("arm", TRAJECTORY_REPRESENTATIONS)
def test_full_alignment_text_never_enters_any_optimizer_arm(arm):
    episode, item = _case()
    item["reference_evidence"]["facts"][0]["mappings"] = [{"unit_text": "UNSEEN_MAPPING_FULLTEXT_CANARY"}]
    output = _render(episode, item, arm)
    complete_input = json.dumps(_native_reflection_conversation(output)) + build_training_reference_text(item)
    assert "UNSEEN_MAPPING_FULLTEXT_CANARY" not in complete_input
    assert "biopsy confirms diagnosis" in complete_input


@pytest.mark.parametrize("arm", TRAJECTORY_REPRESENTATIONS)
def test_eval_projection_never_receives_reference_labels_or_secret_gt(arm):
    episode, item = _case()
    item["answer"] = "HIDDEN_ANSWER_CANARY"
    item["reference_evidence"]["facts"][0]["text"] = "HIDDEN_FACT_CANARY"
    output = _render(episode, item, arm, phase="eval")
    serialized = json.dumps(output)
    assert output["hidden_reference"] is None
    assert "HIDDEN_ANSWER_CANARY" not in serialized
    assert "HIDDEN_FACT_CANARY" not in serialized
    assert "reference_progress" not in json.dumps(output["trajectory"])


def test_terminal_read_has_no_presented_progress_or_retrospective_read_credit():
    episode, item = _case()
    episode.trajectory = episode.trajectory[:-1]
    episode.final_state = episode.trajectory[-1].state_after.model_copy(deep=True)
    episode.answer = None
    episode.evidence_refs = []
    episode.resolved_evidence = []
    episode.termination_reason = TerminationReason.BUDGET_EXHAUSTED
    labels = _render(episode, item, "organized_support_labels")
    abstract = _render(episode, item, "progress_abstracted")
    assert [row["reference_progress"] for row in _rows(labels)] == [row["reference_progress"] for row in _rows(abstract)]
    terminal = _rows(abstract)[-1]["reference_progress"]
    assert terminal["next_policy_saw_result"] is False
    assert terminal["reason"] == "terminal_result_not_presented"
    assert terminal["visible"]["delta"] == terminal["eligible"]["delta"] == 0
    use = _rows(abstract)[0]["later_expand_uses_of_acquired_information"][0]
    assert use["later_reads_of_direct_children"][0]["reference_progress"]["visible_delta"] == 0


@pytest.mark.parametrize("status,validation", [
    (ObservationStatus.INVALID_ACTION, ValidationStatus.INVALID),
    (ObservationStatus.DUPLICATE_ACTION, ValidationStatus.INVALID),
    (ObservationStatus.ERROR, ValidationStatus.VALID),
])
def test_invalid_or_failed_read_is_na_and_never_a_successful_later_read(status, validation):
    episode, item = _case()
    read = episode.trajectory[2]
    read.observation.status = status
    read.validation_status = validation
    read.observation.error_code = "synthetic_execution_error"
    labels = _render(episode, item, "organized_support_labels")
    abstract = _render(episode, item, "progress_abstracted")
    for output in (labels, abstract):
        row = _rows(output)[2]
        assert row["reference_progress"]["status"] == "unavailable"
        assert row["reference_progress"]["visible"] is None
        assert row["information_progress"] is None
    use = _rows(abstract)[0]["later_expand_uses_of_acquired_information"][0]
    assert use["later_reads_of_direct_children"] == []


def test_no_frozen_map_fallback_does_not_turn_sentence_preview_into_eligible_evidence():
    episode, item = _case("lexical_repeat")
    step = episode.trajectory[1]
    step.context_reference_map = None
    step.state_before.eligible_sentence_ids.clear()
    output = _render(episode, item, "progress_abstracted")
    context = _rows(output)[1]["decision_context"]
    sentence = next(ref for ref in context["visible_references"] if ref["node_type"] == "SENTENCE")
    assert sentence["can_use_as_evidence"] is False


def test_fourth_error_text_does_not_reintroduce_retrieval_passages():
    episode, item = _case()
    read = episode.trajectory[2]
    read.observation.status = ObservationStatus.ERROR
    read.observation.error_code = "synthetic_tool_error"
    read.observation.message = "ERROR_RETRIEVAL_TEXT_CANARY"
    episode.error_message = "EPISODE_RETRIEVAL_TEXT_CANARY"
    fourth = _render(episode, item, "progress_abstracted")
    complete = json.dumps(_native_reflection_conversation(fourth)) + build_training_reference_text(item)
    assert "ERROR_RETRIEVAL_TEXT_CANARY" not in complete
    assert "EPISODE_RETRIEVAL_TEXT_CANARY" not in complete
    raw = _render(episode, item, "raw")
    assert "ERROR_RETRIEVAL_TEXT_CANARY" in json.dumps(raw)
    assert "EPISODE_RETRIEVAL_TEXT_CANARY" in json.dumps(raw)


@pytest.mark.parametrize("phase", ["train", "eval"])
def test_rollout_boundary_passes_only_question_scope_and_id_to_target(tmp_path: Path, phase):
    episode, item = _case()
    item["answer"] = "EVALUATOR_ONLY_ANSWER_CANARY"
    item["reference_evidence"]["facts"][0]["text"] = "REFLECT_ONLY_FACT_CANARY"
    item["reference_evidence"]["facts"][0]["mappings"] = [{"unit_text": "MAPPER_ONLY_TEXT_CANARY"}]
    calls = []
    evaluated = []

    class Harness:
        def __init__(self, root):
            self.root = root

        def run(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            result = episode.model_copy(deep=True)
            task = self.root / kwargs["episode_id"]
            task.mkdir(parents=True)
            result.artifact_dir = str(task)
            return result

        def build_io_trace(self, result):
            return {"synthetic": True}

    def harness_factory(**kwargs):
        assert set(kwargs) == {"skill_content", "output_root"}
        assert kwargs["skill_content"] == "same synthetic skill"
        return Harness(kwargs["output_root"])

    class Evaluator:
        def evaluate(self, **kwargs):
            evaluated.append(kwargs)
            return EvaluationResult(
                question=kwargs["question"], predicted_answer=kwargs["predicted_answer"],
                gold_answer=kwargs["gold_answer"], normalized_prediction="synthetic prediction",
                normalized_gold="synthetic gold", status="answered", llm_acc=0, contain_acc=0, hard=0, soft=0,
            )

    output = run_rollout_batch(
        batch=RolloutBatch(items=(item,), phase=phase, split="train" if phase == "train" else "test"),
        out_root=tmp_path, skill_content="same synthetic skill", harness_factory=harness_factory,
        evaluator=Evaluator(), resume=False, trajectory_representation="progress_abstracted",
    )
    assert calls == [{"args": (item["question"], item["scope_id"]), "kwargs": {"episode_id": item["id"]}}]
    assert evaluated[0]["gold_answer"] == item["answer"]
    assert "CANARY" not in json.dumps(calls)
    if phase == "train":
        assert "REFLECT_ONLY_FACT_CANARY" in output[0]["reference_text"]
        assert "MAPPER_ONLY_TEXT_CANARY" not in output[0]["reference_text"]
    else:
        assert "reference_text" not in output[0]
