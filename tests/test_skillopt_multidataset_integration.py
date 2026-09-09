from __future__ import annotations

import json

from agentic_rag.agent.models import ChunkMemoryItem, EntityMemoryItem, ObservationStatus, PolicyStateView, PolicyView, SentenceMemoryItem
from agentic_rag.skillopt.adapter import _native_reflection_conversation
from agentic_rag.skillopt.trajectory import build_reflection_input, build_training_reference_text
from test_skillopt_trajectory import _episode, _item


def reference_episode():
    episode = _episode()
    for step in episode.trajectory:
        state = step.state_before
        memory = []
        if "entity:bridge" in state.visible_entity_ids:
            memory.append(EntityMemoryItem(ref="E1", canonical_name="Bridge Person"))
        if "chunk:1" in state.visible_chunk_ids:
            read = "chunk:1" in state.read_chunk_ids
            memory.append(ChunkMemoryItem(
                ref="C1", chunk_position=0, has_been_read=read,
                text="The bridge reached the gold answer." if read else None,
                previews=[] if read else ["The bridge reached the gold answer."],
            ))
            memory.append(SentenceMemoryItem(ref="S1", parent_chunk_ref="C1", text="The bridge reached the gold answer."))
        step.policy_view = PolicyView(policy_state=PolicyStateView(
            step=state.step, policy_attempts=state.policy_attempts,
            semantic_memory=memory, budget="recorded budget",
        ))
    return episode


def reference_item():
    item = _item()
    item.pop("supporting_facts")
    item["source"] = "medical"
    item["reference_evidence"] = {
        "kind": "reference_text", "method": "lexical_f1", "status": "ready",
        "facts": [{
            "fact_id": "reference:0000", "text": "gold answer", "source_indices": [0],
            "mappings": [{"unit_text": "UNSEEN_CORPUS_MAPPING_SENTINEL"}],
        }],
    }
    return item


def render(arm, phase="train"):
    return build_reflection_input(
        episode=reference_episode(), item=reference_item(), skill_content="fixed skill",
        rollout_phase=phase, rollout_split="train" if phase == "train" else "test",
        trajectory_representation=arm,
    )[0]


def test_four_arms_share_reference_but_do_not_leak_alignment_corpus():
    outputs = [render(arm) for arm in ("raw", "organized", "organized_support_labels", "progress_abstracted")]
    assert all(row["hidden_reference"] == outputs[0]["hidden_reference"] for row in outputs)
    for row in outputs:
        assert "UNSEEN_CORPUS_MAPPING_SENTINEL" not in json.dumps(row)
    shared = build_training_reference_text(reference_item())
    assert "UNSEEN_CORPUS_MAPPING_SENTINEL" not in shared
    raw, organized, labels, abstract = outputs
    assert "reference_progress" not in json.dumps(raw["trajectory"])
    assert "reference_progress" not in json.dumps(organized["trajectory"])
    for left, right in zip(labels["trajectory"]["step_diagnostics"], abstract["trajectory"]["abstract_steps"], strict=True):
        assert left["reference_progress"] == right["reference_progress"]
        assert left["later_expand_uses_of_acquired_information"] == right["later_expand_uses_of_acquired_information"]
    combined_native = json.dumps(_native_reflection_conversation(abstract)) + shared
    assert "The bridge reached the gold answer." not in combined_native
    assert "gold answer" in combined_native


def test_later_read_is_linked_without_crediting_its_progress_to_expand():
    abstract = render("progress_abstracted")
    steps = abstract["trajectory"]["abstract_steps"]
    expand = steps[1]["reference_progress"]
    read = steps[2]["reference_progress"]
    assert expand["eligible"]["delta"] == 0
    assert read["eligible"]["delta"] > 0
    assert read["visible"]["delta"] == 0
    uses = steps[0]["later_expand_uses_of_acquired_information"]
    assert uses[0]["reference_progress"]["eligible_delta"] == 0
    assert uses[0]["later_reads_of_direct_children"][0]["read_step"] == 3
    assert uses[0]["later_reads_of_direct_children"][0]["reference_progress"]["eligible_delta"] == read["eligible"]["delta"]


def test_eval_never_receives_new_hidden_reference_or_gt_derived_progress():
    for arm in ("raw", "organized", "organized_support_labels", "progress_abstracted"):
        output = render(arm, phase="eval")
        assert output["hidden_reference"] is None
        assert "reference:0000" not in json.dumps(output)
        assert "UNSEEN_CORPUS_MAPPING_SENTINEL" not in json.dumps(output)
        assert "reference_progress" not in json.dumps(output["trajectory"])


def test_reference_relations_do_not_enter_shared_reference():
    item = reference_item()
    item["evidence_relations"] = ["RELATION_ARCHIVE_ONLY_SENTINEL"]
    assert "RELATION_ARCHIVE_ONLY_SENTINEL" not in build_training_reference_text(item)


def test_abstract_error_text_cannot_echo_retrieved_body():
    episode = reference_episode()
    canary = "RETRIEVED_BODY_ECHOED_IN_EXCEPTION_ONLY"
    episode.error_message = canary
    episode.trajectory[0].observation.status = ObservationStatus.ERROR
    episode.trajectory[0].observation.message = canary
    for arm in ("raw", "organized", "organized_support_labels", "progress_abstracted"):
        output, _ = build_reflection_input(
            episode=episode, item=reference_item(), skill_content="fixed skill",
            rollout_phase="train", rollout_split="train", trajectory_representation=arm,
        )
        assert (canary in json.dumps(output)) is (arm != "progress_abstracted")


def test_legacy_policy_view_does_not_make_ineligible_sentence_citable():
    from agentic_rag.skillopt.trajectory import _decision_context

    step = reference_episode().trajectory[2]
    step.context_reference_map = None
    step.state_before.eligible_sentence_ids.clear()
    refs = _decision_context(step)["visible_references"]
    sentence = next(item for item in refs if item["ref"] == "S1")
    assert sentence["can_use_as_evidence"] is False
