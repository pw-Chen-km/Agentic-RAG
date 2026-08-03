from __future__ import annotations

import json
import re
from pathlib import Path

from agentic_rag.agent.answer import FakeAnswerGenerator
from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    AssessmentStatus,
    EvidenceAssessment,
    EpisodeResult,
    FinishAction,
    PolicyDecision,
    ReadAction,
    SearchAction,
    SentenceRef,
    TerminationReason,
)
from agentic_rag.agent.policy import ScriptedPolicy
from agentic_rag.paths import portable_path_component
from agentic_rag.storage import Substrate


def _decision(action, *, sufficient: bool = False) -> PolicyDecision:
    evidence_refs = (
        list(action.evidence_refs)
        if isinstance(action, FinishAction)
        else []
    )
    return PolicyDecision(
        assessment=EvidenceAssessment(
            status=(
                AssessmentStatus.SUFFICIENT
                if sufficient
                else AssessmentStatus.INSUFFICIENT
            ),
            supported_facts=(
                ["The evidence answers the question."]
                if sufficient
                else []
            ),
            missing_information=(
                [] if sufficient else ["Find the birthplace."]
            ),
            selected_evidence_refs=evidence_refs,
        ),
        action=action,
    )


def _decision_with_selected(
    action,
    selected_refs,
) -> PolicyDecision:
    return PolicyDecision(
        assessment=EvidenceAssessment(
            status=AssessmentStatus.INSUFFICIENT,
            supported_facts=["Marie Curie was born in Warsaw."],
            missing_information=["Confirm the surrounding context."],
            selected_evidence_refs=list(selected_refs),
        ),
        action=action,
    )


def test_skill_content_harness_persists_candidate_and_logical_io_trace(
    built_substrate: Path,
    fake_embedder,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence_id = next(
        sentence.sentence_id
        for sentence in substrate.sentences
        if sentence.text == "Marie Curie was born in Warsaw."
    )
    question = "Where was Marie Curie born?"
    sentence = substrate.sentence_by_id[sentence_id]
    stable_chunk_id = sentence.chunk_id
    stable_doc_id = substrate.chunk_by_id[stable_chunk_id].doc_id
    stable_node_ids = (
        set(substrate.doc_ids_by_scope["q1"])
        | set(substrate.chunk_ids_by_scope["q1"])
        | set(substrate.sentence_ids_by_scope["q1"])
        | set(substrate.entity_ids_by_scope["q1"])
    )
    candidate_skill = (
        "# Candidate skill\n\n"
        "Search with the complete question and finish from direct evidence.\n"
    )
    selected_handle: str | None = None
    selected_chunk_handle: str | None = None

    def select_and_read_from_handles(messages) -> PolicyDecision:
        nonlocal selected_handle, selected_chunk_handle
        serialized_messages = json.dumps(
            [message.model_dump(mode="json") for message in messages],
            ensure_ascii=False,
        )
        assert all(
            stable_id not in serialized_messages
            for stable_id in stable_node_ids
        )
        payload = json.loads(messages[-1].content)

        def find_sentence(value):
            if isinstance(value, list):
                return next(
                    (
                        found
                        for item in value
                        if (found := find_sentence(item)) is not None
                    ),
                    None,
                )
            if not isinstance(value, dict):
                return None
            if (
                value.get("text")
                == "Marie Curie was born in Warsaw."
                and isinstance(value.get("sentence_id"), str)
            ):
                return value
            return next(
                (
                    found
                    for item in value.values()
                    if (found := find_sentence(item)) is not None
                ),
                None,
            )

        result = find_sentence(payload)
        assert result is not None
        selected_handle = result["sentence_id"]
        selected_chunk_handle = result["parent_chunk_id"]
        assert re.fullmatch(r"S[1-9][0-9]*", selected_handle)
        assert re.fullmatch(r"C[1-9][0-9]*", selected_chunk_handle)
        return _decision_with_selected(
            ReadAction(chunk_id=selected_chunk_handle),
            [SentenceRef(id=selected_handle)],
        )

    def finish_from_handle(messages) -> PolicyDecision:
        assert selected_handle is not None
        serialized_messages = json.dumps(
            [message.model_dump(mode="json") for message in messages],
            ensure_ascii=False,
        )
        assert all(
            stable_id not in serialized_messages
            for stable_id in stable_node_ids
        )
        return _decision(
            FinishAction(
                evidence_refs=[SentenceRef(id=selected_handle)]
            ),
            sufficient=True,
        )

    policy = ScriptedPolicy(
        [
            _decision(
                SearchAction(
                    query=question,
                    method="BM25",
                    target="SENTENCE",
                )
            ),
            select_and_read_from_handles,
            finish_from_handle,
        ]
    )
    answer_generator = FakeAnswerGenerator("Warsaw")
    output_root = tmp_path / "predictions"
    harness = AgentHarness.from_skill_content(
        substrate_path=built_substrate,
        config=AgentConfig(),
        skill_content=candidate_skill,
        skill_source_path="skillopt:candidate-v1",
        output_root=output_root,
        policy=policy,
        answer_generator=answer_generator,
        embedding_backend=fake_embedder,
    )

    episode_id = "hotpotqa:dev:q1"
    result = harness.run(question, "q1", episode_id=episode_id)

    assert result.termination_reason is TerminationReason.FINISH
    assert result.answer == "Warsaw"
    assert selected_handle is not None
    assert selected_chunk_handle is not None
    artifact_dir = output_root / portable_path_component(episode_id)
    assert Path(result.artifact_dir or "") == artifact_dir
    assert (artifact_dir / "skill.md").read_text(encoding="utf-8") == (
        candidate_skill
    )
    persisted_episode = json.loads(
        (artifact_dir / "episode.json").read_text(encoding="utf-8")
    )
    assert persisted_episode["episode_id"] == episode_id
    assert persisted_episode["artifact_dir"] == artifact_dir.as_posix()

    trace = harness.build_io_trace(result)
    assert trace["trace_format"] == "agentic-rag-logical-io-v2"
    assert trace["target_input"] == {
        "question": question,
        "scope_id": "q1",
        "skill_sha256": harness.skill.sha256,
    }
    assert len(trace["policy_calls"]) == 3
    assert trace["policy_calls"][0]["output"]["action"]["type"] == (
        "SEARCH"
    )
    assert trace["policy_calls"][0]["observation"]["status"] == "ok"
    read_call = trace["policy_calls"][1]
    assert read_call["raw_handle_action"] == {
        "type": "READ",
        "chunk_id": selected_chunk_handle,
    }
    assert read_call["resolved_stable_action"] == {
        "type": "READ",
        "chunk_id": stable_chunk_id,
    }
    assert read_call["raw_policy_decision"]["assessment"][
        "selected_evidence_refs"
    ] == [{"unit": "SENTENCE", "id": selected_handle}]
    assert read_call["resolved_decision"]["assessment"][
        "selected_evidence_refs"
    ] == [{"unit": "SENTENCE", "id": sentence_id}]
    assert trace["policy_calls"][2]["output"]["action"]["type"] == (
        "FINISH"
    )
    finish_call = trace["policy_calls"][2]
    assert finish_call["raw_handle_action"]["evidence_refs"] == [
        {"unit": "SENTENCE", "id": selected_handle}
    ]
    assert finish_call["resolved_stable_action"]["evidence_refs"] == [
        {"unit": "SENTENCE", "id": sentence_id}
    ]
    assert finish_call["raw_policy_decision"]["assessment"][
        "selected_evidence_refs"
    ] == [{"unit": "SENTENCE", "id": selected_handle}]
    assert finish_call["resolved_decision"]["assessment"][
        "selected_evidence_refs"
    ] == [{"unit": "SENTENCE", "id": sentence_id}]
    assert trace["answer_generation"]["output"] == {
        "answer": "Warsaw"
    }
    answer_messages = trace["answer_generation"]["input"]
    answer_payload = json.loads(answer_messages[1]["content"])
    assert answer_payload["question"] == question
    assert answer_payload["evidence"][0]["text"] == (
        "Marie Curie was born in Warsaw."
    )

    policy_inputs = json.dumps(
        [call["input"] for call in trace["policy_calls"]],
        ensure_ascii=False,
    )
    assert all(stable_id not in policy_inputs for stable_id in stable_node_ids)
    assert stable_chunk_id not in policy_inputs
    assert stable_doc_id not in policy_inputs
    assert sentence_id in json.dumps(
        trace["policy_calls"][0]["raw_observation"],
        ensure_ascii=False,
    )
    visible_feedback = json.dumps(
        [
            call["agent_visible_observation"]
            for call in trace["policy_calls"]
        ],
        ensure_ascii=False,
    )
    assert all(
        stable_id not in visible_feedback for stable_id in stable_node_ids
    )

    episode = json.loads((artifact_dir / "episode.json").read_text("utf-8"))
    final_registry = episode["final_state"]["handle_registry"]
    assert final_registry["handle_to_stable_id"][selected_handle] == sentence_id
    assert final_registry["stable_id_to_handle"][sentence_id] == selected_handle
    assert episode["trajectory"][2]["decision"]["action"][
        "evidence_refs"
    ] == [{"unit": "SENTENCE", "id": selected_handle}]
    assert episode["trajectory"][2]["resolved_decision"]["action"][
        "evidence_refs"
    ] == [{"unit": "SENTENCE", "id": sentence_id}]

    conversation = json.loads(
        (artifact_dir / "conversation.json").read_text("utf-8")
    )
    conversation_payload = json.dumps(conversation, ensure_ascii=False)
    assert all(
        stable_id not in conversation_payload for stable_id in stable_node_ids
    )
    assert conversation[0]["env_feedback"] == trace["policy_calls"][0][
        "agent_visible_observation"
    ]
    assert conversation[1]["action"] == {
        "type": "READ",
        "chunk_id": selected_chunk_handle,
    }
    assert conversation[2]["action"]["evidence_refs"] == [
        {"unit": "SENTENCE", "id": selected_handle}
    ]

    # A persisted Episode is sufficient to reconstruct the exact same logical
    # provider trace, including both sides of every handle resolution.
    reloaded = EpisodeResult.model_validate_json(
        (artifact_dir / "episode.json").read_text("utf-8")
    )
    assert harness.build_io_trace(reloaded) == trace

    serialized = json.dumps(trace, ensure_ascii=False)
    assert "OPENAI_API_KEY" not in serialized
    assert "Authorization" not in serialized
    assert "NEVER_SHOW_GOLD_ANSWER" not in serialized
