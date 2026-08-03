from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_rag.agent.answer import (
    FakeAnswerGenerator,
    OpenAIResponsesAnswerGenerator,
    ScriptedAnswerGenerator,
)
from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    AssessmentStatus,
    ChunkRef,
    ControllerState,
    EntityHandle,
    EvidenceAssessment,
    ExpandAction,
    ExpansionKind,
    FinishAction,
    ObservationStatus,
    PolicyDecision,
    ReadAction,
    SearchAction,
    SentenceHandle,
    SentenceRef,
    TerminationReason,
    ValidationStatus,
)
from agentic_rag.agent.ollama import (
    OllamaChatAnswerGenerator,
    OllamaChatPolicy,
)
from agentic_rag.errors import (
    AgenticRAGError,
    EvidenceEligibilityError,
    NodeNotFoundError,
)
from agentic_rag.agent.policy import (
    OpenAIResponsesPolicy,
    PolicyResponseError,
    ScriptedPolicy,
    policy_decision_model,
)
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.storage import Substrate


QUESTION_BIRTHPLACE = "Where was Marie Curie born?"
QUESTION_COUNTRY = "In which country was Marie Curie born?"
SKILL_TEXT = (
    "# Initial retrieval strategy\n\n"
    "Search with a complete natural-language question, inspect the evidence, "
    "and finish only when the evidence is sufficient.\n"
)


def _decision(
    action: SearchAction | ExpandAction | ReadAction | FinishAction,
    *,
    status: AssessmentStatus = AssessmentStatus.INSUFFICIENT,
    selected_refs: list[SentenceRef | ChunkRef] | None = None,
) -> PolicyDecision:
    if selected_refs is None and isinstance(action, FinishAction):
        selected_refs = list(action.evidence_refs)
    return PolicyDecision(
        assessment=EvidenceAssessment(
            status=status,
            supported_facts=(
                ["The selected evidence directly answers the question."]
                if status is AssessmentStatus.SUFFICIENT
                else []
            ),
            missing_information=(
                ["The answer is not yet established."]
                if status is not AssessmentStatus.SUFFICIENT
                else []
            ),
            selected_evidence_refs=selected_refs or [],
        ),
        action=action,
    )


def _sentence_id(substrate: Substrate, text: str) -> str:
    return next(
        sentence.sentence_id
        for sentence in substrate.sentences
        if sentence.text == text
    )


def _entity_id(substrate: Substrate, canonical_name: str) -> str:
    return next(
        entity.entity_id
        for entity in substrate.entities
        if entity.canonical_name == canonical_name
    )


def _harness(
    substrate_path: Path,
    fake_embedder: Any,
    output_root: Path,
    decisions: list[PolicyDecision | dict[str, Any] | Exception],
    *,
    answer: str = "Warsaw",
    max_steps: int = 10,
    max_retrieved_tokens: int = 12_000,
    enabled_expansions: tuple[ExpansionKind, ...] = (
        DEFAULT_ENABLED_EXPANSIONS
    ),
) -> tuple[AgentHarness, ScriptedPolicy, FakeAnswerGenerator]:
    policy = ScriptedPolicy(decisions)
    answer_generator = FakeAnswerGenerator(answer)
    harness = AgentHarness(
        substrate=Substrate.open(substrate_path),
        config=AgentConfig(
            max_steps=max_steps,
            max_retrieved_tokens=max_retrieved_tokens,
            enabled_expansions=enabled_expansions,
        ),
        skill=SkillDocument.from_text(SKILL_TEXT),
        policy=policy,
        answer_generator=answer_generator,
        output_root=output_root,
        embedding_backend=fake_embedder,
    )
    return harness, policy, answer_generator


def test_bm25_sentence_finish_preserves_parent_provenance_and_enables_read(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    born_sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    parent_chunk = substrate.chunk_for_sentence(born_sentence_id)

    search = _decision(
        SearchAction(
            query=QUESTION_BIRTHPLACE,
            method="BM25",
            target="SENTENCE",
        )
    )
    finish = _decision(
        FinishAction(
            evidence_refs=[SentenceRef(id=born_sentence_id)]
        ),
        status=AssessmentStatus.SUFFICIENT,
    )
    harness, policy, answer_generator = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "runs",
        [search, finish],
    )

    result = harness.controller.run_episode(
        QUESTION_BIRTHPLACE, "q1", episode_id="sentence-finish"
    )

    assert result.termination_reason is TerminationReason.FINISH
    assert result.answer == "Warsaw"
    assert len(policy.calls) == 2
    assert len(answer_generator.calls) == 1
    assert answer_generator.calls[0][0] == QUESTION_BIRTHPLACE
    assert answer_generator.calls[0][1][0].text == (
        "Marie Curie was born in Warsaw."
    )

    search_step = result.trajectory[0]
    assert search_step.validation_status is ValidationStatus.VALID
    sentence_result = next(
        item
        for item in search_step.observation.results
        if item.get("sentence_id") == born_sentence_id
    )
    assert sentence_result["text"] == "Marie Curie was born in Warsaw."
    assert sentence_result["parent_chunk_id"] == parent_chunk.chunk_id
    assert sentence_result["document_id"] == parent_chunk.doc_id
    assert sentence_result["title"] == "Marie Curie"

    after_search = search_step.state_after
    assert born_sentence_id in after_search.visible_sentence_ids
    assert born_sentence_id in after_search.eligible_sentence_ids
    assert parent_chunk.chunk_id in after_search.visible_chunk_ids
    assert parent_chunk.chunk_id not in after_search.read_chunk_ids

    read = _decision(ReadAction(chunk_id=parent_chunk.chunk_id))
    validation = harness.controller.validator.validate(
        read, after_search, "q1"
    )
    assert validation.ok
    read_observation = harness.controller.router.execute(
        read.action,
        after_search,
        question=QUESTION_BIRTHPLACE,
        scope_id="q1",
        action_id="probe-read",
    )
    assert read_observation.status is ObservationStatus.OK
    assert read_observation.results[0]["chunk_id"] == parent_chunk.chunk_id


def test_chunk_preview_is_navigation_only_until_read(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    born_sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    parent_chunk_id = substrate.chunk_for_sentence(
        born_sentence_id
    ).chunk_id

    search = _decision(
        SearchAction(
            query=QUESTION_BIRTHPLACE,
            method="BM25",
            target="CHUNK",
        )
    )
    read = _decision(ReadAction(chunk_id=parent_chunk_id))
    finish = _decision(
        FinishAction(evidence_refs=[ChunkRef(id=parent_chunk_id)]),
        status=AssessmentStatus.SUFFICIENT,
    )
    harness, _, _ = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "runs",
        [search, read, finish],
    )

    result = harness.controller.run_episode(
        QUESTION_BIRTHPLACE, "q1", episode_id="chunk-read-finish"
    )

    assert result.termination_reason is TerminationReason.FINISH
    search_step, read_step, finish_step = result.trajectory
    chunk_result = next(
        item
        for item in search_step.observation.results
        if item.get("chunk_id") == parent_chunk_id
    )
    assert 1 <= len(chunk_result["previews"]) <= 2
    preview_id = chunk_result["previews"][0]["sentence_id"]

    after_search = search_step.state_after
    assert parent_chunk_id in after_search.visible_chunk_ids
    assert parent_chunk_id not in after_search.read_chunk_ids
    assert preview_id in after_search.visible_sentence_ids
    assert preview_id not in after_search.eligible_sentence_ids

    unread_chunk_finish = _decision(
        FinishAction(evidence_refs=[ChunkRef(id=parent_chunk_id)]),
        status=AssessmentStatus.SUFFICIENT,
    )
    unread_validation = harness.controller.validator.validate(
        unread_chunk_finish, after_search, "q1"
    )
    assert not unread_validation.ok
    assert unread_validation.code == "selected_evidence_not_eligible"

    preview_finish = _decision(
        FinishAction(evidence_refs=[SentenceRef(id=preview_id)]),
        status=AssessmentStatus.SUFFICIENT,
    )
    preview_validation = harness.controller.validator.validate(
        preview_finish, after_search, "q1"
    )
    assert not preview_validation.ok
    assert preview_validation.code == "selected_evidence_not_eligible"

    preview_expand = _decision(
        ExpandAction(
            kind=ExpansionKind.SENTENCE_MENTIONS_ENTITY,
            source_id=preview_id,
            query=QUESTION_BIRTHPLACE,
        )
    )
    preview_expand_validation = harness.controller.validator.validate(
        preview_expand, after_search, "q1"
    )
    assert not preview_expand_validation.ok
    assert preview_expand_validation.code == "source_not_complete"

    after_read = read_step.state_after
    assert parent_chunk_id in after_read.read_chunk_ids
    assert {
        sentence.sentence_id
        for sentence in substrate.sentences_by_chunk[parent_chunk_id]
    } <= after_read.eligible_sentence_ids
    read_expand_validation = harness.controller.validator.validate(
        preview_expand, after_read, "q1"
    )
    assert read_expand_validation.ok
    read_expand_observation = harness.controller.router.execute(
        preview_expand.action,
        after_read,
        question=QUESTION_BIRTHPLACE,
        scope_id="q1",
        action_id="probe-read-sentence-expand",
    )
    assert read_expand_observation.status is ObservationStatus.OK
    assert finish_step.validation_status is ValidationStatus.VALID
    assert result.selected_evidence_refs == [ChunkRef(id=parent_chunk_id)]


def test_finish_requires_same_turn_selected_evidence(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    second_sentence_id = _sentence_id(
        substrate, "Warsaw is the capital of Poland."
    )
    harness, _, _ = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "runs",
        [
            _decision(
                SearchAction(
                    query=QUESTION_BIRTHPLACE,
                    method="BM25",
                    target="SENTENCE",
                )
            )
        ],
        max_steps=1,
    )
    result = harness.controller.run_episode(
        QUESTION_BIRTHPLACE, "q1", episode_id="selection-prerequisite"
    )
    state = result.final_state
    assert state is not None

    unselected_finish = PolicyDecision(
        assessment=EvidenceAssessment(
            status=AssessmentStatus.SUFFICIENT,
            supported_facts=["Marie Curie was born in Warsaw."],
            selected_evidence_refs=[],
        ),
        action=FinishAction(
            evidence_refs=[SentenceRef(id=sentence_id)]
        ),
    )
    rejected = harness.controller.validator.validate(
        unselected_finish, state, "q1"
    )
    assert not rejected.ok
    assert rejected.code == "finish_evidence_not_selected"

    selected_finish = unselected_finish.model_copy(
        update={
            "assessment": unselected_finish.assessment.model_copy(
                update={
                    "selected_evidence_refs": [
                        SentenceRef(id=sentence_id),
                        SentenceRef(id=second_sentence_id),
                    ]
                }
            )
        },
        deep=True,
    )
    assert harness.controller.validator.validate(
        selected_finish, state, "q1"
    ).ok


def test_selection_replaces_atomically_and_invalid_action_preserves_memory(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    invisible_chunk = next(
        iter(substrate.chunk_ids_by_scope["q2"])
    )
    decisions = [
        _decision(
            SearchAction(
                query=QUESTION_BIRTHPLACE,
                method="BM25",
                target="SENTENCE",
            )
        ),
        _decision(
            SearchAction(
                query=QUESTION_COUNTRY,
                method="DENSE",
                target="SENTENCE",
            ),
            selected_refs=[SentenceRef(id=sentence_id)],
        ),
        _decision(
            SearchAction(
                query=QUESTION_COUNTRY,
                method="DENSE",
                target="SENTENCE",
            ),
            selected_refs=[],
        ),
        PolicyResponseError("malformed structured decision"),
        _decision(ReadAction(chunk_id=invisible_chunk)),
        _decision(
            SearchAction(
                query="What evidence is still missing?",
                method="BM25",
                target="CHUNK",
            ),
            selected_refs=[],
        ),
    ]
    harness, _, _ = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "runs",
        decisions,
        max_steps=6,
    )
    result = harness.controller.run_episode(
        QUESTION_COUNTRY, "q1", episode_id="atomic-selection"
    )

    selected_step = result.trajectory[1]
    duplicate_step = result.trajectory[2]
    schema_invalid_step = result.trajectory[3]
    invalid_step = result.trajectory[4]
    cleared_step = result.trajectory[5]
    expected = [SentenceRef(id=sentence_id)]
    assert selected_step.state_after.selected_evidence_refs == expected
    assert duplicate_step.observation.status is (
        ObservationStatus.DUPLICATE_ACTION
    )
    assert duplicate_step.state_after.selected_evidence_refs == expected
    assert schema_invalid_step.observation.error_code == (
        "invalid_policy_response"
    )
    assert schema_invalid_step.state_after.selected_evidence_refs == expected
    assert invalid_step.validation_status is ValidationStatus.INVALID
    assert invalid_step.state_after.selected_evidence_refs == expected
    assert (
        invalid_step.state_after.last_assessment
        == selected_step.state_after.last_assessment
    )
    assert cleared_step.state_after.selected_evidence_refs == []


def test_selected_evidence_rejects_cross_scope_nodes(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    q2_sentence_id = next(iter(substrate.sentence_ids_by_scope["q2"]))
    forged = ControllerState.initial()
    forged.visible_sentence_ids.add(q2_sentence_id)
    forged.eligible_sentence_ids.add(q2_sentence_id)
    decision = _decision(
        SearchAction(
            query=QUESTION_BIRTHPLACE,
            method="BM25",
            target="SENTENCE",
        ),
        selected_refs=[SentenceRef(id=q2_sentence_id)],
    )
    harness, _, _ = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "runs",
        [],
    )
    validation = harness.controller.validator.validate(
        decision, forged, "q1"
    )
    assert not validation.ok
    assert validation.code == "selected_evidence_out_of_scope"


def test_default_policy_schema_and_validator_expose_only_four_expansions(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    chunk_id = substrate.chunk_for_sentence(
        _sentence_id(substrate, "Marie Curie was born in Warsaw.")
    ).chunk_id
    harness, _, _ = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "runs",
        [],
    )

    assert len(ExpansionKind) == 8
    assert set(harness.config.enabled_expansions) == set(
        DEFAULT_ENABLED_EXPANSIONS
    )
    assert len(harness.config.enabled_expansions) == 4

    schema = policy_decision_model(
        DEFAULT_ENABLED_EXPANSIONS
    ).model_json_schema()
    serialized_schema = json.dumps(schema)
    assert all(
        kind.value in serialized_schema for kind in DEFAULT_ENABLED_EXPANSIONS
    )
    assert all(
        kind.value not in serialized_schema
        for kind in set(ExpansionKind) - set(DEFAULT_ENABLED_EXPANSIONS)
    )
    assert "SENTENCE_PART_OF_CHUNK" not in serialized_schema

    state = ControllerState.initial()
    state.visible_chunk_ids.add(chunk_id)
    disabled_internal_action = ExpandAction(
        kind=ExpansionKind.CHUNK_CONTAINS_SENTENCE,
        source_id=chunk_id,
    )
    disabled = harness.controller.validator.validate(
        _decision(disabled_internal_action), state, "q1"
    )
    assert not disabled.ok
    assert disabled.code == "disabled_expansion"

    internal_observation = (
        harness.controller.router.expansion_engine.expand(
            disabled_internal_action, QUESTION_COUNTRY, "q1"
        )
    )
    assert internal_observation.kind == "CHUNK_CONTAINS_SENTENCE"
    assert all(
        result.navigation_only and not result.evidence_eligible
        for result in internal_observation.results
    )

    with pytest.raises(ValidationError):
        ExpandAction.model_validate(
            {
                "type": "EXPAND",
                "kind": "SENTENCE_PART_OF_CHUNK",
                "source_id": _sentence_id(
                    substrate, "Marie Curie was born in Warsaw."
                ),
                "top_k": 5,
            }
        )
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(
            {
                "enabled_expansions": ["SENTENCE_PART_OF_CHUNK"],
            }
        )


def test_entity_sentence_entity_bridge_reaches_poland_with_provenance(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    warsaw_id = _entity_id(substrate, "Warsaw")
    poland_id = _entity_id(substrate, "Poland")
    bridge_sentence_id = _sentence_id(
        substrate, "Warsaw is the capital of Poland."
    )

    search_warsaw = _decision(
        SearchAction(
            query="Warsaw",
            method="LEXICAL",
            target="ENTITY",
        )
    )
    bridge = _decision(
        ExpandAction(
            kind=ExpansionKind.ENTITY_CO_OCCURS_ENTITY_SENTENCE,
            source_id=warsaw_id,
            query=QUESTION_COUNTRY,
        )
    )
    finish = _decision(
        FinishAction(
            evidence_refs=[SentenceRef(id=bridge_sentence_id)]
        ),
        status=AssessmentStatus.SUFFICIENT,
    )
    harness, _, answer_generator = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "runs",
        [search_warsaw, bridge, finish],
        answer="Poland",
    )

    result = harness.controller.run_episode(
        QUESTION_COUNTRY, "q1", episode_id="bridge-to-poland"
    )

    assert result.termination_reason is TerminationReason.FINISH
    assert result.answer == "Poland"
    bridge_step = result.trajectory[1]
    poland_result = next(
        item
        for item in bridge_step.observation.results
        if item.get("entity_id") == poland_id
    )
    assert poland_result["canonical_name"] == "Poland"
    assert poland_result["bridge_sentence_ids"] == [bridge_sentence_id]
    bridge_sentence = poland_result["bridge_sentences"][0]
    assert bridge_sentence["sentence_id"] == bridge_sentence_id
    assert bridge_sentence["text"] == "Warsaw is the capital of Poland."
    assert bridge_sentence["parent_chunk_id"] == (
        substrate.chunk_for_sentence(bridge_sentence_id).chunk_id
    )
    assert bridge_sentence["document_id"].startswith("hotpotqa:dev:q1:")
    assert bridge_sentence["title"] == "Marie Curie"

    after_bridge = bridge_step.state_after
    assert poland_id in after_bridge.visible_entity_ids
    assert bridge_sentence_id in after_bridge.visible_sentence_ids
    assert bridge_sentence_id in after_bridge.eligible_sentence_ids
    assert isinstance(after_bridge.node_handles[poland_id], EntityHandle)
    assert isinstance(
        after_bridge.node_handles[bridge_sentence_id], SentenceHandle
    )
    assert after_bridge.node_handles[bridge_sentence_id].eligible is True
    assert answer_generator.calls[0][1][0].text == (
        "Warsaw is the capital of Poland."
    )


def test_duplicate_and_invalid_steps_consume_budget_without_answer(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    search = _decision(
        SearchAction(
            query=QUESTION_BIRTHPLACE,
            method="BM25",
            target="SENTENCE",
        )
    )
    duplicate_harness, duplicate_policy, duplicate_answer = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "duplicate-runs",
        [search, search],
        max_steps=2,
    )
    duplicate_result = duplicate_harness.controller.run_episode(
        QUESTION_BIRTHPLACE, "q1", episode_id="duplicate"
    )

    assert duplicate_result.termination_reason is (
        TerminationReason.BUDGET_EXHAUSTED
    )
    assert duplicate_result.answer is None
    assert duplicate_result.error_code == "budget_exhausted"
    assert len(duplicate_policy.calls) == 2
    assert duplicate_answer.calls == []
    assert len(duplicate_result.trajectory) == 2
    assert duplicate_result.trajectory[1].observation.status is (
        ObservationStatus.DUPLICATE_ACTION
    )
    assert duplicate_result.final_state.step == 2
    assert duplicate_result.final_state.remaining_step_budget == 0

    substrate = Substrate.open(built_substrate)
    invisible_chunk_id = substrate.chunk_for_sentence(
        _sentence_id(substrate, "Marie Curie was born in Warsaw.")
    ).chunk_id
    invalid_harness, _, invalid_answer = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "invalid-runs",
        [_decision(ReadAction(chunk_id=invisible_chunk_id))],
        max_steps=1,
    )
    invalid_result = invalid_harness.controller.run_episode(
        QUESTION_BIRTHPLACE, "q1", episode_id="invalid"
    )
    invalid_step = invalid_result.trajectory[0]
    assert invalid_result.termination_reason is (
        TerminationReason.BUDGET_EXHAUSTED
    )
    assert invalid_result.answer is None
    assert invalid_answer.calls == []
    assert invalid_step.validation_status is ValidationStatus.INVALID
    assert invalid_step.observation.status is ObservationStatus.INVALID_ACTION
    assert invalid_step.observation.error_code == "source_not_visible"
    assert invalid_step.state_after.step == 1
    assert invalid_step.state_after.remaining_step_budget == 0

    schema_harness, _, schema_answer = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "schema-runs",
        [PolicyResponseError("malformed structured decision")],
        max_steps=1,
    )
    schema_result = schema_harness.controller.run_episode(
        QUESTION_BIRTHPLACE, "q1", episode_id="schema-invalid"
    )
    assert schema_result.termination_reason is (
        TerminationReason.BUDGET_EXHAUSTED
    )
    assert schema_result.answer is None
    assert schema_answer.calls == []
    assert schema_result.trajectory[0].observation.error_code == (
        "invalid_policy_response"
    )
    assert schema_result.final_state.step == 1


def test_state_dependent_invalid_action_can_be_retried_after_prerequisite(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    born_sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    parent_chunk_id = substrate.chunk_for_sentence(
        born_sentence_id
    ).chunk_id
    read = _decision(ReadAction(chunk_id=parent_chunk_id))
    search = _decision(
        SearchAction(
            query=QUESTION_BIRTHPLACE,
            method="BM25",
            target="SENTENCE",
        )
    )
    harness, _, answer_generator = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "retry-runs",
        [read, search, read],
        max_steps=3,
    )

    result = harness.controller.run_episode(
        QUESTION_BIRTHPLACE,
        "q1",
        episode_id="retry-after-prerequisite",
    )

    assert result.termination_reason is TerminationReason.BUDGET_EXHAUSTED
    assert answer_generator.calls == []
    invalid_read, search_step, successful_read = result.trajectory
    assert invalid_read.observation.error_code == "source_not_visible"
    assert invalid_read.validation_status is ValidationStatus.INVALID
    assert search_step.validation_status is ValidationStatus.VALID
    assert successful_read.validation_status is ValidationStatus.VALID
    assert successful_read.observation.status is ObservationStatus.OK
    assert parent_chunk_id in successful_read.state_after.read_chunk_ids


def test_harness_writes_six_skillopt_compatible_artifacts(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    born_sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    decisions = [
        _decision(
            SearchAction(
                query=QUESTION_BIRTHPLACE,
                method="BM25",
                target="SENTENCE",
            )
        ),
        _decision(
            FinishAction(
                evidence_refs=[SentenceRef(id=born_sentence_id)]
            ),
            status=AssessmentStatus.SUFFICIENT,
        ),
    ]
    policy = ScriptedPolicy(decisions)
    answer_generator = FakeAnswerGenerator("Warsaw")
    skill_path = tmp_path / "initial.md"
    skill_path.write_text(SKILL_TEXT, encoding="utf-8")
    output_root = tmp_path / "runs"
    harness = AgentHarness.from_config(
        built_substrate,
        AgentConfig(),
        skill_path,
        output_root,
        policy=policy,
        answer_generator=answer_generator,
        embedding_backend=fake_embedder,
    )

    result = harness.run(
        QUESTION_BIRTHPLACE,
        "q1",
        episode_id="marie-curie-artifacts",
    )

    run_dir = output_root / "marie-curie-artifacts"
    assert result.artifact_dir == run_dir.as_posix()
    assert {path.name for path in run_dir.iterdir()} == {
        "episode.json",
        "conversation.json",
        "target_system_prompt.txt",
        "target_user_prompt.txt",
        "skill.md",
        "effective_config.json",
    }

    episode = json.loads((run_dir / "episode.json").read_text("utf-8"))
    assert episode["query"] == QUESTION_BIRTHPLACE
    assert episode["scope_id"] == "q1"
    assert episode["termination_reason"] == "finish"
    assert episode["answer"] == "Warsaw"
    assert len(episode["trajectory"]) == 2
    assert episode["trajectory"][0]["policy_view"]["context_mode"] == (
        "compact_evidence"
    )
    assert (
        episode["trajectory"][0]["observation"]["results"][0]["text"]
        == "Marie Curie was born in Warsaw."
    )

    conversation = json.loads(
        (run_dir / "conversation.json").read_text("utf-8")
    )
    assert len(conversation) == 2
    assert all(
        set(step) == {"step", "action", "reasoning", "env_feedback"}
        for step in conversation
    )
    assert conversation[0]["action"] == decisions[0].action.model_dump(
        mode="json"
    )
    assert conversation[0]["reasoning"] == (
        decisions[0].assessment.model_dump(mode="json")
    )
    assert conversation[0]["env_feedback"]["status"] == "ok"
    assert conversation[1]["action"]["type"] == "FINISH"
    assert conversation[1]["reasoning"]["selected_evidence_refs"] == [
        {"unit": "SENTENCE", "id": born_sentence_id}
    ]

    assert (run_dir / "skill.md").read_text("utf-8") == SKILL_TEXT
    assert QUESTION_BIRTHPLACE in (
        run_dir / "target_user_prompt.txt"
    ).read_text("utf-8")
    system_prompt = (
        run_dir / "target_system_prompt.txt"
    ).read_text("utf-8")
    assert "Return exactly one PolicyDecision" in system_prompt
    assert all(
        kind.value in system_prompt
        for kind in DEFAULT_ENABLED_EXPANSIONS
    )
    assert "# Initial retrieval strategy" not in system_prompt
    assert SKILL_TEXT not in system_prompt

    effective = json.loads(
        (run_dir / "effective_config.json").read_text("utf-8")
    )
    assert effective["agent"]["enabled_expansions"] == [
        kind.value for kind in DEFAULT_ENABLED_EXPANSIONS
    ]
    assert effective["agent"]["context_mode"] == "compact_evidence"
    assert effective["policy_context"] == {
        "handle_summary_max_chars": 160,
        "node_reference_scheme": "episode_local_typed_handles_v1",
        "selection_semantics": "full_set_replacement",
        "stable_node_ids_visible_to_policy": False,
    }
    assert effective["runtime_components"] == {
        "policy_client": "ScriptedPolicy",
        "answer_generator": "FakeAnswerGenerator",
    }
    assert effective["skill"]["sha256"] == harness.skill.sha256


def test_harness_instantiates_configured_ollama_providers(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    skill_path = tmp_path / "initial.md"
    skill_path.write_text(SKILL_TEXT, encoding="utf-8")
    config = AgentConfig(
        policy={
            "provider": "ollama",
            "model": "qwen3:8b",
            "temperature": 0,
            "think": "low",
            "timeout_seconds": 45,
            "keep_alive": "10m",
            "max_retries": 1,
            "num_ctx": 8_192,
        },
        answer={
            "provider": "ollama",
            "model": "qwen3:4b",
            "temperature": 0,
            "think": False,
            "timeout_seconds": 60,
            "keep_alive": 0,
            "max_retries": 0,
            "num_ctx": 4_096,
        },
    )

    harness = AgentHarness.from_config(
        built_substrate,
        config,
        skill_path,
        tmp_path / "runs",
        embedding_backend=fake_embedder,
    )

    assert isinstance(harness.policy, OllamaChatPolicy)
    assert harness.policy.model == "qwen3:8b"
    assert harness.policy.think == "low"
    assert harness.policy.timeout_seconds == 45
    assert harness.policy.keep_alive == "10m"
    assert harness.policy.max_retries == 1
    assert harness.policy.num_ctx == 8_192
    assert isinstance(
        harness.answer_generator, OllamaChatAnswerGenerator
    )
    assert harness.answer_generator.model == "qwen3:4b"
    assert harness.answer_generator.think is False
    assert harness.answer_generator.timeout_seconds == 60
    assert harness.answer_generator.keep_alive == 0
    assert harness.answer_generator.max_retries == 0
    assert harness.answer_generator.num_ctx == 4_096


@pytest.mark.parametrize(
    ("policy_provider", "answer_provider", "policy_type", "answer_type"),
    [
        (
            "ollama",
            "openai",
            OllamaChatPolicy,
            OpenAIResponsesAnswerGenerator,
        ),
        (
            "openai",
            "ollama",
            OpenAIResponsesPolicy,
            OllamaChatAnswerGenerator,
        ),
    ],
)
def test_harness_instantiates_mixed_provider_directions(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy_provider: str,
    answer_provider: str,
    policy_type: type,
    answer_type: type,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    skill_path = tmp_path / "initial.md"
    skill_path.write_text(SKILL_TEXT, encoding="utf-8")

    def provider_config(provider: str, model: str) -> dict[str, str]:
        return {"provider": provider, "model": model}

    harness = AgentHarness.from_config(
        built_substrate,
        AgentConfig(
            policy=provider_config(policy_provider, "policy-model"),
            answer=provider_config(answer_provider, "answer-model"),
        ),
        skill_path,
        tmp_path / "runs",
        embedding_backend=fake_embedder,
    )

    assert isinstance(harness.policy, policy_type)
    assert isinstance(harness.answer_generator, answer_type)
    assert harness.policy.model == "policy-model"
    assert harness.answer_generator.model == "answer-model"


@pytest.mark.parametrize("injected_component", ["policy", "answer"])
def test_harness_injected_provider_skips_configured_client_creation(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injected_component: str,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    skill_path = tmp_path / "initial.md"
    skill_path.write_text(SKILL_TEXT, encoding="utf-8")
    scripted_policy = ScriptedPolicy([])
    scripted_answer = FakeAnswerGenerator("unused")

    if injected_component == "policy":
        config = AgentConfig(
            policy={"provider": "openai", "model": "must-not-build"},
            answer={"provider": "ollama", "model": "answer-model"},
        )
        kwargs: dict[str, Any] = {"policy": scripted_policy}
    else:
        config = AgentConfig(
            policy={"provider": "ollama", "model": "policy-model"},
            answer={"provider": "openai", "model": "must-not-build"},
        )
        kwargs = {"answer_generator": scripted_answer}

    harness = AgentHarness.from_config(
        built_substrate,
        config,
        skill_path,
        tmp_path / "runs",
        embedding_backend=fake_embedder,
        **kwargs,
    )

    if injected_component == "policy":
        assert harness.policy is scripted_policy
        assert isinstance(
            harness.answer_generator, OllamaChatAnswerGenerator
        )
    else:
        assert isinstance(harness.policy, OllamaChatPolicy)
        assert harness.answer_generator is scripted_answer


def test_read_and_evidence_resolution_remain_scope_safe_when_called_directly(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    q2_sentence = _sentence_id(
        substrate, "Paris is the capital of France."
    )
    q2_chunk = substrate.chunk_for_sentence(q2_sentence).chunk_id
    q1_sentence = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    q1_chunk = substrate.chunk_for_sentence(q1_sentence).chunk_id
    harness, _, _ = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "runs",
        [],
    )
    forged_state = ControllerState.initial()
    forged_state.visible_chunk_ids.add(q2_chunk)

    with pytest.raises(NodeNotFoundError, match="not present in scope q1"):
        harness.controller.router.execute(
            ReadAction(chunk_id=q2_chunk),
            forged_state,
            question=QUESTION_BIRTHPLACE,
            scope_id="q1",
            action_id="cross-scope-read",
        )
    with pytest.raises(NodeNotFoundError, match="not present in scope q1"):
        harness.controller.evidence_resolver.resolve(
            [SentenceRef(id=q2_sentence)], forged_state, "q1"
        )

    empty_state = ControllerState.initial()
    with pytest.raises(EvidenceEligibilityError, match="complete evidence"):
        harness.controller.evidence_resolver.resolve(
            [SentenceRef(id=q1_sentence)], empty_state, "q1"
        )
    with pytest.raises(EvidenceEligibilityError, match="must be READ"):
        harness.controller.evidence_resolver.resolve(
            [ChunkRef(id=q1_chunk)], empty_state, "q1"
        )

    eligible_state = ControllerState.initial()
    eligible_state.eligible_sentence_ids.add(q1_sentence)
    eligible_state.read_chunk_ids.add(q1_chunk)
    resolved = harness.controller.evidence_resolver.resolve(
        [SentenceRef(id=q1_sentence), ChunkRef(id=q1_chunk)],
        eligible_state,
        "q1",
    )
    assert [item.ref for item in resolved] == [ChunkRef(id=q1_chunk)]


def test_answer_generation_error_usage_is_recorded_on_finish_step(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    born_sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    decisions = [
        _decision(
            SearchAction(
                query=QUESTION_BIRTHPLACE,
                method="BM25",
                target="SENTENCE",
            )
        ),
        _decision(
            FinishAction(
                evidence_refs=[SentenceRef(id=born_sentence_id)]
            ),
            status=AssessmentStatus.SUFFICIENT,
        ),
    ]
    policy = ScriptedPolicy(decisions)
    answer_generator = ScriptedAnswerGenerator([])
    harness = AgentHarness(
        substrate=substrate,
        config=AgentConfig(),
        skill=SkillDocument.from_text(SKILL_TEXT),
        policy=policy,
        answer_generator=answer_generator,
        output_root=tmp_path / "runs",
        embedding_backend=fake_embedder,
    )

    result = harness.controller.run_episode(
        QUESTION_BIRTHPLACE,
        "q1",
        episode_id="answer-generation-error",
    )

    assert result.termination_reason is (
        TerminationReason.ANSWER_GENERATION_ERROR
    )
    assert result.usage.answer_calls == 1
    assert result.trajectory[-1].usage.answer_calls == 1
    assert sum(step.usage.answer_calls for step in result.trajectory) == 1


def test_router_runtime_error_is_recorded_and_consumes_one_step(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    first_search = _decision(
        SearchAction(
            query=QUESTION_BIRTHPLACE,
            method="BM25",
            target="SENTENCE",
        )
    )
    selected_search = _decision(
        SearchAction(
            query=QUESTION_COUNTRY,
            method="DENSE",
            target="SENTENCE",
        ),
        selected_refs=[SentenceRef(id=sentence_id)],
    )
    harness, _, answer_generator = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "runs",
        [first_search, selected_search],
        max_steps=3,
    )

    original_execute = harness.controller.router.execute
    calls = 0

    def fail_second_router(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise AgenticRAGError("simulated router failure")
        return original_execute(*args, **kwargs)

    monkeypatch.setattr(
        harness.controller.router, "execute", fail_second_router
    )
    result = harness.controller.run_episode(
        QUESTION_BIRTHPLACE,
        "q1",
        episode_id="router-runtime-error",
    )

    assert result.termination_reason is TerminationReason.RUNTIME_ERROR
    assert answer_generator.calls == []
    assert len(result.trajectory) == 2
    failed_step = result.trajectory[1]
    assert failed_step.observation.status is ObservationStatus.ERROR
    assert failed_step.observation.error_code == "agentic_rag_error"
    assert failed_step.state_after.step == 2
    assert failed_step.state_after.remaining_step_budget == 1
    assert failed_step.state_after.selected_evidence_refs == [
        SentenceRef(id=sentence_id)
    ]
    assert (
        failed_step.state_after.last_assessment
        == selected_search.assessment
    )
    assert result.final_state == failed_step.state_after


def test_resolved_evidence_deduplicates_sentence_contained_by_selected_chunk(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    chunk_id = substrate.chunk_for_sentence(sentence_id).chunk_id
    selected_refs = [
        SentenceRef(id=sentence_id),
        ChunkRef(id=chunk_id),
    ]
    decisions = [
        _decision(
            SearchAction(
                query=QUESTION_BIRTHPLACE,
                method="BM25",
                target="SENTENCE",
            )
        ),
        _decision(ReadAction(chunk_id=chunk_id)),
        _decision(
            FinishAction(evidence_refs=selected_refs),
            status=AssessmentStatus.SUFFICIENT,
        ),
    ]
    harness, _, answer_generator = _harness(
        built_substrate,
        fake_embedder,
        tmp_path / "runs",
        decisions,
    )

    result = harness.controller.run_episode(
        QUESTION_BIRTHPLACE,
        "q1",
        episode_id="selected-vs-resolved",
    )

    assert result.termination_reason is TerminationReason.FINISH
    assert result.selected_evidence_refs == selected_refs
    assert [item.ref for item in result.resolved_evidence] == [
        ChunkRef(id=chunk_id)
    ]
    assert [item.ref for item in answer_generator.calls[0][1]] == [
        ChunkRef(id=chunk_id)
    ]
