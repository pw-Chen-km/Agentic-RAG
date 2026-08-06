from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rag.agent.answer import ScriptedAnswerGenerator
from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.context_resolution import (
    ContextIndexResolutionError,
    resolve_v31_decision,
)
from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    AssessmentStatus,
    ControllerState,
    Observation,
    ObservationStatus,
    SearchAction,
    TypedContextNodeReference,
    TypedContextReferenceMap,
    V31ExpandAction,
    V31FinishAction,
    V31PolicyDecision,
    V31ReadAction,
    V3EvidenceAssessment,
)
from agentic_rag.agent.policy import ScriptedPolicy, policy_decision_model
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.state import StateUpdater
from agentic_rag.storage import Substrate


def _assessment(
    status: AssessmentStatus = AssessmentStatus.INSUFFICIENT,
) -> V3EvidenceAssessment:
    return V3EvidenceAssessment(
        status=status,
        supported_facts=[],
        missing_information=[] if status is AssessmentStatus.SUFFICIENT else [
            "The answer is not established."
        ],
    )


def _semantic_state(substrate: Substrate) -> tuple[ControllerState, str, str, str]:
    sentence = next(
        item
        for item in substrate.sentences
        if "Marie Curie was born in Warsaw" in item.text
    )
    chunk = substrate.chunk_by_id[sentence.chunk_id]
    entity = next(
        item for item in substrate.entities if item.canonical_name == "Marie Curie"
    )
    state = ControllerState.initial()
    state.visible_entity_ids.add(entity.entity_id)
    state.visible_sentence_ids.add(sentence.sentence_id)
    state.eligible_sentence_ids.add(sentence.sentence_id)
    state.visible_chunk_ids.add(chunk.chunk_id)
    state.semantic_memory_node_ids = [
        entity.entity_id,
        sentence.sentence_id,
        chunk.chunk_id,
    ]
    StateUpdater(substrate)._update_node_handles(
        state,
        Observation(status=ObservationStatus.OK),
    )
    return state, entity.entity_id, sentence.sentence_id, chunk.chunk_id


def test_v31_schema_uses_one_typed_reference_namespace() -> None:
    provider_model = policy_decision_model(
        direct_answer=True,
        semantic_memory_v31=True,
    )
    schema = json.dumps(provider_model.model_json_schema(), sort_keys=True)
    for field in ("source_ref", "chunk_ref", "evidence_refs"):
        assert field in schema
    for legacy in (
        "source_context_index",
        "chunk_context_index",
        "citation_index",
        "citations",
        "selected_evidence_refs",
    ):
        assert legacy not in schema
    assert '"SEARCH"' in schema
    assert '"EXPAND"' in schema
    assert '"READ"' in schema
    assert '"FINISH"' in schema

    normalized = V31PolicyDecision(
        assessment=_assessment(),
        action=V31ReadAction(chunk_ref=" c1 "),
    )
    assert normalized.action.chunk_ref == "C1"


def test_v31_context_exposes_semantics_and_stable_typed_refs(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state, entity_id, sentence_id, chunk_id = _semantic_state(substrate)
    builder = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate),
        single_agent_v3_typed_refs=True,
    )

    first = builder.build(
        "Where was Marie Curie born?",
        "# Skill\nSearch when information is missing.",
        state,
        [],
        scope_id="q1",
    )
    payload = json.loads(first.messages[-1].content)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "Marie Curie was born in Warsaw" in serialized
    assert "context_index" not in serialized
    assert "citation_index" not in serialized
    assert "valid_action_catalog" not in serialized
    assert entity_id not in serialized
    assert sentence_id not in serialized
    assert chunk_id not in serialized

    assert isinstance(first.reference_map, TypedContextReferenceMap)
    typed_refs = first.reference_map.typed_refs
    assert {item.node_type for item in typed_refs.values()} == {
        "ENTITY",
        "SENTENCE",
        "CHUNK",
    }
    stable_refs = {
        item.stable_id: ref for ref, item in typed_refs.items()
    }
    assert stable_refs[entity_id].startswith("E")
    assert stable_refs[sentence_id].startswith("S")
    assert stable_refs[chunk_id].startswith("C")

    another_entity = next(
        item
        for item in substrate.entities
        if item.entity_id != entity_id
    )
    state.visible_entity_ids.add(another_entity.entity_id)
    state.semantic_memory_node_ids.append(another_entity.entity_id)
    StateUpdater(substrate)._update_node_handles(
        state,
        Observation(status=ObservationStatus.OK),
    )
    second = builder.build(
        "Where was Marie Curie born?",
        "# Skill",
        state,
        [],
        scope_id="q1",
    )
    assert isinstance(second.reference_map, TypedContextReferenceMap)
    for stable_id, ref in stable_refs.items():
        assert second.reference_map.typed_refs[ref].stable_id == stable_id


def test_v31_can_omit_latest_event_from_policy_input(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state, _, _, _ = _semantic_state(substrate)
    state.newest_observation = Observation(
        action_id="attempt-1",
        status=ObservationStatus.OK,
        retrieved_tokens=17,
    )

    included = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate),
        single_agent_v3_typed_refs=True,
    ).build(
        "Where was Marie Curie born?",
        "# Skill",
        state,
        [],
        scope_id="q1",
    )
    omitted = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate),
        single_agent_v3_typed_refs=True,
        include_v3_latest_event=False,
    ).build(
        "Where was Marie Curie born?",
        "# Skill",
        state,
        [],
        scope_id="q1",
    )

    included_payload = json.loads(included.messages[-1].content)
    omitted_payload = json.loads(omitted.messages[-1].content)
    assert "latest_event" in included_payload["policy_state"]
    assert "latest_event" not in omitted_payload["policy_state"]
    assert (
        omitted_payload["policy_state"]["semantic_memory"]
        == included_payload["policy_state"]["semantic_memory"]
    )
    assert (
        omitted_payload["policy_state"]["attempted_actions"]
        == included_payload["policy_state"]["attempted_actions"]
    )


@pytest.mark.parametrize(
    ("builder_option", "omitted_field"),
    [
        ("include_v3_last_assessment", "last_assessment"),
        ("include_v3_attempted_actions", "attempted_actions"),
        ("include_v3_budget", "budget"),
    ],
)
def test_v31_can_omit_one_policy_state_feature(
    built_substrate: Path,
    builder_option: str,
    omitted_field: str,
) -> None:
    substrate = Substrate.open(built_substrate)
    state, _, _, _ = _semantic_state(substrate)
    state.last_assessment = _assessment()
    options = {
        "evidence_resolver": EvidenceResolver(substrate),
        "single_agent_v3_typed_refs": True,
        builder_option: False,
    }

    built = PolicyContextBuilder(**options).build(
        "Where was Marie Curie born?",
        "# Skill",
        state,
        [],
        scope_id="q1",
    )
    policy_state = json.loads(built.messages[-1].content)["policy_state"]

    assert omitted_field not in policy_state
    for retained_field in {
        "last_assessment",
        "semantic_memory",
        "latest_event",
        "attempted_actions",
        "budget",
    } - {omitted_field}:
        assert retained_field in policy_state


def test_v32_omits_latest_event_and_uses_one_line_budget(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state, _, _, _ = _semantic_state(substrate)
    state.remaining_step_budget = 4
    state.remaining_policy_attempt_budget = 5
    state.remaining_retrieved_token_budget = 3200
    state.newest_observation = Observation(
        action_id="attempt-1",
        status=ObservationStatus.OK,
        retrieved_tokens=17,
    )
    builder = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate),
        single_agent_v3_typed_refs=True,
        include_v3_latest_event=False,
        compact_v3_budget=True,
    )

    built = builder.build(
        "Where was Marie Curie born?",
        "# Skill",
        state,
        [],
        scope_id="q1",
    )
    policy_state = json.loads(built.messages[-1].content)["policy_state"]

    assert "latest_event" not in policy_state
    assert policy_state["budget"] == (
        "Budget: 4 steps, 5 attempts, 3200 retrieval tokens left"
    )
    assert "last_assessment" in policy_state
    assert "semantic_memory" in policy_state
    assert "attempted_actions" in policy_state
    assert built.policy_view.context_mode == "semantic_memory_v3_2"


def test_v31_resolves_and_validates_typed_refs() -> None:
    refs = TypedContextReferenceMap(
        typed_refs={
            "E1": TypedContextNodeReference(
                node_type="ENTITY", stable_id="entity-stable"
            ),
            "S1": TypedContextNodeReference(
                node_type="SENTENCE",
                stable_id="sentence-stable",
                can_use_as_evidence=True,
            ),
            "C1": TypedContextNodeReference(
                node_type="CHUNK", stable_id="chunk-unread", can_read=True
            ),
            "C2": TypedContextNodeReference(
                node_type="CHUNK",
                stable_id="chunk-read",
                can_use_as_evidence=True,
            ),
        }
    )
    expand = resolve_v31_decision(
        V31PolicyDecision(
            assessment=_assessment(),
            action=V31ExpandAction(
                kind="ENTITY_MENTIONED_IN_SENTENCE",
                source_ref="E1",
            ),
        ),
        refs,
    )
    assert expand.action.source_id == "entity-stable"

    read = resolve_v31_decision(
        V31PolicyDecision(
            assessment=_assessment(),
            action=V31ReadAction(chunk_ref="C1"),
        ),
        refs,
    )
    assert read.action.chunk_id == "chunk-unread"

    finish = resolve_v31_decision(
        V31PolicyDecision(
            assessment=_assessment(AssessmentStatus.SUFFICIENT),
            action=V31FinishAction(
                answer="Warsaw", evidence_refs=["S1", "C2"]
            ),
        ),
        refs,
    )
    assert [ref.id for ref in finish.action.evidence_refs] == [
        "sentence-stable",
        "chunk-read",
    ]

    cases = [
        (
            V31ReadAction(chunk_ref="S1"),
            "reference_type_mismatch",
        ),
        (
            V31ReadAction(chunk_ref="C2"),
            "chunk_not_readable",
        ),
        (
            V31ReadAction(chunk_ref="C9"),
            "reference_not_available",
        ),
        (
            V31FinishAction(answer="x", evidence_refs=["E1"]),
            "reference_not_evidence",
        ),
    ]
    for action, expected_code in cases:
        with pytest.raises(ContextIndexResolutionError) as exc_info:
            resolve_v31_decision(
                V31PolicyDecision(assessment=_assessment(), action=action),
                refs,
            )
        assert exc_info.value.code == expected_code


def test_v31_read_chunk_folds_sentence_and_promotes_chunk(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state, _, sentence_id, chunk_id = _semantic_state(substrate)
    sentence_ref = state.handle_registry.handle_for(sentence_id, "SENTENCE")
    chunk_ref = state.handle_registry.handle_for(chunk_id, "CHUNK")
    state.read_chunk_ids.add(chunk_id)
    StateUpdater(substrate)._update_node_handles(
        state,
        Observation(status=ObservationStatus.OK),
    )
    builder = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate),
        single_agent_v3_typed_refs=True,
    )
    built = builder.build("question", "# Skill", state, [], scope_id="q1")
    assert isinstance(built.reference_map, TypedContextReferenceMap)
    assert sentence_ref not in built.reference_map.typed_refs
    assert chunk_ref in built.reference_map.typed_refs
    chunk_mapping = built.reference_map.typed_refs[chunk_ref]
    assert chunk_mapping.can_read is False
    assert chunk_mapping.can_use_as_evidence is True
    chunk_item = next(
        item
        for item in built.policy_view.policy_state.semantic_memory
        if item.node_type == "CHUNK"
    )
    assert chunk_item.ref == chunk_ref
    assert chunk_item.has_been_read is True
    assert chunk_item.text == substrate.chunk_by_id[chunk_id].text


def test_v31_end_to_end_search_then_finish_uses_typed_ref(
    built_substrate: Path,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    search = V31PolicyDecision(
        assessment=_assessment(),
        action=SearchAction(
            query="Where was Marie Curie born?",
            method="BM25",
            target="SENTENCE",
        ),
    )

    def finish_from_context(messages):
        payload = json.loads(messages[-1].content)
        item = next(
            value
            for value in payload["policy_state"]["semantic_memory"]
            if value.get("node_type") == "SENTENCE"
            and "Marie Curie was born in Warsaw" in value.get("text", "")
        )
        return V31PolicyDecision(
            assessment=V3EvidenceAssessment(
                status="SUFFICIENT",
                supported_facts=["Marie Curie was born in Warsaw."],
                missing_information=[],
            ),
            action=V31FinishAction(
                answer="Warsaw",
                evidence_refs=[item["ref"]],
            ),
        )

    policy = ScriptedPolicy([search, finish_from_context])
    fallback_answer = ScriptedAnswerGenerator("must not be called")
    harness = AgentHarness(
        substrate=substrate,
        config=AgentConfig(workflow_mode="single_agent_v3_typed_refs"),
        skill=SkillDocument.from_text("# V3.1\nSearch missing facts."),
        policy=policy,
        answer_generator=fallback_answer,
        output_root=tmp_path / "runs",
    )
    result = harness.run(
        "Where was Marie Curie born?",
        "q1",
        episode_id="single-agent-v31",
    )
    assert result.answer == "Warsaw"
    assert result.usage.policy_calls == 2
    assert result.usage.answer_calls == 0
    assert all(
        step.context_reference_map is not None for step in result.trajectory
    )
    trace = harness.build_io_trace(result)
    assert trace["policy_calls"][-1]["output"]["action"] == {
        "type": "FINISH",
        "answer": "Warsaw",
        "evidence_refs": [
            result.trajectory[-1].decision.action.evidence_refs[0]
        ],
    }


def test_v31_invalid_ref_attempt_is_audited_without_consuming_step(
    built_substrate: Path,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    search = V31PolicyDecision(
        assessment=_assessment(),
        action=SearchAction(
            query="Where was Marie Curie born?",
            method="BM25",
            target="SENTENCE",
        ),
    )
    visible_sentence_ref: dict[str, str] = {}

    def invalid_read(messages):
        payload = json.loads(messages[-1].content)
        sentence = next(
            item
            for item in payload["policy_state"]["semantic_memory"]
            if item.get("node_type") == "SENTENCE"
            and "Marie Curie was born in Warsaw" in item.get("text", "")
        )
        visible_sentence_ref["value"] = sentence["ref"]
        return V31PolicyDecision(
            assessment=_assessment(),
            action=V31ReadAction(chunk_ref=sentence["ref"]),
        )

    def finish(messages):
        return V31PolicyDecision(
            assessment=_assessment(AssessmentStatus.SUFFICIENT),
            action=V31FinishAction(
                answer="Warsaw",
                evidence_refs=[visible_sentence_ref["value"]],
            ),
        )

    harness = AgentHarness(
        substrate=substrate,
        config=AgentConfig(workflow_mode="single_agent_v3_typed_refs"),
        skill=SkillDocument.from_text("# V3.1"),
        policy=ScriptedPolicy([search, invalid_read, finish]),
        answer_generator=ScriptedAnswerGenerator("unused"),
        output_root=tmp_path / "runs-invalid",
    )
    result = harness.run(
        "Where was Marie Curie born?",
        "q1",
        episode_id="single-agent-v31-invalid",
    )
    assert result.answer == "Warsaw"
    assert result.usage.policy_calls == 3
    assert result.trajectory[1].observation.error_code == (
        "reference_type_mismatch"
    )
    assert result.trajectory[1].step == result.trajectory[2].step
    assert result.trajectory[1].context_reference_map is not None


def test_v31_workflow_config_is_accepted() -> None:
    config = AgentConfig(workflow_mode="single_agent_v3_typed_refs")
    assert config.workflow_mode == "single_agent_v3_typed_refs"
    assert config.v3_include_last_assessment is True
    assert config.v3_include_latest_event is True
    assert config.v3_include_attempted_actions is True
    assert config.v3_include_budget is True


def test_v32_workflow_config_and_harness_wiring(
    built_substrate: Path,
    tmp_path: Path,
) -> None:
    config = AgentConfig(workflow_mode="single_agent_v3_2")
    harness = AgentHarness(
        substrate=Substrate.open(built_substrate),
        config=config,
        skill=SkillDocument.from_text("# V3.2"),
        policy=ScriptedPolicy([]),
        answer_generator=ScriptedAnswerGenerator("unused"),
        output_root=tmp_path / "runs-v32",
    )

    assert config.workflow_mode == "single_agent_v3_2"
    assert harness.context_builder.single_agent_v3_typed_refs is True
    assert harness.context_builder.include_v3_latest_event is False
    assert harness.context_builder.compact_v3_budget is True
    assert harness.effective_config()["policy_context"][
        "node_reference_scheme"
    ] == "episode_local_typed_refs_with_frozen_visibility_v1"
    assert harness.effective_config()["policy_context"][
        "latest_event_visible"
    ] is False
    assert harness.effective_config()["policy_context"][
        "budget_representation"
    ] == "compact_text_v1"
