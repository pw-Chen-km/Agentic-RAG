from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.agent.answer import ScriptedAnswerGenerator
from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.context_resolution import (
    ContextIndexResolutionError,
    resolve_v3_decision,
)
from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    AssessmentStatus,
    ChunkRef,
    ContextNodeReference,
    ContextReferenceMap,
    ControllerState,
    Observation,
    ObservationStatus,
    SearchAction,
    SentenceRef,
    V3EvidenceAssessment,
    V3ExpandAction,
    V3FinishAction,
    V3PolicyDecision,
    V3ReadAction,
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
        missing_information=["The answer is not established."],
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


def test_v3_provider_schema_keeps_search_and_removes_legacy_refs() -> None:
    provider_model = policy_decision_model(
        (),
        direct_answer=True,
        semantic_memory_v3=True,
    )
    search_payload = {
        "assessment": {
            "status": "INSUFFICIENT",
            "supported_facts": [],
            "missing_information": ["Need the birthplace."],
        },
        "action": {
            "type": "SEARCH",
            "query": "Where was Marie Curie born?",
            "method": "BM25",
            "target": "SENTENCE",
            "top_k": 5,
        },
    }
    parsed = provider_model.model_validate(search_payload)
    decision = V3PolicyDecision.model_validate(parsed.model_dump(mode="json"))
    assert isinstance(decision.action, SearchAction)

    schema = json.dumps(provider_model.model_json_schema(), sort_keys=True)
    assert '"SEARCH"' in schema
    assert "selected_evidence_refs" not in schema
    assert "evidence_refs" not in schema
    assert "source_id" not in schema

    finish_payload = {
        "assessment": {
            "status": "SUFFICIENT",
            "supported_facts": ["Marie Curie was born in Warsaw."],
            "missing_information": [],
        },
        "action": {
            "type": "FINISH",
            "answer": "Warsaw",
            "citations": [1],
        },
    }
    provider_model.model_validate(finish_payload)
    finish_payload["action"]["citations"] = [1, 1]
    with pytest.raises(ValidationError):
        V3PolicyDecision.model_validate(finish_payload)


def test_v3_context_is_semantic_and_maps_are_not_in_policy_input(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state, entity_id, sentence_id, chunk_id = _semantic_state(substrate)
    builder = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate),
        single_agent_v3=True,
    )

    built = builder.build(
        "Where was Marie Curie born?",
        "# Skill\nSearch when information is missing.",
        state,
        [],
        scope_id="q1",
    )

    payload = json.loads(built.messages[-1].content)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "semantic_memory" in payload["policy_state"]
    assert "actionable_handles" not in serialized
    assert "allowed_expansions" not in serialized
    assert "Action Targets" not in serialized
    assert entity_id not in serialized
    assert sentence_id not in serialized
    assert chunk_id not in serialized
    assert "Marie Curie" in serialized
    assert "Marie Curie was born in Warsaw" in serialized

    assert built.reference_map is not None
    assert built.decision_format is not None
    decision_schema = json.dumps(
        built.decision_format.model_json_schema(), sort_keys=True
    )
    assert '"SEARCH"' in decision_schema
    assert "chunk_context_index" in decision_schema
    assert "citations" in decision_schema
    assert {
        ref.stable_id for ref in built.reference_map.memory.values()
    } == {entity_id, sentence_id, chunk_id}
    assert list(built.reference_map.citations.values()) == [
        SentenceRef(id=sentence_id)
    ]


def test_v3_empty_context_schema_keeps_all_actions_parallel(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    builder = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate),
        single_agent_v3=True,
    )
    built = builder.build(
        "Where was Marie Curie born?",
        "# Skill",
        ControllerState.initial(),
        [],
        scope_id="q1",
    )

    assert built.decision_format is not None
    schema = json.dumps(built.decision_format.model_json_schema())
    assert '"SEARCH"' in schema
    assert '"EXPAND"' in schema
    assert '"READ"' in schema
    assert '"FINISH"' in schema
    assert "source_context_index" in schema
    assert "chunk_context_index" in schema
    assert "citations" in schema


def test_v3_resolves_memory_and_citation_namespaces_independently() -> None:
    refs = ContextReferenceMap(
        memory={
            1: ContextNodeReference(
                node_type="ENTITY", stable_id="entity-stable"
            ),
            2: ContextNodeReference(
                node_type="CHUNK", stable_id="chunk-stable", can_read=True
            ),
        },
        citations={1: SentenceRef(id="sentence-stable")},
    )
    expand = resolve_v3_decision(
        V3PolicyDecision(
            assessment=_assessment(),
            action=V3ExpandAction(
                kind="ENTITY_MENTIONED_IN_SENTENCE",
                source_context_index=1,
            ),
        ),
        refs,
    )
    assert expand.action.source_id == "entity-stable"

    read = resolve_v3_decision(
        V3PolicyDecision(
            assessment=_assessment(),
            action=V3ReadAction(chunk_context_index=2),
        ),
        refs,
    )
    assert read.action.chunk_id == "chunk-stable"

    finish = resolve_v3_decision(
        V3PolicyDecision(
            assessment=_assessment(AssessmentStatus.SUFFICIENT),
            action=V3FinishAction(answer="Warsaw", citations=[1]),
        ),
        refs,
    )
    assert finish.action.evidence_refs == [SentenceRef(id="sentence-stable")]
    assert finish.assessment.selected_evidence_refs == [
        SentenceRef(id="sentence-stable")
    ]

    with pytest.raises(ContextIndexResolutionError) as exc_info:
        resolve_v3_decision(
            V3PolicyDecision(
                assessment=_assessment(),
                action=V3ReadAction(chunk_context_index=1),
            ),
            refs,
        )
    assert exc_info.value.code == "memory_node_type_mismatch"

    with pytest.raises(ContextIndexResolutionError) as exc_info:
        resolve_v3_decision(
            V3PolicyDecision(
                assessment=_assessment(AssessmentStatus.SUFFICIENT),
                action=V3FinishAction(answer="Warsaw", citations=[2]),
            ),
            refs,
        )
    assert exc_info.value.code == "citation_index_out_of_range"


def test_v3_read_chunk_folds_contained_sentences(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state, _, sentence_id, chunk_id = _semantic_state(substrate)
    state.read_chunk_ids.add(chunk_id)
    builder = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate),
        single_agent_v3=True,
    )

    built = builder.build(
        "Where was Marie Curie born?",
        "# Skill",
        state,
        [],
        scope_id="q1",
    )
    items = built.policy_view.policy_state.semantic_memory
    assert not any(item.node_type == "SENTENCE" for item in items)
    chunk_item = next(item for item in items if item.node_type == "CHUNK")
    assert chunk_item.has_been_read is True
    assert chunk_item.text == substrate.chunk_by_id[chunk_id].text
    assert chunk_item.citation_index is not None
    assert built.reference_map is not None
    assert sentence_id not in {
        ref.id for ref in built.reference_map.citations.values()
    }
    assert list(built.reference_map.citations.values()) == [ChunkRef(id=chunk_id)]
    assert built.decision_format is not None
    schema = json.dumps(built.decision_format.model_json_schema())
    assert "citations" in schema
    assert "chunk_context_index" in schema


def test_v3_end_to_end_search_then_finish_uses_policy_answer(
    built_substrate: Path,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    seen_second_context: dict = {}

    search = V3PolicyDecision(
        assessment=_assessment(),
        action=SearchAction(
            query="Where was Marie Curie born?",
            method="BM25",
            target="SENTENCE",
        ),
    )

    def finish_from_context(messages):
        payload = json.loads(messages[-1].content)
        seen_second_context.update(payload)
        item = next(
            value
            for value in payload["policy_state"]["semantic_memory"]
            if value.get("node_type") == "SENTENCE"
            and "Marie Curie was born in Warsaw" in value.get("text", "")
        )
        return V3PolicyDecision(
            assessment=V3EvidenceAssessment(
                status="SUFFICIENT",
                supported_facts=["Marie Curie was born in Warsaw."],
                missing_information=[],
            ),
            action=V3FinishAction(
                answer="Warsaw",
                citations=[item["citation_index"]],
            ),
        )

    policy = ScriptedPolicy([search, finish_from_context])
    fallback_answer = ScriptedAnswerGenerator("must not be called")
    harness = AgentHarness(
        substrate=substrate,
        config=AgentConfig(workflow_mode="single_agent_v3"),
        skill=SkillDocument.from_text("# V3\nSearch when a fact is missing."),
        policy=policy,
        answer_generator=fallback_answer,
        output_root=tmp_path / "runs",
    )

    result = harness.run(
        "Where was Marie Curie born?",
        "q1",
        episode_id="single-agent-v3",
    )

    assert result.answer == "Warsaw"
    assert result.usage.policy_calls == 2
    assert result.usage.answer_calls == 0
    assert fallback_answer.calls == []
    assert seen_second_context["policy_state"]["semantic_memory"]
    assert result.trajectory[0].context_reference_map is not None
    assert result.trajectory[1].context_reference_map is not None
    trace = harness.build_io_trace(result)
    assert trace["answer_generation"] is None
    assert trace["policy_calls"][-1]["output"]["action"] == {
        "type": "FINISH",
        "answer": "Warsaw",
        "citations": [
            result.trajectory[-1].decision.action.citations[0]
        ],
    }
