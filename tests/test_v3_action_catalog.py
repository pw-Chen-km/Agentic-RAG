from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.agent.answer import ScriptedAnswerGenerator
from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    ControllerState,
    Observation,
    ObservationStatus,
    SearchAction,
    V3EvidenceAssessment,
    V3FinishAction,
    V3PolicyDecision,
)
from agentic_rag.agent.policy import ScriptedPolicy
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.state import StateUpdater
from agentic_rag.storage import Substrate


def _semantic_state(substrate: Substrate) -> tuple[ControllerState, set[str]]:
    sentence = next(
        item
        for item in substrate.sentences
        if "Marie Curie was born in Warsaw" in item.text
    )
    chunk = substrate.chunk_by_id[sentence.chunk_id]
    entity = next(
        item
        for item in substrate.entities
        if item.canonical_name == "Marie Curie"
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
        state, Observation(status=ObservationStatus.OK)
    )
    return state, {entity.entity_id, sentence.sentence_id, chunk.chunk_id}


def test_v3_action_catalog_lists_all_structurally_valid_templates(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state, stable_ids = _semantic_state(substrate)
    builder = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate),
        single_agent_v3=True,
        single_agent_v3_action_catalog=True,
    )

    built = builder.build(
        "Where was Marie Curie born?",
        "# Skill\nSearch when information is missing.",
        state,
        [],
        scope_id="q1",
    )

    payload = json.loads(built.messages[-1].content)
    catalog = payload["policy_state"]["valid_action_catalog"]
    assert len(catalog["SEARCH"]) == 6
    assert {
        (item["method"], item["target"])
        for item in catalog["SEARCH"]
    } == {
        ("LEXICAL", "ENTITY"),
        ("BM25", "SENTENCE"),
        ("BM25", "CHUNK"),
        ("DENSE", "ENTITY"),
        ("DENSE", "SENTENCE"),
        ("DENSE", "CHUNK"),
    }
    assert {
        (item["kind"], item["source_context_index"])
        for item in catalog["EXPAND"]
    } == {
        ("ENTITY_MENTIONED_IN_SENTENCE", 1),
        ("ENTITY_CO_OCCURS_ENTITY_SENTENCE", 1),
        ("SENTENCE_MENTIONS_ENTITY", 2),
        ("CHUNK_ADJACENT_CHUNK", 3),
    }
    adjacent = next(
        item
        for item in catalog["EXPAND"]
        if item["kind"] == "CHUNK_ADJACENT_CHUNK"
    )
    assert adjacent["valid_directions"] == ["PREV", "NEXT", "BOTH"]
    assert catalog["READ"] == [
        {"type": "READ", "chunk_context_index": 3}
    ]
    assert catalog["FINISH"] == [
        {"type": "FINISH", "available_citation_indices": [1]}
    ]

    serialized = json.dumps(payload, ensure_ascii=False)
    assert "valid_action_catalog" in serialized
    assert all(stable_id not in serialized for stable_id in stable_ids)
    assert built.reference_map is not None
    assert built.reference_map.memory[3].can_read is True
    assert "structurally executable" in built.messages[0].content


def test_v3_action_catalog_keeps_search_visible_with_empty_memory(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    builder = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate),
        single_agent_v3=True,
        single_agent_v3_action_catalog=True,
    )
    built = builder.build(
        "Where was Marie Curie born?",
        "# Skill",
        ControllerState.initial(),
        [],
        scope_id="q1",
    )
    catalog = json.loads(built.messages[-1].content)["policy_state"][
        "valid_action_catalog"
    ]

    assert len(catalog["SEARCH"]) == 6
    assert catalog["EXPAND"] == []
    assert catalog["READ"] == []
    assert catalog["FINISH"] == []
    assert built.decision_format is not None
    schema = json.dumps(built.decision_format.model_json_schema())
    assert all(
        action in schema
        for action in ('"SEARCH"', '"EXPAND"', '"READ"', '"FINISH"')
    )


def test_v3_baseline_does_not_receive_action_catalog(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state, _ = _semantic_state(substrate)
    built = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate),
        single_agent_v3=True,
    ).build(
        "Where was Marie Curie born?",
        "# Skill",
        state,
        [],
        scope_id="q1",
    )
    assert "valid_action_catalog" not in built.messages[-1].content


def test_v3_action_catalog_is_a_supported_workflow() -> None:
    config = AgentConfig(workflow_mode="single_agent_v3_action_catalog")
    assert config.workflow_mode == "single_agent_v3_action_catalog"


def test_v3_action_catalog_harness_keeps_v3_resolution_and_direct_finish(
    built_substrate: Path,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    seen_catalog: dict = {}
    search = V3PolicyDecision(
        assessment=V3EvidenceAssessment(
            status="INSUFFICIENT",
            supported_facts=[],
            missing_information=["Need the birthplace."],
        ),
        action=SearchAction(
            query="Where was Marie Curie born?",
            method="BM25",
            target="SENTENCE",
        ),
    )

    def finish_from_catalog(messages):
        payload = json.loads(messages[-1].content)
        seen_catalog.update(
            payload["policy_state"]["valid_action_catalog"]
        )
        citation = seen_catalog["FINISH"][0][
            "available_citation_indices"
        ][0]
        return V3PolicyDecision(
            assessment=V3EvidenceAssessment(
                status="SUFFICIENT",
                supported_facts=["Marie Curie was born in Warsaw."],
                missing_information=[],
            ),
            action=V3FinishAction(answer="Warsaw", citations=[citation]),
        )

    fallback = ScriptedAnswerGenerator("must not be called")
    harness = AgentHarness(
        substrate=substrate,
        config=AgentConfig(
            workflow_mode="single_agent_v3_action_catalog",
            max_steps=4,
            max_policy_attempts=6,
        ),
        skill=SkillDocument.from_text("# V3\nSearch, then finish."),
        policy=ScriptedPolicy([search, finish_from_catalog]),
        answer_generator=fallback,
        output_root=tmp_path / "runs",
    )

    result = harness.run(
        "Where was Marie Curie born?",
        "q1",
        episode_id="v3-action-catalog",
    )

    assert result.answer == "Warsaw"
    assert seen_catalog["SEARCH"]
    assert seen_catalog["FINISH"]
    assert result.usage.answer_calls == 0
    assert fallback.calls == []
