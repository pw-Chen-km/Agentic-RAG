from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.agent.answer import ScriptedAnswerGenerator
from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    AssessmentStatus,
    ChunkHandle,
    ControllerState,
    EntityHandle,
    EvidenceAssessment,
    ExpansionKind,
    FinishAction,
    PolicyDecision,
    SearchAction,
    SentenceHandle,
    SentenceRef,
)
from agentic_rag.agent.policy import ScriptedPolicy
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.storage import Substrate


def test_v2_compact_keeps_v2_handles_but_removes_repeated_ui_fields() -> None:
    state = ControllerState.initial(max_steps=6, max_retrieved_tokens=500)
    state.node_handles = {
        "entity:marie": EntityHandle(
            id="E1", label="Marie Curie", entity_type="PERSON"
        ),
        "sentence:birth": SentenceHandle(
            id="S1",
            text="Marie Curie was born in Warsaw.",
            parent_chunk_id="C1",
            document_id="document:marie",
            title="Marie Curie",
            can_use_as_evidence=True,
        ),
        "chunk:marie": ChunkHandle(
            id="C1",
            document_id="document:marie",
            title="Marie Curie",
        ),
    }
    builder = PolicyContextBuilder(
        enabled_expansions=(
            ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
            ExpansionKind.SENTENCE_MENTIONS_ENTITY,
            ExpansionKind.CHUNK_ADJACENT_CHUNK,
        ),
        single_agent_v2=True,
        single_agent_v2_compact=True,
    )

    built = builder.build(
        "Where was Marie Curie born?",
        SkillDocument.from_text("# Strategy\nRetrieve the missing fact."),
        state,
        [],
        scope_id=None,
    )

    payload = json.loads(built.messages[-1].content)
    serialized = built.messages[-1].content
    assert set(payload) == {"instruction", "state"}
    assert set(payload["state"]) == {
        "step",
        "policy_attempts",
        "last_assessment",
        "selected_evidence",
        "latest_observation",
        "known_nodes",
        "expand_sources",
        "action_history",
        "budget",
    }
    assert {item["id"] for item in payload["state"]["known_nodes"]} == {
        "E1",
        "S1",
        "C1",
    }
    assert payload["state"]["expand_sources"] == {
        "CHUNK_ADJACENT_CHUNK": ["C1"],
        "ENTITY_MENTIONED_IN_SENTENCE": ["E1"],
        "SENTENCE_MENTIONS_ENTITY": ["S1"],
    }
    for removed in (
        "actionable_handles",
        "allowed_expansions",
        "attempted_actions",
        "can_expand",
        "can_read",
    ):
        assert removed not in serialized

    # Controller-side resolution still receives the unmodified V2 PolicyView.
    assert len(built.policy_view.policy_state.actionable_handles) == 3
    assert built.policy_view.policy_state.allowed_expansions
    assert "Action Targets" not in "\n".join(
        message.content for message in built.messages
    )


def test_v2_compact_is_a_supported_v2_workflow() -> None:
    config = AgentConfig(workflow_mode="single_agent_v2_compact")
    assert config.workflow_mode == "single_agent_v2_compact"


def test_v2_compact_harness_preserves_v2_selected_evidence_semantics(
    built_substrate: Path,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence = next(
        item
        for item in substrate.sentences
        if "Marie Curie was born in Warsaw" in item.text
    )
    search = PolicyDecision(
        assessment=EvidenceAssessment(status=AssessmentStatus.INSUFFICIENT),
        action=SearchAction(
            query="Where was Marie Curie born?",
            method="BM25",
            target="SENTENCE",
        ),
    )
    finish = PolicyDecision(
        assessment=EvidenceAssessment(
            status=AssessmentStatus.SUFFICIENT,
            selected_evidence_refs=[SentenceRef(id=sentence.sentence_id)],
        ),
        action=FinishAction(
            evidence_refs=[SentenceRef(id=sentence.sentence_id)],
            answer="Warsaw",
        ),
    )
    fallback = ScriptedAnswerGenerator("must not be called")
    harness = AgentHarness(
        substrate=substrate,
        config=AgentConfig(
            workflow_mode="single_agent_v2_compact",
            max_steps=4,
            max_policy_attempts=6,
        ),
        skill=SkillDocument.from_text("# Strategy\nSearch, then finish."),
        policy=ScriptedPolicy([search, finish]),
        answer_generator=fallback,
        output_root=tmp_path / "runs",
    )

    result = harness.run(
        "Where was Marie Curie born?",
        "q1",
        episode_id="v2-compact",
    )

    assert result.answer == "Warsaw"
    assert result.final_state.selected_evidence_refs == [
        SentenceRef(id=sentence.sentence_id)
    ]
    assert result.usage.answer_calls == 0
    assert fallback.calls == []
