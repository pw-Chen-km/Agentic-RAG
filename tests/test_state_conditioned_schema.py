from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.agent.action_schema import ActionSchemaBuilder
from agentic_rag.agent.action_space import AvailableActionSpaceBuilder
from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.models import (
    ActionSpaceMode,
    Assessment,
    ContextReferenceMap,
    EpisodeState,
    ExpansionKind,
    PolicyDecision,
    ReadAction,
)
from agentic_rag.agent.policy import ScriptedPolicy, policy_decision_model
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.substrate.storage import Substrate


ASSESSMENT = {"supported_facts": [], "missing_information": ["answer"]}


def _decision(action: dict) -> dict:
    return {"assessment": ASSESSMENT, "action": action}


def _state_with_refs(substrate: Substrate) -> EpisodeState:
    state = EpisodeState.initial()
    chunk_ids = sorted(substrate.chunk_ids_by_scope["q1"])
    unread_chunk_id, read_chunk_id = chunk_ids[:2]
    sentence_id = substrate.sentences_by_chunk[unread_chunk_id][0].sentence_id
    entity_id = sorted(substrate.entity_ids_by_scope["q1"])[0]
    state.visible_entity_ids.add(entity_id)
    state.visible_sentence_ids.add(sentence_id)
    state.visible_chunk_ids.update({unread_chunk_id, read_chunk_id})
    state.eligible_sentence_ids.add(sentence_id)
    state.read_chunk_ids.add(read_chunk_id)
    state.semantic_memory_node_ids.extend(
        [entity_id, sentence_id, unread_chunk_id, read_chunk_id]
    )
    state.reference_registry.register(entity_id, "ENTITY")
    state.reference_registry.register(sentence_id, "SENTENCE")
    state.reference_registry.register(unread_chunk_id, "CHUNK")
    state.reference_registry.register(read_chunk_id, "CHUNK")
    return state


def test_empty_state_schema_contains_only_six_legal_search_pairs() -> None:
    state = EpisodeState.initial()
    space = AvailableActionSpaceBuilder(()).build(state, ContextReferenceMap())
    model = ActionSchemaBuilder().build(space)
    serialized = json.dumps(model.model_json_schema())

    assert len(space.search_options) == 6
    assert "SEARCH" in serialized
    assert all(item not in serialized for item in ("EXPAND", "READ", "FINISH"))
    model.model_validate(
        _decision(
            {
                "type": "SEARCH",
                "query": "Marie Curie birthplace",
                "method": "BM25",
                "target": "SENTENCE",
                "top_k": 5,
            }
        )
    )
    with pytest.raises(ValidationError):
        model.model_validate(
            _decision(
                {
                    "type": "SEARCH",
                    "query": "Marie Curie",
                    "method": "LEXICAL",
                    "target": "CHUNK",
                    "top_k": 5,
                }
            )
        )
    with pytest.raises(ValidationError):
        model.model_validate(_decision({"type": "READ", "chunk_ref": "C1"}))


def test_schema_and_prompt_share_reference_affordances(built_substrate: Path) -> None:
    substrate = Substrate.open(built_substrate)
    state = _state_with_refs(substrate)
    enabled = (
        ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
        ExpansionKind.SENTENCE_MENTIONS_ENTITY,
    )
    built = PolicyContextBuilder(substrate, enabled).build(
        "Question?",
        SkillDocument.from_text("Use evidence."),
        state,
        [],
        scope_id="q1",
    )
    space = built.available_action_space
    prompt = "\n".join(message.content for message in built.messages)

    assert space.read_refs == ("C1",)
    assert space.finish_evidence_refs == ("S1", "C2")
    assert "chunk_ref in [C1]" in prompt
    assert "subset of [S1, C2]" in prompt

    built.decision_format.model_validate(
        _decision({"type": "READ", "chunk_ref": "C1"})
    )
    built.decision_format.model_validate(
        _decision(
            {"type": "FINISH", "answer": "answer", "evidence_refs": ["S1", "C2"]}
        )
    )
    built.decision_format.model_validate(
        _decision(
            {
                "type": "EXPAND",
                "kind": "ENTITY_MENTIONED_IN_SENTENCE",
                "source_ref": "E1",
                "direction": None,
                "query": None,
                "top_k": 5,
            }
        )
    )
    for action in (
        {"type": "READ", "chunk_ref": "C2"},
        {"type": "FINISH", "answer": "answer", "evidence_refs": ["C1"]},
        {
            "type": "EXPAND",
            "kind": "ENTITY_MENTIONED_IN_SENTENCE",
            "source_ref": "S1",
            "direction": None,
            "query": None,
            "top_k": 5,
        },
    ):
        with pytest.raises(ValidationError):
            built.decision_format.model_validate(_decision(action))


def test_read_chunk_moves_from_read_to_finish_pool(built_substrate: Path) -> None:
    substrate = Substrate.open(built_substrate)
    state = _state_with_refs(substrate)
    builder = PolicyContextBuilder(substrate)
    before = builder.build(
        "Question?", SkillDocument.from_text("Use evidence."), state, [], scope_id="q1"
    )
    unread_id = before.reference_map.typed_refs["C1"].stable_id
    state.read_chunk_ids.add(unread_id)
    after = builder.build(
        "Question?", SkillDocument.from_text("Use evidence."), state, [], scope_id="q1"
    )

    assert "C1" in before.available_action_space.read_refs
    assert "C1" not in before.available_action_space.finish_evidence_refs
    assert "C1" not in after.available_action_space.read_refs
    assert "C1" in after.available_action_space.finish_evidence_refs


def test_budget_finalize_schema_is_finish_only(built_substrate: Path) -> None:
    substrate = Substrate.open(built_substrate)
    state = _state_with_refs(substrate)
    built = PolicyContextBuilder(substrate).build(
        "Question?",
        SkillDocument.from_text("Use evidence."),
        state,
        [],
        scope_id="q1",
        action_space_mode=ActionSpaceMode.BUDGET_FINALIZE,
    )
    serialized = json.dumps(built.decision_format.model_json_schema())

    assert built.available_action_space.mode is ActionSpaceMode.BUDGET_FINALIZE
    assert "FINISH" in serialized
    assert all(item not in serialized for item in ("SEARCH", "EXPAND", "READ"))


def test_prompt_and_schema_ablation_switches_are_independent(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state = _state_with_refs(substrate)
    skill = SkillDocument.from_text("Use evidence.")
    combinations = {}
    for prompt_options in (False, True):
        for state_schema in (False, True):
            built = PolicyContextBuilder(
                substrate,
                show_available_action_options=prompt_options,
                use_state_conditioned_schema=state_schema,
            ).build("Question?", skill, state, [], scope_id="q1")
            combinations[(prompt_options, state_schema)] = built
            prompt = "\n".join(message.content for message in built.messages)
            assert ("Currently available action options" in prompt) is prompt_options
            assert built.available_action_space == combinations.get(
                (False, state_schema), built
            ).available_action_space
            if state_schema:
                with pytest.raises(ValidationError):
                    built.decision_format.model_validate(
                        _decision({"type": "READ", "chunk_ref": "C2"})
                    )
            else:
                assert (
                    built.decision_format.model_json_schema()
                    == policy_decision_model().model_json_schema()
                )


def test_scripted_policy_is_still_checked_by_dynamic_schema() -> None:
    decision = PolicyDecision(
        assessment=Assessment(missing_information=["context"]),
        action=ReadAction(chunk_ref="C99"),
    )
    policy = ScriptedPolicy([decision])
    space = AvailableActionSpaceBuilder(()).build(
        EpisodeState.initial(), ContextReferenceMap()
    )
    with pytest.raises(Exception, match="failed PolicyDecision validation"):
        policy.decide(
            [],
            decision_format=ActionSchemaBuilder().build(space),
        )
