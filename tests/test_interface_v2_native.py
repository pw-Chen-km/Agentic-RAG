from __future__ import annotations

from pathlib import Path
import pytest

from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.action_schema import policy_decision_from_constrained
from agentic_rag.agent.interface import get_interface_contract
from agentic_rag.agent.models import EpisodeState
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.tool_calling import (
    build_tool_definitions,
    decision_from_tool_call,
)
from agentic_rag.agent.interface_action_catalog import ACTION_CARDS
from agentic_rag.agent.policy import PolicyResponseError, PolicyStateError
from agentic_rag.agent.references import ReferenceResolutionError, resolve_decision
from agentic_rag.substrate.storage import Substrate
from agentic_rag.substrate.models import Entity


def _tool_names(tools: list[dict]) -> list[str]:
    return [str(item["function"]["name"]) for item in tools]


def test_v2_condition_tool_registry_isolated(built_substrate: Path) -> None:
    substrate = Substrate.open(built_substrate)
    expected = {
        "C0": ["find_passages", "finish"],
        "C1": ["find_passages", "find_sentences", "finish"],
        "C2": ["find_passages", "finish"],
        "C3": ["find_passages", "finish"],
        "C5": ["find_passages", "find_sentences", "finish"],
        "C4": ["find_passages", "find_sentences", "finish"],
        "A1": ["find_passages", "find_sentences", "finish"],
    }
    for name, names in expected.items():
        contract = get_interface_contract(name)
        built = PolicyContextBuilder(
            substrate, interface_contract=contract
        ).build(
            "Question?",
            SkillDocument.from_text("Answer using available evidence."),
            EpisodeState.initial(),
            [],
            scope_id="q1",
        )
        assert _tool_names(built.tool_definitions) == names
        assert "read_passage" not in _tool_names(built.tool_definitions)
        assert built.provider_tools == built.tool_definitions
        decision_schema = built.decision_format.model_json_schema()
        assert set(decision_schema["required"]) == {"supported_facts", "missing_information", "action"}
        for tool in built.tool_definitions:
            parameters = tool["function"]["parameters"]
            assert "assessment" in parameters["properties"]


def test_native_tool_call_decodes_to_existing_action_model() -> None:
    decision = decision_from_tool_call(
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "find_sentences",
                "arguments": '{"assessment":{"supported_facts":[],"missing_information":["Where Marie Curie was born"]},"query":"Marie Curie"}',
            },
        }
    )
    assert decision.action.type == "SEARCH"
    assert decision.action.target.value == "SENTENCE"
    assert decision.action.query == "Marie Curie"
    assert decision.assessment.missing_information == ["Where Marie Curie was born"]


def test_tool_schema_uses_the_same_capability_card_descriptions(built_substrate: Path) -> None:
    substrate = Substrate.open(built_substrate)
    built = PolicyContextBuilder(
        substrate, interface_contract=get_interface_contract("C4")
    ).build("Question?", SkillDocument.from_text("Answer."), EpisodeState.initial(), [], scope_id="q1")
    descriptions = {
        item["function"]["name"]: item["function"]["description"]
        for item in build_tool_definitions(built.available_action_space)
    }
    for name in ("find_passages", "find_sentences", "finish"):
        assert descriptions[name] == ACTION_CARDS[name].schema_description


def test_constrained_decision_is_single_flattened_action(built_substrate: Path) -> None:
    substrate = Substrate.open(built_substrate)
    built = PolicyContextBuilder(substrate, interface_contract=get_interface_contract("C1")).build(
        "Question?", SkillDocument.from_text("Answer."), EpisodeState.initial(), [], scope_id="q1")
    model = built.decision_format
    valid = model.model_validate({"supported_facts": [], "missing_information": ["answer"],
                                  "action": {"name": "find_sentences", "query": "specific fact"}})
    decision = policy_decision_from_constrained(valid)
    assert decision.action.target.value == "SENTENCE"
    with pytest.raises(Exception):
        model.model_validate({"assessment": {"supported_facts": [], "missing_information": []},
                              "action": {"name": "find_passages", "query": "fact"}})
    with pytest.raises(Exception):
        model.model_validate({"supported_facts": "[]", "missing_information": [],
                              "action": {"name": "find_passages", "query": "fact"}})
    with pytest.raises(Exception):
        model.model_validate({"supported_facts": [], "missing_information": [],
                              "action": [{"name": "find_passages", "query": "one"},
                                         {"name": "find_passages", "query": "two"}]})


def test_entity_tools_are_created_only_from_visible_entity_cards(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    entity_id = sorted(substrate.entity_ids_by_scope["q1"])[0]
    state = EpisodeState.initial()
    state.visible_entity_ids.add(entity_id)
    state.semantic_memory_node_ids.append(entity_id)
    state.reference_registry.register(entity_id, "ENTITY")
    for name, expected_tool in (
        ("C2", "follow_entity_to_passages"),
        ("C3", "follow_entity_to_sentences"),
        ("C5", "follow_entity_to_passages"),
        ("C4", "follow_entity_to_sentences"),
    ):
        contract = get_interface_contract(name)
        built = PolicyContextBuilder(substrate, interface_contract=contract).build(
            "Question?",
            SkillDocument.from_text("Answer."),
            state,
            [],
            scope_id="q1",
        )
        follow = next(
            item
            for item in built.tool_definitions
            if item["function"]["name"] == expected_tool
        )
        entity_schema = follow["function"]["parameters"]["properties"]["entity_ref"]
        assert "enum" not in entity_schema
        assert entity_schema["type"] == "string"
        assert "query" not in follow["function"]["parameters"]["properties"]
    annotation_only = PolicyContextBuilder(
        substrate, interface_contract=get_interface_contract("A1")
    ).build(
        "Question?",
        SkillDocument.from_text("Answer."),
        state,
        [],
        scope_id="q1",
    )
    assert not any(
        item["function"]["name"].startswith("follow_entity")
        for item in annotation_only.tool_definitions
    )


def test_native_observation_uses_minimal_entity_cards(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    entity_id = sorted(substrate.entity_ids_by_scope["q1"])[0]
    state = EpisodeState.initial()
    state.visible_entity_ids.add(entity_id)
    state.semantic_memory_node_ids.append(entity_id)
    state.reference_registry.register(entity_id, "ENTITY")
    built = PolicyContextBuilder(
        substrate, interface_contract=get_interface_contract("C2")
    ).build(
        "Question?", SkillDocument.from_text("Answer."), state, [], scope_id="q1"
    )
    prompt = "\n".join(message.content or "" for message in built.messages)
    entity = substrate.entity_by_id[entity_id]
    assert f"E1 — {entity.canonical_name}" in prompt
    assert "stable_id" not in prompt
    assert "gold" not in prompt.casefold()


def test_v62_entity_filter_excludes_ner_types_but_keeps_single_entity_hop(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    entity = next(item for item in substrate.entities if item.canonical_name == "Marie Curie")
    # The entity is the only distinct entity in its synthetic sentence in this
    # test; it remains eligible because it has an unseen linked sentence.
    state = EpisodeState.initial()
    state.visible_entity_ids.add(entity.entity_id)
    state.semantic_memory_node_ids.append(entity.entity_id)
    state.reference_registry.register(entity.entity_id, "ENTITY")
    built = PolicyContextBuilder(
        substrate, interface_contract=get_interface_contract("C3")
    ).build("Where was Marie Curie born?", SkillDocument.from_text("Answer."), state, [], scope_id="q1")
    prompt = "\n".join(message.content or "" for message in built.messages)
    assert f"E1 — {entity.canonical_name}" in prompt

    substrate.entity_by_id[entity.entity_id] = Entity(
        entity_id=entity.entity_id,
        canonical_name=entity.canonical_name,
        normalized_name=entity.normalized_name,
        entity_type="DATE",
    )
    filtered = PolicyContextBuilder(
        substrate, interface_contract=get_interface_contract("C3")
    ).build("Where was Marie Curie born?", SkillDocument.from_text("Answer."), state, [], scope_id="q1")
    filtered_prompt = "\n".join(message.content or "" for message in filtered.messages)
    assert "Visible entity references:" not in filtered_prompt


def test_finish_is_always_available_but_empty_evidence_is_explicit(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    contract = get_interface_contract("C0")
    built = PolicyContextBuilder(substrate, interface_contract=contract).build(
        "Question?",
        SkillDocument.from_text("Answer."),
        EpisodeState.initial(),
        [],
        scope_id="q1",
    )
    finish = next(
        item for item in built.tool_definitions if item["function"]["name"] == "finish"
    )
    params = finish["function"]["parameters"]
    assert params["properties"]["evidence_refs"]["maxItems"] == 0
    decision = decision_from_tool_call(
        {
            "id": "call-finish",
            "type": "function",
            "function": {
                "name": "finish",
                "arguments": {"assessment": {"supported_facts": ["The source does not answer the question"], "missing_information": []}, "answer": "Unknown", "evidence_refs": []},
            },
        }
    )
    assert decision.action.type == "FINISH"
    assert decision.action.evidence_refs == []


def test_native_parser_rejects_unknown_or_hidden_references(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    contract = get_interface_contract("C3")
    state = EpisodeState.initial()
    entity_id = sorted(substrate.entity_ids_by_scope["q1"])[0]
    state.visible_entity_ids.add(entity_id)
    state.semantic_memory_node_ids.append(entity_id)
    state.reference_registry.register(entity_id, "ENTITY")
    built = PolicyContextBuilder(substrate, interface_contract=contract).build(
        "Question?", SkillDocument.from_text("Answer."), state, [], scope_id="q1"
    )
    native_tools = build_tool_definitions(built.available_action_space)
    follow = next(
        item for item in native_tools
        if item["function"]["name"] == "follow_entity_to_sentences"
    )
    call = {
        "id": "call-hidden",
        "type": "function",
        "function": {
            "name": "follow_entity_to_sentences",
            "arguments": {
                "assessment": {
                    "supported_facts": [],
                    "missing_information": ["Connected evidence"],
                },
                "entity_ref": "E99",
            },
        },
    }
    try:
        parsed = decision_from_tool_call(call, native_tools)
        resolve_decision(parsed, built.reference_map)
    except (PolicyResponseError, PolicyStateError, ReferenceResolutionError):
        pass
    else:
        raise AssertionError("hidden entity reference must be rejected")
    assert "enum" not in follow["function"]["parameters"]["properties"]["entity_ref"]
