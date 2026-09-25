from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.interface import get_interface_contract
from agentic_rag.agent.models import EpisodeState
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.substrate.storage import Substrate


EXPECTED_INITIAL = {
    "C0": {"find_passages", "finish"},
    "C1": {"find_passages", "find_sentences", "finish"},
    "C2": {"find_passages", "finish"},
    "C3": {"find_passages", "finish"},
    "C5": {"find_passages", "find_sentences", "finish"},
    "C4": {"find_passages", "find_sentences", "finish"},
    "A1": {"find_passages", "find_sentences", "finish"},
}


def _action_names_in_prompt(prompt: str) -> list[str]:
    action_section = prompt.split("OPERATIONS AVAILABLE NOW\n\n", 1)[1].split(
        "\n\nREFERENCE RULES", 1
    )[0]
    return re.findall(r"(?m)^Operation: (find_passages|find_sentences|follow_entity_to_passages|follow_entity_to_sentences|finish)\(", action_section)


def _action_names_in_schema(schema: dict) -> set[str]:
    return {
        value["properties"]["name"]["const"]
        for value in schema["$defs"].values()
        if "name" in value.get("properties", {})
    }


@pytest.mark.parametrize("name", list(EXPECTED_INITIAL))
def test_initial_action_guide_matches_schema_without_backend_result_count(
    name: str, built_substrate: Path
) -> None:
    substrate = Substrate.open(built_substrate)
    contract = get_interface_contract(name)
    built = PolicyContextBuilder(substrate, interface_contract=contract).build(
        "Question?",
        SkillDocument.from_text("TASK\n\nAnswer the question.\n\nPOLICY\n\nNo action is preferred."),
        EpisodeState.initial(),
        [],
        scope_id="q1",
    )
    prompt = built.messages[0].content or ""
    names = _action_names_in_prompt(prompt)
    assert set(names) == EXPECTED_INITIAL[name]
    assert set(names) == _action_names_in_schema(built.decision_schema)
    assert names == [
        action for action in ("find_passages", "find_sentences", "finish")
        if action in EXPECTED_INITIAL[name]
    ]
    assert "up to five" not in prompt
    assert "top_k" not in prompt
    assert "DENSE" not in prompt
    assert "HOW THE SEARCH OPERATIONS WORK" in prompt
    if name == "C0":
        assert "stored vectors for sentences" not in prompt
    assert "Search scope: all passages in the collection." in prompt
    assert "Search scope: all sentences in the collection." not in prompt if name == "C0" else True
    assert "The order of this list has no meaning." in prompt
    assert contract.compile()["protocol"] == contract.protocol


@pytest.mark.parametrize(
    ("name", "expected_follow"),
    [
        ("C2", "follow_entity_to_passages"),
        ("C3", "follow_entity_to_sentences"),
        ("C5", "follow_entity_to_passages"),
        ("C4", "follow_entity_to_sentences"),
        ("A1", None),
    ],
)
def test_entity_guide_explains_navigation_only_where_available(
    name: str, expected_follow: str | None, built_substrate: Path
) -> None:
    substrate = Substrate.open(built_substrate)
    entity_id = sorted(substrate.entity_ids_by_scope["q1"])[0]
    state = EpisodeState.initial()
    state.visible_entity_ids.add(entity_id)
    state.semantic_memory_node_ids.append(entity_id)
    state.reference_registry.register(entity_id, "ENTITY")
    built = PolicyContextBuilder(
        substrate, interface_contract=get_interface_contract(name)
    ).build("Question?", SkillDocument.from_text("Answer."), state, [], scope_id="q1")
    prompt = built.messages[0].content or ""
    names = _action_names_in_prompt(prompt)
    assert set(names) == _action_names_in_schema(built.decision_schema)
    if expected_follow:
        assert expected_follow in names
        assert "follows stored links" in prompt
        assert "unlinked" in prompt
        assert "using the original question" in prompt
        assert "query is null" not in prompt
    else:
        assert not any(action.startswith("follow_entity_") for action in names)
        assert "E# labels are annotations only; no action can follow them." in prompt


def test_assessment_off_guide_matches_schema(built_substrate: Path) -> None:
    substrate = Substrate.open(built_substrate)
    built = PolicyContextBuilder(
        substrate,
        interface_contract=get_interface_contract("C0"),
        require_evidence_assessment=False,
    ).build("Question?", SkillDocument.from_text("Answer."), EpisodeState.initial(), [], scope_id="q1")
    prompt = built.messages[0].content or ""
    assert "supported_facts" not in prompt
    assert "missing_information" not in prompt
    assert set(built.decision_schema["required"]) == {"action"}
