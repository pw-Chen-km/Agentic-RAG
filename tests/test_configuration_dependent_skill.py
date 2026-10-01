from __future__ import annotations

from pathlib import Path

from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.interface import get_interface_contract
from agentic_rag.agent.models import (
    EpisodeState,
    ExpansionKind,
    ResolvedExpandAction,
)
from agentic_rag.agent.routing import routing_metadata
from agentic_rag.agent.skill import load_skill_version
from agentic_rag.substrate.storage import Substrate


def test_builtin_skill_versions_are_distinct_and_immutable() -> None:
    root = Path(__file__).resolve().parents[1]
    neutral = load_skill_version("neutral", repo_root=root)
    controlled = load_skill_version("configuration-dependent", repo_root=root)

    assert neutral.name == "neutral"
    assert neutral.routing_policy == "neutral"
    assert controlled.name == "configuration-dependent"
    assert controlled.routing_policy == "configuration-dependent"
    assert neutral.document.sha256 != controlled.document.sha256
    assert "There is no required order" in neutral.document.content
    assert "CONFIGURATION-DEPENDENT ROUTING" in controlled.document.content
    assert "entity-follow" in controlled.document.content


def test_controlled_action_guide_changes_policy_wording_only(built_substrate) -> None:
    substrate = Substrate.open(built_substrate)
    contract = get_interface_contract("C4")
    state = EpisodeState.initial()
    neutral = PolicyContextBuilder(
        substrate, interface_contract=contract, routing_policy="neutral"
    ).build("Question?", "Answer.", state, [], scope_id="q1")
    controlled = PolicyContextBuilder(
        substrate,
        interface_contract=contract,
        routing_policy="configuration-dependent",
    ).build("Question?", "Answer.", state, [], scope_id="q1")

    neutral_prompt = neutral.messages[0].content or ""
    controlled_prompt = controlled.messages[0].content or ""
    assert neutral.tool_definitions == controlled.tool_definitions
    assert "do not require an order or prefer an operation" in neutral_prompt
    assert "gives priority to a visible" in controlled_prompt
    assert "gives priority to a visible" not in neutral_prompt


def test_routing_telemetry_does_not_claim_entity_relevance(built_substrate) -> None:
    substrate = Substrate.open(built_substrate)
    contract = get_interface_contract("C4")
    state = EpisodeState.initial()
    built = PolicyContextBuilder(
        substrate,
        interface_contract=contract,
        routing_policy="configuration-dependent",
    ).build("Question?", "Answer.", state, [], scope_id="q1")
    telemetry = routing_metadata(
        policy="configuration-dependent",
        action=None,
        space=built.available_action_space,
        state=state,
    )

    assert telemetry["routing_policy"] == "configuration-dependent"
    assert telemetry["selected_route"] is None
    assert telemetry["route_compliance"] == "undetermined"
    assert telemetry["entity_navigation_available"] is False


def test_routing_telemetry_recognizes_resolved_entity_hop(built_substrate) -> None:
    substrate = Substrate.open(built_substrate)
    contract = get_interface_contract("C4")
    state = EpisodeState.initial()
    builder = PolicyContextBuilder(
        substrate,
        interface_contract=contract,
        routing_policy="configuration-dependent",
    )
    entity_id = sorted(substrate.entity_ids_by_scope["q1"])[0]
    state.visible_entity_ids.add(entity_id)
    state.semantic_memory_node_ids.append(entity_id)
    state.reference_registry.register(entity_id, "ENTITY")
    built = builder.build("Question?", "Answer.", state, [], scope_id="q1")
    entity_ref = next(iter(built.available_action_space.expand_options[0].source_refs))
    action = ResolvedExpandAction(
        kind=ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
        source_id=entity_id,
    )
    telemetry = routing_metadata(
        policy="configuration-dependent",
        action=action,
        space=built.available_action_space,
        state=state,
    )

    assert entity_ref.startswith("E")
    assert telemetry["selected_route"] == "entity_navigation"
    assert telemetry["selected_entity_ref"] == entity_id
    assert telemetry["route_compliance"] == "local_selected"
