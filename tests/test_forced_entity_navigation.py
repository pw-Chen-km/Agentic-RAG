from __future__ import annotations

from agentic_rag.agent.action_space import AvailableActionSpaceBuilder
from agentic_rag.agent.interface import get_interface_contract
from agentic_rag.agent.models import (
    Assessment,
    ContextNodeReference,
    ContextReferenceMap,
    EpisodeState,
    ExpansionKind,
    Observation,
    ObservationStatus,
    ResolvedFinishAction,
    SearchAction,
    SearchMethod,
    SearchTarget,
)
from agentic_rag.agent.state import StateUpdater
from agentic_rag.substrate.storage import Substrate


def _state_with_visible_entity(built_substrate) -> tuple[EpisodeState, ContextReferenceMap]:
    substrate = Substrate.open(built_substrate)
    entity_id = sorted(substrate.entity_ids_by_scope["q1"])[0]
    state = EpisodeState.initial()
    state.visible_entity_ids.add(entity_id)
    state.semantic_memory_node_ids.append(entity_id)
    state.reference_registry.register(entity_id, "ENTITY")
    ref = state.reference_registry.ref_for(entity_id, "ENTITY")
    assert ref is not None
    return state, ContextReferenceMap(
        typed_refs={
            ref: ContextNodeReference(node_type="ENTITY", stable_id=entity_id)
        }
    )


def test_two_unresolved_searches_expose_only_entity_navigation_when_available(
    built_substrate,
) -> None:
    state, references = _state_with_visible_entity(built_substrate)
    state.last_assessment = Assessment(missing_information=["the connected fact"])
    state.consecutive_unresolved_searches = 2

    contract = get_interface_contract("C4")
    space = AvailableActionSpaceBuilder(
        contract.enabled_expansions, contract
    ).build(state, references)

    assert space.forced_entity_navigation is True
    assert space.search_options == ()
    assert len(space.expand_options) == 1
    assert space.expand_options[0].kind is ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE


def test_c0_and_c1_keep_search_affordances_without_entity_navigation(
    built_substrate,
) -> None:
    state, references = _state_with_visible_entity(built_substrate)
    state.last_assessment = Assessment(missing_information=["the connected fact"])
    state.consecutive_unresolved_searches = 2

    for name in ("C0", "C1"):
        contract = get_interface_contract(name)
        space = AvailableActionSpaceBuilder(
            contract.enabled_expansions, contract
        ).build(state, references)
        assert space.forced_entity_navigation is False
        assert space.search_options
        assert space.expand_options == ()


def test_state_counts_only_executed_unresolved_searches_and_resets_on_other_action(
    built_substrate,
) -> None:
    substrate = Substrate.open(built_substrate)
    updater = StateUpdater(substrate, get_interface_contract("C4"))
    state = EpisodeState.initial()
    search = SearchAction(
        query="Marie Curie",
        method=SearchMethod.DENSE,
        target=SearchTarget.CHUNK,
    )
    assessment = Assessment(missing_information=["the connected fact"])

    for signature, status in (
        ("search-1", ObservationStatus.OK),
        ("search-2-duplicate", ObservationStatus.DUPLICATE_ACTION),
    ):
        state = updater.apply(
            state,
            assessment=assessment,
            observation=Observation(status=status, action=search),
            action_signature=signature,
            scope_id="q1",
        )
    assert state.consecutive_unresolved_searches == 2

    state = updater.apply(
        state,
        assessment=assessment,
        observation=Observation(
            status=ObservationStatus.OK,
            action=ResolvedFinishAction(answer="answer", evidence_refs=[]),
        ),
        action_signature="finish-1",
        scope_id="q1",
    )
    assert state.consecutive_unresolved_searches == 0


def test_forced_navigation_gate_is_recorded_in_policy_context_audit(built_substrate) -> None:
    from agentic_rag.agent.context import PolicyContextBuilder

    substrate = Substrate.open(built_substrate)
    state, _ = _state_with_visible_entity(built_substrate)
    state.last_assessment = Assessment(missing_information=["the connected fact"])
    state.consecutive_unresolved_searches = 2

    built = PolicyContextBuilder(
        substrate,
        interface_contract=get_interface_contract("C4"),
    ).build("What is connected?", "Use the available sources.", state, [], scope_id="q1")

    assert built.available_action_space.forced_entity_navigation is True
    assert built.policy_view.context_audit["consecutive_unresolved_searches"] == 2
    assert built.policy_view.context_audit["information_gap_remaining"] is True
    assert built.policy_view.context_audit["forced_entity_navigation"] is True
    assert built.policy_view.context_audit["available_entity_navigation"] is True
