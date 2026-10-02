from types import SimpleNamespace

from agentic_rag.agent.models import (
    Assessment,
    EpisodeState,
    Observation,
    ObservationStatus,
    ValidationStatus,
    SearchAction,
    SearchMethod,
    SearchTarget,
    action_signature,
)
from agentic_rag.agent.state import StateUpdater
from agentic_rag.agent.state_management import EpisodeStateManager


def test_begin_new_phase_clears_only_current_phase_sources() -> None:
    state = EpisodeState.initial(max_steps=15, max_policy_attempts=12)
    state.phase_index = 2
    state.current_phase_source_keys.update({"passage:C7", "sentence:S14"})
    state.all_source_keys.update({"passage:C1", "passage:C7", "sentence:S14"})
    state.action_signatures.add("find_passages:town")
    state.visible_entity_ids.add("entity:Mary_Town")
    state.remaining_step_budget = 9
    state.consecutive_unresolved_searches = 2
    state.answer_stage_pending = False

    next_state = state.begin_new_phase()

    assert next_state.phase_index == 3
    assert next_state.current_phase_source_keys == set()
    assert next_state.all_source_keys == {
        "passage:C1",
        "passage:C7",
        "sentence:S14",
    }
    assert next_state.action_signatures == {"find_passages:town"}
    assert next_state.visible_entity_ids == {"entity:Mary_Town"}
    assert next_state.remaining_step_budget == 9
    assert next_state.consecutive_unresolved_searches == 0
    assert next_state.answer_stage_pending is False


def test_phase_state_sets_have_stable_json_serialization() -> None:
    state = EpisodeState.initial()
    state.current_phase_source_keys.update({"sentence:S14", "passage:C7"})
    state.all_source_keys.update({"sentence:S14", "passage:C1", "passage:C7"})

    payload = state.model_dump(mode="json")

    assert payload["current_phase_source_keys"] == ["passage:C7", "sentence:S14"]
    assert payload["all_source_keys"] == [
        "passage:C1",
        "passage:C7",
        "sentence:S14",
    ]


def test_resolved_gap_with_new_source_starts_a_fresh_policy_window(built_substrate) -> None:
    from agentic_rag.substrate.storage import Substrate

    substrate = Substrate.open(built_substrate)
    chunk = substrate.chunks[0]
    before = EpisodeState.initial()
    before.last_assessment = Assessment(
        missing_information=["identify the town where the festival was held"]
    )
    before.current_phase_source_keys.add("passage:old")
    before.all_source_keys.add("passage:old")

    updater = StateUpdater(substrate)
    observation = Observation(
        status=ObservationStatus.OK,
        retrieved_tokens=1,
        novel_node_ids=[chunk.chunk_id],
        metadata={"visibility_delta": {"visible_passage_ids": [chunk.chunk_id]}},
    )
    updated = updater.apply(
        before,
        assessment=Assessment(
            resolved_gaps=[
                "identify the town where the festival was held — Mary Town"
            ],
            missing_information=["determine Mary Town's main industry"],
        ),
        observation=observation,
        action_signature="search:town",
    )
    event = SimpleNamespace(
        validation_status=ValidationStatus.VALID,
        observation=observation,
        policy_view=SimpleNamespace(context_audit={"context_mode": "gap_bounded"}),
        assessment=Assessment(
            resolved_gaps=[
                "identify the town where the festival was held — Mary Town"
            ],
            missing_information=["determine Mary Town's main industry"],
        ),
    )
    transitioned, did_transition, reason = EpisodeStateManager._apply_phase_transition(
        before, updated, event
    )

    assert did_transition is True
    assert reason == "resolved_gap_with_new_source"
    assert transitioned.phase_index == 1
    assert transitioned.current_phase_source_keys == {f"passage:{chunk.chunk_id}"}
    assert "passage:old" in transitioned.all_source_keys
    assert f"passage:{chunk.chunk_id}" in transitioned.all_source_keys


def test_phase_reset_does_not_reintroduce_previously_seen_returned_units(built_substrate) -> None:
    from agentic_rag.substrate.storage import Substrate

    substrate = Substrate.open(built_substrate)
    chunks = list(substrate.chunks[:2])
    old_chunk, new_chunk = chunks[0], chunks[1]
    before = EpisodeState.initial()
    before.last_assessment = Assessment(missing_information=["identify the town"])
    before.current_phase_source_keys.add(f"passage:{old_chunk.chunk_id}")
    before.all_source_keys.add(f"passage:{old_chunk.chunk_id}")

    updater = StateUpdater(substrate)
    observation = Observation(
        status=ObservationStatus.OK,
        retrieved_tokens=1,
        novel_node_ids=[new_chunk.chunk_id],
        metadata={"visibility_delta": {"visible_passage_ids": [new_chunk.chunk_id]}},
    )
    updated = updater.apply(
        before,
        assessment=Assessment(
            resolved_gaps=["identify the town — Mary Town"],
            missing_information=["determine Mary Town's industry"],
        ),
        observation=observation,
        action_signature="search:industry",
    )
    # The backend may return a mix of old and new units.  This audit field is
    # intentionally broader than the compact source window.
    observation.metadata["returned_source_keys"] = [
        f"passage:{old_chunk.chunk_id}",
        f"passage:{new_chunk.chunk_id}",
    ]
    event = SimpleNamespace(
        validation_status=ValidationStatus.VALID,
        observation=observation,
        policy_view=SimpleNamespace(context_audit={"context_mode": "gap_bounded"}),
        assessment=Assessment(
            resolved_gaps=["identify the town — Mary Town"],
            missing_information=["determine Mary Town's industry"],
        ),
    )

    transitioned, did_transition, reason = EpisodeStateManager._apply_phase_transition(
        before, updated, event
    )

    assert did_transition is True
    assert reason == "resolved_gap_with_new_source"
    assert transitioned.current_phase_source_keys == {
        f"passage:{new_chunk.chunk_id}"
    }
    assert f"passage:{old_chunk.chunk_id}" in transitioned.all_source_keys


def test_new_source_without_resolved_gap_keeps_current_policy_window(built_substrate) -> None:
    from agentic_rag.substrate.storage import Substrate

    substrate = Substrate.open(built_substrate)
    chunk = substrate.chunks[0]
    before = EpisodeState.initial()
    before.last_assessment = Assessment(missing_information=["identify the town"])
    updater = StateUpdater(substrate)
    observation = Observation(
        status=ObservationStatus.OK,
        retrieved_tokens=1,
        novel_node_ids=[chunk.chunk_id],
        metadata={"visibility_delta": {"visible_passage_ids": [chunk.chunk_id]}},
    )
    updated = updater.apply(
        before,
        assessment=Assessment(missing_information=["identify the town"]),
        observation=observation,
        action_signature="search:town",
    )
    event = SimpleNamespace(
        validation_status=ValidationStatus.VALID,
        observation=observation,
        policy_view=SimpleNamespace(context_audit={"context_mode": "gap_bounded"}),
        assessment=Assessment(missing_information=["identify the town"]),
    )
    transitioned, did_transition, reason = EpisodeStateManager._apply_phase_transition(
        before, updated, event
    )

    assert did_transition is False
    assert reason is None
    assert transitioned.phase_index == 0
    assert transitioned.current_phase_source_keys


def test_three_duplicate_rejections_enter_recovery_without_answer_stage(
    built_substrate,
) -> None:
    from agentic_rag.substrate.storage import Substrate

    substrate = Substrate.open(built_substrate)
    state = EpisodeState.initial()
    state.all_source_keys.add("passage:already-visible")
    action = SearchAction(
        query="same query",
        method=SearchMethod.DENSE,
        target=SearchTarget.CHUNK,
    )
    assessment = Assessment(missing_information=["the remaining fact"])
    updater = StateUpdater(substrate)
    observations = []
    for _ in range(3):
        observation = Observation(
            status=ObservationStatus.DUPLICATE_ACTION,
            action=action,
            error_code="duplicate_action",
        )
        observations.append(observation)
        state = updater.apply(
            state,
            assessment=assessment,
            observation=observation,
                action_signature=action_signature(action),
        )

    assert state.consecutive_duplicate_actions == 3
    assert state.interface_cannot_express_new_route is True
    assert state.recovery_mode is True
    assert action_signature(action) in state.blocked_action_signatures
    event = SimpleNamespace(
        validation_status=ValidationStatus.INVALID,
        observation=observations[-1],
        assessment=assessment,
        policy_view=SimpleNamespace(context_audit={"context_mode": "full"}),
    )
    transitioned, did_transition, reason = EpisodeStateManager._apply_phase_transition(
        EpisodeState.model_validate(state.model_dump()), state, event
    )
    assert did_transition is False
    assert reason is None
    assert transitioned.answer_stage_pending is False


def test_answer_stage_reopens_complete_memory_and_closes_retrieval(built_substrate) -> None:
    from agentic_rag.agent.context import PolicyContextBuilder
    from agentic_rag.agent.interface import get_interface_contract
    from agentic_rag.agent.models import ActionSpaceMode
    from agentic_rag.substrate.storage import Substrate

    substrate = Substrate.open(built_substrate)
    chunks = list(substrate.chunks[:2])
    state = EpisodeState.initial()
    for chunk in chunks:
        state.visible_chunk_ids.add(chunk.chunk_id)
        state.visible_passage_ids.add(chunk.chunk_id)
        state.semantic_memory_node_ids.append(chunk.chunk_id)
        state.reference_registry.register(chunk.chunk_id, "CHUNK")
        state.all_source_keys.add(f"passage:{chunk.chunk_id}")
    state.current_phase_source_keys.add(f"passage:{chunks[-1].chunk_id}")
    state.answer_stage_pending = True

    builder = PolicyContextBuilder(
        substrate,
        interface_contract=get_interface_contract("C0"),
        context_mode="gap_bounded",
    )
    built = builder.build(
        "What happened?",
        "Answer from sources.",
        state,
        [],
        action_space_mode=ActionSpaceMode.ANSWER,
    )
    content = built.messages[-1].content

    assert "ALL SOURCE MEMORY (ANSWER STAGE)" in content
    assert all(chunk.text in content for chunk in chunks)
    # Answer mode reopens the complete episode memory, but the renderer must
    # still emit each canonical passage once (not once in both a current and
    # previous-source section, and not once per nested sentence).
    assert all(content.count(chunk.text) == 1 for chunk in chunks)
    assert "PREVIOUSLY SHOWN SOURCE TEXT" not in content
    assert built.available_action_space.mode is ActionSpaceMode.ANSWER
    assert built.available_action_space.search_options == ()
    assert built.available_action_space.finish_available is True
    assert built.policy_view.context_audit["answer_stage"] is True
    assert built.policy_view.context_audit["assessment_requested"] is False
    assert "resolved_gaps" not in str(built.tool_definitions)
    assert "missing_information" not in str(built.tool_definitions)


def test_gap_bounded_phase_reset_hides_previous_source_until_next_retrieval(built_substrate) -> None:
    from agentic_rag.agent.context import PolicyContextBuilder
    from agentic_rag.agent.interface import get_interface_contract
    from agentic_rag.substrate.storage import Substrate

    substrate = Substrate.open(built_substrate)
    chunk = substrate.chunks[0]
    state = EpisodeState.initial()
    state.visible_chunk_ids.add(chunk.chunk_id)
    state.visible_passage_ids.add(chunk.chunk_id)
    state.semantic_memory_node_ids.append(chunk.chunk_id)
    state.reference_registry.register(chunk.chunk_id, "CHUNK")
    state.all_source_keys.add(f"passage:{chunk.chunk_id}")
    # This is the state immediately after begin_new_phase().
    state.phase_index = 1
    state.current_phase_source_keys.clear()

    built = PolicyContextBuilder(
        substrate,
        interface_contract=get_interface_contract("C0"),
        context_mode="gap_bounded",
    ).build("What happened?", "Answer from sources.", state, [])
    content = built.messages[-1].content

    assert "CURRENT PHASE SOURCE TEXT\nNone." in content
    assert chunk.text not in content
    assert built.available_action_space.search_options
