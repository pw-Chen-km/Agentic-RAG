from __future__ import annotations

from pathlib import Path

from agentic_rag.agent.action_space import AvailableActionSpaceBuilder
from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.interface import get_interface_contract
from agentic_rag.agent.models import EpisodeState, Observation, ObservationStatus
from agentic_rag.agent.observation_projection import ObservationProjector
from agentic_rag.agent.state import StateUpdater
from agentic_rag.substrate.storage import Substrate


def test_interface_capability_isolation(built_substrate: Path) -> None:
    substrate = Substrate.open(built_substrate)
    state = EpisodeState.initial()
    for name, expected_pairs, expected_expansions in (
        ("C0", {"DENSE->CHUNK"}, []),
        ("C1", {"DENSE->CHUNK", "DENSE->SENTENCE"}, []),
        ("C2", {"DENSE->CHUNK"}, ["ENTITY_MENTIONED_IN_CHUNK"]),
        ("C3", {"DENSE->CHUNK"}, ["ENTITY_MENTIONED_IN_SENTENCE"]),
        ("C5", {"DENSE->CHUNK", "DENSE->SENTENCE"}, ["ENTITY_MENTIONED_IN_CHUNK"]),
        ("C4", {"DENSE->CHUNK", "DENSE->SENTENCE"}, ["ENTITY_MENTIONED_IN_SENTENCE"]),
        ("A1", {"DENSE->CHUNK", "DENSE->SENTENCE"}, []),
    ):
        contract = get_interface_contract(name)
        space = AvailableActionSpaceBuilder((), contract).build(
            state, PolicyContextBuilder(substrate, interface_contract=contract).build(
                "question", "skill", state, [], scope_id="q1"
            ).reference_map
        )
        assert {f"{item.method.value}->{item.target.value}" for item in space.search_options} == expected_pairs
        assert [item.value for item in contract.enabled_expansions] == expected_expansions


def test_hidden_and_cut_through_mentions_never_become_visible(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    mention = substrate.mentions[0]
    sentence = substrate.sentence_by_id[mention.sentence_id]
    projector = ObservationProjector(substrate, expose_entities=True)
    partial = sentence.text[: max(0, mention.mention_end - 1)]
    _, delta_partial, audit_partial = projector.project(
        [{"target": "SENTENCE", "sentence_id": sentence.sentence_id, "text": partial}],
        action_type="SEARCH",
    )
    assert mention.entity_id not in delta_partial["visible_entity_ids"]
    assert any(item["visible"] is False for item in audit_partial["visible_source_spans"])

    _, delta_full, _ = projector.project(
        [{"target": "SENTENCE", "sentence_id": sentence.sentence_id, "text": sentence.text}],
        action_type="SEARCH",
    )
    assert mention.entity_id in delta_full["visible_entity_ids"]


def test_invalid_decision_consumes_normal_budget() -> None:
    state = EpisodeState.initial(max_steps=15, max_policy_attempts=15)
    updated = StateUpdater.__new__(StateUpdater)
    # StateUpdater.apply only needs the substrate for reference bookkeeping;
    # an empty object is sufficient when the observation has no visibility.
    updated.substrate = None
    result = updated.apply(
        state,
        assessment=None,
        observation=Observation(status=ObservationStatus.INVALID_ACTION),
        action_signature=None,
        commit_assessment=False,
        consume_step=True,
    )
    assert result.policy_attempts == 1
    assert result.step == 1
    assert result.remaining_policy_attempt_budget == 14
