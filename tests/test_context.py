from __future__ import annotations

from pathlib import Path

from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.models import EpisodeState, ExpansionKind
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.substrate.storage import Substrate


def test_empty_context_is_neutral_and_has_compact_budget(built_substrate: Path) -> None:
    substrate = Substrate.open(built_substrate)
    state = EpisodeState.initial(
        max_steps=10,
        max_policy_attempts=12,
        max_retrieved_tokens=12_000,
    )
    built = PolicyContextBuilder(substrate).build(
        "Where was Marie Curie born?",
        SkillDocument.from_text("Use evidence."),
        state,
        [],
        scope_id="q1",
    )
    prompt = "\n".join(message.content for message in built.messages)
    assert "Original question" in prompt
    assert "Action interface" in prompt
    assert "Current retrieval skill" in prompt
    assert "Currently available action options" in prompt
    options = prompt.split("Currently available action options", 1)[1]
    assert "SEARCH:" in options
    assert "LEXICAL -> ENTITY" in options
    assert "\nEXPAND:\n" not in options
    assert "\nREAD:\n" not in options
    assert "\nFINISH:\n" not in options
    assert '"last_assessment"' in prompt
    assert '"semantic_memory"' in prompt
    assert '"latest_attempt":null' in prompt
    assert '"attempted_actions"' in prompt
    assert "Budget: 10 steps, 12 attempts, 12000 retrieval tokens left" in prompt
    assert "Latest event" not in prompt
    assert "Action Targets" not in prompt
    assert 'source_ref: "E2"' in prompt
    assert 'chunk_ref: "C4"' in prompt
    assert "Never output E#, S#, C#" in prompt
    assert "Recovery mode" not in prompt
    assert built.reference_map.typed_refs == {}


def test_dynamic_options_follow_visible_reference_affordances(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state = EpisodeState.initial()
    chunk_ids = sorted(substrate.chunk_ids_by_scope["q1"])
    assert len(chunk_ids) >= 2
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

    skill = SkillDocument.from_text("Use evidence.")
    enabled = (ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,)
    with_options = PolicyContextBuilder(
        substrate,
        enabled,
        show_available_action_options=True,
    ).build("Question?", skill, state, [], scope_id="q1")
    without_options = PolicyContextBuilder(
        substrate,
        enabled,
        show_available_action_options=False,
    ).build("Question?", skill, state, [], scope_id="q1")

    with_prompt = "\n".join(message.content for message in with_options.messages)
    without_prompt = "\n".join(
        message.content for message in without_options.messages
    )
    options = with_prompt.split("Currently available action options", 1)[1]
    assert "ENTITY_MENTIONED_IN_SENTENCE: source_ref in [E1]" in options
    assert "SENTENCE_MENTIONS_ENTITY" not in options
    assert "chunk_ref in [C1]" in options
    assert "evidence_refs may use any non-empty subset of [S1, C2]" in options
    assert "Currently available action options" not in without_prompt
    assert with_options.policy_view == without_options.policy_view
    assert with_options.messages[1:] == without_options.messages[1:]
    assert (
        with_options.decision_format.model_json_schema()
        == without_options.decision_format.model_json_schema()
    )
