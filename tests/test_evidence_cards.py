from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.models import EpisodeState, ExpansionKind, Observation, ObservationStatus
from agentic_rag.agent.reader import ReaderItem, ReaderResult, ScriptedReader
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.substrate.storage import Substrate


def _state_with_visible_sources(substrate: Substrate) -> EpisodeState:
    state = EpisodeState.initial(max_steps=10, max_policy_attempts=12, max_retrieved_tokens=12000)
    chunk_id = sorted(substrate.chunk_ids_by_scope["q1"])[0]
    sentence_id = substrate.sentences_by_chunk[chunk_id][0].sentence_id
    entity_id = sorted(substrate.entity_ids_by_scope["q1"])[0]
    state.visible_entity_ids.add(entity_id)
    state.visible_sentence_ids.add(sentence_id)
    state.visible_chunk_ids.add(chunk_id)
    state.eligible_sentence_ids.add(sentence_id)
    state.semantic_memory_node_ids.extend([entity_id, sentence_id, chunk_id])
    state.reference_registry.register(entity_id, "ENTITY")
    state.reference_registry.register(sentence_id, "SENTENCE")
    state.reference_registry.register(chunk_id, "CHUNK")
    state.newest_observation = Observation(
        status=ObservationStatus.OK,
        results=[{"text": "The source sentence."}],
        novel_node_ids=[entity_id, sentence_id, chunk_id],
        metadata={
            "visibility_delta": {
                "visible_entity_ids": [entity_id],
                "visible_sentence_ids": [sentence_id],
                "visible_chunk_ids": [chunk_id],
                "eligible_sentence_ids": [sentence_id],
            }
        },
    )
    return state


def test_program_context_without_action_menu_uses_one_card_block(built_substrate):
    substrate = Substrate.open(built_substrate)
    state = _state_with_visible_sources(substrate)
    built = PolicyContextBuilder(
        substrate,
        (ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,),
        show_available_action_options=False,
        observation_mode="program",
    ).build("Which city?", SkillDocument.from_text("Use evidence."), state, [], scope_id="q1")
    prompt = "\n".join(message.content for message in built.messages)
    assert "Evidence cards:" in prompt
    assert "can do:" in prompt
    assert "Currently available action options" not in prompt
    assert "returned_units" not in prompt
    assert "retrieved_tokens" not in prompt
    assert prompt.count("S1") <= 3


def test_program_context_can_add_the_same_centralized_action_menu(
    built_substrate,
):
    substrate = Substrate.open(built_substrate)
    state = _state_with_visible_sources(substrate)
    built = PolicyContextBuilder(
        substrate,
        (ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,),
        show_available_action_options=True,
        observation_mode="program",
    ).build("Which city?", SkillDocument.from_text("Use evidence."), state, [], scope_id="q1")
    prompt = "\n".join(message.content for message in built.messages)
    assert "Evidence cards:" in prompt
    assert "Currently available action options" in prompt
    assert "READ:" in prompt
    assert "FINISH:" in prompt
    assert built.available_action_space.read_refs
    assert built.available_action_space.finish_evidence_refs


def test_reader_items_are_source_grounded_and_rendered_once(built_substrate):
    substrate = Substrate.open(built_substrate)
    state = _state_with_visible_sources(substrate)
    sentence_ref = state.reference_registry.ref_for(
        next(iter(state.visible_sentence_ids)), "SENTENCE"
    )
    reader = ScriptedReader([
        ReaderResult(items=[ReaderItem(
            source_ref=sentence_ref,
            role="direct_answer",
            claim="The source describes the requested fact.",
            quote="The source sentence.",
            confidence="high",
        )])
    ])
    built = PolicyContextBuilder(
        substrate,
        (ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,),
        show_available_action_options=False,
        observation_mode="reader",
        reader=reader,
    ).build("Which city?", SkillDocument.from_text("Use evidence."), state, [], scope_id="q1")
    prompt = "\n".join(message.content for message in built.messages)
    assert "reader summary: The source describes the requested fact." in prompt
    assert "reader quote: The source sentence." in prompt
    # Reader mode is a filter: original sentence/chunk bodies are not
    # included separately (the quote is the only source text shown).
    assert "text:" not in prompt
    assert "preview:" not in prompt
    assert "Currently available action options" not in prompt
    assert built.reader_usage.policy_calls == 1


def test_reader_mode_hides_unselected_sources_but_keeps_full_action_space(
    built_substrate,
):
    substrate = Substrate.open(built_substrate)
    state = _state_with_visible_sources(substrate)
    sentence_id = next(iter(state.visible_sentence_ids))
    sentence_ref = state.reference_registry.ref_for(sentence_id, "SENTENCE")
    reader = ScriptedReader([
        ReaderResult(items=[ReaderItem(
            source_ref=sentence_ref,
            role="direct_answer",
            claim="Only the selected sentence matters here.",
            quote="Selected source quote.",
            confidence="high",
        )])
    ])
    built = PolicyContextBuilder(
        substrate,
        (ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,),
        observation_mode="reader",
        reader=reader,
    ).build("Which city?", SkillDocument.from_text("Use evidence."), state, [], scope_id="q1")
    prompt = "\n".join(message.content for message in built.messages)
    chunk_ref = state.reference_registry.ref_for(
        next(iter(state.visible_chunk_ids)), "CHUNK"
    )
    entity_ref = state.reference_registry.ref_for(
        next(iter(state.visible_entity_ids)), "ENTITY"
    )
    assert "Only the selected sentence matters here." in prompt
    rendered_refs = {
        item["ref"]
        for item in built.rendered_context["evidence_cards"]["cards"]
    }
    assert chunk_ref not in rendered_refs
    assert entity_ref not in rendered_refs
    # Reader filters what the model sees, not what the Controller considers
    # legal.  The complete state-conditioned action space is preserved so a
    # selective Reader cannot accidentally remove all legal actions.
    assert chunk_ref in set(built.available_action_space.read_refs)
    assert sentence_ref in set(built.available_action_space.finish_evidence_refs)
