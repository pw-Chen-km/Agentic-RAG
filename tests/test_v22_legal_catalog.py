from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.agent.models import (
    ActionSelection,
    AssessmentStatus,
    ChunkRef,
    ControllerState,
    EvidenceAssessment,
    ExpandAction,
    ExpansionKind,
    FinishAction,
    PolicyDecision,
    ReadAction,
    SearchAction,
    SentenceRef,
    VALID_SEARCH_PAIRS,
    action_signature,
)
from agentic_rag.agent.validator import DecisionValidator
from agentic_rag.storage import Substrate


def _sentence_id(substrate: Substrate, text: str) -> str:
    return next(
        sentence.sentence_id
        for sentence in substrate.sentences
        if sentence.text == text
    )


def _entity_id(substrate: Substrate, canonical_name: str) -> str:
    return next(
        entity.entity_id
        for entity in substrate.entities
        if entity.canonical_name == canonical_name
    )


def _decision(action, *, selected=()) -> PolicyDecision:
    is_finish = isinstance(action, FinishAction)
    return PolicyDecision(
        assessment=EvidenceAssessment(
            status=(
                AssessmentStatus.SUFFICIENT
                if is_finish
                else AssessmentStatus.INSUFFICIENT
            ),
            selected_evidence_refs=list(selected),
        ),
        action=action,
    )


def _register_visible_nodes(
    substrate: Substrate, state: ControllerState
) -> dict[str, str]:
    sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    preview_id = _sentence_id(
        substrate, "Warsaw is the capital of Poland."
    )
    q2_sentence_id = _sentence_id(
        substrate, "Paris is the capital of France."
    )
    entity_id = _entity_id(substrate, "Marie Curie")
    q2_entity_id = _entity_id(substrate, "Paris")
    chunk_id = substrate.chunk_for_sentence(sentence_id).chunk_id
    q2_chunk_id = substrate.chunk_for_sentence(q2_sentence_id).chunk_id

    values = {
        "sentence": sentence_id,
        "preview": preview_id,
        "q2_sentence": q2_sentence_id,
        "entity": entity_id,
        "q2_entity": q2_entity_id,
        "chunk": chunk_id,
        "q2_chunk": q2_chunk_id,
    }
    for key, node_type in (
        ("entity", "ENTITY"),
        ("q2_entity", "ENTITY"),
        ("sentence", "SENTENCE"),
        ("preview", "SENTENCE"),
        ("q2_sentence", "SENTENCE"),
        ("chunk", "CHUNK"),
        ("q2_chunk", "CHUNK"),
    ):
        state.handle_registry.register(values[key], node_type)

    state.visible_entity_ids.update({entity_id, q2_entity_id})
    state.visible_sentence_ids.update(
        {sentence_id, preview_id, q2_sentence_id}
    )
    state.visible_chunk_ids.update({chunk_id, q2_chunk_id})
    state.eligible_sentence_ids.update({sentence_id, q2_sentence_id})
    return values


def _assert_no_stable_ids(
    options: list[dict[str, object]], substrate: Substrate
) -> None:
    serialized = json.dumps(options, ensure_ascii=False)
    for stable_id in (
        *substrate.entity_by_id,
        *substrate.sentence_by_id,
        *substrate.chunk_by_id,
    ):
        assert stable_id not in serialized


def test_search_catalog_exposes_exactly_the_six_model_pairs(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    validator = DecisionValidator(substrate, tuple(ExpansionKind))

    options = validator.legal_action_options(
        "SEARCH", ControllerState.initial(), "q1"
    )

    assert len(options) == 6
    assert {
        (option["method"], option["target"])
        for option in options
    } == {
        (method.value, target.value) for method, target in VALID_SEARCH_PAIRS
    }
    assert all(option["top_k"] == 5 for option in options)
    for option in options:
        decision = _decision(SearchAction(query="Marie Curie", **option))
        assert validator.validate(
            decision, ControllerState.initial(), "q1"
        ).ok


def test_expand_catalog_uses_validator_predicates_and_policy_semantics(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state = ControllerState.initial()
    ids = _register_visible_nodes(substrate, state)
    enabled = (
        ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
        ExpansionKind.SENTENCE_MENTIONS_ENTITY,
        ExpansionKind.ENTITY_CO_OCCURS_ENTITY_SENTENCE,
        ExpansionKind.CHUNK_ADJACENT_CHUNK,
    )
    validator = DecisionValidator(substrate, enabled)

    options = validator.legal_action_options("EXPAND", state, "q1")

    kinds = {option["kind"] for option in options}
    assert kinds == {kind.value for kind in enabled}
    assert not any(
        option["source_id"]
        == state.handle_registry.handle_for(ids["q2_entity"], "ENTITY")
        for option in options
    )
    assert not any(
        option["source_id"]
        == state.handle_registry.handle_for(ids["preview"], "SENTENCE")
        for option in options
        if option["kind"]
        == ExpansionKind.SENTENCE_MENTIONS_ENTITY.value
    )
    entity_option = next(
        option
        for option in options
        if option["kind"]
        == ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE.value
    )
    assert entity_option["source"] == {
        "node_type": "ENTITY",
        "id": entity_option["source_id"],
        "label": "Marie Curie",
        "entity_type": "PERSON",
    }
    adjacency = [
        option
        for option in options
        if option["kind"] == ExpansionKind.CHUNK_ADJACENT_CHUNK.value
    ]
    assert {option["direction"] for option in adjacency} == {
        "PREV",
        "NEXT",
        "BOTH",
    }
    _assert_no_stable_ids(options, substrate)

    for option in options:
        action = ExpandAction(
            kind=option["kind"],
            source_id=option["source_id"],
            direction=option["direction"],
            top_k=option["top_k"],
        )
        assert validator.validate(_decision(action), state, "q1").ok


def test_read_catalog_excludes_read_and_duplicate_chunks(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state = ControllerState.initial()
    ids = _register_visible_nodes(substrate, state)
    validator = DecisionValidator(substrate, tuple(ExpansionKind))
    chunk_handle = state.handle_registry.handle_for(ids["chunk"], "CHUNK")
    q2_chunk_handle = state.handle_registry.handle_for(
        ids["q2_chunk"], "CHUNK"
    )
    assert chunk_handle is not None
    assert q2_chunk_handle is not None

    options = validator.legal_action_options("READ", state, "q1")
    assert [option["chunk_id"] for option in options] == [chunk_handle]
    _assert_no_stable_ids(options, substrate)
    assert validator.validate(
        _decision(ReadAction(chunk_id=chunk_handle)), state, "q1"
    ).ok

    state.action_signatures.add(
        action_signature(ReadAction(chunk_id=ids["chunk"]))
    )
    assert validator.legal_action_options("READ", state, "q1") == []

    state.read_chunk_ids.add(ids["chunk"])
    repeated = validator.validate(
        _decision(ReadAction(chunk_id=chunk_handle)), state, "q1"
    )
    assert not repeated.ok
    assert repeated.code == "chunk_already_read"


def test_stage1_evidence_validation_resolves_handles_and_rejects_preview(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state = ControllerState.initial()
    ids = _register_visible_nodes(substrate, state)
    validator = DecisionValidator(substrate, tuple(ExpansionKind))
    sentence_handle = state.handle_registry.handle_for(
        ids["sentence"], "SENTENCE"
    )
    preview_handle = state.handle_registry.handle_for(
        ids["preview"], "SENTENCE"
    )
    q2_sentence_handle = state.handle_registry.handle_for(
        ids["q2_sentence"], "SENTENCE"
    )
    assert sentence_handle is not None
    assert preview_handle is not None
    assert q2_sentence_handle is not None

    accepted = validator.validate_selected_evidence(
        [SentenceRef(id=sentence_handle)], state, "q1"
    )
    assert accepted.ok
    assert accepted.resolved_refs == (SentenceRef(id=ids["sentence"]),)

    preview = validator.validate_selected_evidence(
        [SentenceRef(id=preview_handle)], state, "q1"
    )
    assert not preview.ok
    assert preview.code == "selected_evidence_not_eligible"
    assert preview_handle in (preview.message or "")
    assert ids["preview"] not in (preview.message or "")

    outside = validator.validate_selected_evidence(
        [SentenceRef(id=q2_sentence_handle)], state, "q1"
    )
    assert not outside.ok
    assert outside.code == "selected_evidence_out_of_scope"

    duplicate = validator.validate_selected_evidence(
        [
            SentenceRef(id=sentence_handle),
            SentenceRef(id=ids["sentence"]),
        ],
        state,
        "q1",
    )
    assert not duplicate.ok
    assert duplicate.code == "selected_evidence_duplicate"

    opaque_intent = validator.validate_action_selection(
        ActionSelection(
            action_type="EXPAND",
            action_intent=f"Expand {sentence_handle} to find more evidence.",
            selected_evidence_refs=[],
        ),
        state,
        "q1",
    )
    assert not opaque_intent.ok
    assert opaque_intent.code == "action_intent_uses_handle"

    semantic_intent = validator.validate_action_selection(
        ActionSelection(
            action_type="EXPAND",
            action_intent=(
                "Expand the Marie Curie birthplace sentence to find its "
                "named entities."
            ),
            selected_evidence_refs=[],
        ),
        state,
        "q1",
    )
    assert semantic_intent.ok

    grounded_handle_intent = validator.validate_action_selection(
        ActionSelection(
            action_type="EXPAND",
            action_intent=(
                f"Sentence {sentence_handle} states that Marie Curie was "
                "born in Warsaw; expand it to find related entities."
            ),
            selected_evidence_refs=[],
        ),
        state,
        "q1",
    )
    assert grounded_handle_intent.ok

    handle_query = _decision(
        SearchAction(
            query=f"Find facts related to {sentence_handle}",
            method="BM25",
            target="SENTENCE",
        )
    )
    assert validator.validate(handle_query, state, "q1").ok
    semantic_field = validator.validate(
        handle_query,
        state,
        "q1",
        enforce_semantic_handles=True,
    )
    assert not semantic_field.ok
    assert semantic_field.code == "semantic_field_uses_handle"


def test_finish_catalog_lists_only_selected_eligible_evidence(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    state = ControllerState.initial()
    ids = _register_visible_nodes(substrate, state)
    state.read_chunk_ids.add(ids["chunk"])
    validator = DecisionValidator(substrate, tuple(ExpansionKind))
    sentence_handle = state.handle_registry.handle_for(
        ids["sentence"], "SENTENCE"
    )
    preview_handle = state.handle_registry.handle_for(
        ids["preview"], "SENTENCE"
    )
    chunk_handle = state.handle_registry.handle_for(ids["chunk"], "CHUNK")
    assert sentence_handle is not None
    assert preview_handle is not None
    assert chunk_handle is not None
    selected = [
        SentenceRef(id=sentence_handle),
        SentenceRef(id=preview_handle),
        ChunkRef(id=chunk_handle),
    ]

    options = validator.legal_action_options(
        "FINISH",
        state,
        "q1",
        selected_evidence_refs=selected,
    )

    refs = [option["evidence_ref"] for option in options]
    assert refs == [
        {"unit": "SENTENCE", "id": sentence_handle},
        {"unit": "CHUNK", "id": chunk_handle},
    ]
    assert options[0]["evidence"]["text"] == (
        "Marie Curie was born in Warsaw."
    )
    assert "text" in options[1]["evidence"]
    _assert_no_stable_ids(options, substrate)

    valid_selection = [
        SentenceRef(id=sentence_handle), ChunkRef(id=chunk_handle)
    ]
    for option in options:
        ref = option["evidence_ref"]
        action_ref = (
            SentenceRef(id=ref["id"])
            if ref["unit"] == "SENTENCE"
            else ChunkRef(id=ref["id"])
        )
        assert validator.validate(
            _decision(
                FinishAction(evidence_refs=[action_ref]),
                selected=valid_selection,
            ),
            state,
            "q1",
        ).ok
