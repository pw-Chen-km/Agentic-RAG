from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.context import (
    AppendOnlyContextBuilder,
    PolicyContextBuilder,
)
from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.expansion import ExpansionEngine
from agentic_rag.agent.models import (
    AssessmentStatus,
    ChunkHandle,
    ContextMode,
    ControllerState,
    EntityHandle,
    EvidenceAssessment,
    Observation,
    ObservationStatus,
    PolicyDecision,
    ReadAction,
    SearchAction,
    SentenceHandle,
    SentenceRef,
    StepRecord,
    ValidationStatus,
)
from agentic_rag.agent.router import ActionRouter
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.state import StateUpdater, _summary
from agentic_rag.retrieval import Retriever
from agentic_rag.storage import Substrate


QUESTION = "Where was Marie Curie born?"
SCOPE_ID = "q1"
OLD_RESULT_SENTINEL = "OLD RESULT MUST NOT BE REPLAYED"


def _skill() -> SkillDocument:
    return SkillDocument.from_text("# Strategy\nSearch with the full question.")


def _substrate_nodes(
    substrate: Substrate,
) -> tuple[Any, Any, Any, Any]:
    sentence = next(
        item
        for item in substrate.sentences
        if item.text == "Marie Curie was born in Warsaw."
    )
    chunk = substrate.chunk_by_id[sentence.chunk_id]
    document = substrate.document_by_id[chunk.doc_id]
    entity = next(
        item
        for item in substrate.entities
        if item.canonical_name == "Marie Curie"
    )
    return sentence, chunk, document, entity


def _sentence_observation(substrate: Substrate) -> Observation:
    sentence, chunk, document, _ = _substrate_nodes(substrate)
    action = SearchAction(
        query=QUESTION,
        method="BM25",
        target="SENTENCE",
    )
    return Observation(
        action_id="step-1",
        status=ObservationStatus.OK,
        action=action,
        results=[
            {
                "target": "SENTENCE",
                "sentence_id": sentence.sentence_id,
                "text": sentence.text,
                "parent_chunk_id": chunk.chunk_id,
                "document_id": document.doc_id,
                "title": document.title,
                "score": 1.0,
            }
        ],
        retrieved_tokens=8,
        novel_node_ids=[sentence.sentence_id, chunk.chunk_id],
        metadata={
            "method": "BM25",
            "target": "SENTENCE",
            "query_used": QUESTION,
            "visibility_delta": {
                "visible_entity_ids": [],
                "visible_sentence_ids": [sentence.sentence_id],
                "visible_chunk_ids": [chunk.chunk_id],
                "eligible_sentence_ids": [sentence.sentence_id],
                "read_chunk_ids": [],
            },
            "truncated_by_retrieved_token_budget": False,
        },
    )


def _old_history_record() -> tuple[StepRecord, ControllerState]:
    initial = ControllerState.initial(
        max_steps=4, max_retrieved_tokens=100
    )
    decision = PolicyDecision(
        assessment=EvidenceAssessment(
            status=AssessmentStatus.INSUFFICIENT,
            missing_information=["The birthplace is unknown."],
        ),
        action=SearchAction(
            query=QUESTION,
            method="BM25",
            target="SENTENCE",
        ),
    )
    observation = Observation(
        action_id="step-1",
        status=ObservationStatus.OK,
        action=decision.action,
        results=[
            {
                "sentence_id": "sentence:old",
                "text": OLD_RESULT_SENTINEL,
                "parent_chunk_id": "chunk:old",
                "document_id": "document:old",
                "title": "Old",
            }
        ],
        retrieved_tokens=7,
        novel_node_ids=["sentence:old", "chunk:old"],
        metadata={
            "visibility_delta": {
                "visible_entity_ids": [],
                "visible_sentence_ids": ["sentence:old"],
                "visible_chunk_ids": ["chunk:old"],
                "eligible_sentence_ids": ["sentence:old"],
                "read_chunk_ids": [],
            }
        },
    )
    after = initial.model_copy(
        update={
            "step": 1,
            "remaining_step_budget": 3,
            "remaining_retrieved_token_budget": 93,
            "last_assessment": decision.assessment,
            # Deliberately no newest observation: the compact view must summarize
            # the attempted action without replaying this historical payload.
            "newest_observation": None,
        },
        deep=True,
    )
    return (
        StepRecord(
            step=1,
            decision=decision,
            validation_status=ValidationStatus.VALID,
            observation=observation,
            state_before=initial,
            state_after=after,
        ),
        after,
    )


def _payload(builder_result: Any) -> dict[str, Any]:
    return json.loads(builder_result.messages[-1].content)


def test_handle_summary_is_deterministically_limited_to_160_characters() -> None:
    text = "x" * 200
    summarized = _summary(text)
    assert len(summarized) == 160
    assert summarized == ("x" * 159) + "…"
    assert _summary("short") == "short"


def test_default_compact_context_has_four_messages_and_no_history_payload(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    record, state = _old_history_record()
    builder = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate)
    )

    built = builder.build(
        QUESTION,
        _skill(),
        state,
        [record],
        scope_id=SCOPE_ID,
    )
    payload = _payload(built)
    serialized = json.dumps(payload, sort_keys=True)

    assert AgentConfig().context_mode is ContextMode.COMPACT_EVIDENCE
    assert builder.context_mode is ContextMode.COMPACT_EVIDENCE
    assert [message.role for message in built.messages] == [
        "system",
        "system",
        "user",
        "user",
    ]
    assert len(built.messages) == 4
    assert set(payload) == {"instruction", "policy_state"}
    assert set(payload["policy_state"]) == {
        "step",
        "last_valid_assessment",
        "selected_evidence",
        "latest_observation",
        "actionable_handles",
        "attempted_actions",
        "budget",
    }
    assert payload["policy_state"]["last_valid_assessment"] == {
        "status": "INSUFFICIENT",
        "missing_information": ["The birthplace is unknown."],
    }
    assert OLD_RESULT_SENTINEL not in serialized
    for excluded in (
        "policy_decision",
        "environment_observation",
        "state_before",
        "state_after",
        "usage",
        "visibility_delta",
        "action_signatures",
    ):
        assert excluded not in serialized


def test_selected_sentence_is_resolved_with_full_text_and_removed_from_handles(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence, chunk, _, _ = _substrate_nodes(substrate)
    updater = StateUpdater(substrate)
    state = updater.apply(
        ControllerState.initial(),
        assessment=None,
        observation=_sentence_observation(substrate),
        action_signature=None,
    )
    sentence_handle = state.node_handles[sentence.sentence_id]
    assert isinstance(sentence_handle, SentenceHandle)
    state.node_handles[sentence.sentence_id] = sentence_handle.model_copy(
        update={"text": "intentionally compact handle text"}
    )
    state.selected_evidence_refs = [
        SentenceRef(id=sentence.sentence_id)
    ]
    state.newest_observation = None

    view = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate)
    ).build_policy_view(state, [], scope_id=SCOPE_ID)

    assert len(view.policy_state.selected_evidence) == 1
    selected = view.policy_state.selected_evidence[0]
    assert selected.ref == SentenceRef(id=sentence.sentence_id)
    assert selected.text == sentence.text
    assert selected.text != "intentionally compact handle text"
    assert selected.parent_chunk_id == chunk.chunk_id
    handle_ids = {
        handle.id for handle in view.policy_state.actionable_handles
    }
    assert sentence.sentence_id not in handle_ids
    assert chunk.chunk_id not in handle_ids


def test_sentence_projection_exposes_system_derived_evidence_usability(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence, _, _, _ = _substrate_nodes(substrate)
    raw_observation = _sentence_observation(substrate)
    state = StateUpdater(substrate).apply(
        ControllerState.initial(),
        assessment=None,
        observation=raw_observation,
        action_signature=None,
    )

    built = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate)
    ).build(QUESTION, _skill(), state, [], scope_id=SCOPE_ID)
    projected_result = _payload(built)["policy_state"][
        "latest_observation"
    ]["results"][0]

    assert "can_use_as_evidence" not in raw_observation.results[0]
    assert projected_result["sentence_id"] == sentence.sentence_id
    assert projected_result["can_use_as_evidence"] is True
    assert "eligible_sentence_ids" not in json.dumps(
        _payload(built), sort_keys=True
    )


def test_read_projection_omits_duplicate_chunk_text_but_keeps_sentences(
    built_substrate: Path,
    fake_embedder: Any,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence, chunk, _, _ = _substrate_nodes(substrate)
    router = ActionRouter(
        substrate,
        Retriever(substrate, embedding_backend=fake_embedder),
        ExpansionEngine(substrate, embedding_backend=fake_embedder),
    )
    before = ControllerState.initial()
    before.visible_chunk_ids.add(chunk.chunk_id)
    raw_observation = router.execute(
        ReadAction(chunk_id=chunk.chunk_id),
        before,
        question=QUESTION,
        scope_id=SCOPE_ID,
        action_id="step-read",
    )
    state = StateUpdater(substrate).apply(
        before,
        assessment=None,
        observation=raw_observation,
        action_signature=None,
    )

    built = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate)
    ).build(QUESTION, _skill(), state, [], scope_id=SCOPE_ID)
    projected_result = _payload(built)["policy_state"][
        "latest_observation"
    ]["results"][0]

    assert raw_observation.results[0]["text"] == chunk.text
    assert "text" not in projected_result
    assert projected_result["chunk_id"] == chunk.chunk_id
    assert projected_result["read"] is True
    assert projected_result["can_use_as_evidence"] is True
    assert projected_result["sentences"]
    assert all(
        item["text"] for item in projected_result["sentences"]
    )
    assert all(
        item["can_use_as_evidence"] is True
        for item in projected_result["sentences"]
    )
    assert sentence.sentence_id in {
        item["sentence_id"] for item in projected_result["sentences"]
    }


def test_attempted_actions_are_summaries_without_historical_results(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    record, state = _old_history_record()

    payload = _payload(
        PolicyContextBuilder(
            evidence_resolver=EvidenceResolver(substrate)
        ).build(
            QUESTION,
            _skill(),
            state,
            [record],
            scope_id=SCOPE_ID,
        )
    )
    attempts = payload["policy_state"]["attempted_actions"]

    assert attempts == [
        {
            "step": 1,
            "action": {
                "type": "SEARCH",
                "query": QUESTION,
                "method": "BM25",
                "target": "SENTENCE",
                "top_k": 5,
            },
            "validation_status": "valid",
            "observation_status": "ok",
            "error_code": None,
            "message": None,
            "new_node_count": 2,
            "retrieved_tokens": 7,
        }
    ]
    serialized_attempts = json.dumps(attempts, sort_keys=True)
    assert '"results"' not in serialized_attempts
    assert OLD_RESULT_SENTINEL not in serialized_attempts


def test_handle_registry_keeps_semantic_entity_sentence_and_chunk_handles(
    built_substrate: Path,
    fake_embedder: Any,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence, chunk, document, entity = _substrate_nodes(substrate)
    updater = StateUpdater(substrate)
    state = updater.apply(
        ControllerState.initial(),
        assessment=None,
        observation=_sentence_observation(substrate),
        action_signature=None,
    )
    router = ActionRouter(
        substrate,
        Retriever(substrate, embedding_backend=fake_embedder),
        ExpansionEngine(substrate, embedding_backend=fake_embedder),
    )
    entity_observation = router.execute(
        SearchAction(
            query="Marie Curie",
            method="LEXICAL",
            target="ENTITY",
        ),
        state,
        question=QUESTION,
        scope_id=SCOPE_ID,
        action_id="step-entity",
    )
    state = updater.apply(
        state,
        assessment=None,
        observation=entity_observation,
        action_signature=None,
    )

    sentence_handle = state.node_handles[sentence.sentence_id]
    chunk_handle = state.node_handles[chunk.chunk_id]
    entity_handle = state.node_handles[entity.entity_id]
    assert isinstance(sentence_handle, SentenceHandle)
    assert sentence_handle.text == sentence.text
    assert sentence_handle.parent_chunk_id == chunk.chunk_id
    assert sentence_handle.document_id == document.doc_id
    assert sentence_handle.eligible is True
    assert sentence_handle.can_use_as_evidence is True
    assert isinstance(chunk_handle, ChunkHandle)
    assert chunk_handle.document_id == document.doc_id
    assert chunk_handle.read is False
    assert chunk_handle.can_use_as_evidence is False
    assert isinstance(entity_handle, EntityHandle)
    assert entity_handle.label == "Marie Curie"
    assert entity_handle.entity_type == "PERSON"

    # A result-less latest error represents no nodes, so every persistent
    # semantic handle remains actionable in the compact view.
    state.newest_observation = Observation(
        status=ObservationStatus.ERROR,
        action=SearchAction(
            query="What else should be retrieved?",
            method="BM25",
            target="SENTENCE",
        ),
        error_code="backend_error",
        message="temporary failure",
    )
    view = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate)
    ).build_policy_view(state, [], scope_id=SCOPE_ID)
    handles = {
        handle.id: handle
        for handle in view.policy_state.actionable_handles
    }
    assert sentence.sentence_id in handles
    assert chunk.chunk_id in handles
    assert entity.entity_id in handles


def test_chunk_preview_handles_remain_ineligible_and_read_upgrades_chunk(
    built_substrate: Path,
    fake_embedder: Any,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence, chunk, _, _ = _substrate_nodes(substrate)
    router = ActionRouter(
        substrate,
        Retriever(substrate, embedding_backend=fake_embedder),
        ExpansionEngine(substrate, embedding_backend=fake_embedder),
    )
    updater = StateUpdater(substrate)
    state = ControllerState.initial()
    search_observation = router.execute(
        SearchAction(
            query=QUESTION,
            method="BM25",
            target="CHUNK",
        ),
        state,
        question=QUESTION,
        scope_id=SCOPE_ID,
        action_id="step-chunk-search",
    )
    state = updater.apply(
        state,
        assessment=None,
        observation=search_observation,
        action_signature=None,
    )
    chunk_handle = state.node_handles[chunk.chunk_id]
    assert isinstance(chunk_handle, ChunkHandle)
    assert chunk_handle.read is False
    assert chunk_handle.can_use_as_evidence is False
    assert 1 <= len(chunk_handle.previews) <= 2
    preview_id = chunk_handle.previews[0].sentence_id
    preview_handle = state.node_handles[preview_id]
    assert isinstance(preview_handle, SentenceHandle)
    assert preview_handle.eligible is False
    assert preview_handle.can_use_as_evidence is False

    search_payload = _payload(
        PolicyContextBuilder(
            evidence_resolver=EvidenceResolver(substrate)
        ).build(QUESTION, _skill(), state, [], scope_id=SCOPE_ID)
    )
    projected_chunk = next(
        item
        for item in search_payload["policy_state"][
            "latest_observation"
        ]["results"]
        if item["chunk_id"] == chunk.chunk_id
    )
    assert projected_chunk["read"] is False
    assert projected_chunk["can_use_as_evidence"] is False
    assert all(
        preview["can_use_as_evidence"] is False
        for preview in projected_chunk["previews"]
    )
    assert "evidence_eligible" not in json.dumps(
        projected_chunk, sort_keys=True
    )

    read_observation = router.execute(
        ReadAction(chunk_id=chunk.chunk_id),
        state,
        question=QUESTION,
        scope_id=SCOPE_ID,
        action_id="step-read",
    )
    state = updater.apply(
        state,
        assessment=None,
        observation=read_observation,
        action_signature=None,
    )
    upgraded = state.node_handles[chunk.chunk_id]
    assert isinstance(upgraded, ChunkHandle)
    assert upgraded.read is True
    assert upgraded.can_use_as_evidence is True
    assert upgraded.previews == chunk_handle.previews
    assert isinstance(
        state.node_handles[sentence.sentence_id], SentenceHandle
    )
    assert state.node_handles[sentence.sentence_id].eligible is True


def test_append_only_mode_preserves_full_history_baseline(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    record, state = _old_history_record()
    builder = AppendOnlyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate)
    )

    built = builder.build(
        QUESTION,
        _skill(),
        state,
        [record],
        scope_id=SCOPE_ID,
    )

    assert built.policy_view.context_mode is ContextMode.APPEND_ONLY
    assert [message.role for message in built.messages] == [
        "system",
        "system",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert len(built.messages) == 6
    assert OLD_RESULT_SENTINEL in built.messages[4].content
    final_payload = json.loads(built.messages[-1].content)
    assert final_payload["instruction"] == "Produce the next PolicyDecision."
    assert final_payload["current_state"]["remaining_step_budget"] == 3
    assert "policy_state" not in final_payload

    compact = PolicyContextBuilder(
        evidence_resolver=EvidenceResolver(substrate)
    ).build(
        QUESTION,
        _skill(),
        state,
        [record],
        scope_id=SCOPE_ID,
    )
    assert sum(len(item.content) for item in compact.messages) < sum(
        len(item.content) for item in built.messages
    )
