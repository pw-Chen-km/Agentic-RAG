from __future__ import annotations

from pathlib import Path

from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.expansion import ExpansionEngine
from agentic_rag.agent.models import (
    AssessmentStatus,
    ChunkRef,
    ControllerState,
    EvidenceAssessment,
    ExpandAction,
    ExpansionKind,
    FinishAction,
    PolicyDecision,
    ReadAction,
    SentenceRef,
)
from agentic_rag.agent.router import ActionRouter
from agentic_rag.agent.repair import DecisionRepairer
from agentic_rag.agent.validator import DecisionValidator
from agentic_rag.retrieval import Retriever
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


def _finish_decision(sentence_id: str) -> PolicyDecision:
    ref = SentenceRef(id=sentence_id)
    return PolicyDecision(
        assessment=EvidenceAssessment(
            status=AssessmentStatus.SUFFICIENT,
            selected_evidence_refs=[ref],
        ),
        action=FinishAction(evidence_refs=[ref]),
    )


def test_validator_resolves_finish_handles_before_internal_rules(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    chunk_id = substrate.chunk_for_sentence(sentence_id).chunk_id
    state = ControllerState.initial()
    sentence_handle = state.handle_registry.register(
        sentence_id, "SENTENCE"
    )
    state.handle_registry.register(chunk_id, "CHUNK")
    state.visible_sentence_ids.add(sentence_id)
    state.eligible_sentence_ids.add(sentence_id)
    state.visible_chunk_ids.add(chunk_id)

    result = DecisionValidator(
        substrate, tuple(ExpansionKind)
    ).validate(_finish_decision(sentence_handle), state, "q1")

    assert result.ok
    assert result.resolved_decision is not None
    assert result.resolved_decision.assessment.selected_evidence_refs == [
        SentenceRef(id=sentence_id)
    ]
    assert result.resolved_decision.action == FinishAction(
        evidence_refs=[SentenceRef(id=sentence_id)]
    )


def test_validator_returns_typed_handle_errors(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    state = ControllerState.initial()
    sentence_handle = state.handle_registry.register(
        sentence_id, "SENTENCE"
    )
    validator = DecisionValidator(substrate, tuple(ExpansionKind))

    mismatch = validator.validate(
        PolicyDecision(
            assessment=EvidenceAssessment(
                status=AssessmentStatus.INSUFFICIENT
            ),
            action=ReadAction(chunk_id=sentence_handle),
        ),
        state,
        "q1",
    )
    unknown = validator.validate(
        PolicyDecision(
            assessment=EvidenceAssessment(
                status=AssessmentStatus.INSUFFICIENT
            ),
            action=ReadAction(chunk_id="C999"),
        ),
        state,
        "q1",
    )

    assert not mismatch.ok
    assert mismatch.code == "handle_type_mismatch"
    assert "SENTENCE" in (mismatch.message or "")
    assert "CHUNK" in (mismatch.message or "")
    assert mismatch.resolved_decision is None
    assert not unknown.ok
    assert unknown.code == "unknown_handle"


def test_repairer_flips_only_registered_unambiguous_inverse_expansion(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    state = ControllerState.initial()
    sentence_handle = state.handle_registry.register(
        sentence_id, "SENTENCE"
    )
    state.visible_sentence_ids.add(sentence_id)
    state.eligible_sentence_ids.add(sentence_id)
    raw = PolicyDecision(
        assessment=EvidenceAssessment(
            status=AssessmentStatus.INSUFFICIENT
        ),
        action=ExpandAction(
            kind=ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
            source_id=sentence_handle,
        ),
    )
    repairer = DecisionRepairer(tuple(ExpansionKind))

    repaired = repairer.repair(raw, state)

    assert repaired.code == "expand_direction_repaired_from_handle_type"
    assert repaired.decision.action == ExpandAction(
        kind=ExpansionKind.SENTENCE_MENTIONS_ENTITY,
        source_id=sentence_handle,
    )
    assert raw.action.kind is ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE
    assert DecisionValidator(substrate, tuple(ExpansionKind)).validate(
        repaired.decision, state, "q1"
    ).ok

    unknown = raw.model_copy(
        update={"action": raw.action.model_copy(update={"source_id": "S999"})}
    )
    untouched = repairer.repair(unknown, state)
    assert untouched.code is None
    assert untouched.decision == unknown


def test_router_and_evidence_resolver_accept_handles_defensively(
    built_substrate: Path,
    fake_embedder,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    chunk_id = substrate.chunk_for_sentence(sentence_id).chunk_id
    state = ControllerState.initial()
    sentence_handle = state.handle_registry.register(
        sentence_id, "SENTENCE"
    )
    chunk_handle = state.handle_registry.register(chunk_id, "CHUNK")
    state.visible_sentence_ids.add(sentence_id)
    state.eligible_sentence_ids.add(sentence_id)
    state.visible_chunk_ids.add(chunk_id)
    state.read_chunk_ids.add(chunk_id)
    router = ActionRouter(
        substrate,
        Retriever(substrate, embedding_backend=fake_embedder),
        ExpansionEngine(substrate, embedding_backend=fake_embedder),
    )

    observation = router.execute(
        ReadAction(chunk_id=chunk_handle),
        state,
        question="Where was Marie Curie born?",
        scope_id="q1",
        action_id="read-handle",
    )
    evidence = EvidenceResolver(substrate).resolve(
        [SentenceRef(id=sentence_handle)], state, "q1"
    )

    assert observation.action == ReadAction(chunk_id=chunk_id)
    assert observation.results[0]["chunk_id"] == chunk_id
    assert evidence[0].ref == SentenceRef(id=sentence_id)
    assert evidence[0].text == "Marie Curie was born in Warsaw."


def test_router_resolves_expand_source_handle(
    built_substrate: Path,
    fake_embedder,
) -> None:
    substrate = Substrate.open(built_substrate)
    entity_id = _entity_id(substrate, "Marie Curie")
    state = ControllerState.initial()
    entity_handle = state.handle_registry.register(entity_id, "ENTITY")
    state.visible_entity_ids.add(entity_id)
    router = ActionRouter(
        substrate,
        Retriever(substrate, embedding_backend=fake_embedder),
        ExpansionEngine(substrate, embedding_backend=fake_embedder),
    )

    observation = router.execute(
        ExpandAction(
            kind=ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
            source_id=entity_handle,
        ),
        state,
        question="Where was Marie Curie born?",
        scope_id="q1",
        action_id="expand-handle",
    )

    assert isinstance(observation.action, ExpandAction)
    assert observation.action.source_id == entity_id
    assert observation.status.value == "ok"
    assert observation.results


def test_chunk_handle_evidence_resolves_only_after_read(
    built_substrate: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence_id = _sentence_id(
        substrate, "Marie Curie was born in Warsaw."
    )
    chunk_id = substrate.chunk_for_sentence(sentence_id).chunk_id
    state = ControllerState.initial()
    chunk_handle = state.handle_registry.register(chunk_id, "CHUNK")
    state.visible_chunk_ids.add(chunk_id)
    state.read_chunk_ids.add(chunk_id)

    evidence = EvidenceResolver(substrate).resolve(
        [ChunkRef(id=chunk_handle)], state, "q1"
    )

    assert evidence[0].ref == ChunkRef(id=chunk_id)
    assert evidence[0].parent_chunk_id == chunk_id
