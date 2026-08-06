"""Resolve V3 context-local indices against the exact prompt snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_rag.agent.models import (
    ContextReferenceMap,
    EvidenceAssessment,
    ExpandAction,
    ExpansionKind,
    FinishAction,
    PolicyDecision,
    ReadAction,
    SearchAction,
    SentenceRef,
    ChunkRef,
    TypedContextReferenceMap,
    V31ExpandAction,
    V31FinishAction,
    V31PolicyDecision,
    V31ReadAction,
    V3EvidenceAssessment,
    V3ExpandAction,
    V3FinishAction,
    V3PolicyDecision,
    V3ReadAction,
)

ContextResolutionCode = Literal[
    "memory_index_out_of_range",
    "citation_index_out_of_range",
    "memory_node_type_mismatch",
    "chunk_not_readable",
    "expansion_not_valid_for_node",
    "reference_not_available",
    "reference_type_mismatch",
    "reference_not_evidence",
]


@dataclass(frozen=True, slots=True)
class ContextIndexResolutionError(ValueError):
    code: ContextResolutionCode
    message: str

    def __str__(self) -> str:
        return self.message


_ENTITY_SOURCE_KINDS = frozenset(
    {
        ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
        ExpansionKind.ENTITY_CO_OCCURS_ENTITY_SENTENCE,
        ExpansionKind.ENTITY_MENTIONED_IN_CHUNK,
        ExpansionKind.ENTITY_CO_OCCURS_ENTITY_CHUNK,
    }
)
_SENTENCE_SOURCE_KINDS = frozenset(
    {ExpansionKind.SENTENCE_MENTIONS_ENTITY}
)
_CHUNK_SOURCE_KINDS = frozenset(
    {
        ExpansionKind.CHUNK_ADJACENT_CHUNK,
        ExpansionKind.CHUNK_CONTAINS_SENTENCE,
        ExpansionKind.CHUNK_MENTIONS_ENTITY,
    }
)


def resolve_v3_decision(
    decision: V3PolicyDecision,
    references: ContextReferenceMap,
) -> PolicyDecision:
    """Convert a V3 Policy decision to the stable internal contract."""

    action = decision.action
    selected_refs = []
    if isinstance(action, SearchAction):
        resolved_action = action
    elif isinstance(action, V3ExpandAction):
        expected_type = _expected_expansion_source(action.kind)
        source = references.memory.get(action.source_context_index)
        if source is None:
            raise ContextIndexResolutionError(
                "memory_index_out_of_range",
                (
                    "Unknown memory index: "
                    f"{action.source_context_index}"
                ),
            )
        if source.node_type != expected_type:
            raise ContextIndexResolutionError(
                "memory_node_type_mismatch",
                (
                    f"Memory item {action.source_context_index} is "
                    f"{source.node_type}, but {action.kind.value} requires "
                    f"{expected_type}"
                ),
            )
        resolved_action = ExpandAction(
            kind=action.kind,
            source_id=source.stable_id,
            direction=action.direction,
            query=action.query,
            top_k=action.top_k,
        )
    elif isinstance(action, V3ReadAction):
        source = references.memory.get(action.chunk_context_index)
        if source is None:
            raise ContextIndexResolutionError(
                "memory_index_out_of_range",
                f"Unknown memory index: {action.chunk_context_index}",
            )
        if source.node_type != "CHUNK":
            raise ContextIndexResolutionError(
                "memory_node_type_mismatch",
                (
                    f"Memory item {action.chunk_context_index} is "
                    f"{source.node_type}, but READ requires CHUNK"
                ),
            )
        if not source.can_read:
            raise ContextIndexResolutionError(
                "chunk_not_readable",
                (
                    f"Memory item {action.chunk_context_index} is already "
                    "read and cannot be READ again"
                ),
            )
        resolved_action = ReadAction(chunk_id=source.stable_id)
    elif isinstance(action, V3FinishAction):
        for citation in action.citations:
            ref = references.citations.get(citation)
            if ref is None:
                raise ContextIndexResolutionError(
                    "citation_index_out_of_range",
                    f"Unknown evidence citation: {citation}",
                )
            selected_refs.append(ref.model_copy(deep=True))
        resolved_action = FinishAction(
            evidence_refs=list(selected_refs),
            answer=action.answer,
        )
    else:  # pragma: no cover - the discriminated union prevents this
        raise TypeError(f"Unsupported V3 action: {type(action).__name__}")

    return PolicyDecision(
        assessment=assessment_for_internal(
            decision.assessment,
            selected_refs=selected_refs,
        ),
        action=resolved_action,
    )


def resolve_v31_decision(
    decision: V31PolicyDecision,
    references: TypedContextReferenceMap,
) -> PolicyDecision:
    """Resolve one V3.1 typed-ref decision against its frozen visible map."""

    action = decision.action
    selected_refs = []
    if isinstance(action, SearchAction):
        resolved_action = action
    elif isinstance(action, V31ExpandAction):
        expected_type = _expected_expansion_source(action.kind)
        source = references.typed_refs.get(action.source_ref)
        if source is None:
            raise ContextIndexResolutionError(
                "reference_not_available",
                f"Reference {action.source_ref} is not visible in this snapshot",
            )
        if source.node_type != expected_type:
            raise ContextIndexResolutionError(
                "reference_type_mismatch",
                (
                    f"Reference {action.source_ref} is {source.node_type}, but "
                    f"{action.kind.value} requires {expected_type}"
                ),
            )
        resolved_action = ExpandAction(
            kind=action.kind,
            source_id=source.stable_id,
            direction=action.direction,
            query=action.query,
            top_k=action.top_k,
        )
    elif isinstance(action, V31ReadAction):
        source = references.typed_refs.get(action.chunk_ref)
        if source is None:
            raise ContextIndexResolutionError(
                "reference_not_available",
                f"Reference {action.chunk_ref} is not visible in this snapshot",
            )
        if source.node_type != "CHUNK":
            raise ContextIndexResolutionError(
                "reference_type_mismatch",
                (
                    f"Reference {action.chunk_ref} is {source.node_type}, "
                    "but READ requires CHUNK"
                ),
            )
        if not source.can_read:
            raise ContextIndexResolutionError(
                "chunk_not_readable",
                f"Chunk {action.chunk_ref} is already read and cannot be READ again",
            )
        resolved_action = ReadAction(chunk_id=source.stable_id)
    elif isinstance(action, V31FinishAction):
        for typed_ref in action.evidence_refs:
            source = references.typed_refs.get(typed_ref)
            if source is None:
                raise ContextIndexResolutionError(
                    "reference_not_available",
                    f"Reference {typed_ref} is not visible in this snapshot",
                )
            if not source.can_use_as_evidence:
                raise ContextIndexResolutionError(
                    "reference_not_evidence",
                    f"Reference {typed_ref} is not eligible answer evidence",
                )
            if source.node_type == "SENTENCE":
                selected_refs.append(SentenceRef(id=source.stable_id))
            elif source.node_type == "CHUNK":
                selected_refs.append(ChunkRef(id=source.stable_id))
            else:
                raise ContextIndexResolutionError(
                    "reference_not_evidence",
                    f"Entity reference {typed_ref} cannot be answer evidence",
                )
        resolved_action = FinishAction(
            evidence_refs=list(selected_refs),
            answer=action.answer,
        )
    else:  # pragma: no cover - the discriminated union prevents this
        raise TypeError(f"Unsupported V3.1 action: {type(action).__name__}")

    return PolicyDecision(
        assessment=assessment_for_internal(
            decision.assessment,
            selected_refs=selected_refs,
        ),
        action=resolved_action,
    )


def assessment_for_internal(
    assessment: V3EvidenceAssessment,
    *,
    selected_refs: list | None = None,
) -> EvidenceAssessment:
    """Create an internal assessment without exposing reference mechanics."""

    return EvidenceAssessment(
        status=assessment.status,
        supported_facts=list(assessment.supported_facts),
        missing_information=list(assessment.missing_information),
        selected_evidence_refs=list(selected_refs or []),
    )


def _expected_expansion_source(kind: ExpansionKind) -> str:
    if kind in _ENTITY_SOURCE_KINDS:
        return "ENTITY"
    if kind in _SENTENCE_SOURCE_KINDS:
        return "SENTENCE"
    if kind in _CHUNK_SOURCE_KINDS:
        return "CHUNK"
    raise ContextIndexResolutionError(
        "expansion_not_valid_for_node",
        f"Unsupported expansion kind: {kind.value}",
    )
