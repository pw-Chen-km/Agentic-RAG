"""Resolve Policy typed refs against the exact frozen prompt snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_rag.agent.models import (
    ChunkRef,
    ContextReferenceMap,
    ExpandAction,
    ExpansionKind,
    FinishAction,
    PolicyDecision,
    ReadAction,
    ResolvedDecision,
    ResolvedExpandAction,
    ResolvedFinishAction,
    ResolvedReadAction,
    SearchAction,
    SentenceRef,
)

ReferenceResolutionCode = Literal[
    "reference_not_available",
    "reference_type_mismatch",
    "reference_not_evidence",
    "chunk_not_readable",
    "expansion_not_valid_for_node",
]


@dataclass(frozen=True, slots=True)
class ReferenceResolutionError(ValueError):
    code: ReferenceResolutionCode
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
_SENTENCE_SOURCE_KINDS = frozenset({ExpansionKind.SENTENCE_MENTIONS_ENTITY})
_CHUNK_SOURCE_KINDS = frozenset(
    {
        ExpansionKind.CHUNK_ADJACENT_CHUNK,
        ExpansionKind.CHUNK_CONTAINS_SENTENCE,
        ExpansionKind.CHUNK_MENTIONS_ENTITY,
    }
)


def resolve_decision(
    decision: PolicyDecision,
    references: ContextReferenceMap,
) -> ResolvedDecision:
    """Resolve one decision without consulting a newer context snapshot."""

    action = decision.action
    if isinstance(action, SearchAction):
        resolved = action
    elif isinstance(action, ExpandAction):
        expected_type = expected_expansion_source(action.kind)
        source = references.typed_refs.get(action.source_ref)
        if source is None:
            raise ReferenceResolutionError(
                "reference_not_available",
                f"Reference {action.source_ref} is not visible in this snapshot",
            )
        if source.node_type != expected_type:
            raise ReferenceResolutionError(
                "reference_type_mismatch",
                f"Reference {action.source_ref} is {source.node_type}, but {action.kind.value} requires {expected_type}",
            )
        resolved = ResolvedExpandAction(
            kind=action.kind,
            source_id=source.stable_id,
            direction=action.direction,
            query=action.query,
            top_k=action.top_k,
        )
    elif isinstance(action, ReadAction):
        source = references.typed_refs.get(action.chunk_ref)
        if source is None:
            raise ReferenceResolutionError(
                "reference_not_available",
                f"Reference {action.chunk_ref} is not visible in this snapshot",
            )
        if source.node_type != "CHUNK":
            raise ReferenceResolutionError(
                "reference_type_mismatch",
                f"Reference {action.chunk_ref} is {source.node_type}, but READ requires CHUNK",
            )
        if not source.can_read:
            raise ReferenceResolutionError(
                "chunk_not_readable",
                f"Chunk {action.chunk_ref} is already read and cannot be READ again",
            )
        resolved = ResolvedReadAction(chunk_id=source.stable_id)
    elif isinstance(action, FinishAction):
        evidence = []
        for ref in action.evidence_refs:
            source = references.typed_refs.get(ref)
            if source is None:
                raise ReferenceResolutionError(
                    "reference_not_available",
                    f"Reference {ref} is not visible in this snapshot",
                )
            if not source.can_use_as_evidence:
                raise ReferenceResolutionError(
                    "reference_not_evidence",
                    f"Reference {ref} is not eligible answer evidence",
                )
            if source.node_type == "SENTENCE":
                evidence.append(SentenceRef(id=source.stable_id))
            elif source.node_type == "CHUNK":
                evidence.append(ChunkRef(id=source.stable_id))
            else:
                raise ReferenceResolutionError(
                    "reference_not_evidence",
                    f"Entity reference {ref} cannot be answer evidence",
                )
        resolved = ResolvedFinishAction(
            answer=action.answer,
            evidence_refs=evidence,
        )
    else:  # pragma: no cover - discriminated union prevents this
        raise TypeError(f"unsupported action: {type(action).__name__}")
    return ResolvedDecision(
        assessment=decision.assessment.model_copy(deep=True) if decision.assessment is not None else None,
        action=resolved,
    )


def expected_expansion_source(kind: ExpansionKind) -> str:
    if kind in _ENTITY_SOURCE_KINDS:
        return "ENTITY"
    if kind in _SENTENCE_SOURCE_KINDS:
        return "SENTENCE"
    if kind in _CHUNK_SOURCE_KINDS:
        return "CHUNK"
    raise ReferenceResolutionError(
        "expansion_not_valid_for_node",
        f"Unsupported expansion kind: {kind.value}",
    )
