"""State-dependent validation for policy decisions."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.agent.models import (
    AssessmentStatus,
    ChunkRef,
    ControllerState,
    ExpandAction,
    ExpansionKind,
    FinishAction,
    PolicyDecision,
    ReadAction,
    SearchAction,
    SentenceRef,
    action_signature,
)
from agentic_rag.storage import Substrate


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    code: str | None = None
    message: str | None = None
    signature: str | None = None

    @classmethod
    def valid(cls, signature: str) -> "ValidationResult":
        return cls(ok=True, signature=signature)

    @classmethod
    def invalid(
        cls, code: str, message: str, *, signature: str | None = None
    ) -> "ValidationResult":
        return cls(ok=False, code=code, message=message, signature=signature)


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


class DecisionValidator:
    """Validate policy choices without consulting gold evidence."""

    def __init__(
        self,
        substrate: Substrate,
        enabled_expansions: tuple[ExpansionKind, ...],
    ) -> None:
        self.substrate = substrate
        self.enabled_expansions = frozenset(enabled_expansions)

    def validate(
        self,
        decision: PolicyDecision,
        state: ControllerState,
        scope_id: str,
    ) -> ValidationResult:
        self.substrate.require_scope(scope_id)
        action = decision.action
        signature = action_signature(action)

        if signature in state.action_signatures:
            return ValidationResult.invalid(
                "duplicate_action",
                "This exact action has already been attempted",
                signature=signature,
            )

        assessment_error = self._validate_assessment_refs(
            decision, state, scope_id
        )
        if assessment_error is not None:
            return ValidationResult.invalid(
                *assessment_error, signature=signature
            )

        if isinstance(action, FinishAction):
            if decision.assessment.status is not AssessmentStatus.SUFFICIENT:
                return ValidationResult.invalid(
                    "finish_requires_sufficient",
                    "FINISH requires assessment.status=SUFFICIENT",
                    signature=signature,
                )
            evidence_error = self._validate_finish_refs(
                decision, state, scope_id
            )
            if evidence_error is not None:
                return ValidationResult.invalid(
                    *evidence_error, signature=signature
                )
        elif decision.assessment.status is AssessmentStatus.SUFFICIENT:
            return ValidationResult.invalid(
                "sufficient_requires_finish",
                "A SUFFICIENT assessment must choose FINISH",
                signature=signature,
            )
        elif isinstance(action, ExpandAction):
            expansion_error = self._validate_expand(action, state, scope_id)
            if expansion_error is not None:
                return ValidationResult.invalid(
                    *expansion_error, signature=signature
                )
        elif isinstance(action, ReadAction):
            read_error = self._validate_read(action, state, scope_id)
            if read_error is not None:
                return ValidationResult.invalid(
                    *read_error, signature=signature
                )
        elif not isinstance(action, SearchAction):
            return ValidationResult.invalid(
                "unsupported_action",
                f"Unsupported action type: {type(action).__name__}",
                signature=signature,
            )

        return ValidationResult.valid(signature)

    def _validate_expand(
        self,
        action: ExpandAction,
        state: ControllerState,
        scope_id: str,
    ) -> tuple[str, str] | None:
        if action.kind not in self.enabled_expansions:
            return (
                "disabled_expansion",
                f"Expansion {action.kind.value} is disabled for this run",
            )

        if action.kind in _ENTITY_SOURCE_KINDS:
            if action.source_id not in state.visible_entity_ids:
                return (
                    "source_not_visible",
                    f"Entity source is not visible to the agent: {action.source_id}",
                )
            if action.source_id not in self.substrate.entity_ids_by_scope[scope_id]:
                return (
                    "source_out_of_scope",
                    f"Entity source is outside scope {scope_id}: {action.source_id}",
                )
        elif action.kind in _SENTENCE_SOURCE_KINDS:
            if action.source_id not in state.visible_sentence_ids:
                return (
                    "source_not_visible",
                    f"Sentence source is not visible to the agent: {action.source_id}",
                )
            if action.source_id not in state.eligible_sentence_ids:
                return (
                    "source_not_complete",
                    (
                        "Sentence source is only visible as a navigation preview; "
                        "READ its parent Chunk before expanding it: "
                        f"{action.source_id}"
                    ),
                )
            if action.source_id not in self.substrate.sentence_ids_by_scope[scope_id]:
                return (
                    "source_out_of_scope",
                    f"Sentence source is outside scope {scope_id}: {action.source_id}",
                )
        elif action.kind in _CHUNK_SOURCE_KINDS:
            if action.source_id not in state.visible_chunk_ids:
                return (
                    "source_not_visible",
                    f"Chunk source is not visible to the agent: {action.source_id}",
                )
            if action.source_id not in self.substrate.chunk_ids_by_scope[scope_id]:
                return (
                    "source_out_of_scope",
                    f"Chunk source is outside scope {scope_id}: {action.source_id}",
                )
        else:
            return (
                "unsupported_expansion",
                f"Unknown expansion kind: {action.kind}",
            )
        return None

    def _validate_read(
        self,
        action: ReadAction,
        state: ControllerState,
        scope_id: str,
    ) -> tuple[str, str] | None:
        if action.chunk_id not in state.visible_chunk_ids:
            return (
                "source_not_visible",
                f"Chunk is not visible to the agent: {action.chunk_id}",
            )
        if action.chunk_id not in self.substrate.chunk_ids_by_scope[scope_id]:
            return (
                "source_out_of_scope",
                f"Chunk is outside scope {scope_id}: {action.chunk_id}",
            )
        return None

    def _validate_assessment_refs(
        self,
        decision: PolicyDecision,
        state: ControllerState,
        scope_id: str,
    ) -> tuple[str, str] | None:
        for ref in decision.assessment.selected_evidence_refs:
            if isinstance(ref, SentenceRef):
                if ref.id not in state.visible_sentence_ids:
                    return (
                        "selected_evidence_not_visible",
                        f"Selected SentenceRef is not visible: {ref.id}",
                    )
                if ref.id not in self.substrate.sentence_ids_by_scope[scope_id]:
                    return (
                        "selected_evidence_out_of_scope",
                        f"Selected SentenceRef is outside scope {scope_id}: {ref.id}",
                    )
                if ref.id not in state.eligible_sentence_ids:
                    return (
                        "selected_evidence_not_eligible",
                        (
                            "Selected SentenceRef has not been shown as "
                            f"complete evidence: {ref.id}"
                        ),
                    )
            elif isinstance(ref, ChunkRef):
                if ref.id not in state.visible_chunk_ids:
                    return (
                        "selected_evidence_not_visible",
                        f"Selected ChunkRef is not visible: {ref.id}",
                    )
                if ref.id not in self.substrate.chunk_ids_by_scope[scope_id]:
                    return (
                        "selected_evidence_out_of_scope",
                        f"Selected ChunkRef is outside scope {scope_id}: {ref.id}",
                    )
                if ref.id not in state.read_chunk_ids:
                    return (
                        "selected_evidence_not_eligible",
                        f"Selected ChunkRef must already be READ: {ref.id}",
                    )
        return None

    def _validate_finish_refs(
        self,
        decision: PolicyDecision,
        state: ControllerState,
        scope_id: str,
    ) -> tuple[str, str] | None:
        action = decision.action
        if not isinstance(action, FinishAction):
            raise TypeError("_validate_finish_refs requires a FINISH decision")
        selected = {
            (ref.unit, ref.id)
            for ref in decision.assessment.selected_evidence_refs
        }
        for ref in action.evidence_refs:
            if (ref.unit, ref.id) not in selected:
                return (
                    "finish_evidence_not_selected",
                    (
                        "Every FINISH evidence ref must occur in this turn's "
                        f"selected_evidence_refs: {ref.unit}:{ref.id}"
                    ),
                )
            if isinstance(ref, SentenceRef):
                if ref.id not in state.eligible_sentence_ids:
                    return (
                        "evidence_not_eligible",
                        f"Sentence has not been shown as complete evidence: {ref.id}",
                    )
                if ref.id not in self.substrate.sentence_ids_by_scope[scope_id]:
                    return (
                        "evidence_out_of_scope",
                        f"Sentence evidence is outside scope {scope_id}: {ref.id}",
                    )
            elif isinstance(ref, ChunkRef):
                if ref.id not in state.read_chunk_ids:
                    return (
                        "evidence_not_eligible",
                        f"Chunk must be READ before FINISH: {ref.id}",
                    )
                if ref.id not in self.substrate.chunk_ids_by_scope[scope_id]:
                    return (
                        "evidence_out_of_scope",
                        f"Chunk evidence is outside scope {scope_id}: {ref.id}",
                    )
        return None
