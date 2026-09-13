"""Validate already-resolved actions without choosing a retrieval strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    ChunkRef,
    EpisodeState,
    ExpansionKind,
    ResolvedDecision,
    ResolvedExpandAction,
    ResolvedFinishAction,
    ResolvedReadAction,
    SearchAction,
    SentenceRef,
    action_signature,
)
from agentic_rag.agent.interface import InterfaceContract
from agentic_rag.agent.references import expected_expansion_source
from agentic_rag.substrate.storage import Substrate


ValidationCode = Literal[
    "search_pair_not_enabled",
    "duplicate_action",
    "expansion_not_enabled",
    "expansion_not_valid_for_node",
    "source_not_visible",
    "source_out_of_scope",
    "chunk_not_readable",
    "reference_not_evidence",
]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    code: ValidationCode | None = None
    message: str | None = None
    signature: str | None = None
    resolved_decision: ResolvedDecision | None = None


class DecisionValidator:
    """Enforce structural legality after the Policy has selected an action."""

    def __init__(
        self,
        substrate: Substrate,
        enabled_expansions: tuple[ExpansionKind, ...] = DEFAULT_ENABLED_EXPANSIONS,
        interface_contract: InterfaceContract | None = None,
    ) -> None:
        if isinstance(enabled_expansions, InterfaceContract) and interface_contract is None:
            interface_contract = enabled_expansions
            enabled_expansions = interface_contract.enabled_expansions
        self.substrate = substrate
        self.interface_contract = interface_contract
        self.enabled_expansions = tuple(
            interface_contract.enabled_expansions
            if interface_contract is not None
            else enabled_expansions
        )

    def validate(
        self,
        decision: ResolvedDecision,
        state: EpisodeState,
        scope_id: str,
    ) -> ValidationResult:
        self.substrate.require_scope(scope_id)
        action = decision.action
        if isinstance(action, SearchAction) and self.interface_contract is not None:
            if not self.interface_contract.allows_search(action.method, action.target):
                return self._invalid(
                    "search_pair_not_enabled",
                    f"SEARCH pair is not enabled by {self.interface_contract.name}: "
                    f"{action.method.value}->{action.target.value}",
                    decision,
                )
        if isinstance(action, ResolvedExpandAction):
            invalid = self._validate_expand(action, state, scope_id)
            if invalid is not None:
                return self._invalid(*invalid, decision)
        elif isinstance(action, ResolvedReadAction):
            invalid = self._validate_read(action, state, scope_id)
            if invalid is not None:
                return self._invalid(*invalid, decision)
        elif isinstance(action, ResolvedFinishAction):
            invalid = self._validate_finish(action, state, scope_id)
            if invalid is not None:
                return self._invalid(*invalid, decision)
        elif not isinstance(action, SearchAction):
            return self._invalid(
                "expansion_not_valid_for_node",
                f"Unsupported action: {type(action).__name__}",
                decision,
            )

        signature = action_signature(action)
        if signature in state.action_signatures:
            return ValidationResult(
                ok=False,
                code="duplicate_action",
                message="The same semantic action was already executed",
                signature=signature,
                resolved_decision=decision,
            )
        return ValidationResult(
            ok=True,
            signature=signature,
            resolved_decision=decision,
        )

    def _validate_expand(
        self,
        action: ResolvedExpandAction,
        state: EpisodeState,
        scope_id: str,
    ) -> tuple[ValidationCode, str] | None:
        if action.kind not in self.enabled_expansions:
            return "expansion_not_enabled", f"Expansion kind is disabled: {action.kind.value}"
        expected = expected_expansion_source(action.kind)
        visible = {
            "ENTITY": state.visible_entity_ids,
            "SENTENCE": state.visible_sentence_ids,
            "CHUNK": state.visible_chunk_ids,
        }[expected]
        if action.source_id not in visible:
            return "source_not_visible", f"{expected} source is not currently visible"
        allowed = {
            "ENTITY": self.substrate.entity_ids_by_scope[scope_id],
            "SENTENCE": self.substrate.sentence_ids_by_scope[scope_id],
            "CHUNK": self.substrate.chunk_ids_by_scope[scope_id],
        }[expected]
        if action.source_id not in allowed:
            return "source_out_of_scope", f"{expected} source is outside the query scope"
        if expected == "SENTENCE" and action.source_id not in state.eligible_sentence_ids:
            return (
                "expansion_not_valid_for_node",
                "Sentence expansion requires a complete eligible Sentence",
            )
        return None

    def _validate_read(
        self,
        action: ResolvedReadAction,
        state: EpisodeState,
        scope_id: str,
    ) -> tuple[ValidationCode, str] | None:
        if action.chunk_id not in state.visible_chunk_ids:
            return "source_not_visible", "READ Chunk is not currently visible"
        if action.chunk_id not in self.substrate.chunk_ids_by_scope[scope_id]:
            return "source_out_of_scope", "READ Chunk is outside the query scope"
        if action.chunk_id in state.read_chunk_ids:
            return "chunk_not_readable", "Chunk was already READ"
        return None

    def _validate_finish(
        self,
        action: ResolvedFinishAction,
        state: EpisodeState,
        scope_id: str,
    ) -> tuple[ValidationCode, str] | None:
        seen: set[tuple[str, str]] = set()
        for ref in action.evidence_refs:
            key = (ref.unit, ref.id)
            if key in seen:
                return "reference_not_evidence", "FINISH evidence contains duplicates"
            seen.add(key)
            if isinstance(ref, SentenceRef):
                if (
                    ref.id not in state.eligible_sentence_ids
                    or ref.id not in self.substrate.sentence_ids_by_scope[scope_id]
                ):
                    return "reference_not_evidence", "FINISH Sentence is not eligible evidence"
            elif isinstance(ref, ChunkRef):
                if (
                    ref.id not in state.read_chunk_ids
                    or ref.id not in self.substrate.chunk_ids_by_scope[scope_id]
                ):
                    return "reference_not_evidence", "FINISH Chunk has not been READ"
        return None

    @staticmethod
    def _invalid(
        code: ValidationCode,
        message: str,
        decision: ResolvedDecision,
    ) -> ValidationResult:
        return ValidationResult(
            ok=False,
            code=code,
            message=message,
            resolved_decision=decision,
        )
