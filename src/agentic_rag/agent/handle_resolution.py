"""Resolve short policy handles at the controller/environment boundary.

The policy operates on episode-local handles such as ``S1``, ``C2``, and
``E3``.  Internal substrate, visibility, scope, and evidence rules continue to
operate on stable corpus IDs.  This module is the deliberately small boundary
between those two representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_rag.agent.models import (
    AgentAction,
    ChunkRef,
    ControllerState,
    EvidenceAssessment,
    EvidenceRef,
    ExpandAction,
    FinishAction,
    PolicyDecision,
    ReadAction,
    SearchAction,
    SentenceRef,
)
from agentic_rag.storage import Substrate

NodeType = Literal["ENTITY", "SENTENCE", "CHUNK"]


@dataclass(frozen=True, slots=True)
class HandleResolutionError(ValueError):
    """A policy supplied an unknown handle or a handle of the wrong type."""

    code: Literal["unknown_handle", "handle_type_mismatch"]
    value: str
    expected_type: NodeType
    actual_type: str | None = None

    def __str__(self) -> str:
        if self.code == "handle_type_mismatch":
            return (
                f"Handle {self.value} identifies {self.actual_type}, "
                f"not {self.expected_type}"
            )
        return f"Unknown {self.expected_type} handle: {self.value}"


def resolve_node_id(
    value: str,
    expected_type: NodeType,
    state: ControllerState,
    substrate: Substrate,
) -> str:
    """Resolve one policy handle to a stable substrate ID.

    Exact stable IDs remain accepted as an internal/programmatic compatibility
    path.  They are never projected into the online policy context.  An
    invented or misspelled stable-looking value is therefore rejected as an
    unknown handle rather than reaching visibility validation.
    """

    registry = getattr(state, "handle_registry", None)
    if registry is not None:
        stable_id = registry.stable_id_for(value, expected_type)
        if stable_id is not None:
            return stable_id
        actual_type = (
            registry.node_type_for_handle(value)
            if value in registry.handle_to_stable_id
            else None
        )
        if actual_type is not None:
            raise HandleResolutionError(
                code="handle_type_mismatch",
                value=value,
                expected_type=expected_type,
                actual_type=actual_type,
            )

    if _is_stable_id(value, substrate, expected_type):
        return value

    raise HandleResolutionError(
        code="unknown_handle",
        value=value,
        expected_type=expected_type,
    )


def resolve_evidence_ref(
    ref: EvidenceRef,
    state: ControllerState,
    substrate: Substrate,
) -> EvidenceRef:
    if isinstance(ref, SentenceRef):
        return ref.model_copy(
            update={
                "id": resolve_node_id(
                    ref.id, "SENTENCE", state, substrate
                )
            }
        )
    return ref.model_copy(
        update={"id": resolve_node_id(ref.id, "CHUNK", state, substrate)}
    )


def resolve_assessment(
    assessment: EvidenceAssessment,
    state: ControllerState,
    substrate: Substrate,
) -> EvidenceAssessment:
    return assessment.model_copy(
        update={
            "selected_evidence_refs": [
                resolve_evidence_ref(ref, state, substrate)
                for ref in assessment.selected_evidence_refs
            ]
        }
    )


def resolve_action(
    action: AgentAction,
    state: ControllerState,
    substrate: Substrate,
) -> AgentAction:
    if isinstance(action, SearchAction):
        return action
    if isinstance(action, ExpandAction):
        return action.model_copy(
            update={
                "source_id": resolve_node_id(
                    action.source_id,
                    _expand_source_type(action),
                    state,
                    substrate,
                )
            }
        )
    if isinstance(action, ReadAction):
        return action.model_copy(
            update={
                "chunk_id": resolve_node_id(
                    action.chunk_id, "CHUNK", state, substrate
                )
            }
        )
    if isinstance(action, FinishAction):
        return action.model_copy(
            update={
                "evidence_refs": [
                    resolve_evidence_ref(ref, state, substrate)
                    for ref in action.evidence_refs
                ]
            }
        )
    raise TypeError(f"Unsupported action type: {type(action).__name__}")


def resolve_policy_decision(
    decision: PolicyDecision,
    state: ControllerState,
    substrate: Substrate,
) -> PolicyDecision:
    """Return an internal decision whose node references are stable IDs."""

    return decision.model_copy(
        update={
            "assessment": resolve_assessment(
                decision.assessment, state, substrate
            ),
            "action": resolve_action(decision.action, state, substrate),
        }
    )


def _expand_source_type(action: ExpandAction) -> NodeType:
    kind = action.kind.value
    if kind.startswith("ENTITY_"):
        return "ENTITY"
    if kind.startswith("SENTENCE_"):
        return "SENTENCE"
    if kind.startswith("CHUNK_"):
        return "CHUNK"
    raise TypeError(f"Cannot determine source type for expansion {kind}")


def _is_stable_id(
    value: str,
    substrate: Substrate,
    node_type: NodeType,
) -> bool:
    if node_type == "ENTITY":
        return value in substrate.entity_by_id
    if node_type == "SENTENCE":
        return value in substrate.sentence_by_id
    return value in substrate.chunk_by_id
