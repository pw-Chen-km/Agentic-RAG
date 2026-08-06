"""Narrow, auditable repairs at the Policy/controller boundary."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.agent.models import (
    AssessmentStatus,
    ControllerState,
    ExpandAction,
    ExpansionKind,
    FinishAction,
    PolicyDecision,
)


@dataclass(frozen=True, slots=True)
class DecisionRepairResult:
    decision: PolicyDecision
    code: str | None = None


class DecisionRepairer:
    """Repair only an unambiguous inverse sentence/entity expansion.

    Unknown handles, evidence references, READ/FINISH actions, and ambiguous
    entity/chunk expansions are deliberately left untouched. Every repaired
    decision still passes through normal handle resolution and validation.
    """

    _INVERSE_KIND = {
        (
            ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
            "SENTENCE",
        ): ExpansionKind.SENTENCE_MENTIONS_ENTITY,
        (
            ExpansionKind.SENTENCE_MENTIONS_ENTITY,
            "ENTITY",
        ): ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
    }

    def __init__(
        self,
        enabled_expansions: tuple[ExpansionKind, ...],
        *,
        derive_status_from_action: bool = False,
    ) -> None:
        self.enabled_expansions = frozenset(enabled_expansions)
        self.derive_status_from_action = derive_status_from_action

    def repair(
        self, decision: PolicyDecision, state: ControllerState
    ) -> DecisionRepairResult:
        repaired = decision
        codes: list[str] = []
        if self.derive_status_from_action:
            action_is_finish = isinstance(repaired.action, FinishAction)
            status_is_sufficient = (
                repaired.assessment.status is AssessmentStatus.SUFFICIENT
            )
            if action_is_finish and not status_is_sufficient:
                repaired = repaired.model_copy(
                    update={
                        "assessment": repaired.assessment.model_copy(
                            update={"status": AssessmentStatus.SUFFICIENT}
                        )
                    }
                )
                codes.append("finish_status_derived_from_action")
            elif not action_is_finish and status_is_sufficient:
                repaired = repaired.model_copy(
                    update={
                        "assessment": repaired.assessment.model_copy(
                            update={"status": AssessmentStatus.INSUFFICIENT}
                        )
                    }
                )
                codes.append("continue_status_derived_from_action")

        action = repaired.action
        if not isinstance(action, ExpandAction):
            return DecisionRepairResult(
                repaired,
                code="+".join(codes) or None,
            )

        registry = state.handle_registry
        if action.source_id not in registry.handle_to_stable_id:
            return DecisionRepairResult(
                repaired,
                code="+".join(codes) or None,
            )
        actual_type = registry.node_type_for_handle(action.source_id)
        replacement = self._INVERSE_KIND.get((action.kind, actual_type))
        if replacement is None or replacement not in self.enabled_expansions:
            return DecisionRepairResult(
                repaired,
                code="+".join(codes) or None,
            )

        repaired_action = action.model_copy(update={"kind": replacement})
        repaired = repaired.model_copy(update={"action": repaired_action})
        codes.append("expand_direction_repaired_from_handle_type")
        return DecisionRepairResult(
            repaired,
            code="+".join(codes),
        )
