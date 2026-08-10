"""The sole owner of episode state, history, budgets, and usage."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.agent.context import project_observation_for_audit
from agentic_rag.agent.models import (
    Assessment,
    AvailableActionSpace,
    ContextReferenceMap,
    EpisodeResult,
    EpisodeState,
    EvidenceRef,
    Observation,
    PolicyDecision,
    PolicyView,
    ResolvedDecision,
    ResolvedEvidence,
    StepRecord,
    TerminationReason,
    Usage,
    ValidationStatus,
)
from agentic_rag.agent.state import StateUpdater


@dataclass(frozen=True, slots=True)
class AttemptEvent:
    decision: PolicyDecision | None
    resolved_decision: ResolvedDecision | None
    validation_status: ValidationStatus
    validation_error: str | None
    observation: Observation
    assessment: Assessment | None
    action_signature: str | None
    usage: Usage
    policy_view: PolicyView
    context_reference_map: ContextReferenceMap
    available_action_space: AvailableActionSpace
    decision_schema_sha256: str
    commit_assessment: bool = True
    consume_step: bool = True
    invalid_attempt: bool = False


class EpisodeStateManager:
    """Single mutable owner; all collaborators receive isolated snapshots."""

    def __init__(
        self,
        *,
        episode_id: str,
        question: str,
        scope_id: str,
        state_updater: StateUpdater,
        max_steps: int,
        max_policy_attempts: int,
        max_retrieved_tokens: int,
    ) -> None:
        self.episode_id = episode_id
        self.question = question
        self.scope_id = scope_id
        self._state_updater = state_updater
        self._state = EpisodeState.initial(
            max_steps=max_steps,
            max_policy_attempts=max_policy_attempts,
            max_retrieved_tokens=max_retrieved_tokens,
        )
        self._trajectory: list[StepRecord] = []
        self._total_usage = Usage()

    def snapshot(self) -> EpisodeState:
        return self._state.model_copy(deep=True)

    def trajectory_snapshot(self) -> tuple[StepRecord, ...]:
        return tuple(self._trajectory)

    @property
    def next_action_id(self) -> str:
        return f"attempt-{self._state.policy_attempts + 1}"

    @property
    def can_continue(self) -> bool:
        return (
            self._state.remaining_step_budget > 0
            and self._state.remaining_retrieved_token_budget > 0
            and self._state.remaining_policy_attempt_budget > 0
        )

    @property
    def can_attempt_budget_finalize(self) -> bool:
        return (
            self._state.remaining_policy_attempt_budget > 0
            and bool(self._state.eligible_sentence_ids or self._state.read_chunk_ids)
        )

    @property
    def policy_attempt_budget_exhausted(self) -> bool:
        return (
            self._state.remaining_policy_attempt_budget == 0
            and self._state.remaining_step_budget > 0
            and self._state.remaining_retrieved_token_budget > 0
        )

    def record_attempt(self, event: AttemptEvent) -> StepRecord:
        state_before = self._state.model_copy(deep=True)
        updated = self._state_updater.apply(
            self._state,
            assessment=event.assessment,
            observation=event.observation,
            action_signature=event.action_signature,
            commit_assessment=event.commit_assessment,
            consume_step=event.consume_step,
        )
        record = StepRecord(
            step=updated.step if event.consume_step else state_before.step + 1,
            policy_attempt=updated.policy_attempts,
            decision=event.decision,
            resolved_decision=event.resolved_decision,
            validation_status=event.validation_status,
            validation_error=event.validation_error,
            observation=event.observation,
            agent_visible_observation=project_observation_for_audit(
                event.observation, self._state_updater.substrate
            ),
            state_before=state_before,
            state_after=updated.model_copy(deep=True),
            usage=event.usage,
            policy_view=event.policy_view,
            context_reference_map=event.context_reference_map,
            available_action_space=event.available_action_space,
            decision_schema_sha256=event.decision_schema_sha256,
        )
        self._state = updated
        self._trajectory.append(record)
        self._total_usage = self._total_usage + event.usage
        return record

    def add_usage(self, usage: Usage) -> None:
        self._total_usage = self._total_usage + usage

    def result(
        self,
        *,
        reason: TerminationReason,
        answer: str | None = None,
        evidence_refs: list[EvidenceRef] | None = None,
        resolved_evidence: list[ResolvedEvidence] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> EpisodeResult:
        return EpisodeResult(
            episode_id=self.episode_id,
            query=self.question,
            scope_id=self.scope_id,
            termination_reason=reason,
            answer=answer,
            evidence_refs=evidence_refs or [],
            resolved_evidence=resolved_evidence or [],
            trajectory=list(self._trajectory),
            usage=self._total_usage.model_copy(deep=True),
            final_state=self._state.model_copy(deep=True),
            error_code=error_code,
            error_message=error_message,
        )


class EpisodeStateManagerFactory:
    def __init__(self, state_updater: StateUpdater) -> None:
        self._state_updater = state_updater

    def create(
        self,
        *,
        episode_id: str,
        question: str,
        scope_id: str,
        max_steps: int,
        max_policy_attempts: int,
        max_retrieved_tokens: int,
    ) -> EpisodeStateManager:
        return EpisodeStateManager(
            episode_id=episode_id,
            question=question,
            scope_id=scope_id,
            state_updater=self._state_updater,
            max_steps=max_steps,
            max_policy_attempts=max_policy_attempts,
            max_retrieved_tokens=max_retrieved_tokens,
        )
