"""Episode-local state, history, budget, and usage ownership."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.agent.context import (
    project_observation_for_policy,
    project_v3_observation_for_policy,
)
from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    ControllerState,
    ContextReferenceMap,
    TypedContextReferenceMap,
    EpisodeResult,
    EvidenceAssessment,
    EvidenceRef,
    ExpansionKind,
    Observation,
    PolicyDecision,
    PolicyDecisionOutput,
    PolicyStageRecord,
    PolicyViewOutput,
    ResolvedEvidence,
    StepRecord,
    TerminationReason,
    Usage,
    ValidationStatus,
)
from agentic_rag.agent.state import StateUpdater


@dataclass(frozen=True, slots=True)
class AttemptEvent:
    """Everything needed to atomically record one Policy attempt."""

    decision: PolicyDecisionOutput | None
    repaired_decision: PolicyDecision | None
    repair_code: str | None
    resolved_decision: PolicyDecision | None
    validation_status: ValidationStatus
    validation_error: str | None
    observation: Observation
    assessment: EvidenceAssessment | None
    action_signature: str | None
    usage: Usage
    policy_view: PolicyViewOutput
    context_reference_map: (
        ContextReferenceMap | TypedContextReferenceMap | None
    ) = None
    policy_stages: tuple[PolicyStageRecord, ...] = ()
    commit_assessment: bool = True
    consume_step: bool = True
    invalid_attempt: bool = False


class EpisodeStateManager:
    """Single mutable owner of all state for one agent episode.

    The Controller may inspect immutable snapshots and submit AttemptEvents,
    but it does not maintain a second state, history, budget, or usage copy.
    """

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
        enabled_expansions: tuple[ExpansionKind, ...] = (
            DEFAULT_ENABLED_EXPANSIONS
        ),
        semantic_memory_v3: bool = False,
    ) -> None:
        self.episode_id = episode_id
        self.question = question
        self.scope_id = scope_id
        self._state_updater = state_updater
        self._enabled_expansions = tuple(enabled_expansions)
        self._semantic_memory_v3 = semantic_memory_v3
        self._state = ControllerState.initial(
            max_steps=max_steps,
            max_policy_attempts=max_policy_attempts,
            max_retrieved_tokens=max_retrieved_tokens,
        )
        self._trajectory: list[StepRecord] = []
        self._total_usage = Usage()
        self._consecutive_invalid_attempts = 0

    def snapshot(self) -> ControllerState:
        """Return an isolated state value for read-only collaborators."""

        return self._state.model_copy(deep=True)

    def trajectory_snapshot(self) -> tuple[StepRecord, ...]:
        """Return the append-only history as a read-only sequence."""

        return tuple(self._trajectory)

    @property
    def next_action_id(self) -> str:
        return f"attempt-{self._state.policy_attempts + 1}"

    @property
    def consecutive_invalid_attempts(self) -> int:
        return self._consecutive_invalid_attempts

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
            and bool(
                self._state.eligible_sentence_ids
                or self._state.read_chunk_ids
            )
        )

    @property
    def policy_attempt_budget_exhausted(self) -> bool:
        return (
            self._state.remaining_policy_attempt_budget == 0
            and self._state.remaining_step_budget > 0
            and self._state.remaining_retrieved_token_budget > 0
        )

    def record_attempt(self, event: AttemptEvent) -> StepRecord:
        """Apply one transition and append its audit record atomically."""

        state_before = self._state.model_copy(deep=True)
        updated = self._state_updater.apply(
            self._state,
            assessment=event.assessment,
            observation=event.observation,
            action_signature=event.action_signature,
            commit_assessment=event.commit_assessment,
            consume_step=event.consume_step,
        )
        record_step = (
            updated.step if event.consume_step else state_before.step + 1
        )
        record = StepRecord(
            step=record_step,
            policy_attempt=updated.policy_attempts,
            decision=event.decision,
            repaired_decision=event.repaired_decision,
            repair_code=event.repair_code,
            resolved_decision=event.resolved_decision,
            validation_status=event.validation_status,
            validation_error=event.validation_error,
            observation=event.observation,
            agent_visible_observation=(
                project_v3_observation_for_policy(
                    event.observation,
                    self._state_updater.substrate,
                )
                if self._semantic_memory_v3
                and self._state_updater.substrate is not None
                else project_observation_for_policy(
                    event.observation,
                    updated,
                    enabled_expansions=self._enabled_expansions,
                )
            ),
            state_before=state_before,
            state_after=updated.model_copy(deep=True),
            usage=event.usage,
            policy_view=event.policy_view,
            context_reference_map=event.context_reference_map,
            policy_stages=list(event.policy_stages),
        )
        self._state = updated
        self._trajectory.append(record)
        self._total_usage = self._total_usage + event.usage
        if event.invalid_attempt:
            self._consecutive_invalid_attempts += 1
        else:
            self._consecutive_invalid_attempts = 0
        return record

    def add_usage(
        self,
        usage: Usage,
        *,
        attach_to_latest_attempt: bool = False,
    ) -> None:
        """Account for provider usage not represented by a new transition."""

        self._total_usage = self._total_usage + usage
        if attach_to_latest_attempt:
            if not self._trajectory:
                raise RuntimeError("cannot attach usage without an attempt")
            latest = self._trajectory[-1]
            latest.usage = latest.usage + usage

    def result(
        self,
        *,
        reason: TerminationReason,
        answer: str | None = None,
        selected_refs: list[EvidenceRef] | None = None,
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
            selected_evidence_refs=selected_refs or [],
            resolved_evidence=resolved_evidence or [],
            trajectory=list(self._trajectory),
            usage=self._total_usage.model_copy(deep=True),
            final_state=self._state.model_copy(deep=True),
            error_code=error_code,
            error_message=error_message,
        )


class EpisodeStateManagerFactory:
    """Fixed dependency that creates isolated state owners per episode."""

    def __init__(
        self,
        *,
        state_updater: StateUpdater,
        enabled_expansions: tuple[ExpansionKind, ...] = (
            DEFAULT_ENABLED_EXPANSIONS
        ),
        semantic_memory_v3: bool = False,
    ) -> None:
        self._state_updater = state_updater
        self._enabled_expansions = tuple(enabled_expansions)
        self._semantic_memory_v3 = semantic_memory_v3

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
            enabled_expansions=self._enabled_expansions,
            semantic_memory_v3=self._semantic_memory_v3,
        )
