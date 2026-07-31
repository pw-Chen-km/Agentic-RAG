"""The provider-neutral one-decision-per-step agent controller."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from agentic_rag.agent.answer import (
    AnswerGenerationError,
    AnswerGenerator,
)
from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.models import (
    ControllerState,
    EpisodeResult,
    FinishAction,
    Observation,
    ObservationStatus,
    PolicyView,
    StepRecord,
    TerminationReason,
    Usage,
    ValidationStatus,
)
from agentic_rag.agent.policy import (
    PolicyClient,
    PolicyConfigurationError,
    PolicyResponseError,
    PolicyTransportError,
)
from agentic_rag.agent.router import ActionRouter
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.state import StateUpdater
from agentic_rag.agent.validator import DecisionValidator
from agentic_rag.errors import AgenticRAGError


class AgentController:
    def __init__(
        self,
        *,
        policy: PolicyClient,
        answer_generator: AnswerGenerator,
        context_builder: PolicyContextBuilder,
        validator: DecisionValidator,
        router: ActionRouter,
        evidence_resolver: EvidenceResolver,
        skill: SkillDocument,
        max_steps: int = 10,
        max_retrieved_tokens: int = 12_000,
        state_updater: StateUpdater | None = None,
    ) -> None:
        self.policy = policy
        self.answer_generator = answer_generator
        self.context_builder = context_builder
        self.validator = validator
        self.router = router
        self.evidence_resolver = evidence_resolver
        self.skill = skill
        self.max_steps = max_steps
        self.max_retrieved_tokens = max_retrieved_tokens
        self.state_updater = state_updater or StateUpdater(router.substrate)

    def run_episode(
        self,
        question: str,
        scope_id: str,
        *,
        episode_id: str | None = None,
    ) -> EpisodeResult:
        if not question.strip():
            raise ValueError("question must not be blank")
        self.router.substrate.require_scope(scope_id)
        episode_id = episode_id or f"{scope_id}-{uuid4().hex}"
        state = ControllerState.initial(
            max_steps=self.max_steps,
            max_retrieved_tokens=self.max_retrieved_tokens,
        )
        trajectory: list[StepRecord] = []
        total_usage = Usage()

        while (
            state.remaining_step_budget > 0
            and state.remaining_retrieved_token_budget > 0
        ):
            state_before = state.model_copy(deep=True)
            built_context = self.context_builder.build(
                question,
                self.skill,
                state,
                trajectory,
                scope_id=scope_id,
            )
            messages = built_context.messages
            policy_view = built_context.policy_view
            try:
                decision = self.policy.decide(messages)
            except PolicyResponseError as exc:
                policy_usage = self._policy_usage()
                total_usage = total_usage + policy_usage
                observation = Observation(
                    action_id=f"step-{state.step + 1}",
                    status=ObservationStatus.INVALID_ACTION,
                    error_code="invalid_policy_response",
                    message=str(exc),
                )
                state = self.state_updater.apply(
                    state,
                    assessment=None,
                    observation=observation,
                    action_signature=None,
                    commit_assessment=False,
                )
                trajectory.append(
                    StepRecord(
                        step=state.step,
                        decision=None,
                        validation_status=ValidationStatus.INVALID,
                        validation_error=str(exc),
                        observation=observation,
                        state_before=state_before,
                        state_after=state.model_copy(deep=True),
                        usage=policy_usage,
                        policy_view=policy_view,
                    )
                )
                continue
            except (PolicyConfigurationError, PolicyTransportError) as exc:
                total_usage = total_usage + self._policy_usage()
                return self._result(
                    episode_id=episode_id,
                    question=question,
                    scope_id=scope_id,
                    reason=TerminationReason.POLICY_ERROR,
                    state=state,
                    trajectory=trajectory,
                    usage=total_usage,
                    error_code="policy_error",
                    error_message=str(exc),
                )
            except Exception as exc:  # defensive provider boundary
                total_usage = total_usage + self._policy_usage()
                return self._result(
                    episode_id=episode_id,
                    question=question,
                    scope_id=scope_id,
                    reason=TerminationReason.POLICY_ERROR,
                    state=state,
                    trajectory=trajectory,
                    usage=total_usage,
                    error_code="policy_error",
                    error_message=str(exc),
                )

            policy_usage = self._policy_usage()
            total_usage = total_usage + policy_usage
            validation = self.validator.validate(decision, state, scope_id)
            if not validation.ok:
                status = (
                    ObservationStatus.DUPLICATE_ACTION
                    if validation.code == "duplicate_action"
                    else ObservationStatus.INVALID_ACTION
                )
                observation = Observation(
                    action_id=f"step-{state.step + 1}",
                    status=status,
                    action=decision.action,
                    error_code=validation.code,
                    message=validation.message,
                )
                state = self.state_updater.apply(
                    state,
                    assessment=decision.assessment,
                    observation=observation,
                    # Only validated/executed actions participate in duplicate
                    # suppression. A state-dependent invalid action (for
                    # example READ before its Chunk becomes visible) must be
                    # retryable after the missing prerequisite is satisfied.
                    action_signature=None,
                    commit_assessment=False,
                )
                trajectory.append(
                    StepRecord(
                        step=state.step,
                        decision=decision,
                        validation_status=ValidationStatus.INVALID,
                        validation_error=validation.message,
                        observation=observation,
                        state_before=state_before,
                        state_after=state.model_copy(deep=True),
                        usage=policy_usage,
                        policy_view=policy_view,
                    )
                )
                continue

            if isinstance(decision.action, FinishAction):
                resolved = self.evidence_resolver.resolve(
                    decision.action.evidence_refs, state, scope_id
                )
                observation = Observation(
                    action_id=f"step-{state.step + 1}",
                    status=ObservationStatus.OK,
                    action=decision.action,
                    results=[
                        item.model_dump(mode="json") for item in resolved
                    ],
                    metadata={"finish": True},
                )
                state = self.state_updater.apply(
                    state,
                    assessment=decision.assessment,
                    observation=observation,
                    action_signature=validation.signature,
                )
                trajectory.append(
                    StepRecord(
                        step=state.step,
                        decision=decision,
                        validation_status=ValidationStatus.VALID,
                        observation=observation,
                        state_before=state_before,
                        state_after=state.model_copy(deep=True),
                        usage=policy_usage,
                        policy_view=policy_view,
                    )
                )
                try:
                    answer = self.answer_generator.generate(
                        question, resolved
                    )
                except AnswerGenerationError as exc:
                    answer_usage = self._answer_usage()
                    total_usage = total_usage + answer_usage
                    trajectory[-1].usage = (
                        trajectory[-1].usage + answer_usage
                    )
                    return self._result(
                        episode_id=episode_id,
                        question=question,
                        scope_id=scope_id,
                        reason=TerminationReason.ANSWER_GENERATION_ERROR,
                        state=state,
                        trajectory=trajectory,
                        usage=total_usage,
                        selected_refs=list(
                            decision.action.evidence_refs
                        ),
                        resolved_evidence=resolved,
                        error_code="answer_generation_error",
                        error_message=str(exc),
                    )
                except Exception as exc:  # defensive generator boundary
                    answer_usage = self._answer_usage()
                    total_usage = total_usage + answer_usage
                    trajectory[-1].usage = (
                        trajectory[-1].usage + answer_usage
                    )
                    return self._result(
                        episode_id=episode_id,
                        question=question,
                        scope_id=scope_id,
                        reason=TerminationReason.ANSWER_GENERATION_ERROR,
                        state=state,
                        trajectory=trajectory,
                        usage=total_usage,
                        selected_refs=list(
                            decision.action.evidence_refs
                        ),
                        resolved_evidence=resolved,
                        error_code="answer_generation_error",
                        error_message=str(exc),
                    )
                answer_usage = self._answer_usage()
                total_usage = total_usage + answer_usage
                trajectory[-1].usage = trajectory[-1].usage + answer_usage
                return self._result(
                    episode_id=episode_id,
                    question=question,
                    scope_id=scope_id,
                    reason=TerminationReason.FINISH,
                    state=state,
                    trajectory=trajectory,
                    usage=total_usage,
                    answer=answer,
                    selected_refs=list(decision.action.evidence_refs),
                    resolved_evidence=resolved,
                )

            try:
                observation = self.router.execute(
                    decision.action,
                    state,
                    question=question,
                    scope_id=scope_id,
                    action_id=f"step-{state.step + 1}",
                )
            except AgenticRAGError as exc:
                state, trajectory = self._record_runtime_error(
                    state=state,
                    state_before=state_before,
                    trajectory=trajectory,
                    decision=decision,
                    validation_signature=validation.signature,
                    policy_usage=policy_usage,
                    policy_view=policy_view,
                    error_code=exc.code,
                    error_message=exc.message,
                )
                return self._result(
                    episode_id=episode_id,
                    question=question,
                    scope_id=scope_id,
                    reason=TerminationReason.RUNTIME_ERROR,
                    state=state,
                    trajectory=trajectory,
                    usage=total_usage,
                    error_code=exc.code,
                    error_message=exc.message,
                )
            except Exception as exc:
                state, trajectory = self._record_runtime_error(
                    state=state,
                    state_before=state_before,
                    trajectory=trajectory,
                    decision=decision,
                    validation_signature=validation.signature,
                    policy_usage=policy_usage,
                    policy_view=policy_view,
                    error_code="runtime_error",
                    error_message=str(exc),
                )
                return self._result(
                    episode_id=episode_id,
                    question=question,
                    scope_id=scope_id,
                    reason=TerminationReason.RUNTIME_ERROR,
                    state=state,
                    trajectory=trajectory,
                    usage=total_usage,
                    error_code="runtime_error",
                    error_message=str(exc),
                )

            step_usage = policy_usage + Usage(
                retrieved_tokens=observation.retrieved_tokens
            )
            total_usage = total_usage + Usage(
                retrieved_tokens=observation.retrieved_tokens
            )
            state = self.state_updater.apply(
                state,
                assessment=decision.assessment,
                observation=observation,
                action_signature=validation.signature,
            )
            trajectory.append(
                StepRecord(
                    step=state.step,
                    decision=decision,
                    validation_status=ValidationStatus.VALID,
                    observation=observation,
                    state_before=state_before,
                    state_after=state.model_copy(deep=True),
                    usage=step_usage,
                    policy_view=policy_view,
                )
            )

        return self._result(
            episode_id=episode_id,
            question=question,
            scope_id=scope_id,
            reason=TerminationReason.BUDGET_EXHAUSTED,
            state=state,
            trajectory=trajectory,
            usage=total_usage,
            error_code="budget_exhausted",
            error_message="The episode exhausted its step or retrieval budget",
        )

    def initial_messages(
        self, question: str, scope_id: str | None = None
    ) -> Sequence:
        state = ControllerState.initial(
            max_steps=self.max_steps,
            max_retrieved_tokens=self.max_retrieved_tokens,
        )
        return self.context_builder.build(
            question,
            self.skill,
            state,
            [],
            scope_id=scope_id,
        ).messages

    def _policy_usage(self) -> Usage:
        usage = getattr(self.policy, "last_usage", Usage())
        return usage if isinstance(usage, Usage) else Usage()

    def _answer_usage(self) -> Usage:
        usage = getattr(self.answer_generator, "last_usage", Usage())
        return usage if isinstance(usage, Usage) else Usage()

    def _record_runtime_error(
        self,
        *,
        state: ControllerState,
        state_before: ControllerState,
        trajectory: list[StepRecord],
        decision,
        validation_signature: str | None,
        policy_usage: Usage,
        policy_view: PolicyView,
        error_code: str,
        error_message: str,
    ) -> tuple[ControllerState, list[StepRecord]]:
        observation = Observation(
            action_id=f"step-{state.step + 1}",
            status=ObservationStatus.ERROR,
            action=decision.action,
            error_code=error_code,
            message=error_message,
        )
        updated = self.state_updater.apply(
            state,
            assessment=decision.assessment,
            observation=observation,
            action_signature=validation_signature,
        )
        trajectory.append(
            StepRecord(
                step=updated.step,
                decision=decision,
                validation_status=ValidationStatus.VALID,
                observation=observation,
                state_before=state_before,
                state_after=updated.model_copy(deep=True),
                usage=policy_usage,
                policy_view=policy_view,
            )
        )
        return updated, trajectory

    @staticmethod
    def _result(
        *,
        episode_id: str,
        question: str,
        scope_id: str,
        reason: TerminationReason,
        state: ControllerState,
        trajectory: list[StepRecord],
        usage: Usage,
        answer: str | None = None,
        selected_refs: list | None = None,
        resolved_evidence: list | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> EpisodeResult:
        return EpisodeResult(
            episode_id=episode_id,
            query=question,
            scope_id=scope_id,
            termination_reason=reason,
            answer=answer,
            selected_evidence_refs=selected_refs or [],
            resolved_evidence=resolved_evidence or [],
            trajectory=trajectory,
            usage=usage,
            final_state=state.model_copy(deep=True),
            error_code=error_code,
            error_message=error_message,
        )
