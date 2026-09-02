"""Stateless orchestration for the one-decision-per-turn agent loop."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.models import (
    ActionSpaceMode,
    EpisodeResult,
    EpisodeState,
    Message,
    Observation,
    ObservationStatus,
    ResolvedFinishAction,
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
from agentic_rag.agent.references import ReferenceResolutionError, resolve_decision
from agentic_rag.agent.router import ActionRouter
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.state import StateUpdater
from agentic_rag.agent.state_management import (
    AttemptEvent,
    EpisodeStateManager,
    EpisodeStateManagerFactory,
)
from agentic_rag.agent.validator import DecisionValidator


BUDGET_FINALIZE_INSTRUCTION = (
    "Retrieval is closed because its budget is exhausted. Return FINISH using "
    "only currently visible eligible evidence. Do not SEARCH, EXPAND, or READ."
)


class AgentController:
    """Coordinate collaborators without owning episode state or history."""

    def __init__(
        self,
        *,
        policy: PolicyClient,
        context_builder: PolicyContextBuilder,
        validator: DecisionValidator,
        router: ActionRouter,
        evidence_resolver: EvidenceResolver,
        skill: SkillDocument,
        max_steps: int = 10,
        max_policy_attempts: int = 12,
        max_retrieved_tokens: int = 12_000,
        state_updater: StateUpdater | None = None,
    ) -> None:
        self.policy = policy
        self.context_builder = context_builder
        self.validator = validator
        self.router = router
        self.evidence_resolver = evidence_resolver
        self.skill = skill
        self.max_steps = max_steps
        self.max_policy_attempts = max_policy_attempts
        self.max_retrieved_tokens = max_retrieved_tokens
        self.state_manager_factory = EpisodeStateManagerFactory(
            state_updater or StateUpdater(router.substrate)
        )

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
        manager = self.state_manager_factory.create(
            episode_id=episode_id or f"{scope_id}-{uuid4().hex}",
            question=question,
            scope_id=scope_id,
            max_steps=self.max_steps,
            max_policy_attempts=self.max_policy_attempts,
            max_retrieved_tokens=self.max_retrieved_tokens,
        )

        while (
            manager.can_continue
            and not manager.should_reserve_policy_attempt_for_finalize
        ):
            state = manager.snapshot()
            built = self.context_builder.build(
                question,
                self.skill,
                state,
                manager.trajectory_snapshot(),
                scope_id=scope_id,
            )
            try:
                decision = self.policy.decide(
                    built.messages,
                    decision_format=built.decision_format,
                )
            except PolicyResponseError as exc:
                usage = self._policy_usage()
                observation = Observation(
                    action_id=manager.next_action_id,
                    status=ObservationStatus.INVALID_ACTION,
                    error_code="invalid_policy_response",
                    message=str(exc),
                )
                manager.record_attempt(
                    AttemptEvent(
                        decision=None,
                        resolved_decision=None,
                        validation_status=ValidationStatus.INVALID,
                        validation_error=str(exc),
                        observation=observation,
                        assessment=None,
                        action_signature=None,
                        usage=usage,
                        policy_view=built.policy_view,
                        context_reference_map=built.reference_map,
                        available_action_space=built.available_action_space,
                        decision_schema_sha256=built.decision_schema_sha256,
                        commit_assessment=False,
                        consume_step=False,
                        invalid_attempt=True,
                    )
                )
                continue
            except (PolicyConfigurationError, PolicyTransportError) as exc:
                manager.add_usage(self._policy_usage())
                return manager.result(
                    reason=TerminationReason.POLICY_ERROR,
                    error_code="policy_error",
                    error_message=str(exc),
                )
            except Exception as exc:
                manager.add_usage(self._policy_usage())
                return manager.result(
                    reason=TerminationReason.POLICY_ERROR,
                    error_code="policy_error",
                    error_message=str(exc),
                )

            usage = self._policy_usage()
            try:
                resolved = resolve_decision(decision, built.reference_map)
            except ReferenceResolutionError as exc:
                observation = Observation(
                    action_id=manager.next_action_id,
                    status=ObservationStatus.INVALID_ACTION,
                    error_code=exc.code,
                    message=exc.message,
                )
                manager.record_attempt(
                    AttemptEvent(
                        decision=decision,
                        resolved_decision=None,
                        validation_status=ValidationStatus.INVALID,
                        validation_error=exc.message,
                        observation=observation,
                        assessment=decision.assessment,
                        action_signature=None,
                        usage=usage,
                        policy_view=built.policy_view,
                        context_reference_map=built.reference_map,
                        available_action_space=built.available_action_space,
                        decision_schema_sha256=built.decision_schema_sha256,
                        commit_assessment=False,
                        consume_step=False,
                        invalid_attempt=True,
                    )
                )
                continue

            validation = self.validator.validate(resolved, state, scope_id)
            if not validation.ok:
                observation = Observation(
                    action_id=manager.next_action_id,
                    status=(
                        ObservationStatus.DUPLICATE_ACTION
                        if validation.code == "duplicate_action"
                        else ObservationStatus.INVALID_ACTION
                    ),
                    action=resolved.action,
                    error_code=validation.code,
                    message=validation.message,
                )
                manager.record_attempt(
                    AttemptEvent(
                        decision=decision,
                        resolved_decision=resolved,
                        validation_status=ValidationStatus.INVALID,
                        validation_error=validation.message,
                        observation=observation,
                        assessment=resolved.assessment,
                        action_signature=None,
                        usage=usage,
                        policy_view=built.policy_view,
                        context_reference_map=built.reference_map,
                        available_action_space=built.available_action_space,
                        decision_schema_sha256=built.decision_schema_sha256,
                        commit_assessment=False,
                        consume_step=False,
                        invalid_attempt=True,
                    )
                )
                continue

            if isinstance(resolved.action, ResolvedFinishAction):
                return self._finish_result(
                    manager=manager,
                    state=state,
                    decision=decision,
                    resolved=resolved,
                    signature=validation.signature,
                    usage=usage,
                    built=built,
                )

            try:
                observation = self.router.execute(
                    resolved.action,
                    state,
                    question=question,
                    scope_id=scope_id,
                    action_id=manager.next_action_id,
                )
            except Exception as exc:
                self._record_runtime_error(
                    manager=manager,
                    decision=decision,
                    resolved=resolved,
                    signature=validation.signature,
                    usage=usage,
                    built=built,
                    error_code=getattr(exc, "code", "runtime_error"),
                    error_message=str(exc),
                )
                return manager.result(
                    reason=TerminationReason.RUNTIME_ERROR,
                    error_code=getattr(exc, "code", "runtime_error"),
                    error_message=str(exc),
                )

            manager.record_attempt(
                AttemptEvent(
                    decision=decision,
                    resolved_decision=resolved,
                    validation_status=ValidationStatus.VALID,
                    validation_error=None,
                    observation=observation,
                    assessment=resolved.assessment,
                    action_signature=validation.signature,
                    usage=usage + Usage(retrieved_tokens=observation.retrieved_tokens),
                    policy_view=built.policy_view,
                    context_reference_map=built.reference_map,
                    available_action_space=built.available_action_space,
                    decision_schema_sha256=built.decision_schema_sha256,
                )
            )

        if manager.can_attempt_budget_finalize:
            finalized = self._try_budget_finalize(manager)
            if finalized is not None:
                return finalized
        return manager.result(
            reason=TerminationReason.BUDGET_EXHAUSTED,
            error_code=(
                "policy_attempt_budget_exhausted"
                if manager.policy_attempt_budget_exhausted
                else "budget_exhausted"
            ),
            error_message="The episode exhausted its configured budget",
        )

    def initial_messages(self, question: str, scope_id: str | None = None) -> Sequence[Message]:
        state = EpisodeState.initial(
            max_steps=self.max_steps,
            max_policy_attempts=self.max_policy_attempts,
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

    def _try_budget_finalize(self, manager: EpisodeStateManager) -> EpisodeResult | None:
        state = manager.snapshot()
        built = self.context_builder.build(
            manager.question,
            self.skill,
            state,
            manager.trajectory_snapshot(),
            scope_id=manager.scope_id,
            action_space_mode=ActionSpaceMode.BUDGET_FINALIZE,
        )
        messages = [*built.messages, Message(role="user", content=BUDGET_FINALIZE_INSTRUCTION)]
        try:
            decision = self.policy.decide(messages, decision_format=built.decision_format)
            usage = self._policy_usage()
            resolved = resolve_decision(decision, built.reference_map)
            validation = self.validator.validate(resolved, state, manager.scope_id)
        except Exception as exc:
            manager.add_usage(self._policy_usage())
            return manager.result(
                reason=TerminationReason.BUDGET_EXHAUSTED,
                error_code="budget_finalize_failed",
                error_message=str(exc),
            )
        if validation.ok and isinstance(resolved.action, ResolvedFinishAction):
            return self._finish_result(
                manager=manager,
                state=state,
                decision=decision,
                resolved=resolved,
                signature=validation.signature,
                usage=usage,
                built=built,
                consume_step=False,
            )
        manager.add_usage(usage)
        return manager.result(
            reason=TerminationReason.BUDGET_EXHAUSTED,
            error_code="budget_finalize_requires_finish",
            error_message=(validation.message or "Budget finalization requires FINISH"),
        )

    def _finish_result(
        self,
        *,
        manager: EpisodeStateManager,
        state: EpisodeState,
        decision,
        resolved,
        signature: str | None,
        usage: Usage,
        built,
        consume_step: bool = True,
    ) -> EpisodeResult:
        evidence_refs = list(resolved.action.evidence_refs)
        evidence = self.evidence_resolver.resolve(evidence_refs, state, manager.scope_id)
        observation = Observation(
            action_id=manager.next_action_id,
            status=ObservationStatus.OK,
            action=resolved.action,
            results=[item.model_dump(mode="json") for item in evidence],
            metadata={"finish": True, "budget_finalize": not consume_step},
        )
        manager.record_attempt(
            AttemptEvent(
                decision=decision,
                resolved_decision=resolved,
                validation_status=ValidationStatus.VALID,
                validation_error=None,
                observation=observation,
                assessment=resolved.assessment,
                action_signature=signature,
                usage=usage,
                policy_view=built.policy_view,
                context_reference_map=built.reference_map,
                available_action_space=built.available_action_space,
                decision_schema_sha256=built.decision_schema_sha256,
                consume_step=consume_step,
            )
        )
        return manager.result(
            reason=TerminationReason.FINISH,
            answer=resolved.action.answer,
            evidence_refs=evidence_refs,
            resolved_evidence=evidence,
        )

    def _record_runtime_error(
        self,
        *,
        manager: EpisodeStateManager,
        decision,
        resolved,
        signature: str | None,
        usage: Usage,
        built,
        error_code: str,
        error_message: str,
    ) -> None:
        manager.record_attempt(
            AttemptEvent(
                decision=decision,
                resolved_decision=resolved,
                validation_status=ValidationStatus.VALID,
                validation_error=None,
                observation=Observation(
                    action_id=manager.next_action_id,
                    status=ObservationStatus.ERROR,
                    action=resolved.action,
                    error_code=error_code,
                    message=error_message,
                ),
                assessment=resolved.assessment,
                action_signature=signature,
                usage=usage,
                policy_view=built.policy_view,
                context_reference_map=built.reference_map,
                available_action_space=built.available_action_space,
                decision_schema_sha256=built.decision_schema_sha256,
            )
        )
