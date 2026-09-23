"""Two-level closed-loop controller for Options v1.

The legacy controller remains the implementation for the original one-call
policy.  This controller deliberately owns only the option lifecycle and
delegates retrieval execution, reference resolution and validation to the
existing components.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.models import (
    EpisodeResult,
    EpisodeState,
    FinishAction,
    Message,
    Observation,
    ObservationStatus,
    PolicyDecision,
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
from agentic_rag.options.catalog import OptionCatalog
from agentic_rag.options.context import OptionContextBuilder
from agentic_rag.options.models import (
    OptionEvent,
    OptionPolicyDecision,
    OptionSelectorDecision,
    OptionTrace,
)


class OptionsAgentController:
    """Run selector -> option policy -> environment in a closed loop."""

    def __init__(
        self,
        *,
        policy: PolicyClient,
        context_builder: OptionContextBuilder,
        base_context_builder: PolicyContextBuilder,
        validator: DecisionValidator,
        router: ActionRouter,
        evidence_resolver: EvidenceResolver,
        skill: SkillDocument,
        catalog: OptionCatalog,
        max_steps: int = 10,
        max_policy_attempts: int = 12,
        max_retrieved_tokens: int = 12_000,
    ) -> None:
        self.policy = policy
        self.context_builder = context_builder
        self.base_context_builder = base_context_builder
        self.validator = validator
        self.router = router
        self.evidence_resolver = evidence_resolver
        self.skill = skill
        self.catalog = catalog
        self.max_steps = max_steps
        self.max_policy_attempts = max_policy_attempts
        self.max_retrieved_tokens = max_retrieved_tokens
        self.state_manager_factory = EpisodeStateManagerFactory(StateUpdater(router.substrate))

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
        trace = OptionTrace(catalog_sha256=self.catalog.sha256())
        current_option: str | None = None
        last_option: str | None = None
        last_option_status: str | None = None
        self._pending_result: EpisodeResult | None = None

        while manager.can_continue:
            state = manager.snapshot()
            trajectory = manager.trajectory_snapshot()
            try:
                if current_option is None:
                    selected, last_option = self._select_option(
                        manager, trace, question, scope_id, state, trajectory,
                        last_option, last_option_status,
                    )
                    current_option = selected
                    if selected == "FALLBACK":
                        self._run_fallback(
                            manager, trace, question, scope_id,
                            manager.snapshot(), manager.trajectory_snapshot(), last_option,
                        )
                        last_option_status = "CONTINUE"
                        current_option = None
                    continue

                outcome = self._run_option_policy(
                    manager,
                    trace,
                    question,
                    scope_id,
                    state,
                    trajectory,
                    current_option,
                    last_option,
                    last_option_status,
                )
                if outcome == "__FINISH__":
                    if self._pending_result is None:
                        raise RuntimeError("FINISH result was not recorded")
                    return self._pending_result
                if outcome in {"COMPLETE", "BLOCKED"}:
                    trace.events.append(
                        OptionEvent(
                            step=manager.snapshot().step,
                            policy_attempt=manager.snapshot().policy_attempts,
                            event="completed" if outcome == "COMPLETE" else "blocked",
                            option_id=current_option,
                            status=outcome,
                        )
                    )
                    last_option = current_option
                    last_option_status = outcome
                    current_option = None
                elif outcome == "CONTINUE":
                    last_option = current_option
                    last_option_status = "CONTINUE"
            except (PolicyConfigurationError, PolicyTransportError) as exc:
                manager.add_usage(self._policy_usage())
                return self._finish_result(manager.result(
                    reason=TerminationReason.POLICY_ERROR,
                    error_code="policy_error",
                    error_message=str(exc),
                ), trace)
            except PolicyResponseError as exc:
                manager.add_usage(self._policy_usage())
                return self._finish_result(manager.result(
                    reason=TerminationReason.POLICY_ERROR,
                    error_code="invalid_policy_response",
                    error_message=str(exc),
                ), trace)
            except Exception as exc:
                manager.add_usage(self._policy_usage())
                return self._finish_result(manager.result(
                    reason=TerminationReason.RUNTIME_ERROR,
                    error_code="options_runtime_error",
                    error_message=str(exc),
                ), trace)

        if manager.can_attempt_budget_finalize:
            final = self._try_answer_finalize(manager, trace, question, scope_id, last_option)
            if final is not None:
                return final
        return self._finish_result(manager.result(
            reason=TerminationReason.BUDGET_EXHAUSTED,
            error_code="budget_exhausted",
            error_message="The Options v1 episode exhausted its configured budget",
        ), trace)

    def initial_messages(self, question: str, scope_id: str) -> Sequence[Message]:
        state = EpisodeState.initial(
            max_steps=self.max_steps,
            max_policy_attempts=self.max_policy_attempts,
            max_retrieved_tokens=self.max_retrieved_tokens,
        )
        _, messages, _, _, _ = self.context_builder.selector(
            question, self.skill, state, [], scope_id=scope_id,
            last_option_id=None, last_option_status=None,
        )
        return messages

    def _select_option(
        self,
        manager: EpisodeStateManager,
        trace: OptionTrace,
        question: str,
        scope_id: str,
        state: EpisodeState,
        trajectory: Sequence[Any],
        last_option: str | None,
        last_option_status: str | None,
    ) -> tuple[str, str]:
        _, messages, fmt, _, ids = self.context_builder.selector(
            question, self.skill, state, trajectory, scope_id=scope_id,
            last_option_id=last_option, last_option_status=last_option_status,
        )
        raw = self.policy.decide(messages, decision_format=fmt)
        usage = self._policy_usage()
        selected = OptionSelectorDecision.model_validate(raw.model_dump(mode="json"))
        if selected.option_id not in ids:
            raise PolicyResponseError(f"selector returned unavailable option {selected.option_id!r}")
        manager.consume_policy_attempt(usage)
        trace.selector_calls += 1
        trace.events.append(
            OptionEvent(
                step=manager.snapshot().step,
                policy_attempt=manager.snapshot().policy_attempts,
                event="selected",
                option_id=selected.option_id,
                reason="selector decision",
                usage=usage.model_dump(mode="json"),
            )
        )
        return selected.option_id, selected.option_id

    def _run_option_policy(
        self,
        manager: EpisodeStateManager,
        trace: OptionTrace,
        question: str,
        scope_id: str,
        state: EpisodeState,
        trajectory: Sequence[Any],
        option_id: str,
        last_option: str | None,
        last_option_status: str | None,
    ) -> str:
        built = self.context_builder.policy(
            question, self.skill, state, trajectory, scope_id=scope_id,
            option_id=option_id, last_option_id=last_option,
            last_option_status=last_option_status,
        )
        raw = self.policy.decide(built.messages, decision_format=built.decision_format)
        usage = self._policy_usage()
        decision = OptionPolicyDecision.model_validate(raw.model_dump(mode="json"))
        if decision.option_status != "CONTINUE":
            manager.consume_policy_attempt(usage)
            manager.commit_assessment(decision.assessment)
            trace.policy_calls += 1
            return decision.option_status
        if decision.action is None:
            raise PolicyResponseError("CONTINUE option policy response has no action")
        allowed = set(self.catalog.get(option_id).primitive_actions)
        action_type = str(getattr(decision.action, "type", ""))
        if action_type not in allowed or (action_type == "FINISH" and option_id != "O5_ANSWER"):
            raise PolicyResponseError(f"{option_id} cannot execute {action_type}")
        trace.policy_calls += 1
        legacy_decision = PolicyDecision(assessment=decision.assessment, action=decision.action)
        try:
            resolved = resolve_decision(legacy_decision, built.base.reference_map)
        except ReferenceResolutionError as exc:
            self._record_invalid(manager, legacy_decision, built.base, usage, option_id, "CONTINUE", exc.code, exc.message)
            return "CONTINUE"
        validation = self.validator.validate(resolved, state, scope_id)
        if not validation.ok:
            self._record_invalid(manager, legacy_decision, built.base, usage, option_id, "CONTINUE", validation.code, validation.message)
            return "CONTINUE"
        if isinstance(resolved.action, ResolvedFinishAction):
            result = self._finish_with_action(manager, trace, legacy_decision, resolved, validation.signature, usage, built.base, option_id)
            # The caller will return the result only through the episode loop;
            # FINISH is signalled with a private marker below.
            self._pending_result = result
            return "__FINISH__"
        try:
            observation = self.router.execute(
                resolved.action, state, question=question, scope_id=scope_id,
                action_id=manager.next_action_id,
            )
        except Exception as exc:
            observation = Observation(
                action_id=manager.next_action_id,
                status=ObservationStatus.ERROR,
                action=resolved.action,
                error_code=getattr(exc, "code", "runtime_error"),
                message=str(exc),
            )
        record = manager.record_attempt(AttemptEvent(
            decision=legacy_decision, resolved_decision=resolved,
            validation_status=ValidationStatus.VALID,
            validation_error=None, observation=observation,
            assessment=decision.assessment, action_signature=validation.signature,
            usage=usage + Usage(retrieved_tokens=observation.retrieved_tokens),
            policy_view=built.base.policy_view,
            context_reference_map=built.base.reference_map,
            available_action_space=built.base.available_action_space,
            decision_schema_sha256=built.decision_schema_sha256,
            option_id=option_id, option_status="CONTINUE",
        ))
        trace.events.append(OptionEvent(
            step=record.step, policy_attempt=record.policy_attempt,
            event="continue", option_id=option_id, option_status="CONTINUE",
            primitive_action=resolved.action.model_dump(mode="json"),
            outcome=observation.model_dump(mode="json"),
            usage=usage.model_dump(mode="json"),
        ))
        return "CONTINUE"

    def _run_fallback(
        self, manager: EpisodeStateManager, trace: OptionTrace, question: str,
        scope_id: str, state: EpisodeState, trajectory: Sequence[Any], last_option: str | None,
    ) -> None:
        # FALLBACK is deliberately a single primitive-action escape hatch; it
        # should not receive the full option catalogue and accidentally turn
        # back into a second option-selection prompt.
        base = self.base_context_builder.build(
            question,
            "Fallback mode: choose exactly one legal primitive retrieval action.",
            state,
            trajectory,
            scope_id=scope_id,
        )
        raw = self.policy.decide(base.messages, decision_format=base.decision_format)
        usage = self._policy_usage()
        decision = PolicyDecision.model_validate(raw.model_dump(mode="json"))
        if isinstance(decision.action, FinishAction):
            raise PolicyResponseError("FALLBACK cannot execute FINISH")
        try:
            resolved = resolve_decision(decision, base.reference_map)
            validation = self.validator.validate(resolved, state, scope_id)
            if not validation.ok or isinstance(resolved.action, ResolvedFinishAction):
                raise ReferenceResolutionError("fallback_action_invalid", validation.message if not validation.ok else "FALLBACK cannot FINISH")
            observation = self.router.execute(resolved.action, state, question=question, scope_id=scope_id, action_id=manager.next_action_id)
        except Exception as exc:
            observation = Observation(action_id=manager.next_action_id, status=ObservationStatus.INVALID_ACTION, error_code=getattr(exc, "code", "fallback_error"), message=str(exc))
            resolved = None
            validation = None
        record = manager.record_attempt(AttemptEvent(
            decision=decision, resolved_decision=resolved,
            validation_status=ValidationStatus.VALID if validation is not None else ValidationStatus.INVALID,
            validation_error=None if validation is not None else observation.message,
            observation=observation, assessment=decision.assessment,
            action_signature=validation.signature if validation is not None else None,
            usage=usage + Usage(retrieved_tokens=observation.retrieved_tokens),
            policy_view=base.policy_view, context_reference_map=base.reference_map,
            available_action_space=base.available_action_space,
            decision_schema_sha256=base.decision_schema_sha256,
            option_id="FALLBACK", option_status="CONTINUE",
        ))
        trace.policy_calls += 1
        trace.events.append(OptionEvent(
            step=record.step, policy_attempt=record.policy_attempt,
            event="fallback", option_id="FALLBACK", option_status="CONTINUE",
            primitive_action=(resolved.action.model_dump(mode="json") if resolved is not None else None),
            outcome=observation.model_dump(mode="json"), usage=usage.model_dump(mode="json"),
        ))

    def _record_invalid(self, manager: EpisodeStateManager, decision: PolicyDecision, built: Any, usage: Usage, option_id: str, status: str, code: str, message: str) -> None:
        observation = Observation(action_id=manager.next_action_id, status=ObservationStatus.INVALID_ACTION, action=None, error_code=code, message=message)
        manager.record_attempt(AttemptEvent(
            decision=decision, resolved_decision=None, validation_status=ValidationStatus.INVALID,
            validation_error=message, observation=observation, assessment=decision.assessment,
            action_signature=None, usage=usage, policy_view=built.policy_view,
            context_reference_map=built.reference_map, available_action_space=built.available_action_space,
            decision_schema_sha256=built.decision_schema_sha256, commit_assessment=False,
            consume_step=False, invalid_attempt=True, option_id=option_id, option_status=status,
        ))

    def _try_answer_finalize(self, manager: EpisodeStateManager, trace: OptionTrace, question: str, scope_id: str, last_option: str | None) -> EpisodeResult | None:
        state = manager.snapshot()
        if not (state.eligible_sentence_ids or state.read_chunk_ids):
            return None
        # The selected O5 policy schema exposes only FINISH, even though the
        # base builder still records the current structurally legal actions.
        built = self.context_builder.policy(
            question, self.skill, state, manager.trajectory_snapshot(),
            scope_id=scope_id, option_id="O5_ANSWER", last_option_id=last_option,
            last_option_status="COMPLETE",
        )
        try:
            raw = self.policy.decide(built.messages, decision_format=built.decision_format)
            usage = self._policy_usage()
            decision = OptionPolicyDecision.model_validate(raw.model_dump(mode="json"))
            if decision.option_status != "CONTINUE" or decision.action is None:
                manager.consume_policy_attempt(usage)
                return None
            legacy = PolicyDecision(assessment=decision.assessment, action=decision.action)
            resolved = resolve_decision(legacy, built.base.reference_map)
            validation = self.validator.validate(resolved, state, scope_id)
            if not validation.ok or not isinstance(resolved.action, ResolvedFinishAction):
                return None
            trace.policy_calls += 1
            return self._finish_with_action(manager, trace, legacy, resolved, validation.signature, usage, built.base, "O5_ANSWER")
        except Exception:
            manager.add_usage(self._policy_usage())
            return None

    def _finish_with_action(self, manager: EpisodeStateManager, trace: OptionTrace, decision: PolicyDecision, resolved: Any, signature: str | None, usage: Usage, built: Any, option_id: str) -> EpisodeResult:
        evidence = self.evidence_resolver.resolve(list(resolved.action.evidence_refs), manager.snapshot(), manager.scope_id)
        observation = Observation(action_id=manager.next_action_id, status=ObservationStatus.OK, action=resolved.action, results=[item.model_dump(mode="json") for item in evidence], metadata={"finish": True})
        record = manager.record_attempt(AttemptEvent(
            decision=decision, resolved_decision=resolved, validation_status=ValidationStatus.VALID,
            validation_error=None, observation=observation, assessment=decision.assessment,
            action_signature=signature, usage=usage, policy_view=built.policy_view,
            context_reference_map=built.reference_map, available_action_space=built.available_action_space,
            decision_schema_sha256=built.decision_schema_sha256, option_id=option_id, option_status="COMPLETE",
        ))
        trace.events.append(OptionEvent(step=record.step, policy_attempt=record.policy_attempt, event="complete", option_id=option_id, option_status="COMPLETE", primitive_action=resolved.action.model_dump(mode="json"), outcome=observation.model_dump(mode="json"), usage=usage.model_dump(mode="json")))
        return self._finish_result(manager.result(reason=TerminationReason.FINISH, answer=resolved.action.answer, evidence_refs=list(resolved.action.evidence_refs), resolved_evidence=evidence), trace)

    @staticmethod
    def _finish_result(result: EpisodeResult, trace: OptionTrace) -> EpisodeResult:
        result.option_trace = trace.model_dump(mode="json")
        return result

    def _policy_usage(self) -> Usage:
        usage = getattr(self.policy, "last_usage", Usage())
        return usage if isinstance(usage, Usage) else Usage()
