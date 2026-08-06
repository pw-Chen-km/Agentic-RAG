"""The provider-neutral one-decision-per-step agent controller."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from agentic_rag.agent.answer import AnswerGenerator
from agentic_rag.agent.context import (
    PolicyContextBuilder,
)
from agentic_rag.agent.context_resolution import (
    ContextIndexResolutionError,
    assessment_for_internal,
    resolve_v31_decision,
    resolve_v3_decision,
)
from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.models import (
    ActionSelection,
    ActionType,
    ControllerState,
    ContextReferenceMap,
    TypedContextReferenceMap,
    EpisodeResult,
    FinishAction,
    Message,
    Observation,
    ObservationStatus,
    PolicyViewOutput,
    PolicyStagePhase,
    PolicyStageRecord,
    TerminationReason,
    Usage,
    ValidationStatus,
    V31PolicyDecision,
    V3PolicyDecision,
    assemble_policy_decision,
)
from agentic_rag.agent.policy import (
    PolicyClient,
    PolicyConfigurationError,
    PolicyResponseError,
    PolicyTransportError,
    action_parameters_model,
)
from agentic_rag.agent.router import ActionRouter
from agentic_rag.agent.repair import DecisionRepairer
from agentic_rag.agent.skill import ProgressiveSkillBundle, SkillDocument
from agentic_rag.agent.state import StateUpdater
from agentic_rag.agent.state_management import (
    AttemptEvent,
    EpisodeStateManager,
    EpisodeStateManagerFactory,
)
from agentic_rag.agent.validator import DecisionValidator
from agentic_rag.agent.v22_context import V22BuiltContext, V22ContextBuilder
from agentic_rag.errors import AgenticRAGError


BUDGET_FINALIZE_INSTRUCTION = (
    "Retrieval is now closed because its budget is exhausted. "
    "Return a FINISH PolicyDecision using only the strongest "
    "currently eligible evidence. Do not return SEARCH, "
    "EXPAND, or READ, and do not invent supported_facts that "
    "are absent from the selected evidence."
)

V22_BUDGET_FINALIZE_INSTRUCTION = (
    "Retrieval is closed because its budget is exhausted. In this Stage 1 "
    "response, return an ActionSelection with action_type=FINISH, a semantic "
    "action_intent, and the strongest currently eligible evidence. Do not "
    "emit the answer or FINISH parameters yet; the FINISH skill will be "
    "loaded in Stage 2."
)


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
        skill_bundle: ProgressiveSkillBundle | None = None,
        max_steps: int = 10,
        max_policy_attempts: int | None = None,
        max_consecutive_invalid_attempts: int = 2,
        max_retrieved_tokens: int = 12_000,
        single_agent_v2: bool = False,
        single_agent_v22: bool = False,
        single_agent_v3: bool = False,
        single_agent_v3_typed_refs: bool = False,
        state_updater: StateUpdater | None = None,
    ) -> None:
        self.policy = policy
        self.answer_generator = answer_generator
        self.context_builder = context_builder
        self.validator = validator
        self.router = router
        self.evidence_resolver = evidence_resolver
        self.skill = skill
        self.skill_bundle = skill_bundle
        self.max_steps = max_steps
        self.max_policy_attempts = (
            max_steps + 2
            if max_policy_attempts is None
            else max_policy_attempts
        )
        self.max_consecutive_invalid_attempts = (
            max_consecutive_invalid_attempts
        )
        self.max_retrieved_tokens = max_retrieved_tokens
        self.single_agent_v2 = single_agent_v2
        self.single_agent_v22 = single_agent_v22
        self.single_agent_v3 = single_agent_v3
        self.single_agent_v3_typed_refs = single_agent_v3_typed_refs
        semantic_memory_v3 = single_agent_v3 or single_agent_v3_typed_refs
        enabled_single_agent_modes = sum(
            int(value)
            for value in (
                single_agent_v2,
                single_agent_v22,
                semantic_memory_v3,
            )
        )
        if enabled_single_agent_modes > 1:
            raise ValueError("single-agent workflow modes are exclusive")
        if single_agent_v22 and skill_bundle is None:
            raise ValueError("single_agent_v2_2 requires a progressive skill bundle")
        self.single_agent = (
            single_agent_v2 or single_agent_v22 or semantic_memory_v3
        )
        self.accumulate_selected_evidence = single_agent_v2 or single_agent_v22
        self.v22_context_builder = (
            V22ContextBuilder(context_builder) if single_agent_v22 else None
        )
        self.state_manager_factory = EpisodeStateManagerFactory(
            state_updater=state_updater
            or StateUpdater(
                router.substrate,
                accumulate_selected_evidence=self.accumulate_selected_evidence,
            ),
            enabled_expansions=tuple(validator.enabled_expansions),
            semantic_memory_v3=semantic_memory_v3,
        )
        self.decision_repairer = DecisionRepairer(
            tuple(validator.enabled_expansions),
            derive_status_from_action=(single_agent_v2 or semantic_memory_v3),
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
        episode_id = episode_id or f"{scope_id}-{uuid4().hex}"
        if self.single_agent_v22:
            return self._run_episode_v22(
                question=question,
                scope_id=scope_id,
                episode_id=episode_id,
            )
        state_manager = self.state_manager_factory.create(
            episode_id=episode_id,
            question=question,
            scope_id=scope_id,
            max_steps=self.max_steps,
            max_policy_attempts=self.max_policy_attempts,
            max_retrieved_tokens=self.max_retrieved_tokens,
        )

        while state_manager.can_continue:
            state = state_manager.snapshot()
            built_context = self.context_builder.build(
                question,
                self.skill,
                state,
                state_manager.trajectory_snapshot(),
                scope_id=scope_id,
            )
            messages = built_context.messages
            policy_view = built_context.policy_view
            try:
                decision = self.policy.decide(
                    messages,
                    decision_format=built_context.decision_format,
                )
            except PolicyResponseError as exc:
                policy_usage = self._policy_usage()
                observation = Observation(
                    action_id=state_manager.next_action_id,
                    status=ObservationStatus.INVALID_ACTION,
                    error_code="invalid_policy_response",
                    message=str(exc),
                )
                state_manager.record_attempt(
                    AttemptEvent(
                        decision=None,
                        repaired_decision=None,
                        repair_code=None,
                        resolved_decision=None,
                        validation_status=ValidationStatus.INVALID,
                        validation_error=str(exc),
                        observation=observation,
                        assessment=None,
                        action_signature=None,
                        usage=policy_usage,
                        policy_view=policy_view,
                        context_reference_map=built_context.reference_map,
                        commit_assessment=False,
                        consume_step=False,
                        invalid_attempt=True,
                    )
                )
                if (
                    not self.single_agent
                    and state_manager.consecutive_invalid_attempts
                    >= self.max_consecutive_invalid_attempts
                ):
                    return state_manager.result(
                        reason=TerminationReason.BUDGET_EXHAUSTED,
                        error_code="invalid_action_retry_exhausted",
                        error_message=(
                            "The policy exhausted its consecutive invalid "
                            "decision retry budget"
                        ),
                    )
                continue
            except (PolicyConfigurationError, PolicyTransportError) as exc:
                state_manager.add_usage(self._policy_usage())
                return state_manager.result(
                    reason=TerminationReason.POLICY_ERROR,
                    error_code="policy_error",
                    error_message=str(exc),
                )
            except Exception as exc:  # defensive provider boundary
                state_manager.add_usage(self._policy_usage())
                return state_manager.result(
                    reason=TerminationReason.POLICY_ERROR,
                    error_code="policy_error",
                    error_message=str(exc),
                )

            policy_usage = self._policy_usage()
            try:
                if self.single_agent_v3_typed_refs:
                    if not isinstance(decision, V31PolicyDecision):
                        raise ContextIndexResolutionError(
                            "reference_not_available",
                            "V3.1 workflow requires a V31PolicyDecision",
                        )
                    if built_context.reference_map is None:
                        raise RuntimeError(
                            "V3.1 context is missing its frozen reference map"
                        )
                    internal_decision = resolve_v31_decision(
                        decision,
                        built_context.reference_map,
                    )
                elif self.single_agent_v3:
                    if not isinstance(decision, V3PolicyDecision):
                        raise ContextIndexResolutionError(
                            "memory_index_out_of_range",
                            "V3 workflow requires a V3PolicyDecision",
                        )
                    if built_context.reference_map is None:
                        raise RuntimeError(
                            "V3 context is missing its frozen reference map"
                        )
                    internal_decision = resolve_v3_decision(
                        decision,
                        built_context.reference_map,
                    )
                else:
                    if isinstance(
                        decision,
                        (V3PolicyDecision, V31PolicyDecision),
                    ):
                        raise RuntimeError(
                            "Legacy workflows cannot execute semantic-memory "
                            "PolicyDecision"
                        )
                    internal_decision = decision
            except ContextIndexResolutionError as exc:
                observation = Observation(
                    action_id=state_manager.next_action_id,
                    status=ObservationStatus.INVALID_ACTION,
                    error_code=exc.code,
                    message=exc.message,
                )
                state_manager.record_attempt(
                    AttemptEvent(
                        decision=decision,
                        repaired_decision=None,
                        repair_code=None,
                        resolved_decision=None,
                        validation_status=ValidationStatus.INVALID,
                        validation_error=exc.message,
                        observation=observation,
                        assessment=assessment_for_internal(
                            decision.assessment
                        ),
                        action_signature=None,
                        usage=policy_usage,
                        policy_view=policy_view,
                        context_reference_map=built_context.reference_map,
                        commit_assessment=False,
                        consume_step=False,
                        invalid_attempt=True,
                    )
                )
                continue

            repair = self.decision_repairer.repair(internal_decision, state)
            effective_decision = repair.decision
            validation = self.validator.validate(
                effective_decision, state, scope_id
            )
            resolved_decision = validation.resolved_decision
            if not validation.ok:
                status = (
                    ObservationStatus.DUPLICATE_ACTION
                    if validation.code == "duplicate_action"
                    else ObservationStatus.INVALID_ACTION
                )
                observation = Observation(
                    action_id=state_manager.next_action_id,
                    status=status,
                    action=(
                        resolved_decision.action
                        if resolved_decision is not None
                        else effective_decision.action
                    ),
                    error_code=validation.code,
                    message=validation.message,
                )
                state_manager.record_attempt(
                    AttemptEvent(
                        decision=decision,
                        repaired_decision=(
                            effective_decision if repair.code else None
                        ),
                        repair_code=repair.code,
                        resolved_decision=resolved_decision,
                        validation_status=ValidationStatus.INVALID,
                        validation_error=validation.message,
                        observation=observation,
                        assessment=effective_decision.assessment,
                        # Only validated/executed actions participate in
                        # duplicate suppression. State-dependent invalid
                        # actions remain retryable after a prerequisite.
                        action_signature=None,
                        usage=policy_usage,
                        policy_view=policy_view,
                        context_reference_map=built_context.reference_map,
                        commit_assessment=False,
                        consume_step=False,
                        invalid_attempt=True,
                    )
                )
                if (
                    not self.single_agent
                    and state_manager.consecutive_invalid_attempts
                    >= self.max_consecutive_invalid_attempts
                ):
                    return state_manager.result(
                        reason=TerminationReason.BUDGET_EXHAUSTED,
                        error_code="invalid_action_retry_exhausted",
                        error_message=(
                            "The policy exhausted its consecutive invalid "
                            "decision retry budget"
                        ),
                    )
                continue

            if resolved_decision is None:
                raise RuntimeError(
                    "valid decisions must include a resolved decision"
                )

            if isinstance(resolved_decision.action, FinishAction):
                return self._finish_result(
                    state_manager=state_manager,
                    state=state,
                    decision=decision,
                    repaired_decision=(
                        effective_decision if repair.code else None
                    ),
                    repair_code=repair.code,
                    resolved_decision=resolved_decision,
                    validation_signature=validation.signature,
                    policy_usage=policy_usage,
                    policy_view=policy_view,
                    context_reference_map=built_context.reference_map,
                )

            try:
                observation = self.router.execute(
                    resolved_decision.action,
                    state,
                    question=question,
                    scope_id=scope_id,
                    action_id=state_manager.next_action_id,
                )
            except AgenticRAGError as exc:
                self._record_runtime_error(
                    state_manager=state_manager,
                    decision=decision,
                    repaired_decision=(
                        effective_decision if repair.code else None
                    ),
                    repair_code=repair.code,
                    resolved_decision=resolved_decision,
                    validation_signature=validation.signature,
                    policy_usage=policy_usage,
                    policy_view=policy_view,
                    context_reference_map=built_context.reference_map,
                    error_code=exc.code,
                    error_message=exc.message,
                )
                return state_manager.result(
                    reason=TerminationReason.RUNTIME_ERROR,
                    error_code=exc.code,
                    error_message=exc.message,
                )
            except Exception as exc:
                self._record_runtime_error(
                    state_manager=state_manager,
                    decision=decision,
                    repaired_decision=(
                        effective_decision if repair.code else None
                    ),
                    repair_code=repair.code,
                    resolved_decision=resolved_decision,
                    validation_signature=validation.signature,
                    policy_usage=policy_usage,
                    policy_view=policy_view,
                    context_reference_map=built_context.reference_map,
                    error_code="runtime_error",
                    error_message=str(exc),
                )
                return state_manager.result(
                    reason=TerminationReason.RUNTIME_ERROR,
                    error_code="runtime_error",
                    error_message=str(exc),
                )

            step_usage = policy_usage + Usage(
                retrieved_tokens=observation.retrieved_tokens
            )
            state_manager.record_attempt(
                AttemptEvent(
                    decision=decision,
                    repaired_decision=(
                        effective_decision if repair.code else None
                    ),
                    repair_code=repair.code,
                    resolved_decision=resolved_decision,
                    validation_status=ValidationStatus.VALID,
                    validation_error=None,
                    observation=observation,
                    assessment=resolved_decision.assessment,
                    action_signature=validation.signature,
                    usage=step_usage,
                    policy_view=policy_view,
                    context_reference_map=built_context.reference_map,
                )
            )

        if state_manager.can_attempt_budget_finalize:
            finalized = self._try_budget_finalize(
                state_manager=state_manager,
            )
            if finalized is not None:
                return finalized

        attempt_budget_exhausted = (
            state_manager.policy_attempt_budget_exhausted
        )
        return state_manager.result(
            reason=TerminationReason.BUDGET_EXHAUSTED,
            error_code=(
                "policy_attempt_budget_exhausted"
                if attempt_budget_exhausted
                else "budget_exhausted"
            ),
            error_message=(
                "The episode exhausted its policy-attempt budget"
                if attempt_budget_exhausted
                else "The episode exhausted its step or retrieval budget"
            ),
        )

    def _run_episode_v22(
        self,
        *,
        question: str,
        scope_id: str,
        episode_id: str,
    ) -> EpisodeResult:
        """Run one progressive two-stage V2-2 episode."""

        if self.skill_bundle is None or self.v22_context_builder is None:
            raise RuntimeError("V2-2 controller is missing its skill bundle")
        state_manager = self.state_manager_factory.create(
            episode_id=episode_id,
            question=question,
            scope_id=scope_id,
            max_steps=self.max_steps,
            max_policy_attempts=self.max_policy_attempts,
            max_retrieved_tokens=self.max_retrieved_tokens,
        )

        while state_manager.can_continue:
            state = state_manager.snapshot()
            trajectory = state_manager.trajectory_snapshot()
            stages: list[PolicyStageRecord] = []
            cycle_usage = Usage()

            selection_context = self.v22_context_builder.build_selection(
                question,
                self.skill_bundle,
                state,
                trajectory,
                scope_id=scope_id,
            )
            selection, usage, stage, error = self._v22_policy_stage(
                selection_context,
                ActionSelection,
                phase=PolicyStagePhase.ACTION_SELECTION,
            )
            cycle_usage = cycle_usage + usage
            stages.append(stage)
            if error is not None:
                if not isinstance(error, PolicyResponseError):
                    self._record_v22_policy_error(
                        state_manager=state_manager,
                        policy_view=selection_context.policy_view,
                        stages=stages,
                        usage=cycle_usage,
                        message=str(error),
                    )
                    return state_manager.result(
                        reason=TerminationReason.POLICY_ERROR,
                        error_code="policy_error",
                        error_message=str(error),
                    )
                self._record_v22_invalid(
                    state_manager=state_manager,
                    policy_view=selection_context.policy_view,
                    stages=stages,
                    usage=cycle_usage,
                    code="invalid_action_selection_response",
                    message=str(error),
                )
                continue
            if not isinstance(selection, ActionSelection):
                raise RuntimeError("V2-2 selection stage returned the wrong model")
            stages[-1] = stages[-1].model_copy(
                update={"action_type": selection.action_type}
            )

            evidence_validation = self.validator.validate_action_selection(
                selection,
                state,
                scope_id,
            )
            if not evidence_validation.ok:
                stages[-1] = self._v22_stage_error(
                    stages[-1],
                    evidence_validation.code,
                    evidence_validation.message,
                    source="validator",
                )
                self._record_v22_invalid(
                    state_manager=state_manager,
                    policy_view=selection_context.policy_view,
                    stages=stages,
                    usage=cycle_usage,
                    code=(
                        evidence_validation.code
                        or "invalid_selected_evidence"
                    ),
                    message=(
                        evidence_validation.message
                        or "Stage-1 evidence selection is invalid"
                    ),
                )
                continue

            action_context = self.v22_context_builder.build_action(
                question,
                self.skill_bundle,
                selection,
                state,
                trajectory,
                scope_id=scope_id,
            )
            try:
                parameter_model = action_parameters_model(
                    tuple(self.validator.enabled_expansions),
                    selection.action_type,
                )
            except (TypeError, ValueError) as exc:
                stage = PolicyStageRecord(
                    phase=PolicyStagePhase.ACTION_DRAFT,
                    messages=list(action_context.messages),
                    error=str(exc),
                    action_type=selection.action_type,
                    disclosed_skill_paths=[
                        path for path, _ in action_context.disclosed_documents
                    ],
                    disclosed_skill_hashes=dict(
                        action_context.disclosed_documents
                    ),
                )
                stages.append(stage)
                self._record_v22_invalid(
                    state_manager=state_manager,
                    policy_view=selection_context.policy_view,
                    stages=stages,
                    usage=cycle_usage,
                    code="action_family_unavailable",
                    message=str(exc),
                )
                continue

            parameters, usage, stage, draft_error = self._v22_policy_stage(
                action_context,
                parameter_model,
                phase=PolicyStagePhase.ACTION_DRAFT,
                action_type=selection.action_type,
            )
            cycle_usage = cycle_usage + usage
            stages.append(stage)
            if draft_error is not None and not isinstance(
                draft_error, PolicyResponseError
            ):
                self._record_v22_policy_error(
                    state_manager=state_manager,
                    policy_view=selection_context.policy_view,
                    stages=stages,
                    usage=cycle_usage,
                    message=str(draft_error),
                )
                return state_manager.result(
                    reason=TerminationReason.POLICY_ERROR,
                    error_code="policy_error",
                    error_message=str(draft_error),
                )

            initial_decision = None
            initial_validation = None
            initial_code: str | None = None
            initial_message: str | None = None
            if draft_error is not None:
                initial_code = "invalid_action_parameters_response"
                initial_message = str(draft_error)
            else:
                try:
                    initial_decision = assemble_policy_decision(
                        selection, parameters
                    )
                except (TypeError, ValueError) as exc:
                    initial_code = "invalid_action_parameters"
                    initial_message = str(exc)
                else:
                    initial_validation = self.validator.validate(
                        initial_decision,
                        state,
                        scope_id,
                        enforce_semantic_handles=True,
                    )
                    if not initial_validation.ok:
                        initial_code = initial_validation.code
                        initial_message = initial_validation.message

            effective_decision = initial_decision
            final_validation = initial_validation
            failure_code = initial_code
            failure_message = initial_message
            repaired_decision = None
            repair_code = None
            if initial_code is not None:
                stages[-1] = self._v22_stage_error(
                    stages[-1],
                    initial_code,
                    initial_message,
                    source=(
                        "schema"
                        if draft_error is not None
                        else "validator"
                    ),
                )
                legal_options = self.validator.legal_action_options(
                    selection.action_type,
                    state,
                    scope_id,
                    selected_evidence_refs=selection.selected_evidence_refs,
                )
                if legal_options:
                    repair_context = self.v22_context_builder.build_action(
                        question,
                        self.skill_bundle,
                        selection,
                        state,
                        trajectory,
                        scope_id=scope_id,
                        recovery=True,
                        original_parameters=(
                            parameters.model_dump(mode="json")
                            if parameters is not None
                            else None
                        ),
                        error_code=initial_code,
                        error_message=initial_message,
                        legal_action_options=legal_options,
                    )
                    repaired_parameters, usage, stage, repair_error = (
                        self._v22_policy_stage(
                            repair_context,
                            parameter_model,
                            phase=PolicyStagePhase.ACTION_REPAIR,
                            action_type=selection.action_type,
                        )
                    )
                    cycle_usage = cycle_usage + usage
                    stages.append(stage)
                    repair_code = "v22_action_repair"
                    if repair_error is not None and not isinstance(
                        repair_error, PolicyResponseError
                    ):
                        self._record_v22_policy_error(
                            state_manager=state_manager,
                            policy_view=selection_context.policy_view,
                            stages=stages,
                            usage=cycle_usage,
                            message=str(repair_error),
                            decision=initial_decision,
                        )
                        return state_manager.result(
                            reason=TerminationReason.POLICY_ERROR,
                            error_code="policy_error",
                            error_message=str(repair_error),
                        )
                    if repair_error is None:
                        try:
                            repaired_decision = assemble_policy_decision(
                                selection, repaired_parameters
                            )
                        except (TypeError, ValueError) as exc:
                            repair_error = exc
                    if repair_error is not None:
                        failure_code = "invalid_action_repair_response"
                        failure_message = str(repair_error)
                        stages[-1] = self._v22_stage_error(
                            stages[-1],
                            "invalid_action_repair_response",
                            str(repair_error),
                            source="schema",
                        )
                        final_validation = None
                    else:
                        effective_decision = repaired_decision
                        final_validation = self.validator.validate(
                            repaired_decision,
                            state,
                            scope_id,
                            enforce_semantic_handles=True,
                        )
                        if not final_validation.ok:
                            failure_code = final_validation.code
                            failure_message = final_validation.message
                            stages[-1] = self._v22_stage_error(
                                stages[-1],
                                final_validation.code,
                                final_validation.message,
                                source="validator",
                            )

            if (
                effective_decision is None
                or final_validation is None
                or not final_validation.ok
                or final_validation.resolved_decision is None
            ):
                final_code = failure_code or "unresolved_invalid_action"
                final_message = failure_message or (
                    "The V2-2 action remained invalid after recovery"
                )
                self._record_v22_invalid(
                    state_manager=state_manager,
                    policy_view=selection_context.policy_view,
                    stages=stages,
                    usage=cycle_usage,
                    code=final_code,
                    message=final_message,
                    decision=initial_decision,
                    repaired_decision=repaired_decision,
                    repair_code=repair_code,
                    resolved_decision=(
                        final_validation.resolved_decision
                        if final_validation is not None
                        else (
                            initial_validation.resolved_decision
                            if initial_validation is not None
                            else None
                        )
                    ),
                )
                continue

            resolved_decision = final_validation.resolved_decision
            # Keep a schema-invalid raw draft as None. The corrected action is
            # represented only by repaired_decision and the stage trace.
            recorded_decision = initial_decision
            if isinstance(resolved_decision.action, FinishAction):
                return self._finish_result(
                    state_manager=state_manager,
                    state=state,
                    decision=recorded_decision,
                    repaired_decision=(
                        repaired_decision if repair_code else None
                    ),
                    repair_code=repair_code,
                    resolved_decision=resolved_decision,
                    validation_signature=final_validation.signature,
                    policy_usage=cycle_usage,
                    policy_view=selection_context.policy_view,
                    context_reference_map=None,
                    policy_stages=tuple(stages),
                )

            try:
                observation = self.router.execute(
                    resolved_decision.action,
                    state,
                    question=question,
                    scope_id=scope_id,
                    action_id=state_manager.next_action_id,
                )
            except AgenticRAGError as exc:
                self._record_runtime_error(
                    state_manager=state_manager,
                    decision=recorded_decision,
                    repaired_decision=(
                        repaired_decision if repair_code else None
                    ),
                    repair_code=repair_code,
                    resolved_decision=resolved_decision,
                    validation_signature=final_validation.signature,
                    policy_usage=cycle_usage,
                    policy_view=selection_context.policy_view,
                    context_reference_map=None,
                    error_code=exc.code,
                    error_message=exc.message,
                    policy_stages=tuple(stages),
                )
                return state_manager.result(
                    reason=TerminationReason.RUNTIME_ERROR,
                    error_code=exc.code,
                    error_message=exc.message,
                )
            except Exception as exc:
                self._record_runtime_error(
                    state_manager=state_manager,
                    decision=recorded_decision,
                    repaired_decision=(
                        repaired_decision if repair_code else None
                    ),
                    repair_code=repair_code,
                    resolved_decision=resolved_decision,
                    validation_signature=final_validation.signature,
                    policy_usage=cycle_usage,
                    policy_view=selection_context.policy_view,
                    context_reference_map=None,
                    error_code="runtime_error",
                    error_message=str(exc),
                    policy_stages=tuple(stages),
                )
                return state_manager.result(
                    reason=TerminationReason.RUNTIME_ERROR,
                    error_code="runtime_error",
                    error_message=str(exc),
                )

            state_manager.record_attempt(
                AttemptEvent(
                    decision=recorded_decision,
                    repaired_decision=(
                        repaired_decision if repair_code else None
                    ),
                    repair_code=repair_code,
                    resolved_decision=resolved_decision,
                    validation_status=ValidationStatus.VALID,
                    validation_error=None,
                    observation=observation,
                    assessment=resolved_decision.assessment,
                    action_signature=final_validation.signature,
                    usage=cycle_usage
                    + Usage(retrieved_tokens=observation.retrieved_tokens),
                    policy_view=selection_context.policy_view,
                    policy_stages=tuple(stages),
                )
            )

        if state_manager.can_attempt_budget_finalize:
            finalized = self._try_budget_finalize_v22(
                state_manager=state_manager
            )
            if finalized is not None:
                return finalized

        attempt_budget_exhausted = state_manager.policy_attempt_budget_exhausted
        return state_manager.result(
            reason=TerminationReason.BUDGET_EXHAUSTED,
            error_code=(
                "policy_attempt_budget_exhausted"
                if attempt_budget_exhausted
                else "budget_exhausted"
            ),
            error_message=(
                "The episode exhausted its V2-2 action-cycle budget"
                if attempt_budget_exhausted
                else "The episode exhausted its step or retrieval budget"
            ),
        )

    def _v22_policy_stage(
        self,
        context: V22BuiltContext,
        response_model: type,
        *,
        phase: PolicyStagePhase,
        action_type: ActionType | None = None,
    ) -> tuple[object | None, Usage, PolicyStageRecord, Exception | None]:
        output = None
        error: Exception | None = None
        try:
            output = self.policy.generate_structured(
                context.messages, response_model
            )
        except Exception as exc:  # provider boundary; caller classifies it
            error = exc
        usage = self._policy_usage()
        if error is not None and usage.policy_calls == 0:
            # Legacy provider adapters intentionally expose zero usage on a
            # failed call. V2-2's stage trace counts the attempted structured
            # response call while still leaving all token counters at zero.
            usage = usage + Usage(policy_calls=1)
        stage = PolicyStageRecord(
            phase=phase,
            messages=list(context.messages),
            output=(
                output.model_dump(mode="json")
                if output is not None
                else None
            ),
            error=str(error) if error is not None else None,
            action_type=action_type,
            disclosed_skill_paths=[
                path for path, _ in context.disclosed_documents
            ],
            disclosed_skill_hashes=dict(context.disclosed_documents),
            usage=usage,
        )
        return output, usage, stage, error

    @staticmethod
    def _v22_stage_error(
        stage: PolicyStageRecord,
        code: str | None,
        message: str | None,
        *,
        source: str,
    ) -> PolicyStageRecord:
        detail = ": ".join(
            part for part in (source, code, message) if part
        )
        return stage.model_copy(update={"error": detail})

    def _record_v22_invalid(
        self,
        *,
        state_manager: EpisodeStateManager,
        policy_view: PolicyViewOutput,
        stages: Sequence[PolicyStageRecord],
        usage: Usage,
        code: str,
        message: str,
        decision=None,
        repaired_decision=None,
        repair_code: str | None = None,
        resolved_decision=None,
    ) -> None:
        action = None
        if resolved_decision is not None:
            action = resolved_decision.action
        elif repaired_decision is not None:
            action = repaired_decision.action
        elif decision is not None:
            action = decision.action
        observation = Observation(
            action_id=state_manager.next_action_id,
            status=(
                ObservationStatus.DUPLICATE_ACTION
                if code == "duplicate_action"
                else ObservationStatus.INVALID_ACTION
            ),
            action=action,
            error_code=code,
            message=message,
            metadata={"workflow_mode": "single_agent_v2_2"},
        )
        state_manager.record_attempt(
            AttemptEvent(
                decision=decision,
                repaired_decision=repaired_decision,
                repair_code=repair_code,
                resolved_decision=resolved_decision,
                validation_status=ValidationStatus.INVALID,
                validation_error=message,
                observation=observation,
                assessment=(
                    (repaired_decision or decision).assessment
                    if (repaired_decision or decision) is not None
                    else None
                ),
                action_signature=None,
                usage=usage,
                policy_view=policy_view,
                commit_assessment=False,
                consume_step=False,
                invalid_attempt=True,
                policy_stages=tuple(stages),
            )
        )

    def _record_v22_policy_error(
        self,
        *,
        state_manager: EpisodeStateManager,
        policy_view: PolicyViewOutput,
        stages: Sequence[PolicyStageRecord],
        usage: Usage,
        message: str,
        decision=None,
    ) -> None:
        """Persist terminal provider failures without consuming an env step."""

        observation = Observation(
            action_id=state_manager.next_action_id,
            status=ObservationStatus.ERROR,
            action=(decision.action if decision is not None else None),
            error_code="policy_error",
            message=message,
            metadata={"workflow_mode": "single_agent_v2_2"},
        )
        state_manager.record_attempt(
            AttemptEvent(
                decision=decision,
                repaired_decision=None,
                repair_code=None,
                resolved_decision=None,
                validation_status=ValidationStatus.INVALID,
                validation_error=message,
                observation=observation,
                assessment=None,
                action_signature=None,
                usage=usage,
                policy_view=policy_view,
                commit_assessment=False,
                consume_step=False,
                invalid_attempt=False,
                policy_stages=tuple(stages),
            )
        )

    def initial_messages(
        self, question: str, scope_id: str | None = None
    ) -> Sequence:
        state = ControllerState.initial(
            max_steps=self.max_steps,
            max_retrieved_tokens=self.max_retrieved_tokens,
            max_policy_attempts=self.max_policy_attempts,
        )
        if self.single_agent_v22:
            if self.skill_bundle is None or self.v22_context_builder is None:
                raise RuntimeError("V2-2 controller is missing its skill bundle")
            return self.v22_context_builder.build_selection(
                question,
                self.skill_bundle,
                state,
                [],
                scope_id=scope_id or "__initial__",
            ).messages
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

    def _try_budget_finalize(
        self,
        *,
        state_manager: EpisodeStateManager,
    ) -> EpisodeResult | None:
        """Make one evidence-only FINISH attempt after retrieval stops."""

        state = state_manager.snapshot()
        built_context = self.context_builder.build(
            state_manager.question,
            self.skill,
            state,
            state_manager.trajectory_snapshot(),
            scope_id=state_manager.scope_id,
        )
        messages = list(built_context.messages)
        messages.append(
            Message(
                role="user",
                content=BUDGET_FINALIZE_INSTRUCTION,
            )
        )
        try:
            decision = self.policy.decide(
                messages,
                decision_format=built_context.decision_format,
            )
        except PolicyResponseError as exc:
            policy_usage = self._policy_usage()
            observation = Observation(
                action_id=state_manager.next_action_id,
                status=ObservationStatus.INVALID_ACTION,
                error_code="invalid_budget_finalize_response",
                message=str(exc),
                metadata={"budget_finalize": True},
            )
            state_manager.record_attempt(
                AttemptEvent(
                    decision=None,
                    repaired_decision=None,
                    repair_code=None,
                    resolved_decision=None,
                    validation_status=ValidationStatus.INVALID,
                    validation_error=str(exc),
                    observation=observation,
                    assessment=None,
                    action_signature=None,
                    usage=policy_usage,
                    policy_view=built_context.policy_view,
                    context_reference_map=built_context.reference_map,
                    commit_assessment=False,
                    consume_step=False,
                    invalid_attempt=True,
                )
            )
            return state_manager.result(
                reason=TerminationReason.BUDGET_EXHAUSTED,
                error_code="budget_exhausted",
                error_message=(
                    "The retrieval budget was exhausted and the finalize-only "
                    "policy response was invalid"
                ),
            )
        except Exception:
            state_manager.add_usage(self._policy_usage())
            return state_manager.result(
                reason=TerminationReason.BUDGET_EXHAUSTED,
                error_code="budget_exhausted",
                error_message=(
                    "The retrieval budget was exhausted and the finalize-only "
                    "policy call failed"
                ),
            )

        policy_usage = self._policy_usage()
        try:
            if self.single_agent_v3_typed_refs:
                if not isinstance(decision, V31PolicyDecision):
                    raise ContextIndexResolutionError(
                        "reference_not_available",
                        "V3.1 workflow requires a V31PolicyDecision",
                    )
                if built_context.reference_map is None:
                    raise RuntimeError(
                        "V3.1 context is missing its frozen reference map"
                    )
                internal_decision = resolve_v31_decision(
                    decision,
                    built_context.reference_map,
                )
            elif self.single_agent_v3:
                if not isinstance(decision, V3PolicyDecision):
                    raise ContextIndexResolutionError(
                        "memory_index_out_of_range",
                        "V3 workflow requires a V3PolicyDecision",
                    )
                if built_context.reference_map is None:
                    raise RuntimeError(
                        "V3 context is missing its frozen reference map"
                    )
                internal_decision = resolve_v3_decision(
                    decision,
                    built_context.reference_map,
                )
            else:
                if isinstance(
                    decision,
                    (V3PolicyDecision, V31PolicyDecision),
                ):
                    raise RuntimeError(
                        "Legacy workflows cannot execute semantic-memory "
                        "PolicyDecision"
                    )
                internal_decision = decision
        except ContextIndexResolutionError as exc:
            observation = Observation(
                action_id=state_manager.next_action_id,
                status=ObservationStatus.INVALID_ACTION,
                error_code=exc.code,
                message=exc.message,
                metadata={"budget_finalize": True},
            )
            state_manager.record_attempt(
                AttemptEvent(
                    decision=decision,
                    repaired_decision=None,
                    repair_code=None,
                    resolved_decision=None,
                    validation_status=ValidationStatus.INVALID,
                    validation_error=exc.message,
                    observation=observation,
                    assessment=assessment_for_internal(decision.assessment),
                    action_signature=None,
                    usage=policy_usage,
                    policy_view=built_context.policy_view,
                    context_reference_map=built_context.reference_map,
                    commit_assessment=False,
                    consume_step=False,
                    invalid_attempt=True,
                )
            )
            return state_manager.result(
                reason=TerminationReason.BUDGET_EXHAUSTED,
                error_code="budget_exhausted",
                error_message=(
                    "The retrieval budget was exhausted and the finalize-only "
                    f"decision used an invalid context index: {exc.message}"
                ),
            )

        repair = self.decision_repairer.repair(internal_decision, state)
        effective_decision = repair.decision
        validation = self.validator.validate(
            effective_decision, state, state_manager.scope_id
        )
        resolved_decision = validation.resolved_decision
        if (
            validation.ok
            and resolved_decision is not None
            and isinstance(resolved_decision.action, FinishAction)
        ):
            return self._finish_result(
                state_manager=state_manager,
                state=state,
                decision=decision,
                repaired_decision=(
                    effective_decision if repair.code else None
                ),
                repair_code=repair.code,
                resolved_decision=resolved_decision,
                validation_signature=validation.signature,
                policy_usage=policy_usage,
                policy_view=built_context.policy_view,
                context_reference_map=built_context.reference_map,
                consume_step=False,
            )

        message = (
            validation.message
            if not validation.ok
            else "Budget finalization requires a FINISH action"
        )
        code = (
            validation.code
            if not validation.ok
            else "budget_finalize_requires_finish"
        )
        observation = Observation(
            action_id=state_manager.next_action_id,
            status=ObservationStatus.INVALID_ACTION,
            action=(
                resolved_decision.action
                if resolved_decision is not None
                else effective_decision.action
            ),
            error_code=code,
            message=message,
            metadata={"budget_finalize": True},
        )
        state_manager.record_attempt(
            AttemptEvent(
                decision=decision,
                repaired_decision=(
                    effective_decision if repair.code else None
                ),
                repair_code=repair.code,
                resolved_decision=resolved_decision,
                validation_status=ValidationStatus.INVALID,
                validation_error=message,
                observation=observation,
                assessment=effective_decision.assessment,
                action_signature=None,
                usage=policy_usage,
                policy_view=built_context.policy_view,
                context_reference_map=built_context.reference_map,
                commit_assessment=False,
                consume_step=False,
                invalid_attempt=True,
            )
        )
        return state_manager.result(
            reason=TerminationReason.BUDGET_EXHAUSTED,
            error_code="budget_exhausted",
            error_message=(
                "The retrieval budget was exhausted and the finalize-only "
                f"decision was invalid: {message}"
            ),
        )

    def _try_budget_finalize_v22(
        self,
        *,
        state_manager: EpisodeStateManager,
    ) -> EpisodeResult | None:
        """Run the V2-2 root/FINISH stages once after retrieval closes."""

        if self.skill_bundle is None or self.v22_context_builder is None:
            return None
        state = state_manager.snapshot()
        trajectory = state_manager.trajectory_snapshot()
        stages: list[PolicyStageRecord] = []
        usage = Usage()
        selection_context = self.v22_context_builder.build_selection(
            state_manager.question,
            self.skill_bundle,
            state,
            trajectory,
            scope_id=state_manager.scope_id,
            extra_instruction=V22_BUDGET_FINALIZE_INSTRUCTION,
        )
        selection, stage_usage, stage, error = self._v22_policy_stage(
            selection_context,
            ActionSelection,
            phase=PolicyStagePhase.ACTION_SELECTION,
        )
        usage = usage + stage_usage
        stages.append(stage)
        if error is not None or not isinstance(selection, ActionSelection):
            message = str(error or "Invalid budget-finalize selection")
            if error is not None and not isinstance(error, PolicyResponseError):
                self._record_v22_policy_error(
                    state_manager=state_manager,
                    policy_view=selection_context.policy_view,
                    stages=stages,
                    usage=usage,
                    message=message,
                )
                return state_manager.result(
                    reason=TerminationReason.POLICY_ERROR,
                    error_code="policy_error",
                    error_message=message,
                )
            self._record_v22_invalid(
                state_manager=state_manager,
                policy_view=selection_context.policy_view,
                stages=stages,
                usage=usage,
                code="invalid_budget_finalize_response",
                message=message,
            )
            return state_manager.result(
                reason=TerminationReason.BUDGET_EXHAUSTED,
                error_code="budget_exhausted",
                error_message=message,
            )
        stages[-1] = stages[-1].model_copy(
            update={"action_type": selection.action_type}
        )
        if selection.action_type is not ActionType.FINISH:
            message = "Budget finalization requires FINISH"
            stages[-1] = self._v22_stage_error(
                stages[-1],
                "budget_finalize_requires_finish",
                message,
                source="validator",
            )
            self._record_v22_invalid(
                state_manager=state_manager,
                policy_view=selection_context.policy_view,
                stages=stages,
                usage=usage,
                code="budget_finalize_requires_finish",
                message=message,
            )
            return state_manager.result(
                reason=TerminationReason.BUDGET_EXHAUSTED,
                error_code="budget_exhausted",
                error_message=message,
            )
        evidence_validation = self.validator.validate_action_selection(
            selection,
            state,
            state_manager.scope_id,
        )
        if not evidence_validation.ok:
            message = evidence_validation.message or "Invalid finalize evidence"
            stages[-1] = self._v22_stage_error(
                stages[-1],
                evidence_validation.code,
                message,
                source="validator",
            )
            self._record_v22_invalid(
                state_manager=state_manager,
                policy_view=selection_context.policy_view,
                stages=stages,
                usage=usage,
                code=evidence_validation.code or "invalid_selected_evidence",
                message=message,
            )
            return state_manager.result(
                reason=TerminationReason.BUDGET_EXHAUSTED,
                error_code="budget_exhausted",
                error_message=message,
            )

        parameter_model = action_parameters_model(ActionType.FINISH)
        action_context = self.v22_context_builder.build_action(
            state_manager.question,
            self.skill_bundle,
            selection,
            state,
            trajectory,
            scope_id=state_manager.scope_id,
        )
        parameters, stage_usage, stage, error = self._v22_policy_stage(
            action_context,
            parameter_model,
            phase=PolicyStagePhase.ACTION_DRAFT,
            action_type=ActionType.FINISH,
        )
        usage = usage + stage_usage
        stages.append(stage)
        decision = None
        validation = None
        code: str | None = None
        message: str | None = None
        if error is not None:
            if not isinstance(error, PolicyResponseError):
                self._record_v22_policy_error(
                    state_manager=state_manager,
                    policy_view=selection_context.policy_view,
                    stages=stages,
                    usage=usage,
                    message=str(error),
                )
                return state_manager.result(
                    reason=TerminationReason.POLICY_ERROR,
                    error_code="policy_error",
                    error_message=str(error),
                )
            code = "invalid_action_parameters_response"
            message = str(error)
        else:
            decision = assemble_policy_decision(selection, parameters)
            validation = self.validator.validate(
                decision,
                state,
                state_manager.scope_id,
                enforce_semantic_handles=True,
            )
            if not validation.ok:
                code, message = validation.code, validation.message

        repaired_decision = None
        repair_code = None
        failure_code = code
        failure_message = message
        if code is not None:
            stages[-1] = self._v22_stage_error(
                stages[-1],
                code,
                message,
                source=("schema" if error is not None else "validator"),
            )
            legal_options = self.validator.legal_action_options(
                ActionType.FINISH,
                state,
                state_manager.scope_id,
                selected_evidence_refs=selection.selected_evidence_refs,
            )
            if legal_options:
                repair_context = self.v22_context_builder.build_action(
                    state_manager.question,
                    self.skill_bundle,
                    selection,
                    state,
                    trajectory,
                    scope_id=state_manager.scope_id,
                    recovery=True,
                    original_parameters=(
                        parameters.model_dump(mode="json")
                        if parameters is not None
                        else None
                    ),
                    error_code=code,
                    error_message=message,
                    legal_action_options=legal_options,
                )
                repaired_parameters, stage_usage, stage, repair_error = (
                    self._v22_policy_stage(
                        repair_context,
                        parameter_model,
                        phase=PolicyStagePhase.ACTION_REPAIR,
                        action_type=ActionType.FINISH,
                    )
                )
                usage = usage + stage_usage
                stages.append(stage)
                repair_code = "v22_action_repair"
                if repair_error is not None and not isinstance(
                    repair_error, PolicyResponseError
                ):
                    self._record_v22_policy_error(
                        state_manager=state_manager,
                        policy_view=selection_context.policy_view,
                        stages=stages,
                        usage=usage,
                        message=str(repair_error),
                        decision=decision,
                    )
                    return state_manager.result(
                        reason=TerminationReason.POLICY_ERROR,
                        error_code="policy_error",
                        error_message=str(repair_error),
                    )
                if repair_error is None:
                    try:
                        repaired_decision = assemble_policy_decision(
                            selection, repaired_parameters
                        )
                    except (TypeError, ValueError) as exc:
                        repair_error = exc
                if repair_error is not None:
                    failure_code = "invalid_action_repair_response"
                    failure_message = str(repair_error)
                    validation = None
                    stages[-1] = self._v22_stage_error(
                        stages[-1],
                        "invalid_action_repair_response",
                        str(repair_error),
                        source="schema",
                    )
                else:
                    validation = self.validator.validate(
                        repaired_decision,
                        state,
                        state_manager.scope_id,
                        enforce_semantic_handles=True,
                    )
                    if not validation.ok:
                        failure_code = validation.code
                        failure_message = validation.message
                        stages[-1] = self._v22_stage_error(
                            stages[-1],
                            validation.code,
                            validation.message,
                            source="validator",
                        )

        effective_decision = repaired_decision or decision
        if (
            effective_decision is not None
            and validation is not None
            and validation.ok
            and validation.resolved_decision is not None
            and isinstance(validation.resolved_decision.action, FinishAction)
        ):
            return self._finish_result(
                state_manager=state_manager,
                state=state,
                decision=decision,
                repaired_decision=(
                    repaired_decision if repair_code else None
                ),
                repair_code=repair_code,
                resolved_decision=validation.resolved_decision,
                validation_signature=validation.signature,
                policy_usage=usage,
                policy_view=selection_context.policy_view,
                context_reference_map=None,
                consume_step=False,
                policy_stages=tuple(stages),
            )

        failure_message = failure_message or (
            "Budget finalization remained invalid"
        )
        failure_code = failure_code or "invalid_budget_finalize_response"
        self._record_v22_invalid(
            state_manager=state_manager,
            policy_view=selection_context.policy_view,
            stages=stages,
            usage=usage,
            code=failure_code,
            message=failure_message,
            decision=decision,
            repaired_decision=repaired_decision,
            repair_code=repair_code,
            resolved_decision=(
                validation.resolved_decision
                if validation is not None
                else None
            ),
        )
        return state_manager.result(
            reason=TerminationReason.BUDGET_EXHAUSTED,
            error_code="budget_exhausted",
            error_message=failure_message,
        )

    def _finish_result(
        self,
        *,
        state_manager: EpisodeStateManager,
        state: ControllerState,
        decision,
        repaired_decision,
        repair_code: str | None,
        resolved_decision,
        validation_signature: str | None,
        policy_usage: Usage,
        policy_view: PolicyViewOutput,
        context_reference_map: (
            ContextReferenceMap | TypedContextReferenceMap | None
        ),
        consume_step: bool = True,
        policy_stages: tuple[PolicyStageRecord, ...] = (),
    ) -> EpisodeResult:
        selected_refs = list(resolved_decision.action.evidence_refs)
        if self.single_agent_v22:
            selected_refs = _merge_evidence_refs(
                state.selected_evidence_refs,
                list(resolved_decision.assessment.selected_evidence_refs),
            )
        elif self.accumulate_selected_evidence:
            selected_refs = _merge_evidence_refs(
                state.selected_evidence_refs,
                selected_refs,
            )
        resolved = self.evidence_resolver.resolve(
            selected_refs, state, state_manager.scope_id
        )
        observation = Observation(
            action_id=state_manager.next_action_id,
            status=ObservationStatus.OK,
            action=resolved_decision.action,
            results=[item.model_dump(mode="json") for item in resolved],
            metadata={
                "finish": True,
                "budget_finalize": not consume_step,
                **(
                    {
                        "answer_source": (
                            "policy"
                            if resolved_decision.action.answer is not None
                            else "answer_generator"
                        )
                    }
                    if self.single_agent
                    else {}
                ),
            },
        )
        state_manager.record_attempt(
            AttemptEvent(
                decision=decision,
                repaired_decision=repaired_decision,
                repair_code=repair_code,
                resolved_decision=resolved_decision,
                validation_status=ValidationStatus.VALID,
                validation_error=None,
                observation=observation,
                assessment=resolved_decision.assessment,
                action_signature=validation_signature,
                usage=policy_usage,
                policy_view=policy_view,
                context_reference_map=context_reference_map,
                consume_step=consume_step,
                policy_stages=policy_stages,
            )
        )
        direct_answer = (
            resolved_decision.action.answer
            if self.single_agent
            else None
        )
        if direct_answer is not None:
            return state_manager.result(
                reason=TerminationReason.FINISH,
                answer=direct_answer,
                selected_refs=selected_refs,
                resolved_evidence=resolved,
            )

        try:
            answer = self.answer_generator.generate(
                state_manager.question, resolved
            )
        except Exception as exc:
            answer_usage = self._answer_usage()
            state_manager.add_usage(
                answer_usage, attach_to_latest_attempt=True
            )
            return state_manager.result(
                reason=TerminationReason.ANSWER_GENERATION_ERROR,
                selected_refs=selected_refs,
                resolved_evidence=resolved,
                error_code="answer_generation_error",
                error_message=str(exc),
            )
        answer_usage = self._answer_usage()
        state_manager.add_usage(
            answer_usage, attach_to_latest_attempt=True
        )
        return state_manager.result(
            reason=TerminationReason.FINISH,
            answer=answer,
            selected_refs=selected_refs,
            resolved_evidence=resolved,
        )
    def _record_runtime_error(
        self,
        *,
        state_manager: EpisodeStateManager,
        decision,
        repaired_decision,
        repair_code: str | None,
        resolved_decision,
        validation_signature: str | None,
        policy_usage: Usage,
        policy_view: PolicyViewOutput,
        context_reference_map: (
            ContextReferenceMap | TypedContextReferenceMap | None
        ),
        error_code: str,
        error_message: str,
        policy_stages: tuple[PolicyStageRecord, ...] = (),
    ) -> None:
        observation = Observation(
            action_id=state_manager.next_action_id,
            status=ObservationStatus.ERROR,
            action=resolved_decision.action,
            error_code=error_code,
            message=error_message,
        )
        state_manager.record_attempt(
            AttemptEvent(
                decision=decision,
                repaired_decision=repaired_decision,
                repair_code=repair_code,
                resolved_decision=resolved_decision,
                validation_status=ValidationStatus.VALID,
                validation_error=None,
                observation=observation,
                assessment=resolved_decision.assessment,
                action_signature=validation_signature,
                usage=policy_usage,
                policy_view=policy_view,
                context_reference_map=context_reference_map,
                policy_stages=policy_stages,
            )
        )


def _merge_evidence_refs(first: list, second: list) -> list:
    merged = []
    seen: set[tuple[object, str]] = set()
    for ref in [*first, *second]:
        key = (ref.unit, ref.id)
        if key in seen:
            continue
        seen.add(key)
        merged.append(ref)
    return merged
