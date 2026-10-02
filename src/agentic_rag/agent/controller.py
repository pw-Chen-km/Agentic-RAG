"""Stateless orchestration for the one-decision-per-turn agent loop."""

from __future__ import annotations

from collections.abc import Sequence
import re
import json
import time
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
    PolicyStateError,
    PolicyTransportError,
)
from agentic_rag.agent.references import ReferenceResolutionError, resolve_decision
from agentic_rag.agent.routing import routing_metadata
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
        episode_timeout_seconds: float = 3_600.0,
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
        self.episode_timeout_seconds = episode_timeout_seconds
        self.state_manager_factory = EpisodeStateManagerFactory(
            state_updater or StateUpdater(
                router.substrate, context_builder.interface_contract
            )
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

        started = time.monotonic()
        while manager.can_continue:
            if time.monotonic() - started >= self.episode_timeout_seconds:
                return manager.result(
                    reason=TerminationReason.RUNTIME_ERROR,
                    error_code="episode_timeout",
                    error_message="episode exceeded its configured timeout",
                )
            state = manager.snapshot()
            built = self.context_builder.build(
                question,
                self.skill,
                state,
                manager.trajectory_snapshot(),
                scope_id=scope_id,
                action_space_mode=(
                    ActionSpaceMode.ANSWER
                    if state.answer_stage_pending
                    else ActionSpaceMode.NORMAL
                ),
            )
            try:
                decision = self.policy.decide(
                    built.messages,
                    decision_format=built.decision_format,
                    tools=built.provider_tools,
                )
            except PolicyResponseError as exc:
                usage = self._policy_usage()
                failure_category = (
                    "state_invalid" if isinstance(exc, PolicyStateError) else "protocol_invalid"
                )
                observation = Observation(
                    action_id=manager.next_action_id,
                    status=ObservationStatus.INVALID_ACTION,
                    error_code=failure_category,
                    message=str(exc),
                    metadata={"failure_category": failure_category},
                )
                self._attach_routing_metadata(observation, None, built, state, manager)
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
                        decision_schema=built.decision_schema,
                        commit_assessment=False,
                        consume_step=True,
                        invalid_attempt=True,
                        messages=tuple(built.messages),
                        tool_definitions=built.tool_definitions,
                        visible_source_spans=built.visible_source_spans,
                        provider_metadata={
                            **self._provider_metadata(built.messages),
                            "failure_category": failure_category,
                            "legacy_error_code": "invalid_policy_response",
                        },
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
                    metadata={"failure_category": "state_invalid"},
                )
                self._attach_routing_metadata(observation, None, built, state, manager)
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
                        decision_schema=built.decision_schema,
                        commit_assessment=False,
                        consume_step=True,
                        invalid_attempt=True,
                        messages=tuple(built.messages),
                        tool_definitions=built.tool_definitions,
                        visible_source_spans=built.visible_source_spans,
                        provider_metadata={
                            **self._provider_metadata(built.messages),
                            "failure_category": "state_invalid",
                        },
                    )
                )
                continue

            # A retrieval call and its assessment arrive in the same model
            # response. If that response closes the information gap while
            # evidence is already visible, do not execute a retrieval action
            # from the stale half of the response. Record the inconsistency,
            # commit the assessment, and make the next turn answer-only.
            if (
                not isinstance(resolved.action, ResolvedFinishAction)
                and resolved.assessment is not None
                and not resolved.assessment.missing_information
                and state.all_source_keys
            ):
                observation = Observation(
                    action_id=manager.next_action_id,
                    status=ObservationStatus.INVALID_ACTION,
                    action=resolved.action,
                    error_code="assessment_closed_retrieval",
                    message=(
                        "The assessment reports no remaining information gap. "
                        "Retrieval was not executed; the next turn is answer-only."
                    ),
                    metadata={"failure_category": "state_invalid"},
                )
                self._attach_routing_metadata(
                    observation, resolved.action, built, state, manager
                )
                manager.record_attempt(
                    AttemptEvent(
                        decision=decision,
                        resolved_decision=resolved,
                        validation_status=ValidationStatus.INVALID,
                        validation_error=observation.message,
                        observation=observation,
                        assessment=resolved.assessment,
                        action_signature=None,
                        usage=usage,
                        policy_view=built.policy_view,
                        context_reference_map=built.reference_map,
                        available_action_space=built.available_action_space,
                        decision_schema_sha256=built.decision_schema_sha256,
                        decision_schema=built.decision_schema,
                        commit_assessment=True,
                        consume_step=True,
                        invalid_attempt=True,
                        messages=tuple(built.messages),
                        tool_definitions=built.tool_definitions,
                        visible_source_spans=built.visible_source_spans,
                        provider_metadata={
                            **self._provider_metadata(built.messages),
                            "failure_category": "state_invalid",
                        },
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
                    metadata={"failure_category": "state_invalid"},
                )
                self._attach_routing_metadata(
                    observation,
                    resolved.action,
                    built,
                    state,
                    manager,
                    exact_duplicate_blocked=validation.code == "duplicate_action",
                )
                if validation.code == "duplicate_action":
                    observation.metadata["blocked_call"] = _action_call_for_display(
                        resolved.action
                    )
                manager.record_attempt(
                    AttemptEvent(
                        decision=decision,
                        resolved_decision=resolved,
                        validation_status=ValidationStatus.INVALID,
                        validation_error=validation.message,
                        observation=observation,
                        assessment=resolved.assessment,
                        action_signature=validation.signature,
                        usage=usage,
                        policy_view=built.policy_view,
                        context_reference_map=built.reference_map,
                        available_action_space=built.available_action_space,
                        decision_schema_sha256=built.decision_schema_sha256,
                        decision_schema=built.decision_schema,
                        # A duplicate action is a state-level rejection, but
                        # the assessment is still a structurally valid part
                        # of the model response. Keep it for the next
                        # recovery-aware decision context.
                        commit_assessment=(validation.code == "duplicate_action"),
                        consume_step=True,
                        invalid_attempt=True,
                        messages=tuple(built.messages),
                        tool_definitions=built.tool_definitions,
                        visible_source_spans=built.visible_source_spans,
                        provider_metadata={
                            **self._provider_metadata(built.messages),
                            "failure_category": "state_invalid",
                        },
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
                observation.metadata.setdefault(
                    "selected_tool", _tool_name_for_action(decision.action)
                )
                observation.metadata.setdefault(
                    "parsed_arguments", decision.action.model_dump(mode="json")
                )
                self._attach_routing_metadata(
                    observation, resolved.action, built, state, manager
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
                    decision_schema=built.decision_schema,
                    messages=tuple(built.messages),
                    tool_definitions=built.tool_definitions,
                    visible_source_spans=built.visible_source_spans,
                    provider_metadata=self._provider_metadata(built.messages),
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

    def _attach_routing_metadata(
        self,
        observation: Observation,
        action,
        built,
        state: EpisodeState,
        manager: EpisodeStateManager,
        *,
        exact_duplicate_blocked: bool = False,
    ) -> None:
        observation.metadata.update(
            routing_metadata(
                policy=self.context_builder.routing_policy,
                action=action,
                space=built.available_action_space,
                state=state,
                trajectory=manager.trajectory_snapshot(),
                exact_duplicate_blocked=exact_duplicate_blocked,
            )
        )

    def _provider_metadata(self, messages: Sequence[Message]) -> dict:
        metadata = dict(getattr(self.policy, "last_usage_metadata", {}) or {})
        metadata.setdefault(
            "visible_payload_token_estimate",
            sum(
                len(re.findall(r"(?u)\b\w+\b|[^\w\s]", message.content or ""))
                for message in messages
            ),
        )
        metadata.setdefault("query_encoding", "unavailable")
        metadata.setdefault("candidate_scoring", "unavailable")
        metadata.setdefault("annotation_lookup_ms", "unavailable")
        contract = getattr(self.context_builder, "interface_contract", None)
        if contract is not None:
            metadata["interface_contract_digest"] = contract.compile()["digest"]
        ranking = getattr(self.router.retriever, "ranking_service", None)
        if ranking is not None:
            metadata["query_encoding"] = ranking.query_encodes
            metadata["candidate_scoring"] = ranking.candidate_scores
            metadata["ranking_wall_time_ms"] = ranking.last_wall_time_ms
        return metadata

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
        manager.mark_budget_finalize_used()
        try:
            decision = self.policy.decide(
                messages,
                decision_format=built.decision_format,
                tools=built.provider_tools,
            )
            usage = self._policy_usage()
            resolved = resolve_decision(decision, built.reference_map)
            validation = self.validator.validate(
                resolved,
                state,
                manager.scope_id,
                allow_incomplete_finish=True,
            )
        except Exception as exc:
            usage = self._policy_usage()
            manager.record_attempt(
                AttemptEvent(
                    decision=locals().get("decision"),
                    resolved_decision=locals().get("resolved"),
                    validation_status=ValidationStatus.INVALID,
                    validation_error=str(exc),
                    observation=Observation(
                        action_id=manager.next_action_id,
                        status=ObservationStatus.INVALID_ACTION,
                        error_code="budget_finalize_failed",
                        message=str(exc),
                        metadata={
                            "budget_finalize": True,
                            "failure_category": (
                                "protocol_invalid"
                                if isinstance(exc, PolicyResponseError)
                                else "state_invalid"
                            ),
                        },
                    ),
                    assessment=(locals().get("decision").assessment if locals().get("decision") is not None else None),
                    action_signature=None,
                    usage=usage,
                    policy_view=built.policy_view,
                    context_reference_map=built.reference_map,
                    available_action_space=built.available_action_space,
                    decision_schema_sha256=built.decision_schema_sha256,
                    decision_schema=built.decision_schema,
                    commit_assessment=False,
                    consume_step=False,
                    consume_policy_attempt=False,
                    messages=tuple(messages),
                    tool_definitions=built.tool_definitions,
                    visible_source_spans=built.visible_source_spans,
                    provider_metadata={
                        **self._provider_metadata(messages),
                        "failure_category": (
                            "protocol_invalid"
                            if isinstance(exc, PolicyResponseError)
                            else "state_invalid"
                        ),
                    },
                )
            )
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
                messages=messages,
            )
        budget_observation = Observation(
            action_id=manager.next_action_id,
            status=ObservationStatus.INVALID_ACTION,
            action=resolved.action,
            error_code="budget_finalize_requires_finish",
            message=validation.message or "Budget finalization requires FINISH",
            metadata={
                "budget_finalize": True,
                "failure_category": "state_invalid",
            },
        )
        self._attach_routing_metadata(
            budget_observation,
            resolved.action,
            built,
            state,
            manager,
        )
        manager.record_attempt(
            AttemptEvent(
                decision=decision,
                resolved_decision=resolved,
                validation_status=ValidationStatus.INVALID,
                validation_error=validation.message,
                observation=budget_observation,
                assessment=resolved.assessment,
                action_signature=None,
                usage=usage,
                policy_view=built.policy_view,
                context_reference_map=built.reference_map,
                available_action_space=built.available_action_space,
                decision_schema_sha256=built.decision_schema_sha256,
                decision_schema=built.decision_schema,
                commit_assessment=False,
                consume_step=False,
                consume_policy_attempt=False,
                messages=tuple(messages),
                tool_definitions=built.tool_definitions,
                visible_source_spans=built.visible_source_spans,
                provider_metadata={
                    **self._provider_metadata(messages or built.messages),
                    "failure_category": "state_invalid",
                },
            )
        )
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
        messages: Sequence[Message] | None = None,
    ) -> EpisodeResult:
        # FINISH no longer asks the model to choose a citation subset.  The
        # answer-stage context is the authoritative source snapshot, so use
        # every complete source unit actually shown there, deduplicated with
        # passage-over-sentence precedence.
        visible_source_refs = self.evidence_resolver.canonicalize_refs(
            built.visible_source_refs, state, manager.scope_id
        )
        requested_evidence_refs = list(resolved.action.evidence_refs)
        content_overlap_removed = [
            ref for ref in requested_evidence_refs if ref not in visible_source_refs
        ]
        canonical_action = resolved.action.model_copy(
            update={"evidence_refs": visible_source_refs}
        )
        canonical_resolved = resolved.model_copy(
            update={"action": canonical_action}
        )
        evidence = self.evidence_resolver.resolve(
            visible_source_refs, state, manager.scope_id
        )
        observation = Observation(
            action_id=manager.next_action_id,
            status=ObservationStatus.OK,
            action=canonical_action,
            results=[item.model_dump(mode="json") for item in evidence],
            metadata={
                "finish": True,
                "budget_finalize": not consume_step,
                "selected_tool": "finish",
                "parsed_arguments": canonical_action.model_dump(mode="json"),
                "evidence_refs_source": "programmatic_visible_source_refs",
                "visible_source_refs": [
                    ref.model_dump(mode="json") for ref in visible_source_refs
                ],
                "requested_evidence_refs": [
                    ref.model_dump(mode="json") for ref in requested_evidence_refs
                ],
                "canonical_evidence_refs": [
                    ref.model_dump(mode="json") for ref in visible_source_refs
                ],
                "content_overlap_removed_refs": [
                    ref.model_dump(mode="json") for ref in content_overlap_removed
                ],
            },
        )
        built.policy_view.context_audit["visible_source_refs"] = [
            ref.model_dump(mode="json") for ref in visible_source_refs
        ]
        built.policy_view.context_audit["evidence_refs_source"] = (
            "programmatic_visible_source_refs"
        )
        self._attach_routing_metadata(
            observation, canonical_action, built, state, manager,
        )
        manager.record_attempt(
            AttemptEvent(
                decision=decision,
                resolved_decision=canonical_resolved,
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
                decision_schema=built.decision_schema,
                consume_step=consume_step,
                consume_policy_attempt=consume_step,
                messages=tuple(messages or built.messages),
                tool_definitions=built.tool_definitions,
                visible_source_spans=built.visible_source_spans,
                provider_metadata=self._provider_metadata(messages or built.messages),
            )
        )
        return manager.result(
            reason=TerminationReason.FINISH,
            answer=resolved.action.answer,
            evidence_refs=visible_source_refs,
            visible_source_refs=visible_source_refs,
            evidence_refs_source="programmatic_visible_source_refs",
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
        observation = Observation(
            action_id=manager.next_action_id,
            status=ObservationStatus.ERROR,
            action=resolved.action,
            error_code=error_code,
            message=error_message,
            metadata={"failure_category": "execution_error"},
        )
        self._attach_routing_metadata(
            observation,
            resolved.action,
            built,
            manager.snapshot(),
            manager,
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
                decision_schema=built.decision_schema,
                messages=tuple(built.messages),
                tool_definitions=built.tool_definitions,
                visible_source_spans=built.visible_source_spans,
                provider_metadata={
                    **self._provider_metadata(built.messages),
                    "failure_category": "execution_error",
                },
            )
        )


def _tool_name_for_action(action) -> str:
    """Stable native-tool label retained alongside the legacy action model."""

    action_type = getattr(action, "type", None)
    if action_type == "SEARCH":
        method = getattr(getattr(action, "method", None), "value", "").lower()
        target = getattr(getattr(action, "target", None), "value", "").lower()
        if (method, target) == ("dense", "chunk"):
            return "find_passages"
        if (method, target) == ("dense", "sentence"):
            return "find_sentences"
        return f"search_{method}_{target}"
    if action_type == "EXPAND":
        kind = getattr(getattr(action, "kind", None), "value", "")
        if kind == "ENTITY_MENTIONED_IN_CHUNK":
            return "follow_entity_to_passages"
        if kind == "ENTITY_MENTIONED_IN_SENTENCE":
            return "follow_entity_to_sentences"
        return f"follow_{kind.lower()}"
    if action_type == "READ":
        return "read_passage"
    if action_type == "FINISH":
        return "finish"
    return "unknown"


def _action_call_for_display(action) -> str:
    """Render a compact call signature for the recovery overlay."""

    name = _tool_name_for_action(action)
    action_type = getattr(action, "type", None)
    if action_type == "SEARCH":
        arguments = {"query": getattr(action, "query", None)}
    elif action_type == "EXPAND":
        arguments = {"entity_ref": getattr(action, "source_id", None)}
    elif action_type == "READ":
        arguments = {"passage_ref": getattr(action, "chunk_id", None)}
    else:
        arguments = {}
    return f"{name}({json.dumps(arguments, ensure_ascii=False, sort_keys=True)})"
