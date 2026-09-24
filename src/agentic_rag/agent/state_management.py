"""The sole owner of episode state, history, budgets, and usage."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from agentic_rag.agent.context import project_observation_for_audit
from agentic_rag.agent.tool_calling import build_tool_definitions
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
    Message,
)
from agentic_rag.agent.state import StateUpdater
from agentic_rag.agent.context_rendering import output_delivery


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
    decision_schema: dict = field(default_factory=dict)
    commit_assessment: bool = True
    consume_step: bool = True
    invalid_attempt: bool = False
    consume_policy_attempt: bool = True
    messages: tuple[Message, ...] = ()
    provider_metadata: dict = field(default_factory=dict)
    tool_definitions: list[dict] = field(default_factory=list)
    visible_source_spans: list[dict] | None = None


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
        self._finalize_used = False

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
    def should_reserve_policy_attempt_for_finalize(self) -> bool:
        """Keep the last policy call for FINISH once eligible evidence exists."""

        return (
            self._state.remaining_policy_attempt_budget == 1
            and bool(self._state.eligible_sentence_ids or self._state.read_chunk_ids)
        )

    @property
    def can_attempt_budget_finalize(self) -> bool:
        return not self._finalize_used

    def mark_budget_finalize_used(self) -> None:
        self._finalize_used = True

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
            scope_id=self.scope_id,
            commit_assessment=event.commit_assessment,
            consume_step=event.consume_step,
            consume_policy_attempt=event.consume_policy_attempt,
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
            decision_schema=dict(event.decision_schema),
            tool_definitions=(
                list(event.tool_definitions)
                if event.tool_definitions
                else build_tool_definitions(event.available_action_space)
            ),
            messages=list(event.messages),
            provider_metadata=dict(event.provider_metadata or {}),
            visible_source_spans=list(
                event.visible_source_spans
                if event.visible_source_spans is not None
                else (event.observation.metadata.get("visible_source_spans", [])
                    if event.observation is not None else [])
            ),
            telemetry=_telemetry(event),
            context_audit={
                **event.policy_view.context_audit,
                "output_delivery": output_delivery(
                    event.observation, updated, event.visible_source_spans or [],
                    event.context_reference_map.typed_refs if event.context_reference_map else (),
                ),
            },
            assessment_status=("provided" if event.assessment is not None else
                               "unavailable" if event.policy_view.context_audit.get("assessment_requested", True)
                               else "not_requested"),
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


def _telemetry(event: AttemptEvent) -> dict:
    observation = event.observation
    metadata = dict(event.provider_metadata or {})
    if observation is not None:
        metadata.update(
            {
                "selected_tool": observation.metadata.get("selected_tool"),
                "parsed_arguments": observation.metadata.get("parsed_arguments"),
                "failure_category": observation.metadata.get("failure_category"),
                "duplicate_or_noop_reason": observation.message
                if observation.status.value == "duplicate_action"
                else None,
                "execution_wall_time_ms": observation.metadata.get("wall_time_ms"),
                "annotation_lookup_ms": observation.metadata.get("annotation_lookup_ms"),
            }
        )
    metadata["available_tools"] = [
        item.get("function", {}).get("name")
        for item in build_tool_definitions(event.available_action_space)
    ]
    metadata["decision_schema_token_estimate"] = len(
        re.findall(r"(?u)\b\w+\b|[^\w\s]", str(event.decision_schema))
    )
    metadata["tool_schema_token_estimate"] = metadata["decision_schema_token_estimate"]
    return metadata


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
