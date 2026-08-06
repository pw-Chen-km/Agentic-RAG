from __future__ import annotations

from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.models import (
    AssessmentStatus,
    EvidenceAssessment,
    Observation,
    ObservationStatus,
    PolicyDecision,
    SearchAction,
    TerminationReason,
    Usage,
    ValidationStatus,
)
from agentic_rag.agent.state import StateUpdater
from agentic_rag.agent.state_management import (
    AttemptEvent,
    EpisodeStateManager,
)


def _manager() -> EpisodeStateManager:
    return EpisodeStateManager(
        episode_id="episode-1",
        question="Where was Marie Curie born?",
        scope_id="q1",
        state_updater=StateUpdater(),
        max_steps=3,
        max_policy_attempts=5,
        max_retrieved_tokens=100,
    )


def _policy_view(manager: EpisodeStateManager):
    return PolicyContextBuilder().build_policy_view(
        manager.snapshot(),
        manager.trajectory_snapshot(),
        scope_id=None,
    )


def test_state_manager_owns_invalid_history_budget_and_usage() -> None:
    manager = _manager()
    observation = Observation(
        action_id=manager.next_action_id,
        status=ObservationStatus.INVALID_ACTION,
        error_code="invalid_policy_response",
        message="Malformed structured response",
    )

    record = manager.record_attempt(
        AttemptEvent(
            decision=None,
            repaired_decision=None,
            repair_code=None,
            resolved_decision=None,
            validation_status=ValidationStatus.INVALID,
            validation_error="Malformed structured response",
            observation=observation,
            assessment=None,
            action_signature=None,
            usage=Usage(
                policy_calls=1,
                input_tokens=10,
                total_tokens=12,
            ),
            policy_view=_policy_view(manager),
            commit_assessment=False,
            consume_step=False,
            invalid_attempt=True,
        )
    )

    state = manager.snapshot()
    assert record.observation.action_id == "attempt-1"
    assert record.agent_visible_observation["outcome"] == "invalid_action"
    assert state.step == 0
    assert state.policy_attempts == 1
    assert state.remaining_step_budget == 3
    assert state.remaining_policy_attempt_budget == 4
    assert manager.consecutive_invalid_attempts == 1
    assert len(manager.trajectory_snapshot()) == 1

    result = manager.result(
        reason=TerminationReason.BUDGET_EXHAUSTED,
        error_code="test_end",
    )
    assert result.usage.policy_calls == 1
    assert result.usage.total_tokens == 12
    assert result.final_state == state
    assert result.trajectory == [record]


def test_state_manager_resets_invalid_counter_and_isolates_snapshots() -> None:
    manager = _manager()
    decision = PolicyDecision(
        assessment=EvidenceAssessment(
            status=AssessmentStatus.INSUFFICIENT,
            missing_information=["birthplace"],
        ),
        action=SearchAction(
            query="Where was Marie Curie born?",
            method="BM25",
            target="SENTENCE",
        ),
    )
    observation = Observation(
        action_id=manager.next_action_id,
        status=ObservationStatus.OK,
        action=decision.action,
        results=[],
        retrieved_tokens=7,
    )

    manager.record_attempt(
        AttemptEvent(
            decision=decision,
            repaired_decision=None,
            repair_code=None,
            resolved_decision=decision,
            validation_status=ValidationStatus.VALID,
            validation_error=None,
            observation=observation,
            assessment=decision.assessment,
            action_signature="search-signature",
            usage=Usage(
                policy_calls=1,
                retrieved_tokens=7,
                total_tokens=20,
            ),
            policy_view=_policy_view(manager),
        )
    )

    state = manager.snapshot()
    assert state.step == 1
    assert state.policy_attempts == 1
    assert state.remaining_step_budget == 2
    assert state.remaining_retrieved_token_budget == 93
    assert state.action_signatures == {"search-signature"}
    assert state.last_assessment == decision.assessment
    assert manager.consecutive_invalid_attempts == 0
    assert manager.next_action_id == "attempt-2"

    state.remaining_step_budget = 0
    assert manager.snapshot().remaining_step_budget == 2
