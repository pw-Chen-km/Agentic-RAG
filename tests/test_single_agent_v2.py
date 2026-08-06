import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.agent.answer import ScriptedAnswerGenerator
from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    AssessmentStatus,
    ControllerState,
    EvidenceAssessment,
    FinishAction,
    Observation,
    ObservationStatus,
    PolicyDecision,
    SearchAction,
    SentenceRef,
)
from agentic_rag.agent.policy import ScriptedPolicy, policy_decision_model
from agentic_rag.agent.repair import DecisionRepairer
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.state import StateUpdater
from agentic_rag.storage import Substrate


def _assessment(
    *,
    status: AssessmentStatus = AssessmentStatus.INSUFFICIENT,
    refs: list[SentenceRef] | None = None,
) -> EvidenceAssessment:
    return EvidenceAssessment(
        status=status,
        selected_evidence_refs=refs or [],
    )


def test_v2_provider_finish_requires_direct_answer() -> None:
    provider_model = policy_decision_model((), direct_answer=True)
    payload = {
        "assessment": {
            "status": "SUFFICIENT",
            "supported_facts": ["Marie Curie was born in Warsaw."],
            "missing_information": [],
            "selected_evidence_refs": [
                {"unit": "SENTENCE", "id": "S1"}
            ],
        },
        "action": {
            "type": "FINISH",
            "evidence_refs": [{"unit": "SENTENCE", "id": "S1"}],
            "answer": "Warsaw",
        },
    }

    parsed = provider_model.model_validate(payload)
    decision = PolicyDecision.model_validate(parsed.model_dump(mode="json"))
    assert isinstance(decision.action, FinishAction)
    assert decision.action.answer == "Warsaw"

    del payload["action"]["answer"]
    with pytest.raises(ValidationError):
        provider_model.model_validate(payload)

    schema = json.dumps(provider_model.model_json_schema(), sort_keys=True)
    assert '"answer"' in schema


def test_v2_repair_derives_terminal_status_from_finish() -> None:
    decision = PolicyDecision(
        assessment=_assessment(),
        action=FinishAction(
            evidence_refs=[SentenceRef(id="S1")],
            answer="Warsaw",
        ),
    )

    repaired = DecisionRepairer(
        (), derive_status_from_action=True
    ).repair(decision, ControllerState.initial())

    assert repaired.code == "finish_status_derived_from_action"
    assert repaired.decision.assessment.status is AssessmentStatus.SUFFICIENT


def test_v2_state_accumulates_selected_evidence() -> None:
    updater = StateUpdater(accumulate_selected_evidence=True)
    state = ControllerState.initial()
    observation = Observation(status=ObservationStatus.OK)
    first = SentenceRef(id="sentence-1")
    second = SentenceRef(id="sentence-2")

    state = updater.apply(
        state,
        assessment=_assessment(refs=[first]),
        observation=observation,
        action_signature="one",
    )
    state = updater.apply(
        state,
        assessment=_assessment(refs=[second]),
        observation=observation,
        action_signature="two",
    )

    assert state.selected_evidence_refs == [first, second]


def test_v2_recovers_after_two_invalids_and_uses_policy_answer(
    built_substrate: Path,
    tmp_path: Path,
) -> None:
    substrate = Substrate.open(built_substrate)
    sentence = next(
        item
        for item in substrate.sentences
        if "Marie Curie was born in Warsaw" in item.text
    )
    search = PolicyDecision(
        assessment=_assessment(),
        action=SearchAction(
            query="Where was Marie Curie born?",
            method="BM25",
            target="SENTENCE",
        ),
    )
    finish = PolicyDecision(
        # Deliberately inconsistent legacy status: V2 derives SUFFICIENT from
        # the terminal action instead of wasting an invalid retry.
        assessment=_assessment(refs=[SentenceRef(id=sentence.sentence_id)]),
        action=FinishAction(
            evidence_refs=[SentenceRef(id=sentence.sentence_id)],
            answer="Warsaw",
        ),
    )
    policy = ScriptedPolicy([search, search, search, finish])
    fallback_answer = ScriptedAnswerGenerator("must not be called")
    config = AgentConfig(
        workflow_mode="single_agent_v2",
        max_steps=10,
        max_policy_attempts=6,
        max_consecutive_invalid_attempts=2,
    )
    harness = AgentHarness(
        substrate=substrate,
        config=config,
        skill=SkillDocument.from_text("# V2\nRetrieve, then finish."),
        policy=policy,
        answer_generator=fallback_answer,
        output_root=tmp_path / "runs",
    )

    result = harness.run(
        "Where was Marie Curie born?",
        "q1",
        episode_id="single-agent-v2",
    )

    assert result.answer == "Warsaw"
    assert result.usage.policy_calls == 4
    assert result.usage.answer_calls == 0
    assert fallback_answer.calls == []
    assert sum(
        step.validation_status.value == "invalid"
        for step in result.trajectory
    ) == 2
    assert result.trajectory[-1].repair_code == (
        "finish_status_derived_from_action"
    )
    trace = harness.build_io_trace(result)
    assert trace["answer_generation"] is None
    assert trace["policy_calls"][-1]["output"]["action"]["answer"] == (
        "Warsaw"
    )
