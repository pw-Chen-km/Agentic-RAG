from __future__ import annotations

from pathlib import Path

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    Assessment,
    FinishAction,
    PolicyDecision,
    ReadAction,
    SearchAction,
    TerminationReason,
)
from agentic_rag.agent.policy import ScriptedPolicy
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.substrate.storage import Substrate
from conftest import FakeEmbeddingBackend


def _search() -> PolicyDecision:
    return PolicyDecision(
        assessment=Assessment(
            missing_information=["Marie Curie's birthplace"],
        ),
        action=SearchAction(
            query="Where was Marie Curie born?",
            method="BM25",
            target="SENTENCE",
        ),
    )


def _finish() -> PolicyDecision:
    return PolicyDecision(
        assessment=Assessment(
            supported_facts=["Marie Curie was born in Warsaw."],
        ),
        action=FinishAction(answer="Warsaw", evidence_refs=["S1"]),
    )


def _read() -> PolicyDecision:
    return PolicyDecision(
        assessment=Assessment(
            missing_information=["complete context"],
        ),
        action=ReadAction(chunk_ref="C1"),
    )


def _harness(
    built_substrate: Path,
    fake_embedder: FakeEmbeddingBackend,
    tmp_path: Path,
    policy: ScriptedPolicy,
) -> AgentHarness:
    return AgentHarness(
        substrate=Substrate.open(built_substrate),
        config=AgentConfig(),
        skill=SkillDocument.from_text("Find explicit evidence, then answer."),
        policy=policy,
        output_root=tmp_path / "runs",
        embedding_backend=fake_embedder,
    )


def test_search_observation_updates_memory_then_policy_finishes(
    built_substrate: Path,
    fake_embedder: FakeEmbeddingBackend,
    tmp_path: Path,
) -> None:
    policy = ScriptedPolicy([_search(), _finish()])
    harness = _harness(built_substrate, fake_embedder, tmp_path, policy)
    result = harness.run("Where was Marie Curie born?", "q1", episode_id="e2e")

    assert result.termination_reason is TerminationReason.FINISH
    assert result.answer == "Warsaw"
    assert result.usage.policy_calls == 2
    assert result.final_state is not None
    assert result.final_state.step == 2
    assert result.evidence_refs[0].unit == "SENTENCE"
    assert len(policy.calls) == 2

    second_prompt = "\n".join(message.content for message in policy.calls[1])
    assert '"semantic_memory"' in second_prompt
    assert "Marie Curie was born in Warsaw." in second_prompt
    assert '"ref":"S1"' in second_prompt
    assert "Latest event" not in second_prompt
    assert "Action Catalog" not in second_prompt
    assert result.evidence_refs[0].id not in second_prompt


def test_invalid_ref_costs_attempt_but_not_retrieval_step(
    built_substrate: Path,
    fake_embedder: FakeEmbeddingBackend,
    tmp_path: Path,
) -> None:
    invalid_read = PolicyDecision(
        assessment=Assessment(
            missing_information=["context"],
        ),
        action=ReadAction(chunk_ref="C99"),
    )
    policy = ScriptedPolicy([invalid_read, _search(), _finish()])
    result = _harness(
        built_substrate, fake_embedder, tmp_path, policy
    ).run("Where was Marie Curie born?", "q1", episode_id="invalid-ref")

    assert result.termination_reason is TerminationReason.FINISH
    assert result.final_state is not None
    assert result.final_state.policy_attempts == 3
    assert result.final_state.step == 2
    assert result.trajectory[0].observation.error_code == "reference_not_available"
    assert result.trajectory[0].observation.retrieved_tokens == 0
    assert result.trajectory[0].context_reference_map.typed_refs == {}
    second_prompt = "\n".join(message.content for message in policy.calls[1])
    assert '"latest_attempt"' in second_prompt
    assert '"submitted_action":{"chunk_ref":"C99","type":"READ"}' in second_prompt
    assert '"error_code":"reference_not_available"' in second_prompt
    assert '"outcome":"invalid_action"' in second_prompt
    assert '"message":"Reference C99 is not visible in this snapshot"' in second_prompt


def test_semantic_action_history_is_visible_without_stale_refs(
    built_substrate: Path,
    fake_embedder: FakeEmbeddingBackend,
    tmp_path: Path,
) -> None:
    policy = ScriptedPolicy([_search(), _finish()])
    _harness(built_substrate, fake_embedder, tmp_path, policy).run(
        "Where was Marie Curie born?", "q1", episode_id="history"
    )
    second_prompt = "\n".join(message.content for message in policy.calls[1])
    assert '"type":"SEARCH"' in second_prompt
    assert '"method":"BM25"' in second_prompt
    assert '"target":"SENTENCE"' in second_prompt
    history = second_prompt.split('"attempted_actions"', 1)[1]
    assert "sentence:" not in history


def test_read_keeps_sentence_evidence_visible_and_latest_ref_explicit(
    built_substrate: Path,
    fake_embedder: FakeEmbeddingBackend,
    tmp_path: Path,
) -> None:
    policy = ScriptedPolicy([_search(), _read(), _finish()])
    result = _harness(built_substrate, fake_embedder, tmp_path, policy).run(
        "Where was Marie Curie born?", "q1", episode_id="read-stable-evidence"
    )

    assert result.termination_reason is TerminationReason.FINISH
    assert result.answer == "Warsaw"
    assert result.final_state is not None
    assert result.final_state.step == 3
    assert len(policy.calls) == 3

    third_prompt = "\n".join(message.content for message in policy.calls[2])
    assert '"latest_attempt"' in third_prompt
    assert '"submitted_action":{"chunk_ref":"C1","type":"READ"}' in third_prompt
    assert '"outcome":"success"' in third_prompt
    assert '"ref":"S1"' in third_prompt
    options = third_prompt.split("Currently available action options", 1)[1]
    assert "chunk_ref in [C1" not in options
    finish_line = next(
        line for line in options.splitlines() if line.startswith("- evidence_refs")
    )
    assert "S1" in finish_line and "C1" in finish_line
