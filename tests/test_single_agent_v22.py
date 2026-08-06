from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentic_rag.agent.answer import ScriptedAnswerGenerator
from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    Message,
    TerminationReason,
    ValidationStatus,
)
from agentic_rag.agent.policy import PolicyTransportError, ScriptedPolicy
from agentic_rag.agent.skill import ProgressiveSkillBundle, SkillDocument
from agentic_rag.storage import Substrate


BUNDLE_ROOT = (
    Path(__file__).parents[1] / "skills" / "agentic-rag-v2-2" / "SKILL.md"
)


def _payload(messages: Sequence[Message | dict[str, str]]) -> dict[str, Any]:
    for message in reversed(messages):
        content = (
            message.content if isinstance(message, Message) else message["content"]
        )
        if content.startswith("{"):
            return json.loads(content)
    raise AssertionError("V2-2 context did not contain a JSON state payload")


def _handle(
    messages: Sequence[Message | dict[str, str]],
    node_type: str,
    text: str,
) -> str:
    handles = _payload(messages)["current_state"]["visible_handles"]
    for node in handles:
        semantic_text = " ".join(
            str(node.get(key) or "") for key in ("label", "text", "title")
        )
        if node["node_type"] == node_type and text in semantic_text:
            return str(node["handle"])
    raise AssertionError(f"No visible {node_type} matched {text!r}: {handles!r}")


def _finish_selection(
    messages: Sequence[Message | dict[str, str]],
) -> dict[str, Any]:
    sentence_handle = _handle(
        messages, "SENTENCE", "Marie Curie was born in Warsaw"
    )
    return {
        "action_type": "FINISH",
        "action_intent": (
            "The complete sentence states that Marie Curie was born in "
            "Warsaw, so the birthplace question can now be answered."
        ),
        "selected_evidence_refs": [
            {"unit": "SENTENCE", "id": sentence_handle}
        ],
    }


def _finish_parameters(
    messages: Sequence[Message | dict[str, str]],
) -> dict[str, Any]:
    selection = _payload(messages)["action_selection"]
    return {
        "answer": "Warsaw",
        "evidence_refs": selection["selected_evidence_refs"],
    }


def _harness(
    *,
    built_substrate: Path,
    output_root: Path,
    policy: ScriptedPolicy,
    fake_embedder: Any,
    max_steps: int = 8,
    max_policy_attempts: int = 10,
) -> tuple[AgentHarness, ScriptedAnswerGenerator]:
    bundle = ProgressiveSkillBundle.load(BUNDLE_ROOT)
    fallback = ScriptedAnswerGenerator("must not be called")
    harness = AgentHarness(
        substrate=Substrate.open(built_substrate),
        config=AgentConfig(
            workflow_mode="single_agent_v2_2",
            max_steps=max_steps,
            max_policy_attempts=max_policy_attempts,
        ),
        skill=bundle.root,
        skill_bundle=bundle,
        policy=policy,
        answer_generator=fallback,
        output_root=output_root,
        embedding_backend=fake_embedder,
    )
    return harness, fallback


def test_v22_normal_cycle_uses_two_calls_and_writes_bundle_artifact(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    policy = ScriptedPolicy(
        [
            {
                "action_type": "SEARCH",
                "action_intent": (
                    "No birthplace evidence is visible, so search for the "
                    "sentence that states where Marie Curie was born."
                ),
                "selected_evidence_refs": [],
            },
            {
                "query": "Marie Curie birthplace Warsaw",
                "method": "BM25",
                "target": "SENTENCE",
                "top_k": 5,
            },
            _finish_selection,
            _finish_parameters,
        ]
    )
    harness, fallback = _harness(
        built_substrate=built_substrate,
        output_root=tmp_path / "runs",
        policy=policy,
        fake_embedder=fake_embedder,
    )

    result = harness.run(
        "Where was Marie Curie born?", "q1", episode_id="v22-normal"
    )

    assert result.answer == "Warsaw"
    assert result.usage.policy_calls == 4
    assert result.usage.answer_calls == 0
    assert fallback.calls == []
    assert [len(step.policy_stages) for step in result.trajectory] == [2, 2]
    assert all(
        step.validation_status is ValidationStatus.VALID
        for step in result.trajectory
    )
    assert result.final_state is not None
    assert len(result.final_state.selected_evidence_refs) == 1

    artifact_dir = Path(result.artifact_dir or "")
    bundle_artifact = json.loads(
        (artifact_dir / "skill_bundle.json").read_text(encoding="utf-8")
    )
    assert bundle_artifact["sha256"] == harness.skill_bundle.sha256
    assert len(bundle_artifact["documents"]) == 15
    assert all("content" in item for item in bundle_artifact["documents"])

    trace = harness.build_io_trace(result)
    assert trace["trace_format"] == "agentic-rag-logical-io-v3"
    assert len(trace["policy_calls"]) == 4
    assert trace["v2_2_metrics"]["action_cycles"] == 2
    assert trace["v2_2_metrics"]["policy_calls"] == 4
    assert trace["v2_2_metrics"]["repair_attempts"] == 0
    assert trace["policy_calls"][0]["disclosed_skill_paths"] == [
        "SKILL.md"
    ]
    assert "actions/search/SKILL.md" in trace["policy_calls"][1][
        "disclosed_skill_paths"
    ]

    # V2-2 reconstructs COMPLETE evidence from the substrate, even if the
    # legacy compact handle contains only a shortened summary.
    sentence = next(
        item
        for item in harness.substrate.sentences
        if item.text == "Marie Curie was born in Warsaw."
    )
    frozen_state = result.trajectory[0].state_after.model_copy(deep=True)
    stored = frozen_state.node_handles[sentence.sentence_id]
    frozen_state.node_handles[sentence.sentence_id] = stored.model_copy(
        update={"text": "TRUNCATED SUMMARY"}
    )
    context = harness.controller.v22_context_builder.build_selection(
        result.query,
        harness.skill_bundle,
        frozen_state,
        result.trajectory[:1],
        scope_id=result.scope_id,
    )
    complete = next(
        node
        for node in _payload(context.messages)["current_state"][
            "visible_handles"
        ]
        if node["node_type"] == "SENTENCE"
        and node["handle"]
        == frozen_state.handle_registry.handle_for(
            sentence.sentence_id, "SENTENCE"
        )
    )
    assert complete["text"] == sentence.text


def test_v22_repair_is_third_call_and_only_recovery_sees_legal_options(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    policy = ScriptedPolicy(
        [
            {
                "action_type": "SEARCH",
                "action_intent": "Search for direct birthplace evidence.",
                "selected_evidence_refs": [],
            },
            {
                "query": "Marie Curie birthplace",
                "method": "LEXICAL",
                "target": "SENTENCE",
                "top_k": 5,
            },
            {
                "query": "Marie Curie birthplace Warsaw",
                "method": "BM25",
                "target": "SENTENCE",
                "top_k": 5,
            },
            _finish_selection,
            _finish_parameters,
        ]
    )
    harness, _ = _harness(
        built_substrate=built_substrate,
        output_root=tmp_path / "runs",
        policy=policy,
        fake_embedder=fake_embedder,
    )

    result = harness.run(
        "Where was Marie Curie born?", "q1", episode_id="v22-repair"
    )

    assert result.answer == "Warsaw"
    assert result.usage.policy_calls == 5
    assert len(result.trajectory[0].policy_stages) == 3
    assert result.trajectory[0].decision is None
    assert result.trajectory[0].repaired_decision is not None
    draft_input = "\n".join(
        message.content for message in policy.calls[1] if isinstance(message, Message)
    )
    repair_input = "\n".join(
        message.content for message in policy.calls[2] if isinstance(message, Message)
    )
    assert "legal_action_options" not in draft_input
    assert "--- actions/search/references/recovery.md ---" not in draft_input
    assert "legal_action_options" in repair_input
    assert "--- actions/search/references/recovery.md ---" in repair_input
    selection_payload = _payload(policy.calls[0])
    draft_payload = _payload(policy.calls[1])
    repair_payload = _payload(policy.calls[2])
    assert selection_payload["current_state"] == draft_payload["current_state"]
    assert draft_payload["current_state"] == repair_payload["current_state"]
    assert draft_payload["action_selection"]["action_intent"] == (
        "Search for direct birthplace evidence."
    )
    trace = harness.build_io_trace(result)
    assert trace["v2_2_metrics"]["initial_invalid"] == 1
    assert trace["v2_2_metrics"]["repair_attempts"] == 1
    assert trace["v2_2_metrics"]["repair_success"] == 1


def test_v22_failed_repair_rolls_back_pending_evidence_and_reselects(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    policy = ScriptedPolicy(
        [
            {
                "action_type": "SEARCH",
                "action_intent": "Search for direct birthplace evidence.",
                "selected_evidence_refs": [],
            },
            {
                "query": "Marie Curie birthplace Warsaw",
                "method": "BM25",
                "target": "SENTENCE",
                "top_k": 5,
            },
            lambda messages: {
                "action_type": "READ",
                "action_intent": (
                    "Keep the complete birthplace sentence while attempting "
                    "to read more context."
                ),
                "selected_evidence_refs": [
                    {
                        "unit": "SENTENCE",
                        "id": _handle(
                            messages,
                            "SENTENCE",
                            "Marie Curie was born in Warsaw",
                        ),
                    }
                ],
            },
            {"chunk_id": "C999"},
            {"chunk_id": "C999"},
            _finish_selection,
            _finish_parameters,
        ]
    )
    harness, _ = _harness(
        built_substrate=built_substrate,
        output_root=tmp_path / "runs",
        policy=policy,
        fake_embedder=fake_embedder,
    )

    result = harness.run(
        "Where was Marie Curie born?", "q1", episode_id="v22-rollback"
    )

    assert result.answer == "Warsaw"
    assert result.usage.policy_calls == 7
    invalid = result.trajectory[1]
    assert invalid.validation_status is ValidationStatus.INVALID
    assert invalid.state_before.step == invalid.state_after.step == 1
    assert invalid.state_before.selected_evidence_refs == []
    assert invalid.state_after.selected_evidence_refs == []
    assert invalid.observation is not None
    assert invalid.observation.retrieved_tokens == 0
    assert len(invalid.policy_stages) == 3
    assert result.final_state is not None
    assert len(result.final_state.selected_evidence_refs) == 1
    metrics = harness.build_io_trace(result)["v2_2_metrics"]
    assert metrics["repair_attempts"] == 1
    assert metrics["repair_success"] == 0
    assert metrics["reselections"] == 1
    assert metrics["unresolved_invalid"] == 1


def test_v22_sentence_entity_sentence_requires_two_expand_steps(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    policy = ScriptedPolicy(
        [
            {
                "action_type": "SEARCH",
                "action_intent": "Find a complete sentence about Marie Curie.",
                "selected_evidence_refs": [],
            },
            {
                "query": "Marie Curie birthplace Warsaw",
                "method": "BM25",
                "target": "SENTENCE",
                "top_k": 5,
            },
            lambda messages: {
                "action_type": "EXPAND",
                "action_intent": (
                    "Use the complete Marie Curie sentence to discover its "
                    "named entities."
                ),
                "selected_evidence_refs": [],
            },
            lambda messages: {
                "kind": "SENTENCE_MENTIONS_ENTITY",
                "source_id": _handle(
                    messages,
                    "SENTENCE",
                    "Marie Curie was born in Warsaw",
                ),
                "direction": None,
                "query": None,
                "top_k": 5,
            },
            {
                "action_type": "EXPAND",
                "action_intent": (
                    "Follow the Marie Curie entity back to all complete "
                    "sentences that mention her."
                ),
                "selected_evidence_refs": [],
            },
            lambda messages: {
                "kind": "ENTITY_MENTIONED_IN_SENTENCE",
                "source_id": _handle(messages, "ENTITY", "Marie Curie"),
                "direction": None,
                "query": None,
                "top_k": 5,
            },
            _finish_selection,
            _finish_parameters,
        ]
    )
    harness, _ = _harness(
        built_substrate=built_substrate,
        output_root=tmp_path / "runs",
        policy=policy,
        fake_embedder=fake_embedder,
    )

    result = harness.run(
        "Where was Marie Curie born?", "q1", episode_id="v22-expand-chain"
    )

    assert result.answer == "Warsaw"
    assert result.usage.policy_calls == 8
    assert [
        step.resolved_decision.action.type
        for step in result.trajectory
        if step.resolved_decision is not None
    ] == ["SEARCH", "EXPAND", "EXPAND", "FINISH"]
    expand_steps = [
        step
        for step in result.trajectory
        if step.resolved_decision is not None
        and step.resolved_decision.action.type == "EXPAND"
    ]
    assert len(expand_steps) == 2
    assert expand_steps[0].state_after.step + 1 == expand_steps[1].state_after.step


def test_v22_invalid_stage1_evidence_never_loads_an_action_skill(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    policy = ScriptedPolicy(
        [
            {
                "action_type": "SEARCH",
                "action_intent": "Find a complete birthplace sentence.",
                "selected_evidence_refs": [],
            },
            {
                "query": "Marie Curie birthplace Warsaw",
                "method": "BM25",
                "target": "SENTENCE",
                "top_k": 5,
            },
            {
                "action_type": "READ",
                "action_intent": "Keep evidence, then read more context.",
                "selected_evidence_refs": [
                    {"unit": "SENTENCE", "id": "S999"}
                ],
            },
            _finish_selection,
            _finish_parameters,
        ]
    )
    harness, _ = _harness(
        built_substrate=built_substrate,
        output_root=tmp_path / "runs",
        policy=policy,
        fake_embedder=fake_embedder,
    )

    result = harness.run(
        "Where was Marie Curie born?", "q1", episode_id="v22-stage1-gate"
    )

    assert result.answer == "Warsaw"
    assert result.usage.policy_calls == 5
    rejected = result.trajectory[1]
    assert rejected.validation_status is ValidationStatus.INVALID
    assert rejected.state_before.step == rejected.state_after.step == 1
    assert len(rejected.policy_stages) == 1
    assert rejected.policy_stages[0].disclosed_skill_paths == ["SKILL.md"]
    assert "unknown_handle" in (rejected.policy_stages[0].error or "")


def test_v22_repair_success_commits_pending_evidence_once_by_stable_id(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    def read_selection(messages: Sequence[Message | dict[str, str]]) -> dict[str, Any]:
        sentence = _handle(
            messages, "SENTENCE", "Marie Curie was born in Warsaw"
        )
        return {
            "action_type": "READ",
            "action_intent": (
                "Retain the complete Marie Curie birthplace sentence while "
                "reading its visible parent chunk for context."
            ),
            "selected_evidence_refs": [
                {"unit": "SENTENCE", "id": sentence}
            ],
        }

    def legal_read(messages: Sequence[Message | dict[str, str]]) -> dict[str, Any]:
        options = _payload(messages)["legal_action_options"]
        return {"chunk_id": options[0]["chunk_id"]}

    policy = ScriptedPolicy(
        [
            {
                "action_type": "SEARCH",
                "action_intent": "Find a complete birthplace sentence.",
                "selected_evidence_refs": [],
            },
            {
                "query": "Marie Curie birthplace Warsaw",
                "method": "BM25",
                "target": "SENTENCE",
                "top_k": 5,
            },
            read_selection,
            {"chunk_id": "C999"},
            legal_read,
            _finish_selection,
            _finish_parameters,
        ]
    )
    harness, _ = _harness(
        built_substrate=built_substrate,
        output_root=tmp_path / "runs",
        policy=policy,
        fake_embedder=fake_embedder,
    )

    result = harness.run(
        "Where was Marie Curie born?", "q1", episode_id="v22-repair-commit"
    )

    assert result.answer == "Warsaw"
    repaired = result.trajectory[1]
    assert repaired.repair_code == "v22_action_repair"
    assert repaired.validation_status is ValidationStatus.VALID
    assert len(repaired.state_after.selected_evidence_refs) == 1
    stable_ref = repaired.state_after.selected_evidence_refs[0]
    assert not stable_ref.id.startswith(("S", "C"))
    assert result.final_state is not None
    assert result.final_state.selected_evidence_refs == [stable_ref]


def test_v22_three_call_repair_cycle_consumes_one_policy_attempt(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    policy = ScriptedPolicy(
        [
            {
                "action_type": "SEARCH",
                "action_intent": "Find direct birthplace evidence.",
                "selected_evidence_refs": [],
            },
            {
                "query": "Marie Curie",
                "method": "LEXICAL",
                "target": "SENTENCE",
                "top_k": 5,
            },
            {
                "query": "Marie Curie birthplace Warsaw",
                "method": "BM25",
                "target": "SENTENCE",
                "top_k": 5,
            },
        ]
    )
    harness, _ = _harness(
        built_substrate=built_substrate,
        output_root=tmp_path / "runs",
        policy=policy,
        fake_embedder=fake_embedder,
        max_policy_attempts=1,
    )

    result = harness.run(
        "Where was Marie Curie born?", "q1", episode_id="v22-cycle-budget"
    )

    assert result.termination_reason is TerminationReason.BUDGET_EXHAUSTED
    assert result.usage.policy_calls == 3
    assert result.final_state is not None
    assert result.final_state.policy_attempts == 1
    assert result.final_state.step == 1
    assert len(result.trajectory) == 1


def test_v22_finish_subset_is_repaired_and_full_selection_is_accumulated(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    def finish_two(messages: Sequence[Message | dict[str, str]]) -> dict[str, Any]:
        handles = [
            node["handle"]
            for node in _payload(messages)["current_state"]["visible_handles"]
            if node["node_type"] == "SENTENCE"
            and node["completeness"] == "COMPLETE"
        ]
        assert len(handles) >= 3
        finish_two.third = handles[2]
        return {
            "action_type": "FINISH",
            "action_intent": "Two complete sentences are retained for the answer.",
            "selected_evidence_refs": [
                {"unit": "SENTENCE", "id": handles[0]},
                {"unit": "SENTENCE", "id": handles[1]},
            ],
        }

    finish_two.third = ""

    def invalid_finish(messages: Sequence[Message | dict[str, str]]) -> dict[str, Any]:
        return {
            "answer": "Warsaw",
            "evidence_refs": [
                {"unit": "SENTENCE", "id": finish_two.third}
            ],
        }

    def repaired_finish(messages: Sequence[Message | dict[str, str]]) -> dict[str, Any]:
        options = _payload(messages)["legal_action_options"]
        return {
            "answer": "Warsaw",
            "evidence_refs": [options[0]["evidence_ref"]],
        }

    policy = ScriptedPolicy(
        [
            {
                "action_type": "SEARCH",
                "action_intent": "Find complete evidence.",
                "selected_evidence_refs": [],
            },
            {
                "query": "Marie Curie birthplace Warsaw Poland",
                "method": "BM25",
                "target": "SENTENCE",
                "top_k": 5,
            },
            finish_two,
            invalid_finish,
            repaired_finish,
        ]
    )
    harness, _ = _harness(
        built_substrate=built_substrate,
        output_root=tmp_path / "runs",
        policy=policy,
        fake_embedder=fake_embedder,
    )

    result = harness.run(
        "Where was Marie Curie born?", "q1", episode_id="v22-finish-subset"
    )

    assert result.answer == "Warsaw"
    assert result.usage.policy_calls == 5
    finish_step = result.trajectory[-1]
    assert finish_step.repair_code == "v22_action_repair"
    assert finish_step.resolved_decision is not None
    assert len(finish_step.resolved_decision.action.evidence_refs) == 1
    assert result.final_state is not None
    assert len(result.final_state.selected_evidence_refs) == 2
    assert result.selected_evidence_refs == result.final_state.selected_evidence_refs


def test_v22_terminal_stage2_transport_error_preserves_both_call_traces(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    policy = ScriptedPolicy(
        [
            {
                "action_type": "SEARCH",
                "action_intent": "Find direct evidence.",
                "selected_evidence_refs": [],
            },
            PolicyTransportError("local provider disconnected"),
        ]
    )
    harness, _ = _harness(
        built_substrate=built_substrate,
        output_root=tmp_path / "runs",
        policy=policy,
        fake_embedder=fake_embedder,
    )

    result = harness.run(
        "Where was Marie Curie born?", "q1", episode_id="v22-transport"
    )

    assert result.termination_reason is TerminationReason.POLICY_ERROR
    assert result.usage.policy_calls == 2
    assert result.final_state is not None
    assert result.final_state.step == 0
    assert result.final_state.policy_attempts == 1
    assert len(result.trajectory) == 1
    assert len(result.trajectory[0].policy_stages) == 2
    trace = harness.build_io_trace(result)
    assert len(trace["policy_calls"]) == 2
    assert trace["policy_calls"][0]["stage"] == "action_selection"
    assert trace["policy_calls"][1]["stage"] == "action_draft"
    assert "local provider disconnected" in (
        trace["policy_calls"][1]["stage_error"] or ""
    )


def test_v22_constructor_rejects_skill_bundle_root_mismatch(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    bundle = ProgressiveSkillBundle.load(BUNDLE_ROOT)
    policy = ScriptedPolicy([])
    try:
        AgentHarness(
            substrate=Substrate.open(built_substrate),
            config=AgentConfig(workflow_mode="single_agent_v2_2"),
            skill=SkillDocument.from_text("# Different root"),
            skill_bundle=bundle,
            policy=policy,
            answer_generator=ScriptedAnswerGenerator("unused"),
            output_root=tmp_path / "runs",
            embedding_backend=fake_embedder,
        )
    except ValueError as error:
        assert "root document" in str(error)
    else:
        raise AssertionError("mismatched skill root should be rejected")
