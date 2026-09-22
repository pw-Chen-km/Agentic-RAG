from pathlib import Path

import pytest

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.context_rendering import action_summary, render_context
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.interface import get_interface_contract
from agentic_rag.agent.models import (Assessment, EpisodeState, FinishAction, Usage, StepRecord, Observation,
    ObservationStatus, ValidationStatus, PolicyDecision, SearchAction, SearchMethod, SearchTarget)
from agentic_rag.agent.policy import PolicyResponseError
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.tool_calling import decision_from_tool_call
from agentic_rag.substrate.storage import Substrate


@pytest.mark.parametrize("condition", ["C0", "C1", "C2", "C3", "C5", "C4", "A1"])
@pytest.mark.parametrize("assessment", [False, True])
def test_live_shaped_duplicate_feedback_and_assessment_modes(built_substrate, fake_embedder, tmp_path, condition, assessment):
    class Policy:
        count = 0
        last_usage = Usage(policy_calls=1)
        last_usage_metadata = {}
        def decide(self, messages, *, tools=None, **kwargs):
            assert tools is None
            self.count += 1
            if self.count <= 2:
                action = SearchAction(query="Marie Curie", method=SearchMethod.DENSE, target=SearchTarget.CHUNK)
            else:
                action = FinishAction(answer="Insufficient evidence", evidence_refs=[])
            current = Assessment(missing_information=["Unresolved information"]) if assessment else None
            self.last_usage_metadata = {"constrained_single_decision": True, "decision_count": 1}
            return PolicyDecision(assessment=current, action=action)
    harness = AgentHarness(substrate=Substrate.open(built_substrate),
        config=AgentConfig(interface=condition, require_evidence_assessment=assessment),
        skill=SkillDocument.from_text("Answer using sources."), policy=Policy(),
        output_root=tmp_path / "out", embedding_backend=fake_embedder)
    result = harness.run("Where was Marie Curie born?", "q1")
    assert result.termination_reason.value == "finish"
    initial, duplicate, final = result.trajectory
    assert all(ref.startswith("C") for ref in initial.context_audit["output_delivery"]["returned_references"])
    assert "No action has been taken" in initial.messages[-1].content
    assert duplicate.observation.error_code == "duplicate_action"
    content = final.messages[-1].content
    assert "This request was not executed. No new source text was added." in content
    assert "NEW SOURCE TEXT\nNone." in content
    assert "PREVIOUSLY SHOWN SOURCE TEXT" in content
    assert final.context_audit["newly_visible_source_spans"] == []
    assert "DENSE" not in content and '"type": "SEARCH"' not in content
    assert ("PREVIOUS ASSESSMENT" in content) == assessment
    assert final.assessment_status == ("provided" if assessment else "not_requested")
    assert (final.decision.assessment is not None) == assessment
    if assessment:
        assert final.context_audit["previous_assessment"]["turn"] == 2
        assert final.decision.assessment.missing_information == ["Unresolved information"]
    for step in result.trajectory:
        audit = step.context_audit
        assert not set(audit["new_section_references"]) & set(audit["old_section_references"])
        assert "Unresolved information" not in [s["text"] for s in step.visible_source_spans]


@pytest.mark.parametrize("assessment", [False, True])
def test_duplicate_loop_is_bounded_and_finalize_retains_gaps(built_substrate, fake_embedder, tmp_path, assessment):
    class Policy:
        last_usage = Usage(policy_calls=1)
        last_usage_metadata = {}
        count = 0
        def decide(self, messages, *, tools=None, **kwargs):
            assert tools is None
            self.count += 1
            if self.count <= 15:
                action = SearchAction(query="Marie Curie", method=SearchMethod.DENSE, target=SearchTarget.CHUNK)
            else:
                action = FinishAction(answer="Insufficient evidence", evidence_refs=[])
            current = Assessment(missing_information=["Unresolved gap"]) if assessment else None
            self.last_usage_metadata = {"constrained_single_decision": True, "decision_count": 1}
            return PolicyDecision(assessment=current, action=action)
    harness = AgentHarness(substrate=Substrate.open(built_substrate),
        config=AgentConfig(interface="C0", require_evidence_assessment=assessment),
        skill=SkillDocument.from_text("Answer using sources."), policy=Policy(),
        output_root=tmp_path / "bounded", embedding_backend=fake_embedder)
    result = harness.run("Where was Marie Curie born?", "q1")
    assert len(result.trajectory) == 16
    assert result.trajectory[-1].observation.metadata["budget_finalize"]
    assert sum(s.observation.error_code == "duplicate_action" for s in result.trajectory) == 14
    if assessment:
        assert result.trajectory[-1].decision.assessment.missing_information == ["Unresolved gap"]


def test_sentence_passage_promotion_preserves_alias_and_only_new_spans(built_substrate):
    substrate = Substrate.open(built_substrate)
    chunk = next(c for c in substrate.chunks if len(substrate.sentences_by_chunk[c.chunk_id]) > 1)
    sentences = substrate.sentences_by_chunk[chunk.chunk_id]
    state = EpisodeState.initial()
    state.visible_chunk_ids.add(chunk.chunk_id)
    state.reference_registry.register(chunk.chunk_id, "CHUNK")
    state.visible_sentence_ids.add(sentences[0].sentence_id)
    state.eligible_sentence_ids.add(sentences[0].sentence_id)
    state.semantic_memory_node_ids.append(sentences[0].sentence_id)
    sref = state.reference_registry.register(sentences[0].sentence_id, "SENTENCE")
    builder = PolicyContextBuilder(substrate, interface_contract=get_interface_contract("C1"))
    first = builder.build("Question", "Answer.", state, [])
    record = StepRecord.model_construct(policy_attempt=1, decision=None, resolved_decision=None,
        validation_status=ValidationStatus.VALID, observation=Observation(status=ObservationStatus.OK),
        visible_source_spans=first.visible_source_spans, context_reference_map=first.reference_map,
        provider_metadata={})
    state.visible_passage_ids.add(chunk.chunk_id)
    state.visible_chunk_ids.add(chunk.chunk_id)
    state.semantic_memory_node_ids.append(chunk.chunk_id)
    cref = state.reference_registry.register(chunk.chunk_id, "CHUNK")
    state.eligible_sentence_ids.update(s.sentence_id for s in sentences)
    promoted = builder.build("Question", "Answer.", state, [record])
    content = promoted.messages[-1].content
    assert content.count(sentences[0].text) == 1
    assert f"{sref}: sentence" in content
    assert promoted.reference_map.typed_refs[sref].can_use_as_evidence
    assert sref in promoted.available_action_space.finish_evidence_refs
    assert cref in promoted.available_action_space.finish_evidence_refs
    assert sentences[0].sentence_id not in {s["sentence_id"] for s in promoted.policy_view.context_audit["newly_visible_source_spans"]}
    assert len(promoted.visible_source_spans) == len(sentences)


@pytest.mark.parametrize("status,code", [("invalid_action", "source_not_visible"),
    ("invalid_action", "expansion_not_enabled"), ("invalid_action", "protocol_invalid"),
    ("error", "runtime_error"), ("error", "retrieved_token_budget_exceeded")])
def test_safe_error_feedback(status, code):
    record = StepRecord.model_construct(policy_attempt=1, decision=None, resolved_decision=None,
        validation_status=ValidationStatus.INVALID if status == "invalid_action" else ValidationStatus.VALID,
        observation=Observation(status=status, error_code=code, message="SECRET_INTERNAL_ID"))
    summary = action_summary(record)
    assert "SECRET_INTERNAL_ID" not in str(summary)
    assert "No new source text was added" in summary["explanation"]


def test_wrong_assessment_keys_receive_specific_safe_guidance():
    record = StepRecord.model_construct(policy_attempt=1, decision=None, resolved_decision=None,
        validation_status=ValidationStatus.INVALID,
        validation_error="arguments.assessment is missing a required argument",
        observation=Observation(status="invalid_action", error_code="protocol_invalid"))
    text = action_summary(record)["explanation"]
    assert "supported_facts and missing_information" in text
    assert "lists of strings" in text


def test_assessment_required_and_unknown_field_rejected(built_substrate):
    substrate = Substrate.open(built_substrate)
    for required in (False, True):
        built = PolicyContextBuilder(substrate, interface_contract=get_interface_contract("C0"),
            require_evidence_assessment=required).build("Q", "Skill", EpisodeState.initial(), [])
        payload = {"action": {"name": "find_passages", "query": "Q"}}
        if required:
            with pytest.raises(Exception):
                built.decision_format.model_validate(payload)
        else:
            built.decision_format.model_validate(payload)
            with pytest.raises(Exception):
                built.decision_format.model_validate({**payload, "supported_facts": [], "missing_information": []})


def test_success_with_no_new_source_and_reorganized_source():
    state = EpisodeState.initial()
    span = {"sentence_id": "s1", "chunk_id": "c1", "start": 0, "end": 4, "text": "Text"}
    record = StepRecord.model_construct(policy_attempt=1, decision=None, resolved_decision=None,
        validation_status=ValidationStatus.VALID, observation=Observation(status=ObservationStatus.OK),
        visible_source_spans=[span])
    blocks = [{"ref": "C1", "text": "Passage C1\nText", "sentence_ids": ["s1"]}]
    text, audit, _ = render_context(blocks, [span], [], [record], state, require_assessment=False)
    assert "The tool completed" in text
    assert "NEW SOURCE TEXT\nNone." in text
    assert audit["newly_visible_source_spans"] == []
    assert audit["old_section_references"] == ["C1"]


@pytest.mark.parametrize("field", ["renderer_sha256", "provider_protocol_sha256", "require_evidence_assessment"])
def test_resume_rejects_changed_context_or_assessment(tmp_path, monkeypatch, field):
    import importlib.util
    import json
    from types import SimpleNamespace
    script = Path(__file__).resolve().parents[1] / "scripts/run_interface_study.py"
    spec = importlib.util.spec_from_file_location("test_run_context_manifest", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = tmp_path / "config.yaml"
    config.write_text("agent:\n  interface: C0\n  require_evidence_assessment: true\n")
    questions = tmp_path / "questions.json"
    questions.write_text('[{"_id":"q1","question":"Q","answer":"A"}]')
    source = tmp_path / "source.json"
    source.write_text('{}')
    skill = tmp_path / "skill.md"
    skill.write_text('Answer from sources.')
    substrate = tmp_path / "substrate"
    substrate.mkdir()
    (substrate / "manifest.json").write_text('{"embedding_model":"qwen3-embedding:4b"}')
    output = tmp_path / "out"
    output.mkdir()
    args = SimpleNamespace(conditions=["C0"], questions=questions, source_manifest=source,
        substrate=substrate, config=config, skill=skill, output=output, dataset="hotpotqa",
        seed=20260805, limit=1, judge_config=None, resume=True)
    manifest = runner._manifest(args, AgentConfig.from_yaml(config), ("C0",))
    manifest[field] = False if field == "require_evidence_assessment" else "different-code"
    (output / "run_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runner.EvaluationSidecars, "open", lambda p: SimpleNamespace(benchmark_questions=[], gold_support=[]))
    monkeypatch.setattr(runner, "prepare_question_rows", lambda rows, *args: rows)
    with pytest.raises(ValueError, match="resume refused"):
        runner.run(args)
