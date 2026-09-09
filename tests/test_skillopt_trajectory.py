from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rag.agent.models import (
    Assessment,
    AvailableActionSpace,
    EpisodeResult,
    EpisodeState,
    ExpandAction,
    ExpansionKind,
    Observation,
    ObservationStatus,
    PolicyDecision,
    PolicyStateView,
    PolicyView,
    ReadAction,
    ResolvedDecision,
    ResolvedExpandAction,
    ResolvedReadAction,
    SearchAction,
    SearchMethod,
    SearchTarget,
    StepRecord,
    TerminationReason,
    Usage,
    ValidationStatus,
)
from agentic_rag.evaluation import EvaluationResult
from agentic_rag.skillopt.rollout import RolloutBatch, run_rollout_batch
from agentic_rag.skillopt.trajectory import (
    TrajectoryRepresentation,
    build_reflection_input,
)


def _assessment() -> Assessment:
    return Assessment(supported_facts=[], missing_information=["answer"])


def _advance(
    state: EpisodeState,
    *,
    consume_step: bool = True,
) -> EpisodeState:
    result = state.model_copy(deep=True)
    result.policy_attempts += 1
    result.remaining_policy_attempt_budget -= 1
    if consume_step:
        result.step += 1
        result.remaining_step_budget -= 1
    return result


def _episode() -> EpisodeResult:
    initial = EpisodeState.initial(
        max_steps=5,
        max_policy_attempts=6,
        max_retrieved_tokens=1_000,
    )

    search = SearchAction(
        query="Bridge Person",
        method=SearchMethod.LEXICAL,
        target=SearchTarget.ENTITY,
        top_k=5,
    )
    after_search = _advance(initial)
    after_search.visible_entity_ids.add("entity:bridge")
    after_search.semantic_memory_node_ids.append("entity:bridge")
    after_search.reference_registry.register("entity:bridge", "ENTITY")
    search_observation = Observation(
        action_id="a1",
        status=ObservationStatus.OK,
        action=search,
        results=[
            {
                "target": "ENTITY",
                "entity_id": "entity:bridge",
                "score": 0.99,
                "canonical_name": "Bridge Person",
                "entity_type": "PERSON",
                "mention_count": 3,
            }
        ],
        novel_node_ids=["entity:bridge"],
        metadata={"visibility_delta": {"visible_entity_ids": ["entity:bridge"]}},
    )
    step1 = StepRecord(
        step=1,
        policy_attempt=1,
        decision=PolicyDecision(assessment=_assessment(), action=search),
        resolved_decision=ResolvedDecision(assessment=_assessment(), action=search),
        validation_status=ValidationStatus.VALID,
        observation=search_observation,
        state_before=initial,
        state_after=after_search,
        policy_view=PolicyView(
            policy_state=PolicyStateView(
                step=0,
                policy_attempts=0,
                semantic_memory=[],
                latest_attempt=None,
                attempted_actions=[],
                budget="steps=5; policy_attempts=6; retrieval_tokens=1000",
            )
        ),
        available_action_space=AvailableActionSpace(),
        decision_schema_sha256="a" * 64,
    )

    expand_submitted = ExpandAction(
        kind=ExpansionKind.ENTITY_MENTIONED_IN_CHUNK,
        source_ref="E1",
        top_k=5,
    )
    expand_resolved = ResolvedExpandAction(
        kind=ExpansionKind.ENTITY_MENTIONED_IN_CHUNK,
        source_id="entity:bridge",
        top_k=5,
    )
    after_expand = _advance(after_search)
    after_expand.visible_chunk_ids.add("chunk:1")
    after_expand.visible_sentence_ids.add("sentence:gold")
    after_expand.semantic_memory_node_ids.extend(["chunk:1", "sentence:gold"])
    after_expand.reference_registry.register("chunk:1", "CHUNK")
    after_expand.reference_registry.register("sentence:gold", "SENTENCE")
    expand_observation = Observation(
        action_id="a2",
        status=ObservationStatus.OK,
        action=expand_resolved,
        results=[
            {
                "target": "CHUNK",
                "chunk_id": "chunk:1",
                "score": 0.9,
                "document_id": "doc:1",
                "title": "Gold Document",
                "navigation_only": True,
                "evidence_eligible": False,
                "content_read": False,
                "previews": [
                    {
                        "target": "SENTENCE_PREVIEW",
                        "sentence_id": "sentence:gold",
                        "score": 0.8,
                        "text": "The bridge reached the gold answer.",
                        "parent_chunk_id": "chunk:1",
                        "document_id": "doc:1",
                        "title": "Gold Document",
                        "navigation_only": True,
                        "evidence_eligible": False,
                        "paths": [],
                    }
                ],
                "paths": [
                    {
                        "relation": "MENTIONED_IN",
                        "node_ids": ["entity:bridge", "chunk:1"],
                        "bridge_sentence_ids": [],
                        "bridge_chunk_ids": [],
                    }
                ],
            }
        ],
        novel_node_ids=["chunk:1", "sentence:gold"],
        metadata={
            "visibility_delta": {
                "visible_chunk_ids": ["chunk:1"],
                "visible_sentence_ids": ["sentence:gold"],
            }
        },
    )
    step2 = StepRecord(
        step=2,
        policy_attempt=2,
        decision=PolicyDecision(
            assessment=_assessment(), action=expand_submitted
        ),
        resolved_decision=ResolvedDecision(
            assessment=_assessment(), action=expand_resolved
        ),
        validation_status=ValidationStatus.VALID,
        observation=expand_observation,
        state_before=after_search,
        state_after=after_expand,
    )

    read_submitted = ReadAction(chunk_ref="C1")
    read_resolved = ResolvedReadAction(chunk_id="chunk:1")
    after_read = _advance(after_expand)
    after_read.read_chunk_ids.add("chunk:1")
    after_read.eligible_sentence_ids.add("sentence:gold")
    read_observation = Observation(
        action_id="a3",
        status=ObservationStatus.OK,
        action=read_resolved,
        results=[
            {
                "chunk_id": "chunk:1",
                "doc_id": "doc:1",
                "title": "Gold Document",
                "chunk_pos": 0,
                "text": "The bridge reached the gold answer.",
                "sentences": [
                    {
                        "sentence_id": "sentence:gold",
                        "text": "The bridge reached the gold answer.",
                        "parent_chunk_id": "chunk:1",
                        "document_id": "doc:1",
                        "title": "Gold Document",
                    }
                ],
            }
        ],
        already_seen_node_ids=["chunk:1", "sentence:gold"],
        metadata={
            "visibility_delta": {
                "visible_chunk_ids": ["chunk:1"],
                "visible_sentence_ids": ["sentence:gold"],
                "eligible_sentence_ids": ["sentence:gold"],
                "read_chunk_ids": ["chunk:1"],
            }
        },
    )
    step3 = StepRecord(
        step=3,
        policy_attempt=3,
        decision=PolicyDecision(
            assessment=_assessment(), action=read_submitted
        ),
        resolved_decision=ResolvedDecision(
            assessment=_assessment(), action=read_resolved
        ),
        validation_status=ValidationStatus.VALID,
        observation=read_observation,
        state_before=after_expand,
        state_after=after_read,
    )

    after_invalid = _advance(after_read, consume_step=False)
    invalid_observation = Observation(
        action_id="a4",
        status=ObservationStatus.DUPLICATE_ACTION,
        action=search,
        error_code="duplicate_action",
        message="The same action was already executed.",
    )
    step4 = StepRecord(
        step=4,
        policy_attempt=4,
        decision=PolicyDecision(assessment=_assessment(), action=search),
        resolved_decision=ResolvedDecision(assessment=_assessment(), action=search),
        validation_status=ValidationStatus.INVALID,
        validation_error="The same action was already executed.",
        observation=invalid_observation,
        state_before=after_read,
        state_after=after_invalid,
    )
    return EpisodeResult(
        episode_id="episode-1",
        query="What did the bridge reach?",
        scope_id="scope",
        termination_reason=TerminationReason.BUDGET_EXHAUSTED,
        trajectory=[step1, step2, step3, step4],
        usage=Usage(policy_calls=4, retrieved_tokens=20),
        final_state=after_invalid,
        error_code="policy_attempt_budget_exhausted",
    )


def _item() -> dict:
    return {
        "id": "episode-1",
        "question": "What did the bridge reach?",
        "answer": "the gold answer",
        "source": "hotpotqa",
        "question_type": "bridge",
        "scope_id": "scope",
        "supporting_facts": [
            {
                "title": "Gold Document",
                "sentence_id": 0,
                "text": "The bridge reached the gold answer.",
            }
        ],
    }


def _build(representation: str, *, phase: str = "train") -> tuple[dict, dict]:
    return build_reflection_input(
        episode=_episode(),
        item=_item(),
        skill_content="Search, inspect, then answer.",
        rollout_phase=phase,
        rollout_split="train" if phase == "train" else "validation",
        trajectory_representation=representation,
        evaluation={
            "gold_answer": "the gold answer",
            "normalized_gold": "gold answer",
            "judge_messages": [{"content": "the gold answer"}],
            "status": "failed",
            "llm_acc": 0,
            "contain_acc": 0,
            "hard": 0,
            "soft": 0.0,
            "dataset": "hotpotqa",
        },
        target_system_prompt="Protocol used by the target.",
        effective_config={"agent": {"policy": {"model": "qwen"}}},
        sentence_provenance={
            "sentence:gold": {
                "title": "Gold Document",
                "original_sentence_id": 0,
            }
        },
    )


def test_raw_keeps_tool_audit_separate_from_policy_visible_delta() -> None:
    rendered, manifest = _build("raw")
    first = rendered["trajectory"]["raw_steps"][0]
    assert first["policy_input_state"]["policy_state"]["step"] == 0
    assert first["interface_context"]["mode"] == "NORMAL"
    assert first["interface_context"]["decision_schema_sha256"] == "a" * 64
    raw_step = rendered["trajectory"]["raw_steps"][1]
    assert raw_step["tool_raw"]["results"][0]["score"] == 0.9
    chunk = next(
        unit
        for unit in raw_step["policy_visible_delta"]["new_visible_units"]
        if unit["unit_type"] == "CHUNK"
    )
    assert chunk["canonical_text"] is None
    assert "step_diagnostics" not in rendered["trajectory"]
    assert manifest["support_labels_included"] is False


def test_organized_deduplicates_text_but_preserves_paths_and_problems() -> None:
    rendered, _ = _build("organized")
    trajectory = rendered["trajectory"]
    sentence = next(
        unit
        for unit in trajectory["retrieved_units"]
        if unit["stable_id"] == "sentence:gold"
    )
    assert sentence["canonical_text"] == "The bridge reached the gold answer."
    assert sentence["acquisition_steps"] == [2, 3]
    assert [path["mechanism"] for path in sentence["acquisition_paths"]] == [
        "EXPAND",
        "READ",
    ]
    assert "step_diagnostics" not in trajectory
    duplicate_problem = next(
        item
        for item in trajectory["problem_index"]
        if item["category"] == "duplicate_action"
    )
    assert duplicate_problem["steps"] == [4]
    assert len(trajectory["action_ledger"]) == 4
    assert trajectory["action_ledger"][0]["interface_context"][
        "decision_schema_sha256"
    ] == "a" * 64


def test_support_labels_distinguish_visible_eligible_and_read_upgrade() -> None:
    rendered, manifest = _build("organized_support_labels")
    diagnostics = rendered["trajectory"]["step_diagnostics"]

    expand = diagnostics[1]
    assert expand["next_policy_saw_result"] is True
    assert expand["supporting_fact_progress"]["policy_visible"]["coverage"]["recall"] == 1.0
    assert expand["supporting_fact_progress"]["policy_eligible"]["coverage"]["recall"] == 0.0
    assert expand["direct_expand_utility"]["new_policy_visible_gold_fact_ids"] == [
        "G1"
    ]
    assert expand["direct_expand_utility"]["direct_child_count"] == 1

    read = diagnostics[2]
    assert read["supporting_fact_progress"]["policy_visible"]["new_gold_fact_ids"] == []
    assert read["supporting_fact_progress"]["policy_eligible"]["new_gold_fact_ids"] == [
        "G1"
    ]
    assert {
        item["type"] for item in read["information_progress"]["visibility_upgrades"]
    } == {"chunk_handle_to_read", "sentence_preview_to_eligible"}

    invalid = diagnostics[3]
    assert invalid["execution"] == {
        "state": "not_executed",
        "outcome": "duplicate_action",
        "progress_evaluable": False,
    }
    assert invalid["information_progress"] is None
    assert invalid["supporting_fact_progress"] is None
    assert "answer_progress" not in json.dumps(rendered)
    assert manifest["support_labels_included"] is True


def test_progress_abstracted_removes_retrieval_text_and_links_later_bridge() -> None:
    rendered, manifest = _build("progress_abstracted")
    trajectory = rendered["trajectory"]
    steps = trajectory["abstract_steps"]

    assert trajectory["retrieval_text_included"] is False
    assert "retrieved_units" not in trajectory
    assert "action_ledger" not in trajectory
    assert "problem_index" not in trajectory
    assert "tool_raw" not in json.dumps(trajectory)
    assert "canonical_text" not in json.dumps(trajectory)
    assert "The bridge reached the gold answer." not in json.dumps(trajectory)

    acquisition = steps[0]
    assert acquisition["information_progress"]["new_unit_count"] == 1
    bridge_uses = acquisition[
        "later_expand_uses_of_acquired_information"
    ]
    assert len(bridge_uses) == 1
    assert bridge_uses[0]["evidence_stable_id"] == "entity:bridge"
    assert bridge_uses[0]["expand_step"] == 2
    assert bridge_uses[0]["new_unit_count"] == 2
    assert bridge_uses[0]["produced_new_information"] is True
    assert bridge_uses[0]["improved_supporting_fact_progress"] is True

    expand = steps[1]
    assert expand["direct_expand_progress"][
        "new_policy_visible_gold_fact_ids"
    ] == ["G1"]
    assert expand["supporting_fact_progress"]["policy_visible"][
        "coverage"
    ]["recall"] == 1.0
    assert trajectory["unattributed_later_expand_uses"] == []
    assert manifest["support_labels_included"] is True


def test_progress_abstracted_eval_keeps_progress_but_omits_gt_labels() -> None:
    rendered, manifest = _build("progress_abstracted", phase="eval")
    steps = rendered["trajectory"]["abstract_steps"]

    assert steps[0]["information_progress"]["new_unit_count"] == 1
    assert all(step["supporting_fact_progress"] is None for step in steps)
    bridge_use = steps[0]["later_expand_uses_of_acquired_information"][0]
    assert bridge_use["produced_new_information"] is True
    assert bridge_use["improved_supporting_fact_progress"] is None
    assert manifest["hidden_reference_included"] is False
    assert manifest["support_labels_included"] is False


def test_source_counts_use_document_ids_and_distinguish_top_level_hits() -> None:
    initial = EpisodeState.initial(
        max_steps=2,
        max_policy_attempts=3,
        max_retrieved_tokens=1_000,
    )
    action = SearchAction(
        query="Popular Science publication frequency",
        method=SearchMethod.BM25,
        target=SearchTarget.SENTENCE,
        top_k=5,
    )
    after = _advance(initial)
    chunk_ids = {"chunk:concept", "chunk:magazine"}
    sentence_ids = {"sentence:concept", "sentence:magazine"}
    after.visible_chunk_ids.update(chunk_ids)
    after.visible_sentence_ids.update(sentence_ids)
    after.semantic_memory_node_ids.extend(sorted(chunk_ids | sentence_ids))
    for stable_id in sorted(chunk_ids):
        after.reference_registry.register(stable_id, "CHUNK")
    for stable_id in sorted(sentence_ids):
        after.reference_registry.register(stable_id, "SENTENCE")
    observation = Observation(
        action_id="source-count",
        status=ObservationStatus.OK,
        action=action,
        results=[
            {
                "target": "SENTENCE",
                "sentence_id": "sentence:concept",
                "parent_chunk_id": "chunk:concept",
                "document_id": "doc:concept",
                "title": "Popular science",
                "text": "Popular science explains science for a general audience.",
            },
            {
                "target": "SENTENCE",
                "sentence_id": "sentence:magazine",
                "parent_chunk_id": "chunk:magazine",
                "document_id": "doc:magazine",
                "title": "Popular Science",
                "text": "Popular Science is an American magazine.",
            },
        ],
        novel_node_ids=sorted(chunk_ids | sentence_ids),
        metadata={
            "visibility_delta": {
                "visible_chunk_ids": sorted(chunk_ids),
                "visible_sentence_ids": sorted(sentence_ids),
            }
        },
    )
    step = StepRecord(
        step=1,
        policy_attempt=1,
        decision=PolicyDecision(assessment=_assessment(), action=action),
        resolved_decision=ResolvedDecision(assessment=_assessment(), action=action),
        validation_status=ValidationStatus.VALID,
        observation=observation,
        state_before=initial,
        state_after=after,
    )
    episode = EpisodeResult(
        episode_id="source-count",
        query="What is the publication frequency?",
        scope_id="scope",
        termination_reason=TerminationReason.BUDGET_EXHAUSTED,
        trajectory=[step],
        usage=Usage(policy_calls=1, retrieved_tokens=20),
        final_state=after,
        error_code="step_budget_exhausted",
    )
    rendered, _ = build_reflection_input(
        episode=episode,
        item=_item(),
        skill_content="Skill.",
        rollout_phase="train",
        rollout_split="train",
        trajectory_representation="organized_support_labels",
        sentence_provenance={},
    )
    trajectory = rendered["trajectory"]
    progress = trajectory["step_diagnostics"][0]["information_progress"]
    ledger = trajectory["action_ledger"][0]

    assert rendered["schema_version"] == "agentic-rag-skillopt-reflection-v5"
    assert progress["top_level_result_count"] == 2
    assert progress["returned_unit_count"] == 4
    assert progress["new_unit_count"] == 4
    assert progress["duplicate_unit_count"] == 0
    assert progress["returned_source_count"] == 2
    assert progress["new_source_count"] == 2
    assert ledger["top_level_result_count"] == 2
    assert ledger["returned_unit_count"] == 4
    assert ledger["new_source_count"] == 2
    assert {
        unit["source_document_id"] for unit in trajectory["retrieved_units"]
    } == {"doc:concept", "doc:magazine"}


def test_eval_reflection_omits_hidden_reference_and_derived_labels() -> None:
    rendered, manifest = _build("organized_support_labels", phase="eval")
    assert rendered["hidden_reference"] is None
    assert rendered["episode_outcome"]["evaluation"] == {
        "status": "failed",
        "llm_acc": 0,
        "contain_acc": 0,
        "hard": 0,
        "soft": 0.0,
        "dataset": "hotpotqa",
    }
    assert all(
        step["supporting_fact_progress"] is None
        for step in rendered["trajectory"]["step_diagnostics"]
    )
    assert manifest["hidden_reference_included"] is False
    assert manifest["support_labels_included"] is False
    assert manifest["supporting_fact_count"] is None
    assert manifest["sentence_provenance_entry_count"] == 0
    assert manifest["sentence_provenance_sha256"] is None


def test_executed_empty_has_zero_progress_not_na() -> None:
    episode = _episode()
    before = episode.final_state.model_copy(deep=True)
    action = SearchAction(
        query="nothing",
        method=SearchMethod.BM25,
        target=SearchTarget.SENTENCE,
        top_k=5,
    )
    after = _advance(before)
    step = StepRecord(
        step=5,
        policy_attempt=5,
        decision=PolicyDecision(assessment=_assessment(), action=action),
        resolved_decision=ResolvedDecision(assessment=_assessment(), action=action),
        validation_status=ValidationStatus.VALID,
        observation=Observation(
            action_id="a5",
            status=ObservationStatus.OK,
            action=action,
            results=[],
            metadata={"visibility_delta": {}},
        ),
        state_before=before,
        state_after=after,
    )
    episode.trajectory.append(step)
    episode.final_state = after
    rendered, _ = build_reflection_input(
        episode=episode,
        item=_item(),
        skill_content="Skill.",
        rollout_phase="train",
        rollout_split="train",
        trajectory_representation="organized_support_labels",
        sentence_provenance={
            "sentence:gold": {
                "title": "Gold Document",
                "original_sentence_id": 0,
            }
        },
    )
    last = rendered["trajectory"]["step_diagnostics"][-1]
    assert last["execution"]["progress_evaluable"] is True
    assert last["execution"]["outcome"] == "empty"
    assert last["information_progress"]["returned_unit_count"] == 0
    assert last["information_progress"]["new_unit_count"] == 0
    assert last["supporting_fact_progress"]["policy_visible"]["new_gold_fact_ids"] == []


def test_terminal_retrieval_is_environment_progress_not_policy_progress() -> None:
    episode = _episode()
    episode.trajectory = episode.trajectory[:2]
    episode.final_state = episode.trajectory[-1].state_after
    rendered, _ = build_reflection_input(
        episode=episode,
        item=_item(),
        skill_content="Skill.",
        rollout_phase="train",
        rollout_split="train",
        trajectory_representation="organized_support_labels",
        sentence_provenance={
            "sentence:gold": {
                "title": "Gold Document",
                "original_sentence_id": 0,
            }
        },
    )
    diagnostic = rendered["trajectory"]["step_diagnostics"][-1]
    progress = diagnostic["supporting_fact_progress"]
    assert diagnostic["next_policy_saw_result"] is False
    assert progress["environment_visible"]["coverage"]["recall"] == 1.0
    assert progress["policy_visible"]["coverage"]["recall"] == 0.0
    assert progress["policy_visible"]["new_gold_fact_ids"] == []


def test_provenance_mapping_selects_strict_mode_without_text_fallback() -> None:
    rendered, manifest = build_reflection_input(
        episode=_episode(),
        item=_item(),
        skill_content="Skill.",
        rollout_phase="train",
        rollout_split="train",
        trajectory_representation="organized_support_labels",
        sentence_provenance={},
    )
    metadata = rendered["trajectory"]["support_diagnostic_metadata"]
    assert metadata["provenance_mode"] == "strict_title_and_original_sentence_id"
    assert metadata["matched_stable_sentence_count"] == 0
    assert metadata["unmatched_visible_sentence_ids"] == ["sentence:gold"]
    assert manifest["unmatched_visible_sentence_count"] == 1


class _FakeHarness:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def run(
        self,
        question: str,
        scope_id: str,
        *,
        episode_id: str | None = None,
    ) -> EpisodeResult:
        del question, scope_id
        result = _episode()
        result.episode_id = str(episode_id)
        task_dir = self.output_root / str(episode_id)
        task_dir.mkdir(parents=True)
        result.artifact_dir = task_dir.as_posix()
        return result

    def build_io_trace(self, result: EpisodeResult) -> dict:
        return {"episode_id": result.episode_id}


class _FakeEvaluator:
    def evaluate(
        self,
        *,
        question: str,
        predicted_answer: str | None,
        gold_answer: str,
    ) -> EvaluationResult:
        return EvaluationResult(
            question=question,
            predicted_answer=predicted_answer or "",
            gold_answer=gold_answer,
            normalized_prediction="",
            normalized_gold="gold answer",
            status="failed",
            llm_acc=0,
            contain_acc=0,
            hard=0,
            soft=0.0,
            dataset="hotpotqa",
        )


def test_rollout_writes_condition_specific_reflection_artifacts(
    tmp_path: Path,
) -> None:
    def factory(*, skill_content: str, output_root: Path) -> _FakeHarness:
        assert skill_content == "Skill."
        return _FakeHarness(output_root)

    results = run_rollout_batch(
        batch=RolloutBatch(items=(_item(),), phase="train", split="train"),
        out_root=tmp_path,
        skill_content="Skill.",
        harness_factory=factory,
        evaluator=_FakeEvaluator(),  # type: ignore[arg-type]
        trajectory_representation="organized_support_labels",
        sentence_provenance={
            "sentence:gold": {
                "title": "Gold Document",
                "original_sentence_id": 0,
            }
        },
    )
    task_dir = tmp_path / "predictions" / "episode-1"
    conversation = json.loads(
        (task_dir / "reflection_conversation.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (task_dir / "reflection_input_manifest.json").read_text(encoding="utf-8")
    )
    assert conversation["trajectory_representation"] == "organized_support_labels"
    assert manifest["trajectory_representation"] == "organized_support_labels"
    assert manifest["truncated"] is False
    reference = json.loads(results[0]["reference_text"])
    assert reference == conversation["hidden_reference"]
    assert reference["answer"] == "the gold answer"
    assert reference["supporting_facts"][0]["sentence_id"] == 0


def test_native_reflect_reads_condition_mirror_not_legacy_conversation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from agentic_rag.skillopt.adapter import AgenticRAGSkillOptAdapter
    from skillopt.gradient.reflect import fmt_minibatch_trajectories
    import skillopt.gradient.reflect as reflect_module

    formatted: list[str] = []

    def fake_reflect(**kwargs):
        formatted.append(
            fmt_minibatch_trajectories(
                kwargs["results"], kwargs["prediction_dir"]
            )
        )
        return []

    monkeypatch.setattr(reflect_module, "run_minibatch_reflect", fake_reflect)

    for representation in (
        "raw",
        "organized",
        "organized_support_labels",
        "progress_abstracted",
    ):
        batch_dir = tmp_path / representation
        source_dir = batch_dir / "predictions" / "episode-1"
        source_dir.mkdir(parents=True)
        rendered, _ = _build(representation)
        (source_dir / "reflection_conversation.json").write_text(
            json.dumps(rendered), encoding="utf-8"
        )
        (source_dir / "conversation.json").write_text(
            json.dumps([{"role": "user", "content": "LEGACY_SENTINEL"}]),
            encoding="utf-8",
        )
        (source_dir / "target_system_prompt.txt").write_text(
            "exact target system", encoding="utf-8"
        )
        (source_dir / "target_user_prompt.txt").write_text(
            "exact target user", encoding="utf-8"
        )

        adapter = object.__new__(AgenticRAGSkillOptAdapter)
        adapter.trajectory_representation = TrajectoryRepresentation(representation)
        adapter.analyst_workers = 1
        adapter.failure_only = False
        adapter.minibatch_size = 1
        adapter.edit_budget = 1
        adapter._cfg = {}
        adapter.reflect(
            [
                {
                    "id": "episode-1",
                    "hard": 0,
                    "soft": 0.0,
                    "artifact_dir": source_dir.as_posix(),
                    "reference_text": json.dumps(
                        rendered["hidden_reference"], sort_keys=True
                    ),
                }
            ],
            "Skill.",
            batch_dir.as_posix(),
        )

        mirror = batch_dir / "reflection_predictions" / "episode-1"
        native_conversation = json.loads(
            (mirror / "conversation.json").read_text(encoding="utf-8")
        )
        assert isinstance(native_conversation, list)
        assert "representation_legend" in native_conversation[0]["content"]
        assert "top_level_result_count" in native_conversation[0]["content"]
        assert "returned_unit_count" in native_conversation[0]["content"]
        assert "stable document IDs" in native_conversation[0]["content"]
        assert "rollout_context" in native_conversation[1]["content"]
        assert "episode_outcome" in native_conversation[2]["content"]
        assert "trajectory" in native_conversation[3]["content"]
        assert (mirror / "target_system_prompt.txt").read_text() == "exact target system"
        assert (mirror / "target_user_prompt.txt").read_text() == "exact target user"

    assert all("LEGACY_SENTINEL" not in value for value in formatted)
    assert "raw_steps" in formatted[0]
    assert "action_ledger" in formatted[1]
    assert "step_diagnostics" in formatted[2]
    assert "abstract_steps" in formatted[3]
    assert "retrieved_units" not in formatted[3]
    assert "tool_raw" not in formatted[3]
    assert formatted[0] != formatted[1]


def _render_episode_for_test(episode, representation):
    return build_reflection_input(
        episode=episode,
        item=_item(),
        skill_content="Skill.",
        rollout_phase="train",
        rollout_split="train",
        trajectory_representation=representation,
    )[0]


def _steps_for_view(rendered):
    trajectory = rendered["trajectory"]
    return trajectory.get("raw_steps", trajectory.get("action_ledger", trajectory.get("abstract_steps")))


def test_all_four_views_share_decision_time_context_without_future_references():
    episode = _episode()
    views = [
        _render_episode_for_test(episode, representation)
        for representation in TrajectoryRepresentation
    ]
    contexts = [[step["decision_context"] for step in _steps_for_view(view)] for view in views]
    assert all(value == contexts[0] for value in contexts)
    first, expand, read, duplicate = contexts[0]
    assert first["missing_information"] == ["answer"]
    assert first["visible_references"] == []
    assert {row["stable_id"] for row in expand["visible_references"]} == {"entity:bridge"}
    chunk_before_read = next(row for row in read["visible_references"] if row["node_type"] == "CHUNK")
    chunk_after_read = next(row for row in duplicate["visible_references"] if row["node_type"] == "CHUNK")
    assert chunk_before_read["can_read"] is True
    assert chunk_before_read["can_use_as_evidence"] is False
    assert chunk_after_read["has_been_read"] is True
    assert chunk_after_read["can_read"] is False
    assert chunk_after_read["can_use_as_evidence"] is True
    assert first["remaining_budget"]["before"]["steps"] == 5
    assert first["remaining_budget"]["after"]["steps"] == 4
    assert first["available_action_space"] == episode.trajectory[0].available_action_space.model_dump(mode="json")


def test_common_context_respects_frozen_map_even_when_state_has_other_items():
    from agentic_rag.agent.models import ContextNodeReference, ContextReferenceMap

    episode = _episode()
    step = episode.trajectory[3]
    step.context_reference_map = ContextReferenceMap(typed_refs={
        "C1": ContextNodeReference(
            node_type="CHUNK", stable_id="chunk:1", can_read=False,
            can_use_as_evidence=True,
        ),
    })
    step.decision = None
    for representation in TrajectoryRepresentation:
        context = _steps_for_view(_render_episode_for_test(episode, representation))[3]["decision_context"]
        assert context["reference_source"] == "context_reference_map"
        assert [row["ref"] for row in context["visible_references"]] == ["C1"]
        assert context["missing_information"] is None


def test_common_context_policy_view_fallback_contains_no_memory_text():
    from agentic_rag.agent.models import ChunkMemoryItem

    episode = _episode()
    step = episode.trajectory[3]
    step.policy_view = PolicyView(policy_state=PolicyStateView(
        step=3, policy_attempts=3, budget="saved budget",
        semantic_memory=[ChunkMemoryItem(
            ref="C1", chunk_position=0, has_been_read=True, text="PRIVATE_MEMORY_TEXT",
        )],
    ))
    context = _steps_for_view(_render_episode_for_test(episode, "progress_abstracted"))[3]["decision_context"]
    assert context["reference_source"] == "policy_input_state"
    assert [row["ref"] for row in context["visible_references"]] == ["C1"]
    assert context["visible_references"][0]["has_been_read"] is True
    assert "PRIVATE_MEMORY_TEXT" not in json.dumps(context)


def test_organized_units_are_chronological_with_all_acquisition_paths():
    for representation in ("organized", "organized_support_labels"):
        units = _build(representation)[0]["trajectory"]["retrieved_units"]
        order = [(unit["first_seen_step"], unit["stable_id"]) for unit in units]
        assert order == sorted(order)
        assert units[0]["stable_id"] == "entity:bridge"
        assert len({unit["stable_id"] for unit in units}) == len(units)
        sentence = next(unit for unit in units if unit["stable_id"] == "sentence:gold")
        assert sentence["acquisition_steps"] == [2, 3]
        assert sentence["eligible_since_step"] == 3
        chunk = next(unit for unit in units if unit["stable_id"] == "chunk:1")
        assert chunk["read_since_step"] == 3
        assert [path["mechanism"] for path in chunk["acquisition_paths"]] == ["EXPAND", "READ"]


def test_actual_optimizer_input_removes_final_evidence_text_only_in_fourth_arm(tmp_path, monkeypatch):
    from agentic_rag.agent.models import SentenceRef, ResolvedEvidence
    from agentic_rag.skillopt.adapter import AgenticRAGSkillOptAdapter
    from skillopt.gradient.reflect import fmt_minibatch_trajectories
    import skillopt.gradient.reflect as reflect_module

    episode = _episode()
    sentinel = "RETRIEVED_FINAL_EVIDENCE_MUST_NOT_REACH_ABSTRACT_ANALYST"
    evidence = ResolvedEvidence(
        ref=SentenceRef(id="sentence:gold"),
        text=sentinel, document_id="doc:1", title="RETRIEVED_TITLE",
    )
    episode.answer = "ALLOWED_FINAL_ANSWER"
    episode.resolved_evidence = [evidence]
    episode.evidence_refs = [evidence.ref]
    step_text = "STEP_RETRIEVED_TEXT_MUST_NOT_REACH_ABSTRACT_ANALYST"
    episode.trajectory[2].observation.results[0]["text"] = step_text
    episode.trajectory[2].observation.results[0]["sentences"][0]["text"] = step_text
    episode.trajectory[0].decision.assessment.supported_facts = ["KNOWN_FACT_SUMMARY"]
    episode.trajectory[0].decision.assessment.missing_information = ["ALLOWED_MISSING_INFORMATION"]
    before = episode.model_dump_json()
    formatted = []

    def fake_reflect(**kwargs):
        formatted.append(fmt_minibatch_trajectories(kwargs["results"], kwargs["prediction_dir"]))
        return []

    monkeypatch.setattr(reflect_module, "run_minibatch_reflect", fake_reflect)
    for representation in TrajectoryRepresentation:
        rendered = _render_episode_for_test(episode, representation)
        source = tmp_path / representation / "predictions" / "episode-1"
        source.mkdir(parents=True)
        (source / "reflection_conversation.json").write_text(json.dumps(rendered))
        (source / "target_system_prompt.txt").write_text("ALLOWED_SYSTEM_PROMPT")
        (source / "target_user_prompt.txt").write_text("ALLOWED_ORIGINAL_QUERY")
        adapter = object.__new__(AgenticRAGSkillOptAdapter)
        adapter.trajectory_representation = representation
        adapter.analyst_workers = 1
        adapter.failure_only = False
        adapter.minibatch_size = 5
        adapter.edit_budget = 1
        adapter._cfg = {}
        adapter.reflect([{
            "id": "episode-1", "hard": 1, "artifact_dir": str(source),
            "reference_text": json.dumps(rendered["hidden_reference"]),
        }], "Skill.", str(tmp_path / representation))
        actual = formatted[-1]
        assert (sentinel in actual) == (representation != TrajectoryRepresentation.PROGRESS_ABSTRACTED)
        assert (step_text in actual) == (representation != TrajectoryRepresentation.PROGRESS_ABSTRACTED)
        for allowed in (
            "ALLOWED_FINAL_ANSWER", "ALLOWED_MISSING_INFORMATION", "ALLOWED_SYSTEM_PROMPT",
            "ALLOWED_ORIGINAL_QUERY", "the gold answer", "The bridge reached the gold answer.",
        ):
            assert allowed in actual
        if representation == TrajectoryRepresentation.PROGRESS_ABSTRACTED:
            assert "KNOWN_FACT_SUMMARY" not in actual
            assert "RETRIEVED_TITLE" not in actual
            assert set(rendered["episode_outcome"]["resolved_finish_evidence"][0]) == {
                "ref", "document_id", "parent_chunk_id", "contained_sentence_ids",
            }
    assert episode.model_dump_json() == before


def test_adapter_rejects_old_reflection_input_without_overwriting_it(tmp_path):
    from agentic_rag.skillopt.adapter import AgenticRAGSkillOptAdapter

    source = tmp_path / "predictions" / "episode-1"
    source.mkdir(parents=True)
    rendered, _ = _build("progress_abstracted")
    rendered["schema_version"] = "agentic-rag-skillopt-reflection-v2"
    path = source / "reflection_conversation.json"
    original = json.dumps(rendered)
    path.write_text(original)
    adapter = object.__new__(AgenticRAGSkillOptAdapter)
    adapter.trajectory_representation = TrajectoryRepresentation.PROGRESS_ABSTRACTED
    with pytest.raises(ValueError, match="use a new output directory"):
        adapter.reflect([{"id": "episode-1", "artifact_dir": str(source)}], "Skill.", str(tmp_path))
    assert path.read_text() == original
