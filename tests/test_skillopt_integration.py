from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentic_rag.agent.artifacts import ArtifactWriter
from agentic_rag.agent.models import (
    ControllerState,
    EpisodeResult,
    TerminationReason,
    Usage,
)
from agentic_rag.evaluation import (
    EpisodeEvaluator,
    JudgeTransportError,
    ScriptedJudge,
)
from agentic_rag.paths import portable_path_component
from agentic_rag.skillopt.adapter import AgenticRAGSkillOptAdapter
from agentic_rag.skillopt.dataloader import AgenticRAGSkillOptDataLoader
from agentic_rag.skillopt.rollout import RolloutBatch
from agentic_rag.skillopt.trainer import (
    _use_pinned_skillopt_prompts,
    run_skillopt_training,
    summarize_rollout_usage,
)


class _FakeHarness:
    def __init__(
        self,
        skill_content: str,
        output_root: Path,
        *,
        answer: str = "Warsaw",
    ) -> None:
        self.skill_content = skill_content
        self.output_root = output_root
        self.answer = answer
        self.run_calls: list[tuple[str, str, str]] = []

    def run(
        self,
        question: str,
        scope_id: str,
        *,
        episode_id: str | None = None,
    ) -> EpisodeResult:
        assert episode_id is not None
        self.run_calls.append((question, scope_id, episode_id))
        destination = self.output_root / portable_path_component(episode_id)
        result = EpisodeResult(
            episode_id=episode_id,
            query=question,
            scope_id=scope_id,
            termination_reason=TerminationReason.FINISH,
            answer=self.answer,
            usage=Usage(
                policy_calls=2,
                answer_calls=1,
                retrieved_tokens=7,
                policy_input_tokens=10,
                policy_output_tokens=3,
                policy_reasoning_tokens=2,
                answer_input_tokens=4,
                answer_output_tokens=1,
                total_tokens=20,
            ),
            artifact_dir=str(destination),
            final_state=ControllerState.initial(),
        )
        ArtifactWriter(self.output_root).write_episode(
            episode_id=episode_id,
            episode=result,
            target_system_prompt="fixed protocol",
            target_user_prompt=f"Original question:\n{question}",
            skill_content=self.skill_content,
            effective_config={"context_mode": "compact_evidence"},
        )
        return result

    def build_io_trace(self, result: EpisodeResult) -> dict[str, Any]:
        return {
            "trace_format": "agentic-rag-logical-io-v2",
            "target_input": {
                "question": result.query,
                "scope_id": result.scope_id,
            },
            "policy_calls": [],
            "answer_generation": {
                "input": {"question": result.query, "evidence": []},
                "output": {"answer": result.answer},
            },
            "total_usage": result.usage.model_dump(mode="json"),
        }


class _HarnessFactory:
    def __init__(self, answer_for_skill: Any | None = None) -> None:
        self.harnesses: list[_FakeHarness] = []
        self.answer_for_skill = answer_for_skill

    def __call__(
        self, *, skill_content: str, output_root: Path
    ) -> _FakeHarness:
        answer = (
            self.answer_for_skill(skill_content)
            if self.answer_for_skill is not None
            else "Warsaw"
        )
        harness = _FakeHarness(skill_content, output_root, answer=answer)
        self.harnesses.append(harness)
        return harness


def _item(prefix: str, index: int) -> dict[str, str]:
    return {
        "id": f"{prefix}-{index}",
        "question": f"Where was subject {prefix}-{index} born?",
        "scope_id": "hotpotqa:benchmark_exact:dev",
        "answer": "Warsaw",
        "question_type": "bridge" if index < 4 else "comparison",
        "source": "hotpotqa",
    }


def _write_splits(root: Path, *, count: int = 6) -> None:
    root.mkdir(parents=True)
    filenames = {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
    }
    manifest = {"seed": 42, "counts": {key: count for key in filenames}}
    (root / "split_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for split, filename in filenames.items():
        payload = "\n".join(
            json.dumps(_item(split, index)) for index in range(count)
        )
        (root / filename).write_text(payload + "\n", encoding="utf-8")


def test_dataloader_plans_two_disjoint_batches_of_three(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    _write_splits(split_dir)
    loader = AgenticRAGSkillOptDataLoader(split_dir)
    loader.setup({})

    batches = loader.plan_train_epoch(
        epoch=0,
        steps_per_epoch=2,
        accumulation=1,
        batch_size=3,
        seed=42,
    )

    assert loader.get_train_size() == 6
    assert [batch.batch_size for batch in batches] == [3, 3]
    ids = [item["id"] for batch in batches for item in batch.payload]
    assert len(ids) == len(set(ids)) == 6
    assert len(loader.get_split_items("valid_seen")) == 6
    assert len(loader.get_split_items("valid_unseen")) == 6


def test_dataloader_rejects_ids_shared_across_splits(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    _write_splits(split_dir, count=1)
    duplicate = _item("train", 0)
    (split_dir / "validation.jsonl").write_text(
        json.dumps(duplicate) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        AgenticRAGSkillOptDataLoader(split_dir).setup({})


def test_rollout_writes_per_task_contract_and_hides_eval_reference(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "splits"
    _write_splits(split_dir)
    factory = _HarnessFactory()
    judge = ScriptedJudge([True, True])
    adapter = AgenticRAGSkillOptAdapter(
        split_dir=split_dir,
        harness_factory=factory,
        evaluator=EpisodeEvaluator(judge),
    )
    adapter.setup({})

    train_result = adapter.rollout(
        RolloutBatch((_item("train", 0),), "train", "train"),
        "# Candidate skill\n",
        str(tmp_path / "train_step"),
    )[0]
    eval_result = adapter.rollout(
        RolloutBatch((_item("validation", 0),), "eval", "valid_seen"),
        "# Candidate skill\n",
        str(tmp_path / "validation"),
    )[0]

    assert train_result["reference_text"] == "Warsaw"
    assert "reference_text" not in eval_result
    assert train_result["hard"] == 1
    assert train_result["soft"] == 1.0
    task_dir = tmp_path / "train_step" / "predictions" / "train-0"
    assert {path.name for path in task_dir.iterdir()} == {
        "episode.json",
        "conversation.json",
        "target_system_prompt.txt",
        "target_user_prompt.txt",
        "skill.md",
        "effective_config.json",
        "io_trace.json",
        "evaluation.json",
        "rollout_result.json",
    }
    trace = json.loads((task_dir / "io_trace.json").read_text("utf-8"))
    assert set(trace["target_input"]) == {"question", "scope_id"}
    assert "answer" not in trace["target_input"]
    persisted_eval = json.loads(
        (task_dir / "evaluation.json").read_text("utf-8")
    )
    assert persisted_eval["gold_answer"] == "Warsaw"

    resumed = adapter.rollout(
        RolloutBatch((_item("train", 0),), "train", "train"),
        "# Candidate skill\n",
        str(tmp_path / "train_step"),
    )
    assert resumed == [train_result]
    assert factory.harnesses[-1].run_calls == []

    # A transport interruption can leave the six Harness files and evaluation
    # behind before rollout_result.json is published. Resume completes the
    # contract without paying for a second Agent or Judge call.
    (task_dir / "rollout_result.json").unlink()
    resumed_partial = adapter.rollout(
        RolloutBatch((_item("train", 0),), "train", "train"),
        "# Candidate skill\n",
        str(tmp_path / "train_step"),
    )
    assert resumed_partial == [train_result]
    assert factory.harnesses[-1].run_calls == []
    assert len(judge.calls) == 2


def test_adapter_rejects_parallel_smoke_workers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers=1"):
        AgenticRAGSkillOptAdapter(
            split_dir=tmp_path,
            harness_factory=_HarnessFactory(),
            evaluator=EpisodeEvaluator(ScriptedJudge(True)),
            workers=2,
        )


def test_judge_transport_retry_recovers_episode_without_rerunning_target(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "splits"
    _write_splits(split_dir)
    factory = _HarnessFactory()
    judge = ScriptedJudge(
        [JudgeTransportError("temporary judge outage"), True]
    )
    adapter = AgenticRAGSkillOptAdapter(
        split_dir=split_dir,
        harness_factory=factory,
        evaluator=EpisodeEvaluator(judge),
    )
    adapter.setup({})
    batch = RolloutBatch((_item("train", 1),), "train", "train")
    out_root = tmp_path / "judge_retry"

    with pytest.raises(JudgeTransportError, match="temporary judge outage"):
        adapter.rollout(batch, "# Candidate skill\n", str(out_root))

    task_dir = out_root / "predictions" / "train-1"
    assert (task_dir / "episode.json").is_file()
    assert (task_dir / "io_trace.json").is_file()
    assert not (task_dir / "evaluation.json").exists()
    assert not (task_dir / "rollout_result.json").exists()
    episode_before = (task_dir / "episode.json").read_bytes()
    assert factory.harnesses[0].run_calls == [
        (
            "Where was subject train-1 born?",
            "hotpotqa:benchmark_exact:dev",
            "train-1",
        )
    ]

    recovered = adapter.rollout(
        batch,
        "# Candidate skill\n",
        str(out_root),
    )

    assert recovered[0]["hard"] == 1
    assert factory.harnesses[-1].run_calls == []
    assert len(judge.calls) == 2
    assert (task_dir / "episode.json").read_bytes() == episode_before
    assert (task_dir / "evaluation.json").is_file()
    assert (task_dir / "rollout_result.json").is_file()


def test_attach_reference_context_is_train_only(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    _write_splits(split_dir)
    adapter = AgenticRAGSkillOptAdapter(
        split_dir=split_dir,
        harness_factory=_HarnessFactory(),
        evaluator=EpisodeEvaluator(ScriptedJudge(True)),
    )
    item = _item("train", 0)
    attached = adapter.attach_reference_context(
        [
            {"id": item["id"], "rollout_phase": "train"},
            {
                "id": item["id"],
                "rollout_phase": "eval",
                "reference_text": "must disappear",
            },
        ],
        [item],
    )
    assert attached[0]["reference_text"] == "Warsaw"
    assert "reference_text" not in attached[1]


def test_pinned_prompt_bundle_supplies_missing_skillopt_package_data(
    tmp_path: Path,
) -> None:
    prompts = pytest.importorskip("skillopt.prompts")
    original_prompt_dir = prompts._PROMPTS_DIR

    with _use_pinned_skillopt_prompts(tmp_path):
        assert "failure-analysis agent" in prompts.load_prompt(
            "analyst_error"
        )
        assert "FINAL merge" in prompts.load_prompt("merge_final")
        assert "selected_indices" in prompts.load_prompt("ranking")

    assert prompts._PROMPTS_DIR == original_prompt_dir
    metadata = json.loads(
        (tmp_path / "skillopt_prompt_bundle.json").read_text("utf-8")
    )
    assert metadata["tag"] == "v0.2.0"
    assert set(metadata["files"]) == {
        "analyst_error.md",
        "analyst_success.md",
        "merge_failure.md",
        "merge_success.md",
        "merge_final.md",
        "ranking.md",
    }


class _FakeSkillOptimizer:
    """Zero-cost stand-in for SkillOpt reflect/update/gate stages."""

    def __init__(self) -> None:
        self.reflect_calls: list[list[dict[str, Any]]] = []
        self.update_calls: list[tuple[str, dict[str, str]]] = []
        self.gate_calls: list[tuple[float, float]] = []

    def reflect(self, results: list[dict[str, Any]]) -> dict[str, str]:
        self.reflect_calls.append(results)
        return {
            "op": "append",
            "content": "Use complete questions.",
        }

    def update(self, skill: str, patch: dict[str, str]) -> str:
        self.update_calls.append((skill, patch))
        marker = patch["content"]
        if marker in skill:
            return skill + "\nKeep selected evidence across hops.\n"
        return skill + f"\n{marker}\n"

    def gate(self, current_score: float, candidate_score: float) -> bool:
        self.gate_calls.append((current_score, candidate_score))
        return candidate_score > current_score


class _FakeSkillOptTrainer:
    def __init__(
        self,
        cfg: dict[str, Any],
        adapter: Any,
        optimizer: _FakeSkillOptimizer,
    ) -> None:
        self.cfg = cfg
        self.adapter = adapter
        self.optimizer = optimizer

    def train(self) -> dict[str, Any]:
        self.adapter.setup(self.cfg)
        loader = self.adapter.get_dataloader()
        initial_skill = "# Initial\n"
        selection = self.adapter.build_eval_env(6, "valid_seen", 42)
        baseline_selection = self.adapter.rollout(
            selection,
            initial_skill,
            str(Path(self.cfg["out_root"]) / "selection_baseline"),
        )
        current_skill = initial_skill
        current_score = sum(row["hard"] for row in baseline_selection) / 6
        best_skill = initial_skill
        best_score = current_score
        best_step = 0
        batches = loader.plan_train_epoch(
            epoch=0,
            steps_per_epoch=2,
            accumulation=1,
            batch_size=3,
            seed=42,
        )
        for index, batch in enumerate(batches):
            train_results = self.adapter.rollout(
                self.adapter.build_env_from_batch(batch),
                current_skill,
                str(
                    Path(self.cfg["out_root"])
                    / f"step_{index + 1:04d}"
                    / "rollout"
                ),
            )
            patch = self.optimizer.reflect(train_results)
            candidate = self.optimizer.update(current_skill, patch)
            validation = self.adapter.build_eval_env(6, "valid_seen", 42)
            candidate_results = self.adapter.rollout(
                validation,
                candidate,
                str(
                    Path(self.cfg["out_root"])
                    / f"step_{index + 1:04d}"
                    / "selection"
                ),
            )
            candidate_score = (
                sum(row["hard"] for row in candidate_results) / 6
            )
            if self.optimizer.gate(current_score, candidate_score):
                current_skill = candidate
                current_score = candidate_score
            if candidate_score > best_score:
                best_skill = candidate
                best_score = candidate_score
                best_step = index + 1

        (Path(self.cfg["out_root"]) / "best_skill.md").write_text(
            best_skill, encoding="utf-8"
        )
        test = self.adapter.build_eval_env(6, "valid_unseen", 42)
        baseline_test = self.adapter.rollout(
            test,
            initial_skill,
            str(Path(self.cfg["out_root"]) / "test_baseline"),
        )
        best_test = self.adapter.rollout(
            test,
            best_skill,
            str(Path(self.cfg["out_root"]) / "test"),
        )
        baseline_test_score = sum(row["hard"] for row in baseline_test) / 6
        best_test_score = sum(row["hard"] for row in best_test) / 6
        return {
            "baseline_selection_hard": sum(
                row["hard"] for row in baseline_selection
            )
            / 6,
            "best_selection_hard": best_score,
            "final_selection_hard": current_score,
            "final_selection_soft": current_score,
            "baseline_test_hard": baseline_test_score,
            "baseline_test_soft": baseline_test_score,
            "test_hard": best_test_score,
            "test_soft": best_test_score,
            "final_test_hard": best_test_score,
            "final_test_soft": best_test_score,
            "best_step": best_step,
            "token_summary": {
                "reflection": {
                    "calls": len(self.optimizer.reflect_calls),
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                }
            },
        }


def test_fake_skillopt_workflow_runs_train_gate_and_test(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    _write_splits(split_dir)
    factory = _HarnessFactory(
        lambda skill: (
            "Warsaw" if "Use complete questions." in skill else "Unknown"
        )
    )
    optimizer = _FakeSkillOptimizer()
    adapter = AgenticRAGSkillOptAdapter(
        split_dir=split_dir,
        harness_factory=factory,
        evaluator=EpisodeEvaluator(
            ScriptedJudge(
                [False] * 9
                + [True] * 15
                + [False] * 6
                + [True] * 6
            )
        ),
    )
    out_root = tmp_path / "skillopt_run"

    summary = run_skillopt_training(
        {"out_root": str(out_root), "split_dir": str(split_dir)},
        adapter,
        trainer_cls=lambda cfg, env: _FakeSkillOptTrainer(
            cfg, env, optimizer
        ),
    )

    assert summary["test_hard"] == 1.0
    assert summary["baseline_selection_hard"] == 0.0
    assert summary["baseline_test_hard"] == 0.0
    assert summary["best_step"] == 1
    assert len(optimizer.reflect_calls) == 2
    assert len(optimizer.update_calls) == 2
    assert optimizer.gate_calls == [(0.0, 1.0), (1.0, 1.0)]
    assert (out_root / "best_skill.md").is_file()
    assert "Use complete questions." in (
        out_root / "best_skill.md"
    ).read_text("utf-8")
    assert (out_root / "split_manifest.json").is_file()
    workflow = json.loads(
        (out_root / "workflow_summary.json").read_text("utf-8")
    )
    assert workflow["run_kind"] == "workflow_smoke"
    assert workflow["paper_parity"] is False
    assert workflow["unique_question_count"] == 18
    assert workflow["episode_execution_count"] == 36
    assert workflow["target_and_judge_usage"]["episodes"] == 36
    assert workflow["target_and_judge_usage"]["policy"]["calls"] == 72
    assert workflow["optimizer_usage"]["reflection"]["total_tokens"] == 25
    initial_metrics = json.loads(
        (out_root / "initial_metrics.json").read_text("utf-8")
    )
    final_metrics = json.loads(
        (out_root / "final_metrics.json").read_text("utf-8")
    )
    token_summary = json.loads(
        (out_root / "token_summary.json").read_text("utf-8")
    )
    assert initial_metrics["test"]["hard_llm_acc"] == 0.0
    assert final_metrics["test"]["best_hard_llm_acc"] == 1.0
    assert token_summary["optimizer"]["reflection"]["total_tokens"] == 25
    assert len(list(out_root.rglob("rollout_result.json"))) == 36

    usage = summarize_rollout_usage(out_root)
    assert usage["answer"]["calls"] == 36
    assert usage["judge"]["calls"] == 36
    assert usage["retrieved_tokens"] == 36 * 7
