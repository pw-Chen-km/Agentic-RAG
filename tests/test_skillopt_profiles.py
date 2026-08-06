from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from agentic_rag import cli
from agentic_rag.agent.answer import answer_provider_input
from agentic_rag.agent.config import AgentConfig
from agentic_rag.benchmark_profiles import ARAG_DATASET_PROFILES, AnswerMode
from agentic_rag.cli import app
from agentic_rag.evaluation import EpisodeEvaluator, ScriptedJudge
from agentic_rag.models import SourceArtifact
from agentic_rag.skillopt.adapter import AgenticRAGSkillOptAdapter
from agentic_rag.skillopt.benchmark import (
    load_arag_split,
    prepare_arag_smoke_splits,
    smoke_type_quotas,
    validate_arag_smoke_lineage,
)
from agentic_rag.skillopt.trainer import run_skillopt_training


runner = CliRunner()


EXPECTED_QUOTAS = {
    "musique": {"2_hop": 3, "3_hop": 2, "4_hop": 1},
    "hotpotqa": {"bridge": 4, "comparison": 2},
    "2wikimultihop": {
        "bridge_comparison": 1,
        "comparison": 2,
        "compositional": 2,
        "inference": 1,
    },
    "medical": {
        "complex_reasoning": 2,
        "contextual_summarize": 1,
        "creative_generation": 1,
        "fact_retrieval": 2,
    },
    "novel": {
        "complex_reasoning": 2,
        "contextual_summarize": 1,
        "creative_generation": 1,
        "fact_retrieval": 2,
    },
}


def _write_profile_dataset(root: Path, dataset: str) -> Path:
    profile = ARAG_DATASET_PROFILES[dataset]
    quotas = EXPECTED_QUOTAS[dataset]
    data_dir = root / dataset
    data_dir.mkdir(parents=True)
    questions: list[dict[str, Any]] = []
    row = 0
    for task_type in profile.allowed_task_types:
        count = max(quotas.get(task_type, 0) * 3, 1)
        for index in range(count):
            source_id = f"{dataset}-{task_type}-{index}"
            if dataset == "novel" and row < 2:
                source_id = "duplicate-novel-source-id"
            questions.append(
                {
                    "id": source_id,
                    "source": dataset,
                    "question": f"Question {row} for {dataset}?",
                    "answer": f"Gold answer {row} for {dataset}.",
                    "question_type": task_type,
                    "evidence": f"MUST_NOT_ENTER_SPLIT_MANIFEST_{row}",
                }
            )
            row += 1
    (data_dir / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False), encoding="utf-8"
    )
    (data_dir / "chunks.json").write_text(
        json.dumps(
            ["0:First source Chunk.", "1:Second source Chunk."],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def _substrate_manifest(manifest: dict[str, Any]) -> SimpleNamespace:
    dataset = manifest["dataset"]
    return SimpleNamespace(
        dataset=dataset["subset"],
        split=dataset["benchmark_split"],
        source_format="arag_benchmark_exact",
        record_counts={
            "chunks": dataset["chunk_count"],
            "benchmark_questions": dataset["question_count"],
        },
        source_artifacts=[
            SourceArtifact(
                role=role,
                path=metadata["path"],
                sha256=metadata["sha256"],
                size_bytes=metadata["size_bytes"],
            )
            for role, metadata in dataset["source_files"].items()
        ],
    )


@pytest.mark.parametrize("dataset", sorted(ARAG_DATASET_PROFILES))
def test_profile_driven_prepare_load_and_lineage(
    tmp_path: Path, dataset: str
) -> None:
    dataset_dir = _write_profile_dataset(tmp_path / "source", dataset)
    split_dir = tmp_path / "splits"

    manifest = prepare_arag_smoke_splits(
        dataset_dir,
        split_dir,
        dataset=dataset,
        validate_reference_counts=False,
    )

    assert smoke_type_quotas(dataset) == EXPECTED_QUOTAS[dataset]
    assert manifest["schema_version"] == "2.0"
    assert manifest["dataset"]["subset"] == dataset
    assert manifest["dataset"]["metric_contract"] == {
        "reported": [
            metric.value
            for metric in ARAG_DATASET_PROFILES[dataset].reported_metrics
        ],
        "hard": "llm_acc",
        "soft": (
            "llm_acc" if dataset in {"medical", "novel"} else "contain_acc"
        ),
    }
    assert manifest["selection"]["per_split_by_task_type"] == (
        EXPECTED_QUOTAS[dataset]
    )
    assert "MUST_NOT_ENTER_SPLIT_MANIFEST" not in json.dumps(manifest)

    all_ids: list[str] = []
    for split_name in ("train", "validation", "test"):
        items = load_arag_split(
            split_dir / f"{split_name}.jsonl",
            dataset=dataset,
        )
        assert len(items) == 6
        assert {item.scope_id for item in items} == {
            f"{dataset}:benchmark_exact:dev"
        }
        assert all(
            item.id
            == f"{dataset}:benchmark_exact:q:{item.source_row_index:06d}"
            for item in items
        )
        all_ids.extend(item.id for item in items)
    assert len(all_ids) == len(set(all_ids)) == 18

    report = validate_arag_smoke_lineage(
        split_dir,
        _substrate_manifest(manifest),
        dataset=dataset,
    )
    assert report["valid"] is True
    assert report["dataset"] == dataset
    assert report["total_question_count"] == 18


def test_profile_evaluator_changes_only_skillopt_metric_mapping() -> None:
    hotpot = EpisodeEvaluator(
        ScriptedJudge(False), profile="hotpotqa"
    ).evaluate(
        question="Where?",
        predicted_answer="Warsaw with a contradiction.",
        gold_answer="Warsaw",
    )
    medical = EpisodeEvaluator(
        ScriptedJudge(False), profile="medical"
    ).evaluate(
        question="Explain the diagnosis.",
        predicted_answer="The gold diagnosis, but contradicted.",
        gold_answer="gold diagnosis",
    )

    assert hotpot.llm_acc == 0 and hotpot.contain_acc == 1
    assert hotpot.hard == 0 and hotpot.soft == 1.0
    assert medical.llm_acc == 0 and medical.contain_acc == 1
    assert medical.hard == 0 and medical.soft == 0.0
    assert medical.soft_metric.value == "llm_acc"
    assert [metric.value for metric in medical.reported_metrics] == ["llm_acc"]


def test_answer_prompt_respects_profile_answer_mode() -> None:
    short_prompt = answer_provider_input("Question?", [])[0]["content"]
    long_prompt = answer_provider_input(
        "Question?", [], answer_mode=AnswerMode.LONG
    )[0]["content"]

    assert "Return the shortest complete answer" in short_prompt
    assert "Do not shorten a compound answer" in short_prompt
    assert "complete, well-supported answer" in long_prompt
    assert "unsupported claims" in long_prompt


class _UnusedHarnessFactory:
    def __call__(self, **kwargs: object) -> object:
        raise AssertionError(f"rollout should not run: {kwargs}")


class _SummaryOnlyTrainer:
    def __init__(self, cfg: dict[str, Any], adapter: Any) -> None:
        self.cfg = cfg
        self.adapter = adapter

    def train(self) -> dict[str, Any]:
        self.adapter.setup(self.cfg)
        return {
            "baseline_selection_hard": 0.25,
            "baseline_test_hard": 0.5,
            "baseline_test_soft": 0.5,
            "best_selection_hard": 0.75,
            "final_selection_hard": 0.75,
            "final_selection_soft": 0.75,
            "test_hard": 1.0,
            "test_soft": 1.0,
            "final_test_hard": 1.0,
            "final_test_soft": 1.0,
            "best_step": 1,
            "token_summary": {},
        }


def test_trainer_artifacts_use_profile_metric_names(tmp_path: Path) -> None:
    dataset_dir = _write_profile_dataset(tmp_path / "source", "medical")
    split_dir = tmp_path / "splits"
    prepare_arag_smoke_splits(
        dataset_dir,
        split_dir,
        dataset="medical",
        validate_reference_counts=False,
    )
    adapter = AgenticRAGSkillOptAdapter(
        split_dir=split_dir,
        harness_factory=_UnusedHarnessFactory(),
        evaluator=EpisodeEvaluator(ScriptedJudge(True), profile="medical"),
        dataset="medical",
    )
    out_root = tmp_path / "run"

    run_skillopt_training(
        {
            "out_root": str(out_root),
            "split_dir": str(split_dir),
            "dataset": "medical",
        },
        adapter,
        trainer_cls=_SummaryOnlyTrainer,
    )

    initial = json.loads(
        (out_root / "initial_metrics.json").read_text(encoding="utf-8")
    )
    final = json.loads(
        (out_root / "final_metrics.json").read_text(encoding="utf-8")
    )
    workflow = json.loads(
        (out_root / "workflow_summary.json").read_text(encoding="utf-8")
    )
    assert initial["test"] == {
        "hard_llm_acc": 0.5,
        "soft_llm_acc": 0.5,
    }
    assert final["test"]["best_soft_llm_acc"] == 1.0
    assert "best_soft_contain_acc" not in final["test"]
    assert workflow["dataset"] == "medical"
    assert workflow["unique_question_count"] == 18


def test_skillopt_train_cli_infers_generic_dataset_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = _write_profile_dataset(tmp_path / "source", "medical")
    split_dir = tmp_path / "splits"
    split_manifest = prepare_arag_smoke_splits(
        dataset_dir,
        split_dir,
        dataset="medical",
        validate_reference_counts=False,
    )
    substrate_path = tmp_path / "substrate"
    substrate_path.mkdir()
    required_scopes: list[str] = []
    fake_substrate = SimpleNamespace(
        manifest=_substrate_manifest(split_manifest),
        require_scope=lambda scope: required_scopes.append(scope),
    )
    monkeypatch.setattr(
        cli,
        "Substrate",
        SimpleNamespace(open=lambda path: fake_substrate),
    )

    agent_config_path = tmp_path / "agent.yaml"
    agent_config_path.write_text("agent: {}\n", encoding="utf-8")
    skillopt_config_path = tmp_path / "skillopt.yaml"
    skillopt_config_path.write_text("train: {}\n", encoding="utf-8")
    skill_file = tmp_path / "initial.md"
    skill_file.write_text("# Initial skill\n", encoding="utf-8")
    output = tmp_path / "run"
    agent_config = AgentConfig.model_validate(
        {
            "policy": {"provider": "openai", "model": "gpt-5.6-luna"},
            "answer": {"provider": "openai", "model": "gpt-5.6-luna"},
        }
    )
    monkeypatch.setattr(
        cli.AgentConfig, "from_yaml", lambda path: agent_config
    )
    flat_config = {
        "optimizer_model": "gpt-5.6-luna",
        "target_model": "gpt-5.6-luna",
        "batch_size": 3,
        "num_epochs": 1,
        "train_size": 6,
        "accumulation": 1,
        "seed": 42,
        "minibatch_size": 3,
        "merge_batch_size": 3,
        "edit_budget": 1,
        "min_edit_budget": 1,
        "lr_scheduler": "constant",
        "skill_update_mode": "patch",
        "use_slow_update": False,
        "use_meta_skill": False,
        "analyst_workers": 1,
        "max_analyst_rounds": 1,
        "failure_only": False,
        "use_gate": True,
        "sel_env_num": 6,
        "test_env_num": 6,
        "eval_test": True,
        "workers": 1,
        "judge_model": "gpt-5.6-luna",
        "azure_openai_endpoint": "https://api.openai.com/v1",
        "azure_openai_auth_mode": "openai_compatible",
    }
    monkeypatch.setattr(
        cli, "load_skillopt_config", lambda path: dict(flat_config)
    )
    monkeypatch.setattr(
        cli, "OpenAIResponsesJudge", lambda model: SimpleNamespace()
    )
    captured: dict[str, Any] = {}

    class FakeAdapter:
        def __init__(self, **kwargs: Any) -> None:
            captured["adapter_kwargs"] = kwargs

    monkeypatch.setattr(cli, "AgenticRAGSkillOptAdapter", FakeAdapter)

    def fake_training(config: dict[str, Any], adapter: object) -> dict[str, Any]:
        captured["config"] = config
        captured["adapter"] = adapter
        return {"test_hard": 1.0}

    monkeypatch.setattr(cli, "run_skillopt_training", fake_training)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    result = runner.invoke(
        app,
        [
            "skillopt-train",
            str(substrate_path),
            "--split-dir",
            str(split_dir),
            "--agent-config",
            str(agent_config_path),
            "--skillopt-config",
            str(skillopt_config_path),
            "--skill-file",
            str(skill_file),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dataset"] == "medical"
    assert payload["scope_id"] == "medical:benchmark_exact:dev"
    assert payload["metric_contract"] == {
        "reported": ["llm_acc"],
        "hard": "llm_acc",
        "soft": "llm_acc",
    }
    assert required_scopes == ["medical:benchmark_exact:dev"]
    assert captured["config"]["dataset"] == "medical"
    adapter_kwargs = captured["adapter_kwargs"]
    assert adapter_kwargs["dataset"].key == "medical"
    assert adapter_kwargs["evaluator"].profile.key == "medical"
    assert "AZURE_OPENAI_API_KEY" not in os.environ


def test_skillopt_prepare_cli_routes_non_hotpot_profile(tmp_path: Path) -> None:
    dataset_dir = _write_profile_dataset(tmp_path / "source", "novel")
    split_dir = tmp_path / "splits"

    result = runner.invoke(
        app,
        [
            "skillopt-prepare",
            "--dataset-dir",
            str(dataset_dir),
            "--dataset",
            "novel",
            "--split-dir",
            str(split_dir),
            "--allow-subset",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "2.0"
    assert payload["dataset"]["subset"] == "novel"
    assert payload["dataset"]["reference_validation"] == "schema_only"
    assert payload["selection"]["per_split_by_task_type"] == (
        EXPECTED_QUOTAS["novel"]
    )
