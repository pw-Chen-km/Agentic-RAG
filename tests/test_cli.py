from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from typer.testing import CliRunner

from agentic_rag.agent.answer import FakeAnswerGenerator
from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    AssessmentStatus,
    EvidenceAssessment,
    FinishAction,
    PolicyDecision,
    SearchAction,
    SentenceRef,
)
from agentic_rag.agent.policy import ScriptedPolicy
from agentic_rag.agent.skill import SkillDocument
from agentic_rag import cli
from agentic_rag.cli import app
from agentic_rag.models import SourceArtifact
from agentic_rag.retrieval import Retriever
from agentic_rag.storage import Substrate
from agentic_rag.skillopt.data import (
    ARAG_HOTPOTQA_CHUNKS_SHA256,
    ARAG_HOTPOTQA_QUESTIONS_SHA256,
    HOTPOTQA_BENCHMARK_SCOPE_ID,
    prepare_hotpotqa_smoke_splits,
)
from conftest import FakeEmbeddingBackend

runner = CliRunner()


def test_skillopt_environment_restores_derived_credentials_and_module_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "temporary-target-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://previous.invalid")
    monkeypatch.setenv(
        "OPTIMIZER_AZURE_OPENAI_API_KEY", "previous-optimizer-key"
    )
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TARGET_AZURE_OPENAI_API_KEY", raising=False)

    provider = ModuleType("skillopt.model.azure_openai")
    provider.API_KEY = "previous-shared-global"
    provider.OPTIMIZER_API_KEY = "previous-optimizer-global"
    provider.TARGET_API_KEY = "previous-target-global"
    provider._optimizer_client = object()
    provider._target_client = object()
    provider._AZ_CLI_TOKEN_CACHE = {"previous": {"token": "value"}}
    previous_state = {
        name: (
            dict(value) if isinstance(value, dict) else value
        )
        for name, value in vars(provider).items()
        if name.startswith("_") or name.endswith("API_KEY")
    }
    monkeypatch.setitem(
        sys.modules, "skillopt.model.azure_openai", provider
    )

    with cli._skillopt_openai_environment(
        endpoint="https://api.openai.com/v1",
        auth_mode="openai_compatible",
    ):
        assert os.environ["AZURE_OPENAI_API_KEY"] == "temporary-target-key"
        os.environ["OPTIMIZER_AZURE_OPENAI_API_KEY"] = "derived-key"
        os.environ["TARGET_AZURE_OPENAI_API_KEY"] = "derived-key"
        provider.API_KEY = "derived-key"
        provider.OPTIMIZER_API_KEY = "derived-key"
        provider.TARGET_API_KEY = "derived-key"
        provider._optimizer_client = object()
        provider._target_client = object()
        provider._AZ_CLI_TOKEN_CACHE["new"] = {"token": "derived-key"}

    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://previous.invalid"
    assert "AZURE_OPENAI_API_KEY" not in os.environ
    assert (
        os.environ["OPTIMIZER_AZURE_OPENAI_API_KEY"]
        == "previous-optimizer-key"
    )
    assert "TARGET_AZURE_OPENAI_API_KEY" not in os.environ
    for name, value in previous_state.items():
        assert getattr(provider, name) == value


def test_skillopt_environment_scrubs_provider_module_loaded_inside_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "temporary-target-key")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delitem(
        sys.modules, "skillopt.model.azure_openai", raising=False
    )

    provider = ModuleType("skillopt.model.azure_openai")
    provider.API_KEY = "temporary-target-key"
    provider.OPTIMIZER_API_KEY = "temporary-target-key"
    provider.TARGET_API_KEY = "temporary-target-key"
    provider._optimizer_client = object()
    provider._target_client = object()
    provider._AZ_CLI_TOKEN_CACHE = {"request": {"token": "secret"}}
    with cli._skillopt_openai_environment(
        endpoint="https://api.openai.com/v1",
        auth_mode="openai_compatible",
    ):
        monkeypatch.setitem(
            sys.modules, "skillopt.model.azure_openai", provider
        )
        os.environ["OPTIMIZER_AZURE_OPENAI_API_KEY"] = (
            "temporary-target-key"
        )
        os.environ["TARGET_AZURE_OPENAI_API_KEY"] = "temporary-target-key"

    assert "AZURE_OPENAI_API_KEY" not in os.environ
    assert "OPTIMIZER_AZURE_OPENAI_API_KEY" not in os.environ
    assert "TARGET_AZURE_OPENAI_API_KEY" not in os.environ
    assert provider.API_KEY == ""
    assert provider.OPTIMIZER_API_KEY == ""
    assert provider.TARGET_API_KEY == ""
    assert provider._optimizer_client is None
    assert provider._target_client is None
    assert provider._AZ_CLI_TOKEN_CACHE == {}


def test_validate_and_read_cli(built_substrate: Path) -> None:
    validation = runner.invoke(app, ["validate", str(built_substrate)])
    assert validation.exit_code == 0, validation.output
    assert json.loads(validation.output)["valid"] is True

    retriever = Retriever(
        built_substrate, embedding_backend=FakeEmbeddingBackend()
    )
    chunk = retriever.search(
        "Where was Marie Curie born?", "BM25", "CHUNK", "q1", top_k=1
    )[0]
    read = runner.invoke(
        app, ["read", str(built_substrate), chunk.chunk_id]
    )
    assert read.exit_code == 0, read.output
    assert json.loads(read.output)["chunk_id"] == chunk.chunk_id


def test_lexical_search_and_bridge_cli(built_substrate: Path) -> None:
    search = runner.invoke(
        app,
        [
            "search",
            str(built_substrate),
            "Warsaw",
            "--scope-id",
            "q1",
            "--method",
            "lexical",
            "--target",
            "entity",
        ],
    )
    assert search.exit_code == 0, search.output
    entity_id = json.loads(search.output)[0]["entity_id"]

    bridge = runner.invoke(
        app,
        [
            "bridge",
            str(built_substrate),
            entity_id,
            "--scope-id",
            "q1",
        ],
    )
    assert bridge.exit_code == 0, bridge.output
    assert any(
        item["target_canonical_name"] == "Poland"
        for item in json.loads(bridge.output)
    )


def test_cli_domain_error_is_nonzero(built_substrate: Path) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            str(built_substrate),
            "Warsaw",
            "--scope-id",
            "q1",
            "--method",
            "bm25",
            "--target",
            "entity",
        ],
    )
    assert result.exit_code == 2
    assert json.loads(result.stderr)["error"] == "invalid_retrieval_pair"


def test_run_cli_executes_scripted_episode_and_writes_artifacts(
    built_substrate: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "Where was Marie Curie born?"
    substrate = Substrate.open(built_substrate)
    sentence_id = next(
        sentence.sentence_id
        for sentence in substrate.sentences
        if sentence.text == "Marie Curie was born in Warsaw."
    )
    policy = ScriptedPolicy(
        [
            PolicyDecision(
                assessment=EvidenceAssessment(
                    status=AssessmentStatus.INSUFFICIENT,
                    missing_information=["Marie Curie's birthplace"],
                ),
                action=SearchAction(
                    query=question,
                    method="BM25",
                    target="SENTENCE",
                ),
            ),
            PolicyDecision(
                assessment=EvidenceAssessment(
                    status=AssessmentStatus.SUFFICIENT,
                    supported_facts=[
                        "Marie Curie was born in Warsaw."
                    ],
                    selected_evidence_refs=[
                        SentenceRef(id=sentence_id)
                    ],
                ),
                action=FinishAction(
                    evidence_refs=[SentenceRef(id=sentence_id)]
                ),
            ),
        ]
    )
    output_root = tmp_path / "runs"
    harness = AgentHarness(
        substrate=substrate,
        config=AgentConfig(),
        skill=SkillDocument.from_text(
            "# Initial retrieval strategy\n\nSearch sentences first.\n"
        ),
        policy=policy,
        answer_generator=FakeAnswerGenerator("Warsaw"),
        output_root=output_root,
        embedding_backend=FakeEmbeddingBackend(),
    )

    def fake_from_config(**kwargs: object) -> AgentHarness:
        assert kwargs["substrate_path"] == built_substrate
        assert kwargs["output_root"] == output_root
        return harness

    monkeypatch.setattr(
        cli,
        "AgentHarness",
        SimpleNamespace(from_config=fake_from_config),
    )
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("agent: {}\n", encoding="utf-8")
    skill_path = tmp_path / "initial.md"
    skill_path.write_text("# Initial retrieval strategy\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "run",
            str(built_substrate),
            question,
            "--scope-id",
            "q1",
            "--skill-file",
            str(skill_path),
            "--config",
            str(config_path),
            "--output",
            str(output_root),
            "--episode-id",
            "cli-scripted-success",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["answer"] == "Warsaw"
    assert payload["termination_reason"] == "finish"
    assert payload["query"] == question
    run_dir = output_root / "cli-scripted-success"
    assert {path.name for path in run_dir.iterdir()} == {
        "episode.json",
        "conversation.json",
        "target_system_prompt.txt",
        "target_user_prompt.txt",
        "skill.md",
        "effective_config.json",
    }


def test_run_cli_without_openai_key_returns_typed_nonzero_error(
    built_substrate: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("agent: {}\n", encoding="utf-8")
    skill_path = tmp_path / "initial.md"
    skill_path.write_text("# Initial retrieval strategy\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "run",
            str(built_substrate),
            "Where was Marie Curie born?",
            "--scope-id",
            "q1",
            "--skill-file",
            str(skill_path),
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "runs"),
            "--episode-id",
            "cli-missing-api-key",
        ],
    )

    assert result.exit_code == 2
    error = json.loads(result.stderr)
    assert error["error"] == "policy_error"
    assert "OPENAI_API_KEY" in error["message"]


def test_skillopt_prepare_cli_only_materializes_split_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "arag-dataset"
    dataset_dir.mkdir()
    split_dir = tmp_path / "smoke-splits"
    calls: list[tuple[Path, Path]] = []

    def fake_prepare(
        dataset_dir: Path, split_dir: Path
    ) -> dict[str, object]:
        calls.append((dataset_dir, split_dir))
        return {
            "purpose": "workflow_smoke",
            "dataset": {"revision": "pinned-revision"},
        }

    monkeypatch.setattr(
        cli, "prepare_hotpotqa_smoke_splits", fake_prepare
    )
    result = runner.invoke(
        app,
        [
            "skillopt-prepare",
            "--dataset-dir",
            str(dataset_dir),
            "--split-dir",
            str(split_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(dataset_dir, split_dir)]
    assert json.loads(result.output)["purpose"] == "workflow_smoke"


def test_skillopt_train_cli_wires_native_trainer_without_gold_target_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    substrate_path = tmp_path / "substrate"
    substrate_path.mkdir()
    dataset_dir = tmp_path / "arag-hotpotqa"
    dataset_dir.mkdir()
    questions = [
        {
            "id": f"benchmark-{index:04d}",
            "question": f"Question benchmark-{index:04d}?",
            "answer": f"SECRET-GOLD-benchmark-{index:04d}",
            "question_type": "bridge" if index < 811 else "comparison",
            "source": "hotpotqa",
            "evidence": [],
        }
        for index in range(1000)
    ]
    (dataset_dir / "questions.json").write_text(
        json.dumps(questions), encoding="utf-8"
    )
    (dataset_dir / "chunks.json").write_text(
        json.dumps([f"{index}:Chunk {index}" for index in range(1311)]),
        encoding="utf-8",
    )
    split_dir = tmp_path / "splits"
    split_manifest = prepare_hotpotqa_smoke_splits(
        dataset_dir,
        split_dir,
        expected_question_count=None,
        expected_chunk_count=None,
    )
    split_manifest["dataset"]["source_files"]["chunks"]["sha256"] = (
        ARAG_HOTPOTQA_CHUNKS_SHA256
    )
    split_manifest["dataset"]["source_files"]["questions"]["sha256"] = (
        ARAG_HOTPOTQA_QUESTIONS_SHA256
    )
    (split_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    agent_config_path = tmp_path / "agent.yaml"
    agent_config_path.write_text("agent: {}\n", encoding="utf-8")
    skillopt_config_path = tmp_path / "skillopt.yaml"
    skillopt_config_path.write_text("train: {}\n", encoding="utf-8")
    skill_file = tmp_path / "initial.md"
    skill_file.write_text("# Initial skill\n", encoding="utf-8")
    output = tmp_path / "run"

    fake_substrate = SimpleNamespace(
        manifest=SimpleNamespace(
            source_format="hotpotqa_benchmark_exact",
            record_counts={
                "chunks": 1311,
                "benchmark_questions": 1000,
            },
            source_artifacts=[
                SourceArtifact(
                    role=role,
                    path=split_manifest["dataset"]["source_files"][role][
                        "path"
                    ],
                    sha256=split_manifest["dataset"]["source_files"][role][
                        "sha256"
                    ],
                    size_bytes=split_manifest["dataset"]["source_files"][role][
                        "size_bytes"
                    ],
                )
                for role in ("chunks", "questions")
            ],
        ),
        require_scope=lambda scope: None,
    )
    monkeypatch.setattr(
        cli,
        "Substrate",
        SimpleNamespace(open=lambda path: fake_substrate),
    )
    agent_config = AgentConfig.model_validate(
        {
            "policy": {"provider": "openai", "model": "gpt-5.6-luna"},
            "answer": {"provider": "openai", "model": "gpt-5.6-luna"},
        }
    )
    monkeypatch.setattr(
        cli.AgentConfig,
        "from_yaml",
        lambda path: agent_config,
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
    captured: dict[str, object] = {}

    class FakeAdapter:
        def __init__(self, **kwargs: object) -> None:
            captured["adapter_kwargs"] = kwargs

    monkeypatch.setattr(cli, "AgenticRAGSkillOptAdapter", FakeAdapter)
    monkeypatch.setattr(
        cli.AgentHarness,
        "from_skill_content",
        lambda **kwargs: captured.setdefault("harness_kwargs", kwargs),
    )

    def fake_training(config: dict, adapter: object) -> dict[str, object]:
        captured["config"] = dict(config)
        captured["adapter"] = adapter
        factory = captured["adapter_kwargs"]["harness_factory"]
        factory(
            skill_content="# Candidate\n",
            output_root=output / "predictions",
        )
        assert os.environ["AZURE_OPENAI_API_KEY"] == "test-openai-key"
        return {"test_hard": 1.0}

    monkeypatch.setattr(cli, "run_skillopt_training", fake_training)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
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
    assert json.loads(result.output)["run_kind"] == "workflow_smoke"
    assert captured["config"]["out_root"] == str(output.resolve())
    harness_kwargs = captured["harness_kwargs"]
    assert set(harness_kwargs) == {
        "substrate_path",
        "config",
        "skill_content",
        "skill_source_path",
        "output_root",
    }
    serialized_target_factory_input = json.dumps(
        harness_kwargs, default=str
    )
    assert "SECRET-GOLD" not in serialized_target_factory_input
    assert "test-openai-key" not in serialized_target_factory_input
    assert "AZURE_OPENAI_API_KEY" not in os.environ
