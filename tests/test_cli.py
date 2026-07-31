from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
from agentic_rag.retrieval import Retriever
from agentic_rag.storage import Substrate
from conftest import FakeEmbeddingBackend

runner = CliRunner()


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
