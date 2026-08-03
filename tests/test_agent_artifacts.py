from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, SecretStr

from agentic_rag.agent import artifacts
from agentic_rag.agent.artifacts import ArtifactWriter, REDACTED_SECRET
from agentic_rag.paths import portable_path_component


class ExpansionKind(str, Enum):
    ENTITY_TO_SENTENCE = "ENTITY_MENTIONED_IN_SENTENCE"
    SENTENCE_TO_ENTITY = "SENTENCE_MENTIONS_ENTITY"


class Assessment(BaseModel):
    status: str
    missing_information: list[str] = []


class EffectiveConfig(BaseModel):
    model: str
    cache_dir: Path
    enabled_expansions: set[ExpansionKind]
    max_retrieved_tokens: int
    openai_api_key: str
    nested: dict[str, Any]
    wrapped_secret: SecretStr


@dataclass(frozen=True)
class Episode:
    episode_id: str
    question: str
    trajectory: tuple[dict[str, Any], ...]
    provider_metadata: dict[str, Any]


def _sample_inputs() -> dict[str, Any]:
    trajectory = (
        {
            "step": 1,
            "assessment": Assessment(
                status="INSUFFICIENT",
                missing_information=["birthplace"],
            ),
            "action": {
                "type": "SEARCH",
                "method": "BM25",
                "target": "SENTENCE",
                "query": "Where was Marie Curie born?",
            },
            "observation": {
                "results": [
                    {
                        "sentence_id": "sentence:S12",
                        "text": "Marie Curie was born in Warsaw.",
                    }
                ]
            },
            "agent_visible_observation": {
                "status": "ok",
                "results": [
                    {
                        "sentence_id": "S1",
                        "text": "Marie Curie was born in Warsaw.",
                    }
                ],
            },
        },
        {
            "step_index": 2,
            "decision": {
                "assessment": {
                    "status": "UNCERTAIN",
                    "missing_information": ["country"],
                },
                "action": {
                    "type": "EXPAND",
                    "kind": "SENTENCE_PART_OF_CHUNK",
                    "source_id": "S1",
                },
            },
            "resolved_decision": {
                "assessment": {
                    "status": "UNCERTAIN",
                    "missing_information": ["country"],
                },
                "action": {
                    "type": "EXPAND",
                    "kind": "SENTENCE_PART_OF_CHUNK",
                    "source_id": "sentence:S12",
                },
            },
            "agent_visible_observation": {
                "status": "invalid_action",
                "action": {
                    "type": "EXPAND",
                    "kind": "SENTENCE_PART_OF_CHUNK",
                    "source_id": "S1",
                },
                "error_code": "unknown_expansion",
                "message": "This expansion does not exist.",
            },
            "outcome": {
                "validation_error": {
                    "code": "unknown_expansion",
                    "message": "This expansion does not exist.",
                }
            },
        },
    )
    return {
        "episode_id": "episode-0001",
        "episode": Episode(
            episode_id="episode-0001",
            question="居禮夫人在哪裡出生？",
            trajectory=trajectory,
            provider_metadata={"authorization": "Bearer do-not-persist"},
        ),
        "target_system_prompt": "Use exactly one action per step.",
        "target_user_prompt": "Question: Where was Marie Curie born?",
        "skill_content": "# Retrieval skill\n\nStart with sentence search.\n",
        "effective_config": EffectiveConfig(
            model="gpt-5.6-luna",
            cache_dir=Path("/tmp/agent-cache"),
            enabled_expansions={
                ExpansionKind.SENTENCE_TO_ENTITY,
                ExpansionKind.ENTITY_TO_SENTENCE,
            },
            max_retrieved_tokens=12_000,
            openai_api_key="sk-must-not-be-written",
            nested={
                "clientSecret": "also-secret",
                "ordinary_token_count": 7,
            },
            wrapped_secret=SecretStr("secret-wrapper-value"),
        ),
    }


def test_writes_complete_skillopt_compatible_run_directory(
    tmp_path: Path,
) -> None:
    inputs = _sample_inputs()
    run_dir = ArtifactWriter(tmp_path / "runs").write_episode(**inputs)

    assert run_dir == tmp_path / "runs" / "episode-0001"
    assert {path.name for path in run_dir.iterdir()} == {
        "episode.json",
        "conversation.json",
        "target_system_prompt.txt",
        "target_user_prompt.txt",
        "skill.md",
        "effective_config.json",
    }

    episode = json.loads((run_dir / "episode.json").read_text("utf-8"))
    assert episode["question"] == "居禮夫人在哪裡出生？"
    assert episode["provider_metadata"]["authorization"] == REDACTED_SECRET
    assert episode["trajectory"][1]["decision"]["action"]["source_id"] == (
        "S1"
    )
    assert episode["trajectory"][1]["resolved_decision"]["action"][
        "source_id"
    ] == "sentence:S12"

    conversation = json.loads(
        (run_dir / "conversation.json").read_text("utf-8")
    )
    assert conversation == [
        {
            "step": 1,
            "action": inputs["episode"].trajectory[0]["action"],
            "reasoning": {
                "status": "INSUFFICIENT",
                "missing_information": ["birthplace"],
            },
            "env_feedback": inputs["episode"].trajectory[0][
                "agent_visible_observation"
            ],
        },
        {
            "step": 2,
            "action": inputs["episode"].trajectory[1]["decision"]["action"],
            "reasoning": inputs["episode"].trajectory[1]["decision"][
                "assessment"
            ],
            "env_feedback": inputs["episode"].trajectory[1][
                "agent_visible_observation"
            ],
        },
    ]
    assert "sentence:S12" not in json.dumps(conversation)
    assert all(
        set(record) == {"step", "action", "reasoning", "env_feedback"}
        for record in conversation
    )

    config = json.loads(
        (run_dir / "effective_config.json").read_text("utf-8")
    )
    assert config["cache_dir"] == "/tmp/agent-cache"
    assert config["enabled_expansions"] == [
        "ENTITY_MENTIONED_IN_SENTENCE",
        "SENTENCE_MENTIONS_ENTITY",
    ]
    assert config["max_retrieved_tokens"] == 12_000
    assert config["openai_api_key"] == REDACTED_SECRET
    assert config["nested"]["clientSecret"] == REDACTED_SECRET
    assert config["nested"]["ordinary_token_count"] == 7
    assert config["wrapped_secret"] == REDACTED_SECRET

    assert (run_dir / "target_system_prompt.txt").read_text(
        "utf-8"
    ) == inputs["target_system_prompt"]
    assert (run_dir / "target_user_prompt.txt").read_text(
        "utf-8"
    ) == inputs["target_user_prompt"]
    assert (run_dir / "skill.md").read_text("utf-8") == inputs["skill_content"]

    raw_episode = (run_dir / "episode.json").read_bytes()
    assert "居禮夫人".encode() in raw_episode
    assert b"do-not-persist" not in raw_episode
    raw_config = (run_dir / "effective_config.json").read_bytes()
    assert b"sk-must-not-be-written" not in raw_config
    assert b"also-secret" not in raw_config


def test_write_is_deterministic_idempotent_and_immutable(
    tmp_path: Path,
) -> None:
    writer = ArtifactWriter(tmp_path / "runs")
    inputs = _sample_inputs()
    first = writer.write(**inputs)
    before = {
        path.name: path.read_bytes()
        for path in sorted(first.iterdir())
    }

    second = writer.write_episode(**inputs)
    assert second == first
    assert {
        path.name: path.read_bytes()
        for path in sorted(second.iterdir())
    } == before

    different = {**inputs, "skill_content": "# Different skill\n"}
    with pytest.raises(FileExistsError):
        writer.write_episode(**different)
    assert {
        path.name: path.read_bytes()
        for path in sorted(first.iterdir())
    } == before


def test_windows_unsafe_episode_id_uses_portable_directory(
    tmp_path: Path,
) -> None:
    logical_id = "hotpotqa:benchmark_exact:q:000001"
    inputs = {
        **_sample_inputs(),
        "episode_id": logical_id,
        "episode": Episode(
            episode_id=logical_id,
            question="Where was Marie Curie born?",
            trajectory=(),
            provider_metadata={},
        ),
    }

    run_dir = ArtifactWriter(tmp_path / "runs").write_episode(**inputs)

    assert run_dir.name == portable_path_component(logical_id)
    assert ":" not in run_dir.name
    assert len(run_dir.name) <= 120
    episode = json.loads((run_dir / "episode.json").read_text("utf-8"))
    assert episode["episode_id"] == logical_id


def test_directory_fsync_is_skipped_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifacts.sys, "platform", "win32")

    def unexpected_open(*_args, **_kwargs):
        raise AssertionError("Windows must not call os.open on a directory")

    monkeypatch.setattr(artifacts.os, "open", unexpected_open)
    artifacts._fsync_directory(tmp_path)


@pytest.mark.parametrize("logical_id", ["CON", "report.", "A:B", "資料:一"])
def test_portable_path_component_handles_windows_names(logical_id: str) -> None:
    component = portable_path_component(logical_id)

    assert component not in {"CON", "report."}
    assert all(character not in component for character in '<>:"/\\|?*')
    assert component == portable_path_component(logical_id)


def test_failed_write_never_publishes_partial_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = ArtifactWriter(tmp_path / "runs")
    inputs = _sample_inputs()
    real_write = artifacts._write_file
    writes = 0

    def fail_during_write(path: Path, content: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated disk failure")
        real_write(path, content)

    monkeypatch.setattr(artifacts, "_write_file", fail_during_write)
    with pytest.raises(OSError, match="simulated disk failure"):
        writer.write_episode(**inputs)

    runs_root = tmp_path / "runs"
    assert not (runs_root / "episode-0001").exists()
    assert list(runs_root.iterdir()) == []


@pytest.mark.parametrize(
    "episode_id",
    ["", " ", ".", "..", "../escape", "nested/run", r"nested\\run"],
)
def test_rejects_unsafe_episode_ids(
    tmp_path: Path,
    episode_id: str,
) -> None:
    inputs = {**_sample_inputs(), "episode_id": episode_id}
    with pytest.raises(ValueError):
        ArtifactWriter(tmp_path / "runs").write_episode(**inputs)
    assert not (tmp_path / "runs").exists()
