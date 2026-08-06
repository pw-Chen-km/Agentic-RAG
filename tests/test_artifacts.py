from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.agent.artifacts import ArtifactWriter
from agentic_rag.agent.models import EpisodeResult, TerminationReason


def test_canonical_artifact_set_has_no_answer_stage(tmp_path: Path) -> None:
    result = EpisodeResult(
        episode_id="episode-1",
        query="Question?",
        scope_id="q1",
        termination_reason=TerminationReason.BUDGET_EXHAUSTED,
    )
    destination = ArtifactWriter(tmp_path).write_episode(
        episode_id=result.episode_id,
        episode=result,
        trajectory=[],
        target_system_prompt="system",
        target_user_prompt="user",
        skill_content="skill",
        effective_config={"architecture": "semantic_memory_typed_refs_compact"},
    )
    files = {path.name for path in destination.iterdir()}
    assert files == {
        "conversation.json",
        "effective_config.json",
        "episode.json",
        "skill.md",
        "target_system_prompt.txt",
        "target_user_prompt.txt",
    }
    episode = json.loads((destination / "episode.json").read_text(encoding="utf-8"))
    assert "answer_usage" not in episode
