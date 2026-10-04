"""Render all policy turns from one saved episode without running the Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import EpisodeResult
from agentic_rag.agent.policy import ScriptedPolicy
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.substrate.storage import Substrate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episode = EpisodeResult.model_validate_json(args.episode.read_text(encoding="utf-8"))
    config = AgentConfig.from_yaml(args.config)
    substrate = Substrate.open(args.substrate)
    harness = AgentHarness(
        substrate=substrate,
        config=config,
        skill=SkillDocument.load(args.skill),
        policy=ScriptedPolicy([]),  # context rendering never calls the target Policy
        output_root=args.output.parent / "unused",
    )

    rows = []
    for position, step in enumerate(episode.trajectory):
        built = harness.context_builder.build(
            episode.query,
            harness.skill,
            step.state_before,
            episode.trajectory[:position],
            scope_id=episode.scope_id,
        )
        rows.append({
            "step": step.step,
            "policy_attempt": step.policy_attempt,
            "messages": [message.model_dump(mode="json") for message in built.messages],
            "rendered_context": built.rendered_context,
            "reader_usage": built.reader_usage.model_dump(mode="json"),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"episode_id": episode.episode_id, "mode": config.observation_mode, "turns": rows},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
