"""Run a benchmark SkillOpt workflow with local Ollama Qwen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ollama import Client

from agentic_rag.agent.config import AgentConfig, OllamaPolicyConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.evaluation import (
    EpisodeEvaluator,
    JudgeMessage,
    JudgeResponse,
    JudgeResponseError,
    JudgeUsage,
)
from agentic_rag.skillopt.adapter import AgenticRAGSkillOptAdapter
from agentic_rag.skillopt.data import (
    HOTPOTQA_BENCHMARK_SCOPE_ID,
    validate_hotpotqa_smoke_lineage,
)
from agentic_rag.skillopt.benchmark import (
    split_manifest_profile,
    validate_benchmark_lineage,
)
from agentic_rag.skillopt.trainer import load_skillopt_config, run_skillopt_training
from agentic_rag.substrate.storage import Substrate


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--skillopt-config", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


class LocalQwenJudge:
    """Structured semantic judge using the same local Ollama Qwen runtime."""

    def __init__(self, config: OllamaPolicyConfig) -> None:
        self.model = config.model
        self.host = config.host
        self.think = config.think
        self.num_ctx = config.num_ctx
        self.keep_alive = config.keep_alive
        self.client = Client(host=config.host, timeout=config.timeout_seconds)

    def judge(self, messages: Sequence[JudgeMessage]) -> JudgeResponse:
        schema = {
            "type": "object",
            "properties": {"correct": {"type": "boolean"}},
            "required": ["correct"],
            "additionalProperties": False,
        }
        response = self.client.chat(
            model=self.model,
            messages=[message.as_provider_input() for message in messages],
            stream=False,
            format=schema,
            think=self.think,
            options={"temperature": 0, "num_ctx": self.num_ctx},
            keep_alive=self.keep_alive,
        )
        content = response.message.content
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise JudgeResponseError("local Qwen judge returned invalid JSON") from exc
        correct = payload.get("correct")
        if type(correct) is not bool:
            raise JudgeResponseError("local Qwen judge omitted boolean correct")
        input_tokens = max(int(response.prompt_eval_count or 0), 0)
        output_tokens = max(int(response.eval_count or 0), 0)
        return JudgeResponse(
            correct=correct,
            raw_output=payload,
            model=self.model,
            usage=JudgeUsage(
                calls=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            request_options={
                "provider": "ollama",
                "host": self.host,
                "think": self.think,
                "temperature": 0,
                "num_ctx": self.num_ctx,
            },
        )


def main() -> None:
    args = arguments()
    substrate = Substrate.open(args.substrate)
    profile = split_manifest_profile(args.split_dir)
    split_manifest = json.loads(
        (args.split_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    split_metadata = split_manifest.get("splits")
    if not isinstance(split_metadata, dict):
        raise ValueError("split manifest has no splits mapping")

    def split_count(name: str) -> int:
        metadata = split_metadata.get(name)
        value = metadata.get("count") if isinstance(metadata, dict) else None
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"split manifest {name} count must be positive")
        return value
    if split_manifest.get("schema_version") in {"1.0", "1.1"}:
        if profile.key != "hotpotqa":
            raise ValueError("schema 1.x splits support only HotpotQA")
        scope_id = HOTPOTQA_BENCHMARK_SCOPE_ID
        validate_hotpotqa_smoke_lineage(args.split_dir, substrate.manifest)
    else:
        lineage = validate_benchmark_lineage(
            args.split_dir,
            substrate.manifest,
            dataset=profile,
        )
        scope_id = str(lineage["scope_id"])
    substrate.require_scope(scope_id)

    agent_config = AgentConfig.from_yaml(args.agent_config)
    if not isinstance(agent_config.policy, OllamaPolicyConfig):
        raise ValueError("local Qwen SkillOpt requires an Ollama Policy config")
    if not agent_config.policy.think:
        raise ValueError("local Qwen SkillOpt requires thinking mode")
    initial_skill = SkillDocument.load(args.skill)

    config = load_skillopt_config(args.skillopt_config)
    config.update(
        {
            "out_root": str(args.output.resolve()),
            "split_dir": str(args.split_dir.resolve()),
            "skill_init": str(args.skill.resolve()),
            "dataset": profile.key,
            "train_size": split_count("train"),
            "sel_env_num": split_count("validation"),
            "test_env_num": split_count("test"),
        }
    )

    def harness_factory(*, skill_content: str, output_root: Path) -> AgentHarness:
        return AgentHarness.from_skill_content(
            substrate_path=args.substrate,
            config=agent_config,
            skill_content=skill_content,
            skill_source_path="skillopt:local-qwen-candidate",
            output_root=output_root,
        )

    evaluator = EpisodeEvaluator(LocalQwenJudge(agent_config.policy), profile=profile)
    adapter = AgenticRAGSkillOptAdapter(
        split_dir=args.split_dir,
        harness_factory=harness_factory,
        evaluator=evaluator,
        dataset=profile,
        workers=int(config.get("workers", 1)),
        analyst_workers=int(config["analyst_workers"]),
        failure_only=bool(config.get("failure_only", False)),
        minibatch_size=int(config["minibatch_size"]),
        edit_budget=int(config["edit_budget"]),
        seed=int(config["seed"]),
        resume=True,
        fixed_answer_contract=initial_skill.fixed_answer_contract,
    )
    summary = run_skillopt_training(config, adapter)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
