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
from agentic_rag.skillopt.trajectory import TRAJECTORY_REPRESENTATIONS
from agentic_rag.skillopt.provenance import (
    PROVENANCE_SPLIT_SCHEMA_VERSION,
    validate_hotpotqa_provenance_lineage,
)
from agentic_rag.substrate.storage import EvaluationSidecars, Substrate


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--skillopt-config", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectory-representation",
        choices=TRAJECTORY_REPRESENTATIONS,
        required=True,
        help="Reflect-only trajectory view; target rollouts remain unchanged.",
    )
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
    evaluation_sidecars = EvaluationSidecars.open(args.substrate)
    sentence_provenance = _sentence_provenance_map(evaluation_sidecars)
    if (
        args.trajectory_representation
        in {"organized_support_labels", "progress_abstracted"}
        and not sentence_provenance
    ):
        raise ValueError(
            f"{args.trajectory_representation} requires deterministic source "
            "sentence provenance in the evaluation sidecars"
        )
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
    split_schema = split_manifest.get("schema_version")
    if split_schema == PROVENANCE_SPLIT_SCHEMA_VERSION:
        if profile.key != "hotpotqa":
            raise ValueError("provenance reflection splits support only HotpotQA")
        scope_id = str(split_manifest["dataset"]["scope_id"])
        validate_hotpotqa_provenance_lineage(args.split_dir, args.substrate)
    elif split_schema in {"1.0", "1.1"}:
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
    if agent_config.policy.think:
        raise ValueError(
            "trajectory ablation requires Target Agent thinking to be disabled"
        )
    initial_skill = SkillDocument.load(args.skill)

    config = load_skillopt_config(args.skillopt_config)
    optimizer_model = str(config.get("optimizer_model") or "")
    target_model = str(config.get("target_model") or "")
    if optimizer_model != agent_config.policy.model or target_model != agent_config.policy.model:
        raise ValueError(
            "Target Agent and SkillOpt optimizer metadata must use the same model"
        )
    optimizer_thinking = config.get(
        "optimizer_qwen_chat_enable_thinking",
        config.get("qwen_chat_enable_thinking"),
    )
    if optimizer_thinking is not True:
        raise ValueError("Qwen optimizer thinking must be explicitly enabled")
    config.update(
        {
            "out_root": str(args.output.resolve()),
            "split_dir": str(args.split_dir.resolve()),
            "skill_init": str(args.skill.resolve()),
            "dataset": profile.key,
            "train_size": split_count("train"),
            "sel_env_num": split_count("validation"),
            "test_env_num": split_count("test"),
            "trajectory_representation": args.trajectory_representation,
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
        trajectory_representation=args.trajectory_representation,
        sentence_provenance=sentence_provenance,
    )
    summary = run_skillopt_training(config, adapter)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _sentence_provenance_map(
    sidecars: EvaluationSidecars,
) -> dict[str, dict[str, object]]:
    """Index evaluation-only source coordinates by stable Sentence ID."""

    result: dict[str, dict[str, object]] = {}
    for item in sidecars.source_sentence_provenance:
        if item.sentence_id is None:
            continue
        if item.sentence_id in result:
            raise ValueError(
                "source sentence provenance contains a duplicate stable ID: "
                f"{item.sentence_id}"
            )
        result[item.sentence_id] = {
            "original_title": item.original_title,
            "original_sentence_id": item.original_sentence_id,
            "text": item.original_sentence_text,
        }
    return result


if __name__ == "__main__":
    main()
