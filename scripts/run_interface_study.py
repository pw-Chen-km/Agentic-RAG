"""Run the paired C0--C4 HotpotQA interface pilot with resume safety."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.harness import _policy_from_config
from agentic_rag.agent.interface import get_interface_contract
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.evaluation.interface_study import aggregate, evaluate_episode
from agentic_rag.substrate.storage import EvaluationSidecars, Substrate


CONDITIONS = ("C0", "C1", "C2", "C3", "C4")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(args: argparse.Namespace, config: AgentConfig, conditions: tuple[str, ...]) -> dict[str, Any]:
    source_manifest = args.source_manifest
    repo_root = Path(__file__).resolve().parents[1]
    renderer_bytes = b"".join(
        (repo_root / item).read_bytes()
        for item in (
            Path("src/agentic_rag/agent/observation_projection.py"),
            Path("src/agentic_rag/agent/interface.py"),
        )
    )
    return {
        "manifest_version": "interface-study-run-v1",
        "dataset": args.dataset,
        "seed": args.seed,
        "question_limit": args.limit,
        "conditions": list(conditions),
        "question_file": args.questions.resolve().as_posix(),
        "question_sha256": sha256(args.questions),
        "source_manifest": source_manifest.resolve().as_posix(),
        "source_manifest_sha256": sha256(source_manifest),
        "substrate": args.substrate.resolve().as_posix(),
        "substrate_manifest_sha256": sha256(args.substrate / "manifest.json"),
        "config": args.config.resolve().as_posix(),
        "config_sha256": sha256(args.config),
        "skill": args.skill.resolve().as_posix(),
        "skill_sha256": sha256(args.skill),
        "judge_config_sha256": sha256(args.judge_config) if args.judge_config else None,
        "renderer_version": "observation-projector-v1",
        "renderer_sha256": hashlib.sha256(renderer_bytes).hexdigest(),
        "budget": {
            "normal_policy_decisions": 15,
            "finalize_calls": 1,
            "retrieved_token_guard": 12000,
            "request_timeout_seconds": 600,
            "episode_timeout_seconds": 3600,
        },
        "contracts": {name: get_interface_contract(name).compile() for name in conditions},
    }


def _load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["episode_id"])] = row
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    conditions = tuple(args.conditions)
    if not conditions:
        raise ValueError("at least one condition is required")
    for name in conditions:
        get_interface_contract(name)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(args, AgentConfig.from_yaml(args.config), conditions)
    manifest_path = args.output / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("resume refused: source/config/substrate/renderer manifest changed")
    else:
        write_json(manifest_path, manifest)

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("pilot questions must be a JSON array")
    sidecars = EvaluationSidecars.open(args.substrate)
    gold = [asdict(item) for item in sidecars.gold_support]
    base_config = AgentConfig.from_yaml(args.config)
    if (
        base_config.max_steps != 15
        or base_config.max_policy_attempts != 15
        or base_config.max_retrieved_tokens != 12000
    ):
        raise ValueError("pilot config must use 15 decisions and a 12000-token retrieval guard")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    substrate = Substrate.open(args.substrate)
    skill = SkillDocument.load(args.skill)
    harnesses: dict[str, AgentHarness] = {}
    for condition in conditions:
        condition_config = base_config.model_copy(update={"interface": condition})
        harnesses[condition] = AgentHarness(
            substrate=substrate,
            config=condition_config,
            skill=skill,
            policy=_policy_from_config(condition_config),
            output_root=args.output / "episodes",
        )
    completed = _load_progress(args.output / "progress.jsonl")
    if args.limit is not None:
        questions = questions[: args.limit]
    schedule = [(str(row.get("_id") or row.get("id")), condition, row) for row in questions for condition in conditions]
    random.Random(args.seed).shuffle(schedule)
    progress_path = args.output / "progress.jsonl"
    summaries: list[dict[str, Any]] = []
    scope_id = sidecars.benchmark_questions[0].scope_id if sidecars.benchmark_questions else "hotpotqa:benchmark_exact:dev"
    semantic_evaluator = None
    if args.semantic_eval:
        from agentic_rag.evaluation.graphrag_bench import GraphRAGSemanticEvaluator, OllamaSemanticJudge
        judge_model, judge_host, embedding_model = args.judge_model, args.judge_host, args.embedding_model
        if args.judge_config:
            import yaml
            raw = yaml.safe_load(args.judge_config.read_text(encoding="utf-8")) or {}
            judge_model = raw.get("model", judge_model); judge_host = raw.get("host", judge_host); embedding_model = raw.get("embedding_model", embedding_model)
        judge = OllamaSemanticJudge(model=judge_model, host=judge_host, embedding_model=embedding_model)
        semantic_evaluator = GraphRAGSemanticEvaluator(judge)
    for question_id, condition, row in schedule:
        episode_id = f"{condition}--{question_id}"
        if episode_id in completed:
            summaries.append(completed[episode_id])
            continue
        harness = harnesses[condition]
        started = time.perf_counter()
        try:
            result = harness.run(row["question"], scope_id, episode_id=episode_id)
            episode_payload = json.loads((harness.artifact_writer.path_for_episode(episode_id) / "episode.json").read_text(encoding="utf-8"))
            scored = evaluate_episode(episode_payload, question=row, gold_support=gold)
            if semantic_evaluator is not None:
                visible_spans = [span for step in episode_payload.get("trajectory", []) for span in step.get("visible_source_spans", [])]
                semantic = semantic_evaluator.evaluate(
                    question=row,
                    answer=episode_payload.get("answer"),
                    contexts=[str(span.get("text") or span.get("source_text") or "") for span in visible_spans],
                    evidence=row.get("evidence") if isinstance(row.get("evidence"), list) else ([row.get("evidence")] if row.get("evidence") else []),
                )
                semantic.update({"episode_id": episode_id, "question_id": question_id, "condition": condition})
                write_json(args.output / "semantic_evaluations" / f"{episode_id}.json", semantic)
            scored.update({"episode_id": episode_id, "condition": condition, "wall_time_seconds": time.perf_counter() - started})
        except Exception as exc:
            scored = {
                "episode_id": episode_id,
                "condition": condition,
                "question_id": question_id,
                "not_evaluable": True,
                "termination_reason": "runtime_error",
                "error_code": type(exc).__name__,
                "error_message": str(exc),
                "wall_time_seconds": time.perf_counter() - started,
            }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(scored, ensure_ascii=False, sort_keys=True) + "\n")
        summaries.append(scored)
    by_condition = {
        condition: aggregate([row for row in summaries if row.get("condition") == condition])
        for condition in conditions
    }
    result = {"run_manifest": manifest, "conditions": by_condition, "results": summaries}
    write_json(args.output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/interface_study_qwen.yaml"))
    parser.add_argument("--skill", type=Path, default=Path("skills/interface_study.md"))
    parser.add_argument("--output", type=Path, default=Path("runs/interface-study-v1"))
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=("C0", "C1", "C2", "C3", "C4", "A1"))
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--semantic-eval", action="store_true")
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--judge-host", default="http://localhost:11434")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--judge-config", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="limit questions for a smoke run")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
