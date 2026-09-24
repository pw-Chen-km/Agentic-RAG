"""Run the paired Agentic RAG interface study with resume safety."""

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
from agentic_rag.evaluation.gold_sidecars import sha256 as file_sha256, validate_gold_sidecars
from agentic_rag.evaluation.question_identity import IDENTITY_VERSION, prepare_question_rows
from agentic_rag.substrate.storage import EvaluationSidecars, Substrate


CONDITIONS = ("C0", "C1", "C2", "C3", "C5", "C4", "A1")


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
            Path("src/agentic_rag/agent/interface_action_catalog.py"),
            Path("src/agentic_rag/agent/context.py"),
            Path("src/agentic_rag/agent/context_rendering.py"),
            Path("src/agentic_rag/agent/entity_visibility.py"),
            Path("src/agentic_rag/agent/router.py"),
            Path("src/agentic_rag/agent/state_management.py"),
        )
    )
    provider_protocol_bytes = b"".join(
        (repo_root / item).read_bytes()
        for item in (
            Path("src/agentic_rag/agent/tool_calling.py"),
            Path("src/agentic_rag/agent/action_schema.py"),
            Path("src/agentic_rag/agent/models.py"),
            Path("src/agentic_rag/agent/validator.py"),
            Path("src/agentic_rag/agent/expansion.py"),
            Path("src/agentic_rag/agent/providers/openai_compatible.py"),
            Path("src/agentic_rag/agent/providers/ollama.py"),
        )
    )
    action_catalog_sha256 = sha256(repo_root / "src/agentic_rag/agent/interface_action_catalog.py")
    entity_schema_sha256 = hashlib.sha256(
        (repo_root / "src/agentic_rag/agent/tool_calling.py").read_bytes()
        + (repo_root / "src/agentic_rag/agent/action_schema.py").read_bytes()
    ).hexdigest()
    substrate_manifest_data = json.loads((args.substrate / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "manifest_version": "interface-study-run-v2",
        "dataset": args.dataset,
        "seed": args.seed,
        "question_limit": args.limit,
        "question_indices": getattr(args, "question_indices", None),
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
        "evaluator_code_sha256": sha256(repo_root / "src/agentic_rag/evaluation/graphrag_bench.py"),
        "semantic_metric_version": "graphrag-benchmark-logic-v1",
        "prompt_template_digest": hashlib.sha256(
            (repo_root / "src/agentic_rag/evaluation/graphrag_bench.py").read_bytes()
        ).hexdigest(),
        "embedding_model_identity": substrate_manifest_data.get("embedding_model"),
        "model_identity": config.policy.model,
        "provider_name": config.policy.provider,
        "decoding": {
            "temperature": config.policy.temperature,
            "seed": getattr(config.policy, "seed", None),
            "think": getattr(config.policy, "think", None),
            "num_ctx": config.policy.num_ctx,
            "max_output_tokens": config.policy.max_output_tokens,
        },
        "embedding_validation_override": bool(
            getattr(args, "allow_substrate_embedding_mismatch", False)
        ),
        "renderer_version": "sectioned-context-v6.2-entity-navigation-filter",
        "renderer_sha256": hashlib.sha256(renderer_bytes).hexdigest(),
        "protocol_type": "native_tool_calling",
        "native_tool_calling": True,
        "provider_protocol": "native-tool-calling-v1.1-original-question-hop",
        "require_evidence_assessment": config.require_evidence_assessment,
        "target_prompt_digest": hashlib.sha256(
            (repo_root / "src/agentic_rag/agent/interface.py").read_bytes()
            + (repo_root / "src/agentic_rag/agent/interface_action_catalog.py").read_bytes()
            + args.skill.read_bytes()
        ).hexdigest(),
        "provider_protocol_sha256": hashlib.sha256(provider_protocol_bytes).hexdigest(),
        "tool_schema_sha256": {
            name: hashlib.sha256(
                json.dumps(get_interface_contract(name).compile(), ensure_ascii=False,
                           sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for name in conditions
        },
        "action_catalog_sha256": action_catalog_sha256,
        "entity_schema_sha256": entity_schema_sha256,
        "entity_schema_version": "entity-navigation-filter-v1",
        "entity_filter_policy_sha256": sha256(
            repo_root / "src/agentic_rag/agent/entity_visibility.py"
        ),
        "budget": {
            "normal_policy_decisions": 15,
            "finalize_calls": 1,
            "retrieved_token_guard": 12000,
            "request_timeout_seconds": 600,
            "episode_timeout_seconds": 3600,
        },
        "contracts": {name: get_interface_contract(name).compile() for name in conditions},
    }
    sidecar_manifest_path = args.substrate / "evaluation" / "gold_evidence_manifest.json"
    manifest["gold_sidecar_manifest_sha256"] = (
        sha256(sidecar_manifest_path) if sidecar_manifest_path.exists() else None
    )
    manifest["embedding_backend"] = substrate_manifest_data.get("embedding_backend")
    manifest["embedding_dimension"] = substrate_manifest_data.get("embedding_dimension")
    if args.dataset in {"novel", "medical"}:
        manifest["question_identity_version"] = IDENTITY_VERSION
        manifest["question_identity_sha256"] = sha256(
            repo_root / "src/agentic_rag/evaluation/question_identity.py"
        )
        manifest["runner_sha256"] = sha256(Path(__file__))
    return manifest


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
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("pilot questions must be a JSON array")
    sidecars = EvaluationSidecars.open(args.substrate)
    substrate_manifest = json.loads((args.substrate / "manifest.json").read_text(encoding="utf-8"))
    if getattr(args, "allow_substrate_embedding_mismatch", False):
        raise ValueError("embedding mismatch override is diagnostic-only and cannot be used for formal runs")
    embedding_model = substrate_manifest.get("embedding_model")
    embedding_name = embedding_model.get("name") if isinstance(embedding_model, dict) else embedding_model
    embedding_backend = str(substrate_manifest.get("embedding_backend") or "")
    embedding_dimension = int(substrate_manifest.get("embedding_dimension") or 0)
    questions = prepare_question_rows(questions, sidecars.benchmark_questions, args.dataset)
    if getattr(args, "question_indices", None) is not None:
        if args.limit is not None:
            raise ValueError("--question-indices and --limit cannot be combined")
        if len(set(args.question_indices)) != len(args.question_indices):
            raise ValueError("--question-indices must be unique")
        if any(index < 0 or index >= len(questions) for index in args.question_indices):
            raise ValueError("--question-indices contains an out-of-range source row")
        questions = [questions[index] for index in args.question_indices]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(args, AgentConfig.from_yaml(args.config), conditions)
    sidecar_manifest_path = args.substrate / "evaluation" / "gold_evidence_manifest.json"
    sidecar_report = None
    if sidecar_manifest_path.exists():
        sidecar_report = validate_gold_sidecars(args.substrate, args.dataset)
        manifest["gold_sidecar_manifest_sha256"] = file_sha256(sidecar_manifest_path)
    manifest["gold_sidecar_report"] = sidecar_report
    manifest["embedding_backend"] = embedding_backend or manifest.get("embedding_backend")
    manifest["embedding_dimension"] = embedding_dimension or manifest.get("embedding_dimension")
    manifest_path = args.output / "run_manifest.json"
    if args.resume and not manifest_path.exists():
        raise ValueError("resume requested but run_manifest.json does not exist")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("resume refused: source/config/substrate/renderer manifest changed")
    if not sidecar_manifest_path.exists():
        raise ValueError("gold evidence sidecar is missing; run repair_gold_sidecars.py")
    if not embedding_name or not embedding_backend or embedding_dimension <= 0:
        raise ValueError("substrate manifest must declare embedding model, backend, and positive dimension")
    if sidecar_report is None:
        raise ValueError("gold evidence sidecar validation did not run")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)

    gold = [asdict(item) for item in sidecars.gold_support]
    base_config = AgentConfig.from_yaml(args.config)
    if base_config.protocol != "native_tool_calling":
        raise ValueError("formal interface study requires protocol: native_tool_calling")
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
        if args.dataset in {"novel", "medical"}:
            scored.update({key: row[key] for key in (
                "source_question_id", "source_row_index", "substrate_question_id"
            )})
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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/interface_study_v2_qwen38_vllm.yaml"),
    )
    parser.add_argument("--skill", type=Path, default=Path("skills/interface_study.md"))
    parser.add_argument("--output", type=Path, default=Path("runs/interface-study-v2"))
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--semantic-eval", action="store_true")
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--judge-host", default="http://localhost:11434")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--judge-config", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="limit questions for a smoke run")
    parser.add_argument("--question-indices", nargs="+", type=int, default=None,
                        help="source row indices to run after full source/sidecar validation")
    parser.add_argument(
        "--allow-substrate-embedding-mismatch",
        action="store_true",
        help="diagnostic only: run with the substrate's recorded embedding backend; never use for formal runs",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
