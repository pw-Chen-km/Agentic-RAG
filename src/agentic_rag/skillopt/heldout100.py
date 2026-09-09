"""External held-out-100 evaluation for the four SkillOpt conditions.

This module deliberately separates preflight validation from execution.  All
four completed experiment outputs, the fixed Sample-100 lineage, the provenance
substrate, and the no-thinking target configuration are validated before the
first target rollout can start.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_rag.agent.config import AgentConfig, OllamaPolicyConfig
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.skillopt.provenance import validate_hotpotqa_provenance_lineage
from agentic_rag.substrate.storage import EvaluationSidecars, Substrate


CONDITIONS = (
    "raw",
    "organized",
    "organized_support_labels",
    "progress_abstracted",
)
CONDITION_LABELS = {
    "raw": "Raw Trajectory",
    "organized": "Result-Driven Trajectory",
    "organized_support_labels": "Organized + Supporting-Fact Labels",
    "progress_abstracted": "Progress-Abstracted Trajectory",
}
EXPECTED_MODEL = "qwen3.6:35b-a3b-bf16"
EXPECTED_HOST = "http://127.0.0.1:11435"
EXPECTED_SCOPE = "hotpotqa:benchmark_exact:dev"
EXPECTED_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_SAMPLE_SIZE = 100


@dataclass(frozen=True, slots=True)
class ConditionPlan:
    name: str
    label: str
    pilot_output: Path
    best_skill: Path
    best_skill_sha256: str
    workflow_summary_sha256: str
    final_metrics_sha256: str
    output: Path
    log: Path
    timing: Path
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Heldout100Plan:
    repo_root: Path
    pilot_root: Path
    pilot_manifest: Path
    pilot_split_dir: Path
    sample_dir: Path
    sample_json: Path
    sample_jsonl: Path
    sample_manifest: Path
    sample_rows: tuple[dict[str, Any], ...]
    sample_ids: tuple[str, ...]
    substrate: Path
    agent_config: Path
    output_root: Path
    conditions: tuple[ConditionPlan, ...]
    model: str
    host: str
    substrate_manifest_sha256: str
    agent_config_sha256: str
    sample_json_sha256: str
    sample_jsonl_sha256: str
    sample_manifest_sha256: str
    pilot_manifest_sha256: str
    pilot_split_manifest_sha256: str

    def manifest(self, *, mode: str, status: str) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "study": "skillopt_trajectory_best_skill_external_heldout100",
            "status": status,
            "execution": {
                "mode": mode,
                "condition_order": list(CONDITIONS),
                "max_concurrent_conditions": 1,
                "sequential": True,
                "resume_supported": True,
            },
            "contract": {
                "dataset": "hotpotqa",
                "question_count": EXPECTED_SAMPLE_SIZE,
                "question_ids": list(self.sample_ids),
                "scope_id": EXPECTED_SCOPE,
                "model": self.model,
                "ollama_host": self.host,
                "temperature": 0,
                "thinking": False,
                "embedding_model": EXPECTED_EMBEDDING,
                "embedding_dimension": 384,
                "metrics": [
                    "llm_accuracy",
                    "normalized_exact_match",
                    "contain_accuracy",
                    "blank_answers",
                    "errors",
                    "policy_calls",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "judge_tokens",
                    "retrieved_tokens",
                    "elapsed_seconds",
                ],
            },
            "inputs": {
                "pilot_root": self.pilot_root.as_posix(),
                "pilot_manifest": {
                    "path": self.pilot_manifest.as_posix(),
                    "sha256": self.pilot_manifest_sha256,
                },
                "pilot_split_dir": self.pilot_split_dir.as_posix(),
                "pilot_split_manifest_sha256": (self.pilot_split_manifest_sha256),
                "sample_dir": self.sample_dir.as_posix(),
                "sample_manifest_sha256": self.sample_manifest_sha256,
                "sample_json_sha256": self.sample_json_sha256,
                "sample_jsonl_sha256": self.sample_jsonl_sha256,
                "substrate": self.substrate.as_posix(),
                "substrate_manifest_sha256": (self.substrate_manifest_sha256),
                "agent_config": self.agent_config.as_posix(),
                "agent_config_sha256": self.agent_config_sha256,
            },
            "conditions": [
                {
                    "name": item.name,
                    "label": item.label,
                    "pilot_output": item.pilot_output.as_posix(),
                    "best_skill": item.best_skill.as_posix(),
                    "best_skill_sha256": item.best_skill_sha256,
                    "workflow_summary_sha256": (item.workflow_summary_sha256),
                    "final_metrics_sha256": item.final_metrics_sha256,
                    "output": item.output.as_posix(),
                    "log": item.log.as_posix(),
                    "timing": item.timing.as_posix(),
                    "argv": list(item.command),
                }
                for item in self.conditions
            ],
        }


def build_heldout100_plan(
    *,
    repo_root: str | Path,
    pilot_root: str | Path,
    sample_dir: str | Path,
    substrate: str | Path,
    agent_config: str | Path,
    output_root: str | Path,
    python_executable: str,
) -> Heldout100Plan:
    """Validate every immutable input and return the sequential run plan."""

    root = _require_directory(Path(repo_root), "repository root")
    pilot = _require_directory(Path(pilot_root), "pilot root")
    sample = _require_directory(Path(sample_dir), "Sample-100 directory")
    substrate_path = _require_directory(Path(substrate), "provenance substrate")
    config_path = _require_file(Path(agent_config), "Agent config")
    output = Path(output_root).expanduser().resolve()
    if output == pilot:
        raise ValueError("held-out output root must not equal the pilot root")

    pilot_manifest_path = _require_file(
        pilot / "ablation_manifest.json", "pilot ablation manifest"
    )
    pilot_manifest = _load_json_object(pilot_manifest_path, "pilot ablation manifest")
    if pilot_manifest.get("study") != (
        "skillopt_end_to_end_iterative_trajectory_representation"
    ):
        raise ValueError("pilot manifest declares an unexpected study")
    manifest_conditions = pilot_manifest.get("conditions")
    if not isinstance(manifest_conditions, list):
        raise ValueError("pilot manifest has no conditions list")
    condition_rows: dict[str, Mapping[str, Any]] = {}
    for row in manifest_conditions:
        if not isinstance(row, Mapping):
            raise ValueError("pilot condition entries must be objects")
        name = str(row.get("condition") or "")
        if not name or name in condition_rows:
            raise ValueError("pilot conditions contain a missing or duplicate name")
        condition_rows[name] = row
    if set(condition_rows) != set(CONDITIONS):
        raise ValueError(
            "pilot manifest must contain exactly these conditions: "
            + ", ".join(CONDITIONS)
        )

    inputs = pilot_manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("pilot manifest has no inputs mapping")
    pilot_substrate = _resolve_manifest_path(inputs.get("substrate"), root)
    pilot_split_dir = _resolve_manifest_path(inputs.get("split_dir"), root)
    if pilot_substrate != substrate_path:
        raise ValueError(
            "external evaluation substrate does not match the pilot manifest"
        )
    pilot_agent_config_value = inputs.get("agent_config")
    if isinstance(pilot_agent_config_value, str) and pilot_agent_config_value.strip():
        pilot_agent_config = _resolve_manifest_path(pilot_agent_config_value, root)
        if pilot_agent_config != config_path:
            raise ValueError(
                "external evaluation Agent config does not match the pilot manifest"
            )
        _require_declared_sha256(
            inputs,
            "agent_config_sha256",
            config_path,
            "pilot Agent config",
        )
    elif inputs.get("base_agent_config_sha256") != _sha256(config_path):
        raise ValueError(
            "external evaluation Agent config does not match the pilot base config"
        )
    pilot_split_dir = _require_directory(pilot_split_dir, "pilot split directory")
    _require_declared_sha256(
        inputs,
        "substrate_manifest_sha256",
        substrate_path / "manifest.json",
        "pilot substrate manifest",
    )
    _require_declared_sha256(
        inputs,
        "split_manifest_sha256",
        pilot_split_dir / "split_manifest.json",
        "pilot split manifest",
    )

    agent = AgentConfig.from_yaml(config_path)
    policy = _validate_target_policy(agent)
    runtime, sidecars = _validate_provenance_substrate(substrate_path)
    lineage = validate_hotpotqa_provenance_lineage(pilot_split_dir, substrate_path)
    pilot_ids = _pilot_question_ids(pilot_split_dir)
    sample_rows, sample_ids, sample_paths = _validate_sample100(
        sample,
        pilot_ids=pilot_ids,
        sidecars=sidecars,
    )
    del runtime

    benchmark_runner = _require_file(
        root / "scripts" / "run_benchmark_eval.py",
        "benchmark evaluation runner",
    )
    condition_plans: list[ConditionPlan] = []
    missing: list[str] = []
    incomplete: list[str] = []
    for name in CONDITIONS:
        row = condition_rows[name]
        declared_output = _resolve_manifest_path(row.get("output"), root)
        expected_pilot_output = (pilot / name).resolve()
        if declared_output != expected_pilot_output:
            raise ValueError(
                f"pilot output for {name} is {declared_output}; expected the "
                f"declared condition directory {expected_pilot_output}"
            )
        best_skill = expected_pilot_output / "best_skill.md"
        workflow_summary = expected_pilot_output / "workflow_summary.json"
        final_metrics = expected_pilot_output / "final_metrics.json"
        if not best_skill.is_file():
            missing.append(f"{name}: {best_skill}")
            continue
        if best_skill.stat().st_size == 0:
            incomplete.append(f"{name}: best_skill.md is empty")
            continue
        completion_files = [
            path for path in (workflow_summary, final_metrics) if not path.is_file()
        ]
        if completion_files:
            incomplete.append(
                f"{name}: missing completion file(s) "
                + ", ".join(path.name for path in completion_files)
            )
            continue
        _load_json_object(workflow_summary, f"{name} workflow summary")
        _load_json_object(final_metrics, f"{name} final metrics")
        SkillDocument.load(best_skill)
        condition_output = output / name
        if condition_output == expected_pilot_output:
            raise ValueError(
                f"held-out output for {name} would overwrite its pilot output"
            )
        command = (
            python_executable,
            benchmark_runner.as_posix(),
            "--substrate",
            substrate_path.as_posix(),
            "--split",
            sample_paths["jsonl"].as_posix(),
            "--config",
            config_path.as_posix(),
            "--skill",
            best_skill.as_posix(),
            "--output",
            condition_output.as_posix(),
            "--expected-count",
            str(EXPECTED_SAMPLE_SIZE),
            "--dataset",
            "hotpotqa",
        )
        condition_plans.append(
            ConditionPlan(
                name=name,
                label=CONDITION_LABELS[name],
                pilot_output=expected_pilot_output,
                best_skill=best_skill,
                best_skill_sha256=_sha256(best_skill),
                workflow_summary_sha256=_sha256(workflow_summary),
                final_metrics_sha256=_sha256(final_metrics),
                output=condition_output,
                log=output / "logs" / f"{name}.log",
                timing=output / "timing" / f"{name}.json",
                command=command,
            )
        )
    if missing:
        raise FileNotFoundError(
            "pilot best_skill.md is missing; no held-out condition was started. "
            "Complete all four experiment runs first. Missing: " + "; ".join(missing)
        )
    if incomplete:
        raise RuntimeError(
            "pilot output is incomplete; no held-out condition was started. "
            + "; ".join(incomplete)
        )

    return Heldout100Plan(
        repo_root=root,
        pilot_root=pilot,
        pilot_manifest=pilot_manifest_path,
        pilot_split_dir=pilot_split_dir,
        sample_dir=sample,
        sample_json=sample_paths["json"],
        sample_jsonl=sample_paths["jsonl"],
        sample_manifest=sample_paths["manifest"],
        sample_rows=tuple(sample_rows),
        sample_ids=tuple(sample_ids),
        substrate=substrate_path,
        agent_config=config_path,
        output_root=output,
        conditions=tuple(condition_plans),
        model=policy.model,
        host=policy.host,
        substrate_manifest_sha256=_sha256(substrate_path / "manifest.json"),
        agent_config_sha256=_sha256(config_path),
        sample_json_sha256=_sha256(sample_paths["json"]),
        sample_jsonl_sha256=_sha256(sample_paths["jsonl"]),
        sample_manifest_sha256=_sha256(sample_paths["manifest"]),
        pilot_manifest_sha256=_sha256(pilot_manifest_path),
        pilot_split_manifest_sha256=str(lineage["split_manifest_sha256"]),
    )


def execute_heldout100_plan(
    plan: Heldout100Plan,
    *,
    resume: bool = False,
    run_command: Callable[..., Any] = subprocess.run,
    check_endpoint: Callable[[str, str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run each condition serially, then write the common comparison."""

    _preflight_outputs(plan, resume=resume)
    endpoint_check = check_endpoint or verify_ollama_model
    endpoint_check(plan.host, plan.model)
    plan.output_root.mkdir(parents=True, exist_ok=True)
    (plan.output_root / "logs").mkdir(exist_ok=True)
    (plan.output_root / "timing").mkdir(exist_ok=True)
    manifest_path = plan.output_root / "evaluation_manifest.json"
    _atomic_write_json(
        manifest_path,
        plan.manifest(mode="sequential", status="running"),
    )

    active = 0
    try:
        for condition in plan.conditions:
            summary_path = condition.output / "summary.json"
            if summary_path.is_file():
                _validated_results(condition, plan)
                continue
            command = list(condition.command)
            if condition.output.is_dir():
                command.append("--resume")
            condition.log.parent.mkdir(parents=True, exist_ok=True)
            started_at = datetime.now(timezone.utc).isoformat()
            started = clock()
            return_code: int | None = None
            error: str | None = None
            active += 1
            if active != 1:
                raise RuntimeError("held-out conditions must never run in parallel")
            try:
                with condition.log.open("a", encoding="utf-8") as log_handle:
                    completed = run_command(
                        command,
                        cwd=plan.repo_root,
                        check=True,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                return_code = int(getattr(completed, "returncode", 0))
            except Exception as exc:
                value = getattr(exc, "returncode", None)
                return_code = int(value) if isinstance(value, int) else None
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                active -= 1
                _append_timing(
                    condition,
                    started_at=started_at,
                    elapsed_seconds=max(clock() - started, 0.0),
                    return_code=return_code,
                    error=error,
                )
            if not summary_path.is_file():
                raise RuntimeError(
                    f"{condition.name} runner completed without summary.json: "
                    f"{summary_path}"
                )
            _validated_results(condition, plan)

        comparison = write_heldout100_comparison(plan)
    except Exception as exc:
        failed = plan.manifest(mode="sequential", status="failed")
        failed["failure"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(manifest_path, failed)
        raise
    _atomic_write_json(
        manifest_path,
        plan.manifest(mode="sequential", status="complete"),
    )
    return comparison


def write_heldout100_comparison(plan: Heldout100Plan) -> dict[str, Any]:
    """Validate all result orderings and write JSON plus a Markdown table."""

    systems = [_condition_metrics(condition, plan) for condition in plan.conditions]
    results_by_condition = {
        condition.name: _validated_results(condition, plan)
        for condition in plan.conditions
    }
    per_question = []
    for index, source in enumerate(plan.sample_rows):
        row: dict[str, Any] = {
            "id": plan.sample_ids[index],
            "question": source["question"],
            "gold_answer": source["answer"],
            "conditions": {},
        }
        for condition in plan.conditions:
            result = results_by_condition[condition.name][index]
            row["conditions"][condition.name] = {
                "predicted_answer": result.get("predicted_answer") or "",
                "llm_acc": int(result.get("llm_acc") or 0),
                "normalized_exact": int(result.get("normalized_exact") or 0),
                "contain_acc": int(result.get("contain_acc") or 0),
                "error_code": result.get("error_code"),
                "termination_reason": result.get("termination_reason"),
            }
        per_question.append(row)
    payload = {
        "schema_version": "1.0",
        "contract": plan.manifest(mode="sequential", status="complete")["contract"],
        "condition_order": list(CONDITIONS),
        "systems": systems,
        "per_question": per_question,
    }
    _atomic_write_json(plan.output_root / "summary.json", payload)
    _atomic_write_text(plan.output_root / "report.md", _comparison_markdown(systems))
    return payload


def verify_ollama_model(host: str, model: str) -> None:
    """Check the configured local Ollama endpoint without generating text."""

    url = f"{host.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot reach Ollama endpoint {url}: {exc}") from exc
    models = payload.get("models") if isinstance(payload, Mapping) else None
    names = {
        str(item.get("name") or item.get("model") or "")
        for item in models or []
        if isinstance(item, Mapping)
    }
    if model not in names:
        raise RuntimeError(
            f"Ollama endpoint {host} does not list required model {model!r}"
        )


def _validate_target_policy(agent: AgentConfig) -> OllamaPolicyConfig:
    policy = agent.policy
    if not isinstance(policy, OllamaPolicyConfig):
        raise ValueError("external evaluation requires the Ollama policy provider")
    if policy.model != EXPECTED_MODEL:
        raise ValueError(
            f"target model must be {EXPECTED_MODEL!r}, found {policy.model!r}"
        )
    if policy.host != EXPECTED_HOST:
        raise ValueError(
            f"Ollama host must be {EXPECTED_HOST!r}, found {policy.host!r}"
        )
    if policy.think is not False:
        raise ValueError("Target Agent thinking must be explicitly disabled")
    if policy.temperature != 0:
        raise ValueError("Target Agent temperature must be zero")
    return policy


def _validate_provenance_substrate(
    path: Path,
) -> tuple[Substrate, EvaluationSidecars]:
    runtime = Substrate.open(path)
    manifest = runtime.manifest
    if manifest.dataset != "hotpotqa":
        raise ValueError("external evaluation substrate must declare HotpotQA")
    if manifest.source_format != "hotpotqa_global_provenance":
        raise ValueError(
            "external evaluation requires hotpotqa_global_provenance substrate"
        )
    if manifest.schema_version != "2.1":
        raise ValueError("provenance substrate schema version must be 2.1")
    if manifest.embedding_model.name != EXPECTED_EMBEDDING:
        raise ValueError(f"embedding model must be {EXPECTED_EMBEDDING!r}")
    if manifest.embedding_dimension != 384:
        raise ValueError("embedding dimension must be 384")
    runtime.require_scope(EXPECTED_SCOPE)
    sidecars = EvaluationSidecars.open(path)
    return runtime, sidecars


def _validate_sample100(
    sample_dir: Path,
    *,
    pilot_ids: set[str],
    sidecars: EvaluationSidecars,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Path]]:
    paths = {
        "json": _require_file(sample_dir / "questions.json", "Sample-100 JSON"),
        "jsonl": _require_file(sample_dir / "questions.jsonl", "Sample-100 JSONL"),
        "manifest": _require_file(
            sample_dir / "sample_manifest.json", "Sample-100 manifest"
        ),
    }
    raw_json = json.loads(paths["json"].read_text(encoding="utf-8"))
    if not isinstance(raw_json, list) or not all(
        isinstance(row, dict) for row in raw_json
    ):
        raise ValueError("Sample-100 questions.json must contain an object list")
    rows: list[dict[str, Any]] = list(raw_json)
    jsonl_rows = _load_jsonl(paths["jsonl"])
    if rows != jsonl_rows:
        raise ValueError(
            "Sample-100 JSON and JSONL rows differ or have different ordering"
        )
    ids = [str(row.get("id") or "") for row in rows]
    if len(ids) != EXPECTED_SAMPLE_SIZE or len(set(ids)) != EXPECTED_SAMPLE_SIZE:
        raise ValueError("Sample-100 must contain exactly 100 unique IDs")
    manifest = _load_json_object(paths["manifest"], "Sample-100 manifest")
    if manifest.get("count") != EXPECTED_SAMPLE_SIZE:
        raise ValueError("Sample-100 manifest count must be 100")
    if manifest.get("selected_ids") != ids:
        raise ValueError("Sample-100 manifest selected_ids order does not match")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Sample-100 manifest has no outputs mapping")
    for filename, key in (("questions.json", "json"), ("questions.jsonl", "jsonl")):
        record = outputs.get(filename)
        if not isinstance(record, Mapping) or record.get("sha256") != _sha256(
            paths[key]
        ):
            raise ValueError(f"Sample-100 {filename} SHA-256 does not match")
    overlap = sorted(set(ids) & pilot_ids)
    if overlap:
        raise ValueError(
            "external Sample-100 overlaps pilot train/validation/test IDs: "
            + ", ".join(overlap[:10])
        )
    type_counts = Counter(str(row.get("question_type") or "") for row in rows)
    if type_counts != Counter({"bridge": 81, "comparison": 19}):
        raise ValueError(
            "Sample-100 question types must contain 81 bridge and 19 comparison"
        )
    question_by_id = {item.question_id: item for item in sidecars.benchmark_questions}
    for row in rows:
        question_id = str(row["id"])
        if row.get("scope_id") != EXPECTED_SCOPE or row.get("source") != "hotpotqa":
            raise ValueError(
                f"Sample-100 item {question_id} has the wrong source or scope"
            )
        source = question_by_id.get(question_id)
        if source is None:
            raise ValueError(
                f"Sample-100 item {question_id} is absent from substrate sidecars"
            )
        if (
            row.get("question") != source.question
            or row.get("answer") != source.answer
            or row.get("question_type") != source.question_type
            or row.get("scope_id") != source.scope_id
        ):
            raise ValueError(
                f"Sample-100 item {question_id} does not match substrate sidecars"
            )
    return rows, ids, paths


def _pilot_question_ids(split_dir: Path) -> set[str]:
    ids: list[str] = []
    for split in ("train", "validation", "test"):
        ids.extend(
            str(row.get("id") or "")
            for row in _load_jsonl(split_dir / f"{split}.jsonl")
        )
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("pilot split IDs must be non-empty and disjoint")
    return set(ids)


def _preflight_outputs(plan: Heldout100Plan, *, resume: bool) -> None:
    for condition in plan.conditions:
        if not condition.output.exists():
            continue
        if not resume:
            raise FileExistsError(
                f"held-out output already exists for {condition.name}: "
                f"{condition.output}; use --resume"
            )
        summary = condition.output / "summary.json"
        progress = condition.output / "progress.json"
        if not summary.is_file() and not progress.is_file():
            raise FileNotFoundError(
                f"cannot resume {condition.name}: neither summary.json nor "
                f"progress.json exists under {condition.output}"
            )
        if not condition.timing.is_file():
            raise FileNotFoundError(
                f"cannot resume {condition.name}: timing ledger is missing: "
                f"{condition.timing}"
            )
        if summary.is_file():
            _validated_results(condition, plan)


def _validated_results(
    condition: ConditionPlan, plan: Heldout100Plan
) -> list[dict[str, Any]]:
    summary_path = condition.output / "summary.json"
    payload = _load_json_object(summary_path, f"{condition.name} summary")
    contract = payload.get("run_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{condition.name} summary has no run contract")
    expected_contract = {
        "policy_model": plan.model,
        "judge_model": plan.model,
        "judge_host": plan.host,
        "judge_thinking": False,
        "split_sha256": plan.sample_jsonl_sha256,
        "skill_sha256": condition.best_skill_sha256,
        "config_sha256": plan.agent_config_sha256,
        "substrate_manifest_sha256": plan.substrate_manifest_sha256,
        "expected_count": EXPECTED_SAMPLE_SIZE,
        "dataset": "hotpotqa",
        "scope_id": EXPECTED_SCOPE,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"{condition.name} summary contract {key} does not match")
    rows = payload.get("results")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{condition.name} summary results must be an object list")
    ids = [str(row.get("id") or "") for row in rows]
    if ids != list(plan.sample_ids):
        raise ValueError(
            f"{condition.name} result IDs do not exactly match Sample-100 order"
        )
    for source, row in zip(plan.sample_rows, rows, strict=True):
        if (
            row.get("question") != source["question"]
            or row.get("gold_answer") != source["answer"]
        ):
            raise ValueError(
                f"{condition.name} result {row.get('id')} does not match sample"
            )
    return rows


def _condition_metrics(
    condition: ConditionPlan, plan: Heldout100Plan
) -> dict[str, Any]:
    rows = _validated_results(condition, plan)
    timing = _load_json_object(condition.timing, f"{condition.name} timing")
    elapsed = float(timing.get("elapsed_seconds_total") or 0.0)

    def total(key: str) -> int:
        return sum(int((row.get("usage") or {}).get(key) or 0) for row in rows)

    count = len(rows)
    llm_correct = sum(int(row.get("llm_acc") or 0) for row in rows)
    exact = sum(int(row.get("normalized_exact") or 0) for row in rows)
    contain = sum(int(row.get("contain_acc") or 0) for row in rows)
    calls = total("policy_calls")
    total_tokens = total("total_tokens")

    def judge_total(key: str) -> int:
        return sum(
            int((row.get("judge_usage") or {}).get(key) or 0) for row in rows
        )

    return {
        "condition": condition.name,
        "label": condition.label,
        "best_skill": condition.best_skill.as_posix(),
        "best_skill_sha256": condition.best_skill_sha256,
        "result_summary": (condition.output / "summary.json").as_posix(),
        "question_count": count,
        "llm_acc_correct": llm_correct,
        "llm_accuracy": llm_correct / count,
        "normalized_exact_correct": exact,
        "normalized_exact_accuracy": exact / count,
        "contain_correct": contain,
        "contain_accuracy": contain / count,
        "blank_answers": sum(
            not str(row.get("predicted_answer") or "").strip() for row in rows
        ),
        "errors": sum(bool(row.get("error_code")) for row in rows),
        "termination_reasons": dict(
            sorted(
                Counter(
                    str(row.get("termination_reason") or "unknown") for row in rows
                ).items()
            )
        ),
        "invalid_attempts_total": sum(
            int(row.get("invalid_attempts") or 0) for row in rows
        ),
        "policy_calls_total": calls,
        "policy_calls_average": calls / count,
        "input_tokens_total": total("input_tokens"),
        "output_tokens_total": total("output_tokens"),
        "reasoning_tokens_total": total("reasoning_tokens"),
        "total_tokens": total_tokens,
        "total_tokens_average": total_tokens / count,
        "judge_calls_total": judge_total("calls"),
        "judge_tokens_total": judge_total("total_tokens"),
        "retrieved_tokens_total": total("retrieved_tokens"),
        "elapsed_seconds": elapsed,
        "average_seconds_per_question": elapsed / count,
        "execution_attempts": len(timing.get("attempts") or []),
    }


def _comparison_markdown(systems: list[dict[str, Any]]) -> str:
    lines = [
        "# SkillOpt Best-Skill External Held-out-100 Comparison",
        "",
        (
            "All four best skills were evaluated sequentially on the same "
            "ordered HotpotQA Sample-100 with Qwen 3.6 thinking disabled."
        ),
        "",
        "| Metric | "
        + " | ".join(CONDITION_LABELS[name] for name in CONDITIONS)
        + " |",
        "|---|" + "---:|" * len(CONDITIONS),
    ]
    metrics = (
        ("LLM accuracy", "llm_accuracy", ".1%"),
        ("Normalized exact", "normalized_exact_accuracy", ".1%"),
        ("Contain accuracy", "contain_accuracy", ".1%"),
        ("Blank answers", "blank_answers", "d"),
        ("Errors", "errors", "d"),
        ("Policy calls", "policy_calls_total", "d"),
        ("Input tokens", "input_tokens_total", "d"),
        ("Output tokens", "output_tokens_total", "d"),
        ("Total tokens", "total_tokens", "d"),
        ("Judge tokens", "judge_tokens_total", "d"),
        ("Elapsed seconds", "elapsed_seconds", ".1f"),
        ("Seconds/question", "average_seconds_per_question", ".2f"),
    )
    for label, key, format_spec in metrics:
        values = [format(system[key], format_spec) for system in systems]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Each condition directory contains its complete 100-row result summary.",
            "The root `summary.json` contains the common contract and aligned per-question comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def _append_timing(
    condition: ConditionPlan,
    *,
    started_at: str,
    elapsed_seconds: float,
    return_code: int | None,
    error: str | None,
) -> None:
    if condition.timing.is_file():
        value = _load_json_object(condition.timing, f"{condition.name} timing")
    else:
        value = {
            "schema_version": "1.0",
            "condition": condition.name,
            "elapsed_seconds_total": 0.0,
            "attempts": [],
        }
    attempts = value.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError(f"{condition.name} timing attempts must be a list")
    attempts.append(
        {
            "started_at_utc": started_at,
            "elapsed_seconds": elapsed_seconds,
            "return_code": return_code,
            "error": error,
            "completed": error is None and return_code == 0,
        }
    )
    value["elapsed_seconds_total"] = (
        float(value.get("elapsed_seconds_total") or 0.0) + elapsed_seconds
    )
    _atomic_write_json(condition.timing, value)


def _resolve_manifest_path(value: object, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("pilot manifest contains a missing path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _require_declared_sha256(
    inputs: Mapping[str, Any], key: str, path: Path, role: str
) -> None:
    declared = inputs.get(key)
    if not isinstance(declared, str) or declared != _sha256(path):
        raise ValueError(f"{role} SHA-256 does not match the pilot manifest")


def _require_file(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} is missing: {resolved}")
    return resolved


def _require_directory(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{role} is missing: {resolved}")
    return resolved


def _load_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {role}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    required = _require_file(path, "JSONL input")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        required.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL at {required}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {required}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "CONDITIONS",
    "ConditionPlan",
    "Heldout100Plan",
    "build_heldout100_plan",
    "execute_heldout100_plan",
    "verify_ollama_model",
    "write_heldout100_comparison",
]
