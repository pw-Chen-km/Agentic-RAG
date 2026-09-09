from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rag.skillopt import heldout100
from agentic_rag.skillopt.heldout100 import (
    CONDITIONS,
    build_heldout100_plan,
    execute_heldout100_plan,
)


def test_plan_requires_every_completed_pilot_best_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    missing = paths["pilot"] / "organized" / "best_skill.md"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="organized:.*best_skill.md"):
        _build(paths)

    assert not paths["output"].exists()


def test_plan_preserves_exact_sample_order_and_official_skill_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    plan = _build(paths)

    assert [item.name for item in plan.conditions] == list(CONDITIONS)
    assert list(plan.sample_ids) == [f"sample-{index:03d}" for index in range(100)]
    for condition in plan.conditions:
        assert condition.best_skill == (
            paths["pilot"] / condition.name / "best_skill.md"
        )
        assert "--resume" not in condition.command
        assert condition.command[condition.command.index("--split") + 1] == str(
            paths["sample"] / "questions.jsonl"
        )
        assert condition.command[condition.command.index("--substrate") + 1] == str(
            paths["substrate"]
        )


def test_plan_rejects_sample_order_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    jsonl = paths["sample"] / "questions.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows)),
        encoding="utf-8",
    )
    manifest_path = paths["sample"] / "sample_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["questions.jsonl"]["sha256"] = _sha256(jsonl)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="different ordering"):
        _build(paths)


def test_plan_rejects_pilot_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    train = paths["split"] / "train.jsonl"
    row = json.loads(train.read_text(encoding="utf-8"))
    row["id"] = "sample-000"
    train.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="overlaps pilot"):
        _build(paths)


def test_plan_rejects_target_thinking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    config = paths["config"]
    config.write_text(
        config.read_text(encoding="utf-8").replace("think: false", "think: true"),
        encoding="utf-8",
    )
    pilot_manifest_path = paths["pilot"] / "ablation_manifest.json"
    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    pilot_manifest["inputs"]["agent_config_sha256"] = _sha256(config)
    pilot_manifest_path.write_text(json.dumps(pilot_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="thinking must be explicitly disabled"):
        _build(paths)


def test_execution_is_sequential_and_writes_common_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    plan = _build(paths)
    calls: list[str] = []
    active = 0
    maximum_active = 0

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal active, maximum_active
        assert kwargs["check"] is True
        assert kwargs["cwd"] == paths["repo"]
        active += 1
        maximum_active = max(maximum_active, active)
        output = Path(command[command.index("--output") + 1])
        skill = Path(command[command.index("--skill") + 1])
        condition = output.name
        calls.append(condition)
        condition_index = list(CONDITIONS).index(condition)
        llm_correct = 89 + condition_index
        exact_correct = 90 + condition_index
        contain_correct = 92 + condition_index
        rows = []
        for index, source in enumerate(plan.sample_rows):
            rows.append(
                {
                    "id": source["id"],
                    "question": source["question"],
                    "gold_answer": source["answer"],
                    "predicted_answer": "" if index == 0 else source["answer"],
                    "llm_acc": int(index < llm_correct),
                    "hard": int(index < llm_correct),
                    "soft": int(index < contain_correct),
                    "normalized_exact": int(index < exact_correct),
                    "contain_acc": int(index < contain_correct),
                    "termination_reason": "error" if index == 0 else "finish",
                    "error_code": "fixture_error" if index == 0 else None,
                    "invalid_attempts": index % 2,
                    "usage": {
                        "policy_calls": 2,
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "reasoning_tokens": 0,
                        "total_tokens": 13,
                        "retrieved_tokens": 5,
                    },
                    "judge_usage": {
                        "calls": 1,
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "reasoning_tokens": 0,
                        "total_tokens": 3,
                    },
                }
            )
        payload = {
            "run_contract": {
                "policy_model": plan.model,
                "judge_model": plan.model,
                "judge_host": plan.host,
                "judge_thinking": False,
                "split_sha256": plan.sample_jsonl_sha256,
                "skill_sha256": hashlib.sha256(skill.read_bytes()).hexdigest(),
                "config_sha256": plan.agent_config_sha256,
                "substrate_manifest_sha256": plan.substrate_manifest_sha256,
                "expected_count": 100,
                "dataset": "hotpotqa",
                "scope_id": "hotpotqa:benchmark_exact:dev",
            },
            "results": rows,
        }
        output.mkdir(parents=True)
        (output / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
        active -= 1
        return SimpleNamespace(returncode=0)

    times = iter((0.0, 10.0, 20.0, 32.0, 40.0, 55.0, 60.0, 78.0))
    endpoint_checks: list[tuple[str, str]] = []
    summary = execute_heldout100_plan(
        plan,
        run_command=fake_run,
        check_endpoint=lambda host, model: endpoint_checks.append((host, model)),
        clock=lambda: next(times),
    )

    assert calls == list(CONDITIONS)
    assert maximum_active == 1
    assert endpoint_checks == [("http://127.0.0.1:11435", "qwen3.6:35b-a3b-bf16")]
    assert [item["normalized_exact_correct"] for item in summary["systems"]] == [
        90,
        91,
        92,
        93,
    ]
    assert [item["llm_acc_correct"] for item in summary["systems"]] == [
        89,
        90,
        91,
        92,
    ]
    assert [item["contain_correct"] for item in summary["systems"]] == [
        92,
        93,
        94,
        95,
    ]
    assert [item["elapsed_seconds"] for item in summary["systems"]] == [
        10.0,
        12.0,
        15.0,
        18.0,
    ]
    assert all(item["blank_answers"] == 1 for item in summary["systems"])
    assert all(item["errors"] == 1 for item in summary["systems"])
    assert all(item["policy_calls_total"] == 200 for item in summary["systems"])
    assert all(item["total_tokens"] == 1300 for item in summary["systems"])
    assert all(item["judge_tokens_total"] == 300 for item in summary["systems"])
    assert (paths["output"] / "summary.json").is_file()
    assert (paths["output"] / "report.md").is_file()
    manifest = json.loads(
        (paths["output"] / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["execution"]["max_concurrent_conditions"] == 1


def test_plan_accepts_parallel4_manifest_base_agent_config_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    manifest_path = paths["pilot"] / "ablation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = manifest["inputs"]
    inputs.pop("agent_config")
    inputs.pop("agent_config_sha256")
    inputs["base_agent_config_sha256"] = _sha256(paths["config"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = _build(paths)

    assert [item.name for item in plan.conditions] == list(CONDITIONS)
    assert plan.agent_config == paths["config"].resolve()


def _build(paths: dict[str, Path]):
    return build_heldout100_plan(
        repo_root=paths["repo"],
        pilot_root=paths["pilot"],
        sample_dir=paths["sample"],
        substrate=paths["substrate"],
        agent_config=paths["config"],
        output_root=paths["output"],
        python_executable="/fixture/python",
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo = tmp_path / "repo"
    pilot = tmp_path / "pilot"
    split = tmp_path / "pilot_split"
    sample = tmp_path / "sample100"
    substrate = tmp_path / "substrate"
    output = tmp_path / "heldout_output"
    for path in (repo / "scripts", pilot, split, sample, substrate):
        path.mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "run_benchmark_eval.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    config = repo / "agent.yaml"
    config.write_text(
        """agent:
  max_steps: 10
  max_policy_attempts: 12
  max_retrieved_tokens: 12000
policy:
  provider: ollama
  model: qwen3.6:35b-a3b-bf16
  host: http://127.0.0.1:11435
  temperature: 0
  think: false
  num_ctx: 262144
""",
        encoding="utf-8",
    )
    (substrate / "manifest.json").write_text(
        json.dumps({"fixture": True}), encoding="utf-8"
    )
    for split_name in ("train", "validation", "test"):
        (split / f"{split_name}.jsonl").write_text(
            json.dumps(
                {
                    "id": f"pilot-{split_name}",
                    "question": f"{split_name}?",
                    "answer": split_name,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (split / "split_manifest.json").write_text(
        json.dumps({"fixture": True}), encoding="utf-8"
    )

    sample_rows = [
        {
            "id": f"sample-{index:03d}",
            "question": f"Question {index}?",
            "answer": f"Answer {index}",
            "question_type": "bridge" if index < 81 else "comparison",
            "scope_id": "hotpotqa:benchmark_exact:dev",
            "source": "hotpotqa",
        }
        for index in range(100)
    ]
    questions_json = sample / "questions.json"
    questions_jsonl = sample / "questions.jsonl"
    questions_json.write_text(json.dumps(sample_rows), encoding="utf-8")
    questions_jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in sample_rows),
        encoding="utf-8",
    )
    sample_manifest = {
        "count": 100,
        "selected_ids": [row["id"] for row in sample_rows],
        "outputs": {
            "questions.json": {"sha256": _sha256(questions_json)},
            "questions.jsonl": {"sha256": _sha256(questions_jsonl)},
        },
    }
    (sample / "sample_manifest.json").write_text(
        json.dumps(sample_manifest), encoding="utf-8"
    )

    conditions = []
    skill = """# Fixture skill

## Trainable retrieval workflow

Search carefully.

## Fixed answer contract

Answer briefly.
"""
    for name in CONDITIONS:
        condition = pilot / name
        condition.mkdir()
        (condition / "best_skill.md").write_text(skill, encoding="utf-8")
        (condition / "workflow_summary.json").write_text(
            json.dumps({"complete": True}), encoding="utf-8"
        )
        (condition / "final_metrics.json").write_text(
            json.dumps({"complete": True}), encoding="utf-8"
        )
        conditions.append(
            {
                "condition": name,
                "output": condition.as_posix(),
                "argv": ["runner", "--trajectory-representation", name],
            }
        )
    (pilot / "ablation_manifest.json").write_text(
        json.dumps(
            {
                "study": "skillopt_end_to_end_iterative_trajectory_representation",
                "conditions": conditions,
                "inputs": {
                    "substrate": substrate.as_posix(),
                    "substrate_manifest_sha256": _sha256(substrate / "manifest.json"),
                    "agent_config": config.as_posix(),
                    "agent_config_sha256": _sha256(config),
                    "split_dir": split.as_posix(),
                    "split_manifest_sha256": _sha256(split / "split_manifest.json"),
                },
            }
        ),
        encoding="utf-8",
    )

    sidecar_questions = [
        SimpleNamespace(
            question_id=row["id"],
            question=row["question"],
            answer=row["answer"],
            question_type=row["question_type"],
            scope_id=row["scope_id"],
        )
        for row in sample_rows
    ]
    monkeypatch.setattr(
        heldout100,
        "_validate_provenance_substrate",
        lambda path: (
            SimpleNamespace(),
            SimpleNamespace(benchmark_questions=sidecar_questions),
        ),
    )
    monkeypatch.setattr(
        heldout100,
        "validate_hotpotqa_provenance_lineage",
        lambda split_dir, substrate: {
            "split_manifest_sha256": _sha256(split_dir / "split_manifest.json")
        },
    )
    return {
        "repo": repo.resolve(),
        "pilot": pilot.resolve(),
        "split": split.resolve(),
        "sample": sample.resolve(),
        "substrate": substrate.resolve(),
        "config": config.resolve(),
        "output": output.resolve(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
