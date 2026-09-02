"""Run or dry-run the end-to-end SkillOpt renderer conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


CONDITIONS = (
    "raw",
    "organized",
    "organized_support_labels",
    "progress_abstracted",
)


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=root / "configs" / "hotpotqa_qwen36_amd_nothink.yaml",
    )
    parser.add_argument(
        "--skillopt-config",
        type=Path,
        default=(
            root / "configs" / "hotpotqa_skillopt_qwen36_amd_trajectory_ablation.yaml"
        ),
    )
    parser.add_argument("--skill", type=Path, default=root / "skills" / "baseline.md")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
        help=(
            "Conditions to run in the listed order. Defaults to all; use "
            "progress_abstracted to supplement an existing three-arm study."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate controls and write the study manifest without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = Path(__file__).resolve().parents[1]
    substrate = _require_directory(args.substrate, "substrate")
    split_dir = _require_directory(args.split_dir, "split directory")
    agent_config = _require_file(args.agent_config, "Agent config")
    skillopt_config = _require_file(args.skillopt_config, "SkillOpt config")
    skill = _require_file(args.skill, "initial skill")
    if skill.name != "baseline.md":
        raise ValueError("all ablation conditions must start from skills/baseline.md")

    agent_yaml = _load_yaml(agent_config)
    skillopt_yaml = _load_yaml(skillopt_config)
    split_manifest = _load_json(split_dir / "split_manifest.json")
    split_counts = _validate_controls(agent_yaml, skillopt_yaml, split_manifest)

    output_root = args.output_root.resolve()
    runner = root / "scripts" / "run_local_qwen_skillopt.py"
    commands: list[dict[str, Any]] = []
    selected_conditions = tuple(dict.fromkeys(args.conditions))
    for condition in selected_conditions:
        condition_output = output_root / condition
        command = [
            sys.executable,
            str(runner),
            "--substrate",
            str(substrate),
            "--split-dir",
            str(split_dir),
            "--agent-config",
            str(agent_config),
            "--skillopt-config",
            str(skillopt_config),
            "--skill",
            str(skill),
            "--trajectory-representation",
            condition,
            "--output",
            str(condition_output),
        ]
        commands.append(
            {
                "condition": condition,
                "output": condition_output.as_posix(),
                "argv": command,
            }
        )

    model = str(agent_yaml["policy"]["model"])
    manifest = {
        "schema_version": "1.0",
        "study": "skillopt_end_to_end_iterative_trajectory_representation",
        "pilot": sum(split_counts.values()) <= 32,
        "available_conditions": list(CONDITIONS),
        "selected_conditions": list(selected_conditions),
        "conditions": commands,
        "shared_controls": {
            "initial_skill": skill.as_posix(),
            "initial_skill_sha256": _sha256_file(skill),
            "target_model": model,
            "target_thinking": False,
            "optimizer_model": model,
            "optimizer_thinking": True,
            "optimizer_roles": ["reflect", "merge", "rank"],
            "temperature": 0,
            "train_size": split_counts["train"],
            "validation_size": split_counts["validation"],
            "test_size": split_counts["test"],
            "epochs": 1,
            "validation_gate": True,
        },
        "inputs": {
            "substrate": substrate.as_posix(),
            "substrate_manifest_sha256": _sha256_file(substrate / "manifest.json"),
            "split_dir": split_dir.as_posix(),
            "split_manifest_sha256": _sha256_file(split_dir / "split_manifest.json"),
            "agent_config": agent_config.as_posix(),
            "agent_config_sha256": _sha256_file(agent_config),
            "skillopt_config": skillopt_config.as_posix(),
            "skillopt_config_sha256": _sha256_file(skillopt_config),
        },
        "execution": {
            "mode": "dry_run" if args.dry_run else "sequential",
            "rollouts_shared_across_conditions": False,
            "note": (
                "This is the end-to-end iterative study. Each condition begins "
                "from the same skill but produces its own later rollouts."
            ),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "ablation_manifest.json"
    _atomic_replace_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return
    for row in commands:
        subprocess.run(row["argv"], cwd=root, check=True)


def _validate_controls(
    agent: dict[str, Any],
    skillopt: dict[str, Any],
    split_manifest: dict[str, Any],
) -> dict[str, int]:
    policy = agent.get("policy")
    model = skillopt.get("model")
    train = skillopt.get("train")
    evaluation = skillopt.get("evaluation")
    if not all(isinstance(value, dict) for value in (policy, model, train, evaluation)):
        raise ValueError("ablation configs are missing required sections")
    assert isinstance(policy, dict)
    assert isinstance(model, dict)
    assert isinstance(train, dict)
    assert isinstance(evaluation, dict)
    target_model = policy.get("model")
    if policy.get("provider") != "ollama" or policy.get("think") is not False:
        raise ValueError("Target Agent must use Ollama with thinking disabled")
    if model.get("optimizer") != target_model or model.get("target") != target_model:
        raise ValueError("Target and optimizer must declare the same Qwen model")
    if model.get("optimizer_qwen_chat_enable_thinking") is not True:
        raise ValueError("optimizer thinking must be enabled")
    if model.get("target_qwen_chat_enable_thinking") is not False:
        raise ValueError("native target metadata must keep thinking disabled")
    if model.get("qwen_chat_temperature") != 0:
        raise ValueError("optimizer temperature must be zero")
    train_size = train.get("train_size")
    validation_size = evaluation.get("sel_env_num")
    test_size = evaluation.get("test_env_num")
    requested = {
        "train": train_size,
        "validation": validation_size,
        "test": test_size,
    }
    if train.get("num_epochs") != 1:
        raise ValueError("trajectory ablation must use exactly one epoch")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in requested.values()
    ):
        raise ValueError("train/validation/test sizes must be positive integers")
    if evaluation.get("use_gate") is not True:
        raise ValueError("native SkillOpt validation gate must remain enabled")
    splits = split_manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("split manifest has no splits mapping")
    actual = {
        name: value.get("count") if isinstance(value, dict) else None
        for name, value in splits.items()
        if name in requested
    }
    if actual != requested:
        raise ValueError(f"split counts are {actual}; config requests {requested}")
    return {name: int(value) for name, value in requested.items()}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_file(path: Path, role: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} is missing: {resolved}")
    return resolved


def _require_directory(path: Path, role: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{role} is missing: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_replace_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
