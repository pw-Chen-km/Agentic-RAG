from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.skillopt import split_manifest_profile, summarize_rollout_usage


def test_split_manifest_selects_declared_dataset_profile(tmp_path: Path) -> None:
    (tmp_path / "split_manifest.json").write_text(
        json.dumps({"dataset": {"subset": "hotpotqa"}}),
        encoding="utf-8",
    )
    assert split_manifest_profile(tmp_path).key == "hotpotqa"
    (tmp_path / "split_manifest.json").write_text(
        json.dumps({"dataset": {"subset": "medical"}}),
        encoding="utf-8",
    )
    assert split_manifest_profile(tmp_path).key == "medical"


def test_skillopt_usage_uses_single_policy_counter(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "rollout_result.json").write_text(
        json.dumps(
            {
                "usage": {
                    "policy_calls": 2,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "reasoning_tokens": 5,
                    "total_tokens": 120,
                    "retrieved_tokens": 42,
                },
                "judge_usage": {
                    "calls": 1,
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "reasoning_tokens": 0,
                    "total_tokens": 11,
                },
            }
        ),
        encoding="utf-8",
    )
    usage = summarize_rollout_usage(tmp_path)
    assert usage["policy"] == {
        "calls": 2,
        "input_tokens": 100,
        "output_tokens": 20,
        "reasoning_tokens": 5,
        "total_tokens": 120,
    }
    assert usage["retrieved_tokens"] == 42
    assert "answer" not in usage
