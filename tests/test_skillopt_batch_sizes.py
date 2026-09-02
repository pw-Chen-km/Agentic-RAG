"""Regression checks for the 100/50/50 experiment's 20/5 batching recipe.

No Target Agent, Judge, or Optimizer model calls are made by these tests.
"""

from __future__ import annotations

from collections import Counter
import importlib
import math
from pathlib import Path

import pytest
import yaml

from agentic_rag.skillopt.dataloader import AgenticRAGSkillOptDataLoader


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/hotpotqa_skillopt_qwen36_amd_trajectory_ablation_100_50_50.yaml"
)


def test_expanded_experiment_batching_recipe() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["train"] == {
        "num_epochs": 1,
        "train_size": 100,
        "batch_size": 20,
        "accumulation": 1,
        "seed": 42,
    }
    assert config["gradient"]["minibatch_size"] == 5
    assert config["gradient"]["failure_only"] is False
    # These are independent controls, not reflection minibatch size.
    assert config["gradient"]["merge_batch_size"] == 3
    assert config["optimizer"]["learning_rate"] == 1
    assert config["evaluation"]["use_gate"] is True
    assert config["evaluation"]["sel_env_num"] == 50
    assert config["evaluation"]["test_env_num"] == 50


def test_native_config_parser_forwards_twenty_and_five() -> None:
    pytest.importorskip("skillopt")
    from agentic_rag.skillopt.trainer import load_skillopt_config

    config = load_skillopt_config(CONFIG)
    assert config["batch_size"] == 20
    assert config["minibatch_size"] == 5
    assert config["accumulation"] == 1
    assert math.ceil(
        config["train_size"] / (config["batch_size"] * config["accumulation"])
    ) == 5


def test_hundred_training_items_become_five_disjoint_batches(tmp_path: Path) -> None:
    loader = AgenticRAGSkillOptDataLoader(tmp_path)
    items = [{"id": f"question-{index}"} for index in range(100)]
    loader._splits = {"train": items, "val": [], "test": []}
    batches = loader.plan_train_epoch(
        epoch=1, steps_per_epoch=5, accumulation=1, batch_size=20, seed=42,
    )
    assert [batch.batch_size for batch in batches] == [20] * 5
    actual = [item["id"] for batch in batches for item in batch.payload]
    assert Counter(actual) == Counter(item["id"] for item in items)
    repeated = loader.plan_train_epoch(
        epoch=1, steps_per_epoch=5, accumulation=1, batch_size=20, seed=42,
    )
    assert [batch.payload for batch in batches] == [batch.payload for batch in repeated]


@pytest.mark.parametrize("failures", range(21))
def test_native_reflection_covers_whole_batch_in_groups_at_most_five(
    failures: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("skillopt")
    reflect = importlib.import_module("skillopt.gradient.reflect")
    results = [
        {"id": str(index), "hard": int(index >= failures)}
        for index in range(20)
    ]
    seen: list[tuple[str, str, list[dict]]] = []
    frozen_skill = "The same current Skill for all twenty trajectories."

    def analyst(kind: str):
        def record(skill: str, batch: list[dict], prediction_dir: str, **kwargs):
            seen.append((kind, skill, list(batch)))
            # An analyst may propose no edit; no model service is invoked.
            return None
        return record

    monkeypatch.setattr(reflect, "run_error_analyst_minibatch", analyst("failure"))
    monkeypatch.setattr(reflect, "run_success_analyst_minibatch", analyst("success"))
    reflect.run_minibatch_reflect(
        results=results,
        skill_content=frozen_skill,
        prediction_dir=str(tmp_path / "predictions"),
        patches_dir=str(tmp_path / "patches"),
        workers=1,
        failure_only=False,
        minibatch_size=5,
        edit_budget=1,
        random_seed=42,
        skill_aware_reflection=False,
    )
    assert len(seen) == math.ceil(failures / 5) + math.ceil((20 - failures) / 5)
    assert all(skill == frozen_skill and 1 <= len(batch) <= 5 for _, skill, batch in seen)
    assert Counter(row["id"] for _, _, batch in seen for row in batch) == Counter(
        row["id"] for row in results
    )
    for kind, _, batch in seen:
        assert all(bool(row["hard"]) == (kind == "success") for row in batch)
