"""Exercise the installed dispatcher for the new 40 / 5 configuration offline."""
import importlib
import math
from collections import Counter

import pytest


@pytest.mark.parametrize("total", [40, 12, 2])
@pytest.mark.parametrize("requested_failures", [0, 1, 4, 5, 6, 20, 39, 40])
def test_reflection_five_preserves_every_episode_and_frozen_skill(
    total, requested_failures, tmp_path, monkeypatch,
):
    pytest.importorskip("skillopt")
    reflect = importlib.import_module("skillopt.gradient.reflect")
    failures = min(total, requested_failures)
    results = [{"id": str(i), "hard": int(i >= failures)} for i in range(total)]
    captured = []

    def record(skill, batch, prediction_dir, **kwargs):
        captured.append((skill, list(batch)))
        return None

    monkeypatch.setattr(reflect, "run_error_analyst_minibatch", record)
    monkeypatch.setattr(reflect, "run_success_analyst_minibatch", record)
    reflect.run_minibatch_reflect(
        results=results, skill_content="Frozen batch Skill",
        prediction_dir=str(tmp_path / "predictions"), patches_dir=str(tmp_path / "patches"),
        workers=1, failure_only=False, minibatch_size=5, edit_budget=1,
        random_seed=42, skill_aware_reflection=False,
    )
    assert len(captured) == math.ceil(failures / 5) + math.ceil((total - failures) / 5)
    assert all(skill == "Frozen batch Skill" and 1 <= len(batch) <= 5 for skill, batch in captured)
    assert Counter(row["id"] for _, batch in captured for row in batch) == Counter(row["id"] for row in results)
