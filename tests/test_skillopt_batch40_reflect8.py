"""Test the installed reflection dispatcher without any model calls."""
import importlib
import math
from collections import Counter
import pytest


@pytest.mark.parametrize("failures", [0, 1, 7, 8, 9, 20, 31, 32, 39, 40])
def test_every_result_is_analyzed_once_under_same_skill(failures, tmp_path, monkeypatch):
    pytest.importorskip("skillopt")
    reflect = importlib.import_module("skillopt.gradient.reflect")
    results = [{"id": str(i), "hard": int(i >= failures)} for i in range(40)]
    captured = []
    def record(skill, batch, prediction_dir, **kwargs):
        captured.append((skill, list(batch)))
        return None
    monkeypatch.setattr(reflect, "run_error_analyst_minibatch", record)
    monkeypatch.setattr(reflect, "run_success_analyst_minibatch", record)
    reflect.run_minibatch_reflect(results=results, skill_content="Frozen batch Skill",
        prediction_dir=str(tmp_path/"predictions"), patches_dir=str(tmp_path/"patches"),
        workers=1, failure_only=False, minibatch_size=8, edit_budget=1,
        random_seed=42, skill_aware_reflection=False)
    assert len(captured) == math.ceil(failures/8)+math.ceil((40-failures)/8)
    assert all(skill == "Frozen batch Skill" and 1 <= len(batch) <= 8 for skill,batch in captured)
    assert Counter(r["id"] for _,batch in captured for r in batch) == Counter(r["id"] for r in results)
