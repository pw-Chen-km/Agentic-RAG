from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentic_rag.skillopt.runtime_checkpoint import read_checkpoint, use_atomic_skillopt_checkpoints


def native_stub():
    def untouched(*args):
        raise AssertionError("Unpatched native write")
    return SimpleNamespace(_format_step_buffer=lambda rows: json.dumps(rows), **{name: untouched for name in (
        "_load_history", "_save_history", "_load_runtime_state", "_save_runtime_state", "_save_skill")})


def state(step, score=0):
    return dict(last_completed_step=step, current_score=score, best_score=score, best_step=step)


def test_zero_score_and_restoration(tmp_path):
    native = native_stub()
    original = native._load_history
    with use_atomic_skillopt_checkpoints(tmp_path, native=native):
        native._save_skill(str(tmp_path), 0, "original")
        native._save_runtime_state(str(tmp_path), state(0))
        resumed = native._load_runtime_state(str(tmp_path))
        assert float(resumed.get("current_score", -1) or -1) == 0
        assert float(resumed.get("best_score", 7) or 7) == 0
        assert native._load_history(str(tmp_path)) == []
    assert native._load_history is original


def test_resume_uses_committed_skill_history_not_partial_native_files(tmp_path):
    native = native_stub()
    with use_atomic_skillopt_checkpoints(tmp_path, native=native):
        native._save_skill(str(tmp_path), 0, "initial")
        native._save_runtime_state(str(tmp_path), state(0, .5))
        native._save_skill(str(tmp_path), 1, "new")
        native._save_history(str(tmp_path), [{"step": 1}])
        native._save_runtime_state(str(tmp_path), state(1, .7))
        # Simulate death after next update's convenience files, before commit.
        (tmp_path / "runtime_state.json").write_text("broken json")
        (tmp_path / "best_skill.md").write_text("uncommitted")
        native._save_history(str(tmp_path), [{"step": 1}, {"step": 2}])
        resumed = native._load_runtime_state(str(tmp_path))
        from pathlib import Path
        assert resumed["last_completed_step"] == 1
        assert Path(resumed["best_skill_path"]).read_text() == "new"
        assert native._load_history(str(tmp_path)) == [{"step": 1}]
        assert json.loads((tmp_path / "steps/step_0001/step_record.json").read_text()) == {"step": 1}


def test_partial_history_cannot_commit_or_move_backwards(tmp_path):
    native = native_stub()
    with use_atomic_skillopt_checkpoints(tmp_path, native=native):
        native._save_skill(str(tmp_path), 0, "initial")
        native._save_runtime_state(str(tmp_path), state(0))
        with pytest.raises(ValueError, match="partially written"):
            native._save_runtime_state(str(tmp_path), state(1))
        native._save_skill(str(tmp_path), 1, "new")
        native._save_history(str(tmp_path), [{"step": 1}])
        native._save_runtime_state(str(tmp_path), state(1))
        with pytest.raises(ValueError, match="backwards"):
            native._save_runtime_state(str(tmp_path), state(0))


def test_corrupt_checkpoint_never_falls_back_to_initial(tmp_path):
    native = native_stub()
    with use_atomic_skillopt_checkpoints(tmp_path, native=native):
        native._save_skill(str(tmp_path), 0, "original")
        native._save_runtime_state(str(tmp_path), state(0))
        path = tmp_path / "durable_checkpoint.json"
        data = json.loads(path.read_text())
        data["payload"]["runtime_state"]["last_completed_step"] = 9
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="hash mismatch"):
            native._load_runtime_state(str(tmp_path))


def test_old_state_and_foreign_root_rejected_and_context_restored(tmp_path):
    native = native_stub()
    original = native._load_runtime_state
    with pytest.raises(ValueError, match="atomic checkpoint"):
        with use_atomic_skillopt_checkpoints(tmp_path, native=native):
            (tmp_path / "runtime_state.json").write_text("{}")
            native._load_runtime_state(str(tmp_path))
    assert native._load_runtime_state is original
    with use_atomic_skillopt_checkpoints(tmp_path, native=native):
        with pytest.raises(ValueError, match="different training"):
            native._load_history(str(tmp_path / "other"))


def test_resume_restores_exact_committed_prior_reflection_context(tmp_path):
    native = native_stub()
    with use_atomic_skillopt_checkpoints(tmp_path, native=native):
        native._save_skill(str(tmp_path), 0, "initial")
        native._save_runtime_state(str(tmp_path), state(0, .5))
        buffer = []
        assert native._format_step_buffer(buffer) == "[]"
        buffer.append({"step": 1, "accepted": False, "rejected_edits": ["test patch"]})
        native._save_skill(str(tmp_path), 1, "initial")
        native._save_history(str(tmp_path), [{"step": 1}])
        native._save_runtime_state(str(tmp_path), state(1, .5))
        # An uncommitted later append must not affect the saved context.
        buffer.append({"step": 2, "uncommitted": True})
    with use_atomic_skillopt_checkpoints(tmp_path, native=native):
        native._load_runtime_state(str(tmp_path))
        new_buffer = []
        text = native._format_step_buffer(new_buffer)
        assert json.loads(text) == [{"step": 1, "accepted": False, "rejected_edits": ["test patch"]}]
        assert new_buffer == json.loads(text)
        # The buffer is restored only once, then used normally by the trainer.
        native._format_step_buffer(new_buffer)
        assert len(new_buffer) == 1
