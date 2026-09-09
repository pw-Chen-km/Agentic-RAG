"""Project-scoped, atomic update checkpoints for the pinned SkillOpt trainer.

The installed package is never edited. A checkpoint commits a completed update,
not an in-flight LLM request. Interrupted, uncommitted phases require an explicit
resume; completed rollout episodes are reused by the existing rollout adapter.
"""
from __future__ import annotations

import hashlib
import copy
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

VERSION = "skillopt-atomic-update-checkpoint-v1"


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()


def read_checkpoint(root: str | Path) -> dict[str, Any] | None:
    path = Path(root) / "durable_checkpoint.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("version") != VERSION:
        raise ValueError("Unsupported durable checkpoint")
    payload = value["payload"]
    if hashlib.sha256(_json(payload)).hexdigest() != value["payload_sha256"]:
        raise ValueError("Durable checkpoint hash mismatch")
    step = payload["runtime_state"]["last_completed_step"]
    history = payload["history"]
    if type(step) is not int or step < 0 or len(history) != step:
        raise ValueError("Durable checkpoint step/history mismatch")
    if [r.get("step") for r in history] != list(range(1, step + 1)):
        raise ValueError("Durable checkpoint history is not contiguous")
    if step and not isinstance(payload.get("step_buffer"), list):
        raise ValueError("Completed checkpoint lacks prior reflection context")
    for name in ("current_skill", "best_skill"):
        record = payload[name]
        if hashlib.sha256(record["text"].encode()).hexdigest() != record["sha256"]:
            raise ValueError("Durable checkpoint Skill hash mismatch")
    return value


class _PresentZero(float):
    """Preserve numeric 0 through upstream's ``value or fallback`` idiom."""
    def __bool__(self) -> bool:
        return True


@contextmanager
def use_atomic_skillopt_checkpoints(root: str | Path, *, native: Any = None) -> Iterator[None]:
    if native is None:
        import skillopt.engine.trainer as native
    root = Path(root).resolve()
    names = ("_load_history", "_save_history", "_load_runtime_state", "_save_runtime_state", "_save_skill", "_format_step_buffer")
    original = {name: getattr(native, name) for name in names}
    live_buffer: list | None = None
    pending_buffer: list | None = None

    def format_buffer(buffer: list) -> str:
        nonlocal live_buffer, pending_buffer
        if pending_buffer is not None:
            if buffer:
                raise ValueError("Refusing to merge unknown prior reflection context")
            buffer.extend(copy.deepcopy(pending_buffer))
            pending_buffer = None
        # The native trainer appends to this same list before committing the
        # update. Keep the actual list, not an earlier shallow snapshot.
        live_buffer = buffer
        return original["_format_step_buffer"](buffer)

    def check_root(value: str) -> None:
        if Path(value).resolve() != root:
            raise ValueError("Checkpoint belongs to a different training run")

    def load_history(value: str) -> list[dict]:
        check_root(value)
        checkpoint = read_checkpoint(root)
        if checkpoint:
            return checkpoint["payload"]["history"]
        path = root / "history.json"
        if path.exists() and json.loads(path.read_text()):
            raise ValueError("History exists without an atomic checkpoint; refusing unsafe resume")
        return []

    def save_history(value: str, history: Any) -> None:
        check_root(value)
        _atomic(root / "history.json", _json(history))

    def save_skill(value: str, step: int, text: str) -> None:
        check_root(value)
        _atomic(root / "skills" / f"skill_v{step:04d}.md", text.encode())

    def save_runtime(value: str, state: dict) -> None:
        check_root(value)
        step = state["last_completed_step"]
        previous = read_checkpoint(root)
        if previous and step < previous["payload"]["runtime_state"]["last_completed_step"]:
            raise ValueError("Refusing to move a completed update backwards")
        history_path = root / "history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        if type(step) is not int or step < 0 or [r.get("step") for r in history] != list(range(1, step + 1)):
            raise ValueError("Cannot commit a partially written update")
        def skill(number: int) -> dict[str, str]:
            text = (root / "skills" / f"skill_v{number:04d}.md").read_text()
            return {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}
        payload = {"runtime_state": dict(state), "history": history,
                   "current_skill": skill(step), "best_skill": skill(int(state["best_step"])),
                   "step_buffer": copy.deepcopy(live_buffer or [])}
        checkpoint = {"version": VERSION, "payload": payload,
                      "payload_sha256": hashlib.sha256(_json(payload)).hexdigest()}
        # This one replace is the commit point. Native convenience files can be
        # reconstructed from it even if the process dies immediately afterwards.
        _atomic(root / "durable_checkpoint.json", _json(checkpoint))
        _atomic(root / "runtime_state.json", _json(state))
        if step:
            _atomic(root / "steps" / f"step_{step:04d}" / "step_record.json", _json(history[-1]))

    def load_runtime(value: str) -> dict | None:
        nonlocal pending_buffer
        check_root(value)
        checkpoint = read_checkpoint(root)
        if checkpoint is None:
            if (root / "runtime_state.json").exists():
                raise ValueError("Runtime state has no atomic checkpoint; refusing unsafe resume")
            return None
        payload = checkpoint["payload"]
        pending_buffer = copy.deepcopy(payload.get("step_buffer", []))
        state = dict(payload["runtime_state"])
        resume_root = root / "checkpoint_skills" / checkpoint["payload_sha256"]
        for name in ("current", "best"):
            path = resume_root / f"{name}_skill.md"
            text = payload[f"{name}_skill"]["text"].encode()
            if path.exists() and path.read_bytes() != text:
                raise ValueError("Checkpoint Skill copy changed")
            if not path.exists():
                _atomic(path, text)
            state[f"{name}_skill_path"] = str(path)
        for key in ("current_score", "best_score"):
            if state.get(key) == 0:
                state[key] = _PresentZero(0)
        return state

    replacements = dict(_load_history=load_history, _save_history=save_history,
                        _load_runtime_state=load_runtime, _save_runtime_state=save_runtime,
                        _save_skill=save_skill, _format_step_buffer=format_buffer)
    try:
        for name, function in replacements.items():
            setattr(native, name, function)
        yield
    finally:
        for name, function in original.items():
            setattr(native, name, function)
