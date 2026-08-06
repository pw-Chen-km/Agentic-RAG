"""Deterministic, atomic persistence for one agent episode.

This module intentionally depends only on Python's standard library.  The
writer accepts dictionaries, dataclasses, and Pydantic models through a small
duck-typed serialization boundary so the controller's concrete model classes
do not leak into artifact persistence.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from agentic_rag.paths import portable_path_component

REDACTED_SECRET = "[REDACTED]"

_ARTIFACT_FILENAMES = frozenset(
    {
        "episode.json",
        "conversation.json",
        "target_system_prompt.txt",
        "target_user_prompt.txt",
        "skill.md",
        "effective_config.json",
    }
)
_SECRET_KEY_PARTS = frozenset(
    {
        "apikey",
        "authorization",
        "authtoken",
        "bearertoken",
        "clientsecret",
        "credential",
        "credentials",
        "password",
        "privatekey",
        "refreshtoken",
        "secret",
        "token",
    }
)


class ArtifactWriter:
    """Write an immutable run directory below ``runs_root``.

    A complete sibling temporary directory is built and fsynced before it is
    atomically renamed below ``runs_root``. Logical IDs that are not valid
    Windows filenames use a deterministic portable directory name. Repeating
    an identical write is idempotent. Reusing an episode ID for different
    content raises ``FileExistsError`` and leaves the first run untouched.
    """

    def __init__(self, runs_root: str | Path) -> None:
        self.runs_root = Path(runs_root)

    def write_episode(
        self,
        *,
        episode_id: str,
        episode: Any,
        target_system_prompt: str,
        target_user_prompt: str,
        skill_content: str,
        effective_config: Any,
        trajectory: Sequence[Any] | None = None,
    ) -> Path:
        """Persist all public artifacts for one episode and return its path."""

        _validate_episode_id(episode_id)
        _require_text("target_system_prompt", target_system_prompt)
        _require_text("target_user_prompt", target_user_prompt)
        _require_text("skill_content", skill_content)

        episode_value = _to_jsonable(episode)
        trajectory_value = (
            trajectory
            if trajectory is not None
            else _field(episode, "trajectory", _field(episode, "steps", ()))
        )
        conversation_value = _skillopt_conversation(trajectory_value)
        config_value = _to_jsonable(effective_config)

        payloads = {
            "episode.json": _json_bytes(episode_value),
            "conversation.json": _json_bytes(conversation_value),
            "target_system_prompt.txt": target_system_prompt.encode("utf-8"),
            "target_user_prompt.txt": target_user_prompt.encode("utf-8"),
            "skill.md": skill_content.encode("utf-8"),
            "effective_config.json": _json_bytes(config_value),
        }
        return self._publish(episode_id, payloads)

    def path_for_episode(self, episode_id: str) -> Path:
        """Return the cross-platform artifact path for a logical Episode ID."""

        _validate_episode_id(episode_id)
        return self.runs_root / portable_path_component(episode_id)

    def write(
        self,
        *,
        episode_id: str,
        episode: Any,
        target_system_prompt: str,
        target_user_prompt: str,
        skill_content: str,
        effective_config: Any,
        trajectory: Sequence[Any] | None = None,
    ) -> Path:
        """Alias for :meth:`write_episode` for concise harness integration."""

        return self.write_episode(
            episode_id=episode_id,
            episode=episode,
            target_system_prompt=target_system_prompt,
            target_user_prompt=target_user_prompt,
            skill_content=skill_content,
            effective_config=effective_config,
            trajectory=trajectory,
        )

    def _publish(self, episode_id: str, payloads: Mapping[str, bytes]) -> Path:
        if frozenset(payloads) != _ARTIFACT_FILENAMES:
            raise ValueError("artifact payload set does not match the public contract")

        self.runs_root.mkdir(parents=True, exist_ok=True)
        portable_id = portable_path_component(episode_id)
        destination = self.runs_root / portable_id
        if destination.exists():
            return _accept_identical_or_raise(destination, payloads)

        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{portable_id}.",
                suffix=".tmp",
                dir=self.runs_root,
            )
        )
        published = False
        try:
            for filename in sorted(payloads):
                _write_file(temporary / filename, payloads[filename])
            _fsync_directory(temporary)

            try:
                os.rename(temporary, destination)
                published = True
                _fsync_directory(self.runs_root)
                return destination
            except OSError:
                # Another process may have published this episode concurrently.
                if destination.exists():
                    return _accept_identical_or_raise(destination, payloads)
                raise
        finally:
            if not published and temporary.exists():
                shutil.rmtree(temporary)


def _skillopt_conversation(trajectory: Any) -> list[dict[str, Any]]:
    """Project controller trajectory records into SkillOpt step records."""

    if trajectory is None:
        return []
    if isinstance(trajectory, (str, bytes, bytearray, Mapping)):
        raise TypeError("trajectory must be a sequence of step records")
    if not isinstance(trajectory, Sequence):
        raise TypeError("trajectory must be a sequence of step records")

    conversation: list[dict[str, Any]] = []
    for position, raw_step in enumerate(trajectory, start=1):
        step = _to_jsonable(raw_step)
        if not isinstance(step, Mapping):
            raise TypeError("each trajectory step must serialize to an object")

        decision = step.get("decision")
        decision = decision if isinstance(decision, Mapping) else {}
        conversation.append(
            {
                "step": step.get("step", position),
                "action": decision.get("action"),
                "reasoning": decision.get("assessment"),
                "env_feedback": step.get("agent_visible_observation"),
            }
        )
    return conversation


def _to_jsonable(value: Any) -> Any:
    """Convert supported records to deterministic JSON values and redact secrets."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("artifact JSON cannot contain NaN or infinity")
        return value
    if isinstance(value, Enum):
        return _to_jsonable(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("binary values are not supported in JSON artifacts")

    # Pydantic SecretStr/SecretBytes and compatible secret wrappers.
    if callable(getattr(value, "get_secret_value", None)):
        return REDACTED_SECRET

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_jsonable(model_dump(mode="python", by_alias=True))

    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(
            {field.name: getattr(value, field.name) for field in fields(value)}
        )

    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _json_key(raw_key)
            if key in converted:
                raise ValueError(f"duplicate JSON object key after conversion: {key!r}")
            converted[key] = (
                REDACTED_SECRET
                if _is_secret_key(key)
                else _to_jsonable(raw_value)
            )
        return converted

    if isinstance(value, (set, frozenset)):
        converted_items = [_to_jsonable(item) for item in value]
        return sorted(converted_items, key=_canonical_sort_key)
    if isinstance(value, Sequence):
        return [_to_jsonable(item) for item in value]

    raise TypeError(
        "artifact values must be JSON-compatible, mappings, dataclasses, "
        f"or Pydantic models; received {type(value).__name__}"
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _json_key(key: Any) -> str:
    if isinstance(key, Enum):
        key = key.value
    if isinstance(key, (str, int, float, bool)) or key is None:
        return str(key)
    raise TypeError(f"unsupported JSON object key type: {type(key).__name__}")


def _is_secret_key(key: str) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", key.casefold())
    return any(
        compact == part or compact.endswith(part)
        for part in _SECRET_KEY_PARTS
    )


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_file(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    # Windows cannot open directories through os.open(), and directory fsync
    # is not part of its durability contract.  File fsync + same-directory
    # rename remains the strongest portable sequence available here.
    if sys.platform == "win32":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _accept_identical_or_raise(
    destination: Path,
    payloads: Mapping[str, bytes],
) -> Path:
    if destination.is_dir():
        existing_names = {path.name for path in destination.iterdir()}
        if existing_names == set(payloads) and all(
            (destination / filename).is_file()
            and (destination / filename).read_bytes() == content
            for filename, content in payloads.items()
        ):
            return destination
    raise FileExistsError(
        f"episode artifact path already exists with different content: {destination}"
    )


def _validate_episode_id(episode_id: str) -> None:
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    if (
        episode_id in {".", ".."}
        or "/" in episode_id
        or "\\" in episode_id
        or "\x00" in episode_id
    ):
        raise ValueError("episode_id must be a single safe path component")


def _require_text(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
