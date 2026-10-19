"""JSON-friendly candidate and trajectory archive records."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Mapping
import json
from pathlib import Path


@dataclass
class CandidateArchive:
    parent_skill_hash: str
    candidate_skill_hash: str
    source_stage: str
    edits: list[Mapping[str, Any]]
    gate_decision: str | None = None
    replay_question_ids: list[str] = field(default_factory=list)
    trajectories: list[Mapping[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonlArchive:
    """Append-only JSONL store for rollout/proposal audit records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
