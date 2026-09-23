"""Auditable option lifecycle events."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


EVENT_TYPES = {"start", "continue", "complete", "blocked", "interrupted", "fallback"}


@dataclass(frozen=True)
class OptionEvent:
    step: int
    event: str
    option_id: str
    subgoal: str | None = None
    primitive_action: Mapping[str, Any] | None = None
    outcome: Mapping[str, Any] | None = None
    option_status: str | None = None

    def __post_init__(self) -> None:
        if self.event not in EVENT_TYPES:
            raise ValueError(f"invalid option event: {self.event}")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not self.option_id:
            raise ValueError("option_id is required")

    def to_mapping(self) -> dict[str, Any]:
        value: dict[str, Any] = {"step": self.step, "event": self.event, "option_id": self.option_id}
        if self.subgoal is not None:
            value["subgoal"] = self.subgoal
        if self.primitive_action is not None:
            value["primitive_action"] = dict(self.primitive_action)
        if self.outcome is not None:
            value["outcome"] = dict(self.outcome)
        if self.option_status is not None:
            value["option_status"] = self.option_status
        return value


@dataclass
class OptionTrajectory:
    """Mutable builder used by a controller; exported records are copies."""

    events: list[OptionEvent] = field(default_factory=list)

    def record(self, event: OptionEvent) -> None:
        if self.events and event.step < self.events[-1].step:
            raise ValueError("option events must be chronological")
        self.events.append(event)

    def record_start(self, step: int, option_id: str, *, subgoal: str | None = None) -> None:
        self.record(OptionEvent(step, "start", option_id, subgoal=subgoal))

    def record_action(self, step: int, option_id: str, action: Mapping[str, Any], outcome: Mapping[str, Any], *, status: str = "CONTINUE") -> None:
        event = "fallback" if option_id == "FALLBACK" else "continue"
        self.record(OptionEvent(step, event, option_id, primitive_action=dict(action), outcome=dict(outcome), option_status=status))

    def record_end(self, step: int, option_id: str, status: str, *, outcome: Mapping[str, Any] | None = None) -> None:
        if status not in {"COMPLETE", "BLOCKED"}:
            raise ValueError("option end status must be COMPLETE or BLOCKED")
        event = "complete" if status == "COMPLETE" else "blocked"
        self.record(OptionEvent(step, event, option_id, outcome=dict(outcome or {}), option_status=status))

    def record_interrupted(self, step: int, option_id: str, *, reason: str) -> None:
        self.record(OptionEvent(step, "interrupted", option_id, outcome={"reason": reason}))

    def to_list(self) -> list[dict[str, Any]]:
        return [event.to_mapping() for event in self.events]

