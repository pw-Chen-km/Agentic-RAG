"""Route completed trajectories to the appropriate optimizer reflection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class RoutedCases:
    selection: tuple[dict[str, Any], ...] = ()
    policy: tuple[dict[str, Any], ...] = ()
    termination: tuple[dict[str, Any], ...] = ()

    def all(self) -> tuple[dict[str, Any], ...]:
        return self.selection + self.policy + self.termination


def route_episode(episode: Mapping[str, Any]) -> RoutedCases:
    """Classify only recorded events; no future state is inferred."""
    steps = episode.get("trajectory") or episode.get("steps") or []
    selection: list[dict[str, Any]] = []
    policy: list[dict[str, Any]] = []
    termination: list[dict[str, Any]] = []
    for raw in steps:
        if not isinstance(raw, Mapping):
            continue
        event = raw.get("event")
        status = raw.get("option_status")
        if event in {"start", "interrupted"} or raw.get("option_selection") is not None:
            selection.append(dict(raw))
        if raw.get("primitive_action") is not None or raw.get("action") is not None:
            policy.append(dict(raw))
        if event in {"complete", "blocked", "interrupted"} or status in {"COMPLETE", "BLOCKED"}:
            termination.append(dict(raw))
    return RoutedCases(tuple(selection), tuple(policy), tuple(termination))


def route_episodes(episodes: Iterable[Mapping[str, Any]]) -> RoutedCases:
    selection: list[dict[str, Any]] = []
    policy: list[dict[str, Any]] = []
    termination: list[dict[str, Any]] = []
    for episode in episodes:
        cases = route_episode(episode)
        selection.extend(cases.selection)
        policy.extend(cases.policy)
        termination.extend(cases.termination)
    return RoutedCases(tuple(selection), tuple(policy), tuple(termination))

