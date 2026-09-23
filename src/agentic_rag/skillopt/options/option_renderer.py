"""Compact renderers for Agent and Optimizer inputs.

All functions are pure transformations.  They never infer evidence, append
future observations, or mutate an audit trajectory.  In particular, repeated
stable ids are retained while their long text is emitted only on first sight.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from .option_store import OptionStore


def render_selector_catalog(store: OptionStore) -> str:
    return store.markdown()


def render_selected_option(store: OptionStore, option_id: str) -> str:
    return store.markdown(selected_option=option_id)


def _stable_id(value: Mapping[str, Any]) -> str | None:
    for key in ("stable_id", "evidence_id", "ref", "id"):
        item = value.get(key)
        if isinstance(item, str) and item:
            return item
    return None


def _compact_step(step: Mapping[str, Any], seen_text: set[str]) -> dict[str, Any]:
    """Copy one audit step without repeating evidence text."""
    output: dict[str, Any] = {}
    for key in (
        "step", "option_id", "option_status", "assessment", "action", "outcome",
        "validation_status", "remaining_steps", "remaining_policy_attempts",
        "retrieval_tokens_remaining", "usage", "termination_reason",
    ):
        if key in step:
            output[key] = step[key]
    # New text is allowed only when this stable id has not been shown before.
    evidence = step.get("new_evidence")
    if evidence is None:
        evidence = step.get("visible_evidence")
    compact_evidence: list[Any] = []
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, Mapping):
                compact_evidence.append(item)
                continue
            copied = dict(item)
            stable_id = _stable_id(copied)
            text = copied.get("text")
            if stable_id and stable_id in seen_text:
                copied.pop("text", None)
                copied["text_repeated"] = True
            elif stable_id:
                seen_text.add(stable_id)
            compact_evidence.append(copied)
    if compact_evidence:
        output["new_evidence"] = compact_evidence
    for key in ("evidence_ids", "visible_evidence_ids", "read_refs", "eligible_refs", "available_actions", "available_options"):
        if key in step:
            output[key] = step[key]
    return output


def build_optimizer_trajectory(steps: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build an ordered, compact trajectory view from raw audit records."""
    seen_text: set[str] = set()
    return [_compact_step(step, seen_text) for step in steps]


def build_selection_view(episode: Mapping[str, Any]) -> dict[str, Any]:
    steps = episode.get("trajectory") or episode.get("steps") or []
    selected = []
    for step in steps:
        if isinstance(step, Mapping) and step.get("option_id"):
            selected.append({
                key: step[key]
                for key in ("step", "option_id", "option_status", "assessment", "outcome", "available_options", "remaining_steps")
                if key in step
            })
    return {"question_id": episode.get("episode_id", episode.get("id")), "steps": selected}


def build_policy_view(episode: Mapping[str, Any]) -> dict[str, Any]:
    steps = episode.get("trajectory") or episode.get("steps") or []
    return {
        "question_id": episode.get("episode_id", episode.get("id")),
        "steps": build_optimizer_trajectory(steps),
    }


def build_termination_view(episode: Mapping[str, Any]) -> dict[str, Any]:
    steps = episode.get("trajectory") or episode.get("steps") or []
    return {
        "question_id": episode.get("episode_id", episode.get("id")),
        "steps": [
            {
                key: step[key]
                for key in ("step", "option_id", "option_status", "assessment", "action", "outcome", "termination_reason", "remaining_steps")
                if key in step
            }
            for step in steps
            if isinstance(step, Mapping) and step.get("option_status") in {"COMPLETE", "BLOCKED"}
        ],
    }


def render_json_input(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)

