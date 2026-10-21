"""Canonical A-RAG action-space and Phase 1 optimization objective.

The action menu is part of the experiment contract.  Keeping it in one small
module prevents the optimizer prompt, trajectory summaries, and runtime
preflight from silently disagreeing about which actions the agent can choose.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence


ACTION_SPACE_VERSION = "arag-action-space-v2-linear-hop"
TERMINAL_ACTION = "stop_and_answer"
RETRIEVAL_ACTIONS = (
    "semantic_search",
    "entity_hop",       # Direct-BM25 question/visible-entity hop.
    "linear_hop",       # LinearRAG-style corpus graph hop.
    "read_chunk",
)
OPTIONAL_ACTIONS = ("keyword_search",)
DEFAULT_ACTION_SPACE = RETRIEVAL_ACTIONS + (TERMINAL_ACTION,)


def configured_actions(config: Mapping | None = None) -> tuple[str, ...]:
    """Return the public action menu for a runtime configuration.

    The current no-keyword + Direct-BM25 + LinearHop arm has four retrieval
    actions.  ``keyword_search`` is retained only when explicitly enabled so
    old A-RAG controls remain representable without leaking into the new arm.
    """

    config = config or {}
    actions = []
    if (config.get("keyword_search") or {}).get("enabled", False):
        actions.append("keyword_search")
    actions.append("semantic_search")
    if (config.get("entity_hop") or {}).get("enabled", True):
        actions.append("entity_hop")
    if (config.get("linear_hop") or {}).get("enabled", False):
        actions.append("linear_hop")
    actions.append("read_chunk")
    actions.append(TERMINAL_ACTION)
    return tuple(dict.fromkeys(actions))


def normalize_action_name(name) -> str:
    """Normalize runtime/tool labels to the contract vocabulary."""

    text = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "finish": TERMINAL_ACTION,
        "answer": TERMINAL_ACTION,
        "stop": TERMINAL_ACTION,
        "stop_and_answer": TERMINAL_ACTION,
        "entityhop": "entity_hop",
        "linearhop": "linear_hop",
        "read": "read_chunk",
    }
    return aliases.get(text, text)


def action_space_contract(actions: Sequence[str] | None = None) -> dict:
    """Serialize the immutable menu and objective shown to Phase 1."""

    actions = tuple(actions or DEFAULT_ACTION_SPACE)
    normalized = tuple(dict.fromkeys(normalize_action_name(a) for a in actions))
    unknown = set(normalized) - set(RETRIEVAL_ACTIONS) - set(OPTIONAL_ACTIONS) - {TERMINAL_ACTION}
    if unknown:
        raise ValueError(f"unknown action-space members: {sorted(unknown)}")
    if TERMINAL_ACTION not in normalized:
        raise ValueError("action space must include stop_and_answer")
    return {
        "version": ACTION_SPACE_VERSION,
        "actions": list(normalized),
        "retrieval_actions": [a for a in normalized if a != TERMINAL_ACTION],
        "terminal_action": TERMINAL_ACTION,
        "objective": {
            "maximize": ["answer_accuracy", "useful_new_evidence"],
            "minimize": ["policy_calls", "tool_calls", "retrieved_tokens"],
            "action_usage": "report every enabled action separately; zero is a measured value",
            "unknown_policy": "runtime/Judge failure is unknown, never false or zero",
            "selection_scope": "Phase 1 train diagnosis; candidate acceptance uses validation only",
            "test_isolation": "validation/test trajectories never enter optimization or gate decisions",
        },
    }


def action_usage(raw: Mapping) -> dict[str, int]:
    """Count each action in one saved trajectory, including terminal answer."""

    counts = Counter()
    for decision in raw.get("trajectory", []) or []:
        if not isinstance(decision, Mapping):
            continue
        calls = decision.get("calls") or decision.get("tool_calls") or []
        if isinstance(calls, Mapping):
            calls = [calls]
        if not calls and (decision.get("tool_name") or decision.get("tool") or decision.get("action")):
            calls = [decision]
        for call in calls:
            if isinstance(call, Mapping):
                name = call.get("tool") or call.get("tool_name") or call.get("action") or call.get("name")
            else:
                name = call
            counts[normalize_action_name(name)] += 1
        resolved = decision.get("resolved_decision")
        if isinstance(resolved, Mapping) and not calls:
            counts[normalize_action_name(resolved.get("action") or resolved.get("tool"))] += 1
        if not calls and (decision.get("response") is not None or decision.get("force_answer_instruction")):
            counts[TERMINAL_ACTION] += 1
    if raw.get("answer") is not None and not counts.get(TERMINAL_ACTION):
        # Legacy artifacts may store the answer outside trajectory.
        counts[TERMINAL_ACTION] += 1
    return dict(sorted(counts.items()))


def validate_action_usage(usage: Mapping, actions: Sequence[str]) -> None:
    """Reject silent action-menu drift in deterministic summaries."""

    allowed = set(actions)
    unknown = set(usage) - allowed
    if unknown:
        raise ValueError(f"trajectory contains actions outside the configured action space: {sorted(unknown)}")

