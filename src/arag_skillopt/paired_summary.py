"""Deterministic paired trajectory summaries for HRO Meta analysis.

HRO (Hindsight Reflection Optimization) is deliberately an offline layer.  It
does not add an Agent call or change the policy/retriever/state workflow: it
normalizes two already-recorded TRAIN episodes into a compact, auditable
comparison that can be supplied to the existing Meta diagnosis/merge stages.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from .observations import result_units
from .trajectory import decisions_from_raw
from .action_space import action_usage, normalize_action_name

HRO_VERSION = "hro-paired-trajectory-v1"


def _as_mapping(value):
    return value if isinstance(value, Mapping) else {}


def _json_scalar(value):
    """Return a deterministic, compact representation of an action field."""
    if isinstance(value, Mapping):
        return {
            str(k): _json_scalar(value[k])
            for k in sorted(value)
            if k not in {"observation", "tool_result", "result", "text"}
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_scalar(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_target(arguments, step):
    """Find substrate IDs without treating per-episode E/S/C labels as IDs."""
    arguments = _as_mapping(arguments)
    step = _as_mapping(step)
    nested_arguments = _as_mapping(arguments.get("arguments"))
    for source in (arguments, nested_arguments, step):
        for key in (
            "stable_substrate_id",
            "stable_substrate_ids",
            "substrate_id",
            "substrate_ids",
            "stable_node_id",
            "stable_node_ids",
            "node_id",
            "node_ids",
            "chunk_id",
            "chunk_ids",
            "novel_node_ids",
        ):
            value = source.get(key)
            if value not in (None, "", [], {}):
                if isinstance(value, (list, tuple, set)):
                    return ",".join(sorted(str(v) for v in value))
                return str(value)
    # Query/entity text is useful for deterministic comparison when no
    # substrate identifier was recorded, but observations/evidence are not.
    for source in (arguments, step):
        for key in ("entity", "entity_name", "query", "keywords", "target"):
            value = source.get(key)
            if value not in (None, "", [], {}):
                if isinstance(value, (list, tuple, set)):
                    return ",".join(sorted(str(v) for v in value))
                return str(value)
    return None


def _stable_action(call, step=None):
    """Normalize one resolved decision to a stable, human-readable action."""
    call = _as_mapping(call)
    step = _as_mapping(step)
    resolved = step.get("resolved_decision")
    if isinstance(resolved, Mapping):
        call = {**call, **resolved}
    elif isinstance(resolved, str) and resolved:
        return resolved
    tool = (
        call.get("tool")
        or call.get("tool_name")
        or call.get("action")
        or step.get("tool_name")
        or step.get("action")
    )
    if tool is None:
        tool = call.get("name") or step.get("name") or "UNKNOWN"
    tool_text = str(tool).strip()
    if normalize_action_name(tool_text) == "stop_and_answer":
        return "stop_and_answer"
    tool = tool_text.upper()
    target = _stable_target(call, step)
    return f"{tool}:{target}" if target is not None else tool


def _fallback_decisions(raw):
    """Use the canonical parser, with a conservative legacy fallback."""
    try:
        return decisions_from_raw(raw)
    except (KeyError, TypeError, ValueError):
        decisions = []
        for index, step in enumerate(raw.get("trajectory", []) or [], 1):
            if not isinstance(step, Mapping):
                continue
            decisions.append(
                {
                    "decision": index,
                    "calls": [
                        {
                            "tool": step.get("tool_name") or step.get("tool"),
                            "arguments": step.get("arguments", {}),
                            **(
                                step.get("resolved_decision")
                                if isinstance(step.get("resolved_decision"), Mapping)
                                else {}
                            ),
                        }
                    ],
                    "results": [{"step": step, "visible_at": None}],
                    "response": step.get("response"),
                }
            )
        return decisions


def action_sequence(raw):
    """Return action-only trajectory strings, including a terminal FINISH."""
    actions = []
    for decision in _fallback_decisions(raw):
        results = decision.get("results") or []
        calls = decision.get("calls") or []
        for index, call in enumerate(calls):
            step = results[index].get("step", {}) if index < len(results) else {}
            actions.append(_stable_action(call, step))
        if not calls and (
            decision.get("response") is not None
            or decision.get("force_answer_instruction")
        ):
            actions.append("FINISH")
    return actions


def action_bundles(raw):
    """Return one stable action bundle per policy turn.

    HRO compares the behavior selected by a policy call, rather than local
    E/S/C labels or result text.  Multiple tools selected in one decision stay
    together so ``first_divergence`` remains aligned with policy turns.
    """
    bundles = []
    for decision in _fallback_decisions(raw):
        calls = decision.get("calls") or []
        bundle = []
        results = decision.get("results") or []
        for index, call in enumerate(calls):
            step = results[index].get("step", {}) if index < len(results) else {}
            bundle.append(_stable_action(call, step))
        if bundle:
            bundles.append(bundle)
        elif decision.get("response") is not None or decision.get("force_answer_instruction"):
            bundles.append(["stop_and_answer"])
    return bundles


def _novel_ids_from_step(step):
    step = _as_mapping(step)
    ids = set()
    for key in (
        "novel_node_ids",
        "new_node_ids",
        "novel_evidence_ids",
        "new_evidence_ids",
        "novel_chunk_ids",
    ):
        value = step.get(key)
        if isinstance(value, (list, tuple, set)):
            ids.update(str(item) for item in value)
        elif value not in (None, ""):
            ids.add(str(value))
    return ids


def novel_evidence_ids(raw):
    """Collect unique newly visible evidence IDs from recorded decisions."""
    found = set()
    for decision in _fallback_decisions(raw):
        for result in decision.get("results") or []:
            step = _as_mapping(result.get("step"))
            found.update(_novel_ids_from_step(step))
            # Older artifacts do not expose novel_node_ids.  result_units()
            # gives the same conservative unique substrate count without
            # retaining evidence text in the summary.
            if not _novel_ids_from_step(step):
                try:
                    units, _ = result_units(
                        step.get("tool_name") or step.get("tool"),
                        step.get("tool_result") or step.get("result"),
                    )
                    for unit in units:
                        unit_id = (
                            unit.get("chunk_id")
                            or unit.get("node_id")
                            or unit.get("id")
                        )
                        if unit_id is not None:
                            found.add(str(unit_id))
                except (AttributeError, KeyError, TypeError, ValueError):
                    continue
    return found


def _correct(evaluation):
    evaluation = _as_mapping(evaluation)
    for key in ("hard", "correct"):
        value = evaluation.get(key)
        if type(value) is bool:
            return value
    return None


def _usage(raw):
    usage = _as_mapping(raw.get("usage"))
    calls = raw.get("policy_calls")
    if not isinstance(calls, (int, float)):
        calls = usage.get("policy_calls")
    if not isinstance(calls, (int, float)):
        calls = usage.get("model_calls")
    tokens = usage.get("total_tokens")
    if not isinstance(tokens, (int, float)):
        tokens = raw.get("total_tokens")
    return calls, tokens


def _first_divergence(parent_actions, candidate_actions):
    limit = max(len(parent_actions), len(candidate_actions))
    for index in range(limit):
        parent_action = parent_actions[index] if index < len(parent_actions) else None
        candidate_action = (
            candidate_actions[index] if index < len(candidate_actions) else None
        )
        if parent_action != candidate_action:
            return {
                "step": index + 1,
                "parent_action": parent_action,
                "candidate_action": candidate_action,
            }
    return None


def is_informative_pair(summary):
    """Whether a pair contains an observable action/outcome/cost difference."""
    return bool(
        summary.get("first_divergence") is not None
        or summary.get("parent_answer_correct")
        != summary.get("candidate_answer_correct")
        or summary.get("token_delta", 0) != 0
        or summary.get("policy_call_delta", 0) != 0
    )


def summarize_pair(
    question_id,
    parent_episode,
    candidate_episode,
    parent_evaluation=None,
    candidate_evaluation=None,
):
    """Create the HRO paired summary without retaining prompts or evidence text."""
    parent_episode = _as_mapping(parent_episode)
    candidate_episode = _as_mapping(candidate_episode)
    parent_actions = action_sequence(parent_episode)
    candidate_actions = action_sequence(candidate_episode)
    parent_ids = novel_evidence_ids(parent_episode)
    candidate_ids = novel_evidence_ids(candidate_episode)
    parent_calls, parent_tokens = _usage(parent_episode)
    candidate_calls, candidate_tokens = _usage(candidate_episode)
    parent_correct = _correct(parent_evaluation)
    candidate_correct = _correct(candidate_evaluation)
    first = _first_divergence(parent_actions, candidate_actions)
    summary = {
        "schema_version": HRO_VERSION,
        "question_id": str(question_id),
        "parent_answer_correct": parent_correct,
        "candidate_answer_correct": candidate_correct,
        "first_divergence": first,
        "parent_actions": parent_actions,
        "candidate_actions": candidate_actions,
        "parent_action_bundles": action_bundles(parent_episode),
        "candidate_action_bundles": action_bundles(candidate_episode),
        "parent_action_usage": action_usage(parent_episode),
        "candidate_action_usage": action_usage(candidate_episode),
        "parent_new_evidence_ids": sorted(parent_ids),
        "candidate_new_evidence_ids": sorted(candidate_ids),
        "parent_new_evidence_count": len(parent_ids),
        "candidate_new_evidence_count": len(candidate_ids),
        "new_evidence_delta": len(candidate_ids) - len(parent_ids),
        "parent_policy_calls": parent_calls,
        "candidate_policy_calls": candidate_calls,
        "policy_call_delta": (
            candidate_calls - parent_calls
            if isinstance(parent_calls, (int, float))
            and isinstance(candidate_calls, (int, float))
            else None
        ),
        "parent_total_tokens": parent_tokens,
        "candidate_total_tokens": candidate_tokens,
        "token_delta": (
            candidate_tokens - parent_tokens
            if isinstance(parent_tokens, (int, float))
            and isinstance(candidate_tokens, (int, float))
            else None
        ),
    }
    summary["informative"] = is_informative_pair(summary)
    summary["status"] = (
        "unknown"
        if parent_correct is None or candidate_correct is None
        else "informative"
        if summary["informative"]
        else "identical"
    )
    return summary


def compact_pair_record(comparison):
    """Remove rendered observations before writing comparison_pairs.json."""
    return {
        key: comparison.get(key)
        for key in (
            "question_id",
            "question",
            "parent_predicted_answer",
            "candidate_predicted_answer",
            "train_batch",
            "parent_skill_hash",
            "candidate_skill_hash",
            "transition_group",
            "group",
            "paired_summary",
            "eligible_for_reflection",
            "skip_reason",
            "outcomes",
        )
        if key in comparison
    }


def encode_summary(summary):
    """Stable JSON helper used by tests and audit tooling."""
    return json.dumps(summary, sort_keys=True, ensure_ascii=False)
