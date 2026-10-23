"""HRO v1 grouped reflection and single-proposer contracts.

This module is deliberately deterministic.  It prepares bounded Stage A
contexts from already-saved TRAIN artifacts and validates the model-facing
Stage A/Stage B envelopes; it never evaluates an answer or mutates a Skill.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from .action_space import DEFAULT_ACTION_SPACE, action_space_contract
from .paired_summary import HRO_VERSION


HRO_V1_VERSION = "hro-group-reflection-v2"
UNSCORED = "UNSCORED"
GROUPS = ("G00", "G01", "G10", "G11")
STAGE_A_FIELDS = (
    "observed_pattern",
    "evidence_effect",
    "cost_effect",
    "applicability_hint",
    "risks",
)


def transition_group(parent_correct, candidate_correct) -> str:
    """Classify a pair from backend evaluation labels only."""
    if type(parent_correct) is not bool or type(candidate_correct) is not bool:
        return UNSCORED
    return {
        (False, False): "G00",
        (False, True): "G01",
        (True, False): "G10",
        (True, True): "G11",
    }[(parent_correct, candidate_correct)]


def _number(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(values) / len(values) if values else None


def _pair_summary(pair):
    return pair.get("paired_summary", pair) if isinstance(pair, Mapping) else {}


def pair_group(pair) -> str:
    summary = _pair_summary(pair)
    return transition_group(
        summary.get("parent_answer_correct"),
        summary.get("candidate_answer_correct"),
    )


def group_summary(pairs: Sequence[Mapping], group: str) -> dict:
    """Calculate backend statistics for one outcome group."""
    if group not in GROUPS:
        raise ValueError(f"unknown HRO group: {group}")
    selected = [p for p in pairs if pair_group(p) == group]
    summaries = [_pair_summary(p) for p in selected]
    divergences = [
        s.get("first_divergence", {}).get("step")
        for s in summaries
        if isinstance(s.get("first_divergence"), Mapping)
    ]
    action_changed = [s for s in summaries if s.get("first_divergence") is not None]
    return {
        "group": group,
        "count": len(selected),
        "action_changed_count": len(action_changed),
        "action_changed_rate": (len(action_changed) / len(selected)) if selected else 0.0,
        "mean_first_divergence_turn": _number(divergences),
        "mean_new_evidence_count": _number(
            [s.get("candidate_new_evidence_count") for s in summaries]
        ),
        "mean_new_evidence_delta": _number(
            [s.get("new_evidence_delta") for s in summaries]
        ),
        "mean_full_read_delta": _number(
            [s.get("full_read_delta") for s in summaries]
        ),
        "mean_snippet_to_full_delta": _number(
            [s.get("snippet_to_full_delta") for s in summaries]
        ),
        "mean_policy_call_delta": _number(
            [s.get("policy_call_delta") for s in summaries]
        ),
        "mean_tool_call_delta": _number(
            [s.get("tool_call_delta") for s in summaries]
        ),
        "mean_retrieved_token_delta": _number(
            [s.get("retrieved_token_delta") for s in summaries]
        ),
        "mean_model_token_delta": _number(
            [s.get("model_token_delta", s.get("token_delta")) for s in summaries]
        ),
        "informative_count": sum(bool(s.get("informative")) for s in summaries),
    }


def _clean_rule(rule):
    if not isinstance(rule, Mapping):
        return rule
    return {k: rule[k] for k in ("rule_id", "title", "when", "action_sequence") if k in rule}


def _case(pair):
    """Project a comparison pair into the Stage A evidence schema.

    Gold answers, support maps, raw evidence text, prompts, and evaluator
    explanations are intentionally not copied into this context.
    """
    summary = _pair_summary(pair)
    group = pair_group(pair)
    if group == UNSCORED:
        raise ValueError("UNSCORED pairs are never sent to Stage A")
    parent_ids = set(summary.get("parent_new_evidence_ids") or [])
    candidate_ids = set(summary.get("candidate_new_evidence_ids") or [])
    parent_snippet_ids = set(summary.get("parent_snippet_ids") or [])
    candidate_snippet_ids = set(summary.get("candidate_snippet_ids") or [])
    parent_full_read_ids = set(summary.get("parent_full_read_ids") or [])
    candidate_full_read_ids = set(summary.get("candidate_full_read_ids") or [])
    parent_promotions = set(summary.get("parent_snippet_to_full_ids") or [])
    candidate_promotions = set(summary.get("candidate_snippet_to_full_ids") or [])
    return {
        "question_id": str(pair.get("question_id", summary.get("question_id"))),
        "question": pair.get("question"),
        "parent": {
            "action_bundles": summary.get("parent_action_bundles") or [],
            "predicted_answer": pair.get("parent_predicted_answer"),
            "answer_correct": summary.get("parent_answer_correct"),
        },
        "candidate": {
            "action_bundles": summary.get("candidate_action_bundles") or [],
            "predicted_answer": pair.get("candidate_predicted_answer"),
            "answer_correct": summary.get("candidate_answer_correct"),
        },
        "answer_transition": group,
        "action_changed": summary.get("first_divergence") is not None,
        "first_divergence_turn": (
            summary.get("first_divergence", {}).get("step")
            if isinstance(summary.get("first_divergence"), Mapping)
            else None
        ),
        "evidence_diff": {
            "status": "available" if parent_ids or candidate_ids else "unavailable",
            "added_visible_ids": sorted(candidate_ids - parent_ids),
            "removed_visible_ids": sorted(parent_ids - candidate_ids),
            "visible_id_delta": summary.get("new_visible_evidence_delta", summary.get("new_evidence_delta")),
            "parent_snippet_ids": sorted(parent_snippet_ids),
            "candidate_snippet_ids": sorted(candidate_snippet_ids),
            "parent_full_read_ids": sorted(parent_full_read_ids),
            "candidate_full_read_ids": sorted(candidate_full_read_ids),
            "parent_snippet_to_full_ids": sorted(parent_promotions),
            "candidate_snippet_to_full_ids": sorted(candidate_promotions),
            "full_read_delta": summary.get("full_read_delta"),
            "snippet_to_full_delta": summary.get("snippet_to_full_delta"),
        },
        "cost_diff": {
            "policy_call_delta": summary.get("policy_call_delta"),
            "tool_call_delta": summary.get("tool_call_delta"),
            "retrieved_token_delta": summary.get("retrieved_token_delta"),
            "model_token_delta": summary.get("model_token_delta", summary.get("token_delta")),
        },
    }


def build_stage_a_context(
    group: str,
    pairs: Sequence[Mapping],
    *,
    changed_rule_ids: Sequence[str] = (),
    parent_rules: Sequence[Mapping] = (),
    candidate_rules: Sequence[Mapping] = (),
    phase1_rationale: str | None = None,
    action_space: Sequence[str] | None = None,
    max_cases: int = 5,
) -> dict:
    """Build the four-part bounded Stage A diagnosis context."""
    if group not in GROUPS:
        raise ValueError(f"unknown HRO group: {group}")
    if len(pairs) > max_cases:
        raise ValueError("Stage A accepts at most five paired cases")
    if any(pair_group(pair) != group for pair in pairs):
        raise ValueError("Stage A cases must belong to exactly one G group")
    cases = [_case(pair) for pair in pairs]
    changed = {str(x) for x in changed_rule_ids}
    # Stage A only needs the rules that the historical Phase 1 proposal
    # actually changed.  Keeping the full Skill in the checkpoint is still
    # possible elsewhere; this model-facing envelope stays bounded.
    rules_a = [
        _clean_rule(r) for r in parent_rules
        if not changed or str(r.get("rule_id")) in changed
    ]
    rules_b = [
        _clean_rule(r) for r in candidate_rules
        if not changed or str(r.get("rule_id")) in changed
    ]
    return {
        "schema_version": HRO_V1_VERSION,
        "meta_task": {
            "stage": "group_reflection",
            "objective": f"Analyze candidate Skill behavior in {group}; diagnosis only",
            "output": "diagnosis_only",
        },
        "group": group,
        "constraints": {
            "do_not_write_skill": True,
            "do_not_use_gold_answer": True,
            "use_only_observed_artifacts": True,
            "do_not_compare_other_groups": True,
        },
        "action_space": action_space_contract(action_space or DEFAULT_ACTION_SPACE),
        "changed_rule_ids": [str(x) for x in changed_rule_ids],
        "parent_rules": rules_a,
        "candidate_rules": rules_b,
        "phase1_rationale": phase1_rationale,
        "group_summary": group_summary(pairs, group),
        "paired_cases": cases,
    }


def stage_a_output_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(STAGE_A_FIELDS),
        "properties": {key: {"type": "string"} for key in STAGE_A_FIELDS},
    }


_FORBIDDEN_STAGE_A = re.compile(
    r"\b(?:keep|revise|reject|add_rule|replace_rule|delete_rule|gold|reference_answer|answer_key)\b",
    re.IGNORECASE,
)


def validate_stage_a_output(value: Mapping) -> dict:
    """Reject executable edits, decisions, gold labels, and cross-group text."""
    if not isinstance(value, Mapping) or set(value) != set(STAGE_A_FIELDS):
        raise ValueError("Stage A must return exactly the five diagnosis fields")
    normalized = {}
    for key in STAGE_A_FIELDS:
        if not isinstance(value[key], str):
            raise ValueError(f"Stage A field {key} must be text")
        if _FORBIDDEN_STAGE_A.search(value[key]):
            raise ValueError(f"Stage A field {key} contains a forbidden edit/gold decision")
        normalized[key] = value[key].strip()
    return normalized


def build_stage_b_context(
    parent_skill: Mapping,
    candidate_skill: Mapping,
    changed_rule_ids: Sequence[str],
    group_summaries: Mapping[str, Mapping],
    reflections: Sequence[Mapping],
    representative_pairs: Sequence[Mapping],
    train_statistics: Mapping,
    *,
    action_space: Sequence[str] | None = None,
) -> dict:
    """Build the single Skill Proposer input after all Stage A groups."""
    if any(group not in GROUPS for group in group_summaries):
        raise ValueError("Stage B group summaries contain an unknown group")
    if len(representative_pairs) > 20:
        raise ValueError("Stage B representative pairs exceed the four-group budget")
    candidate_policy = list((candidate_skill or {}).get("retrieval_policy") or [])
    changed = {str(x) for x in changed_rule_ids}
    candidate_view = dict(candidate_skill)
    candidate_view["retrieval_policy"] = [
        rule for rule in candidate_policy
        if not changed or str(rule.get("rule_id")) in changed
    ]
    return {
        "schema_version": HRO_V1_VERSION,
        "meta_task": {"stage": "skill_proposer", "output": "KEEP_REVISE_REJECT"},
        "parent_skill": parent_skill,
        "candidate_skill": candidate_view,
        "changed_rule_ids": [str(x) for x in changed_rule_ids],
        "group_summaries": {k: group_summaries[k] for k in GROUPS if k in group_summaries},
        "group_reflections": list(reflections),
        "representative_pairs": list(representative_pairs),
        "train_statistics": dict(train_statistics),
        "action_space": action_space_contract(action_space or DEFAULT_ACTION_SPACE),
        "constraints": {
            "one_logical_edit_max": True,
            "validation_only_for_gate": True,
            "do_not_use_validation_or_test_trajectories": True,
        },
    }


def validate_stage_b_output(value: Mapping, *, case_ids: Sequence[str] = ()) -> dict:
    """Validate the one-proposer decision and its optional single edit."""
    if not isinstance(value, Mapping):
        raise ValueError("Stage B output must be an object")
    allowed = {"decision", "reason", "supporting_case_ids", "edit"}
    if set(value) - allowed or {"decision", "reason"} - set(value):
        raise ValueError("Stage B output has unexpected or missing fields")
    decision = value["decision"]
    if decision not in {"KEEP", "REVISE", "REJECT"}:
        raise ValueError("Stage B decision must be KEEP, REVISE, or REJECT")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Stage B reason is required")
    ids = value.get("supporting_case_ids", [])
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError("supporting_case_ids must be a list of strings")
    unknown = set(ids) - set(case_ids)
    if unknown:
        raise ValueError(f"Stage B cites unavailable cases: {sorted(unknown)}")
    edit = value.get("edit")
    if decision == "REVISE":
        if not isinstance(edit, Mapping):
            raise ValueError("REVISE requires exactly one logical edit")
        if set(edit) - {"operation", "rule_id", "rule", "reason"}:
            raise ValueError("Stage B edit has unexpected fields")
        if edit.get("operation") not in {"add", "replace", "delete"}:
            raise ValueError("Stage B edit operation is invalid")
    elif edit is not None:
        raise ValueError("KEEP/REJECT must not contain a Skill edit")
    return dict(value)


def encode_context(context: Mapping) -> str:
    return json.dumps(context, sort_keys=True, ensure_ascii=False)

