"""Model-facing renderers for optimizer contexts.

The optimizer keeps canonical JSON checkpoints for replay and audit, but a
model does not need to parse a deeply nested implementation object.  These
renderers project the same deterministic data into short, labelled sections.
The projection is intentionally lossless with respect to the fields used by
the HRO contract; omitted provenance belongs in the sidecar checkpoint, not
in the model-facing prompt.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from .action_space import action_space_contract
from .compact_rendering import (
    DISPLAY_GUIDANCE,
    PLAIN_DISPLAY_GUIDANCE,
)
from .trajectory import render_view

# This is the model-facing context version.  The canonical JSON checkpoints
# remain unchanged and continue to carry the full replay/audit provenance.
# Every active optimizer path (HRO and Phase 1) identifies its model input with
# this same version.
CONTEXT_RENDERER_VERSION = "arag-optimizer-context-unified-v3"


_GROUP_MEANINGS = {
    "G00": "parent incorrect -> candidate incorrect",
    "G01": "parent incorrect -> candidate correct",
    "G10": "parent correct -> candidate incorrect",
    "G11": "parent correct -> candidate correct",
}


def _value(value, fallback="unknown"):
    return fallback if value is None else str(value)


def _number(value, digits=3):
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _rule_lines(rules: Sequence[Mapping] | None):
    lines = []
    for rule in rules or []:
        if not isinstance(rule, Mapping):
            continue
        rid = _value(rule.get("rule_id"), "unknown-rule")
        lines.append(f"- {rid}: {rule.get('title', '')}")
        if rule.get("when") is not None:
            lines.append(f"  when: {rule['when']}")
        actions = rule.get("action_sequence") or []
        if actions:
            lines.append("  actions: " + " -> ".join(str(action) for action in actions))
    return lines or ["- none supplied"]


def _action_lines(contract: Mapping | None):
    if isinstance(contract, Mapping):
        actions = contract.get("actions") or []
        terminal = contract.get("terminal_action", "stop_and_answer")
    else:
        actions = list(contract or ())
        terminal = "stop_and_answer"
    lines = []
    descriptions = {
        "keyword_search": "lexical search over the corpus",
        "semantic_search": "semantic search over the corpus",
        "entity_hop": "follow an entity-linked retrieval edge",
        "linear_hop": "follow a LinearRAG-style graph edge and return snippets",
        "read_chunk": "read selected chunks in full",
        "stop_and_answer": "stop retrieval and answer from observed evidence",
    }
    for action in actions:
        lines.append(f"- {action}: {descriptions.get(action, 'configured action')}")
    if terminal not in actions:
        lines.append(f"- {terminal}: terminal answer action")
    return lines or ["- action contract unavailable"]


def _action_contract(action_space):
    """Normalize tuple/list action spaces and already-built contracts."""
    if isinstance(action_space, Mapping):
        return action_space
    return action_space_contract(action_space or ())


def _skill_lines(skill):
    skill = skill or {}
    rules = skill.get("retrieval_policy") if isinstance(skill, Mapping) else None
    return _rule_lines(rules)


def _context_header(title, task, skill, action_space):
    return [
        f"[OPTIMIZER CONTEXT / {CONTEXT_RENDERER_VERSION}]",
        f"[TASK / {title}]",
        task,
        "[CURRENT SKILL]",
        *_skill_lines(skill),
        "[ACTION CONTRACT]",
        *_action_lines(_action_contract(action_space)),
    ]


def render_phase1_reflection_context(
    skill: Mapping,
    episodes: Sequence[Mapping],
    kind: str,
    *,
    action_space=None,
    plain_language: bool = False,
):
    """Render the Phase 1 reflection payload through the unified renderer.

    The detailed decision view is still the exact, lossless trajectory view;
    this function only gives it a stable sectioned envelope.  In particular,
    no evidence is selected or summarized here.
    """
    guidance = PLAIN_DISPLAY_GUIDANCE if plain_language else DISPLAY_GUIDANCE
    lines = _context_header(
        "PHASE 1 REFLECTION",
        (
            f"Analyze the observed {kind} cases and report reusable policy evidence. "
            "Do not edit the Skill; the batch merge handles edits."
        ),
        skill,
        action_space,
    )
    lines += ["[READING CONVENTION]", guidance, "[OBSERVED CASES]"]
    for episode in episodes:
        qid = _value(episode.get("question_id"))
        view = episode.get("view") or {}
        rendered = render_view(view, guidance=False, plain_language=plain_language)
        # render_view already carries the canonical Case ID.  Only add the
        # fallback marker for minimal/synthetic views that do not.
        if not any(line.startswith("Case ID:") for line in rendered.splitlines()):
            lines.append(f"Case ID: {qid}")
        if view.get("question") is not None and "Question:" not in rendered:
            lines.append("Question: " + str(view["question"]))
        lines.append(rendered)
        lines.append("")
    lines += [
        "[OUTPUT CONTRACT]",
        "Report observations only with the configured diagnosis tool. "
        "Do not emit Skill edits, outcome decisions, or unsupported gold/reference claims.",
    ]
    return "\n".join(lines).strip() + "\n"


def _render_diagnosis(diagnosis, indent="  "):
    if not isinstance(diagnosis, Mapping):
        return [indent + _value(diagnosis)]
    lines = []
    for key, value in diagnosis.items():
        label = str(key).replace("_", " ").capitalize()
        if isinstance(value, (Mapping, list, tuple)):
            lines.append(f"{indent}{label}: {_json(value)}")
        else:
            lines.append(f"{indent}{label}: {_value(value)}")
    return lines or [indent + "(empty diagnosis)"]


def render_phase1_merge_context(
    skill: Mapping,
    proposals: Sequence[Mapping],
    supporting_case_ids: Sequence[str],
    *,
    action_space=None,
    state_action: bool = False,
):
    """Render Phase 1/Meta merge input with the same section contract.

    The final canonical JSON block is intentionally retained at the end for
    replay/debug consumers.  The preceding labelled sections are what a
    model should read first.
    """
    lines = _context_header(
        "PHASE 1 MERGE",
        "Combine the supplied diagnoses into bounded Skill edits. "
        "Only documented supporting cases may be cited; no new trajectory evidence is available here.",
        skill,
        action_space,
    )
    lines += [
        "[ELIGIBLE SUPPORT CASES]",
        "Eligible supporting_case_ids (documented SUPPORT; classifications are advisory):",
        ", ".join(str(qid) for qid in supporting_case_ids) or "none",
        "[DIAGNOSIS REPORTS]",
    ]
    for proposal in proposals:
        lines.append(
            f"Proposal {proposal.get('proposal_id', 'unknown')} "
            f"({proposal.get('analysis_kind', 'unknown')}; status={proposal.get('status', 'unknown')})"
        )
        ids = proposal.get("case_ids") or []
        lines.append("  Cases: " + ", ".join(str(qid) for qid in ids))
        if proposal.get("reason"):
            lines.append("  Reason: " + str(proposal["reason"]))
        diagnoses = proposal.get("diagnoses") or []
        if diagnoses:
            lines.append("  Diagnoses:")
            for diagnosis in diagnoses:
                lines.extend(_render_diagnosis(diagnosis, indent="    "))
        else:
            lines.append("  Diagnoses: none")
    lines += [
        "[OUTPUT CONTRACT]",
        "Use the native edit tools only. Preserve no_change when observations do not justify a reusable edit.",
        "[CANONICAL DIAGNOSES / REPLAY DATA]",
        "Batch diagnoses (all groups):",
        _json(list(proposals)),
    ]
    return "\n".join(lines).strip() + "\n"


def _case_lines(case: Mapping):
    question_id = _value(case.get("question_id"))
    parent = case.get("parent") or {}
    candidate = case.get("candidate") or {}
    evidence = case.get("evidence_diff") or {}
    cost = case.get("cost_diff") or {}
    lines = [f"Case {question_id}"]
    if case.get("question") is not None:
        lines.append(f"Question: {case['question']}")
    lines.append(
        "Parent actions: "
        + _json(parent.get("action_bundles") or [])
        + f"; answer status: {_value(parent.get('answer_correct'))}"
    )
    lines.append(
        "Candidate actions: "
        + _json(candidate.get("action_bundles") or [])
        + f"; answer status: {_value(candidate.get('answer_correct'))}"
    )
    lines.append(f"Outcome group: {_value(case.get('answer_transition'))} ({_GROUP_MEANINGS.get(case.get('answer_transition'), 'unknown group')})")
    lines.append(f"Action changed: {_value(case.get('action_changed'))}; first divergence turn: {_value(case.get('first_divergence_turn'))}")
    lines.append(
        "Evidence difference: "
        f"new visible={_json(evidence.get('added_visible_ids') or evidence.get('added_ids') or [])}, "
        f"removed visible={_json(evidence.get('removed_visible_ids') or evidence.get('removed_ids') or [])}, "
        f"visible-ID delta={_value(evidence.get('visible_id_delta', evidence.get('new_evidence_count')))}"
    )
    lines.append(
        "Evidence use: "
        f"parent full reads={_json(evidence.get('parent_full_read_ids') or [])}, "
        f"candidate full reads={_json(evidence.get('candidate_full_read_ids') or [])}, "
        f"candidate snippet-to-full promotions={_json(evidence.get('candidate_snippet_to_full_ids') or [])}, "
        f"full-read delta={_value(evidence.get('full_read_delta'))}"
    )
    lines.append(
        "Cost difference: "
        f"policy calls={_value(cost.get('policy_call_delta'))}, "
        f"tool calls={_value(cost.get('tool_call_delta'))}, "
        f"retrieved tokens={_value(cost.get('retrieved_token_delta'))}"
    )
    return lines


def _human_value(value):
    """Render nested diagnostic values as short prose instead of JSON."""
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            parts.append(f"{str(key).replace('_', ' ')}: {_human_value(item)}")
        return "; ".join(parts) or "none"
    if isinstance(value, (list, tuple, set)):
        return "; ".join(_human_value(item) for item in value) or "none"
    return str(value)


def _stage_b_reflection_lines(reflection: Mapping, index: int):
    """Explain one Stage A report for the Stage B proposer."""
    reflection = reflection if isinstance(reflection, Mapping) else {}
    group = reflection.get("group") or "unassigned group"
    status = reflection.get("status") or "unknown status"
    lines = [f"Reflection {index} ({group}; {status})"]
    case_ids = reflection.get("case_ids") or []
    lines.append(
        "  Cases considered: " + ", ".join(str(x) for x in case_ids)
        if case_ids
        else "  Cases considered: none"
    )
    if reflection.get("reason"):
        lines.append("  Report status: " + str(reflection["reason"]))
    diagnoses = reflection.get("diagnoses") or []
    if not diagnoses:
        lines.append("  Diagnosis: no reusable pattern was reported.")
        return lines
    for diagnosis_index, diagnosis in enumerate(diagnoses, 1):
        if not isinstance(diagnosis, Mapping):
            lines.append(f"  Diagnosis {diagnosis_index}: {_human_value(diagnosis)}")
            continue
        lines.append(f"  Diagnosis {diagnosis_index}:")
        observations = diagnosis.get("observations") or []
        if observations:
            for observation in observations:
                if isinstance(observation, Mapping):
                    case_id = observation.get("case_id", "case")
                    details = {
                        key: value
                        for key, value in observation.items()
                        if key != "case_id"
                    }
                    lines.append(f"    Observed in {case_id}: {_human_value(details)}")
                else:
                    lines.append("    Observed: " + _human_value(observation))
        for key, label in (
            ("analysis", "Interpretation"),
            ("observed_pattern", "Observed pattern"),
            ("possible_skill_improvement", "Possible improvement"),
            ("evidence_effect", "Evidence effect"),
            ("cost_effect", "Cost effect"),
            ("applicability_hint", "Applicability hint"),
            ("limitations", "Limitations"),
            ("risks", "Risks"),
        ):
            if diagnosis.get(key) not in (None, "", [], {}):
                lines.append(f"    {label}: {_human_value(diagnosis[key])}")
    return lines


def _stage_b_pair_lines(pair: Mapping, index: int):
    """Render one representative parent/candidate pair as evidence prose."""
    pair = pair if isinstance(pair, Mapping) else {}
    summary = pair.get("paired_summary") or pair
    qid = pair.get("question_id", summary.get("question_id", f"case-{index}"))
    lines = [f"Representative case {index}: {qid}"]
    if pair.get("question") is not None:
        lines.append("  Question: " + str(pair["question"]))
    group = pair.get("transition_group") or pair.get("group")
    if group:
        lines.append(f"  Outcome: {group} ({_GROUP_MEANINGS.get(group, 'unknown outcome')})")
    lines.append(
        "  Answer status: parent="
        + _value(summary.get("parent_answer_correct"))
        + ", candidate="
        + _value(summary.get("candidate_answer_correct"))
    )
    lines.append(
        "  Action changed: "
        + _value(summary.get("action_changed", summary.get("first_divergence") is not None))
    )
    divergence = summary.get("first_divergence")
    if isinstance(divergence, Mapping):
        lines.append(
            "  First action difference: turn "
            + _value(divergence.get("step"))
            + "; parent chose "
            + _human_value(divergence.get("parent_action"))
            + "; candidate chose "
            + _human_value(divergence.get("candidate_action"))
        )
    for label, key in (("Parent", "parent_action_bundles"), ("Candidate", "candidate_action_bundles")):
        bundles = summary.get(key) or []
        if bundles:
            rendered = " | ".join(
                f"turn {turn}: " + ", ".join(str(action) for action in bundle)
                for turn, bundle in enumerate(bundles, 1)
            )
            lines.append(f"  {label} action path: {rendered}")
    lines.append(
        "  Evidence use: parent full reads="
        + _human_value(summary.get("parent_full_read_ids"))
        + "; candidate full reads="
        + _human_value(summary.get("candidate_full_read_ids"))
        + "; full-read change="
        + _value(summary.get("full_read_delta"))
        + "; snippet-to-full change="
        + _value(summary.get("snippet_to_full_delta"))
    )
    lines.append(
        "  Cost change (candidate minus parent): policy calls="
        + _value(summary.get("policy_call_delta"))
        + "; tool calls="
        + _value(summary.get("tool_call_delta"))
        + "; retrieved tokens="
        + _value(summary.get("retrieved_token_delta"))
        + "; model tokens="
        + _value(summary.get("model_token_delta", summary.get("token_delta")))
    )
    return lines


def _stage_b_statistics_lines(statistics: Mapping, group_summaries: Mapping):
    """Explain train coverage and outcomes in human-readable terms."""
    statistics = statistics if isinstance(statistics, Mapping) else {}
    lines = [
        "Training comparison coverage:",
        f"  Paired training cases: {_value(statistics.get('paired_count'))}",
        f"  Informative cases sent to reflection/proposal: {_value(statistics.get('informative_pair_count'))}",
        f"  Unscored or unavailable cases: {_value(statistics.get('unscored_count'))}",
    ]
    outcomes = statistics.get("outcomes") or {}
    if outcomes:
        lines.append(
            "  Outcome counts: "
            + "; ".join(
                f"{str(key).replace('_', ' ')}={value}"
                for key, value in outcomes.items()
            )
        )
    skipped = statistics.get("skipped") or {}
    if skipped:
        lines.append(
            "  Excluded cases: "
            + "; ".join(
                f"{str(key).replace('_', ' ')}={value}"
                for key, value in skipped.items()
            )
        )
    for group in ("G00", "G01", "G10", "G11"):
        summary = (group_summaries or {}).get(group)
        if not summary:
            continue
        lines.append(
            f"  {group} ({_GROUP_MEANINGS[group]}): "
            f"{summary.get('count', 0)} cases; "
            f"action changed in {summary.get('action_changed_count', 0)}; "
            f"mean retrieved-token change {_number(summary.get('mean_retrieved_token_delta'))}; "
            f"mean full-read change {_number(summary.get('mean_full_read_delta'))}."
        )
    return lines


def render_stage_a_context(context: Mapping):
    """Render one bounded HRO Stage A group for model consumption."""
    group = context.get("group")
    task = context.get("meta_task") or {}
    constraints = context.get("constraints") or {}
    summary = context.get("group_summary") or {}
    lines = [
        f"[HRO CONTEXT / {CONTEXT_RENDERER_VERSION}]",
        "[TASK]",
        "This is HRO Stage A group diagnosis. Explain observed behavior only; do not edit the Skill.",
        f"Objective: {_value(task.get('objective'), 'diagnose the supplied group')}.",
        "Use only observed artifacts. Do not use gold/reference answers or compare another group.",
        "[OUTCOME GROUP]",
        f"{group}: {_GROUP_MEANINGS.get(group, 'unknown group')}",
        "[ACTION CONTRACT]",
        *_action_lines(context.get("action_space")),
        "[CHANGED RULES]",
        f"Changed rule IDs: {_json(context.get('changed_rule_ids') or [])}",
        "Parent rules:",
        *_rule_lines(context.get("parent_rules")),
        "Candidate rules:",
        *_rule_lines(context.get("candidate_rules")),
        "[GROUP SUMMARY]",
        f"{summary.get('count', 0)} paired cases are in this group.",
        f"{summary.get('action_changed_count', 0)}/{summary.get('count', 0)} cases changed action ({_number(summary.get('action_changed_rate', 0.0) * 100, 1)}%).",
        f"Mean first divergence turn: {_number(summary.get('mean_first_divergence_turn'))}.",
        f"Mean candidate new-evidence count: {_number(summary.get('mean_new_evidence_count'))}.",
        f"Mean new-evidence delta: {_number(summary.get('mean_new_evidence_delta'))}.",
        f"Mean full-read delta: {_number(summary.get('mean_full_read_delta'))}.",
        f"Mean snippet-to-full promotion delta: {_number(summary.get('mean_snippet_to_full_delta'))}.",
        f"Mean policy-call delta (candidate - parent): {_number(summary.get('mean_policy_call_delta'))}.",
        f"Mean retrieved-token delta (candidate - parent): {_number(summary.get('mean_retrieved_token_delta'))}.",
        f"Informative cases: {summary.get('informative_count', 0)}.",
        "[PAIRED CASES]",
    ]
    for case in context.get("paired_cases") or []:
        lines.extend(_case_lines(case))
        lines.append("")
    lines += [
        "[OUTPUT]",
        "Describe the shared observed pattern, evidence effect, cost effect, applicability hint, and risks.",
        "Do not choose KEEP, REVISE, or REJECT and do not propose a Skill edit.",
    ]
    return "\n".join(lines).strip() + "\n"


def render_stage_b_context(context: Mapping):
    """Render the single HRO Stage B proposer input."""
    lines = [
        f"[HRO CONTEXT / {CONTEXT_RENDERER_VERSION}]",
        "[TASK]",
        "This is HRO Stage B Skill proposal. Decide KEEP, REVISE, or REJECT for the current Skill.",
        "At most one logical edit is allowed. Use validation only for the gate; never use validation/test trajectories as evidence.",
        "[CURRENT SKILL]",
        *_rule_lines((context.get("parent_skill") or {}).get("retrieval_policy")),
        "[HISTORICAL CANDIDATE CHANGES]",
        f"Changed rule IDs: {_json(context.get('changed_rule_ids') or [])}",
        *_rule_lines((context.get("candidate_skill") or {}).get("retrieval_policy")),
        "[ACTION CONTRACT]",
        *_action_lines(context.get("action_space")),
        "[GROUP FINDINGS]",
    ]
    summaries = context.get("group_summaries") or {}
    for group in ("G00", "G01", "G10", "G11"):
        summary = summaries.get(group)
        if not summary:
            lines.append(f"{group}: no eligible cases")
            continue
        lines.append(
            f"{group} ({_GROUP_MEANINGS[group]}): {summary.get('count', 0)} cases; "
            f"action changed in {summary.get('action_changed_count', 0)}; "
            f"mean retrieved-token delta {_number(summary.get('mean_retrieved_token_delta'))}; "
            f"mean model-token delta {_number(summary.get('mean_model_token_delta'))}; "
            f"mean tool-call delta {_number(summary.get('mean_tool_call_delta'))}; "
            f"mean policy-call delta {_number(summary.get('mean_policy_call_delta'))}."
        )
    lines += ["[STAGE A REFLECTIONS]"]
    for index, reflection in enumerate(context.get("group_reflections") or [], 1):
        lines.extend(_stage_b_reflection_lines(reflection, index))
        lines.append("")
    lines += ["[REPRESENTATIVE CASES]"]
    for index, pair in enumerate(context.get("representative_pairs") or [], 1):
        lines.extend(_stage_b_pair_lines(pair, index))
        lines.append("")
    lines += ["[TRAINING COVERAGE]"]
    lines.extend(_stage_b_statistics_lines(
        context.get("train_statistics") or {},
        summaries,
    ))
    lines += [
        "[OUTPUT]",
        "Return a decision, a reason grounded in the supplied observations, supporting case IDs, and at most one logical edit.",
    ]
    return "\n".join(lines).strip() + "\n"

