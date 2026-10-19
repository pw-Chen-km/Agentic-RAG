"""Reflection evidence, not executable Skill edits.

Singleton observations are legal here. Cross-question support is checked only
when the existing batch merger creates a candidate. Structural validation cannot
prove that a model's interpretation of a trajectory is correct.
"""

import copy
import json
from collections.abc import Mapping

MAX_DIAGNOSES = 2
VERSION = "diagnosis-then-merge-v2-classification-advisory"
STATE_ACTION_PROTOCOL = "state-action-diagnosis-to-edits-v6"
FORMAT_REPAIR_VERSION = "redundant-outer-case-id-v1"


def _check_fields(value, expected, path, label):
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label} at {path}: expected an object")
    extra, missing = sorted(set(value) - expected), sorted(expected - set(value))
    if extra or missing:
        hint = ("; case_id belongs only inside observations, not beside analysis"
                if "case_id" in extra and label == "state-action diagnosis fields" else "")
        raise ValueError(f"invalid {label} at {path}: unexpected fields {extra}; missing fields {missing}{hint}")


def normalize_redundant_case_id(call, case_ids):
    """Copy, never mutate, one provably redundant field; strict parsing follows.

    Only an otherwise four-field diagnosis with exactly one observation can be
    normalized. Its two IDs must be the same supplied string. Never move an ID,
    choose between conflicting IDs, drop other fields, or repair truncated JSON.
    The caller must reject generation-limit replies before calling this helper.
    """
    normalized = copy.deepcopy(call)
    fn = normalized.get("function", normalized)
    args = fn.get("arguments")
    was_string = isinstance(args, str)
    if was_string:
        args = json.loads(args)
    events = []
    if isinstance(args, Mapping) and isinstance(args.get("diagnoses"), list):
        for index, diagnosis in enumerate(args["diagnoses"]):
            if not isinstance(diagnosis, dict) or "case_id" not in diagnosis:
                continue
            path = f"diagnoses[{index}].case_id"
            expected = {"observations", "analysis", "possible_skill_improvement", "limitations", "case_id"}
            if set(diagnosis) != expected:
                continue  # Do not hide any other missing or unexpected field.
            observations = diagnosis["observations"]
            if (not isinstance(observations, list) or len(observations) != 1
                    or not isinstance(observations[0], dict)):
                raise ValueError(f"cannot normalize {path}: exactly one observation is required")
            outer, inner = diagnosis["case_id"], observations[0].get("case_id")
            if not isinstance(outer, str) or outer != inner:
                raise ValueError(f"cannot normalize {path}: it conflicts with observations[0].case_id")
            if outer not in case_ids:
                raise ValueError(f"cannot normalize {path}: the question ID was not supplied")
            del diagnosis["case_id"]
            events.append({"version": FORMAT_REPAIR_VERSION, "path": path,
                           "removed_value": outer,
                           "matched_path": f"diagnoses[{index}].observations[0].case_id",
                           "reason": "Exact redundant copy of the only observation's supplied question ID"})
    if was_string:
        fn["arguments"] = json.dumps(args, ensure_ascii=False)
    return normalized, events


def state_action_tools():
    """Case-grounded prose; rule selection belongs to Merge."""
    observation = {
        "type": "object", "additionalProperties": False,
        "required": ["case_id", "state", "action", "result"],
        "properties": {
            "case_id": {"type": "string", "description": "Exact ID of a supplied question that supports this finding."},
            "state": {"type": "string", "description": "What information the agent had and what was still missing before its choice, as shown by the record."},
            "action": {"type": "string", "description": "What the agent actually did, with relevant tool arguments. Identify run A and run B when comparing two runs."},
            "result": {"type": "string", "description": "What happened afterward, including relevant measured costs and answer evaluation."},
        },
    }
    diagnosis = {
        "type": "object", "additionalProperties": False,
        "required": ["observations", "analysis", "possible_skill_improvement", "limitations"],
        "properties": {
            "observations": {"type": "array", "items": observation, "minItems": 1, "maxItems": 5,
                             "description": "One observation per question directly supporting the finding; explain contrary examples in limitations."},
            "analysis": {"type": "string", "description": "Explain what the behavior shows, why it matters, and what remains an interpretation."},
            "possible_skill_improvement": {"type": "string", "description": "An idea for when and how written guidance might help. May be empty. Do not choose rule IDs or edit operations."},
            "limitations": {"type": "string", "description": "Relevant uncertainty, other causes, or contrary examples. May be empty."},
        },
    }
    return [{"type": "function", "function": {
        "name": "report_diagnoses",
        "description": "Save explanations of what the agent knew, what it did, what happened, and ideas for improving its written instructions. This tool does not change the rules.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": ["diagnoses", "reason"], "properties": {
                           "diagnoses": {"type": "array", "items": diagnosis, "maxItems": MAX_DIAGNOSES},
                           "reason": {"type": "string", "description": "Briefly explain why these findings were reported, or why the list is empty."},
                       }},
    }}]


def diagnostic_tools(protocol=VERSION):
    if protocol == STATE_ACTION_PROTOCOL:
        return state_action_tools()
    observation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["case_id", "decisions", "behavior", "outcome", "interpretation"],
        "properties": {
            "case_id": {"type": "string"},
            "decisions": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                          "description": "Copy exact references from the supplied list: Phase 1 uses 1, 2; Meta uses A1, B2. Do not add D or Decision prefixes."},
            "behavior": {"type": "string"},
            "outcome": {"type": "string"},
            "interpretation": {
                "type": "string",
                "enum": ["support", "neutral", "counterexample"],
            },
        },
    }
    diagnosis = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "pattern",
            "failure_layer",
            "observations",
            "policy_limitation",
            "hypothesis",
            "basis",
            "cost_tradeoff",
            "target_rule_ids",
        ],
        "properties": {
            "pattern": {"type": "string"},
            "failure_layer": {
                "type": "string",
                "enum": [
                    "policy",
                    "backend",
                    "interface",
                    "answer",
                    "evaluation",
                    "uncertain",
                ],
            },
            "observations": {
                "type": "array",
                "description": "Use one observation per case_id within a diagnosis. Combine multiple decisions of that case in its decisions list.",
                "items": observation,
                "minItems": 1,
                "maxItems": 5,
            },
            "policy_limitation": {"type": "string"},
            "hypothesis": {"type": "string"},
            "basis": {
                "type": "string",
                "enum": ["observed_pattern", "policy_hypothesis", "uncertain"],
            },
            "cost_tradeoff": {"type": "string"},
            "target_rule_ids": {"type": "array", "items": {"type": "string"}},
        },
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "report_diagnoses",
                "description": "Save up to two concise diagnoses for the batch merger; do not edit the Skill. An empty list is valid.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["diagnoses", "reason"],
                    "properties": {
                        "diagnoses": {
                            "type": "array",
                            "items": diagnosis,
                            "maxItems": MAX_DIAGNOSES,
                        },
                        "reason": {"type": "string"},
                    },
                },
            },
        }
    ]


def parse_diagnoses(call, case_ids, rule_ids, decision_refs=None, *, protocol=VERSION):
    """Check supplied IDs and real decision positions; no semantic auto-judge."""
    fn = call.get("function", call)
    if fn.get("name") != "report_diagnoses":
        raise ValueError("reflection must call report_diagnoses, not a Skill edit tool")
    args = fn.get("arguments")
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(args, Mapping) or set(args) != {"diagnoses", "reason"}:
        raise ValueError("diagnoses and reason are the only report arguments")
    if not isinstance(args["reason"], str) or not args["reason"].strip():
        raise ValueError("diagnostic reason is required")
    diagnoses = args["diagnoses"]
    if not isinstance(diagnoses, list) or len(diagnoses) > MAX_DIAGNOSES:
        raise ValueError("at most two diagnoses per reflection")
    if protocol == STATE_ACTION_PROTOCOL:
        for index, d in enumerate(diagnoses):
            path = f"diagnoses[{index}]"
            _check_fields(d, {
                "observations", "analysis", "possible_skill_improvement", "limitations"
            }, path, "state-action diagnosis fields")
            for key in ("analysis", "possible_skill_improvement", "limitations"):
                if not isinstance(d[key], str) or (key == "analysis" and not d[key].strip()):
                    raise ValueError(f"invalid diagnostic {key} at {path}.{key}")
            observations = d["observations"]
            if not isinstance(observations, list) or not 1 <= len(observations) <= 5:
                raise ValueError("diagnosis needs 1..5 case observations")
            seen = set()
            for observation_index, o in enumerate(observations):
                observation_path = f"{path}.observations[{observation_index}]"
                _check_fields(o, {"case_id", "state", "action", "result"},
                              observation_path, "state-action observation fields")
                qid = o["case_id"]
                if not isinstance(qid, str) or qid not in case_ids or qid in seen:
                    raise ValueError(f"diagnostic case must be supplied and unique at {observation_path}.case_id")
                seen.add(qid)
                if not all(isinstance(o[k], str) and o[k].strip() for k in ("state", "action", "result")):
                    raise ValueError(f"observed state, action and result required at {observation_path}")
        return dict(args)
    fields = {
        "pattern",
        "failure_layer",
        "observations",
        "policy_limitation",
        "hypothesis",
        "basis",
        "cost_tradeoff",
        "target_rule_ids",
    }
    for d in diagnoses:
        if not isinstance(d, dict) or set(d) != fields:
            raise ValueError("invalid diagnosis fields")
        for key in ("pattern", "policy_limitation", "hypothesis", "cost_tradeoff"):
            if not isinstance(d[key], str) or (
                key != "hypothesis" and not d[key].strip()
            ):
                raise ValueError(f"invalid diagnostic {key}")
        if d["failure_layer"] not in {
            "policy",
            "backend",
            "interface",
            "answer",
            "evaluation",
            "uncertain",
        }:
            raise ValueError("invalid failure layer")
        if d["basis"] not in {"observed_pattern", "policy_hypothesis", "uncertain"}:
            raise ValueError("invalid evidence basis")
        targets = d["target_rule_ids"]
        if (
            not isinstance(targets, list)
            or not all(isinstance(x, str) for x in targets)
            or not set(targets) <= set(rule_ids)
        ):
            raise ValueError("diagnostic targets must be current rule IDs")
        observations = d["observations"]
        if not isinstance(observations, list) or not 1 <= len(observations) <= 5:
            raise ValueError("diagnosis needs 1..5 case observations")
        seen = set()
        for o in observations:
            if not isinstance(o, dict) or set(o) != {
                "case_id",
                "decisions",
                "behavior",
                "outcome",
                "interpretation",
            }:
                raise ValueError("invalid case observation fields")
            qid = o["case_id"]
            if not isinstance(qid, str) or qid not in case_ids:
                raise ValueError("diagnostic case must be supplied and unique")
            if qid in seen:
                raise ValueError("use one observation per case_id within a diagnosis; combine its decisions into that one observation")
            seen.add(qid)
            if o["interpretation"] not in {"support", "neutral", "counterexample"}:
                raise ValueError("invalid observation interpretation")
            if not all(
                isinstance(o[k], str) and o[k].strip() for k in ("behavior", "outcome")
            ):
                raise ValueError("observed behavior and outcome required")
            refs = o["decisions"]
            if (
                not isinstance(refs, list)
                or not refs
                or not all(isinstance(x, str) for x in refs)
            ):
                raise ValueError(
                    "decision references must be a nonempty list of strings"
                )
            if decision_refs is not None and not set(refs) <= set(
                decision_refs.get(qid, [])
            ):
                raise ValueError("diagnosis refers to an unrecorded decision")
    return dict(args)


def support_ids(reports):
    """Documented SUPPORT, regardless of a model's layer/basis classification.

    The classification is an interpretation, not a provenance check or proof of
    whether textual guidance can affect the decision. Neutral/counterexample
    observations still cannot be used to pad an edit's support.
    """
    return {
        o["case_id"]
        for report in reports
        for d in report.get("diagnoses", [])
        for o in d["observations"]
        if ("analysis" in d or o.get("interpretation") == "support")
    }


def citation_advisories(reports):
    """Keep disputed classifications visible in audit; never veto an edit."""
    return [
        {
            "case_id": o["case_id"],
            "proposal_id": report.get("proposal_id"),
            "failure_layer": d["failure_layer"],
            "basis": d["basis"],
            "decisions": o["decisions"],
        }
        for report in reports
        for d in report.get("diagnoses", [])
        if "failure_layer" in d and (d["failure_layer"] != "policy" or d["basis"] == "uncertain")
        for o in d["observations"]
        if o["interpretation"] == "support"
    ]
