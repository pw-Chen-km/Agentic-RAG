"""JSON Schemas for the two model calls in Options v1.

The schemas are provider-neutral dictionaries.  The Agent runtime can pass
them directly to Ollama ``format`` or use the same shape with another
structured-output provider.  References are supplied by the current state,
so stale or out-of-scope identifiers never become part of the policy call.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from .option_store import OptionSpec


def _option_object_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "option_id": {"type": "string", "pattern": "^(?:O[1-9][0-9]*_[A-Z0-9_]+|FALLBACK)$"},
            "name": {"type": "string", "minLength": 1},
            "goal": {"type": "string", "minLength": 1},
            "initiation": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
            "primitive_actions": {"type": "array", "items": {"type": "string", "enum": ["SEARCH", "EXPAND", "READ", "FINISH"]}, "minItems": 1},
            "policy": {"type": "object", "additionalProperties": {"type": "string", "minLength": 1}, "minProperties": 1},
            "termination": {"type": "object", "additionalProperties": {"type": "string", "minLength": 1}, "minProperties": 1},
            "interrupt_when": {"type": "string", "minLength": 1},
        },
        "required": ["option_id", "name", "goal", "initiation", "primitive_actions", "policy", "termination", "interrupt_when"],
        "additionalProperties": False,
    }


def optimizer_edit_schema(stage: str = "meta") -> dict[str, Any]:
    """Schema for the constrained Option SkillOpt response."""
    if stage not in {"selection", "policy", "termination", "meta"}:
        raise ValueError(f"unknown optimizer stage: {stage}")
    common = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["refine", "replace", "add_policy_case", "add_option", "delete_option"]},
            "option_id": {"type": "string"},
            "field": {"type": "string"},
            "new_value": {},
            "observation": {"type": "string"},
            "response": {"type": "string"},
            "option": _option_object_schema(),
            "reason": {"type": "string", "minLength": 1},
            "supporting_case_ids": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 2, "maxItems": 16},
        },
        "required": ["operation", "reason", "supporting_case_ids"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "edits": {"type": "array", "items": common, "maxItems": 2},
            "no_change": {"type": "boolean"},
            "reason": {"type": "string", "minLength": 1},
        },
        "required": ["edits", "no_change", "reason"],
        "additionalProperties": False,
    }


def option_selector_schema(option_ids: Iterable[str]) -> dict[str, Any]:
    values = list(dict.fromkeys(option_ids))
    if not values:
        raise ValueError("option selector needs at least one option")
    return {
        "type": "object",
        "properties": {"option_id": {"type": "string", "enum": values}},
        "required": ["option_id"],
        "additionalProperties": False,
    }


def _assessment_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "supported_facts": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "missing_information": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        },
        "required": ["supported_facts", "missing_information"],
        "additionalProperties": False,
    }


def _action_schema(action: str, *, references: Mapping[str, Iterable[str]] | None = None) -> dict[str, Any]:
    references = references or {}
    common: dict[str, Any] = {"type": "object", "properties": {"type": {"type": "string", "const": action}}, "required": ["type"], "additionalProperties": False}
    properties = common["properties"]
    if action == "SEARCH":
        properties.update({
            "method": {"type": "string", "enum": ["LEXICAL", "BM25", "DENSE"]},
            "target": {"type": "string", "enum": ["ENTITY", "SENTENCE", "CHUNK"]},
            "query": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "const": 5},
        })
        common["required"] += ["method", "target", "query", "top_k"]
    elif action == "EXPAND":
        props = {"kind": {"type": "string"}, "source_ref": {"type": "string"}}
        if references.get("expand_source_refs"):
            props["source_ref"] = {"type": "string", "enum": list(references["expand_source_refs"])}
        properties.update(props)
        common["required"] += ["kind", "source_ref"]
    elif action == "READ":
        props: dict[str, Any] = {"chunk_ref": {"type": "string"}}
        if references.get("read_refs"):
            props["chunk_ref"] = {"type": "string", "enum": list(references["read_refs"])}
        properties.update(props)
        common["required"] += ["chunk_ref"]
    elif action == "FINISH":
        props = {"evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1}, "answer": {"type": "string", "minLength": 1}}
        if references.get("finish_evidence_refs"):
            props["evidence_refs"] = {"type": "array", "items": {"type": "string", "enum": list(references["finish_evidence_refs"])}, "minItems": 1}
        properties.update(props)
        common["required"] += ["evidence_refs", "answer"]
    else:
        raise ValueError(f"unknown action: {action}")
    return common


def option_policy_schema(
    option: OptionSpec,
    *,
    references: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Build the policy output schema for one selected option."""
    actions = [_action_schema(action, references=references) for action in option.primitive_actions]
    assessment = _assessment_schema()
    # CONTINUE executes one primitive action.  COMPLETE/BLOCKED are explicit
    # hand-back decisions and therefore intentionally have no action field.
    continue_branch = {
        "type": "object",
        "properties": {"assessment": assessment, "option_status": {"const": "CONTINUE"}, "action": {"oneOf": actions}},
        "required": ["assessment", "option_status", "action"],
        "additionalProperties": False,
    }
    terminal_branch = {
        "type": "object",
        "properties": {"assessment": assessment, "option_status": {"type": "string", "enum": ["COMPLETE", "BLOCKED"]}},
        "required": ["assessment", "option_status"],
        "additionalProperties": False,
    }
    return {"oneOf": [continue_branch, terminal_branch]}
