"""Native-tool schemas used by the A-RAG SkillOpt optimizer."""

from __future__ import annotations

from typing import Any, Mapping


def tool_schemas(
    existing_rule_ids: list[str],
    supporting_case_ids: list[str] | None = None,
    min_support_cases: int = 2,
    *, plain_language: bool = False,
) -> list[dict[str, Any]]:
    """Build compact Ollama/OpenAI-compatible tool definitions."""

    if type(min_support_cases) is not int or min_support_cases < 1:
        raise ValueError("min_support_cases must be a positive integer")

    rule_id_enum = {"type": "string", "enum": existing_rule_ids} if existing_rule_ids else {"type": "string"}
    case_item = {"type": "string"}
    if supporting_case_ids is not None:
        case_item["enum"] = sorted(set(supporting_case_ids))
    case_schema = {
        "type": "array",
        "items": case_item,
        "minItems": min_support_cases,
        "uniqueItems": True,
    }
    rule = {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "when", "action_sequence"],
        "properties": {
            "title": {"type": "string"},
            "when": {"type": "string"},
            "action_sequence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
    }
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "add_rule",
                "description": "Add one new general retrieval rule.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["section", "rule_id", "rule", "reason", "supporting_case_ids"],
                    "properties": {
                        "section": {"type": "string", "enum": ["retrieval_policy"]},
                        "rule_id": {"type": "string", "pattern": "^R[0-9]+$"},
                        "rule": rule,
                        "reason": {"type": "string"},
                        "supporting_case_ids": case_schema,
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "replace_rule",
                "description": "Replace one existing retrieval rule.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["rule_id", "rule", "reason", "supporting_case_ids"],
                    "properties": {
                        "rule_id": rule_id_enum,
                        "rule": rule,
                        "reason": {"type": "string"},
                        "supporting_case_ids": case_schema,
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_rule",
                "description": "Delete one existing retrieval rule when evidence supports removal.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["rule_id", "reason", "supporting_case_ids"],
                    "properties": {
                        "rule_id": rule_id_enum,
                        "reason": {"type": "string"},
                        "supporting_case_ids": case_schema,
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "no_change",
                "description": "Choose no skill edit when evidence is insufficient or mixed.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["reason"],
                    "properties": {"reason": {"type": "string"}},
                },
            },
        },
    ]
    if plain_language:
        descriptions = {
            "add_rule": "Add one new instruction for how the agent finds or reads information.",
            "replace_rule": "Rewrite an existing instruction, identified by rule_id.",
            "delete_rule": "Remove an existing instruction when the reported observations justify removing it.",
            "no_change": "Explain why the available findings do not justify changing the instructions.",
        }
        for schema in schemas:
            function = schema["function"]
            function["description"] = descriptions[function["name"]]
            properties = function["parameters"]["properties"]
            properties["reason"]["description"] = (
                "Explain why the supplied findings do not justify a change."
                if function["name"] == "no_change" else
                "Explain the finding behind this choice and how the agent's future choices should change."
            )
            if "supporting_case_ids" in properties:
                properties["supporting_case_ids"]["description"] = "Exact supplied question IDs with observations that support the finding. This is not proof that the new instruction works."
        rule["properties"]["title"]["description"] = "A short name for the instruction."
        rule["properties"]["when"]["description"] = "The situation the agent can recognize when it should use this instruction."
        rule["properties"]["action_sequence"]["description"] = "Written instructions for choosing the next action, including if/then choices when useful."
    return schemas


def parse_tool_call(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Ollama and OpenAI tool-call envelopes."""

    function = tool_call.get("function", tool_call)
    name = function.get("name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        import json

        arguments = json.loads(arguments)
    if not isinstance(arguments, Mapping):
        raise ValueError("optimizer tool arguments must be an object")
    if name not in {"add_rule", "replace_rule", "delete_rule", "no_change"}:
        raise ValueError(f"unknown optimizer tool: {name}")
    operation = {
        "add_rule": "add",
        "replace_rule": "replace",
        "delete_rule": "delete",
        "no_change": "no_change",
    }[name]
    allowed = {"reason"} if operation == "no_change" else {"rule_id", "reason", "supporting_case_ids"}
    if operation in {"add", "replace"}:
        allowed |= {"rule"}
    if operation == "add":
        allowed |= {"section"}
    if set(arguments) - allowed:
        raise ValueError("unexpected optimizer arguments")
    if not isinstance(arguments.get("reason"), str) or not arguments["reason"].strip():
        raise ValueError("reason required")
    if operation != "no_change" and (not isinstance(arguments.get("supporting_case_ids"), list)
            or not all(isinstance(x, str) for x in arguments["supporting_case_ids"])):
        raise ValueError("supporting_case_ids must be a list of strings")
    normalized = {**dict(arguments), "operation": operation}
    if operation in {"add", "replace"}:
        rule = normalized.get("rule")
        if not isinstance(rule, Mapping):
            raise ValueError(f"{name} requires a rule object")
        # The outer id is authoritative.  An older model may still emit an
        # inner id; if present, it must agree, then it is normalized into the
        # rule object consumed by RuleStore.
        inner_id = rule.get("rule_id")
        outer_id = normalized.get("rule_id")
        if not outer_id or not isinstance(outer_id, str):
            raise ValueError(f"{name} requires an outer rule_id")
        if inner_id is not None and inner_id != outer_id:
            raise ValueError("outer and inner rule_id differ")
        rule = dict(rule)
        rule["rule_id"] = outer_id
        normalized["rule"] = rule
        if operation == "add":
            if normalized.get("section") != "retrieval_policy":
                raise ValueError("add_rule section must be retrieval_policy")
    return normalized
