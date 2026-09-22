"""State-conditioned native tool definitions and tool-call decoding."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from agentic_rag.agent.models import (
    Assessment,
    AvailableActionSpace,
    ExpandAction,
    ExpansionKind,
    FinishAction,
    PolicyDecision,
    SearchAction,
    SearchMethod,
    SearchTarget,
)
from agentic_rag.agent.policy import PolicyResponseError, PolicyStateError


def build_tool_definitions(
    action_space: AvailableActionSpace,
    *, require_evidence_assessment: bool = True,
) -> list[dict[str, Any]]:
    """Build the exact native tool registry for one policy snapshot."""

    tools: list[dict[str, Any]] = []
    for option in sorted(
        action_space.search_options,
        key=lambda item: (item.method.value, 0 if item.target is SearchTarget.CHUNK else 1),
    ):
        method = option.method.value.lower()
        target = option.target.value.lower()
        if option.method is SearchMethod.DENSE and option.target is SearchTarget.CHUNK:
            name = "find_passages"
            description = "Search the collection and return complete passages related to a query."
        elif option.method is SearchMethod.DENSE and option.target is SearchTarget.SENTENCE:
            name = "find_sentences"
            description = "Search the collection and return complete sentences related to a query."
        else:
            name = f"search_{method}_{target}"
            description = f"Search the collection and return complete {target.lower()} results related to a query."
        tools.append(_function(name, description, {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        }))

    for option in action_space.expand_options:
        if option.kind is ExpansionKind.ENTITY_MENTIONED_IN_CHUNK:
            name = "follow_entity_to_passages"
            description = "Return up to five complete passages that mention the displayed name. Select its entity reference from the visible name list. An optional query ranks these passages; null uses the original question."
        elif option.kind is ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE:
            name = "follow_entity_to_sentences"
            description = "Return up to five complete sentences that mention the displayed name. Select its entity reference from the visible name list. An optional query ranks these sentences; null uses the original question."
        else:
            name = f"follow_{option.kind.value.lower()}"
            description = "Follow the displayed structural reference and return the available connected units."
        properties: dict[str, Any] = {
            "entity_ref": {
                "type": "string",
                "enum": list(option.source_refs),
                "description": "A reference displayed with an entity name in the current observation.",
            }
        }
        if option.kind not in {
            ExpansionKind.ENTITY_MENTIONED_IN_CHUNK,
            ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
        }:
            properties = {
                "source_ref": {
                    "type": "string",
                    "enum": list(option.source_refs),
                    "description": "A reference displayed in the current observation.",
                }
            }
        if option.kind is not ExpansionKind.CHUNK_ADJACENT_CHUNK:
            properties["query"] = {"type": ["string", "null"], "minLength": 1}
        else:
            properties["direction"] = {"type": "string", "enum": list(item.value for item in option.directions)}
        required = ["entity_ref"] if "entity_ref" in properties else ["source_ref"]
        tools.append(_function(name, description, {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }))

    if action_space.read_refs:
        tools.append(_function(
            "read_passage",
            "Return the complete text of a visible passage reference.",
            {
                "type": "object",
                "properties": {"passage_ref": {"type": "string", "enum": list(action_space.read_refs)}},
                "required": ["passage_ref"],
                "additionalProperties": False,
            },
        ))

    if action_space.finish_available:
        evidence_refs = list(action_space.finish_evidence_refs)
        evidence_items: dict[str, Any] = (
            {"type": "string", "enum": evidence_refs}
            if evidence_refs
            else {"type": "string"}
        )
        tools.append(_function(
            "finish",
            "Return the answer supported by references shown in the current observation.",
            {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "minLength": 1},
                    "evidence_refs": {
                        "type": "array",
                        "items": evidence_items,
                        "minItems": 0,
                        "maxItems": min(20, len(evidence_refs)),
                    },
                },
                "required": ["answer", "evidence_refs"],
                "additionalProperties": False,
            },
        ))
    if not require_evidence_assessment:
        for tool in tools:
            parameters = tool["function"]["parameters"]
            parameters["properties"].pop("assessment")
            parameters["required"].remove("assessment")
    return tools


def tool_schema_sha256(tools: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(list(tools), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def decision_from_tool_call(tool_call: Mapping[str, Any], tools=None) -> PolicyDecision:
    """Convert one provider-native function call into the existing action model."""

    function = tool_call.get("function") or {}
    name = str(function.get("name") or "")
    raw_arguments = function.get("arguments", {})
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments, object_pairs_hook=_unique_keys)
        except json.JSONDecodeError as exc:
            raise PolicyResponseError("native tool arguments were not valid JSON") from exc
    else:
        arguments = raw_arguments
    if not isinstance(arguments, Mapping):
        raise PolicyResponseError("native tool arguments must be an object")
    args = dict(arguments)
    if tools is not None:
        definition = next((item["function"] for item in tools if item["function"]["name"] == name), None)
        if definition is None:
            raise PolicyStateError(f"unavailable native tool: {name}")
        _validate_arguments(args, definition["parameters"])

    try:
        raw_assessment = args.pop("assessment", None)
        assessment = Assessment.model_validate(raw_assessment) if raw_assessment is not None else None
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyResponseError(
            "native tool call must include a valid evidence assessment"
        ) from exc

    try:
        if name == "find_passages":
            action = SearchAction(query=args["query"], method=SearchMethod.DENSE, target=SearchTarget.CHUNK)
        elif name == "find_sentences":
            action = SearchAction(query=args["query"], method=SearchMethod.DENSE, target=SearchTarget.SENTENCE)
        elif name == "follow_entity_to_passages":
            action = ExpandAction(
                kind=ExpansionKind.ENTITY_MENTIONED_IN_CHUNK,
                source_ref=str(args["entity_ref"]),
                query=args.get("query"),
            )
        elif name == "follow_entity_to_sentences":
            action = ExpandAction(
                kind=ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
                source_ref=str(args["entity_ref"]),
                query=args.get("query"),
            )
        elif name == "read_passage":
            from agentic_rag.agent.models import ReadAction
            action = ReadAction(chunk_ref=str(args["passage_ref"]))
        elif name == "finish":
            action = FinishAction(answer=args["answer"], evidence_refs=args["evidence_refs"])
        else:
            raise PolicyResponseError(f"unknown native tool: {name}")
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyResponseError(f"invalid arguments for native tool {name!r}") from exc
    return PolicyDecision(assessment=assessment, action=action)


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise PolicyResponseError(f"duplicate argument key: {key}")
        value[key] = item
    return value


def _validate_arguments(value, schema, path="arguments"):
    """Validate the small JSON-schema subset emitted by this module."""
    types = schema.get("type")
    types = types if isinstance(types, list) else [types]
    matches = {"object": isinstance(value, dict), "array": isinstance(value, list),
               "string": isinstance(value, str), "null": value is None}
    if not any(matches.get(kind, False) for kind in types):
        raise PolicyResponseError(f"{path} has an invalid type")
    if "enum" in schema and value not in schema["enum"]:
        raise PolicyStateError(f"{path} is not a currently visible reference")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if set(schema.get("required", [])) - set(value):
            raise PolicyResponseError(f"{path} is missing a required argument")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise PolicyResponseError(f"{path} contains unexpected arguments")
        for key, child in value.items():
            _validate_arguments(child, properties[key], f"{path}.{key}")
    elif isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 10**9):
            raise PolicyResponseError(f"{path} has an invalid number of items")
        for item in value:
            _validate_arguments(item, schema["items"], path)
    elif isinstance(value, str) and len(value.strip()) < schema.get("minLength", 0):
        raise PolicyResponseError(f"{path} must not be blank")


def _function(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    parameters = {
        **parameters,
        "properties": {
            "assessment": {
                "type": "object",
                "description": (
                    "Before selecting the tool, summarize what the shown source text "
                    "already supports and what information is still needed. Use exactly "
                    "supported_facts and missing_information as keys, with lists of strings as values."
                ),
                "properties": {
                    "supported_facts": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 5,
                        "description": (
                            "Brief facts already supported by the source text shown to you. "
                            "Use an empty list if no relevant fact is supported yet."
                        ),
                    },
                    "missing_information": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 3,
                        "description": (
                            "Specific information still needed to answer the question. "
                            "Use an empty list if no information is missing. Keep unresolved gaps "
                            "when finishing with insufficient evidence or an exhausted budget."
                        ),
                    },
                },
                "required": ["supported_facts", "missing_information"],
                "additionalProperties": False,
            },
            **parameters["properties"],
        },
        "required": ["assessment", *parameters.get("required", [])],
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
            "strict": True,
        },
    }
