"""Native Ollama tool adapter used by the Workflow-Aware SkillOpt optimizer.

This module is intentionally separate from the Agent policy provider.  The
Agent consumes one structured ``PolicyDecision``; the optimizer consumes one
or more small edit tools.  Keeping the two adapters separate makes failures in
the optimizer observable without changing the Agent wire format.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .runner import write_json


_EDIT_TOOL_NAMES = {"add_rule", "replace_rule", "delete_rule", "no_change"}
_RULE_FIELDS = ("title", "when", "action_sequence", "stop_or_recovery", "exceptions")


def _value(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _response_dump(response: Any) -> Any:
    """Keep a serializable provider response for native-call audits."""
    if response is None:
        return None
    dumper = getattr(response, "model_dump", None)
    if callable(dumper):
        try:
            return dumper(mode="json")
        except Exception:
            pass
    if isinstance(response, Mapping):
        return dict(response)
    return repr(response)


def _usage(response: Any) -> dict[str, int]:
    """Extract Ollama usage without requiring a particular SDK response type."""
    def integer(name: str) -> int:
        try:
            return max(int(_value(response, name) or 0), 0)
        except (TypeError, ValueError):
            return 0

    prompt = integer("prompt_eval_count")
    completion = integer("eval_count")
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _tool_calls(response: Any) -> list[tuple[str, dict[str, Any]]]:
    """Extract all Ollama function calls, decoding stringified arguments."""
    message = _value(response, "message")
    calls = _value(message, "tool_calls") or []
    result: list[tuple[str, dict[str, Any]]] = []
    for call in calls:
        function = _value(call, "function")
        name = _value(function, "name")
        arguments = _value(function, "arguments")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("native tool call has no function name")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, Mapping):
            raise ValueError("native tool arguments must be an object")
        result.append((name.strip(), _decode_nested(dict(arguments))))
    if not result:
        raise ValueError("optimizer response did not contain a native tool call")
    return result


def _decode_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _decode_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_nested(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return _decode_nested(json.loads(stripped))
            except json.JSONDecodeError:
                return value
    return value


def _string(value: Any, field: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _rule_schema(*, include_id: bool = True) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "title": {"type": "string", "minLength": 1, "maxLength": 240},
        "when": {"type": "string", "minLength": 1, "maxLength": 1000},
        "action_sequence": {"type": "array", "minItems": 1, "maxItems": 6,
                             "items": {"type": "string", "minLength": 1, "maxLength": 500}},
        "stop_or_recovery": {"type": "string", "minLength": 1, "maxLength": 1000},
        "exceptions": {"type": "array", "maxItems": 6,
                        "items": {"type": "string", "minLength": 1, "maxLength": 500}},
    }
    required = list(_RULE_FIELDS)
    if include_id:
        properties["rule_id"] = {"type": "string", "pattern": "^[RA][0-9]+$"}
        required.insert(0, "rule_id")
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


def build_optimizer_tools(stage: str, rule_catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build small, stage-aware tool schemas for the optimizer.

    Existing rule IDs are enums for replacement/deletion.  New IDs cannot be
    enumerated because they are not known in advance, so ``add_rule`` uses a
    prefix pattern and RuleStore performs the final uniqueness check.
    """
    allowed = list(rule_catalog.get("editable_sections") or (
        ("answer_policy",) if stage == "answer" else ("retrieval_policy", "recovery_policy")
    ))
    rules = list(rule_catalog.get("rules") or [])
    ids = [str(rule.get("rule_id")) for rule in rules if rule.get("rule_id")]
    sections = {"type": "string", "enum": allowed}
    id_enum = {"type": "string", "enum": ids} if ids else {"type": "string", "enum": ["__none__"]}
    audit = {
        "reason": {"type": "string", "minLength": 1, "maxLength": 1000},
        "supporting_case_ids": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
    }
    add = {"type": "object", "properties": {
        "section": sections,
        "rule_id": {"type": "string", "pattern": "^[RA][0-9]+$"},
        "rule": _rule_schema(include_id=False),
        **audit,
    }, "required": ["section", "rule_id", "rule", "reason", "supporting_case_ids"],
              "additionalProperties": False}
    replace = {"type": "object", "properties": {
        "rule_id": id_enum,
        "rule": _rule_schema(include_id=False),
        **audit,
    }, "required": ["rule_id", "rule", "reason", "supporting_case_ids"],
                  "additionalProperties": False}
    delete = {"type": "object", "properties": {"rule_id": id_enum, **audit},
              "required": ["rule_id", "reason", "supporting_case_ids"],
              "additionalProperties": False}
    no_change = {"type": "object", "properties": {"reason": audit["reason"]},
                 "required": ["reason"], "additionalProperties": False}
    descriptions = {
        "add_rule": "Add one general reusable rule to an editable Skill section.",
        "replace_rule": "Replace one existing rule after finding repeated evidence.",
        "delete_rule": "Delete one existing redundant or harmful rule.",
        "no_change": "Do not modify the Skill when evidence is insufficient.",
    }
    schemas = {"add_rule": add, "replace_rule": replace,
               "delete_rule": delete, "no_change": no_change}
    return [{"type": "function", "function": {"name": name,
             "description": descriptions[name], "parameters": schemas[name]}}
            for name in ("add_rule", "replace_rule", "delete_rule", "no_change")]


def _edit_from_call(name: str, args: Mapping[str, Any], *, stage: str,
                    existing_ids: set[str]) -> dict[str, Any]:
    if name not in _EDIT_TOOL_NAMES:
        raise ValueError(f"unknown optimizer tool: {name}")
    reason = _string(args.get("reason"), "reason", required=(name != "no_change"))
    supporting = args.get("supporting_case_ids", [])
    if name != "no_change" and (not isinstance(supporting, list) or
                                 any(not isinstance(x, str) for x in supporting)):
        raise ValueError("supporting_case_ids must be a list of strings")
    if name == "no_change":
        return {"edits": [], "no_change": True,
                "reason": _string(args.get("reason"), "reason")}
    rule_id = args.get("rule_id")
    if name in {"replace_rule", "delete_rule"}:
        if not isinstance(rule_id, str) or rule_id not in existing_ids:
            raise ValueError(f"{name} requires an existing rule_id")
    rule = args.get("rule")
    if name in {"add_rule", "replace_rule"}:
        if not isinstance(rule, Mapping):
            raise ValueError(f"{name} requires a rule object")
        for field in _RULE_FIELDS:
            if field not in rule:
                raise ValueError(f"rule missing {field}")
        for field in ("title", "when", "stop_or_recovery"):
            _string(rule.get(field), f"rule.{field}")
        if (not isinstance(rule.get("action_sequence"), list) or
                not rule["action_sequence"] or
                any(not isinstance(item, str) or not item.strip() for item in rule["action_sequence"])):
            raise ValueError("rule.action_sequence must be a non-empty list of strings")
        if (not isinstance(rule.get("exceptions"), list) or
                any(not isinstance(item, str) or not item.strip() for item in rule["exceptions"])):
            raise ValueError("rule.exceptions must be a list of strings")
    operation = name.removesuffix("_rule")
    edit: dict[str, Any] = {"operation": operation, "reason": reason or "",
                            "supporting_case_ids": supporting}
    if rule_id is not None:
        edit["rule_id"] = rule_id
    if name == "add_rule":
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ValueError("add_rule requires a new rule_id")
        if rule_id in existing_ids:
            raise ValueError(f"add_rule uses an existing rule_id: {rule_id}")
        if not rule_id.startswith("R") and stage != "answer":
            raise ValueError("retrieval add_rule must use an R rule_id")
        if stage == "answer" and not rule_id.startswith("A"):
            raise ValueError("answer add_rule must use an A rule_id")
        section = args.get("section")
        allowed = {"answer_policy"} if stage == "answer" else {"retrieval_policy", "recovery_policy"}
        if section not in allowed:
            raise ValueError("add_rule targets a forbidden section")
        edit["section"] = section
    if name in {"add_rule", "replace_rule"}:
        edit["rule"] = dict(rule)
    return edit


def _normalize_calls(calls: Sequence[tuple[str, dict[str, Any]]], *, stage: str,
                     existing_ids: set[str]) -> dict[str, Any]:
    if len(calls) > 2:
        raise ValueError("optimizer returned more than two edit tool calls")
    if any(name == "no_change" for name, _ in calls):
        if len(calls) != 1:
            raise ValueError("no_change cannot be combined with edits")
        return _edit_from_call(calls[0][0], calls[0][1], stage=stage,
                               existing_ids=existing_ids)
    edits = [_edit_from_call(name, args, stage=stage, existing_ids=existing_ids)
             for name, args in calls]
    return {"edits": edits, "no_change": not edits,
            "reason": "native optimizer edits" if edits else "no_change"}


def native_optimizer_call(*, client: Any, model: str, messages: Sequence[Mapping[str, Any]],
                          stage: str, payload: Mapping[str, Any], output: Path,
                          think: Any = True, temperature: float = 0.0,
                          num_ctx: int = 32_768, max_attempts: int = 3) -> dict[str, Any]:
    """Call native optimizer tools, retrying invalid calls and returning no-op.

    The audit file always includes the raw tool arguments and validation error
    details.  After three failures, the caller receives an explicit no-op
    instead of an exception or an arbitrary first candidate.
    """
    catalog = payload.get("rule_catalog") if isinstance(payload, Mapping) else {}
    existing_ids = {str(rule.get("rule_id")) for rule in (catalog or {}).get("rules", [])
                    if rule.get("rule_id")}
    tools = build_optimizer_tools(stage, catalog or {})
    attempts: list[dict[str, Any]] = []
    attempt_messages = list(messages)
    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        response = None
        try:
            kwargs = {"model": model, "messages": attempt_messages, "stream": False,
                      "tools": tools, "options": {"temperature": temperature, "num_ctx": num_ctx}}
            if think is not None:
                kwargs["think"] = think
            response = client.chat(**kwargs)
            calls = _tool_calls(response)
            normalized = _normalize_calls(calls, stage=stage, existing_ids=existing_ids)
            attempts.append({"attempt": attempt, "status": "valid",
                             "tool_calls": [{"name": name, "arguments": args} for name, args in calls],
                             "provider_response": _response_dump(response),
                             "usage": _usage(response),
                             "seconds": time.monotonic() - started})
            audit = {"stage": stage, "native": True, "messages": attempt_messages,
                     "tools": tools, "attempts": attempts, "response": normalized}
            write_json(output, audit)
            return normalized
        except Exception as exc:
            detail = str(exc)
            raw_calls: list[dict[str, Any]] = []
            try:
                raw_calls = [{"name": name, "arguments": args} for name, args in _tool_calls(response)]  # type: ignore[name-defined]
            except Exception:
                pass
            attempts.append({"attempt": attempt, "status": "error", "error": detail,
                             "raw_tool_calls": raw_calls,
                             "provider_response": _response_dump(response),
                             "usage": _usage(response) if response is not None else None,
                             "seconds": time.monotonic() - started})
            write_json(output, {"stage": stage, "native": True, "messages": attempt_messages,
                                "tools": tools, "attempts": attempts})
            if attempt < max_attempts:
                attempt_messages = [
                    *attempt_messages,
                    {"role": "user", "content": (
                        "The previous native optimizer call failed validation: "
                        f"{detail}. Retry using exactly one valid edit tool, at most two "
                        "edit calls total, or no_change alone."
                    )},
                ]
    result = {"edits": [], "no_change": True,
              "reason": "native_optimizer_failed_after_retries"}
    write_json(output, {"stage": stage, "native": True, "messages": attempt_messages,
                        "tools": tools, "attempts": attempts, "response": result})
    return result
