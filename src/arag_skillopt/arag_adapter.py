"""Normalize an A-RAG result into the SkillOpt audit shape.

The adapter is intentionally tolerant because official A-RAG versions have
used slightly different names for response and usage fields.  It never creates
E/S/C references or invents a result that is absent from the runtime audit.
"""

from __future__ import annotations

from typing import Any, Mapping


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def normalize_trajectory(raw: Mapping[str, Any], *, question_id: str | None = None, skill_hash: str | None = None) -> dict[str, Any]:
    """Return a stable, JSON-serializable A-RAG trajectory record."""

    # Official A-RAG BaseAgent stores executed calls under ``trajectory``;
    # retain the aliases used by older adapters, preferring the official key.
    actions = _first(raw, "trajectory", "tool_calls", "actions", "steps", default=[])
    if not isinstance(actions, list):
        actions = []
    normalized_actions = []
    read_ids: list[str] = []
    entity_hop_calls: list[dict[str, Any]] = []
    for index, action in enumerate(actions, start=1):
        if not isinstance(action, Mapping):
            normalized_actions.append({"step": index, "raw": action})
            continue
        name = _first(action, "name", "tool", "tool_name", "function", "action")
        args = _first(action, "arguments", "args", "parameters", default={})
        result = _first(action, "result", "tool_result", "observation", "output")
        if name == "read_chunk" and isinstance(args, Mapping):
            chunk_ids = args.get("chunk_ids")
            if isinstance(chunk_ids, list):
                read_ids.extend(str(chunk_id) for chunk_id in chunk_ids)
            elif args.get("chunk_id") is not None:
                read_ids.append(str(args["chunk_id"]))
        if name == "entity_hop":
            record = {
                "entity": args.get("entity") if isinstance(args, Mapping) else None,
                "top_k": args.get("top_k") if isinstance(args, Mapping) else None,
            }
            if isinstance(action.get("log"), Mapping):
                for key in (
                    "hop_backend",
                    "entity_candidates_internal",
                    "chunks_found",
                    "retrieved_tokens",
                    "status",
                ):
                    if key in action["log"]:
                        record[key] = action["log"][key]
            for key in (
                "hop_backend",
                "entity_candidates_internal",
                "candidate_chunk_count",
                "chunks_found",
                "retrieved_tokens",
                "status",
            ):
                if key in action:
                    record[key] = action[key]
            entity_hop_calls.append(record)
        normalized = {"step": index, "tool": name, "arguments": args, "result": result}
        if name == "entity_hop" and isinstance(action, Mapping):
            for key in (
                "hop_backend",
                "entity_candidates_internal",
                "candidate_chunk_count",
                "chunks_found",
                "retrieved_tokens",
                "status",
                "chunk_ids",
            ):
                if key in action:
                    normalized[key] = action[key]
        normalized_actions.append(normalized)
    usage = _first(raw, "usage", "token_usage", default={})
    if not isinstance(usage, Mapping):
        usage = {}
    answer = _first(raw, "answer", "final_answer", "response")
    return {
        "question_id": question_id if question_id is not None else _first(raw, "question_id", "id"),
        "skill_hash": skill_hash if skill_hash is not None else _first(raw, "skill_hash"),
        "actions": normalized_actions,
        # Keep a descriptive alias for downstream reflection code.  It is the
        # same normalized sequence, not a second copy of retrieved text.
        "trajectory": normalized_actions,
        "read_chunk_ids": list(dict.fromkeys(read_ids)),
        "entity_hop_calls": entity_hop_calls,
        "entity_hop_count": len(entity_hop_calls),
        "final_answer": answer,
        "loop_count": _first(raw, "loop_count", "loops", "num_loops", default=len(normalized_actions)),
        "policy_calls": _first(raw, "policy_calls", "llm_calls", "calls", default=None),
        "input_tokens": _first(usage, "input_tokens", "prompt_tokens", default=None),
        "output_tokens": _first(usage, "output_tokens", "completion_tokens", default=None),
        "total_tokens": _first(usage, "total_tokens", default=None),
        "termination_reason": _first(raw, "termination_reason", "stop_reason", "status", default=None),
        "total_cost": _first(raw, "total_cost", "cost", default=None),
    }
