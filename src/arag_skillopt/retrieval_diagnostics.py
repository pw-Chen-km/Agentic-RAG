"""Offline hop diagnostics derived from an immutable A-RAG raw episode.

The diagnostic record is an analysis sidecar.  It is never sent to the policy
LLM and is not used by HRO Stage A.  It records what a hop returned, the graph
bridge metadata exposed by the tool, and whether a later observed action
explicitly consumed one of those returned IDs.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping

from .observations import result_units


DIAGNOSTICS_VERSION = "arag-retrieval-diagnostics-v1"
HOP_TOOLS = frozenset({"entity_hop", "linear_hop"})


def _actions(raw):
    value = raw.get("trajectory", raw.get("actions", []))
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _tool(action):
    value = action.get("tool_name", action.get("tool", action.get("name", "")))
    return str(value)


def _result(action):
    value = action.get("tool_result", action.get("result", ""))
    return value if isinstance(value, str) else str(value)


def _metadata(action):
    log = action.get("log", {})
    return {**(log if isinstance(log, Mapping) else {}), **action}


def _loop(action, index):
    value = action.get("loop", action.get("decision"))
    return int(value) if isinstance(value, int) and value > 0 else index + 1


def _arguments(action):
    value = action.get("arguments", action.get("args", action.get("parameters", {})))
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return {"unparsed_arguments": value}
    return value if isinstance(value, Mapping) else {}


def _ids_from_units(tool, text):
    try:
        units, parse_status = result_units(tool, text)
    except (AttributeError, KeyError, TypeError, ValueError):
        return set(), set(), "unparsed"
    snippets = {
        str(unit["chunk_id"])
        for unit in units
        if unit.get("available") and unit.get("kind") == "snippet" and unit.get("chunk_id") is not None
    }
    full = {
        str(unit["chunk_id"])
        for unit in units
        if unit.get("available") and unit.get("kind") == "full" and unit.get("chunk_id") is not None
    }
    return snippets, full, parse_status


def _metadata_ids(metadata, key):
    value = metadata.get(key, [])
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)} if value not in (None, "") else set()


def _status(action, metadata, parsed_status, returned_ids):
    if metadata.get("error") or _result(action).startswith("Error"):
        return "error"
    explicit = metadata.get("status")
    if explicit:
        return str(explicit)
    if returned_ids:
        return "ok"
    if _result(action).startswith(("No results", "No entity", "No graph")):
        return "no_entity_match"
    if parsed_status == "empty":
        return "no_entity_match"
    return "unknown"


def _bridge_rows(metadata):
    trace = metadata.get("hop_trace")
    if isinstance(trace, list):
        rows = []
        for row in trace:
            if not isinstance(row, Mapping):
                continue
            rows.append({
                key: row[key]
                for key in (
                    "hop",
                    "sentence_id",
                    "chunk_id",
                    "source_entity_ids",
                    "activated_entity_ids",
                    "similarity",
                    "path_score",
                )
                if key in row
            })
        if rows:
            return rows

    # Older/compact logs may only contain parallel ID lists.  Preserve those
    # IDs without inventing a path between them.
    sentence_ids = list(_metadata_ids(metadata, "sentence_ids"))
    chunk_ids = list(_metadata_ids(metadata, "chunk_ids"))
    activated = sorted(_metadata_ids(metadata, "activated_entities"))
    if not sentence_ids and not chunk_ids and not activated:
        return []
    return [
        {
            "sentence_id": sentence_ids[index] if index < len(sentence_ids) else None,
            "chunk_id": chunk_ids[index] if index < len(chunk_ids) else None,
            "activated_entity_ids": activated,
            "bridge_detail": "summary_only",
        }
        for index in range(max(len(sentence_ids), len(chunk_ids), 1))
    ]


def _full_read_ids(action):
    if _tool(action) != "read_chunk":
        return set()
    _, full, _ = _ids_from_units("read_chunk", _result(action))
    metadata = _metadata(action)
    # The tool log is authoritative when parsing an older or nonstandard
    # read response; requested IDs alone are never treated as successful reads.
    logged = _metadata_ids(metadata, "new_chunks_read")
    return full | logged


def _contains_identifier(arguments, identifiers):
    if not identifiers:
        return False
    try:
        text = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(arguments)
    return any(identifier and str(identifier) in text for identifier in identifiers)


def build_retrieval_diagnostics(raw: Mapping) -> dict:
    """Build a deterministic per-hop diagnostic record from one raw episode."""
    raw = raw if isinstance(raw, Mapping) else {}
    actions = _actions(raw)
    grouped = defaultdict(list)
    for index, action in enumerate(actions):
        grouped[_loop(action, index)].append((index, action))

    trace_complete = isinstance(raw.get("decision_audit"), list) and bool(raw.get("decision_audit"))
    if raw.get("execution_errors") or raw.get("error"):
        trace_complete = False

    events = []
    status_counts = Counter()
    adoption_counts = Counter()
    for index, action in enumerate(actions):
        tool = _tool(action)
        if tool not in HOP_TOOLS:
            continue
        metadata = _metadata(action)
        parsed_snippets, _, parsed_status = _ids_from_units(tool, _result(action))
        returned_ids = parsed_snippets | _metadata_ids(metadata, "chunk_ids")
        status = _status(action, metadata, parsed_status, returned_ids)
        bridge = _bridge_rows(metadata)
        activated = sorted(_metadata_ids(metadata, "activated_entities"))
        bridge_ids = set(returned_ids)
        for row in bridge:
            if row.get("chunk_id") is not None:
                bridge_ids.add(str(row["chunk_id"]))

        later_full_reads = set()
        later_references = []
        stable_ids = set(bridge_ids) | set(activated)
        current_loop = _loop(action, index)
        for later_index, later_action in enumerate(actions):
            if _loop(later_action, later_index) <= current_loop:
                continue
            full_ids = _full_read_ids(later_action)
            matched_reads = full_ids & returned_ids
            later_full_reads.update(matched_reads)
            if matched_reads:
                later_references.append({
                    "kind": "full_read",
                    "tool": _tool(later_action),
                    "ids": sorted(matched_reads),
                    "turn": _loop(later_action, later_index),
                })
            if _contains_identifier(_arguments(later_action), stable_ids):
                later_references.append({
                    "kind": "argument_reference",
                    "tool": _tool(later_action),
                    "turn": _loop(later_action, later_index),
                })

        if not returned_ids:
            adoption = None
            adoption_status = "not_applicable"
            adoption_basis = "hop_returned_no_stable_evidence_id"
        elif not trace_complete:
            adoption = None
            adoption_status = "unknown"
            adoption_basis = "trajectory_visibility_not_verified"
        elif later_references:
            adoption = True
            adoption_status = "observed"
            adoption_basis = "later_action_referenced_hop_output"
        else:
            adoption = False
            adoption_status = "not_observed"
            adoption_basis = "no_later_action_referenced_hop_output"

        event = {
            "call_index": index + 1,
            "turn": current_loop,
            "tool": tool,
            "status": status,
            "returned_snippet_ids": sorted(returned_ids),
            "activated_entities": activated,
            "bridge": bridge,
            "hop_backend": metadata.get("hop_backend"),
            "max_hops": metadata.get("max_hops"),
            "top_k": metadata.get("top_k"),
            "later_full_reads": sorted(later_full_reads),
            "later_references": later_references,
            "adoption": adoption,
            "adoption_status": adoption_status,
            "adoption_basis": adoption_basis,
        }
        events.append(event)
        status_counts[status] += 1
        adoption_counts[adoption_status] += 1

    return {
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "trace_complete": trace_complete,
        "question_id": raw.get("question_id"),
        "hop_tools": sorted(HOP_TOOLS),
        "events": events,
        "summary": {
            "hop_calls": len(events),
            "status_counts": dict(sorted(status_counts.items())),
            "adoption_counts": dict(sorted(adoption_counts.items())),
            "bridge_event_count": sum(bool(event["bridge"]) for event in events),
            "returned_snippet_count": sum(len(event["returned_snippet_ids"]) for event in events),
        },
    }
