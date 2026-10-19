"""Offline metrics for the fixed-skill EntityHop ablation.

Metrics are derived only from the saved runtime trace.  A chunk is newly seen
when it is first returned in a parsed snippet/full result, not merely when its
ID was requested.  Batched calls share a pre-decision state.  In particular a
read in the *same* decision cannot be credited as using a hop's unseen result.
Unknown usage is kept as ``None`` and is never imputed as zero.
"""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .observations import result_units


METRICS_VERSION = "entityhop-metrics-v1"
RETRIEVAL_TOOLS = frozenset({"keyword_search", "semantic_search", "entity_hop"})


def _number(value: Any) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _canonical_args(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if isinstance(value, Mapping):
        return {str(key): _canonical_args(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        # Lists remain ordered: exact argument repetition is not set equality.
        return [_canonical_args(item) for item in value]
    return value


def normalized_call_key(tool: str, arguments: Any) -> str:
    """Exact call identity after Unicode/case/whitespace normalization."""

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass
    return json.dumps([tool, _canonical_args(arguments)], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _actions(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = raw.get("trajectory", raw.get("actions", []))
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _tool(action: Mapping[str, Any]) -> str:
    value = action.get("tool_name", action.get("tool", action.get("name", "")))
    if isinstance(value, Mapping):
        value = value.get("name", "")
    if not value and isinstance(action.get("function"), Mapping):
        value = action["function"].get("name", "")
    return str(value)


def _arguments(action: Mapping[str, Any]) -> Any:
    value = action.get("arguments", action.get("args", action.get("parameters")))
    if value is None and isinstance(action.get("function"), Mapping):
        value = action["function"].get("arguments", {})
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value if value is not None else {}


def _result(action: Mapping[str, Any]) -> str:
    value = action.get("tool_result", action.get("result", ""))
    return value if isinstance(value, str) else str(value)


def _metadata(action: Mapping[str, Any]) -> dict[str, Any]:
    log = action.get("log", {})
    return {**(log if isinstance(log, Mapping) else {}), **action}


def _decision(action: Mapping[str, Any], index: int) -> int:
    value = action.get("loop", action.get("decision"))
    return int(value) if isinstance(value, int) and value > 0 else index + 1


def _usage(raw: Mapping[str, Any]) -> dict[str, int | float | None]:
    usage = raw.get("usage", {})
    usage = usage if isinstance(usage, Mapping) else {}
    result: dict[str, int | float | None] = {
        "input_tokens": _number(usage.get("input_tokens", usage.get("prompt_tokens"))),
        "output_tokens": _number(usage.get("output_tokens", usage.get("completion_tokens"))),
        "total_tokens": _number(usage.get("total_tokens")),
    }
    if usage.get("complete") is False:
        # Some transports report a partial sum; not a complete episode cost.
        return {key: None for key in result}
    if result["total_tokens"] is None and result["input_tokens"] is not None and result["output_tokens"] is not None:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def episode_metrics(raw: Mapping[str, Any], evaluation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Extract auditable episode metrics from unmodified A-RAG output."""

    actions = _actions(raw)
    audit = raw.get("decision_audit")
    audit = audit if isinstance(audit, list) else None
    grouped: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    parsed: dict[int, tuple[list[dict[str, Any]], str | None]] = {}
    unknown_formats: list[int] = []
    for index, action in enumerate(actions):
        grouped[_decision(action, index)].append((index, action))
        parsed[index] = result_units(_tool(action), _result(action))
        if parsed[index][1] == "unparsed" and _tool(action) in RETRIEVAL_TOOLS | {"read_chunk"}:
            text = _result(action)
            if not text.startswith(("Error", "No entity-linked chunks")):
                unknown_formats.append(index + 1)

    seen: set[str] = set()
    full_reads: set[str] = set()
    decision_rows: list[dict[str, Any]] = []
    successful_reads: list[tuple[int, set[str]]] = []
    hop_rows: list[dict[str, Any]] = []
    repeated = 0
    repeated_hops = 0
    repeated_searches = 0
    call_keys: set[str] = set()
    tool_errors: list[dict[str, Any]] = []

    for decision in sorted(grouped):
        # Update source visibility only after all co-selected calls execute.
        returned_ids: set[str] = set()
        newly_read: set[str] = set()
        decision_tools: list[str] = []
        for index, action in grouped[decision]:
            tool = _tool(action)
            decision_tools.append(tool)
            meta = _metadata(action)
            units, _status = parsed[index]
            ids = {str(unit["chunk_id"]) for unit in units if unit.get("available") and unit.get("kind") in {"snippet", "full"}}
            returned_ids.update(ids)
            if tool == "read_chunk":
                read_ids = {str(unit["chunk_id"]) for unit in units if unit.get("available") and unit.get("kind") == "full"}
                newly_read.update(read_ids - full_reads)
                successful_reads.append((decision, read_ids))
                full_reads.update(read_ids)
            if tool in RETRIEVAL_TOOLS:
                key = normalized_call_key(tool, _arguments(action))
                if key in call_keys:
                    repeated += 1
                    if tool == "entity_hop":
                        repeated_hops += 1
                    else:
                        repeated_searches += 1
                call_keys.add(key)
            if meta.get("error") or _result(action).startswith("Error"):
                tool_errors.append({"call": index + 1, "decision": decision, "tool": tool, "error": meta.get("error", _result(action))})
            if tool == "entity_hop":
                status = meta.get("status")
                if status is None:
                    if ids:
                        status = "ok"
                    elif _result(action).startswith(("No results", "No entity-linked chunks")):
                        status = "no_entity_match"
                    else:
                        status = "unknown"
                hop_rows.append({
                    "call": index + 1,
                    "decision": decision,
                    "status": status,
                    "chunk_ids": sorted(ids),
                    "new_chunk_ids": sorted(ids - seen),
                    "hop_backend": meta.get("hop_backend"),
                })
        new_ids = returned_ids - seen
        decision_rows.append({
            "decision": decision,
            "calls_selected_together": decision_tools,
            "returned_chunk_ids": sorted(returned_ids),
            "returned_chunk_count": len(returned_ids),
            "new_chunk_ids": sorted(new_ids),
            "new_chunk_count": len(new_ids),
            "new_source_ratio": len(new_ids) / len(returned_ids) if returned_ids else None,
            "cumulative_unique_chunks": len(seen | returned_ids),
            "first_full_read_chunk_ids": sorted(newly_read),
            "first_full_read_count": len(newly_read),
        })
        seen.update(returned_ids)

    for hop in hop_rows:
        hop_ids = set(hop["chunk_ids"])
        later_reads = set().union(*(ids for decision, ids in successful_reads if decision > hop["decision"]))
        credited = hop_ids & later_reads
        hop["read_after_hop"] = bool(credited)
        hop["read_after_hop_chunk_ids"] = sorted(credited)
    hop_count = len(hop_rows)
    ok_hops = sum(row["status"] == "ok" and bool(row["chunk_ids"]) for row in hop_rows)
    no_matches = sum(row["status"] == "no_entity_match" for row in hop_rows)
    hops_with_chunks = sum(bool(row["chunk_ids"]) for row in hop_rows)
    later_read_hops = sum(row["read_after_hop"] for row in hop_rows)
    action_token_values = [_number(_metadata(action).get("retrieved_tokens")) for action in actions]
    retrieved_tokens = _number(raw.get("total_retrieved_tokens"))
    if retrieved_tokens is None and all(value is not None for value in action_token_values):
        retrieved_tokens = sum(action_token_values)

    execution_errors = raw.get("execution_errors")
    service_errors = []
    if audit is not None:
        service_errors.extend(
            {"decision": row.get("decision", index + 1), "error": row["error"]}
            for index, row in enumerate(audit)
            if isinstance(row, Mapping) and row.get("error")
        )
    if isinstance(execution_errors, list):
        service_errors.extend({"error": error} for error in execution_errors)
    if raw.get("error") and str(raw["error"]) not in {str(row["error"]) for row in service_errors}:
        service_errors.append({"error": raw["error"]})
    correct = evaluation.get("correct") if isinstance(evaluation, Mapping) else None
    correct = correct if type(correct) is bool else None
    model_calls = len(audit) if audit is not None else _number(raw.get("policy_calls"))
    return {
        "metrics_version": METRICS_VERSION,
        "correct": correct,
        "model_calls": model_calls,
        "tool_calls": len(actions),
        "read_calls": sum(_tool(action) == "read_chunk" for action in actions),
        "unique_full_chunks_read": len(full_reads),
        "unique_chunks_seen": len(seen),
        "new_chunk_count": len(seen),
        "entity_hop_calls": hop_count,
        "hop_successes": ok_hops,
        "hop_success_rate": ok_hops / hop_count if hop_count else None,
        "no_entity_match_count": no_matches,
        "no_entity_match_ratio": no_matches / hop_count if hop_count else None,
        "hop_read_after_count": later_read_hops,
        "hop_read_after_ratio": later_read_hops / hops_with_chunks if hops_with_chunks else None,
        "hop_read_after_denominator": hops_with_chunks,
        "hop_new_chunk_count": len(set().union(*(set(row["new_chunk_ids"]) for row in hop_rows))),
        "repeated_search_or_hop_calls": repeated,
        "repeated_search_calls": repeated_searches,
        "repeated_hop_calls": repeated_hops,
        "retrieved_tokens": retrieved_tokens,
        **_usage(raw),
        "latency_seconds": _number(raw.get("elapsed_seconds")),
        "service_error_count": len(service_errors),
        "service_errors": service_errors,
        "tool_error_count": len(tool_errors),
        "tool_errors": tool_errors,
        "unparsed_result_calls": unknown_formats,
        "unknown_evaluation": correct is None,
        "evidence_recall": None,
        "evidence_recall_status": "N/A: Novel has no reliable sentence-level gold coordinates",
        "decisions": decision_rows,
        "hops": hop_rows,
    }


NUMERIC_METRICS = (
    "model_calls", "tool_calls", "read_calls", "unique_full_chunks_read", "unique_chunks_seen",
    "entity_hop_calls", "hop_successes", "no_entity_match_count", "hop_read_after_count",
    "hop_new_chunk_count", "repeated_search_or_hop_calls", "repeated_search_calls", "repeated_hop_calls",
    "retrieved_tokens", "input_tokens", "output_tokens", "total_tokens", "latency_seconds",
    "service_error_count", "tool_error_count",
)


def aggregate_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate observed costs while explicitly reporting missingness."""

    metrics = [row.get("metrics") if isinstance(row.get("metrics"), Mapping) else episode_metrics(row.get("raw", row), row.get("evaluation")) for row in episodes]
    known = [row["correct"] for row in metrics if type(row.get("correct")) is bool]
    result: dict[str, Any] = {
        "episodes": len(metrics),
        "known_evaluations": len(known),
        "unknown_evaluations": len(metrics) - len(known),
        "accuracy": sum(known) / len(known) if known else None,
        "evidence_recall": "N/A",
    }
    for key in NUMERIC_METRICS:
        values = [_number(row.get(key)) for row in metrics]
        observed = [value for value in values if value is not None]
        result[key] = {
            "observed_episodes": len(observed),
            "missing_episodes": len(metrics) - len(observed),
            "observed_sum": sum(observed) if observed else None,
            "mean_observed": sum(observed) / len(observed) if observed else None,
            "complete_sum": sum(observed) if len(observed) == len(metrics) else None,
        }
    def ratio(numerator: str, denominator: str):
        n = sum(row.get(numerator, 0) for row in metrics)
        d = sum(row.get(denominator, 0) for row in metrics)
        return n / d if d else None
    result["hop_success_rate"] = ratio("hop_successes", "entity_hop_calls")
    result["no_entity_match_ratio"] = ratio("no_entity_match_count", "entity_hop_calls")
    result["hop_read_after_ratio"] = ratio("hop_read_after_count", "hop_read_after_denominator")
    return result


def paired_comparison(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare only same-ID measurements; never impute missing/unknown values."""

    def indexed(rows):
        result = {}
        for row in rows:
            qid = str(row.get("question_id", row.get("raw", {}).get("question_id", "")))
            if not qid or qid in result:
                raise ValueError(f"missing or duplicate question ID: {qid!r}")
            result[qid] = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else episode_metrics(row.get("raw", row), row.get("evaluation"))
        return result
    a, b = indexed(left), indexed(right)
    common = sorted(set(a) & set(b))
    accuracy_ids = [qid for qid in common if type(a[qid].get("correct")) is bool and type(b[qid].get("correct")) is bool]
    outcome = {
        "matched_ids": len(common),
        "left_only_ids": sorted(set(a) - set(b)),
        "right_only_ids": sorted(set(b) - set(a)),
        "accuracy_pair_count": len(accuracy_ids),
        "accuracy_excluded_ids": sorted(set(common) - set(accuracy_ids)),
        "accuracy_left": sum(a[qid]["correct"] for qid in accuracy_ids) / len(accuracy_ids) if accuracy_ids else None,
        "accuracy_right": sum(b[qid]["correct"] for qid in accuracy_ids) / len(accuracy_ids) if accuracy_ids else None,
        "right_wins": sum(not a[qid]["correct"] and b[qid]["correct"] for qid in accuracy_ids),
        "left_wins": sum(a[qid]["correct"] and not b[qid]["correct"] for qid in accuracy_ids),
        "costs": {},
    }
    for key in NUMERIC_METRICS:
        ids = [qid for qid in common if _number(a[qid].get(key)) is not None and _number(b[qid].get(key)) is not None]
        outcome["costs"][key] = {
            "pair_count": len(ids),
            "excluded_ids": sorted(set(common) - set(ids)),
            "left_mean": sum(a[qid][key] for qid in ids) / len(ids) if ids else None,
            "right_mean": sum(b[qid][key] for qid in ids) / len(ids) if ids else None,
            "mean_delta_right_minus_left": sum(b[qid][key] - a[qid][key] for qid in ids) / len(ids) if ids else None,
        }
    return outcome


def markdown_report(arms: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """Compact report with explicit evaluation and cost populations."""

    lines = ["# Novel EntityHop ablation", "", "Evidence recall: N/A. Accuracy uses known evaluations only; cost means use observed episodes, with missing counts retained in summary JSON.", "", "| Arm | Episodes | Evaluated | Unknown | Accuracy | Model calls | Tool calls | Reads | Hops | Total tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
    for arm, episodes in arms.items():
        s = aggregate_metrics(episodes)
        values = [arm, s["episodes"], s["known_evaluations"], s["unknown_evaluations"], s["accuracy"]]
        values += [s[key]["mean_observed"] for key in ("model_calls", "tool_calls", "read_calls", "entity_hop_calls", "total_tokens")]
        lines.append("| " + " | ".join(fmt(value) for value in values) + " |")
    return "\n".join(lines) + "\n"
