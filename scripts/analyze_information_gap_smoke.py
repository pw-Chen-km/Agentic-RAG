"""Read raw episode JSON and report information-gap workflow diagnostics.

No Assessment model, semantic judge, or target-agent input is used or changed.
Generic-gap flags are text heuristics; action/gap alignment requires manual review.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VERSION = "information-gap-smoke-analysis-v1"
KNOWN_DATASETS = {"hotpotqa", "novel", "medical"}
GENERIC_GAP = re.compile(
    r"^(?:(?:i|we)\s+)?(?:(?:still\s+)?need(?:s|ed)?\s+)?(?:more\s+)?"
    r"(?:information|evidence|context|details|sources|facts|answer)(?:\s+(?:is\s+)?(?:needed|required|missing))?[.!]?\s*$",
    re.IGNORECASE,
)
TOKENS = ("input", "output", "total")


def _object(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _calls(provider: dict) -> list:
    calls = provider.get("raw_tool_calls")
    if isinstance(calls, list):
        return calls
    raw = _object(provider.get("raw_provider_output"))
    calls = _object(raw.get("message")).get("tool_calls")
    if isinstance(calls, list):
        return calls
    choices = raw.get("choices") or []
    if choices and isinstance(choices[0], dict):
        calls = _object(choices[0].get("message")).get("tool_calls")
        if isinstance(calls, list):
            return calls
    return []


def _assessment(value: Any, requested: bool) -> dict:
    result = {"status": "not_requested" if not requested else "missing_assessment",
              "schema_valid": False, "gaps": []}
    if value is None:
        return result
    if isinstance(value, dict) and isinstance(value.get("missing_information"), list):
        result["gaps"] = [item for item in value["missing_information"] if isinstance(item, str)]
    if not requested:
        result["status"] = "unexpected_assessment"
        return result
    valid = (isinstance(value, dict) and set(value) == {"missing_information"}
             and isinstance(value["missing_information"], list)
             and len(value["missing_information"]) <= 3
             and all(isinstance(item, str) and bool(item.strip()) for item in value["missing_information"]))
    result.update(status="valid" if valid else "invalid_assessment", schema_valid=valid)
    return result


def _raw_call(call: Any, requested: bool) -> dict:
    function = _object(_object(call).get("function"))
    name = function.get("name") or "unavailable"
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args, object_pairs_hook=_strict_object)
        except (ValueError, TypeError):
            return {"name": name, "arguments": None, "assessment": {
                "status": "invalid_json", "schema_valid": False, "gaps": []}}
    if not isinstance(args, dict):
        return {"name": name, "arguments": None, "assessment": {
            "status": "invalid_arguments", "schema_valid": False, "gaps": []}}
    return {"name": name, "arguments": {key: value for key, value in args.items() if key != "assessment"},
            "assessment": _assessment(args.get("assessment"), requested)}


def _canonical_action(value: Any) -> str | None:
    action = _object(value)
    if action.get("name"):
        return str(action["name"])
    kind = action.get("type")
    if kind == "SEARCH":
        return {"CHUNK": "find_passages", "SENTENCE": "find_sentences"}.get(action.get("target"), "search")
    if kind == "EXPAND":
        return {"ENTITY_MENTIONED_IN_CHUNK": "follow_entity_to_passages",
                "ENTITY_MENTIONED_IN_SENTENCE": "follow_entity_to_sentences"}.get(action.get("kind"), "follow")
    return {"FINISH": "finish", "READ": "read_passage"}.get(kind)


def _entity_cards(step: dict) -> int | None:
    messages = step.get("messages")
    if isinstance(messages, list):
        users = [message.get("content") for message in messages
                 if isinstance(message, dict) and message.get("role") == "user"
                 and isinstance(message.get("content"), str)]
        if users:
            current = users[-1]
            if "Visible entity references:" in current:
                section = current.split("Visible entity references:", 1)[1].split("\n\n", 1)[0]
                return len(set(re.findall(r"(?m)^\s*(E[1-9][0-9]*)\s*[—-]", section)))
            if "NEW SOURCE TEXT" in current:
                return 0
    refs = _object(_object(step.get("context_reference_map")).get("typed_refs"))
    if refs or "context_reference_map" in step:
        return sum(_object(value).get("node_type") == "ENTITY" for value in refs.values())
    return None


def _known_number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _sum_known(values: list) -> int | float | None:
    return sum(values) if values and all(value is not None for value in values) else None


def analyze_episode(episode: dict, *, dataset: str, path: Path, requested: bool = True) -> dict:
    episode_id = str(episode.get("episode_id") or path.parent.name)
    condition = str(episode.get("condition") or episode_id.split("--", 1)[0])
    turns, failures, actions, executed_actions, raw_statuses = [], Counter(), Counter(), Counter(), Counter()
    duplicate_streak = max_duplicate_streak = 0
    previous_gaps = None
    gap_changes = gap_comparisons = 0
    for index, step in enumerate(episode.get("trajectory") or [], 1):
        if not isinstance(step, dict):
            continue
        provider = _object(step.get("provider_metadata"))
        observation = _object(step.get("observation"))
        metadata, audit = _object(observation.get("metadata")), _object(step.get("context_audit"))
        required = bool(audit.get("assessment_requested", requested))
        raw = [_raw_call(call, required) for call in _calls(provider)]
        raw_statuses.update(call["assessment"]["status"] for call in raw)
        decision = _object(step.get("decision") or step.get("resolved_decision"))
        parsed = _assessment(decision.get("assessment"), required)
        if not raw and decision.get("assessment") is not None:
            raw_statuses[parsed["status"]] += 1
        chosen = parsed if parsed["schema_valid"] else next(
            (call["assessment"] for call in raw if call["assessment"]["schema_valid"]), parsed)
        # Retain malformed raw gap strings for burden diagnostics without treating them as valid.
        diagnostics = chosen if chosen["schema_valid"] else next(
            (call["assessment"] for call in raw if call["assessment"]["gaps"]), chosen)
        gaps = diagnostics["gaps"]
        code, status = observation.get("error_code"), observation.get("status")
        category = (provider.get("failure_category") or metadata.get("failure_category")
                    or (code if code in {"protocol_invalid", "state_invalid", "execution_error"} else None))
        duplicate = code == "duplicate_action" or status == "duplicate_action"
        if category:
            failures[str(category)] += 1
        if duplicate:
            failures["duplicate_action"] += 1
        duplicate_streak = duplicate_streak + 1 if duplicate else 0
        max_duplicate_streak = max(max_duplicate_streak, duplicate_streak)
        rejected = duplicate or status == "invalid_action" or step.get("validation_status") == "invalid"
        names = [call["name"] for call in raw]
        if not names and _canonical_action(decision.get("action")):
            names = [_canonical_action(decision["action"])]
        actions.update(names)
        if step.get("validation_status") == "valid" and status == "ok":
            executed_actions.update(names)
        if chosen["schema_valid"]:
            if previous_gaps is not None:
                gap_comparisons += 1
                gap_changes += previous_gaps != gaps
            previous_gaps = list(gaps)
        tokens = {name: _known_number(provider.get(f"provider_{name}_tokens")) for name in TOKENS}
        turns.append({"turn": step.get("policy_attempt", index), "assessment_requested": required,
                      "assessment_recorded_status": step.get("assessment_status", "unavailable"),
                      "raw_assessments": raw, "assessment_schema_valid": chosen["schema_valid"],
                      "missing_information": gaps, "gap_count": len(gaps),
                      "gap_character_count": sum(map(len, gaps)),
                      "blank_gap_count": sum(not gap.strip() for gap in gaps),
                      "generic_gap_flags": [bool(GENERIC_GAP.fullmatch(gap.strip())) for gap in gaps],
                      "selected_actions": names, "failure_category": category, "error_code": code,
                      "duplicate": duplicate, "rejected": rejected,
                      "assessment_available_on_rejected_action": chosen["schema_valid"] if rejected else None,
                      "entity_card_count": _entity_cards(step), "provider_tokens": tokens})
    provider_tokens = {name: _sum_known([turn["provider_tokens"][name] for turn in turns]) for name in TOKENS}
    usage = _object(episode.get("usage"))
    return {"dataset": dataset, "condition": condition, "episode_id": episode_id, "artifact": str(path),
            "question": episode.get("query"),
            "terminal": bool(episode.get("termination_reason") or episode.get("error_code")),
            "termination_reason": episode.get("termination_reason"), "error_code": episode.get("error_code"),
            "policy_calls": _known_number(usage.get("policy_calls")) if usage.get("policy_calls") is not None else len(turns),
            "provider_tokens": provider_tokens, "failure_categories": dict(failures),
            "attempted_actions": dict(actions), "executed_actions": dict(executed_actions),
            "raw_assessment_statuses": dict(raw_statuses),
            "raw_assessment_parse_failures": sum(count for status, count in raw_statuses.items()
                if status in {"invalid_json", "invalid_arguments", "invalid_assessment", "missing_assessment", "unexpected_assessment"}),
            "max_consecutive_duplicates": max_duplicate_streak,
            "gap_changes": gap_changes, "gap_comparisons": gap_comparisons, "turns": turns}


def _aggregate(rows: list[dict]) -> dict:
    turns = [turn for row in rows for turn in row["turns"]]
    gaps = [gap for turn in turns for gap in turn["missing_information"]]
    cards = [turn["entity_card_count"] for turn in turns if turn["entity_card_count"] is not None]
    rejected = [turn for turn in turns if turn["rejected"]]
    failures, actions, executed_actions, raw_statuses, terminals = Counter(), Counter(), Counter(), Counter(), Counter()
    for row in rows:
        failures.update(row["failure_categories"]); actions.update(row["attempted_actions"])
        executed_actions.update(row["executed_actions"])
        raw_statuses.update(row["raw_assessment_statuses"])
        if row["terminal"]:
            terminals[row["termination_reason"]] += 1
    return {"dataset": rows[0]["dataset"], "condition": rows[0]["condition"], "episodes": len(rows),
            "terminal_artifacts": sum(row["terminal"] for row in rows), "terminal_reasons": dict(terminals),
            "policy_calls": _sum_known([row["policy_calls"] for row in rows]), "turn_count": len(turns),
            "provider_tokens": {name: _sum_known([row["provider_tokens"][name] for row in rows]) for name in TOKENS},
            "token_availability": {name: sum(row["provider_tokens"][name] is not None for row in rows) for name in TOKENS},
            "failure_categories": dict(failures),
            "protocol_invalid_count": failures.get("protocol_invalid", 0),
            "state_invalid_count": failures.get("state_invalid", 0),
            "execution_error_count": failures.get("execution_error", 0),
            "duplicate_action_count": failures.get("duplicate_action", 0),
            "attempted_actions": dict(actions), "executed_actions": dict(executed_actions),
            "raw_assessment_statuses": dict(raw_statuses),
            "raw_assessment_parse_failures": sum(row["raw_assessment_parse_failures"] for row in rows),
            "assessment_valid_turns": sum(turn["assessment_schema_valid"] for turn in turns),
            "gap_item_count": len(gaps), "gap_character_count": sum(map(len, gaps)),
            "mean_gap_items_per_observed_turn": len(gaps) / len(turns) if turns else None,
            "mean_gap_characters_per_item": sum(map(len, gaps)) / len(gaps) if gaps else None,
            "blank_gap_count": sum(not gap.strip() for gap in gaps),
            "generic_gap_heuristic_count": sum(sum(turn["generic_gap_flags"]) for turn in turns),
            "max_consecutive_duplicates": max((row["max_consecutive_duplicates"] for row in rows), default=0),
            "gap_changes": sum(row["gap_changes"] for row in rows),
            "gap_comparisons": sum(row["gap_comparisons"] for row in rows),
            "rejected_turns": len(rejected),
            "rejected_turns_with_available_assessment": sum(turn["assessment_available_on_rejected_action"] for turn in rejected),
            "entity_card_observed_turns": len(cards),
            "mean_entity_cards": sum(cards) / len(cards) if cards else None,
            "max_entity_cards": max(cards) if cards else None}


def analyze(run: Path) -> dict:
    root = run.resolve()
    rows, errors = [], []
    for path in sorted(root.rglob("episode.json")):
        if "episodes" not in path.relative_to(root).parts:
            continue
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(episode, dict):
                raise ValueError("episode JSON must be an object")
            owner = path.parent.parent.parent
            manifest_path = owner / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            dataset = str(manifest.get("dataset") or episode.get("dataset") or next(
                (part for part in reversed(path.relative_to(root).parts) if part.lower() in KNOWN_DATASETS), "unavailable")).lower()
            rows.append(analyze_episode(episode, dataset=dataset, path=path,
                        requested=bool(manifest.get("require_evidence_assessment", True))))
        except (ValueError, OSError, TypeError) as exc:
            errors.append({"artifact": str(path), "reason": str(exc)})
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["condition"])].append(row)
    samples = [{"status": "needs_review", "dataset": row["dataset"], "condition": row["condition"],
                "episode_id": row["episode_id"], "artifact": row["artifact"], "question": row["question"],
                "turn": turn["turn"], "missing_information": turn["missing_information"],
                "selected_actions": turn["selected_actions"], "raw_calls": turn["raw_assessments"],
                "rejected": turn["rejected"], "semantic_alignment_score": None}
               for row in rows for turn in row["turns"] if turn["assessment_schema_valid"]]
    return {"analysis_version": VERSION, "run": str(root),
            "status": "completed_with_read_errors" if errors else "completed",
            "episode_artifacts": len(rows), "terminal_artifacts": sum(row["terminal"] for row in rows),
            "artifact_read_errors": errors,
            "dataset_condition_aggregates": [_aggregate(value) for _, value in sorted(groups.items())],
            "episodes": rows, "manual_action_gap_review": samples,
            "notes": {"generic_gap_flags": "Literal text heuristic only; not a semantic specificity score.",
                      "action_gap_alignment": "needs_review; no automatic semantic alignment score.",
                      "gap_counts": "Include raw string gaps from malformed assessments for burden diagnostics; validity is reported separately.",
                      "tokens": "Provider-reported tokens only; unavailable fields remain null and are not replaced by zero.",
                      "assessment_parse_failures": "Count malformed or missing assessments in raw calls; no raw call is unavailable, not counted as a schema parse failure.",
                      "failure_categories": "Recorded artifact categories; duplicate is also counted separately and may overlap state_invalid."}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run.is_dir():
        parser.error("--run must name an existing run directory")
    result = analyze(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "episode_artifacts", "terminal_artifacts", "artifact_read_errors")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
