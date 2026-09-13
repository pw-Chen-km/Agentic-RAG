"""Evidence-sidecar metrics for the interface study."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from collections import Counter
from typing import Any


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().strip())


def score_answer(predicted: str | None, gold: str) -> dict[str, bool]:
    predicted = predicted or ""
    p = normalize_answer(predicted)
    g = normalize_answer(gold)
    return {"exact": p == g, "contain": bool(g) and g in p}


def evaluate_episode(
    episode: Mapping[str, Any],
    *,
    question: Mapping[str, Any],
    gold_support: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate only against source spans actually exposed to Policy."""

    facts = [item for item in gold_support if item.get("question_id") == question.get("_id")]
    mapping_missing = any(not item.get("sentence_id") for item in facts)
    visible_sentence_ids: set[str] = set()
    projected_sentence_ids: set[str] = set()
    candidate_count = 0
    policy_calls = 0
    invalid_attempts = 0
    retrieval_actions = 0
    first_complete_decision: int | None = None
    first_complete_tokens: int | None = None
    for step in episode.get("trajectory", []):
        policy_calls += 1
        status = step.get("validation_status")
        if status == "invalid":
            invalid_attempts += 1
        observation = step.get("observation") or {}
        action = (observation.get("action") or {}).get("type")
        if action in {"SEARCH", "EXPAND", "READ"}:
            retrieval_actions += 1
        candidate_count += len(observation.get("results") or [])
        for span in step.get("visible_source_spans") or []:
            sid = span.get("sentence_id")
            if sid:
                projected_sentence_ids.add(sid)
                if span.get("visible") is True:
                    visible_sentence_ids.add(sid)
        gold_ids = {item.get("sentence_id") for item in facts if item.get("sentence_id")}
        if gold_ids and gold_ids.issubset(visible_sentence_ids) and first_complete_decision is None:
            first_complete_decision = int(step.get("policy_attempt") or policy_calls)
            usage = step.get("usage") or {}
            first_complete_tokens = usage.get("total_tokens")

    final_refs = {
        (item.get("id") if isinstance(item, Mapping) else item)
        for item in (episode.get("evidence_refs") or [])
    }
    cited_sentence_ids: set[str] = set()
    for step in episode.get("trajectory", []):
        refs = (step.get("context_reference_map") or {}).get("typed_refs", {})
        for ref, node in refs.items():
            if node.get("stable_id") in final_refs and node.get("node_type") == "SENTENCE":
                cited_sentence_ids.add(node.get("stable_id"))
    gold_ids = {item.get("sentence_id") for item in facts if item.get("sentence_id")}
    answer = score_answer(episode.get("answer"), str(question.get("answer") or ""))
    evaluable = bool(facts) and not mapping_missing and bool(gold_ids)
    support_recall = (
        len(gold_ids & visible_sentence_ids) / len(gold_ids) if evaluable else None
    )
    curves = {
        str(prefix): sum(
            1
            for row in rows
            if row.get("first_complete_support_decision") is not None
            and int(row["first_complete_support_decision"]) <= prefix
        )
        for prefix in range(1, 16)
    }
    return {
        "question_id": question.get("_id"),
        "question_type": question.get("type"),
        "answer_correct_exact": answer["exact"],
        "answer_correct_contain": answer["contain"],
        "candidate_retrieved": candidate_count,
        "text_projected": len(projected_sentence_ids),
        "text_seen_by_policy": len(visible_sentence_ids),
        "evidence_eligible": len(visible_sentence_ids),
        "evidence_cited": len(gold_ids & cited_sentence_ids),
        "support_recall": support_recall,
        "complete_support": (bool(evaluable) and gold_ids.issubset(visible_sentence_ids)) if evaluable else None,
        "complete_support_cited": (bool(evaluable) and gold_ids.issubset(cited_sentence_ids)) if evaluable else None,
        "first_complete_support_decision": first_complete_decision,
        "first_complete_support_tokens": first_complete_tokens,
        "policy_calls": policy_calls,
        "invalid_attempts": invalid_attempts,
        "retrieval_actions": retrieval_actions,
        "not_evaluable": not evaluable,
        "termination_reason": episode.get("termination_reason"),
        "error_code": episode.get("error_code"),
        "usage": episode.get("usage") or {},
    }


def aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "episodes": len(rows),
        "evaluable_episodes": sum(not row.get("not_evaluable", True) for row in rows),
        "exact_rate": mean("answer_correct_exact"),
        "contain_rate": mean("answer_correct_contain"),
        "support_recall": mean("support_recall"),
        "complete_support_rate": mean("complete_support"),
        "complete_support_cited_rate": mean("complete_support_cited"),
        "mean_first_complete_support_decision": mean("first_complete_support_decision"),
        "mean_first_complete_support_tokens": mean("first_complete_support_tokens"),
        "policy_calls": sum(int(row.get("policy_calls") or 0) for row in rows),
        "invalid_attempts": sum(int(row.get("invalid_attempts") or 0) for row in rows),
        "retrieval_actions": sum(int(row.get("retrieval_actions") or 0) for row in rows),
        "failure_categories": dict(
            Counter(
                str(row.get("error_code") or row.get("termination_reason") or "unknown")
                for row in rows
                if row.get("error_code") or row.get("termination_reason") != "finish"
            )
        ),
        "prefix_acquisition_curves": curves,
    }
