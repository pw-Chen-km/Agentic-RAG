"""Gate on complete SAME-question validation pairs; unknown is not wrong/zero.

Legacy defaults still require 100% coverage. New configs explicitly permit a
small unavailable fraction, always excluding a pair from BOTH metric populations.
"""

import math

GATE_VERSION = "complete-same-question-pairs-v2"


def unavailable_reasons(episode):
    evaluation = episode.get("evaluation", {})
    raw = episode.get("raw", {})
    usage = raw.get("usage", {})
    reasons = []
    if evaluation.get("status") != "ok" or type(evaluation.get("correct")) is not bool:
        reasons.append("answer_evaluation_unavailable")
    if raw.get("execution_errors") or raw.get("error"):
        reasons.append("runtime_error")
    tokens, calls = usage.get("total_tokens"), raw.get("policy_calls")
    if (
        usage.get("complete") is not True
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in (tokens, calls)
        )
        or not isinstance(calls, int)
        or calls < 1
    ):
        reasons.append("cost_measurement_unavailable")
    return reasons


def metrics(episodes):
    valid = all(
        type(e["evaluation"].get("correct")) is bool
        and e["evaluation"].get("status") == "ok"
        and e["raw"].get("usage", {}).get("complete")
        and not e["raw"].get("execution_errors")
        for e in episodes
    )
    return {
        "count": len(episodes),
        "complete": valid and bool(episodes),
        "accuracy": sum(e["evaluation"]["correct"] for e in episodes) / len(episodes)
        if valid and episodes
        else None,
        "target_tokens": sum(e["raw"]["usage"]["total_tokens"] for e in episodes)
        if valid
        else None,
        "policy_calls": sum(e["raw"]["policy_calls"] for e in episodes)
        if valid
        else None,
    }


def decide(
    parent,
    candidate,
    accuracy_tolerance=0.02,
    min_efficiency_gain=0.05,
    *,
    max_token_cost_increase=0.05,
    max_call_cost_increase=0.05,
    min_validation_pair_coverage=1.0,
):
    for value in (max_token_cost_increase, max_call_cost_increase):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
        ):
            raise ValueError("cost increase limits must be finite fractions in [0,1]")
    if (
        isinstance(min_validation_pair_coverage, bool)
        or not isinstance(min_validation_pair_coverage, (int, float))
        or not 0 < min_validation_pair_coverage <= 1
    ):
        raise ValueError("validation pair coverage must be a finite fraction in (0,1]")
    ids_a = [e["question_id"] for e in parent]
    ids_b = [e["question_id"] for e in candidate]
    result = {
        "accepted": False,
        "gate_version": GATE_VERSION,
        "metric_population": "complete_same_question_pairs",
        "max_token_cost_increase": max_token_cost_increase,
        "max_call_cost_increase": max_call_cost_increase,
        "min_validation_pair_coverage": min_validation_pair_coverage,
        "expected_count": len(parent),
    }
    # Unknown observations can be excluded. Corrupt IDs/splits cannot be.
    if (
        not ids_a
        or any(not isinstance(i, str) or not i for i in ids_a + ids_b)
        or len(ids_a) != len(set(ids_a))
        or len(ids_b) != len(set(ids_b))
        or set(ids_a) != set(ids_b)
        or (min_validation_pair_coverage == 1.0 and ids_a != ids_b)
    ):
        return {**result, "reason": "incomplete_or_unpaired_validation"}
    by_a, by_b = dict(zip(ids_a, parent)), dict(zip(ids_b, candidate))
    paired_ids, excluded = [], []
    for qid in ids_a:
        ra, rb = unavailable_reasons(by_a[qid]), unavailable_reasons(by_b[qid])
        if ra or rb:
            excluded.append(
                {"question_id": qid, "parent_reasons": ra, "candidate_reasons": rb}
            )
        else:
            paired_ids.append(qid)
    a, b = (
        metrics([by_a[i] for i in paired_ids]),
        metrics([by_b[i] for i in paired_ids]),
    )
    required = math.ceil(len(parent) * min_validation_pair_coverage)
    result.update(
        parent=a,
        candidate=b,
        paired_count=len(paired_ids),
        pair_coverage=len(paired_ids) / len(parent),
        required_paired_count=required,
        paired_question_ids=paired_ids,
        excluded_count=len(excluded),
        excluded_pairs=excluded,
        parent_unavailable_count=sum(bool(x["parent_reasons"]) for x in excluded),
        candidate_unavailable_count=sum(bool(x["candidate_reasons"]) for x in excluded),
    )
    if len(paired_ids) < required or not paired_ids:
        return {
            **result,
            "reason": "incomplete_or_unpaired_validation"
            if min_validation_pair_coverage == 1.0
            else "insufficient_paired_validation_coverage",
        }
    if a["target_tokens"] <= 0 or a["policy_calls"] <= 0:
        return {**result, "reason": "unusable_paired_validation_cost"}
    da = b["accuracy"] - a["accuracy"]
    tg = (a["target_tokens"] - b["target_tokens"]) / a["target_tokens"]
    cg = (a["policy_calls"] - b["policy_calls"]) / a["policy_calls"]
    def at_least(value, boundary):
        return value >= boundary or math.isclose(value, boundary, rel_tol=0, abs_tol=1e-12)

    accepted = (
        (da > 0 or (at_least(da, -accuracy_tolerance)
                    and at_least(max(tg, cg), min_efficiency_gain)))
        and at_least(tg, -max_token_cost_increase)
        and at_least(cg, -max_call_cost_increase)
    )
    return {
        **result,
        "accuracy_delta": da,
        "token_gain": tg,
        "call_gain": cg,
        "accepted": accepted,
        "reason": "passed_existing_gate"
        if accepted
        else "accuracy_efficiency_gate_rejected",
    }
