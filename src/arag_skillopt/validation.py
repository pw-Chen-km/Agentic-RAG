"""Small validation-gate interface kept separate from the A-RAG runtime."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    metrics: Mapping[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_gate(
    candidate_skill: Mapping[str, Any],
    question_ids: Iterable[str],
    evaluator: Callable[[Mapping[str, Any], list[str]], Mapping[str, Any]],
    *,
    accept_if: Callable[[Mapping[str, Any]], bool] | None = None,
) -> GateResult:
    metrics = dict(evaluator(candidate_skill, list(question_ids)))
    accepted = bool(accept_if(metrics) if accept_if else metrics.get("accepted", False))
    return GateResult(accepted=accepted, metrics=metrics, reason="accepted" if accepted else "rejected")


def accept_candidate(metrics: Mapping[str, Any], *, minimum_accuracy: float | None = None) -> bool:
    """Compatibility helper for small smoke runners."""
    if "accepted" in metrics:
        return bool(metrics["accepted"])
    if minimum_accuracy is not None:
        return float(metrics.get("accuracy", 0.0)) >= minimum_accuracy
    return False
