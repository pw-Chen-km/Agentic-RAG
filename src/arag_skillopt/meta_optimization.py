"""Meta comparison records for parent/candidate A-RAG paths."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping


@dataclass(frozen=True)
class WorkflowComparison:
    question_id: str
    parent_skill_hash: str
    candidate_skill_hash: str
    parent_trajectory: Mapping[str, Any]
    candidate_trajectory: Mapping[str, Any]
    train_batch_id: str
    match_level: str = "same_question"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_comparison(
    *, question_id: str, parent_skill_hash: str, candidate_skill_hash: str,
    parent_trajectory: Mapping[str, Any], candidate_trajectory: Mapping[str, Any],
    train_batch_id: str, match_level: str = "same_question",
) -> WorkflowComparison:
    return WorkflowComparison(question_id, parent_skill_hash, candidate_skill_hash,
                              parent_trajectory, candidate_trajectory, train_batch_id, match_level)


def compare_parent_candidate(parent: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, non-causal comparison for a paired workflow."""
    return {
        "question_id": parent.get("question_id") or candidate.get("question_id"),
        "parent_skill_hash": parent.get("skill_hash"),
        "candidate_skill_hash": candidate.get("skill_hash"),
        "parent_calls": parent.get("policy_calls"),
        "candidate_calls": candidate.get("policy_calls"),
        "parent_tokens": parent.get("total_tokens"),
        "candidate_tokens": candidate.get("total_tokens"),
        "same_question": (parent.get("question_id") == candidate.get("question_id")),
    }
