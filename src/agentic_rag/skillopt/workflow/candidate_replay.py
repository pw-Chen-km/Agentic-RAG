"""Train-only replay contract for rejected candidates."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class ReplayRequest:
    candidate_skill_sha256: str
    parent_skill_sha256: str
    question_ids: tuple[str, ...]
    split: str = "train"
    purpose: str = "meta_analysis_only"

    def __post_init__(self) -> None:
        if self.split != "train": raise ValueError("rejected candidate replay must use train split")
        if self.purpose != "meta_analysis_only": raise ValueError("replay cannot update active Skill")
        if not self.question_ids: raise ValueError("replay requires the completed batch question IDs")

def replay_purpose(candidate_skill_sha256: str, parent_skill_sha256: str, question_ids: Sequence[str]) -> ReplayRequest:
    return ReplayRequest(candidate_skill_sha256, parent_skill_sha256, tuple(str(x) for x in question_ids))
