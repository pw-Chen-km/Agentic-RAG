"""Immutable proposal receipts, independent of the LLM implementation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
@dataclass(frozen=True)
class Proposal:
    candidate_id: str
    stage: str
    parent_skill_sha256: str
    candidate_skill_sha256: str
    text: str
    supporting_question_ids: tuple[str, ...] = ()
    def __post_init__(self):
        if self.stage not in {'retrieval','meta','answer'}: raise ValueError('invalid proposal stage')
        if not self.candidate_id: raise ValueError('candidate id required')
