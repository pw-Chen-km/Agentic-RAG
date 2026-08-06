"""Pinned HotpotQA benchmark facts used by evaluation and SkillOpt."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


HOTPOTQA_DATASET_REPO_ID: Final = "Ayanami0730/rag_test"
HOTPOTQA_DATASET_REVISION: Final = (
    "b9198a5a8702cc35c6df7542529357a9af95d928"
)


class EvaluationMetric(StrEnum):
    LLM_ACC = "llm_acc"
    CONTAIN_ACC = "contain_acc"


class DatasetProfile(BaseModel):
    """Immutable facts for the single supported HotpotQA evaluation set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = "hotpotqa"
    display_name: str = "HotpotQA"
    repo_id: str = HOTPOTQA_DATASET_REPO_ID
    revision: str = HOTPOTQA_DATASET_REVISION
    reference_question_count: int = Field(default=1_000, gt=0)
    reference_chunk_count: int = Field(default=1_311, gt=0)
    reference_unique_question_ids: int = Field(default=1_000, gt=0)
    reference_task_type_counts: tuple[tuple[str, int], ...] = (
        ("bridge", 811),
        ("comparison", 189),
    )
    reported_metrics: tuple[EvaluationMetric, ...] = (
        EvaluationMetric.LLM_ACC,
        EvaluationMetric.CONTAIN_ACC,
    )
    skillopt_hard_metric: EvaluationMetric = EvaluationMetric.LLM_ACC
    skillopt_soft_metric: EvaluationMetric = EvaluationMetric.CONTAIN_ACC

    @model_validator(mode="after")
    def validate_reference_contract(self) -> Self:
        names = [name for name, _ in self.reference_task_type_counts]
        if len(names) != len(set(names)):
            raise ValueError("reference task types must be unique")
        if sum(count for _, count in self.reference_task_type_counts) != self.reference_question_count:
            raise ValueError("reference task type counts must sum to question count")
        return self

    @property
    def allowed_task_types(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.reference_task_type_counts)

    def scope_id(self, split: str) -> str:
        return f"hotpotqa:benchmark_exact:{split}"

    def document_id(self, split: str) -> str:
        return f"hotpotqa:{split}:benchmark_exact:corpus"

    def chunk_id(self, split: str, source_id: int) -> str:
        return f"hotpotqa:{split}:benchmark_exact:c:{source_id:06d}"


HOTPOTQA_PROFILE: Final = DatasetProfile()


def get_dataset_profile(name: str = "hotpotqa") -> DatasetProfile:
    normalized = name.strip().casefold().replace("-", "").replace("_", "")
    if normalized not in {"hotpotqa", "hotpot"}:
        raise ValueError("only the HotpotQA benchmark profile is supported")
    return HOTPOTQA_PROFILE


__all__ = [
    "DatasetProfile",
    "EvaluationMetric",
    "HOTPOTQA_DATASET_REPO_ID",
    "HOTPOTQA_DATASET_REVISION",
    "HOTPOTQA_PROFILE",
    "get_dataset_profile",
]
