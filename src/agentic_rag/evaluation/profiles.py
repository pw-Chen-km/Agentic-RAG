"""Typed profiles for the five benchmark datasets used by V3.2."""

from __future__ import annotations

import re
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal, Mapping, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_rag.errors import InputFormatError

DATASET_REPO_ID: Final = "Ayanami0730/rag_test"
DATASET_REVISION: Final = (
    "b9198a5a8702cc35c6df7542529357a9af95d928"
)
HOTPOTQA_DATASET_REPO_ID: Final = DATASET_REPO_ID
HOTPOTQA_DATASET_REVISION: Final = DATASET_REVISION

DatasetKey = Literal[
    "musique",
    "hotpotqa",
    "2wikimultihop",
    "medical",
    "novel",
]


class EvaluationMetric(StrEnum):
    LLM_ACC = "llm_acc"
    CONTAIN_ACC = "contain_acc"


class AnswerMode(StrEnum):
    SHORT = "short"
    LONG = "long"


class DuplicateQuestionIdPolicy(StrEnum):
    REJECT = "reject"
    DISAMBIGUATE_WITH_ROW = "disambiguate_with_row"


class TaskTypeStrategy(StrEnum):
    FIELD = "field"
    MUSIQUE_ID_HOPS = "musique_id_hops"


class ChunkAdjacencyMode(StrEnum):
    NUMERIC_SOURCE_ORDER = "numeric_source_order"


class DatasetProfile(BaseModel):
    """Immutable benchmark facts and dataset-specific normalization rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: DatasetKey
    display_name: str
    source_family: Literal["linear_rag", "graphrag_bench"]
    repo_id: str = DATASET_REPO_ID
    revision: str = DATASET_REVISION
    reference_question_count: int = Field(gt=0)
    reference_chunk_count: int = Field(gt=0)
    reference_unique_question_ids: int = Field(gt=0)
    reference_task_type_counts: tuple[tuple[str, int], ...]
    reported_metrics: tuple[EvaluationMetric, ...]
    skillopt_hard_metric: EvaluationMetric = EvaluationMetric.LLM_ACC
    skillopt_soft_metric: EvaluationMetric
    answer_mode: AnswerMode
    task_type_strategy: TaskTypeStrategy = TaskTypeStrategy.FIELD
    duplicate_question_id_policy: DuplicateQuestionIdPolicy = (
        DuplicateQuestionIdPolicy.REJECT
    )
    chunk_adjacency: ChunkAdjacencyMode = (
        ChunkAdjacencyMode.NUMERIC_SOURCE_ORDER
    )

    @model_validator(mode="after")
    def validate_reference_contract(self) -> Self:
        task_names = [name for name, _ in self.reference_task_type_counts]
        task_total = sum(count for _, count in self.reference_task_type_counts)
        if len(task_names) != len(set(task_names)) or any(
            not name or count <= 0
            for name, count in self.reference_task_type_counts
        ):
            raise ValueError("reference task types must be unique and positive")
        if task_total != self.reference_question_count:
            raise ValueError(
                "reference task type counts must sum to question count"
            )
        if self.reference_unique_question_ids > self.reference_question_count:
            raise ValueError(
                "unique question IDs cannot exceed the question count"
            )
        if self.skillopt_hard_metric not in self.reported_metrics:
            raise ValueError("SkillOpt hard metric must be a reported metric")
        if self.skillopt_soft_metric not in self.reported_metrics:
            raise ValueError("SkillOpt soft metric must be a reported metric")
        return self

    @property
    def allowed_task_types(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.reference_task_type_counts)

    @property
    def supports_long_answers(self) -> bool:
        return self.answer_mode is AnswerMode.LONG

    def scope_id(self, split: str) -> str:
        return f"{self.key}:benchmark_exact:{split}"

    def document_id(self, split: str) -> str:
        return f"{self.key}:{split}:benchmark_exact:corpus"

    def chunk_id(self, split: str, source_id: int) -> str:
        return f"{self.key}:{split}:benchmark_exact:c:{source_id:06d}"


def _profile(
    *,
    key: DatasetKey,
    display_name: str,
    source_family: Literal["linear_rag", "graphrag_bench"],
    questions: int,
    chunks: int,
    task_types: tuple[tuple[str, int], ...],
    metrics: tuple[EvaluationMetric, ...],
    soft_metric: EvaluationMetric,
    answer_mode: AnswerMode = AnswerMode.SHORT,
    task_type_strategy: TaskTypeStrategy = TaskTypeStrategy.FIELD,
    unique_question_ids: int | None = None,
    duplicate_policy: DuplicateQuestionIdPolicy = (
        DuplicateQuestionIdPolicy.REJECT
    ),
) -> DatasetProfile:
    return DatasetProfile(
        key=key,
        display_name=display_name,
        source_family=source_family,
        reference_question_count=questions,
        reference_chunk_count=chunks,
        reference_unique_question_ids=unique_question_ids or questions,
        reference_task_type_counts=task_types,
        reported_metrics=metrics,
        skillopt_soft_metric=soft_metric,
        answer_mode=answer_mode,
        task_type_strategy=task_type_strategy,
        duplicate_question_id_policy=duplicate_policy,
    )


_PROFILES: Final[dict[DatasetKey, DatasetProfile]] = {
    "musique": _profile(
        key="musique",
        display_name="MuSiQue",
        source_family="linear_rag",
        questions=1_000,
        chunks=1_354,
        task_types=(("2_hop", 518), ("3_hop", 316), ("4_hop", 166)),
        metrics=(EvaluationMetric.LLM_ACC, EvaluationMetric.CONTAIN_ACC),
        soft_metric=EvaluationMetric.CONTAIN_ACC,
        task_type_strategy=TaskTypeStrategy.MUSIQUE_ID_HOPS,
    ),
    "hotpotqa": _profile(
        key="hotpotqa",
        display_name="HotpotQA",
        source_family="linear_rag",
        questions=1_000,
        chunks=1_311,
        task_types=(("bridge", 811), ("comparison", 189)),
        metrics=(EvaluationMetric.LLM_ACC, EvaluationMetric.CONTAIN_ACC),
        soft_metric=EvaluationMetric.CONTAIN_ACC,
    ),
    "2wikimultihop": _profile(
        key="2wikimultihop",
        display_name="2WikiMultiHopQA",
        source_family="linear_rag",
        questions=1_000,
        chunks=658,
        task_types=(
            ("compositional", 413),
            ("inference", 108),
            ("comparison", 244),
            ("bridge_comparison", 235),
        ),
        metrics=(EvaluationMetric.LLM_ACC, EvaluationMetric.CONTAIN_ACC),
        soft_metric=EvaluationMetric.CONTAIN_ACC,
    ),
    "medical": _profile(
        key="medical",
        display_name="Medical (GraphRAG-Bench)",
        source_family="graphrag_bench",
        questions=2_062,
        chunks=225,
        task_types=(
            ("fact_retrieval", 1_098),
            ("complex_reasoning", 509),
            ("contextual_summarize", 289),
            ("creative_generation", 166),
        ),
        metrics=(EvaluationMetric.LLM_ACC,),
        soft_metric=EvaluationMetric.LLM_ACC,
        answer_mode=AnswerMode.LONG,
    ),
    "novel": _profile(
        key="novel",
        display_name="Novel (GraphRAG-Bench)",
        source_family="graphrag_bench",
        questions=2_010,
        chunks=1_117,
        unique_question_ids=2_009,
        task_types=(
            ("fact_retrieval", 971),
            ("complex_reasoning", 610),
            ("contextual_summarize", 362),
            ("creative_generation", 67),
        ),
        metrics=(EvaluationMetric.LLM_ACC,),
        soft_metric=EvaluationMetric.LLM_ACC,
        answer_mode=AnswerMode.LONG,
        duplicate_policy=(
            DuplicateQuestionIdPolicy.DISAMBIGUATE_WITH_ROW
        ),
    ),
}

DATASET_PROFILES: Final[Mapping[str, DatasetProfile]] = (
    MappingProxyType(_PROFILES)
)
HOTPOTQA_PROFILE: Final = _PROFILES["hotpotqa"]

_ALIASES: Final[dict[str, DatasetKey]] = {
    "2wiki": "2wikimultihop",
    "2wiki_multihop": "2wikimultihop",
    "2wikimultihop": "2wikimultihop",
    "2wikimultihopqa": "2wikimultihop",
    "graphrag_medical": "medical",
    "graphrag_novel": "novel",
    "hotpot": "hotpotqa",
    "hotpot_qa": "hotpotqa",
    "hotpotqa": "hotpotqa",
    "medical": "medical",
    "med": "medical",
    "musique": "musique",
    "novel": "novel",
}


def get_dataset_profile(key_or_alias: str = "hotpotqa") -> DatasetProfile:
    normalized = re.sub(r"[^a-z0-9]+", "_", key_or_alias.strip().casefold()).strip(
        "_"
    )
    key = _ALIASES.get(normalized)
    if key is None:
        supported = ", ".join(sorted(_PROFILES))
        raise InputFormatError(
            f"Unsupported benchmark dataset {key_or_alias!r}; expected one of: "
            f"{supported}"
        )
    return _PROFILES[key]


def normalize_task_type(
    profile: DatasetProfile,
    raw_type: object,
    *,
    question_id: str,
) -> str | None:
    """Return a stable task type without leaking any gold evidence."""

    raw = str(raw_type).strip() if raw_type is not None else ""
    if not raw and profile.task_type_strategy is TaskTypeStrategy.MUSIQUE_ID_HOPS:
        match = re.search(
            r"(?:^|_)([234])hop\d*(?:__|_|$)", question_id.casefold()
        )
        if match is None:
            raise InputFormatError(
                "MuSiQue question_type is blank and its ID does not encode a "
                f"2/3/4-hop task: {question_id!r}"
            )
        normalized = f"{match.group(1)}_hop"
    elif not raw:
        return None
    else:
        normalized = re.sub(r"[^a-z0-9]+", "_", raw.casefold()).strip("_")

    if normalized not in profile.allowed_task_types:
        raise InputFormatError(
            f"Benchmark {profile.key} question {question_id!r} has unsupported "
            f"question_type {raw_type!r}; expected one of "
            f"{profile.allowed_task_types}"
        )
    return normalized


__all__ = [
    "DATASET_PROFILES",
    "DATASET_REPO_ID",
    "DATASET_REVISION",
    "AnswerMode",
    "DatasetKey",
    "DatasetProfile",
    "ChunkAdjacencyMode",
    "DuplicateQuestionIdPolicy",
    "EvaluationMetric",
    "HOTPOTQA_PROFILE",
    "HOTPOTQA_DATASET_REPO_ID",
    "HOTPOTQA_DATASET_REVISION",
    "TaskTypeStrategy",
    "get_dataset_profile",
    "normalize_task_type",
]
