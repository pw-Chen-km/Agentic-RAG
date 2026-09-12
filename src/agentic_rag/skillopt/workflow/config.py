"""Configuration contracts for the staged workflow optimizer."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping

@dataclass(frozen=True)
class StageConfig:
    name: str
    editable_sections: tuple[str, ...]
    reflection_minibatch_size: int = 5

@dataclass(frozen=True)
class WorkflowConfig:
    """A complete, validated staged-training configuration."""
    rollout_batch_size: int = 40
    reflection_minibatch_size: int = 5
    stages: tuple[StageConfig, ...] = field(default_factory=lambda: (
        StageConfig("retrieval", ("retrieval_policy", "recovery_policy")),
        StageConfig("meta", ("retrieval_policy", "recovery_policy")),
        StageConfig("answer", ("answer_policy",)),
    ))
    replay_rejected_candidates: bool = True
    replay_split: str = "train"
    no_test_in_meta: bool = True

    def __post_init__(self) -> None:
        if self.rollout_batch_size <= 0 or self.reflection_minibatch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.reflection_minibatch_size > self.rollout_batch_size:
            raise ValueError("reflection minibatch cannot exceed rollout batch")
        names = tuple(s.name for s in self.stages)
        if names != ("retrieval", "meta", "answer"):
            raise ValueError("stages must be retrieval, meta, answer")
        if self.replay_split != "train":
            raise ValueError("rejected-candidate replay is train-only")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkflowConfig":
        stages = value.get("stages")
        parsed = None
        if stages is not None:
            parsed = tuple(StageConfig(str(s["name"]), tuple(s.get("editable_sections", ())), int(s.get("reflection_minibatch_size", value.get("reflection_minibatch_size", 5)))) for s in stages)
        return cls(rollout_batch_size=int(value.get("rollout_batch_size", 40)), reflection_minibatch_size=int(value.get("reflection_minibatch_size", 5)), stages=parsed or cls().stages, replay_rejected_candidates=bool(value.get("replay_rejected_candidates", True)), replay_split=str(value.get("replay_split", "train")), no_test_in_meta=bool(value.get("no_test_in_meta", True)))
