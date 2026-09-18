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
    enable_meta: bool = True
    replay_split: str = "train"
    no_test_in_meta: bool = True
    # The v2 seed contains answer rules for rendering, but Retrieval and Meta
    # runs keep them locked until the answer stage gets its own experiment.
    enable_answer_updates: bool = False
    # When false, candidate Skills are still measured on validation and test,
    # but validation is observational only and cannot block adoption.  This
    # is an explicit ablation, not the production safety gate.
    use_validation_gate: bool = True
    # Ollama may reject a very large reflection request.  The runner keeps the
    # configured minibatch when possible and splits only requests over this
    # serialized-character limit (normally down to one case).
    max_reflection_input_chars: int = 120_000
    max_optimizer_input_tokens: int = 16_000
    max_edits_per_reflection: int = 2
    max_edits_per_batch: int = 2
    max_edit_tokens: int = 250
    max_trainable_skill_tokens: int = 1_500

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
        if self.max_reflection_input_chars <= 0:
            raise ValueError("max_reflection_input_chars must be positive")
        if self.max_optimizer_input_tokens <= 0:
            raise ValueError("max_optimizer_input_tokens must be positive")
        if self.max_edits_per_reflection <= 0 or self.max_edits_per_batch <= 0:
            raise ValueError("edit limits must be positive")
        if self.max_edit_tokens <= 0 or self.max_trainable_skill_tokens <= 0:
            raise ValueError("token limits must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkflowConfig":
        stages = value.get("stages")
        parsed = None
        if stages is not None:
            parsed = tuple(StageConfig(str(s["name"]), tuple(s.get("editable_sections", ())), int(s.get("reflection_minibatch_size", value.get("reflection_minibatch_size", 5)))) for s in stages)
        return cls(
            rollout_batch_size=int(value.get("rollout_batch_size", 40)),
            reflection_minibatch_size=int(value.get("reflection_minibatch_size", 5)),
            stages=parsed or cls().stages,
            replay_rejected_candidates=bool(value.get("replay_rejected_candidates", True)),
            enable_meta=bool(value.get("enable_meta", True)),
            replay_split=str(value.get("replay_split", "train")),
            no_test_in_meta=bool(value.get("no_test_in_meta", True)),
            enable_answer_updates=bool(value.get("enable_answer_updates", False)),
            use_validation_gate=bool(value.get("use_validation_gate", True)),
            max_reflection_input_chars=int(value.get("max_reflection_input_chars", 120_000)),
            max_optimizer_input_tokens=int(value.get("max_optimizer_input_tokens", 16_000)),
            max_edits_per_reflection=int(value.get("max_edits_per_reflection", 2)),
            max_edits_per_batch=int(value.get("max_edits_per_batch", 2)),
            max_edit_tokens=int(value.get("max_edit_tokens", 250)),
            max_trainable_skill_tokens=int(value.get("max_trainable_skill_tokens", 1_500)),
        )
