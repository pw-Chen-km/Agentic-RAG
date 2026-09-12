"""Stage coordinator: runtime callbacks are injected by the experiment runner."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable
from .config import WorkflowConfig
from .candidate_replay import replay_purpose
from .trajectory_views import batch_views

@dataclass(frozen=True)
class BatchResult:
    batch_index: int
    question_ids: tuple[str, ...]
    active_skill_sha256: str
    trajectories: tuple[Any, ...]

class WorkflowCoordinator:
    """Orchestrate one complete batch without embedding policy in renderers."""
    def __init__(self, config: WorkflowConfig, *, rollout: Callable[..., Iterable[Any]], reflect: Callable[..., Any], validate: Callable[..., Any], apply: Callable[..., str], replay: Callable[..., Any] | None = None) -> None:
        self.config, self.rollout, self.reflect, self.validate, self.apply = config, rollout, reflect, validate, apply
        self.replay = replay

    def run_batch(self, *, batch_index: int, questions: list[dict[str, Any]], skill: str, skill_sha256: str) -> tuple[BatchResult, list[Any]]:
        if len(questions) > self.config.rollout_batch_size: raise ValueError("questions exceed rollout batch size")
        trajectories = tuple(self.rollout(questions=questions, skill=skill, purpose="training"))
        analyses = [self.reflect(view) for view in batch_views(trajectories, "workflow", minibatch_size=self.config.reflection_minibatch_size)]
        return BatchResult(batch_index, tuple(str(q["id"]) for q in questions), skill_sha256, trajectories), analyses

    def replay_rejected(self, *, candidate_skill_sha256: str, parent_skill_sha256: str, question_ids: list[str]) -> Any:
        request = replay_purpose(candidate_skill_sha256, parent_skill_sha256, question_ids)
        if self.replay is None: return request
        return self.replay(request=request)
