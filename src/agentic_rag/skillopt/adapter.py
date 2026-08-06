"""SkillOpt v0.2.0 environment adapter for the existing Agent Harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only with the optional research runtime
    from skillopt.envs.base import EnvAdapter as _EnvAdapter

    SKILLOPT_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback behavior is tested instead
    SKILLOPT_AVAILABLE = False

    class _EnvAdapter:
        """Minimal import fallback; real reflection requires SkillOpt."""

        def setup(self, cfg: dict[str, Any]) -> None:
            self._cfg = dict(cfg)

        def requires_ray(self) -> bool:
            return False


from agentic_rag.evaluation import EpisodeEvaluator
from agentic_rag.evaluation.profiles import (
    DatasetProfile,
    get_dataset_profile,
)
from agentic_rag.skillopt.dataloader import (
    AgenticRAGSkillOptDataLoader,
    BatchSpec,
    item_to_dict,
)
from agentic_rag.skillopt.rollout import (
    HarnessFactory,
    RolloutBatch,
    run_rollout_batch,
)


class AgenticRAGSkillOptAdapter(_EnvAdapter):
    """Wire HotpotQA to SkillOpt's rollout/reflection loop."""

    def __init__(
        self,
        *,
        split_dir: str | Path,
        harness_factory: HarnessFactory,
        evaluator: EpisodeEvaluator,
        dataset: str | DatasetProfile = "hotpotqa",
        workers: int = 1,
        analyst_workers: int = 1,
        failure_only: bool = False,
        minibatch_size: int = 3,
        edit_budget: int = 1,
        seed: int = 42,
        limit: int = 0,
        resume: bool = True,
    ) -> None:
        if workers != 1:
            raise ValueError(
                "the inspectable smoke workflow requires workers=1"
            )
        if analyst_workers <= 0:
            raise ValueError("analyst_workers must be positive")
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if edit_budget <= 0:
            raise ValueError("edit_budget must be positive")
        self.profile = (
            dataset
            if isinstance(dataset, DatasetProfile)
            else get_dataset_profile(dataset)
        )
        if evaluator.profile.key != self.profile.key:
            raise ValueError(
                f"evaluator profile is {evaluator.profile.key!r}; expected "
                f"{self.profile.key!r}"
            )
        self.harness_factory = harness_factory
        self.evaluator = evaluator
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = bool(failure_only)
        self.minibatch_size = int(minibatch_size)
        self.edit_budget = int(edit_budget)
        self.resume = bool(resume)
        self.dataloader = AgenticRAGSkillOptDataLoader(
            split_dir,
            dataset=self.profile.key,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict[str, Any]) -> None:
        super().setup(cfg)
        configured_workers = int(cfg.get("workers", self.workers))
        if configured_workers != 1:
            raise ValueError(
                "the inspectable smoke workflow requires workers=1"
            )
        self.workers = configured_workers
        self.analyst_workers = int(
            cfg.get("analyst_workers", self.analyst_workers)
        )
        self.failure_only = bool(
            cfg.get("failure_only", self.failure_only)
        )
        self.minibatch_size = int(
            cfg.get("minibatch_size", self.minibatch_size)
        )
        self.edit_budget = int(cfg.get("edit_budget", self.edit_budget))
        if min(
            self.analyst_workers,
            self.minibatch_size,
            self.edit_budget,
        ) <= 0:
            raise ValueError(
                "analyst_workers, minibatch_size, and edit_budget must be positive"
            )
        self.dataloader.setup(cfg)

    def get_dataloader(self) -> AgenticRAGSkillOptDataLoader:
        return self.dataloader

    def requires_ray(self) -> bool:
        return False

    def build_env_from_batch(
        self, batch: BatchSpec, **kwargs: Any
    ) -> RolloutBatch:
        del kwargs
        raw_items = batch.payload or []
        if not isinstance(raw_items, (list, tuple)):
            raise TypeError("dataset-backed SkillOpt batch payload must be a list")
        return RolloutBatch(
            items=tuple(item_to_dict(item) for item in raw_items),
            phase=str(batch.phase),
            split=str(batch.split),
        )

    def build_train_env(
        self, batch_size: int, seed: int, **kwargs: Any
    ) -> RolloutBatch:
        batch = self.dataloader.build_train_batch(
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )
        return self.build_env_from_batch(batch)

    def build_eval_env(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs: Any,
    ) -> RolloutBatch:
        batch = self.dataloader.build_eval_batch(
            env_num=env_num,
            split=split,
            seed=seed,
            **kwargs,
        )
        return self.build_env_from_batch(batch)

    def rollout(
        self,
        env_manager: RolloutBatch | list[dict[str, Any]],
        skill_content: str,
        out_dir: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return run_rollout_batch(
            batch=env_manager,
            out_root=out_dir,
            skill_content=skill_content,
            harness_factory=self.harness_factory,
            evaluator=self.evaluator,
            workers=self.workers,
            resume=bool(kwargs.get("resume", self.resume)),
        )

    def attach_reference_context(
        self,
        results: list[dict[str, Any]],
        items: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Expose answer references to reflection for train rollouts only."""

        item_by_id = {
            str(item["id"]): item
            for item in (items or [])
            if isinstance(item, dict) and item.get("id") is not None
        }
        enriched: list[dict[str, Any]] = []
        for row in results:
            merged = dict(row)
            if merged.get("rollout_phase") == "train":
                item = item_by_id.get(str(merged.get("id")))
                if item is not None and item.get("answer") is not None:
                    merged["reference_text"] = str(item["answer"])
            else:
                merged.pop("reference_text", None)
            enriched.append(merged)
        return enriched

    def get_task_types(self) -> list[str]:
        return list(self.profile.allowed_task_types)


__all__ = ["AgenticRAGSkillOptAdapter", "SKILLOPT_AVAILABLE"]
