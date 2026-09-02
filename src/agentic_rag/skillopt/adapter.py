"""SkillOpt v0.2.0 environment adapter for the existing Agent Harness."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

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

        def reflect(self, *args: Any, **kwargs: Any) -> list[dict | None]:
            del args, kwargs
            raise RuntimeError("SkillOpt is required for reflection")


from agentic_rag.evaluation import EpisodeEvaluator
from agentic_rag.agent.skill import SkillDocument
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
from agentic_rag.skillopt.trajectory import (
    REFLECTION_SCHEMA_VERSION,
    TrajectoryRepresentation,
    build_training_reference_text,
    normalize_trajectory_representation,
)


class AgenticRAGSkillOptAdapter(_EnvAdapter):
    """Wire one declared benchmark profile to SkillOpt's rollout loop."""

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
        fixed_answer_contract: str | None = None,
        trajectory_representation: str | TrajectoryRepresentation = "raw",
        sentence_provenance: Mapping[str, Mapping[str, Any]] | None = None,
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
        self.fixed_answer_contract = fixed_answer_contract
        self.trajectory_representation = normalize_trajectory_representation(
            trajectory_representation
        )
        self.sentence_provenance = (
            dict(sentence_provenance) if sentence_provenance is not None else None
        )
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
        self.trajectory_representation = normalize_trajectory_representation(
            cfg.get(
                "trajectory_representation",
                self.trajectory_representation.value,
            )
        )
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
        effective_skill = SkillDocument.freeze_answer_contract(
            skill_content,
            self.fixed_answer_contract,
        )
        return run_rollout_batch(
            batch=env_manager,
            out_root=out_dir,
            skill_content=effective_skill,
            harness_factory=self.harness_factory,
            evaluator=self.evaluator,
            workers=self.workers,
            resume=bool(kwargs.get("resume", self.resume)),
            trajectory_representation=kwargs.get(
                "trajectory_representation",
                self.trajectory_representation,
            ),
            sentence_provenance=kwargs.get(
                "sentence_provenance",
                self.sentence_provenance,
            ),
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
                    merged["reference_text"] = build_training_reference_text(item)
            else:
                merged.pop("reference_text", None)
            enriched.append(merged)
        return enriched

    def reflect(
        self,
        results: list[dict[str, Any]],
        skill_content: str,
        out_dir: str,
        **kwargs: Any,
    ) -> list[dict | None]:
        """Run native SkillOpt reflection over the selected trajectory view.

        SkillOpt v0.2.0 reads ``<prediction_dir>/<id>/conversation.json``.
        Target rollouts deliberately do not write that legacy file.  Instead,
        this hook creates an auditable, persistent mirror from each task's
        condition-specific ``reflection_conversation.json`` and points the
        otherwise unchanged native reflect stage at that mirror.
        """

        batch_dir = Path(out_dir)
        source_root = Path(
            kwargs.get("prediction_dir", batch_dir / "predictions")
        )
        mirror_root = batch_dir / "reflection_predictions"
        native_results: list[dict[str, Any]] = []
        for row in results:
            task_id = str(row.get("id") or "")
            if not task_id:
                raise ValueError("reflection result is missing id")
            source_dir = Path(str(row.get("artifact_dir") or ""))
            if not source_dir.is_dir():
                source_dir = source_root / task_id
            source_path = source_dir / "reflection_conversation.json"
            if not source_path.is_file():
                raise FileNotFoundError(
                    "condition-specific reflection input is missing: "
                    f"{source_path}"
                )
            rendered = json.loads(source_path.read_text(encoding="utf-8"))
            if not isinstance(rendered, Mapping):
                raise ValueError(f"invalid reflection input: {source_path}")
            if rendered.get("schema_version") != REFLECTION_SCHEMA_VERSION:
                raise ValueError(
                    "reflection input version differs from this implementation; "
                    f"use a new output directory, do not resume old inputs: {source_path}"
                )
            actual_representation = str(
                rendered.get("trajectory_representation") or ""
            )
            if actual_representation != self.trajectory_representation.value:
                raise ValueError(
                    "reflection representation differs from adapter condition: "
                    f"{actual_representation!r} != "
                    f"{self.trajectory_representation.value!r}"
                )

            mirror_dir = mirror_root / task_id
            mirror_dir.mkdir(parents=True, exist_ok=True)
            _write_json_if_same(
                mirror_dir / "conversation.json",
                _native_reflection_conversation(rendered),
            )
            native_row = dict(row)
            for prompt_name in (
                "target_system_prompt",
                "target_user_prompt",
            ):
                filename = f"{prompt_name}.txt"
                source_prompt = source_dir / filename
                destination_prompt = mirror_dir / filename
                inline_prompt = native_row.pop(prompt_name, None)
                if source_prompt.is_file():
                    _copy_if_same(source_prompt, destination_prompt)
                elif isinstance(inline_prompt, str) and inline_prompt:
                    _write_text_if_same(destination_prompt, inline_prompt)
            native_results.append(native_row)

        forwarded = dict(kwargs)
        forwarded["prediction_dir"] = mirror_root.as_posix()
        return super().reflect(
            native_results,
            skill_content,
            out_dir,
            **forwarded,
        )

    def get_task_types(self) -> list[str]:
        return list(self.profile.allowed_task_types)


__all__ = ["AgenticRAGSkillOptAdapter", "SKILLOPT_AVAILABLE"]


def _native_reflection_conversation(
    rendered: Mapping[str, Any],
) -> list[dict[str, str]]:
    representation = str(rendered["trajectory_representation"])
    legends = {
        "raw": (
            "Unorganized chronological audit. tool_raw is the environment "
            "record; policy_input_state and policy_visible_delta delimit what "
            "the Target Policy actually saw. No GT-derived step labels."
        ),
        "organized": (
            "The same rollout reorganized into a problem index, complete "
            "action ledger, and deduplicated retrieved-unit memory with every "
            "acquisition path, ordered by first acquisition step. "
            "No GT-derived step labels."
        ),
        "organized_support_labels": (
            "The organized rollout plus evaluator-derived supporting-fact "
            "progress. Retrieval text organization is unchanged from the "
            "organized arm; no answer-progress labels are present."
        ),
        "progress_abstracted": (
            "A step-by-step abstract progress record with all retrieved result "
            "text removed, including the final cited evidence text. "
            "Each step shows whether the action ran, whether it "
            "added new information, whether it moved toward the known "
            "supporting facts, and whether information acquired at that step "
            "was later used by EXPAND and produced direct progress. It does "
            "not claim that a bridge caused the final outcome, and it contains "
            "no answer-progress labels."
        ),
    }
    sections: list[tuple[str, Any]] = [
        (
            "representation_legend",
            {
                "trajectory_representation": representation,
                "meaning": legends[representation],
                "shared_decision_context": (
                    "Each step includes the model's missing_information from "
                    "that decision (null if unavailable), recorded pre-action "
                    "references and read/evidence eligibility, available "
                    "actions, and before/after budgets. No supported-fact "
                    "summaries or retrieved text are added to this common "
                    "context. Missing information may name an acquired clue. "
                    "Raw additionally keeps its original full audit; these "
                    "views are not identical in total information content."
                ),
                "execution_na_vs_zero": (
                    "not_executed/tool_error => progress N/A; an executed "
                    "empty result => zero progress"
                ),
                "count_definitions": {
                    "top_level_result_count": (
                        "Number of top-level hits directly returned by the "
                        "retriever or tool."
                    ),
                    "returned_unit_count": (
                        "Number of distinct visible Entity, Sentence, and "
                        "Chunk units recursively exposed by those hits; this "
                        "can exceed top_level_result_count because previews "
                        "inside a Chunk are separate units."
                    ),
                    "new_unit_count": (
                        "Returned visible units not visible before this step."
                    ),
                    "duplicate_unit_count": (
                        "Returned visible units already visible before this "
                        "step."
                    ),
                    "new_source_count": (
                        "Previously unseen stable document IDs returned by "
                        "this step; titles are display text only."
                    ),
                    "visibility_upgrades": (
                        "Existing units upgraded from Sentence preview to "
                        "eligible evidence or from Chunk handle to READ."
                    ),
                },
            },
        ),
        ("rollout_context", rendered.get("rollout_context")),
        ("episode_outcome", rendered.get("episode_outcome")),
        ("trajectory", rendered.get("trajectory")),
    ]
    return [
        {
            "role": "system" if index == 0 else "user",
            "content": f"## {name}\n{json.dumps(value, ensure_ascii=False, sort_keys=True)}",
        }
        for index, (name, value) in enumerate(sections)
    ]


def _write_json_if_same(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    _write_text_if_same(path, payload)


def _write_text_if_same(path: Path, payload: str) -> None:
    if path.exists():
        if path.is_file() and path.read_text(encoding="utf-8") == payload:
            return
        raise FileExistsError(f"reflection mirror already differs: {path}")
    path.write_text(payload, encoding="utf-8")


def _copy_if_same(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == source.read_bytes():
            return
        raise FileExistsError(
            f"reflection prompt mirror already differs: {destination}"
        )
    shutil.copy2(source, destination)
